"""Fused chunk forward: hidden-state propagation + output in a single Pallas kernel.

Fuses ``_chunk_fwd_h_kernel`` and ``_chunk_fwd_o_kernel`` so that each
(BK, BV) tile of the hidden state stays in VMEM scratch across chunks,
eliminating the HBM round-trip for the full ``h: [B, NT, H, K, V]`` tensor.

Grid: (B, H, K // BK, V // BV, NT)
  - First 4 dims parallel, NT arbitrary (sequential).
  - Each grid point handles one (BK, BV) tile of the hidden state.

At each time step the kernel:
  1. Computes a *partial* output contribution from this K tile:
       partial = q_k @ h_k * gate + causal(q_k @ k_k^T * gate) @ v
  2. Updates this K tile's hidden state:
       h_k = h_k * decay + k_k^T @ v

The launcher sums partials across K tiles and applies scale:
  o = sum_over_K(partial) * scale
"""

import functools

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

from tops.ops.utils import exp, is_tpu_runtime
from tops.utils import assert_shape, assert_shape_or_none


# ---------------------------------------------------------------------------
# Pallas kernel
# ---------------------------------------------------------------------------

def _fused_chunk_fwd_kernel(
    q_ref,          # (1, 1, BT, BK)  — K-tiled
    k_ref,          # (1, 1, BT, BK)  — K-tiled
    v_ref,          # (1, 1, BT, BV)  — V-tiled
    h0_ref,         # (1, 1, BK, BV)  or None
    g_ref,          # (1, 1, BT, 128) or None
    g_gamma_ref,    # [H] via SMEM/ANY, or None
    o_ref,          # (1, 1, 1, BT, BV)  — partial output per K tile
    ht_ref,         # (1, 1, BK, BV)   — output, or None
    scratch_ref,    # (BK, BV) VMEM float32
    *,
    BT: int,
    NT: int,
):
    """Fused Pallas kernel: hidden-state propagation + output computation.

    Grid: (B, H, NK, NV, NT)
      - First 4 dims are parallel; NT is arbitrary (sequential over time).
      - NK = K // BK, NV = V // BV.

    Each grid point processes one (BK, BV) tile of the hidden state
    across all time steps.  At each time step:
      1. Compute partial output contribution from this K tile.
      2. Update this K tile's hidden state.

    The partial outputs are summed across K tiles and scaled by the launcher.

    Refs (after block-spec indexing):
      q_ref / k_ref : (1, 1, BT, BK)  — K-tiled
      v_ref         : (1, 1, BT, BV)  — V-tiled
      h0_ref        : (1, 1, BK, BV)  — initial state tile, or None
      g_ref         : (1, 1, BT, 128)  — scalar gate (broadcast to 4D for TPU alignment), or None
      g_gamma_ref   : [H]             — per-head fixed decay, or None
      o_ref         : (1, 1, 1, BT, BV) — partial output tile
      ht_ref        : (1, 1, BK, BV)  — final-state tile, or None
      scratch_ref   : (BK, BV)        — running hidden state in VMEM
    """
    BK = q_ref.shape[3]
    BV = v_ref.shape[3]
    i_t = pl.program_id(4)

    # ---- Precompute g_gamma ramp (constant across chunks) ----
    if g_gamma_ref is not None:
        head_idx = pl.program_id(1)
        b_gamma = g_gamma_ref[head_idx].astype(jnp.float32)
        b_g_gamma = b_gamma * (jnp.arange(BT) + 1).astype(jnp.float32)  # [BT]

    # ---- Initialize hidden state on first chunk ----
    @pl.when(i_t == 0)
    def _init():
        if h0_ref is not None:
            scratch_ref[:, :] = h0_ref[0, 0].astype(jnp.float32)
        else:
            scratch_ref[:, :] = jnp.zeros((BK, BV), dtype=jnp.float32)

    # ---- Load current chunk ----
    b_q = q_ref[0, 0]     # (BT, BK)
    b_k = k_ref[0, 0]     # (BT, BK)
    b_v = v_ref[0, 0]     # (BT, BV)
    b_h = scratch_ref[...]  # (BK, BV) — state *before* this chunk's update

    # ===== Stage 1: Partial output for this K tile =====
    # Partial inter-chunk: q_k @ h_k
    partial_o = jnp.dot(
        b_q, b_h,
        preferred_element_type=jnp.float32,
    )  # (BT, BV)

    # Partial intra-chunk attention: q_k @ k_k^T
    partial_A = jnp.dot(
        b_q, b_k.T,
        preferred_element_type=jnp.float32,
    )  # (BT, BT)

    # Apply scalar gate g
    if g_ref is not None:
        b_g = g_ref[0, 0, :, 0].astype(jnp.float32)  # (BT,)
        partial_o = partial_o * exp(b_g)[:, None]
        g_diff = b_g[:, None] - b_g[None, :]
        fwd_mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
        safe_g_diff = jnp.where(fwd_mask, g_diff, 0.0)
        partial_A = partial_A * exp(safe_g_diff)

    # Apply per-head fixed decay g_gamma
    if g_gamma_ref is not None:
        partial_o = partial_o * exp(b_g_gamma)[:, None]
        g_gamma_diff = b_g_gamma[:, None] - b_g_gamma[None, :]
        fwd_mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
        safe_g_gamma_diff = jnp.where(fwd_mask, g_gamma_diff, 0.0)
        partial_A = partial_A * exp(safe_g_gamma_diff)

    # Causal mask (lower triangular: i >= j)
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    partial_A = jnp.where(mask, partial_A, 0.0)

    # Partial contribution (inter + intra); scale applied after K reduction
    partial = partial_o + jnp.dot(
        partial_A, b_v.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )  # (BT, BV)
    o_ref[0, 0, 0] = partial.astype(o_ref.dtype)

    # ===== Stage 2: Update hidden state for this K tile =====
    b_v_upd = b_v

    # Decay state and adjust v for scalar gate g
    if g_ref is not None:
        b_g_last = b_g[BT - 1]
        scratch_ref[...] *= exp(b_g_last)
        b_v_upd = (b_v_upd * exp(b_g_last - b_g)[:, None]).astype(b_v_upd.dtype)

    # Decay state and adjust v for g_gamma
    if g_gamma_ref is not None:
        b_g_gamma_last = b_gamma * BT
        scratch_ref[...] *= exp(b_g_gamma_last)
        b_v_upd = (b_v_upd * exp(b_g_gamma_last - b_g_gamma)[:, None]).astype(b_v_upd.dtype)

    # State update: h_k += k_k^T @ v_upd
    scratch_ref[...] = scratch_ref[...] + jnp.dot(
        b_k.astype(jnp.float32).T,
        b_v_upd.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # Store final hidden state
    @pl.when(i_t == NT - 1)
    def _store_ht():
        if ht_ref is not None:
            ht_ref[0, 0] = scratch_ref[...].astype(ht_ref.dtype)


# ---------------------------------------------------------------------------
# Pallas launcher
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=("output_final_state", "chunk_size", "interpret"),
)
def fused_chunk_fwd_kernel(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    h0: jax.Array | None = None,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float,
    output_final_state: bool = False,
    chunk_size: int = 64,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """Pallas launcher for the fused chunk forward kernel.

    Reshapes inputs to (B, H, T, D) layout, launches the fused kernel with
    K-tiled and V-tiled grid, reduces partial outputs across K tiles, and
    transposes the result back to (B, T, H, D).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    BK = 128
    BV = 128
    NK = K // BK
    NV = V // BV

    # Transpose to (B, H, T, D) layout for the kernel
    _q = q.transpose(0, 2, 1, 3)   # (B, H, T, K)
    _k = k.transpose(0, 2, 1, 3)   # (B, H, T, K)
    _v = v.transpose(0, 2, 1, 3)   # (B, H, T, V)

    _g = None
    if g is not None:
        _g = g.transpose(0, 2, 1)   # (B, H, T)
        _g = jnp.broadcast_to(_g[:, :, :, None], (B, H, T, 128))  # (B, H, T, 128)

    grid = (B, H, NK, NV, NT)

    # ---- Index maps ----
    def qk_map(b, h, ik, iv, t):
        return (b, h, t, ik)

    def v_map(b, h, ik, iv, t):
        return (b, h, t, iv)

    def state_map(b, h, ik, iv, t):
        return (b, h, ik, iv)

    def g_map(b, h, ik, iv, t):
        return (b, h, t, 0)

    def o_map(b, h, ik, iv, t):
        return (b, h, ik, t, iv)

    # ---- Specs ----
    smem = pltpu.ANY if interpret else pltpu.SMEM

    spec_qk    = pl.BlockSpec((1, 1, BT, BK), qk_map)
    spec_v     = pl.BlockSpec((1, 1, BT, BV), v_map)
    spec_h0    = pl.BlockSpec((1, 1, BK, BV), state_map) if h0 is not None else None
    spec_g     = pl.BlockSpec((1, 1, BT, 128), g_map) if _g is not None else None
    spec_gamma = pl.BlockSpec(memory_space=smem) if g_gamma is not None else None

    # o_partial: (B, H, NK, T, V) — partial contribution per K tile, float32
    spec_o     = pl.BlockSpec((1, 1, 1, BT, BV), o_map)
    spec_ht    = pl.BlockSpec((1, 1, BK, BV), state_map) if output_final_state else None

    # ---- Output shapes ----
    out_shapes = [
        jax.ShapeDtypeStruct((B, H, NK, T, V), jnp.float32),   # o_partial
        jax.ShapeDtypeStruct((B, H, K, V), jnp.float32)        # ht
        if output_final_state else None,
    ]

    # ---- Launch ----
    o_partial, ht = pl.pallas_call(
        functools.partial(_fused_chunk_fwd_kernel, BT=BT, NT=NT),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[spec_qk, spec_qk, spec_v, spec_h0,
                      spec_g, spec_gamma],
            out_specs=[spec_o, spec_ht],
            scratch_shapes=[pltpu.VMEM((BK, BV), jnp.float32)],
        ),
        out_shape=out_shapes,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel", "parallel", "parallel", "parallel", "arbitrary",
            ),
        ),
        interpret=interpret,
    )(_q, _k, _v, h0, _g, g_gamma)

    # Reduce across K tiles and apply scale
    o = o_partial.sum(axis=2) * scale   # (B, H, T, V)

    # Transpose back: (B, H, T, V) -> (B, T, H, V)
    o = o.transpose(0, 2, 1, 3).astype(v.dtype)

    return o, ht


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_chunk_fwd(
    q: jax.Array,                        # [B, T, H, K]
    k: jax.Array,                        # [B, T, H, K]
    v: jax.Array,                        # [B, T, H, V]
    *,
    g: jax.Array | None = None,          # [B, T, H]  chunk-local cumsum of scalar gate
    g_gamma: jax.Array | None = None,    # [H]         per-head fixed decay rate
    h0: jax.Array | None = None,         # [B, H, K, V] initial hidden state
    scale: float | None = None,
    use_ht: bool = False,
    chunk_size: int = 64,
    interpret: bool | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Fused chunk forward: compute output *o* and optionally final state *ht*.

    Merges hidden-state propagation (``chunk_fwd_h``) and output computation
    (``chunk_fwd_o``) into a single Pallas kernel.  Each (BK, BV) tile of the
    hidden state is kept in VMEM scratch, avoiding the HBM materialisation of
    ``h: [B, NT, H, K, V]``.

    Both K and V are tiled (BK = BV = 128).  The kernel writes per-K-tile
    partial output contributions; the launcher reduces across K tiles and
    applies the attention scale.

    Recurrence per chunk c (chunk_size positions):
        h_c = h_{c-1} * decay(g, g_gamma) + k_c^T @ v_c
        o_c = (q_c @ h_{c-1} + causal(q_c @ k_c^T) @ v_c) * scale

    Args:
        q:  [B, T, H, K] — queries.
        k:  [B, T, H, K] — keys.
        v:  [B, T, H, V] — values.
        g:  [B, T, H]    — chunk-local cumsum of scalar gate (optional).
        g_gamma: [H]     — per-head fixed decay rate (optional).
        h0: [B, H, K, V] — initial hidden state (optional).
        scale: attention scale. Defaults to K ** -0.5.
        output_final_state: if True, also return the final hidden state.
        chunk_size: block size.  T must be divisible by chunk_size.
        interpret: Pallas interpret mode.  None = auto-detect.

    Returns:
        o:  [B, T, H, V]      — output tensor.
        ht: [B, H, K, V] or None — final hidden state.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    if scale is None:
        scale = K ** -0.5

    # =================== assert kernel requirements start ===================
    assert_shape(q, (B, T, H, K))
    assert_shape(k, (B, T, H, K))
    assert_shape(v, (B, T, H, V))
    assert_shape_or_none(g, (B, T, H))
    assert_shape_or_none(g_gamma, (H,))
    assert_shape_or_none(h0, (B, H, K, V))
    assert K % 128 == 0, f"K={K} must be a multiple of 128"
    assert V % 128 == 0, f"V={V} must be a multiple of 128"
    assert T % chunk_size == 0, (
        f"Sequence length T={T} must be divisible by chunk_size={chunk_size}"
    )
    assert scale is not None
    # =================== assert kernel requirements done ===================

    if interpret is None:
        interpret = not is_tpu_runtime()

    return fused_chunk_fwd_kernel(
        q=q,
        k=k,
        v=v,
        h0=h0,
        g=g,
        g_gamma=g_gamma,
        scale=scale,
        output_final_state=use_ht,
        chunk_size=chunk_size,
        interpret=interpret,
    )
