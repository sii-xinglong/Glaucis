# Source: primatrix/pallas-kernel @ branch: main
# Commit: cbeeae7953583f8b5a406a0a76d233f8df569352
# Initialized: 2026-04-01
"""Reference fused_chunk_simple_gla kernel from primatrix/pallas-kernel.

Upstream code copied verbatim into a single self-contained file.
Used as the correctness baseline for evolutionary optimization.

Forward: fused_chunk_simple_gla_fwd — delegates to fused_chunk_fwd (common module).
Backward: fuses dh propagation + dq/dk/dv in a single Pallas kernel.
Forward pre-computes h via chunk_fwd_h and saves as residual (no recomputation).
"""

# --- All imports (deduplicated from all upstream files) ---
import os
import functools

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu


# --- Copied upstream code (in dependency order, leaves first) ---


# From: tops/ops/utils.py
def exp(x):
    return jnp.exp(x.astype(jnp.float32))


# From: tops/ops/utils.py
_IS_TPU_RUNTIME_CACHED: bool | None = None

def is_tpu_runtime() -> bool:
    """Return True if the current JAX runtime is on TPU devices.

    Prefer checking actual devices; fall back to default backend if necessary.
    """
    global _IS_TPU_RUNTIME_CACHED
    if _IS_TPU_RUNTIME_CACHED is not None:
        return _IS_TPU_RUNTIME_CACHED
    try:
        devs = jax.devices()
        _IS_TPU_RUNTIME_CACHED = len(devs) > 0 and all(
            d.platform == "tpu" for d in devs
        )
    except Exception:
        _IS_TPU_RUNTIME_CACHED = jax.default_backend() == "tpu"
    return _IS_TPU_RUNTIME_CACHED


# From: tops/ops/utils.py
def get_interpret() -> bool:
    """Determine the ``interpret`` flag for ``pallas_call``.

    Reads the environment variable ``PALLAS_INTERPRET``.  When set to
    ``"1"`` or ``"true"`` (case-insensitive) interpret mode is enabled;
    every other value (including unset) disables it.
    """
    env = os.environ.get("PALLAS_INTERPRET", "")
    return env.strip().lower() in ("1", "true")


# From: tops/utils.py
def assert_shape_or_none(x: jax.Array | list[jax.Array | None] | tuple[jax.Array | None, ...] | None,
                         expected_shape: tuple[int, ...], name: str | list[str] | tuple[str, ...] = "tensor"):
    """
    Concise helper to assert tensor shapes.
    Skips assertion for any element that is None.
    Supports a single array or an iterable of arrays that should all match the expected shape.
    """
    if x is None:
        return

    if isinstance(x, (list, tuple)):
        has_names = isinstance(name, (list, tuple)) and len(name) == len(x)
        for i, tensor in enumerate(x):
            if tensor is not None:
                curr_name = name[i] if has_names else f"{name}_{i}"
                assert tensor.shape == expected_shape, f"[{curr_name}] Expected shape {expected_shape}, got {tensor.shape}"
    else:
        assert x.shape == expected_shape, f"[{name}] Expected shape {expected_shape}, got {x.shape}"

# From: tops/utils.py
def assert_shape(x: jax.Array | list[jax.Array] | tuple[jax.Array, ...],
                 expected_shape: tuple[int, ...], name: str | list[str] | tuple[str, ...] = "tensor"):
    """
    Concise helper to assert tensor shapes.
    Supports a single array or an iterable of arrays that should all match the expected shape.
    """
    if isinstance(x, (list, tuple)):
        # If name is not a sequence or has mismatched length, use a generic numbered name
        has_names = isinstance(name, (list, tuple)) and len(name) == len(x)
        for i, tensor in enumerate(x):
            curr_name = name[i] if has_names else f"{name}_{i}"
            assert tensor.shape == expected_shape, f"[{curr_name}] Expected shape {expected_shape}, got {tensor.shape}"
    else:
        assert x.shape == expected_shape, f"[{name}] Expected shape {expected_shape}, got {x.shape}"


# =============================================================================
# Forward: Fused chunk forward kernel (from common module)
# =============================================================================


# From: tops/ops/common/fused_chunk.py
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


# From: tops/ops/common/fused_chunk.py
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


# From: tops/ops/common/fused_chunk.py
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


# From: tops/ops/simple_gla/fused_chunk.py
def fused_chunk_simple_gla_fwd(
    q: jax.Array,                       # [B, T, H, K]
    k: jax.Array,                       # [B, T, H, K]
    v: jax.Array,                       # [B, T, H, V]
    *,
    g: jax.Array | None = None,         # [B, T, H]  chunk-local cumsum of scalar gate
    g_gamma: jax.Array | None = None,   # [H]         per-head fixed decay rate
    h0: jax.Array | None = None,        # [B, H, K, V] initial hidden state
    scale: float | None = None,
    use_ht: bool = False,
    chunk_size: int = 64,
    interpret: bool | None = None,
):
    return fused_chunk_fwd(
        q,
        k,
        v,
        g=g,
        g_gamma=g_gamma,
        h0=h0,
        scale=scale,
        use_ht=use_ht,
        chunk_size=chunk_size,
        interpret=interpret,
    )


# =============================================================================
# State propagation: chunk_fwd_h (needed by backward for h recomputation)
# =============================================================================


# From: tops/ops/common/chunk_h.py
def _chunk_fwd_h_kernel(
    k_ref,  # [1, 1, BT, BK]
    v_ref,  # [1, 1, BT, BV]
    h0_ref,  # [1, 1, BK, BV]
    gk_ref,  # [1, 1, BT, BK]
    g_ref,   # [1, 1, BT, 128]
    g_gamma,  # [H]
    h_ref,  # [1, NS, 1, BK, BV] outputs
    ht_ref,  # [1, 1, BK , BV]
    scratch_ref, #[BK, BV]
    *,
    BT,
    BS,
    NT,
):

    BK = k_ref.shape[3]
    BV = v_ref.shape[3]
    NTS = BS // BT
    T = NT * BT
    i_b, i_h, i_k, i_v, i_t = pl.program_id(0), pl.program_id(1), pl.program_id(2),pl.program_id(3),pl.program_id(4)

    if g_gamma is not None:
        b_g = g_gamma[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)

    @pl.when(i_t == 0)
    def init():
        if h0_ref is not None:
            scratch_ref[:,:] = h0_ref[0, 0].astype(jnp.float32)
        else:
            scratch_ref[:,:] = jnp.zeros((BK, BV), dtype=jnp.float32)

    @pl.when((i_t % NTS) == 0)
    def store_fn():
        i_s = i_t // NTS
        h_ref[0, i_s, 0] = scratch_ref[...].astype(h_ref.dtype)

    k_tile = k_ref[(0, 0, slice(None), slice(None))] # BT * BK
    v_tile = v_ref[(0, 0, slice(None), slice(None))] # BT * BV

    if g_ref is not None:
        b_g_scalar = g_ref[0, 0, slice(None), 0]  # [BT]
        b_g_scalar_last = b_g_scalar[BT - 1]       # scalar
        scratch_ref[...] *= exp(b_g_scalar_last)                 # uniform decay
        v_tile = (v_tile * exp(b_g_scalar_last - b_g_scalar)[:, None]).astype(v_tile.dtype)

    if g_gamma is not None:
        # tpu not support scalar bf16 mul
        b_g_last = (g_gamma[i_h].astype(jnp.float32) * jnp.minimum(BT, T - i_t * BT)).astype(g_gamma.dtype)
        scratch_ref[...] *= exp(b_g_last)
        v_tile = (v_tile * exp(b_g_last - b_g)[:, None]).astype(v_tile.dtype)


    if gk_ref is not None:
        gk_tile = gk_ref[(0, 0, slice(None), slice(None))] # BT * BK
        g_last = gk_tile[-1, :]
        decay = exp(g_last)
        scratch_ref[...] = scratch_ref[...] * decay[:, None]  # [BK, BV] * [BK,1]
        k_tile = (k_tile * exp(g_last[None, :] - gk_tile)).astype(k_tile.dtype)

    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
            k_tile.astype(jnp.float32).T,
            v_tile.astype(jnp.float32),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
    )

    @pl.when(i_t == NT - 1)
    def end():
        if ht_ref is not None:
            ht_ref[0, 0] = scratch_ref[...]


# From: tops/ops/common/chunk_h.py
# note: The precision difference between this kernel on the TPU and FLA on the GPU is 5e-2.
@functools.partial(
    jax.jit,
    static_argnames=[
        "output_final_state",
        "chunk_size",
        "split_size",
        "states_in_fp32",
    ],
)
def chunk_fwd_h_kernel(
    k: jax.Array,
    v: jax.Array,  # [B,T,H,V]
    *,
    g: jax.Array | None = None,  # [B,T,H]
    g_gamma: jax.Array | None = None,  # (H,)
    gk: jax.Array | None = None,  # [B,T,H,K]
    gv: jax.Array | None = None,  # [B,T,H,V]
    h0: jax.Array | None = None,  # [N,H,K,V]
    output_final_state: bool = False,
    cu_seqlens_cpu: jax.Array | None = None,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 64,
    split_size: int | None = None,
    states_in_fp32: bool = False,
):
    # todo: tune bk and bv for bast performance
    BK = 128
    BV = 128
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens_cpu is None else cu_seqlens_cpu.shape[-1] - 1
    BT = chunk_size
    BS = BT if split_size is None else split_size

    # =================== assert kernel requirements start ===================
    assert_shape(k, (B, T, H, K))
    assert_shape(v, (B, T, H, V))
    assert_shape_or_none(g, (B, T, H))
    assert_shape_or_none(g_gamma, (H,))
    assert_shape_or_none(gk, (B, T, H, K))
    # assert_shape_or_none(gv, (B, T, H, V))
    assert gv is None, "gv is currently not supported"
    assert cu_seqlens_cpu is None, "cu_seqlens_cpu is currently not supported"
    assert cu_seqlens_dev is None, "cu_seqlens_dev is currently not supported"
    assert_shape_or_none(h0, (N, H, K, V))

    assert K % 128 == 0, "K % 128 must equal to 0."
    assert V % 128 == 0, "V % 128 must equal to 0."
    assert T % chunk_size == 0, "T mod chunk_size must equal to 0."
    if cu_seqlens_cpu is not None:
        assert cu_seqlens_cpu[0] == 0, "cu_seqlens_cpu must start with 0."
        assert (cu_seqlens_cpu % chunk_size == 0).all(), "cu_seqlens_cpu must be multiples of chunk_size."

    assert BS % BT == 0, (
        f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"
    )
    # =================== assert kernel requirements done ===================

    # N: the actual number of sequences in the batch with either equal or variable lengths

    N, NS = (
        B,
        T // BS,
    )  # split_offsets[-1] # NS number of chunk_size
    NT = T // BT

    k = jnp.transpose(k, (0, 2, 1, 3))  # (B,H,T,K)
    v = jnp.transpose(v, (0, 2, 1, 3))  # (B,H,T,V)
    if gk is not None:
        gk = jnp.transpose(gk, (0, 2, 1, 3))  # (B,H,T,K)

    if g is not None:
        g = jnp.transpose(g, (0, 2, 1))  # (B, H, T)
        g = jnp.broadcast_to(g[:, :, :, None], (B, H, T, 128))  # (B, H, T, 128)

    grid = (B, H, pl.cdiv(K, BK), pl.cdiv(V, BV), NT)

    def k_index_map(batch_index, head_index, k_index, _, t_index):
        return batch_index, head_index, t_index, k_index

    def gk_index_map(batch_index, head_index,  k_index, _, t_index):
        return batch_index, head_index, t_index, k_index

    def g_index_map(batch_index, head_index,  k_index, _, t_index):
        return batch_index, head_index, t_index, 0

    def v_index_map(batch_index, head_index, _, v_index, t_index):
        return batch_index, head_index, t_index, v_index

    def h0_index_map(batch_index, head_index, k_index, v_index, _):
        return batch_index, head_index, k_index, v_index

    def h_index_map(batch_index, head_index, k_index, v_index, _):
        return batch_index, 0, head_index, k_index, v_index

    def ht_index_map(batch_index, head_index, k_index, v_index, _):
        return batch_index, head_index, k_index, v_index


    out_shape = [
        jax.ShapeDtypeStruct(
            shape=(N, NS, H, K, V), dtype=k.dtype if not states_in_fp32 else jnp.float32
        )
    ]
    out_specs = [pl.BlockSpec((1, NS, 1, BK, BV), h_index_map)]
    if output_final_state:
        out_shape.append(jax.ShapeDtypeStruct(shape=(N, H, K, V), dtype=jnp.float32))
        out_specs.append(pl.BlockSpec((1, 1, BK, BV), ht_index_map))
    else:
        out_shape.append(None)
        out_specs.append(None)

    in_specs = [
        pl.BlockSpec((1, 1, BT, BK), k_index_map),
        pl.BlockSpec((1, 1, BT, BV), v_index_map),
    ]
    scratch = pltpu.VMEM((BK, BV), jnp.float32)
    scratch_shapes = [scratch]
    if h0 is not None:
        in_specs.append(pl.BlockSpec((1, 1, BK, BV), h0_index_map))
    else:
        in_specs.append(None)
    if gk is not None:
        in_specs.append(pl.BlockSpec((1, 1, BT, BK), gk_index_map))
    else:
        in_specs.append(None)

    if g is not None:
        in_specs.append(pl.BlockSpec((1, 1, BT, 128), g_index_map))
    else:
        in_specs.append(None)

    if g_gamma is not None:
        in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    else:
        in_specs.append(None)

    kernel = functools.partial(
        _chunk_fwd_h_kernel,
        BT=BT,
        BS=BS,
        NT=NT,
    )
    interpret = get_interpret()
    h, ht = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes
        ),
        out_shape=out_shape,
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel",
                "parallel",
                "parallel",
                "parallel",
                "arbitrary",
            ),
            # vmem_limit_bytes=32 * 1024 * 1024,
            disable_bounds_checks=True,
        ),
    )(k, v, h0, gk, g, g_gamma)

    h = h.reshape(B, -1, H, K, V)
    ht = ht.reshape(N, H, K, V) if ht is not None else None

    if output_final_state:
        return h, ht
    return h, None

# Alias used by fused_chunk_simple_gla_bwd
chunk_fwd_h = chunk_fwd_h_kernel


# =============================================================================
# Backward: Fused dh propagation + dq/dk/dv kernel
# =============================================================================


# From: tops/ops/simple_gla/fused_chunk.py
def _fused_chunk_bwd_kernel(
    q_ref,          # (1, 1, BT, K)    — reversed chunk order
    k_ref,          # (1, 1, BT, K)    — reversed chunk order
    v_ref,          # (1, 1, BT, V)    — reversed chunk order
    h_ref,          # (1, 1, 1, K, V)  — from forward, reversed chunk order
    do_ref,         # (1, 1, BT, V)    — reversed chunk order
    g_gamma_ref,    # [H] via SMEM/ANY
    dht_ref,        # (1, 1, K, V) or None — terminal state gradient
    # --- outputs ---
    dq_ref,         # (1, 1, BT, K)    — reversed chunk order
    dk_ref,         # (1, 1, BT, K)    — reversed chunk order
    dv_ref,         # (1, 1, BT, V)    — reversed chunk order
    dh0_ref,        # (1, 1, K, V) or None — initial state gradient
    # --- scratch ---
    scratch_ref,    # (K, V) VMEM float32 — carries dh across chunks
    *,
    BT: int,
    NT: int,
    scale: float,
):
    """Fused backward kernel: dh propagation + dq/dk/dv computation.

    Merges ``chunk_bwd_dh`` and ``chunk_simple_gla_bwd_o`` so that the
    full dh tensor [B, NT, H, K, V] never materialises in HBM.

    Grid: (B, H, NT) — B, H parallel, NT arbitrary (backward).
    Iteration: i_t = 0 processes the *last* chunk (NT-1), i_t = NT-1
    processes chunk 0.  BlockSpec index maps reverse the chunk order.

    At each step:
      1. Load dh from VMEM scratch (gradient flowing from future chunks).
      2. Compute dq, dk, dv using both h (from HBM) and dh (from scratch).
      3. Update dh for the previous chunk and save to scratch.
    """
    K = q_ref.shape[3]
    V = v_ref.shape[3]
    i_t = pl.program_id(2)       # 0 → last chunk, NT-1 → first chunk
    head_idx = pl.program_id(1)

    # ---- Initialize dh on first backward step ----
    @pl.when(i_t == 0)
    def _init():
        if dht_ref is not None:
            scratch_ref[:, :] = dht_ref[0, 0].astype(jnp.float32)
        else:
            scratch_ref[:, :] = jnp.zeros((K, V), dtype=jnp.float32)

    # ---- Load data ----
    b_q = q_ref[0, 0]                              # (BT, K)
    b_k = k_ref[0, 0]                              # (BT, K)
    b_v = v_ref[0, 0]                              # (BT, V)
    b_h = h_ref[0, 0, 0].astype(jnp.float32)       # (K, V)
    b_do = do_ref[0, 0]                             # (BT, V)
    b_dh = scratch_ref[...].astype(jnp.float32)     # (K, V)

    # ---- Per-position decay from g_gamma ----
    b_gamma = g_gamma_ref[head_idx].astype(jnp.float32)
    b_g = b_gamma * (jnp.arange(BT) + 1).astype(jnp.float32)   # [BT]
    b_gn = b_g[BT - 1]                                           # g_gamma * BT

    # ---- Recompute A in-kernel ----
    pos = (jnp.arange(BT) + 1).astype(jnp.float32)
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    safe_diff = jnp.where(mask, b_gamma * (pos[:, None] - pos[None, :]), 0.0)
    decay = jnp.exp(safe_diff)
    b_a = jnp.dot(
        b_q, b_k.T,
        preferred_element_type=jnp.float32,
    ) * scale * decay

    # ---- dA = (do @ v^T) * scale, lower-triangular ----
    b_dA = jnp.dot(
        b_do, b_v.T,
        preferred_element_type=jnp.float32,
    ) * scale
    b_dA = jnp.where(mask, b_dA, 0.0)

    # ---- dv = A^T @ do  +  k_decay @ dh ----
    b_a_masked = jnp.where(mask, b_a, 0.0).astype(b_do.dtype)
    b_dv_intra = jnp.dot(
        b_a_masked.T, b_do,
        preferred_element_type=jnp.float32,
    )
    k_decay = (b_k * jnp.exp(b_gn - b_g)[:, None]).astype(b_k.dtype)
    b_dv_inter = jnp.dot(
        k_decay, b_dh.astype(b_k.dtype),
        preferred_element_type=jnp.float32,
    )
    dv_ref[0, 0] = (b_dv_intra + b_dv_inter).astype(dv_ref.dtype)

    # ---- Gated dA: dA * exp(g_i - g_j) ----
    safe_g_diff = jnp.where(mask, b_g[:, None] - b_g[None, :], 0.0)
    b_dA_gated = b_dA * jnp.exp(safe_g_diff)

    # ---- dq = gated_dA @ k  +  do @ h^T * scale * exp(g) ----
    b_dq_intra = jnp.dot(
        b_dA_gated.astype(b_k.dtype), b_k,
        preferred_element_type=jnp.float32,
    )
    b_dq_inter = jnp.dot(
        b_do, b_h.astype(b_do.dtype).T,
        preferred_element_type=jnp.float32,
    ) * (scale * jnp.exp(b_g)[:, None])
    dq_ref[0, 0] = (b_dq_intra + b_dq_inter).astype(dq_ref.dtype)

    # ---- dk = gated_dA^T @ q  +  v @ dh^T * exp(g_last - g) ----
    b_dk_intra = jnp.dot(
        b_dA_gated.T.astype(b_q.dtype), b_q,
        preferred_element_type=jnp.float32,
    )
    b_dk_inter = jnp.dot(
        b_v, b_dh.astype(b_v.dtype).T,
        preferred_element_type=jnp.float32,
    ) * jnp.exp(b_gn - b_g)[:, None]
    dk_ref[0, 0] = (b_dk_intra + b_dk_inter).astype(dk_ref.dtype)

    # ---- Update dh for previous chunk ----
    # Recurrence: dh_n = dh_{n+1} * exp(g_gamma * BT) + (q*scale*exp(g))^T @ do
    b_dh = b_dh * jnp.exp(b_gn)
    b_q_hat = (b_q * (scale * jnp.exp(b_g)[:, None])).astype(jnp.float32)
    b_dh = b_dh + jnp.dot(
        b_q_hat.T, b_do.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    scratch_ref[...] = b_dh

    # ---- Store dh0 at last backward step (chunk 0) ----
    @pl.when(i_t == NT - 1)
    def _store_dh0():
        if dh0_ref is not None:
            dh0_ref[0, 0] = scratch_ref[...].astype(dh0_ref.dtype)


# From: tops/ops/simple_gla/fused_chunk.py
@functools.partial(
    jax.jit,
    static_argnames=("scale", "output_dh0", "chunk_size", "interpret"),
)
def _fused_chunk_bwd_launcher(
    q: jax.Array,           # [B, T, H, K]
    k: jax.Array,           # [B, T, H, K]
    v: jax.Array,           # [B, T, H, V]
    h: jax.Array,           # [B, NT, H, K, V]
    do: jax.Array,          # [B, T, H, V]
    g_gamma: jax.Array,     # [H]
    dht: jax.Array | None,  # [B, H, K, V]
    *,
    scale: float,
    output_dh0: bool,
    chunk_size: int,
    interpret: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
    """Pallas launcher for the fused backward kernel.

    Transposes inputs to (B, H, T, D) layout, launches the kernel with
    reversed chunk order, and transposes results back to (B, T, H, D).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT

    # Transpose to (B, H, ...) layout
    _q = q.transpose(0, 2, 1, 3)        # (B, H, T, K)
    _k = k.transpose(0, 2, 1, 3)        # (B, H, T, K)
    _v = v.transpose(0, 2, 1, 3)        # (B, H, T, V)
    _do = do.transpose(0, 2, 1, 3)      # (B, H, T, V)
    _h = h.transpose(0, 2, 1, 3, 4)     # (B, H, NT, K, V)

    grid = (B, H, NT)

    # Reversed chunk order for backward traversal
    def rev_qk_map(b, h, t):
        return (b, h, NT - 1 - t, 0)

    def rev_v_map(b, h, t):
        return (b, h, NT - 1 - t, 0)

    def rev_h_map(b, h, t):
        return (b, h, NT - 1 - t, 0, 0)

    def state_map(b, h, t):
        return (b, h, 0, 0)

    smem = pltpu.ANY if interpret else pltpu.SMEM

    in_specs = [
        pl.BlockSpec((1, 1, BT, K), rev_qk_map),                            # q
        pl.BlockSpec((1, 1, BT, K), rev_qk_map),                            # k
        pl.BlockSpec((1, 1, BT, V), rev_v_map),                             # v
        pl.BlockSpec((1, 1, 1, K, V), rev_h_map),                           # h
        pl.BlockSpec((1, 1, BT, V), rev_v_map),                             # do
        pl.BlockSpec(memory_space=smem),                                     # g_gamma
        pl.BlockSpec((1, 1, K, V), state_map) if dht is not None else None,  # dht
    ]

    out_specs = [
        pl.BlockSpec((1, 1, BT, K), rev_qk_map),                            # dq
        pl.BlockSpec((1, 1, BT, K), rev_qk_map),                            # dk
        pl.BlockSpec((1, 1, BT, V), rev_v_map),                             # dv
        pl.BlockSpec((1, 1, K, V), state_map) if output_dh0 else None,       # dh0
    ]

    out_shapes = [
        jax.ShapeDtypeStruct((B, H, T, K), q.dtype),                        # dq
        jax.ShapeDtypeStruct((B, H, T, K), k.dtype),                        # dk
        jax.ShapeDtypeStruct((B, H, T, V), v.dtype),                        # dv
        jax.ShapeDtypeStruct((B, H, K, V), jnp.float32)                     # dh0
        if output_dh0 else None,
    ]

    dq, dk, dv, dh0 = pl.pallas_call(
        functools.partial(_fused_chunk_bwd_kernel, BT=BT, NT=NT, scale=scale),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=[pltpu.VMEM((K, V), jnp.float32)],
        ),
        out_shape=out_shapes,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
        interpret=interpret,
    )(_q, _k, _v, _h, _do, g_gamma, dht)

    # Transpose back: (B, H, T, D) -> (B, T, H, D)
    dq = dq.transpose(0, 2, 1, 3)
    dk = dk.transpose(0, 2, 1, 3)
    dv = dv.transpose(0, 2, 1, 3)

    return dq, dk, dv, dh0


# From: tops/ops/simple_gla/fused_chunk.py
def fused_chunk_simple_gla_bwd(
    q: jax.Array,                        # [B, T, H, K]
    k: jax.Array,                        # [B, T, H, K]
    v: jax.Array,                        # [B, T, H, V]
    do: jax.Array,                       # [B, T, H, V]
    *,
    g: jax.Array | None = None,          # [B, T, H]  (must be None)
    g_gamma: jax.Array | None = None,    # [H]  per-head fixed decay rate
    h0: jax.Array | None = None,         # [B, H, K, V] initial hidden state
    dht: jax.Array | None = None,        # [B, H, K, V] terminal state gradient
    scale: float | None = None,
    chunk_size: int = 64,
    interpret: bool | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
    """Fused chunk backward for simple GLA (g_gamma only).

    Merges hidden-state gradient propagation (``chunk_bwd_dh``) and per-chunk
    gradient computation (``chunk_simple_gla_bwd_o``) into a single Pallas
    kernel.  The dh tensor [B, NT, H, K, V] stays in VMEM scratch, avoiding
    HBM materialisation.

    The forward hidden states h are recomputed via ``chunk_fwd_h``.

    Args:
        q:  [B, T, H, K] -- queries.
        k:  [B, T, H, K] -- keys.
        v:  [B, T, H, V] -- values.
        do: [B, T, H, V] -- output gradient.
        g:  must be None (per-element gating not supported).
        g_gamma: [H] -- per-head fixed decay rate.
        h0: [B, H, K, V] -- initial hidden state (optional).
        dht: [B, H, K, V] -- terminal state gradient (optional).
        scale: attention scale (default K ** -0.5).
        chunk_size: block size.  T must be divisible by chunk_size.
        interpret: Pallas interpret mode.  None = auto-detect.

    Returns:
        (dq, dk, dv, dh0)
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    if scale is None:
        scale = K ** -0.5

    # =================== assert kernel requirements start ===================
    assert_shape(q, (B, T, H, K))
    assert_shape(k, (B, T, H, K))
    assert_shape(v, (B, T, H, V))
    assert_shape(do, (B, T, H, V))
    assert g is None, "per-element gating not supported in fused_chunk_simple_gla_bwd"
    assert g_gamma is not None, "g_gamma is required for fused_chunk_simple_gla_bwd"
    assert_shape(g_gamma, (H,))
    assert_shape_or_none(h0, (B, H, K, V))
    assert_shape_or_none(dht, (B, H, K, V))
    assert K % 128 == 0, f"K={K} must be a multiple of 128"
    assert V % 128 == 0, f"V={V} must be a multiple of 128"
    assert T % C == 0, f"T={T} must be divisible by chunk_size={C}"
    # =================== assert kernel requirements done ===================

    if interpret is None:
        interpret = not is_tpu_runtime()

    # 1. Recompute forward hidden states h
    h, _ = chunk_fwd_h(
        k=k,
        v=v,
        g=None,
        g_gamma=g_gamma,
        gk=None,
        h0=h0,
        output_final_state=False,
        chunk_size=C,
        states_in_fp32=True,
    )

    # 2. Fused dh + dq/dk/dv backward kernel
    dq, dk, dv, dh0 = _fused_chunk_bwd_launcher(
        q, k, v, h, do, g_gamma, dht,
        scale=scale,
        output_dh0=h0 is not None,
        chunk_size=C,
        interpret=interpret,
    )

    return dq, dk, dv, dh0


# =============================================================================
# Public API for evaluate.py
# =============================================================================


def _make_test_data(B, T, H, K, V, chunk_size, seed=42):
    """Create deterministic test data for the fused_chunk_simple_gla kernel."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    q = jax.random.normal(k1, (B, T, H, K), dtype=jnp.bfloat16)
    k_arr = jax.random.normal(k2, (B, T, H, K), dtype=jnp.bfloat16)
    v = jax.random.normal(k3, (B, T, H, V), dtype=jnp.bfloat16)
    g_gamma = -jnp.abs(jax.random.normal(k4, (H,), dtype=jnp.float32)) * 0.1
    return q, k_arr, v, g_gamma


def simple_compute(B=2, T=4096, H=16, K=128, V=128, chunk_size=64):
    """Forward + backward fused_chunk_simple_gla (reference, no recomputation).

    Returns the loss scalar. Both forward and backward are computed,
    so timing captures the full training step performance.
    """
    q, k_arr, v, g_gamma = _make_test_data(B, T, H, K, V, chunk_size)
    scale = K ** -0.5

    @jax.custom_vjp
    def _compute(q, k, v):
        o, _ = fused_chunk_simple_gla_fwd(
            q, k, v, g_gamma=g_gamma, scale=scale, chunk_size=chunk_size,
        )
        return o

    def _fwd(q, k, v):
        o, _ = fused_chunk_simple_gla_fwd(
            q, k, v, g_gamma=g_gamma, scale=scale, chunk_size=chunk_size,
        )
        # Pre-compute h for backward (trade memory for compute — no recomputation)
        h, _ = chunk_fwd_h(
            k=k, v=v, g=None, g_gamma=g_gamma, gk=None,
            h0=None, output_final_state=False,
            chunk_size=chunk_size, states_in_fp32=True,
        )
        return o, (q, k, v, h)

    def _bwd(residuals, do):
        q, k, v, h = residuals
        B, T, H, K = q.shape
        C = chunk_size
        s = scale if scale is not None else K ** -0.5
        interp = not is_tpu_runtime()

        # Use pre-computed h — skip recomputation
        dq, dk, dv, _ = _fused_chunk_bwd_launcher(
            q, k, v, h, do, g_gamma, None,
            scale=s, output_dh0=False,
            chunk_size=C, interpret=interp,
        )
        return dq, dk, dv

    _compute.defvjp(_fwd, _bwd)

    def loss_fn(q, k, v):
        return _compute(
            q.astype(jnp.float32), k.astype(jnp.float32),
            v.astype(jnp.float32),
        ).sum()

    loss, _ = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k_arr, v)
    return loss


def reference_fn(**kwargs):
    """Generic entry point for evaluate.py function discovery."""
    return simple_compute(**kwargs)
