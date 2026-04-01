"""Fused chunk kernels for Simple GLA.

Forward: delegates to ``fused_chunk_fwd`` (common module).
Backward: fuses ``chunk_bwd_dh`` and ``chunk_simple_gla_bwd_o`` into a
single Pallas kernel so that the full dh tensor [B, NT, H, K, V] never
materialises in HBM.  The hidden-state gradient is carried across chunks
in VMEM scratch, analogous to the forward fusion for h.
"""

import functools

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

from tops.ops.common.chunk_h import chunk_fwd_h_kernel as chunk_fwd_h
from tops.ops.common.fused_chunk import fused_chunk_fwd
from tops.ops.utils import is_tpu_runtime
from tops.utils import assert_shape, assert_shape_or_none


# =========================================================================
# Forward (delegates to common fused kernel)
# =========================================================================


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


# =========================================================================
# Fused backward: dh propagation + dq/dk/dv in a single Pallas kernel
# =========================================================================


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


# =========================================================================
# Pallas launcher
# =========================================================================


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


# =========================================================================
# Public API
# =========================================================================


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
