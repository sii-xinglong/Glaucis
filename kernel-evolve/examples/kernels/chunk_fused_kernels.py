# Source: primatrix/pallas-kernel @ branch: feat/chunk-gla-fused-kernels
# Commit: 1de541558ce12b8fc6c439a85f020422a4ba2c6a
# Initialized: 2026-04-01
"""Fused chunk-GLA Pallas TPU kernel — template for evolutionary optimization.

Merges three separate pallas_calls (h propagation + A recomputation + output
computation) into a single pallas_call for both forward and backward passes.
Uses g_gamma (per-head constant gate) mode.

Eliminates:
  1. Two kernel launch overheads
  2. The h tensor HBM round-trip (h is produced and consumed in VMEM)
  3. The A tensor HBM round-trip (A is recomputed inline from q, k, g)
  4. The g_cumsum tensor entirely (gating computed from g_gamma scalar)

Optimization targets within the EVOLVE-BLOCK:
  - Kernel body compute ordering and pipelining
  - Precision choices for intermediate dot products
  - Scratch memory layout
  - Grid dimension ordering
  - Forward/backward kernel fusion strategies
  - Block sizes (currently BK=128, BV=128 fixed to match K,V dims)

Reference dimensions from upstream tests:
  q, k: [B, T, H, 128]   v: [B, T, H, 128]   g_gamma: [H]
  Default: B=2, T=256, H=4, K=128, V=128, chunk_size=64
"""

import functools

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _make_test_data(B, T, H, K, V, chunk_size, seed=42):
    """Create deterministic (q, k, v, g_gamma) for a fused GLA test case."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    q = jax.random.normal(k1, (B, T, H, K), dtype=jnp.bfloat16)
    k_arr = jax.random.normal(k2, (B, T, H, K), dtype=jnp.bfloat16)
    v = jax.random.normal(k3, (B, T, H, V), dtype=jnp.bfloat16)
    g_gamma = -jnp.abs(jax.random.normal(k4, (H,), dtype=jnp.float32)) * 0.1
    return q, k_arr, v, g_gamma


def exp(x):
    """exp in float32."""
    return jnp.exp(x.astype(jnp.float32))


# EVOLVE-BLOCK-START
# ============================================================
# Forward: Fused kernel (h propagation + A recompute + output)
# ============================================================


def _chunk_fwd_fused_kernel(
    q_ref,
    k_ref,
    v_ref,
    g_gamma,
    h_ref,
    o_ref,
    scratch_ref,
    *,
    BT,
    NT,
    scale,
):
    """Fused forward kernel: h propagation + A recomputation + output.

    At each time step t:
      1. Save h[t] to output (for backward residuals)
      2. Compute o[t] = q_gated @ h[t] * scale + A_masked @ v
         where A is recomputed inline from q, k, g_gamma
      3. Update h[t+1] = h[t] * decay + k.T @ (v * gating)
    """
    BK = k_ref.shape[3]
    BV = v_ref.shape[3]
    i_b, i_h, i_k, i_v, i_t = (
        pl.program_id(0),
        pl.program_id(1),
        pl.program_id(2),
        pl.program_id(3),
        pl.program_id(4),
    )

    # Per-position gating: g_gamma * (1, 2, ..., BT)
    b_g_ramp = g_gamma[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)
    # State decay for one full chunk
    b_g_last = g_gamma[i_h].astype(jnp.float32) * BT

    # --- Initialize h state in scratch at t=0 ---
    @pl.when(i_t == 0)
    def init():
        scratch_ref[:, :] = jnp.zeros((BK, BV), dtype=jnp.float32)

    # --- Step 1: Save current h to output (for backward residuals) ---
    h_ref[0, i_t, 0] = scratch_ref[...].astype(h_ref.dtype)

    # --- Step 2: Compute output o[t] ---
    b_q = q_ref[0, 0]   # [BT, BK]
    b_k = k_ref[0, 0]   # [BT, BK]
    b_v = v_ref[0, 0]   # [BT, BV]

    # Compute gated q and k for A recomputation
    exp_g = exp(b_g_ramp)       # [BT]
    exp_neg_g = exp(-b_g_ramp)  # [BT]
    b_qg = (b_q.astype(jnp.float32) * exp_g[:, None])       # [BT, BK] f32
    b_kg = (b_k.astype(jnp.float32) * exp_neg_g[:, None])   # [BT, BK] f32

    # Recompute A = b_qg @ b_kg.T * scale  [BT, BT]
    b_A = (
        jnp.dot(
            b_qg,
            b_kg.T,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * scale
    )

    # Causal mask
    m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_A_masked = jnp.where(m_s, b_A, 0.0)

    # Inter-chunk: b_qg @ h * scale  [BT, BV]
    b_o_inter = (
        jnp.dot(
            b_qg,
            scratch_ref[...],
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * scale
    )

    # Intra-chunk: A_masked @ v  [BT, BV]
    b_o_intra = jnp.dot(
        b_A_masked,
        b_v.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    o_ref[0, 0] = (b_o_inter + b_o_intra).astype(o_ref.dtype)

    # --- Step 3: Update h for next time step ---
    scratch_ref[...] *= exp(b_g_last)

    v_gated = (b_v * exp(b_g_last - b_g_ramp)[:, None]).astype(b_v.dtype)

    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        b_k.astype(jnp.float32).T,
        v_gated.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


@functools.partial(jax.jit, static_argnames=["chunk_size", "scale"])
def chunk_fwd_fused_g_gamma(q, k, v, g_gamma, scale, chunk_size):
    """Fused chunked GLA forward pass for g_gamma mode.

    Args:
        q:          [B, T, H, K] queries in bfloat16.
        k:          [B, T, H, K] keys in bfloat16.
        v:          [B, T, H, V] values in bfloat16.
        g_gamma:    [H] per-head constant log-space gate in float32.
        scale:      Scaling factor, typically K**-0.5.
        chunk_size: Block size along the time dimension.

    Returns:
        h: [B, NT, H, K, V] hidden states at each chunk boundary (bfloat16).
        o: [B, T, H, V] output (bfloat16).
    """
    BK, BV, BT = 128, 128, chunk_size
    B, T, H, K_dim = q.shape
    V = v.shape[-1]
    NT = T // BT

    assert T % BT == 0, f"T ({T}) must be a multiple of chunk_size ({BT})"
    assert K_dim == BK, f"K must be {BK}, got {K_dim}"
    assert V == BV, f"V must be {BV}, got {V}"

    q_t = jnp.transpose(q, (0, 2, 1, 3))
    k_t = jnp.transpose(k, (0, 2, 1, 3))
    v_t = jnp.transpose(v, (0, 2, 1, 3))

    grid = (B, H, 1, 1, NT)

    def q_map(b, h, ki, vi, t): return b, h, t, ki
    def k_map(b, h, ki, vi, t): return b, h, t, ki
    def v_map(b, h, ki, vi, t): return b, h, t, vi
    def h_map(b, h, ki, vi, t): return b, 0, h, ki, vi
    def o_map(b, h, ki, vi, t): return b, h, t, vi

    h_all, o_t = pl.pallas_call(
        functools.partial(_chunk_fwd_fused_kernel, BT=BT, NT=NT, scale=scale),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, 1, BT, BK), q_map),
                pl.BlockSpec((1, 1, BT, BK), k_map),
                pl.BlockSpec((1, 1, BT, BV), v_map),
                pl.BlockSpec(memory_space=pltpu.SMEM),
            ],
            out_specs=[
                pl.BlockSpec((1, NT, 1, BK, BV), h_map),
                pl.BlockSpec((1, 1, BT, BV), o_map),
            ],
            scratch_shapes=[pltpu.VMEM((BK, BV), jnp.float32)],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((B, NT, H, K_dim, V), q.dtype),
            jax.ShapeDtypeStruct((B, H, T, V), q.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel", "parallel", "parallel", "parallel", "arbitrary",
            ),
            disable_bounds_checks=True,
        ),
    )(q_t, k_t, v_t, g_gamma)

    o = jnp.transpose(o_t, (0, 2, 1, 3))
    return h_all, o


# ============================================================
# Backward: Fused dh reverse propagation + dq/dk/dv
# ============================================================


def _chunk_bwd_fused_kernel(
    q_ref,
    k_ref,
    v_ref,
    h_ref,
    do_ref,
    g_gamma,
    dq_ref,
    dk_ref,
    dv_ref,
    scratch_ref,
    *,
    BT,
    NT,
    scale,
):
    """Fused backward kernel with g_cumsum eliminated.

    Processes chunks in REVERSE time order via reverse index_maps.
    """
    BK = q_ref.shape[3]
    BV = do_ref.shape[3]
    i_b, i_h, i_k, i_v, i_t = (
        pl.program_id(0),
        pl.program_id(1),
        pl.program_id(2),
        pl.program_id(3),
        pl.program_id(4),
    )

    b_g_ramp = g_gamma[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)
    b_g_last = g_gamma[i_h].astype(jnp.float32) * BT

    @pl.when(i_t == 0)
    def init():
        scratch_ref[:, :] = jnp.zeros((BK, BV), dtype=jnp.float32)

    b_q = q_ref[0, 0]
    b_k = k_ref[0, 0]
    b_v = v_ref[0, 0]
    b_h = h_ref[0, 0, 0].astype(jnp.float32)
    b_do = do_ref[0, 0]
    b_dh = scratch_ref[...]

    # Phase 0: Pre-compute exp/gate values
    exp_pos = exp(b_g_ramp)
    exp_neg = exp(-b_g_ramp)
    exp_gn_minus = exp(b_g_last - b_g_ramp)

    k_neg = (b_k.astype(jnp.float32) * exp_neg[:, None])
    k_decay = (b_k.astype(jnp.float32) * exp_gn_minus[:, None])
    q_pos = (b_q.astype(jnp.float32) * exp_pos[:, None])

    # Phase 1: Recompute A and compute dA
    b_a = (
        jnp.dot(
            q_pos, k_neg.T,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ) * scale
    )

    b_dA_raw = (
        jnp.dot(
            b_do.astype(jnp.float32), b_v.astype(jnp.float32).T,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ) * scale
    )

    # Phase 2: Apply causal masks
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_dA = jnp.where(mask, b_dA_raw, 0.0)
    b_a_masked = jnp.where(mask, b_a, 0.0)

    # Phase 3: Four independent dot products
    b_dv_intra = jnp.dot(
        b_a_masked.T, b_do.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b_dv_inter = jnp.dot(
        k_decay, b_dh,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b_dq_inter = jnp.dot(
        b_do.astype(jnp.float32), b_h.T,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b_dk_inter = jnp.dot(
        b_v.astype(jnp.float32), b_dh.T,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # Phase 4: Intra-chunk dq and dk
    b_dq_intra_raw = jnp.dot(
        b_dA, k_neg,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b_dk_intra_raw = jnp.dot(
        b_dA.T, q_pos,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # Phase 5: Combine and write outputs at NT-1-i_t (forward time order)
    i_out = NT - 1 - i_t
    dv_ref[0, i_out, 0] = (b_dv_intra + b_dv_inter).astype(dv_ref.dtype)

    b_dq = b_dq_intra_raw * exp_pos[:, None] + b_dq_inter * (scale * exp_pos[:, None])
    dq_ref[0, i_out, 0] = b_dq.astype(dq_ref.dtype)

    b_dk = b_dk_intra_raw * exp_neg[:, None] + b_dk_inter * exp_gn_minus[:, None]
    dk_ref[0, i_out, 0] = b_dk.astype(dk_ref.dtype)

    # Phase 6: Update dh state for next reverse step
    scratch_ref[...] *= exp(b_g_last)

    q_hat = (b_q * exp(b_g_ramp)[:, None] * scale).astype(jnp.float32)

    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        q_hat.T,
        b_do.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


@functools.partial(jax.jit, static_argnames=["chunk_size", "scale"])
def chunk_bwd_fused_g_gamma(q, k, v, h, do, g_gamma, scale, chunk_size):
    """Fused chunked GLA backward pass for g_gamma mode.

    Args:
        q:          [B, T, H, K] queries in bfloat16.
        k:          [B, T, H, K] keys in bfloat16.
        v:          [B, T, H, V] values in bfloat16.
        h:          [B, NT, H, K, V] hidden states from forward pass.
        do:         [B, T, H, V] upstream output gradients in bfloat16.
        g_gamma:    [H] per-head constant log-space gate in float32.
        scale:      Scaling factor, typically K**-0.5.
        chunk_size: Block size along the time dimension.

    Returns:
        dq: [B, T, H, K] query gradients (bfloat16).
        dk: [B, T, H, K] key gradients (bfloat16).
        dv: [B, T, H, V] value gradients (bfloat16).
    """
    BK, BV, BT = 128, 128, chunk_size
    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = T // BT

    assert T % BT == 0
    assert K == BK
    assert V == BV

    q_t = jnp.transpose(q, (0, 2, 1, 3))
    k_t = jnp.transpose(k, (0, 2, 1, 3))
    v_t = jnp.transpose(v, (0, 2, 1, 3))
    do_t = jnp.transpose(do, (0, 2, 1, 3))
    h_bhntKV = jnp.transpose(h, (0, 2, 1, 3, 4))

    grid = (B, H, 1, 1, NT)

    # Reverse input index maps
    def q_map(b, h, ki, vi, t): return b, h, NT - 1 - t, ki
    def k_map(b, h, ki, vi, t): return b, h, NT - 1 - t, ki
    def v_map(b, h, ki, vi, t): return b, h, NT - 1 - t, vi
    def h_map(b, h, ki, vi, t): return b, h, NT - 1 - t, 0, 0
    def do_map(b, h, ki, vi, t): return b, h, NT - 1 - t, vi

    def out_k_map(b, h, ki, vi, t): return b, 0, h, 0, 0
    def out_v_map(b, h, ki, vi, t): return b, 0, h, 0, 0

    dq_5d, dk_5d, dv_5d = pl.pallas_call(
        functools.partial(_chunk_bwd_fused_kernel, BT=BT, NT=NT, scale=scale),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, 1, BT, BK), q_map),
                pl.BlockSpec((1, 1, BT, BK), k_map),
                pl.BlockSpec((1, 1, BT, BV), v_map),
                pl.BlockSpec((1, 1, 1, BK, BV), h_map),
                pl.BlockSpec((1, 1, BT, BV), do_map),
                pl.BlockSpec(memory_space=pltpu.SMEM),
            ],
            out_specs=[
                pl.BlockSpec((1, NT, 1, BT, BK), out_k_map),
                pl.BlockSpec((1, NT, 1, BT, BK), out_k_map),
                pl.BlockSpec((1, NT, 1, BT, BV), out_v_map),
            ],
            scratch_shapes=[pltpu.VMEM((BK, BV), jnp.float32)],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((B, NT, H, BT, K), q.dtype),
            jax.ShapeDtypeStruct((B, NT, H, BT, K), k.dtype),
            jax.ShapeDtypeStruct((B, NT, H, BT, V), v.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel", "parallel", "parallel", "parallel", "arbitrary",
            ),
            disable_bounds_checks=True,
        ),
    )(q_t, k_t, v_t, h_bhntKV, do_t, g_gamma)

    dq = dq_5d.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K)
    dk = dk_5d.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K)
    dv = dv_5d.transpose(0, 1, 3, 2, 4).reshape(B, T, H, V)

    return dq, dk, dv


# ============================================================
# custom_vjp wrapper
# ============================================================


def chunk_fused_gla(q, k, v, g_gamma, scale, chunk_size):
    """Fused chunked GLA with custom_vjp (Pallas TPU kernels)."""
    @jax.custom_vjp
    def _compute(q, k, v):
        _, o = chunk_fwd_fused_g_gamma(q, k, v, g_gamma, scale, chunk_size)
        return o

    def _fwd(q, k, v):
        h, o = chunk_fwd_fused_g_gamma(q, k, v, g_gamma, scale, chunk_size)
        return o, (q, k, v, h)

    def _bwd(residuals, do):
        q, k, v, h = residuals
        dq, dk, dv = chunk_bwd_fused_g_gamma(
            q, k, v, h, do, g_gamma, scale, chunk_size
        )
        return dq, dk, dv

    _compute.defvjp(_fwd, _bwd)
    return _compute(q, k, v)


# ============================================================
# Entry point
# ============================================================


def optimized_compute(B=2, T=256, H=4, K=128, V=128, chunk_size=64):
    """Forward + backward fused chunk-GLA with Pallas TPU kernels.

    Returns the loss scalar. Both forward and backward are computed,
    so timing captures the full training step performance.
    """
    q, k_arr, v, g_gamma = _make_test_data(B, T, H, K, V, chunk_size)
    scale = K ** -0.5

    def loss_fn(q, k, v):
        return chunk_fused_gla(
            q.astype(jnp.float32), k.astype(jnp.float32),
            v.astype(jnp.float32), g_gamma, scale, chunk_size,
        ).sum()

    loss, _ = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k_arr, v)
    return loss
# EVOLVE-BLOCK-END
