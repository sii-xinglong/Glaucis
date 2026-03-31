"""Chunked GLA (Gated Linear Attention) Pallas TPU kernel — template for evolutionary optimization.

Implements chunked GLA forward and backward passes using Pallas kernels
targeting TPU, with g_gamma (per-head constant gate) mode.

Optimization targets within the EVOLVE-BLOCK:
  - Kernel fusion strategies (merge phases)
  - Block sizes and tiling within kernels
  - Memory layout and transpose strategies
  - Loop structure and pipelining
  - Grid dimensions and BlockSpec configurations
  - Accumulator precision choices

AL model reference dimensions:
  q, k, v: [2, 4096, 16, 128]
  g_gamma:  (16,)
  chunk_size: 64
"""

import functools

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _make_test_data(B, T, H, K, V, chunk_size, seed=42):
    """Create deterministic (q, k, v, g_gamma) for a GLA test case."""
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
# Forward: Inter-chunk state propagation (Pallas kernel)
# ============================================================


def _chunk_fwd_h_kernel(
    k_ref, v_ref, h0_ref, g_gamma,
    h_ref, ht_ref, scratch_ref,
    *, BT, NT,
):
    BK = k_ref.shape[3]
    BV = v_ref.shape[3]
    i_b, i_h, i_k, i_v, i_t = (
        pl.program_id(0), pl.program_id(1), pl.program_id(2),
        pl.program_id(3), pl.program_id(4),
    )

    b_g = g_gamma[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)

    @pl.when(i_t == 0)
    def init():
        if h0_ref is not None:
            scratch_ref[:, :] = h0_ref[0, 0].astype(jnp.float32)
        else:
            scratch_ref[:, :] = jnp.zeros((BK, BV), dtype=jnp.float32)

    h_ref[0, i_t, 0] = scratch_ref[...]

    k_tile = k_ref[0, 0]
    v_tile = v_ref[0, 0]

    b_g_last = g_gamma[i_h] * BT
    scratch_ref[...] *= exp(b_g_last)
    v_tile = (v_tile * exp(b_g_last - b_g)[:, None]).astype(v_tile.dtype)

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


def chunk_fwd_h(k, v, g_gamma, chunk_size):
    """Launch inter-chunk state propagation Pallas kernel."""
    BK, BV, BT = 128, 128, chunk_size
    B, T, H, K_dim = k.shape
    V = v.shape[-1]
    NT = T // BT

    k_t = jnp.transpose(k, (0, 2, 1, 3))
    v_t = jnp.transpose(v, (0, 2, 1, 3))

    grid = (B, H, pl.cdiv(K_dim, BK), pl.cdiv(V, BV), NT)

    def k_map(b, h, ki, vi, t): return b, h, t, ki
    def v_map(b, h, ki, vi, t): return b, h, t, vi
    def h_map(b, h, ki, vi, t): return b, 0, h, ki, vi
    def ht_map(b, h, ki, vi, t): return b, h, ki, vi

    h_all, ht = pl.pallas_call(
        functools.partial(_chunk_fwd_h_kernel, BT=BT, NT=NT),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, 1, BT, BK), k_map),
                pl.BlockSpec((1, 1, BT, BV), v_map),
                None,
                pl.BlockSpec(memory_space=pltpu.SMEM),
            ],
            out_specs=[
                pl.BlockSpec((1, NT, 1, BK, BV), h_map),
                None,
            ],
            scratch_shapes=[pltpu.VMEM((BK, BV), jnp.float32)],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((B, NT, H, K_dim, V), k.dtype),
            None,
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary", "arbitrary", "arbitrary"),
            disable_bounds_checks=True,
        ),
    )(k_t, v_t, None, g_gamma)

    return h_all


# ============================================================
# Forward: Intra-chunk attention matrix (Pallas kernel)
# RETAINED for backward recomputation path, but no longer
# called during forward pass.
# ============================================================


def _chunk_gla_fwd_intra_gk_pl(q_ref, k_ref, g_ref, A_ref, *, BT, scale):
    b_q = q_ref[0, 0]
    b_k = k_ref[0, 0]
    b_g = g_ref[0, 0].astype(jnp.float32)

    b_qg = (b_q * jnp.exp(b_g)).astype(b_q.dtype)
    b_kg = (b_k * jnp.exp(-b_g)).astype(b_k.dtype)

    b_A = (
        jnp.dot(b_qg, b_kg.T,
                precision=jax.lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32)
        * scale
    )
    A_ref[0, 0] = b_A.astype(A_ref.dtype)


def chunk_gla_fwd_intra_gk(q, k, g_cumsum, scale, chunk_size):
    """Launch intra-chunk attention Pallas kernel."""
    B, T, H, K = q.shape
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _k = k.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _g = g_cumsum.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)

    spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    A_spec = pl.BlockSpec([1, 1, BT, BT], index_map=lambda h, nt: (h, nt, 0, 0))

    A = pl.pallas_call(
        functools.partial(_chunk_gla_fwd_intra_gk_pl, BT=BT, scale=scale),
        grid=(H, total_NT),
        out_shape=jax.ShapeDtypeStruct([H, total_NT, BT, BT], jnp.float32),
        in_specs=[spec, spec, spec],
        out_specs=A_spec,
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
    )(_q, _k, _g)

    A = A.reshape(H, B, NT, BT, BT).transpose(1, 0, 2, 3, 4)
    A = A.reshape(B, H, NT * BT, BT).transpose(0, 2, 1, 3)
    return A


# ============================================================
# Forward: Output combination (Pallas kernel)
#
# MUTATION (fuse_fwd_A): Recompute A inside this kernel instead
# of receiving it as a pre-computed input. This eliminates:
#   1. The chunk_gla_fwd_intra_gk pallas_call (kernel launch overhead)
#   2. One full HBM round-trip for the A matrix
#   3. One intermediate tensor allocation (B*H*T*BT float32)
#
# k is added as a new input to recompute: A = (q*exp(g)) @ (k*exp(-g)).T * scale
# A is removed from inputs (net change: same number of inputs, but
# we save the entire separate kernel launch + HBM write/read of A).
#
# MUTATION (bf16_uniform): Cast ALL matmul operands to bfloat16
# before each dot product. Accumulators remain float32 via
# preferred_element_type. This halves register footprint for
# matmul inputs, reducing register spills.
#
# MXU-VPU overlap strategy:
#   Phase 0 (VPU): Compute exp(g), exp(-g), gate q and k
#   Phase 1 (MXU): Recompute A = b_qg @ b_kg.T * scale
#   Phase 2 (VPU): Apply causal mask to A (overlaps with MXU drain)
#   Phase 3 (MXU): Inter-chunk: b_qg @ h * scale
#   Phase 4 (MXU): Intra-chunk: A_masked @ v
#   Phase 5 (VPU): Sum and write output
# ============================================================


def _chunk_gla_fwd_o_gk_pl(q_ref, k_ref, v_ref, g_ref, h_ref, o_ref, *, BT, scale):
    b_q = q_ref[0, 0]
    b_k = k_ref[0, 0]
    b_v = v_ref[0, 0]
    b_g = g_ref[0, 0]
    b_h = h_ref[0, 0]

    # --- Phase 0 (VPU): Pre-compute all gating scalars ---
    b_g_f32 = b_g.astype(jnp.float32)
    exp_g = jnp.exp(b_g_f32)              # [BT, K]
    exp_neg_g = jnp.exp(-b_g_f32)         # [BT, K]
    # Cast gated variants directly to bf16 to reduce register footprint
    b_qg = (b_q * exp_g).astype(jnp.bfloat16)    # [BT, K]: q * exp(g)
    b_kg = (b_k * exp_neg_g).astype(jnp.bfloat16) # [BT, K]: k * exp(-g)

    # --- Phase 1 (MXU): Recompute A = b_qg @ b_kg.T * scale ---
    # MUTATION (bf16_uniform): both operands already bf16
    b_A = jnp.dot(b_qg, b_kg.T,
                  precision=jax.lax.Precision.HIGHEST,
                  preferred_element_type=jnp.float32) * scale  # [BT, BT]

    # --- Phase 2 (VPU): Causal mask (overlaps with MXU drain) ---
    m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_A_masked = jnp.where(m_s, b_A, 0.0).astype(jnp.bfloat16)

    # --- Phase 3 (MXU): Inter-chunk contribution (b_qg @ h) ---
    # MUTATION (bf16_uniform): cast b_h to bf16, b_qg already bf16
    b_o_inter = jnp.dot(b_qg, b_h.astype(jnp.bfloat16),
                        precision=jax.lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32)
    b_o_inter = b_o_inter * scale

    # --- Phase 4 (MXU): Intra-chunk contribution (A_masked @ v) ---
    # MUTATION (bf16_uniform): both operands bf16 (b_A_masked cast above, b_v is input bf16)
    b_o_intra = jnp.dot(b_A_masked, b_v.astype(jnp.bfloat16),
                        precision=jax.lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32)

    # --- Phase 5 (VPU): Sum and write ---
    o_ref[0, 0] = (b_o_inter + b_o_intra).astype(o_ref.dtype)


def chunk_gla_fwd_o_gk(q, k, v, g_cumsum, h, scale, chunk_size):
    """Launch output combination Pallas kernel.

    MUTATION (fuse_fwd_A): k is now an input; A is recomputed inside the kernel.
    This eliminates the separate chunk_gla_fwd_intra_gk pallas_call.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _k = k.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _v = v.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _g = g_cumsum.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _h = h.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)

    q_spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    k_spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    v_spec = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))
    g_spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    h_spec = pl.BlockSpec([1, 1, K, V], index_map=lambda h, nt: (h, nt, 0, 0))
    o_spec = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))

    o = pl.pallas_call(
        functools.partial(_chunk_gla_fwd_o_gk_pl, BT=BT, scale=scale),
        grid=(H, total_NT),
        out_shape=jax.ShapeDtypeStruct([H, total_NT, BT, V], v.dtype),
        in_specs=[q_spec, k_spec, v_spec, g_spec, h_spec],
        out_specs=o_spec,
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
    )(_q, _k, _v, _g, _h)

    o = o.reshape(H, B, NT, BT, V).transpose(1, 0, 2, 3, 4)
    o = o.reshape(B, H, NT * BT, V).transpose(0, 2, 1, 3)
    return o


# ============================================================
# Forward orchestrator
#
# MUTATION (fuse_fwd_A): Removed chunk_gla_fwd_intra_gk call.
# A is no longer computed or returned -- it is recomputed inside
# the output kernel from q, k, g_cumsum.
# This reduces forward from 3 pallas_calls to 2.
# ============================================================


def chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size):
    """Chunked GLA forward pass."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C

    pos = jnp.arange(1, C + 1, dtype=jnp.float32)
    pos = jnp.tile(pos, NT).reshape(1, T, 1, 1)
    g_cumsum = jnp.broadcast_to(g_gamma.reshape(1, 1, -1, 1) * pos, q.shape)

    h = chunk_fwd_h(k, v, g_gamma, C)
    # REMOVED: A = chunk_gla_fwd_intra_gk(q, k, g_cumsum, scale, C)
    # A is now recomputed inside chunk_gla_fwd_o_gk
    o = chunk_gla_fwd_o_gk(q, k, v, g_cumsum, h, scale, C)

    return g_cumsum, h, o


# ============================================================
# Backward: State gradient propagation (Pallas kernel)
#
# MUTATION (fold_dh_pallas): Replace lax.scan with pallas_call.
#
# MUTATION (bf16_uniform): Cast both matmul operands to bfloat16
# before the dot product. Accumulator remains float32.
# ============================================================


def _chunk_bwd_dh_kernel(
    q_ref, do_ref, h0_ref, g_gamma,
    dh_ref, dht_ref, scratch_ref,
    *, BT, NT, scale,
):
    """Backward dh propagation kernel body.

    Mirrors _chunk_fwd_h_kernel structure but computes:
      dh[t] = dh[t+1] * state_decay + (q[t] * exp(g_ramp) * scale).T @ do[t]

    Because inputs are time-reversed before being passed in, the kernel
    iterates forward but effectively processes chunks from last to first.
    """
    BK = q_ref.shape[3]
    BV = do_ref.shape[3]
    i_b, i_h, i_k, i_v, i_t = (
        pl.program_id(0), pl.program_id(1), pl.program_id(2),
        pl.program_id(3), pl.program_id(4),
    )

    # g_ramp: per-position gating within a chunk = g_gamma * (1, 2, ..., BT)
    b_g_ramp = g_gamma[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)

    # State decay for one full chunk: exp(g_gamma * BT)
    b_g_last = g_gamma[i_h] * BT

    # Initialize scratch (dh state) to zeros at t=0 (which is the END
    # of the original sequence because inputs are reversed)
    @pl.when(i_t == 0)
    def init():
        if h0_ref is not None:
            scratch_ref[:, :] = h0_ref[0, 0].astype(jnp.float32)
        else:
            scratch_ref[:, :] = jnp.zeros((BK, BV), dtype=jnp.float32)

    # Emit current dh state BEFORE updating (matches scan semantics
    # where dh_out is emitted before the state update)
    dh_ref[0, i_t, 0] = scratch_ref[...]

    # Load q and do tiles for this (reversed) time step
    q_tile = q_ref[0, 0]    # [BT, BK]
    do_tile = do_ref[0, 0]  # [BT, BV]

    # Update dh: dh = dh * state_decay + q_hat.T @ do
    # where q_hat = q * exp(g_ramp) * scale
    scratch_ref[...] *= exp(b_g_last)

    # q_hat = q * exp(g_ramp) * scale, shape [BT, BK]
    # MUTATION (bf16_uniform): cast to bf16 before dot
    q_hat = (q_tile * exp(b_g_ramp)[:, None] * scale).astype(jnp.bfloat16)

    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        q_hat.T,                                   # [BK, BT] bf16
        do_tile.astype(jnp.bfloat16),              # [BT, BV] bf16
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    @pl.when(i_t == NT - 1)
    def end():
        if dht_ref is not None:
            dht_ref[0, 0] = scratch_ref[...]


def chunk_bwd_dh_pallas(q, do, g_gamma, scale, chunk_size):
    """Launch backward dh propagation as a Pallas kernel.

    MUTATION (fold_dh_pallas): Replaces _chunk_bwd_dh_scan (lax.scan)
    with a single pallas_call, reducing ~NT computation events to ~1.

    Reverse scan is implemented by flipping q and do along the time
    axis before the call, and flipping dh after.
    """
    BK, BV, BT = 128, 128, chunk_size
    B, T, H, K = q.shape
    V = do.shape[-1]
    NT = T // BT

    # Reshape to (B, H, NT, BT, dim) then transpose to (B, H, T_chunks, BT, dim)
    q_t = jnp.transpose(q, (0, 2, 1, 3))  # [B, H, T, K]
    do_t = jnp.transpose(do, (0, 2, 1, 3))  # [B, H, T, V]

    # Reshape time into chunks: [B, H, NT, BT, dim]
    q_chunked = q_t.reshape(B, H, NT, BT, K)
    do_chunked = do_t.reshape(B, H, NT, BT, V)

    # REVERSE the chunk order to implement reverse scan
    # After flip: position 0 = last chunk, position NT-1 = first chunk
    q_flipped = jnp.flip(q_chunked, axis=2)   # [B, H, NT, BT, K]
    do_flipped = jnp.flip(do_chunked, axis=2)  # [B, H, NT, BT, V]

    # Collapse back to [B, H, NT*BT, dim] for BlockSpec compatibility
    q_flat = q_flipped.reshape(B, H, NT * BT, K)
    do_flat = do_flipped.reshape(B, H, NT * BT, V)

    grid = (B, H, pl.cdiv(K, BK), pl.cdiv(V, BV), NT)

    def q_map(b, h, ki, vi, t): return b, h, t, ki
    def do_map(b, h, ki, vi, t): return b, h, t, vi
    def dh_map(b, h, ki, vi, t): return b, 0, h, ki, vi
    def dht_map(b, h, ki, vi, t): return b, h, ki, vi

    dh_flipped, dht = pl.pallas_call(
        functools.partial(_chunk_bwd_dh_kernel, BT=BT, NT=NT, scale=scale),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, 1, BT, BK), q_map),
                pl.BlockSpec((1, 1, BT, BV), do_map),
                None,
                pl.BlockSpec(memory_space=pltpu.SMEM),
            ],
            out_specs=[
                pl.BlockSpec((1, NT, 1, BK, BV), dh_map),
                None,
            ],
            scratch_shapes=[pltpu.VMEM((BK, BV), jnp.float32)],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((B, NT, H, K, V), jnp.float32),
            None,
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary", "arbitrary", "arbitrary"),
            disable_bounds_checks=True,
        ),
    )(q_flat, do_flat, None, g_gamma)

    # Reverse the output back to original time order
    # dh_flipped is [B, NT, H, K, V] with reversed time
    dh_out = jnp.flip(dh_flipped, axis=1)  # [B, NT, H, K, V]

    return dh_out


# ============================================================
# Backward: Fused dq, dk, dv (Pallas kernel)
#
# MUTATION (combined: reduce_inputs + skip_dg + bf16_uniform):
#   1. A is recomputed inside the kernel (from L1_reduce_inputs parent)
#   2. dg computation is completely removed (skip_dg)
#   3. ALL matmul operands cast to bfloat16 (bf16_uniform)
#
# bf16_uniform rationale: The backward kernel has 2.5M register spills.
# All intermediate arrays are float32, consuming significant register
# space. By casting BOTH matmul operands to bfloat16 before each dot
# product (keeping the accumulator in float32 via preferred_element_type),
# we halve the register footprint of matmul inputs. The key insight
# from FP25: ALL matmul operands must be the SAME dtype. Previous
# L1_bf16_intermediates failed because it cast only ONE operand.
#
# MXU-VPU overlap strategy:
#   Phase 0 (VPU): Pre-compute exp values and gated key/query variants
#   Phase 1 (MXU): Recompute A = q_pos @ k_neg.T * scale
#                  Then compute dA = do @ v.T * scale
#   Phase 2 (VPU): Apply causal masks to both A and dA
#   Phase 3 (MXU batch): Four independent dot products back-to-back
#   Phase 4 (MXU): Intra-chunk dq and dk
#   Phase 5 (VPU): Combine results and write outputs (NO dg)
# ============================================================


def _chunk_gla_bwd_fused_kernel(
    q_ref, k_ref, v_ref, g_ref, h_ref, do_ref, dh_ref,
    dq_ref, dk_ref, dv_ref,
    *, BT, scale,
):
    b_q = q_ref[0, 0]
    b_k = k_ref[0, 0]
    b_v = v_ref[0, 0]
    b_g = g_ref[0, 0].astype(jnp.float32)
    b_h = h_ref[0, 0].astype(jnp.float32)
    b_do = do_ref[0, 0]
    b_dh = dh_ref[0, 0].astype(jnp.float32)

    b_gn = b_g[BT - 1, :]  # last row: [K]

    # -------------------------------------------------------
    # Phase 0 (VPU): Pre-compute ALL exp/gate values upfront
    # Compute in float32, then cast intermediates to bf16 early
    # to reduce register footprint.
    # -------------------------------------------------------
    exp_pos_g = jnp.exp(b_g)               # [BT, K] float32
    exp_neg_g = jnp.exp(-b_g)              # [BT, K] float32
    exp_gn_minus_g = jnp.exp(b_gn[None, :] - b_g)  # [BT, K] float32

    # Pre-cast gated key/query variants to bf16 immediately
    # This reduces their register footprint from float32 to bf16
    k_neg = (b_k * exp_neg_g).astype(jnp.bfloat16)          # [BT, K]: k * exp(-g)
    k_decay = (b_k * exp_gn_minus_g).astype(jnp.bfloat16)   # [BT, K]: k * exp(gn-g)
    q_pos = (b_q * exp_pos_g).astype(jnp.bfloat16)          # [BT, K]: q * exp(g)

    # Also cast b_do and b_v to bf16 for use as matmul operands
    b_do_bf16 = b_do.astype(jnp.bfloat16)   # [BT, V]
    b_v_bf16 = b_v.astype(jnp.bfloat16)     # [BT, V]

    # -------------------------------------------------------
    # Phase 1 (MXU): Recompute A from q_pos and k_neg, then
    # compute dA. Both operands bf16 for each dot.
    # -------------------------------------------------------
    # MUTATION (bf16_uniform): both q_pos, k_neg already bf16
    b_a = jnp.dot(q_pos, k_neg.T,
                  precision=jax.lax.Precision.HIGHEST,
                  preferred_element_type=jnp.float32) * scale  # [BT, BT]

    # MUTATION (bf16_uniform): both b_do_bf16, b_v_bf16 are bf16
    b_dA_raw = jnp.dot(b_do_bf16, b_v_bf16.T,
                       precision=jax.lax.Precision.HIGHEST,
                       preferred_element_type=jnp.float32) * scale  # [BT, BT]

    # -------------------------------------------------------
    # Phase 2 (VPU): Apply causal masks
    # Cast masked results to bf16 to reduce register footprint
    # -------------------------------------------------------
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_dA = jnp.where(mask, b_dA_raw, 0.0).astype(jnp.bfloat16)    # masked dA [BT, BT] bf16
    b_a_masked = jnp.where(mask, b_a, 0.0).astype(jnp.bfloat16)   # masked a  [BT, BT] bf16

    # -------------------------------------------------------
    # Phase 3 (MXU batch): Four independent dot products
    # MUTATION (bf16_uniform): ALL operands cast to bf16
    # -------------------------------------------------------
    # dv_intra: b_a_masked.T @ b_do  (both bf16)
    b_dv_intra = jnp.dot(b_a_masked.T, b_do_bf16,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)   # [BT, V]

    # dv_inter: k_decay @ b_dh  (k_decay bf16, cast b_dh to bf16)
    b_dv_inter = jnp.dot(k_decay, b_dh.astype(jnp.bfloat16),
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)   # [BT, V]

    # dq_inter: b_do @ b_h.T  (both cast to bf16)
    b_dq_inter = jnp.dot(b_do_bf16, b_h.astype(jnp.bfloat16).T,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)   # [BT, K]

    # dk_inter: b_v @ b_dh.T  (both cast to bf16)
    b_dk_inter = jnp.dot(b_v_bf16, b_dh.astype(jnp.bfloat16).T,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)   # [BT, K]

    # -------------------------------------------------------
    # Phase 4 (MXU): Intra-chunk dq and dk
    # MUTATION (bf16_uniform): ALL operands bf16
    # -------------------------------------------------------
    # b_dA and k_neg already bf16
    b_dq_intra_raw = jnp.dot(b_dA, k_neg,
                             precision=jax.lax.Precision.HIGHEST,
                             preferred_element_type=jnp.float32)  # [BT, K]

    # b_dA.T and q_pos already bf16
    b_dk_intra_raw = jnp.dot(b_dA.T, q_pos,
                             precision=jax.lax.Precision.HIGHEST,
                             preferred_element_type=jnp.float32)  # [BT, K]

    # -------------------------------------------------------
    # Phase 5 (VPU): Combine results and write outputs
    # REMOVED: All dg computation (dgk_inter, dg_raw, mask_upper,
    #          M_upper, dg_rev_cumsum, dg_ref write)
    # -------------------------------------------------------

    # dv: combine intra + inter
    dv_ref[0, 0] = (b_dv_intra + b_dv_inter).astype(dv_ref.dtype)

    # dq: scale intra by exp(g), scale inter by scale*exp(g)
    b_dq = b_dq_intra_raw * exp_pos_g + b_dq_inter * (scale * exp_pos_g)
    dq_ref[0, 0] = b_dq.astype(dq_ref.dtype)

    # dk: scale intra by exp(-g), inter already scaled correctly
    b_dk = b_dk_intra_raw * exp_neg_g + b_dk_inter * exp_gn_minus_g
    dk_ref[0, 0] = b_dk.astype(dk_ref.dtype)


def chunk_gla_bwd_fused(q, k, v, g_cumsum, h, do, dh, scale, chunk_size):
    """Launch fused backward Pallas kernel.

    MUTATION (combined):
      - A is recomputed inside the kernel (inherited from L1_reduce_inputs)
      - dg output is removed (skip_dg): 3 outputs instead of 4
      - ALL matmul operands cast to bf16 (bf16_uniform)
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _k = k.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _v = v.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _g = g_cumsum.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _do = do.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _h = h.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)
    _dh = dh.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)

    grid = (H, total_NT)
    spec_K = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_V = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_h = pl.BlockSpec([1, 1, K, V], index_map=lambda h, nt: (h, nt, 0, 0))

    dq, dk, dv = pl.pallas_call(
        functools.partial(_chunk_gla_bwd_fused_kernel, BT=BT, scale=scale),
        grid=grid,
        out_shape=[
            jax.ShapeDtypeStruct([H, total_NT, BT, K], q.dtype),
            jax.ShapeDtypeStruct([H, total_NT, BT, K], k.dtype),
            jax.ShapeDtypeStruct([H, total_NT, BT, V], v.dtype),
        ],
        in_specs=[spec_K, spec_K, spec_V, spec_K, spec_h, spec_V, spec_h],
        out_specs=[spec_K, spec_K, spec_V],
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
    )(_q, _k, _v, _g, _h, _do, _dh)

    def _unreshape(x, last_dim):
        x = x.reshape(H, B, NT, BT, last_dim)
        x = x.transpose(1, 0, 2, 3, 4)
        x = x.reshape(B, H, T, last_dim)
        return x.transpose(0, 2, 1, 3)

    return _unreshape(dq, K), _unreshape(dk, K), _unreshape(dv, V)


# ============================================================
# custom_vjp wrapper
#
# MUTATION (combined + fold_dh_pallas):
#   _fwd: residuals no longer include A (fuse_fwd_A)
#   _bwd: uses chunk_bwd_dh_pallas instead of _chunk_bwd_dh_scan,
#          chunk_gla_bwd_fused returns 3 values (skip_dg)
# ============================================================


def chunk_gla(q, k, v, g_gamma, scale, chunk_size):
    """Chunked GLA with custom_vjp (Pallas TPU kernels)."""
    @jax.custom_vjp
    def _compute(q, k, v):
        _, _, o = chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size)
        return o

    def _fwd(q, k, v):
        g_cumsum, h, o = chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size)
        return o, (q, k, v, g_cumsum, h)

    def _bwd(residuals, do):
        q, k, v, g_cumsum, h = residuals
        B, T, H, K = q.shape
        V = v.shape[-1]
        C = chunk_size
        NT = T // C
        # MUTATION (fold_dh_pallas): Use pallas_call instead of lax.scan
        dh = chunk_bwd_dh_pallas(q, do, g_gamma, scale, C)
        dq, dk, dv = chunk_gla_bwd_fused(q, k, v, g_cumsum, h, do, dh, scale, C)
        return dq, dk, dv

    _compute.defvjp(_fwd, _bwd)
    return _compute(q, k, v)


# ============================================================
# Entry point
# ============================================================


def optimized_compute(B=2, T=4096, H=16, K=128, V=128, chunk_size=64):
    """Forward + backward chunked GLA with Pallas TPU kernels.

    Returns the loss scalar. Both forward and backward are computed,
    so timing captures the full training step performance.
    """
    q, k_arr, v, g_gamma = _make_test_data(B, T, H, K, V, chunk_size)
    scale = K ** -0.5

    def loss_fn(q, k, v):
        return chunk_gla(q.astype(jnp.float32), k.astype(jnp.float32),
                        v.astype(jnp.float32), g_gamma, scale, chunk_size).sum()

    loss, _ = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k_arr, v)
    return loss
# EVOLVE-BLOCK-END
