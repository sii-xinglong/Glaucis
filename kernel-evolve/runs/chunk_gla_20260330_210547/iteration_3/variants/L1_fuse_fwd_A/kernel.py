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
# KEPT for code completeness but NOT called in forward path.
# A is now recomputed inside _chunk_gla_fwd_o_gk_fused_pl.
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
    """Launch intra-chunk attention Pallas kernel.

    NOTE: This function is retained for code completeness but is no longer
    called in the forward path. A is recomputed inside the fused output kernel.
    """
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
# Forward: Fused output combination with A recomputation
# (Pallas kernel)
#
# MUTATION (L1_fuse_fwd_A): Recompute A inside the output kernel
# instead of loading it as a separate input. This eliminates the
# separate chunk_gla_fwd_intra_gk pallas_call entirely.
#
# Changes from L1_reduce_inputs base:
#   1. Added k as input (k_ref), removed A (a_ref)
#   2. Recompute A inside kernel: b_qg @ b_kg.T * scale
#   3. Apply causal mask to recomputed A
#   4. Use masked A for intra-chunk output contribution
#
# Benefits:
#   - Eliminate 1 pallas_call from forward (3 -> 2 kernel launches)
#   - Save kernel launch overhead + HBM round-trip for A
#   - A no longer needs to be stored as a residual (backward
#     already recomputes it from SO11/L1_reduce_inputs)
#   - Reduce total HBM traffic: A is [B,H,T,BT] float32
#
# Cost:
#   - One additional MXU dot in the output kernel: [BT,K] @ [K,BT]
#   - This is small relative to the existing inter-chunk dot
#
# MXU-VPU overlap strategy:
#   VPU phase: pre-compute exp(g), exp(-g), form b_qg, b_kg
#   MXU phase 1: recompute A = b_qg @ b_kg.T * scale
#   MXU phase 2: inter-chunk b_qg @ h (overlaps with VPU causal mask)
#   VPU phase: apply causal mask to A
#   MXU phase 3: intra-chunk A_masked @ v
#   VPU phase: sum and write
# ============================================================


def _chunk_gla_fwd_o_gk_fused_pl(q_ref, k_ref, v_ref, g_ref, h_ref, o_ref, *, BT, scale):
    b_q = q_ref[0, 0]
    b_k = k_ref[0, 0]
    b_g = g_ref[0, 0]
    b_v = v_ref[0, 0]
    b_h = h_ref[0, 0]

    # --- VPU phase: pre-compute all gating scalars ---
    b_g_f32 = b_g.astype(jnp.float32)
    exp_g = jnp.exp(b_g_f32)            # [BT, K]
    exp_neg_g = jnp.exp(-b_g_f32)       # [BT, K]
    b_qg = (b_q * exp_g).astype(b_q.dtype)      # q * exp(g)
    b_kg = (b_k * exp_neg_g).astype(b_k.dtype)   # k * exp(-g)

    # --- MXU phase 1: recompute A = b_qg @ b_kg.T * scale ---
    # This replaces loading A from HBM, eliminating one pallas_call
    b_A = jnp.dot(b_qg, b_kg.T,
                  precision=jax.lax.Precision.HIGHEST,
                  preferred_element_type=jnp.float32) * scale   # [BT, BT]

    # --- MXU phase 2: inter-chunk contribution (b_qg @ h) ---
    b_o_inter = jnp.dot(b_qg, b_h.astype(b_qg.dtype),
                        precision=jax.lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32)
    b_o_inter = b_o_inter * scale

    # --- VPU phase: causal mask (overlaps with MXU pipeline drain) ---
    m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_A_masked = jnp.where(m_s, b_A, 0.0).astype(b_v.dtype)

    # --- MXU phase 3: intra-chunk contribution (A_masked @ v) ---
    b_o_intra = jnp.dot(b_A_masked, b_v,
                        precision=jax.lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32)

    # --- VPU phase: sum and write ---
    o_ref[0, 0] = (b_o_inter + b_o_intra).astype(o_ref.dtype)


def chunk_gla_fwd_o_gk(q, k, v, g_cumsum, h, scale, chunk_size):
    """Launch fused output combination Pallas kernel.

    MUTATION: Takes k instead of A. Recomputes A inside the kernel
    from q, k, g, eliminating the separate chunk_gla_fwd_intra_gk call.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _k = k.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _g = g_cumsum.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _v = v.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _h = h.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)

    q_spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    k_spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    v_spec = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))
    g_spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    h_spec = pl.BlockSpec([1, 1, K, V], index_map=lambda h, nt: (h, nt, 0, 0))
    o_spec = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))

    o = pl.pallas_call(
        functools.partial(_chunk_gla_fwd_o_gk_fused_pl, BT=BT, scale=scale),
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
# MUTATION: Removed chunk_gla_fwd_intra_gk call. A is now
# recomputed inside chunk_gla_fwd_o_gk. Forward goes from
# 3 pallas_calls to 2. A is no longer returned (backward
# already recomputes it via L1_reduce_inputs optimization).
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
# Backward: State gradient propagation (lax.scan)
# ============================================================


def _chunk_bwd_dh_scan(q, do, g_gamma, scale, C, B, T, H, K, V, NT):
    """Backward state gradient propagation via reverse lax.scan."""
    q_scan = q.reshape(B, NT, C, H, K).transpose(1, 0, 2, 3, 4)
    do_scan = do.reshape(B, NT, C, H, V).transpose(1, 0, 2, 3, 4)

    g_gamma_f32 = g_gamma.astype(jnp.float32)
    state_decay = jnp.exp(g_gamma_f32 * C)
    b_g_ramp = g_gamma_f32[None, :] * (jnp.arange(C, dtype=jnp.float32) + 1)[:, None]

    def scan_fn(dh, chunk_data):
        b_q, b_do = chunk_data
        dh_out = dh
        dh = dh * state_decay[None, :, None, None]
        b_q_hat = (b_q * jnp.exp(b_g_ramp)[None, :, :, None] * scale)
        dh = dh + lax.dot_general(
            b_q_hat.astype(jnp.float32), b_do.astype(jnp.float32),
            dimension_numbers=(((1,), (1,)), ((0, 2), (0, 2))),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        return dh, dh_out

    dh_init = jnp.zeros((B, H, K, V), dtype=jnp.float32)
    _, dh_all = lax.scan(scan_fn, dh_init, (q_scan, do_scan), reverse=True)
    return dh_all.transpose(1, 0, 2, 3, 4)


# ============================================================
# Backward: Fused dq, dk, dv, dg (Pallas kernel)
#
# UNCHANGED from L1_reduce_inputs base. A is recomputed inside
# the backward kernel from q, k, g (SO11 optimization).
# ============================================================


def _chunk_gla_bwd_fused_kernel(
    q_ref, k_ref, v_ref, g_ref, h_ref, do_ref, dh_ref,
    dq_ref, dk_ref, dv_ref, dg_ref,
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
    # -------------------------------------------------------
    exp_pos_g = jnp.exp(b_g)               # [BT, K]
    exp_neg_g = jnp.exp(-b_g)              # [BT, K]
    exp_gn_minus_g = jnp.exp(b_gn[None, :] - b_g)  # [BT, K]

    k_neg = (b_k * exp_neg_g).astype(b_k.dtype)
    k_decay = (b_k * exp_gn_minus_g).astype(b_k.dtype)
    q_pos = (b_q * exp_pos_g).astype(b_q.dtype)

    # -------------------------------------------------------
    # Phase 1 (MXU): Recompute A from q_pos and k_neg, then
    # compute dA.
    # -------------------------------------------------------
    b_a = jnp.dot(q_pos, k_neg.T,
                  precision=jax.lax.Precision.HIGHEST,
                  preferred_element_type=jnp.float32) * scale

    b_dA_raw = jnp.dot(b_do.astype(b_v.dtype), b_v.T,
                       precision=jax.lax.Precision.HIGHEST,
                       preferred_element_type=jnp.float32) * scale

    # -------------------------------------------------------
    # Phase 2 (VPU): Apply causal masks
    # -------------------------------------------------------
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_dA = jnp.where(mask, b_dA_raw, 0.0)
    b_a_masked = jnp.where(mask, b_a, 0.0)

    # -------------------------------------------------------
    # Phase 3 (MXU batch): Four independent dot products
    # -------------------------------------------------------
    b_dv_intra = jnp.dot(b_a_masked.T.astype(b_do.dtype), b_do,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)

    b_dv_inter = jnp.dot(k_decay, b_dh.astype(b_k.dtype),
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)

    b_dq_inter = jnp.dot(b_do, b_h.astype(b_do.dtype).T,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)

    b_dk_inter = jnp.dot(b_v, b_dh.astype(b_v.dtype).T,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)

    # -------------------------------------------------------
    # Phase 4 (MXU): Intra-chunk dq and dk
    # -------------------------------------------------------
    b_dq_intra_raw = jnp.dot(b_dA.astype(k_neg.dtype), k_neg,
                             precision=jax.lax.Precision.HIGHEST,
                             preferred_element_type=jnp.float32)

    b_dk_intra_raw = jnp.dot(b_dA.T.astype(q_pos.dtype), q_pos,
                             precision=jax.lax.Precision.HIGHEST,
                             preferred_element_type=jnp.float32)

    # -------------------------------------------------------
    # Phase 5 (VPU): Combine results and write outputs
    # -------------------------------------------------------
    dv_ref[0, 0] = (b_dv_intra + b_dv_inter).astype(dv_ref.dtype)

    b_dq = b_dq_intra_raw * exp_pos_g + b_dq_inter * (scale * exp_pos_g)
    dq_ref[0, 0] = b_dq.astype(dq_ref.dtype)

    b_dk = b_dk_intra_raw * exp_neg_g + b_dk_inter * exp_gn_minus_g
    dk_ref[0, 0] = b_dk.astype(dk_ref.dtype)

    dgk_inter = (jnp.exp(b_gn) * jnp.sum(b_h * b_dh, axis=1)
                 + jnp.sum(b_dk_inter * b_k.astype(jnp.float32), axis=0))
    dg_raw = b_q.astype(jnp.float32) * b_dq - b_k.astype(jnp.float32) * b_dk
    mask_upper = jnp.arange(BT)[None, :] >= jnp.arange(BT)[:, None]
    M_upper = jnp.where(mask_upper, 1.0, 0.0).astype(jnp.float32)
    dg_rev_cumsum = jnp.dot(M_upper, dg_raw,
                           precision=jax.lax.Precision.HIGHEST,
                           preferred_element_type=jnp.float32)
    dg_ref[0, 0] = (dg_rev_cumsum + dgk_inter[None, :]).astype(dg_ref.dtype)


def chunk_gla_bwd_fused(q, k, v, g_cumsum, h, do, dh, scale, chunk_size):
    """Launch fused backward Pallas kernel.

    UNCHANGED from L1_reduce_inputs. A is recomputed inside the
    kernel from q, k, g.
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

    dq, dk, dv, dg = pl.pallas_call(
        functools.partial(_chunk_gla_bwd_fused_kernel, BT=BT, scale=scale),
        grid=grid,
        out_shape=[
            jax.ShapeDtypeStruct([H, total_NT, BT, K], q.dtype),
            jax.ShapeDtypeStruct([H, total_NT, BT, K], k.dtype),
            jax.ShapeDtypeStruct([H, total_NT, BT, V], v.dtype),
            jax.ShapeDtypeStruct([H, total_NT, BT, K], g_cumsum.dtype),
        ],
        in_specs=[spec_K, spec_K, spec_V, spec_K, spec_h, spec_V, spec_h],
        out_specs=[spec_K, spec_K, spec_V, spec_K],
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
    )(_q, _k, _v, _g, _h, _do, _dh)

    def _unreshape(x, last_dim):
        x = x.reshape(H, B, NT, BT, last_dim)
        x = x.transpose(1, 0, 2, 3, 4)
        x = x.reshape(B, H, T, last_dim)
        return x.transpose(0, 2, 1, 3)

    return _unreshape(dq, K), _unreshape(dk, K), _unreshape(dv, V), _unreshape(dg, K)


# ============================================================
# custom_vjp wrapper
#
# MUTATION: Residuals no longer include A. chunk_gla_fwd now
# returns (g_cumsum, h, o) instead of (g_cumsum, A, h, o).
# Backward already recomputes A internally (L1_reduce_inputs).
# ============================================================


def chunk_gla(q, k, v, g_gamma, scale, chunk_size):
    """Chunked GLA with custom_vjp (Pallas TPU kernels)."""
    @jax.custom_vjp
    def _compute(q, k, v):
        _, _, o = chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size)
        return o

    def _fwd(q, k, v):
        g_cumsum, h, o = chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size)
        # A removed from residuals — backward recomputes it internally
        return o, (q, k, v, g_cumsum, h)

    def _bwd(residuals, do):
        q, k, v, g_cumsum, h = residuals
        B, T, H, K = q.shape
        V = v.shape[-1]
        C = chunk_size
        NT = T // C
        dh = _chunk_bwd_dh_scan(q, do, g_gamma, scale, C, B, T, H, K, V, NT)
        dq, dk, dv, _ = chunk_gla_bwd_fused(q, k, v, g_cumsum, h, do, dh, scale, C)
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
