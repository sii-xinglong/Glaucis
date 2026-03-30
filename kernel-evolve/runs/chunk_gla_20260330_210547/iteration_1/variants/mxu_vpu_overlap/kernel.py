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
# MXU-VPU overlap optimization:
#   - Pre-compute all VPU exp/masking values before MXU section
#   - Issue inter-chunk MXU dot first, then apply mask (VPU) to
#     prepare intra-chunk inputs, then issue intra-chunk MXU dot
#   - The mask (VPU comparison + where) overlaps with inter-chunk
#     MXU result draining through the pipeline
# ============================================================


def _chunk_gla_fwd_o_gk_pl(q_ref, v_ref, g_ref, h_ref, A_ref, o_ref, *, BT, scale):
    b_q = q_ref[0, 0]
    b_g = g_ref[0, 0]
    b_v = v_ref[0, 0]
    b_h = h_ref[0, 0]
    b_A = A_ref[0, 0]

    # --- VPU phase: pre-compute all gating scalars before MXU section ---
    b_g_f32 = b_g.astype(jnp.float32)
    exp_g = jnp.exp(b_g_f32)          # VPU: [BT, K] element-wise
    b_qg = (b_q * exp_g).astype(b_q.dtype)  # VPU: scale q by gate

    # --- MXU phase 1: inter-chunk contribution (b_qg @ h) ---
    # This is the larger MXU op (BT x K) @ (K x V) = BT x V
    b_o_inter = jnp.dot(b_qg, b_h.astype(b_qg.dtype),
                        precision=jax.lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32)
    b_o_inter = b_o_inter * scale

    # --- VPU phase: causal mask and A masking ---
    # These VPU ops (index comparison, where) can overlap with the
    # inter-chunk MXU result flowing through the pipeline
    m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_A_masked = jnp.where(m_s, b_A, 0.0).astype(b_v.dtype)

    # --- MXU phase 2: intra-chunk contribution (A_masked @ v) ---
    # Second independent MXU op issued after VPU mask is ready
    b_o_intra = jnp.dot(b_A_masked, b_v,
                        precision=jax.lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32)

    # --- VPU phase: sum and write ---
    o_ref[0, 0] = (b_o_inter + b_o_intra).astype(o_ref.dtype)


def chunk_gla_fwd_o_gk(q, v, g_cumsum, A, h, scale, chunk_size):
    """Launch output combination Pallas kernel."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _g = g_cumsum.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _v = v.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _A = A.reshape(B, NT, BT, H, BT).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, BT)
    _h = h.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)

    q_spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    v_spec = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))
    g_spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    h_spec = pl.BlockSpec([1, 1, K, V], index_map=lambda h, nt: (h, nt, 0, 0))
    A_spec = pl.BlockSpec([1, 1, BT, BT], index_map=lambda h, nt: (h, nt, 0, 0))
    o_spec = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))

    o = pl.pallas_call(
        functools.partial(_chunk_gla_fwd_o_gk_pl, BT=BT, scale=scale),
        grid=(H, total_NT),
        out_shape=jax.ShapeDtypeStruct([H, total_NT, BT, V], v.dtype),
        in_specs=[q_spec, v_spec, g_spec, h_spec, A_spec],
        out_specs=o_spec,
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
    )(_q, _v, _g, _h, _A)

    o = o.reshape(H, B, NT, BT, V).transpose(1, 0, 2, 3, 4)
    o = o.reshape(B, H, NT * BT, V).transpose(0, 2, 1, 3)
    return o


# ============================================================
# Forward orchestrator
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
    A = chunk_gla_fwd_intra_gk(q, k, g_cumsum, scale, C)
    o = chunk_gla_fwd_o_gk(q, v, g_cumsum, A, h, scale, C)

    return g_cumsum, A, h, o


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
# MXU-VPU overlap optimization strategy:
#
# Baseline issue: VPU ops (exp, masking, element-wise multiply)
# are interleaved with MXU ops (dot products) in a way that creates
# a sequential dependency chain: VPU result -> MXU -> VPU -> MXU...
#
# Optimization: Hoist ALL VPU exp computations to a single pre-compute
# phase before ANY MXU work begins. This gives the compiler maximum
# freedom to schedule MXU and VPU units concurrently:
#
#   Phase 0 (VPU): Pre-compute exp_pos_g, exp_neg_g, exp_gn_minus_g,
#                  k_neg, k_decay, q_pos — all element-wise, independent
#
#   Phase 1 (MXU): dA = do @ v.T  [BT x BT dot — feeds mask then dq/dk]
#
#   Phase 2 (VPU): Apply causal mask to dA, apply mask to b_a
#                  These overlap with any MXU pipeline drain from Phase 1
#
#   Phase 3 (MXU batch): Issue independent dot products back-to-back:
#     - dv_intra = a_masked.T @ do   [BT x BT] @ [BT x V]
#     - dv_inter = k_decay @ dh      [BT x K] @ [K x V]  -- independent
#     - dq_inter = do @ h.T          [BT x V] @ [V x K]  -- independent
#     - dk_inter = v @ dh.T          [BT x V] @ [V x K]  -- independent
#   Back-to-back MXU ops allow compiler to pipeline on dual-MXU paths
#
#   Phase 4 (MXU): dq_intra = dA_masked @ k_neg  [BT x BT] @ [BT x K]
#                  dk_intra = dA_masked.T @ q_pos [BT x BT] @ [BT x K]
#   These depend on dA from Phase 1, issued together for pipelining
#
#   Phase 5 (VPU): Combine: dq = dq_intra*exp_g + dq_inter*scale*exp_g
#                           dk = dk_intra*exp_neg_g + dk_inter
#                           dg = rev_cumsum(q*dq - k*dk) + dgk_inter
#   VPU combination can overlap with MXU from Phase 4
# ============================================================


def _chunk_gla_bwd_fused_kernel(
    q_ref, k_ref, v_ref, g_ref, h_ref, a_ref, do_ref, dh_ref,
    dq_ref, dk_ref, dv_ref, dg_ref,
    *, BT, scale,
):
    b_q = q_ref[0, 0]
    b_k = k_ref[0, 0]
    b_v = v_ref[0, 0]
    b_g = g_ref[0, 0].astype(jnp.float32)
    b_h = h_ref[0, 0].astype(jnp.float32)
    b_a = a_ref[0, 0].astype(jnp.float32)
    b_do = do_ref[0, 0]
    b_dh = dh_ref[0, 0].astype(jnp.float32)

    b_gn = b_g[BT - 1, :]  # last row: [K]

    # -------------------------------------------------------
    # Phase 0 (VPU): Pre-compute ALL exp/gate values upfront
    # This batches all VPU element-wise ops together so the
    # compiler can keep VPU busy during MXU pipeline latency.
    # -------------------------------------------------------
    exp_pos_g = jnp.exp(b_g)               # [BT, K] — used by dq_intra, dk_intra, dg
    exp_neg_g = jnp.exp(-b_g)              # [BT, K] — used by k_neg (dq), dk_intra
    exp_gn_minus_g = jnp.exp(b_gn[None, :] - b_g)  # [BT, K] — used by k_decay (dv), dk_inter

    # Pre-cast gated key/query variants (VPU multiply)
    k_neg = (b_k * exp_neg_g).astype(b_k.dtype)          # [BT, K]: k * exp(-g)
    k_decay = (b_k * exp_gn_minus_g).astype(b_k.dtype)   # [BT, K]: k * exp(gn-g)
    q_pos = (b_q * exp_pos_g).astype(b_q.dtype)          # [BT, K]: q * exp(g)

    # -------------------------------------------------------
    # Phase 1 (MXU): Compute dA — feeds everything downstream
    # dA[i,j] = (do[i,:] . v[j,:]) * scale, causal-masked
    # -------------------------------------------------------
    b_dA_raw = jnp.dot(b_do.astype(b_v.dtype), b_v.T,
                       precision=jax.lax.Precision.HIGHEST,
                       preferred_element_type=jnp.float32) * scale  # [BT, BT]

    # -------------------------------------------------------
    # Phase 2 (VPU): Apply causal masks
    # Overlaps with any MXU drain from Phase 1
    # -------------------------------------------------------
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_dA = jnp.where(mask, b_dA_raw, 0.0)          # masked dA [BT, BT]
    b_a_masked = jnp.where(mask, b_a, 0.0)         # masked a  [BT, BT]

    # -------------------------------------------------------
    # Phase 3 (MXU batch): Issue four independent dot products
    # back-to-back. None depend on each other — compiler can
    # pipeline across dual-MXU execution slots.
    #
    # dv_intra: [BT,BT]^T @ [BT,V]  =>  [BT, V]
    # dv_inter: [BT,K]   @ [K, V]   =>  [BT, V]  (independent of dv_intra)
    # dq_inter: [BT,V]   @ [V, K]   =>  [BT, K]  (fully independent)
    # dk_inter: [BT,V]   @ [V, K]   =>  [BT, K]  (fully independent)
    # -------------------------------------------------------
    b_dv_intra = jnp.dot(b_a_masked.T.astype(b_do.dtype), b_do,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)   # [BT, V]

    b_dv_inter = jnp.dot(k_decay, b_dh.astype(b_k.dtype),
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)   # [BT, V]

    b_dq_inter = jnp.dot(b_do, b_h.astype(b_do.dtype).T,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)   # [BT, K]

    b_dk_inter = jnp.dot(b_v, b_dh.astype(b_v.dtype).T,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)   # [BT, K]

    # -------------------------------------------------------
    # Phase 4 (MXU): Intra-chunk dq and dk — both depend on
    # b_dA from Phase 1. Issue back-to-back for pipelining.
    # dq_intra: [BT,BT] @ [BT,K]  =>  [BT, K]
    # dk_intra: [BT,BT]^T @ [BT,K] => [BT, K]
    # -------------------------------------------------------
    b_dq_intra_raw = jnp.dot(b_dA.astype(k_neg.dtype), k_neg,
                             precision=jax.lax.Precision.HIGHEST,
                             preferred_element_type=jnp.float32)  # [BT, K]

    b_dk_intra_raw = jnp.dot(b_dA.T.astype(q_pos.dtype), q_pos,
                             precision=jax.lax.Precision.HIGHEST,
                             preferred_element_type=jnp.float32)  # [BT, K]

    # -------------------------------------------------------
    # Phase 5 (VPU): Combine results and write outputs
    # These element-wise ops can overlap with MXU from Phase 4
    # draining through the accumulator pipeline.
    # -------------------------------------------------------

    # dv: combine intra + inter (simple addition, VPU)
    dv_ref[0, 0] = (b_dv_intra + b_dv_inter).astype(dv_ref.dtype)

    # dq: scale intra by exp(g), scale inter by scale*exp(g)
    b_dq = b_dq_intra_raw * exp_pos_g + b_dq_inter * (scale * exp_pos_g)
    dq_ref[0, 0] = b_dq.astype(dq_ref.dtype)

    # dk: scale intra by exp(-g), inter already scaled correctly
    b_dk = b_dk_intra_raw * exp_neg_g + b_dk_inter * exp_gn_minus_g
    dk_ref[0, 0] = b_dk.astype(dk_ref.dtype)

    # dg: reverse cumsum of (q*dq - k*dk) plus inter-chunk term
    # This MXU (M_upper @ dg_raw) is the dg cumsum — unavoidable dependency
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
    """Launch fused backward Pallas kernel."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    A = chunk_gla_fwd_intra_gk(q, k, g_cumsum, scale, BT)

    _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _k = k.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _v = v.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _g = g_cumsum.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _do = do.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _h = h.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)
    _dh = dh.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)
    _A = A.reshape(B, NT, BT, H, BT).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, BT)

    grid = (H, total_NT)
    spec_K = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_V = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_h = pl.BlockSpec([1, 1, K, V], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_A = pl.BlockSpec([1, 1, BT, BT], index_map=lambda h, nt: (h, nt, 0, 0))

    dq, dk, dv, dg = pl.pallas_call(
        functools.partial(_chunk_gla_bwd_fused_kernel, BT=BT, scale=scale),
        grid=grid,
        out_shape=[
            jax.ShapeDtypeStruct([H, total_NT, BT, K], q.dtype),
            jax.ShapeDtypeStruct([H, total_NT, BT, K], k.dtype),
            jax.ShapeDtypeStruct([H, total_NT, BT, V], v.dtype),
            jax.ShapeDtypeStruct([H, total_NT, BT, K], g_cumsum.dtype),
        ],
        in_specs=[spec_K, spec_K, spec_V, spec_K, spec_h, spec_A, spec_V, spec_h],
        out_specs=[spec_K, spec_K, spec_V, spec_K],
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
    )(_q, _k, _v, _g, _h, _A, _do, _dh)

    def _unreshape(x, last_dim):
        x = x.reshape(H, B, NT, BT, last_dim)
        x = x.transpose(1, 0, 2, 3, 4)
        x = x.reshape(B, H, T, last_dim)
        return x.transpose(0, 2, 1, 3)

    return _unreshape(dq, K), _unreshape(dk, K), _unreshape(dv, V), _unreshape(dg, K)


# ============================================================
# custom_vjp wrapper
# ============================================================


def chunk_gla(q, k, v, g_gamma, scale, chunk_size):
    """Chunked GLA with custom_vjp (Pallas TPU kernels)."""
    @jax.custom_vjp
    def _compute(q, k, v):
        _, _, _, o = chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size)
        return o

    def _fwd(q, k, v):
        g_cumsum, A, h, o = chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size)
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
