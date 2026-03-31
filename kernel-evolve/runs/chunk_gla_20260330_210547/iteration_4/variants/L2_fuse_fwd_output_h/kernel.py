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
# Forward: Combined h-propagation + output computation
#
# MUTATION (L2 fuse_fwd_output_h):
# Fuses the two forward pallas_calls into ONE:
#   1. chunk_fwd_h: h state propagation across time chunks
#   2. chunk_gla_fwd_o_gk: output computation from q,k,v,g,h
#
# These were previously separate pallas_calls, forcing h through
# HBM between them. Now h stays in VMEM scratch.
#
# Grid: (B, H, NT) with dimension_semantics = ("parallel", "parallel", "arbitrary")
# Since K=128=BK and V=128=BV for our shapes, no K/V tiling needed.
#
# At each time step i_t:
#   1. Read current h from scratch -> write to h_out[i_t]
#   2. Compute output: recompute A from q,k,g, compute o_inter
#      from q,h, compute o_intra from A,v
#   3. Update h for next step: h = h * decay + k.T @ (v * decay)
#   4. Write o to output
#
# Expected impact: Eliminate one forward pallas_call (~20+ fewer
# computation events), keep h in VMEM instead of HBM round-trip.
# ============================================================


def _chunk_fwd_combined_kernel(
    k_ref, v_ref, q_ref, g_ref,
    g_gamma,
    h_ref, o_ref, scratch_ref,
    *, BT, NT, K, V, scale,
):
    """Combined forward kernel: h propagation + output computation.

    Per grid element (b, h, t), receives one BT-sized time chunk of
    k, v, q, g_cumsum. Maintains h state in VMEM scratch across the
    "arbitrary" time dimension.

    Inputs (4 Refs, tiled per time step):
      k_ref: [1, 1, BT, K]   - key tile for this time step
      v_ref: [1, 1, BT, V]   - value tile for this time step
      q_ref: [1, 1, BT, K]   - query tile for this time step
      g_ref: [1, 1, BT, K]   - g_cumsum tile for this time step

    Scalar prefetch (1):
      g_gamma: [H]           - per-head gate scalar (in SMEM)

    Outputs (2 Refs, tiled per time step):
      h_ref: [1, 1, 1, K, V] - h state snapshot (for backward)
      o_ref: [1, 1, BT, V]   - output for this time step

    Scratch (1):
      scratch_ref: [K, V]    - h state in VMEM (float32)
    """
    i_b = pl.program_id(0)
    i_h = pl.program_id(1)
    i_t = pl.program_id(2)

    # --- Initialize h state at t=0 ---
    @pl.when(i_t == 0)
    def init():
        scratch_ref[:, :] = jnp.zeros((K, V), dtype=jnp.float32)

    # --- Step 1: Save current h to output (backward needs it) ---
    h_ref[0, 0, 0] = scratch_ref[...]

    # --- Step 2: Compute output for this time step ---
    b_q = q_ref[0, 0]      # [BT, K]
    b_k = k_ref[0, 0]      # [BT, K]
    b_v = v_ref[0, 0]      # [BT, V]
    b_g = g_ref[0, 0]      # [BT, K]
    b_h = scratch_ref[...]  # [K, V] float32

    # Phase 0 (VPU): Pre-compute gating scalars
    b_g_f32 = b_g.astype(jnp.float32)
    exp_g = jnp.exp(b_g_f32)                          # [BT, K]
    exp_neg_g = jnp.exp(-b_g_f32)                     # [BT, K]
    b_qg = (b_q * exp_g).astype(b_q.dtype)            # [BT, K]: q * exp(g)
    b_kg = (b_k * exp_neg_g).astype(b_k.dtype)        # [BT, K]: k * exp(-g)

    # Phase 1 (MXU): Recompute A = b_qg @ b_kg.T * scale
    b_A = jnp.dot(b_qg, b_kg.T,
                  precision=jax.lax.Precision.HIGHEST,
                  preferred_element_type=jnp.float32) * scale  # [BT, BT]

    # Phase 2 (VPU): Causal mask
    m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_A_masked = jnp.where(m_s, b_A, 0.0).astype(b_v.dtype)

    # Phase 3 (MXU): Inter-chunk contribution (b_qg @ h * scale)
    b_o_inter = jnp.dot(b_qg, b_h.astype(b_qg.dtype),
                        precision=jax.lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32)
    b_o_inter = b_o_inter * scale

    # Phase 4 (MXU): Intra-chunk contribution (A_masked @ v)
    b_o_intra = jnp.dot(b_A_masked, b_v,
                        precision=jax.lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32)

    # Phase 5 (VPU): Sum and write output
    o_ref[0, 0] = (b_o_inter + b_o_intra).astype(o_ref.dtype)

    # --- Step 3: Update h state for next time step ---
    # h_{t+1} = h_t * exp(g_gamma * BT) + k.T @ (v * exp(g_gamma*BT - g_ramp))
    b_g_gamma_h = g_gamma[i_h].astype(jnp.float32)
    b_g_last = b_g_gamma_h * BT
    b_g_ramp = b_g_gamma_h * (jnp.arange(0, BT, dtype=jnp.float32) + 1)

    scratch_ref[...] = scratch_ref[...] * jnp.exp(b_g_last)

    # v_decay = v * exp(g_gamma*BT - g_ramp)
    v_decay = (b_v * jnp.exp(b_g_last - b_g_ramp)[:, None]).astype(b_v.dtype)

    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        b_k.astype(jnp.float32).T,
        v_decay.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


def chunk_fwd_combined(q, k, v, g_cumsum, g_gamma, scale, chunk_size):
    """Launch combined forward Pallas kernel (h propagation + output).

    MUTATION (L2 fuse_fwd_output_h): Single pallas_call replaces two.
    Eliminates HBM round-trip for h between chunk_fwd_h and chunk_gla_fwd_o_gk.
    """
    BT = chunk_size
    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = T // BT

    # Reshape to [B, H, T, K/V] so BlockSpec can tile BT along T dimension
    # Same layout as original chunk_fwd_h: transpose to [B, H, T, dim]
    k_t = jnp.transpose(k, (0, 2, 1, 3))       # [B, H, T, K]
    v_t = jnp.transpose(v, (0, 2, 1, 3))       # [B, H, T, V]
    q_t = jnp.transpose(q, (0, 2, 1, 3))       # [B, H, T, K]
    g_t = jnp.transpose(g_cumsum, (0, 2, 1, 3)) # [B, H, T, K]

    # h output: [B, H, NT, K, V] - one [K,V] state per time chunk
    # Reshape for BlockSpec tiling: [B, NT, H, K, V] so we can tile
    # by (b, h, t) -> (b, h, t, 0, 0)
    # But we need the output shape to match the BlockSpec dims.
    # Use 5D with (1, 1, K, V) blocks indexed by (b, h, t)
    # Array shape: [B, H, NT, K, V]

    grid = (B, H, NT)

    def kqg_map(b, h, t): return (b, h, t, 0)
    def v_map(b, h, t): return (b, h, t, 0)
    def h_map(b, h, t): return (b, h, t, 0, 0)
    def o_map(b, h, t): return (b, h, t, 0)

    h_all, o = pl.pallas_call(
        functools.partial(
            _chunk_fwd_combined_kernel,
            BT=BT, NT=NT, K=K, V=V, scale=scale,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, 1, BT, K), kqg_map),   # k
                pl.BlockSpec((1, 1, BT, V), v_map),     # v
                pl.BlockSpec((1, 1, BT, K), kqg_map),   # q
                pl.BlockSpec((1, 1, BT, K), kqg_map),   # g_cumsum
            ],
            out_specs=[
                pl.BlockSpec((1, 1, 1, K, V), h_map),   # h_all: [B, H, NT, K, V]
                pl.BlockSpec((1, 1, BT, V), o_map),     # o: [B, H, T, V]
            ],
            scratch_shapes=[pltpu.VMEM((K, V), jnp.float32)],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((B, H, NT, K, V), k.dtype),
            jax.ShapeDtypeStruct((B, H, T, V), v.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            disable_bounds_checks=True,
        ),
    )(g_gamma, k_t, v_t, q_t, g_t)

    # Reshape h_all: [B, H, NT, K, V] -> [B, NT, H, K, V]
    h_all = h_all.transpose(0, 2, 1, 3, 4)

    # Reshape o: [B, H, T, V] -> [B, T, H, V]
    o = o.transpose(0, 2, 1, 3)

    return h_all, o


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
# Forward orchestrator
#
# MUTATION (L2 fuse_fwd_output_h): Single combined pallas_call
# replaces both chunk_fwd_h and chunk_gla_fwd_o_gk.
# Forward is now 1 pallas_call instead of 2.
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

    # MUTATION: Single combined kernel replaces chunk_fwd_h + chunk_gla_fwd_o_gk
    h, o = chunk_fwd_combined(q, k, v, g_cumsum, g_gamma, scale, C)

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
# Backward: Fused dq, dk, dv (Pallas kernel)
#
# MUTATION (combined: reduce_inputs + skip_dg):
#   1. A is recomputed inside the kernel (from L1_reduce_inputs parent)
#   2. dg computation is completely removed (skip_dg)
#
# skip_dg rationale: The caller discards dg via `dq, dk, dv, _ = ...`.
# Removing dg saves:
#   - 1 MXU matmul: M_upper @ dg_raw ([BT,BT] @ [BT,K] = [64,64] @ [64,128])
#   - 1 VPU mask construction (mask_upper, M_upper)
#   - 1 BT*BT intermediate matrix (16KB at float32)
#   - Inter-chunk dg term computation (dgk_inter)
#   - 1 output Ref write (dg_ref)
#
# Combined savings: 3 outputs -> 3 (was 4), 7 inputs (unchanged),
# fewer MXU ops, less register pressure.
#
# MXU-VPU overlap strategy:
#   Phase 0 (VPU): Pre-compute exp values and gated key/query variants
#   Phase 1 (MXU): Recompute A from q_pos and k_neg, then
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
    # -------------------------------------------------------
    exp_pos_g = jnp.exp(b_g)               # [BT, K]
    exp_neg_g = jnp.exp(-b_g)              # [BT, K]
    exp_gn_minus_g = jnp.exp(b_gn[None, :] - b_g)  # [BT, K]

    # Pre-cast gated key/query variants (VPU multiply)
    k_neg = (b_k * exp_neg_g).astype(b_k.dtype)          # [BT, K]: k * exp(-g)
    k_decay = (b_k * exp_gn_minus_g).astype(b_k.dtype)   # [BT, K]: k * exp(gn-g)
    q_pos = (b_q * exp_pos_g).astype(b_q.dtype)          # [BT, K]: q * exp(g)

    # -------------------------------------------------------
    # Phase 1 (MXU): Recompute A from q_pos and k_neg, then
    # compute dA. Both are [BT,K] @ [K,BT] = [BT,BT] dots.
    # -------------------------------------------------------
    b_a = jnp.dot(q_pos, k_neg.T,
                  precision=jax.lax.Precision.HIGHEST,
                  preferred_element_type=jnp.float32) * scale  # [BT, BT]

    b_dA_raw = jnp.dot(b_do.astype(b_v.dtype), b_v.T,
                       precision=jax.lax.Precision.HIGHEST,
                       preferred_element_type=jnp.float32) * scale  # [BT, BT]

    # -------------------------------------------------------
    # Phase 2 (VPU): Apply causal masks
    # -------------------------------------------------------
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_dA = jnp.where(mask, b_dA_raw, 0.0)          # masked dA [BT, BT]
    b_a_masked = jnp.where(mask, b_a, 0.0)         # masked a  [BT, BT]

    # -------------------------------------------------------
    # Phase 3 (MXU batch): Four independent dot products
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
    # Phase 4 (MXU): Intra-chunk dq and dk
    # -------------------------------------------------------
    b_dq_intra_raw = jnp.dot(b_dA.astype(k_neg.dtype), k_neg,
                             precision=jax.lax.Precision.HIGHEST,
                             preferred_element_type=jnp.float32)  # [BT, K]

    b_dk_intra_raw = jnp.dot(b_dA.T.astype(q_pos.dtype), q_pos,
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
# MUTATION (combined):
#   _fwd: residuals no longer include A (fuse_fwd_A)
#   _bwd: chunk_gla_bwd_fused returns 3 values, not 4 (skip_dg)
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
        dh = _chunk_bwd_dh_scan(q, do, g_gamma, scale, C, B, T, H, K, V, NT)
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
