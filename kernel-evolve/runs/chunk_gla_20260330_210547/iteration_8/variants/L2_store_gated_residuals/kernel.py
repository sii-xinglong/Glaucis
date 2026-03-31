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
# Forward: Combined h propagation + output computation
#
# MUTATION (store_gated_residuals): Extends the fused forward kernel
# to also output q_pos [B,H,T,K] and k_neg [B,H,T,K] as residuals.
#
# These are the pre-gated arrays already computed in the forward:
#   q_pos = q * exp(g_ramp)
#   k_neg = k * exp(-g_ramp)
#
# Storing them eliminates 3 exp() calls and ~3 large intermediate
# arrays from the backward kernel, reducing register pressure.
#
# Additional change: the backward no longer needs g_cumsum as input.
# All gating is recomputed from g_gamma (a scalar per head in SMEM)
# using b_g_ramp = g_gamma[i_h] * (arange+1), which is a [BT] vector
# broadcast to [BT,K] — much cheaper than loading a full [BT,K] array.
# ============================================================


def _chunk_fwd_combined_kernel(
    q_ref, k_ref, v_ref, g_gamma,
    h_ref, o_ref, qpos_ref, kneg_ref, scratch_ref,
    *, BT, NT, scale,
):
    """Combined forward kernel: h propagation + output + gated residuals.

    At each time step t:
      1. Save h[t] to output (for backward)
      2. Compute o[t] = q_gated @ h[t] * scale + A_masked @ v
         where A is recomputed inline from q, k, g
      3. Save q_pos and k_neg as residuals for backward
      4. Update h[t+1] = h[t] * decay + k.T @ (v * gating)
    """
    BK = k_ref.shape[3]
    BV = v_ref.shape[3]
    i_b, i_h, i_k, i_v, i_t = (
        pl.program_id(0), pl.program_id(1), pl.program_id(2),
        pl.program_id(3), pl.program_id(4),
    )

    # Per-position gating: g_gamma * (1, 2, ..., BT)
    b_g_ramp = g_gamma[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)
    # State decay for one full chunk
    b_g_last = g_gamma[i_h] * BT

    # --- Initialize h state in scratch at t=0 ---
    @pl.when(i_t == 0)
    def init():
        scratch_ref[:, :] = jnp.zeros((BK, BV), dtype=jnp.float32)

    # --- Step 1: Save current h to output (for backward residuals) ---
    h_ref[0, i_t, 0] = scratch_ref[...]

    # --- Step 2: Compute output o[t] ---
    b_q = q_ref[0, 0]   # [BT, BK]
    b_k = k_ref[0, 0]   # [BT, BK]
    b_v = v_ref[0, 0]   # [BT, BV]

    # Compute gated q and k for A recomputation
    exp_g = jnp.exp(b_g_ramp)               # [BT]
    exp_neg_g = jnp.exp(-b_g_ramp)          # [BT]
    b_qg = (b_q * exp_g[:, None]).astype(b_q.dtype)      # [BT, BK]
    b_kg = (b_k * exp_neg_g[:, None]).astype(b_k.dtype)   # [BT, BK]

    # --- Step 2b: Store q_pos and k_neg as residuals ---
    # These are [BT, BK] blocks written to [B, H, T, K] outputs
    qpos_ref[0, 0] = b_qg
    kneg_ref[0, 0] = b_kg

    # Recompute A = b_qg @ b_kg.T * scale  [BT, BT]
    b_A = jnp.dot(b_qg, b_kg.T,
                  precision=lax.Precision.HIGHEST,
                  preferred_element_type=jnp.float32) * scale

    # Causal mask
    m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_A_masked = jnp.where(m_s, b_A, 0.0).astype(b_v.dtype)

    # Inter-chunk: b_qg @ h * scale  [BT, BV]
    b_o_inter = jnp.dot(b_qg, scratch_ref[...].astype(b_qg.dtype),
                        precision=lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32) * scale

    # Intra-chunk: A_masked @ v  [BT, BV]
    b_o_intra = jnp.dot(b_A_masked, b_v,
                        precision=lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32)

    o_ref[0, 0] = (b_o_inter + b_o_intra).astype(o_ref.dtype)

    # --- Step 3: Update h for next time step ---
    # h = h * exp(g_gamma * BT) + k.T @ (v * exp(g_gamma*BT - g_ramp))
    scratch_ref[...] *= exp(b_g_last)

    # v_gated = v * exp(g_gamma*BT - g_ramp)  [BT, BV]
    v_gated = (b_v * jnp.exp(b_g_last - b_g_ramp)[:, None]).astype(b_v.dtype)

    # k.T @ v_gated: [BK, BT] @ [BT, BV] = [BK, BV]
    # Contraction is on BT=64
    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        b_k.astype(jnp.float32).T,       # [BK, BT]
        v_gated.astype(jnp.float32),      # [BT, BV]
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


def chunk_fwd_combined(q, k, v, g_gamma, scale, chunk_size):
    """Launch combined forward Pallas kernel (h propagation + output + gated residuals).

    MUTATION (store_gated_residuals): Extended to also output q_pos and k_neg
    as residuals for the backward pass, eliminating exp() calls in backward.

    Returns (h, o, q_pos, k_neg) where:
      h is [B, NT, H, K, V] for backward residuals
      q_pos is [B, H, T, K] = q * exp(g_ramp)
      k_neg is [B, H, T, K] = k * exp(-g_ramp)
    """
    BK, BV, BT = 128, 128, chunk_size
    B, T, H, K_dim = q.shape
    V = v.shape[-1]
    NT = T // BT

    # Layout: (B, H, T, dim) -- time axis will be "arbitrary"
    q_t = jnp.transpose(q, (0, 2, 1, 3))   # [B, H, T, K]
    k_t = jnp.transpose(k, (0, 2, 1, 3))   # [B, H, T, K]
    v_t = jnp.transpose(v, (0, 2, 1, 3))   # [B, H, T, V]

    grid = (B, H, pl.cdiv(K_dim, BK), pl.cdiv(V, BV), NT)

    # Index maps: all take 5 grid dims (b, h, ki, vi, t)
    def q_map(b, h, ki, vi, t):  return b, h, t, ki
    def k_map(b, h, ki, vi, t):  return b, h, t, ki
    def v_map(b, h, ki, vi, t):  return b, h, t, vi
    def h_map(b, h, ki, vi, t):  return b, 0, h, ki, vi
    def o_map(b, h, ki, vi, t):  return b, h, t, vi
    def qpos_map(b, h, ki, vi, t): return b, h, t, ki
    def kneg_map(b, h, ki, vi, t): return b, h, t, ki

    h_all, o_t, q_pos, k_neg = pl.pallas_call(
        functools.partial(_chunk_fwd_combined_kernel, BT=BT, NT=NT, scale=scale),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, 1, BT, BK), q_map),   # q
                pl.BlockSpec((1, 1, BT, BK), k_map),   # k
                pl.BlockSpec((1, 1, BT, BV), v_map),   # v
                pl.BlockSpec(memory_space=pltpu.SMEM),  # g_gamma
            ],
            out_specs=[
                pl.BlockSpec((1, NT, 1, BK, BV), h_map),  # h output
                pl.BlockSpec((1, 1, BT, BV), o_map),      # o output
                pl.BlockSpec((1, 1, BT, BK), qpos_map),   # q_pos output
                pl.BlockSpec((1, 1, BT, BK), kneg_map),   # k_neg output
            ],
            scratch_shapes=[pltpu.VMEM((BK, BV), jnp.float32)],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((B, NT, H, K_dim, V), q.dtype),  # h
            jax.ShapeDtypeStruct((B, H, T, V), q.dtype),          # o
            jax.ShapeDtypeStruct((B, H, T, K_dim), q.dtype),      # q_pos
            jax.ShapeDtypeStruct((B, H, T, K_dim), q.dtype),      # k_neg
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary", "arbitrary", "arbitrary"),
            disable_bounds_checks=True,
        ),
    )(q_t, k_t, v_t, g_gamma)

    # o_t is [B, H, T, V], need to transpose to [B, T, H, V]
    o = jnp.transpose(o_t, (0, 2, 1, 3))   # [B, T, H, V]

    return h_all, o, q_pos, k_neg


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
# MUTATION (store_gated_residuals): chunk_fwd_combined now returns
# (h, o, q_pos, k_neg). g_cumsum is still computed for API
# compatibility but is no longer passed to backward.
# ============================================================


def chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size):
    """Chunked GLA forward pass."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    # MUTATION (store_gated_residuals): Single pallas_call for
    # h propagation, output computation, AND gated residual storage
    h, o, q_pos, k_neg = chunk_fwd_combined(q, k, v, g_gamma, scale, C)

    return h, o, q_pos, k_neg


# ============================================================
# Backward: Fused dh + dq/dk/dv (single Pallas kernel)
#
# MUTATION (store_gated_residuals): Major changes from recompute_dh_v2:
#   1. REMOVED g_cumsum as an input (was [B,H,NT*BT,K] = large HBM load)
#   2. ADDED q_pos [B,H,T,K] and k_neg [B,H,T,K] as inputs (from forward)
#   3. All exp-gating in backward is now from g_gamma (scalar per head)
#      via b_g_ramp = g_gamma[i_h] * (arange+1), broadcast [BT]->[BT,K]
#   4. k_decay derived from k_neg: k_decay = k_neg * exp(g_gamma*BT)
#   5. Eliminates 3 exp() calls on [BT,K] arrays in backward kernel body
#
# Net HBM change:
#   Removed: g_cumsum load [BT,K] per chunk = 1 tensor load
#   Added:   q_pos load [BT,K] + k_neg load [BT,K] = 2 tensor loads
#   Net: +1 tensor load, BUT 3 fewer exp() calls and fewer live registers
#
# Expected impact: Significant reduction in backward register spills
# (from 2.72M baseline) due to fewer simultaneously live intermediates.
# ============================================================


def _chunk_gla_bwd_store_gated_residuals_kernel(
    q_ref, k_ref, v_ref, qpos_ref, kneg_ref, h_ref, do_ref, g_gamma,
    dq_ref, dk_ref, dv_ref, scratch_ref,
    *, BT, NT, scale,
):
    """Fused backward kernel using stored gated residuals.

    Processes chunks in REVERSE time order (inputs pre-flipped).
    At each step i_t (in flipped/reversed time):
      1. Load current dh state from scratch_ref
      2. Compute dq/dk/dv using current dh, q_pos, k_neg (no g_cumsum)
      3. Write outputs to dq_ref[0, i_t, 0] (direct 5D slot indexing)
      4. Update dh: dh = dh * state_decay + q_pos.T @ do

    Key difference from recompute_dh_v2:
      - q_pos and k_neg come as inputs (pre-computed in forward)
      - exp_pos_g, exp_neg_g, exp_gn_minus_g derived from g_gamma scalar
        via b_g_ramp broadcast, NOT from g_cumsum [BT,K] array
      - k_decay = k_neg * exp(g_gamma * BT) -- no separate exp(gn-g) needed
    """
    BK = q_ref.shape[3]
    BV = do_ref.shape[3]
    i_b, i_h, i_k, i_v, i_t = (
        pl.program_id(0), pl.program_id(1), pl.program_id(2),
        pl.program_id(3), pl.program_id(4),
    )

    # Per-position gating from g_gamma scalar (NO g_cumsum load needed)
    b_g_ramp = g_gamma[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)  # [BT]
    # State decay for one full chunk: exp(g_gamma * BT)
    b_g_last = g_gamma[i_h] * BT  # scalar

    # Initialize dh state to zeros at t=0 (first step of REVERSE scan)
    @pl.when(i_t == 0)
    def init():
        scratch_ref[:, :] = jnp.zeros((BK, BV), dtype=jnp.float32)

    # Load inputs for this (reversed) time step
    b_q = q_ref[0, 0]                         # [BT, BK]
    b_k = k_ref[0, 0]                         # [BT, BK]
    b_v = v_ref[0, 0]                         # [BT, BV]
    # Pre-gated arrays from forward residuals
    b_qpos = qpos_ref[0, 0]                   # [BT, BK]: q * exp(g_ramp)
    b_kneg = kneg_ref[0, 0]                   # [BT, BK]: k * exp(-g_ramp)
    # h_ref is 5D [B, H, NT, K, V]; BlockSpec (1,1,1,BK,BV) maps t->NT dim
    b_h = h_ref[0, 0, 0].astype(jnp.float32)  # [BK, BV]
    b_do = do_ref[0, 0]                        # [BT, BV]

    # Current dh from scratch (accumulated so far in reverse scan)
    b_dh = scratch_ref[...]    # [BK, BV]

    # -------------------------------------------------------
    # Phase 0 (VPU): Compute gating from g_gamma scalar
    # Instead of loading g_cumsum [BT,K] and computing exp(), we
    # compute from b_g_ramp [BT] and broadcast to [BT,K].
    # This eliminates 3 exp() calls on [BT,K] arrays.
    # -------------------------------------------------------
    # exp_pos_g [BT] for final dq gating
    exp_pos_g_vec = jnp.exp(b_g_ramp)          # [BT]
    # exp_neg_g [BT] for final dk gating
    exp_neg_g_vec = jnp.exp(-b_g_ramp)         # [BT]
    # exp(gn - g) [BT] for k_decay and dk gating
    b_gn = b_g_ramp[BT - 1]  # g_gamma * BT = last element of ramp (scalar)
    exp_gn_minus_g_vec = jnp.exp(b_gn - b_g_ramp)  # [BT]

    # k_decay = k * exp(gn - g) = k_neg * exp(gn)
    # Since k_neg[i] = k[i] * exp(-g[i]) and gn = g_gamma * BT:
    #   k_neg[i] * exp(gn) = k[i] * exp(gn - g[i]) = k_decay[i]
    # b_gn is a scalar (last element of b_g_ramp), exp(gn) is scalar broadcast
    k_decay = (b_kneg.astype(jnp.float32) * jnp.exp(b_gn)).astype(b_k.dtype)  # [BT, BK]

    # -------------------------------------------------------
    # Phase 1 (MXU): Recompute A and compute dA
    # Use q_pos and k_neg directly (no exp() needed here)
    # -------------------------------------------------------
    b_a = jnp.dot(b_qpos, b_kneg.T,
                  precision=jax.lax.Precision.HIGHEST,
                  preferred_element_type=jnp.float32) * scale    # [BT, BT]

    b_dA_raw = jnp.dot(b_do.astype(b_v.dtype), b_v.T,
                       precision=jax.lax.Precision.HIGHEST,
                       preferred_element_type=jnp.float32) * scale  # [BT, BT]

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
                         preferred_element_type=jnp.float32)    # [BT, V]

    b_dv_inter = jnp.dot(k_decay, b_dh.astype(b_k.dtype),
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)    # [BT, V]

    b_dq_inter = jnp.dot(b_do, b_h.astype(b_do.dtype).T,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)    # [BT, K]

    b_dk_inter = jnp.dot(b_v, b_dh.astype(b_v.dtype).T,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)    # [BT, K]

    # -------------------------------------------------------
    # Phase 4 (MXU): Intra-chunk dq and dk using pre-gated arrays
    # -------------------------------------------------------
    b_dq_intra_raw = jnp.dot(b_dA.astype(b_kneg.dtype), b_kneg,
                              precision=jax.lax.Precision.HIGHEST,
                              preferred_element_type=jnp.float32)  # [BT, K]

    b_dk_intra_raw = jnp.dot(b_dA.T.astype(b_qpos.dtype), b_qpos,
                              precision=jax.lax.Precision.HIGHEST,
                              preferred_element_type=jnp.float32)  # [BT, K]

    # -------------------------------------------------------
    # Phase 5 (VPU): Combine results and write to 5D output slots
    #
    # MUTATION (store_gated_residuals): exp gating uses [BT] vectors
    # broadcast to [BT,K] instead of loading [BT,K] from g_cumsum.
    # -------------------------------------------------------
    dv_ref[0, i_t, 0] = (b_dv_intra + b_dv_inter).astype(dv_ref.dtype)

    # dq: intra needs exp_pos_g broadcast, inter needs scale * exp_pos_g broadcast
    b_dq = b_dq_intra_raw * exp_pos_g_vec[:, None] + b_dq_inter * (scale * exp_pos_g_vec[:, None])
    dq_ref[0, i_t, 0] = b_dq.astype(dq_ref.dtype)

    # dk: intra needs exp_neg_g broadcast, inter needs exp_gn_minus_g broadcast
    b_dk = b_dk_intra_raw * exp_neg_g_vec[:, None] + b_dk_inter * exp_gn_minus_g_vec[:, None]
    dk_ref[0, i_t, 0] = b_dk.astype(dk_ref.dtype)

    # -------------------------------------------------------
    # Phase 6: Update dh state in scratch for next reverse step
    # dh = dh * exp(g_gamma * BT) + q_pos.T @ do
    # q_pos already has exp(g_ramp) baked in, so q_hat = q_pos * scale
    # -------------------------------------------------------
    scratch_ref[...] *= exp(b_g_last)

    q_hat = (b_qpos.astype(jnp.float32) * scale)  # [BT, BK]

    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        q_hat.T,                           # [BK, BT]
        b_do.astype(jnp.float32),          # [BT, BV]
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


def chunk_gla_bwd_store_gated_residuals(q, k, v, q_pos, k_neg, h, do, g_gamma, scale, chunk_size):
    """Fused backward: single Pallas kernel using stored gated residuals.

    MUTATION (store_gated_residuals): Key differences from recompute_dh_v2:
      1. No g_cumsum input -- all gating from g_gamma scalar
      2. q_pos and k_neg from forward residuals replace exp() computations
      3. k_decay derived from k_neg * exp(g_gamma * BT)
      4. exp_pos_g, exp_neg_g, exp_gn_minus_g are [BT] vectors (broadcast)

    Reverse scan: inputs q/k/v/q_pos/k_neg/do pre-flipped along chunk axis,
    h pre-flipped along NT axis.
    """
    BK, BV, BT = 128, 128, chunk_size
    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = T // BT

    # Transpose to (B, H, T, dim) layout
    q_t = jnp.transpose(q, (0, 2, 1, 3))         # [B, H, T, K]
    k_t = jnp.transpose(k, (0, 2, 1, 3))         # [B, H, T, K]
    v_t = jnp.transpose(v, (0, 2, 1, 3))         # [B, H, T, V]
    do_t = jnp.transpose(do, (0, 2, 1, 3))       # [B, H, T, V]
    # q_pos and k_neg are already [B, H, T, K] from forward

    # Reshape time into chunks: [B, H, NT, BT, dim]
    q_chunked = q_t.reshape(B, H, NT, BT, K)
    k_chunked = k_t.reshape(B, H, NT, BT, K)
    v_chunked = v_t.reshape(B, H, NT, BT, V)
    qpos_chunked = q_pos.reshape(B, H, NT, BT, K)
    kneg_chunked = k_neg.reshape(B, H, NT, BT, K)
    do_chunked = do_t.reshape(B, H, NT, BT, V)

    # h is [B, NT, H, K, V]; transpose to [B, H, NT, K, V]
    h_bhntKV = jnp.transpose(h, (0, 2, 1, 3, 4))  # [B, H, NT, K, V]

    # REVERSE the chunk order to implement reverse scan
    q_flipped = jnp.flip(q_chunked, axis=2)
    k_flipped = jnp.flip(k_chunked, axis=2)
    v_flipped = jnp.flip(v_chunked, axis=2)
    qpos_flipped = jnp.flip(qpos_chunked, axis=2)
    kneg_flipped = jnp.flip(kneg_chunked, axis=2)
    do_flipped = jnp.flip(do_chunked, axis=2)
    h_flipped = jnp.flip(h_bhntKV, axis=2)

    # Flatten q/k/v/qpos/kneg/do back to (B, H, NT*BT, dim) for 4D BlockSpec
    q_flat = q_flipped.reshape(B, H, NT * BT, K)
    k_flat = k_flipped.reshape(B, H, NT * BT, K)
    v_flat = v_flipped.reshape(B, H, NT * BT, V)
    qpos_flat = qpos_flipped.reshape(B, H, NT * BT, K)
    kneg_flat = kneg_flipped.reshape(B, H, NT * BT, K)
    do_flat = do_flipped.reshape(B, H, NT * BT, V)
    # h stays as 5D [B, H, NT, K, V] -- matched by 5D BlockSpec

    # Grid: (B, H, K_tiles=1, V_tiles=1, NT)
    grid = (B, H, pl.cdiv(K, BK), pl.cdiv(V, BV), NT)

    # Input index maps (5 grid dims: b, h, ki, vi, t)
    def q_map(b, h, ki, vi, t):    return b, h, t, ki
    def k_map(b, h, ki, vi, t):    return b, h, t, ki
    def v_map(b, h, ki, vi, t):    return b, h, t, vi
    def qpos_map(b, h, ki, vi, t): return b, h, t, ki
    def kneg_map(b, h, ki, vi, t): return b, h, t, ki
    def h_map(b, h, ki, vi, t):    return b, h, t, 0, 0
    def do_map(b, h, ki, vi, t):   return b, h, t, vi

    # Output index maps -- 5D outputs [B, NT, H, BT, K/V]
    def out_k_map(b, h, ki, vi, t): return b, 0, h, 0, 0
    def out_v_map(b, h, ki, vi, t): return b, 0, h, 0, 0

    dq_5d, dk_5d, dv_5d = pl.pallas_call(
        functools.partial(
            _chunk_gla_bwd_store_gated_residuals_kernel,
            BT=BT, NT=NT, scale=scale,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, 1, BT, BK), q_map),        # q: 4D
                pl.BlockSpec((1, 1, BT, BK), k_map),        # k: 4D
                pl.BlockSpec((1, 1, BT, BV), v_map),        # v: 4D
                pl.BlockSpec((1, 1, BT, BK), qpos_map),     # q_pos: 4D
                pl.BlockSpec((1, 1, BT, BK), kneg_map),     # k_neg: 4D
                pl.BlockSpec((1, 1, 1, BK, BV), h_map),     # h: 5D
                pl.BlockSpec((1, 1, BT, BV), do_map),       # do: 4D
                pl.BlockSpec(memory_space=pltpu.SMEM),       # g_gamma
            ],
            out_specs=[
                pl.BlockSpec((1, NT, 1, BT, BK), out_k_map),  # dq: 5D
                pl.BlockSpec((1, NT, 1, BT, BK), out_k_map),  # dk: 5D
                pl.BlockSpec((1, NT, 1, BT, BV), out_v_map),  # dv: 5D
            ],
            scratch_shapes=[pltpu.VMEM((BK, BV), jnp.float32)],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((B, NT, H, BT, K), q.dtype),   # dq
            jax.ShapeDtypeStruct((B, NT, H, BT, K), k.dtype),   # dk
            jax.ShapeDtypeStruct((B, NT, H, BT, V), v.dtype),   # dv
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary", "arbitrary", "arbitrary"),
            disable_bounds_checks=True,
        ),
    )(q_flat, k_flat, v_flat, qpos_flat, kneg_flat, h_flipped, do_flat, g_gamma)

    # dq/dk/dv are [B, NT, H, BT, K/V] in reversed-time chunk order.
    # Flip along NT axis (axis=1) to restore original chunk order.
    dq_ordered = jnp.flip(dq_5d, axis=1)
    dk_ordered = jnp.flip(dk_5d, axis=1)
    dv_ordered = jnp.flip(dv_5d, axis=1)

    # Reshape [B, NT, H, BT, K] -> [B, T, H, K]
    dq = dq_ordered.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K)
    dk = dk_ordered.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K)
    dv = dv_ordered.transpose(0, 1, 3, 2, 4).reshape(B, T, H, V)

    return dq, dk, dv


# ============================================================
# custom_vjp wrapper
#
# MUTATION (store_gated_residuals):
#   _fwd: uses chunk_fwd_combined which now returns (h, o, q_pos, k_neg)
#         Residuals include q_pos and k_neg; g_cumsum is dropped entirely.
#   _bwd: uses chunk_gla_bwd_store_gated_residuals (no g_cumsum input,
#         receives q_pos and k_neg from forward residuals)
#
# Total pallas_calls: 2 (same as recompute_dh_v2)
# HBM residual change: +2 [B,T,H,K] (q_pos, k_neg) but -1 g_cumsum
#   Net: +1 [B,T,H,K] tensor in residuals
# Backward compute savings: -3 exp() calls on [BT,K], -1 [BT,K] load
# ============================================================


def chunk_gla(q, k, v, g_gamma, scale, chunk_size):
    """Chunked GLA with custom_vjp (Pallas TPU kernels)."""
    @jax.custom_vjp
    def _compute(q, k, v):
        _, o, _, _ = chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size)
        return o

    def _fwd(q, k, v):
        h, o, q_pos, k_neg = chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size)
        return o, (q, k, v, q_pos, k_neg, h)

    def _bwd(residuals, do):
        q, k, v, q_pos, k_neg, h = residuals
        # MUTATION (store_gated_residuals): Single fused pallas_call
        # Uses pre-gated q_pos, k_neg from forward; no g_cumsum needed.
        dq, dk, dv = chunk_gla_bwd_store_gated_residuals(
            q, k, v, q_pos, k_neg, h, do, g_gamma, scale, chunk_size,
        )
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
