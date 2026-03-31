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
# MUTATION (fuse_fwd_combined): Merges the two forward pallas_calls
# (chunk_fwd_h + chunk_gla_fwd_o_gk) into a single pallas_call.
#
# This eliminates:
#   1. One kernel launch overhead
#   2. The h tensor HBM round-trip (h is produced and consumed in VMEM)
#   3. ~20 computation events from the separate kernel boundary
#
# The combined kernel uses grid (B, H, K/BK, V/BV, NT) with time
# as "arbitrary" dimension. At each time step t:
#   1. Save current h to h_ref output (for backward residuals)
#   2. Compute output: o = q_gated @ h * scale + A_masked @ v
#   3. Update h: h = h * decay + k.T @ (v * gating)
#
# VMEM scratch holds h state [BK, BV] in float32 across time steps.
#
# For target shape K=128=BK, V=128=BV, the grid is (2, 16, 1, 1, 64).
# Each tile sees the full h, so no cross-tile reduction is needed.
# ============================================================


def _chunk_fwd_combined_kernel(
    q_ref, k_ref, v_ref, g_gamma,
    h_ref, o_ref, scratch_ref,
    *, BT, NT, scale,
):
    """Combined forward kernel: h propagation + output in one pass.

    At each time step t:
      1. Save h[t] to output (for backward)
      2. Compute o[t] = q_gated @ h[t] * scale + A_masked @ v
         where A is recomputed inline from q, k, g
      3. Update h[t+1] = h[t] * decay + k.T @ (v * gating)
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
    """Launch combined forward Pallas kernel (h propagation + output).

    MUTATION (fuse_fwd_combined): Replaces chunk_fwd_h + chunk_gla_fwd_o_gk
    with a single pallas_call. The h tensor stays in VMEM scratch instead
    of making an HBM round-trip between two separate kernels.

    Returns (h, o) where h is [B, NT, H, K, V] for backward residuals.
    """
    BK, BV, BT = 128, 128, chunk_size
    B, T, H, K_dim = q.shape
    V = v.shape[-1]
    NT = T // BT

    # Layout: (B, H, T, dim) — time axis will be "arbitrary"
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

    h_all, o_t = pl.pallas_call(
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
            ],
            scratch_shapes=[pltpu.VMEM((BK, BV), jnp.float32)],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((B, NT, H, K_dim, V), q.dtype),  # h
            jax.ShapeDtypeStruct((B, H, T, V), q.dtype),          # o
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary", "arbitrary", "arbitrary"),
            disable_bounds_checks=True,
        ),
    )(q_t, k_t, v_t, g_gamma)

    # o_t is [B, H, T, V], need to transpose to [B, T, H, V]
    o = jnp.transpose(o_t, (0, 2, 1, 3))   # [B, T, H, V]

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
# MUTATION (fuse_fwd_combined): Replaced chunk_fwd_h + chunk_gla_fwd_o_gk
# with a single chunk_fwd_combined call.
# This reduces forward from 2 pallas_calls to 1.
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

    # MUTATION (fuse_fwd_combined): Single pallas_call for both
    # h propagation and output computation
    h, o = chunk_fwd_combined(q, k, v, g_gamma, scale, C)

    return g_cumsum, h, o


# ============================================================
# Backward: Fused dh + dq/dk/dv (single Pallas kernel)
#
# MUTATION (recompute_dh_v2): Eliminates the separate chunk_bwd_dh_pallas
# call by fusing dh accumulation into the fused backward kernel.
#
# KEY FIX over Round 6 L2_recompute_dh:
#   The h input is 5D [B, H, NT, K, V]. Round 6 failed with:
#     "Block shape (=(Blocked(1), Blocked(1), Blocked(128), Blocked(128)))
#      must have the same number of dimensions as the array shape
#      (2, 16, 64, 128, 128)"
#   Fix: use 5D BlockSpec for h: (1, 1, 1, BK, BV) with
#        h_map(b, h, ki, vi, t): return (b, h, t, 0, 0)
#
# DIFFERENT OUTPUT STRATEGY vs L2_bwd_fuse_dh_v2:
#   Instead of flattening outputs and flipping after, this variant uses
#   5D output arrays [B, NT, H, BT, K/V] with a block that covers ALL NT.
#   The kernel writes directly to the correct time slot via dq_ref[0, i_t, 0].
#   This avoids reshape/flip overhead on the output side.
#
# Architecture:
#   - Inputs q/k/v/g/do: 4D [B, H, NT*BT, dim], pre-flipped for reverse scan
#   - h input: 5D [B, H, NT, K, V], pre-flipped along axis=2
#   - Output: 5D [B, NT, H, BT, K/V], block shape covers all NT
#     BlockSpec: (1, NT, 1, BT, BK) with out_map(b, h, ki, vi, t): (b, 0, h, 0, 0)
#   - Kernel writes: dq_ref[0, i_t, 0] = ... (index into NT dimension)
#   - After pallas_call: flip outputs along axis=1 (NT), reshape to [B, T, H, dim]
#
# Grid: (B, H, 1, 1, NT) — K_tiles=V_tiles=1 since K=BK=128, V=BV=128
# ============================================================


def _chunk_gla_bwd_recompute_dh_v2_kernel(
    q_ref, k_ref, v_ref, g_ref, h_ref, do_ref, g_gamma,
    dq_ref, dk_ref, dv_ref, scratch_ref,
    *, BT, NT, scale,
):
    """Fused backward kernel with 5D h BlockSpec fix and 5D output layout.

    Processes chunks in REVERSE time order (inputs pre-flipped).
    At each step i_t (in flipped/reversed time):
      1. Load current dh state from scratch_ref
      2. Compute dq/dk/dv using current dh (same math as chunk_gla_bwd_fused)
      3. Write outputs to dq_ref[0, i_t, 0] (direct 5D slot indexing)
      4. Update dh: dh = dh * state_decay + q_hat.T @ do

    h_ref is 5D [B, H, NT, K, V] with BlockSpec (1, 1, 1, BK, BV).
    h_ref[0, 0, 0] gives the [BK, BV] block for the current time slot.

    dq/dk/dv refs are 5D [B, NT, H, BT, K/V] with block (1, NT, 1, BT, BK/BV).
    dq_ref[0, i_t, 0] writes [BT, BK] to time slot i_t.
    """
    BK = q_ref.shape[3]
    BV = do_ref.shape[3]
    i_b, i_h, i_k, i_v, i_t = (
        pl.program_id(0), pl.program_id(1), pl.program_id(2),
        pl.program_id(3), pl.program_id(4),
    )

    # Per-position gating: g_gamma * (1, 2, ..., BT)
    b_g_ramp = g_gamma[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)
    # State decay for one full chunk: exp(g_gamma * BT)
    b_g_last = g_gamma[i_h] * BT

    # Initialize dh state to zeros at t=0 (first step of REVERSE scan,
    # i.e. the END of the original sequence because inputs are flipped)
    @pl.when(i_t == 0)
    def init():
        scratch_ref[:, :] = jnp.zeros((BK, BV), dtype=jnp.float32)

    # Load inputs for this (reversed) time step
    # q/k/v/g/do are 4D [B, H, NT*BT, dim]; BlockSpec (1,1,BT,dim) maps t->chunk
    b_q = q_ref[0, 0]                         # [BT, BK]
    b_k = k_ref[0, 0]                         # [BT, BK]
    b_v = v_ref[0, 0]                         # [BT, BV]
    b_g = g_ref[0, 0].astype(jnp.float32)     # [BT, BK]
    # h_ref is 5D [B, H, NT, K, V]; BlockSpec (1,1,1,BK,BV) maps t->NT dim
    b_h = h_ref[0, 0, 0].astype(jnp.float32)  # [BK, BV]
    b_do = do_ref[0, 0]                        # [BT, BV]

    # Current dh from scratch (accumulated so far in reverse scan)
    b_dh = scratch_ref[...]    # [BK, BV]

    b_gn = b_g[BT - 1, :]  # last row of g_cumsum for this chunk: [K]

    # -------------------------------------------------------
    # Phase 0 (VPU): Pre-compute ALL exp/gate values upfront
    # -------------------------------------------------------
    exp_pos_g = jnp.exp(b_g)                           # [BT, K]
    exp_neg_g = jnp.exp(-b_g)                          # [BT, K]
    exp_gn_minus_g = jnp.exp(b_gn[None, :] - b_g)     # [BT, K]

    k_neg = (b_k * exp_neg_g).astype(b_k.dtype)           # [BT, K]: k * exp(-g)
    k_decay = (b_k * exp_gn_minus_g).astype(b_k.dtype)    # [BT, K]: k * exp(gn-g)
    q_pos = (b_q * exp_pos_g).astype(b_q.dtype)           # [BT, K]: q * exp(g)

    # -------------------------------------------------------
    # Phase 1 (MXU): Recompute A and compute dA
    # -------------------------------------------------------
    b_a = jnp.dot(q_pos, k_neg.T,
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
    # Phase 4 (MXU): Intra-chunk dq and dk
    # -------------------------------------------------------
    b_dq_intra_raw = jnp.dot(b_dA.astype(k_neg.dtype), k_neg,
                              precision=jax.lax.Precision.HIGHEST,
                              preferred_element_type=jnp.float32)  # [BT, K]

    b_dk_intra_raw = jnp.dot(b_dA.T.astype(q_pos.dtype), q_pos,
                              precision=jax.lax.Precision.HIGHEST,
                              preferred_element_type=jnp.float32)  # [BT, K]

    # -------------------------------------------------------
    # Phase 5 (VPU): Combine results and write to 5D output slots
    #
    # Output refs are 5D [B, NT, H, BT, K/V] with block (1, NT, 1, BT, BK/BV).
    # dq_ref[0, i_t, 0] indexes into the NT dimension at position i_t.
    # This writes [BT, BK] directly to the correct (reversed) time slot.
    # -------------------------------------------------------
    dv_ref[0, i_t, 0] = (b_dv_intra + b_dv_inter).astype(dv_ref.dtype)

    b_dq = b_dq_intra_raw * exp_pos_g + b_dq_inter * (scale * exp_pos_g)
    dq_ref[0, i_t, 0] = b_dq.astype(dq_ref.dtype)

    b_dk = b_dk_intra_raw * exp_neg_g + b_dk_inter * exp_gn_minus_g
    dk_ref[0, i_t, 0] = b_dk.astype(dk_ref.dtype)

    # -------------------------------------------------------
    # Phase 6: Update dh state in scratch for next reverse step
    # dh = dh * exp(g_gamma * BT) + q_hat.T @ do
    # NOTE: g_ramp here uses g_gamma directly (not g_cumsum)
    # -------------------------------------------------------
    scratch_ref[...] *= exp(b_g_last)

    q_hat = (b_q * exp(b_g_ramp)[:, None] * scale).astype(jnp.float32)

    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        q_hat.T,                           # [BK, BT]
        b_do.astype(jnp.float32),          # [BT, BV]
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


def chunk_gla_bwd_recompute_dh_v2(q, k, v, g_cumsum, h, do, g_gamma, scale, chunk_size):
    """Fused backward: single Pallas kernel combining dh + dq/dk/dv.

    MUTATION (recompute_dh_v2): Fixes Round 6 L2_recompute_dh which failed
    because h is 5D [B, H, NT, K, V] but BlockSpec was 4D.

    Key differences from L2_recompute_dh (Round 6):
      1. h BlockSpec is now 5D: (1, 1, 1, BK, BV) with h_map returning 5 indices
      2. Output arrays are 5D [B, NT, H, BT, K/V] — kernel writes via dq_ref[0, i_t, 0]
      3. No output flipping needed — the 5D block covers all NT slots, kernel
         writes to the correct slot directly; we flip the NT axis after

    Reverse scan: inputs q/k/v/g/do pre-flipped along chunk axis (axis=2 in
    [B, H, NT, BT, dim] view), h pre-flipped along NT axis (axis=2 in
    [B, H, NT, K, V]). Output flipped back along NT axis after the call.
    """
    BK, BV, BT = 128, 128, chunk_size
    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = T // BT

    # Transpose to (B, H, T, dim) layout
    q_t = jnp.transpose(q, (0, 2, 1, 3))         # [B, H, T, K]
    k_t = jnp.transpose(k, (0, 2, 1, 3))         # [B, H, T, K]
    v_t = jnp.transpose(v, (0, 2, 1, 3))         # [B, H, T, V]
    g_t = jnp.transpose(g_cumsum, (0, 2, 1, 3))  # [B, H, T, K]
    do_t = jnp.transpose(do, (0, 2, 1, 3))       # [B, H, T, V]

    # Reshape time into chunks: [B, H, NT, BT, dim]
    q_chunked = q_t.reshape(B, H, NT, BT, K)
    k_chunked = k_t.reshape(B, H, NT, BT, K)
    v_chunked = v_t.reshape(B, H, NT, BT, V)
    g_chunked = g_t.reshape(B, H, NT, BT, K)
    do_chunked = do_t.reshape(B, H, NT, BT, V)

    # h is [B, NT, H, K, V]; transpose to [B, H, NT, K, V]
    h_bhntKV = jnp.transpose(h, (0, 2, 1, 3, 4))  # [B, H, NT, K, V]

    # REVERSE the chunk order to implement reverse scan
    # After flip: position 0 = last chunk, position NT-1 = first chunk
    q_flipped = jnp.flip(q_chunked, axis=2)    # [B, H, NT, BT, K]
    k_flipped = jnp.flip(k_chunked, axis=2)    # [B, H, NT, BT, K]
    v_flipped = jnp.flip(v_chunked, axis=2)    # [B, H, NT, BT, V]
    g_flipped = jnp.flip(g_chunked, axis=2)    # [B, H, NT, BT, K]
    do_flipped = jnp.flip(do_chunked, axis=2)  # [B, H, NT, BT, V]
    h_flipped = jnp.flip(h_bhntKV, axis=2)    # [B, H, NT, K, V]

    # Flatten q/k/v/g/do back to (B, H, NT*BT, dim) for standard 4D BlockSpec
    q_flat = q_flipped.reshape(B, H, NT * BT, K)
    k_flat = k_flipped.reshape(B, H, NT * BT, K)
    v_flat = v_flipped.reshape(B, H, NT * BT, V)
    g_flat = g_flipped.reshape(B, H, NT * BT, K)
    do_flat = do_flipped.reshape(B, H, NT * BT, V)
    # h stays as 5D [B, H, NT, K, V] — matched by 5D BlockSpec

    # Grid: (B, H, K_tiles=1, V_tiles=1, NT)
    # K=128=BK and V=128=BV so K_tiles=V_tiles=1
    grid = (B, H, pl.cdiv(K, BK), pl.cdiv(V, BV), NT)

    # Input index maps (5 grid dims: b, h, ki, vi, t)
    def q_map(b, h, ki, vi, t):  return b, h, t, ki    # 4D [B, H, NT*BT->t, K->ki]
    def k_map(b, h, ki, vi, t):  return b, h, t, ki
    def v_map(b, h, ki, vi, t):  return b, h, t, vi
    def g_map(b, h, ki, vi, t):  return b, h, t, ki
    # h_map: 5D [B, H, NT, K, V] — t maps to NT chunk index
    def h_map(b, h, ki, vi, t):  return b, h, t, 0, 0
    def do_map(b, h, ki, vi, t): return b, h, t, vi

    # Output index maps — 5D outputs [B, NT, H, BT, K/V]
    # Block (1, NT, 1, BT, BK/BV) covers all NT; index_map returns (b, 0, h, 0, 0)
    # The kernel uses dq_ref[0, i_t, 0] to write to the correct time slot.
    def out_k_map(b, h, ki, vi, t): return b, 0, h, 0, 0
    def out_v_map(b, h, ki, vi, t): return b, 0, h, 0, 0

    dq_5d, dk_5d, dv_5d = pl.pallas_call(
        functools.partial(
            _chunk_gla_bwd_recompute_dh_v2_kernel,
            BT=BT, NT=NT, scale=scale,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, 1, BT, BK), q_map),        # q: 4D
                pl.BlockSpec((1, 1, BT, BK), k_map),        # k: 4D
                pl.BlockSpec((1, 1, BT, BV), v_map),        # v: 4D
                pl.BlockSpec((1, 1, BT, BK), g_map),        # g: 4D
                pl.BlockSpec((1, 1, 1, BK, BV), h_map),     # h: 5D FIX
                pl.BlockSpec((1, 1, BT, BV), do_map),       # do: 4D
                pl.BlockSpec(memory_space=pltpu.SMEM),      # g_gamma
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
    )(q_flat, k_flat, v_flat, g_flat, h_flipped, do_flat, g_gamma)

    # dq/dk/dv are [B, NT, H, BT, K/V] in reversed-time chunk order.
    # Flip along NT axis (axis=1) to restore original chunk order.
    dq_ordered = jnp.flip(dq_5d, axis=1)   # [B, NT, H, BT, K]
    dk_ordered = jnp.flip(dk_5d, axis=1)   # [B, NT, H, BT, K]
    dv_ordered = jnp.flip(dv_5d, axis=1)   # [B, NT, H, BT, V]

    # Reshape [B, NT, H, BT, K] -> [B, T, H, K].
    # NT and BT are not adjacent (H sits between them), so we must transpose
    # to bring NT and BT together before merging:
    #   [B, NT, H, BT, K] -> transpose(0,1,3,2,4) -> [B, NT, BT, H, K]
    #   -> reshape(B, T, H, K) merges NT*BT=T correctly.
    dq = dq_ordered.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K)
    dk = dk_ordered.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K)
    dv = dv_ordered.transpose(0, 1, 3, 2, 4).reshape(B, T, H, V)

    return dq, dk, dv


# ============================================================
# custom_vjp wrapper
#
# MUTATION (recompute_dh_v2 + fuse_fwd_combined):
#   _fwd: uses chunk_fwd_combined (single pallas_call for forward)
#   _bwd: uses chunk_gla_bwd_recompute_dh_v2 (single pallas_call
#         for backward — fuses chunk_bwd_dh_pallas + chunk_gla_bwd_fused)
#         With fixed 5D h BlockSpec and 5D output layout.
#
# Total pallas_calls: 2 (was 3 in L2_fuse_fwd_combined lineage)
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
        # MUTATION (recompute_dh_v2): Single fused pallas_call replaces
        # chunk_bwd_dh_pallas + chunk_gla_bwd_fused (2 calls -> 1 call)
        # Uses 5D h BlockSpec fix + 5D output layout for direct slot writes.
        dq, dk, dv = chunk_gla_bwd_recompute_dh_v2(
            q, k, v, g_cumsum, h, do, g_gamma, scale, chunk_size,
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
