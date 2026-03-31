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
# MUTATION (L2_fwd_store_A): Extends the fused forward kernel to also
# store the masked attention matrix A [BT, BT] as an additional output.
# This eliminates A recomputation (q_pos @ k_neg.T + masking) in backward,
# saving 1 matmul and associated intermediates per time step.
#
# Additional output: A_ref [B, H, NT, BT, BT] — the masked intra-chunk
# attention matrix, stored in float32 for numerical fidelity.
#
# HBM cost: [2, 16, 64, 64, 64] float32 = 33.5MB additional residual.
# Net saving: 1 matmul [BT,BK]@[BK,BT] + causal masking in backward.
# ============================================================


def _chunk_fwd_combined_kernel(
    q_ref, k_ref, v_ref, g_gamma,
    h_ref, o_ref, A_out_ref, scratch_ref,
    *, BT, NT, scale,
):
    """Combined forward kernel: h propagation + output + store A_masked.

    At each time step t:
      1. Save h[t] to output (for backward)
      2. Compute A_masked and o[t] = q_gated @ h[t] * scale + A_masked @ v
      3. Store A_masked to A_out_ref (for backward to skip A recomputation)
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

    # Compute gated q and k for A computation
    exp_g = jnp.exp(b_g_ramp)               # [BT]
    exp_neg_g = jnp.exp(-b_g_ramp)          # [BT]
    b_qg = (b_q * exp_g[:, None]).astype(b_q.dtype)      # [BT, BK]
    b_kg = (b_k * exp_neg_g[:, None]).astype(b_k.dtype)   # [BT, BK]

    # Compute A = b_qg @ b_kg.T * scale  [BT, BT]
    b_A = jnp.dot(b_qg, b_kg.T,
                  precision=lax.Precision.HIGHEST,
                  preferred_element_type=jnp.float32) * scale

    # Causal mask
    m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_A_masked = jnp.where(m_s, b_A, 0.0)

    # --- Step 2b: Store A_masked for backward ---
    # A_out_ref is 5D [B, H, NT, BT, BT] with BlockSpec (1, 1, 1, BT, BT)
    # Index map maps (b, h, _, _, t) -> (b, h, t, 0, 0)
    A_out_ref[0, 0, 0] = b_A_masked.astype(A_out_ref.dtype)

    # Inter-chunk: b_qg @ h * scale  [BT, BV]
    b_o_inter = jnp.dot(b_qg, scratch_ref[...].astype(b_qg.dtype),
                        precision=lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32) * scale

    # Intra-chunk: A_masked @ v  [BT, BV]
    b_o_intra = jnp.dot(b_A_masked.astype(b_v.dtype), b_v,
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
    """Launch combined forward Pallas kernel (h propagation + output + store A).

    MUTATION (L2_fwd_store_A): Adds A_masked [B, H, NT, BT, BT] as a third
    output. This pre-masked attention matrix is passed as a residual to
    backward, eliminating A recomputation (1 matmul + masking) per chunk.

    Returns (h, o, A_masked) where:
      h is [B, NT, H, K, V] for backward residuals
      A_masked is [B, H, NT, BT, BT] float32 for backward residuals
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
    # A_out_map: 5D [B, H, NT, BT, BT]; t maps to NT chunk dimension
    def A_out_map(b, h, ki, vi, t): return b, h, t, 0, 0

    h_all, o_t, A_masked = pl.pallas_call(
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
                pl.BlockSpec((1, NT, 1, BK, BV), h_map),       # h output
                pl.BlockSpec((1, 1, BT, BV), o_map),           # o output
                pl.BlockSpec((1, 1, 1, BT, BT), A_out_map),   # A_masked output
            ],
            scratch_shapes=[pltpu.VMEM((BK, BV), jnp.float32)],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((B, NT, H, K_dim, V), q.dtype),  # h
            jax.ShapeDtypeStruct((B, H, T, V), q.dtype),          # o
            jax.ShapeDtypeStruct((B, H, NT, BT, BT), jnp.float32),  # A_masked
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary", "arbitrary", "arbitrary"),
            disable_bounds_checks=True,
        ),
    )(q_t, k_t, v_t, g_gamma)

    # o_t is [B, H, T, V], need to transpose to [B, T, H, V]
    o = jnp.transpose(o_t, (0, 2, 1, 3))   # [B, T, H, V]

    return h_all, o, A_masked


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
# MUTATION (L2_fwd_store_A): Forward now returns A_masked as additional
# residual for backward. chunk_fwd_combined produces (h, o, A_masked).
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

    # MUTATION (L2_fwd_store_A): Single pallas_call for h propagation,
    # output computation, AND storing masked A matrix
    h, o, A_masked = chunk_fwd_combined(q, k, v, g_gamma, scale, C)

    return g_cumsum, h, A_masked, o


# ============================================================
# Backward: Fused dh + dq/dk/dv (single Pallas kernel)
#
# MUTATION (L2_fwd_store_A): Receives pre-computed A_masked from forward
# as an additional input, eliminating the A recomputation matmul
# (q_pos @ k_neg.T) and causal masking in the backward kernel.
#
# What is eliminated:
#   - b_a = q_pos @ k_neg.T * scale (1 matmul [BT,BK]@[BK,BT])
#   - Causal masking of b_a
#
# What is kept:
#   - q_pos = b_q * exp_pos_g (needed for b_dk_intra_raw and dq final gating)
#   - k_neg = b_k * exp_neg_g (needed for b_dq_intra_raw and dk final gating)
#   - exp_gn_minus_g (needed for k_decay and dk final gating)
#   - b_dA computation (do @ v.T * scale * mask)
#   - All inter-chunk contributions unchanged
#
# A_masked input: 5D [B, H, NT, BT, BT] with BlockSpec (1, 1, 1, BT, BT)
# ============================================================


def _chunk_gla_bwd_store_A_kernel(
    q_ref, k_ref, v_ref, g_ref, h_ref, do_ref, A_ref, g_gamma,
    dq_ref, dk_ref, dv_ref, scratch_ref,
    *, BT, NT, scale,
):
    """Fused backward kernel using pre-stored A_masked from forward.

    Processes chunks in REVERSE time order (inputs pre-flipped).
    At each step i_t (in flipped/reversed time):
      1. Load current dh state from scratch_ref
      2. Compute dq/dk/dv using current dh and pre-stored A_masked
         (skips A recomputation matmul)
      3. Write outputs to dq_ref[0, i_t, 0] (direct 5D slot indexing)
      4. Update dh: dh = dh * state_decay + q_hat.T @ do

    h_ref is 5D [B, H, NT, K, V] with BlockSpec (1, 1, 1, BK, BV).
    A_ref is 5D [B, H, NT, BT, BT] with BlockSpec (1, 1, 1, BT, BT).
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

    # Initialize dh state to zeros at t=0 (first step of REVERSE scan)
    @pl.when(i_t == 0)
    def init():
        scratch_ref[:, :] = jnp.zeros((BK, BV), dtype=jnp.float32)

    # Load inputs for this (reversed) time step
    b_q = q_ref[0, 0]                         # [BT, BK]
    b_k = k_ref[0, 0]                         # [BT, BK]
    b_v = v_ref[0, 0]                         # [BT, BV]
    b_g = g_ref[0, 0].astype(jnp.float32)     # [BT, BK]
    b_h = h_ref[0, 0, 0].astype(jnp.float32)  # [BK, BV]
    b_do = do_ref[0, 0]                        # [BT, BV]

    # Load pre-stored A_masked from forward (ELIMINATES A recomputation)
    # A_ref is 5D [B, H, NT, BT, BT] with BlockSpec (1, 1, 1, BT, BT)
    b_a_masked = A_ref[0, 0, 0]               # [BT, BT] float32

    # Current dh from scratch (accumulated so far in reverse scan)
    b_dh = scratch_ref[...]    # [BK, BV]

    b_gn = b_g[BT - 1, :]  # last row of g_cumsum for this chunk: [K]

    # -------------------------------------------------------
    # Phase 0 (VPU): Pre-compute exp/gate values
    # q_pos and k_neg still needed for dq_intra and dk_intra
    # -------------------------------------------------------
    exp_pos_g = jnp.exp(b_g)                           # [BT, K]
    exp_neg_g = jnp.exp(-b_g)                          # [BT, K]
    exp_gn_minus_g = jnp.exp(b_gn[None, :] - b_g)     # [BT, K]

    k_neg = (b_k * exp_neg_g).astype(b_k.dtype)           # [BT, K]: k * exp(-g)
    k_decay = (b_k * exp_gn_minus_g).astype(b_k.dtype)    # [BT, K]: k * exp(gn-g)
    q_pos = (b_q * exp_pos_g).astype(b_q.dtype)           # [BT, K]: q * exp(g)

    # -------------------------------------------------------
    # Phase 1 (MXU): Compute dA (still needed)
    # ELIMINATED: b_a = q_pos @ k_neg.T * scale (use stored b_a_masked)
    # -------------------------------------------------------
    b_dA_raw = jnp.dot(b_do.astype(b_v.dtype), b_v.T,
                       precision=jax.lax.Precision.HIGHEST,
                       preferred_element_type=jnp.float32) * scale  # [BT, BT]

    # -------------------------------------------------------
    # Phase 2 (VPU): Apply causal mask to dA only
    # b_a_masked is already pre-masked from forward
    # -------------------------------------------------------
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_dA = jnp.where(mask, b_dA_raw, 0.0)

    # -------------------------------------------------------
    # Phase 3 (MXU batch): Four independent dot products
    # Uses b_a_masked (from forward) instead of recomputed b_a
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
    # Still uses k_neg and q_pos (needed for gradient computation)
    # -------------------------------------------------------
    b_dq_intra_raw = jnp.dot(b_dA.astype(k_neg.dtype), k_neg,
                              precision=jax.lax.Precision.HIGHEST,
                              preferred_element_type=jnp.float32)  # [BT, K]

    b_dk_intra_raw = jnp.dot(b_dA.T.astype(q_pos.dtype), q_pos,
                              precision=jax.lax.Precision.HIGHEST,
                              preferred_element_type=jnp.float32)  # [BT, K]

    # -------------------------------------------------------
    # Phase 5 (VPU): Combine results and write to 5D output slots
    # -------------------------------------------------------
    dv_ref[0, i_t, 0] = (b_dv_intra + b_dv_inter).astype(dv_ref.dtype)

    b_dq = b_dq_intra_raw * exp_pos_g + b_dq_inter * (scale * exp_pos_g)
    dq_ref[0, i_t, 0] = b_dq.astype(dq_ref.dtype)

    b_dk = b_dk_intra_raw * exp_neg_g + b_dk_inter * exp_gn_minus_g
    dk_ref[0, i_t, 0] = b_dk.astype(dk_ref.dtype)

    # -------------------------------------------------------
    # Phase 6: Update dh state in scratch for next reverse step
    # dh = dh * exp(g_gamma * BT) + q_hat.T @ do
    # -------------------------------------------------------
    scratch_ref[...] *= exp(b_g_last)

    q_hat = (b_q * exp(b_g_ramp)[:, None] * scale).astype(jnp.float32)

    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        q_hat.T,                           # [BK, BT]
        b_do.astype(jnp.float32),          # [BT, BV]
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


def chunk_gla_bwd_store_A(q, k, v, g_cumsum, h, A_masked, do, g_gamma, scale, chunk_size):
    """Fused backward using pre-stored A_masked from forward.

    MUTATION (L2_fwd_store_A): Receives A_masked [B, H, NT, BT, BT] from
    forward, eliminating the q_pos @ k_neg.T matmul and causal masking
    that were previously recomputed in backward.

    Same reverse-scan architecture as L2_recompute_dh_v2:
      - Inputs q/k/v/g/do pre-flipped along chunk axis
      - h pre-flipped along NT axis
      - A_masked pre-flipped along NT axis (to match reversed scan)
      - Outputs are 5D [B, NT, H, BT, K/V], flipped back after call
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

    # A_masked is already [B, H, NT, BT, BT] -- no transpose needed

    # REVERSE the chunk order to implement reverse scan
    q_flipped = jnp.flip(q_chunked, axis=2)    # [B, H, NT, BT, K]
    k_flipped = jnp.flip(k_chunked, axis=2)    # [B, H, NT, BT, K]
    v_flipped = jnp.flip(v_chunked, axis=2)    # [B, H, NT, BT, V]
    g_flipped = jnp.flip(g_chunked, axis=2)    # [B, H, NT, BT, K]
    do_flipped = jnp.flip(do_chunked, axis=2)  # [B, H, NT, BT, V]
    h_flipped = jnp.flip(h_bhntKV, axis=2)     # [B, H, NT, K, V]
    A_flipped = jnp.flip(A_masked, axis=2)     # [B, H, NT, BT, BT]

    # Flatten q/k/v/g/do back to (B, H, NT*BT, dim) for standard 4D BlockSpec
    q_flat = q_flipped.reshape(B, H, NT * BT, K)
    k_flat = k_flipped.reshape(B, H, NT * BT, K)
    v_flat = v_flipped.reshape(B, H, NT * BT, V)
    g_flat = g_flipped.reshape(B, H, NT * BT, K)
    do_flat = do_flipped.reshape(B, H, NT * BT, V)
    # h stays as 5D [B, H, NT, K, V]
    # A stays as 5D [B, H, NT, BT, BT]

    # Grid: (B, H, K_tiles=1, V_tiles=1, NT)
    grid = (B, H, pl.cdiv(K, BK), pl.cdiv(V, BV), NT)

    # Input index maps (5 grid dims: b, h, ki, vi, t)
    def q_map(b, h, ki, vi, t):  return b, h, t, ki
    def k_map(b, h, ki, vi, t):  return b, h, t, ki
    def v_map(b, h, ki, vi, t):  return b, h, t, vi
    def g_map(b, h, ki, vi, t):  return b, h, t, ki
    def h_map(b, h, ki, vi, t):  return b, h, t, 0, 0
    def do_map(b, h, ki, vi, t): return b, h, t, vi
    # A_map: 5D [B, H, NT, BT, BT] -- t maps to NT chunk index
    def A_map(b, h, ki, vi, t):  return b, h, t, 0, 0

    # Output index maps -- 5D outputs [B, NT, H, BT, K/V]
    def out_k_map(b, h, ki, vi, t): return b, 0, h, 0, 0
    def out_v_map(b, h, ki, vi, t): return b, 0, h, 0, 0

    dq_5d, dk_5d, dv_5d = pl.pallas_call(
        functools.partial(
            _chunk_gla_bwd_store_A_kernel,
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
                pl.BlockSpec((1, 1, 1, BK, BV), h_map),     # h: 5D
                pl.BlockSpec((1, 1, BT, BV), do_map),       # do: 4D
                pl.BlockSpec((1, 1, 1, BT, BT), A_map),     # A_masked: 5D
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
    )(q_flat, k_flat, v_flat, g_flat, h_flipped, do_flat, A_flipped, g_gamma)

    # dq/dk/dv are [B, NT, H, BT, K/V] in reversed-time chunk order.
    # Flip along NT axis (axis=1) to restore original chunk order.
    dq_ordered = jnp.flip(dq_5d, axis=1)   # [B, NT, H, BT, K]
    dk_ordered = jnp.flip(dk_5d, axis=1)   # [B, NT, H, BT, K]
    dv_ordered = jnp.flip(dv_5d, axis=1)   # [B, NT, H, BT, V]

    # Reshape [B, NT, H, BT, K] -> [B, T, H, K].
    dq = dq_ordered.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K)
    dk = dk_ordered.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K)
    dv = dv_ordered.transpose(0, 1, 3, 2, 4).reshape(B, T, H, V)

    return dq, dk, dv


# ============================================================
# custom_vjp wrapper
#
# MUTATION (L2_fwd_store_A):
#   _fwd: uses chunk_fwd_combined which now also outputs A_masked
#   _bwd: uses chunk_gla_bwd_store_A which takes A_masked as input,
#         eliminating the A recomputation matmul in backward
#
# Residuals now include A_masked [B, H, NT, BT, BT] (33.5MB additional).
# Total pallas_calls: 2 (1 forward + 1 backward, same count as base)
# ============================================================


def chunk_gla(q, k, v, g_gamma, scale, chunk_size):
    """Chunked GLA with custom_vjp (Pallas TPU kernels)."""
    @jax.custom_vjp
    def _compute(q, k, v):
        _, _, _, o = chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size)
        return o

    def _fwd(q, k, v):
        g_cumsum, h, A_masked, o = chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size)
        return o, (q, k, v, g_cumsum, h, A_masked)

    def _bwd(residuals, do):
        q, k, v, g_cumsum, h, A_masked = residuals
        # MUTATION (L2_fwd_store_A): Pass pre-stored A_masked to backward,
        # eliminating 1 matmul (q_pos @ k_neg.T) per chunk in backward.
        dq, dk, dv = chunk_gla_bwd_store_A(
            q, k, v, g_cumsum, h, A_masked, do, g_gamma, scale, chunk_size,
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
