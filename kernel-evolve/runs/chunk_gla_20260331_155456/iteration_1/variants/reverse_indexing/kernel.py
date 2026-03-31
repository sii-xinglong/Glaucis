"""Chunked GLA (Gated Linear Attention) Pallas TPU kernel — reverse_indexing variant.

Combines TWO HBM-reduction mutations on the baseline:
  1. bf16 residuals: store q/k/v as bf16 in _fwd residuals (saves 96MB),
     cast back to f32 in _bwd before backward computation.
  2. Eliminate flips: replace jnp.flip() + reversed data with reversed
     BlockSpec index_maps (NT-1-t), removing 5 flip HBM copies and the
     chunk->flip->flatten reshape chain from the backward host code.

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
# MUTATION (eliminate_gcumsum): g_cumsum is no longer computed or
# stored as a residual. The backward kernel recomputes gating
# from the g_gamma scalar directly, saving ~67MB HBM.
# ============================================================


def chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size):
    """Chunked GLA forward pass.

    MUTATION (eliminate_gcumsum): g_cumsum removed from residuals.
    The backward kernel reconstructs gating from g_gamma scalar.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    # MUTATION (fuse_fwd_combined): Single pallas_call for both
    # h propagation and output computation
    h, o = chunk_fwd_combined(q, k, v, g_gamma, scale, C)

    return h, o


# ============================================================
# Backward: Fused dh + dq/dk/dv (single Pallas kernel)
#
# MUTATION (reverse_indexing): Combines two optimizations:
#   1. eliminate_gcumsum: g_cumsum removed, gating recomputed
#      from g_gamma scalar as [BT] vectors
#   2. eliminate_flip: reversed BlockSpec index_maps (NT-1-t)
#      replace jnp.flip() calls, eliminating 5 HBM flip copies
#      and the chunk->flip->flatten reshape chain
#
# MUTATION (bf16_residuals): q/k/v stored as bf16 in residuals,
#   cast back to f32 in _bwd before backward computation
# ============================================================


def _chunk_gla_bwd_kernel(
    q_ref, k_ref, v_ref, h_ref, do_ref, g_gamma,
    dq_ref, dk_ref, dv_ref, scratch_ref,
    *, BT, NT, scale,
):
    """Fused backward kernel with reversed index_maps (no flip).

    MUTATION (eliminate_flip): Inputs are NOT flipped. Instead,
    BlockSpec index_maps use NT-1-t to reverse time order.
    Outputs are written to NT-1-i_t position to produce correct order.

    MUTATION (eliminate_gcumsum): g_ref removed from inputs.
    Gating recomputed from g_gamma scalar as [BT] vectors.

    Processes chunks in REVERSE time order via reversed index_maps.
    At each step i_t (grid position 0..NT-1, maps to chunk NT-1-i_t..0):
      1. Load current dh state from scratch_ref
      2. Compute dq/dk/dv using current dh
      3. Write outputs to dq_ref[0, NT-1-i_t, 0] (reversed position)
      4. Update dh: dh = dh * state_decay + q_hat.T @ do
    """
    BK = q_ref.shape[3]
    BV = do_ref.shape[3]
    i_b, i_h, i_k, i_v, i_t = (
        pl.program_id(0), pl.program_id(1), pl.program_id(2),
        pl.program_id(3), pl.program_id(4),
    )

    # MUTATION (eliminate_gcumsum): Recompute gating from g_gamma scalar
    b_g_ramp = g_gamma[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)  # [BT]
    b_g_last = g_gamma[i_h].astype(jnp.float32) * BT  # scalar

    # Initialize dh state to zeros at t=0 (first step of REVERSE scan,
    # i.e. the END of the original sequence via reversed index_maps)
    @pl.when(i_t == 0)
    def init():
        scratch_ref[:, :] = jnp.zeros((BK, BV), dtype=jnp.float32)

    # Load inputs for this (reversed) time step
    b_q = q_ref[0, 0]                         # [BT, BK]
    b_k = k_ref[0, 0]                         # [BT, BK]
    b_v = v_ref[0, 0]                         # [BT, BV]
    b_h = h_ref[0, 0, 0].astype(jnp.float32)  # [BK, BV]
    b_do = do_ref[0, 0]                        # [BT, BV]

    # Current dh from scratch
    b_dh = scratch_ref[...]    # [BK, BV]

    b_gn = b_g_last  # scalar: g_gamma * BT

    # -------------------------------------------------------
    # Phase 0 (VPU): Pre-compute ALL exp/gate values upfront
    # -------------------------------------------------------
    exp_pos = jnp.exp(b_g_ramp)                    # [BT]
    exp_neg = jnp.exp(-b_g_ramp)                   # [BT]
    exp_gn_minus = jnp.exp(b_gn - b_g_ramp)       # [BT]

    k_neg = (b_k * exp_neg[:, None]).astype(b_k.dtype)           # [BT, K]
    k_decay = (b_k * exp_gn_minus[:, None]).astype(b_k.dtype)    # [BT, K]
    q_pos = (b_q * exp_pos[:, None]).astype(b_q.dtype)           # [BT, K]

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
    # Phase 5 (VPU): Combine results and write to REVERSED 5D output slots
    # MUTATION (eliminate_flip): Write to NT-1-i_t instead of i_t
    # -------------------------------------------------------
    dv_ref[0, NT - 1 - i_t, 0] = (b_dv_intra + b_dv_inter).astype(dv_ref.dtype)

    b_dq = b_dq_intra_raw * exp_pos[:, None] + b_dq_inter * (scale * exp_pos[:, None])
    dq_ref[0, NT - 1 - i_t, 0] = b_dq.astype(dq_ref.dtype)

    b_dk = b_dk_intra_raw * exp_neg[:, None] + b_dk_inter * exp_gn_minus[:, None]
    dk_ref[0, NT - 1 - i_t, 0] = b_dk.astype(dk_ref.dtype)

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


def chunk_gla_bwd_eliminate_gcumsum(q, k, v, h, do, g_gamma, scale, chunk_size):
    """Fused backward: single Pallas kernel with g_cumsum eliminated and flips eliminated.

    MUTATION (eliminate_flip): No jnp.flip() calls. Instead, reversed
    BlockSpec index_maps (NT-1-t) feed chunks in reverse order, and
    the kernel writes outputs to NT-1-i_t positions. This eliminates
    5 flip HBM copies and the chunk->flip->flatten reshape chain.

    MUTATION (eliminate_gcumsum): g_cumsum removed from inputs entirely.
    The kernel recomputes gating from g_gamma scalar as [BT] vectors.
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

    # Reshape time into chunks: [B, H, NT, BT, dim]
    q_chunked = q_t.reshape(B, H, NT, BT, K)
    k_chunked = k_t.reshape(B, H, NT, BT, K)
    v_chunked = v_t.reshape(B, H, NT, BT, V)
    do_chunked = do_t.reshape(B, H, NT, BT, V)

    # h is [B, NT, H, K, V]; transpose to [B, H, NT, K, V]
    h_bhntKV = jnp.transpose(h, (0, 2, 1, 3, 4))  # [B, H, NT, K, V]

    # MUTATION (eliminate_flip): NO jnp.flip() calls — flatten NON-flipped chunked arrays
    q_flat = q_chunked.reshape(B, H, NT * BT, K)
    k_flat = k_chunked.reshape(B, H, NT * BT, K)
    v_flat = v_chunked.reshape(B, H, NT * BT, V)
    do_flat = do_chunked.reshape(B, H, NT * BT, V)

    grid = (B, H, pl.cdiv(K, BK), pl.cdiv(V, BV), NT)

    # MUTATION (eliminate_flip): Reversed input index_maps use NT_val-1-t
    NT_val = NT
    def q_map(b, h, ki, vi, t):  return b, h, NT_val - 1 - t, ki
    def k_map(b, h, ki, vi, t):  return b, h, NT_val - 1 - t, ki
    def v_map(b, h, ki, vi, t):  return b, h, NT_val - 1 - t, vi
    def h_map(b, h, ki, vi, t):  return b, h, NT_val - 1 - t, 0, 0
    def do_map(b, h, ki, vi, t): return b, h, NT_val - 1 - t, vi

    # Output index maps — 5D outputs [B, NT, H, BT, K/V]
    # Outputs are NOT reversed; the kernel writes to NT-1-i_t position
    def out_k_map(b, h, ki, vi, t): return b, 0, h, 0, 0
    def out_v_map(b, h, ki, vi, t): return b, 0, h, 0, 0

    dq_5d, dk_5d, dv_5d = pl.pallas_call(
        functools.partial(
            _chunk_gla_bwd_kernel,
            BT=BT, NT=NT, scale=scale,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, 1, BT, BK), q_map),        # q: 4D
                pl.BlockSpec((1, 1, BT, BK), k_map),        # k: 4D
                pl.BlockSpec((1, 1, BT, BV), v_map),        # v: 4D
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
    )(q_flat, k_flat, v_flat, h_bhntKV, do_flat, g_gamma)

    # MUTATION (eliminate_flip): NO output jnp.flip() — outputs already in correct order
    # Use dq_5d/dk_5d/dv_5d directly.
    # Reshape [B, NT, H, BT, K] -> [B, T, H, K]
    dq = dq_5d.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K)
    dk = dk_5d.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K)
    dv = dv_5d.transpose(0, 1, 3, 2, 4).reshape(B, T, H, V)

    return dq, dk, dv


# ============================================================
# custom_vjp wrapper
#
# MUTATION (bf16_residuals):
#   _fwd: Stores q/k/v as bf16 in residuals (saves 96MB HBM)
#   _bwd: Casts q/k/v back to f32 before backward computation
#
# MUTATION (eliminate_gcumsum):
#   _fwd: No longer computes or stores g_cumsum in residuals
#   _bwd: Calls backward kernel (no g_cumsum input)
#
# MUTATION (eliminate_flip):
#   _bwd: Backward uses reversed index_maps, no flip copies
# ============================================================


def chunk_gla(q, k, v, g_gamma, scale, chunk_size):
    """Chunked GLA with custom_vjp (Pallas TPU kernels)."""
    @jax.custom_vjp
    def _compute(q, k, v):
        _, o = chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size)
        return o

    def _fwd(q, k, v):
        # MUTATION (bf16_residuals): Store q/k/v as bf16 to save 96MB HBM
        h, o = chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size)
        return o, (q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v.astype(jnp.bfloat16), h)

    def _bwd(residuals, do):
        # MUTATION (bf16_residuals): Cast q/k/v back to f32 for backward computation
        q_bf16, k_bf16, v_bf16, h = residuals
        q = q_bf16.astype(jnp.float32)
        k = k_bf16.astype(jnp.float32)
        v = v_bf16.astype(jnp.float32)
        # MUTATION (eliminate_gcumsum + eliminate_flip): backward with reversed index_maps
        dq, dk, dv = chunk_gla_bwd_eliminate_gcumsum(
            q, k, v, h, do, g_gamma, scale, chunk_size,
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
