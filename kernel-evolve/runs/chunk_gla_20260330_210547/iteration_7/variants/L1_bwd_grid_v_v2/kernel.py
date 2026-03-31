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
# Backward: State gradient propagation (Pallas kernel)
#
# MUTATION (fold_dh_pallas): Replace lax.scan with pallas_call.
#
# The original _chunk_bwd_dh_scan used lax.scan with reverse=True,
# which compiles to ~NT separate computation events on TPU (one per
# scan iteration). This mutation replaces it with a single pallas_call
# that uses the time dimension with "arbitrary" semantics, matching
# the pattern already proven in chunk_fwd_h.
#
# To implement the REVERSE scan direction:
#   - Flip q and do along the time axis before passing to pallas_call
#   - The kernel iterates forward (t=0..NT-1), but because inputs
#     are flipped, t=0 in the kernel corresponds to the LAST chunk
#   - Flip the output dh along the time axis after pallas_call
#
# This produces a single computation event (or a few) instead of ~64,
# which should significantly reduce backward pass overhead.
#
# Expected impact: ~60 fewer computation events in backward pass.
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
    q_hat = (q_tile * exp(b_g_ramp)[:, None] * scale).astype(jnp.float32)

    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        q_hat.T,                          # [BK, BT]
        do_tile.astype(jnp.float32),      # [BT, BV]
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
# Backward: Fused dq, dk, dv with V-tiling via grid dimension
#
# MUTATION (L1_bwd_grid_v_v2): Add a V-dimension to the backward
# fused kernel grid to reduce register pressure.
#
# Problem: The base backward kernel has 2.5M register spills.
# V-dimension arrays (b_v [BT,128], b_do [BT,128], b_dh [K,128],
# b_h [K,128]) contribute heavily to register pressure.
#
# Solution: Process V=128 in two 64-wide passes via the grid.
# grid = (H, total_NT, NV) where NV = V // BV_inner = 2
#
# Key insight from Round 6 failure:
#   Arrays STILL have full V=128 as their last dimension.
#   The BlockSpec must SELECT a 64-wide sub-tile using index_map,
#   not reduce the array's actual last dimension.
#
# Accumulation strategy for dq/dk across V tiles:
#   - dq_inter = do @ h.T is [BT, K] — depends on V
#   - dk_inter = v @ dh.T is [BT, K] — depends on V
#   - dq_intra and dk_intra depend only on q, k, A — NOT on V
#   => Use VMEM scratch to accumulate dq_inter and dk_inter
#   => Only compute intra terms on first V tile (i_v == 0)
#   => On last V tile (i_v == NV-1): finalize and write dq, dk
#
# dv: each V tile writes its own independent slice (no accumulation).
# ============================================================

BV_INNER = 64  # Sub-tile width for V dimension in backward kernel


def _chunk_gla_bwd_fused_v_tiled_kernel(
    q_ref, k_ref, v_ref, g_ref, h_ref, do_ref, dh_ref,
    dq_ref, dk_ref, dv_ref,
    dq_inter_accum_ref, dk_inter_accum_ref,
    *, BT, BV_inner, NV, scale,
):
    """Backward fused kernel with V-tiling via grid dimension.

    Grid: (H, total_NT, NV) where NV = V // BV_inner.

    Each kernel invocation processes one BV_inner=64 slice of the V
    dimension. V-dependent inter terms (dq_inter, dk_inter) are
    accumulated in VMEM scratch across V tiles and finalized on the
    last tile. V-independent intra terms are computed only on i_v==0.
    """
    i_h = pl.program_id(0)
    i_nt = pl.program_id(1)
    i_v = pl.program_id(2)

    # Load K-dimension arrays (do NOT depend on i_v)
    b_q = q_ref[0, 0]              # [BT, K]
    b_k = k_ref[0, 0]              # [BT, K]
    b_g = g_ref[0, 0].astype(jnp.float32)  # [BT, K]

    # Load V-subtile arrays (BlockSpec selects the BV_inner slice)
    b_v = v_ref[0, 0]              # [BT, BV_inner]
    b_do = do_ref[0, 0]            # [BT, BV_inner]
    b_h = h_ref[0, 0].astype(jnp.float32)   # [K, BV_inner]
    b_dh = dh_ref[0, 0].astype(jnp.float32) # [K, BV_inner]

    b_gn = b_g[BT - 1, :]  # last row: [K]

    # -------------------------------------------------------
    # Phase 0 (VPU): Pre-compute ALL exp/gate values upfront
    # -------------------------------------------------------
    exp_pos_g = jnp.exp(b_g)               # [BT, K]
    exp_neg_g = jnp.exp(-b_g)              # [BT, K]
    exp_gn_minus_g = jnp.exp(b_gn[None, :] - b_g)  # [BT, K]

    k_neg = (b_k * exp_neg_g).astype(b_k.dtype)          # [BT, K]
    k_decay = (b_k * exp_gn_minus_g).astype(b_k.dtype)   # [BT, K]
    q_pos = (b_q * exp_pos_g).astype(b_q.dtype)          # [BT, K]

    # -------------------------------------------------------
    # Phase 1 (MXU): Compute dA = do @ v.T * scale
    # Note: A recomputation uses only q, k — NOT v, so it's
    # computed only on first V tile and reused via intra terms.
    # -------------------------------------------------------
    b_dA_raw = jnp.dot(b_do.astype(b_v.dtype), b_v.T,
                       precision=jax.lax.Precision.HIGHEST,
                       preferred_element_type=jnp.float32) * scale  # [BT, BT]

    # -------------------------------------------------------
    # Phase 2 (VPU): Apply causal mask to dA
    # -------------------------------------------------------
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_dA = jnp.where(mask, b_dA_raw, 0.0)  # [BT, BT]

    # -------------------------------------------------------
    # Phase 3 (MXU): V-dependent dot products for dv and inter terms
    # -------------------------------------------------------

    # dv intra: A_masked.T @ do  [BT, BV_inner]
    # Need masked A — computed only on first V tile to reuse a_masked.
    # Actually we can compute A here with a guard, or compute per tile.
    # For simplicity and correctness: compute A per V tile (cheap — K*BT).
    # A only uses q_pos, k_neg which are K-dimension; same for all V tiles.
    b_a = jnp.dot(q_pos, k_neg.T,
                  precision=jax.lax.Precision.HIGHEST,
                  preferred_element_type=jnp.float32) * scale  # [BT, BT]
    b_a_masked = jnp.where(mask, b_a, 0.0)

    b_dv_intra = jnp.dot(b_a_masked.T.astype(b_do.dtype), b_do,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)   # [BT, BV_inner]

    # dv inter: k_decay @ dh  [BT, BV_inner]
    b_dv_inter = jnp.dot(k_decay, b_dh.astype(b_k.dtype),
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)   # [BT, BV_inner]

    # Write dv for this V subtile — independent across V tiles
    dv_ref[0, 0] = (b_dv_intra + b_dv_inter).astype(dv_ref.dtype)

    # dq_inter partial: do @ h.T  [BT, K]  (h here is [K, BV_inner])
    b_dq_inter_partial = jnp.dot(b_do, b_h.astype(b_do.dtype).T,
                                 precision=jax.lax.Precision.HIGHEST,
                                 preferred_element_type=jnp.float32)  # [BT, K]

    # dk_inter partial: v @ dh.T  [BT, K]  (dh here is [K, BV_inner])
    b_dk_inter_partial = jnp.dot(b_v, b_dh.astype(b_v.dtype).T,
                                 precision=jax.lax.Precision.HIGHEST,
                                 preferred_element_type=jnp.float32)  # [BT, K]

    # -------------------------------------------------------
    # Accumulate inter terms across V tiles using VMEM scratch
    # -------------------------------------------------------

    @pl.when(i_v == 0)
    def init_accum():
        dq_inter_accum_ref[:, :] = b_dq_inter_partial
        dk_inter_accum_ref[:, :] = b_dk_inter_partial

    @pl.when(i_v != 0)
    def accumulate():
        dq_inter_accum_ref[:, :] = dq_inter_accum_ref[...] + b_dq_inter_partial
        dk_inter_accum_ref[:, :] = dk_inter_accum_ref[...] + b_dk_inter_partial

    # -------------------------------------------------------
    # Phase 4 (MXU): Intra-chunk dq and dk (V-independent)
    # Compute only on last V tile to avoid redundant work.
    # Finalize dq, dk on last V tile using accumulated inter terms.
    # -------------------------------------------------------

    @pl.when(i_v == NV - 1)
    def finalize():
        # Intra terms (no dependence on V — same result regardless of i_v)
        b_dq_intra_raw = jnp.dot(b_dA.astype(k_neg.dtype), k_neg,
                                 precision=jax.lax.Precision.HIGHEST,
                                 preferred_element_type=jnp.float32)  # [BT, K]

        b_dk_intra_raw = jnp.dot(b_dA.T.astype(q_pos.dtype), q_pos,
                                 precision=jax.lax.Precision.HIGHEST,
                                 preferred_element_type=jnp.float32)  # [BT, K]

        # Read fully accumulated inter terms
        b_dq_inter = dq_inter_accum_ref[...]  # [BT, K]
        b_dk_inter = dk_inter_accum_ref[...]  # [BT, K]

        # Combine intra + inter for dq
        b_dq = b_dq_intra_raw * exp_pos_g + b_dq_inter * (scale * exp_pos_g)
        dq_ref[0, 0] = b_dq.astype(dq_ref.dtype)

        # Combine intra + inter for dk
        b_dk = b_dk_intra_raw * exp_neg_g + b_dk_inter * exp_gn_minus_g
        dk_ref[0, 0] = b_dk.astype(dk_ref.dtype)


def chunk_gla_bwd_fused(q, k, v, g_cumsum, h, do, dh, scale, chunk_size):
    """Launch fused backward Pallas kernel with V-tiling via grid.

    MUTATION (L1_bwd_grid_v_v2):
      - grid = (H, total_NT, NV) where NV = V // BV_inner
      - V-dependent arrays (v, do, h, dh, dv) use BlockSpec with
        index_map to select BV_inner=64 sub-tile from V=128
      - K-dimension arrays (q, k, g, dq, dk) are NOT V-tiled;
        their BlockSpec ignores the i_v grid dimension
      - dq/dk inter terms are accumulated in VMEM scratch across V tiles
      - dv is written independently per V tile
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT
    BV_inner = BV_INNER
    NV = V // BV_inner

    _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _k = k.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _v = v.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _g = g_cumsum.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _do = do.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _h = h.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)
    _dh = dh.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)

    # Grid: (H, total_NT, NV)
    grid = (H, total_NT, NV)

    # BlockSpec index maps.
    # K-dimension arrays: q, k, g, dq, dk — ignore i_v (3rd grid dim)
    def k_map(h, nt, iv):  return (h, nt, 0, 0)    # full [BT, K] block

    # V-dimension arrays: v, do, dv — select BV_inner sub-tile from V=128
    def v_map(h, nt, iv):  return (h, nt, 0, iv)   # selects [BT, BV_inner]

    # [K, V] arrays: h, dh — select BV_inner sub-tile from V=128
    def kv_map(h, nt, iv): return (h, nt, 0, iv)   # selects [K, BV_inner]

    # BlockSpecs — note: arrays still have full V=128 last dim,
    # but BlockSpec block shape is BV_inner=64 to select the sub-tile.
    spec_K = pl.BlockSpec((1, 1, BT, K), k_map)           # [BT, K] — K-dim arrays
    spec_V = pl.BlockSpec((1, 1, BT, BV_inner), v_map)    # [BT, BV_inner] — V sub-tile
    spec_KV = pl.BlockSpec((1, 1, K, BV_inner), kv_map)   # [K, BV_inner] — h/dh sub-tile

    dq, dk, dv = pl.pallas_call(
        functools.partial(
            _chunk_gla_bwd_fused_v_tiled_kernel,
            BT=BT, BV_inner=BV_inner, NV=NV, scale=scale,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                spec_K,   # q:  [H, total_NT, BT, K]
                spec_K,   # k:  [H, total_NT, BT, K]
                spec_V,   # v:  [H, total_NT, BT, V] → sub-tile [BT, BV_inner]
                spec_K,   # g:  [H, total_NT, BT, K]
                spec_KV,  # h:  [H, total_NT, K, V]  → sub-tile [K, BV_inner]
                spec_V,   # do: [H, total_NT, BT, V] → sub-tile [BT, BV_inner]
                spec_KV,  # dh: [H, total_NT, K, V]  → sub-tile [K, BV_inner]
            ],
            out_specs=[
                spec_K,   # dq: [H, total_NT, BT, K] — one write per NV (last tile)
                spec_K,   # dk: [H, total_NT, BT, K] — one write per NV (last tile)
                spec_V,   # dv: [H, total_NT, BT, V] → sub-tile per V tile
            ],
            scratch_shapes=[
                pltpu.VMEM((BT, K), jnp.float32),  # dq_inter accumulator
                pltpu.VMEM((BT, K), jnp.float32),  # dk_inter accumulator
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct([H, total_NT, BT, K], q.dtype),
            jax.ShapeDtypeStruct([H, total_NT, BT, K], k.dtype),
            jax.ShapeDtypeStruct([H, total_NT, BT, V], v.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel"),
            disable_bounds_checks=True,
        ),
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
# MUTATION (combined + fold_dh_pallas + fuse_fwd_combined):
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
