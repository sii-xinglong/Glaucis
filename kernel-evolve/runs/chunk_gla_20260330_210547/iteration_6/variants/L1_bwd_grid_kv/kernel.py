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
# (Pallas kernel - FUSED)
#
# MUTATION (fuse_fwd): Merges the two forward pallas_calls
# (chunk_fwd_h + chunk_gla_fwd_o_gk) into a single kernel.
#
# This eliminates:
#   1. One pallas_call launch overhead
#   2. One full HBM round-trip for the h tensor between kernels
#      (h is now kept in VMEM scratch, never written to HBM
#       except as output for backward)
#   3. The h tensor read in the output kernel
#
# Grid: (B, H, K/BK, V/BV, NT) with time as "arbitrary"
# At each time step t:
#   1. Save h to h_ref output (for backward residuals)
#   2. Compute output: o = (q*exp(g)) @ h * scale + A_masked @ v
#      where A = (q*exp(g)) @ (k*exp(-g)).T * scale (recomputed)
#   3. Update h: h = h * exp(g_gamma*BT) + k.T @ (v*exp(g_gamma*BT - g))
# ============================================================


def _chunk_fwd_combined_kernel(
    q_ref, k_ref, v_ref, h0_ref, g_gamma,
    h_ref, o_ref, ht_ref, scratch_ref,
    *, BT, NT, scale,
):
    """Combined forward kernel: h-propagation + output in one pass.

    For each time step:
      1. Emit h state (for backward)
      2. Compute output from h state and intra-chunk attention
      3. Update h state for next chunk
    """
    BK = q_ref.shape[3]
    BV = v_ref.shape[3]
    i_b, i_h, i_k, i_v, i_t = (
        pl.program_id(0), pl.program_id(1), pl.program_id(2),
        pl.program_id(3), pl.program_id(4),
    )

    # Per-position gating within chunk: g_gamma * (1, 2, ..., BT)
    b_g = g_gamma[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)
    b_g_last = g_gamma[i_h] * BT  # decay for full chunk

    # --- Initialize h state at t=0 ---
    @pl.when(i_t == 0)
    def init():
        if h0_ref is not None:
            scratch_ref[:, :] = h0_ref[0, 0].astype(jnp.float32)
        else:
            scratch_ref[:, :] = jnp.zeros((BK, BV), dtype=jnp.float32)

    # --- Step 1: Save current h state for backward ---
    h_ref[0, i_t, 0] = scratch_ref[...]

    # --- Step 2: Compute output for this chunk ---
    # Load tiles
    b_q = q_ref[0, 0]   # [BT, BK]
    b_k = k_ref[0, 0]   # [BT, BK]
    b_v = v_ref[0, 0]   # [BT, BV]

    # Gating
    exp_g = jnp.exp(b_g)[:, None]          # [BT, 1]
    exp_neg_g = jnp.exp(-b_g)[:, None]     # [BT, 1]
    b_qg = (b_q * exp_g).astype(b_q.dtype)     # [BT, BK]: q * exp(g)
    b_kg = (b_k * exp_neg_g).astype(b_k.dtype)  # [BT, BK]: k * exp(-g)

    # Inter-chunk: b_qg @ h * scale -> [BT, BV]
    b_o_inter = jnp.dot(b_qg, scratch_ref[...].astype(b_qg.dtype),
                        precision=lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32)
    b_o_inter = b_o_inter * scale

    # Intra-chunk: A = b_qg @ b_kg.T * scale, then A_masked @ v
    b_A = jnp.dot(b_qg, b_kg.T,
                  precision=lax.Precision.HIGHEST,
                  preferred_element_type=jnp.float32) * scale  # [BT, BT]
    m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_A_masked = jnp.where(m_s, b_A, 0.0).astype(b_v.dtype)

    b_o_intra = jnp.dot(b_A_masked, b_v,
                        precision=lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32)  # [BT, BV]

    # Write output
    o_ref[0, 0] = (b_o_inter + b_o_intra).astype(o_ref.dtype)

    # --- Step 3: Update h state for next chunk ---
    # h = h * exp(g_gamma * BT) + k.T @ (v * exp(g_gamma*BT - g))
    scratch_ref[...] *= exp(b_g_last)

    v_decay = (b_v * exp(b_g_last - b_g)[:, None]).astype(b_v.dtype)  # [BT, BV]
    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        b_k.astype(jnp.float32).T,       # [BK, BT]
        v_decay.astype(jnp.float32),      # [BT, BV]
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    @pl.when(i_t == NT - 1)
    def end():
        if ht_ref is not None:
            ht_ref[0, 0] = scratch_ref[...]


def chunk_fwd_combined(q, k, v, g_gamma, scale, chunk_size):
    """Launch combined forward Pallas kernel (h-propagation + output).

    MUTATION (fuse_fwd): Replaces chunk_fwd_h + chunk_gla_fwd_o_gk
    with a single pallas_call. h state is kept in VMEM scratch.

    Returns:
        h: [B, NT, H, K, V] - inter-chunk states (for backward)
        o: [B, T, H, V] - output
    """
    BK, BV, BT = 128, 128, chunk_size
    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = T // BT

    # Transpose to [B, H, T, dim] layout for BlockSpec
    q_t = jnp.transpose(q, (0, 2, 1, 3))   # [B, H, T, K]
    k_t = jnp.transpose(k, (0, 2, 1, 3))   # [B, H, T, K]
    v_t = jnp.transpose(v, (0, 2, 1, 3))   # [B, H, T, V]

    grid = (B, H, pl.cdiv(K, BK), pl.cdiv(V, BV), NT)

    def q_map(b, h, ki, vi, t): return b, h, t, ki   # [BT, BK]
    def k_map(b, h, ki, vi, t): return b, h, t, ki   # [BT, BK]
    def v_map(b, h, ki, vi, t): return b, h, t, vi   # [BT, BV]
    def h_map(b, h, ki, vi, t): return b, 0, h, ki, vi  # [NT, BK, BV]
    def o_map(b, h, ki, vi, t): return b, h, t, vi   # [BT, BV]

    h_all, o_t, ht = pl.pallas_call(
        functools.partial(_chunk_fwd_combined_kernel, BT=BT, NT=NT, scale=scale),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, 1, BT, BK), q_map),   # q
                pl.BlockSpec((1, 1, BT, BK), k_map),   # k
                pl.BlockSpec((1, 1, BT, BV), v_map),   # v
                None,                                    # h0 (unused)
                pl.BlockSpec(memory_space=pltpu.SMEM),  # g_gamma
            ],
            out_specs=[
                pl.BlockSpec((1, NT, 1, BK, BV), h_map),  # h_all
                pl.BlockSpec((1, 1, BT, BV), o_map),      # o
                None,                                       # ht (unused)
            ],
            scratch_shapes=[pltpu.VMEM((BK, BV), jnp.float32)],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((B, NT, H, K, V), q.dtype),   # h_all
            jax.ShapeDtypeStruct((B, H, T, V), v.dtype),       # o
            None,                                                # ht
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary", "arbitrary", "arbitrary"),
            disable_bounds_checks=True,
        ),
    )(q_t, k_t, v_t, None, g_gamma)

    # o_t is [B, H, T, V], need [B, T, H, V]
    o = jnp.transpose(o_t, (0, 2, 1, 3))

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
# MUTATION (fuse_fwd + fuse_fwd_A): Forward is now a SINGLE
# pallas_call. chunk_fwd_combined does h-propagation, A-recompute,
# and output computation all in one kernel.
# ============================================================


def chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size):
    """Chunked GLA forward pass.

    MUTATION: Reduced from 2 pallas_calls to 1 via chunk_fwd_combined.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C

    pos = jnp.arange(1, C + 1, dtype=jnp.float32)
    pos = jnp.tile(pos, NT).reshape(1, T, 1, 1)
    g_cumsum = jnp.broadcast_to(g_gamma.reshape(1, 1, -1, 1) * pos, q.shape)

    # Single fused forward kernel: computes h and o together
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
# Backward: Fused dq, dk, dv (Pallas kernel)
#
# MUTATION (bwd_grid_kv): Add V tiling to the backward fused
# kernel's grid to reduce per-tile register footprint.
#
# Profile motivation: _chunk_gla_bwd_fused_kernel has 2.5M register
# spills. The kernel processes full V=128 in each tile, making all
# [BT,V], [K,V] arrays simultaneously live.
#
# Strategy: Only tile V (not K), using grid=(H, total_NT, NV) where
# NV = V // BV_inner and BV_inner = 64. This avoids the A/dA
# cross-tile accumulation issue since A only depends on K, not V.
#
# V-tiling impact:
#   - V-dependent arrays halve: b_v [BT,64], b_do [BT,64],
#     b_h [K,64], b_dh [K,64], dv output [BT,64]
#   - K-dependent arrays unchanged: b_q [BT,128], b_k [BT,128], etc.
#   - A/dA [BT,BT] computed per V-tile (same cost but only once needed)
#
# dq_inter = do @ h.T [BT,K] and dk_inter = v @ dh.T [BT,K] are
# partial sums across V tiles and must be accumulated. We use
# VMEM scratch buffers for these partial [BT,K] accumulators, plus
# the K-wide gate precomputations (exp_pos_g, exp_neg_g, etc.).
#
# Grid: (H, total_NT, NV) where NV=2 for V=128, BV_inner=64
# ============================================================

BV_INNER = 64  # V sub-tile size for backward kernel


def _chunk_gla_bwd_fused_kernel_vtiled(
    q_ref, k_ref, v_ref, g_ref, h_ref, do_ref, dh_ref,
    dq_ref, dk_ref, dv_ref,
    dq_inter_acc_ref, dk_inter_acc_ref,
    *, BT, scale, BV_inner, NV,
):
    """Backward fused kernel with V tiling to reduce register pressure.

    Grid: (H, total_NT, NV)
    - i_v indexes into V sub-tiles of size BV_inner

    V-dependent arrays (b_v, b_do, b_h, b_dh, dv) are halved per tile.
    K-dependent arrays (b_q, b_k, b_g, exp arrays) remain full K=128.

    dq_inter and dk_inter are [BT,K] accumulators in VMEM scratch.
    On the first V tile (i_v==0), they are initialized to zero.
    On the last V tile (i_v==NV-1), they are combined and written.
    """
    i_h = pl.program_id(0)
    i_nt = pl.program_id(1)
    i_v = pl.program_id(2)

    # -------------------------------------------------------
    # Load K-wide inputs (same for all V tiles)
    # -------------------------------------------------------
    b_q = q_ref[0, 0]                            # [BT, K]
    b_k = k_ref[0, 0]                            # [BT, K]
    b_g = g_ref[0, 0].astype(jnp.float32)        # [BT, K]

    b_gn = b_g[BT - 1, :]                        # last row: [K]

    # -------------------------------------------------------
    # Phase 0 (VPU): Pre-compute ALL exp/gate values (K-wide)
    # -------------------------------------------------------
    exp_pos_g = jnp.exp(b_g)                     # [BT, K]
    exp_neg_g = jnp.exp(-b_g)                    # [BT, K]
    exp_gn_minus_g = jnp.exp(b_gn[None, :] - b_g)  # [BT, K]

    k_neg = (b_k * exp_neg_g).astype(b_k.dtype)          # [BT, K]
    k_decay = (b_k * exp_gn_minus_g).astype(b_k.dtype)   # [BT, K]
    q_pos = (b_q * exp_pos_g).astype(b_q.dtype)          # [BT, K]

    # -------------------------------------------------------
    # Load V-tile-specific inputs
    # -------------------------------------------------------
    b_v = v_ref[0, 0]                            # [BT, BV_inner]
    b_do = do_ref[0, 0]                          # [BT, BV_inner]
    b_h = h_ref[0, 0].astype(jnp.float32)        # [K, BV_inner]
    b_dh = dh_ref[0, 0].astype(jnp.float32)      # [K, BV_inner]

    # -------------------------------------------------------
    # Phase 1 (MXU): Compute A and dA from K-wide arrays.
    # A only depends on K, so it's the same for all V tiles.
    # We compute it every V tile (same cost, needed for dv_intra).
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
    b_dA = jnp.where(mask, b_dA_raw, 0.0)          # [BT, BT]
    b_a_masked = jnp.where(mask, b_a, 0.0)         # [BT, BT]

    # -------------------------------------------------------
    # Phase 3 (MXU): V-tile-specific computations
    # -------------------------------------------------------
    # dv: fully V-local, write directly
    b_dv_intra = jnp.dot(b_a_masked.T.astype(b_do.dtype), b_do,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)   # [BT, BV_inner]

    b_dv_inter = jnp.dot(k_decay, b_dh.astype(b_k.dtype),
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)   # [BT, BV_inner]

    dv_ref[0, 0] = (b_dv_intra + b_dv_inter).astype(dv_ref.dtype)

    # dq_inter and dk_inter are [BT, K] partial sums across V tiles
    b_dq_inter_partial = jnp.dot(b_do, b_h.astype(b_do.dtype).T,
                                  precision=jax.lax.Precision.HIGHEST,
                                  preferred_element_type=jnp.float32)  # [BT, K]

    b_dk_inter_partial = jnp.dot(b_v, b_dh.astype(b_v.dtype).T,
                                  precision=jax.lax.Precision.HIGHEST,
                                  preferred_element_type=jnp.float32)  # [BT, K]

    # -------------------------------------------------------
    # Accumulate dq_inter and dk_inter across V tiles
    # -------------------------------------------------------
    @pl.when(i_v == 0)
    def init_acc():
        dq_inter_acc_ref[:, :] = b_dq_inter_partial
        dk_inter_acc_ref[:, :] = b_dk_inter_partial

    @pl.when(i_v != 0)
    def accum():
        dq_inter_acc_ref[:, :] = dq_inter_acc_ref[...] + b_dq_inter_partial
        dk_inter_acc_ref[:, :] = dk_inter_acc_ref[...] + b_dk_inter_partial

    # -------------------------------------------------------
    # On the last V tile: compute intra terms and write dq/dk
    # -------------------------------------------------------
    @pl.when(i_v == NV - 1)
    def write_dk_dq():
        # Phase 4 (MXU): Intra-chunk dq and dk (only on last V tile)
        b_dq_intra_raw = jnp.dot(b_dA.astype(k_neg.dtype), k_neg,
                                  precision=jax.lax.Precision.HIGHEST,
                                  preferred_element_type=jnp.float32)  # [BT, K]

        b_dk_intra_raw = jnp.dot(b_dA.T.astype(q_pos.dtype), q_pos,
                                  precision=jax.lax.Precision.HIGHEST,
                                  preferred_element_type=jnp.float32)  # [BT, K]

        # Phase 5 (VPU): Combine and write dq, dk
        b_dq = b_dq_intra_raw * exp_pos_g + dq_inter_acc_ref[...] * (scale * exp_pos_g)
        dq_ref[0, 0] = b_dq.astype(dq_ref.dtype)

        b_dk = b_dk_intra_raw * exp_neg_g + dk_inter_acc_ref[...] * exp_gn_minus_g
        dk_ref[0, 0] = b_dk.astype(dk_ref.dtype)


def chunk_gla_bwd_fused(q, k, v, g_cumsum, h, do, dh, scale, chunk_size):
    """Launch fused backward Pallas kernel with V tiling.

    MUTATION (bwd_grid_kv): Add V tiling to backward kernel grid.
    Grid becomes (H, total_NT, NV) where NV = V // BV_inner.

    V-tiled arrays (BV_inner=64 instead of V=128):
      - b_v, b_do, b_h, b_dh, dv: [*, 64] instead of [*, 128]
    Accumulated across V tiles via VMEM scratch:
      - dq_inter_acc [BT, K], dk_inter_acc [BT, K]
    Written only on last V tile:
      - dq, dk outputs
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT
    BV = BV_INNER
    NV = V // BV

    _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _k = k.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _v = v.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _g = g_cumsum.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _do = do.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _h = h.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)
    _dh = dh.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)

    # Grid now includes NV dimension for V tiling
    grid = (H, total_NT, NV)

    # BlockSpecs: q, k, g are K-full (no V indexing needed)
    # v, do, dv use BV sub-tiles indexed by i_v (grid dim 2)
    # h, dh use BV sub-tiles indexed by i_v (grid dim 2)
    def spec_K_map(h, nt, iv): return (h, nt, 0, 0)
    def spec_V_map(h, nt, iv): return (h, nt, 0, iv)   # V dim tiled by iv
    def spec_h_map(h, nt, iv): return (h, nt, 0, iv)   # [K, BV] tile

    # dq, dk: written only on last V tile but BlockSpec must cover full output
    # We use spec_K_map since dq/dk have shape [H, total_NT, BT, K] (no V)
    def spec_dq_map(h, nt, iv): return (h, nt, 0, 0)
    def spec_dk_map(h, nt, iv): return (h, nt, 0, 0)

    dq, dk, dv = pl.pallas_call(
        functools.partial(
            _chunk_gla_bwd_fused_kernel_vtiled,
            BT=BT, scale=scale, BV_inner=BV, NV=NV,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec([1, 1, BT, K], spec_K_map),    # q
                pl.BlockSpec([1, 1, BT, K], spec_K_map),    # k
                pl.BlockSpec([1, 1, BT, BV], spec_V_map),   # v  (V-tiled)
                pl.BlockSpec([1, 1, BT, K], spec_K_map),    # g
                pl.BlockSpec([1, 1, K, BV], spec_h_map),    # h  (V-tiled)
                pl.BlockSpec([1, 1, BT, BV], spec_V_map),   # do (V-tiled)
                pl.BlockSpec([1, 1, K, BV], spec_h_map),    # dh (V-tiled)
            ],
            out_specs=[
                pl.BlockSpec([1, 1, BT, K], spec_dq_map),   # dq (K-full)
                pl.BlockSpec([1, 1, BT, K], spec_dk_map),   # dk (K-full)
                pl.BlockSpec([1, 1, BT, BV], spec_V_map),   # dv (V-tiled)
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
# MUTATION (combined + fold_dh_pallas + fuse_fwd):
#   _fwd: residuals no longer include A (fuse_fwd_A),
#         forward uses single fused kernel (fuse_fwd)
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
