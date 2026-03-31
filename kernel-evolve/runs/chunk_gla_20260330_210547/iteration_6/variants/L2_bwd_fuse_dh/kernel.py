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
    b_qg = (b_q * exp_g).astype(b_q.dtype)    # [BT, K]: q * exp(g)
    b_kg = (b_k * exp_neg_g).astype(b_k.dtype) # [BT, K]: k * exp(-g)

    # --- Phase 1 (MXU): Recompute A = b_qg @ b_kg.T * scale ---
    b_A = jnp.dot(b_qg, b_kg.T,
                  precision=jax.lax.Precision.HIGHEST,
                  preferred_element_type=jnp.float32) * scale  # [BT, BT]

    # --- Phase 2 (VPU): Causal mask (overlaps with MXU drain) ---
    m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_A_masked = jnp.where(m_s, b_A, 0.0).astype(b_v.dtype)

    # --- Phase 3 (MXU): Inter-chunk contribution (b_qg @ h) ---
    b_o_inter = jnp.dot(b_qg, b_h.astype(b_qg.dtype),
                        precision=jax.lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32)
    b_o_inter = b_o_inter * scale

    # --- Phase 4 (MXU): Intra-chunk contribution (A_masked @ v) ---
    b_o_intra = jnp.dot(b_A_masked, b_v,
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
# Backward: Fused dh propagation + dq/dk/dv (single Pallas kernel)
#
# MUTATION (bwd_fuse_dh): Merge chunk_bwd_dh_pallas and
# chunk_gla_bwd_fused into a single pallas_call.
#
# The previous architecture:
#   1. chunk_bwd_dh_pallas  — reverse time scan, emits dh[B,NT,H,K,V] to HBM
#   2. chunk_gla_bwd_fused  — reads dh[b,nt,h,:,:] from HBM, computes dq/dk/dv
#
# This merger eliminates:
#   - One kernel launch (overhead)
#   - The dh tensor HBM round-trip: write (kernel 1) → read (kernel 2)
#     = 2 * B * NT * H * K * V * 4 bytes = 2 * 2 * 64 * 16 * 128 * 128 * 4 = 536MB
#
# Fusion mechanism (Option A from brief):
#   - Use same 5D grid: (B, H, K_tiles, V_tiles, NT) with "arbitrary" time
#   - Flip q, k, v, g, h, do inputs along the NT axis (same as chunk_bwd_dh_pallas)
#   - VMEM scratch holds dh state [BK, BV] in float32
#   - At each reverse-time step i_t (original chunk = NT-1-i_t):
#       a. Read current dh from VMEM scratch
#       b. Compute dq/dk/dv for this chunk using dh_current
#       c. Write dq/dk/dv to output (in reversed time slot i_t)
#       d. Update dh: dh = dh * decay + q_hat.T @ do
#   - After pallas_call: flip dq/dk/dv outputs along NT axis to restore order
#
# Key correctness insight:
#   The dh kernel emits dh BEFORE updating (so dh_out[t] is the dh that
#   propagates INTO chunk t from chunk t+1, i.e. what the fused kernel needs).
#   By reading scratch BEFORE the update and using it for dq/dk/dv, we
#   preserve this exact semantics.
#
# Since K_tiles = V_tiles = 1 for K=V=128 with BK=BV=128, each tile
# processes a full (BK=128, BV=128) block — matching the fused kernel's
# per-(h,nt) tile exactly.
#
# Expected impact: eliminate 1 pallas_call + 536MB HBM dh traffic.
# Profile motivation: profile_brief.md lists this as Priority 1 optimization.
# ============================================================


def _chunk_gla_bwd_fused_dh_kernel(
    q_ref, k_ref, v_ref, g_ref, h_ref, do_ref, g_gamma,
    dq_ref, dk_ref, dv_ref, scratch_ref,
    *, BT, NT, scale,
):
    """Fused backward kernel: computes dh state (VMEM) + dq/dk/dv per reverse chunk.

    Grid: (B, H, K_tiles=1, V_tiles=1, NT)
    Time dimension is "arbitrary" (processed in reverse via pre-flipped inputs).

    At step i_t (reverse order):
      - scratch_ref holds dh[t+1] (the dh state flowing into original chunk t)
      - We compute dq/dk/dv for this chunk using dh = scratch_ref
      - Then update: dh = dh * exp(g_gamma * BT) + q_hat.T @ do
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

    # Initialize dh scratch to zeros at t=0 (= last chunk in original time,
    # because inputs are time-reversed). No dht boundary condition needed.
    @pl.when(i_t == 0)
    def init():
        scratch_ref[:, :] = jnp.zeros((BK, BV), dtype=jnp.float32)

    # ---------------------------------------------------------------
    # Load inputs for this (reversed) time step.
    # All tensors have been pre-flipped along NT before the pallas_call,
    # so i_t=0 corresponds to original chunk NT-1, i_t=1 to NT-2, etc.
    # ---------------------------------------------------------------
    b_q = q_ref[0, 0]     # [BT, BK]
    b_k = k_ref[0, 0]     # [BT, BK]
    b_v = v_ref[0, 0]     # [BT, BV]
    b_g = g_ref[0, 0].astype(jnp.float32)   # [BT, BK] — g_cumsum for this chunk
    b_h = h_ref[0, 0].astype(jnp.float32)   # [BK, BV] — forward h state
    b_do = do_ref[0, 0]   # [BT, BV]

    # Current dh state (= dh flowing into this chunk from the future)
    b_dh = scratch_ref[...]   # [BK, BV] float32

    # ---------------------------------------------------------------
    # Compute dq/dk/dv using b_dh (same math as chunk_gla_bwd_fused_kernel)
    # ---------------------------------------------------------------
    b_gn = b_g[BT - 1, :]                      # [BK]: last row of g_cumsum
    exp_pos_g = jnp.exp(b_g)                    # [BT, BK]
    exp_neg_g = jnp.exp(-b_g)                   # [BT, BK]
    exp_gn_minus_g = jnp.exp(b_gn[None, :] - b_g)  # [BT, BK]

    k_neg = (b_k * exp_neg_g).astype(b_k.dtype)          # [BT, BK]
    k_decay = (b_k * exp_gn_minus_g).astype(b_k.dtype)   # [BT, BK]
    q_pos = (b_q * exp_pos_g).astype(b_q.dtype)          # [BT, BK]

    # Causal mask [BT, BT]
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]

    # Recompute A and dA_raw [BT, BT]
    b_a = jnp.dot(q_pos, k_neg.T,
                  precision=jax.lax.Precision.HIGHEST,
                  preferred_element_type=jnp.float32) * scale

    b_dA_raw = jnp.dot(b_do.astype(b_v.dtype), b_v.T,
                       precision=jax.lax.Precision.HIGHEST,
                       preferred_element_type=jnp.float32) * scale

    b_a_masked = jnp.where(mask, b_a, 0.0)     # [BT, BT]
    b_dA = jnp.where(mask, b_dA_raw, 0.0)      # [BT, BT]

    # dv = A.T @ do + k_decay @ dh
    b_dv_intra = jnp.dot(b_a_masked.T.astype(b_do.dtype), b_do,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)   # [BT, BV]
    b_dv_inter = jnp.dot(k_decay, b_dh.astype(b_k.dtype),
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)   # [BT, BV]
    dv_ref[0, i_t, 0] = (b_dv_intra + b_dv_inter).astype(dv_ref.dtype)

    # dq = (dA @ k_neg) * exp(g) + (do @ h.T) * scale * exp(g)
    b_dq_intra = jnp.dot(b_dA.astype(k_neg.dtype), k_neg,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)   # [BT, BK]
    b_dq_inter = jnp.dot(b_do, b_h.astype(b_do.dtype).T,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)   # [BT, BK]
    b_dq = b_dq_intra * exp_pos_g + b_dq_inter * (scale * exp_pos_g)
    dq_ref[0, i_t, 0] = b_dq.astype(dq_ref.dtype)

    # dk = (dA.T @ q_pos) * exp(-g) + (v @ dh.T) * exp(gn-g)
    b_dk_intra = jnp.dot(b_dA.T.astype(q_pos.dtype), q_pos,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)   # [BT, BK]
    b_dk_inter = jnp.dot(b_v, b_dh.astype(b_v.dtype).T,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)   # [BT, BK]
    b_dk = b_dk_intra * exp_neg_g + b_dk_inter * exp_gn_minus_g
    dk_ref[0, i_t, 0] = b_dk.astype(dk_ref.dtype)

    # ---------------------------------------------------------------
    # Update dh state: dh = dh * exp(g_gamma * BT) + q_hat.T @ do
    # q_hat = q * exp(g_ramp) * scale  [BT, BK]
    # This mirrors _chunk_bwd_dh_kernel's update exactly.
    # ---------------------------------------------------------------
    scratch_ref[...] *= exp(b_g_last)
    q_hat = (b_q * exp(b_g_ramp)[:, None] * scale).astype(jnp.float32)
    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        q_hat.T,                             # [BK, BT]
        b_do.astype(jnp.float32),            # [BT, BV]
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


def chunk_gla_bwd_fused_dh(q, k, v, g_cumsum, h, do, g_gamma, scale, chunk_size):
    """Fused backward: dh propagation + dq/dk/dv in a single pallas_call.

    MUTATION (bwd_fuse_dh): Merges chunk_bwd_dh_pallas + chunk_gla_bwd_fused.
    Eliminates the dh tensor HBM round-trip (256MB write + 256MB read = 512MB).

    Implementation:
      1. Flip q, k, v, g, h, do along the NT dimension (chunk axis)
      2. Run pallas_call with 5D grid (B, H, 1, 1, NT) and arbitrary time
      3. VMEM scratch holds dh state; at each step compute dq/dk/dv, then update dh
      4. Outputs dq/dk/dv are in reversed-time order → flip back after call
    """
    BK, BV, BT = 128, 128, chunk_size
    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = T // BT

    # Transpose to (B, H, NT, BT, dim) layout
    q_t = jnp.transpose(q, (0, 2, 1, 3))    # [B, H, T, K]
    k_t = jnp.transpose(k, (0, 2, 1, 3))    # [B, H, T, K]
    v_t = jnp.transpose(v, (0, 2, 1, 3))    # [B, H, T, V]
    do_t = jnp.transpose(do, (0, 2, 1, 3))  # [B, H, T, V]

    # Reshape time into chunks
    q_chunked = q_t.reshape(B, H, NT, BT, K)
    k_chunked = k_t.reshape(B, H, NT, BT, K)
    v_chunked = v_t.reshape(B, H, NT, BT, V)
    do_chunked = do_t.reshape(B, H, NT, BT, V)

    # g_cumsum: already [B, T, H, K], need [B, H, NT, BT, K]
    g_t = jnp.transpose(g_cumsum, (0, 2, 1, 3))       # [B, H, T, K]
    g_chunked = g_t.reshape(B, H, NT, BT, K)

    # h: [B, NT, H, K, V] -> for pallas need [B, H, NT, K, V]
    h_bhntKV = jnp.transpose(h, (0, 2, 1, 3, 4))      # [B, H, NT, K, V]

    # FLIP along NT axis (axis=2) to implement reverse scan
    q_flip = jnp.flip(q_chunked, axis=2)   # [B, H, NT, BT, K]
    k_flip = jnp.flip(k_chunked, axis=2)   # [B, H, NT, BT, K]
    v_flip = jnp.flip(v_chunked, axis=2)   # [B, H, NT, BT, V]
    do_flip = jnp.flip(do_chunked, axis=2) # [B, H, NT, BT, V]
    g_flip = jnp.flip(g_chunked, axis=2)   # [B, H, NT, BT, K]
    h_flip = jnp.flip(h_bhntKV, axis=2)   # [B, H, NT, K, V]

    # Flatten NT*BT back for BlockSpec indexing
    q_flat = q_flip.reshape(B, H, NT * BT, K)    # [B, H, NT*BT, K]
    k_flat = k_flip.reshape(B, H, NT * BT, K)
    v_flat = v_flip.reshape(B, H, NT * BT, V)
    do_flat = do_flip.reshape(B, H, NT * BT, V)
    g_flat = g_flip.reshape(B, H, NT * BT, K)
    # h_flip stays as [B, H, NT, K, V] — indexed per chunk

    grid = (B, H, pl.cdiv(K, BK), pl.cdiv(V, BV), NT)

    # Index maps: (b, h, ki, vi, t) -> tensor indices
    def qkvg_map(b, h, ki, vi, t): return b, h, t, ki    # slice along NT dimension
    def do_map(b, h, ki, vi, t): return b, h, t, vi
    def h_map(b, h, ki, vi, t): return b, h, t, 0        # h is [B, H, NT, K, V], block [1,1,K,V]

    # Output specs: dq/dk/dv stored in reversed-time layout.
    # Mirror the dh_ref pattern from chunk_bwd_dh_pallas exactly:
    #   out_shape (B, NT, H, BT, K) with block (1, NT, 1, BT, BK)
    #   index_map ignores `t` (block spans all NT) — kernel writes dq_ref[0, i_t, 0]
    # This matches: dh_map(b,h,ki,vi,t) -> (b, 0, h, ki, vi) ignoring t.
    def out_dq_map(b, h, ki, vi, t): return b, 0, h, 0, ki   # k-tiled, t unused (block=all NT)
    def out_dk_map(b, h, ki, vi, t): return b, 0, h, 0, ki
    def out_dv_map(b, h, ki, vi, t): return b, 0, h, 0, vi   # v-tiled

    # h_spec: [B, H, NT, K, V] accessible as (1, 1, K, V) blocks indexed by (b,h,t,*)
    h_spec = pl.BlockSpec((1, 1, K, V), h_map)

    dq_flipped, dk_flipped, dv_flipped = pl.pallas_call(
        functools.partial(_chunk_gla_bwd_fused_dh_kernel, BT=BT, NT=NT, scale=scale),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, 1, BT, BK), qkvg_map),   # q [B,H,NT*BT,K]
                pl.BlockSpec((1, 1, BT, BK), qkvg_map),   # k [B,H,NT*BT,K]
                pl.BlockSpec((1, 1, BT, BV), do_map),     # v [B,H,NT*BT,V]
                pl.BlockSpec((1, 1, BT, BK), qkvg_map),   # g [B,H,NT*BT,K]
                h_spec,                                     # h [B,H,NT,K,V]
                pl.BlockSpec((1, 1, BT, BV), do_map),     # do [B,H,NT*BT,V]
                pl.BlockSpec(memory_space=pltpu.SMEM),     # g_gamma [H] scalar
            ],
            out_specs=[
                pl.BlockSpec((1, NT, 1, BT, BK), out_dq_map),
                pl.BlockSpec((1, NT, 1, BT, BK), out_dk_map),
                pl.BlockSpec((1, NT, 1, BT, BV), out_dv_map),
            ],
            scratch_shapes=[pltpu.VMEM((BK, BV), jnp.float32)],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((B, NT, H, BT, K), q.dtype),
            jax.ShapeDtypeStruct((B, NT, H, BT, K), k.dtype),
            jax.ShapeDtypeStruct((B, NT, H, BT, V), v.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary", "arbitrary", "arbitrary"),
            disable_bounds_checks=True,
        ),
    )(q_flat, k_flat, v_flat, g_flat, h_flip, do_flat, g_gamma)

    # dq/dk/dv are in reversed-time order [B, NT, H, BT, K/V]
    # Flip NT axis back to original order
    dq_out = jnp.flip(dq_flipped, axis=1)   # [B, NT, H, BT, K]
    dk_out = jnp.flip(dk_flipped, axis=1)   # [B, NT, H, BT, K]
    dv_out = jnp.flip(dv_flipped, axis=1)   # [B, NT, H, BT, V]

    # Reshape back to [B, T, H, K/V]
    # [B, NT, H, BT, K] -> [B, H, NT, BT, K] -> [B, H, T, K] -> [B, T, H, K]
    def _unreshape(x, last_dim):
        x = x.transpose(0, 2, 1, 3, 4)    # [B, H, NT, BT, last_dim]
        x = x.reshape(B, H, T, last_dim)   # [B, H, T, last_dim]
        return x.transpose(0, 2, 1, 3)     # [B, T, H, last_dim]

    return _unreshape(dq_out, K), _unreshape(dk_out, K), _unreshape(dv_out, V)


# ============================================================
# custom_vjp wrapper
#
# MUTATION (bwd_fuse_dh):
#   _fwd: unchanged from fuse_fwd_A + fold_dh_pallas
#   _bwd: uses chunk_gla_bwd_fused_dh (single call) instead of
#         chunk_bwd_dh_pallas + chunk_gla_bwd_fused (two calls)
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
        # MUTATION (bwd_fuse_dh): single fused pallas_call replacing 2 calls
        dq, dk, dv = chunk_gla_bwd_fused_dh(
            q, k, v, g_cumsum, h, do, g_gamma, scale, C
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
