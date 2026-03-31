"""Chunked GLA (Gated Linear Attention) Pallas TPU kernel — activation checkpoint variant.

Combines THREE HBM reduction mutations:
  1. bf16 residuals for q/k/v — store as bf16 (saves 96MB)
  2. h recomputation — don't store h, recompute in backward (saves 32MB)
  3. Eliminate flips — reversed BlockSpec indexing (saves ~288MB temporaries)

Total residuals: only (q_bf16, k_bf16, v_bf16) = 48MB (was 224MB, -79% reduction!)

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
# In this activation_checkpoint variant, the forward kernel still
# produces h for output computation, but h is NOT saved to HBM
# as a residual. Only bf16 inputs are stored.
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
    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        b_k.astype(jnp.float32).T,       # [BK, BT]
        v_gated.astype(jnp.float32),      # [BT, BV]
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


def chunk_fwd_combined(q, k, v, g_gamma, scale, chunk_size):
    """Launch combined forward Pallas kernel (h propagation + output).

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
# h recomputation kernel (Change 2)
#
# Recomputes h from k, v, g_gamma in backward instead of storing
# it as a residual. Saves 32MB HBM.
# ============================================================


def _h_only_kernel(k_ref, v_ref, g_gamma, h_ref, scratch_ref, *, BT, NT):
    """Kernel that recomputes h state propagation from k, v, g_gamma.

    Same logic as forward h propagation, but runs standalone
    during backward to avoid storing h as a residual.
    """
    BK = k_ref.shape[3]
    BV = v_ref.shape[3]
    i_b, i_h, i_k, i_v, i_t = (
        pl.program_id(0), pl.program_id(1), pl.program_id(2),
        pl.program_id(3), pl.program_id(4),
    )
    b_g_last = g_gamma[i_h] * BT
    b_g_ramp = g_gamma[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)

    @pl.when(i_t == 0)
    def init():
        scratch_ref[:, :] = jnp.zeros((BK, BV), dtype=jnp.float32)

    h_ref[0, i_t, 0] = scratch_ref[...]

    b_k = k_ref[0, 0]
    b_v = v_ref[0, 0]

    scratch_ref[...] *= exp(b_g_last)
    v_gated = (b_v * jnp.exp(b_g_last - b_g_ramp)[:, None]).astype(b_v.dtype)

    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        b_k.astype(jnp.float32).T,
        v_gated.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


def recompute_h(k, v, g_gamma, chunk_size):
    """Recompute h from k, v, g_gamma using a dedicated Pallas kernel.

    Called during backward to avoid storing h [B, NT, H, K, V] as a residual.
    Saves 32MB HBM for reference dimensions.
    """
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

    h_all = pl.pallas_call(
        functools.partial(_h_only_kernel, BT=BT, NT=NT),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, 1, BT, BK), k_map),
                pl.BlockSpec((1, 1, BT, BV), v_map),
                pl.BlockSpec(memory_space=pltpu.SMEM),
            ],
            out_specs=[pl.BlockSpec((1, NT, 1, BK, BV), h_map)],
            scratch_shapes=[pltpu.VMEM((BK, BV), jnp.float32)],
        ),
        out_shape=[jax.ShapeDtypeStruct((B, NT, H, K_dim, V), k.dtype)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary", "arbitrary", "arbitrary"),
            disable_bounds_checks=True,
        ),
    )(k_t, v_t, g_gamma)

    return h_all


# ============================================================
# Forward orchestrator
# ============================================================


def chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size):
    """Chunked GLA forward pass.

    In the activation_checkpoint variant, h is still computed for
    output but will NOT be stored as a residual.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    h, o = chunk_fwd_combined(q, k, v, g_gamma, scale, C)

    return h, o


# ============================================================
# Backward: Fused dh + dq/dk/dv with REVERSED BlockSpec indexing
# (Change 3: Eliminate flips)
#
# Instead of flipping arrays before the kernel and flipping outputs
# after, we use reversed index maps: (b, h, NT-1-t, ki) to read
# chunks in reverse order directly, and write outputs to
# dq_ref[0, NT-1-i_t, 0] to place them in correct order.
#
# This eliminates 8 flip operations (~288MB of temporaries).
# ============================================================


def _chunk_gla_bwd_kernel(
    q_ref, k_ref, v_ref, h_ref, do_ref, g_gamma,
    dq_ref, dk_ref, dv_ref, scratch_ref,
    *, BT, NT, scale,
):
    """Fused backward kernel with reversed indexing (no flips).

    Processes chunks in REVERSE time order via reversed BlockSpec
    index maps. Writes outputs to NT-1-i_t to restore original order.
    """
    BK = q_ref.shape[3]
    BV = do_ref.shape[3]
    i_b, i_h, i_k, i_v, i_t = (
        pl.program_id(0), pl.program_id(1), pl.program_id(2),
        pl.program_id(3), pl.program_id(4),
    )

    # Recompute gating from g_gamma scalar
    b_g_ramp = g_gamma[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)  # [BT]
    b_g_last = g_gamma[i_h].astype(jnp.float32) * BT  # scalar

    # Initialize dh state to zeros at t=0 (first step of REVERSE scan,
    # i.e. the END of the original sequence — inputs come reversed via index maps)
    @pl.when(i_t == 0)
    def init():
        scratch_ref[:, :] = jnp.zeros((BK, BV), dtype=jnp.float32)

    # Load inputs (already reversed via BlockSpec index maps)
    b_q = q_ref[0, 0]                         # [BT, BK]
    b_k = k_ref[0, 0]                         # [BT, BK]
    b_v = v_ref[0, 0]                         # [BT, BV]
    b_h = h_ref[0, 0, 0].astype(jnp.float32)  # [BK, BV]
    b_do = do_ref[0, 0]                        # [BT, BV]

    # Current dh from scratch
    b_dh = scratch_ref[...]    # [BK, BV]

    b_gn = b_g_last  # scalar: g_gamma * BT

    # Pre-compute exp/gate values
    exp_pos = jnp.exp(b_g_ramp)                    # [BT]
    exp_neg = jnp.exp(-b_g_ramp)                   # [BT]
    exp_gn_minus = jnp.exp(b_gn - b_g_ramp)       # [BT]

    k_neg = (b_k * exp_neg[:, None]).astype(b_k.dtype)           # [BT, K]
    k_decay = (b_k * exp_gn_minus[:, None]).astype(b_k.dtype)    # [BT, K]
    q_pos = (b_q * exp_pos[:, None]).astype(b_q.dtype)           # [BT, K]

    # Phase 1: Recompute A and compute dA
    b_a = jnp.dot(q_pos, k_neg.T,
                  precision=jax.lax.Precision.HIGHEST,
                  preferred_element_type=jnp.float32) * scale    # [BT, BT]

    b_dA_raw = jnp.dot(b_do.astype(b_v.dtype), b_v.T,
                       precision=jax.lax.Precision.HIGHEST,
                       preferred_element_type=jnp.float32) * scale  # [BT, BT]

    # Phase 2: Causal masks
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_dA = jnp.where(mask, b_dA_raw, 0.0)
    b_a_masked = jnp.where(mask, b_a, 0.0)

    # Phase 3: Four independent dot products
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

    # Phase 4: Intra-chunk dq and dk
    b_dq_intra_raw = jnp.dot(b_dA.astype(k_neg.dtype), k_neg,
                              precision=jax.lax.Precision.HIGHEST,
                              preferred_element_type=jnp.float32)  # [BT, K]

    b_dk_intra_raw = jnp.dot(b_dA.T.astype(q_pos.dtype), q_pos,
                              precision=jax.lax.Precision.HIGHEST,
                              preferred_element_type=jnp.float32)  # [BT, K]

    # Phase 5: Combine and write to REVERSED output slots (NT-1-i_t)
    # This eliminates the need for flipping outputs after the kernel
    dv_ref[0, NT - 1 - i_t, 0] = (b_dv_intra + b_dv_inter).astype(dv_ref.dtype)

    b_dq = b_dq_intra_raw * exp_pos[:, None] + b_dq_inter * (scale * exp_pos[:, None])
    dq_ref[0, NT - 1 - i_t, 0] = b_dq.astype(dq_ref.dtype)

    b_dk = b_dk_intra_raw * exp_neg[:, None] + b_dk_inter * exp_gn_minus[:, None]
    dk_ref[0, NT - 1 - i_t, 0] = b_dk.astype(dk_ref.dtype)

    # Phase 6: Update dh state for next reverse step
    scratch_ref[...] *= exp(b_g_last)

    q_hat = (b_q * exp(b_g_ramp)[:, None] * scale).astype(jnp.float32)

    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        q_hat.T,                           # [BK, BT]
        b_do.astype(jnp.float32),          # [BT, BV]
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


def chunk_gla_bwd_eliminate_gcumsum(q, k, v, h, do, g_gamma, scale, chunk_size):
    """Fused backward with reversed BlockSpec indexing (no flips).

    Change 3: Instead of flipping 5 input arrays and 3 output arrays,
    we use reversed index maps to read chunks in reverse order.
    This eliminates 8 flip operations and ~288MB of temporaries.
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

    # NO FLIPS! Instead, reversed index maps read chunks in reverse order.
    # Flatten back to (B, H, NT*BT, dim) for 4D BlockSpec
    q_flat = q_chunked.reshape(B, H, NT * BT, K)
    k_flat = k_chunked.reshape(B, H, NT * BT, K)
    v_flat = v_chunked.reshape(B, H, NT * BT, V)
    do_flat = do_chunked.reshape(B, H, NT * BT, V)

    grid = (B, H, pl.cdiv(K, BK), pl.cdiv(V, BV), NT)

    # REVERSED input index maps: read chunk NT-1-t instead of t
    def q_map(b, h, ki, vi, t):  return b, h, NT - 1 - t, ki
    def k_map(b, h, ki, vi, t):  return b, h, NT - 1 - t, ki
    def v_map(b, h, ki, vi, t):  return b, h, NT - 1 - t, vi
    def h_map(b, h, ki, vi, t):  return b, h, NT - 1 - t, 0, 0
    def do_map(b, h, ki, vi, t): return b, h, NT - 1 - t, vi

    # Output index maps — 5D outputs [B, NT, H, BT, K/V]
    # Outputs are written to NT-1-i_t inside the kernel, so no flip needed
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
                pl.BlockSpec((1, 1, BT, BK), q_map),        # q: reversed
                pl.BlockSpec((1, 1, BT, BK), k_map),        # k: reversed
                pl.BlockSpec((1, 1, BT, BV), v_map),        # v: reversed
                pl.BlockSpec((1, 1, 1, BK, BV), h_map),     # h: reversed
                pl.BlockSpec((1, 1, BT, BV), do_map),       # do: reversed
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

    # NO OUTPUT FLIPS needed — kernel wrote to NT-1-i_t directly

    # Reshape [B, NT, H, BT, K] -> [B, T, H, K].
    dq = dq_5d.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K)
    dk = dk_5d.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K)
    dv = dv_5d.transpose(0, 1, 3, 2, 4).reshape(B, T, H, V)

    return dq, dk, dv


# ============================================================
# custom_vjp wrapper — ACTIVATION CHECKPOINT variant
#
# Three combined mutations:
#   1. bf16 residuals: store q/k/v as bf16 (saves 96MB)
#   2. h recomputation: don't store h, recompute in backward (saves 32MB)
#   3. Flip elimination: reversed BlockSpec in backward (saves ~288MB temp)
#
# Total residuals: (q_bf16, k_bf16, v_bf16) = 48MB
# Baseline stored: (q, k, v, h) = 224MB  ->  79% reduction
# ============================================================


def chunk_gla(q, k, v, g_gamma, scale, chunk_size):
    """Chunked GLA with activation checkpointing custom_vjp."""
    @jax.custom_vjp
    def _compute(q, k, v):
        _, o = chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size)
        return o

    def _fwd(q, k, v):
        # Only run forward for output — do NOT store h
        _, o = chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size)
        # Store only bf16 inputs as residuals (Change 1 + Change 2)
        return o, (q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v.astype(jnp.bfloat16))

    def _bwd(residuals, do):
        # Upcast bf16 residuals back to float32 (Change 1)
        q_bf16, k_bf16, v_bf16 = residuals
        q = q_bf16.astype(jnp.float32)
        k = k_bf16.astype(jnp.float32)
        v = v_bf16.astype(jnp.float32)
        # Recompute h from k, v, g_gamma (Change 2)
        h = recompute_h(k, v, g_gamma, chunk_size)
        # Backward with reversed BlockSpec indexing (Change 3)
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
