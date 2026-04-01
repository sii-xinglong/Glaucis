import functools
import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _make_test_data(B, T, H, K, V, chunk_size, seed=42):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    q = jax.random.normal(k1, (B, T, H, K), dtype=jnp.bfloat16)
    k_arr = jax.random.normal(k2, (B, T, H, K), dtype=jnp.bfloat16)
    v = jax.random.normal(k3, (B, T, H, V), dtype=jnp.bfloat16)
    g_gamma = -jnp.abs(jax.random.normal(k4, (H,), dtype=jnp.float32)) * 0.1
    return q, k_arr, v, g_gamma


def exp(x):
    return jnp.exp(x.astype(jnp.float32))


# EVOLVE-BLOCK-START
# ============================================================
# Forward: Fused kernel -- 4-step unrolled (process 4 time steps per grid iteration)
# ============================================================
#
# Profile motivation: building on L2 bf16_h_residual (1.152x) and the
# 2-step unrolling pattern. By processing 4 sub-steps per grid iteration
# the kernel body contains 4x the matmuls. The Mosaic compiler can
# interleave independent matmuls from different sub-steps across mxu0/mxu1,
# since after one sub-step's h-update the next sub-step's independent ops
# (A recompute, output computation) can begin while the h-update dot is
# finishing. This reduces grid overhead by 4x and gives the compiler more
# scheduling freedom.
#
# Grid changes from (B, H, 1, 1, NT) to (B, H, 1, 1, NT//4).
# Block shapes quadrupled along the time axis to load all 4 sub-steps.
# h output: 4 snapshots per iteration via block (1, 4, 1, BK, BV).
# Keeps L2's output scratch buffer for output accumulation.
# Keeps L2's deferred b_h load pattern in backward.
# ============================================================


def _chunk_fwd_fused_kernel_4step(
    q_ref,
    k_ref,
    v_ref,
    g_gamma,
    h_ref,
    o_ref,
    scratch_ref,
    o_scratch_ref,
    *,
    BT,
    NT,
    scale,
):
    """Fused forward kernel -- 4-step unrolled.

    Each grid iteration processes 4 consecutive time steps (4*BT rows).
    q_ref/k_ref/v_ref/o_ref have block shape (1, 1, 4*BT, ...).
    h_ref has block shape (1, 4, 1, BK, BV) to write 4 h snapshots.
    scratch_ref [BK, BV]: h state (f32), persists across time steps.
    o_scratch_ref [BT, BV]: output accumulator (f32), per sub-step.
    """
    BK = k_ref.shape[3]
    BV = v_ref.shape[3]
    i_b, i_h, i_k, i_v, i_t = (
        pl.program_id(0),
        pl.program_id(1),
        pl.program_id(2),
        pl.program_id(3),
        pl.program_id(4),
    )

    # Per-position gating ramp for one BT-sized sub-step
    g_val = g_gamma[i_h].astype(jnp.float32)
    b_g_ramp = g_val * (jnp.arange(0, BT) + 1)
    b_g_last = g_val * BT

    # Precompute exp values used in all 4 sub-steps
    exp_g_ramp = exp(b_g_ramp)           # [BT]
    exp_neg_g_ramp = exp(-b_g_ramp)      # [BT]
    exp_gn_minus = exp(b_g_last - b_g_ramp)  # [BT]
    exp_g_last = exp(b_g_last)           # scalar

    # Causal mask (same for all sub-steps)
    m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]

    # --- Initialize h state in scratch at first grid iteration ---
    @pl.when(i_t == 0)
    def init():
        scratch_ref[:, :] = jnp.zeros((BK, BV), dtype=jnp.float32)

    # =============================================
    # SUB-STEP 0: process rows [0, BT) of the 4*BT block
    # =============================================

    # Save h snapshot for sub-step 0
    h_ref[0, 0, 0] = scratch_ref[...].astype(h_ref.dtype)

    # Load sub-step 0 data
    b_q_0 = q_ref[0, 0, pl.ds(0, BT), :]      # [BT, BK]
    b_k_0 = k_ref[0, 0, pl.ds(0, BT), :]      # [BT, BK]
    b_v_0 = v_ref[0, 0, pl.ds(0, BT), :]      # [BT, BV]

    # Gated q and k
    b_qg_0 = b_q_0.astype(jnp.float32) * exp_g_ramp[:, None]
    b_kg_0 = b_k_0.astype(jnp.float32) * exp_neg_g_ramp[:, None]

    # A = qg @ kg.T * scale  [BT, BT]
    b_A_0 = (
        jnp.dot(
            b_qg_0,
            b_kg_0.T,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * scale
    )
    b_A_masked_0 = jnp.where(m_s, b_A_0, 0.0)

    # Inter-chunk: qg @ h * scale  [BT, BV] -- into output scratch
    o_scratch_ref[...] = (
        jnp.dot(
            b_qg_0,
            scratch_ref[...],
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * scale
    )

    # Intra-chunk: A_masked @ v  [BT, BV] -- add to output scratch
    o_scratch_ref[...] = o_scratch_ref[...] + jnp.dot(
        b_A_masked_0,
        b_v_0.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # Write accumulated output
    o_ref[0, 0, pl.ds(0, BT), :] = o_scratch_ref[...].astype(o_ref.dtype)

    # Update h for sub-step 0 -> sub-step 1
    scratch_ref[...] *= exp_g_last
    v_gated_0 = b_v_0.astype(jnp.float32) * exp_gn_minus[:, None]
    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        b_k_0.astype(jnp.float32).T,
        v_gated_0,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # =============================================
    # SUB-STEP 1: process rows [BT, 2*BT) of the 4*BT block
    # =============================================

    h_ref[0, 1, 0] = scratch_ref[...].astype(h_ref.dtype)

    b_q_1 = q_ref[0, 0, pl.ds(BT, BT), :]
    b_k_1 = k_ref[0, 0, pl.ds(BT, BT), :]
    b_v_1 = v_ref[0, 0, pl.ds(BT, BT), :]

    b_qg_1 = b_q_1.astype(jnp.float32) * exp_g_ramp[:, None]
    b_kg_1 = b_k_1.astype(jnp.float32) * exp_neg_g_ramp[:, None]

    b_A_1 = (
        jnp.dot(
            b_qg_1,
            b_kg_1.T,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * scale
    )
    b_A_masked_1 = jnp.where(m_s, b_A_1, 0.0)

    o_scratch_ref[...] = (
        jnp.dot(
            b_qg_1,
            scratch_ref[...],
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * scale
    )
    o_scratch_ref[...] = o_scratch_ref[...] + jnp.dot(
        b_A_masked_1,
        b_v_1.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    o_ref[0, 0, pl.ds(BT, BT), :] = o_scratch_ref[...].astype(o_ref.dtype)

    scratch_ref[...] *= exp_g_last
    v_gated_1 = b_v_1.astype(jnp.float32) * exp_gn_minus[:, None]
    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        b_k_1.astype(jnp.float32).T,
        v_gated_1,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # =============================================
    # SUB-STEP 2: process rows [2*BT, 3*BT) of the 4*BT block
    # =============================================

    h_ref[0, 2, 0] = scratch_ref[...].astype(h_ref.dtype)

    b_q_2 = q_ref[0, 0, pl.ds(2 * BT, BT), :]
    b_k_2 = k_ref[0, 0, pl.ds(2 * BT, BT), :]
    b_v_2 = v_ref[0, 0, pl.ds(2 * BT, BT), :]

    b_qg_2 = b_q_2.astype(jnp.float32) * exp_g_ramp[:, None]
    b_kg_2 = b_k_2.astype(jnp.float32) * exp_neg_g_ramp[:, None]

    b_A_2 = (
        jnp.dot(
            b_qg_2,
            b_kg_2.T,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * scale
    )
    b_A_masked_2 = jnp.where(m_s, b_A_2, 0.0)

    o_scratch_ref[...] = (
        jnp.dot(
            b_qg_2,
            scratch_ref[...],
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * scale
    )
    o_scratch_ref[...] = o_scratch_ref[...] + jnp.dot(
        b_A_masked_2,
        b_v_2.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    o_ref[0, 0, pl.ds(2 * BT, BT), :] = o_scratch_ref[...].astype(o_ref.dtype)

    scratch_ref[...] *= exp_g_last
    v_gated_2 = b_v_2.astype(jnp.float32) * exp_gn_minus[:, None]
    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        b_k_2.astype(jnp.float32).T,
        v_gated_2,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # =============================================
    # SUB-STEP 3: process rows [3*BT, 4*BT) of the 4*BT block
    # =============================================

    h_ref[0, 3, 0] = scratch_ref[...].astype(h_ref.dtype)

    b_q_3 = q_ref[0, 0, pl.ds(3 * BT, BT), :]
    b_k_3 = k_ref[0, 0, pl.ds(3 * BT, BT), :]
    b_v_3 = v_ref[0, 0, pl.ds(3 * BT, BT), :]

    b_qg_3 = b_q_3.astype(jnp.float32) * exp_g_ramp[:, None]
    b_kg_3 = b_k_3.astype(jnp.float32) * exp_neg_g_ramp[:, None]

    b_A_3 = (
        jnp.dot(
            b_qg_3,
            b_kg_3.T,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * scale
    )
    b_A_masked_3 = jnp.where(m_s, b_A_3, 0.0)

    o_scratch_ref[...] = (
        jnp.dot(
            b_qg_3,
            scratch_ref[...],
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * scale
    )
    o_scratch_ref[...] = o_scratch_ref[...] + jnp.dot(
        b_A_masked_3,
        b_v_3.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    o_ref[0, 0, pl.ds(3 * BT, BT), :] = o_scratch_ref[...].astype(o_ref.dtype)

    scratch_ref[...] *= exp_g_last
    v_gated_3 = b_v_3.astype(jnp.float32) * exp_gn_minus[:, None]
    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        b_k_3.astype(jnp.float32).T,
        v_gated_3,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


@functools.partial(jax.jit, static_argnames=["chunk_size", "scale"])
def chunk_fwd_fused_g_gamma(q, k, v, g_gamma, scale, chunk_size):
    """Fused chunked GLA forward pass -- 4-step unrolled.

    Grid iterates over NT//4 time steps; each iteration processes 4 chunks.
    Requires NT = T // chunk_size to be divisible by 4.
    """
    BK, BV, BT = 128, 128, chunk_size
    B, T, H, K_dim = q.shape
    V = v.shape[-1]
    NT = T // BT

    assert T % BT == 0, f"T ({T}) must be a multiple of chunk_size ({BT})"
    assert K_dim == BK, f"K must be {BK}, got {K_dim}"
    assert V == BV, f"V must be {BV}, got {V}"
    assert NT % 4 == 0, f"NT ({NT}) must be divisible by 4 for 4-step unrolling"

    q_t = jnp.transpose(q, (0, 2, 1, 3))   # [B, H, T, K]
    k_t = jnp.transpose(k, (0, 2, 1, 3))
    v_t = jnp.transpose(v, (0, 2, 1, 3))

    NT4 = NT // 4
    grid = (B, H, 1, 1, NT4)

    # BlockSpec multiplies the returned index by the block_shape to get
    # the start position. With block (1, 1, 4*BT, BK), returning t gives
    # start = t * 4*BT along the T axis -- exactly the group of 4
    # BT-sized chunks we need for grid iteration t.
    def q_map(b, h, ki, vi, t): return b, h, t, ki
    def k_map(b, h, ki, vi, t): return b, h, t, ki
    def v_map(b, h, ki, vi, t): return b, h, t, vi
    def o_map(b, h, ki, vi, t): return b, h, t, vi

    # h output: [B, NT, H, K, V], block (1, 4, 1, BK, BV).
    # Returning t gives start = t * 4 along the NT axis, covering
    # h[b, 4*t:4*t+4, h, :, :] -- exactly the 4 snapshots we write.
    def h_map(b, h, ki, vi, t): return b, t, h, ki, vi

    h_all, o_t = pl.pallas_call(
        functools.partial(
            _chunk_fwd_fused_kernel_4step, BT=BT, NT=NT, scale=scale
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, 1, 4 * BT, BK), q_map),   # q
                pl.BlockSpec((1, 1, 4 * BT, BK), k_map),   # k
                pl.BlockSpec((1, 1, 4 * BT, BV), v_map),   # v
                pl.BlockSpec(memory_space=pltpu.SMEM),      # g_gamma
            ],
            out_specs=[
                pl.BlockSpec((1, 4, 1, BK, BV), h_map),    # h: 4 snapshots
                pl.BlockSpec((1, 1, 4 * BT, BV), o_map),   # o: 4*BT rows
            ],
            scratch_shapes=[
                pltpu.VMEM((BK, BV), jnp.float32),   # h state scratch
                pltpu.VMEM((BT, BV), jnp.float32),   # output accumulator scratch
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((B, NT, H, K_dim, V), q.dtype),
            jax.ShapeDtypeStruct((B, H, T, V), q.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel", "parallel", "parallel", "parallel", "arbitrary",
            ),
            disable_bounds_checks=True,
        ),
    )(q_t, k_t, v_t, g_gamma)

    o = jnp.transpose(o_t, (0, 2, 1, 3))
    return h_all, o


# ============================================================
# Backward: Fused dh reverse propagation + dq/dk/dv -- 4-step unrolled
#
# Keeps L2's deferred b_h load pattern within each sub-step:
#   - Compute dv first (does NOT need b_h)
#   - Load b_h just before dq_inter/dk_inter
# ============================================================


def _chunk_bwd_fused_kernel_4step(
    q_ref,
    k_ref,
    v_ref,
    h_ref,
    do_ref,
    g_gamma,
    dq_ref,
    dk_ref,
    dv_ref,
    scratch_ref,
    *,
    BT,
    NT,
    scale,
):
    """Fused backward kernel -- 4-step unrolled, reverse time.

    Each grid iteration processes 4 consecutive time steps in reverse order.
    Inputs are loaded in reverse via index_maps that flip the time axis.
    Within each iteration we process sub-steps in reverse: 3, 2, 1, 0,
    matching the reverse scan order.

    Within the loaded 4*BT block:
      rows [0, BT)       = sub-step 0 (earliest forward time)
      rows [BT, 2*BT)    = sub-step 1
      rows [2*BT, 3*BT)  = sub-step 2
      rows [3*BT, 4*BT)  = sub-step 3 (latest forward time)

    h_ref block (1, 1, 4, BK, BV):
      h_ref[0, 0, s] = h at sub-step s (s=0 earliest, s=3 latest)

    Output indices: grid iteration i_t processes reversed block at NT4-1-i_t.
    Forward-time base = 4*(NT4-1-i_t). Sub-step s_orig within block has
    forward-time index = 4*(NT4-1-i_t) + s_orig.
    """
    BK = q_ref.shape[3]
    BV = do_ref.shape[3]
    NT4 = NT // 4
    i_b, i_h, i_k, i_v, i_t = (
        pl.program_id(0),
        pl.program_id(1),
        pl.program_id(2),
        pl.program_id(3),
        pl.program_id(4),
    )

    g_val = g_gamma[i_h].astype(jnp.float32)
    b_g_ramp = g_val * (jnp.arange(0, BT) + 1)
    b_g_last = g_val * BT

    # Precompute exp values
    exp_pos = exp(b_g_ramp)
    exp_neg = exp(-b_g_ramp)
    exp_gn_minus = exp(b_g_last - b_g_ramp)
    exp_g_last = exp(b_g_last)

    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]

    @pl.when(i_t == 0)
    def init():
        scratch_ref[:, :] = jnp.zeros((BK, BV), dtype=jnp.float32)

    # Compute forward-time output slot indices for this grid iteration.
    # Grid iteration i_t (reversed) processes the block at original
    # forward-time base = 4*(NT4-1-i_t).
    # Sub-step s_orig has forward-time index = base + s_orig.
    i_base = 4 * (NT4 - 1 - i_t)
    i_slot_0 = i_base + 0  # earliest in this block
    i_slot_1 = i_base + 1
    i_slot_2 = i_base + 2
    i_slot_3 = i_base + 3  # latest in this block

    # =============================================
    # REVERSE SUB-STEP: process sub-step 3 (LATEST, rows [3*BT, 4*BT))
    # =============================================

    b_q = q_ref[0, 0, pl.ds(3 * BT, BT), :]
    b_k = k_ref[0, 0, pl.ds(3 * BT, BT), :]
    b_v = v_ref[0, 0, pl.ds(3 * BT, BT), :]
    b_do = do_ref[0, 0, pl.ds(3 * BT, BT), :]
    b_dh = scratch_ref[...]

    k_neg = b_k.astype(jnp.float32) * exp_neg[:, None]
    q_pos = b_q.astype(jnp.float32) * exp_pos[:, None]

    # Recompute A and dA
    b_a = (
        jnp.dot(
            q_pos, k_neg.T,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ) * scale
    )
    b_dA_raw = (
        jnp.dot(
            b_do.astype(jnp.float32), b_v.astype(jnp.float32).T,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ) * scale
    )

    b_dA = jnp.where(mask, b_dA_raw, 0.0)
    b_a_masked = jnp.where(mask, b_a, 0.0)

    # dv first (L2 deferred b_h pattern: dv does NOT need b_h)
    b_dv_intra = jnp.dot(
        b_a_masked.T, b_do.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    k_decay = b_k.astype(jnp.float32) * exp_gn_minus[:, None]
    b_dv_inter = jnp.dot(
        k_decay, b_dh,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    dv_ref[0, i_slot_3, 0] = (b_dv_intra + b_dv_inter).astype(dv_ref.dtype)

    # NOW load b_h for dq_inter/dk_inter
    b_h = h_ref[0, 0, 3].astype(jnp.float32)

    b_dq_inter = jnp.dot(
        b_do.astype(jnp.float32), b_h.T,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b_dk_inter = jnp.dot(
        b_v.astype(jnp.float32), b_dh.T,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # Intra-chunk dq and dk
    b_dq_intra_raw = jnp.dot(
        b_dA, k_neg,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b_dk_intra_raw = jnp.dot(
        b_dA.T, q_pos,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    b_dq = b_dq_intra_raw * exp_pos[:, None] + b_dq_inter * (scale * exp_pos[:, None])
    dq_ref[0, i_slot_3, 0] = b_dq.astype(dq_ref.dtype)

    b_dk = b_dk_intra_raw * exp_neg[:, None] + b_dk_inter * exp_gn_minus[:, None]
    dk_ref[0, i_slot_3, 0] = b_dk.astype(dk_ref.dtype)

    # Update dh state
    scratch_ref[...] *= exp_g_last
    q_hat = (b_q.astype(jnp.float32) * exp_pos[:, None] * scale)
    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        q_hat.T,
        b_do.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # =============================================
    # REVERSE SUB-STEP: process sub-step 2 (rows [2*BT, 3*BT))
    # =============================================

    b_q2 = q_ref[0, 0, pl.ds(2 * BT, BT), :]
    b_k2 = k_ref[0, 0, pl.ds(2 * BT, BT), :]
    b_v2 = v_ref[0, 0, pl.ds(2 * BT, BT), :]
    b_do2 = do_ref[0, 0, pl.ds(2 * BT, BT), :]
    b_dh2 = scratch_ref[...]

    k_neg2 = b_k2.astype(jnp.float32) * exp_neg[:, None]
    q_pos2 = b_q2.astype(jnp.float32) * exp_pos[:, None]

    b_a2 = (
        jnp.dot(
            q_pos2, k_neg2.T,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ) * scale
    )
    b_dA_raw2 = (
        jnp.dot(
            b_do2.astype(jnp.float32), b_v2.astype(jnp.float32).T,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ) * scale
    )

    b_dA2 = jnp.where(mask, b_dA_raw2, 0.0)
    b_a_masked2 = jnp.where(mask, b_a2, 0.0)

    # dv first (deferred b_h)
    b_dv_intra2 = jnp.dot(
        b_a_masked2.T, b_do2.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    k_decay2 = b_k2.astype(jnp.float32) * exp_gn_minus[:, None]
    b_dv_inter2 = jnp.dot(
        k_decay2, b_dh2,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    dv_ref[0, i_slot_2, 0] = (b_dv_intra2 + b_dv_inter2).astype(dv_ref.dtype)

    # NOW load b_h
    b_h2 = h_ref[0, 0, 2].astype(jnp.float32)

    b_dq_inter2 = jnp.dot(
        b_do2.astype(jnp.float32), b_h2.T,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b_dk_inter2 = jnp.dot(
        b_v2.astype(jnp.float32), b_dh2.T,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    b_dq_intra_raw2 = jnp.dot(
        b_dA2, k_neg2,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b_dk_intra_raw2 = jnp.dot(
        b_dA2.T, q_pos2,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    b_dq2 = b_dq_intra_raw2 * exp_pos[:, None] + b_dq_inter2 * (scale * exp_pos[:, None])
    dq_ref[0, i_slot_2, 0] = b_dq2.astype(dq_ref.dtype)

    b_dk2 = b_dk_intra_raw2 * exp_neg[:, None] + b_dk_inter2 * exp_gn_minus[:, None]
    dk_ref[0, i_slot_2, 0] = b_dk2.astype(dk_ref.dtype)

    # Update dh state
    scratch_ref[...] *= exp_g_last
    q_hat2 = (b_q2.astype(jnp.float32) * exp_pos[:, None] * scale)
    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        q_hat2.T,
        b_do2.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # =============================================
    # REVERSE SUB-STEP: process sub-step 1 (rows [BT, 2*BT))
    # =============================================

    b_q3 = q_ref[0, 0, pl.ds(BT, BT), :]
    b_k3 = k_ref[0, 0, pl.ds(BT, BT), :]
    b_v3 = v_ref[0, 0, pl.ds(BT, BT), :]
    b_do3 = do_ref[0, 0, pl.ds(BT, BT), :]
    b_dh3 = scratch_ref[...]

    k_neg3 = b_k3.astype(jnp.float32) * exp_neg[:, None]
    q_pos3 = b_q3.astype(jnp.float32) * exp_pos[:, None]

    b_a3 = (
        jnp.dot(
            q_pos3, k_neg3.T,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ) * scale
    )
    b_dA_raw3 = (
        jnp.dot(
            b_do3.astype(jnp.float32), b_v3.astype(jnp.float32).T,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ) * scale
    )

    b_dA3 = jnp.where(mask, b_dA_raw3, 0.0)
    b_a_masked3 = jnp.where(mask, b_a3, 0.0)

    # dv first (deferred b_h)
    b_dv_intra3 = jnp.dot(
        b_a_masked3.T, b_do3.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    k_decay3 = b_k3.astype(jnp.float32) * exp_gn_minus[:, None]
    b_dv_inter3 = jnp.dot(
        k_decay3, b_dh3,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    dv_ref[0, i_slot_1, 0] = (b_dv_intra3 + b_dv_inter3).astype(dv_ref.dtype)

    # NOW load b_h
    b_h3 = h_ref[0, 0, 1].astype(jnp.float32)

    b_dq_inter3 = jnp.dot(
        b_do3.astype(jnp.float32), b_h3.T,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b_dk_inter3 = jnp.dot(
        b_v3.astype(jnp.float32), b_dh3.T,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    b_dq_intra_raw3 = jnp.dot(
        b_dA3, k_neg3,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b_dk_intra_raw3 = jnp.dot(
        b_dA3.T, q_pos3,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    b_dq3 = b_dq_intra_raw3 * exp_pos[:, None] + b_dq_inter3 * (scale * exp_pos[:, None])
    dq_ref[0, i_slot_1, 0] = b_dq3.astype(dq_ref.dtype)

    b_dk3 = b_dk_intra_raw3 * exp_neg[:, None] + b_dk_inter3 * exp_gn_minus[:, None]
    dk_ref[0, i_slot_1, 0] = b_dk3.astype(dk_ref.dtype)

    # Update dh state
    scratch_ref[...] *= exp_g_last
    q_hat3 = (b_q3.astype(jnp.float32) * exp_pos[:, None] * scale)
    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        q_hat3.T,
        b_do3.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # =============================================
    # REVERSE SUB-STEP: process sub-step 0 (EARLIEST, rows [0, BT))
    # =============================================

    b_q4 = q_ref[0, 0, pl.ds(0, BT), :]
    b_k4 = k_ref[0, 0, pl.ds(0, BT), :]
    b_v4 = v_ref[0, 0, pl.ds(0, BT), :]
    b_do4 = do_ref[0, 0, pl.ds(0, BT), :]
    b_dh4 = scratch_ref[...]

    k_neg4 = b_k4.astype(jnp.float32) * exp_neg[:, None]
    q_pos4 = b_q4.astype(jnp.float32) * exp_pos[:, None]

    b_a4 = (
        jnp.dot(
            q_pos4, k_neg4.T,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ) * scale
    )
    b_dA_raw4 = (
        jnp.dot(
            b_do4.astype(jnp.float32), b_v4.astype(jnp.float32).T,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ) * scale
    )

    b_dA4 = jnp.where(mask, b_dA_raw4, 0.0)
    b_a_masked4 = jnp.where(mask, b_a4, 0.0)

    # dv first (deferred b_h)
    b_dv_intra4 = jnp.dot(
        b_a_masked4.T, b_do4.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    k_decay4 = b_k4.astype(jnp.float32) * exp_gn_minus[:, None]
    b_dv_inter4 = jnp.dot(
        k_decay4, b_dh4,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    dv_ref[0, i_slot_0, 0] = (b_dv_intra4 + b_dv_inter4).astype(dv_ref.dtype)

    # NOW load b_h
    b_h4 = h_ref[0, 0, 0].astype(jnp.float32)

    b_dq_inter4 = jnp.dot(
        b_do4.astype(jnp.float32), b_h4.T,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b_dk_inter4 = jnp.dot(
        b_v4.astype(jnp.float32), b_dh4.T,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    b_dq_intra_raw4 = jnp.dot(
        b_dA4, k_neg4,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    b_dk_intra_raw4 = jnp.dot(
        b_dA4.T, q_pos4,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    b_dq4 = b_dq_intra_raw4 * exp_pos[:, None] + b_dq_inter4 * (scale * exp_pos[:, None])
    dq_ref[0, i_slot_0, 0] = b_dq4.astype(dq_ref.dtype)

    b_dk4 = b_dk_intra_raw4 * exp_neg[:, None] + b_dk_inter4 * exp_gn_minus[:, None]
    dk_ref[0, i_slot_0, 0] = b_dk4.astype(dk_ref.dtype)

    # Update dh state for next grid iteration
    scratch_ref[...] *= exp_g_last
    q_hat4 = (b_q4.astype(jnp.float32) * exp_pos[:, None] * scale)
    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        q_hat4.T,
        b_do4.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


@functools.partial(jax.jit, static_argnames=["chunk_size", "scale"])
def chunk_bwd_fused_g_gamma(q, k, v, h, do, g_gamma, scale, chunk_size):
    """Fused chunked GLA backward pass -- 4-step unrolled.

    Grid iterates over NT//4 time steps in reverse; each iteration
    processes 4 consecutive chunks. Keeps L2's deferred b_h load
    pattern within each sub-step.
    """
    BK, BV, BT = 128, 128, chunk_size
    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = T // BT

    assert T % BT == 0
    assert K == BK
    assert V == BV
    assert NT % 4 == 0, f"NT ({NT}) must be divisible by 4 for 4-step unrolling"

    NT4 = NT // 4

    q_t = jnp.transpose(q, (0, 2, 1, 3))    # [B, H, T, K]
    k_t = jnp.transpose(k, (0, 2, 1, 3))
    v_t = jnp.transpose(v, (0, 2, 1, 3))
    do_t = jnp.transpose(do, (0, 2, 1, 3))
    h_bhntKV = jnp.transpose(h, (0, 2, 1, 3, 4))  # [B, H, NT, K, V]

    grid = (B, H, 1, 1, NT4)

    # Reverse input index maps -- each iteration loads 4*BT rows.
    # BlockSpec multiplies returned index by block_shape, so returning
    # (NT4-1-t) gives start = (NT4-1-t)*4*BT along T -- the correct
    # reversed group of 4 chunks.
    # Within the loaded 4*BT block:
    #   rows [0, BT)       = sub-step 0 (earliest forward time)
    #   rows [BT, 2*BT)    = sub-step 1
    #   rows [2*BT, 3*BT)  = sub-step 2
    #   rows [3*BT, 4*BT)  = sub-step 3 (latest forward time)
    # Kernel processes 3 first, then 2, 1, 0 (reverse scan order).

    def q_map(b, h, ki, vi, t): return b, h, NT4 - 1 - t, ki
    def k_map(b, h, ki, vi, t): return b, h, NT4 - 1 - t, ki
    def v_map(b, h, ki, vi, t): return b, h, NT4 - 1 - t, vi
    def do_map(b, h, ki, vi, t): return b, h, NT4 - 1 - t, vi

    # h input: [B, H, NT, K, V], block (1, 1, 4, BK, BV).
    # Returning (NT4-1-t) gives start = (NT4-1-t)*4 along NT dim,
    # loading h[b, h, 4*(NT4-1-t):4*(NT4-1-t)+4, :, :].
    # Within block: h_ref[0,0,s] = h at sub-step s.
    def h_map(b, h, ki, vi, t): return b, h, NT4 - 1 - t, ki, vi

    def out_k_map(b, h, ki, vi, t): return b, 0, h, 0, 0
    def out_v_map(b, h, ki, vi, t): return b, 0, h, 0, 0

    dq_5d, dk_5d, dv_5d = pl.pallas_call(
        functools.partial(
            _chunk_bwd_fused_kernel_4step, BT=BT, NT=NT, scale=scale
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, 1, 4 * BT, BK), q_map),    # q
                pl.BlockSpec((1, 1, 4 * BT, BK), k_map),    # k
                pl.BlockSpec((1, 1, 4 * BT, BV), v_map),    # v
                pl.BlockSpec((1, 1, 4, BK, BV), h_map),     # h: 4 snapshots
                pl.BlockSpec((1, 1, 4 * BT, BV), do_map),   # do
                pl.BlockSpec(memory_space=pltpu.SMEM),       # g_gamma
            ],
            out_specs=[
                pl.BlockSpec((1, NT, 1, BT, BK), out_k_map),   # dq
                pl.BlockSpec((1, NT, 1, BT, BK), out_k_map),   # dk
                pl.BlockSpec((1, NT, 1, BT, BV), out_v_map),   # dv
            ],
            scratch_shapes=[pltpu.VMEM((BK, BV), jnp.float32)],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((B, NT, H, BT, K), q.dtype),
            jax.ShapeDtypeStruct((B, NT, H, BT, K), k.dtype),
            jax.ShapeDtypeStruct((B, NT, H, BT, V), v.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel", "parallel", "parallel", "parallel", "arbitrary",
            ),
            disable_bounds_checks=True,
        ),
    )(q_t, k_t, v_t, h_bhntKV, do_t, g_gamma)

    dq = dq_5d.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K)
    dk = dk_5d.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K)
    dv = dv_5d.transpose(0, 1, 3, 2, 4).reshape(B, T, H, V)

    return dq, dk, dv


# ============================================================
# custom_vjp wrapper
# ============================================================


def chunk_fused_gla(q, k, v, g_gamma, scale, chunk_size):
    """Fused chunked GLA with custom_vjp (Pallas TPU kernels)."""
    @jax.custom_vjp
    def _compute(q, k, v):
        _, o = chunk_fwd_fused_g_gamma(q, k, v, g_gamma, scale, chunk_size)
        return o

    def _fwd(q, k, v):
        h, o = chunk_fwd_fused_g_gamma(q, k, v, g_gamma, scale, chunk_size)
        return o, (q, k, v, h)

    def _bwd(residuals, do):
        q, k, v, h = residuals
        dq, dk, dv = chunk_bwd_fused_g_gamma(
            q, k, v, h, do, g_gamma, scale, chunk_size
        )
        return dq, dk, dv

    _compute.defvjp(_fwd, _bwd)
    return _compute(q, k, v)


# ============================================================
# Entry point
# ============================================================


def optimized_compute(B=2, T=256, H=4, K=128, V=128, chunk_size=64):
    """Forward + backward fused chunk-GLA with Pallas TPU kernels.

    Returns the loss scalar. Both forward and backward are computed,
    so timing captures the full training step performance.
    """
    q, k_arr, v, g_gamma = _make_test_data(B, T, H, K, V, chunk_size)
    scale = K ** -0.5

    def loss_fn(q, k, v):
        return chunk_fused_gla(
            q.astype(jnp.float32), k.astype(jnp.float32),
            v.astype(jnp.float32), g_gamma, scale, chunk_size,
        ).sum()

    loss, _ = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k_arr, v)
    return loss
# EVOLVE-BLOCK-END
