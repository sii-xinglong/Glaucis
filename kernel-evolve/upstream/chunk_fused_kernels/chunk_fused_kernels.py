"""Fused chunk-GLA Pallas TPU kernels for g_gamma (per-head constant gate) mode.

Merges three separate pallas_calls (h propagation + A recomputation + output
computation) into a single pallas_call.  This eliminates:

  1. Two kernel launch overheads
  2. The h tensor HBM round-trip (h is produced and consumed in VMEM)
  3. The A tensor HBM round-trip (A is recomputed inline from q, k, g)
  4. The g_cumsum tensor entirely (gating is computed from the g_gamma scalar)

The combined kernel uses grid (B, H, K/BK, V/BV, NT) with time as an
"arbitrary" dimension.  VMEM scratch holds the h state [BK, BV] in float32
across time steps.  At each time step t:

  1. Save current h to h_ref output (for backward residuals)
  2. Compute output: o = q_gated @ h * scale + A_masked @ v
  3. Update h: h = h * decay + k.T @ (v * gating)

For the target shape K=128=BK, V=128=BV the grid is (B, H, 1, 1, NT).
Each tile sees the full h, so no cross-tile reduction is needed.
"""

import functools

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tops.ops.utils import exp
from tops.utils import assert_shape, export_public


def _chunk_fwd_fused_kernel(
  q_ref,
  k_ref,
  v_ref,
  g_gamma,
  h_ref,
  o_ref,
  scratch_ref,
  *,
  BT,
  NT,
  scale,
):
  """Fused forward kernel: h propagation + A recomputation + output.

  At each time step t:
    1. Save h[t] to output (for backward residuals)
    2. Compute o[t] = q_gated @ h[t] * scale + A_masked @ v
       where A is recomputed inline from q, k, g_gamma
    3. Update h[t+1] = h[t] * decay + k.T @ (v * gating)
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

  # Per-position gating: g_gamma * (1, 2, ..., BT)
  b_g_ramp = g_gamma[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)
  # State decay for one full chunk
  b_g_last = g_gamma[i_h].astype(jnp.float32) * BT

  # --- Initialize h state in scratch at t=0 ---
  @pl.when(i_t == 0)
  def init():
    scratch_ref[:, :] = jnp.zeros((BK, BV), dtype=jnp.float32)

  # --- Step 1: Save current h to output (for backward residuals) ---
  h_ref[0, i_t, 0] = scratch_ref[...].astype(h_ref.dtype)

  # --- Step 2: Compute output o[t] ---
  b_q = q_ref[0, 0]  # [BT, BK]
  b_k = k_ref[0, 0]  # [BT, BK]
  b_v = v_ref[0, 0]  # [BT, BV]

  # Compute gated q and k for A recomputation (keep in f32 to avoid
  # Mosaic bf16 matmul compilation issues on TPU v7x)
  exp_g = exp(b_g_ramp)  # [BT]
  exp_neg_g = exp(-b_g_ramp)  # [BT]
  b_qg = (b_q.astype(jnp.float32) * exp_g[:, None])  # [BT, BK] f32
  b_kg = (b_k.astype(jnp.float32) * exp_neg_g[:, None])  # [BT, BK] f32

  # Recompute A = b_qg @ b_kg.T * scale  [BT, BT]
  b_A = (
    jnp.dot(
      b_qg,
      b_kg.T,
      precision=lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
    )
    * scale
  )

  # Causal mask
  m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
  b_A_masked = jnp.where(m_s, b_A, 0.0)

  # Inter-chunk: b_qg @ h * scale  [BT, BV]
  b_o_inter = (
    jnp.dot(
      b_qg,
      scratch_ref[...],
      precision=lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
    )
    * scale
  )

  # Intra-chunk: A_masked @ v  [BT, BV]
  b_o_intra = jnp.dot(
    b_A_masked,
    b_v.astype(jnp.float32),
    precision=lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )

  o_ref[0, 0] = (b_o_inter + b_o_intra).astype(o_ref.dtype)

  # --- Step 3: Update h for next time step ---
  # h = h * exp(g_gamma * BT) + k.T @ (v * exp(g_gamma*BT - g_ramp))
  scratch_ref[...] *= exp(b_g_last)

  # v_gated = v * exp(g_gamma*BT - g_ramp)  [BT, BV]
  v_gated = (b_v * exp(b_g_last - b_g_ramp)[:, None]).astype(b_v.dtype)

  # k.T @ v_gated: [BK, BT] @ [BT, BV] = [BK, BV]
  scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
    b_k.astype(jnp.float32).T,  # [BK, BT]
    v_gated.astype(jnp.float32),  # [BT, BV]
    precision=lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )


@functools.partial(jax.jit, static_argnames=["chunk_size", "scale"])
def chunk_fwd_fused_g_gamma(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  g_gamma: jax.Array,
  scale: float,
  chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
  """Fused chunked GLA forward pass for g_gamma (per-head constant gate) mode.

  Replaces the three-kernel forward pipeline (chunk_fwd_h + intra_gk +
  fwd_o_gk) with a single Pallas kernel.  The hidden state ``h`` stays
  in VMEM scratch instead of making an HBM round-trip, and the attention
  matrix ``A`` is recomputed inline rather than materialised.

  Args:
    q:          [B, T, H, K] queries in bfloat16.
    k:          [B, T, H, K] keys in bfloat16.
    v:          [B, T, H, V] values in bfloat16.
    g_gamma:    [H] per-head constant log-space gate in float32.
    scale:      Scaling factor, typically K**-0.5.
    chunk_size: Block size along the time dimension.

  Returns:
    h: [B, NT, H, K, V] hidden states at each chunk boundary (bfloat16),
       retained for the backward pass.
    o: [B, T, H, V] output (bfloat16).
  """
  BK, BV, BT = 128, 128, chunk_size
  B, T, H, K_dim = q.shape
  V = v.shape[-1]
  NT = T // BT

  assert_shape(q, (B, T, H, K_dim), "q")
  assert_shape(k, (B, T, H, K_dim), "k")
  assert_shape(v, (B, T, H, V), "v")
  assert_shape(g_gamma, (H,), "g_gamma")
  assert T % BT == 0, f"T ({T}) must be a multiple of chunk_size ({BT})"
  assert K_dim == BK, (
    f"Fused forward kernel requires K == {BK}; "
    f"multi-tile reduction not implemented (got K={K_dim})"
  )
  assert V == BV, (
    f"Fused forward kernel requires V == {BV}; "
    f"multi-tile reduction not implemented (got V={V})"
  )

  # Layout: (B, H, T, dim) -- time axis will be "arbitrary"
  q_t = jnp.transpose(q, (0, 2, 1, 3))  # [B, H, T, K]
  k_t = jnp.transpose(k, (0, 2, 1, 3))  # [B, H, T, K]
  v_t = jnp.transpose(v, (0, 2, 1, 3))  # [B, H, T, V]

  grid = (B, H, 1, 1, NT)

  # Index maps: all take 5 grid dims (b, h, ki, vi, t)
  def q_map(b, h, ki, vi, t):
    return b, h, t, ki

  def k_map(b, h, ki, vi, t):
    return b, h, t, ki

  def v_map(b, h, ki, vi, t):
    return b, h, t, vi

  def h_map(b, h, ki, vi, t):
    return b, 0, h, ki, vi

  def o_map(b, h, ki, vi, t):
    return b, h, t, vi

  h_all, o_t = pl.pallas_call(
    functools.partial(_chunk_fwd_fused_kernel, BT=BT, NT=NT, scale=scale),
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      grid=grid,
      in_specs=[
        pl.BlockSpec((1, 1, BT, BK), q_map),  # q
        pl.BlockSpec((1, 1, BT, BK), k_map),  # k
        pl.BlockSpec((1, 1, BT, BV), v_map),  # v
        pl.BlockSpec(memory_space=pltpu.SMEM),  # g_gamma
      ],
      out_specs=[
        pl.BlockSpec((1, NT, 1, BK, BV), h_map),  # h output
        pl.BlockSpec((1, 1, BT, BV), o_map),  # o output
      ],
      scratch_shapes=[pltpu.VMEM((BK, BV), jnp.float32)],
    ),
    out_shape=[
      jax.ShapeDtypeStruct((B, NT, H, K_dim, V), q.dtype),  # h
      jax.ShapeDtypeStruct((B, H, T, V), q.dtype),  # o
    ],
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=(
        "parallel",
        "parallel",
        "parallel",
        "parallel",
        "arbitrary",
      ),
      disable_bounds_checks=True,
    ),
  )(q_t, k_t, v_t, g_gamma)

  # o_t is [B, H, T, V], transpose to [B, T, H, V]
  o = jnp.transpose(o_t, (0, 2, 1, 3))  # [B, T, H, V]

  return h_all, o


# ============================================================
# Backward: Fused dh reverse propagation + dq/dk/dv computation
#
# Merges the separate backward kernels (dh propagation, dA, dv,
# dq/dk intra, dq/dk/dg inter) into a single pallas_call.
#
# The kernel processes chunks in REVERSE time order via reverse
# BlockSpec index_maps (t → NT-1-t) — no jnp.flip copies needed.
# VMEM scratch holds the dh state [BK, BV] in float32, accumulated
# in reverse.  g_cumsum is NOT loaded — gating is recomputed from
# g_gamma scalar as BT-length vectors.
# dg is NOT computed (dead output elimination for g_gamma mode).
# ============================================================


def _chunk_bwd_fused_kernel(
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
  """Fused backward kernel with g_cumsum eliminated.

  Processes chunks in REVERSE time order via reverse index_maps.
  Grid step i_t=0 reads the LAST chunk (NT-1), i_t=NT-1 reads chunk 0.
  At each step:
    1. Load current dh state from scratch_ref
    2. Compute dq/dk/dv using current dh
    3. Write outputs to dq_ref[0, NT-1-i_t, 0] (correct time order)
    4. Update dh: dh = dh * state_decay + q_hat.T @ do
  """
  BK = q_ref.shape[3]
  BV = do_ref.shape[3]
  i_b, i_h, i_k, i_v, i_t = (
    pl.program_id(0),
    pl.program_id(1),
    pl.program_id(2),
    pl.program_id(3),
    pl.program_id(4),
  )

  # Recompute gating from g_gamma scalar
  b_g_ramp = g_gamma[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)  # [BT]
  # State decay for one full chunk: scalar
  b_g_last = g_gamma[i_h].astype(jnp.float32) * BT  # scalar

  # Initialize dh state to zeros at t=0 (first step of REVERSE scan,
  # i.e. the END of the original sequence because inputs are flipped)
  @pl.when(i_t == 0)
  def init():
    scratch_ref[:, :] = jnp.zeros((BK, BV), dtype=jnp.float32)

  # Load inputs for this (reversed) time step
  b_q = q_ref[0, 0]  # [BT, BK]
  b_k = k_ref[0, 0]  # [BT, BK]
  b_v = v_ref[0, 0]  # [BT, BV]
  # h_ref is 5D [B, H, NT, K, V]; BlockSpec (1,1,1,BK,BV) maps t->NT dim
  b_h = h_ref[0, 0, 0].astype(jnp.float32)  # [BK, BV]
  b_do = do_ref[0, 0]  # [BT, BV]

  # Current dh from scratch (accumulated so far in reverse scan)
  b_dh = scratch_ref[...]  # [BK, BV]

  # Phase 0 (VPU): Pre-compute ALL exp/gate values upfront
  exp_pos = exp(b_g_ramp)  # [BT]
  exp_neg = exp(-b_g_ramp)  # [BT]
  exp_gn_minus = exp(b_g_last - b_g_ramp)  # [BT]

  # Broadcast [BT] -> [BT, K] at point of use (keep in f32 to avoid
  # Mosaic bf16 matmul compilation issues on TPU v7x)
  k_neg = (b_k.astype(jnp.float32) * exp_neg[:, None])  # [BT, K] f32
  k_decay = (b_k.astype(jnp.float32) * exp_gn_minus[:, None])  # [BT, K] f32
  q_pos = (b_q.astype(jnp.float32) * exp_pos[:, None])  # [BT, K] f32

  # Phase 1 (MXU): Recompute A and compute dA
  b_a = (
    jnp.dot(
      q_pos,
      k_neg.T,
      precision=lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
    )
    * scale
  )  # [BT, BT]

  b_dA_raw = (
    jnp.dot(
      b_do.astype(jnp.float32),
      b_v.astype(jnp.float32).T,
      precision=lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
    )
    * scale
  )  # [BT, BT]

  # Phase 2 (VPU): Apply causal masks
  mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
  b_dA = jnp.where(mask, b_dA_raw, 0.0)
  b_a_masked = jnp.where(mask, b_a, 0.0)

  # Phase 3 (MXU batch): Four independent dot products
  b_dv_intra = jnp.dot(
    b_a_masked.T,
    b_do.astype(jnp.float32),
    precision=lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )  # [BT, V]

  b_dv_inter = jnp.dot(
    k_decay,
    b_dh,
    precision=lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )  # [BT, V]

  b_dq_inter = jnp.dot(
    b_do.astype(jnp.float32),
    b_h.T,
    precision=lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )  # [BT, K]

  b_dk_inter = jnp.dot(
    b_v.astype(jnp.float32),
    b_dh.T,
    precision=lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )  # [BT, K]

  # Phase 4 (MXU): Intra-chunk dq and dk
  b_dq_intra_raw = jnp.dot(
    b_dA,
    k_neg,
    precision=lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )  # [BT, K]

  b_dk_intra_raw = jnp.dot(
    b_dA.T,
    q_pos,
    precision=lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )  # [BT, K]

  # Phase 5 (VPU): Combine results and write to 5D output slots.
  # Write at NT-1-i_t so outputs land in original (forward) time order
  # without needing a post-kernel jnp.flip.
  i_out = NT - 1 - i_t
  dv_ref[0, i_out, 0] = (b_dv_intra + b_dv_inter).astype(dv_ref.dtype)

  b_dq = b_dq_intra_raw * exp_pos[:, None] + b_dq_inter * (scale * exp_pos[:, None])
  dq_ref[0, i_out, 0] = b_dq.astype(dq_ref.dtype)

  b_dk = b_dk_intra_raw * exp_neg[:, None] + b_dk_inter * exp_gn_minus[:, None]
  dk_ref[0, i_out, 0] = b_dk.astype(dk_ref.dtype)

  # Phase 6: Update dh state in scratch for next reverse step
  # dh = dh * exp(g_gamma * BT) + q_hat.T @ do
  scratch_ref[...] *= exp(b_g_last)

  q_hat = (b_q * exp(b_g_ramp)[:, None] * scale).astype(jnp.float32)

  scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
    q_hat.T,  # [BK, BT]
    b_do.astype(jnp.float32),  # [BT, BV]
    precision=lax.Precision.HIGHEST,
    preferred_element_type=jnp.float32,
  )


@functools.partial(jax.jit, static_argnames=["chunk_size", "scale"])
def chunk_bwd_fused_g_gamma(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  h: jax.Array,
  do: jax.Array,
  g_gamma: jax.Array,
  scale: float,
  chunk_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Fused chunked GLA backward pass for g_gamma (per-head constant gate) mode.

  Replaces the multi-kernel backward pipeline (dh propagation + dA + dv +
  dq/dk intra + dq/dk/dg inter) with a single Pallas kernel.  The hidden
  state gradient ``dh`` stays in VMEM scratch instead of making an HBM
  round-trip, and gating is recomputed from the ``g_gamma`` scalar.

  The kernel processes chunks in reverse time order via reverse BlockSpec
  index_maps (``t → NT-1-t``), eliminating all ``jnp.flip`` copies.
  Outputs are written to their correct (forward) time positions inside
  the kernel, so no post-processing flip is needed either.

  Args:
    q:          [B, T, H, K] queries in bfloat16.
    k:          [B, T, H, K] keys in bfloat16.
    v:          [B, T, H, V] values in bfloat16.
    h:          [B, NT, H, K, V] hidden states from the forward pass.
    do:         [B, T, H, V] upstream output gradients in bfloat16.
    g_gamma:    [H] per-head constant log-space gate in float32.
    scale:      Scaling factor, typically K**-0.5.
    chunk_size: Block size along the time dimension.

  Returns:
    dq: [B, T, H, K] query gradients (bfloat16).
    dk: [B, T, H, K] key gradients (bfloat16).
    dv: [B, T, H, V] value gradients (bfloat16).
  """
  BK, BV, BT = 128, 128, chunk_size
  B, T, H, K = q.shape
  V = v.shape[-1]
  NT = T // BT

  assert_shape(q, (B, T, H, K), "q")
  assert_shape(k, (B, T, H, K), "k")
  assert_shape(v, (B, T, H, V), "v")
  assert_shape(h, (B, NT, H, K, V), "h")
  assert_shape(do, (B, T, H, V), "do")
  assert_shape(g_gamma, (H,), "g_gamma")
  assert T % BT == 0, f"T ({T}) must be a multiple of chunk_size ({BT})"
  assert K == BK, (
    f"Fused backward kernel requires K == {BK}; "
    f"multi-tile reduction not implemented (got K={K})"
  )
  assert V == BV, (
    f"Fused backward kernel requires V == {BV}; "
    f"multi-tile reduction not implemented (got V={V})"
  )

  # Transpose to (B, H, T, dim) layout — zero-copy views
  q_t = jnp.transpose(q, (0, 2, 1, 3))  # [B, H, T, K]
  k_t = jnp.transpose(k, (0, 2, 1, 3))  # [B, H, T, K]
  v_t = jnp.transpose(v, (0, 2, 1, 3))  # [B, H, T, V]
  do_t = jnp.transpose(do, (0, 2, 1, 3))  # [B, H, T, V]

  # h is [B, NT, H, K, V]; transpose to [B, H, NT, K, V] — zero-copy view
  h_bhntKV = jnp.transpose(h, (0, 2, 1, 3, 4))  # [B, H, NT, K, V]

  grid = (B, H, 1, 1, NT)

  # Reverse input index maps: t → NT-1-t reads chunks in reverse time order.
  # No jnp.flip needed — the index_map does the reversal at tile-fetch time.
  def q_map(b, h, ki, vi, t):
    return b, h, NT - 1 - t, ki

  def k_map(b, h, ki, vi, t):
    return b, h, NT - 1 - t, ki

  def v_map(b, h, ki, vi, t):
    return b, h, NT - 1 - t, vi

  def h_map(b, h, ki, vi, t):
    return b, h, NT - 1 - t, 0, 0

  def do_map(b, h, ki, vi, t):
    return b, h, NT - 1 - t, vi

  # Output index maps — 5D outputs [B, NT, H, BT, K/V].
  # The kernel writes at NT-1-i_t internally, so no output flip needed.
  def out_k_map(b, h, ki, vi, t):
    return b, 0, h, 0, 0

  def out_v_map(b, h, ki, vi, t):
    return b, 0, h, 0, 0

  dq_5d, dk_5d, dv_5d = pl.pallas_call(
    functools.partial(
      _chunk_bwd_fused_kernel,
      BT=BT,
      NT=NT,
      scale=scale,
    ),
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      grid=grid,
      in_specs=[
        pl.BlockSpec((1, 1, BT, BK), q_map),  # q: 4D
        pl.BlockSpec((1, 1, BT, BK), k_map),  # k: 4D
        pl.BlockSpec((1, 1, BT, BV), v_map),  # v: 4D
        pl.BlockSpec((1, 1, 1, BK, BV), h_map),  # h: 5D
        pl.BlockSpec((1, 1, BT, BV), do_map),  # do: 4D
        pl.BlockSpec(memory_space=pltpu.SMEM),  # g_gamma
      ],
      out_specs=[
        pl.BlockSpec((1, NT, 1, BT, BK), out_k_map),  # dq: 5D
        pl.BlockSpec((1, NT, 1, BT, BK), out_k_map),  # dk: 5D
        pl.BlockSpec((1, NT, 1, BT, BV), out_v_map),  # dv: 5D
      ],
      scratch_shapes=[pltpu.VMEM((BK, BV), jnp.float32)],
    ),
    out_shape=[
      jax.ShapeDtypeStruct((B, NT, H, BT, K), q.dtype),  # dq
      jax.ShapeDtypeStruct((B, NT, H, BT, K), k.dtype),  # dk
      jax.ShapeDtypeStruct((B, NT, H, BT, V), v.dtype),  # dv
    ],
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=(
        "parallel",
        "parallel",
        "parallel",
        "parallel",
        "arbitrary",
      ),
      disable_bounds_checks=True,
    ),
  )(q_t, k_t, v_t, h_bhntKV, do_t, g_gamma)

  # dq/dk/dv are [B, NT, H, BT, K/V] already in correct time order
  # (kernel wrote at NT-1-i_t). Reshape to [B, T, H, K/V].
  dq = dq_5d.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K)
  dk = dk_5d.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K)
  dv = dv_5d.transpose(0, 1, 3, 2, 4).reshape(B, T, H, V)

  return dq, dk, dv


__all__ = export_public(globals())
