"""FP8 backward transposed grouped matrix multiply (tGMM) Pallas kernel for TPU.

Computes per-group: output[g] = lhs[:, slice_g]^T @ rhs[slice_g, :]
  - lhs: [K, M] quantized FP8 E4M3 with 1x128 block scales
  - rhs: [M, N] quantized FP8 E4M3 with 1x128 block scales
  - output: [num_groups, K, N] bfloat16

Self-contained -- depends only on JAX + Pallas.  Uses the same RNG seeds
and quantization as gmm_fp8_bwd_ref.py so outputs match.

The kernel transposes lhs internally: loads [tm, tk] tiles and computes
lhs_tile.T @ rhs_tile = [tk, tn].
"""

from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# ---------------------------------------------------------------------------
# FP8 block-wise quantization helpers (matching gmm_fp8_bwd_ref.py)
# ---------------------------------------------------------------------------

QArray = namedtuple("QArray", ["qvalue", "scale"])

BLOCK_SIZE = 128
FP8_MAX = 448.0


def _quantize_fp8_blockwise(x, block_size=BLOCK_SIZE):
    """Quantize a float array to FP8 E4M3 with per-block scales.

    Reshapes the last dimension into blocks of ``block_size``, computes
    per-block absmax, and scales into the float8_e4m3fn range.

    Returns a QArray(qvalue, scale) where:
      - qvalue has the same shape as x, dtype float8_e4m3fn
      - scale has shape (*x.shape[:-1], x.shape[-1] // block_size)
    """
    orig_shape = x.shape
    num_blocks = orig_shape[-1] // block_size
    blocked = x.reshape(*orig_shape[:-1], num_blocks, block_size)
    absmax = jnp.max(jnp.abs(blocked), axis=-1)
    absmax = jnp.maximum(absmax, 1e-12)
    scale = absmax / FP8_MAX
    scale_expanded = scale[..., jnp.newaxis]
    scaled = blocked / scale_expanded
    scaled = jnp.clip(scaled, -FP8_MAX, FP8_MAX)
    qvalue = scaled.reshape(orig_shape).astype(jnp.float8_e4m3fn)
    return QArray(qvalue=qvalue, scale=scale)


def _dequantize(qarray):
    """Dequantize a QArray back to float32."""
    orig_shape = qarray.qvalue.shape
    num_blocks = qarray.scale.shape[-1]
    blocked = qarray.qvalue.reshape(
        *orig_shape[:-1], num_blocks, BLOCK_SIZE
    ).astype(jnp.float32)
    scale_expanded = qarray.scale[..., jnp.newaxis].astype(jnp.float32)
    return (blocked * scale_expanded).reshape(orig_shape)


# ---------------------------------------------------------------------------
# Group metadata
# ---------------------------------------------------------------------------

def make_group_metadata(
    group_sizes,
    m_tile_size,
    num_tiles_m,
    visit_empty_groups=False,
):
    """Build tile-to-group mapping metadata for the Pallas grid.

    Args:
        group_sizes: 1-D int array of length num_groups.
        m_tile_size: number of rows per tile along M.
        num_tiles_m: total number of M-tiles (M // m_tile_size).
        visit_empty_groups: if True, emit at least one tile per group.

    Returns:
        (m_tile_ids, group_ids, group_offsets, num_active_tiles)
    """
    group_sizes_np = np.asarray(group_sizes)
    num_groups = len(group_sizes_np)
    group_offsets = np.zeros(num_groups + 1, dtype=np.int32)
    group_offsets[1:] = np.cumsum(group_sizes_np)

    m_tile_ids = []
    group_ids_list = []

    for g in range(num_groups):
        start_row = group_offsets[g]
        end_row = group_offsets[g + 1]
        start_tile = start_row // m_tile_size
        end_tile = (end_row + m_tile_size - 1) // m_tile_size if end_row > start_row else start_tile

        if start_row == end_row and visit_empty_groups:
            # Empty group: emit one dummy tile so output gets zeroed.
            m_tile_ids.append(0)
            group_ids_list.append(g)
        else:
            for t in range(start_tile, end_tile):
                m_tile_ids.append(t)
                group_ids_list.append(g)

    num_active_tiles = len(m_tile_ids)
    m_tile_ids = jnp.array(m_tile_ids, dtype=jnp.int32)
    group_ids_arr = jnp.array(group_ids_list, dtype=jnp.int32)
    group_offsets_arr = jnp.array(group_offsets, dtype=jnp.int32)

    return m_tile_ids, group_ids_arr, group_offsets_arr, num_active_tiles


def _get_store_mask(grid_id, group_offsets, group_ids, m_tile_ids, tm, tile_cols):
    """Create a boolean mask for rows that belong to the current group.

    Returns a [tm, tile_cols] mask that is True for valid rows.
    """
    group = group_ids[grid_id]
    start = group_offsets[group]
    end = group_offsets[group + 1]
    tile_start = m_tile_ids[grid_id] * tm
    row_indices = tile_start + jnp.arange(tm)
    valid = (row_indices >= start) & (row_indices < end)
    return valid[:, None] * jnp.ones((1, tile_cols), dtype=jnp.bool_)


# ---------------------------------------------------------------------------
# Pallas tGMM kernel
# ---------------------------------------------------------------------------

def _tgmm_kernel(
    lhs_qv_ref,
    lhs_scale_ref,
    rhs_qv_ref,
    rhs_scale_ref,
    out_ref,
    acc_scratch,
    *,
    group_ids,
    group_offsets,
    m_tile_ids,
    num_active_tiles,
    tm,
    tk,
    tn,
):
    # EVOLVE-BLOCK-START
    grid_id = pl.program_id(2)  # m-tile iterator
    num_programs = num_active_tiles

    # Group transition tracking
    group = group_ids[grid_id]
    prev_group = group_ids[jnp.maximum(0, grid_id - 1)]
    last = num_programs - 1
    next_group = group_ids[jnp.minimum(grid_id + 1, last)]
    is_prologue = (grid_id == 0) | (group != prev_group)
    is_epilogue = (grid_id == last) | (group != next_group)
    group_size = group_offsets[group + 1] - group_offsets[group]

    @pl.when(is_prologue)
    def _zero():
        acc_scratch[...] = jnp.zeros_like(acc_scratch)

    @pl.when(group_size > 0)
    def _compute():
        lhs_tile = lhs_qv_ref[...].astype(jnp.float32)     # [tm, tk]
        rhs_tile = rhs_qv_ref[...].astype(jnp.float32)     # [tm, tn]

        # Group boundary mask
        tile_start = m_tile_ids[grid_id] * tm
        group_start = group_offsets[group]
        group_end = group_offsets[group + 1]
        row_indices = tile_start + jnp.arange(tm)
        valid = (row_indices >= group_start) & (row_indices < group_end)

        mask_lhs = jnp.broadcast_to(valid[:, None], (tm, tk))
        mask_rhs = jnp.broadcast_to(valid[:, None], (tm, tn))
        lhs_tile = jnp.where(mask_lhs, lhs_tile, 0.0)
        rhs_tile = jnp.where(mask_rhs, rhs_tile, 0.0)

        # Transposed dot: [tm, tk]^T @ [tm, tn] = [tk, tn]
        out = jnp.dot(lhs_tile.T, rhs_tile)  # [tk, tn] in f32

        # Apply block-wise scales
        # lhs_scale_ref: [tm, tk // 128] -> after transpose: [tk // 128, tm]
        # rhs_scale_ref: [tm, tn // 128]
        lhs_s = lhs_scale_ref[...].astype(jnp.float32)   # [tm, tk // 128]
        rhs_s = rhs_scale_ref[...].astype(jnp.float32)   # [tm, tn // 128]

        # Expand scales to match tile dimensions.
        # lhs_s [tm, tk//128] -> broadcast each scale block across 128 cols -> [tm, tk]
        lhs_s_exp = jnp.repeat(lhs_s, BLOCK_SIZE, axis=1)  # [tm, tk]
        rhs_s_exp = jnp.repeat(rhs_s, BLOCK_SIZE, axis=1)  # [tm, tn]

        # Mask scales too
        lhs_s_exp = jnp.where(mask_lhs, lhs_s_exp, 0.0)
        rhs_s_exp = jnp.where(mask_rhs, rhs_s_exp, 0.0)

        # Scale the result: we need to apply per-element scaling.
        # For exact scaling: sum_m (lhs[m,k] * lhs_scale[m,k_block]) * (rhs[m,n] * rhs_scale[m,n_block])
        # = sum_m (lhs_scaled[m,k] * rhs_scaled[m,n])
        # So we scale before the dot product.
        lhs_scaled = lhs_tile * lhs_s_exp   # [tm, tk]
        rhs_scaled = rhs_tile * rhs_s_exp   # [tm, tn]

        out = jnp.dot(lhs_scaled.T, rhs_scaled)  # [tk, tn]

        acc_scratch[...] += out

    @pl.when(is_epilogue)
    def _store():
        out_ref[...] = acc_scratch[...].astype(jnp.bfloat16)
    # EVOLVE-BLOCK-END


def optimized_compute(M=2048, K=512, N=1024, num_groups=4):
    """Generate FP8-quantized inputs and compute tGMM backward pass via Pallas.

    Args:
        M: inner (contracted) dimension, divisible by num_groups and BLOCK_SIZE.
        K: rows of lhs / first dim of output per group.
        N: cols of rhs / second dim of output.
        num_groups: number of groups along M.

    Returns:
        jnp.array of shape [num_groups, K, N] in bfloat16.
    """
    # --- Generate inputs with same RNG seeds as reference ---
    key = jax.random.PRNGKey(43)
    k1, k2 = jax.random.split(key)

    lhs_fp = jax.random.normal(k1, (K, M), dtype=jnp.float32)
    rhs_fp = jax.random.normal(k2, (M, N), dtype=jnp.float32)

    # Quantize with 1x128 block scales along last dimension
    lhs_q = _quantize_fp8_blockwise(lhs_fp)  # qvalue [K, M], scale [K, M//128]
    rhs_q = _quantize_fp8_blockwise(rhs_fp)  # qvalue [M, N], scale [M, N//128]

    # --- Prepare inputs for Pallas ---
    # The kernel needs lhs as [M, K] for tile-based loading.
    # Transpose qvalue and scale accordingly.
    # lhs_q.qvalue: [K, M] -> [M, K]
    # lhs_q.scale:  [K, M//128] -> need [M, K//128] after transpose
    # We re-quantize the transposed version for correct block-scale alignment.
    lhs_deq = _dequantize(lhs_q)  # [K, M] float32
    lhs_t = lhs_deq.T  # [M, K]
    lhs_t_q = _quantize_fp8_blockwise(lhs_t)  # qvalue [M, K], scale [M, K//128]

    rhs_deq = _dequantize(rhs_q)  # [M, N] float32
    rhs_t_q = _quantize_fp8_blockwise(rhs_deq)  # qvalue [M, N], scale [M, N//128]

    lhs_qv = lhs_t_q.qvalue   # [M, K] fp8
    lhs_scale = lhs_t_q.scale  # [M, K//128] f32
    rhs_qv = rhs_t_q.qvalue   # [M, N] fp8
    rhs_scale = rhs_t_q.scale  # [M, N//128] f32

    # --- Group metadata ---
    group_size_each = M // num_groups
    group_sizes = [group_size_each] * num_groups

    # Tile sizes
    tm = 128
    tk = min(K, 128)
    tn = min(N, 128)

    tiles_m = M // tm
    tiles_k = K // tk
    tiles_n = N // tn

    m_tile_ids, group_ids, group_offsets, num_active_tiles = make_group_metadata(
        group_sizes=group_sizes,
        m_tile_size=tm,
        num_tiles_m=tiles_m,
        visit_empty_groups=True,
    )

    # --- Pallas call ---
    grid = (tiles_n, tiles_k, num_active_tiles)

    def lhs_index_map(n_i, k_i, grid_id):
        return (m_tile_ids[grid_id], k_i)

    def lhs_scale_index_map(n_i, k_i, grid_id):
        return (m_tile_ids[grid_id], k_i)

    def rhs_index_map(n_i, k_i, grid_id):
        return (m_tile_ids[grid_id], n_i)

    def rhs_scale_index_map(n_i, k_i, grid_id):
        return (m_tile_ids[grid_id], n_i)

    def out_index_map(n_i, k_i, grid_id):
        return (group_ids[grid_id], k_i, n_i)

    lhs_scale_block_k = max(tk // BLOCK_SIZE, 1)
    rhs_scale_block_n = max(tn // BLOCK_SIZE, 1)

    kernel_fn = partial(
        _tgmm_kernel,
        group_ids=group_ids,
        group_offsets=group_offsets,
        m_tile_ids=m_tile_ids,
        num_active_tiles=num_active_tiles,
        tm=tm,
        tk=tk,
        tn=tn,
    )

    result = pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct((num_groups, K, N), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((tm, tk), lhs_index_map),
                pl.BlockSpec((tm, lhs_scale_block_k), lhs_scale_index_map),
                pl.BlockSpec((tm, tn), rhs_index_map),
                pl.BlockSpec((tm, rhs_scale_block_n), rhs_scale_index_map),
            ],
            out_specs=pl.BlockSpec((None, tk, tn), out_index_map),
            scratch_shapes=[pltpu.VMEM((tk, tn), jnp.float32)],
        ),
        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
        ),
    )(lhs_qv, lhs_scale, rhs_qv, rhs_scale)

    return result
