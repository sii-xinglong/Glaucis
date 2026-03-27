"""FP8 forward grouped matrix multiplication Pallas kernel for TPU.

Self-contained -- depends only on JAX + Pallas.  Exports the entry point
that evaluate.py looks for:

  * ``optimized_compute(M, K, N, num_groups)``

Uses the exact same RNG seeds and FP8 quantisation as gmm_fp8_fwd_ref.py
so outputs are bit-comparable (within FP8 precision).

The code between EVOLVE-BLOCK-START / EVOLVE-BLOCK-END is the mutable
region that the evolution engine will rewrite.
"""

from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# ---------------------------------------------------------------------------
# FP8 block-wise quantisation helpers  (identical to gmm_fp8_fwd_ref.py)
# ---------------------------------------------------------------------------

QArray = namedtuple("QArray", ["qvalue", "scale"])

_FP8_E4M3_MAX = 448.0


def _quantize_1x128(x):
    """Quantise a 2-D tensor with 1x128 block scaling to float8_e4m3fn."""
    rows, K = x.shape
    block_k = 128
    assert K % block_k == 0

    x_f32 = x.astype(jnp.float32)
    blocks = x_f32.reshape(rows, K // block_k, block_k)
    absmax = jnp.max(jnp.abs(blocks), axis=-1)
    scale = absmax / _FP8_E4M3_MAX
    scale = jnp.maximum(scale, jnp.finfo(jnp.float32).tiny)

    inv_scale = (1.0 / scale)[:, :, None]
    scaled = blocks * inv_scale
    scaled = jnp.clip(scaled, -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    qvalue = scaled.reshape(rows, K).astype(jnp.float8_e4m3fn)

    return QArray(qvalue=qvalue, scale=scale)


def _quantize_128x128(x):
    """Quantise a 3-D tensor with 128x128 block scaling to float8_e4m3fn."""
    G, K, N = x.shape
    bk, bn = 128, 128
    assert K % bk == 0
    assert N % bn == 0

    x_f32 = x.astype(jnp.float32)
    blocks = x_f32.reshape(G, K // bk, bk, N // bn, bn)
    absmax = jnp.max(jnp.abs(blocks), axis=(2, 4))
    scale = absmax / _FP8_E4M3_MAX
    scale = jnp.maximum(scale, jnp.finfo(jnp.float32).tiny)

    inv_scale = (1.0 / scale)[:, :, None, :, None]
    scaled = blocks * inv_scale
    scaled = jnp.clip(scaled, -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    qvalue = scaled.reshape(G, K, N).astype(jnp.float8_e4m3fn)

    return QArray(qvalue=qvalue, scale=scale)


# ---------------------------------------------------------------------------
# Group metadata  (equivalent to jax_triton make_group_metadata)
# ---------------------------------------------------------------------------

def _make_group_metadata(
    group_sizes: np.ndarray,
    tm: int,
    num_groups: int,
):
    """Compute group metadata arrays for the Pallas GMM grid.

    Args:
        group_sizes: 1-D int array of length ``num_groups``, number of rows
            per group.
        tm: tile size along M.
        num_groups: number of groups.

    Returns:
        group_offsets: int32 [num_groups + 1] — cumulative row offsets.
        group_ids: int32 [num_active_tiles] — which group each M-tile
            belongs to.
        m_tile_ids: int32 [num_active_tiles] — the local M-tile index
            within the group for each active tile.
    """
    group_offsets = np.zeros(num_groups + 1, dtype=np.int32)
    group_offsets[1:] = np.cumsum(group_sizes)

    group_ids = []
    m_tile_ids = []
    for g in range(num_groups):
        tiles_for_g = int(np.ceil(group_sizes[g] / tm))
        group_ids.extend([g] * tiles_for_g)
        m_tile_ids.extend(range(tiles_for_g))

    return (
        jnp.array(group_offsets, dtype=jnp.int32),
        jnp.array(group_ids, dtype=jnp.int32),
        jnp.array(m_tile_ids, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# Pallas kernel + pallas_call wrapper
# ---------------------------------------------------------------------------

def optimized_compute(M=2048, K=1024, N=1024, num_groups=4):
    """Generate FP8-quantised inputs and compute forward GMM via Pallas.

    Inputs are generated with the same RNG seeds and quantisation as
    ``gmm_fp8_fwd_ref.simple_compute`` so that outputs can be compared.

    Returns:
        jnp.ndarray of shape [M, N] in bfloat16.
    """
    assert M % num_groups == 0

    # -- 1. Input generation (same as reference) ---------------------------
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    lhs_fp = jax.random.normal(k1, (M, K), dtype=jnp.float32)
    rhs_fp = jax.random.normal(k2, (num_groups, K, N), dtype=jnp.float32)

    lhs_q = _quantize_1x128(lhs_fp)
    rhs_q = _quantize_128x128(rhs_fp)

    # Separate qvalue and scale arrays for Pallas.
    lhs_qv = lhs_q.qvalue          # [M, K]          fp8
    lhs_sc = lhs_q.scale            # [M, K//128]     f32
    rhs_qv = rhs_q.qvalue          # [G, K, N]       fp8
    rhs_sc = rhs_q.scale            # [G, K//128, N//128] f32

    # -- 2. Tiling parameters ----------------------------------------------
    tm, tk, tn = 128, 128, 128
    tiles_k = K // tk
    tiles_n = N // tn

    group_size = M // num_groups
    group_sizes = np.full(num_groups, group_size, dtype=np.int32)

    group_offsets, group_ids, m_tile_ids = _make_group_metadata(
        group_sizes, tm, num_groups,
    )
    num_active_tiles = len(group_ids)

    # -- 3. Prefetch / scalar arrays ---------------------------------------
    # Pack group_ids and m_tile_ids together for scalar prefetch.
    # group_metadata: [2, num_active_tiles]  row-0 = group_ids, row-1 = m_tile_ids
    group_metadata = jnp.stack([group_ids, m_tile_ids], axis=0)  # [2, T]

    # -- 4. Index maps -----------------------------------------------------
    # Grid: (tiles_n, num_active_tiles, tiles_k)
    #   axis 0 -> n-tile
    #   axis 1 -> active m-tile  (mapped via group_metadata)
    #   axis 2 -> k-tile

    def lhs_qv_index_map(n_i, tile_i, k_i, group_metadata_ref, group_offsets_ref):
        del n_i
        g_id = group_metadata_ref[0, tile_i]
        m_tid = group_metadata_ref[1, tile_i]
        m_start = group_offsets_ref[g_id] + m_tid * tm
        return (m_start, k_i * tk)

    def lhs_sc_index_map(n_i, tile_i, k_i, group_metadata_ref, group_offsets_ref):
        del n_i
        g_id = group_metadata_ref[0, tile_i]
        m_tid = group_metadata_ref[1, tile_i]
        m_start = group_offsets_ref[g_id] + m_tid * tm
        # Scale shape [M, K//128]; tile (tm, 1) since tk/128 = 1
        return (m_start, k_i)

    def rhs_qv_index_map(n_i, tile_i, k_i, group_metadata_ref, group_offsets_ref):
        del group_offsets_ref
        g_id = group_metadata_ref[0, tile_i]
        return (g_id, k_i * tk, n_i * tn)

    def rhs_sc_index_map(n_i, tile_i, k_i, group_metadata_ref, group_offsets_ref):
        del group_offsets_ref
        g_id = group_metadata_ref[0, tile_i]
        # Scale shape [G, K//128, N//128]; tile (1, 1, 1)
        return (g_id, k_i, n_i)

    def out_index_map(n_i, tile_i, k_i, group_metadata_ref, group_offsets_ref):
        del k_i
        g_id = group_metadata_ref[0, tile_i]
        m_tid = group_metadata_ref[1, tile_i]
        m_start = group_offsets_ref[g_id] + m_tid * tm
        return (m_start, n_i * tn)

    # -- 5. BlockSpecs -----------------------------------------------------
    lhs_qv_spec = pl.BlockSpec((tm, tk), lhs_qv_index_map)
    lhs_sc_spec = pl.BlockSpec((tm, 1), lhs_sc_index_map)
    rhs_qv_spec = pl.BlockSpec((1, tk, tn), rhs_qv_index_map)
    rhs_sc_spec = pl.BlockSpec((1, 1, 1), rhs_sc_index_map)
    out_spec = pl.BlockSpec((tm, tn), out_index_map)

    # -- 6. Kernel function ------------------------------------------------
    def _kernel(
        group_metadata_ref,
        group_offsets_ref,
        lhs_qv_ref,
        lhs_sc_ref,
        rhs_qv_ref,
        rhs_sc_ref,
        out_ref,
        acc_scratch,
    ):
        # EVOLVE-BLOCK-START
        grid_id = pl.program_id(1)
        k_i = pl.program_id(2)

        @pl.when(k_i == 0)
        def _zero_acc():
            acc_scratch[...] = jnp.zeros_like(acc_scratch)

        # Load FP8 quantised-value tiles.
        lhs_tile = lhs_qv_ref[...].astype(jnp.bfloat16)   # [tm, tk]
        rhs_tile = rhs_qv_ref[0, :, :].astype(jnp.bfloat16)  # [tk, tn]

        # FP8 matmul: [tm, tk] @ [tk, tn] -> [tm, tn] in f32.
        result = jax.lax.dot_general(
            lhs_tile,
            rhs_tile,
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

        # Apply blockwise scales.
        # lhs_sc_ref: [tm, 1] f32 scale per row per K-block.
        # rhs_sc_ref: [1, 1, 1] f32 scale for this (group, k-block, n-block).
        lhs_scale = lhs_sc_ref[...].astype(jnp.float32)     # [tm, 1]
        rhs_scale = rhs_sc_ref[0, 0, 0].astype(jnp.float32) # scalar

        result = result * lhs_scale * rhs_scale

        acc_scratch[...] += result

        @pl.when(k_i == tiles_k - 1)
        def _store():
            # Mask: rows beyond the group boundary should not be overwritten.
            g_id = group_metadata_ref[0, grid_id]
            m_tid = group_metadata_ref[1, grid_id]
            g_start = group_offsets_ref[g_id]
            g_end = group_offsets_ref[g_id + 1]
            tile_row_start = g_start + m_tid * tm

            # Number of valid rows in this tile.
            valid_rows = g_end - tile_row_start
            row_ids = jnp.arange(tm, dtype=jnp.int32)
            mask = row_ids[:, None] < valid_rows

            acc = acc_scratch[...]
            existing = out_ref[...].astype(jnp.float32)
            acc = jnp.where(mask, acc, existing)
            out_ref[...] = acc.astype(jnp.bfloat16)
        # EVOLVE-BLOCK-END

    # -- 7. pallas_call ----------------------------------------------------
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(tiles_n, num_active_tiles, tiles_k),
        in_specs=[
            lhs_qv_spec,
            lhs_sc_spec,
            rhs_qv_spec,
            rhs_sc_spec,
        ],
        out_specs=out_spec,
        scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],
    )

    kernel = pl.pallas_call(
        _kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
        ),
    )

    # Initialise output to zeros so masked regions are well-defined.
    out_init = jnp.zeros((M, N), dtype=jnp.bfloat16)

    result = kernel(group_metadata, group_offsets, lhs_qv, lhs_sc, rhs_qv, rhs_sc)
    return result
