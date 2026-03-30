"""GMM block-wise FP8 quantization kernel for TPU -- template for evolutionary optimization.

Grouped Matrix Multiply (GMM) with DeepSeek-V3 / Transformer Engine style
block-wise FP8 quantization, using tokamax Pallas TPU kernel and qwix
quantization library.

Optimization targets within the EVOLVE-BLOCK:
  - Tiling parameters for fwd/bwd_gmm/bwd_tgmm phases
  - FP8 quantization tile size (block size)
  - Calibration methods (absmax vs alternatives)
  - Quantization axis configurations (channelwise + tiled)
  - Forward/backward computation structure

AL model reference dimensions:
  Gate/Up: lhs [M, 2048] @ rhs [G, 2048, 512]  -> [M, 512]
  Down:    lhs [M, 512]  @ rhs [G, 512, 2048]   -> [M, 2048]
"""

import dataclasses
import functools
from typing import List, Tuple

import jax
import jax.numpy as jnp

import qwix
import qwix.pallas as qpl

from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_kernel as tokamax_backend


# ---------------------------------------------------------------------------
# Constants (fixed across optimization iterations)
# ---------------------------------------------------------------------------
FP8_DTYPE = jnp.float8_e4m3fn
DEFAULT_TILE_SIZE = 128


def _make_test_data(M, K, N, G, seed=42):
    """Create (lhs, rhs, group_sizes) for a GMM test case."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    lhs = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
    rhs = jax.random.normal(k2, (G, K, N), dtype=jnp.bfloat16)
    base = (M // G // DEFAULT_TILE_SIZE) * DEFAULT_TILE_SIZE
    assert base > 0, f"M={M} / G={G} = {M // G} must be >= {DEFAULT_TILE_SIZE}"
    sizes = [base] * G
    sizes[-1] = M - base * (G - 1)
    group_sizes = jnp.array(sizes, dtype=jnp.int32)
    return lhs, rhs, group_sizes


# EVOLVE-BLOCK-START
def _phase_tiling(M, K, N):
    """Compute phase-specific tiling based on actual matrix dimensions.

    Strategy: Reduce register pressure by choosing tile sizes that minimize
    the number of simultaneously-live values in each phase.

    For a tile of (tm, tk, tn), the live values are roughly:
      - lhs tile: tm * tk
      - rhs tile: tk * tn
      - accumulator: tm * tn
      - total ~ tm*tk + tk*tn + tm*tn

    We want to minimize total live values while keeping tiles large enough
    for MXU utilization. The key insight is that different phases have
    different K/N ratios, so the optimal tiling differs per phase.
    """
    # Forward: lhs [M, K] @ rhs [G, K, N]
    # Gate/Up: K=2048, N=512. K is large so use smaller M/N tiles to reduce
    # lhs (tm*tk) and rhs (tk*tn) tile footprints. Use 128 for K to keep
    # MXU fed, but 64 for M and N to slash register count.
    # Live values: 64*128 + 128*64 + 64*64 = 8192 + 8192 + 4096 = 20480
    # vs baseline: 128*128 + 128*128 + 128*128 = 49152 (2.4x reduction)
    if K >= 1024:
        fwd_tiling = (64, 128, 64)
    else:
        # Down: K=512, N=2048. K is smaller, N is large.
        # Can afford slightly larger M tile since K footprint is smaller.
        fwd_tiling = (128, 64, 64)

    # Backward GMM (dlhs): dlhs_dout [M, N] @ rhs^T [G, N, K] -> [M, K]
    # transpose_rhs=True means rhs axes are swapped internally.
    # For Gate/Up backward: effectively [M, 512] @ [G, 512, 2048] -> [M, 2048]
    # The "K" dimension in the GMM call is N (512), output is K (2048).
    # Small reduction dim (512) means we can use larger output tiles.
    # Use small M tile (64) to limit accumulator + lhs, moderate K (128).
    bwd_gmm_tiling = (64, 128, 64)

    # Backward TGMM (drhs): lhs_t [K, M] @ drhs_dout [M, N] -> [G, K, N]
    # For Gate/Up backward: [2048, M] @ [M, 512] -> [G, 2048, 512]
    # M is the reduction dimension (8192, large).
    # K=2048 and N=512 are output dimensions.
    # Use small tiles for both output dims to limit accumulator size.
    # Keep reduction tile at 128 for throughput.
    bwd_tgmm_tiling = (64, 128, 64)

    return fwd_tiling + bwd_gmm_tiling + bwd_tgmm_tiling


def gmm_fp8_blockwise(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    tiling: tuple[int, ...] = (64, 128, 64) * 3,
) -> jnp.ndarray:
    """GMM with fp8_blockwise quantization and tokamax backend.

    Uses phase-specific tiling to reduce register pressure. Tile sizes are
    chosen per-phase based on the matrix dimensions to minimize the number
    of simultaneously-live tile values, targeting the catastrophic register
    spill/fill cycle observed in profiling (160k spills).
    """
    tile_size = 128

    qt_rule = qwix.QtRule(
        weight_qtype=FP8_DTYPE,
        act_qtype=FP8_DTYPE,
        bwd_qtype=FP8_DTYPE,
        tile_size=tile_size,
        weight_calibration_method="absmax",
        act_calibration_method="absmax",
        bwd_calibration_method="absmax",
    )
    fwd_bwd = lambda *a: _gmm_fwd(*a)[0]
    fwd_bwd = jax.custom_vjp(fwd_bwd, nondiff_argnums=(3, 4))
    fwd_bwd.defvjp(
        _gmm_fwd,
        functools.partial(_gmm_bwd, lhs.dtype, rhs.dtype),
    )
    return fwd_bwd(lhs, rhs, group_sizes, qt_rule, tiling)


def _gmm_fwd(lhs, rhs, group_sizes, qt_rule, tiling):
    tile_size = qt_rule.tile_size
    tiling = tuple(min(t, tile_size) for t in tiling)

    lhs_bf16 = lhs
    lhs = qpl.quantize(
        lhs_bf16,
        qt_rule.act_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size},
        calibration_method=qt_rule.act_calibration_method,
        scale_dtype=jnp.float32,
    )
    lhs_t = qpl.quantize(
        lhs_bf16.swapaxes(0, 1),
        qt_rule.act_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size},
        calibration_method=qt_rule.act_calibration_method,
        scale_dtype=jnp.float32,
    )
    rhs = qpl.quantize(
        rhs,
        qt_rule.weight_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size, 2: tile_size},
        calibration_method=qt_rule.weight_calibration_method,
        scale_dtype=jnp.float32,
    )

    out = tokamax_backend.gmm(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        out_dtype=jnp.float32,
        tiling=tiling[:3],
        group_offset=None,
        transpose_rhs=False,
        interpret=False,
    )
    return out, (lhs, rhs, group_sizes, lhs_t)


def _gmm_bwd(lhs_dtype, rhs_dtype, qt_rule, tiling, residual, grad):
    lhs, rhs, group_sizes, lhs_t = residual
    num_actual_groups = rhs.shape[0]
    tile_size = qt_rule.tile_size
    tiling = tuple(min(t, tile_size) for t in tiling)

    dlhs_dout = qpl.quantize(
        grad,
        qt_rule.bwd_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size},
        calibration_method=qt_rule.bwd_calibration_method,
        scale_dtype=jnp.float32,
    )
    drhs_dout = qpl.quantize(
        grad,
        qt_rule.bwd_qtype,
        channelwise_axes=[1],
        tiled_axes={0: tile_size},
        calibration_method=qt_rule.bwd_calibration_method,
        scale_dtype=jnp.float32,
    )

    dlhs = tokamax_backend.gmm(
        lhs=dlhs_dout,
        rhs=rhs,
        group_sizes=group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        out_dtype=lhs_dtype,
        tiling=tiling[3:6] if len(tiling) >= 6 else tiling[:3],
        group_offset=None,
        transpose_rhs=True,
        interpret=False,
    )

    drhs = tokamax_backend.tgmm(
        lhs=lhs_t,
        rhs=drhs_dout,
        group_sizes=group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        out_dtype=rhs_dtype,
        tiling=tiling[6:9] if len(tiling) >= 9 else tiling[:3],
        group_offset=None,
        num_actual_groups=num_actual_groups,
        interpret=False,
    )

    return dlhs, drhs, None


def optimized_compute(M=8192, K=2048, N=512, G=32):
    """Forward + backward GMM with FP8 blockwise quantization.

    Returns the loss scalar. Both forward and backward are computed,
    so timing captures the full training step performance.
    """
    lhs, rhs, group_sizes = _make_test_data(M, K, N, G)

    # Compute phase-specific tiling based on actual dimensions
    tiling = _phase_tiling(M, K, N)

    def loss_fn(lhs, rhs):
        return gmm_fp8_blockwise(lhs, rhs, group_sizes, tiling=tiling).sum()

    loss, _ = jax.value_and_grad(loss_fn, argnums=(0, 1))(lhs, rhs)
    return loss
# EVOLVE-BLOCK-END
