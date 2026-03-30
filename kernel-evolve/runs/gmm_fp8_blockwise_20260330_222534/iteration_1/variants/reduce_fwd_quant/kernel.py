"""GMM block-wise FP8 quantization kernel for TPU -- reduce_fwd_quant variant.

Minimal quantization variant: only rhs (weights) are quantized to FP8.
All activations (lhs, grad) stay bf16 throughout forward and backward passes.

Optimization strategy:
  - Eliminate ALL quantization except rhs weight quantization in forward
  - Forward: bf16 lhs + fp8 rhs (mixed precision, proven by SO7)
  - Backward bwd_gmm: bf16 grad + fp8 rhs (reuse forward's quantized rhs)
  - Backward tgmm: bf16 lhs_t + bf16 grad (no quantization at all)
  - Tiling: (1024, 256, 128, 1024, 256, 128, 2048, 256, 128) per FP6
  - _clamp_tiling helper (F004: no min(t, tile_size) clamping for tiling > 128)
  - tile_size=128 for remaining fp8 quantization (F005)

Expected impact: Only 1 quantize call total (rhs). Should dramatically reduce
VLIW bundles and register pressure from 160K spills. Target: >2.0x speedup.

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
def _clamp_tiling(tiling, tile_size):
    """Clamp tiling values that are for fp8 ops to tile_size.

    F004: Do NOT clamp tiling values > 128 -- large tiles are valid for
    non-quantized (bf16) operations. Only clamp for fp8 quantized dimensions
    where tile_size alignment is required.
    """
    return tuple(tiling)


def gmm_fp8_blockwise(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    tiling: tuple[int, ...] = (1024, 256, 128, 1024, 256, 128, 2048, 256, 128),
) -> jnp.ndarray:
    """GMM with minimal fp8 quantization -- only rhs (weights) quantized.

    Uses mixed precision: bf16 lhs + fp8 rhs for forward gmm and
    bf16 grad + fp8 rhs for backward bwd_gmm. Backward tgmm uses
    pure bf16 (bf16 lhs_t + bf16 grad). This eliminates 4 out of 5
    quantize calls from the baseline.
    """
    tile_size = 128  # F005: tile_size=128 for fp8 quantization

    qt_rule = qwix.QtRule(
        weight_qtype=FP8_DTYPE,
        act_qtype=FP8_DTYPE,        # kept for qt_rule structure, not used
        bwd_qtype=FP8_DTYPE,        # kept for qt_rule structure, not used
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
    """Forward pass with minimal quantization.

    Only rhs (weights) are quantized to fp8. lhs stays bf16.
    Mixed precision gmm: bf16 lhs + fp8 rhs (SO7).
    lhs_t is NOT quantized -- stored as bf16 for backward tgmm.
    """
    tile_size = qt_rule.tile_size
    tiling = _clamp_tiling(tiling, tile_size)

    # Keep lhs as bf16 -- NO quantization of activations
    lhs_bf16 = lhs

    # Quantize ONLY rhs (weights) to fp8 -- the single quantize call
    rhs = qpl.quantize(
        rhs,
        qt_rule.weight_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size, 2: tile_size},
        calibration_method=qt_rule.weight_calibration_method,
        scale_dtype=jnp.float32,
    )

    # Forward gmm: bf16 lhs + fp8 rhs (mixed precision, proven by SO7)
    out = tokamax_backend.gmm(
        lhs=lhs_bf16,
        rhs=rhs,
        group_sizes=group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        out_dtype=jnp.float32,
        tiling=tiling[:3],
        group_offset=None,
        transpose_rhs=False,
        interpret=False,
    )

    # Residuals: (bf16 lhs, fp8 rhs, group_sizes, bf16 lhs_bf16)
    # lhs_bf16 stored for backward tgmm (will be swapaxed there)
    return out, (lhs_bf16, rhs, group_sizes, lhs_bf16)


def _gmm_bwd(lhs_dtype, rhs_dtype, qt_rule, tiling, residual, grad):
    """Backward pass with zero quantization.

    bwd_gmm: bf16 grad + fp8 rhs (reuse forward's quantized weights, SO7)
    tgmm: bf16 lhs_t + bf16 grad (pure bf16, no quantization needed)
    """
    lhs_bf16, rhs, group_sizes, lhs_bf16_orig = residual
    num_actual_groups = rhs.shape[0]
    tile_size = qt_rule.tile_size
    tiling = _clamp_tiling(tiling, tile_size)

    # bwd_gmm: bf16 grad + fp8 rhs (mixed precision, reuse quantized rhs)
    # NO quantization of grad -- pass bf16 directly
    dlhs = tokamax_backend.gmm(
        lhs=grad,
        rhs=rhs,
        group_sizes=group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        out_dtype=lhs_dtype,
        tiling=tiling[3:6] if len(tiling) >= 6 else tiling[:3],
        group_offset=None,
        transpose_rhs=True,
        interpret=False,
    )

    # tgmm: bf16 lhs_t + bf16 grad (pure bf16, no quantization)
    lhs_t = lhs_bf16_orig.swapaxes(0, 1)
    drhs = tokamax_backend.tgmm(
        lhs=lhs_t,
        rhs=grad,
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
    """Forward + backward GMM with minimal FP8 quantization.

    Returns the loss scalar. Both forward and backward are computed,
    so timing captures the full training step performance.
    """
    lhs, rhs, group_sizes = _make_test_data(M, K, N, G)

    def loss_fn(lhs, rhs):
        return gmm_fp8_blockwise(lhs, rhs, group_sizes).sum()

    loss, _ = jax.value_and_grad(loss_fn, argnums=(0, 1))(lhs, rhs)
    return loss
# EVOLVE-BLOCK-END
