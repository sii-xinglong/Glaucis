"""GMM block-wise FP8 quantization kernel for TPU -- fwd_mixed_precision variant.

Forward mixed precision optimization: eliminates 2 of 3 forward quantization
calls by leveraging SO7 (tokamax gmm accepts mixed bf16 lhs + fp8 rhs).

Key changes from baseline:
  - Forward: only quantize rhs to fp8; pass bf16 lhs directly (mixed precision)
  - Forward: skip lhs_t quantization entirely (only used in bf16 backward tgmm)
  - Backward: zero quantization (SO8) -- bf16 grad + fp8 rhs for bwd_gmm,
    bf16 lhs_t + bf16 grad for tgmm
  - Tiling: (1024, 256, 128) fwd, (1024, 256, 128) bwd_gmm, (2048, 256, 128) tgmm
  - _clamp_tiling helper: clamp to tile_size only for fp8 dimensions

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
def _clamp_tiling(tiling_triple, tile_size):
    """Clamp tiling dimensions to tile_size for fp8 quantized phases.

    For phases using fp8 operands, the K dimension (index 2) must be clamped
    to tile_size to match the fp8 quantization block size. M and N dimensions
    are unconstrained (F004: TM <= 256, TK <= 512, TN <= 512).
    """
    tm, tn, tk = tiling_triple
    return (tm, tn, min(tk, tile_size))


def gmm_fp8_blockwise(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    tiling: tuple[int, ...] = (1024, 256, 128, 1024, 256, 128, 2048, 256, 128),
) -> jnp.ndarray:
    """GMM with forward mixed precision: bf16 lhs + fp8 rhs, zero bwd quantization."""
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

    # Clamp forward tiling for fp8 rhs dimension
    fwd_tiling = _clamp_tiling(tiling[:3], tile_size)

    # Keep bf16 lhs for both forward gmm (mixed precision) and backward tgmm
    lhs_bf16 = lhs

    # Only quantize rhs to fp8 -- lhs stays bf16 (SO7: mixed bf16+fp8 proven)
    rhs = qpl.quantize(
        rhs,
        qt_rule.weight_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size, 2: tile_size},
        calibration_method=qt_rule.weight_calibration_method,
        scale_dtype=jnp.float32,
    )

    # Forward gmm: bf16 lhs + fp8 rhs (mixed precision via SO7)
    out = tokamax_backend.gmm(
        lhs=lhs_bf16,
        rhs=rhs,
        group_sizes=group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        out_dtype=jnp.float32,
        tiling=fwd_tiling,
        group_offset=None,
        transpose_rhs=False,
        interpret=False,
    )
    # Residuals: fp8 rhs for bwd_gmm, bf16 lhs for tgmm (no lhs_t quantized)
    return out, (rhs, group_sizes, lhs_bf16)


def _gmm_bwd(lhs_dtype, rhs_dtype, qt_rule, tiling, residual, grad):
    rhs, group_sizes, lhs_bf16 = residual
    num_actual_groups = rhs.shape[0]
    tile_size = qt_rule.tile_size

    # Clamp bwd_gmm tiling for fp8 rhs dimension
    bwd_tiling = _clamp_tiling(tiling[3:6] if len(tiling) >= 6 else tiling[:3], tile_size)

    # tgmm tiling unconstrained -- both operands are bf16
    tgmm_tiling = tiling[6:9] if len(tiling) >= 9 else tiling[:3]

    # Zero backward quantization (SO8): pass bf16 grad directly
    # bwd_gmm: bf16 grad + fp8 rhs (SO7 mixed precision)
    dlhs = tokamax_backend.gmm(
        lhs=grad,
        rhs=rhs,
        group_sizes=group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        out_dtype=lhs_dtype,
        tiling=bwd_tiling,
        group_offset=None,
        transpose_rhs=True,
        interpret=False,
    )

    # tgmm: bf16 lhs_t + bf16 grad (fully bf16, no quantization needed)
    lhs_t = lhs_bf16.swapaxes(0, 1)
    drhs = tokamax_backend.tgmm(
        lhs=lhs_t,
        rhs=grad,
        group_sizes=group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        out_dtype=rhs_dtype,
        tiling=tgmm_tiling,
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

    def loss_fn(lhs, rhs):
        return gmm_fp8_blockwise(lhs, rhs, group_sizes).sum()

    loss, _ = jax.value_and_grad(loss_fn, argnums=(0, 1))(lhs, rhs)
    return loss
# EVOLVE-BLOCK-END
