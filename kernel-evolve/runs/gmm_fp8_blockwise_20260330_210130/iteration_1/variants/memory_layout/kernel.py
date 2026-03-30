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
def gmm_fp8_blockwise(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    tiling: tuple[int, ...] = (128, 128, 128) * 3,
) -> jnp.ndarray:
    """GMM with fp8_blockwise quantization and tokamax backend.

    Memory-optimized variant: saves only lhs_bf16 + quantized rhs as residuals,
    recomputes quantized lhs and lhs_t in backward to reduce register pressure.
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


def _quantize_lhs(lhs_bf16, qt_rule):
    """Quantize lhs activation (factored out for reuse in fwd and bwd)."""
    return qpl.quantize(
        lhs_bf16,
        qt_rule.act_qtype,
        channelwise_axes=[0],
        tiled_axes={1: qt_rule.tile_size},
        calibration_method=qt_rule.act_calibration_method,
        scale_dtype=jnp.float32,
    )


def _quantize_lhs_t(lhs_bf16, qt_rule):
    """Quantize transposed lhs activation (factored out for reuse in bwd)."""
    return qpl.quantize(
        lhs_bf16.swapaxes(0, 1),
        qt_rule.act_qtype,
        channelwise_axes=[0],
        tiled_axes={1: qt_rule.tile_size},
        calibration_method=qt_rule.act_calibration_method,
        scale_dtype=jnp.float32,
    )


def _gmm_fwd(lhs, rhs, group_sizes, qt_rule, tiling):
    tile_size = qt_rule.tile_size
    tiling = tuple(min(t, tile_size) for t in tiling)

    # Keep a reference to the bf16 input before quantization
    lhs_bf16 = lhs

    # Quantize lhs for forward GMM
    lhs_q = _quantize_lhs(lhs_bf16, qt_rule)

    # Quantize rhs (weights) -- must be saved since rhs comes from parameters
    rhs_q = qpl.quantize(
        rhs,
        qt_rule.weight_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size, 2: tile_size},
        calibration_method=qt_rule.weight_calibration_method,
        scale_dtype=jnp.float32,
    )

    out = tokamax_backend.gmm(
        lhs=lhs_q,
        rhs=rhs_q,
        group_sizes=group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        out_dtype=jnp.float32,
        tiling=tiling[:3],
        group_offset=None,
        transpose_rhs=False,
        interpret=False,
    )

    # CRITICAL CHANGE: Save only lhs_bf16 (compact bf16) and rhs_q (needed for dlhs).
    # Do NOT save lhs_q or lhs_t -- recompute them in backward.
    # This reduces residuals from 4 quantized tensors to 1 bf16 + 1 quantized tensor,
    # dramatically cutting the memory footprint carried between fwd and bwd.
    return out, (lhs_bf16, rhs_q, group_sizes)


def _gmm_bwd(lhs_dtype, rhs_dtype, qt_rule, tiling, residual, grad):
    lhs_bf16, rhs_q, group_sizes = residual
    num_actual_groups = rhs_q.shape[0]
    tile_size = qt_rule.tile_size
    tiling = tuple(min(t, tile_size) for t in tiling)

    # Recompute lhs_t from saved lhs_bf16 instead of loading from residuals.
    # This trades cheap quantization compute for massive memory savings.
    # With only 6.68% compute utilization, we have ~93% compute headroom.
    lhs_t = _quantize_lhs_t(lhs_bf16, qt_rule)

    # Quantize grad for dlhs path
    dlhs_dout = qpl.quantize(
        grad,
        qt_rule.bwd_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size},
        calibration_method=qt_rule.bwd_calibration_method,
        scale_dtype=jnp.float32,
    )
    # Quantize grad for drhs path
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
        rhs=rhs_q,
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

    def loss_fn(lhs, rhs):
        return gmm_fp8_blockwise(lhs, rhs, group_sizes).sum()

    loss, _ = jax.value_and_grad(loss_fn, argnums=(0, 1))(lhs, rhs)
    return loss
# EVOLVE-BLOCK-END
