"""GMM block-wise FP8 quantization kernel for TPU -- L2_fwd_tn256_tk512 variant.

Builds on L2_fwd_tk512 (2.570x) which proved forward TK=512 effective with
phase-specialized tiling (256,512,128, 256,512,128, 2048,512,128).

This variant increases forward TN from 128 to 256, doubling the N-tile size
in the forward gmm pass while keeping TK=512. For Gate/Up (N=512), this means
2 N-iterations instead of 4. For Down (N=2048), 8 iterations instead of 16.

Tiling changes vs parent:
  - Forward gmm:  (256, 512, 256) -- TN 128->256
  - Backward gmm: (256, 512, 128) -- unchanged
  - tgmm:         (2048, 512, 128) -- unchanged

The two evaluated shapes have different K/N dimensions:
  Gate/Up: lhs [256, 2048] @ rhs [2048, 512]   (per-group, K=2048, N=512)
  Down:    lhs [256, 512]  @ rhs [512, 2048]    (per-group, K=512, N=2048)

Key changes vs baseline:
  - Forward: quantize only lhs and rhs (skip lhs_t quantization)
  - Forward residuals: store lhs_bf16 instead of quantized lhs_t
  - Backward bwd_gmm: bf16 grad + fp8 rhs (mixed precision, SO7)
  - Backward tgmm: compute lhs_t from stored lhs_bf16, pass bf16 directly
  - Zero backward quantization (SO8)
  - Phase-specialized tiling: (256,512,256, 256,512,128, 2048,512,128)
  - Remove min(t, tile_size) clamping to allow tiling > 128

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


def _clamp_tiling(tiling, tile_size):
    """Clamp each tiling dimension to be >= 128 (the minimum) but allow > tile_size.

    Unlike the baseline min(t, tile_size) which caps at 128, this only
    enforces the lower bound, enabling larger tiles like 1024 or 2048.
    """
    return tuple(max(t, 128) for t in tiling)


# EVOLVE-BLOCK-START
def gmm_fp8_blockwise(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    tiling: tuple[int, ...] = (256, 512, 256, 256, 512, 128, 2048, 512, 128),
) -> jnp.ndarray:
    """GMM with fp8_blockwise quantization and tokamax backend.

    L2_fwd_tn256_tk512 variant: forward TN 128->256 on L2 base.
    fwd (256,512,256), bwd_gmm (256,512,128), tgmm (2048,512,128).
    Zero backward quantization with mixed-precision bwd_gmm.
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
    # Use _clamp_tiling instead of min(t, tile_size) to allow tiling > 128
    tiling = _clamp_tiling(tiling, tile_size)

    # Store bf16 lhs for backward (needed to compute lhs_t in backward)
    lhs_bf16 = lhs

    # Quantize lhs to fp8 for forward gmm
    lhs = qpl.quantize(
        lhs_bf16,
        qt_rule.act_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size},
        calibration_method=qt_rule.act_calibration_method,
        scale_dtype=jnp.float32,
    )

    # SKIP lhs_t quantization entirely -- SO8 showed tgmm uses bf16 anyway.
    # This saves VPU cycles and reduces register pressure (160K spills in baseline).

    # Quantize rhs to fp8 for forward gmm
    rhs = qpl.quantize(
        rhs,
        qt_rule.weight_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size, 2: tile_size},
        calibration_method=qt_rule.weight_calibration_method,
        scale_dtype=jnp.float32,
    )

    # Forward gmm with phase-specialized tiling: (256, 512, 256)
    # TM=256 matches per-group M exactly (M=8192 / G=32 = 256)
    # TK=512 halves K-loop iterations (K=2048: 4 iters instead of 8)
    # TN=256 halves N-loop iterations vs TN=128
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

    # Residuals: store lhs_bf16 (not quantized lhs_t), plus fp8 rhs and group_sizes
    return out, (lhs_bf16, rhs, group_sizes)


def _gmm_bwd(lhs_dtype, rhs_dtype, qt_rule, tiling, residual, grad):
    lhs_bf16, rhs, group_sizes = residual
    num_actual_groups = rhs.shape[0]
    tile_size = qt_rule.tile_size
    # Use _clamp_tiling instead of min(t, tile_size) to allow tiling > 128
    tiling = _clamp_tiling(tiling, tile_size)

    # Zero backward quantization (SO8): pass bf16 grad and fp8 rhs directly
    # to bwd_gmm. Mixed precision: bf16 grad + fp8 rhs (SO7).
    # Backward gmm tiling: (256, 512, 128) -- TM=256 for per-group alignment,
    # TK=512 halves K-loop iterations (K=2048: 4 iters instead of 8)
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

    # Compute lhs_t from stored bf16 lhs in backward (no quantization needed)
    lhs_t = lhs_bf16.swapaxes(0, 1)

    # tgmm with bf16 lhs_t + bf16 grad directly (SO5: tgmm is bf16-safe)
    # tgmm tiling: (2048, 512, 128) -- TK=512 to reduce K-loop iterations.
    # Since tgmm operates in bf16 (not fp8), larger K tiles are safe and
    # reduce loop overhead without fp8 quantization tile constraints.
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
