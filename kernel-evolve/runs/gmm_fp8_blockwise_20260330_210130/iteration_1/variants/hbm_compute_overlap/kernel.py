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

    Mutation: hbm_compute_overlap — defer lhs_t quantization to backward pass
    to halve live quantized tensors during forward, reducing register pressure.
    Save lhs_bf16 in residuals instead of pre-quantized lhs_t; re-quantize
    lhs_t on demand in backward. This trades a small re-quantization cost for
    a large reduction in simultaneously-live values, cutting spill/fill traffic.
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
    """Forward pass: quantize only lhs and rhs. Defer lhs_t to backward.

    By NOT quantizing lhs_t here, we avoid creating a third large quantized
    tensor that must be kept alive through the forward GMM computation.
    The compiler sees fewer simultaneously-live values, reducing register
    pressure and the catastrophic spill/fill cycle.

    We save lhs_bf16 (the original bf16 input) in residuals so the backward
    pass can quantize lhs_t on demand. lhs_bf16 is a single bf16 tensor
    vs the quantized lhs_t which is a QuantizedTensor (data + scales),
    so residual memory is comparable but forward-pass live values are halved.
    """
    tile_size = qt_rule.tile_size
    tiling = tuple(min(t, tile_size) for t in tiling)

    # Keep original bf16 for backward pass re-quantization
    lhs_bf16 = lhs

    # Phase 1: Quantize lhs for forward GMM
    lhs_q = qpl.quantize(
        lhs_bf16,
        qt_rule.act_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size},
        calibration_method=qt_rule.act_calibration_method,
        scale_dtype=jnp.float32,
    )

    # Phase 2: Quantize rhs for forward GMM (lhs_q quantization is done,
    # compiler can begin scheduling its DMA while we quantize rhs)
    rhs_q = qpl.quantize(
        rhs,
        qt_rule.weight_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size, 2: tile_size},
        calibration_method=qt_rule.weight_calibration_method,
        scale_dtype=jnp.float32,
    )

    # Phase 3: Forward GMM — only lhs_q and rhs_q are live quantized tensors
    # (previously lhs_q, rhs_q, AND lhs_t_q were all live here)
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

    # Residuals: save lhs_bf16 instead of pre-quantized lhs_t.
    # This means backward will re-quantize lhs_t, but forward has
    # dramatically fewer live values.
    return out, (lhs_q, rhs_q, group_sizes, lhs_bf16)


def _gmm_bwd(lhs_dtype, rhs_dtype, qt_rule, tiling, residual, grad):
    """Backward pass: quantize lhs_t here (deferred from forward).

    The re-quantization of lhs_t is a small cost compared to the massive
    spill/fill savings in the forward pass. The backward pass naturally
    needs lhs_t only for the tgmm, so quantizing it here means it's live
    only when needed, not carried through the entire forward computation.
    """
    lhs_q, rhs_q, group_sizes, lhs_bf16 = residual
    num_actual_groups = rhs_q.shape[0]
    tile_size = qt_rule.tile_size
    tiling = tuple(min(t, tile_size) for t in tiling)

    # Phase 1: Quantize grad for dlhs computation
    dlhs_dout = qpl.quantize(
        grad,
        qt_rule.bwd_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size},
        calibration_method=qt_rule.bwd_calibration_method,
        scale_dtype=jnp.float32,
    )

    # Phase 2: Compute dlhs — uses dlhs_dout and rhs_q (already quantized)
    # Do this BEFORE quantizing lhs_t and drhs_dout to keep live values low
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

    # Phase 3: Now quantize lhs_t (deferred from forward pass).
    # At this point dlhs computation is done, so dlhs_dout and rhs_q
    # can be released by the compiler, freeing registers before we
    # create lhs_t_q and drhs_dout.
    lhs_t_q = qpl.quantize(
        lhs_bf16.swapaxes(0, 1),
        qt_rule.act_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size},
        calibration_method=qt_rule.act_calibration_method,
        scale_dtype=jnp.float32,
    )

    # Phase 4: Quantize grad for drhs computation (separate from dlhs_dout
    # quantization to avoid both being live simultaneously)
    drhs_dout = qpl.quantize(
        grad,
        qt_rule.bwd_qtype,
        channelwise_axes=[1],
        tiled_axes={0: tile_size},
        calibration_method=qt_rule.bwd_calibration_method,
        scale_dtype=jnp.float32,
    )

    # Phase 5: Compute drhs — only lhs_t_q and drhs_dout need to be live
    drhs = tokamax_backend.tgmm(
        lhs=lhs_t_q,
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
