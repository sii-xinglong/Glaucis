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
    """GMM with fp8_blockwise quantization — MXU/VPU overlap variant.

    Key changes vs baseline:
    1. Defer lhs_t quantization from forward to backward pass, reducing
       the number of live quantized tensors and register pressure.
    2. Quantize rhs first (weight, can be done independently) then lhs,
       so JAX can overlap VPU quantization work with MXU dispatch.
    3. In backward pass, overlap lhs_t quantization with the dlhs GMM
       by placing them in sequence so the compiler can pipeline VPU work
       for lhs_t while MXU executes the dlhs matmul.
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
    """Forward pass: quantize rhs and lhs only; defer lhs_t to backward.

    By NOT quantizing lhs_t here, we:
    - Eliminate one full qpl.quantize pass from the forward (saves VPU time)
    - Reduce the number of live tensors from 3 quantized to 2 quantized + 1 bf16
    - Cut register pressure roughly by 1/3 for quantized tensor storage
    - Allow the compiler to better schedule MXU vs VPU work with fewer conflicts

    We save lhs_bf16 in residuals so the backward can quantize lhs_t on demand,
    right when it's actually needed for tgmm.
    """
    tile_size = qt_rule.tile_size
    tiling = tuple(min(t, tile_size) for t in tiling)

    lhs_bf16 = lhs

    # Quantize rhs FIRST — this is the weight tensor, independent of lhs.
    # The compiler can start MXU weight-loading/preparation while we
    # subsequently quantize lhs on the VPU.
    rhs = qpl.quantize(
        rhs,
        qt_rule.weight_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size, 2: tile_size},
        calibration_method=qt_rule.weight_calibration_method,
        scale_dtype=jnp.float32,
    )

    # Quantize lhs (activation) — by this point rhs quantization results
    # may already be flowing to MXU tile buffers via DMA.
    lhs = qpl.quantize(
        lhs_bf16,
        qt_rule.act_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size},
        calibration_method=qt_rule.act_calibration_method,
        scale_dtype=jnp.float32,
    )

    # GMM: MXU can begin as soon as lhs and rhs quantized data is ready.
    # No lhs_t quantization blocking the pipeline anymore.
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

    # Save lhs_bf16 (not lhs_t quantized) in residuals.
    # lhs_t will be quantized on-demand in backward, overlapping with MXU work.
    return out, (lhs, rhs, group_sizes, lhs_bf16)


def _gmm_bwd(lhs_dtype, rhs_dtype, qt_rule, tiling, residual, grad):
    """Backward pass: overlap lhs_t quantization with dlhs GMM on MXU.

    Restructured flow:
    1. Quantize grad for dlhs (VPU work)
    2. Start dlhs GMM (MXU work) — compiler can pipeline this
    3. While MXU executes dlhs GMM, quantize lhs_t and grad for drhs (VPU work)
       This is the KEY overlap: MXU does matmul while VPU does quantization
    4. Execute drhs tgmm (MXU work) — lhs_t and drhs_dout are now ready
    """
    lhs, rhs, group_sizes, lhs_bf16 = residual
    num_actual_groups = rhs.shape[0]
    tile_size = qt_rule.tile_size
    tiling = tuple(min(t, tile_size) for t in tiling)

    # Step 1: Quantize grad for the dlhs path (VPU)
    dlhs_dout = qpl.quantize(
        grad,
        qt_rule.bwd_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size},
        calibration_method=qt_rule.bwd_calibration_method,
        scale_dtype=jnp.float32,
    )

    # Step 2: Launch dlhs GMM (MXU) — rhs is already quantized from forward
    # The compiler can start executing this on MXU immediately.
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

    # Step 3: While MXU is busy with dlhs, quantize lhs_t and drhs_dout on VPU.
    # This is the main overlap opportunity: MXU computes dlhs matmul while
    # VPU concurrently handles these two quantizations.
    #
    # Quantize lhs_t from saved bf16 data (deferred from forward pass).
    lhs_t = qpl.quantize(
        lhs_bf16.swapaxes(0, 1),
        qt_rule.act_qtype,
        channelwise_axes=[0],
        tiled_axes={1: tile_size},
        calibration_method=qt_rule.act_calibration_method,
        scale_dtype=jnp.float32,
    )

    # Quantize grad for the drhs path — different axis configuration than dlhs
    drhs_dout = qpl.quantize(
        grad,
        qt_rule.bwd_qtype,
        channelwise_axes=[1],
        tiled_axes={0: tile_size},
        calibration_method=qt_rule.bwd_calibration_method,
        scale_dtype=jnp.float32,
    )

    # Step 4: drhs tgmm (MXU) — both lhs_t and drhs_dout are ready now
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
