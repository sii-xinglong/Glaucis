"""Reference implementation for backward FP8 transposed grouped matrix multiply (tGMM).

Computes per-group matmul of transposed-activation [K, M] with gradient [M, N],
using FP8 E4M3 block-wise quantization with 1x128 block scales.

Self-contained: depends only on JAX and numpy.
"""

from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np

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
    absmax = jnp.max(jnp.abs(blocked), axis=-1)  # [..., num_blocks]
    absmax = jnp.maximum(absmax, 1e-12)
    scale = absmax / FP8_MAX  # [..., num_blocks]
    scale_expanded = scale[..., jnp.newaxis]  # [..., num_blocks, 1]
    scaled = blocked / scale_expanded
    scaled = jnp.clip(scaled, -FP8_MAX, FP8_MAX)
    qvalue = scaled.reshape(orig_shape).astype(jnp.float8_e4m3fn)
    return QArray(qvalue=qvalue, scale=scale)


def _dequantize(qarray):
    """Dequantize a QArray back to float32."""
    orig_shape = qarray.qvalue.shape
    num_blocks = qarray.scale.shape[-1]
    blocked = qarray.qvalue.reshape(*orig_shape[:-1], num_blocks, BLOCK_SIZE).astype(
        jnp.float32
    )
    scale_expanded = qarray.scale[..., jnp.newaxis].astype(jnp.float32)
    return (blocked * scale_expanded).reshape(orig_shape)


def simple_compute(M=2048, K=512, N=1024, num_groups=4):
    """Generate FP8-quantized inputs and compute tGMM backward pass.

    Args:
        M: inner (contracted) dimension, must be divisible by num_groups and BLOCK_SIZE.
        K: rows of lhs / output columns-per-group first dim.
        N: cols of rhs / output second dim.
        num_groups: number of groups along M.

    Returns:
        jnp.array of shape [num_groups, K, N] in bfloat16.
    """
    key = jax.random.PRNGKey(43)
    k1, k2 = jax.random.split(key)

    # Generate full-precision inputs then quantize
    lhs_fp = jax.random.normal(k1, (K, M), dtype=jnp.float32)
    rhs_fp = jax.random.normal(k2, (M, N), dtype=jnp.float32)

    # Quantize with 1x128 block scales along last dimension
    # lhs [K, M] -> scale shape [K, M//128]
    # rhs [M, N] -> scale shape [M, N//128]
    lhs_q = _quantize_fp8_blockwise(lhs_fp)
    rhs_q = _quantize_fp8_blockwise(rhs_fp)

    # Dequantize for reference computation
    lhs = _dequantize(lhs_q)
    rhs = _dequantize(rhs_q)

    # Per-group matmul: lhs[:, slice_g] @ rhs[slice_g, :] -> [K, N]
    group_size = M // num_groups
    results = []
    start = 0
    for _ in range(num_groups):
        lhs_slice = lhs[:, start : start + group_size]  # [K, gs]
        rhs_slice = rhs[start : start + group_size, :]  # [gs, N]
        out_g = jnp.dot(lhs_slice, rhs_slice)  # [K, N] in f32
        results.append(out_g)
        start += group_size

    return jnp.stack(results).astype(jnp.bfloat16)  # [num_groups, K, N]


def reference_fn(**kwargs):
    """Entry point used by evaluate.py."""
    return simple_compute(**kwargs)
