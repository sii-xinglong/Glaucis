"""Reference implementation for FP8 block-wise grouped matrix multiplication (forward).

Self-contained -- depends only on JAX and numpy.  Exports the two entry
points that evaluate.py looks for:

  * ``simple_compute(M, K, N, num_groups)``
  * ``reference_fn(**kwargs)``

FP8 E4M3 quantisation uses per-block absmax scaling:
  - lhs  [M, K]              -> 1x128   blocks -> scales [M, K//128]
  - rhs  [num_groups, K, N]  -> 128x128 blocks -> scales [num_groups, K//128, N//128]

The reference multiplies dequantised slices per group and concatenates
the results into an [M, N] bfloat16 tensor.
"""

from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# FP8 block-wise quantisation helpers
# ---------------------------------------------------------------------------

QArray = namedtuple("QArray", ["qvalue", "scale"])

_FP8_E4M3_MAX = 448.0


def _quantize_1x128(x):
    """Quantise a 2-D tensor with 1x128 block scaling to float8_e4m3fn.

    Args:
        x: array of shape [rows, K]  (K must be divisible by 128).

    Returns:
        QArray with qvalue [rows, K] fp8 and scale [rows, K//128] float32.
    """
    rows, K = x.shape
    block_k = 128
    assert K % block_k == 0, f"K={K} must be divisible by {block_k}"

    x_f32 = x.astype(jnp.float32)
    # Reshape into blocks of size 128 along K.
    blocks = x_f32.reshape(rows, K // block_k, block_k)
    absmax = jnp.max(jnp.abs(blocks), axis=-1)           # [rows, K//128]
    scale = absmax / _FP8_E4M3_MAX                        # [rows, K//128]
    scale = jnp.maximum(scale, jnp.finfo(jnp.float32).tiny)

    # Quantise: divide each element by its block scale, clip, cast.
    inv_scale = (1.0 / scale)[:, :, None]                 # [rows, K//128, 1]
    scaled = blocks * inv_scale
    scaled = jnp.clip(scaled, -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    qvalue = scaled.reshape(rows, K).astype(jnp.float8_e4m3fn)

    return QArray(qvalue=qvalue, scale=scale)


def _quantize_128x128(x):
    """Quantise a 3-D tensor with 128x128 block scaling to float8_e4m3fn.

    Args:
        x: array of shape [G, K, N]  (K and N must be divisible by 128).

    Returns:
        QArray with qvalue [G, K, N] fp8 and scale [G, K//128, N//128] float32.
    """
    G, K, N = x.shape
    bk, bn = 128, 128
    assert K % bk == 0, f"K={K} must be divisible by {bk}"
    assert N % bn == 0, f"N={N} must be divisible by {bn}"

    x_f32 = x.astype(jnp.float32)
    blocks = x_f32.reshape(G, K // bk, bk, N // bn, bn)  # [G, K//128, 128, N//128, 128]
    absmax = jnp.max(jnp.abs(blocks), axis=(2, 4))        # [G, K//128, N//128]
    scale = absmax / _FP8_E4M3_MAX
    scale = jnp.maximum(scale, jnp.finfo(jnp.float32).tiny)

    inv_scale = (1.0 / scale)[:, :, None, :, None]        # broadcast over block dims
    scaled = blocks * inv_scale
    scaled = jnp.clip(scaled, -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    qvalue = scaled.reshape(G, K, N).astype(jnp.float8_e4m3fn)

    return QArray(qvalue=qvalue, scale=scale)


# ---------------------------------------------------------------------------
# Dequantisation helpers
# ---------------------------------------------------------------------------

def _dequantize_1x128(q):
    """Dequantise a QArray produced by ``_quantize_1x128``.

    Returns a float32 tensor of the original shape.
    """
    rows, K = q.qvalue.shape
    block_k = 128
    # scale: [rows, K//128] -> broadcast to [rows, K//128, 128]
    scale_bc = q.scale[:, :, None]
    blocks = q.qvalue.astype(jnp.float32).reshape(rows, K // block_k, block_k)
    return (blocks * scale_bc).reshape(rows, K)


def _dequantize_128x128(q):
    """Dequantise a QArray produced by ``_quantize_128x128``.

    Returns a float32 tensor of the original shape.
    """
    G, K, N = q.qvalue.shape
    bk, bn = 128, 128
    # scale: [G, K//128, N//128] -> broadcast over block dims
    scale_bc = q.scale[:, :, None, :, None]  # [G, K//128, 1, N//128, 1]
    blocks = q.qvalue.astype(jnp.float32).reshape(G, K // bk, bk, N // bn, bn)
    return (blocks * scale_bc).reshape(G, K, N)


# ---------------------------------------------------------------------------
# Reference grouped matmul
# ---------------------------------------------------------------------------

def simple_compute(M=2048, K=1024, N=1024, num_groups=4):
    """Generate FP8-quantised inputs and compute grouped matmul.

    All random state is derived from ``jax.random.PRNGKey(42)`` so that
    the kernel template can reproduce identical inputs.

    Returns:
        jnp.ndarray of shape [M, N] in bfloat16.
    """
    assert M % num_groups == 0, f"M={M} must be divisible by num_groups={num_groups}"

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    # Generate full-precision inputs.
    lhs_fp = jax.random.normal(k1, (M, K), dtype=jnp.float32)
    rhs_fp = jax.random.normal(k2, (num_groups, K, N), dtype=jnp.float32)

    # Quantise to FP8.
    lhs_q = _quantize_1x128(lhs_fp)
    rhs_q = _quantize_128x128(rhs_fp)

    # Dequantise back to float32 for reference matmul.
    lhs_deq = _dequantize_1x128(lhs_q)     # [M, K] float32
    rhs_deq = _dequantize_128x128(rhs_q)   # [num_groups, K, N] float32

    group_size = M // num_groups
    results = []
    for g in range(num_groups):
        offset = g * group_size
        lhs_slice = lhs_deq[offset:offset + group_size, :]   # [group_size, K]
        out_g = jnp.dot(lhs_slice, rhs_deq[g])               # [group_size, N]
        results.append(out_g)

    return jnp.concatenate(results, axis=0).astype(jnp.bfloat16)


def reference_fn(**kwargs):
    """Entry point used by evaluate.py."""
    return simple_compute(**kwargs)
