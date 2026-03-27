"""Tiled matmul Pallas kernel for TPU -- template for evolutionary optimization."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def matmul_kernel(x_ref, y_ref, o_ref):
  # EVOLVE-BLOCK-START
  BLOCK_K = 128
  acc = jnp.zeros(o_ref.shape, dtype=jnp.float32)
  def body(i, acc):
    x = pl.load(x_ref, (pl.dslice(None), pl.dslice(i * BLOCK_K, BLOCK_K)))
    y = pl.load(y_ref, (pl.dslice(i * BLOCK_K, BLOCK_K), pl.dslice(None)))
    acc += jnp.dot(x, y)
    return acc
  k_tiles = x_ref.shape[1] // BLOCK_K
  acc = jax.lax.fori_loop(0, k_tiles, body, acc)
  o_ref[...] = acc.astype(o_ref.dtype)
  # EVOLVE-BLOCK-END


def optimized_compute(M=1024, N=1024, K=1024):
  x = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=jnp.float16)
  y = jax.random.normal(jax.random.PRNGKey(1), (K, N), dtype=jnp.float16)
  BLOCK_M, BLOCK_N = 128, 128
  result = pl.pallas_call(
    matmul_kernel,
    out_shape=jax.ShapeDtypeStruct((M, N), jnp.float16),
    grid=(M // BLOCK_M, N // BLOCK_N),
    in_specs=[
      pl.BlockSpec((BLOCK_M, K), lambda i, j: (i, 0)),
      pl.BlockSpec((K, BLOCK_N), lambda i, j: (0, j)),
    ],
    out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),
  )(x, y)
  return result
