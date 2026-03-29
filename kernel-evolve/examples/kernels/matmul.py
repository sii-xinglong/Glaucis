"""Tiled matmul Pallas kernel for TPU -- template for evolutionary optimization."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


# EVOLVE-BLOCK-START
def matmul_kernel(x_ref, y_ref, o_ref):
  acc = jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)
  o_ref[...] = acc.astype(o_ref.dtype)


def optimized_compute(M=1024, N=1024, K=1024):
  x = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=jnp.bfloat16)
  y = jax.random.normal(jax.random.PRNGKey(1), (K, N), dtype=jnp.bfloat16)
  BLOCK_M, BLOCK_N = 128, 128
  result = pl.pallas_call(
    matmul_kernel,
    out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
    grid=(M // BLOCK_M, N // BLOCK_N),
    in_specs=[
      pl.BlockSpec((BLOCK_M, K), lambda i, j: (i, 0)),
      pl.BlockSpec((K, BLOCK_N), lambda i, j: (0, j)),
    ],
    out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),
  )(x, y)
  return result
# EVOLVE-BLOCK-END
