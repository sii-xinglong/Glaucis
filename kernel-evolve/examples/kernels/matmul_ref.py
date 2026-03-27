"""Simple JAX reference matmul for correctness comparison."""

import jax
import jax.numpy as jnp


def simple_compute(M=1024, N=1024, K=1024):
  x = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=jnp.bfloat16)
  y = jax.random.normal(jax.random.PRNGKey(1), (K, N), dtype=jnp.bfloat16)
  return jnp.dot(x, y)


def reference_fn(**kwargs):
  return simple_compute(**kwargs)
