"""Reference BF16 GMM for correctness comparison (no FP8 quantization).

Uses tokamax backend directly with bfloat16 precision.
Provides reference_fn(M, K, N, G) that runs forward + backward,
matching the optimized kernel's interface.
"""

import jax
import jax.numpy as jnp

from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_kernel as tokamax_backend


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


def gmm_bf16(lhs, rhs, group_sizes):
    """Reference GMM in BF16 using tokamax backend (no quantization)."""
    return tokamax_backend.gmm(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        out_dtype=jnp.float32,
        tiling=(128, 128, 128),
        transpose_rhs=False,
        interpret=False,
    )


def simple_compute(M=8192, K=2048, N=512, G=32):
    """Forward-only BF16 GMM reference.

    Returns the loss scalar for correctness comparison.
    Note: no backward pass because tokamax pallas_call does not
    support automatic JVP differentiation (only custom_vjp works).
    """
    lhs, rhs, group_sizes = _make_test_data(M, K, N, G)
    return gmm_bf16(lhs, rhs, group_sizes).sum()


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
