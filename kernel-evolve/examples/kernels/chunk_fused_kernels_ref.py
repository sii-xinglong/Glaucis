# Source: primatrix/pallas-kernel @ branch: feat/chunk-gla-fused-kernels
# Commit: 1de541558ce12b8fc6c439a85f020422a4ba2c6a
# Initialized: 2026-04-01
"""Reference chunked GLA fused kernels (g_gamma mode) for correctness comparison.

Pure JAX reference implementation of the fused chunk-GLA forward and backward
passes. Uses naive recurrent GLA as the ground truth, wrapped with custom_vjp
to produce forward+backward scalar loss for the evaluate.py pipeline.

The fused kernel merges three separate pallas_calls into one:
  1. h propagation (inter-chunk state)
  2. A recomputation (intra-chunk attention)
  3. Output computation

This reference uses the naive step-by-step recurrence (no Pallas) to serve
as the correctness baseline.
"""

import jax
import jax.lax as lax
import jax.numpy as jnp


def _make_test_data(B, T, H, K, V, chunk_size, seed=42):
    """Create deterministic (q, k, v, g_gamma) for a fused GLA test case."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    q = jax.random.normal(k1, (B, T, H, K), dtype=jnp.bfloat16)
    k_arr = jax.random.normal(k2, (B, T, H, K), dtype=jnp.bfloat16)
    v = jax.random.normal(k3, (B, T, H, V), dtype=jnp.bfloat16)
    g_gamma = -jnp.abs(jax.random.normal(k4, (H,), dtype=jnp.float32)) * 0.1
    return q, k_arr, v, g_gamma


# ============================================================
# Naive recurrent GLA (step-by-step reference, no Pallas)
# ============================================================


def naive_recurrent_gla(q, k, v, gk, scale):
    """Naive step-by-step recurrent GLA.

    Core recurrence:
        h_t = h_{t-1} * exp(gk_t) + k_t^T @ v_t
        o_t = (q_t * scale) . h_t  (then sum along K dimension)

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        gk: [B, T, H, K] — per-step log-space gates
        scale: scaling factor

    Returns:
        o: [B, T, H, V]
    """
    dtype = q.dtype
    q, k, v, gk = (
        jnp.transpose(x, (0, 2, 1, 3)).astype(jnp.float32) for x in (q, k, v, gk)
    )
    B, H, T, K = q.shape
    V = v.shape[-1]

    o = jnp.zeros_like(v)
    h = jnp.zeros((B, H, K, V), dtype=jnp.float32)

    for t in range(T):
        q_t = q[:, :, t] * scale
        k_t = k[:, :, t]
        v_t = v[:, :, t]
        gk_t = jnp.exp(gk[:, :, t])
        kv_t = k_t[..., None] * v_t[..., None, :]
        h = h * gk_t[..., None] + kv_t
        o = o.at[:, :, t].set((q_t[..., None] * h).sum(-2))

    return jnp.transpose(o, (0, 2, 1, 3)).astype(dtype)


# ============================================================
# Reference forward+backward via custom_vjp on naive recurrence
# ============================================================


def chunk_fused_ref(q, k, v, g_gamma, scale, chunk_size):
    """Chunked GLA fused reference with custom_vjp (pure JAX naive recurrence).

    Uses naive_recurrent_gla as the compute function, wrapped with custom_vjp
    to enable gradient computation.
    """
    B, T, H, K = q.shape
    gk = jnp.broadcast_to(g_gamma.reshape(1, 1, -1, 1), (B, T, H, K))

    @jax.custom_vjp
    def _compute(q, k, v):
        return naive_recurrent_gla(q, k, v, gk, scale)

    def _fwd(q, k, v):
        o = naive_recurrent_gla(q, k, v, gk, scale)
        return o, (q, k, v)

    def _bwd(residuals, do):
        q_r, k_r, v_r = residuals

        def loss_fn(q_, k_, v_):
            return (naive_recurrent_gla(q_, k_, v_, gk, scale)
                    * do.astype(jnp.float32)).sum()

        dq, dk, dv = jax.grad(loss_fn, argnums=(0, 1, 2))(
            q_r.astype(jnp.float32),
            k_r.astype(jnp.float32),
            v_r.astype(jnp.float32),
        )
        return dq.astype(q_r.dtype), dk.astype(k_r.dtype), dv.astype(v_r.dtype)

    _compute.defvjp(_fwd, _bwd)
    return _compute(q, k, v)


# ============================================================
# Public API for evaluate.py
# ============================================================


def simple_compute(B=2, T=256, H=4, K=128, V=128, chunk_size=64):
    """Forward + backward fused chunk-GLA (pure JAX naive reference).

    Returns the loss scalar. Both forward and backward are computed,
    so timing captures the full training step performance.
    """
    q, k_arr, v, g_gamma = _make_test_data(B, T, H, K, V, chunk_size)
    scale = K ** -0.5

    def loss_fn(q, k, v):
        return chunk_fused_ref(
            q.astype(jnp.float32), k.astype(jnp.float32),
            v.astype(jnp.float32), g_gamma, scale, chunk_size,
        ).sum()

    loss, _ = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k_arr, v)
    return loss


def reference_fn(**kwargs):
    """Generic entry point for evaluate.py function discovery."""
    return simple_compute(**kwargs)
