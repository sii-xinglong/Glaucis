"""Reference pure JAX implementation of chunked GLA for correctness comparison.

Uses lax.scan for sequential state propagation and einsum for attention.
g_gamma-only path (per-head constant gate, no per-element gates).
"""

import functools

import jax
import jax.lax as lax
import jax.numpy as jnp


def _make_test_data(B, T, H, K, V, chunk_size, seed=42):
    """Create deterministic (q, k, v, g_gamma) for a GLA test case."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    q = jax.random.normal(k1, (B, T, H, K), dtype=jnp.bfloat16)
    k_arr = jax.random.normal(k2, (B, T, H, K), dtype=jnp.bfloat16)
    v = jax.random.normal(k3, (B, T, H, V), dtype=jnp.bfloat16)
    g_gamma = -jnp.abs(jax.random.normal(k4, (H,), dtype=jnp.float32)) * 0.1
    return q, k_arr, v, g_gamma


def _chunk_fwd_h_scan(k, v, g_gamma, h0, output_final_state, C, B, T, H, K, V, NT):
    """lax.scan-based forward state propagation."""
    k_scan = k.reshape(B, NT, C, H, K).transpose(1, 0, 2, 3, 4)
    v_scan = v.reshape(B, NT, C, H, V).transpose(1, 0, 2, 3, 4)

    g_gamma_f32 = g_gamma.astype(jnp.float32)
    state_decay = jnp.exp(g_gamma_f32 * C)
    b_g = g_gamma_f32[None, :] * (jnp.arange(C, dtype=jnp.float32) + 1)[:, None]
    v_decay = jnp.exp(g_gamma_f32[None, :] * C - b_g)

    def scan_fn(h, chunk_data):
        b_k, b_v = chunk_data
        h_out = h
        h = h * state_decay[None, :, None, None]
        b_v = (b_v * v_decay[None, :, :, None]).astype(b_v.dtype)
        kv = lax.dot_general(
            b_k.astype(jnp.float32), b_v.astype(jnp.float32),
            dimension_numbers=(((1,), (1,)), ((0, 2), (0, 2))),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        h = h + kv
        return h, h_out

    h_init = jnp.zeros((B, H, K, V), dtype=jnp.float32)
    if h0 is not None:
        h_init = h0.reshape(B, H, K, V).astype(jnp.float32)

    h_final, h_all = lax.scan(scan_fn, h_init, (k_scan, v_scan))
    h_all = h_all.transpose(1, 0, 2, 3, 4)

    ht = h_final.astype(jnp.float32) if output_final_state else None
    return h_all, ht


def _chunk_gla_fwd_intra(q, k, g_cumsum, scale, chunk_size):
    """Intra-chunk attention matrix."""
    B, T, H, K = q.shape
    C = chunk_size
    NT = T // C

    q_c = q.reshape(B, NT, C, H, K)
    k_c = k.reshape(B, NT, C, H, K)
    gc_c = g_cumsum.reshape(B, NT, C, H, K)

    q_gated = q_c * jnp.exp(gc_c)
    k_gated = k_c * jnp.exp(-gc_c)

    A = jnp.einsum(
        "bnihk,bnjhk->bnihj", q_gated, k_gated,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ) * scale
    A = A.reshape(B, T, H, C)
    return A


def _chunk_gla_fwd_o(q, v, g_cumsum, A, h, scale, chunk_size):
    """Combine inter-chunk and intra-chunk output."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    q_flat = q.reshape(-1, C, H, K)
    v_flat = v.reshape(-1, C, H, V)
    gc_flat = g_cumsum.reshape(-1, C, H, K)
    h_flat = h.reshape(-1, H, K, V)
    A_flat = A.reshape(-1, C, H, C)

    qg = q_flat * jnp.exp(gc_flat)
    o_inter = scale * jnp.einsum("nchk,nhkv->nchv", qg, h_flat)

    causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))[:, None, :]
    n_A = jnp.where(causal_mask, A_flat, 0.0)
    o_intra = jnp.einsum("nihj,njhv->nihv", n_A, v_flat)

    return (o_inter + o_intra).reshape(B, T, H, V)


def chunk_gla_fwd_ref(q, k, v, g_gamma, scale, chunk_size):
    """Forward orchestrator."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C

    pos = jnp.arange(1, C + 1, dtype=jnp.float32)
    pos = jnp.tile(pos, NT).reshape(1, T, 1, 1)
    g_cumsum = jnp.broadcast_to(g_gamma * pos, q.shape)

    h, ht = _chunk_fwd_h_scan(k, v, g_gamma, None, False, C, B, T, H, K, V, NT)
    A = _chunk_gla_fwd_intra(q, k, g_cumsum, scale, C)
    o = _chunk_gla_fwd_o(q, v, g_cumsum, A, h, scale, C)

    return g_cumsum, A, h, o


def _chunk_bwd_dh_scan(q, do, g_gamma, scale, C, B, T, H, K, V, NT):
    """lax.scan-based backward state gradient propagation (reverse)."""
    q_scan = q.reshape(B, NT, C, H, K).transpose(1, 0, 2, 3, 4)
    do_scan = do.reshape(B, NT, C, H, V).transpose(1, 0, 2, 3, 4)

    g_gamma_f32 = g_gamma.astype(jnp.float32)
    state_decay = jnp.exp(g_gamma_f32 * C)
    b_g_ramp = g_gamma_f32[None, :] * (jnp.arange(C, dtype=jnp.float32) + 1)[:, None]

    def scan_fn(dh, chunk_data):
        b_q, b_do = chunk_data
        dh_out = dh
        dh = dh * state_decay[None, :, None, None]
        b_q_hat = (b_q * jnp.exp(b_g_ramp)[None, :, :, None] * scale)
        dh = dh + lax.dot_general(
            b_q_hat.astype(jnp.float32), b_do.astype(jnp.float32),
            dimension_numbers=(((1,), (1,)), ((0, 2), (0, 2))),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        return dh, dh_out

    dh_init = jnp.zeros((B, H, K, V), dtype=jnp.float32)
    _, dh_all = lax.scan(scan_fn, dh_init, (q_scan, do_scan), reverse=True)
    dh_all = dh_all.transpose(1, 0, 2, 3, 4)
    return dh_all


def _chunk_gla_bwd_ref(q, k, v, g_cumsum, h, A, do, dh, scale, chunk_size):
    """Backward: compute dq, dk, dv from saved intermediates."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C

    q_c = q.reshape(B, NT, C, H, K)
    k_c = k.reshape(B, NT, C, H, K)
    v_c = v.reshape(B, NT, C, H, V)
    gc_c = g_cumsum.reshape(B, NT, C, H, K)
    do_c = do.reshape(B, NT, C, H, V)
    A_c = A.reshape(B, NT, C, H, C)
    gn = gc_c[:, :, -1, :, :]

    # dA
    dA_c = (
        jnp.einsum("bnihv,bnjhv->bnihj", do_c, v_c, precision=lax.Precision.HIGHEST)
        * scale
    )
    causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
    dA_c = jnp.where(causal_mask[None, None, :, None, :], dA_c, 0.0)

    # dv
    A_masked = jnp.where(causal_mask[None, None, :, None, :], A_c, 0.0)
    dv_intra = jnp.einsum("bnihj,bnihv->bnjhv", A_masked, do_c, precision=lax.Precision.HIGHEST)
    k_decay = k_c * jnp.exp(gn[:, :, None, :, :] - gc_c)
    dv_inter = jnp.einsum("bnchk,bnhkv->bnchv", k_decay, dh, precision=lax.Precision.HIGHEST)
    dv = (dv_intra + dv_inter).reshape(B, T, H, V)

    # dq intra + inter
    k_neg = k_c * jnp.exp(-gc_c)
    dq_intra = jnp.exp(gc_c) * jnp.einsum(
        "bnihj,bnjhk->bnihk", dA_c, k_neg, precision=lax.Precision.HIGHEST
    )
    dq_inter = (
        scale * jnp.exp(gc_c)
        * jnp.einsum("bnchv,bnhkv->bnchk", do_c, h, precision=lax.Precision.HIGHEST)
    )
    dq = (dq_intra + dq_inter).reshape(B, T, H, K)

    # dk intra + inter
    q_pos = q_c * jnp.exp(gc_c)
    dk_intra = jnp.exp(-gc_c) * jnp.einsum(
        "bnihj,bnihk->bnjhk", dA_c, q_pos, precision=lax.Precision.HIGHEST
    )
    dk_inter = jnp.exp(gn[:, :, None, :, :] - gc_c) * jnp.einsum(
        "bnchv,bnhkv->bnchk", v_c, dh, precision=lax.Precision.HIGHEST
    )
    dk = (dk_intra + dk_inter).reshape(B, T, H, K)

    return dq, dk, dv


def chunk_gla_ref(q, k, v, g_gamma, scale, chunk_size):
    """Chunked GLA with custom_vjp (pure JAX reference)."""
    @jax.custom_vjp
    def _compute(q, k, v):
        g_cumsum, A, h, o = chunk_gla_fwd_ref(q, k, v, g_gamma, scale, chunk_size)
        return o

    def _fwd(q, k, v):
        g_cumsum, A, h, o = chunk_gla_fwd_ref(q, k, v, g_gamma, scale, chunk_size)
        return o, (q, k, v, g_cumsum, h, A)

    def _bwd(residuals, do):
        q, k, v, g_cumsum, h, A = residuals
        B, T, H, K = q.shape
        V = v.shape[-1]
        C = chunk_size
        NT = T // C
        dh = _chunk_bwd_dh_scan(q, do, g_gamma, scale, C, B, T, H, K, V, NT)
        dq, dk, dv = _chunk_gla_bwd_ref(q, k, v, g_cumsum, h, A, do, dh, scale, C)
        return dq, dk, dv

    _compute.defvjp(_fwd, _bwd)
    return _compute(q, k, v)


def simple_compute(B=2, T=4096, H=16, K=128, V=128, chunk_size=64):
    """Forward + backward chunked GLA (pure JAX reference).

    Returns the loss scalar. Both forward and backward are computed,
    so timing captures the full training step performance.
    """
    q, k_arr, v, g_gamma = _make_test_data(B, T, H, K, V, chunk_size)
    scale = K ** -0.5

    def loss_fn(q, k, v):
        return chunk_gla_ref(q.astype(jnp.float32), k.astype(jnp.float32),
                            v.astype(jnp.float32), g_gamma, scale, chunk_size).sum()

    loss, _ = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k_arr, v)
    return loss


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
