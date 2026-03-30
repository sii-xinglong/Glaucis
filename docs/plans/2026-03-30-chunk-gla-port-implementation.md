# Chunked GLA Kernel Port Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port the chunked GLA (Gated Linear Attention) forward+backward Pallas kernels from primatrix/pallas-kernel into kernel-evolve's evolvable format, targeting AL model dimensions (B=2, T=4096, H=16, K=128, V=128, chunk_size=64) with g_gamma-only gate mode.

**Architecture:** Single-file monolith for both template and reference. Template contains all Pallas kernels + orchestration in one EVOLVE-BLOCK. Reference uses pure JAX (lax.scan + einsum). Both wrap fwd+bwd with custom_vjp. Entry points return a loss scalar via value_and_grad.

**Tech Stack:** JAX, jax.experimental.pallas, jax.experimental.pallas.tpu, bfloat16 inputs with float32 accumulators.

**Source:** `primatrix/pallas-kernel` — `tops/ops/gla/chunk.py`, `tops/ops/common/chunk_h.py`

---

### Task 1: Create Reference Kernel (Pure JAX)

**Files:**
- Create: `kernel-evolve/examples/kernels/chunk_gla_ref.py`

**Step 1: Write the reference kernel file**

This is the pure JAX reference implementation. No Pallas kernels. All functions use lax.scan or einsum. g_gamma-only path.

Key functions:
- `_make_test_data(B, T, H, K, V, chunk_size, seed=42)` — shared input generation
- `_chunk_fwd_h_scan(k, v, g_gamma, h0, ...)` — lax.scan-based inter-chunk state propagation
- `_chunk_gla_fwd_intra(q, k, g_cumsum, scale, chunk_size)` — intra-chunk attention matrix via einsum
- `_chunk_gla_fwd_o(q, v, g_cumsum, A, h, scale, chunk_size)` — output combination
- `chunk_gla_fwd_ref(q, k, v, g_gamma, scale, chunk_size)` — forward orchestrator
- `_chunk_bwd_dh_scan(q, do, g_gamma, scale, chunk_size)` — backward state grads via reverse lax.scan
- `_chunk_gla_bwd_ref(q, k, v, g_cumsum, h, A, do, dh, scale, chunk_size)` — backward orchestrator
- `chunk_gla_ref(q, k, v, g_gamma, scale, chunk_size)` — custom_vjp wrapper
- `simple_compute(B, T, H, K, V, chunk_size)` — entry point

The exact code is ported from `chunk.py` + `chunk_h.py` from the source repo, stripped to g_gamma-only path, with all imports from `tops.*` replaced by inline implementations.

Critical implementation details from the source:
- `g_cumsum = g_gamma * pos` where `pos = tile([1,2,...,C], T//C)` — analytical cumsum for constant gate
- Forward state update per chunk: `h = h * exp(g_gamma*C) + (k*exp(g_gamma*C - g_gamma*[1..C]))^T @ v`
- Backward dh propagation (reverse scan): `dh = dh * exp(g_gamma*C) + (q*exp(g_gamma*[1..C])*scale)^T @ do`
- Backward fused: computes dA, dv, dq, dk, dg per chunk, then sum-reduces dg → g_gamma shape
- `simple_compute` returns loss scalar from `jax.value_and_grad(loss_fn)(q, k, v)` where `loss_fn = chunk_gla_ref(...).sum()`

```python
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
    # Negative g_gamma for decay (like logsigmoid output)
    g_gamma = -jnp.abs(jax.random.normal(k4, (H,), dtype=jnp.float32)) * 0.1
    return q, k_arr, v, g_gamma


def _compute_g_cumsum(g_gamma, T, chunk_size):
    """Analytical chunk-local cumsum for constant g_gamma."""
    C = chunk_size
    pos = jnp.arange(1, C + 1, dtype=jnp.float32)
    pos = jnp.tile(pos, T // C)  # [T]
    # g_cumsum[t, h] = g_gamma[h] * ((t % C) + 1)
    return pos[:, None] * g_gamma[None, :]  # [T, H] -> broadcast to [B, T, H, K]


def _chunk_fwd_h_scan(k, v, g_gamma, h0, output_final_state, C, B, T, H, K, V, NT):
    """lax.scan-based forward state propagation."""
    k_scan = k.reshape(B, NT, C, H, K).transpose(1, 0, 2, 3, 4)
    v_scan = v.reshape(B, NT, C, H, V).transpose(1, 0, 2, 3, 4)

    g_gamma_f32 = g_gamma.astype(jnp.float32)
    state_decay = jnp.exp(g_gamma_f32 * C)  # [H]
    b_g = g_gamma_f32[None, :] * (jnp.arange(C, dtype=jnp.float32) + 1)[:, None]  # [C, H]
    v_decay = jnp.exp(g_gamma_f32[None, :] * C - b_g)  # [C, H]

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
    h_all = h_all.transpose(1, 0, 2, 3, 4)  # [B, NT, H, K, V]

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
    NT = T // C

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

    # Analytical g_cumsum for g_gamma
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
    """Backward: compute dq, dk, dv, dg from saved intermediates."""
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
    gn = gc_c[:, :, -1, :, :]  # [B, NT, H, K]

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

    # dq intra
    k_neg = k_c * jnp.exp(-gc_c)
    dq_intra = jnp.exp(gc_c) * jnp.einsum(
        "bnihj,bnjhk->bnihk", dA_c, k_neg, precision=lax.Precision.HIGHEST
    )
    # dq inter
    dq_inter = (
        scale * jnp.exp(gc_c)
        * jnp.einsum("bnchv,bnhkv->bnchk", do_c, h, precision=lax.Precision.HIGHEST)
    )
    dq = (dq_intra + dq_inter).reshape(B, T, H, K)

    # dk intra
    q_pos = q_c * jnp.exp(gc_c)
    dk_intra = jnp.exp(-gc_c) * jnp.einsum(
        "bnihj,bnihk->bnjhk", dA_c, q_pos, precision=lax.Precision.HIGHEST
    )
    # dk inter
    dk_inter = jnp.exp(gn[:, :, None, :, :] - gc_c) * jnp.einsum(
        "bnchv,bnhkv->bnchk", v_c, dh, precision=lax.Precision.HIGHEST
    )
    dk = (dk_intra + dk_inter).reshape(B, T, H, K)

    # dg: gate gradient
    dq_total = (dq_intra + dq_inter)
    dk_total = (dk_intra + dk_inter)
    dgk_inter = jnp.exp(gn) * jnp.einsum(
        "bnhkv,bnhkv->bnhk", h, dh, precision=lax.Precision.HIGHEST
    ) + jnp.sum(dk_inter * k_c, axis=2)
    dg_raw = q_c * dq_total - k_c * dk_total
    dg = (
        jnp.cumsum(dg_raw[:, :, ::-1, :, :], axis=2)[:, :, ::-1, :, :]
        + dgk_inter[:, :, None, :, :]
    ).reshape(B, T, H, K)

    # Sum-reduce dg to g_gamma shape [H]
    dg_gamma = dg.sum(axis=(0, 1, 3))  # sum over B, T, K -> [H]

    return dq, dk, dv, dg_gamma


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
        dq, dk, dv, _ = _chunk_gla_bwd_ref(q, k, v, g_cumsum, h, A, do, dh, scale, C)
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
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('kernel-evolve/examples/kernels/chunk_gla_ref.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add kernel-evolve/examples/kernels/chunk_gla_ref.py
git commit -m "feat(kernel-evolve): add chunked GLA reference kernel (pure JAX)"
```

---

### Task 2: Create Template Kernel (Pallas TPU)

**Files:**
- Create: `kernel-evolve/examples/kernels/chunk_gla.py`

**Step 1: Write the template kernel file**

This is the Pallas TPU kernel template. Same `_make_test_data` as the reference. All Pallas kernels and orchestration inside EVOLVE-BLOCK. g_gamma-only path.

The EVOLVE-BLOCK contains three Pallas kernels (fwd_h, fwd_intra_gk, fwd_o_gk) plus the fused backward kernel (chunk_gla_bwd_fused), the backward dh kernel, the forward/backward orchestrators, custom_vjp wrapper, and optimized_compute.

Key Pallas kernels (ported from source with g_gamma-only path):

1. `_chunk_fwd_h_kernel` — inter-chunk state propagation
   - Grid: `(B, H, K//128, V//128, NT)` — last dim is sequential
   - Uses scratch VMEM `[BK, BV]` as accumulator
   - At each chunk: store h → decay by `exp(g_gamma*C)` → update with `k^T @ v`

2. `_chunk_gla_fwd_intra_gk_pl` — intra-chunk attention matrix
   - Grid: `(H, B*NT)` — fully parallel
   - Computes `A = (q * exp(g)) @ (k * exp(-g))^T * scale`

3. `_chunk_gla_fwd_o_gk_pl` — output combination
   - Grid: `(H, B*NT)` — fully parallel
   - Inter: `scale * (q * exp(g)) @ h`
   - Intra: `tril(A) @ v`

4. `_chunk_bwd_dh_scan` — backward state grads (lax.scan, not Pallas)
   - Uses reverse lax.scan for simplicity (evolvable to Pallas kernel)

5. `_chunk_gla_bwd_fused_kernel` — fused backward Pallas kernel
   - Grid: `(H, B*NT)` — fully parallel
   - Computes dA, dv, dq, dk, dg in one pass per chunk
   - Uses upper-triangular matmul for reverse cumsum of dg

```python
"""Chunked GLA (Gated Linear Attention) Pallas TPU kernel — template for evolutionary optimization.

Implements chunked GLA forward and backward passes using Pallas kernels
targeting TPU, with g_gamma (per-head constant gate) mode.

Optimization targets within the EVOLVE-BLOCK:
  - Kernel fusion strategies (merge phases)
  - Block sizes and tiling within kernels
  - Memory layout and transpose strategies
  - Loop structure and pipelining
  - Grid dimensions and BlockSpec configurations
  - Accumulator precision choices

AL model reference dimensions:
  q, k, v: [2, 4096, 16, 128]
  g_gamma:  (16,)
  chunk_size: 64
"""

import functools

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _make_test_data(B, T, H, K, V, chunk_size, seed=42):
    """Create deterministic (q, k, v, g_gamma) for a GLA test case."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    q = jax.random.normal(k1, (B, T, H, K), dtype=jnp.bfloat16)
    k_arr = jax.random.normal(k2, (B, T, H, K), dtype=jnp.bfloat16)
    v = jax.random.normal(k3, (B, T, H, V), dtype=jnp.bfloat16)
    g_gamma = -jnp.abs(jax.random.normal(k4, (H,), dtype=jnp.float32)) * 0.1
    return q, k_arr, v, g_gamma


def exp(x):
    """exp in float32."""
    return jnp.exp(x.astype(jnp.float32))


# EVOLVE-BLOCK-START
# ============================================================
# Forward: Inter-chunk state propagation (Pallas kernel)
# ============================================================


def _chunk_fwd_h_kernel(
    k_ref, v_ref, h0_ref, g_gamma,
    h_ref, ht_ref, scratch_ref,
    *, BT, NT,
):
    BK = k_ref.shape[3]
    BV = v_ref.shape[3]
    i_b, i_h, i_k, i_v, i_t = (
        pl.program_id(0), pl.program_id(1), pl.program_id(2),
        pl.program_id(3), pl.program_id(4),
    )

    b_g = g_gamma[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)

    @pl.when(i_t == 0)
    def init():
        if h0_ref is not None:
            scratch_ref[:, :] = h0_ref[0, 0].astype(jnp.float32)
        else:
            scratch_ref[:, :] = jnp.zeros((BK, BV), dtype=jnp.float32)

    h_ref[0, i_t, 0] = scratch_ref[...]

    k_tile = k_ref[0, 0]
    v_tile = v_ref[0, 0]

    b_g_last = g_gamma[i_h] * BT
    scratch_ref[...] *= exp(b_g_last)
    v_tile = (v_tile * exp(b_g_last - b_g)[:, None]).astype(v_tile.dtype)

    scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
        k_tile.astype(jnp.float32).T,
        v_tile.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    @pl.when(i_t == NT - 1)
    def end():
        if ht_ref is not None:
            ht_ref[0, 0] = scratch_ref[...]


def chunk_fwd_h(k, v, g_gamma, chunk_size):
    """Launch inter-chunk state propagation Pallas kernel."""
    BK, BV, BT = 128, 128, chunk_size
    B, T, H, K_dim = k.shape
    V = v.shape[-1]
    NT = T // BT

    k_t = jnp.transpose(k, (0, 2, 1, 3))  # (B, H, T, K)
    v_t = jnp.transpose(v, (0, 2, 1, 3))  # (B, H, T, V)

    grid = (B, H, pl.cdiv(K_dim, BK), pl.cdiv(V, BV), NT)

    def k_map(b, h, ki, vi, t): return b, h, t, ki
    def v_map(b, h, ki, vi, t): return b, h, t, vi
    def h_map(b, h, ki, vi, t): return b, 0, h, ki, vi
    def ht_map(b, h, ki, vi, t): return b, h, ki, vi

    h_all, ht = pl.pallas_call(
        functools.partial(_chunk_fwd_h_kernel, BT=BT, NT=NT),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, 1, BT, BK), k_map),
                pl.BlockSpec((1, 1, BT, BV), v_map),
                None,  # h0
                pl.BlockSpec(memory_space=pltpu.SMEM),  # g_gamma
            ],
            out_specs=[
                pl.BlockSpec((1, NT, 1, BK, BV), h_map),
                None,  # ht
            ],
            scratch_shapes=[pltpu.VMEM((BK, BV), jnp.float32)],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((B, NT, H, K_dim, V), k.dtype),
            None,
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary", "arbitrary", "arbitrary"),
            disable_bounds_checks=True,
        ),
    )(k_t, v_t, None, g_gamma)

    return h_all


# ============================================================
# Forward: Intra-chunk attention matrix (Pallas kernel)
# ============================================================


def _chunk_gla_fwd_intra_gk_pl(q_ref, k_ref, g_ref, A_ref, *, BT, scale):
    b_q = q_ref[0, 0]
    b_k = k_ref[0, 0]
    b_g = g_ref[0, 0].astype(jnp.float32)

    b_qg = (b_q * jnp.exp(b_g)).astype(b_q.dtype)
    b_kg = (b_k * jnp.exp(-b_g)).astype(b_k.dtype)

    b_A = (
        jnp.dot(b_qg, b_kg.T,
                precision=jax.lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32)
        * scale
    )
    A_ref[0, 0] = b_A.astype(A_ref.dtype)


def chunk_gla_fwd_intra_gk(q, k, g_cumsum, scale, chunk_size):
    """Launch intra-chunk attention Pallas kernel."""
    B, T, H, K = q.shape
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _k = k.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _g = g_cumsum.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)

    spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    A_spec = pl.BlockSpec([1, 1, BT, BT], index_map=lambda h, nt: (h, nt, 0, 0))

    A = pl.pallas_call(
        functools.partial(_chunk_gla_fwd_intra_gk_pl, BT=BT, scale=scale),
        grid=(H, total_NT),
        out_shape=jax.ShapeDtypeStruct([H, total_NT, BT, BT], jnp.float32),
        in_specs=[spec, spec, spec],
        out_specs=A_spec,
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
    )(_q, _k, _g)

    A = A.reshape(H, B, NT, BT, BT).transpose(1, 0, 2, 3, 4)
    A = A.reshape(B, H, NT * BT, BT).transpose(0, 2, 1, 3)
    return A


# ============================================================
# Forward: Output combination (Pallas kernel)
# ============================================================


def _chunk_gla_fwd_o_gk_pl(q_ref, v_ref, g_ref, h_ref, A_ref, o_ref, *, BT, scale):
    b_q = q_ref[0, 0]
    b_g = g_ref[0, 0]
    b_v = v_ref[0, 0]
    b_h = h_ref[0, 0]
    b_A = A_ref[0, 0]

    b_g_f32 = b_g.astype(jnp.float32)
    b_qg = (b_q * jnp.exp(b_g_f32)).astype(b_q.dtype)
    b_o = jnp.dot(b_qg, b_h.astype(b_qg.dtype),
                  precision=jax.lax.Precision.HIGHEST,
                  preferred_element_type=jnp.float32)
    b_o *= scale

    m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_A = jnp.where(m_s, b_A, 0.0).astype(b_A.dtype)
    b_o += jnp.dot(b_A, b_v,
                   precision=jax.lax.Precision.HIGHEST,
                   preferred_element_type=jnp.float32)
    o_ref[0, 0] = b_o.astype(o_ref.dtype)


def chunk_gla_fwd_o_gk(q, v, g_cumsum, A, h, scale, chunk_size):
    """Launch output combination Pallas kernel."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _g = g_cumsum.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _v = v.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _A = A.reshape(B, NT, BT, H, BT).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, BT)
    _h = h.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)

    q_spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    v_spec = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))
    g_spec = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    h_spec = pl.BlockSpec([1, 1, K, V], index_map=lambda h, nt: (h, nt, 0, 0))
    A_spec = pl.BlockSpec([1, 1, BT, BT], index_map=lambda h, nt: (h, nt, 0, 0))
    o_spec = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))

    o = pl.pallas_call(
        functools.partial(_chunk_gla_fwd_o_gk_pl, BT=BT, scale=scale),
        grid=(H, total_NT),
        out_shape=jax.ShapeDtypeStruct([H, total_NT, BT, V], v.dtype),
        in_specs=[q_spec, v_spec, g_spec, h_spec, A_spec],
        out_specs=o_spec,
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
    )(_q, _v, _g, _h, _A)

    o = o.reshape(H, B, NT, BT, V).transpose(1, 0, 2, 3, 4)
    o = o.reshape(B, H, NT * BT, V).transpose(0, 2, 1, 3)
    return o


# ============================================================
# Forward orchestrator
# ============================================================


def chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size):
    """Chunked GLA forward pass."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C

    pos = jnp.arange(1, C + 1, dtype=jnp.float32)
    pos = jnp.tile(pos, NT).reshape(1, T, 1, 1)
    g_cumsum = jnp.broadcast_to(g_gamma * pos, q.shape)

    h = chunk_fwd_h(k, v, g_gamma, C)
    A = chunk_gla_fwd_intra_gk(q, k, g_cumsum, scale, C)
    o = chunk_gla_fwd_o_gk(q, v, g_cumsum, A, h, scale, C)

    return g_cumsum, A, h, o


# ============================================================
# Backward: State gradient propagation (lax.scan)
# ============================================================


def _chunk_bwd_dh_scan(q, do, g_gamma, scale, C, B, T, H, K, V, NT):
    """Backward state gradient propagation via reverse lax.scan."""
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
    return dh_all.transpose(1, 0, 2, 3, 4)


# ============================================================
# Backward: Fused dq, dk, dv, dg (Pallas kernel)
# ============================================================


def _chunk_gla_bwd_fused_kernel(
    q_ref, k_ref, v_ref, g_ref, h_ref, a_ref, do_ref, dh_ref,
    dq_ref, dk_ref, dv_ref, dg_ref,
    *, BT, scale,
):
    b_q = q_ref[0, 0]
    b_k = k_ref[0, 0]
    b_v = v_ref[0, 0]
    b_g = g_ref[0, 0].astype(jnp.float32)
    b_h = h_ref[0, 0].astype(jnp.float32)
    b_a = a_ref[0, 0].astype(jnp.float32)
    b_do = do_ref[0, 0]
    b_dh = dh_ref[0, 0].astype(jnp.float32)

    b_gn = b_g[BT - 1, :]

    # dA
    b_dA = jnp.dot(b_do.astype(b_v.dtype), b_v.T,
                   precision=jax.lax.Precision.HIGHEST,
                   preferred_element_type=jnp.float32) * scale
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_dA = jnp.where(mask, b_dA, 0.0)

    # dv
    b_a_masked = jnp.where(mask, b_a, 0.0)
    b_dv_intra = jnp.dot(b_a_masked.T.astype(b_do.dtype), b_do,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)
    k_decay = (b_k * jnp.exp(b_gn[None, :] - b_g)).astype(b_k.dtype)
    b_dv_inter = jnp.dot(k_decay, b_dh.astype(b_k.dtype),
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32)
    dv_ref[0, 0] = (b_dv_intra + b_dv_inter).astype(dv_ref.dtype)

    # dq
    k_neg = (b_k * jnp.exp(-b_g)).astype(b_k.dtype)
    b_dq_intra = jnp.dot(b_dA.astype(k_neg.dtype), k_neg,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32) * jnp.exp(b_g)
    b_dq_inter = jnp.dot(b_do, b_h.astype(b_do.dtype).T,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32) * (scale * jnp.exp(b_g))
    b_dq = b_dq_intra + b_dq_inter
    dq_ref[0, 0] = b_dq.astype(dq_ref.dtype)

    # dk
    q_pos = (b_q * jnp.exp(b_g)).astype(b_q.dtype)
    b_dk_intra = jnp.dot(b_dA.T.astype(q_pos.dtype), q_pos,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32) * jnp.exp(-b_g)
    b_dk_inter = jnp.dot(b_v, b_dh.astype(b_v.dtype).T,
                         precision=jax.lax.Precision.HIGHEST,
                         preferred_element_type=jnp.float32) * jnp.exp(b_gn[None, :] - b_g)
    b_dk = b_dk_intra + b_dk_inter
    dk_ref[0, 0] = b_dk.astype(dk_ref.dtype)

    # dg
    dgk_inter = (jnp.exp(b_gn) * jnp.sum(b_h * b_dh, axis=1)
                 + jnp.sum(b_dk_inter * b_k.astype(jnp.float32), axis=0))
    dg_raw = b_q.astype(jnp.float32) * b_dq - b_k.astype(jnp.float32) * b_dk
    mask_upper = jnp.arange(BT)[None, :] >= jnp.arange(BT)[:, None]
    M_upper = jnp.where(mask_upper, 1.0, 0.0).astype(jnp.float32)
    dg_rev_cumsum = jnp.dot(M_upper, dg_raw,
                           precision=jax.lax.Precision.HIGHEST,
                           preferred_element_type=jnp.float32)
    dg_ref[0, 0] = (dg_rev_cumsum + dgk_inter[None, :]).astype(dg_ref.dtype)


def chunk_gla_bwd_fused(q, k, v, g_cumsum, h, do, dh, scale, chunk_size):
    """Launch fused backward Pallas kernel."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    A = chunk_gla_fwd_intra_gk(q, k, g_cumsum, scale, BT)

    _q = q.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _k = k.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _v = v.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _g = g_cumsum.reshape(B, NT, BT, H, K).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, K)
    _do = do.reshape(B, NT, BT, H, V).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, V)
    _h = h.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)
    _dh = dh.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)
    _A = A.reshape(B, NT, BT, H, BT).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, BT)

    grid = (H, total_NT)
    spec_K = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_V = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_h = pl.BlockSpec([1, 1, K, V], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_A = pl.BlockSpec([1, 1, BT, BT], index_map=lambda h, nt: (h, nt, 0, 0))

    dq, dk, dv, dg = pl.pallas_call(
        functools.partial(_chunk_gla_bwd_fused_kernel, BT=BT, scale=scale),
        grid=grid,
        out_shape=[
            jax.ShapeDtypeStruct([H, total_NT, BT, K], q.dtype),
            jax.ShapeDtypeStruct([H, total_NT, BT, K], k.dtype),
            jax.ShapeDtypeStruct([H, total_NT, BT, V], v.dtype),
            jax.ShapeDtypeStruct([H, total_NT, BT, K], g_cumsum.dtype),
        ],
        in_specs=[spec_K, spec_K, spec_V, spec_K, spec_h, spec_A, spec_V, spec_h],
        out_specs=[spec_K, spec_K, spec_V, spec_K],
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
    )(_q, _k, _v, _g, _h, _A, _do, _dh)

    def _unreshape(x, last_dim):
        x = x.reshape(H, B, NT, BT, last_dim)
        x = x.transpose(1, 0, 2, 3, 4)
        x = x.reshape(B, H, T, last_dim)
        return x.transpose(0, 2, 1, 3)

    return _unreshape(dq, K), _unreshape(dk, K), _unreshape(dv, V), _unreshape(dg, K)


# ============================================================
# custom_vjp wrapper
# ============================================================


def chunk_gla(q, k, v, g_gamma, scale, chunk_size):
    """Chunked GLA with custom_vjp (Pallas TPU kernels)."""
    @jax.custom_vjp
    def _compute(q, k, v):
        _, _, _, o = chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size)
        return o

    def _fwd(q, k, v):
        g_cumsum, A, h, o = chunk_gla_fwd(q, k, v, g_gamma, scale, chunk_size)
        return o, (q, k, v, g_cumsum, h, A)

    def _bwd(residuals, do):
        q, k, v, g_cumsum, h, A = residuals
        B, T, H, K = q.shape
        V = v.shape[-1]
        C = chunk_size
        NT = T // C
        dh = _chunk_bwd_dh_scan(q, do, g_gamma, scale, C, B, T, H, K, V, NT)
        dq, dk, dv, _ = chunk_gla_bwd_fused(q, k, v, g_cumsum, h, do, dh, scale, C)
        return dq, dk, dv

    _compute.defvjp(_fwd, _bwd)
    return _compute(q, k, v)


# ============================================================
# Entry point
# ============================================================


def optimized_compute(B=2, T=4096, H=16, K=128, V=128, chunk_size=64):
    """Forward + backward chunked GLA with Pallas TPU kernels.

    Returns the loss scalar. Both forward and backward are computed,
    so timing captures the full training step performance.
    """
    q, k_arr, v, g_gamma = _make_test_data(B, T, H, K, V, chunk_size)
    scale = K ** -0.5

    def loss_fn(q, k, v):
        return chunk_gla(q.astype(jnp.float32), k.astype(jnp.float32),
                        v.astype(jnp.float32), g_gamma, scale, chunk_size).sum()

    loss, _ = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k_arr, v)
    return loss
# EVOLVE-BLOCK-END
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('kernel-evolve/examples/kernels/chunk_gla.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Verify EVOLVE-BLOCK markers and extract**

Run: `python -c "from kernel_evolve.mutation import extract_evolve_block; code = open('kernel-evolve/examples/kernels/chunk_gla.py').read(); block = extract_evolve_block(code); print(f'Block: {len(block)} chars, {len(block.splitlines())} lines')"`
Expected: Shows block size (should be ~400+ lines)

**Step 4: Commit**

```bash
git add kernel-evolve/examples/kernels/chunk_gla.py
git commit -m "feat(kernel-evolve): add chunked GLA Pallas TPU kernel template"
```

---

### Task 3: Create YAML Config

**Files:**
- Create: `kernel-evolve/examples/chunk_gla.yaml`

**Step 1: Write the config**

```yaml
kernel:
  name: "chunk_gla"
  template: "kernels/chunk_gla.py"
  reference: "kernels/chunk_gla_ref.py"
  evolve_markers:
    start: "# EVOLVE-BLOCK-START"
    end: "# EVOLVE-BLOCK-END"

# AL model GLA dimensions:
#   q, k, v: [B, T, H, K] = [2, 4096, 16, 128]
#   g_gamma:  (16,)
#   chunk_size: 64
shapes:
  - { B: 2, T: 4096, H: 16, K: 128, V: 128, chunk_size: 64 }

correctness:
  method: "allclose"
  rtol: 1e-2
  atol: 1e-2

evaluator:
  namespace: "default"
  job_template: ".github/ci/kernel-eval-job.yaml"
  repo: "sii-xinglong/Glaucis"
  branch: "main"
  poll_interval: 15
  timeout: 600

tpu:
  cluster: "tpu7x-cluster"
  zone: "us-central1"

session:
  max_iterations: 20
  output_dir: "runs/chunk_gla"
```

**Step 2: Verify config parses**

Run: `cd kernel-evolve && python -c "from kernel_evolve.config import EvolveConfig; c = EvolveConfig.from_yaml('examples/chunk_gla.yaml'); print(f'Kernel: {c.kernel.name}, shapes: {c.shapes}')"`
Expected: `Kernel: chunk_gla, shapes: [{'B': 2, 'T': 4096, 'H': 16, 'K': 128, 'V': 128, 'chunk_size': 64}]`

**Step 3: Commit**

```bash
git add kernel-evolve/examples/chunk_gla.yaml
git commit -m "feat(kernel-evolve): add chunked GLA eval config"
```

---

### Task 4: Create Tests

**Files:**
- Create: `kernel-evolve/tests/test_chunk_gla.py`

**Step 1: Write the tests**

Tests verify:
1. Syntax validity of both files
2. EVOLVE-BLOCK markers present and extractable
3. `optimized_compute` and `simple_compute` functions exist
4. `_make_test_data` produces matching inputs
5. YAML config loads correctly
6. Reference kernel forward-only runs on CPU (small dimensions)

```python
"""Tests for chunked GLA kernel template and reference."""
import ast
import sys
import pytest

sys.path.insert(0, "src")
from kernel_evolve.mutation import extract_evolve_block
from kernel_evolve.config import EvolveConfig


TEMPLATE = "examples/kernels/chunk_gla.py"
REFERENCE = "examples/kernels/chunk_gla_ref.py"
CONFIG = "examples/chunk_gla.yaml"


def test_template_syntax():
    code = open(TEMPLATE).read()
    ast.parse(code)


def test_reference_syntax():
    code = open(REFERENCE).read()
    ast.parse(code)


def test_evolve_block_extractable():
    code = open(TEMPLATE).read()
    block = extract_evolve_block(code)
    assert len(block) > 100, f"Block too small: {len(block)} chars"
    assert "optimized_compute" in block


def test_template_has_optimized_compute():
    ns = {}
    exec(open(TEMPLATE).read(), ns)
    assert "optimized_compute" in ns
    assert callable(ns["optimized_compute"])


def test_reference_has_simple_compute():
    ns = {}
    exec(open(REFERENCE).read(), ns)
    assert "simple_compute" in ns
    assert callable(ns["simple_compute"])


def test_reference_has_reference_fn():
    ns = {}
    exec(open(REFERENCE).read(), ns)
    assert "reference_fn" in ns
    assert callable(ns["reference_fn"])


def test_matching_test_data():
    """Both kernels produce identical test data from the same seed."""
    tmpl_ns = {}
    ref_ns = {}
    exec(open(TEMPLATE).read(), tmpl_ns)
    exec(open(REFERENCE).read(), ref_ns)

    import jax.numpy as jnp

    t_q, t_k, t_v, t_g = tmpl_ns["_make_test_data"](2, 256, 4, 128, 128, 64)
    r_q, r_k, r_v, r_g = ref_ns["_make_test_data"](2, 256, 4, 128, 128, 64)

    assert jnp.allclose(t_q, r_q)
    assert jnp.allclose(t_k, r_k)
    assert jnp.allclose(t_v, r_v)
    assert jnp.allclose(t_g, r_g)


def test_config_loads():
    config = EvolveConfig.from_yaml(CONFIG)
    assert config.kernel.name == "chunk_gla"
    assert len(config.shapes) == 1
    assert config.shapes[0]["B"] == 2
    assert config.shapes[0]["T"] == 4096


def test_reference_forward_small():
    """Reference kernel produces output on CPU (small dims)."""
    ref_ns = {}
    exec(open(REFERENCE).read(), ref_ns)

    import jax.numpy as jnp

    q, k, v, g_gamma = ref_ns["_make_test_data"](1, 64, 2, 128, 128, 16)
    scale = 128 ** -0.5
    o = ref_ns["chunk_gla_ref"](
        q.astype(jnp.float32), k.astype(jnp.float32),
        v.astype(jnp.float32), g_gamma, scale, 16
    )
    assert o.shape == (1, 64, 2, 128)
    assert jnp.isfinite(o).all()
```

**Step 2: Run tests**

Run: `cd kernel-evolve && python -m pytest tests/test_chunk_gla.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add kernel-evolve/tests/test_chunk_gla.py
git commit -m "test(kernel-evolve): add chunked GLA kernel tests"
```

---

### Task 5: Create Standalone Integration Test

**Files:**
- Create: `kernel-evolve/tests/standalone_chunk_gla_test.py`

**Step 1: Write the standalone test**

This test runs on TPU and verifies end-to-end correctness (template vs reference) and basic performance. Mirrors `standalone_gmm_fp8_blockwise_test.py`.

```python
"""Standalone integration test for chunked GLA kernel.

Run on TPU:
    python tests/standalone_chunk_gla_test.py

Verifies:
  1. Both template and reference produce finite results
  2. Template matches reference within atol=1e-2
  3. Measures forward+backward latency
"""
import time
import sys
import numpy as np

import jax
import jax.numpy as jnp


def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"Platform: {jax.default_backend()}")

    # Load kernels
    tmpl_ns = {}
    ref_ns = {}
    exec(open("examples/kernels/chunk_gla.py").read(), tmpl_ns)
    exec(open("examples/kernels/chunk_gla_ref.py").read(), ref_ns)

    shapes = {"B": 2, "T": 4096, "H": 16, "K": 128, "V": 128, "chunk_size": 64}
    print(f"\nShapes: {shapes}")

    # Correctness
    print("\n--- Correctness ---")
    tmpl_out = tmpl_ns["optimized_compute"](**shapes)
    ref_out = ref_ns["simple_compute"](**shapes)
    jax.block_until_ready(tmpl_out)
    jax.block_until_ready(ref_out)

    max_diff = float(np.max(np.abs(np.array(tmpl_out) - np.array(ref_out))))
    print(f"Template output: {float(tmpl_out):.6f}")
    print(f"Reference output: {float(ref_out):.6f}")
    print(f"Max diff: {max_diff:.6e}")

    if max_diff > 1e-2:
        print(f"FAIL: max_diff {max_diff} > atol 1e-2")
        sys.exit(1)
    print("PASS: correctness within tolerance")

    # Performance
    print("\n--- Performance ---")
    warmup = 10
    iters = 50

    for name, fn in [("template", tmpl_ns["optimized_compute"]),
                     ("reference", ref_ns["simple_compute"])]:
        for _ in range(warmup):
            out = fn(**shapes)
            jax.block_until_ready(out)

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = fn(**shapes)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)

        median_ms = np.median(times) * 1000
        print(f"{name}: {median_ms:.2f} ms (median of {iters})")

    print("\nDone.")


if __name__ == "__main__":
    main()
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('kernel-evolve/tests/standalone_chunk_gla_test.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add kernel-evolve/tests/standalone_chunk_gla_test.py
git commit -m "test(kernel-evolve): add standalone chunked GLA TPU integration test"
```

---

### Task 6: Final Verification and Commit

**Step 1: Run all existing tests to verify no regressions**

Run: `cd kernel-evolve && python -m pytest tests/ -v --ignore=tests/standalone_gmm_fp8_blockwise_test.py --ignore=tests/standalone_chunk_gla_test.py`
Expected: All existing tests pass

**Step 2: Verify the mutation system works with the new kernel**

Run: `cd kernel-evolve && python -c "
from kernel_evolve.config import EvolveConfig
from kernel_evolve.mutation import extract_evolve_block, inject_evolve_block, validate_syntax

config = EvolveConfig.from_yaml('examples/chunk_gla.yaml')
code = open('examples/' + config.kernel.template).read()
block = extract_evolve_block(code)
print(f'Extracted block: {len(block.splitlines())} lines')

# Verify round-trip: extract -> inject -> validate
new_code = inject_evolve_block(code, block)
assert validate_syntax(new_code), 'Round-trip syntax check failed'
print('Round-trip inject: OK')

# Verify new code matches original
assert new_code.strip() == code.strip(), 'Round-trip content mismatch'
print('Round-trip content: OK')
print('All checks passed!')
"`
Expected: All checks pass

**Step 3: Verify files exist in the right places**

Run: `ls -la kernel-evolve/examples/kernels/chunk_gla*.py kernel-evolve/examples/chunk_gla.yaml kernel-evolve/tests/*chunk_gla*`
Expected: All 4 files listed

**Step 4: Final commit (if any uncommitted changes)**

```bash
git status
# Only if there are changes:
git add -A kernel-evolve/
git commit -m "chore(kernel-evolve): finalize chunked GLA kernel port"
```
