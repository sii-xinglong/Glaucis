# Source: primatrix/pallas-kernel @ branch: main
# Commit: cbeeae7953583f8b5a406a0a76d233f8df569352
# Initialized: 2026-04-01
"""fused_chunk_simple_gla Pallas TPU kernel — template for evolutionary optimization.

Fused chunk forward (h + o in single kernel) + fused backward (dh + dq/dk/dv
in single kernel) for Simple GLA with per-head g_gamma decay.

Compared to the reference, this template removes h recomputation from backward:
forward pre-computes h via chunk_fwd_h and saves it as a custom_vjp residual.
This trades memory for compute in the backward pass.

Optimization targets within the EVOLVE-BLOCK:
  - Kernel fusion strategies (merge fwd h + o, bwd dh + dq/dk/dv)
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

# --- All imports (deduplicated from all upstream files) ---
import os
import functools

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _make_test_data(B, T, H, K, V, chunk_size, seed=42):
    """Create deterministic test data for the fused_chunk_simple_gla kernel."""
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


_IS_TPU_RUNTIME_CACHED: bool | None = None

def is_tpu_runtime() -> bool:
    global _IS_TPU_RUNTIME_CACHED
    if _IS_TPU_RUNTIME_CACHED is not None:
        return _IS_TPU_RUNTIME_CACHED
    try:
        devs = jax.devices()
        _IS_TPU_RUNTIME_CACHED = len(devs) > 0 and all(
            d.platform == "tpu" for d in devs
        )
    except Exception:
        _IS_TPU_RUNTIME_CACHED = jax.default_backend() == "tpu"
    return _IS_TPU_RUNTIME_CACHED


def get_interpret() -> bool:
    env = os.environ.get("PALLAS_INTERPRET", "")
    return env.strip().lower() in ("1", "true")


def assert_shape_or_none(x, expected_shape, name="tensor"):
    if x is None:
        return
    if isinstance(x, (list, tuple)):
        has_names = isinstance(name, (list, tuple)) and len(name) == len(x)
        for i, tensor in enumerate(x):
            if tensor is not None:
                curr_name = name[i] if has_names else f"{name}_{i}"
                assert tensor.shape == expected_shape, f"[{curr_name}] Expected shape {expected_shape}, got {tensor.shape}"
    else:
        assert x.shape == expected_shape, f"[{name}] Expected shape {expected_shape}, got {x.shape}"


def assert_shape(x, expected_shape, name="tensor"):
    if isinstance(x, (list, tuple)):
        has_names = isinstance(name, (list, tuple)) and len(name) == len(x)
        for i, tensor in enumerate(x):
            curr_name = name[i] if has_names else f"{name}_{i}"
            assert tensor.shape == expected_shape, f"[{curr_name}] Expected shape {expected_shape}, got {tensor.shape}"
    else:
        assert x.shape == expected_shape, f"[{name}] Expected shape {expected_shape}, got {x.shape}"


# EVOLVE-BLOCK-START

# =============================================================================
# Forward: Fused chunk forward kernel (h + o in single Pallas kernel)
# =============================================================================


# From: tops/ops/common/fused_chunk.py
def _fused_chunk_fwd_kernel(
    q_ref,          # (1, 1, BT, BK)  — K-tiled
    k_ref,          # (1, 1, BT, BK)  — K-tiled
    v_ref,          # (1, 1, BT, BV)  — V-tiled
    h0_ref,         # (1, 1, BK, BV)  or None
    g_ref,          # (1, 1, BT, 128) or None
    g_gamma_ref,    # [H] via SMEM/ANY, or None
    o_ref,          # (1, 1, 1, BT, BV)  — partial output per K tile
    ht_ref,         # (1, 1, BK, BV)   — output, or None
    scratch_ref,    # (BK, BV) VMEM float32
    *,
    BT: int,
    NT: int,
):
    BK = q_ref.shape[3]
    BV = v_ref.shape[3]
    i_t = pl.program_id(4)

    if g_gamma_ref is not None:
        head_idx = pl.program_id(1)
        b_gamma = g_gamma_ref[head_idx].astype(jnp.float32)
        b_g_gamma = b_gamma * (jnp.arange(BT) + 1).astype(jnp.float32)

    @pl.when(i_t == 0)
    def _init():
        if h0_ref is not None:
            scratch_ref[:, :] = h0_ref[0, 0].astype(jnp.float32)
        else:
            scratch_ref[:, :] = jnp.zeros((BK, BV), dtype=jnp.float32)

    b_q = q_ref[0, 0]
    b_k = k_ref[0, 0]
    b_v = v_ref[0, 0]
    b_h = scratch_ref[...]

    partial_o = jnp.dot(b_q, b_h, preferred_element_type=jnp.float32)
    partial_A = jnp.dot(b_q, b_k.T, preferred_element_type=jnp.float32)

    if g_ref is not None:
        b_g = g_ref[0, 0, :, 0].astype(jnp.float32)
        partial_o = partial_o * exp(b_g)[:, None]
        g_diff = b_g[:, None] - b_g[None, :]
        fwd_mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
        safe_g_diff = jnp.where(fwd_mask, g_diff, 0.0)
        partial_A = partial_A * exp(safe_g_diff)

    if g_gamma_ref is not None:
        partial_o = partial_o * exp(b_g_gamma)[:, None]
        g_gamma_diff = b_g_gamma[:, None] - b_g_gamma[None, :]
        fwd_mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
        safe_g_gamma_diff = jnp.where(fwd_mask, g_gamma_diff, 0.0)
        partial_A = partial_A * exp(safe_g_gamma_diff)

    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    partial_A = jnp.where(mask, partial_A, 0.0)

    partial = partial_o + jnp.dot(
        partial_A, b_v.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    o_ref[0, 0, 0] = partial.astype(o_ref.dtype)

    b_v_upd = b_v

    if g_ref is not None:
        b_g_last = b_g[BT - 1]
        scratch_ref[...] *= exp(b_g_last)
        b_v_upd = (b_v_upd * exp(b_g_last - b_g)[:, None]).astype(b_v_upd.dtype)

    if g_gamma_ref is not None:
        b_g_gamma_last = b_gamma * BT
        scratch_ref[...] *= exp(b_g_gamma_last)
        b_v_upd = (b_v_upd * exp(b_g_gamma_last - b_g_gamma)[:, None]).astype(b_v_upd.dtype)

    scratch_ref[...] = scratch_ref[...] + jnp.dot(
        b_k.astype(jnp.float32).T,
        b_v_upd.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    @pl.when(i_t == NT - 1)
    def _store_ht():
        if ht_ref is not None:
            ht_ref[0, 0] = scratch_ref[...].astype(ht_ref.dtype)


# From: tops/ops/common/fused_chunk.py
@functools.partial(
    jax.jit,
    static_argnames=("output_final_state", "chunk_size", "interpret"),
)
def fused_chunk_fwd_kernel(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    h0: jax.Array | None = None,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float,
    output_final_state: bool = False,
    chunk_size: int = 64,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    BK = 128
    BV = 128
    NK = K // BK
    NV = V // BV

    _q = q.transpose(0, 2, 1, 3)
    _k = k.transpose(0, 2, 1, 3)
    _v = v.transpose(0, 2, 1, 3)

    _g = None
    if g is not None:
        _g = g.transpose(0, 2, 1)
        _g = jnp.broadcast_to(_g[:, :, :, None], (B, H, T, 128))

    grid = (B, H, NK, NV, NT)

    def qk_map(b, h, ik, iv, t):
        return (b, h, t, ik)
    def v_map(b, h, ik, iv, t):
        return (b, h, t, iv)
    def state_map(b, h, ik, iv, t):
        return (b, h, ik, iv)
    def g_map(b, h, ik, iv, t):
        return (b, h, t, 0)
    def o_map(b, h, ik, iv, t):
        return (b, h, ik, t, iv)

    smem = pltpu.ANY if interpret else pltpu.SMEM

    spec_qk    = pl.BlockSpec((1, 1, BT, BK), qk_map)
    spec_v     = pl.BlockSpec((1, 1, BT, BV), v_map)
    spec_h0    = pl.BlockSpec((1, 1, BK, BV), state_map) if h0 is not None else None
    spec_g     = pl.BlockSpec((1, 1, BT, 128), g_map) if _g is not None else None
    spec_gamma = pl.BlockSpec(memory_space=smem) if g_gamma is not None else None

    spec_o     = pl.BlockSpec((1, 1, 1, BT, BV), o_map)
    spec_ht    = pl.BlockSpec((1, 1, BK, BV), state_map) if output_final_state else None

    out_shapes = [
        jax.ShapeDtypeStruct((B, H, NK, T, V), jnp.float32),
        jax.ShapeDtypeStruct((B, H, K, V), jnp.float32)
        if output_final_state else None,
    ]

    o_partial, ht = pl.pallas_call(
        functools.partial(_fused_chunk_fwd_kernel, BT=BT, NT=NT),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[spec_qk, spec_qk, spec_v, spec_h0,
                      spec_g, spec_gamma],
            out_specs=[spec_o, spec_ht],
            scratch_shapes=[pltpu.VMEM((BK, BV), jnp.float32)],
        ),
        out_shape=out_shapes,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel", "parallel", "parallel", "parallel", "arbitrary",
            ),
        ),
        interpret=interpret,
    )(_q, _k, _v, h0, _g, g_gamma)

    o = o_partial.sum(axis=2) * scale
    o = o.transpose(0, 2, 1, 3).astype(v.dtype)

    return o, ht


# From: tops/ops/common/fused_chunk.py
def fused_chunk_fwd(
    q, k, v, *, g=None, g_gamma=None, h0=None, scale=None,
    use_ht=False, chunk_size=64, interpret=None,
):
    B, T, H, K = q.shape
    V = v.shape[-1]

    if scale is None:
        scale = K ** -0.5

    assert_shape(q, (B, T, H, K))
    assert_shape(k, (B, T, H, K))
    assert_shape(v, (B, T, H, V))
    assert_shape_or_none(g, (B, T, H))
    assert_shape_or_none(g_gamma, (H,))
    assert_shape_or_none(h0, (B, H, K, V))
    assert K % 128 == 0, f"K={K} must be a multiple of 128"
    assert V % 128 == 0, f"V={V} must be a multiple of 128"
    assert T % chunk_size == 0
    assert scale is not None

    if interpret is None:
        interpret = not is_tpu_runtime()

    return fused_chunk_fwd_kernel(
        q=q, k=k, v=v, h0=h0, g=g, g_gamma=g_gamma,
        scale=scale, output_final_state=use_ht,
        chunk_size=chunk_size, interpret=interpret,
    )


# From: tops/ops/simple_gla/fused_chunk.py
def fused_chunk_simple_gla_fwd(
    q, k, v, *, g=None, g_gamma=None, h0=None,
    scale=None, use_ht=False, chunk_size=64, interpret=None,
):
    return fused_chunk_fwd(
        q, k, v, g=g, g_gamma=g_gamma, h0=h0,
        scale=scale, use_ht=use_ht, chunk_size=chunk_size, interpret=interpret,
    )


# =============================================================================
# State propagation: chunk_fwd_h (used in forward to pre-compute h)
# =============================================================================


# From: tops/ops/common/chunk_h.py
def _chunk_fwd_h_kernel(
    k_ref, v_ref, h0_ref, gk_ref, g_ref, g_gamma,
    h_ref, ht_ref, scratch_ref,
    *, BT, BS, NT,
):
    BK = k_ref.shape[3]
    BV = v_ref.shape[3]
    NTS = BS // BT
    T = NT * BT
    i_b, i_h, i_k, i_v, i_t = pl.program_id(0), pl.program_id(1), pl.program_id(2), pl.program_id(3), pl.program_id(4)

    if g_gamma is not None:
        b_g = g_gamma[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)

    @pl.when(i_t == 0)
    def init():
        if h0_ref is not None:
            scratch_ref[:,:] = h0_ref[0, 0].astype(jnp.float32)
        else:
            scratch_ref[:,:] = jnp.zeros((BK, BV), dtype=jnp.float32)

    @pl.when((i_t % NTS) == 0)
    def store_fn():
        i_s = i_t // NTS
        h_ref[0, i_s, 0] = scratch_ref[...].astype(h_ref.dtype)

    k_tile = k_ref[(0, 0, slice(None), slice(None))]
    v_tile = v_ref[(0, 0, slice(None), slice(None))]

    if g_ref is not None:
        b_g_scalar = g_ref[0, 0, slice(None), 0]
        b_g_scalar_last = b_g_scalar[BT - 1]
        scratch_ref[...] *= exp(b_g_scalar_last)
        v_tile = (v_tile * exp(b_g_scalar_last - b_g_scalar)[:, None]).astype(v_tile.dtype)

    if g_gamma is not None:
        b_g_last = (g_gamma[i_h].astype(jnp.float32) * jnp.minimum(BT, T - i_t * BT)).astype(g_gamma.dtype)
        scratch_ref[...] *= exp(b_g_last)
        v_tile = (v_tile * exp(b_g_last - b_g)[:, None]).astype(v_tile.dtype)

    if gk_ref is not None:
        gk_tile = gk_ref[(0, 0, slice(None), slice(None))]
        g_last = gk_tile[-1, :]
        decay = exp(g_last)
        scratch_ref[...] = scratch_ref[...] * decay[:, None]
        k_tile = (k_tile * exp(g_last[None, :] - gk_tile)).astype(k_tile.dtype)

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


# From: tops/ops/common/chunk_h.py
@functools.partial(
    jax.jit,
    static_argnames=[
        "output_final_state",
        "chunk_size",
        "split_size",
        "states_in_fp32",
    ],
)
def chunk_fwd_h(
    k, v, *, g=None, g_gamma=None, gk=None, gv=None,
    h0=None, output_final_state=False,
    cu_seqlens_cpu=None, cu_seqlens_dev=None,
    chunk_size=64, split_size=None, states_in_fp32=False,
):
    BK = 128
    BV = 128
    B, T, H, K_dim, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens_cpu is None else cu_seqlens_cpu.shape[-1] - 1
    BT = chunk_size
    BS = BT if split_size is None else split_size

    assert_shape(k, (B, T, H, K_dim))
    assert_shape(v, (B, T, H, V))
    assert_shape_or_none(g, (B, T, H))
    assert_shape_or_none(g_gamma, (H,))
    assert_shape_or_none(gk, (B, T, H, K_dim))
    assert gv is None, "gv is currently not supported"
    assert cu_seqlens_cpu is None, "cu_seqlens_cpu is currently not supported"
    assert cu_seqlens_dev is None, "cu_seqlens_dev is currently not supported"
    assert_shape_or_none(h0, (N, H, K_dim, V))
    assert K_dim % 128 == 0, "K % 128 must equal to 0."
    assert V % 128 == 0, "V % 128 must equal to 0."
    assert T % chunk_size == 0, "T mod chunk_size must equal to 0."
    assert BS % BT == 0

    N, NS = B, T // BS
    NT = T // BT

    k_t = jnp.transpose(k, (0, 2, 1, 3))
    v_t = jnp.transpose(v, (0, 2, 1, 3))
    if gk is not None:
        gk = jnp.transpose(gk, (0, 2, 1, 3))
    if g is not None:
        g = jnp.transpose(g, (0, 2, 1))
        g = jnp.broadcast_to(g[:, :, :, None], (B, H, T, 128))

    grid = (B, H, pl.cdiv(K_dim, BK), pl.cdiv(V, BV), NT)

    def k_index_map(batch_index, head_index, k_index, _, t_index):
        return batch_index, head_index, t_index, k_index
    def v_index_map(batch_index, head_index, _, v_index, t_index):
        return batch_index, head_index, t_index, v_index
    def h0_index_map(batch_index, head_index, k_index, v_index, _):
        return batch_index, head_index, k_index, v_index
    def h_index_map(batch_index, head_index, k_index, v_index, _):
        return batch_index, 0, head_index, k_index, v_index
    def ht_index_map(batch_index, head_index, k_index, v_index, _):
        return batch_index, head_index, k_index, v_index

    out_shape = [
        jax.ShapeDtypeStruct(
            shape=(N, NS, H, K_dim, V), dtype=k_t.dtype if not states_in_fp32 else jnp.float32
        )
    ]
    out_specs = [pl.BlockSpec((1, NS, 1, BK, BV), h_index_map)]
    if output_final_state:
        out_shape.append(jax.ShapeDtypeStruct(shape=(N, H, K_dim, V), dtype=jnp.float32))
        out_specs.append(pl.BlockSpec((1, 1, BK, BV), ht_index_map))
    else:
        out_shape.append(None)
        out_specs.append(None)

    in_specs = [
        pl.BlockSpec((1, 1, BT, BK), k_index_map),
        pl.BlockSpec((1, 1, BT, BV), v_index_map),
    ]
    scratch = pltpu.VMEM((BK, BV), jnp.float32)
    scratch_shapes = [scratch]
    in_specs.append(pl.BlockSpec((1, 1, BK, BV), h0_index_map) if h0 is not None else None)
    in_specs.append(pl.BlockSpec((1, 1, BT, BK), k_index_map) if gk is not None else None)
    in_specs.append(pl.BlockSpec((1, 1, BT, 128), lambda bi, hi, ki, _, ti: (bi, hi, ti, 0)) if g is not None else None)
    in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM) if g_gamma is not None else None)

    kernel = functools.partial(_chunk_fwd_h_kernel, BT=BT, BS=BS, NT=NT)
    interpret = get_interpret()
    h, ht = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes
        ),
        out_shape=out_shape,
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel", "parallel", "parallel", "parallel", "arbitrary",
            ),
            disable_bounds_checks=True,
        ),
    )(k_t, v_t, h0, gk, g, g_gamma)

    h = h.reshape(B, -1, H, K_dim, V)
    ht = ht.reshape(N, H, K_dim, V) if ht is not None else None

    if output_final_state:
        return h, ht
    return h, None


# =============================================================================
# Backward: Fused dh propagation + dq/dk/dv kernel
# =============================================================================


# From: tops/ops/simple_gla/fused_chunk.py
def _fused_chunk_bwd_kernel(
    q_ref, k_ref, v_ref, h_ref, do_ref, g_gamma_ref, dht_ref,
    dq_ref, dk_ref, dv_ref, dh0_ref,
    scratch_ref,
    *, BT: int, NT: int, scale: float,
):
    K = q_ref.shape[3]
    V = v_ref.shape[3]
    i_t = pl.program_id(2)
    head_idx = pl.program_id(1)

    @pl.when(i_t == 0)
    def _init():
        if dht_ref is not None:
            scratch_ref[:, :] = dht_ref[0, 0].astype(jnp.float32)
        else:
            scratch_ref[:, :] = jnp.zeros((K, V), dtype=jnp.float32)

    b_q = q_ref[0, 0]
    b_k = k_ref[0, 0]
    b_v = v_ref[0, 0]
    b_h = h_ref[0, 0, 0].astype(jnp.float32)
    b_do = do_ref[0, 0]
    b_dh = scratch_ref[...].astype(jnp.float32)

    b_gamma = g_gamma_ref[head_idx].astype(jnp.float32)
    b_g = b_gamma * (jnp.arange(BT) + 1).astype(jnp.float32)
    b_gn = b_g[BT - 1]

    pos = (jnp.arange(BT) + 1).astype(jnp.float32)
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    safe_diff = jnp.where(mask, b_gamma * (pos[:, None] - pos[None, :]), 0.0)
    decay = jnp.exp(safe_diff)
    b_a = jnp.dot(b_q, b_k.T, preferred_element_type=jnp.float32) * scale * decay

    b_dA = jnp.dot(b_do, b_v.T, preferred_element_type=jnp.float32) * scale
    b_dA = jnp.where(mask, b_dA, 0.0)

    b_a_masked = jnp.where(mask, b_a, 0.0).astype(b_do.dtype)
    b_dv_intra = jnp.dot(b_a_masked.T, b_do, preferred_element_type=jnp.float32)
    k_decay = (b_k * jnp.exp(b_gn - b_g)[:, None]).astype(b_k.dtype)
    b_dv_inter = jnp.dot(k_decay, b_dh.astype(b_k.dtype), preferred_element_type=jnp.float32)
    dv_ref[0, 0] = (b_dv_intra + b_dv_inter).astype(dv_ref.dtype)

    safe_g_diff = jnp.where(mask, b_g[:, None] - b_g[None, :], 0.0)
    b_dA_gated = b_dA * jnp.exp(safe_g_diff)

    b_dq_intra = jnp.dot(b_dA_gated.astype(b_k.dtype), b_k, preferred_element_type=jnp.float32)
    b_dq_inter = jnp.dot(b_do, b_h.astype(b_do.dtype).T, preferred_element_type=jnp.float32) * (scale * jnp.exp(b_g)[:, None])
    dq_ref[0, 0] = (b_dq_intra + b_dq_inter).astype(dq_ref.dtype)

    b_dk_intra = jnp.dot(b_dA_gated.T.astype(b_q.dtype), b_q, preferred_element_type=jnp.float32)
    b_dk_inter = jnp.dot(b_v, b_dh.astype(b_v.dtype).T, preferred_element_type=jnp.float32) * jnp.exp(b_gn - b_g)[:, None]
    dk_ref[0, 0] = (b_dk_intra + b_dk_inter).astype(dk_ref.dtype)

    b_dh = b_dh * jnp.exp(b_gn)
    b_q_hat = (b_q * (scale * jnp.exp(b_g)[:, None])).astype(jnp.float32)
    b_dh = b_dh + jnp.dot(
        b_q_hat.T, b_do.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    scratch_ref[...] = b_dh

    @pl.when(i_t == NT - 1)
    def _store_dh0():
        if dh0_ref is not None:
            dh0_ref[0, 0] = scratch_ref[...].astype(dh0_ref.dtype)


# From: tops/ops/simple_gla/fused_chunk.py
@functools.partial(
    jax.jit,
    static_argnames=("scale", "output_dh0", "chunk_size", "interpret"),
)
def _fused_chunk_bwd_launcher(
    q, k, v, h, do, g_gamma, dht, *,
    scale, output_dh0, chunk_size, interpret,
):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT

    _q = q.transpose(0, 2, 1, 3)
    _k = k.transpose(0, 2, 1, 3)
    _v = v.transpose(0, 2, 1, 3)
    _do = do.transpose(0, 2, 1, 3)
    _h = h.transpose(0, 2, 1, 3, 4)

    grid = (B, H, NT)

    def rev_qk_map(b, h, t):
        return (b, h, NT - 1 - t, 0)
    def rev_v_map(b, h, t):
        return (b, h, NT - 1 - t, 0)
    def rev_h_map(b, h, t):
        return (b, h, NT - 1 - t, 0, 0)
    def state_map(b, h, t):
        return (b, h, 0, 0)

    smem = pltpu.ANY if interpret else pltpu.SMEM

    in_specs = [
        pl.BlockSpec((1, 1, BT, K), rev_qk_map),
        pl.BlockSpec((1, 1, BT, K), rev_qk_map),
        pl.BlockSpec((1, 1, BT, V), rev_v_map),
        pl.BlockSpec((1, 1, 1, K, V), rev_h_map),
        pl.BlockSpec((1, 1, BT, V), rev_v_map),
        pl.BlockSpec(memory_space=smem),
        pl.BlockSpec((1, 1, K, V), state_map) if dht is not None else None,
    ]

    out_specs = [
        pl.BlockSpec((1, 1, BT, K), rev_qk_map),
        pl.BlockSpec((1, 1, BT, K), rev_qk_map),
        pl.BlockSpec((1, 1, BT, V), rev_v_map),
        pl.BlockSpec((1, 1, K, V), state_map) if output_dh0 else None,
    ]

    out_shapes = [
        jax.ShapeDtypeStruct((B, H, T, K), q.dtype),
        jax.ShapeDtypeStruct((B, H, T, K), k.dtype),
        jax.ShapeDtypeStruct((B, H, T, V), v.dtype),
        jax.ShapeDtypeStruct((B, H, K, V), jnp.float32) if output_dh0 else None,
    ]

    dq, dk, dv, dh0 = pl.pallas_call(
        functools.partial(_fused_chunk_bwd_kernel, BT=BT, NT=NT, scale=scale),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=[pltpu.VMEM((K, V), jnp.float32)],
        ),
        out_shape=out_shapes,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
        interpret=interpret,
    )(_q, _k, _v, _h, _do, g_gamma, dht)

    dq = dq.transpose(0, 2, 1, 3)
    dk = dk.transpose(0, 2, 1, 3)
    dv = dv.transpose(0, 2, 1, 3)

    return dq, dk, dv, dh0


# =============================================================================
# custom_vjp wrapper + entry point
# =============================================================================


def fused_chunk_simple_gla(q, k, v, g_gamma, scale, chunk_size):
    """Fused chunk Simple GLA with custom_vjp (Pallas TPU kernels).

    Forward uses fused kernel (h stays in VMEM). h is also pre-computed
    via chunk_fwd_h and saved as residual — backward uses saved h directly
    (no recomputation).
    """
    @jax.custom_vjp
    def _compute(q, k, v):
        o, _ = fused_chunk_simple_gla_fwd(
            q, k, v, g_gamma=g_gamma, scale=scale, chunk_size=chunk_size,
        )
        return o

    def _fwd(q, k, v):
        o, _ = fused_chunk_simple_gla_fwd(
            q, k, v, g_gamma=g_gamma, scale=scale, chunk_size=chunk_size,
        )
        # Pre-compute h for backward (trade memory for compute — no recomputation)
        h, _ = chunk_fwd_h(
            k=k, v=v, g=None, g_gamma=g_gamma, gk=None,
            h0=None, output_final_state=False,
            chunk_size=chunk_size, states_in_fp32=True,
        )
        return o, (q, k, v, h)

    def _bwd(residuals, do):
        q, k, v, h = residuals
        B, T, H, K = q.shape
        C = chunk_size
        s = scale if scale is not None else K ** -0.5
        interp = not is_tpu_runtime()

        # Use pre-computed h — skip recomputation
        dq, dk, dv, _ = _fused_chunk_bwd_launcher(
            q, k, v, h, do, g_gamma, None,
            scale=s, output_dh0=False,
            chunk_size=C, interpret=interp,
        )
        return dq, dk, dv

    _compute.defvjp(_fwd, _bwd)
    return _compute(q, k, v)


def optimized_compute(B=2, T=4096, H=16, K=128, V=128, chunk_size=64):
    """Forward + backward fused_chunk_simple_gla with Pallas TPU kernels.

    Returns the loss scalar. Both forward and backward are computed,
    so timing captures the full training step performance.
    """
    q, k_arr, v, g_gamma = _make_test_data(B, T, H, K, V, chunk_size)
    scale = K ** -0.5

    def loss_fn(q, k, v):
        return fused_chunk_simple_gla(
            q.astype(jnp.float32), k.astype(jnp.float32),
            v.astype(jnp.float32), g_gamma, scale, chunk_size,
        ).sum()

    loss, _ = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k_arr, v)
    return loss
# EVOLVE-BLOCK-END
