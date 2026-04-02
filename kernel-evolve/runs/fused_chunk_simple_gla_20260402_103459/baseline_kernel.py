# Source: primatrix/pallas-kernel @ branch: main
# Commit: cbeeae7953583f8b5a406a0a76d233f8df569352
# Initialized: 2026-04-01
# Variant: L1_fwd_4step — 4-step grid unrolling on forward kernel only
"""fused_chunk_simple_gla Pallas TPU kernel — L1 + forward 4-step unrolling.

Fused chunk forward (h + o in single kernel) + split backward (Pass 1: dv+dh,
Pass 2: dq+dk) for Simple GLA with per-head g_gamma decay.

Forward kernel: 4-step grid unrolling.  Grid is (B, H, NK, NV, NT//4) instead
of (B, H, NK, NV, NT).  Each iteration processes 4 consecutive chunks,
reducing grid launches by 4x on the time axis while keeping the backward
passes unchanged.

Backward is split into 2 passes to reduce register pressure:
Pass 1: dv + dh propagation (sequential, "arbitrary" time dim)
Pass 2: dq + dk (parallel, dh_states fully materialized)

Optimization targets within the EVOLVE-BLOCK:
  - Forward kernel: 4-step time unrolling to reduce grid overhead
  - Backward kernel: unchanged (split 2-pass architecture)
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
# Forward: Fused chunk forward kernel with 4-step time unrolling
# Grid: (B, H, NK, NV, NT//4) — each iteration processes 4 consecutive chunks
# =============================================================================


def _fused_chunk_fwd_4step_kernel(
    q_ref,          # (1, 1, 4*BT, BK)  — K-tiled, 4 chunks
    k_ref,          # (1, 1, 4*BT, BK)  — K-tiled, 4 chunks
    v_ref,          # (1, 1, 4*BT, BV)  — V-tiled, 4 chunks
    h0_ref,         # (1, 1, BK, BV)  or None
    g_ref,          # (1, 1, 4*BT, 128) or None
    g_gamma_ref,    # [H] via SMEM/ANY, or None
    o_ref,          # (1, 1, 1, 4*BT, BV)  — partial output, 4 chunks
    ht_ref,         # (1, 1, BK, BV)   — output, or None
    scratch_ref,    # (BK, BV) VMEM float32
    *,
    BT: int,
    NT: int,
    NT_QUARTER: int,
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

    # --- Sub-step 0: first chunk in this 4-step window ---
    b_q0 = q_ref[0, 0, pl.ds(0, BT), :]
    b_k0 = k_ref[0, 0, pl.ds(0, BT), :]
    b_v0 = v_ref[0, 0, pl.ds(0, BT), :]
    b_h0 = scratch_ref[...]

    partial_o0 = jnp.dot(b_q0, b_h0, preferred_element_type=jnp.float32)
    partial_A0 = jnp.dot(b_q0, b_k0.T, preferred_element_type=jnp.float32)

    if g_ref is not None:
        b_g0 = g_ref[0, 0, pl.ds(0, BT), 0].astype(jnp.float32)
        partial_o0 = partial_o0 * exp(b_g0)[:, None]
        g_diff0 = b_g0[:, None] - b_g0[None, :]
        fwd_mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
        safe_g_diff0 = jnp.where(fwd_mask, g_diff0, 0.0)
        partial_A0 = partial_A0 * exp(safe_g_diff0)

    if g_gamma_ref is not None:
        partial_o0 = partial_o0 * exp(b_g_gamma)[:, None]
        g_gamma_diff = b_g_gamma[:, None] - b_g_gamma[None, :]
        fwd_mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
        safe_g_gamma_diff = jnp.where(fwd_mask, g_gamma_diff, 0.0)
        partial_A0 = partial_A0 * exp(safe_g_gamma_diff)

    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    partial_A0 = jnp.where(mask, partial_A0, 0.0)

    partial0 = partial_o0 + jnp.dot(
        partial_A0, b_v0.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    o_ref[0, 0, 0, pl.ds(0, BT), :] = partial0.astype(o_ref.dtype)

    # Update h after sub-step 0
    b_v_upd0 = b_v0

    if g_ref is not None:
        b_g0_last = b_g0[BT - 1]
        scratch_ref[...] *= exp(b_g0_last)
        b_v_upd0 = (b_v_upd0 * exp(b_g0_last - b_g0)[:, None]).astype(b_v_upd0.dtype)

    if g_gamma_ref is not None:
        b_g_gamma_last = b_gamma * BT
        scratch_ref[...] *= exp(b_g_gamma_last)
        b_v_upd0 = (b_v_upd0 * exp(b_g_gamma_last - b_g_gamma)[:, None]).astype(b_v_upd0.dtype)

    scratch_ref[...] = scratch_ref[...] + jnp.dot(
        b_k0.astype(jnp.float32).T,
        b_v_upd0.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # --- Sub-step 1: second chunk in this 4-step window ---
    b_q1 = q_ref[0, 0, pl.ds(BT, BT), :]
    b_k1 = k_ref[0, 0, pl.ds(BT, BT), :]
    b_v1 = v_ref[0, 0, pl.ds(BT, BT), :]
    b_h1 = scratch_ref[...]

    partial_o1 = jnp.dot(b_q1, b_h1, preferred_element_type=jnp.float32)
    partial_A1 = jnp.dot(b_q1, b_k1.T, preferred_element_type=jnp.float32)

    if g_ref is not None:
        b_g1 = g_ref[0, 0, pl.ds(BT, BT), 0].astype(jnp.float32)
        partial_o1 = partial_o1 * exp(b_g1)[:, None]
        g_diff1 = b_g1[:, None] - b_g1[None, :]
        fwd_mask1 = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
        safe_g_diff1 = jnp.where(fwd_mask1, g_diff1, 0.0)
        partial_A1 = partial_A1 * exp(safe_g_diff1)

    if g_gamma_ref is not None:
        partial_o1 = partial_o1 * exp(b_g_gamma)[:, None]
        g_gamma_diff1 = b_g_gamma[:, None] - b_g_gamma[None, :]
        fwd_mask1 = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
        safe_g_gamma_diff1 = jnp.where(fwd_mask1, g_gamma_diff1, 0.0)
        partial_A1 = partial_A1 * exp(safe_g_gamma_diff1)

    mask1 = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    partial_A1 = jnp.where(mask1, partial_A1, 0.0)

    partial1 = partial_o1 + jnp.dot(
        partial_A1, b_v1.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    o_ref[0, 0, 0, pl.ds(BT, BT), :] = partial1.astype(o_ref.dtype)

    # Update h after sub-step 1
    b_v_upd1 = b_v1

    if g_ref is not None:
        b_g1_last = b_g1[BT - 1]
        scratch_ref[...] *= exp(b_g1_last)
        b_v_upd1 = (b_v_upd1 * exp(b_g1_last - b_g1)[:, None]).astype(b_v_upd1.dtype)

    if g_gamma_ref is not None:
        b_g_gamma_last1 = b_gamma * BT
        scratch_ref[...] *= exp(b_g_gamma_last1)
        b_v_upd1 = (b_v_upd1 * exp(b_g_gamma_last1 - b_g_gamma)[:, None]).astype(b_v_upd1.dtype)

    scratch_ref[...] = scratch_ref[...] + jnp.dot(
        b_k1.astype(jnp.float32).T,
        b_v_upd1.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # --- Sub-step 2: third chunk in this 4-step window ---
    b_q2 = q_ref[0, 0, pl.ds(2 * BT, BT), :]
    b_k2 = k_ref[0, 0, pl.ds(2 * BT, BT), :]
    b_v2 = v_ref[0, 0, pl.ds(2 * BT, BT), :]
    b_h2 = scratch_ref[...]

    partial_o2 = jnp.dot(b_q2, b_h2, preferred_element_type=jnp.float32)
    partial_A2 = jnp.dot(b_q2, b_k2.T, preferred_element_type=jnp.float32)

    if g_ref is not None:
        b_g2 = g_ref[0, 0, pl.ds(2 * BT, BT), 0].astype(jnp.float32)
        partial_o2 = partial_o2 * exp(b_g2)[:, None]
        g_diff2 = b_g2[:, None] - b_g2[None, :]
        fwd_mask2 = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
        safe_g_diff2 = jnp.where(fwd_mask2, g_diff2, 0.0)
        partial_A2 = partial_A2 * exp(safe_g_diff2)

    if g_gamma_ref is not None:
        partial_o2 = partial_o2 * exp(b_g_gamma)[:, None]
        g_gamma_diff2 = b_g_gamma[:, None] - b_g_gamma[None, :]
        fwd_mask2 = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
        safe_g_gamma_diff2 = jnp.where(fwd_mask2, g_gamma_diff2, 0.0)
        partial_A2 = partial_A2 * exp(safe_g_gamma_diff2)

    mask2 = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    partial_A2 = jnp.where(mask2, partial_A2, 0.0)

    partial2 = partial_o2 + jnp.dot(
        partial_A2, b_v2.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    o_ref[0, 0, 0, pl.ds(2 * BT, BT), :] = partial2.astype(o_ref.dtype)

    # Update h after sub-step 2
    b_v_upd2 = b_v2

    if g_ref is not None:
        b_g2_last = b_g2[BT - 1]
        scratch_ref[...] *= exp(b_g2_last)
        b_v_upd2 = (b_v_upd2 * exp(b_g2_last - b_g2)[:, None]).astype(b_v_upd2.dtype)

    if g_gamma_ref is not None:
        b_g_gamma_last2 = b_gamma * BT
        scratch_ref[...] *= exp(b_g_gamma_last2)
        b_v_upd2 = (b_v_upd2 * exp(b_g_gamma_last2 - b_g_gamma)[:, None]).astype(b_v_upd2.dtype)

    scratch_ref[...] = scratch_ref[...] + jnp.dot(
        b_k2.astype(jnp.float32).T,
        b_v_upd2.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # --- Sub-step 3: fourth chunk in this 4-step window ---
    b_q3 = q_ref[0, 0, pl.ds(3 * BT, BT), :]
    b_k3 = k_ref[0, 0, pl.ds(3 * BT, BT), :]
    b_v3 = v_ref[0, 0, pl.ds(3 * BT, BT), :]
    b_h3 = scratch_ref[...]

    partial_o3 = jnp.dot(b_q3, b_h3, preferred_element_type=jnp.float32)
    partial_A3 = jnp.dot(b_q3, b_k3.T, preferred_element_type=jnp.float32)

    if g_ref is not None:
        b_g3 = g_ref[0, 0, pl.ds(3 * BT, BT), 0].astype(jnp.float32)
        partial_o3 = partial_o3 * exp(b_g3)[:, None]
        g_diff3 = b_g3[:, None] - b_g3[None, :]
        fwd_mask3 = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
        safe_g_diff3 = jnp.where(fwd_mask3, g_diff3, 0.0)
        partial_A3 = partial_A3 * exp(safe_g_diff3)

    if g_gamma_ref is not None:
        partial_o3 = partial_o3 * exp(b_g_gamma)[:, None]
        g_gamma_diff3 = b_g_gamma[:, None] - b_g_gamma[None, :]
        fwd_mask3 = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
        safe_g_gamma_diff3 = jnp.where(fwd_mask3, g_gamma_diff3, 0.0)
        partial_A3 = partial_A3 * exp(safe_g_gamma_diff3)

    mask3 = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    partial_A3 = jnp.where(mask3, partial_A3, 0.0)

    partial3 = partial_o3 + jnp.dot(
        partial_A3, b_v3.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    o_ref[0, 0, 0, pl.ds(3 * BT, BT), :] = partial3.astype(o_ref.dtype)

    # Update h after sub-step 3
    b_v_upd3 = b_v3

    if g_ref is not None:
        b_g3_last = b_g3[BT - 1]
        scratch_ref[...] *= exp(b_g3_last)
        b_v_upd3 = (b_v_upd3 * exp(b_g3_last - b_g3)[:, None]).astype(b_v_upd3.dtype)

    if g_gamma_ref is not None:
        b_g_gamma_last3 = b_gamma * BT
        scratch_ref[...] *= exp(b_g_gamma_last3)
        b_v_upd3 = (b_v_upd3 * exp(b_g_gamma_last3 - b_g_gamma)[:, None]).astype(b_v_upd3.dtype)

    scratch_ref[...] = scratch_ref[...] + jnp.dot(
        b_k3.astype(jnp.float32).T,
        b_v_upd3.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # Store final h state at the last 4-step iteration
    @pl.when(i_t == NT_QUARTER - 1)
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

    assert NT % 4 == 0, f"NT={NT} must be divisible by 4 for 4-step unrolling"
    NT_QUARTER = NT // 4

    _q = q.transpose(0, 2, 1, 3)
    _k = k.transpose(0, 2, 1, 3)
    _v = v.transpose(0, 2, 1, 3)

    _g = None
    if g is not None:
        _g = g.transpose(0, 2, 1)
        _g = jnp.broadcast_to(_g[:, :, :, None], (B, H, T, 128))

    # 4-step unrolled grid: NT_QUARTER iterations instead of NT
    grid = (B, H, NK, NV, NT_QUARTER)

    # BlockSpecs now tile 4*BT in the time dimension
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

    # Time dimension tiles are 4*BT to cover 4 chunks per iteration
    spec_qk    = pl.BlockSpec((1, 1, 4 * BT, BK), qk_map)
    spec_v     = pl.BlockSpec((1, 1, 4 * BT, BV), v_map)
    spec_h0    = pl.BlockSpec((1, 1, BK, BV), state_map) if h0 is not None else None
    spec_g     = pl.BlockSpec((1, 1, 4 * BT, 128), g_map) if _g is not None else None
    spec_gamma = pl.BlockSpec(memory_space=smem) if g_gamma is not None else None

    # Output covers 4*BT per iteration
    spec_o     = pl.BlockSpec((1, 1, 1, 4 * BT, BV), o_map)
    spec_ht    = pl.BlockSpec((1, 1, BK, BV), state_map) if output_final_state else None

    # out_shape for o_partial: time dim stays T (NT_QUARTER * 4*BT = NT * BT = T)
    out_shapes = [
        jax.ShapeDtypeStruct((B, H, NK, T, V), jnp.float32),
        jax.ShapeDtypeStruct((B, H, K, V), jnp.float32)
        if output_final_state else None,
    ]

    o_partial, ht = pl.pallas_call(
        functools.partial(
            _fused_chunk_fwd_4step_kernel,
            BT=BT, NT=NT, NT_QUARTER=NT_QUARTER,
        ),
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
# Backward Pass 1: dv computation + dh state propagation
# =============================================================================
# This kernel computes dv and propagates dh backwards through time.
# By isolating dv+dh from dq+dk, each kernel has fewer live intermediates,
# reducing register spills compared to the monolithic backward kernel.
#
# Live arrays in this pass:
#   b_q [BT,K], b_k [BT,K], b_v [BT,V], b_do [BT,V], b_dh [K,V]
#   b_a_masked [BT,BT] (for intra-chunk dv), k_decay [BT,K]
# Total peak: ~3 large matrices + 1 [BT,BT] attention matrix
#
# Original monolithic kernel had all of the above PLUS:
#   b_dA [BT,BT], b_dA_gated [BT,BT], dq/dk intermediates
# which caused 310K register spills.


def _bwd_dv_dh_kernel(
    q_ref, k_ref, v_ref, do_ref, g_gamma_ref, dht_ref,
    dv_ref, dh_states_ref,
    scratch_ref,
    *, BT: int, NT: int, scale: float,
):
    """Pass 1: Compute dv and propagate dh state backwards.

    Iterates chunks in reverse time order. For each chunk:
    1. Compute intra-chunk dv = A^T @ do (using causal attention weights)
    2. Compute inter-chunk dv = k_decay @ dh (from accumulated gradient state)
    3. Store dh state for this time step (used by Pass 2)
    4. Update dh: decay + q_hat^T @ do accumulation
    """
    K = q_ref.shape[3]
    V = v_ref.shape[3]
    i_t = pl.program_id(2)
    head_idx = pl.program_id(1)

    # Initialize dh from dht (gradient w.r.t. final hidden state) or zeros
    @pl.when(i_t == 0)
    def _init():
        if dht_ref is not None:
            scratch_ref[:, :] = dht_ref[0, 0].astype(jnp.float32)
        else:
            scratch_ref[:, :] = jnp.zeros((K, V), dtype=jnp.float32)

    # Load tile data
    b_q = q_ref[0, 0]
    b_k = k_ref[0, 0]
    b_v = v_ref[0, 0]
    b_do = do_ref[0, 0]
    b_dh = scratch_ref[...].astype(jnp.float32)

    # Compute gating factors
    b_gamma = g_gamma_ref[head_idx].astype(jnp.float32)
    b_g = b_gamma * (jnp.arange(BT) + 1).astype(jnp.float32)
    b_gn = b_g[BT - 1]

    # Build causal attention matrix with decay
    pos = (jnp.arange(BT) + 1).astype(jnp.float32)
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    safe_diff = jnp.where(mask, b_gamma * (pos[:, None] - pos[None, :]), 0.0)
    decay = jnp.exp(safe_diff)
    b_a = jnp.dot(b_q, b_k.T, preferred_element_type=jnp.float32) * scale * decay
    b_a_masked = jnp.where(mask, b_a, 0.0).astype(b_do.dtype)

    # dv = A^T @ do (intra-chunk) + k_decay @ dh (inter-chunk)
    b_dv_intra = jnp.dot(b_a_masked.T, b_do, preferred_element_type=jnp.float32)
    k_decay = (b_k * jnp.exp(b_gn - b_g)[:, None]).astype(b_k.dtype)
    b_dv_inter = jnp.dot(k_decay, b_dh.astype(b_k.dtype), preferred_element_type=jnp.float32)
    dv_ref[0, 0] = (b_dv_intra + b_dv_inter).astype(dv_ref.dtype)

    # Store dh state BEFORE updating it (Pass 2 needs dh at this time step)
    # The grid iterates in reverse, so i_t=0 is the last chunk, i_t=NT-1 is the first.
    # We store into reverse-mapped index so Pass 2 can read in forward or reverse order.
    dh_states_ref[0, 0, 0] = scratch_ref[...].astype(dh_states_ref.dtype)

    # Update dh for next (earlier) time step:
    # dh_{t-1} = dh_t * exp(gamma * BT) + q_hat^T @ do
    b_dh = b_dh * jnp.exp(b_gn)
    b_q_hat = (b_q * (scale * jnp.exp(b_g)[:, None])).astype(jnp.float32)
    b_dh = b_dh + jnp.dot(
        b_q_hat.T, b_do.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    scratch_ref[...] = b_dh


def _bwd_dq_dk_kernel(
    q_ref, k_ref, v_ref, h_ref, do_ref, g_gamma_ref, dh_states_ref,
    dq_ref, dk_ref, dh0_ref,
    *, BT: int, NT: int, scale: float,
):
    """Pass 2: Compute dq and dk using pre-computed h and dh_states.

    For each chunk (reverse time order matching Pass 1):
    1. Compute intra-chunk dq = dA_gated @ k
    2. Compute inter-chunk dq = do @ h^T * scale * exp(g)
    3. Compute intra-chunk dk = dA_gated^T @ q
    4. Compute inter-chunk dk = v @ dh^T * exp(gn - g)

    No scratch space needed -- dh_states are fully materialized from Pass 1,
    so this kernel has NO sequential state dependency and uses parallel semantics
    on the time dimension.
    """
    K = q_ref.shape[3]
    V = v_ref.shape[3]
    i_t = pl.program_id(2)
    head_idx = pl.program_id(1)

    # Load tile data
    b_q = q_ref[0, 0]
    b_k = k_ref[0, 0]
    b_v = v_ref[0, 0]
    b_h = h_ref[0, 0, 0].astype(jnp.float32)
    b_do = do_ref[0, 0]
    b_dh = dh_states_ref[0, 0, 0].astype(jnp.float32)

    # Compute gating factors
    b_gamma = g_gamma_ref[head_idx].astype(jnp.float32)
    b_g = b_gamma * (jnp.arange(BT) + 1).astype(jnp.float32)
    b_gn = b_g[BT - 1]

    # Build causal mask and gated dA
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_dA = jnp.dot(b_do, b_v.T, preferred_element_type=jnp.float32) * scale
    b_dA = jnp.where(mask, b_dA, 0.0)

    safe_g_diff = jnp.where(mask, b_g[:, None] - b_g[None, :], 0.0)
    b_dA_gated = b_dA * jnp.exp(safe_g_diff)

    # dq = dA_gated @ k (intra) + do @ h^T * scale * exp(g) (inter)
    b_dq_intra = jnp.dot(b_dA_gated.astype(b_k.dtype), b_k, preferred_element_type=jnp.float32)
    b_dq_inter = jnp.dot(b_do, b_h.astype(b_do.dtype).T, preferred_element_type=jnp.float32) * (scale * jnp.exp(b_g)[:, None])
    dq_ref[0, 0] = (b_dq_intra + b_dq_inter).astype(dq_ref.dtype)

    # dk = dA_gated^T @ q (intra) + v @ dh^T * exp(gn - g) (inter)
    b_dk_intra = jnp.dot(b_dA_gated.T.astype(b_q.dtype), b_q, preferred_element_type=jnp.float32)
    b_dk_inter = jnp.dot(b_v, b_dh.astype(b_v.dtype).T, preferred_element_type=jnp.float32) * jnp.exp(b_gn - b_g)[:, None]
    dk_ref[0, 0] = (b_dk_intra + b_dk_inter).astype(dk_ref.dtype)

    # Store dh0 if needed (only from the last iteration = first chunk in time)
    # In Pass 1, after the final iteration (i_t == NT-1), scratch holds the
    # fully-propagated dh0. But we need it here too for output_dh0.
    # Actually, dh0 comes from Pass 1's final scratch state. We handle this
    # in the launcher by running a third mini-kernel or by having Pass 1 output it.
    # For simplicity, dh0 output is handled in the launcher from Pass 1.


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
    def rev_dh_states_map(b, h, t):
        return (b, h, NT - 1 - t, 0, 0)
    def state_map(b, h, t):
        return (b, h, 0, 0)

    smem = pltpu.ANY if interpret else pltpu.SMEM

    # =========================================================================
    # Pass 1: dv + dh propagation (sequential over time, "arbitrary" dim)
    # =========================================================================
    pass1_in_specs = [
        pl.BlockSpec((1, 1, BT, K), rev_qk_map),   # q
        pl.BlockSpec((1, 1, BT, K), rev_qk_map),   # k
        pl.BlockSpec((1, 1, BT, V), rev_v_map),     # v
        pl.BlockSpec((1, 1, BT, V), rev_v_map),     # do
        pl.BlockSpec(memory_space=smem),             # g_gamma
        pl.BlockSpec((1, 1, K, V), state_map) if dht is not None else None,  # dht
    ]

    pass1_out_specs = [
        pl.BlockSpec((1, 1, BT, V), rev_v_map),     # dv
        pl.BlockSpec((1, 1, 1, K, V), rev_h_map),   # dh_states
    ]

    pass1_out_shapes = [
        jax.ShapeDtypeStruct((B, H, T, V), v.dtype),           # dv
        jax.ShapeDtypeStruct((B, H, NT, K, V), jnp.float32),   # dh_states [B,H,NT,K,V]
    ]

    dv, dh_states = pl.pallas_call(
        functools.partial(_bwd_dv_dh_kernel, BT=BT, NT=NT, scale=scale),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=pass1_in_specs,
            out_specs=pass1_out_specs,
            scratch_shapes=[pltpu.VMEM((K, V), jnp.float32)],
        ),
        out_shape=pass1_out_shapes,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
        interpret=interpret,
    )(_q, _k, _v, _do, g_gamma, dht)

    # =========================================================================
    # Pass 2: dq + dk (parallel over time -- dh_states fully materialized)
    # =========================================================================
    pass2_in_specs = [
        pl.BlockSpec((1, 1, BT, K), rev_qk_map),    # q
        pl.BlockSpec((1, 1, BT, K), rev_qk_map),    # k
        pl.BlockSpec((1, 1, BT, V), rev_v_map),      # v
        pl.BlockSpec((1, 1, 1, K, V), rev_h_map),    # h (pre-computed fwd states)
        pl.BlockSpec((1, 1, BT, V), rev_v_map),      # do
        pl.BlockSpec(memory_space=smem),              # g_gamma
        pl.BlockSpec((1, 1, 1, K, V), rev_h_map),    # dh_states from Pass 1
    ]

    pass2_out_specs = [
        pl.BlockSpec((1, 1, BT, K), rev_qk_map),    # dq
        pl.BlockSpec((1, 1, BT, K), rev_qk_map),    # dk
        None,  # dh0 placeholder (not computed in Pass 2)
    ]

    pass2_out_shapes = [
        jax.ShapeDtypeStruct((B, H, T, K), q.dtype),       # dq
        jax.ShapeDtypeStruct((B, H, T, K), k.dtype),       # dk
        None,  # dh0 not needed
    ]

    dq, dk, _ = pl.pallas_call(
        functools.partial(_bwd_dq_dk_kernel, BT=BT, NT=NT, scale=scale),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=pass2_in_specs,
            out_specs=pass2_out_specs,
            scratch_shapes=[],
        ),
        out_shape=pass2_out_shapes,
        compiler_params=pltpu.CompilerParams(
            # Time dimension is now parallel since dh_states are materialized
            dimension_semantics=("parallel", "parallel", "parallel"),
        ),
        interpret=interpret,
    )(_q, _k, _v, _h, _do, g_gamma, dh_states)

    dq = dq.transpose(0, 2, 1, 3)
    dk = dk.transpose(0, 2, 1, 3)
    dv = dv.transpose(0, 2, 1, 3)

    # dh0 not supported in split variant (output_dh0=False in all call sites)
    dh0 = None

    return dq, dk, dv, dh0


# =============================================================================
# custom_vjp wrapper + entry point
# =============================================================================


def fused_chunk_simple_gla(q, k, v, g_gamma, scale, chunk_size):
    """Fused chunk Simple GLA with custom_vjp (Pallas TPU kernels).

    Forward uses fused kernel with 4-step time unrolling (h stays in VMEM,
    single pallas_call, NT//4 grid iterations instead of NT).
    h is also pre-computed via chunk_fwd_h and saved as residual -- backward
    uses saved h directly (no recomputation).

    Backward is split into 2 passes to reduce register pressure:
    Pass 1: dv + dh propagation (sequential, "arbitrary" time dim)
    Pass 2: dq + dk (parallel, dh_states fully materialized)
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
