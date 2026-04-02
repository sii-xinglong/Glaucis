# Source: primatrix/pallas-kernel @ branch: main
# Commit: cbeeae7953583f8b5a406a0a76d233f8df569352
# Initialized: 2026-04-01
# Variant: mixed_unroll_6 — Python for-loop forward + h_buf output (eliminates chunk_fwd_h)
"""fused_chunk_simple_gla Pallas TPU kernel — mixed_unroll_6.

Fused chunk forward (h + o in single kernel) + split backward (Pass 1: dv+dh,
Pass 2: dq+dk) for Simple GLA with per-head g_gamma decay.

Forward kernel: 4-step grid unrolling using Python for-loop (SO21: identical
compilation to manual unroll). Additionally outputs h_buf — the h state BEFORE
each sub-step's update — eliminating the separate chunk_fwd_h pallas_call.

Key change: _fwd now gets h states directly from fused_chunk_fwd_kernel's
h_buf output instead of calling chunk_fwd_h. This removes one full pallas_call
from the forward pass.

Backward is split into 2 passes to reduce register pressure:
Pass 1: dv + dh propagation (sequential, "arbitrary" time dim)
Pass 2: dq + dk (parallel, dh_states fully materialized)

Optimization targets within the EVOLVE-BLOCK:
  - Forward kernel: Python for-loop unrolling (cleaner, same compiled output)
  - Forward kernel: h_buf output eliminates separate chunk_fwd_h call
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
# Forward: Fused chunk forward kernel with 4-step time unrolling (Python loop)
# Grid: (B, H, NK, NV, NT//4) — each iteration processes 4 consecutive chunks
# Also outputs h_buf: h state BEFORE each chunk update (eliminates chunk_fwd_h)
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
    h_buf_ref,      # (1, 4, 1, BK, BV) — h states before each sub-step update
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

    # Python for-loop over 4 sub-steps (SO21: identical compiled output to manual unroll)
    for step in range(4):
        offset = step * BT

        # Load tile for this sub-step
        b_q = q_ref[0, 0, pl.ds(offset, BT), :]
        b_k = k_ref[0, 0, pl.ds(offset, BT), :]
        b_v = v_ref[0, 0, pl.ds(offset, BT), :]
        b_h = scratch_ref[...]

        # Store h state BEFORE this sub-step's update (for backward use)
        h_buf_ref[0, step, 0, :, :] = b_h.astype(h_buf_ref.dtype)

        # Compute partial output: inter-chunk (q @ h) + intra-chunk (q @ k^T @ v)
        partial_o = jnp.dot(b_q, b_h, preferred_element_type=jnp.float32)
        partial_A = jnp.dot(b_q, b_k.T, preferred_element_type=jnp.float32)

        if g_ref is not None:
            b_g = g_ref[0, 0, pl.ds(offset, BT), 0].astype(jnp.float32)
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
        o_ref[0, 0, 0, pl.ds(offset, BT), :] = partial.astype(o_ref.dtype)

        # Update h state for next sub-step
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
) -> tuple[jax.Array, jax.Array | None, jax.Array]:
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
    # h_buf: shape (B, NT, H, K, V), block (1, 4, 1, BK, BV)
    # index_map returns block indices; with block_shape[1]=4, dim-1 slice = [t*4 : t*4+4]
    def h_buf_map(b, h, ik, iv, t):
        return (b, t, h, ik, iv)

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
    # h_buf: (1, 4, 1, BK, BV) — 4 h snapshots (before each sub-step) per iteration
    spec_h_buf = pl.BlockSpec((1, 4, 1, BK, BV), h_buf_map)

    # out_shape for o_partial: time dim stays T (NT_QUARTER * 4*BT = NT * BT = T)
    out_shapes = [
        jax.ShapeDtypeStruct((B, H, NK, T, V), jnp.float32),
        jax.ShapeDtypeStruct((B, H, K, V), jnp.float32)
        if output_final_state else None,
        # h_buf: (B, NT, H, K, V) in float32 — matches backward's expected layout
        jax.ShapeDtypeStruct((B, NT, H, K, V), jnp.float32),
    ]

    o_partial, ht, h_buf = pl.pallas_call(
        functools.partial(
            _fused_chunk_fwd_4step_kernel,
            BT=BT, NT=NT, NT_QUARTER=NT_QUARTER,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[spec_qk, spec_qk, spec_v, spec_h0,
                      spec_g, spec_gamma],
            out_specs=[spec_o, spec_ht, spec_h_buf],
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

    # h_buf already has shape (B, NT, H, K, V) — ready for backward consumption
    return o, ht, h_buf


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

    Forward uses fused kernel with 4-step time unrolling (Python for-loop,
    SO21: identical compilation to manual unroll). h states are captured as
    h_buf from the fused forward kernel's new output — eliminates the separate
    chunk_fwd_h pallas_call entirely.

    Backward is split into 2 passes to reduce register pressure:
    Pass 1: dv + dh propagation (sequential, "arbitrary" time dim)
    Pass 2: dq + dk (parallel, dh_states fully materialized)
    """
    @jax.custom_vjp
    def _compute(q, k, v):
        o, _, _ = fused_chunk_simple_gla_fwd(
            q, k, v, g_gamma=g_gamma, scale=scale, chunk_size=chunk_size,
        )
        return o

    def _fwd(q, k, v):
        # Single fused kernel call: computes o AND h_buf simultaneously
        # h_buf shape: (B, NT, H, K, V) in float32 — ready for backward
        o, _, h = fused_chunk_simple_gla_fwd(
            q, k, v, g_gamma=g_gamma, scale=scale, chunk_size=chunk_size,
        )
        # h already has shape (B, NT, H, K, V) matching backward's expectation
        return o, (q, k, v, h)

    def _bwd(residuals, do):
        q, k, v, h = residuals
        B, T, H, K = q.shape
        C = chunk_size
        s = scale if scale is not None else K ** -0.5
        interp = not is_tpu_runtime()

        # Use pre-computed h from fused forward — no recomputation, no extra pallas_call
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
