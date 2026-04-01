import functools

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

from tops.ops.utils import exp, get_interpret
from tops.utils import assert_shape, assert_shape_or_none, export_public


def _build_chunk_map(cu_seqlens, T_sum, BT):
    NT = T_sum // BT
    chunk_ids = lax.iota(jnp.int32, NT)
    chunk_pos = chunk_ids * BT
    seq_idx = jnp.searchsorted(cu_seqlens[1:], chunk_pos, side="right")
    return seq_idx



def _chunk_fwd_h_kernel(
    k_ref,  # [1, 1, BT, BK]
    v_ref,  # [1, 1, BT, BV]
    h0_ref,  # [1, 1, BK, BV]
    gk_ref,  # [1, 1, BT, BK]
    g_ref,   # [1, 1, BT, 128]
    g_gamma,  # [H]
    h_ref,  # [1, NS, 1, BK, BV] outputs
    ht_ref,  # [1, 1, BK , BV]
    scratch_ref, #[BK, BV]
    *,
    BT,
    BS,
    NT,
):

    BK = k_ref.shape[3]
    BV = v_ref.shape[3]
    NTS = BS // BT
    T = NT * BT
    i_b, i_h, i_k, i_v, i_t = pl.program_id(0), pl.program_id(1), pl.program_id(2),pl.program_id(3),pl.program_id(4)

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

    k_tile = k_ref[(0, 0, slice(None), slice(None))] # BT * BK
    v_tile = v_ref[(0, 0, slice(None), slice(None))] # BT * BV

    if g_ref is not None:
        b_g_scalar = g_ref[0, 0, slice(None), 0]  # [BT]
        b_g_scalar_last = b_g_scalar[BT - 1]       # scalar
        scratch_ref[...] *= exp(b_g_scalar_last)                 # uniform decay
        v_tile = (v_tile * exp(b_g_scalar_last - b_g_scalar)[:, None]).astype(v_tile.dtype)

    if g_gamma is not None:
        # tpu not support scalar bf16 mul
        b_g_last = (g_gamma[i_h].astype(jnp.float32) * jnp.minimum(BT, T - i_t * BT)).astype(g_gamma.dtype)
        scratch_ref[...] *= exp(b_g_last)
        v_tile = (v_tile * exp(b_g_last - b_g)[:, None]).astype(v_tile.dtype)


    if gk_ref is not None:
        gk_tile = gk_ref[(0, 0, slice(None), slice(None))] # BT * BK
        g_last = gk_tile[-1, :]
        decay = exp(g_last)
        scratch_ref[...] = scratch_ref[...] * decay[:, None]  # [BK, BV] * [BK,1]
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


def check_chunk_fwd(x):
    assert x is None, "x should be None."


# note: The precision difference between this kernel on the TPU and FLA on the GPU is 5e-2.
@functools.partial(
    jax.jit,
    static_argnames=[
        "output_final_state",
        "chunk_size",
        "split_size",
        "states_in_fp32",
    ],
)
def chunk_fwd_h_kernel(
    k: jax.Array,
    v: jax.Array,  # [B,T,H,V]
    *,
    g: jax.Array | None = None,  # [B,T,H]
    g_gamma: jax.Array | None = None,  # (H,)
    gk: jax.Array | None = None,  # [B,T,H,K]
    gv: jax.Array | None = None,  # [B,T,H,V]
    h0: jax.Array | None = None,  # [N,H,K,V]
    output_final_state: bool = False,
    cu_seqlens_cpu: jax.Array | None = None,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 64,
    split_size: int | None = None,
    states_in_fp32: bool = False,
):
    # todo: tune bk and bv for bast performance
    BK = 128
    BV = 128
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens_cpu is None else cu_seqlens_cpu.shape[-1] - 1
    BT = chunk_size
    BS = BT if split_size is None else split_size

    # =================== assert kernel requirements start ===================
    assert_shape(k, (B, T, H, K))
    assert_shape(v, (B, T, H, V))
    assert_shape_or_none(g, (B, T, H))
    assert_shape_or_none(g_gamma, (H,))
    assert_shape_or_none(gk, (B, T, H, K))
    # assert_shape_or_none(gv, (B, T, H, V))
    assert gv is None, "gv is currently not supported"
    assert cu_seqlens_cpu is None, "cu_seqlens_cpu is currently not supported"
    assert cu_seqlens_dev is None, "cu_seqlens_dev is currently not supported"
    assert_shape_or_none(h0, (N, H, K, V))

    assert K % 128 == 0, "K % 128 must equal to 0."
    assert V % 128 == 0, "V % 128 must equal to 0."
    assert T % chunk_size == 0, "T mod chunk_size must equal to 0."
    if cu_seqlens_cpu is not None:
        assert cu_seqlens_cpu[0] == 0, "cu_seqlens_cpu must start with 0."
        assert (cu_seqlens_cpu % chunk_size == 0).all(), "cu_seqlens_cpu must be multiples of chunk_size."

    assert BS % BT == 0, (
        f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"
    )
    # =================== assert kernel requirements done ===================

    # N: the actual number of sequences in the batch with either equal or variable lengths

    N, NS = (
        B,
        T // BS,
    )  # split_offsets[-1] # NS number of chunk_size
    NT = T // BT

    k = jnp.transpose(k, (0, 2, 1, 3))  # (B,H,T,K)
    v = jnp.transpose(v, (0, 2, 1, 3))  # (B,H,T,V)
    if gk is not None:
        gk = jnp.transpose(gk, (0, 2, 1, 3))  # (B,H,T,K)

    if g is not None:
        g = jnp.transpose(g, (0, 2, 1))  # (B, H, T)
        g = jnp.broadcast_to(g[:, :, :, None], (B, H, T, 128))  # (B, H, T, 128)

    grid = (B, H, pl.cdiv(K, BK), pl.cdiv(V, BV), NT)

    def k_index_map(batch_index, head_index, k_index, _, t_index):
        return batch_index, head_index, t_index, k_index

    def gk_index_map(batch_index, head_index,  k_index, _, t_index):
        return batch_index, head_index, t_index, k_index

    def g_index_map(batch_index, head_index,  k_index, _, t_index):
        return batch_index, head_index, t_index, 0

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
            shape=(N, NS, H, K, V), dtype=k.dtype if not states_in_fp32 else jnp.float32
        )
    ]
    out_specs = [pl.BlockSpec((1, NS, 1, BK, BV), h_index_map)]
    if output_final_state:
        out_shape.append(jax.ShapeDtypeStruct(shape=(N, H, K, V), dtype=jnp.float32))
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
    if h0 is not None:
        in_specs.append(pl.BlockSpec((1, 1, BK, BV), h0_index_map))
    else:
        in_specs.append(None)
    if gk is not None:
        in_specs.append(pl.BlockSpec((1, 1, BT, BK), gk_index_map))
    else:
        in_specs.append(None)

    if g is not None:
        in_specs.append(pl.BlockSpec((1, 1, BT, 128), g_index_map))
    else:
        in_specs.append(None)

    if g_gamma is not None:
        in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    else:
        in_specs.append(None)

    kernel = functools.partial(
        _chunk_fwd_h_kernel,
        BT=BT,
        BS=BS,
        NT=NT,
    )
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
                "parallel",
                "parallel",
                "parallel",
                "parallel",
                "arbitrary",
            ),
            # vmem_limit_bytes=32 * 1024 * 1024,
            disable_bounds_checks=True,
        ),
    )(k, v, h0, gk, g, g_gamma)

    h = h.reshape(B, -1, H, K, V)
    ht = ht.reshape(N, H, K, V) if ht is not None else None

    if output_final_state:
        return h, ht
    return h, None


def _chunk_fwd_h_scan(k, v, g, g_gamma, gk, h0,
                      output_final_state, states_in_fp32,
                      C, B, T, H, K, V, NT):
    """lax.scan-based forward state propagation for fixed-length sequences.

    Replaces the Python for-loop in chunk_fwd_h_ref to avoid XLA
    trace-time loop unrolling (which creates a huge HLO graph).
    """
    h_dtype = jnp.float32 if states_in_fp32 else k.dtype
    has_g = g is not None
    has_gk = gk is not None

    # Reshape into chunks: [B, NT, C, H, D] then [NT, B, C, H, D] for scan
    k_scan = k.reshape(B, NT, C, H, K).transpose(1, 0, 2, 3, 4)
    v_scan = v.reshape(B, NT, C, H, V).transpose(1, 0, 2, 3, 4)

    scan_inputs = (k_scan, v_scan)
    if has_g:
        g_scan = g.reshape(B, NT, C, H).transpose(1, 0, 2, 3)
        scan_inputs += (g_scan,)
    if has_gk:
        gk_scan = gk.reshape(B, NT, C, H, K).transpose(1, 0, 2, 3, 4)
        scan_inputs += (gk_scan,)

    # Precompute g_gamma decay terms (constant across chunks)
    if g_gamma is not None:
        g_gamma_f32 = g_gamma.astype(jnp.float32)
        g_last_gamma = g_gamma_f32 * C  # [H]
        state_decay = jnp.exp(g_last_gamma)  # [H]
        b_g_gamma = g_gamma_f32[None, :] * (jnp.arange(C, dtype=jnp.float32) + 1)[:, None]  # [C, H]
        v_decay = jnp.exp(g_last_gamma[None, :] - b_g_gamma)  # [C, H]

    def scan_fn(h, chunk_data):
        # Unpack scan inputs (structure determined at trace time)
        idx = 0
        b_k = chunk_data[idx]; idx += 1
        b_v = chunk_data[idx]; idx += 1
        if has_g:
            b_g_scalar = chunk_data[idx]; idx += 1
        if has_gk:
            b_gk = chunk_data[idx]

        h_out = h  # state BEFORE update

        # Scalar gate g: [B, C, H]
        if has_g:
            b_g_last = b_g_scalar[:, -1, :]  # [B, H]
            h = h * jnp.exp(b_g_last.astype(jnp.float32))[:, :, None, None]
            b_v = (b_v * jnp.exp((b_g_last[:, None, :] - b_g_scalar).astype(jnp.float32))[:, :, :, None]).astype(b_v.dtype)

        # Per-head fixed decay g_gamma
        if g_gamma is not None:
            h = h * state_decay[None, :, None, None]
            b_v = (b_v * v_decay[None, :, :, None]).astype(b_v.dtype)

        # Per-K-dim gate gk: [B, C, H, K]
        if has_gk:
            b_gk_last = b_gk[:, -1, :, :]  # [B, H, K]
            h = h * jnp.exp(b_gk_last.astype(jnp.float32))[:, :, :, None]
            b_k = b_k * jnp.exp((b_gk_last[:, None, :, :] - b_gk).astype(jnp.float32))

        # State update: h += k^T @ v  (contract C, batch B and H)
        kv = lax.dot_general(
            b_k,
            b_v,
            dimension_numbers=(((1,), (1,)), ((0, 2), (0, 2))),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        h = h + kv
        return h, h_out

    # Initial state: [B, H, K, V]  (N = B for fixed-length)
    h_init = jnp.zeros((B, H, K, V), dtype=jnp.float32)
    if h0 is not None:
        h_init = h0.reshape(B, H, K, V).astype(jnp.float32)

    h_final, h_all = lax.scan(scan_fn, h_init, scan_inputs)
    # h_all: [NT, B, H, K, V] -> [B, NT, H, K, V]
    h_all = h_all.transpose(1, 0, 2, 3, 4).astype(h_dtype)

    ht = None
    if output_final_state:
        ht = h_final.astype(jnp.float32)  # [B, H, K, V]

    return h_all, ht


def chunk_fwd_h_ref(
    k: jax.Array,
    v: jax.Array,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    gk: jax.Array | None = None,
    gv: jax.Array | None = None,
    h0: jax.Array | None = None,
    output_final_state: bool = False,
    states_in_fp32: bool = False,
    cu_seqlens_cpu: jax.Array | None = None,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array | None]:
    """Inter-chunk hidden state propagation.

    Computes the hidden state at the start of each chunk by
    sequentially propagating through chunks.

    Args:
        k:  [B, T, H, K] — keys (T must be a multiple of chunk_size)
        v:  [B, T, H, V] — values
        g:  [B, T, H] — chunk-local cumsum of scalar gate (optional)
        g_gamma: [H] — per-head fixed decay rate (optional)
        gk: [B, T, H, K] — chunk-local cumsum of K-dim gates (optional)
        gv: [B, T, H, V] — V-dim gate (optional, currently unused)
        h0: [N, H, K, V] — initial hidden state (optional)
        output_final_state: whether to return final state
        states_in_fp32: if True, store h_all in float32 instead of k.dtype
        cu_seqlens_dev: cumulative sequence lengths (optional)
        cu_seqlens_cpu: alias for cu_seqlens (backward compat)
        chunk_size: block size

    Returns:
        h:  [B, NT, H, K, V] — hidden state at the start of each chunk
        ht: [B, H, K, V] or None — final hidden state
    """

    B, T, H, K = k.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C
    N = B if cu_seqlens_dev is None else cu_seqlens_dev.shape[-1] - 1
    assert T % C == 0, "T must be a multiple of chunk_size for chunk_fwd_h"
    assert (cu_seqlens_cpu is None) or (cu_seqlens_cpu % C == 0).all(), (
        "cu_seqlens must be multiples of chunk_size for chunk_fwd_h"
    )

    # Fast path: fixed-length sequences use lax.scan (avoids XLA loop unrolling)
    if cu_seqlens_cpu is None and gv is None:
        return _chunk_fwd_h_scan(
            k, v, g, g_gamma, gk, h0,
            output_final_state, states_in_fp32,
            C, B, T, H, K, V, NT,
        )

    k = k.reshape(-1, H, K)
    v = v.reshape(-1, H, V)
    gk = gk.reshape(-1, H, K) if gk is not None else None
    g = g.reshape(-1, H) if g is not None else None
    h0 = h0.reshape(-1, H, K, V) if h0 is not None else None

    h_dtype = jnp.float32 if states_in_fp32 else k.dtype
    ht = jnp.zeros([N, H, K, V], dtype=jnp.float32)
    h_all = jnp.zeros([B, NT, H, K, V], dtype=h_dtype)
    for i_n in range(N):
        if cu_seqlens_cpu is None:
            bos = i_n * T
            eos = (i_n + 1) * T
        else:
            bos = int(cu_seqlens_cpu[i_n])
            eos = int(cu_seqlens_cpu[i_n + 1])

        h = jnp.zeros((H, K, V), dtype=jnp.float32)
        if h0 is not None:
            h = h + h0[i_n].astype(jnp.float32)

        if g_gamma is not None:
            g_gamma_f32 = g_gamma.astype(jnp.float32)
            b_g = g_gamma_f32[None, :] * (jnp.arange(0, C) + 1)[:, None]  # [C, H] float32

        NT_seq = (eos - bos) // C
        for i_t in range(NT_seq):
            if cu_seqlens_cpu is None:
                h_all = h_all.at[i_n, i_t].set(h.astype(h_all.dtype))
            else:
                h_all = h_all.at[0, bos // C + i_t].set(h.astype(h_all.dtype))
            b_k = k[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, K]
            b_v = v[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, V]

            if g is not None:
                b_g_scalar = g[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H]
                b_g_last = b_g_scalar[-1]  # [H]
                h *= exp(b_g_last)[:, None, None]  # (H, K, V)
                b_v = (b_v * exp(b_g_last[None, :] - b_g_scalar)[:, :, None]).astype(
                    b_v.dtype
                )

            if g_gamma is not None:
                b_g_last = g_gamma_f32 * jnp.minimum(C, (eos - bos) - i_t * C)  # [H] float32
                h *= exp(b_g_last[:, None, None])  # (H, K, V)
                b_v = (b_v * exp(b_g_last[None, :] - b_g)[:, :, None]).astype(
                    b_v.dtype
                )

            if gk is not None:
                b_gk = gk[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, K]
                b_gk_last = b_gk[-1]  # [H, K]
                h *= exp(b_gk_last[:, :, None])  # b_gk_last -> [H, K, V]

                b_k = b_k * exp(
                    b_gk_last[None, :, :] - b_gk
                )  # b_gk_last -> [C, H, K]

            h = h + lax.dot_general(
                b_k,
                b_v,
                dimension_numbers=(((0,), (0,)), ((1,), (1,))),
                precision=lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
        if output_final_state:
            ht = ht.at[i_n].set(h.astype(ht.dtype))
    if output_final_state:
        return h_all, ht
    else:
        return h_all, None


def _chunk_bwd_dh_scan(q, do, g, g_gamma, gk, dht,
                       scale, output_dh0, states_in_fp32,
                       C, B, T, H, K, V, NT):
    """lax.scan-based backward state gradient propagation for fixed-length sequences.

    Mirrors _chunk_fwd_h_scan: replaces the Python for-loop in chunk_bwd_dh_ref
    to avoid XLA trace-time loop unrolling (which creates a huge HLO graph).
    """
    has_g = g is not None
    has_gk = gk is not None

    # Reshape into chunks: [B, NT, C, H, D] then [NT, B, C, H, D] for scan
    q_scan = q.reshape(B, NT, C, H, K).transpose(1, 0, 2, 3, 4)
    do_scan = do.reshape(B, NT, C, H, V).transpose(1, 0, 2, 3, 4)

    scan_inputs = (q_scan, do_scan)
    if has_g:
        g_scan = g.reshape(B, NT, C, H).transpose(1, 0, 2, 3)
        scan_inputs += (g_scan,)
    if has_gk:
        gk_scan = gk.reshape(B, NT, C, H, K).transpose(1, 0, 2, 3, 4)
        scan_inputs += (gk_scan,)

    # Precompute g_gamma decay terms (constant across chunks)
    if g_gamma is not None:
        g_gamma_f32 = g_gamma.astype(jnp.float32)
        g_last_gamma = g_gamma_f32 * C  # [H]
        state_decay = jnp.exp(g_last_gamma)  # [H]
        b_g_ramp = g_gamma_f32[None, :] * (jnp.arange(C, dtype=jnp.float32) + 1)[:, None]  # [C, H]

    def scan_fn(dh, chunk_data):
        # Unpack scan inputs (structure determined at trace time)
        idx = 0
        b_q = chunk_data[idx]; idx += 1
        b_do = chunk_data[idx]; idx += 1
        if has_g:
            b_g_scalar = chunk_data[idx]; idx += 1
        if has_gk:
            b_gk = chunk_data[idx]

        dh_out = dh  # state BEFORE update (stored at this chunk boundary)

        # Per-K-dim gate gk: [B, C, H, K]
        if has_gk:
            b_gk_last = b_gk[:, -1, :, :]  # [B, H, K]
            dh = dh * jnp.exp(b_gk_last.astype(jnp.float32))[:, :, :, None]
            b_q_hat = b_q * jnp.exp(b_gk.astype(jnp.float32)) * scale
        elif has_g:
            b_g_last = b_g_scalar[:, -1, :]  # [B, H]
            dh = dh * jnp.exp(b_g_last.astype(jnp.float32))[:, :, None, None]
            b_q_hat = (b_q * jnp.exp(b_g_scalar.astype(jnp.float32))[:, :, :, None] * scale)
        elif g_gamma is not None:
            dh = dh * state_decay[None, :, None, None]
            b_q_hat = (b_q * jnp.exp(b_g_ramp)[None, :, :, None] * scale)
        else:
            b_q_hat = b_q * scale

        # Accumulate: dh += q_hat^T @ do  (contract C, batch B and H)
        dh = dh + lax.dot_general(
            b_q_hat,
            b_do,
            dimension_numbers=(((1,), (1,)), ((0, 2), (0, 2))),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        return dh, dh_out

    # Initial state: [B, H, K, V]
    dh_init = jnp.zeros((B, H, K, V), dtype=jnp.float32)
    if dht is not None:
        dh_init = dht.reshape(B, H, K, V).astype(jnp.float32)

    dh_final, dh_all = lax.scan(scan_fn, dh_init, scan_inputs, reverse=True)
    # dh_all: [NT, B, H, K, V] -> [B, NT, H, K, V]
    dh_all = dh_all.transpose(1, 0, 2, 3, 4)

    dh0 = None
    if output_dh0:
        dh0 = dh_final.astype(jnp.float32)  # [B, H, K, V]

    return dh_all, dh0


def chunk_bwd_dh_ref(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    g_gamma: jax.Array,
    gk: jax.Array,
    do: jax.Array,
    h0: jax.Array | None = None,
    dht: jax.Array | None = None,
    scale: float = 1.0,
    output_dh0: bool = False,
    states_in_fp32: bool = False,
    cu_seqlens_cpu: jax.Array | None = None,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array | None]:
    """Backward hidden state gradient propagation.

    Propagates gradients backward through chunks to compute dh at each
    chunk boundary and dh0.

    Args:
        q:   [B, T, H, K] — queries
        k:   [B, T, H, K] — keys
        v:   [B, T, H, V] — values
        gk:  [B, T, H, K] — chunk-local cumsum of gates
        do:  [B, T, H, V] — output gradient
        h0:  [N, H, K, V] — initial hidden state (optional)
        dht: [N, H, K, V] — terminal state gradient (optional)
        scale: scaling factor
        cu_seqlens_cpu: unused, kept for interface compatibility
        chunk_size: block size

    Returns:
        dh:  [B, NT, H, K, V] — gradient at start of each chunk
        dh0: [N, H, K, V] or None — initial state gradient
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C
    N = B if cu_seqlens_cpu is None else cu_seqlens_cpu.shape[-1] - 1
    assert T % C == 0, "T must be a multiple of chunk_size for chunk_bwd_dh"

    # Fast path: fixed-length sequences use lax.scan (avoids XLA loop unrolling)
    if cu_seqlens_cpu is None:
        return _chunk_bwd_dh_scan(
            q, do, g, g_gamma, gk, dht,
            scale, output_dh0, states_in_fp32,
            C, B, T, H, K, V, NT,
        )

    is_varlen = cu_seqlens_cpu is not None

    q = q.reshape(-1, H, K)
    do = do.reshape(-1, H, V)
    gk = gk.reshape(-1, H, K) if gk is not None else None

    dh_all = jnp.zeros([B, NT, H, K, V], dtype=jnp.float32)
    dh0_all = (
        jnp.zeros([N, H, K, V], dtype=jnp.float32)
        if (h0 is not None or dht is not None)
        else None
    )

    for i_n in range(N):
        if not is_varlen:
            bos = i_n * T
            eos = (i_n + 1) * T
        else:
            bos = int(cu_seqlens_cpu[i_n])
            eos = int(cu_seqlens_cpu[i_n + 1])

        NT_seq = (eos - bos) // C
        dh = jnp.zeros((H, K, V), dtype=jnp.float32)
        if dht is not None:
            dh = dh + dht[i_n].astype(jnp.float32)

        for i_t in range(NT_seq - 1, -1, -1):
            bi = 0 if is_varlen else i_n
            ti = bos // C + i_t if is_varlen else i_t
            dh_all = dh_all.at[bi, ti].set(dh)

            b_q = q[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, K]
            b_do = do[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, V]

            if gk is not None:
                b_gk = gk[bos + i_t * C : bos + (i_t + 1) * C]  # [C, H, K]
                b_gk_last = b_gk[-1]  # [H, K]
                b_q_hat = b_q * exp(b_gk) * scale  # [C, H, K]
                dh = dh * exp(b_gk_last[:, :, None])
            else:
                b_q_hat = b_q * scale

            # contract over C (dim 0) and H (dim 1): [C,H,K]^T @ [C,H,V] -> [H,K,V]
            dh = dh + lax.dot_general(
                b_q_hat,
                b_do,
                dimension_numbers=(((0,), (0,)), ((1,), (1,))),
                precision=lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )

        if dh0_all is not None:
            dh0_all = dh0_all.at[i_n].set(dh)

    return dh_all, dh0_all


def _chunk_bwd_dh_kernel(
    q_ref,          # [1, 1, BT, BK]
    do_ref,         # [1, 1, BT, BV]
    dht_ref,        # [N, 1, BK, BV]
    gk_ref,         # [1, 1, BT, BK]
    g_ref,          # [1, 1, BT]
    g_gamma,        # [H]
    cu_seqlens_ref, # [num_seq + 1]
    chunk_to_seq,   # [NT]
    dh_ref,         # [1, 1, BK, BV]
    dh0_ref,        # [N, 1, BK, BV] or None
    carry_ref,      # scratch VMEM (BK, BV)
    *,
    BT: int,
    NT: int,
    scale: float,
):
    BK = q_ref.shape[3]
    BV = do_ref.shape[3]

    i_c = pl.program_id(3)  # chunk index (0 = last chunk in time)
    i_t = NT - 1 - i_c      # global chunk index (forward order)
    t0 = i_t * BT            # global time offset

    # Load carry from previous step, or init zeros for first step
    b_dh = lax.cond(
        i_c == 0,
        lambda _: jnp.zeros((BK, BV), dtype=jnp.float32),
        lambda _: carry_ref[...].astype(jnp.float32),
        operand=None,
    )

    if g_gamma is not None:
        head_index = pl.program_id(0)
        # tpu not support scalar bf16 mul
        b_g_ramp = (g_gamma[head_index].astype(jnp.float32) * (jnp.arange(0, BT) + 1)).astype(g_gamma.dtype)  # [BT]

    seq_idx = chunk_to_seq[i_t]
    eos = cu_seqlens_ref[seq_idx + 1]

    # reset dh at sequence boundary (last chunk of each sequence)
    is_last_chunk = (t0 + BT >= eos)

    def reset_state(_):
        if dht_ref is not None:
            return dht_ref[seq_idx, 0].astype(jnp.float32)
        return jnp.zeros((BK, BV), dtype=jnp.float32)

    b_dh = lax.cond(is_last_chunk, reset_state, lambda _: b_dh, operand=None)

    # store dh (after reset, before compute)
    dh_ref[0, 0] = b_dh.astype(dh_ref.dtype)

    b_q = q_ref[(0, 0, slice(None), slice(None))]    # [BT, BK]
    b_do = do_ref[(0, 0, slice(None), slice(None))]   # [BT, BV]
    b_q = (b_q * scale).astype(b_q.dtype)

    # scalar gate (g)
    if g_ref is not None:
        b_g_scalar = g_ref[0, 0, slice(None)]  # [BT]
        b_g_scalar_last = b_g_scalar[BT - 1]
        b_dh *= exp(b_g_scalar_last)
        b_q = (b_q * exp(b_g_scalar)[:, None]).astype(b_q.dtype)

    # per-head fixed decay (g_gamma)
    if g_gamma is not None:
        # tpu not support scalar bf16 mul
        b_g_last = (g_gamma[head_index].astype(jnp.float32) * jnp.minimum(BT, eos - t0)).astype(g_gamma.dtype)
        b_dh *= exp(b_g_last)
        b_q = (b_q * exp(b_g_ramp)[:, None]).astype(b_q.dtype)

    # per-K-dim gate (gk)
    if gk_ref is not None:
        b_gk = gk_ref[(0, 0, slice(None), slice(None))]  # [BT, BK]
        g_last = b_gk[BT - 1, :]
        b_dh = b_dh * exp(g_last)[:, None]  # [BK, BV] * [BK, 1]
        b_q = (b_q * exp(b_gk)).astype(b_q.dtype)

    b_dh = b_dh + jax.lax.dot(
        b_q.astype(jnp.float32).T, b_do.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    # write dh0 at sequence start
    bos = cu_seqlens_ref[seq_idx]

    @pl.when(t0 == bos)
    def _():
        if dh0_ref is not None:
            dh0_ref[seq_idx, 0] = b_dh.astype(dh0_ref.dtype)

    # Save carry for next step
    carry_ref[...] = b_dh.astype(jnp.float32)


@functools.partial(
    jax.jit,
    static_argnames=[
        "scale",
        "output_dh0",
        "chunk_size",
        "states_in_fp32",
    ],
)
def chunk_bwd_dh_kernel(
    q: jax.Array,                # [B, T, H, K]
    k: jax.Array,                # [B, T, H, K] (unused but kept for API compatibility)
    v: jax.Array,                # [B, T, H, V] (unused but kept for API compatibility)
    g: jax.Array | None = None,  # [B, T, H]
    g_gamma: jax.Array | None = None,  # [H]
    gk: jax.Array | None = None, # [B, T, H, K]
    do: jax.Array = None,        # [B, T, H, V]
    dht: jax.Array | None = None,# [N, H, K, V]
    scale: float = 1.0,
    output_dh0: bool = False,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 128,
    states_in_fp32: bool = False,
):
    BK, BV, BT = 128, 128, chunk_size
    B, T, H, K = q.shape
    V = do.shape[-1]
    T_sum = B * T
    NT = T_sum // BT

    assert K % 128 == 0, "K % 128 must equal to 0."
    assert V % 128 == 0, "V % 128 must equal to 0."
    assert T % chunk_size == 0, "T mod chunk_size must equal to 0."

    if cu_seqlens_dev is None:
        cu_seqlens_dev = jnp.arange(T_sum + 1, step=T)
    chunk_to_seq = _build_chunk_map(cu_seqlens=cu_seqlens_dev, T_sum=T_sum, BT=BT)
    N = len(cu_seqlens_dev) - 1

    # Reshape to (H, NT, BT, X) — one chunk per grid step
    q = jnp.reshape(q, (T_sum, H, K)).transpose(1, 0, 2).reshape(H, NT, BT, K)
    do = jnp.reshape(do, (T_sum, H, V)).transpose(1, 0, 2).reshape(H, NT, BT, V)
    if gk is not None:
        gk = jnp.reshape(gk, (T_sum, H, K)).transpose(1, 0, 2).reshape(H, NT, BT, K)
    if g is not None:
        g = jnp.reshape(g, (T_sum, H)).transpose(1, 0).reshape(H, NT, BT)

    grid = (H, pl.cdiv(K, BK), pl.cdiv(V, BV), NT)

    # Reversed chunk order: c=0 → last chunk (processed first in backward)
    def idx_map_K(h, k, v, c): return h, NT - 1 - c, 0, k
    def idx_map_V(h, k, v, c): return h, NT - 1 - c, 0, v
    def idx_map_state(h, k, v, c): return 0, h, k, v

    dtype_out = q.dtype if not states_in_fp32 else jnp.float32

    out_shape = [
        jax.ShapeDtypeStruct(shape=(NT, H, K, V), dtype=dtype_out),
    ]
    out_specs = [
        pl.BlockSpec((1, 1, BK, BV), lambda h, k, v, c: (NT - 1 - c, h, k, v)),
    ]
    if output_dh0:
        out_shape.append(jax.ShapeDtypeStruct(shape=(N, H, K, V), dtype=dtype_out))
        out_specs.append(pl.BlockSpec((N, 1, BK, BV), idx_map_state))
    else:
        out_shape.append(None)
        out_specs.append(None)

    in_specs = [
        pl.BlockSpec((1, 1, BT, BK), idx_map_K),   # q
        pl.BlockSpec((1, 1, BT, BV), idx_map_V),    # do
        pl.BlockSpec((N, 1, BK, BV), idx_map_state) if dht is not None else None,  # dht
        pl.BlockSpec((1, 1, BT, BK), idx_map_K) if gk is not None else None,  # gk
        pl.BlockSpec((1, 1, BT), lambda h, k, v, c: (h, NT - 1 - c, 0)) if g is not None else None,  # g
        pl.BlockSpec(memory_space=pltpu.SMEM) if g_gamma is not None else None,  # g_gamma
        pl.BlockSpec(memory_space=pltpu.SMEM),      # cu_seqlens
        pl.BlockSpec(memory_space=pltpu.SMEM),      # chunk_to_seq
    ]

    kernel = functools.partial(_chunk_bwd_dh_kernel, BT=BT, NT=NT, scale=scale)

    interpret = get_interpret()
    dh_all, dh0 = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=[pltpu.VMEM((BK, BV), jnp.float32)],
        ),
        out_shape=out_shape,
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=32 * 1024 * 1024,
            disable_bounds_checks=True,
        ),
    )(q, do, dht, gk, g, g_gamma, cu_seqlens_dev, chunk_to_seq)

    dh_all = dh_all.reshape(B, -1, H, K, V)
    dh0 = dh0.reshape(N, H, K, V) if dh0 is not None else None
    return dh_all, dh0

__all__ = export_public(globals())
