# FP8 GMM Kernel-Evolve Port Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port the FP8 block-wise GMM Pallas kernel from tokamax (ant-pretrain PR #220) into kernel-evolve as two self-contained, evolvable kernels (forward gmm + backward tgmm).

**Architecture:** Each kernel is a self-contained Python file with inlined helpers from tokamax (no external dependencies beyond JAX/Pallas). The EVOLVE-BLOCK wraps the inner Pallas kernel body. Reference implementations use plain JAX `jnp.dot` on dequantized FP8 inputs. Two YAML configs drive independent evolution runs.

**Tech Stack:** JAX, Pallas (Mosaic TPU), float8_e4m3fn, kernel-evolve framework

**Pallas TPU constraints:** Block shape last two dims must be divisible by 8×128 (or equal array dims). No dynamic_slice inside kernels. VMEM limits: 16MB v4, 128MB v5e, 64MB v6e/v7x. Use static loops, masked operations, coalesced memory access.

---

### Task 1: Create Forward GMM FP8 Reference (`gmm_fp8_fwd_ref.py`)

**Files:**
- Create: `kernel-evolve/kernels/gmm_fp8_fwd_ref.py`

**Step 1: Write the reference file**

This file must export `reference_fn(**kwargs)` that takes shape parameters and returns a JAX array. It generates FP8 block-wise quantized inputs (same RNG seeds as the kernel template), dequantizes them, and computes the grouped matmul using plain JAX.

```python
"""Reference FP8 grouped matrix multiplication for correctness comparison."""

import jax
import jax.numpy as jnp
from collections import namedtuple

QArray = namedtuple("QArray", ["qvalue", "scale"])

TILE_SIZE = 128


def _blockwise_quantize(x, block_shape, key):
    """Quantize a tensor to FP8 E4M3 with block-wise scales.

    Args:
        x: Input tensor in BF16/F32.
        block_shape: Tuple of block sizes for each dim. Use 1 for "per-row" dims.
        key: Unused, kept for API compat.

    Returns:
        QArray with qvalue (float8_e4m3fn) and scale (float32).
    """
    orig_shape = x.shape
    # Reshape to expose blocks: e.g. [M, K] with block_shape=(1, 128)
    # becomes [M, K//128, 128], then absmax over last dims
    reshaped = x
    new_shape = []
    reduce_axes = []
    axis_idx = 0
    for dim_size, blk in zip(orig_shape, block_shape):
        if blk == 1 or blk >= dim_size:
            new_shape.append(dim_size)
            reduce_axes.append(axis_idx)
            axis_idx += 1
        else:
            new_shape.extend([dim_size // blk, blk])
            reduce_axes.append(axis_idx + 1)
            axis_idx += 2
    reshaped = x.reshape(new_shape)
    absmax = jnp.max(jnp.abs(reshaped), axis=reduce_axes, keepdims=True)
    fp8_max = jnp.float32(448.0)  # E4M3 max
    scale = (absmax / fp8_max).astype(jnp.float32)
    scale = jnp.maximum(scale, jnp.float32(1e-12))
    qvalue = (reshaped / scale).astype(jnp.float8_e4m3fn)
    # Reshape back
    qvalue = qvalue.reshape(orig_shape)
    # Scale shape: remove the block-interior dims
    scale_shape = []
    for dim_size, blk in zip(orig_shape, block_shape):
        if blk == 1 or blk >= dim_size:
            scale_shape.append(dim_size)
        else:
            scale_shape.append(dim_size // blk)
    scale = scale.reshape(scale_shape)
    return QArray(qvalue=qvalue, scale=scale)


def _dequantize(q):
    """Dequantize a QArray back to float32."""
    qvalue = q.qvalue.astype(jnp.float32)
    scale = q.scale
    # Broadcast scale to match qvalue shape
    if qvalue.ndim == 2 and scale.ndim == 2:
        # [M, K] with scale [M, K//blk] -> repeat scale
        ratio = qvalue.shape[-1] // scale.shape[-1]
        scale = jnp.repeat(scale, ratio, axis=-1)
        if qvalue.shape[0] != scale.shape[0]:
            ratio0 = qvalue.shape[0] // scale.shape[0]
            scale = jnp.repeat(scale, ratio0, axis=0)
    elif qvalue.ndim == 3 and scale.ndim == 3:
        # [G, K, N] with scale [G, K//blk, N//blk]
        for ax in range(3):
            if qvalue.shape[ax] != scale.shape[ax]:
                ratio = qvalue.shape[ax] // scale.shape[ax]
                scale = jnp.repeat(scale, ratio, axis=ax)
    return qvalue * scale


def _generate_inputs(M, K, N, num_groups):
    """Generate FP8 block-wise quantized inputs (deterministic)."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    lhs_bf16 = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
    rhs_bf16 = jax.random.normal(k2, (num_groups, K, N), dtype=jnp.bfloat16)

    lhs_q = _blockwise_quantize(lhs_bf16, (1, TILE_SIZE), k1)
    # Weight: 128x128 blocks on K and N dims, group dim untouched
    rhs_q = _blockwise_quantize(rhs_bf16, (1, TILE_SIZE, TILE_SIZE), k2)

    group_sizes = jnp.full((num_groups,), M // num_groups, dtype=jnp.int32)
    return lhs_q, rhs_q, group_sizes


def simple_compute(M=4096, K=7168, N=2048, num_groups=8):
    """Reference grouped matmul using dequantized FP8 inputs."""
    lhs_q, rhs_q, group_sizes = _generate_inputs(M, K, N, num_groups)
    lhs = _dequantize(lhs_q)
    rhs = _dequantize(rhs_q)

    # Compute per-group matmul
    offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(group_sizes)])
    results = []
    for g in range(num_groups):
        start = offsets[g]
        end = offsets[g + 1]
        lhs_slice = jax.lax.dynamic_slice(lhs, (start, 0), (group_sizes[g], K))
        out_g = jnp.dot(lhs_slice.astype(jnp.float32), rhs[g].astype(jnp.float32))
        results.append(out_g)
    return jnp.concatenate(results, axis=0).astype(jnp.bfloat16)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
```

**Step 2: Commit**

```bash
git add kernel-evolve/kernels/gmm_fp8_fwd_ref.py
git commit -m "feat(kernel-evolve): add FP8 GMM forward reference implementation"
```

---

### Task 2: Create Forward GMM FP8 Kernel Template (`gmm_fp8_fwd.py`)

**Files:**
- Create: `kernel-evolve/kernels/gmm_fp8_fwd.py`

**Step 1: Write the kernel template**

This is the main evolvable kernel. It inlines all necessary helpers from tokamax and wraps the inner kernel body in EVOLVE-BLOCK markers. The `optimized_compute()` entry point generates the same FP8 inputs as the reference (same RNG seeds) and runs the Pallas kernel.

Key components:
1. `QArray` namedtuple + `_blockwise_quantize()` (same as ref)
2. `make_group_metadata()` (from tokamax `pallas_mosaic_tpu_kernel.py`)
3. `_get_store_mask()` (from tokamax)
4. `_scale_out_by_scale()` (from tokamax)
5. Inner `kernel()` with EVOLVE-BLOCK markers
6. `gmm()` function that builds grid/specs and calls `pallas_call`
7. `optimized_compute()` entry point

The EVOLVE-BLOCK wraps the inner kernel body — the part that loads FP8 tiles, performs the dot product with scale application, and accumulates results. This is approximately 60 lines of code that the LLM will mutate.

```python
"""FP8 block-wise grouped matrix multiplication Pallas kernel for TPU.

Self-contained port from tokamax (primatrix/tokamax) for evolutionary
optimization via kernel-evolve. Forward pass: lhs[sizes[i-1]:sizes[i], :] @ rhs[i]
with FP8 E4M3 block-wise quantization (1x128 activation, 128x128 weight).
"""

import functools
import json
from collections import namedtuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

QArray = namedtuple("QArray", ["qvalue", "scale"])

TILE_SIZE = 128


# ---------------------------------------------------------------------------
# FP8 block-wise quantization (self-contained, no qwix dependency)
# ---------------------------------------------------------------------------

def _blockwise_quantize(x, block_shape, key):
    """Quantize tensor to FP8 E4M3 with block-wise scales."""
    orig_shape = x.shape
    new_shape = []
    reduce_axes = []
    axis_idx = 0
    for dim_size, blk in zip(orig_shape, block_shape):
        if blk == 1 or blk >= dim_size:
            new_shape.append(dim_size)
            reduce_axes.append(axis_idx)
            axis_idx += 1
        else:
            new_shape.extend([dim_size // blk, blk])
            reduce_axes.append(axis_idx + 1)
            axis_idx += 2
    reshaped = x.reshape(new_shape)
    absmax = jnp.max(jnp.abs(reshaped), axis=reduce_axes, keepdims=True)
    fp8_max = jnp.float32(448.0)
    scale = jnp.maximum(absmax / fp8_max, jnp.float32(1e-12)).astype(jnp.float32)
    qvalue = (reshaped / scale).astype(jnp.float8_e4m3fn)
    qvalue = qvalue.reshape(orig_shape)
    scale_shape = []
    for dim_size, blk in zip(orig_shape, block_shape):
        if blk == 1 or blk >= dim_size:
            scale_shape.append(dim_size)
        else:
            scale_shape.append(dim_size // blk)
    scale = scale.reshape(scale_shape)
    return QArray(qvalue=qvalue, scale=scale)


# ---------------------------------------------------------------------------
# Group metadata (from tokamax pallas_mosaic_tpu_kernel.py)
# ---------------------------------------------------------------------------

def make_group_metadata(*, group_sizes, m, tm, start_group, num_nonzero_groups,
                        visit_empty_groups):
    """Create metadata for grouped matmul grid execution."""
    num_groups = group_sizes.shape[0]
    end_group = start_group + num_nonzero_groups - 1
    group_ends = jnp.cumsum(group_sizes)
    group_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends])

    rounded_group_ends = ((group_ends + tm - 1) // tm * tm).astype(jnp.int32)
    group_starts = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]])
    rounded_group_starts = group_starts // tm * tm
    rounded_group_sizes = rounded_group_ends - rounded_group_starts
    rounded_group_sizes = jnp.where(group_sizes == 0, 0, rounded_group_sizes)
    group_tiles = rounded_group_sizes // tm

    if visit_empty_groups:
        group_tiles = jnp.where(group_sizes == 0, 1, group_tiles)

    if m % tm != 0:
        raise NotImplementedError(f"{m=} must be divisible by tile size ({tm}).")

    tiles_m = m // tm
    group_ids = jnp.repeat(
        jnp.arange(num_groups, dtype=jnp.int32),
        group_tiles,
        total_repeat_length=tiles_m + num_groups - 1,
    )

    partial_tile_mask = ((group_offsets[:-1] % tm) == 0) | (group_sizes == 0)
    if visit_empty_groups:
        partial_tile_mask = jnp.where(group_sizes == 0, 0, partial_tile_mask)
    partial_tile_ids = jnp.where(partial_tile_mask, tiles_m, group_offsets[:-1] // tm)
    tile_visits = (
        jnp.histogram(partial_tile_ids, bins=tiles_m, range=(0, tiles_m - 1))[0] + 1
    )
    m_tile_ids = jnp.repeat(
        jnp.arange(tiles_m, dtype=jnp.int32),
        tile_visits.astype(jnp.int32),
        total_repeat_length=tiles_m + num_groups - 1,
    )

    # Account for sharding
    first_tile_in_shard = (group_ids < start_group).sum()
    group_ids = jnp.roll(group_ids, shift=-first_tile_in_shard, axis=0)
    m_tile_ids = jnp.roll(m_tile_ids, shift=-first_tile_in_shard, axis=0)

    iota = jnp.arange(num_groups, dtype=jnp.int32)
    active_group_mask = (iota <= end_group) & (iota >= start_group)
    group_tiles = jnp.where(active_group_mask, group_tiles, 0)
    num_tiles = group_tiles.sum()
    return (group_offsets, group_ids, m_tile_ids), num_tiles


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def _get_store_mask(*, grid_id, group_metadata, tm, tn):
    """Mask for rows belonging to the current group in the current tile."""
    group_offsets, group_ids, m_tile_ids = group_metadata
    group_id = group_ids[grid_id]
    group_start = group_offsets[group_id]
    group_end = group_offsets[group_id + 1]
    m_id = m_tile_ids[grid_id] * tm
    iota = jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0) + m_id
    return (iota >= group_start) & (iota < group_end)


def _scale_out_by_scale(out, scales):
    """Apply block-wise scales to output tensor."""
    if any(s1 % s2 != 0 for s1, s2 in zip(out.shape, scales.shape)):
        raise ValueError(f"Cannot broadcast {scales.shape=} to {out.shape=}")
    for ax, (s1, s2) in enumerate(zip(out.shape, scales.shape)):
        scales = jnp.tile(
            scales, [s1 // s2 if i == ax else 1 for i in range(scales.ndim)]
        )
    return out * scales


def _quant_block_spec_gmm(qvalue, scale, block_spec, reduction_axis):
    """Build BlockSpec pair for QArray in gmm (simplified from tokamax)."""
    eps_list = [pl.cdiv(qs, ss) for qs, ss in zip(qvalue.shape, scale.shape)]
    tile_sizes = block_spec.block_shape

    # Compute subchannel_iters from reduction axis
    eps_red = eps_list[reduction_axis]
    tk = tile_sizes[reduction_axis]
    subchannel_iters = max(1, tk // eps_red) if tk is not None else 1

    # Build scale index map: map tile indices to scale tile indices
    def _scale_index_map(*args):
        idxs = list(block_spec.index_map(*args))
        result = []
        for i, idx in enumerate(idxs):
            eps = eps_list[i]
            tile_size = tile_sizes[i]
            if eps == qvalue.shape[i]:
                result.append(0)
            elif tile_size is not None and eps <= tile_size:
                result.append(idx)
            else:
                result.append(idx * tile_size // eps if tile_size else idx // eps)
        return result

    # Compute scale tile sizes
    LANES = 128
    SUBLANES = 8  # Conservative for v4
    min_addr = ([1] * qvalue.ndim + [SUBLANES, LANES])[-qvalue.ndim:]
    scale_tile_sizes = []
    for i, (eps, ts, s, mas) in enumerate(zip(eps_list, tile_sizes, qvalue.shape, min_addr)):
        if ts is None:
            scale_tile_sizes.append(None)
        elif eps == s:
            scale_tile_sizes.append(mas)
        elif eps > ts:
            scale_tile_sizes.append(mas)
        elif eps == 1:
            scale_tile_sizes.append(max(1, ts // eps) if ts else 1)
        else:
            scale_tile_sizes.append(max(1, ts // eps) * mas)
    scale_block_spec = pl.BlockSpec(scale_tile_sizes, _scale_index_map)

    # Inflate scales so they are individually addressable
    for ax in range(scale.ndim):
        eps = eps_list[ax]
        ts = tile_sizes[ax]
        if ts is None or eps == qvalue.shape[ax]:
            true_per_tile = 1
            sts = min_addr[ax]
        elif eps > ts:
            true_per_tile = 1
            sts = 1
        elif eps == 1:
            true_per_tile = ts
            sts = ts
        else:
            true_per_tile = ts // eps
            sts = true_per_tile * min_addr[ax]
        inflation = sts // true_per_tile
        if inflation > 1:
            scale = jnp.repeat(scale, inflation, axis=ax)

    return qvalue, scale, block_spec, scale_block_spec, subchannel_iters


# ---------------------------------------------------------------------------
# Forward GMM Pallas kernel
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=["tiling", "num_groups", "interpret"])
def gmm(lhs_qvalue, lhs_scale, rhs_qvalue, rhs_scale, group_sizes,
        tiling=(128, 128, 128), num_groups=8, interpret=False):
    """FP8 block-wise grouped matrix multiplication.

    Args:
        lhs_qvalue: [M, K] float8_e4m3fn
        lhs_scale: [M, K//128] float32
        rhs_qvalue: [num_groups, K, N] float8_e4m3fn
        rhs_scale: [num_groups, K//128, N//128] float32
        group_sizes: [num_groups] int32
        tiling: (tm, tk, tn) tile sizes
        num_groups: Number of groups (static)
        interpret: Run in interpret mode for debugging

    Returns:
        [M, N] bfloat16 output
    """
    m, k = lhs_qvalue.shape
    n = rhs_qvalue.shape[2]
    tm, tk, tn = tiling
    tiles_k = pl.cdiv(k, tk)
    tiles_n = pl.cdiv(n, tn)
    out_dtype = jnp.bfloat16

    group_offset = jnp.array([0], dtype=jnp.int32)
    group_metadata, num_active_tiles = make_group_metadata(
        group_sizes=group_sizes, m=m, tm=tm, start_group=0,
        num_nonzero_groups=num_groups, visit_empty_groups=False,
    )

    # Build BlockSpecs
    def lhs_index_map(n_i, grid_id, k_i, group_metadata, group_offset):
        _, _, m_tile_ids = group_metadata
        return m_tile_ids[grid_id], k_i

    def rhs_index_map(n_i, grid_id, k_i, group_metadata, group_offset):
        _, group_ids, _ = group_metadata
        return group_ids[grid_id] - group_offset[0], k_i, n_i

    def out_index_map(n_i, grid_id, k_i, group_metadata, group_offset):
        _, _, m_tile_ids = group_metadata
        return m_tile_ids[grid_id], n_i

    lhs_block_spec = pl.BlockSpec((tm, tk), lhs_index_map)
    rhs_block_spec = pl.BlockSpec((None, tk, tn), rhs_index_map)
    out_block_spec = pl.BlockSpec((tm, tn), out_index_map)

    # Process QArray block specs
    lhs_qv, lhs_sc, lhs_qv_spec, lhs_sc_spec, lhs_sci = _quant_block_spec_gmm(
        lhs_qvalue, lhs_scale, lhs_block_spec, reduction_axis=1)
    rhs_qv, rhs_sc, rhs_qv_spec, rhs_sc_spec, rhs_sci = _quant_block_spec_gmm(
        rhs_qvalue, rhs_scale, rhs_block_spec, reduction_axis=1)
    subchannel_iters = max(lhs_sci, rhs_sci)

    def _kernel(group_metadata_refs, group_offset_ref,
                lhs_qv_ref, lhs_sc_ref, rhs_qv_ref, rhs_sc_ref,
                out_ref, acc_scratch):
        # EVOLVE-BLOCK-START
        grid_id, k_i = pl.program_id(1), pl.program_id(2)

        @pl.when(k_i == 0)
        def _zero_acc():
            acc_scratch[...] = jnp.zeros_like(acc_scratch)

        # Subchannel iteration for block-wise FP8 scale handling.
        # Each subchannel iteration processes a sub-tile of the K dimension
        # where a single scale value applies.
        sc_tile = tk // subchannel_iters

        def accum(is_last_k_tile):
            iterations = subchannel_iters
            if is_last_k_tile:
                iterations = pl.cdiv(k - tk * (tiles_k - 1), sc_tile)

            for it in range(iterations):
                # Load sub-tile of lhs qvalue and scale
                lhs_qv_sub = lhs_qv_ref[:, it * sc_tile:(it + 1) * sc_tile]
                lhs_sc_sub = lhs_sc_ref[...]  # Scale for this sub-tile

                # Load sub-tile of rhs qvalue and scale
                rhs_qv_sub = rhs_qv_ref[:, it * sc_tile:(it + 1) * sc_tile, :]
                rhs_sc_sub = rhs_sc_ref[...]  # Scale for this sub-tile

                # Mask if this is the last sub-tile and K doesn't divide evenly
                is_last_subtile = (
                    is_last_k_tile and (tiles_k - 1) * tk + (it + 1) * sc_tile >= k
                )
                k_rem = (k % tk) % sc_tile
                if is_last_subtile and k_rem != 0:
                    iota_lhs = lax.broadcasted_iota(jnp.int32, lhs_qv_sub.shape, 1)
                    lhs_qv_sub = jnp.where(iota_lhs < k_rem, lhs_qv_sub, 0)
                    iota_rhs = lax.broadcasted_iota(jnp.int32, rhs_qv_sub.shape, 1)
                    rhs_qv_sub = jnp.where(iota_rhs < k_rem, rhs_qv_sub, 0)

                # FP8 dot product: [tm, sc_tile] @ [sc_tile, tn] -> [tm, tn] f32
                out = jax.lax.dot_general(
                    lhs_qv_sub, rhs_qv_sub,
                    dimension_numbers=(((1,), (1,)), ((), ())),
                    precision=(jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
                    preferred_element_type=jnp.float32,
                )

                # Apply block-wise scales
                out = _scale_out_by_scale(out, lhs_sc_sub)
                out = _scale_out_by_scale(out, rhs_sc_sub)

                # Accumulate
                acc_scratch[...] += out.astype(jnp.float32)

                # Store on last sub-tile
                if is_last_subtile:
                    mask = _get_store_mask(
                        grid_id=grid_id, group_metadata=group_metadata_refs,
                        tm=tm, tn=tn,
                    )
                    acc = acc_scratch[...]
                    acc = jax.lax.select(mask, acc, out_ref[...].astype(jnp.float32))
                    out_ref[...] = acc.astype(out_dtype)

        lax.cond(k_i == tiles_k - 1, lambda: accum(True), lambda: accum(False))
        # EVOLVE-BLOCK-END

    out = pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[lhs_qv_spec, lhs_sc_spec, rhs_qv_spec, rhs_sc_spec],
            out_specs=out_block_spec,
            grid=(tiles_n, num_active_tiles, tiles_k),
            scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary")
        ),
        interpret=interpret,
        name="gmm_fp8_fwd",
    )(group_metadata, group_offset, lhs_qv, lhs_sc, rhs_qv, rhs_sc)
    return out


# ---------------------------------------------------------------------------
# Entry point for kernel-evolve evaluator
# ---------------------------------------------------------------------------

def _generate_inputs(M, K, N, num_groups):
    """Generate FP8 block-wise quantized inputs (deterministic)."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    lhs_bf16 = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
    rhs_bf16 = jax.random.normal(k2, (num_groups, K, N), dtype=jnp.bfloat16)
    lhs_q = _blockwise_quantize(lhs_bf16, (1, TILE_SIZE), k1)
    rhs_q = _blockwise_quantize(rhs_bf16, (1, TILE_SIZE, TILE_SIZE), k2)
    group_sizes = jnp.full((num_groups,), M // num_groups, dtype=jnp.int32)
    return lhs_q, rhs_q, group_sizes


def optimized_compute(M=4096, K=7168, N=2048, num_groups=8):
    lhs_q, rhs_q, group_sizes = _generate_inputs(M, K, N, num_groups)
    return gmm(
        lhs_q.qvalue, lhs_q.scale,
        rhs_q.qvalue, rhs_q.scale,
        group_sizes,
        tiling=(TILE_SIZE, TILE_SIZE, TILE_SIZE),
        num_groups=num_groups,
    )
```

**Note:** The `_quant_block_spec_gmm` helper is a simplified version of tokamax's
`quant_block_spec()`. It handles building separate BlockSpecs for qvalue and scale
arrays, computing scale inflation for TPU addressability, and determining
`subchannel_iters`. The actual tokamax code uses a `QArray` dataclass registered as
a JAX pytree; here we pass qvalue and scale as separate arrays since we can't use
the qwix dependency.

**Step 2: Commit**

```bash
git add kernel-evolve/kernels/gmm_fp8_fwd.py
git commit -m "feat(kernel-evolve): add FP8 GMM forward Pallas kernel template"
```

---

### Task 3: Create Backward tGMM FP8 Reference (`gmm_fp8_bwd_ref.py`)

**Files:**
- Create: `kernel-evolve/kernels/gmm_fp8_bwd_ref.py`

**Step 1: Write the reference file**

The backward tGMM computes `lhs[:, slice]^T @ rhs[slice, :]` accumulated per group,
producing output shape `[num_groups, K, N]`.

```python
"""Reference FP8 transposed grouped matrix multiplication for correctness comparison."""

import jax
import jax.numpy as jnp
from collections import namedtuple

QArray = namedtuple("QArray", ["qvalue", "scale"])

TILE_SIZE = 128


def _blockwise_quantize(x, block_shape, key):
    """Quantize tensor to FP8 E4M3 with block-wise scales."""
    orig_shape = x.shape
    new_shape = []
    reduce_axes = []
    axis_idx = 0
    for dim_size, blk in zip(orig_shape, block_shape):
        if blk == 1 or blk >= dim_size:
            new_shape.append(dim_size)
            reduce_axes.append(axis_idx)
            axis_idx += 1
        else:
            new_shape.extend([dim_size // blk, blk])
            reduce_axes.append(axis_idx + 1)
            axis_idx += 2
    reshaped = x.reshape(new_shape)
    absmax = jnp.max(jnp.abs(reshaped), axis=reduce_axes, keepdims=True)
    fp8_max = jnp.float32(448.0)
    scale = jnp.maximum(absmax / fp8_max, jnp.float32(1e-12)).astype(jnp.float32)
    qvalue = (reshaped / scale).astype(jnp.float8_e4m3fn)
    qvalue = qvalue.reshape(orig_shape)
    scale_shape = []
    for dim_size, blk in zip(orig_shape, block_shape):
        if blk == 1 or blk >= dim_size:
            scale_shape.append(dim_size)
        else:
            scale_shape.append(dim_size // blk)
    scale = scale.reshape(scale_shape)
    return QArray(qvalue=qvalue, scale=scale)


def _dequantize(q):
    """Dequantize a QArray back to float32."""
    qvalue = q.qvalue.astype(jnp.float32)
    scale = q.scale
    for ax in range(qvalue.ndim):
        if qvalue.shape[ax] != scale.shape[ax]:
            ratio = qvalue.shape[ax] // scale.shape[ax]
            scale = jnp.repeat(scale, ratio, axis=ax)
    return qvalue * scale


def _generate_inputs(M, K, N, num_groups):
    """Generate FP8 block-wise quantized inputs for tGMM (deterministic).

    tGMM inputs:
      lhs: [K, M] (transposed activation) with 1x128 blocks
      rhs: [M, N] (gradient) with 1x128 blocks
    """
    key = jax.random.PRNGKey(43)  # Different seed from fwd
    k1, k2 = jax.random.split(key)
    # lhs is [K, M] - transposed from forward's [M, K]
    lhs_bf16 = jax.random.normal(k1, (K, M), dtype=jnp.bfloat16)
    # rhs is [M, N] - the gradient
    rhs_bf16 = jax.random.normal(k2, (M, N), dtype=jnp.bfloat16)

    lhs_q = _blockwise_quantize(lhs_bf16, (1, TILE_SIZE), k1)
    rhs_q = _blockwise_quantize(rhs_bf16, (1, TILE_SIZE), k2)

    group_sizes = jnp.full((num_groups,), M // num_groups, dtype=jnp.int32)
    return lhs_q, rhs_q, group_sizes


def simple_compute(M=4096, K=7168, N=2048, num_groups=8):
    """Reference tGMM: lhs[:, slice]^T @ rhs[slice, :] per group."""
    lhs_q, rhs_q, group_sizes = _generate_inputs(M, K, N, num_groups)
    # Dequantize
    lhs = _dequantize(lhs_q)  # [K, M]
    rhs = _dequantize(rhs_q)  # [M, N]

    # lhs is [K, M], for tGMM we compute lhs[:, slice].T @ rhs[slice, :]
    # which is lhs_slice^T @ rhs_slice = [K, group_M]^T ... wait
    # Actually tGMM: lhs is [K, M], we load [M, K] tiles (transposed in memory)
    # and compute lhs_tile.T @ rhs_tile for each m-tile in the group.
    # Result is [num_groups, K, N].

    offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(group_sizes)])
    results = []
    for g in range(num_groups):
        start = offsets[g]
        gs = group_sizes[g]
        # lhs[:, start:start+gs] is [K, gs], transposed to [gs, K]
        lhs_slice = jax.lax.dynamic_slice(lhs, (0, start), (K, gs))  # [K, gs]
        rhs_slice = jax.lax.dynamic_slice(rhs, (start, 0), (gs, N))  # [gs, N]
        # [K, gs] @ [gs, N] -> [K, N] but we need lhs_slice^T @ rhs_slice
        # Actually the memory layout is: lhs [K, M] loaded as [M, K] tiles
        # tGMM computes [M,K]^T @ [M,N] per group = [K,M_g]^T...
        # Simpler: lhs_slice.T = [gs, K].T = [K, gs]
        # We want: lhs_col_slice.T @ rhs_row_slice = [K, gs] @ [gs, N] = [K, N]
        out_g = jnp.dot(lhs_slice.astype(jnp.float32), rhs_slice.astype(jnp.float32))
        results.append(out_g)
    return jnp.stack(results, axis=0).astype(jnp.bfloat16)  # [num_groups, K, N]


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
```

**Step 2: Commit**

```bash
git add kernel-evolve/kernels/gmm_fp8_bwd_ref.py
git commit -m "feat(kernel-evolve): add FP8 tGMM backward reference implementation"
```

---

### Task 4: Create Backward tGMM FP8 Kernel Template (`gmm_fp8_bwd.py`)

**Files:**
- Create: `kernel-evolve/kernels/gmm_fp8_bwd.py`

**Step 1: Write the kernel template**

tGMM kernel computes `lhs[:, slice]^T @ rhs[slice, :]` per group. The inner kernel
has a different pattern from forward: it tracks group prologue/epilogue (zero
accumulator on group start, store on group end), and applies group-boundary masking
to both lhs and rhs before the transposed dot.

Key differences from forward GMM:
- Grid: `(tiles_n, tiles_k, num_active_tiles)` — m-tiles are in the last dim
- lhs is `[M, K]` (transposed in kernel via `.T`), rhs is `[M, N]`
- Output is `[num_groups, K, N]`
- Group boundary handling: prologue (zero acc), epilogue (store)
- `visit_empty_groups=True` (must zero output for empty groups)

```python
"""FP8 block-wise transposed grouped matrix multiplication Pallas kernel for TPU.

Self-contained port from tokamax. Backward pass: computes
lhs[:, sizes[i-1]:sizes[i]]^T @ rhs[sizes[i-1]:sizes[i], :] per group.
Input lhs [K, M] stored as [M, K] tiles (transposed in kernel).
"""

import functools
import json
from collections import namedtuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

QArray = namedtuple("QArray", ["qvalue", "scale"])

TILE_SIZE = 128


def _blockwise_quantize(x, block_shape, key):
    """Quantize tensor to FP8 E4M3 with block-wise scales."""
    orig_shape = x.shape
    new_shape = []
    reduce_axes = []
    axis_idx = 0
    for dim_size, blk in zip(orig_shape, block_shape):
        if blk == 1 or blk >= dim_size:
            new_shape.append(dim_size)
            reduce_axes.append(axis_idx)
            axis_idx += 1
        else:
            new_shape.extend([dim_size // blk, blk])
            reduce_axes.append(axis_idx + 1)
            axis_idx += 2
    reshaped = x.reshape(new_shape)
    absmax = jnp.max(jnp.abs(reshaped), axis=reduce_axes, keepdims=True)
    fp8_max = jnp.float32(448.0)
    scale = jnp.maximum(absmax / fp8_max, jnp.float32(1e-12)).astype(jnp.float32)
    qvalue = (reshaped / scale).astype(jnp.float8_e4m3fn)
    qvalue = qvalue.reshape(orig_shape)
    scale_shape = []
    for dim_size, blk in zip(orig_shape, block_shape):
        if blk == 1 or blk >= dim_size:
            scale_shape.append(dim_size)
        else:
            scale_shape.append(dim_size // blk)
    scale = scale.reshape(scale_shape)
    return QArray(qvalue=qvalue, scale=scale)


def make_group_metadata(*, group_sizes, m, tm, start_group, num_nonzero_groups,
                        visit_empty_groups):
    """Create metadata for grouped matmul grid execution."""
    num_groups = group_sizes.shape[0]
    end_group = start_group + num_nonzero_groups - 1
    group_ends = jnp.cumsum(group_sizes)
    group_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends])

    rounded_group_ends = ((group_ends + tm - 1) // tm * tm).astype(jnp.int32)
    group_starts = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]])
    rounded_group_starts = group_starts // tm * tm
    rounded_group_sizes = rounded_group_ends - rounded_group_starts
    rounded_group_sizes = jnp.where(group_sizes == 0, 0, rounded_group_sizes)
    group_tiles = rounded_group_sizes // tm

    if visit_empty_groups:
        group_tiles = jnp.where(group_sizes == 0, 1, group_tiles)

    if m % tm != 0:
        raise NotImplementedError(f"{m=} must be divisible by tile size ({tm}).")

    tiles_m = m // tm
    group_ids = jnp.repeat(
        jnp.arange(num_groups, dtype=jnp.int32),
        group_tiles,
        total_repeat_length=tiles_m + num_groups - 1,
    )

    partial_tile_mask = ((group_offsets[:-1] % tm) == 0) | (group_sizes == 0)
    if visit_empty_groups:
        partial_tile_mask = jnp.where(group_sizes == 0, 0, partial_tile_mask)
    partial_tile_ids = jnp.where(partial_tile_mask, tiles_m, group_offsets[:-1] // tm)
    tile_visits = (
        jnp.histogram(partial_tile_ids, bins=tiles_m, range=(0, tiles_m - 1))[0] + 1
    )
    m_tile_ids = jnp.repeat(
        jnp.arange(tiles_m, dtype=jnp.int32),
        tile_visits.astype(jnp.int32),
        total_repeat_length=tiles_m + num_groups - 1,
    )

    first_tile_in_shard = (group_ids < start_group).sum()
    group_ids = jnp.roll(group_ids, shift=-first_tile_in_shard, axis=0)
    m_tile_ids = jnp.roll(m_tile_ids, shift=-first_tile_in_shard, axis=0)

    iota = jnp.arange(num_groups, dtype=jnp.int32)
    active_group_mask = (iota <= end_group) & (iota >= start_group)
    group_tiles = jnp.where(active_group_mask, group_tiles, 0)
    num_tiles = group_tiles.sum()
    return (group_offsets, group_ids, m_tile_ids), num_tiles


def _get_store_mask(*, grid_id, group_metadata, tm, tn):
    """Mask for rows belonging to the current group in the current tile."""
    group_offsets, group_ids, m_tile_ids = group_metadata
    group_id = group_ids[grid_id]
    group_start = group_offsets[group_id]
    group_end = group_offsets[group_id + 1]
    m_id = m_tile_ids[grid_id] * tm
    iota = jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0) + m_id
    return (iota >= group_start) & (iota < group_end)


def _scale_out_by_scale(out, scales):
    """Apply block-wise scales to output tensor."""
    for ax, (s1, s2) in enumerate(zip(out.shape, scales.shape)):
        if s1 != s2:
            scales = jnp.tile(
                scales, [s1 // s2 if i == ax else 1 for i in range(scales.ndim)]
            )
    return out * scales


def _quant_block_spec_tgmm(qvalue, scale, block_spec, reduction_axis):
    """Build BlockSpec pair for QArray in tGMM."""
    eps_list = [pl.cdiv(qs, ss) for qs, ss in zip(qvalue.shape, scale.shape)]
    tile_sizes = block_spec.block_shape
    LANES = 128
    SUBLANES = 8
    min_addr = ([1] * qvalue.ndim + [SUBLANES, LANES])[-qvalue.ndim:]

    eps_red = eps_list[reduction_axis]
    tk = tile_sizes[reduction_axis]
    subchannel_iters = max(1, tk // eps_red) if tk is not None else 1

    def _scale_index_map(*args):
        idxs = list(block_spec.index_map(*args))
        result = []
        for i, idx in enumerate(idxs):
            eps = eps_list[i]
            ts = tile_sizes[i]
            if eps == qvalue.shape[i]:
                result.append(0)
            elif ts is not None and eps <= ts:
                result.append(idx)
            else:
                result.append(idx * ts // eps if ts else idx // eps)
        return result

    scale_tile_sizes = []
    for i, (eps, ts, s, mas) in enumerate(zip(eps_list, tile_sizes, qvalue.shape, min_addr)):
        if ts is None:
            scale_tile_sizes.append(None)
        elif eps == s:
            scale_tile_sizes.append(mas)
        elif eps > ts:
            scale_tile_sizes.append(mas)
        elif eps == 1:
            scale_tile_sizes.append(max(1, ts // eps) if ts else 1)
        else:
            scale_tile_sizes.append(max(1, ts // eps) * mas)
    scale_block_spec = pl.BlockSpec(scale_tile_sizes, _scale_index_map)

    for ax in range(scale.ndim):
        eps = eps_list[ax]
        ts = tile_sizes[ax]
        if ts is None or eps == qvalue.shape[ax]:
            true_per_tile = 1
            sts = min_addr[ax]
        elif eps > ts:
            true_per_tile = 1
            sts = 1
        elif eps == 1:
            true_per_tile = ts
            sts = ts
        else:
            true_per_tile = ts // eps
            sts = true_per_tile * min_addr[ax]
        inflation = sts // true_per_tile
        if inflation > 1:
            scale = jnp.repeat(scale, inflation, axis=ax)

    return qvalue, scale, block_spec, scale_block_spec, subchannel_iters


# ---------------------------------------------------------------------------
# Backward tGMM Pallas kernel
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=["tiling", "num_groups", "interpret"])
def tgmm(lhs_qvalue, lhs_scale, rhs_qvalue, rhs_scale, group_sizes,
         tiling=(128, 128, 128), num_groups=8, interpret=False):
    """FP8 block-wise transposed grouped matrix multiplication.

    Computes: for each group g, output[g] = lhs[:, slice_g]^T @ rhs[slice_g, :]
    where lhs is stored as [M, K] (row-major), loaded as [M,K] tiles,
    transposed inside the kernel.

    Args:
        lhs_qvalue: [M, K] float8_e4m3fn (stored as M×K, transposed in kernel)
        lhs_scale: [M, K//128] float32
        rhs_qvalue: [M, N] float8_e4m3fn
        rhs_scale: [M, N//128] float32
        group_sizes: [num_groups] int32
        tiling: (tm, tk, tn) tile sizes
        num_groups: Number of groups (static)
        interpret: Run in interpret mode

    Returns:
        [num_groups, K, N] bfloat16 output
    """
    m, k = lhs_qvalue.shape
    n = rhs_qvalue.shape[1]
    tm, tk, tn = tiling
    tiles_k = pl.cdiv(k, tk)
    tiles_n = pl.cdiv(n, tn)
    out_dtype = jnp.bfloat16

    group_offset = jnp.array([0], dtype=jnp.int32)
    group_metadata, num_active_tiles = make_group_metadata(
        group_sizes=group_sizes, m=m, tm=tm, start_group=0,
        num_nonzero_groups=num_groups, visit_empty_groups=True,
    )

    def lhs_index_map(n_i, k_i, grid_id, group_metadata, group_offset):
        _, _, m_tile_ids = group_metadata
        return m_tile_ids[grid_id], k_i

    def rhs_index_map(n_i, k_i, grid_id, group_metadata, group_offset):
        _, _, m_tile_ids = group_metadata
        return m_tile_ids[grid_id], n_i

    def out_index_map(n_i, k_i, grid_id, group_metadata, group_offset):
        _, group_ids, _ = group_metadata
        return group_ids[grid_id] - group_offset[0], k_i, n_i

    lhs_block_spec = pl.BlockSpec((tm, tk), lhs_index_map)
    rhs_block_spec = pl.BlockSpec((tm, tn), rhs_index_map)
    out_block_spec = pl.BlockSpec((None, tk, tn), out_index_map)

    # Process QArray block specs — reduction axis is 0 for tGMM (M dimension)
    lhs_qv, lhs_sc, lhs_qv_spec, lhs_sc_spec, lhs_sci = _quant_block_spec_tgmm(
        lhs_qvalue, lhs_scale, lhs_block_spec, reduction_axis=0)
    rhs_qv, rhs_sc, rhs_qv_spec, rhs_sc_spec, rhs_sci = _quant_block_spec_tgmm(
        rhs_qvalue, rhs_scale, rhs_block_spec, reduction_axis=0)

    def _kernel(group_metadata_refs, group_offset_ref,
                lhs_qv_ref, lhs_sc_ref, rhs_qv_ref, rhs_sc_ref,
                out_ref, acc_scratch):
        # EVOLVE-BLOCK-START
        grid_id = pl.program_id(2)
        group_offsets_ref, group_ids_ref, _ = group_metadata_refs
        group = group_ids_ref[grid_id]
        prev_group = group_ids_ref[jnp.where(grid_id > 0, grid_id - 1, 0)]
        is_prologue = (grid_id == 0) | (group != prev_group)
        is_end_of_grid = grid_id == (pl.num_programs(2) - 1)
        next_group = group_ids_ref[jnp.where(is_end_of_grid, grid_id, grid_id + 1)]
        is_epilogue = is_end_of_grid | (group != next_group)
        group_size = group_offsets_ref[group + 1] - group_offsets_ref[group]
        nonzero_gs = group_size > 0

        @pl.when(is_prologue)
        def _zero_acc():
            acc_scratch[...] = jnp.zeros_like(acc_scratch)

        @pl.when(nonzero_gs)
        def _compute():
            # Load lhs [tm, tk] and rhs [tm, tn] tiles
            lhs_tile = lhs_qv_ref[...]
            lhs_s = lhs_sc_ref[...]
            rhs_tile = rhs_qv_ref[...]
            rhs_s = rhs_sc_ref[...]

            # Unpack scales
            lhs_scales = lhs_s
            rhs_scales = rhs_s

            # Apply group boundary mask
            mask_kwargs = dict(
                grid_id=grid_id, group_metadata=group_metadata_refs, tm=tm
            )
            lhs_mask = _get_store_mask(**mask_kwargs, tn=tk)
            rhs_mask = _get_store_mask(**mask_kwargs, tn=tn)
            lhs_tile = jnp.where(lhs_mask, lhs_tile, 0)
            rhs_tile = jnp.where(rhs_mask, rhs_tile, 0)

            # Transposed dot: [tm, tk]^T @ [tm, tn] = [tk, tn]
            out = jax.lax.dot(
                lhs_tile.T, rhs_tile,
                precision=(jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
                preferred_element_type=jnp.float32,
            )

            # Apply scales
            out = _scale_out_by_scale(out, lhs_scales.T)
            out = _scale_out_by_scale(out, rhs_scales)

            acc_scratch[...] += out.astype(jnp.float32)

        @pl.when(is_epilogue)
        def _store():
            acc = acc_scratch[...]
            out_ref[...] = acc.astype(out_dtype)
        # EVOLVE-BLOCK-END

    out = pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((num_groups, k, n), out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[lhs_qv_spec, lhs_sc_spec, rhs_qv_spec, rhs_sc_spec],
            out_specs=out_block_spec,
            grid=(tiles_n, tiles_k, num_active_tiles),
            scratch_shapes=[pltpu.VMEM((tk, tn), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary")
        ),
        interpret=interpret,
        name="tgmm_fp8_bwd",
    )(group_metadata, group_offset, lhs_qv, lhs_sc, rhs_qv, rhs_sc)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _generate_inputs(M, K, N, num_groups):
    """Generate FP8 block-wise quantized inputs for tGMM."""
    key = jax.random.PRNGKey(43)
    k1, k2 = jax.random.split(key)
    lhs_bf16 = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
    rhs_bf16 = jax.random.normal(k2, (M, N), dtype=jnp.bfloat16)
    lhs_q = _blockwise_quantize(lhs_bf16, (1, TILE_SIZE), k1)
    rhs_q = _blockwise_quantize(rhs_bf16, (1, TILE_SIZE), k2)
    group_sizes = jnp.full((num_groups,), M // num_groups, dtype=jnp.int32)
    return lhs_q, rhs_q, group_sizes


def optimized_compute(M=4096, K=7168, N=2048, num_groups=8):
    lhs_q, rhs_q, group_sizes = _generate_inputs(M, K, N, num_groups)
    return tgmm(
        lhs_q.qvalue, lhs_q.scale,
        rhs_q.qvalue, rhs_q.scale,
        group_sizes,
        tiling=(TILE_SIZE, TILE_SIZE, TILE_SIZE),
        num_groups=num_groups,
    )
```

**Step 2: Commit**

```bash
git add kernel-evolve/kernels/gmm_fp8_bwd.py
git commit -m "feat(kernel-evolve): add FP8 tGMM backward Pallas kernel template"
```

---

### Task 5: Create YAML Configs

**Files:**
- Create: `kernel-evolve/gmm_fp8_fwd.yaml`
- Create: `kernel-evolve/gmm_fp8_bwd.yaml`

**Step 1: Write forward config**

```yaml
kernel:
  name: "fp8_gmm_forward"
  template: "kernels/gmm_fp8_fwd.py"
  reference: "kernels/gmm_fp8_fwd_ref.py"
  evolve_markers:
    start: "# EVOLVE-BLOCK-START"
    end: "# EVOLVE-BLOCK-END"

shapes:
  - { M: 4096, K: 7168, N: 2048, num_groups: 8 }
  - { M: 8192, K: 7168, N: 2048, num_groups: 64 }

correctness:
  method: "allclose"
  rtol: 5e-2
  atol: 5e-2

evolution:
  population_size: 25
  num_islands: 3
  max_generations: 50
  stagnation_limit: 10
  fitness: "speedup"

llm:
  provider: "anthropic"
  model: "claude-sonnet-4-6"
  temperature: 0.7

tpu:
  cluster: "my-gke-cluster"
  zone: "us-central2-b"
  tpu_type: "v4-8"
  namespace: "default"
  image: "gcr.io/my-project/kernel-eval:latest"
  timeout: 300

logging:
  output_dir: "runs/gmm_fp8_fwd_001"
  perf_log: true
  charts: true
```

**Step 2: Write backward config**

```yaml
kernel:
  name: "fp8_tgmm_backward"
  template: "kernels/gmm_fp8_bwd.py"
  reference: "kernels/gmm_fp8_bwd_ref.py"
  evolve_markers:
    start: "# EVOLVE-BLOCK-START"
    end: "# EVOLVE-BLOCK-END"

shapes:
  - { M: 4096, K: 7168, N: 2048, num_groups: 8 }
  - { M: 8192, K: 7168, N: 2048, num_groups: 64 }

correctness:
  method: "allclose"
  rtol: 5e-2
  atol: 5e-2

evolution:
  population_size: 25
  num_islands: 3
  max_generations: 50
  stagnation_limit: 10
  fitness: "speedup"

llm:
  provider: "anthropic"
  model: "claude-sonnet-4-6"
  temperature: 0.7

tpu:
  cluster: "my-gke-cluster"
  zone: "us-central2-b"
  tpu_type: "v4-8"
  namespace: "default"
  image: "gcr.io/my-project/kernel-eval:latest"
  timeout: 300

logging:
  output_dir: "runs/gmm_fp8_bwd_001"
  perf_log: true
  charts: true
```

**Step 3: Commit**

```bash
git add kernel-evolve/gmm_fp8_fwd.yaml kernel-evolve/gmm_fp8_bwd.yaml
git commit -m "feat(kernel-evolve): add YAML configs for FP8 GMM evolution"
```

---

### Task 6: Validate configs with dry-run

**Step 1: Run dry-run validation for both configs**

```bash
cd kernel-evolve
kernel-evolve run --config gmm_fp8_fwd.yaml --dry-run
kernel-evolve run --config gmm_fp8_bwd.yaml --dry-run
```

Expected: Both print validation success without errors.

**Step 2: Verify EVOLVE-BLOCK extraction works**

```bash
python -c "
from kernel_evolve.mutation import extract_evolve_block
code = open('kernels/gmm_fp8_fwd.py').read()
block = extract_evolve_block(code)
print(f'Forward evolve block: {len(block.splitlines())} lines')
code = open('kernels/gmm_fp8_bwd.py').read()
block = extract_evolve_block(code)
print(f'Backward evolve block: {len(block.splitlines())} lines')
"
```

Expected: Both extract blocks of ~40-60 lines.

**Step 3: Verify syntax validation passes**

```bash
python -c "
from kernel_evolve.mutation import validate_syntax
for f in ['kernels/gmm_fp8_fwd.py', 'kernels/gmm_fp8_fwd_ref.py',
          'kernels/gmm_fp8_bwd.py', 'kernels/gmm_fp8_bwd_ref.py']:
    code = open(f).read()
    ok, err = validate_syntax(code)
    print(f'{f}: {\"OK\" if ok else err}')
"
```

Expected: All four files print "OK".

---

### Task 7: Final commit and summary

**Step 1: Review all new files**

```bash
git status
git diff --stat HEAD~5
```

**Step 2: Verify file structure matches design**

```
kernel-evolve/
├── gmm_fp8_fwd.yaml
├── gmm_fp8_bwd.yaml
└── kernels/
    ├── gmm_fp8_fwd.py
    ├── gmm_fp8_fwd_ref.py
    ├── gmm_fp8_bwd.py
    └── gmm_fp8_bwd_ref.py
```
