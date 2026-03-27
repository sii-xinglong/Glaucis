# FP8 GMM Pallas Kernel Port to kernel-evolve

## Goal

Port the FP8 block-wise GMM (Grouped Matrix Multiplication) Pallas kernel from
the tokamax library (as used in ant-pretrain PR #220) into the kernel-evolve
framework for evolutionary optimization on TPU.

## Source

- **PR**: https://github.com/primatrix/ant-pretrain/pull/220
- **Kernel source**: `tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py`
- **Integration**: `src/MaxText/kernels/megablox/ops.py` (forward/backward quantization flow)
- **TPU utilities**: `tokamax/_src/mosaic_tpu.py` (quant_block_spec, custom_buffered_pallas_call)

## Architecture

### File Structure

```
kernel-evolve/
├── gmm_fp8_fwd.yaml           # Forward GMM evolution config
├── gmm_fp8_bwd.yaml           # Backward tGMM evolution config
└── kernels/
    ├── gmm_fp8_fwd.py          # Forward GMM Pallas kernel (self-contained)
    ├── gmm_fp8_fwd_ref.py      # Forward GMM reference
    ├── gmm_fp8_bwd.py          # Backward tGMM Pallas kernel (self-contained)
    └── gmm_fp8_bwd_ref.py      # Backward tGMM reference
```

### Why Two Separate Kernels

The kernel-evolve framework supports a single EVOLVE-BLOCK per file. Since forward
GMM and backward tGMM have fundamentally different computation patterns (gmm does
`lhs @ rhs[group]`; tgmm does `lhs_slice^T @ rhs_slice` accumulated per group),
they are split into separate files with independent evolution configs.

### Self-contained Design

All files are self-contained with no tokamax/qwix dependency. This is required
because the Docker evaluation container only has JAX/Pallas installed. Key helpers
inlined from tokamax:

- `QArray` as a `namedtuple("QArray", ["qvalue", "scale"])`
- `make_group_metadata()` for group offset/tile mapping
- `_get_store_mask()` for group-boundary row masking
- `_scale_out_by_scale()` for block-wise FP8 scale application
- `quant_block_spec()` simplified for FP8 QArray BlockSpec construction

### FP8 Input Generation

Since qwix is unavailable, FP8 block-wise quantization is done inline:
1. Generate BF16 random matrices
2. Reshape into 128-element blocks
3. Compute per-block absmax
4. Scale and cast to `jnp.float8_e4m3fn`
5. Package as `QArray(qvalue, scale)`

Block patterns (matching DeepSeek-V3 / Transformer Engine Float8BlockScaling):
- Activation: 1x128 blocks → scale shape `[M, K//128]`
- Weight: 128x128 blocks → scale shape `[num_groups, K//128, N//128]`
- Backward gradient: 1x128 blocks → scale shape `[M, N//128]`

## Forward GMM Kernel (`gmm_fp8_fwd.py`)

### EVOLVE-BLOCK Content

The inner Pallas `kernel()` function body (the function passed to `pallas_call`):

```python
def kernel(group_metadata, _, lhs_ref, rhs_ref, out_ref, acc_scratch, *, subchannel_iters):
    # EVOLVE-BLOCK-START
    grid_id, k_i = pl.program_id(1), pl.program_id(2)
    # ... zero accumulator on first k tile
    # ... subchannel iteration: load FP8 tiles, unpack scales
    # ... dot_general for FP8 matmul
    # ... apply scales to output
    # ... accumulate in f32
    # ... mask + store on last k tile
    # EVOLVE-BLOCK-END
```

### What the LLM Can Optimize

- **Subchannel iteration**: How FP8 tiles are loaded and split for scale application
- **Dot precision**: `dot_general` precision settings and accumulation dtype
- **Scale application**: Order and method of multiplying block-wise scales
- **Memory access patterns**: How tiles are loaded from refs
- **Accumulation strategy**: f32 accumulator management
- **Masking**: Group boundary masking efficiency

### `optimized_compute(M, K, N, num_groups)`

1. Generate FP8 inputs (lhs [M,K], rhs [num_groups,K,N]) with block-wise scales
2. Compute group_sizes (uniform: M//num_groups each)
3. Build group_metadata
4. Set tiling (128, 128, 128) and grid
5. Call `pallas_call` with the kernel

## Backward tGMM Kernel (`gmm_fp8_bwd.py`)

### EVOLVE-BLOCK Content

The tGMM inner kernel computes `lhs_slice^T @ rhs_slice` per group:

```python
def kernel(group_metadata, _, lhs_ref, rhs_ref, out_ref, acc_scratch, *, subchannel_iters):
    # EVOLVE-BLOCK-START
    grid_id = pl.program_id(2)
    # ... prologue: zero accumulator for new group
    # ... load tiles, apply group mask
    # ... transpose lhs, dot product
    # ... apply FP8 scales
    # ... epilogue: store accumulated result
    # EVOLVE-BLOCK-END
```

### `optimized_compute(M, K, N, num_groups)`

1. Generate FP8 inputs (lhs [M,K], rhs [M,N]) with block-wise scales
2. Compute group_sizes, group_metadata
3. Call tGMM pallas_call → output [num_groups, K, N]

## Reference Implementations

### Forward (`gmm_fp8_fwd_ref.py`)

```python
def reference_fn(M, K, N, num_groups):
    # Same FP8 input generation as template
    # Dequantize: bf16_val = qvalue * scale
    # For each group: jnp.dot(lhs_slice, rhs[g]) in f32
    # Concatenate results
```

### Backward (`gmm_fp8_bwd_ref.py`)

```python
def reference_fn(M, K, N, num_groups):
    # Same FP8 input generation
    # Dequantize
    # For each group: jnp.dot(lhs_slice.T, rhs_slice) in f32
    # Stack results → [num_groups, K, N]
```

## Shapes (DeepSeek-V3 MoE)

```yaml
shapes:
  - { M: 4096, K: 7168, N: 2048, num_groups: 8 }
  - { M: 8192, K: 7168, N: 2048, num_groups: 64 }
```

These represent typical MoE expert layer dimensions in DeepSeek-V3 architecture.

## Correctness

FP8 block-wise quantization introduces significant precision loss compared to BF16:

```yaml
correctness:
  method: "allclose"
  rtol: 5e-2
  atol: 5e-2
```

## Constraints

1. All dimensions must be multiples of 128 (tile_size for FP8 block scaling)
2. `tgmm` requires `subchannel_iters == 1` (tile_size == eps)
3. TPU Pallas: sublane size is 8 (gen <7) or 16 (gen >=7), lanes = 128
4. FP8 E4M3 range: [-448, 448], requires scale-aware accumulation
