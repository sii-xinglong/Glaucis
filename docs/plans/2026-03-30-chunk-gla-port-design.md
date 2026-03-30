# Chunked GLA Kernel Port Design

## Goal

Port the chunked GLA (Gated Linear Attention) forward and backward Pallas kernels from [primatrix/pallas-kernel](https://github.com/primatrix/pallas-kernel) into the kernel-evolve optimization framework.

## Source

- Repository: `primatrix/pallas-kernel` (package: `tops`)
- Key files: `tops/ops/gla/chunk.py`, `tops/ops/common/chunk_h.py`, `tops/ops/common/chunk_o.py`, `tops/ops/common/cumsum.py`
- Gate mode: `g_gamma` only (per-head constant gate, no per-element gradients needed)

## Target Dimensions (AL Model)

| Parameter | Value |
|-----------|-------|
| B | 2 |
| T | 4096 |
| H | 16 |
| K | 128 |
| V | 128 |
| chunk_size | 64 |
| scale | 1/sqrt(128) |

## File Structure

```
kernel-evolve/examples/
  kernels/
    chunk_gla.py          # Template with EVOLVE-BLOCK
    chunk_gla_ref.py      # Pure JAX reference
  chunk_gla.yaml          # Eval config
```

## Template Kernel (`chunk_gla.py`)

### Frozen Section (outside EVOLVE-BLOCK)

- Imports: jax, jnp, pallas, pallas.tpu
- Constants: BK=128, BV=128
- Utility functions: `cdiv`, `align_up`, `pad_to_multiple`, `exp` (float32 promotion)
- `optimized_compute(B, T, H, K, V, chunk_size)` entry point

### EVOLVE-BLOCK Contents (~600-700 lines)

All Pallas kernels and orchestration, g_gamma-only path:

1. **chunk_local_cumsum_scalar** — constant gate decay within each chunk
2. **chunk_fwd_h_kernel** — Pallas kernel for inter-chunk state propagation (sequential over chunks)
3. **chunk_gla_fwd_o_kernel** — Pallas kernel combining inter/intra-chunk outputs
4. **chunk_gla_fwd** — forward orchestration
5. **chunk_bwd_dh_kernel** — Pallas kernel for reverse state gradient propagation
6. **chunk_gla_bwd_kernel** — fused backward gradients (dq, dk, dv)
7. **chunk_gla_bwd** — backward orchestration
8. **chunk_gla_fwd_bwd** — `jax.custom_vjp` wrapper

### Entry Point Behavior

`optimized_compute` creates deterministic inputs (PRNGKey(42)), runs fwd+bwd, returns `(o, dq, dk, dv)`.

## Reference Kernel (`chunk_gla_ref.py`)

Pure JAX implementations using `jax.lax.scan` and einsum. No Pallas kernels. Same `simple_compute` signature returning `(o, dq, dk, dv)`.

## YAML Config (`chunk_gla.yaml`)

- Single shape: B=2, T=4096, H=16, K=128, V=128, chunk_size=64
- Correctness: allclose with rtol=1e-2, atol=1e-2
- Target: tpu7x-cluster in us-central1

## Evaluation

Correctness is checked on all four output tensors (o, dq, dk, dv). Performance measures full fwd+bwd combined latency. FLOPs formula for chunked GLA: dominated by the matmuls in state propagation and output computation.

## What the Optimizer Can Mutate

- Block sizes within kernels
- Chunk-level tiling strategy
- Kernel fusion (merge phases)
- Loop unrolling / pipelining
- Memory layout (head-first vs batch-first)
- Grid dimensions and BlockSpecs
- Accumulator precision
- Cumsum fusion into state propagation

## Key TPU Constraints

- BK=BV=128 aligned to MXU tiles
- bfloat16 inputs, float32 accumulators (required by Mosaic compiler on v7x)
- `pltpu.CompilerParams(disable_bounds_checks=True)` for performance
- `jax.lax.Precision.HIGHEST` for matmuls
