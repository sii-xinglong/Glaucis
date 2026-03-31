## Profile Brief for Round 0 (Baseline)

### Source

- Kernel: baseline_kernel.py (9x-optimized, from previous session L2_eliminate_gcumsum)
- Speedup: 9.066x | Latency: 845.8ms
- Compute ratio: 1.0 | Memory transfer ratio: 0.0

### Hardware Utilization Summary


| Unit                  | Utilization %       | Assessment                          |
| --------------------- | ------------------- | ----------------------------------- |
| MXU                   | 32.12%              | medium (single MXU, dual_ratio=0.0) |
| Scalar ALU            | 0.04%               | negligible                          |
| Vector ALU            | 16.72%              | moderate — gating exp operations    |
| Vector Load           | 0.009%              | negligible                          |
| Vector Store          | 12.10%              | high — register spills dominate     |
| Vector EUP            | 0.27%               | low — exp() hardware unit underused |
| Register fills/spills | 3,296,400/2,576,844 | significant pressure                |


### Deep Profiling Metrics


| Metric             | Value | Assessment                                |
| ------------------ | ----- | ----------------------------------------- |
| VLIW bundle count  | 9,332 | moderate complexity                       |
| MXU dual ratio     | 0.0   | single MXU only (mxu0=5430, mxu1=0)       |
| Computation events | 150   | 2 pallas_calls (fwd_combined + bwd_fused) |
| DMA transfers      | 18    | no double buffering                       |
| Fusion count       | 0     | ideal for single Pallas kernel            |


### Architecture Summary

- **Forward**: Single fused `pallas_call` (chunk_fwd_combined) — h propagation + output + A recomputation
- **Backward**: Single fused `pallas_call` (bwd_eliminate_gcumsum) — reverse dh scan + dq/dk/dv
- **Total pallas_calls**: 2
- **g_cumsum**: Already eliminated from residuals (~67MB saved)
- **A matrix**: Already recomputed inline (not stored)

### HBM Memory Analysis (PRIMARY OPTIMIZATION TARGET)

**Residuals stored between forward and backward** (`_fwd` returns to `_bwd`):


| Tensor              | Shape                 | Dtype    | Size (MB) |
| ------------------- | --------------------- | -------- | --------- |
| q                   | [2, 4096, 16, 128]    | float32  | 64.0      |
| k                   | [2, 4096, 16, 128]    | float32  | 64.0      |
| v                   | [2, 4096, 16, 128]    | float32  | 64.0      |
| h                   | [2, 64, 16, 128, 128] | bfloat16 | 32.0      |
| **Total residuals** |                       |          | **224.0** |


**Backward temporary allocations** (additional HBM):


| Operation           | Tensor                                                 | Shape            | Dtype   | Size (MB)       |
| ------------------- | ------------------------------------------------------ | ---------------- | ------- | --------------- |
| transpose q/k/v/do  | q_t, k_t, v_t, do_t                                    | [2,16,4096,128]  | f32     | 4 x 64 = 256    |
| reshape chunks      | q_chunked etc                                          | [2,16,64,64,128] | f32     | same memory     |
| flip arrays         | q_flipped, k_flipped, v_flipped, do_flipped, h_flipped | various          | various | ~288            |
| flatten             | q_flat etc                                             | [2,16,4096,128]  | f32     | same as flipped |
| **Total bwd temps** |                                                        |                  |         | **~544**        |


**Peak HBM estimate**: residuals (224MB) + bwd temporaries (~~544MB) + outputs (~~192MB) = **~960MB**

### Bottleneck Diagnosis (for HBM reduction)

**Primary target**: Residual storage (224MB) and backward flip copies (~288MB)
**Evidence**:

- q/k/v stored at f32 despite being bf16 inputs — upcast to f32 happens before `_fwd`, so f32 versions are residuals
- `jnp.flip()` creates full copies of q/k/v/do/h arrays for reverse scan
- h is stored at bf16 (32MB) — already reasonably compact

### Optimization Priorities (HBM memory reduction)

1. **bf16 residuals**: Store q/k/v as bf16 (inputs are bf16), cast to f32 inside backward. Saves 96MB (192MB → 96MB). Risk: minor precision impact on backward gradients.
2. **Eliminate flip copies**: Instead of `jnp.flip()` creating new arrays, use reversed indexing or negative stride to avoid allocating ~288MB of flip copies. Could use `jnp.flip` elimination by reversing the BlockSpec index_map.
3. **h recomputation**: Don't store h as residual. Recompute it in backward by running the forward h-propagation scan again (adds compute but saves 32MB HBM).
4. **In-place backward indexing**: Instead of flip + flatten, have the backward kernel directly index into the non-flipped arrays using `(NT - 1 - i_t)` in BlockSpec index_maps.
5. **Activation checkpointing**: Only store q and k as residuals; recompute v, h, and all intermediates in backward. Most aggressive savings but adds compute.

### What NOT to try (profile evidence)

- **Speed optimization**: The goal is HBM reduction, not speed. Don't pursue MXU utilization or VLIW reduction — maintain ~9x.
- **Register pressure reduction**: FP34 showed register pressure reduction is saturated at this optimization level. Don't chase spills.
- **Kernel splitting**: FP20 showed splitting adds launch overhead that negates gains.

