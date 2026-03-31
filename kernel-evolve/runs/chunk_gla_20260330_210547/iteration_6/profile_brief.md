## Profile Brief for Round 6

### Source
- Kernel: iteration_5/variants/L2_fuse_fwd_combined/kernel.py
- Speedup: 7.845x | Latency: 976ms
- Compute ratio: 1.0 | Memory transfer ratio: 0.0

### Delta vs Baseline
| Metric | Baseline (R0) | Current Best (R5) | Delta |
|--------|---------------|-------------------|-------|
| Speedup | 0.885x | 7.845x | +786% |
| Compute ratio | 1.0 | 1.0 | unchanged |
| VLIW bundles | 8270 | 7930 | -4.1% |
| MXU dual ratio | 0.0 | 0.0 | unchanged |
| Register spills | 1.96M | 2.50M | +27.5% |
| Computation events | 264 | 177 | -33.0% |
| MXU util (runtime) | 22.5% | 33.2% | +47.6% |

### Hardware Utilization Summary
| Unit | Utilization % | Assessment |
|------|--------------|------------|
| MXU  | 33.22% | medium — compute-bound but single-MXU |
| Scalar ALU | 0.04% | negligible — no control-flow bottleneck |
| Vector ALU | 20.20% | moderate — significant VPU work (exp, gating) |
| Vector Load | 0.01% | negligible |
| Vector Store | 9.82% | elevated — register spill traffic |
| Vector EUP | 2.08% | low — exp() hardware utilized |
| Register fills/spills | 3,038,841 / 2,497,509 | severe register pressure |

### Deep Profiling Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | 7930 | stable from R4 |
| MXU dual ratio | 0.0 | poor — ONLY mxu0 active (4656 ops, mxu1=0) |
| DMA transfers | 20 | moderate |
| HLO fusions | 0 | ideal for Pallas |
| Computation events | 177 | forward: 1 pallas_call, backward: 2 pallas_calls |

### Bottleneck Diagnosis

**Primary bottleneck**: Register pressure (2.5M spills) + single-MXU execution (dual_ratio=0.0)

**Evidence**:
- Vector Store util at 9.82% is dominated by register spill traffic (2.5M spills writing to VMEM)
- Vector fills at 3.0M indicate constant reloading of spilled values
- MXU dual_ratio=0.0 means only 1 of 2 MXU units is active — half the matmul capacity is unused
- Despite 33.2% MXU runtime utilization, this is on a single MXU — actual utilization of total MXU capacity is ~16.6%

**Combined patterns**:
- Register pressure + high VLIW count → compiler cannot fit all intermediates in registers, generating spill/fill traffic that slows execution
- Single-MXU + compute-bound → matmul dimensions or data dependencies prevent dual-MXU scheduling (per FP21: this is a structural limitation of chunk_gla's matmul dimensions)

### Current Kernel Architecture
- **Forward**: 1 pallas_call (combined h propagation + output computation with VMEM scratch)
- **Backward dh**: 1 pallas_call (reverse time scan via flipped inputs, VMEM scratch)
- **Backward fused**: 1 pallas_call (dq/dk/dv computation, grid=(H, B*NT))
- **Total**: 3 pallas_calls (down from 5 at baseline)

### Optimization Priorities (derived from profile)

1. **Backward kernel fusion (merge dh + fused)**: The backward still uses 2 separate pallas_calls — chunk_bwd_dh_pallas produces dh in HBM, then chunk_gla_bwd_fused reads it. Merging these would eliminate the dh tensor HBM round-trip and one kernel launch. This is the same pattern that yielded SO15 (+14.8% from forward fusion). However, the dh kernel iterates in REVERSE time order while the fused kernel iterates in original order — this is a structural challenge.

2. **Precision reduction for register pressure**: Remove `precision=lax.Precision.HIGHEST` from matmul calls and use DEFAULT precision. This might allow the compiler to use bf16 intermediate accumulation, reducing register footprint. Each float32 accumulator uses 2x the register space of bf16. Risk: correctness impact from reduced precision.

3. **fori_loop for V dimension in backward**: The backward fused kernel processes full V=128 in one tile. Using `jax.lax.fori_loop` to process V in two 64-wide passes would hide one iteration from the compiler (per FP23 lesson), potentially halving peak register pressure. The compiler currently hoists all V-wide intermediates simultaneously.

4. **Eliminate dh HBM traffic via recomputation**: Instead of computing dh in a separate pallas_call and passing it to the fused backward, recompute each dh[t] value on-the-fly inside the fused backward kernel. This eliminates the dh tensor entirely (saves B*NT*H*K*V = 2*64*16*128*128 = 256MB HBM traffic). Risk: doubles the matmul work for dh accumulation.

5. **Double buffering**: DMA analysis shows no double buffering (double_buffering=false). Adding explicit double buffering via emit_pipeline could overlap data loading with computation. However, the kernel is already compute-bound (compute_ratio=1.0), so DMA overlap may not help much.

### What NOT to try (profile evidence)
- **BT=128 (chunk size increase)**: FP28 proved counterproductive — VLIW bloat +44%, -21% regression
- **bf16 operands with Precision.HIGHEST**: FP25 extended — Mosaic rejects this combination
- **Staged backward output writes**: FP29 — reduces spills but regresses MXU util, net marginal
- **Source-level operation reordering**: FP18/FP8 — compiler ignores Python source order
- **exp() → reciprocal conversion**: FP27 — hardware EUP handles exp() more efficiently
- **Manual loop unrolling**: FP23 — increases register pressure by making both iterations visible
- **BT < 64**: FP24 — critically underutilizes MXU
- **Dual-MXU via register reduction alone**: FP21 — dual scheduling depends on matmul dimensions, not register availability
