## Profile Brief for Round 7

### Source
- Kernel: iteration_6/variants/L2_tgmm_tm1024/kernel.py (L2 best, 2.847x)
- Speedup: 2.847x | Latency: 3.611ms
- Compute ratio: 1.0 | Memory transfer ratio: 0.0

### Delta vs Baseline
| Metric | Baseline | R5 Best | R6 Best (L2_tgmm_tm1024) | Delta R5→R6 |
|--------|----------|---------|--------------------------|-------------|
| Speedup | 1.015x | 2.769x | 2.847x | +2.8% |
| Latency | 10.698ms | 3.800ms | 3.611ms | -5.0% |
| VLIW bundles | 23,462 | 23,462 | 12,951 | -44.8% |
| MXU ops | 1,792 | 1,792 | 896 | -50.0% |
| Spills | 156 | 156 | 156 | 0 |

### Hardware Utilization Summary
| Unit | Utilization % | Assessment |
|------|--------------|------------|
| MXU  | ~0% (event counter) | Counter-based metric unreliable |
| Scalar ALU | ~0% | Low overhead |
| Vector ALU | ~0% | Minimal vector work |
| Vector Load | ~0% | Low |
| Vector Store | ~0% | Low |
| Register fills/spills | 156/156 | Some pressure remains |

### Deep Profiling Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | 12,951 | 44.8% reduction from R5 — major simplification |
| MXU dual ratio | 1.0 | Perfect dual-MXU scheduling |
| MXU ops | 896 (448+448) | Halved from 1,792 — simpler tgmm tiles |
| Compute efficiency | 19.80% | Highest so far |
| DMA transfers | 21 | Double-buffered |
| Fusion count | 0 | Ideal for Pallas |

### Bottleneck Diagnosis
**Primary bottleneck**: Compute-bound (compute_ratio=1.0)
**Evidence**: Zero memory transfer ratio. All time in compute. Spills at 156 — moderate register pressure remains. VLIW at 12,951 is dramatically simpler than before.
**Key observations**:
1. L2_tgmm_tm1024 achieved the best speedup with FEWER MXU ops (896 vs 1,792). Simpler VLIW = better pipeline utilization.
2. L1_scale_bf16 achieved near-best (2.819x) with only 9 spills — bf16 scales almost eliminate register pressure.
3. These two optimizations have NOT been combined yet.

### Optimization Priorities (derived from profile)
1. **Cross-pollinate SO14+SO15**: tgmm TM=1024 (VLIW reduction) + bf16 scales (spill elimination). Neither has been tested with the other. This is the highest-priority direction.
2. **tgmm TM exploration**: TM=2048→1024 was +2.8%. Explore TM=512 and TM=256 to see if the trend continues.
3. **Register pressure elimination**: With bf16 scales, spills drop to 9. Combine with tgmm TM=1024 which already has 156 spills — the compound effect could be significant.

### What NOT to try (profile evidence)
- **Larger tiles (TK=1024, TN=512)**: Confirmed regressions in R6 (FP22, FP23)
- **bwd TN=256**: Catastrophic spills (FP16b)
- **tgmm TM=4096**: VLIW bloat (FP13)
- **Operation reordering**: No effect on Mosaic compiler (FP8, FP18)
