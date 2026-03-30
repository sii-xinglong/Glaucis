## Profile Brief for Round 2

### Delta vs Baseline
| Metric | Baseline | Current Best (L1) | Delta |
|--------|----------|-------------------|-------|
| Speedup | 1.015x | 2.294x | +126% |
| Latency | 10.698ms | 4.692ms | -56% |
| VLIW bundles | 4,065 | 23,462 | +477% (but 2x MXU ops) |
| MXU ops | 72 | 1,792 | +2389% |
| MXU dual ratio | 1.0 | 1.0 | unchanged |
| Register spills | 160,845 | 156 | -99.9% |
| Compute efficiency | 6.68% | 15.24% | +128% |

### Source
- Kernel: iteration_1/variants/tiling_phase_specialization/kernel.py
- Speedup: 2.294x | Latency: 4.692ms
- Compute ratio: 1.0 | Memory transfer ratio: 0.0

### Hardware Utilization Summary
| Unit | Utilization % | Assessment |
|------|--------------|------------|
| MXU  | ~0.003% | extremely low (runtime counter, not reflective of actual throughput) |
| Scalar ALU | 0.008% | very low |
| Vector ALU | 0.002% | very low |
| Vector Load | ~0.003% | very low |
| Vector Store | 0.056% | low |
| Register fills/spills | 156/156 | negligible — solved from 160K baseline |

Note: Runtime hw_utilization counters appear unreliable for this kernel — they show near-zero for all units despite 15.24% compute efficiency and 2.294x speedup. The deep profiling metrics (VLIW, MXU ops) are more trustworthy.

### Deep Profiling Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | 23,462 | high (but 2x MXU ops vs SO8's 17,558) |
| MXU dual ratio | 1.0 | perfect — both MXUs equally loaded |
| MXU ops total | 1,792 (896+896) | 2x more than SO8 (896) |
| Compute efficiency | 15.24% | +13% over SO8, still massive headroom |
| DMA transfers | 21 (6 syncs) | double_buffered: yes |
| HBM bandwidth | N/A | not measured |
| Arithmetic intensity | N/A | not measured |
| Register spills | 156 | negligible |

### Bottleneck Diagnosis
**Primary bottleneck**: Compute-bound with low overall utilization
**Evidence**: compute_ratio=1.0 (no sync waits), compute_efficiency=15.24% (84.8% of peak unused). Register spills eliminated (156 vs 160K baseline). The kernel is entirely compute-limited but at only 15% of peak, suggesting the matmul workload is inherently small relative to TPU v7x capacity, or the fwd+bwd pipeline has scheduling gaps between phases.
**Combined patterns**: The 23,462 VLIW bundles with 1,792 MXU ops means only 7.6% of bundles contain MXU work. The remaining 92.4% are VPU/scalar ops for quantization, data movement, and loop control. This suggests quantization overhead is still a significant portion of total instruction count.

### LLO Key Observations

**MXU Scheduling** (896 mxu0 ops, 896 mxu1 ops, dual_ratio=1.0):
```
%17376 = vmatprep.subr.bf16.mxu0 %v21035_v2
%17440 = vmatprep.subr.bf16.mxu1 %v21036_v2
%17379 = vmatpush1.bf16.msk.msra.mxu0 %vm17378_vm4, %v17377_v21
%17443 = vmatpush1.bf16.msk.msra.mxu1 %vm17442_vm6, %v17441_v22
%13733 = vmatprep.mubr.bf16.mxu0 %v12717_v7
%13734 = vmatmul.mubr.bf16.vlgmr.msra.gmra.mrb[0].mxu0 %v12718_v8
```
Both MXUs co-scheduled with bf16 matmul ops. Dual-port scheduling is effective.

**Scalar Spills** (9,075 spill/fill ops in smem, NOT vector VMEM spills):
```
%s18470 = spseudo.spill_to_hbm %s32952_s0, ...
%s18501_s4 = spseudo.fill_from_hbm %s18486_s4, ...
```
Scalar register spills to smem/HBM for address computations. Low impact (156 vector spills only).

### Optimization Priorities (derived from profile)
1. **Reduce quantization overhead**: 92.4% of VLIW bundles are non-MXU. Forward still has 2 qpl.quantize calls (lhs, rhs). These generate reduce_max, divide, clip, cast ops that dominate instruction count. Explore pre-computed scales or simpler calibration.
2. **Tiling exploration within constraints**: Current best uses (256,256,128) for forward. Try (256, 512, 128) — larger TK=512 could reduce K-loop iterations for forward too. Also try (256, 128, 128) to see if smaller TK reduces overhead.
3. **bwd_gmm tiling refinement**: Currently (1024, 256, 128). Try (256, 256, 128) matching forward's TM=256 for per-group alignment. Or try (256, 512, 128) for larger K tiles.
4. **Precision of forward gmm**: Forward currently uses HIGHEST precision implied by qpl.quantize. Try jax.lax.Precision.HIGHEST for explicit control.

### What NOT to try (profile evidence)
- **MXU scheduling / dual ratio**: Already perfect at 1.0. No benefit.
- **Forward mixed precision (bf16 lhs + fp8 rhs)**: FP10 proved this causes catastrophic correctness failure (max_diff=120,649).
- **Python code restructuring**: F001 proved this has no effect on compiled output.
- **bwd_gmm M > 1024**: FP9 proved this causes VLIW bloat regardless of precision.
- **tile_size != 128**: FP1/F005 proved this breaks FP8 correctness.
