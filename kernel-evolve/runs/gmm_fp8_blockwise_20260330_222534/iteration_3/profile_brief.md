## Profile Brief for Round 3

### Source
- Kernel: iteration_2/variants/L1_fwd_tk512/kernel.py
- Speedup: 2.335x | Latency: 4.403ms
- Compute ratio: 1.0 | Memory transfer ratio: 0.0

### Delta vs Baseline
| Metric | Baseline | Current Best | Delta |
|--------|----------|-------------|-------|
| Speedup | 1.015x | 2.335x | +130% |
| Latency | 10.698ms | 4.403ms | -58.8% |
| Compute ratio | 1.0 | 1.0 | — |
| VLIW bundles | 4,065 | 23,462 | +477% |
| MXU ops | 72 | 1,792 | +2389% |
| MXU dual ratio | 1.0 | 1.0 | — |
| Register spills | 160,845 | 156 | -99.9% |
| Compute efficiency | 6.68% | 16.24% | +143% |
| DMA transfers | 35 | 21 | -40% |

### Hardware Utilization Summary
| Unit | Utilization % | Assessment |
|------|--------------|------------|
| MXU  | ~0% (counter artifact) | Counter measurement unreliable — MXU is clearly active (1,792 ops, dual_ratio=1.0) |
| Scalar ALU | 0.0075% | Very low — minimal control flow overhead |
| Vector ALU | 0.0019% | Very low — quantization ops are minimal |
| Vector Load | 0.000026% | Negligible |
| Vector Store | 0.052% | Low |
| Register fills/spills | 156/156 | Excellent — down from 160,845 baseline |

### Deep Profiling Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | 23,462 | High (5.8x baseline), but dominated by MXU ops — this is good |
| MXU dual ratio | 1.0 | Perfect — both MXUs fully utilized |
| MXU ops total | 1,792 (896 mxu0 + 896 mxu1) | 24.9x baseline — major improvement |
| Avg ops/bundle (ILP) | N/A | Bundle density not available |
| HBM bandwidth | N/A | Not measured |
| Arithmetic intensity | N/A | Not measured |
| Compute efficiency | 16.24% of 2307 TFLOPS peak | 2.4x baseline — improving but far from peak |
| DMA transfers | 21 | Double-buffered: yes |
| Pipeline NOPs | 0 | Excellent — no pipeline bubbles |
| Fusions | 0 | Ideal for single Pallas kernel |

### Bottleneck Diagnosis
**Primary bottleneck**: Compute-bound with 16.24% compute efficiency — far from peak FLOPS
**Evidence**: compute_ratio=1.0 (fully compute-bound), zero sync waits, only 156 register spills. The kernel is spending all time computing but only achieving 16.24% of peak FLOPS. VLIW has 23,462 bundles with 1,792 MXU ops, meaning ~92.4% of bundles are non-MXU (data movement, quantization, index computation).
**Combined patterns**: The 92.4% non-MXU VLIW overhead suggests the remaining 2 forward quantization calls (lhs + rhs) are the major bottleneck. Each quantize call involves absmax reduction, scale computation, and fp8 casting — all VPU/SFU work.

### LLO Key Observations

**Allocation summary**: Kernel has 4 SMEM spill allocations (`spilled sreg/preg`), 1 VMEM scratch (f32[512,128] = 256KB), and 1 internal scratch (u32[144,128]). The VMEM scratch is for tgmm accumulator.

**MXU Scheduling** (896 mxu0 ops, 896 mxu1 ops, dual_ratio=1.0):
Both MXUs perfectly co-scheduled. 1,792 total MXU ops across 23,462 VLIW bundles means ~7.6% of bundles contain MXU ops.

**DMA Pattern** (21 transfers, 6 syncs):
Double-buffered with overlap. DMA is not the bottleneck — only 21 transfers and 6 syncs.

**Spill Count**: Only 4 SMEM register spills (scalar registers), 156 vector fills/spills. Dramatically improved from baseline's 160,845.

### Optimization Priorities (derived from profile)
1. **Forward quantization reduction**: 2 forward quantize calls (lhs + rhs) generate substantial VPU/SFU overhead. The forward lhs quantize involves absmax, scale, and cast on M=8192 x K=2048 = 16M elements. The rhs quantize operates on G=32 x K=2048 x N=512 = 33M elements. These dominate the 92.4% non-MXU VLIW bundles. BUT: FP10 proves forward mixed precision fails, so we cannot skip forward quantization. Alternative: try `calibration_method="max"` (cheaper than absmax which needs abs then max) or `scale_dtype=jnp.bfloat16` (cheaper than float32).
2. **Forward out_dtype optimization**: Currently out_dtype=jnp.float32 for forward, requiring float32 accumulation and float32→bf16 cast for backward. Using out_dtype=jnp.bfloat16 for forward would skip this conversion and may reduce VLIW overhead.
3. **Tiling fine-tuning**: Forward TK=512 was +1.8% over TK=256. Explore bwd_gmm TK=512 to halve backward K-loop iterations. Explore tgmm TM=4096 (now that bf16 has no M constraints).

### What NOT to try (profile evidence)
- **DMA/memory optimization**: compute_ratio=1.0, memory_transfer_ratio=0.0. Kernel is fully compute-bound, not memory-bound.
- **MXU dual scheduling**: dual_ratio=1.0 already perfect.
- **Register pressure reduction**: Only 156 spills (down from 160,845). Not a bottleneck.
- **Forward mixed precision (bf16 lhs + fp8 rhs)**: FP10 proves this causes catastrophic correctness failure (max_diff=120,649).
- **Operation reordering in backward**: FP8 proves XLA normalizes schedule regardless of Python source order.
