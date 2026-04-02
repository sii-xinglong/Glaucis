## Profile Brief for Round 3

### Source
- Kernel: fused_chunk_simple_gla (L1_scratch_A variant, L1 lineage)
- Speedup: 1.4023x | Latency: 9.583ms (variant) vs 13.438ms (reference)
- Compute ratio: 1.0 | Memory transfer ratio: 0.0
- Compile time: 26.6s | Peak memory: 1280 MB

### Delta vs Baseline
| Metric | Baseline (1.0x) | Current Best (L1_scratch_A) | Delta |
|--------|-----------------|---------------------------|-------|
| Speedup | 1.000x | 1.402x | +40.2% |
| Latency | 13.45ms | 9.58ms | -28.8% |
| Compute ratio | 1.0 | 1.0 | unchanged |
| VLIW bundles | 5* | 5,499 | n/a (baseline coarse) |
| MXU dual ratio | n/a | 0.0 | — |
| Register fills | 310,256 | 6,762,360 | +21.8x** |
| Register spills | 310,256 | 6,326,905 | +20.4x** |
| VMEM utilization | n/a | 0.93% | — |
| Peak memory | 1280 MB | 1280 MB | unchanged |

*Baseline profiling had very coarse granularity (34 computation events vs 85 for variants).
**The 20x spill increase reflects different profiling resolution, not actual degradation. The 4-step unrolling with scratch A inherently has more code per compilation unit.

### Delta vs Round 2 Profile (L1 bwd_kv_tiling)
| Metric | Round 2 Best | L1_scratch_A | Delta |
|--------|-------------|-------------|-------|
| VLIW bundles | 5,275 | 5,499 | +224 (+4.2%) |
| VMEM bytes | 589,824 | 622,592 | +32,768 (+5.6%) |
| VMEM utilization | 0.88% | 0.93% | +0.05% |
| DMA count | 464 | 496 | +32 (+6.9%) |
| Spills/fills | 6,326,905/6,762,360 | 6,326,905/6,762,360 | unchanged |
| MXU ops | 640/0 | 640/0 | unchanged |
| Latency | 9.680ms | 9.583ms | -97μs (shared benchmark?) |

### Hardware Utilization Summary
| Unit | Utilization % | Assessment |
|------|--------------|------------|
| MXU  | 27.91% | medium — but note profiling sample variance (was 0.44% in Round 2) |
| Scalar ALU | 0.09% | very low — minimal control-flow overhead |
| Vector ALU | 17.41% | medium — element-wise ops active |
| Vector Load | 0.14% | very low |
| Vector Store | 7.50% | low-medium — store-heavy (scratch staging) |
| Vector EUP | 0.48% | low — exp() is not dominant |
| Register fills/spills | 6,762,360 / 6,326,905 | severe register pressure — dominant performance limiter |

**NOTE**: hw_utilization varies significantly between profiling samples. Round 2 L1 showed mxu_util=0.44%, this round shows 27.91%. The deep profiling metrics (VLIW, MXU ops, spills) are deterministic and reliable.

### Deep Profiling Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | 5,499 | +224 vs Round 2 (from scratch A staging) |
| MXU dual ratio | 0.0 | critical — MXU1 completely idle, only MXU0 used |
| MXU ops | 640 (mxu0=640, mxu1=0) | single-port scheduling |
| Avg ops/bundle (ILP) | 3.30 | good — decent VLIW slot utilization |
| HBM bandwidth | 335 MB per invocation | 0.95% of 3690 GB/s peak — very low |
| Arithmetic intensity | n/a | not available |
| Compute efficiency | n/a | not available |
| VMEM utilization | 0.93% of 64 MiB (622 KB used) | extremely low — 63.4 MiB headroom |
| HBM capacity used | 0.65% of 192 GB (1280 MB) | low — small kernel footprint |
| DMA transfers | 496 (no double buffering) | +32 vs Round 2 (scratch A staging overhead) |
| Pipeline NOPs | 0 | ideal — no pipeline bubbles |
| Fusion count | 0 (HLO has 35 fusion blocks but 1 custom_call) | single Pallas kernel |

### Bottleneck Diagnosis
**Primary bottleneck**: register pressure + single-MXU scheduling
**Evidence**: 6.3M register spills dominate execution. All deep profiling metrics confirm the TPU spends the vast majority of time shuffling data between VMEM and registers. MXU dual_ratio=0.0 means only one of two MXUs is active — half the matrix multiply capacity is wasted. Despite good ILP (3.30 ops/bundle), register pressure creates an internal memory traffic bottleneck.
**Combined patterns**: Low MXU util + massive spills + VMEM only 0.93% utilized = the compiler allocates too little VMEM for scratch, causing register overflow. There is 63 MiB of unused VMEM capacity. The added scratch_A buffer (+32KB) did NOT reduce spills — the attention matrix was not the primary spill source.

### LLO Key Observations

**Kernel Structure** (1 forward kernel in LLO: `_fused_chunk_fwd_4step_kernel`):
- iteration_bounds = [12, 16, 1, 1, 16] → B=12, H=16, NK=1, NV=1, NT=16
- 4-step grid unrolling: each of 16 grid iterations processes 4 time chunks
- scratch_operands = 2: arg10 (128×128 h state) + arg11 (64×128 attention A)
- 6 functions total: 1 main kernel + 5 transform functions

**MXU Scheduling** (640 mxu0 ops, 0 mxu1 ops, dual_ratio=0.0):
```
"llo.vmatprep.subr"(%32) <{mxu_id = 0 : i32}> : (vector<8x128xf32>) -> ()
"llo.vmatprep.subr"(%32) <{mxu_id = 0 : i32}> : (vector<8x128xf32>) -> ()
...  (20 consecutive vmatprep on mxu0 at kernel init, lines 315-353)
```
All matmul ops exclusively on `mxu_id = 0`. No `mxu_id = 1` ops anywhere.
Data format: f32 (HIGHEST precision). Matrix dimensions: [8×128] tiles.

**Scratch A Staging** (new in this variant):
```
llo.vector_store_masked %429 into %arg11 masked %437    ; store A[0] to scratch
llo.vector_store_masked %430 into %arg11 + %27 masked   ; store A[1] to scratch
... (8 stores: 64×128 matrix = 8 [8×128] vectors)
%465 = llo.vector_load %arg11                            ; load A[0] back
%469 = llo.vector_load %arg11 + %27                      ; load A[1] back
... (8 loads: immediately after stores, same addresses)
```
Pattern repeats 4 times (once per sub-step, lines 688-748, 3335+, etc.).
Each sub-step: 8 masked stores + 8 loads = 16 VMEM ops for scratch A.
Total: 4 × 16 = 64 extra VMEM ops. These add overhead without reducing register spills.

### HLO Key Observations
- HLO: 1362 lines, 1 custom_call (single Pallas kernel)
- 35 fusion blocks (these are XLA-level, not within the Pallas kernel)
- 45 transpose/copy/bitcast operations in the HLO graph (outside Pallas kernel)
- Fusion count within the Pallas kernel itself: 0 (ideal)

### Optimization Priorities (derived from profile)
1. **Reduce register pressure via algorithmic change**: With VMEM at 0.93% and spills at 6.3M, the kernel needs fewer live values per iteration, not more scratch staging (FP33 confirmed). The h state scratch (128×128 = 16 [8×128] vectors held across all 4 sub-steps) is the likely dominant spill source. Approaches:
   - **2-step unrolling** instead of 4-step: halves the number of h state intermediates alive simultaneously (but FP50 showed 8-step regresses; 2-step may also regress from fewer grid reductions)
   - **BV=64 output tiling**: Process half the output dimension per grid iteration, reducing live o_ref vectors from 16 to 8 per sub-step (but FP54 showed BK must be 128; BV might have the same constraint)
   - **Recomputation of attention matrix A**: Instead of holding A vectors in registers/scratch across the matmul, recompute A on-the-fly from q·k^T (trading more MXU ops for fewer live values)

2. **Enable dual-MXU scheduling**: dual_ratio=0.0 means half the MXU capacity is wasted. The [8×128] matmul tiles may be too small to trigger dual scheduling. The 640 MXU ops are all sequential on MXU0. Two hypotheses:
   - The matmul dimensions (BT=64 × BK=128) decompose into [8×128] sub-tiles that the compiler schedules sequentially
   - f32 HIGHEST precision may prevent dual-MXU scheduling (dual scheduling may only work with bf16/fp8)
   Neither can be tested without changing block sizes or precision (FP55 blocks precision changes).

3. **Reduce grid iteration overhead**: With 16 time iterations × 4 sub-steps = 64 effective time steps, the loop overhead (h state read/write to scratch per iteration) is multiplied. If the number of time iterations could be reduced further, less scratch traffic would occur. However, FP50 confirmed 8-step regresses and 4-step is the ceiling.

### What NOT to try (profile evidence)
- **VMEM scratch for additional intermediates (FP33)**: L1_scratch_A proved that adding VMEM scratch for A added 224 VLIW bundles and 32 DMA ops with zero spill reduction. The compiler adds store/load on top of its own spill management.
- **Source-level restructuring (FP39, FP53)**: Round 1 and Round 2 confirmed across 10 variants that backward architecture changes, Python loop refactoring, and data access pattern changes all compile identically.
- **disable_bounds_checks (FP56)**: Zero effect on aligned block sizes.
- **8-step or higher unrolling (FP50)**: 8-step regresses from 4-step.
- **Precision reduction (FP55)**: ALL matmuls require HIGHEST precision for correctness at atol=10.0.
- **Block size BT=128 (FP52)**: Fails correctness.
- **BK=64 K-tiling (FP54)**: Violates Pallas TPU block shape constraint.
- **bf16 h residual storage**: L1_bf16_h_scratch showed slight regression or no improvement.
