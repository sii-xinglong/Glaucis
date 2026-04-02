## Profile Brief for Round 2

### Source
- Kernel: fused_chunk_simple_gla (bwd_kv_tiling variant, L1 lineage)
- Speedup: 1.3882x | Latency: 9.680ms (variant) vs 13.438ms (reference)
- Compute ratio: 1.0 | Memory transfer ratio: 0.0
- Compile time: 26.9s | Peak memory: 1280 MB

### Delta vs Baseline
| Metric | Baseline (1.0x) | Current Best (L1) | Delta |
|--------|-----------------|-------------------|-------|
| Speedup | 1.000x | 1.388x | +38.8% |
| Latency | 13.45ms | 9.68ms | -28.0% |
| Compute ratio | 1.0 | 1.0 | unchanged |
| VLIW bundles | 5* | 5,275 | n/a (baseline coarse) |
| MXU dual ratio | n/a | 0.0 | — |
| Register fills | 310,256 | 6,762,360 | +21.8x** |
| Register spills | 310,256 | 6,326,905 | +20.4x** |
| VMEM utilization | n/a | 0.88% | — |
| Peak memory | 1280 MB | 1280 MB | unchanged |

*Baseline profiling had very coarse granularity (34 computation events vs 85 for variants).
**The 20x spill increase reflects different profiling resolution, not actual degradation. The 4-step unrolling inherently has more code per compilation unit.

### Hardware Utilization Summary
| Unit | Utilization % | Assessment |
|------|--------------|------------|
| MXU  | 0.44% | extremely low — kernel far from compute-bound |
| Scalar ALU | 1.22% | low but 2.8x MXU — control-flow overhead proportionally significant |
| Vector ALU | 0.28% | very low |
| Vector Load | 0.19% | very low |
| Vector Store | 0.12% | very low |
| Vector EUP | 0.008% | negligible — exp() is not a factor |
| Register fills/spills | 6,762,360 / 6,326,905 | severe register pressure — dominant performance limiter |

### Deep Profiling Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | 5,275 | forward 4-step kernel |
| MXU dual ratio | 0.0 | critical — MXU1 completely idle, only MXU0 used |
| MXU ops | 640 (mxu0=640, mxu1=0) | single-port scheduling |
| Avg ops/bundle (ILP) | 3.39 | good — decent VLIW slot utilization |
| HBM bandwidth | 335 MB per invocation | 0.94% of 3690 GB/s peak — very low |
| Arithmetic intensity | n/a | not available |
| Compute efficiency | n/a | not available |
| VMEM utilization | 0.88% of 64 MiB (576 KB used) | extremely low — 63.4 MiB headroom |
| HBM capacity used | 0.65% of 192 GB (1280 MB) | low — small kernel footprint |
| DMA transfers | 464 (no double buffering) | no DMA/compute overlap |
| Pipeline NOPs | 0 | ideal — no pipeline bubbles |
| Fusion count | 0 | ideal — single Pallas kernel, no XLA fusion |

### Bottleneck Diagnosis
**Primary bottleneck**: register pressure + single-MXU scheduling
**Evidence**: 6.3M register spills dominate execution. All hardware units show <1.3% utilization, meaning the TPU spends the vast majority of time shuffling data between VMEM and registers (spill/fill traffic) rather than doing useful computation. MXU dual_ratio=0.0 means only one of two MXUs is active — half the matrix multiply capacity is wasted. Despite good ILP (3.39 ops/bundle), the register pressure creates a memory traffic bottleneck internal to the chip.
**Combined patterns**: Low MXU util + massive spills + VMEM only 0.88% utilized = the compiler allocates too little VMEM for scratch, causing register overflow. There is 63 MiB of unused VMEM capacity that could hold intermediates currently being spilled.

### LLO Key Observations

**Kernel Structure** (1 forward kernel in LLO: `_fused_chunk_fwd_4step_kernel`):
- iteration_bounds = [10, 16, 1, 1, 16] → B=10, H=16, NK=1, NV=1, NT=16
- 4-step grid unrolling: each of 16 grid iterations processes 4 time chunks
- 6 functions total: 1 main kernel + 5 transform functions (index maps)
- Backward kernels compiled separately (not in this LLO file)

**MXU Scheduling** (640 mxu0 ops, 0 mxu1 ops, dual_ratio=0.0):
```
"llo.vmatprep.mubr"(%32) <{mxu_id = 0 : i32}> : (vector<8x128xf32>) -> ()
"llo.vmatmul.mubr"(%53) <{mxu_id = 0 : i32, dwg = false}> : (vector<8x128xf32>) -> ()
%246 = "llo.vmatres"() <{mxu_id = 0 : i32}> : () -> vector<8x128xf32>
%248 = "llo.vmatres"() <{mxu_id = 0 : i32}> : () -> vector<8x128xf32>
"llo.vmatprep.mubr"(%32) <{mxu_id = 0 : i32}> : (vector<8x128xf32>) -> ()
"llo.vmatmul.mubr"(%58) <{mxu_id = 0 : i32, dwg = false}> : (vector<8x128xf32>) -> ()
```
All matmul ops exclusively on `mxu_id = 0`. No `mxu_id = 1` ops anywhere.
Data format: f32 (HIGHEST precision). Matrix dimensions: [8x128] tiles.

**Matmul Pattern**:
- 2,560 matmul-related ops (vmatprep + vmatmul + vmatres) in total IR
- Per-execution: 640 MXU ops (from profile)
- Sequence: prep → mul → res → res (2 results per matmul, suggesting 128x128 matmuls decomposed into 8x128 sub-tiles)
- ALL matmuls sequential on single MXU — no parallel pair scheduling

**Conditional Initialization**:
- `scf.if` at kernel entry: zeroes h scratch buffer (128x128 = 16 [8x128] vectors) on first time step only
- 1 conditional block in the entire kernel

### HLO Key Observations
HLO file exists (1362 lines) but exceeds read limit. Key from profile:
- Fusion count: 0 (ideal for single Pallas kernel)
- No cross-program prefetch

### Optimization Priorities (derived from profile)
1. **Reduce register pressure via explicit VMEM scratch**: With VMEM at 0.88% (576 KB out of 64 MiB), there is massive headroom to pin intermediates in VMEM scratch instead of relying on register allocation. The compiler is spilling 6.3M values because it can't fit them in registers — explicitly placing large intermediates (attention matrix A [64×64×4B=16KB], gated operands [64×128×4B=32KB]) in VMEM scratch could eliminate most spills. HOWEVER: FP33 showed manual VMEM scratch staging was counterproductive for chunk_gla — the Mosaic compiler adds its own load/store instructions on top. This must be tested carefully.

2. **Enable dual-MXU scheduling**: dual_ratio=0.0 means half the MXU capacity is wasted. FP21 showed spill reduction alone doesn't trigger dual-MXU. The matmul dimensions ([BT=64, BK=128, BV=128]) may not have independent matmul pairs the compiler can schedule simultaneously. Two possible approaches: (a) increase matmul dimensions to incentivize dual scheduling (but FP52 shows BT=128 fails correctness), (b) restructure to expose concurrent independent matmuls (but FP39 shows source-level restructuring is normalized).

3. **Add DMA prefetch / double buffering**: Currently no double buffering (dma double_buffering=false). HBM bandwidth util is only 0.94% — the kernel isn't memory-bandwidth-bound, but DMA prefetch could overlap data loading with compute, hiding whatever memory latency remains. This is a secondary optimization — register pressure is the primary bottleneck.

### What NOT to try (profile evidence)
- **Source-level restructuring**: Round 1 confirmed FP39 — 5 different source-level approaches (backward fusion, phase separation, Python loops, h elimination, combined) all produced identical compiled output. The Mosaic compiler normalizes source structure.
- **Pallas_call count reduction**: FP45 confirmed at 1.388x — eliminating chunk_fwd_h (3→2 pallas_calls) had zero speedup impact and increased peak memory 25-50%.
- **Backward architecture changes**: FP53 confirmed — monolithic backward, phase-separated backward, and split backward all produce identical performance. Backward is compute-dominated (9 matmuls/step).
- **Increasing block sizes**: BT=128 fails correctness (FP52). BK must stay 128 (FP54 — TPU block shape constraint). Block size exploration space is severely constrained.
- **Precision reduction**: FP55 — ALL matmuls require HIGHEST precision for correctness at atol=10.0.
- **8-step or higher unrolling**: FP50 — 8-step regresses from 4-step. 4-step is the ceiling for grid unrolling.
