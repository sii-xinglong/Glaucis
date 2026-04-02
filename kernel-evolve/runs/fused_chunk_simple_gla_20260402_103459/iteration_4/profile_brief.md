## Profile Brief for Round 4

### Source
- Kernel: fused_chunk_simple_gla (L1_no_scratch_A variant, L1 lineage)
- Speedup: 1.3881x | Latency: 9.692ms (variant) vs 13.454ms (reference)
- Compute ratio: 1.0 | Memory transfer ratio: 0.0
- Compile time: 26.8s | Peak memory: 1280 MB

### Delta vs Baseline
| Metric | Baseline (1.0x) | Current Best (L1_no_scratch_A) | Delta |
|--------|-----------------|-------------------------------|-------|
| Speedup | 1.000x | 1.388x | +38.8% |
| Latency | 13.45ms | 9.69ms | -27.9% |
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
| MXU  | 0.44% | very low — but hw_utilization varies between profiling samples |
| Scalar ALU | 1.23% | low — minimal control-flow overhead |
| Vector ALU | 0.28% | very low |
| Vector Load | 0.19% | very low |
| Vector Store | 0.12% | very low |
| Vector EUP | 0.008% | negligible — exp() is not dominant |
| Register fills/spills | 6,762,360 / 6,326,905 | severe register pressure — dominant performance limiter |

**NOTE**: hw_utilization sampling varies significantly across runs. Round 2 showed mxu_util=27.91% for the same compiled kernel. The deep profiling metrics (VLIW, MXU ops, spills) are deterministic and reliable.

### Deep Profiling Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | 5,275 | clean baseline (scratch_A removed, -224 vs Round 2) |
| MXU dual ratio | 0.0 | critical — MXU1 completely idle, only MXU0 used |
| MXU ops | 640 (mxu0=640, mxu1=0) | single-port scheduling |
| Avg ops/bundle (ILP) | 3.39 | good — decent VLIW slot utilization |
| HBM bandwidth | 335 MB per invocation | 0.94% of 3690 GB/s peak — very low |
| Arithmetic intensity | n/a | not available |
| Compute efficiency | n/a | not available |
| VMEM utilization | 0.88% of 64 MiB (590 KB used) | extremely low — 63.4 MiB headroom |
| HBM capacity used | 0.65% of 192 GB (1280 MB) | low — small kernel footprint |
| DMA transfers | 464 (no double buffering) | moderate DMA count |
| Pipeline NOPs | 0 | ideal — no pipeline bubbles |
| Fusion count | 0 | ideal — single Pallas kernel |

### Bottleneck Diagnosis
**Primary bottleneck**: register pressure + single-MXU scheduling
**Evidence**: 6.3M register spills dominate execution. The TPU spends the vast majority of time shuffling data between VMEM and registers. MXU dual_ratio=0.0 means only one of two MXUs is active — half the matrix multiply capacity is wasted. Despite good ILP (3.39 ops/bundle), register pressure creates an internal memory traffic bottleneck.
**Combined patterns**: Low VMEM utilization (0.88%) + massive spills + compute-bound (compute_ratio=1.0) = the compiler allocates too little VMEM for scratch, causing register overflow. There is 63 MiB of unused VMEM capacity. All source-level restructuring attempts (15 variants across 3 rounds) compile to the same binary — the bottleneck is in the compiler's register allocation, not in the source code.

### LLO Key Observations
LLO not available for L1_no_scratch_A. Reference LLO from L1_2step (same kernel at 2-step):

**Kernel Structure** (from L1_2step LLO, same forward kernel architecture):
- Forward kernel: `_fused_chunk_fwd_Nstep_kernel` with iteration_bounds = [B, H, 1, 1, NT]
- At 4-step: NT=16 grid iterations, each processes 4 time chunks
- scratch_operands = 1: arg10 (128×128 h state only — scratch_A removed)
- h state initialization: 16 zero-stores to scratch (128×128 = 16 [8×128] vectors)

**MXU Scheduling** (from all available LLO across rounds):
- All MXU ops on mxu_id=0, zero mxu1 ops
- Matrix dimensions: [8×128] tiles (f32 HIGHEST precision)
- This is consistent across ALL variants — 4-step, 2-step, with/without scratch_A

**Key constraint**: The [8×128] matmul sub-tile shape and f32 HIGHEST precision may be preventing dual-MXU scheduling. Dual-MXU requires either larger tiles or lower precision (bf16/fp8), both of which are blocked by constraints (FP54: BK=64 fails, FP55: DEFAULT precision fails).

### HLO Key Observations
HLO not available for this variant. Previous analysis (Round 2):
- 1362 lines, 1 custom_call (single Pallas kernel)
- 35 XLA fusion blocks (outside the Pallas kernel, not within it)
- 45 transpose/copy/bitcast operations in the HLO graph
- Fusion count within Pallas kernel: 0 (ideal)

### Optimization Priorities (derived from profile)
1. **Reduce register pressure via algorithmic restructuring**: With 6.3M spills and VMEM at 0.88%, the kernel needs fundamentally fewer live values per iteration. Source-level changes are normalized by the compiler (FP39 confirmed across 15 variants). The remaining options are:
   - **emit_pipeline / prefetch pipelining**: Use Pallas `pltpu.emit_pipeline` to let the compiler manage DMA-compute overlap and potentially restructure register allocation. This changes the compilation strategy itself, not just the source code.
   - **Recompute attention matrix A on demand**: Instead of computing A=q·k^T once and holding it in registers, split the forward into an A-compute phase and an output-compute phase. Each phase has fewer live values.
   - **Reduce output dimension per grid iteration (BV=64)**: Process half the V dimension per grid iteration. This halves the number of live o_ref vectors from 16 to 8 per sub-step. Unlike BK=64 (FP54), BV is the output dimension and may not violate block shape constraints.

2. **Enable dual-MXU scheduling**: dual_ratio=0.0 wastes half the MXU capacity. All 640 MXU ops are on mxu0. Two hypotheses:
   - f32 HIGHEST precision prevents dual scheduling (but FP55 blocks precision changes)
   - [8×128] sub-tile shape is too small for dual scheduling (but BK/BV must be 128 per FP54)
   - **New approach**: Use `preferred_element_type=jnp.float32` with bf16 inputs instead of `Precision.HIGHEST` with f32 inputs. This may produce identical numerical results but change the MXU scheduling (bf16 matmul with f32 accumulator vs f32 matmul).

3. **Explore pallas_call architecture changes**: Since all source-level changes within the current pallas_call architecture are normalized (FP39), the next frontier is changing the pallas_call structure itself:
   - Multiple smaller pallas_calls with intermediate materialization
   - Different grid decomposition (e.g., grid over B×H×T instead of B×H×NT)
   - Pipelining primitives (`emit_pipeline`) that change compiler behavior

### What NOT to try (profile evidence)
- **Source-level intermediate management (FP39)**: 15 variants across 3 rounds confirmed ALL source-level changes compile identically: scratch staging, dtype changes, backward restructuring, loop restructuring, K-splitting.
- **2-step unrolling (SO20)**: 4-step > 2-step confirmed. 2-step reduces spills 35% but doubles grid iterations, netting a 13.6% regression.
- **BT=32 (FP24 extended)**: Fails correctness (max_diff=24.22 > atol=10.0), not just MXU underutilization.
- **BT=128 (FP52)**: Narrowly fails correctness (max_diff=10.22 > atol=10.0).
- **BK=64 (FP54)**: Violates Pallas TPU block shape constraint.
- **Precision.DEFAULT (FP55)**: Fails correctness for all matmuls at atol=10.0.
- **8-step unrolling (FP50)**: Regresses from 4-step.
- **bf16 intermediate storage (FP53)**: Zero impact on compute-bound kernel.
- **Backward architecture changes (FP53)**: Split/monolithic/restructured backward all compile identically.
- **VMEM scratch for additional intermediates (FP33)**: Adds VLIW overhead with zero spill reduction.
- **Backward grid unrolling (FP43)**: Negligible for both split and monolithic backward.

### Exhaustion Assessment
After 3 rounds and 15 variants, the optimization space within the current kernel architecture is heavily explored. The remaining high-impact approaches require changing the compilation strategy (emit_pipeline, pallas_call restructuring) or the algorithm itself (attention recomputation, output tiling), not just the source code within the current structure.
