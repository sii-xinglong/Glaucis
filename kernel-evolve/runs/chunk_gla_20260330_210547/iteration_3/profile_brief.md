## Profile Brief for Round 3

### Source
- Kernel: iteration_2/variants/L1_reduce_inputs/kernel.py (best from Round 2)
- Speedup: 1.097x | Latency: 6904ms (vs 7573ms reference)
- Compute ratio: 1.0 | Memory transfer ratio: 0.0

### Delta vs Baseline
| Metric | Baseline | Current Best (L1 R2) | Delta |
|--------|----------|----------------------|-------|
| Speedup | 0.885x | 1.097x | +24.0% |
| Latency | 8691ms | 6904ms | -20.6% |
| Compute ratio | 1.0 | 1.0 | unchanged |
| VLIW bundles | 8270 | 9058 | +9.5% |
| MXU0 ops | 4656 | 5238 | +12.5% (extra A recomputation dot) |
| MXU dual ratio | 0.0 | 0.0 | unchanged — CRITICAL |
| DMA transfers | 24 | 22 | -8.3% |
| Computation events | 264 | 237 | -10.2% (27 fewer kernel launches) |
| Register spills | 1,907,895 | 2,436,588 | +27.7% WORSE |
| Register fills | 2,363,328 | 3,236,478 | +36.9% WORSE |
| MXU util (runtime) | 22.53% | 30.21% | +7.7pp |
| Vector ALU util | 17.10% | 20.63% | +3.5pp |
| Vector Store util | 5.49% | 8.04% | +2.6pp (spill traffic) |
| EUP util | 1.41% | 1.68% | +0.3pp |
| Xlane ops | 32 | 32 | unchanged |

### Hardware Utilization Summary
| Unit | Utilization % | Assessment |
|------|--------------|------------|
| MXU | 30.21% | medium — improved from 22.53% baseline |
| Scalar ALU | 0.007% | negligible — no scalar bottleneck |
| Vector ALU | 20.63% | medium — vector computation active |
| Vector Load | 0.001% | negligible |
| Vector Store | 8.04% | medium — elevated by register spill traffic |
| Vector EUP | 1.68% | low — minimal transcendentals |
| Register fills/spills | 3,236,478 / 2,436,588 | **HIGH** — 2.4M spills, 27.7% worse than baseline |

### Deep Profiling Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | 9058 | +9.5% vs baseline (8270) — slight increase |
| MXU dual ratio | 0.0 | **CRITICAL** — only mxu0 active, mxu1 completely idle |
| MXU0 ops | 5238 | +12.5% vs baseline (extra dot for A recompute) |
| HBM bandwidth | 134MB | low — memory not the bottleneck |
| HBM BW utilization | 0.0005% | far below 3690 GB/s peak |
| DMA transfers | 22 | -8.3% vs baseline, no double-buffering |
| Pipeline NOPs | 0 | no stalls |
| HLO fusions | 0 | ideal for Pallas |
| Xlane ops | 32 | unchanged |
| Computation events | 237 | -10.2% vs baseline — key improvement |

### Bottleneck Diagnosis

**Primary bottleneck**: Register pressure + single-MXU scheduling
**Evidence**:
- 2.4M register spills with 8% Vector Store util dedicated to spill traffic. Each spill is a round-trip to VMEM, wasting vector store bandwidth.
- dual_ratio=0.0 means 100% of matmul work runs on mxu0 while mxu1 is completely idle. Effectively halving potential MXU throughput.
- Despite compute_ratio=1.0 (fully compute-bound), MXU runtime utilization is only 30.21% — most compute time is spent on vector operations and spill/fill traffic, not on matrix multiplies.

**Combined patterns**:
- **High vector_store + single-MXU**: Data shuffling (spills) dominates, and MXU capacity is underutilized. If both MXUs were active AND spills reduced, theoretical 2-3x improvement on MXU-bound portions.
- **High vector_fills/spills + high vliw_bundle_count**: Register pressure from kernel complexity. The fused backward kernel has 7 inputs + 4 outputs + multiple intermediate arrays all live simultaneously.

### LLO Key Observations

**MXU Scheduling** (5238 mxu0 ops, 0 mxu1 ops, dual_ratio=0.0):
All MXU operations use only mxu0. Representative pattern:
```
%849 = vmatprep.subr.mxu0 0.0
%851 = vmatpush1.xpose.msra.mxu0 %v850
%852 = vmatprep.subr.mxu0 0.0
%854 = vmatpush1.xpose.msra.mxu0 %v853
...
```
No mxu1 ops anywhere in the kernel. The compiler has not found independent matmul pairs to schedule on both MXUs. This is likely because all matmuls in the backward kernel share data dependencies (q, k, v, g are used across multiple dot products).

**DMA Pattern** (22 transfers, pipeline-emitter style):
```
dma.done %s546, 1024 /* pipeline-emitter-dma-wait */
dma.done %s555, 1024 /* pipeline-emitter-dma-wait */
...
```
DMA uses pipeline-emitter pattern (7 input DMAs at start, 4 output DMAs at end). All DMA waits happen in prologue/epilogue regions, not interleaved with compute. No double-buffering detected.

**Register Pressure**:
The loop body has a deeply pipelined structure with many phi nodes (>20 scalar phi values at loop entry). The compiler has organized the main computation into a single loop of 2050 iterations (= 2 * 16 * ~64, covering batch×head×chunk dimensions). High spill count suggests intermediate arrays exceed register file capacity within the loop body.

### Optimization Priorities (derived from profile)

1. **Reduce intermediate array liveness**: The backward kernel computes dq, dk, dv, dg simultaneously, keeping all intermediate arrays (q_pos, k_neg, v, g, dA, do, dh, plus products like dv_partial = dA.T @ do) live concurrently. Finding ways to compute subsets of outputs sequentially (within the same pallas_call) could reduce peak register pressure. Unlike splitting into separate pallas_calls (which adds launch overhead per FP20), sequential computation within a single kernel avoids launch overhead while reducing liveness. Consider using `jax.lax.fori_loop` to hide intermediate computations from the compiler's hoisting, or restructuring the computation order so that dq/dk intermediates are consumed before dv/dg intermediates are created.

2. **Eliminate more pallas_call overhead**: The SO11 breakthrough showed that eliminating one pallas_call produced 25% improvement. The forward pass still has multiple separate pallas_calls (inter-chunk state computation, intra-chunk attention, output combination). If any of these can be fused or their outputs recomputed cheaply in other kernels, similar gains are possible.

3. **Reduce total computation events further**: Currently at 237 (vs 264 baseline). Each computation event has fixed scheduling overhead. If forward kernels can be simplified or combined, further event reduction could yield measurable gains.

4. **Address dual_ratio=0.0**: All matmuls use only mxu0. If independent matmul pairs can be created (e.g., by restructuring the backward computation so that two dot products have no data dependency), the compiler may schedule them on both MXUs. However, this is difficult given the inherent data dependencies in GLA backward. Lower priority because FP21 suggests this may be a structural limitation of the algorithm's matmul dimensions.

### What NOT to try (profile evidence)
- **Scratch memory / VMEM explicit management**: FP23/L1_scratch_accum showed <5% spill reduction, insufficient to matter. The Mosaic compiler already manages VMEM efficiently.
- **Manual loop unrolling**: FP23 proved this INCREASES register pressure by giving the compiler visibility into multiple iterations.
- **Smaller block sizes (BT < 64)**: FP24 shows BT=32 critically underutilizes MXU (12% vs 22.5%).
- **dynamic_slice for in-kernel tiling**: FP22 — not supported in Pallas TPU lowering.
- **Kernel splitting**: FP20 — launch overhead exceeds the register pressure savings.
- **Source-level operation reordering**: FP18 — has no effect on compiled VLIW schedule.
