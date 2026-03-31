## Profile Brief for Round 4

### Source
- Kernel: iteration_3/variants/L1_combined_fuse_skip/kernel.py (best from Round 3)
- Speedup: 1.577x | Latency: 4884ms (vs 7701ms reference)
- Compute ratio: 1.0 | Memory transfer ratio: 0.0

### Delta vs Baseline
| Metric | Baseline | Current Best (L1 R3) | Delta |
|--------|----------|----------------------|-------|
| Speedup | 0.885x | 1.577x | +78.2% |
| Latency | 8691ms | 4884ms | -43.8% |
| Compute ratio | 1.0 | 1.0 | unchanged |
| VLIW bundles | 8270 | 7930 | -4.1% |
| MXU0 ops | 4656 | 4656 | unchanged |
| MXU dual ratio | 0.0 | 0.0 | unchanged — CRITICAL |
| DMA transfers | 24 | 20 | -16.7% |
| Computation events | 264 | 207 | -21.6% |
| Register spills | 1,907,895 | 2,497,509 | +30.9% WORSE |
| Register fills | 2,363,328 | 3,038,841 | +28.6% WORSE |
| MXU util (runtime) | 22.53% | 33.21% | +10.7pp |
| Vector ALU util | 17.10% | 20.22% | +3.1pp |
| Vector Store util | 5.49% | 9.80% | +4.3pp (spill traffic) |

### Hardware Utilization Summary
| Unit | Utilization % | Assessment |
|------|--------------|------------|
| MXU | 33.21% | medium — improved significantly from 22.5% baseline |
| Scalar ALU | 0.010% | negligible |
| Vector ALU | 20.22% | medium — vector computation active |
| Vector Load | 0.001% | negligible |
| Vector Store | 9.80% | elevated — register spill traffic |
| Vector EUP | 2.10% | low |
| Register fills/spills | 3,038,841 / 2,497,509 | **HIGH** — 30.9% worse than baseline |

### Deep Profiling Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | 7930 | -4.1% vs baseline — slightly simpler |
| MXU dual ratio | 0.0 | **CRITICAL** — mxu1 completely idle |
| MXU0 ops | 4656 | same as baseline |
| HBM bandwidth | 134MB | low |
| DMA transfers | 20 | -16.7% vs baseline |
| Pipeline NOPs | 0 | no stalls |
| Computation events | 207 | -21.6% vs baseline — key improvement source |

### Bottleneck Diagnosis

**Primary bottleneck**: Register pressure (2.5M spills) + single-MXU (dual_ratio=0.0)
**Evidence**:
- 2.5M register spills with 9.8% Vector Store util dedicated to spill traffic
- dual_ratio=0.0: mxu1 completely idle, halving potential matmul throughput
- MXU util at 33.2% — improved but still far from theoretical max

**Key constraint**: After 3 rounds of kernel launch optimization (264→207 events), the remaining events are harder to eliminate:
- Forward: chunk_fwd_h (scan dependency) + fused output kernel (already combines A + output)
- Backward: lax.scan for dh + fused backward kernel (already combines dq/dk/dv)
- Each remaining pallas_call has structural reasons for existence

### What has been proven across rounds
1. **Computation event reduction is the dominant lever**: Each 10% reduction → ~25-35% speedup
2. **Intra-kernel simplification (VLIW reduction) has minimal standalone impact** (<1% speedup for 12.5% VLIW reduction)
3. **But it compounds with event reduction** (+7.9% when combined with forward fusion)
4. **Register pressure is painful but NOT the primary bottleneck**: Spills increased 30% while speedup improved 78%

### Optimization Priorities (derived from profile)

1. **Fuse remaining forward kernels**: The forward still has 2 pallas_calls (chunk_fwd_h + fused_output). If they could be merged (hard due to scan dependency), ~20+ events could be saved. Alternative: can the chunk_fwd_h state propagation be restructured to avoid a separate kernel?

2. **Reduce backward computation events**: The backward has lax.scan (producing ~64 events for NT=64 chunks) + 1 pallas_call. The lax.scan dominates event count. If dh computation can be folded into the backward pallas_call (as a grid dimension instead of scan), events could decrease significantly.

3. **Address register pressure**: With 2.5M spills, reducing live values could improve the 9.8% Vector Store utilization. Candidates:
   - Reduce the number of exp() computations (exp_pos_g, exp_neg_g, exp_gn_minus_g — can these be derived from each other?)
   - Use bfloat16 for some intermediates instead of float32
   - Simplify the backward kernel further (are there any remaining unnecessary computations?)

4. **Dual-MXU scheduling**: All 4656 MXU ops are on mxu0. FP21 suggests this is structural (matmul dimensions/dependencies), but worth one more attempt with deliberately independent matmul pairs.

### What NOT to try (profile evidence)
- **V-tiling / manual unrolling**: FP23 — catastrophically worse
- **Kernel splitting**: FP20 — adds launch overhead
- **Smaller blocks (BT < 64)**: FP24 — underutilizes MXU
- **dynamic_slice**: FP22 — not supported
- **Source-level reordering**: FP18 — no effect
- **Scratch memory**: Round 2 showed <5% spill reduction
- **Skip dg alone**: Round 3 showed only +0.5% standalone impact (already applied)
