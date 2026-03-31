## Round 8 Batch Analysis

**Variants evaluated**: 5
**Successes**: 5 | **Failures**: 0
**Best speedup this round**: 9.005x (L2_eliminate_gcumsum)
**Overall best speedup**: 9.005x (lineage L2)
**Previous best**: 8.988x (R7, L2_recompute_dh_v2) — **+0.2% improvement**

### Comparative Ranking

| Rank | Variant | Status | Speedup | Latency (ms) | Compute Ratio | Bottleneck | Direction |
|------|---------|--------|---------|--------------|---------------|------------|-----------|
| 1 | L2_eliminate_gcumsum | SUCCESS | 9.005x | 843.9 | 1.0 | register pressure + single-MXU | eliminate_gcumsum |
| 2 | L2_algebraic_simplify | SUCCESS | 8.917x | 859.1 | 1.0 | register pressure + single-MXU | algebraic_simplify |
| 3 | L2_fwd_store_A | SUCCESS | 8.913x | 854.9 | 1.0 | register pressure + single-MXU | fwd_store_A |
| 4 | L2_store_gated_residuals | SUCCESS | 8.821x | 866.3 | 1.0 | register pressure + single-MXU | store_gated_residuals |
| 5 | L2_extra_scratch | SUCCESS | 8.749x | 874.1 | 1.0 | register pressure + single-MXU | extra_scratch |

### Per-Variant Details

#### L2_eliminate_gcumsum (Rank 1) — NEW BEST

**Status**: SUCCESS
**Speedup**: 9.005x (up from 8.988x, +0.2%)
**Latency**: 843.9ms (down from 855.8ms, -1.4%)
**Lineage**: L2 (round 8 in lineage)

| Metric | Value | vs R7 Best | Assessment |
|--------|-------|------------|------------|
| compute_ratio | 1.0 | unchanged | compute-bound |
| vliw_bundle_count | 9332 | -88 (-0.9%) | slightly simpler |
| MXU dual_ratio | 0.0 | unchanged | poor — single-MXU structural limit |
| MXU ops (mxu0) | 5430 | unchanged | same matmul work |
| DMA transfers | 18 | -2 (-10%) | slightly less data movement |
| HLO fusions | 0 | unchanged | ideal for Pallas |
| computation_events | 150 | -21 (-12.3%) | fewer profiled events |
| MXU util (runtime) | 32.12% | -2.48pp | decreased slightly |
| Scalar ALU util | 0.04% | unchanged | negligible |
| Vector ALU util | 16.71% | -2.22pp | moderate |
| Vector Store util | 12.12% | -2.48pp | improved — less spill traffic |
| Vector EUP util | 0.27% | -1.60pp | lower exp() usage — expected |
| Vector fills | 3,296,400 | +36,900 (+1.1%) | marginally increased |
| Vector spills | 2,576,844 | -141,444 (-5.2%) | meaningful reduction |

**Bottleneck**: Register pressure remains the primary bottleneck (2.58M spills), though reduced 5.2% from R7. The g_cumsum elimination approach successfully removed a 67MB input array and replaced [BT,K] exp() operations with [BT] scalar exp(). This reduced VLIW bundles (-0.9%), DMA transfers (-10%), and Vector Store util (-2.5pp). The modest MXU util decrease (-2.5pp) suggests some scheduling efficiency was lost, but the net latency reduction (-1.4%) confirms the approach is beneficial.

**Key insight**: Dropping g_cumsum from backward and recomputing gating from g_gamma scalar is a net win despite the recomputation cost. The bandwidth savings (67MB per invocation) and reduced exp() call overhead outweigh the scalar recomputation cost. However, the improvement is marginal (+0.2%), suggesting we are approaching the performance ceiling for this kernel architecture.

**Suggestions**:
- Combine g_cumsum elimination with algebraic simplification for potentially additive register pressure relief
- The fills increased slightly (+1.1%) despite spills decreasing — investigate whether the compiler is reloading more aggressively
- Consider combining this approach with L2_fwd_store_A's A-matrix storage to attack both register pressure and matmul count

#### L2_algebraic_simplify (Rank 2)

**Status**: SUCCESS
**Speedup**: 8.917x (down from 8.988x, -0.8% regression)
**Latency**: 859.1ms
**Lineage**: L2 (round 8 variant)

| Metric | Value | vs R7 Best | Assessment |
|--------|-------|------------|------------|
| compute_ratio | 1.0 | unchanged | compute-bound |
| vliw_bundle_count | 9313 | -107 (-1.1%) | slightly simpler |
| MXU ops (mxu0) | 5430 | unchanged | same matmul work |
| computation_events | 165 | -6 (-3.5%) | slightly fewer events |
| MXU util (runtime) | 29.01% | -5.59pp | significant regression |
| Vector ALU util | 14.74% | -4.19pp | decreased |
| Vector Store util | 7.76% | -6.84pp | major improvement — spill traffic halved |
| Vector EUP util | 0.24% | -1.63pp | lower exp() — expected |
| Vector fills | 3,234,900 | -24,600 (-0.8%) | marginal reduction |
| Vector spills | 2,503,032 | -215,256 (-7.9%) | good reduction |

**Bottleneck**: Algebraic simplification achieved the best spill traffic reduction (Vector Store util halved from 14.6% to 7.8%) and 7.9% fewer register spills. However, MXU utilization regressed significantly (-5.6pp from 34.6% to 29.0%), causing a net speedup regression despite less register pressure. The algebraic restructuring likely changed the compiler's scheduling in a way that introduces more pipeline bubbles between MXU operations.

**Key learning**: Register spill reduction does not always translate to speedup improvement. The compiler's VLIW scheduling can be disrupted by algebraic restructuring, even when the restructuring reduces live variables. The MXU util regression suggests the code changes made it harder for the compiler to overlap MXU and VPU operations.

#### L2_fwd_store_A (Rank 3)

**Status**: SUCCESS
**Speedup**: 8.913x (down from 8.988x, -0.8% regression)
**Latency**: 854.9ms
**Lineage**: L2 (round 8 variant)

| Metric | Value | vs R7 Best | Assessment |
|--------|-------|------------|------------|
| compute_ratio | 1.0 | unchanged | compute-bound |
| vliw_bundle_count | 8622 | -798 (-8.5%) | significant simplification |
| MXU ops (mxu0) | 4848 | -582 (-10.7%) | 1 fewer matmul as designed |
| DMA transfers | 22 | +2 (+10%) | extra DMA for A matrix |
| computation_events | 174 | +3 (+1.8%) | slight increase from A output |
| MXU util (runtime) | 33.88% | -0.72pp | essentially unchanged |
| Vector ALU util | 20.03% | +1.10pp | slightly more VPU work |
| Vector Store util | 14.27% | -0.33pp | essentially unchanged |
| Vector EUP util | 2.04% | +0.17pp | slightly more exp() |
| Vector fills | 2,792,100 | -467,400 (-14.3%) | major reduction |
| Vector spills | 2,017,188 | -701,100 (-25.8%) | BIGGEST spill reduction this round |

**Bottleneck**: This variant achieved the largest absolute register pressure reduction: spills down 25.8% (701K fewer) and fills down 14.3% (467K fewer). VLIW bundles reduced 8.5% and MXU ops reduced 10.7% due to eliminating the A-matrix recomputation matmul from backward. Despite these dramatic improvements, the net speedup regressed -0.8% because the extra HBM bandwidth for storing/loading A matrices (+33.5MB) offsets the spill and matmul savings.

**Key learning**: Storing the A matrix as a forward-to-backward residual is a promising direction for register pressure relief. The 25.8% spill reduction is the largest achieved in any round. The approach fails to gain speedup only because the A matrix HBM traffic cost nearly exactly cancels the register pressure savings. If a way can be found to reduce the A matrix size (e.g., compressed representation or lower precision), this could become a net win.

#### L2_store_gated_residuals (Rank 4)

**Status**: SUCCESS
**Speedup**: 8.821x (down from 8.988x, -1.9% regression)
**Latency**: 866.3ms
**Lineage**: L2 (round 8 variant)

| Metric | Value | vs R7 Best | Assessment |
|--------|-------|------------|------------|
| compute_ratio | 1.0 | unchanged | compute-bound |
| vliw_bundle_count | 9426 | +6 (+0.1%) | essentially unchanged |
| MXU ops (mxu0) | 5430 | unchanged | same matmul work |
| DMA transfers | 22 | +2 (+10%) | extra DMA for residuals |
| computation_events | 156 | -15 (-8.8%) | fewer pallas events |
| MXU util (runtime) | 35.16% | +0.56pp | slightly better |
| Vector ALU util | 17.84% | -1.09pp | moderate |
| Vector Store util | 14.35% | -0.25pp | essentially unchanged |
| Vector EUP util | 0.37% | -1.50pp | less exp() — expected |
| Vector fills | 2,976,600 | -282,900 (-8.7%) | good reduction |
| Vector spills | 2,423,088 | -295,200 (-10.9%) | meaningful reduction |

**Bottleneck**: Storing q_pos/k_neg as forward residuals reduced spills by 10.9% and fills by 8.7%, confirming that the gating residuals approach reduces backward register pressure. However, the extra HBM bandwidth for storing/loading the residuals (+2 DMAs) costs more than the register pressure savings. The VLIW bundle count is essentially unchanged, suggesting the compiler replaces spilled exp() intermediates with HBM loads of similar cost.

**Key learning**: Forward-to-backward residual storage has diminishing returns when the residuals themselves are large arrays. The q_pos and k_neg arrays are [B,H,NT,BT,K] = 67MB each, so storing them adds 134MB HBM traffic. This is much more expensive than the g_cumsum elimination (which only saves 67MB) because the approach stores TWO residuals vs eliminating ONE input.

#### L2_extra_scratch (Rank 5)

**Status**: SUCCESS
**Speedup**: 8.749x (down from 8.988x, -2.7% regression)
**Latency**: 874.1ms
**Lineage**: L2 (round 8 variant)

| Metric | Value | vs R7 Best | Assessment |
|--------|-------|------------|------------|
| compute_ratio | 1.0 | unchanged | compute-bound |
| vliw_bundle_count | 9496 | +76 (+0.8%) | slight complexity increase |
| MXU ops (mxu0) | 5430 | unchanged | same matmul work |
| DMA transfers | 20 | unchanged | no extra data movement |
| computation_events | 171 | unchanged | same pallas_call structure |
| MXU util (runtime) | 34.62% | +0.02pp | essentially unchanged |
| Vector ALU util | 19.22% | +0.29pp | essentially unchanged |
| Vector Store util | 14.79% | +0.19pp | essentially unchanged |
| Vector EUP util | 2.45% | +0.58pp | slightly more |
| Vector fills | 3,099,600 | -159,900 (-4.9%) | modest reduction |
| Vector spills | 2,582,988 | -135,300 (-5.0%) | modest reduction |

**Bottleneck**: Adding 3 explicit VMEM scratch buffers for intermediate staging achieved only a 5% spill reduction while increasing VLIW bundles by 0.8% and regressing speedup by 2.7%. The explicit scratch management approach adds more overhead (VMEM store/load instructions for the scratch buffers) than it saves from reduced compiler-generated spills. The compiler's register allocator already manages spills reasonably efficiently for this kernel.

**Key learning**: Manual VMEM scratch staging is counterproductive for this kernel. The compiler's automatic spill management is hard to beat by adding explicit scratch stores/loads — the overhead of the explicit staging negates the spill reduction. This is consistent with the R7 finding that scoped recomputation (L2_reduce_intermediates) also regressed.

### Failed Variants Summary

No failed variants this round — all 5 succeeded.

### Lineage Trends

#### Lineage L2 (bwd_fusion → eliminate_gcumsum)

| Round | Best Variant | Speedup | VLIW | Spills | Fills | MXU Util | Events |
|-------|-------------|---------|------|--------|-------|----------|--------|
| R0 (baseline) | baseline | 0.885x | 8270 | 1.91M | 2.36M | 22.5% | 264 |
| R3 | L2_reduce_inputs | 4.006x | -- | -- | -- | -- | -- |
| R4 | L2_fold_dh_pallas | 6.831x | -- | -- | -- | -- | -- |
| R5 | L2_fuse_fwd_combined | 7.845x | 7930 | 2.50M | 3.04M | 33.2% | 177 |
| R7 | L2_recompute_dh_v2 | 8.988x | 9420 | 2.72M | 3.26M | 34.6% | 171 |
| **R8** | **L2_eliminate_gcumsum** | **9.005x** | **9332** | **2.58M** | **3.30M** | **32.1%** | **150** |

**Trend**: Marginal improvement (+0.2%) after R7's +14.6% breakthrough. The register pressure reduction strategies in R8 achieved their intended goal (spills down in all 5 variants), but the speedup gains are minimal. This round represents **diminishing returns** — the kernel architecture may be near its performance ceiling.

**Warning signs**:
- The improvement from R7→R8 (+0.2%) is an order of magnitude smaller than R5→R7 (+14.6%)
- All 5 register pressure reduction approaches (algebraic simplification, VMEM scratch, A-matrix storage, gating residuals, g_cumsum elimination) produced <1% speedup change
- The best spill reduction (L2_fwd_store_A, -25.8%) actually REGRESSED speedup, showing that register pressure is no longer the primary limiting factor
- MXU utilization appears to be the binding constraint at this performance level

#### Lineage L1 (fold_dh_fuse_fwd)

| Round | Status | Note |
|-------|--------|------|
| R5 | 7.639x | Best ever |
| R6 | Stagnant (1) | BlockSpec rank bug |
| R7 | Stagnant (2) | TPU alignment constraint |
| R8 | Stagnant (3) | No variants submitted (all resources allocated to L2) |

**Trend**: 3 consecutive stagnant rounds. L1 converged to L2's architecture in R5 and has failed to differentiate since. **Pruning recommended.**

### Cross-Variant Insights

**Register pressure reduction is no longer the primary optimization lever**: All 5 variants successfully reduced spills (by 5-26%), but the speedup impact ranged from -2.7% to +0.2%. The correlation between spill reduction and speedup is near-zero this round, suggesting that:
1. The remaining spills are "cheap" — they occur in locations where the compiler can overlap spill/fill traffic with other work
2. The binding constraint has shifted from register pressure to MXU pipeline utilization efficiency
3. HBM bandwidth costs from adding residual storage can cancel spill savings

**The g_cumsum elimination pattern works**: Dropping a large input array and recomputing from a scalar is a net win when the recomputation is cheap (scalar exp) and the eliminated array is expensive (67MB). This is a generalizable pattern: prefer scalar recomputation over large-array inputs when the scalar is already available in the kernel context.

**Forward-to-backward residual storage has a size threshold**: Storing A matrix (33.5MB) or gating residuals (134MB) adds HBM traffic that offsets spill savings. For residual storage to be worthwhile, the stored data must be small enough that HBM traffic < spill savings. The A matrix is close to the break-even point; the gating residuals are clearly too large.
