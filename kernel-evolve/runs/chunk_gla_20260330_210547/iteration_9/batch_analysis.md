## Round 9 Batch Analysis

**Variants evaluated**: 5
**Successes**: 4 | **Failures**: 1 (INCORRECT)
**Best speedup this round**: 9.054x (L2_bf16_h_residual)
**Overall best speedup**: 9.054x (lineage L2)
**Previous best**: 9.005x (R8, L2_eliminate_gcumsum) — **+0.5% improvement**

### Comparative Ranking

| Rank | Variant | Status | Speedup | Latency (ms) | Compute Ratio | Bottleneck | Direction |
|------|---------|--------|---------|--------------|---------------|------------|-----------|
| 1 | L2_bf16_h_residual | SUCCESS | 9.054x | 841.7 | 1.0 | MXU pipeline + single-MXU | bf16_h_residual |
| 2 | L2_simplify_gating | SUCCESS | 8.975x | 860.2 | 1.0 | MXU pipeline + single-MXU | simplify_gating |
| 3 | L2_merge_A_dA | SUCCESS | 8.973x | 849.5 | 1.0 | MXU pipeline + single-MXU | merge_A_dA |
| 4 | L2_precompute_gated | SUCCESS | 8.954x | 846.5 | 1.0 | MXU pipeline + single-MXU | precompute_gated |
| -- | L2_bf16_dh_update | INCORRECT | -- | -- | -- | -- | bf16_dh_update |

### Per-Variant Details

#### L2_bf16_h_residual (Rank 1) — NEW BEST

**Status**: SUCCESS
**Speedup**: 9.054x (up from 9.005x, +0.5%)
**Latency**: 841.7ms (down from 843.9ms, -0.3%)
**Lineage**: L2 (round 9 in lineage)

| Metric | Value | vs R8 Best | Assessment |
|--------|-------|------------|------------|
| compute_ratio | 1.0 | unchanged | compute-bound |
| vliw_bundle_count | 9252 | -80 (-0.9%) | slightly simpler |
| MXU dual_ratio | 0.0 | unchanged | poor — single-MXU structural limit |
| MXU ops (mxu0) | 5430 | unchanged | same matmul work |
| DMA transfers | 18 | unchanged | same data movement |
| HLO fusions | 0 | unchanged | ideal for Pallas |
| computation_events | 150 | unchanged | same pallas_call structure |
| MXU util (runtime) | 32.42% | +0.30pp | marginally better |
| Scalar ALU util | 0.04% | unchanged | negligible |
| Vector ALU util | 16.49% | -0.22pp | marginally less VPU work |
| Vector Store util | 11.86% | -0.26pp | marginally less spill traffic |
| Vector EUP util | 0.27% | unchanged | same exp() usage |
| Vector fills | 3,198,000 | -98,400 (-3.0%) | meaningful reduction |
| Vector spills | 2,503,044 | -73,800 (-2.9%) | meaningful reduction |

**Bottleneck**: MXU pipeline efficiency remains the primary bottleneck at 32.4% utilization (single-MXU). The bf16 h residual approach reduces HBM traffic by halving the h state storage from f32 (128MB) to bf16 (64MB). This saves ~64MB of DMA traffic per invocation (both forward write and backward read). The VLIW bundle count decreased by 80 bundles (-0.9%), suggesting the compiler generates slightly simpler code for bf16 DMA loads. Register spills decreased 2.9% and fills decreased 3.0%, indicating the reduced DMA traffic indirectly eases register pressure by freeing DMA slots.

**LLO observations**: The h residual input is confirmed as `bf16[1,1,1,128,128]` window shape (32KB per tile vs 64KB for f32). The main loop still runs `start=0, step=1, limit=2050`. The scoped memory usage is 13,627,392 bytes (13MB).

**HLO observations**: Single `tpu_custom_call` with `bf16[2,16,64,128,128]` parameter (index 3). 0 fusions — ideal. No transposes or copies outside the custom_call.

**Key insight**: bf16 h residual works because the h state at chunk boundaries is used as an approximation for inter-chunk gradient contributions in the backward pass. The truncation from f32 to bf16 introduces small errors that do not compound — unlike the dh update matmul which accumulates across 64 time steps. This is a generalizable pattern: forward-to-backward residuals that are used once (not accumulated) can tolerate bf16 precision.

#### L2_simplify_gating (Rank 2)

**Status**: SUCCESS
**Speedup**: 8.975x (down from 9.005x, -0.3% regression)
**Latency**: 860.2ms
**Lineage**: L2 (round 9 variant)

| Metric | Value | vs R8 Best | Assessment |
|--------|-------|------------|------------|
| compute_ratio | 1.0 | unchanged | compute-bound |
| vliw_bundle_count | 9313 | -19 (-0.2%) | essentially unchanged |
| MXU ops (mxu0) | 5430 | unchanged | same matmul work |
| DMA transfers | 18 | unchanged | same data movement |
| computation_events | 150 | unchanged | same structure |
| MXU util (runtime) | 32.13% | +0.01pp | unchanged |
| Vector ALU util | 16.66% | -0.05pp | essentially unchanged |
| Vector Store util | 12.11% | -0.01pp | essentially unchanged |
| Vector EUP util | 0.20% | -0.07pp | slightly less exp() |
| Vector fills | 3,284,100 | -12,300 (-0.4%) | marginal reduction |
| Vector spills | 2,564,544 | -12,300 (-0.5%) | marginal reduction |

**Bottleneck**: The VPU exp() call reduction approach (5 exp() calls down to 3) had negligible impact on performance. The exp() operations are handled by the hardware EUP unit (0.20-0.27% utilization) which is not a bottleneck. Eliminating redundant exp() calls saved only 19 VLIW bundles and 12K spills — the compiler was likely already optimizing these away via common subexpression elimination.

**Key learning**: The Mosaic compiler is effective at CSE for exp() operations on TPU v7x. Manual exp() elimination provides marginal benefit when the EUP unit is underutilized (<1%).

#### L2_merge_A_dA (Rank 3)

**Status**: SUCCESS
**Speedup**: 8.973x (down from 9.005x, -0.4% regression)
**Latency**: 849.5ms
**Lineage**: L2 (round 9 variant)

| Metric | Value | vs R8 Best | Assessment |
|--------|-------|------------|------------|
| compute_ratio | 1.0 | unchanged | compute-bound |
| vliw_bundle_count | 9332 | unchanged | same complexity |
| MXU ops (mxu0) | 5430 | unchanged | same matmul work |
| DMA transfers | 18 | unchanged | same data movement |
| computation_events | 150 | unchanged | same structure |
| MXU util (runtime) | 26.11% | -6.01pp | **MAJOR regression** |
| Vector ALU util | 13.85% | -2.86pp | decreased |
| Vector Store util | 6.80% | -5.32pp | **major improvement** — spill traffic halved |
| Vector EUP util | 0.22% | -0.05pp | essentially unchanged |
| Vector fills | 2,570,700 | -725,700 (-22.0%) | **BEST fill reduction this round** |
| Vector spills | 2,121,744 | -455,100 (-17.7%) | **BEST spill reduction this round** |

**Bottleneck**: The merge_A_dA restructuring achieved the best register pressure reduction of any R9 variant: spills down 17.7% and fills down 22.0%. Vector Store util halved from 12.12% to 6.80%. However, MXU utilization regressed severely from 32.1% to 26.1% (-6.0pp), causing a net speedup regression despite dramatically less register pressure. The phase reordering disrupted the compiler's MXU-VPU scheduling overlap.

**Key learning**: Algorithmic phase reordering can dramatically reduce register pressure (by minimizing simultaneously live arrays) but at the cost of MXU pipeline scheduling. The compiler's VLIW scheduler is sensitive to the order in which matmuls and VPU operations appear — reordering can introduce pipeline bubbles between MXU ops even when the total matmul count is unchanged. This reinforces FP34: register pressure reduction does not translate to speedup at ~9x.

#### L2_precompute_gated (Rank 4)

**Status**: SUCCESS
**Speedup**: 8.954x (down from 9.005x, -0.6% regression)
**Latency**: 846.5ms
**Lineage**: L2 (round 9 variant)

| Metric | Value | vs R8 Best | Assessment |
|--------|-------|------------|------------|
| compute_ratio | 1.0 | unchanged | compute-bound |
| vliw_bundle_count | 9313 | -19 (-0.2%) | essentially unchanged |
| MXU ops (mxu0) | 5430 | unchanged | same matmul work |
| DMA transfers | 18 | unchanged | same data movement |
| computation_events | 150 | unchanged | same structure |
| MXU util (runtime) | 31.91% | -0.21pp | essentially unchanged |
| Vector ALU util | 17.06% | +0.35pp | slightly more |
| Vector Store util | 8.83% | -3.29pp | improved — less spill traffic |
| Vector EUP util | 0.20% | -0.07pp | slightly less |
| Vector fills | 3,321,000 | +24,600 (+0.7%) | marginally increased |
| Vector spills | 2,558,400 | -18,444 (-0.7%) | marginal reduction |

**Bottleneck**: Consolidating all VPU gating operations into a single compact block at the start of the backward kernel did not improve MXU-VPU overlap as hoped. The compiler's VLIW scheduler already interleaves VPU and MXU operations effectively — manual consolidation adds artificial serialization (VPU block must complete before MXU phases begin) rather than improving overlap. The Vector Store util improvement (-3.29pp) is likely from reduced intermediate variable lifetime rather than the consolidation itself.

**Key learning**: The Mosaic compiler's VLIW scheduler handles MXU-VPU interleaving well. Manually consolidating VPU operations into a single block is counterproductive because it removes the compiler's freedom to interleave VPU work with MXU pipeline stalls. Let the compiler handle operation scheduling.

#### L2_bf16_dh_update (INCORRECT)

**Status**: INCORRECT
**Error**: `Correctness failed for shape {'B': 2, 'T': 4096, 'H': 16, 'K': 128, 'V': 128, 'chunk_size': 64}: max_diff=22.275390625`
**Lineage**: L2 (round 9 variant)

**Cause**: Using bf16 operands with `lax.Precision.DEFAULT` for the dh update matmul (`q_hat.T @ do` in Phase 6 backward) and forward h update (`k.T @ v_gated`) caused correctness failure. The dh state accumulates across 64 time steps (T/chunk_size = 4096/64 = 64). Each time step's bf16 truncation error compounds multiplicatively through the sequential state propagation, resulting in max_diff=22.28 which exceeds the atol=10.0 threshold.

**Key learning**: This extends FP31 (Precision.DEFAULT fails correctness). Not only does global precision reduction fail, but selectively applying bf16 precision to state-accumulating matmuls also fails. The dh update is structurally different from the h residual: dh accumulates across ALL time steps within the backward pass, while h residual is a point-in-time snapshot used once. **Any matmul whose output feeds into cross-time-step accumulation must use f32/HIGHEST precision.**

### Failed Variants Summary

| Variant | Status | Error | Fix |
|---------|--------|-------|-----|
| L2_bf16_dh_update | INCORRECT | max_diff=22.28 > atol=10.0 | Do not use bf16 for matmuls whose outputs accumulate across time steps (dh state update). Only use bf16 for point-in-time residuals (h state snapshot). |

### Lineage Trends

#### Lineage L2 (bwd_fusion → bf16_h_residual)

| Round | Best Variant | Speedup | VLIW | Spills | Fills | MXU Util | Events |
|-------|-------------|---------|------|--------|-------|----------|--------|
| R0 (baseline) | baseline | 0.885x | 8270 | 1.91M | 2.36M | 22.5% | 264 |
| R3 | L2_reduce_inputs | 4.006x | -- | -- | -- | -- | -- |
| R4 | L2_fold_dh_pallas | 6.831x | -- | -- | -- | -- | -- |
| R5 | L2_fuse_fwd_combined | 7.845x | 7930 | 2.50M | 3.04M | 33.2% | 177 |
| R7 | L2_recompute_dh_v2 | 8.988x | 9420 | 2.72M | 3.26M | 34.6% | 171 |
| R8 | L2_eliminate_gcumsum | 9.005x | 9332 | 2.58M | 3.30M | 32.1% | 150 |
| **R9** | **L2_bf16_h_residual** | **9.054x** | **9252** | **2.50M** | **3.20M** | **32.4%** | **150** |

**Trend**: Continued marginal improvement (+0.5%) via HBM bandwidth reduction. The bf16 h residual approach opens a new optimization axis (precision reduction for non-accumulating residuals) that previous rounds had not explored. VLIW bundles continue decreasing (9332 → 9252), and spills continue decreasing (2.58M → 2.50M), but the improvement rate is declining.

**Trajectory analysis**:
- R5→R7: +14.6% (breakthrough — kernel fusion)
- R7→R8: +0.2% (diminishing — register pressure saturated)
- R8→R9: +0.5% (slight rebound — new axis: HBM bandwidth via bf16)

The kernel is approaching its architectural performance ceiling. Further gains require either:
1. Finding more non-accumulating arrays that can be stored as bf16
2. Reducing the total number of matmuls per backward time step (algorithmic change)
3. Exploiting a fundamentally new optimization axis not yet tried

### Cross-Variant Insights

**bf16 precision has a correctness boundary**: The h residual (point-in-time snapshot, used once in backward) tolerates bf16. The dh update (accumulated across 64 time steps) does not. This defines a clear rule: **bf16 is safe for forward-to-backward residuals that are consumed without accumulation; it is unsafe for any value that feeds into sequential state updates.**

**VPU optimization is a dead end at ~9x**: Both L2_simplify_gating (exp() reduction) and L2_precompute_gated (VPU consolidation) regressed. The compiler already handles VPU scheduling effectively. Manual VPU optimization provides <0.5% benefit and often regresses by constraining the compiler's VLIW scheduler.

**Phase reordering trades MXU util for register pressure**: L2_merge_A_dA achieved the best spill reduction (17.7%) but the worst MXU regression (-6.0pp). This reinforces FP34 with a new dimension: not only does spill reduction not help, phase reordering actively harms MXU scheduling. The compiler's original phase ordering was near-optimal for MXU pipeline utilization.

**HBM bandwidth is a viable optimization axis**: Unlike register pressure (saturated, FP34) and VPU scheduling (compiler-optimized), HBM bandwidth reduction via precision lowering produced measurable improvement (+0.5%). Future rounds should explore: other residual arrays that could use bf16, packed integer representations for mask-like arrays, or compressed storage formats.
