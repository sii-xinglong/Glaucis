## Round 7 Batch Analysis

**Variants evaluated**: 5
**Successes**: 4 | **Failures**: 1 (INCORRECT)
**Best speedup this round**: 8.988x (L2_recompute_dh_v2)
**Overall best speedup**: 8.988x (lineage L2)
**Previous best**: 7.845x (R5, L2_fuse_fwd_combined) — **+14.6% improvement**

### Comparative Ranking

| Rank | Variant | Status | Speedup | Latency (ms) | Compute Ratio | Bottleneck | Direction |
|------|---------|--------|---------|--------------|---------------|------------|-----------|
| 1 | L2_recompute_dh_v2 | SUCCESS | 8.988x | 855.8 | 1.0 | register pressure + single-MXU | bwd_fusion_5d_output |
| 2 | L2_bwd_fuse_dh_v2 | SUCCESS | 8.968x | 859.4 | 1.0 | register pressure + single-MXU | bwd_fusion_flat_output |
| 3 | L2_reduce_intermediates | SUCCESS | 7.700x | 988.8 | 1.0 | register pressure + single-MXU | reduce_intermediates |
| 4 | L2_split_dv_dqdk | SUCCESS | 6.869x | 1128.7 | 1.0 | kernel launch overhead | split_backward |
| -- | L1_bwd_grid_v_v2 | INCORRECT | -- | -- | -- | BlockSpec alignment | v_tiling_grid |

### Per-Variant Details

#### L2_recompute_dh_v2 (Rank 1) — NEW BEST

**Status**: SUCCESS
**Speedup**: 8.988x (up from 7.845x, +14.6%)
**Latency**: 855.8ms (down from 975.7ms, -12.3%)
**Lineage**: L2 (round 7 in lineage)

| Metric | Value | vs R5 Best | Assessment |
|--------|-------|------------|------------|
| compute_ratio | 1.0 | unchanged | compute-bound |
| vliw_bundle_count | 9420 | +1490 (+18.8%) | complexity increase from fusion |
| MXU dual_ratio | 0.0 | unchanged | poor — single-MXU structural limit |
| MXU ops (mxu0) | 5430 | +774 (+16.6%) | more matmul work from fused bwd |
| DMA transfers | 20 | unchanged | moderate |
| HLO fusions | 0 | unchanged | ideal for Pallas |
| computation_events | 171 | -6 (-3.4%) | confirms 1 fewer pallas_call |
| MXU util (runtime) | 34.60% | +1.38pp | medium — improving |
| Scalar ALU util | 0.04% | unchanged | negligible |
| Vector ALU util | 18.93% | -1.27pp | moderate |
| Vector Store util | 14.60% | +4.78pp | elevated — more spill traffic |
| Vector EUP util | 1.87% | -0.21pp | low |
| Vector fills | 3,259,500 | +220,659 (+7.3%) | increased register pressure |
| Vector spills | 2,718,288 | +220,779 (+8.8%) | increased register pressure |

**Bottleneck**: Register pressure (2.72M spills, +8.8% vs R5) combined with single-MXU execution (dual_ratio=0.0). Despite increased spills, the elimination of the dh HBM round-trip (512MB) more than compensates, yielding a net 14.6% speedup. The fused backward kernel now handles both dh state propagation and dq/dk/dv computation in a single pallas_call with VMEM scratch for state.

**Architecture**: 2 pallas_calls total (down from 3):
1. Combined forward (h propagation + output computation)
2. Combined backward (dh state + dq/dk/dv computation)

**Key insight**: The backward fusion approach mirrors the forward fusion (SO15) that yielded +14.8% in R5. Both eliminate a HBM round-trip by merging sequential state-dependent kernels into a single pallas_call with VMEM scratch and "arbitrary" dimension_semantics.

**Suggestions**:
- The fused backward kernel has 9420 VLIW bundles with 2.72M spills — reducing register pressure is the primary remaining optimization target
- Consider splitting the fused backward into scoped computation blocks (like L2_reduce_intermediates attempted, but within the fused kernel)
- The 5D output layout of this variant slightly outperformed L2_bwd_fuse_dh_v2's flat layout — the time-indexed writes may produce better VLIW scheduling

#### L2_bwd_fuse_dh_v2 (Rank 2)

**Status**: SUCCESS
**Speedup**: 8.968x (up from 7.845x, +14.3%)
**Latency**: 859.4ms
**Lineage**: L2 (round 7 variant)

| Metric | Value | vs R5 Best | Assessment |
|--------|-------|------------|------------|
| compute_ratio | 1.0 | unchanged | compute-bound |
| vliw_bundle_count | 9391 | +1461 (+18.4%) | complexity increase from fusion |
| MXU dual_ratio | 0.0 | unchanged | poor — single-MXU |
| MXU ops (mxu0) | 5430 | +774 (+16.6%) | more matmul work |
| computation_events | 171 | -6 (-3.4%) | 1 fewer pallas_call |
| MXU util (runtime) | 37.11% | +3.89pp | medium — best MXU util this round |
| Vector Store util | 15.60% | +5.78pp | highest spill traffic this round |
| Vector fills | 3,259,500 | +220,659 (+7.3%) | increased |
| Vector spills | 2,742,864 | +245,355 (+9.8%) | highest spills this round |

**Bottleneck**: Same as L2_recompute_dh_v2. Flat 4D output layout produces slightly higher MXU utilization (37.1% vs 34.6%) but marginally more spills (2.74M vs 2.72M). The ~3.5ms latency difference likely stems from compiler scheduling differences between flat and 5D output layouts.

**Suggestions**: Consider hybrid: use flat output layout (better MXU) with scoped intermediate reduction (fewer spills).

#### L2_reduce_intermediates (Rank 3)

**Status**: SUCCESS
**Speedup**: 7.700x (down from 7.845x, -1.8% regression)
**Latency**: 988.8ms
**Lineage**: L2 (round 7 variant)

| Metric | Value | vs R5 Best | Assessment |
|--------|-------|------------|------------|
| vliw_bundle_count | 7930 | unchanged | same kernel structure |
| MXU ops (mxu0) | 4656 | unchanged | same matmul work |
| computation_events | 177 | unchanged | same pallas_call count |
| MXU util (runtime) | 19.85% | -13.37pp | significant regression |
| Vector Store util | 5.53% | -4.29pp | improved — less spill traffic |
| Vector fills | 2,386,782 | -652,059 (-21.5%) | meaningful reduction |
| Vector spills | 2,337,570 | -159,939 (-6.4%) | modest reduction |

**Bottleneck**: The scoped compute-and-consume restructure successfully reduced register fills by 21.5% and spills by 6.4%, but the extra recomputation of A-matrix and dA intermediates caused a net slowdown. MXU utilization dropped dramatically (33.2% → 19.9%) suggesting the recomputation work added non-MXU overhead.

**Key learning**: Scoped recomputation within a single kernel can reduce register pressure, but the recomputation cost at this kernel complexity level outweighs the spill savings. The approach might work better if the recomputed intermediates were cheaper (simpler expressions).

#### L2_split_dv_dqdk (Rank 4)

**Status**: SUCCESS
**Speedup**: 6.869x (down from 7.845x, -12.4% regression)
**Latency**: 1128.7ms
**Lineage**: L2 (round 7 variant)

| Metric | Value | vs R5 Best | Assessment |
|--------|-------|------------|------------|
| vliw_bundle_count | 5195 | -2735 (-34.5%) | major simplification |
| MXU ops (mxu0) | 2910 | -1746 (-37.5%) | less matmul work per kernel |
| computation_events | 204 | +27 (+15.3%) | more kernel launches |
| MXU util (runtime) | 17.44% | -15.78pp | significant regression |
| Vector Store util | 3.78% | -6.04pp | much improved |
| Vector fills | 2,047,950 | -990,891 (-32.6%) | major reduction |
| Vector spills | 1,623,600 | -873,909 (-35.0%) | major reduction |

**Bottleneck**: Kernel splitting dramatically reduced register pressure (spills -35%, fills -33%, VLIW -35%) but the cost was too high: 27 more computation events from 2 extra kernel launches, plus A-matrix recomputation overhead in both sub-kernels. The kernel launch overhead dominates at this performance level.

**Key learning**: Kernel splitting as a register pressure relief strategy is counterproductive for chunk_gla when the fused kernel is already fast. The launch overhead and redundant A-matrix computation add more latency than the register spill reduction saves. This confirms FP20's lesson from a different angle — now measured at 7.845x rather than 0.885x.

#### L1_bwd_grid_v_v2 (INCORRECT)

**Error**: `ValueError: The Pallas TPU lowering currently requires that the last two dimensions of your block shape are divisible by 8 and 128 respectively`
**Block spec**: `(1, 1, 64, 64)` for array shape `(16, 128, 64, 128)` — last dimension 64 is NOT divisible by 128.
**Root cause**: V-tiling with BV_inner=64 creates a block shape where the last dim (64) doesn't meet Pallas's alignment requirement of being divisible by 128 or equal to the full array dimension.
**Fix**: BV_inner must be 128 (the full V dimension), making V-tiling along the last dimension impossible with this constraint. V-tiling would need the V dimension to NOT be the last dimension — requires transposing the array layout.

### Failed Variants Summary

| Variant | Status | Error Category | Root Cause |
|---------|--------|---------------|------------|
| L1_bwd_grid_v_v2 | INCORRECT | Pallas TPU alignment | Block shape last dim 64 not divisible by 128 |

### Lineage Trends

#### Lineage L2 (fuse_fwd_combined → bwd_fusion)

| Round | Best Variant | Speedup | VLIW | Spills | MXU Util | Events |
|-------|-------------|---------|------|--------|----------|--------|
| R0 (baseline) | baseline | 0.885x | 8270 | 1.91M | 22.5% | 264 |
| R1 | L2_mxu_util | 0.883x | -- | -- | -- | -- |
| R2 | L2_v_tiling | 1.108x | -- | -- | -- | -- |
| R3 | L2_reduce_inputs | 4.006x | -- | -- | -- | -- |
| R4 | L2_fold_dh_pallas | 6.831x | -- | -- | -- | -- |
| R5 | L2_fuse_fwd_combined | 7.845x | 7930 | 2.50M | 33.2% | 177 |
| R6 | (stagnant) | 7.845x | -- | -- | -- | -- |
| **R7** | **L2_recompute_dh_v2** | **8.988x** | **9420** | **2.72M** | **34.6%** | **171** |

**Trend**: Strong improvement after 1 stagnant round. The backward fusion strategy (merging dh + bwd_fused) mirrors the R5 forward fusion breakthrough. VLIW bloat (+18.8%) is the trade-off for eliminating a HBM round-trip. This is a **sustainable** improvement — the complexity increase maps directly to eliminated data movement, not unnecessary computation.

**Warning**: Register spills continue to grow (1.91M → 2.50M → 2.72M). Each fusion adds more work to a single compiled kernel, pushing register pressure higher. Future optimizations must address this within the fused kernels.

#### Lineage L1 (fold_dh_fuse_fwd)

| Round | Status | Note |
|-------|--------|------|
| R5 | 7.639x | Best ever |
| R6 | Stagnant (1) | L1_bwd_grid_kv failed: BlockSpec rank bug |
| R7 | Stagnant (2) | L1_bwd_grid_v_v2 failed: TPU alignment constraint |

**Trend**: 2 consecutive failed variants. L1 has converged to the same architecture as L2 (both use the same R5 best kernel), so the differentiation is lost. The V-tiling direction continues to hit Pallas constraints.

### IR Analysis

No IR artifacts available for the top two variants (L2_recompute_dh_v2, L2_bwd_fuse_dh_v2). Artifacts available for L2_reduce_intermediates and L2_split_dv_dqdk but these are lower-performing variants. Key observations from metrics alone:

**Backward fusion VLIW increase**: Both fusion variants added ~1460-1490 VLIW bundles (18-19% increase). With 774 additional MXU ops (16.6% increase), the extra bundles are primarily from the dh state propagation loop that was previously in a separate pallas_call. The bundle-to-MXU ratio suggests moderate VLIW packing efficiency in the fused kernel.

**Computation event reduction**: 177 → 171 (-3.4%, -6 events). This confirms the backward went from 2 pallas_calls to 1. The 6 events (not just 3) reduction suggests the profiler counts per-iteration events differently when kernels are fused.

**Register pressure pattern**: Fills grew more than spills (+7.3% vs +8.8%), suggesting the fused kernel reloads previously-spilled values more frequently due to the larger working set from dh state propagation.
