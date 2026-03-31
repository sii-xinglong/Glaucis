## Round 3 Batch Analysis

**Variants evaluated**: 5
**Successes**: 5 | **Failures**: 0
**Best speedup this round**: 1.577x (L1_combined_fuse_skip)
**Overall best speedup**: 1.577x (lineage L1) — **+43.7% OVER PREVIOUS BEST**

### Comparative Ranking

| Rank | Variant | Status | Speedup | Latency (ms) | Compute Ratio | Bottleneck | Direction |
|------|---------|--------|---------|--------------|---------------|------------|-----------|
| 1 | L1_combined_fuse_skip | SUCCESS | 1.577x | 4884 | 1.0 | register-pressure + single-MXU | fuse_fwd_A + skip_dg |
| 2 | L1_fuse_fwd_A | SUCCESS | 1.461x | 5227 | 1.0 | register-pressure + single-MXU | fuse_fwd_A |
| 3 | L1_skip_dg | SUCCESS | 1.102x | 6915 | 1.0 | register-pressure + single-MXU | skip_dg |
| 4 | L2_reduce_inputs | SUCCESS | 1.056x | 7227 | 1.0 | register-pressure + complexity-bloat | reduce_inputs |
| 5 | L2_skip_dg | SUCCESS | 0.877x | 8742 | 1.0 | register-pressure + complexity-bloat | skip_dg |

### Per-Variant Details

#### L1_combined_fuse_skip (Rank 1) — MAJOR BREAKTHROUGH

**Status**: SUCCESS
**Speedup**: 1.577x (4884ms vs 7701ms reference)
**Lineage**: L1 (round 3)

| Metric | Value | Assessment |
|--------|-------|------------|
| compute_ratio | 1.0 | compute-bound |
| vliw_bundle_count | 7930 | -12.5% vs R2 best (9058) — simpler kernel |
| MXU dual_ratio | 0.0 | **CRITICAL** — still single-MXU |
| MXU0 ops | 4656 | -11.1% vs R2 (5238) — removed A recompute + dg matmuls |
| DMA transfers | 20 | -9.1% vs R2 (22) |
| Pipeline NOPs | 0 | no stalls |
| HLO fusions | 0 | ideal for Pallas |
| Xlane ops | 32 | unchanged |
| MXU util (runtime) | 33.21% | medium — improved from 30.2% |
| Scalar ALU util | 0.010% | negligible |
| Vector ALU util | 20.22% | medium |
| Vector Store util | 9.80% | medium — spill traffic |
| EUP util | 2.10% | low |
| Vector fills/spills | 3,038,841 / 2,497,509 | **HIGH** — similar to R2 (2.44M spills) |
| Computation events | 207 | **-12.7% vs R2 (237)** — 30 fewer kernel launches |

**Bottleneck**: Register pressure remains (2.5M spills), single-MXU (dual_ratio=0.0). Despite these, the 1.577x speedup is driven by aggressive kernel launch elimination: 207 computation events vs 264 baseline (-21.6%). The two optimizations compound:
1. Forward A fusion eliminates chunk_gla_fwd_intra_gk pallas_call (~24 fewer events)
2. Skip dg removes 1 matmul + VPU chain, reducing VLIW by 12.5% (~6 fewer events)

**Why it works**: The combined effect is more than additive. With a simpler backward kernel (7930 vs 9058 VLIW), the compiler produces a tighter schedule. MXU utilization improved to 33.2% (from 30.2%) because more of the execution time is spent on useful matmuls rather than dg computation overhead. The 30 fewer computation events translate to ~1800ms latency savings.

**Suggestions**:
- Continue eliminating kernel launch overhead (can chunk_fwd_h be fused with forward output?)
- Address the persistent dual_ratio=0.0
- Try reducing register pressure now that the kernel is simpler (fewer VPU ops → potentially fewer spills)

#### L1_fuse_fwd_A (Rank 2)

**Status**: SUCCESS
**Speedup**: 1.461x (5227ms vs 7639ms reference)
**Lineage**: L1 (round 3)

| Metric | Value | Assessment |
|--------|-------|------------|
| compute_ratio | 1.0 | compute-bound |
| vliw_bundle_count | 9058 | unchanged from R2 (same backward kernel) |
| MXU dual_ratio | 0.0 | still single-MXU |
| MXU0 ops | 5238 | unchanged (backward still has A recompute) |
| DMA transfers | 22 | unchanged |
| Computation events | 213 | -10.1% vs R2 (237) — 24 fewer from fwd A fusion |
| Vector fills/spills | 3,235,689 / 2,435,994 | similar to R2 |

**Bottleneck**: Same backward kernel as R2 best (L1_reduce_inputs), but forward is faster. The 24 fewer computation events come entirely from eliminating the forward chunk_gla_fwd_intra_gk pallas_call. This confirms that forward kernel launch overhead was also significant.

**Why it works**: Recomputing A inside the forward output kernel (adding k as input, computing qg @ kg.T * scale) eliminates one pallas_call from forward. The extra MXU dot is small (64×128 @ 128×64) compared to the kernel launch savings.

#### L1_skip_dg (Rank 3)

**Status**: SUCCESS
**Speedup**: 1.102x (6915ms vs 7617ms reference)
**Lineage**: L1 (round 3)

| Metric | Value | Assessment |
|--------|-------|------------|
| compute_ratio | 1.0 | compute-bound |
| vliw_bundle_count | 7930 | -12.5% vs R2 (9058) — significant reduction |
| MXU0 ops | 4656 | -11.1% vs R2 (5238) — removed dg matmul + A recompute no longer needed |
| DMA transfers | 20 | -9.1% vs R2 (22) |
| Computation events | 231 | -2.5% vs R2 (237) — 6 fewer events |
| Vector fills/spills | 3,039,582 / 2,498,118 | unchanged from R2 — dg removal didn't help spills |

**Bottleneck**: Despite 12.5% fewer VLIW bundles and 11.1% fewer MXU ops, speedup improved only 0.5% (1.097x → 1.102x). The kernel is simpler but the saved computation is a small fraction of total time. The backward kernel execution time is dominated by the grid iterations (231 events), not the per-tile kernel complexity.

**Key insight**: VLIW bundle reduction alone does NOT translate to proportional speedup. What matters is total computation events (kernel launch count). Skip_dg saved only 6 events vs fuse_fwd_A's 24. This confirms that **inter-kernel overhead dominates intra-kernel complexity** for this workload.

#### L2_reduce_inputs (Rank 4)

**Status**: SUCCESS
**Speedup**: 1.056x (7227ms vs 7633ms reference)
**Lineage**: L2 (round 3)

| Metric | Value | Assessment |
|--------|-------|------------|
| vliw_bundle_count | 13928 | +6% vs R2 (13140) — slight increase from A recompute in V-tiled kernel |
| MXU0 ops | 8148 | +7.7% vs R2 (7566) — extra dot for A recompute |
| DMA transfers | 22 | unchanged |
| Computation events | 237 | -10.2% vs R2 (264) — 27 fewer events from removing A call |
| Vector fills/spills | 6,657,546 / 5,623,842 | slightly worse than R2 (5.74M) |

**Bottleneck**: The V-tiled kernel remains fundamentally problematic (5.6M spills, 13928 VLIW bundles). Applying reduce_inputs improved L2 from 0.872x to 1.056x (+21%), crossing the 1.0x threshold. But L2 is structurally limited by the V-tiling complexity bloat (FP23).

#### L2_skip_dg (Rank 5)

**Status**: SUCCESS
**Speedup**: 0.877x (8742ms vs 7667ms reference)
**Lineage**: L2 (round 3)

| Metric | Value | Assessment |
|--------|-------|------------|
| vliw_bundle_count | 11884 | -9.6% vs R2 (13140) |
| MXU0 ops | 6984 | -7.7% vs R2 (7566) |
| Computation events | 258 | -2.3% vs R2 (264) — 6 fewer events |
| Vector fills/spills | 6,265,281 / 5,108,235 | -11% spills vs R2 — modest improvement |

**Bottleneck**: Same pattern as L1_skip_dg: removing dg reduced VLIW/MXU but barely improved speedup. The V-tiled kernel structure remains the bottleneck. L2's skip_dg didn't even cross 1.0x.

### Lineage Trends (Round 1 → Round 2 → Round 3)

**Lineage L1** (mxu_vpu_overlap → reduce_inputs → combined_fuse_skip):

| Metric | Round 1 | Round 2 | Round 3 (best) | R2→R3 Delta | Trend |
|--------|---------|---------|----------------|-------------|-------|
| Speedup | 0.876x | 1.097x | 1.577x | +43.7% | **MAJOR IMPROVEMENT** |
| VLIW bundles | 8270 | 9058 | 7930 | -12.5% | improved |
| MXU ops | 4656 | 5238 | 4656 | -11.1% | returned to baseline |
| Spills | 1,957,131 | 2,436,588 | 2,497,509 | +2.5% | flat (not growing) |
| DMA | 24 | 22 | 20 | -9.1% | improved |
| Comp events | 264 | 237 | 207 | -12.7% | **STRONG improvement** |
| MXU util | 22.53% | 30.21% | 33.21% | +3.0pp | improved |

**Analysis**: L1 has shown consistent, accelerating improvement: 0.876x → 1.097x → 1.577x. Each round targets kernel launch overhead reduction, which has proven to be the dominant performance lever. VLIW bundles decreased, MXU utilization increased, and register spills stabilized (not growing despite added complexity in forward A recompute). The trend is sustainable — there may be room for further event reduction.

**Lineage L2** (mxu_utilization → v_tiling → reduce_inputs/skip_dg):

| Metric | Round 1 | Round 2 | Round 3 (best) | R2→R3 Delta | Trend |
|--------|---------|---------|----------------|-------------|-------|
| Speedup | 0.825x | 0.872x | 1.056x | +21.1% | moderate improvement |
| VLIW bundles | 5226 | 13140 | 13928 | +6.0% | **complexity bloat persists** |
| Spills | 369,000 | 5,735,994 | 5,623,842 | -2.0% | flat (still catastrophic) |
| Comp events | 270 | 264 | 237 | -10.2% | improved (reduce_inputs) |

**Analysis**: L2 improved to 1.056x via reduce_inputs (same optimization that helped L1). But the underlying V-tiled kernel structure is a dead end: 13928 VLIW bundles and 5.6M spills are unsustainable. L2's improvement came entirely from the reduce_inputs technique, not from the V-tiling direction. Consider pivoting L2 to a fundamentally different approach.

### Cross-Variant Insights

1. **Forward A fusion is the biggest win in Round 3**: L1_fuse_fwd_A alone gave +33.2% (1.097x → 1.461x). This is comparable to the backward reduce_inputs win in Round 2 (+25.2%). **Eliminating pallas_calls is the single most effective optimization for this kernel.**

2. **Skip dg has marginal direct impact but compounds well**: L1_skip_dg alone gained only +0.5%. But when combined with fuse_fwd_A, the total gain is +43.7% (vs +33.2% for fuse alone). The dg skip reduces backward VLIW complexity by 12.5%, allowing the compiler to produce a tighter schedule that benefits from the forward event reduction.

3. **Computation events are the dominant performance lever**: The correlation between computation events and speedup is clear:
   - 264 events → 0.885x (baseline)
   - 237 events → 1.097x (R2, -10.2%)
   - 213 events → 1.461x (R3 fuse, -19.3%)
   - 207 events → 1.577x (R3 combined, -21.6%)
   - Each 10% reduction in events yields ~25-35% speedup improvement.

4. **L2's V-tiling direction is exhausted**: Both L2 variants (reduce_inputs and skip_dg) applied L1-style optimizations to L2's V-tiled kernel. The best result (1.056x) is far below L1's 1.577x. The V-tiling adds structural complexity (13928 VLIW, 5.6M spills) that cannot be overcome with launch-overhead optimizations alone.

5. **Register spills stabilized at ~2.5M**: Despite removing dg computation and simplifying the backward kernel, spills stayed at ~2.5M. This suggests the remaining 2.5M spills are structural — caused by the 7 input + 3 output Refs plus the inherent intermediate arrays (q_pos, k_neg, k_decay, exp values, masks, partial products).
