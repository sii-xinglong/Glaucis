## Round 2 Batch Analysis

**Variants evaluated**: 5
**Successes**: 4 | **Failures**: 1 (INCORRECT)
**Best speedup this round**: 1.097x (L1_reduce_inputs)
**Overall best speedup**: 1.097x (lineage L1) — **FIRST TIME BEATING REFERENCE**

### Comparative Ranking

| Rank | Variant | Status | Speedup | Latency (ms) | Compute Ratio | Bottleneck | Direction |
|------|---------|--------|---------|--------------|---------------|------------|-----------|
| 1 | L1_reduce_inputs | SUCCESS | 1.097x | 6904 | 1.0 | register-pressure + single-MXU | reduce_inputs |
| 2 | L1_scratch_accum | SUCCESS | 0.888x | 8755 | 1.0 | register-pressure + single-MXU | scratch_memory |
| 3 | L2_single_fused_k_tile | SUCCESS | 0.872x | 8699 | 1.0 | complexity-bloat + register-pressure | v_tiling |
| 4 | L2_smaller_bt | SUCCESS | 0.811x | 9467 | 1.0 | MXU-underutilization | smaller_blocks |
| -- | L1_k_tiling | INCORRECT | -- | -- | -- | -- | k_tiling |

### Per-Variant Details

#### L1_reduce_inputs (Rank 1) — BREAKTHROUGH

**Status**: SUCCESS
**Speedup**: 1.097x (6904ms vs 7573ms reference)
**Lineage**: L1 (round 2)

| Metric | Value | Assessment |
|--------|-------|------------|
| compute_ratio | 1.0 | compute-bound |
| vliw_bundle_count | 9058 | +9.5% vs baseline (8270) — slight increase from recomputing A |
| MXU dual_ratio | 0.0 | **CRITICAL** — still single-MXU |
| MXU0 ops | 5238 | +12.5% vs baseline (4656) — extra dot for A recomputation |
| MXU1 ops | 0 | idle |
| DMA transfers | 22 | -8% vs baseline (24) — eliminated A input DMA |
| Pipeline NOPs | 0 | no stalls |
| HLO fusions | 0 | ideal for Pallas |
| Xlane ops | 32 | unchanged |
| MXU util (runtime) | 30.21% | medium — improved from 28.09% |
| Scalar ALU util | 0.007% | negligible |
| Vector ALU util | 20.63% | medium |
| Vector Store util | 8.04% | medium — slight increase from 6.99% |
| EUP util | 1.68% | low |
| Vector fills/spills | 3,236,478 / 2,436,588 | **HIGH** — spills increased 24% vs R1 (1.96M) |
| Computation events | 237 | **-10% vs baseline (264)** — key improvement: 27 fewer kernel launches |

**Bottleneck**: Register pressure remains severe (2.4M spills), single-MXU (dual_ratio=0.0). Despite spills increasing, the net effect is strongly positive because eliminating the separate A kernel pallas_call saves 27 computation events (kernel launches), removes one HBM round-trip for the A matrix, and allows the MXU to do more useful work (5238 ops vs 4656).

**Why it works**: The separate `chunk_gla_fwd_intra_gk` pallas_call to compute A had its own compilation + launch overhead, its own DMA transfers (A tiles from HBM), and its own register allocation. By recomputing A inside the backward kernel using already-computed `q_pos` and `k_neg` (just 1 extra dot product), all of that overhead is eliminated. The extra MXU dot product costs ~12% more MXU ops but saves ~10% on total computation events. The HBM bandwidth savings (no A matrix transfer) also helps.

**Suggestions**:
- Combine with scratch_accum approach (L1_scratch_accum showed marginal spill reduction)
- Try eliminating additional kernel inputs (e.g., can dh be recomputed?)
- Focus on further reducing computation events / kernel launch overhead
- Address the persistent dual_ratio=0.0 problem

#### L1_scratch_accum (Rank 2)

**Status**: SUCCESS
**Speedup**: 0.888x (8755ms vs 7778ms reference)
**Lineage**: L1 (round 2)

| Metric | Value | Assessment |
|--------|-------|------------|
| compute_ratio | 1.0 | compute-bound |
| vliw_bundle_count | 8350 | +1% vs baseline (8270) — nearly identical |
| MXU dual_ratio | 0.0 | still single-MXU |
| MXU0 ops | 4656 | identical to baseline |
| DMA transfers | 24 | unchanged |
| Vector fills/spills | 2,141,766 / 1,883,277 | spills down 4% (from 1.96M) — marginal improvement |
| Computation events | 264 | unchanged |
| MXU util | 26.77% | medium |
| Vector Store util | 6.45% | slightly improved from 6.99% |

**Bottleneck**: Scratch accumulators had minimal impact on register pressure. VLIW and MXU ops are nearly identical to baseline, suggesting the compiler already optimized accumulator placement. The marginal 4% spill reduction translates to a marginal 1.4% speedup improvement (0.888x vs 0.876x).

**Why it didn't work well**: The Mosaic compiler likely already places output accumulators efficiently. Explicitly using VMEM scratch may have conflicting VMEM allocation with the compiler's own strategy, preventing meaningful register savings. The PrefetchScalarGridSpec overhead may slightly offset the small spill reduction.

#### L2_single_fused_k_tile (Rank 3)

**Status**: SUCCESS
**Speedup**: 0.872x (8699ms vs 7587ms reference)
**Lineage**: L2 (round 2)

| Metric | Value | Assessment |
|--------|-------|------------|
| compute_ratio | 1.0 | compute-bound |
| vliw_bundle_count | 13140 | **+59% vs baseline (8270)** — complexity bloat |
| MXU dual_ratio | 0.0 | still single-MXU |
| MXU0 ops | 7566 | +62% vs baseline (4656) — doubled MXU work |
| DMA transfers | 24 | unchanged |
| Vector fills/spills | 6,548,388 / 5,735,994 | **CATASTROPHIC** — 3x worse than baseline (1.96M) |
| Xlane ops | 64 | doubled (from 32) |
| Vector Store util | 11.85% | very high — massive spill traffic |
| MXU util | 29.71% | medium |

**Bottleneck**: V-tiling caused massive complexity bloat. The manually unrolled V-loop doubled all operations (VLIW +59%, MXU +62%, xlane +100%) without reducing register pressure — in fact spills tripled (5.74M vs 1.96M). The compiler hoisted both V-halves simultaneously, keeping all intermediate arrays live across both iterations. The approach fundamentally failed to reduce peak liveness.

**Why it failed**: Manual unrolling gives the compiler visibility into both iterations simultaneously, which is counterproductive for register pressure. The compiler treats both V-halves as a single computation, keeping all intermediates live. This is the opposite of the intended effect. To actually reduce liveness, the V-loop would need to use `fori_loop` to hide the second iteration from the compiler — but `fori_loop` has its own overhead.

#### L2_smaller_bt (Rank 4)

**Status**: SUCCESS
**Speedup**: 0.811x (9467ms vs 7677ms reference)
**Lineage**: L2 (round 2)

| Metric | Value | Assessment |
|--------|-------|------------|
| compute_ratio | 1.0 | compute-bound |
| vliw_bundle_count | 5226 | identical to L2 R1 — same kernel code, different chunk_size |
| MXU dual_ratio | 0.0 | still single-MXU |
| MXU0 ops | 2910 | same as L2 R1 |
| DMA transfers | 20 | slightly fewer |
| Vector fills/spills | 369,360 / 369,360 | near-zero — good |
| Computation events | 270 | +2.3% vs baseline (264) — more grid iterations |
| MXU util | 11.96% | **LOW** — halved from 22.53% baseline |
| Vector ALU util | 11.65% | low |
| Vector Store util | 1.43% | very low — no spill traffic |

**Bottleneck**: MXU severely underutilized at 12% (vs 22.5% baseline). With BT=32, matmul dimensions are [32,128]@[128,128] — the 32-wide dimension underutilizes the 128x128 MXU systolic array. Near-zero spills but doubled grid iterations (NT=128 vs 64) add overhead that exceeds the simpler-code benefit.

**Key insight**: Smaller tiles reduce spills but critically underutilize the MXU. The MXU needs tile dimensions >= 64 to have reasonable utilization. BT=32 is below the threshold for efficient MXU use.

### Failed Variants Summary

#### L1_k_tiling (INCORRECT)

**Error**: `NotImplementedError: Unimplemented primitive in Pallas TPU lowering for tc: dynamic_slice`
**Cause**: `jax.lax.dynamic_slice` is NOT supported in Pallas TPU kernel lowering (Mosaic). The K-tiling approach used `dynamic_slice` to slice already-loaded arrays within the kernel body.
**Fix**: Cannot use `dynamic_slice` in Pallas kernels. Alternative approaches:
- Use static indexing with compile-time constants (`array[:, 0:64]` and `array[:, 64:128]`) — but this is just manual unrolling which may not reduce register pressure (as L2_single_fused_k_tile proved)
- Use `fori_loop` with Ref-based tile access via the grid dimension (add K as a grid dimension with BlockSpec)
- Accept that in-kernel tiling via slicing is not feasible on TPU

### Lineage Trends (Round 1 → Round 2)

**Lineage L1** (mxu_vpu_overlap → reduce_inputs):

| Metric | Round 1 | Round 2 (best) | Delta | Trend |
|--------|---------|----------------|-------|-------|
| Speedup | 0.876x | 1.097x | +25.2% | **MAJOR IMPROVEMENT** |
| VLIW bundles | 8270 | 9058 | +9.5% | slight increase |
| MXU ops | 4656 | 5238 | +12.5% | more MXU work |
| Spills | 1,957,131 | 2,436,588 | +24.5% | INCREASED |
| DMA | 24 | 22 | -8.3% | improved |
| Comp events | 264 | 237 | -10.2% | **improved — fewer launches** |
| MXU util | 28.09% | 30.21% | +2.1pp | improved |

**Analysis**: L1's reduce_inputs variant demonstrates a key principle: **reducing kernel launch overhead can be more impactful than reducing register pressure**. Despite spills increasing 24%, the elimination of a separate pallas_call (27 fewer computation events) produced a 25% speedup improvement. This suggests that the chunk_gla program's bottleneck is partially in inter-kernel overhead, not just intra-kernel register pressure.

**Lineage L2** (mxu_utilization → v_tiling):

| Metric | Round 1 | Round 2 (best) | Delta | Trend |
|--------|---------|----------------|-------|-------|
| Speedup | 0.825x | 0.872x | +5.7% | moderate improvement |
| VLIW bundles | 5226 | 13140 | +151% | **COMPLEXITY BLOAT** |
| Spills | 369,000 | 5,735,994 | +1454% | **CATASTROPHIC REGRESSION** |

**Analysis**: L2's V-tiling variant improved speedup slightly (0.825x → 0.872x) but at enormous cost to kernel complexity. The variant is closer to L1's fused approach than L2's split approach. The original L2 split direction appears to be a dead end.

### Cross-Variant Insights

1. **Kernel launch overhead > register pressure**: L1_reduce_inputs proves that eliminating a separate pallas_call (even if it increases register pressure) is net positive. The intra-gk A computation kernel cost ~10% of total computation events.

2. **dynamic_slice not available in Pallas TPU**: K-tiling within kernel bodies via dynamic_slice is not feasible. Only static indexing or grid-level tiling works.

3. **Manual unrolling increases register pressure**: L2_single_fused_k_tile proves that manually unrolling V-dimension loops gives the compiler too much visibility, causing it to hoist all intermediates simultaneously. This tripled spills.

4. **BT < 64 underutilizes MXU**: L2_smaller_bt shows that BT=32 drops MXU utilization to 12% — below the threshold for efficient execution.

5. **Scratch memory has minimal impact**: The Mosaic compiler already manages accumulator placement efficiently. Explicit VMEM scratch provides <5% spill reduction.
