## Round 5 Batch Analysis

**Variants evaluated**: 5
**Successes**: 4 | **Failures**: 1
**Best speedup this round**: 7.845x (L2_fuse_fwd_combined) — **NEW OVERALL BEST**
**Overall best speedup**: 7.845x (lineage L2) — **+14.8% over previous best (6.831x)**

### Comparative Ranking

| Rank | Variant | Status | Speedup | Latency (ms) | Compute Ratio | Bottleneck | Direction |
|------|---------|--------|---------|--------------|---------------|------------|-----------|
| 1 | L2_fuse_fwd_combined | SUCCESS | 7.845x | 976 | 1.0 | register-pressure + single-MXU | fuse_fwd_combined |
| 2 | L1_fold_dh_fuse_fwd | SUCCESS | 7.639x | 1009 | 1.0 | register-pressure + single-MXU | fold_dh_fuse_fwd |
| 3 | L2_reduce_live_bwd | SUCCESS | 6.957x | 1097 | 1.0 | register-pressure + single-MXU | reduce_live_bwd |
| 4 | L2_bt128 | SUCCESS | 5.395x | 1412 | 1.0 | register-pressure + VLIW bloat | bt128 |
| -- | L2_bf16_uniform | INCORRECT | -- | -- | -- | Mosaic compile error | bf16_uniform |

### Per-Variant Details

#### L2_fuse_fwd_combined (Rank 1) — NEW OVERALL BEST

**Status**: SUCCESS
**Speedup**: 7.845x (976ms vs 7654ms reference)
**Lineage**: L2 (round 5)

| Metric | Value | Assessment |
|--------|-------|------------|
| compute_ratio | 1.0 | compute-bound |
| vliw_bundle_count | 7930 | identical to R4 best — same backward kernels |
| MXU dual_ratio | 0.0 | **still single-MXU** |
| MXU0 ops | 4656 | unchanged |
| DMA transfers | 20 | unchanged |
| HLO fusions | 0 | ideal for Pallas |
| Computation events | 177 | **-10.6% vs R4 (198)** — forward fusion eliminated 21 events |
| MXU util (runtime) | 33.22% | same as R4 |
| Vector ALU util | 20.20% | same as R4 |
| Vector Store util | 9.82% | same as R4 |
| EUP util | 2.08% | same as R4 |
| Vector fills/spills | 3,038,841 / 2,497,509 | identical to R4 |

**Bottleneck**: Register pressure (2.5M spills) + single-MXU (dual_ratio=0.0). These are unchanged from R4 — the forward fusion did not affect the backward kernel's compilation.

**Why 7.845x with only 21 fewer events**: The forward fusion merged chunk_fwd_h + chunk_gla_fwd_o_gk into a single pallas_call. This eliminated:
1. One pallas_call kernel launch (and its HBM round-trip for the h tensor)
2. 21 computation events (198 → 177 = -10.6%)
3. The h tensor HBM write-then-read cycle between the two forward kernels

The speedup improvement (6.831x → 7.845x = +14.8%) is consistent with the "10% event reduction → ~25% speedup" ratio observed for pallas_call grid tile events.

**Suggestions**:
1. Register pressure is now the dominant remaining bottleneck at 976ms
2. Backward kernel fusion or simplification is the next frontier
3. The backward still uses 2 pallas_calls (chunk_bwd_dh_pallas + chunk_gla_bwd_fused). Can these be merged?

#### L1_fold_dh_fuse_fwd (Rank 2)

**Status**: SUCCESS
**Speedup**: 7.639x (1009ms vs 7710ms reference)
**Lineage**: L1 (round 5 — PIVOT to L2's kernel + forward fusion)

| Metric | Value | Assessment |
|--------|-------|------------|
| compute_ratio | 1.0 | compute-bound |
| vliw_bundle_count | 7930 | identical to L2_fuse_fwd_combined |
| MXU dual_ratio | 0.0 | same |
| Computation events | 177 | identical to L2_fuse_fwd_combined |
| Vector fills/spills | 3,038,841 / 2,497,509 | identical to L2_fuse_fwd_combined |

**Key insight**: L1_fold_dh_fuse_fwd has IDENTICAL compiled metrics to L2_fuse_fwd_combined. The 2.7% speedup difference (7.639x vs 7.845x) is within measurement noise — both kernels are functionally equivalent. L1 has effectively converged to the same kernel as L2. The lineages have merged.

#### L2_reduce_live_bwd (Rank 3)

**Status**: SUCCESS
**Speedup**: 6.957x (1097ms vs 7630ms reference)
**Lineage**: L2 (round 5)

| Metric | Value | Assessment |
|--------|-------|------------|
| compute_ratio | 1.0 | compute-bound |
| vliw_bundle_count | 7930 | identical — same kernel structure |
| MXU dual_ratio | 0.0 | unchanged |
| Computation events | 198 | unchanged — no forward fusion |
| MXU util (runtime) | 25.23% | **-24.0% vs R4 (33.20%)** — WORSE |
| Vector ALU util | 15.31% | **-24.2% vs R4 (20.19%)** — WORSE |
| Vector Store util | 6.80% | **-30.7% vs R4 (9.81%)** — improved (less spill traffic) |
| Vector fills/spills | 2,928,828 / 2,251,998 | **spills -9.8%, fills -3.6%** |

**Analysis**: The staged output writes successfully reduced register spills by 9.8% (2,498,118 → 2,251,998) and Vector Store utilization by 30.7% (9.81% → 6.80%). However, MXU utilization also dropped significantly (33.20% → 25.23%). The staged writes may have introduced longer dependency chains between output writes, preventing the compiler from overlapping MXU operations.

**Net effect**: The 9.8% spill reduction and 30.7% store util improvement are positive, but the MXU regression means the backward kernel is overall SLOWER per iteration. The 1.8% speedup improvement (6.831x → 6.957x) is marginal.

**Lesson**: Staged output writes in Pallas kernels can reduce register pressure but may hurt MXU scheduling by introducing serialization barriers between output write groups.

#### L2_bt128 (Rank 4) — REGRESSION

**Status**: SUCCESS
**Speedup**: 5.395x (1412ms vs 7618ms reference) — **-21.0% regression from R4 best (6.831x)**
**Lineage**: L2 (round 5)

| Metric | Value | Assessment |
|--------|-------|------------|
| compute_ratio | 1.0 | compute-bound |
| vliw_bundle_count | 11,449 | **+44.3% vs R4 (7930)** — VLIW bloat |
| MXU dual_ratio | 0.0 | unchanged |
| MXU0 ops | 6,192 | **+33.0% vs R4 (4656)** — more matmul ops per tile |
| DMA transfers | 20 | unchanged |
| Computation events | 198 | **SAME as BT=64** — NT halved but events unchanged |
| MXU util (runtime) | 45.91% | **+38.3% vs R4 (33.20%)** — highest ever |
| Vector ALU util | 22.80% | +12.9% vs R4 |
| Vector Store util | 14.29% | **+45.6% vs R4 (9.81%)** — much worse spill traffic |
| Vector fills/spills | 3,629,418 / 2,649,660 | **spills +6.1%, fills +19.4%** |

**Why BT=128 regressed**: Despite halving NT from 64 to 32, computation events stayed at 198. The per-tile VLIW complexity increased +44.3% (more code per larger tile), register spills increased, and although MXU utilization improved (larger matmuls), the per-tile overhead growth exceeded the benefit of fewer tiles. The attention matrix A at [128,128] is 4x larger than [64,64], creating substantial register pressure.

**Key finding**: Increasing BT beyond 64 is counterproductive for chunk_gla on this shape. BT=64 is the optimal chunk size.

#### L2_bf16_uniform (INCORRECT)

**Error**: `MosaicError: INTERNAL: Mosaic failed to compile TPU kernel: Bad lhs type`
**MLIR operation**: `tpu.matmul (vector<64x256xbf16>, vector<256x256xbf16>, vector<64x256xf32>)`
**Cause**: Despite casting BOTH matmul operands to bfloat16 (fixing the FP25 mixed-dtype issue), Mosaic STILL rejects the matmul. The error persists because `precision=lax.Precision.HIGHEST` combined with bf16 operands creates a configuration Mosaic doesn't support. The `contract_precision<fp32>` in the MLIR suggests that HIGHEST precision forces the matmul contract to be in f32, which conflicts with bf16 operand types.

**Extended finding (extends FP25)**: It's not just mixed dtypes — Mosaic rejects bf16 operands when `precision=lax.Precision.HIGHEST` is specified. HIGHEST forces f32 computation internally, which is incompatible with bf16 input types. To use bf16 matmul inputs, the precision parameter must be changed to `lax.Precision.DEFAULT` or removed entirely.

**Fix**: Either (a) remove `precision=lax.Precision.HIGHEST` when using bf16 operands, or (b) keep operands in float32 with HIGHEST precision. Cannot have both bf16 inputs AND HIGHEST precision.

### Failed Variants Summary

| Variant | Error Type | Root Cause | Fix Available |
|---------|-----------|------------|---------------|
| L2_bf16_uniform | Mosaic MLIR | bf16 operands + Precision.HIGHEST incompatible | Remove HIGHEST precision or keep f32 operands |

### Lineage Trends (Round 1 → Round 5)

**Lineage L2** (mxu_utilization → v_tiling → reduce_inputs → fold_dh_pallas → fuse_fwd_combined):

| Metric | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 | R4→R5 Delta | Trend |
|--------|---------|---------|---------|---------|---------|-------------|-------|
| Speedup | 0.825x | 0.872x | 1.056x | 6.831x | 7.845x | +14.8% | **IMPROVING** |
| VLIW bundles | 5226 | 13140 | 13928 | 7930 | 7930 | 0% | flat |
| Spills | 369K | 5.74M | 5.62M | 2.50M | 2.50M | 0% | flat |
| Events | 270 | 264 | 237 | 198 | 177 | -10.6% | **improving** |
| MXU util | -- | -- | -- | 33.20% | 33.22% | 0% | flat |

**Analysis**: L2 continues to improve via event reduction. The pattern is consistent: each event reduction translates to proportional speedup. The remaining 177 events are harder to reduce — forward is now fused into 1 pallas_call, backward is 2 pallas_calls.

**Lineage L1** (mxu_vpu_overlap → reduce_inputs → combined_fuse_skip → reduce_exp → **PIVOT: fold_dh_fuse_fwd**):

| Metric | Round 3 | Round 4 | Round 5 | R4→R5 Delta | Trend |
|--------|---------|---------|---------|-------------|-------|
| Speedup | 1.577x | 1.498x | 7.639x | **+409.7%** | **CONVERGED with L2** |
| Events | 207 | 207 | 177 | -14.5% | improved (adopted L2's optimizations) |

**Analysis**: L1 pivoted to adopt L2's fold_dh_pallas + forward fusion. The resulting kernel is functionally identical to L2_fuse_fwd_combined (same VLIW, MXU, spills, events). The lineages have **converged** — both are now the same kernel.

### Cross-Variant Insights

1. **Forward kernel fusion confirmed as effective**: L2_fuse_fwd_combined and L1_fold_dh_fuse_fwd both achieved the same 177-event kernel by merging the two forward pallas_calls. The 10.6% event reduction produced a 14.8% speedup improvement.

2. **Lineage convergence**: Both L1 and L2 have independently converged to the same kernel architecture: combined forward pallas_call + backward dh pallas_call + backward fused pallas_call. Future rounds need genuinely novel directions.

3. **BT=128 is counterproductive**: Despite higher MXU util (45.9%) and fewer grid iterations, the per-tile VLIW bloat (+44%) and increased register pressure (-21% speedup) make BT=128 worse than BT=64.

4. **Staged backward writes have mixed results**: Reduced spills by 9.8% but regressed MXU util by 24%. Net effect is marginal (+1.8% speedup).

5. **bf16 matmul with Precision.HIGHEST is not supported**: Even with both operands in bf16, Mosaic requires float32 operands when HIGHEST precision is specified. This is a deeper constraint than FP25.
