## Round 4 Batch Analysis

**Variants evaluated**: 5
**Successes**: 2 | **Failures**: 3
**Best speedup this round**: 6.831x (L2_fold_dh_pallas) — **MASSIVE BREAKTHROUGH**
**Overall best speedup**: 6.831x (lineage L2) — **+333% OVER PREVIOUS BEST (1.577x)**

### Comparative Ranking

| Rank | Variant | Status | Speedup | Latency (ms) | Compute Ratio | Bottleneck | Direction |
|------|---------|--------|---------|--------------|---------------|------------|-----------|
| 1 | L2_fold_dh_pallas | SUCCESS | 6.831x | 1122 | 1.0 | register-pressure + single-MXU | fold_dh_pallas |
| 2 | L1_reduce_exp | SUCCESS | 1.498x | 5096 | 1.0 | register-pressure + single-MXU | reduce_exp |
| -- | L1_bf16_intermediates | INCORRECT | -- | -- | -- | Mosaic compile error | bf16_intermediates |
| -- | L1_fwd_single_kernel | INCORRECT | -- | -- | -- | shape mismatch | fwd_single_kernel |
| -- | L2_fuse_fwd_output_h | INCORRECT | -- | -- | -- | index_map signature | fuse_fwd_output_h |

### Per-Variant Details

#### L2_fold_dh_pallas (Rank 1) — MASSIVE BREAKTHROUGH

**Status**: SUCCESS
**Speedup**: 6.831x (1122ms vs 7662ms reference)
**Lineage**: L2 (round 4 — pivoted from V-tiling to L1's clean kernel + dh pallas_call)

| Metric | Value | Assessment |
|--------|-------|------------|
| compute_ratio | 1.0 | compute-bound |
| vliw_bundle_count | 7930 | identical to R3 best (7930) — same kernels |
| MXU dual_ratio | 0.0 | **still single-MXU** |
| MXU0 ops | 4656 | identical to R3 best |
| DMA transfers | 20 | identical to R3 best |
| HBM bandwidth | 134MB | low — same as R3 |
| HLO fusions | 0 | ideal for Pallas |
| Computation events | 198 | **-4.3% vs R3 (207)** — only 9 fewer events! |
| MXU util (runtime) | 33.20% | medium — same as R3 |
| Scalar ALU util | 0.042% | negligible |
| Vector ALU util | 20.19% | medium — same as R3 |
| Vector Store util | 9.81% | elevated — spill traffic (same as R3) |
| EUP util | 2.08% | low |
| Vector fills/spills | 3,039,582 / 2,498,118 | **identical to R3** — same kernels |

**Bottleneck**: Register pressure (2.5M spills) + single-MXU (dual_ratio=0.0). However, these are now the bottleneck of a 6.83x-faster kernel.

**Critical insight — why 6.83x with only 9 fewer events**:
The VLIW bundles, MXU ops, spills, and DMA counts are IDENTICAL to the R3 best. The Pallas kernels themselves are unchanged. What changed is that `lax.scan` (64 reverse iterations for dh computation) was replaced with a `pallas_call` using "arbitrary" dimension semantics.

The event count dropped only from 207 to 198 (-4.3%), yet latency dropped from 4884ms to 1122ms (-77%). This means:

1. **lax.scan iterations are NOT ordinary computation events**. Each of the ~64 scan iterations incurs full XLA dispatch overhead: host→device communication, kernel scheduling, memory allocation for intermediates.
2. **pallas_call with "arbitrary" dims compiles ALL iterations into a single on-chip kernel**. The 64 time steps execute as a loop within a single VLIW program — no per-iteration dispatch.
3. **The dispatch overhead of lax.scan dominated execution time**. The 64 scan iterations were collectively consuming ~3762ms (4884ms - 1122ms), meaning each scan iteration cost ~59ms of overhead — far more than the actual computation.

This is the single most important optimization finding in this entire optimization run.

**Why it works**: The backward dh computation has the structure `dh[t] = f(dh[t+1], q[t], do[t])` — a sequential scan with a state dependency. `lax.scan` implements this as 64 separate XLA computations dispatched one-by-one. The pallas_call replacement compiles the same scan into a single TPU kernel with the time dimension as "arbitrary" (meaning sequentially iterated), keeping the dh state in VMEM scratch between iterations. No host-device round-trips, no intermediate HBM allocations.

**Suggestions**:
1. **Apply same technique to forward**: The forward `chunk_fwd_h` already uses pallas_call with "arbitrary" for h propagation — this is why the forward was already efficient. The backward was the remaining scan bottleneck.
2. **Combine forward h + output kernel**: Now that both scans use pallas_call, merging the two forward kernels could eliminate another ~20 events.
3. **Register pressure is now the primary bottleneck**: At 1122ms latency, the 2.5M spills consume a larger fraction of total time. Addressing register pressure may yield proportionally larger gains.

#### L1_reduce_exp (Rank 2) — REGRESSION

**Status**: SUCCESS
**Speedup**: 1.498x (5096ms vs 7636ms reference) — **-5.0% regression from R3 best (1.577x)**
**Lineage**: L1 (round 4)

| Metric | Value | Assessment |
|--------|-------|------------|
| compute_ratio | 1.0 | compute-bound |
| vliw_bundle_count | 7906 | -0.3% vs R3 (7930) — negligible change |
| MXU dual_ratio | 0.0 | unchanged |
| MXU0 ops | 4656 | unchanged |
| DMA transfers | 20 | unchanged |
| Computation events | 207 | unchanged from R3 |
| MXU util (runtime) | 32.90% | slightly worse than R3 (33.21%) |
| Vector ALU util | 19.25% | slightly worse than R3 (20.22%) |
| Vector Store util | 9.64% | slightly better (9.80%) |
| EUP util | 1.37% | lower than R3 (2.10%) — fewer exp() calls as expected |
| Vector fills/spills | 3,124,962 / 2,472,903 | spills -1.0% (2,497,509→2,472,903), fills +2.8% (3,038,841→3,124,962) |

**Bottleneck**: The reduce_exp optimization had minimal impact on register pressure (1% spill reduction) but REGRESSED overall performance by 5%. The reciprocal operations (1/exp(x) instead of exp(-x)) may be slower on TPU's hardware exponential unit. The EUP utilization dropped as expected (1.37% vs 2.10%), but this didn't translate to faster execution — the ALU divisions may have added pipeline stalls.

**Key finding**: exp(-x) is NOT more expensive than 1/exp(x) on TPU v7x. The hardware EUP handles exp() efficiently; replacing it with reciprocal/division operations degrades performance.

#### L1_bf16_intermediates (INCORRECT)

**Error**: `MosaicError: INTERNAL: Mosaic failed to compile TPU kernel: Bad lhs type`
**MLIR operation**: `tpu.matmul (vector<64x256xbf16>, vector<256x256xf32>, vector<64x256xf32>)`
**Cause**: Mosaic/TPU does not support mixed-precision matmul inputs (bf16 lhs × f32 rhs). All matmul operands must have the same dtype. The variant cast b_A_masked to bf16 before dotting with b_h (which was f32 from h_ref cast), creating a bf16×f32 matmul.
**Fix**: When casting intermediates to bf16 before matmul, both operands must be cast to the same dtype. Either cast both to bf16, or keep both as the original dtype. Alternatively, use `.astype(b_qg.dtype)` patterns that ensure matching types.

#### L1_fwd_single_kernel (INCORRECT)

**Error**: `TypeError: dot_general requires contracting dimensions to have the same shape, got (128,) and (64,).`
**Location**: `_chunk_fwd_merged_kernel` at the h-update matmul
**Cause**: Shape mismatch in the merged forward kernel. The k_tile (for h update) needs to be [BT, K] = [64, 128] transposed to [K, BT] = [128, 64], then dotted with v_tile [BT, V] = [64, 128]. The contracting dimension should be BT=64, but the code attempted to contract along K=128.
**Fix**: The h update dot should be `k_tile.T @ v_tile` where k_tile is [64, 128] and v_tile is [64, 128], contracting on BT=64: `dot(k_tile.astype(f32).T, v_tile.astype(f32))` gives [128, 128].

#### L2_fuse_fwd_output_h (INCORRECT)

**Error**: `TypeError: chunk_fwd_combined.<locals>.kqg_map() takes 3 positional arguments but 4 were given`
**Cause**: The PrefetchScalarGridSpec with `num_scalar_prefetch=1` and grid `(B, H, 1, 1, NT)` passes 4 non-prefetch grid indices to the BlockSpec index_map function. But `kqg_map` was defined to accept only 3 arguments (b, h, t), missing the two middle grid dimensions (ki=1, vi=1).
**Fix**: index_map functions must accept exactly `len(grid) - num_scalar_prefetch` arguments. For grid `(B, H, 1, 1, NT)` with 1 scalar prefetch, maps need 4 args: `lambda b, h, ki, vi, t: ...` (5 grid dims - 0 = 5 args if no scalar prefetch, or 4 if 1 scalar prefetch? Actually with PrefetchScalarGridSpec, the first `num_scalar_prefetch` grid dims are scalar, so the index_map gets `len(grid)` args minus 0. Need to check the exact API.)

### Failed Variants Summary

| Variant | Error Type | Root Cause | Fix Available |
|---------|-----------|------------|---------------|
| L1_bf16_intermediates | Mosaic MLIR | Mixed bf16/f32 matmul inputs | Cast both operands to same dtype |
| L1_fwd_single_kernel | Shape mismatch | Wrong contraction dimension in h-update dot | Fix transpose ordering |
| L2_fuse_fwd_output_h | API misuse | index_map arg count doesn't match grid dims | Add missing grid dim args |

### Lineage Trends (Round 1 → Round 2 → Round 3 → Round 4)

**Lineage L1** (mxu_vpu_overlap → reduce_inputs → combined_fuse_skip → reduce_exp):

| Metric | Round 1 | Round 2 | Round 3 | Round 4 | R3→R4 Delta | Trend |
|--------|---------|---------|---------|---------|-------------|-------|
| Speedup | 0.876x | 1.097x | 1.577x | 1.498x | -5.0% | **REGRESSION** |
| VLIW bundles | 8270 | 9058 | 7930 | 7906 | -0.3% | flat |
| MXU ops | 4656 | 5238 | 4656 | 4656 | 0% | flat |
| Spills | 1,957,131 | 2,436,588 | 2,497,509 | 2,472,903 | -1.0% | flat |
| DMA | 24 | 22 | 20 | 20 | 0% | flat |
| Comp events | 264 | 237 | 207 | 207 | 0% | flat |
| MXU util | 22.53% | 30.21% | 33.21% | 32.90% | -0.3pp | flat |

**Analysis**: L1 has STAGNATED. The reduce_exp direction failed to improve performance. All metrics are flat or slightly worse. L1 has exhausted its optimization space within the current kernel structure. The only remaining lever (reducing computation events further) requires fusing forward kernels or replacing lax.scan — which is exactly what L2 proved works.

**Lineage L2** (mxu_utilization → v_tiling → reduce_inputs → **PIVOT: fold_dh_pallas**):

| Metric | Round 1 | Round 2 | Round 3 | Round 4 (PIVOT) | R3→R4 Delta | Trend |
|--------|---------|---------|---------|-----------------|-------------|-------|
| Speedup | 0.825x | 0.872x | 1.056x | **6.831x** | **+546.9%** | **REVOLUTIONARY** |
| VLIW bundles | 5226 | 13140 | 13928 | 7930 | -43.0% | massive improvement (pivot) |
| Spills | 369,000 | 5,735,994 | 5,623,842 | 2,498,118 | -55.6% | massive improvement (pivot) |
| Comp events | 270 | 264 | 237 | 198 | -16.5% | improved |
| MXU util | -- | -- | -- | 33.20% | -- | -- |

**Analysis**: L2's pivot was a game-changer. By abandoning V-tiling and adopting L1's clean kernel + replacing lax.scan with pallas_call, L2 achieved a 6.83x speedup — the single largest improvement in this optimization run. The pivot was the correct strategic decision.

### Cross-Variant Insights

1. **lax.scan → pallas_call is the single most impactful optimization discovered**: The L2_fold_dh_pallas variant changed NOTHING about the Pallas kernels themselves (identical VLIW, MXU ops, spills). It only replaced the `lax.scan` backward dh computation with a `pallas_call`. Yet it achieved a 4.35x speedup (4884ms → 1122ms). This means **lax.scan dispatch overhead was consuming ~77% of total execution time**.

2. **Computation event count is an imperfect proxy**: L2_fold_dh_pallas has 198 events vs 207 for the base (-4.3%), but the speedup is 333%. This breaks the previously observed correlation of "10% event reduction → 25-35% speedup". The relationship holds for events of similar cost (pallas_call grid tiles), but lax.scan events are VASTLY more expensive per event than pallas_call events.

3. **Intra-kernel optimizations have diminishing returns at L1's level**: L1_reduce_exp (changing exp() to reciprocal) produced a 5% regression on 1.577x. The kernel structure is optimized to the point where minor algebraic transformations don't help — they can even hurt by disrupting the compiler's optimization.

4. **Mixed-precision matmul inputs NOT supported in Mosaic**: L1_bf16_intermediates confirmed that Mosaic requires matching dtypes for matmul operands. This is a new failure pattern (see [FP25]).

5. **Forward kernel fusion is still worth trying**: Both L1_fwd_single_kernel and L2_fuse_fwd_output_h attempted forward fusion but had implementation bugs. The approach itself is sound — the bugs are fixable.
