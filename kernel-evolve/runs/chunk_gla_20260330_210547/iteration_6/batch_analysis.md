## Round 6 Batch Analysis

**Variants evaluated**: 5
**Successes**: 1 | **Failures**: 4
**Best speedup this round**: 7.731x (L2_fori_v_bwd) — REGRESSION from 7.845x
**Overall best speedup**: 7.845x (lineage L2, unchanged)

### Comparative Ranking

| Rank | Variant | Status | Speedup | Latency (ms) | Compute Ratio | Bottleneck | Direction |
|------|---------|--------|---------|--------------|---------------|------------|-----------|
| 1 | L2_fori_v_bwd | SUCCESS | 7.731x | 986 | 1.0 | register-pressure + single-MXU | fori_v_bwd |
| -- | L2_precision_default | INCORRECT | -- | -- | -- | max_diff=23.91 > atol=10.0 | precision_default |
| -- | L2_bwd_fuse_dh | INCORRECT | -- | -- | -- | BlockSpec rank mismatch | bwd_fuse_dh |
| -- | L2_recompute_dh | INCORRECT | -- | -- | -- | BlockSpec rank mismatch | recompute_dh |
| -- | L1_bwd_grid_kv | INCORRECT | -- | -- | -- | Block dim alignment violation | bwd_grid_kv |

### Per-Variant Details

#### L2_fori_v_bwd (Rank 1) — REGRESSION

**Status**: SUCCESS
**Speedup**: 7.731x (986ms vs 7622ms reference)
**Lineage**: L2 (round 6)

| Metric | Value | vs R5 Best | Assessment |
|--------|-------|------------|------------|
| compute_ratio | 1.0 | unchanged | compute-bound |
| MXU util (runtime) | 13.50% | -59.3% (33.22%) | **SEVERE regression** |
| Vector ALU util | 9.50% | -53.0% (20.20%) | major regression |
| Vector Store util | 5.20% | -47.0% (9.82%) | improved (less spill traffic) |
| Vector fills/spills | 5,573,297 / 4,148,640 | **+83.4% / +66.1%** | **MUCH WORSE** |

**Analysis**: The fori_loop approach WORSENED register pressure dramatically (spills +66%, fills +83%) rather than reducing it. The fori_loop + lax.cond pattern creates carry tuples that the compiler materializes in VMEM, adding MORE spill traffic. MXU util dropped to 13.5% — the fori_loop scheduling overhead dominates.

**Key finding**: `jax.lax.fori_loop` with `lax.cond` inside does NOT reduce register pressure in Pallas TPU kernels. The carry tuple and conditional branches create additional VMEM traffic that exceeds the benefit of hiding iterations.

#### L2_precision_default (INCORRECT)

**Error**: Correctness failure, max_diff=23.91 (atol=10.0)
**Cause**: `lax.Precision.DEFAULT` allows bf16 accumulation for matmuls. With BT=64 and K=128, the sum of 64×128=8192 products at bf16 precision loses significant bits. The error accumulates across all chunks and the loss sum, exceeding atol=10.0.

**Key finding**: `Precision.DEFAULT` is NOT safe for chunk_gla. The kernel requires `Precision.HIGHEST` for correctness at the current atol=10.0 threshold.

#### L2_bwd_fuse_dh (INCORRECT)

**Error**: `Block shape (= (Blocked(1), Blocked(1), Blocked(128), Blocked(128))) must have the same number of dimensions as the array shape (2, 16, 64, 128, 128)`
**Cause**: The fused backward+dh kernel uses a 4D BlockSpec `(1, 1, 128, 128)` for the h input, but the h array has 5 dimensions `(B, NT, H, K, V) = (2, 16, 64, 128, 128)`. The BlockSpec needs 5D to match.
**Fix**: Use 5D BlockSpec `(1, 1, 1, 128, 128)` with index_map returning 5 indices.

#### L2_recompute_dh (INCORRECT)

**Error**: Same BlockSpec rank mismatch as L2_bwd_fuse_dh.
**Cause**: Identical bug — 4D BlockSpec for 5D h array.
**Fix**: Same as L2_bwd_fuse_dh.

#### L1_bwd_grid_kv (INCORRECT)

**Error**: `Block shape (= (Blocked(1), Blocked(1), Blocked(64), Blocked(64))) ... array shape (16, 128, 64, 128)` — block last dim 64 != array dim 128.
**Cause**: V-tiling with BV_inner=64 creates block shape `(1, 1, BT, BV_inner)` = `(1, 1, 64, 64)`, but the V-tiled input arrays still have V=128 as the last dimension (the tiling wasn't applied to the actual array reshape). Either the input arrays need to be reshaped/re-indexed for V sub-tiles, or the BlockSpec index_map needs to handle V sub-tile indexing.
**Fix**: The BlockSpec needs to use full array dims with an index_map that selects the V sub-tile range, not a reduced block shape.

### Failed Variants Summary

| Variant | Error Type | Root Cause | Fixable |
|---------|-----------|------------|---------|
| L2_precision_default | Correctness | DEFAULT precision insufficient | No — fundamental |
| L2_bwd_fuse_dh | BlockSpec rank | 4D spec for 5D array | Yes — use 5D BlockSpec |
| L2_recompute_dh | BlockSpec rank | 4D spec for 5D array | Yes — use 5D BlockSpec |
| L1_bwd_grid_kv | Block dim mismatch | V sub-tile not reflected in array shape | Yes — fix array/spec indexing |

### Cross-Variant Insights

1. **Backward kernel fusion bugs are fixable**: Both L2_bwd_fuse_dh and L2_recompute_dh failed with the SAME BlockSpec rank bug. The optimization approach is sound — the implementation just needs corrected array indexing. These should be retried in Round 7.

2. **fori_loop + lax.cond does NOT reduce register pressure**: L2_fori_v_bwd proved that using `jax.lax.fori_loop` with `lax.cond` inside a Pallas kernel INCREASES register pressure (+66% spills) rather than reducing it. The carry tuple and conditional branches add VMEM overhead.

3. **Precision.DEFAULT is not viable**: max_diff=23.91 at DEFAULT precision means the kernel genuinely needs HIGHEST precision for correctness. This closes off precision reduction as an optimization direction.

4. **Round 6 was exploratory**: Despite 4/5 failures, the round produced valuable negative results that constrain the solution space. The backward fusion direction remains the most promising avenue but needs correct BlockSpec implementation.
