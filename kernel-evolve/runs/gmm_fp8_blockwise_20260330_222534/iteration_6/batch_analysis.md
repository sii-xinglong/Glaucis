## Round 6 Batch Analysis

**Variants evaluated**: 5
**Successes**: 5 | **Failures**: 0
**Best speedup this round**: 2.847x (L2_tgmm_tm1024)
**Overall best speedup**: 2.847x (lineage L2, +2.8% over Round 5's 2.769x)

### Delta vs Baseline
| Metric | Baseline | Round 5 Best | Round 6 Best | Delta R5→R6 |
|--------|----------|-------------|-------------|-------------|
| Speedup | 1.015x | 2.769x | 2.847x | +2.8% |
| Latency | 10.698ms | 3.800ms | 3.611ms | -5.0% |
| VLIW | 23,462 | 23,462 | 12,951 | -44.8% |
| MXU ops | 1,792 | 1,792 | 896 | -50.0% |
| Spills | 156 | 156 | 156 | 0 |

### Comparative Ranking

| Rank | Variant | Speedup | Latency (ms) | VLIW | MXU Ops | Spills | Compute Eff. | Tiling (fwd, bwd, tgmm) |
|------|---------|---------|-------------|------|---------|--------|-------------|--------------------------|
| 1 | L2_tgmm_tm1024 | 2.847x | ~3.61ms | 12,951 | 896 | 156 | 19.80% | (256,512,256), (256,512,128), (1024,512,128) |
| 2 | L1_scale_bf16 | 2.819x | ~3.65ms | 23,462 | 896 | 9 | 19.59% | (256,512,256), (256,512,128), (2048,512,128) |
| 3 | L1_bwd_tk1024 | 2.728x | ~3.77ms | 23,462 | 896 | 234 | 18.97% | (256,512,256), (256,1024,128), (2048,512,128) |
| 4 | L1_fwd_tk1024 | 2.498x | ~4.11ms | 23,462 | 896 | 234 | 17.38% | (256,1024,256), (256,512,128), (2048,512,128) |
| 5 | L2_fwd_tn512 | 2.410x | ~4.44ms | 23,462 | 896 | 231 | 16.09% | (256,512,512), (256,512,128), (2048,512,128) |

### Key Findings

1. **tgmm TM=1024 is a major improvement**: L2_tgmm_tm1024 (2.847x) sets a new overall best. VLIW bundles dropped from 23,462 to 12,951 (-44.8%), and MXU ops halved from 1,792 to 896. The smaller tgmm tile (1024 vs 2048) eliminates half the tgmm computation per tile, reducing total kernel complexity dramatically. This is the opposite of FP13 (tgmm TM=4096 bloat) — the sweet spot is TM=1024, not 2048 or 4096.

2. **bf16 scales work on L1 kernel**: L1_scale_bf16 (2.819x, 9 spills) is a significant improvement over L1's R5 best (2.751x, 156 spills). bf16 scale computation massively reduces register pressure (156→9 spills). Previously, bf16 scales interfered with L2+TK=512 compilation (R4, 2.456x), but on L1's code path they are compatible.

3. **bwd TK=1024 slight regression**: L1_bwd_tk1024 (2.728x) is slightly worse than L1's 2.751x. The doubled K-tile doesn't help bwd_gmm — 234 spills (vs 156 baseline) from larger intermediate tensors.

4. **fwd TK=1024 significant regression**: L1_fwd_tk1024 (2.498x) is a major regression. Forward TK=1024 with TN=256 creates too-large tiles for the compiler to handle efficiently. TK=512 is the optimal forward K-tile.

5. **fwd TN=512 regression**: L2_fwd_tn512 (2.410x) — TN=512 for Gate/Up (N=512) means a single N-tile covering the entire N dimension. This is worse than TN=256 (two N-tiles), suggesting the compiler prefers some N-tiling granularity even when N is small.

### Optimal Tiling Update
**New optimal**: (256, 512, 256, 256, 512, 128, **1024**, 512, 128)
- Forward: TM=256 (per-group), TK=512, TN=256 (proven R5)
- bwd_gmm: TM=256, TK=512, TN=128 (proven R5)
- tgmm: **TM=1024** (down from 2048), TK=512, TN=128
