## Round 5 Batch Analysis

**Variants evaluated**: 5
**Successes**: 5 | **Failures**: 0
**Best speedup this round**: 2.769x (L2_fwd_tn256_tk512)
**Overall best speedup**: 2.769x (lineage L2, +4.5% over Round 4's 2.651x)

### Delta vs Baseline
| Metric | Baseline | Round 4 Best | Round 5 Best | Delta R4→R5 |
|--------|----------|-------------|-------------|-------------|
| Speedup | 1.015x | 2.651x | 2.769x | +4.5% |
| Latency | 10.698ms | 3.898ms | 3.800ms | -2.5% |

### Comparative Ranking

| Rank | Variant | Speedup | Latency (ms) | VLIW | MXU Ops | Spills | Compute Eff. | Tiling (fwd, bwd, tgmm) |
|------|---------|---------|-------------|------|---------|--------|-------------|--------------------------|
| 1 | L2_fwd_tn256_tk512 | 2.769x | ~3.80ms | 23,462 | 1,792 | 156 | 19.25% | (256,512,256), (256,512,128), (2048,512,128) |
| 2 | L1_fwd_tn256 | 2.751x | ~3.82ms | 23,462 | 1,792 | 156 | 19.13% | (256,512,256), (256,512,128), (2048,512,128) |
| 3 | L1_all_tn256 | 2.503x | ~4.19ms | 23,462 | 1,792 | 9,450 | 17.28% | (256,512,256), (256,512,256), (2048,512,128) |
| 4 | L1_tgmm_tk1024 | 2.318x | ~4.54ms | 33,116 | 3,328 | 234 | 15.59% | (256,512,128), (256,512,128), (2048,1024,128) |
| 5 | L1_bwd_tn256 | 2.295x | ~4.57ms | 23,462 | 1,792 | 9,447 | 15.41% | (256,512,128), (256,512,256), (2048,512,128) |

### Key Findings

1. **Forward TN=256 is a major win**: Both L1_fwd_tn256 (2.751x) and L2_fwd_tn256_tk512 (2.769x) show significant improvement with fwd TN=256. For Gate/Up (N=512), TN=256 halves N-grid from 4→2. For Down (N=2048), 16→8. Same VLIW/MXU profile but faster execution.

2. **L2 slightly edges L1 with same tiling**: L2_fwd_tn256_tk512 (2.769x) vs L1_fwd_tn256 (2.751x). Both use tiling (256,512,256, 256,512,128, 2048,512,128). The difference is small and may be measurement variance.

3. **Backward TN=256 causes catastrophic register pressure**: L1_bwd_tn256 (2.295x, 9,447 spills) is a massive regression from L1 best (2.651x, 156 spills). bwd_gmm TN=256 with transpose_rhs=True creates large intermediate tensors that overflow registers. bwd_gmm TN MUST stay at 128.

4. **Combined fwd+bwd TN=256 shows mixed results**: L1_all_tn256 (2.503x, 9,450 spills) — the forward TN=256 benefit partially offsets the backward TN=256 regression, but net result is still negative.

5. **tgmm TK=1024 causes complexity bloat**: L1_tgmm_tk1024 (2.318x) has 33,116 VLIW bundles (+41%) and 3,328 MXU ops (+86%). Despite nearly 2x MXU ops, latency is worse. tgmm TK is maximally tiled at 512.

### Optimal Tiling Update
**New optimal**: (256, 512, 256, 256, 512, 128, 2048, 512, 128)
- Forward: TM=256 (per-group), TK=512 (K-loop halving), **TN=256** (N-grid halving)
- bwd_gmm: TM=256 (per-group), TK=512 (K-loop halving), TN=128 (register safety)
- tgmm: TM=2048, TK=512, TN=128 (proven optimal)
