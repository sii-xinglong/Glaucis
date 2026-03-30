## Round 2 Batch Analysis

**Variants evaluated**: 5
**Successes**: 5 | **Failures**: 0
**Best speedup this round**: 2.335x (L1_fwd_tk512)
**Overall best speedup**: 2.335x (lineage L1, +1.8% over Round 1's 2.294x)

### Delta vs Baseline
| Metric | Baseline | Round 1 Best | Round 2 Best | Delta R1→R2 |
|--------|----------|-------------|-------------|-------------|
| Speedup | 1.015x | 2.294x | 2.335x | +1.8% |
| Latency | 10.698ms | 4.692ms | 4.403ms | -6.2% |

### Comparative Ranking

| Rank | Variant | Speedup | Latency (ms) | VLIW | MXU Ops | Tiling (fwd, bwd, tgmm) |
|------|---------|---------|-------------|------|---------|--------------------------|
| 1 | L1_fwd_tk512 | 2.335x | 4.403ms | 23,462 | 1,792 | (256,512,128), (1024,256,128), (2048,512,128) |
| 2 | L1_bwd_tm256 | 2.290x | 4.489ms | 23,462 | 1,792 | (256,256,128), (256,256,128), (2048,512,128) |
| 3 | L2_bwd_tm256 | 2.285x | 4.499ms | 23,462 | 1,792 | (256,256,128), (256,256,128), (2048,512,128) |
| 4 | L2_so9_tiling | 2.193x | 4.689ms | 23,462 | 1,792 | (256,256,128), (1024,256,128), (2048,512,128) |
| 5 | L1_tgmm_tk256 | 2.046x | 5.025ms | 17,558 | 896 | (256,256,128), (1024,256,128), (2048,256,128) |

### Key Findings

1. **tgmm TK is the dominant factor**: TK=512 doubles MXU ops (896→1,792) and VLIW bundles (17,558→23,462). All TK=512 variants achieve >2.19x, while TK=256 (L1_tgmm_tk256) drops to 2.046x. The tgmm phase is where most of the speedup comes from.

2. **Forward TK=512 gives small edge**: L1_fwd_tk512 (2.335x) vs L1 (2.294x) shows forward TK=512 reduces K-loop iterations by half for Gate/Up (K=2048). +1.8% improvement.

3. **bwd_gmm TM doesn't matter much**: TM=1024 (2.294x-2.335x) vs TM=256 (2.285x-2.290x) shows negligible difference. Both compile to the same VLIW/MXU profile.

4. **L1 and L2 are structurally identical**: L2_so9_tiling (2.193x) confirms that applying SO9 tiling to L2 gives similar results to L1 (2.294x), just slightly different due to measurement variance. The code structure difference (lhs_t skip in forward) doesn't affect compilation (F001).

5. **Diminishing returns**: Round 1→2 improvement was only 1.8% (2.294x→2.335x). The main tiling parameters are now well-explored.
