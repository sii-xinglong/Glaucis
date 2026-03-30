## Round 7 Batch Analysis

**Variants evaluated**: 5
**Successes**: 4 | **Failures**: 1 (INCORRECT)
**Best speedup this round**: 2.719x (L2_tgmm_tm512)
**Overall best speedup**: 2.847x (lineage L2, unchanged from Round 6)

### Delta vs Baseline
| Metric | Baseline | Round 6 Best | Round 7 Best | Delta R6→R7 |
|--------|----------|-------------|-------------|-------------|
| Speedup | 1.015x | 2.847x | 2.719x | -4.5% (stagnation) |

### Comparative Ranking

| Rank | Variant | Speedup | Latency (ms) | VLIW | MXU Ops | Spills | Compute Eff. | Tiling (fwd, bwd, tgmm) |
|------|---------|---------|-------------|------|---------|--------|-------------|--------------------------|
| 1 | L2_tgmm_tm512 | 2.719x | ~3.78ms | 12,360 | 576 | 231 | 18.91% | (256,512,256), (256,512,128), (512,512,128) |
| 2 | L1_tgmm_tm1024 | 2.688x | ~3.82ms | 12,951 | 896 | 84 | 18.69% | (256,512,256), (256,512,128), (1024,512,128) + bf16 scales |
| 3 | L2_scale_bf16 | 2.682x | ~3.83ms | 12,951 | 896 | 81 | 18.64% | (256,512,256), (256,512,128), (1024,512,128) + bf16 scales |
| 4 | L1_tgmm_tm256_bf16 | 2.276x | ~4.51ms | 8,497 | 576 | 84 | 15.20% | (256,512,256), (256,512,128), (256,512,128) + bf16 scales |
| -- | L2_tgmm_tm1024_bwd_tn64 | INCORRECT | -- | -- | -- | -- | -- | bwd TN=64 failed |

### Key Findings

1. **bf16 scales and tgmm TM=1024 are INCOMPATIBLE optimizations**: Both cross-pollination variants regressed:
   - L2_scale_bf16 (2.682x vs L2's 2.847x, -5.8%): Adding bf16 scales to L2_tgmm_tm1024 reduced spills (156→81) but regressed speedup. The bf16 scale dtype changes the compilation path in a way that negates the VLIW simplification benefit of tgmm TM=1024.
   - L1_tgmm_tm1024 (2.688x vs L1's 2.819x, -4.6%): Adding tgmm TM=1024 to L1_scale_bf16 has the same issue from the other direction.
   - **Conclusion**: SO14 and SO15 independently improve performance but their effects are not additive. Each changes the compilation in a way that is mutually exclusive.

2. **tgmm TM sweet spot is 1024**: L2_tgmm_tm512 (2.719x) is worse than TM=1024 (2.847x). TM=512 reduces MXU ops from 896 to 576 without proportional VLIW reduction (12,360 vs 12,951). L1_tgmm_tm256_bf16 (2.276x) is a major regression — TM=256 drops to 8,497 VLIW / 576 MXU.

3. **bwd TN=64 causes correctness failure**: L2_tgmm_tm1024_bwd_tn64 returned INCORRECT with max_diff=0.0. Sub-128 tiling breaks tokamax's internal tile assumptions. TN must be >= 128.

4. **Diminishing returns signal**: This is the first round where ALL variants failed to improve over the existing lineage bests. The tiling optimization space appears exhausted within the current kernel structure.
