## Round 8 Batch Analysis

**Variants evaluated**: 5
**Successes**: 5 | **Failures**: 0
**Best speedup this round**: 2.807x (L2_fwd_tm512) — regression from overall best 2.847x
**Overall best speedup**: 2.847x (lineage L2, unchanged — 2nd consecutive stagnation round)

### Comparative Ranking

| Rank | Variant | Speedup | Latency (ms) | Direction | Notable |
|------|---------|---------|-------------|-----------|---------|
| 1 | L2_fwd_tm512 | 2.807x | ~3.66ms | fwd TM=512 | Near-best but regression (-1.4%) |
| 2 | L1_bwd_tk256_bf16 | 2.702x | ~3.80ms | bwd TK=256 | Regression from L1's 2.819x |
| 3 | L2_bwd_tm128 | 2.559x | ~4.01ms | bwd TM=128 | Sub-group TM regression |
| 4 | L2_tgmm_tm128 | 2.283x | ~4.50ms | tgmm TM=128 | Smallest tgmm tile, major regression |
| 5 | L1_tgmm_tm1024_f32 | 2.241x | ~4.58ms | tgmm TM=1024 | TM=1024 ONLY works on L2 code path |

### Key Findings

1. **tgmm TM=1024 is L2-specific**: L1_tgmm_tm1024_f32 (2.241x) is a massive regression from L1's baseline (2.751x with TM=2048). The same tgmm TM=1024 that gives 2.847x on L2 gives only 2.241x on L1. This confirms deep code-path sensitivity — the optimization cannot be ported between lineages.

2. **fwd TM=256 confirmed optimal**: L2_fwd_tm512 (2.807x) is close but still worse than L2's best (2.847x with TM=256). Per-group alignment (TM=256 = M/G) is robust.

3. **bwd TM=128 is significantly worse**: L2_bwd_tm128 (2.559x vs 2.847x). Sub-group bwd tiling creates too many grid iterations. TM=256 is optimal for bwd.

4. **tgmm TM=128 is terrible**: L2_tgmm_tm128 (2.283x) — 64 tgmm tiles per group creates overwhelming grid iteration overhead. Confirms TM=1024 as the tgmm sweet spot (FP25).

5. **bwd TK=256 + bf16 scales don't compound**: L1_bwd_tk256_bf16 (2.702x vs 2.819x). bf16 scales don't compensate for the bwd TK regression.

### Convergence Assessment

**Both lineages have stagnated for 2 consecutive rounds.** The tiling parameter space has been exhaustively explored:
- Forward: TM=256 (per-group), TK=512, TN=256 — all variations tested and boundaries confirmed
- Backward: TM=256, TK=512, TN=128 — all variations tested, TN=64/256 both fail
- tgmm: TM=1024 (L2-specific), TK=512, TN=128 — full range [128-4096] tested

**Recommendation: Terminate optimization. Final best: 2.847x (L2_tgmm_tm1024).**
