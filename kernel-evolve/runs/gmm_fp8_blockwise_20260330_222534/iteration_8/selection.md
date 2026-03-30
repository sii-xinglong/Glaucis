## Round 8 Selection

### No improvement — both lineages stagnant for 2 consecutive rounds

| Lineage | Best Variant | Speedup | Stagnant Rounds | Status |
|---------|-------------|---------|-----------------|--------|
| L2 | L2_tgmm_tm1024 (R6) | 2.847x | 2 | CONVERGED |
| L1 | L1_scale_bf16 (R6) | 2.819x | 2 | CONVERGED |

### Not Selected (All Round 8 Variants)

| Variant | Status | Speedup | Reason |
|---------|--------|---------|--------|
| L2_fwd_tm512 | SUCCESS | 2.807x | Regression (-1.4%) |
| L1_bwd_tk256_bf16 | SUCCESS | 2.702x | Regression (-4.1%) |
| L2_bwd_tm128 | SUCCESS | 2.559x | Sub-group TM regression |
| L2_tgmm_tm128 | SUCCESS | 2.283x | tgmm TM=128 too small |
| L1_tgmm_tm1024_f32 | SUCCESS | 2.241x | TM=1024 only works on L2 |

### Optimization Terminated

Both lineages have been stagnant for 2 consecutive rounds (7 and 8). The tiling parameter space has been exhaustively explored across 8 rounds and 40 variants. No further improvement is expected from tiling changes alone.

**Final result: 2.847x speedup (L2_tgmm_tm1024)**
