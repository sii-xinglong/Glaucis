## Round 3 Selection

### Selected (Top-2)

| Lineage | Variant | Speedup | Direction | Rationale |
|---------|---------|---------|-----------|-----------|
| L2 | L2_bwd_tm256_tk512 | 2.460x | tiling_specialization | New overall best, +5.4% over previous best. bwd_gmm TM=256+TK=512 compound effect |
| L2 | L2_scale_bf16 | 2.342x | quant_precision | Second best, bf16 scales marginally cheaper without correctness loss |

### Not Selected

| Variant | Status | Speedup | Reason |
|---------|--------|---------|--------|
| L1_bwd_tk512 | SUCCESS | 2.307x | Regressed from L1's 2.335x. TK=512 doesn't help with TM=1024 |
| L1_tgmm_tm4096 | SUCCESS | 2.277x | Regressed, complexity bloat (47K VLIW bundles vs 23K) |
| L1_fwd_bf16_out | INCORRECT | — | Forward bf16 accumulation causes max_diff=2065 |

### Lineage Updates

- **L1**: No improvement this round (best variants: 2.307x and 2.277x, both below previous 2.335x). stagnant_rounds → 1
- **L2**: Updated best_speedup 2.290 → 2.460, best_kernel → L2_bwd_tm256_tk512, reset stagnant_rounds

### Lineages.json Status

- Active lineages: 2
- Pruned lineages: 0
- Current round: 3
