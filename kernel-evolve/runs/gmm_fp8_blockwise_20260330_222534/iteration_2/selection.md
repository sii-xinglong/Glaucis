## Round 2 Selection

### Selected (Top-2)

| Lineage | Variant | Speedup | Direction | Rationale |
|---------|---------|---------|-----------|-----------|
| L1 | L1_fwd_tk512 | 2.335x | tiling_specialization | New overall best, forward TK=512 gives edge |
| L2 | L1_bwd_tm256 | 2.290x | tiling_specialization | Second best, uniform TM=256 across phases |

### Not Selected

| Variant | Status | Speedup | Reason |
|---------|--------|---------|--------|
| L2_bwd_tm256 | SUCCESS | 2.285x | Below top-K, structurally same as L1_bwd_tm256 |
| L2_so9_tiling | SUCCESS | 2.193x | Below top-K |
| L1_tgmm_tk256 | SUCCESS | 2.046x | tgmm TK=256 regresses — halves MXU ops |

### Lineage Updates

- **L1**: Updated best_speedup 2.294 → 2.335, best_kernel → L1_fwd_tk512, reset stagnant_rounds
- **L2**: Updated best_speedup 1.944 → 2.290, best_kernel → L1_bwd_tm256, direction → tiling_specialization

### Lineages.json Status

- Active lineages: 2
- Pruned lineages: 0
- Current round: 2
