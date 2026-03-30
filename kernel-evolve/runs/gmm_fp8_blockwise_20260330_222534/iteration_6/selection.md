## Round 6 Selection

### Selected (Top-2)

| Lineage | Variant | Speedup | Direction | Rationale |
|---------|---------|---------|-----------|-----------|
| L2 | L2_tgmm_tm1024 | 2.847x | tiling_specialization | New overall best, tgmm TM=1024 halves VLIW complexity |
| L1 | L1_scale_bf16 | 2.819x | tiling_specialization | Second best, bf16 scales eliminate register pressure |

### Not Selected

| Variant | Status | Speedup | Reason |
|---------|--------|---------|--------|
| L1_bwd_tk1024 | SUCCESS | 2.728x | Slight regression from L1 best (2.751x), 234 spills |
| L1_fwd_tk1024 | SUCCESS | 2.498x | Major regression, fwd TK=1024 too large |
| L2_fwd_tn512 | SUCCESS | 2.410x | TN=512 regression, single N-tile inefficient |

### Lineage Updates

- **L2**: Updated best_speedup 2.769 → 2.847, best_kernel → L2_tgmm_tm1024, reset stagnant_rounds
- **L1**: Updated best_speedup 2.751 → 2.819, best_kernel → L1_scale_bf16, reset stagnant_rounds

### Lineages.json Status

- Active lineages: 2
- Pruned lineages: 0
- Current round: 6
