## Round 5 Selection

### Selected (Top-2)

| Lineage | Variant | Speedup | Direction | Rationale |
|---------|---------|---------|-----------|-----------|
| L2 | L2_fwd_tn256_tk512 | 2.769x | tiling_specialization | New overall best, fwd TN=256 major win |
| L1 | L1_fwd_tn256 | 2.751x | tiling_specialization | Second best, near-identical to L2 |

### Not Selected

| Variant | Status | Speedup | Reason |
|---------|--------|---------|--------|
| L1_all_tn256 | SUCCESS | 2.503x | bwd TN=256 causes 9,450 spills |
| L1_tgmm_tk1024 | SUCCESS | 2.318x | Complexity bloat (33K VLIW) |
| L1_bwd_tn256 | SUCCESS | 2.295x | Catastrophic spills (9,447) — regression |

### Lineage Updates

- **L1**: Updated best_speedup 2.651 → 2.751, best_kernel → L1_fwd_tn256, reset stagnant_rounds
- **L2**: Updated best_speedup 2.570 → 2.769, best_kernel → L2_fwd_tn256_tk512, reset stagnant_rounds

### Lineages.json Status

- Active lineages: 2
- Pruned lineages: 0
- Current round: 5
