## Round 4 Selection

### Selected (Top-2)

| Lineage | Variant | Speedup | Direction | Rationale |
|---------|---------|---------|-----------|-----------|
| L1 | L1_adopt_l2_tiling | 2.651x | tiling_specialization | New overall best, cross-pollination of L1+L2 tilings |
| L2 | L2_fwd_tk512 | 2.570x | tiling_specialization | Second best, fwd TK=512 applied to L2 |

### Not Selected

| Variant | Status | Speedup | Reason |
|---------|--------|---------|--------|
| L2_fwd_tn256 | SUCCESS | 2.536x | Below top-K |
| L2_fwd_tk512_scale_bf16 | SUCCESS | 2.456x | bf16 scales interfere with fwd TK=512 |
| L2_tgmm_tn256 | SUCCESS | 2.452x | tgmm TN=256 halves MXU ops — regression |

### Lineage Updates

- **L1**: Updated best_speedup 2.335 → 2.651, best_kernel → L1_adopt_l2_tiling, reset stagnant_rounds to 0
- **L2**: Updated best_speedup 2.460 → 2.570, best_kernel → L2_fwd_tk512, reset stagnant_rounds to 0

### Lineages.json Status

- Active lineages: 2
- Pruned lineages: 0
- Current round: 4
