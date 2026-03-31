## Round 7 Selection

### Selected (Top-K = 2)

| Lineage | Variant | Speedup | Direction | Rationale |
|---------|---------|---------|-----------|-----------|
| L2 | L2_recompute_dh_v2 | 8.988x | bwd_fusion_5d_output | New best — backward fusion with 5D output layout, +14.6% over previous best |
| L2 | L2_bwd_fuse_dh_v2 | 8.968x | bwd_fusion_flat_output | Runner-up — same fusion approach, flat output layout, marginally higher MXU util |

Both top variants are from L2 lineage. L2_recompute_dh_v2 becomes the new L2 best kernel.

### Not Selected

| Variant | Status | Speedup | Reason |
|---------|--------|---------|--------|
| L2_reduce_intermediates | SUCCESS | 7.700x | Below previous best — recomputation overhead exceeds spill savings |
| L2_split_dv_dqdk | SUCCESS | 6.869x | Regression — kernel launch overhead from splitting exceeds spill savings |
| L1_bwd_grid_v_v2 | INCORRECT | -- | TPU block shape alignment constraint (last dim 64 not divisible by 128) |

### Lineage Updates

- **L2**: Updated best_speedup 7.845x → 8.988x, best_kernel → iteration_7/variants/L2_recompute_dh_v2/kernel.py, reset stagnant_rounds to 0, appended iter-7-recompute_dh_v2 to history
- **L1**: Stagnant for 2 rounds (variant failed INCORRECT). No improvement. L1 has converged to same architecture as L2 — differentiation is lost.

### Lineages.json Status

- Active lineages: 2 (L1 at risk with stagnant_rounds=2)
- Pruned lineages: 3 (from earlier rounds)
- Current round: 7

### Notes

L1 has been stagnant for 2 rounds with both attempts failing on Pallas constraints (R6: BlockSpec rank bug, R7: TPU alignment constraint). L1 converged to the same kernel architecture as L2 in R5, making it effectively a duplicate lineage exploring failed V-tiling/grid-expansion directions. Consider pruning L1 and spawning a fresh lineage from the new R7 best kernel if a novel direction presents itself.
