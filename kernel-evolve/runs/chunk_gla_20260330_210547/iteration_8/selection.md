## Round 8 Selection

### Selected (Top-K = 2)

| Lineage | Variant | Speedup | Direction | Rationale |
|---------|---------|---------|-----------|-----------|
| L2 | L2_eliminate_gcumsum | 9.005x | eliminate_gcumsum | New best — g_cumsum elimination from backward, +0.2% over R7 |
| L2 | L2_algebraic_simplify | 8.917x | algebraic_simplify | Runner-up — best spill traffic reduction (Vec Store 7.76%), but MXU util regression |

Both top variants are from L2 lineage. L2_eliminate_gcumsum becomes the new L2 best kernel.

### Not Selected

| Variant | Status | Speedup | Reason |
|---------|--------|---------|--------|
| L2_fwd_store_A | SUCCESS | 8.913x | Best spill reduction (-25.8%) but HBM traffic from A matrix offsets savings |
| L2_store_gated_residuals | SUCCESS | 8.821x | Gating residual storage adds too much HBM traffic (134MB) |
| L2_extra_scratch | SUCCESS | 8.749x | VMEM scratch staging adds more overhead than it saves — confirmed FP33 |

### Lineage Updates

- **L2**: Updated best_speedup 8.988x → 9.005x, best_kernel → iteration_8/variants/L2_eliminate_gcumsum/kernel.py, reset stagnant_rounds to 0, appended iter-8-eliminate_gcumsum to history
- **L1**: Pruned — stagnant 3 rounds, converged to same architecture as L2, no differentiation path remaining

### Lineages.json Status

- Active lineages: 1 (L2 only, after L1 pruning)
- Pruned lineages: 4 (P1, P2, P3, L1)
- Current round: 8

### Notes

Round 8 explored 5 different register pressure reduction approaches. All succeeded but produced diminishing returns (+0.2% max improvement). This signals a potential performance plateau at ~9x. Future rounds should explore fundamentally different approaches beyond register pressure reduction — the binding constraint appears to have shifted to MXU pipeline efficiency (34.6% util out of 50% theoretical max with single-MXU) and possibly algorithmic restructuring.

L1 has been pruned after 3 stagnant rounds with no viable differentiation from L2. The optimization now proceeds with a single lineage (L2) at 9.005x.
