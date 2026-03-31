## Round 1 Selection

### Selected (Top-2)

| Lineage | Variant | Speedup | Delta vs Baseline | Direction | Rationale |
|---------|---------|---------|-------------------|-----------|-----------|
| L1 | bf16_residuals | 9.058x | -0.09% | bf16 residual precision | Best speedup, ~96MB HBM savings, essentially zero speed cost |
| L2 | eliminate_flip | 9.014x | -0.57% | flip elimination | Second best, ~288MB HBM savings, acceptable 0.57% speed cost |

### Not Selected

| Variant | Status | Speedup | Delta vs Baseline | Reason |
|---------|--------|---------|-------------------|--------|
| reverse_indexing | SUCCESS | 8.969x | -1.07% | Combined bf16+flip; slightly worse than sum of parts due to interaction penalty. Techniques already covered by L1+L2. |
| h_recompute | SUCCESS | 6.567x | -27.6% | Unacceptable speed regression. Recomputation not fused with backward kernel. |
| activation_checkpoint | SUCCESS | 6.389x | -29.5% | Most aggressive but worst speed. Dominated by h_recompute penalty. |

### Lineage Updates

- **L1 (new)**: Created from bf16_residuals. Best speedup 9.058x. Direction: bf16 residual precision reduction.
- **L2 (new)**: Created from eliminate_flip. Best speedup 9.014x. Direction: flip elimination via reversed index maps.

### Selection Rationale

The top-2 selection separates the two independent HBM reduction techniques into distinct lineages:

1. **L1 (bf16_residuals)** preserves the highest speed and provides a clean base for exploring additional memory reductions on top of the bf16 residual change.

2. **L2 (eliminate_flip)** demonstrates the flip elimination approach independently. In Round 2, this lineage can explore combining with bf16 residuals more carefully or applying further HBM optimizations.

The `reverse_indexing` variant (combined approach) was not selected because the individual techniques are already represented in L1 and L2. The combined approach showed an interaction penalty, and future rounds can attempt the combination with more careful implementation.

### Lineages.json Status

- Active lineages: 2
- Pruned lineages: 0
- Current round: 1
