## Round 9 Selection

### Selected (Top-K = 2)

| Lineage | Variant | Speedup | Direction | Rationale |
|---------|---------|---------|-----------|-----------|
| L2 | L2_bf16_h_residual | 9.054x | bf16_h_residual | New best — bf16 h residual halves HBM traffic for state storage, +0.5% over R8 |
| L2 | L2_simplify_gating | 8.975x | simplify_gating | Runner-up — VPU exp() reduction, marginal spill improvement |

Both top variants are from L2 lineage. L2_bf16_h_residual becomes the new L2 best kernel.

### Not Selected

| Variant | Status | Speedup | Reason |
|---------|--------|---------|--------|
| L2_merge_A_dA | SUCCESS | 8.973x | Best spill reduction (-17.7%) but MXU util regression (-6.0pp) cancels register savings |
| L2_precompute_gated | SUCCESS | 8.954x | VPU consolidation counterproductive — constrains compiler's VLIW scheduler |
| L2_bf16_dh_update | INCORRECT | -- | bf16 precision on accumulating dh matmul causes correctness failure (max_diff=22.28) |

### Lineage Updates

- **L2**: Updated best_speedup 9.005x -> 9.054x, best_kernel -> iteration_9/variants/L2_bf16_h_residual/kernel.py, reset stagnant_rounds to 0, appended iter-9-bf16_h_residual to history

### Lineages.json Status

- Active lineages: 1 (L2 only)
- Pruned lineages: 4 (P1, P2, P3, L1)
- Current round: 9

### Notes

Round 9 explored 5 different optimization directions focused on MXU pipeline efficiency and HBM bandwidth reduction. The key discovery is that bf16 precision for non-accumulating forward-to-backward residuals (h state) provides measurable HBM bandwidth savings (+0.5% speedup) without correctness impact. This opens a new optimization axis: precision reduction for residual storage.

The VPU optimization approaches (simplify_gating, precompute_gated) confirmed that the Mosaic compiler handles VPU scheduling effectively — manual VPU optimization is counterproductive at this performance level.

The phase reordering approach (merge_A_dA) reinforces FP34 with additional evidence: even dramatic register pressure reduction (-17.7% spills) does not improve speedup, and phase reordering actively harms MXU scheduling.

The bf16 dh matmul failure extends FP31: bf16 precision is unsafe for any matmul whose output feeds into sequential state accumulation across time steps.
