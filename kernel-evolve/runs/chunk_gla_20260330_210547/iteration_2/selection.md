## Round 2 Selection

### Top-K Selection (K=2)

| Rank | Variant | Lineage | Speedup | Selected |
|------|---------|---------|---------|----------|
| 1 | L1_reduce_inputs | L1 | 1.097x | YES |
| 2 | L1_scratch_accum | L1 | 0.888x | -- (same lineage as #1) |
| 3 | L2_single_fused_k_tile | L2 | 0.872x | YES (best L2) |
| 4 | L2_smaller_bt | L2 | 0.811x | -- (same lineage as #3) |
| -- | L1_k_tiling | L1 | INCORRECT | -- |

### Lineage Updates

**L1** (mxu_vpu_overlap → reduce_inputs):
- Previous best: 0.876x (iteration_1/variants/mxu_vpu_overlap/kernel.py)
- New best: **1.097x** (iteration_2/variants/L1_reduce_inputs/kernel.py)
- Delta: **+25.2%** — MAJOR IMPROVEMENT, FIRST TIME BEATING REFERENCE
- Direction evolved: mxu_vpu_overlap → reduce_inputs
- stagnant_rounds: 0 (reset — improved)
- Key insight: Eliminating the separate `chunk_gla_fwd_intra_gk` pallas_call saved 27 computation events and removed one HBM round-trip for the A matrix, more than compensating for the 24% increase in register spills.

**L2** (mxu_utilization → v_tiling):
- Previous best: 0.825x (iteration_1/variants/mxu_utilization/kernel.py)
- New best: **0.872x** (iteration_2/variants/L2_single_fused_k_tile/kernel.py)
- Delta: +5.7% — moderate improvement
- Direction evolved: mxu_utilization → v_tiling
- stagnant_rounds: 0 (reset — improved)
- Note: Despite catastrophic spill regression (5.74M vs 369K), the fused V-tiling approach is faster than the original L2 split. This suggests the split-kernel approach itself was the L2 bottleneck, not register pressure.

### Pruning

No lineages pruned this round. Both L1 and L2 improved:
- Active lineages: 2 (within max_active_lineages=4)
- L1 is the clear leader at 1.097x
- L2 at 0.872x still has room for improvement

### Strategy for Round 3

**L1 (1.097x)**: Build on reduce_inputs success. Focus on:
1. Further reducing kernel inputs/outputs (can dh be recomputed?)
2. Addressing dual_ratio=0.0 (both MXUs idle despite compute-bound)
3. Reducing the 2.4M register spills introduced by A recomputation

**L2 (0.872x)**: The split approach appears to be a dead end. Consider:
1. Pivoting L2 to a fused approach similar to L1 but with a different optimization angle
2. Or keeping L2 as an exploration lineage trying fundamentally different algorithmic approaches
