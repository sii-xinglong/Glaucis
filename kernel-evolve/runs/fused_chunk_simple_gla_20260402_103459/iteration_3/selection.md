## Round 3 Selection

### Selected (Top-K = 2)

| Lineage | Variant | Speedup | Direction | Rationale |
|---------|---------|---------|-----------|-----------|
| L1 | L1_no_scratch_A | 1.3881x | cleanup | Cleanest L1 variant — removed counterproductive scratch_A. Same compiled output as all other 4-step variants. |
| L2 | L2_bwd_2step | 1.3889x | backward_grid_reduction | Only L2 variant — compiled identically to L1 variants. Marginally higher speedup from reference benchmark variance. |

### Not Selected

| Variant | Status | Speedup | Reason |
|---------|--------|---------|--------|
| L1_k_split_h | SUCCESS | 1.3881x | bf16 dh_states had zero effect — compiled identically to L1_no_scratch_A |
| L1_2step | SUCCESS | 1.2221x | Significant regression — 2-step grid overhead outweighs spill reduction |
| L1_bt32 | INCORRECT | -- | BT=32 fails correctness (max_diff=24.22, atol=10.0) |

### Lineage Updates

- **L1**: Speedup regressed 1.4023x → 1.3881x. Best kernel remains iteration_2/variants/L1_scratch_A/kernel.py (Round 2 measurement). However, since L1_no_scratch_A compiles to a cleaner kernel (5,275 vs 5,499 VLIW), updating best_kernel to L1_no_scratch_A for a cleaner base. stagnant_rounds incremented to 1.
- **L2**: Speedup regressed 1.4023x → 1.3889x. Best kernel remains iteration_2/variants/L2_bwd_fused_arbitrary/kernel.py. stagnant_rounds incremented to 1.

**NOTE**: The 1.4023x measurements from Round 2 for both lineages are now suspected to be reference benchmark variance rather than genuine kernel improvements. All Round 3 variants (with different source structures) compiled to the same binary and measured at 1.388x consistently. The true performance of the 4-step grid unrolling kernel is **~1.388x**, not 1.402x.

### Lineages.json Status

- Active lineages: 2 (L1, L2)
- Pruned lineages: 3 (P1, P2, P3 — from Round 1)
- Current round: 3
- Both lineages stagnant for 1 round
