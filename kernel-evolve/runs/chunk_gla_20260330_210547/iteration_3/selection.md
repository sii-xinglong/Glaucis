## Round 3 Selection

### Top-K Selection (K=2)

| Rank | Variant | Lineage | Speedup | Selected |
|------|---------|---------|---------|----------|
| 1 | L1_combined_fuse_skip | L1 | 1.577x | YES (new L1 best) |
| 2 | L1_fuse_fwd_A | L1 | 1.461x | -- (same lineage as #1) |
| 3 | L1_skip_dg | L1 | 1.102x | -- (same lineage as #1) |
| 4 | L2_reduce_inputs | L2 | 1.056x | YES (new L2 best) |
| 5 | L2_skip_dg | L2 | 0.877x | -- (same lineage as #4) |

### Lineage Updates

**L1** (reduce_inputs → combined_fuse_skip):
- Previous best: 1.097x (iteration_2/variants/L1_reduce_inputs/kernel.py)
- New best: **1.577x** (iteration_3/variants/L1_combined_fuse_skip/kernel.py)
- Delta: **+43.7%** — MAJOR BREAKTHROUGH
- Direction evolved: reduce_inputs → combined_fuse_skip (fwd A fusion + backward dg elimination)
- stagnant_rounds: 0 (reset — massive improvement)
- Key win: Eliminating forward chunk_gla_fwd_intra_gk pallas_call + removing dead dg computation

**L2** (v_tiling → reduce_inputs):
- Previous best: 0.872x (iteration_2/variants/L2_single_fused_k_tile/kernel.py)
- New best: **1.056x** (iteration_3/variants/L2_reduce_inputs/kernel.py)
- Delta: +21.1% — moderate improvement (crossed 1.0x threshold)
- Direction evolved: v_tiling → reduce_inputs
- stagnant_rounds: 0 (reset — improved)
- Note: Improvement came from reduce_inputs (L1-style), not V-tiling innovation

### Pruning

No lineages pruned. Both improved. However, L2 is structurally limited by V-tiling complexity. Consider pivoting L2 direction in Round 4.

### Strategy for Round 4

**L1 (1.577x)**: Continue kernel launch elimination:
1. Can chunk_fwd_h be fused with the forward output kernel? (scan dependency makes this hard)
2. Can the lax.scan for dh be combined with the backward pallas_call?
3. Further simplify the backward kernel to reduce the 2.5M spills

**L2 (1.056x)**: Pivot away from V-tiling:
1. Adopt L1's combined approach (fwd A fusion + skip dg) but on a clean kernel (not V-tiled)
2. Try completely different algorithmic approach
