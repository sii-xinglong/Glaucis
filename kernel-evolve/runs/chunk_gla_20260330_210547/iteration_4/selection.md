## Round 4 Selection

### Top-K Selection (K=2)

| Rank | Variant | Speedup | Lineage | Selected |
|------|---------|---------|---------|----------|
| 1 | L2_fold_dh_pallas | 6.831x | L2 | YES |
| 2 | L1_reduce_exp | 1.498x | L1 | YES (stagnant) |

### Lineage Updates

**L2** (MASSIVE IMPROVEMENT):
- Previous best: 1.056x (reduce_inputs)
- New best: 6.831x (fold_dh_pallas) — **+546.9%**
- Best kernel: iteration_4/variants/L2_fold_dh_pallas/kernel.py
- stagnant_rounds: 0 (reset)
- PIVOTED from V-tiling lineage to L1's clean kernel + pallas_call dh replacement

**L1** (STAGNANT — regression):
- Previous best: 1.577x (combined_fuse_skip)
- This round: 1.498x (reduce_exp) — **-5.0% regression**
- Best kernel unchanged: iteration_3/variants/L1_combined_fuse_skip/kernel.py
- stagnant_rounds: 0 → 1
- reduce_exp direction proved that reciprocal ops are slower than exp() on TPU

### Pruned This Round

None pruned — both lineages survive. L1 has 1 stagnant round (will be pruned if stagnant for 2 more rounds without improvement).

### Failed Variants (not eligible for selection)

| Variant | Error Type | Root Cause |
|---------|-----------|------------|
| L1_bf16_intermediates | Mosaic MLIR | Mixed bf16/f32 matmul inputs not supported |
| L1_fwd_single_kernel | Shape mismatch | Wrong contraction dimension in h-update dot |
| L2_fuse_fwd_output_h | API misuse | index_map arg count doesn't match grid dims |

### Strategic Assessment

The L2 pivot was the single most impactful decision of the entire optimization run. L2 abandoned its failing V-tiling direction, adopted L1's proven clean kernel, and added the pallas_call dh replacement — achieving a 6.83x speedup.

**Round 5 priorities**:
1. Apply similar lax.scan → pallas_call to any remaining scans (forward h propagation already uses pallas_call)
2. Forward kernel fusion — L1_fwd_single_kernel and L2_fuse_fwd_output_h both had fixable implementation bugs
3. Register pressure reduction is now proportionally more impactful at 1122ms base latency
4. Dual-MXU scheduling remains an unexplored opportunity (dual_ratio=0.0 across all rounds)
