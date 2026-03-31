## Round 5 Selection

### Top-K Selection (K=2)

| Rank | Variant | Speedup | Lineage | Selected |
|------|---------|---------|---------|----------|
| 1 | L2_fuse_fwd_combined | 7.845x | L2 | YES |
| 2 | L1_fold_dh_fuse_fwd | 7.639x | L1 | YES |
| 3 | L2_reduce_live_bwd | 6.957x | L2 | no (below top-2) |
| 4 | L2_bt128 | 5.395x | L2 | no (regression) |
| -- | L2_bf16_uniform | INCORRECT | L2 | no (failed) |

### Lineage Updates

**L2** (IMPROVED):
- Previous best: 6.831x (iteration_4/variants/L2_fold_dh_pallas/kernel.py)
- New best: 7.845x (iteration_5/variants/L2_fuse_fwd_combined/kernel.py)
- Delta: +14.8%
- Direction: fuse_fwd_combined
- stagnant_rounds: 0 (reset — improved)

**L1** (IMPROVED — PIVOT):
- Previous best: 1.577x (iteration_3/variants/L1_combined_fuse_skip/kernel.py)
- New best: 7.639x (iteration_5/variants/L1_fold_dh_fuse_fwd/kernel.py)
- Delta: +384.3%
- Direction: fold_dh_fuse_fwd (pivoted to L2's architecture + forward fusion)
- stagnant_rounds: 0 (reset — massive improvement via pivot)

### Notes

Both lineages have converged to functionally identical kernels:
- Same VLIW bundle count (7930)
- Same MXU ops (4656 mxu0, 0 mxu1)
- Same computation events (177)
- Same register fills/spills (3,038,841 / 2,497,509)
- Speedup difference (7.845x vs 7.639x) is within measurement noise (~2.7%)

Future rounds need genuinely novel optimization directions since the lineages have merged architecturally. The remaining bottlenecks are:
1. Register pressure (2.5M spills) — dominant remaining bottleneck
2. Single-MXU execution (dual_ratio=0.0)
3. Backward kernel still uses 2 pallas_calls (can these be merged?)
