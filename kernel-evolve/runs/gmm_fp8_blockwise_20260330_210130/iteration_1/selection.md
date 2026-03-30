## Round 1 Selection

### Selected (Top-2)

| Lineage | Variant | Speedup | Direction | Rationale |
|---------|---------|---------|-----------|-----------|
| L1 | hbm_compute_overlap | 1.0005x | hbm_compute_overlap | Best speedup (marginal). Compiled to identical code as baseline but concept is sound. |
| L2 | mxu_vpu_overlap | 1.0001x | mxu_vpu_overlap | Second best. Also compiled identically. Will serve as base for parameter-level changes. |

### Not Selected

| Variant | Status | Speedup | Reason |
|---------|--------|---------|--------|
| memory_layout | SUCCESS | 0.9614x | Worse than baseline |
| quantization_strategy | INCORRECT | -- | Correctness failure — too many aggressive changes at once |
| tiling_strategy | INCORRECT | -- | Runtime error — tile size 64 likely violated tokamax constraints |

### Lineage Updates

- **L1**: Created from hbm_compute_overlap (speedup 1.0005x)
- **L2**: Created from mxu_vpu_overlap (speedup 1.0001x)

### Lineages.json Status

- Active lineages: 2
- Pruned lineages: 0
- Current round: 1

### Notes for Round 2

All Python-level restructuring (deferred quantization, reordered ops, reduced residuals) had zero effect on compiled output. The XLA compiler normalizes the computation graph. Round 2 must change **mathematical parameters**:
- Quantization tile_size (256, applied individually)
- Tokamax tiling parameters (valid configurations only)
- Different quantization axis configurations
- Precision settings in gmm call
