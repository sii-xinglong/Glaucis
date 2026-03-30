## Round 1 Selection

### Selected (Top-2)

| Lineage | Variant | Speedup | Direction | Rationale |
|---------|---------|---------|-----------|-----------|
| L1 | mxu_vpu_overlap | 0.876x | mxu_vpu_overlap | Closest to baseline; compiled code identical — serves as baseline-equivalent starting point |
| L2 | mxu_utilization | 0.825x | mxu_utilization | 81% spill reduction (1.9M→369K), 37% VLIW reduction (8270→5226) — most architectural improvement despite latency regression |

### Not Selected

| Variant | Status | Speedup | Reason |
|---------|--------|---------|--------|
| tiling_strategy | SUCCESS | 0.823x | Below top-K; 73% WORSE spills (3.3M vs 1.9M baseline) — dA recomputation harmful |
| hbm_compute_overlap | INCORRECT | -- | emit_pipeline body function argument mismatch |
| memory_layout | INCORRECT | -- | BlockSpec dimension mismatch after layout change |

### Lineage Updates

- **L1 (NEW)**: Created from mxu_vpu_overlap (0.876x). Note: compiled code identical to baseline — future mutations should try genuinely different approaches.
- **L2 (NEW)**: Created from mxu_utilization (0.825x). Strong register pressure reduction. Future mutations should reduce launch overhead or find a way to keep the spill reduction within a fused kernel.

### Lineages.json Status

- Active lineages: 2
- Pruned: 3
- Current round: 1
