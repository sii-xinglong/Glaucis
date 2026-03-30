## Round 2 Selection

### Selected (Top-2)

No successful variants this round. Lineages L1 and L2 carry forward from Round 1 unchanged.

### Not Selected

| Variant | Status | Error | Reason |
|---------|--------|-------|--------|
| tiling_512_1024_512 | COMPILE_ERROR | Core dump: tiling exceeds per-group dim | TM=512 > M/G=256 |
| tiling_256_512_256 | COMPILE_ERROR | Core dump: tiling exceeds internal dim | Internal tokamax constraint |
| tiling_512_1024_512_ts256 | COMPILE_ERROR | Same tiling crash | TM=512 > M/G=256 |
| tiling_phased | COMPILE_ERROR | Same tiling crash | TM=512 > M/G=256 |
| ts256_only | INCORRECT | max_diff=94,575 | tile_size=256 breaks FP8 accuracy |

### Lineage Updates

- **L1**: stagnant_rounds incremented to 1 (no improvement this round — all variants failed)
- **L2**: stagnant_rounds incremented to 1

### Lineages.json Status

- Active lineages: 2 (unchanged)
- Pruned lineages: 3 (from Round 1)
- Current round: 2
