## Round 3 Selection

### Selected (Top-2)

| Lineage | Variant | Speedup | Direction | Rationale |
|---------|---------|---------|-----------|-----------|
| L1 | tiling_128_256_128 | 1.3216x | K-tiling | Best speedup, doubled K dimension increases MXU throughput |
| L2 | tiling_256_128_128 | 1.3166x | M-tiling | Near-best speedup with lowest latency (7.812ms) |

Note: tiling_128_128_256 (1.22x) is notable for halving register spills. Its approach may combine with the selected lineages in future rounds.

### Not Selected

| Variant | Status | Speedup | Reason |
|---------|--------|---------|--------|
| tiling_128_128_256 | SUCCESS | 1.2151x | Below top-2 threshold (but valuable for spill reduction insight) |
| tiling_256_256_256 | COMPILE_ERROR | -- | Multi-dimension increase crashes compiler |
| tiling_256_512_512 | COMPILE_ERROR | -- | Multi-dimension increase crashes compiler |

### Lineage Updates

- **L1**: Replaced — new best 1.3216x (was 1.0005x). Reset stagnant_rounds to 0.
- **L2**: Replaced — new best 1.3166x (was 1.0001x). Reset stagnant_rounds to 0.

### Lineages.json Status

- Active lineages: 2
- Pruned lineages: 3
- Current round: 3
