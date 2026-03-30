## Round 3 Batch Analysis

**Variants evaluated**: 5
**Successes**: 3 | **Failures**: 2
**Best speedup this round**: 1.3216x (tiling_128_256_128)
**Overall best speedup**: 1.3216x

### Key Findings

1. **Increasing a single tiling dimension from 128→256 produces real speedups (1.22-1.32x)**
2. **Combining multiple dimension increases causes compiler crashes** — (256,256,256) and (256,512,512) both fail
3. **N-dimension increase halves register spills** (73K vs 161K) while keeping same VLIW count (4065)
4. **M and K dimension increases add more MXU ops** (56 vs 36) but also more VLIW bundles

### Comparative Ranking

| Rank | Variant | Status | Speedup | Latency (ms) | Spills | VLIW Bundles | MXU Ops |
|------|---------|--------|---------|--------------|--------|-------------|---------|
| 1 | tiling_128_256_128 | SUCCESS | 1.3216x | 7.958 | 161,948 | 6457 | 56/56 |
| 2 | tiling_256_128_128 | SUCCESS | 1.3166x | 7.812 | 161,948 | 5248 | 56/56 |
| 3 | tiling_128_128_256 | SUCCESS | 1.2151x | 8.464 | 73,543 | 4065 | 36/36 |
| -- | tiling_256_256_256 | COMPILE_ERROR | -- | -- | -- | -- | -- |
| -- | tiling_256_512_512 | COMPILE_ERROR | -- | -- | -- | -- | -- |

### Per-Variant Analysis

#### tiling_128_256_128 (Rank 1 — 1.32x)
Doubled K-tiling. More MXU ops (56 vs 36 baseline) and higher compute efficiency (9.0% vs 6.7%).
VLIW bundles increased (6457 vs 4065) but the larger K tile processes more data per iteration.
Spills slightly up (162K vs 161K) — larger K tile doesn't help register pressure.

#### tiling_256_128_128 (Rank 2 — 1.32x)
Doubled M-tiling. Similar metrics to K-tiling variant — 56 MXU ops, 9.15% efficiency.
Fewer VLIW bundles (5248 vs 6457) suggesting M-tiling is slightly more efficient.
Actually has the lowest absolute latency (7.812ms).

#### tiling_128_128_256 (Rank 3 — 1.22x)
Doubled N-tiling. **Halved register spills** (73,543 vs 160,845 baseline) — the most impactful change for the spill bottleneck.
Same VLIW bundles and MXU ops as baseline — the N dimension doesn't affect the matmul inner loop.
Lower speedup because it doesn't increase MXU throughput, but dramatically reduces spill overhead.

### Constraint Discovery
**Only one dimension can be increased beyond 128 at a time** in the tokamax backend.
(256,256,256) and (256,512,512) both crash. Valid configurations: change exactly one of TM/TK/TN.

### Next Round Suggestion
Try per-PHASE tiling where each phase changes a DIFFERENT single dimension:
- Fwd: (128, 256, 128) — K-tiling for forward
- Bwd GMM: (256, 128, 128) — M-tiling for backward
- Bwd TGMM: (128, 128, 256) — N-tiling for backward
This gives each phase its optimal single-dimension increase.
