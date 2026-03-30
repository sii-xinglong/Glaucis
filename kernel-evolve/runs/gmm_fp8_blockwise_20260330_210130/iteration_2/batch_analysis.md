## Round 2 Batch Analysis

**Variants evaluated**: 5
**Successes**: 0 | **Failures**: 5
**Best speedup this round**: N/A (all failed)

### Key Finding

GMM tiling is per-GROUP, not per-matrix. With G=32 groups and M=8192, per-group M=256. The requested tiling (512,1024,512) exceeds per-group limits. Additionally, quantization tile_size=256 is catastrophically wrong (max_diff=94,575).

**Valid tiling bounds** for both shapes:
- TM ≤ 256 (M/G = 8192/32)
- TK ≤ 512 (min K across shapes)
- TN ≤ 512 (min N across shapes)

### Comparative Ranking

| Rank | Variant | Status | Error |
|------|---------|--------|-------|
| -- | tiling_512_1024_512 | COMPILE_ERROR | limits[i]<=dim(i) (8 vs 1) — TM=512 > per-group 256 |
| -- | tiling_256_512_256 | COMPILE_ERROR | limits[i]<=dim(i) (4 vs 1) — TM=256 ok but TK=512 may exceed for some internal dim |
| -- | tiling_512_1024_512_ts256 | COMPILE_ERROR | Same tiling crash + ts256 |
| -- | tiling_phased | COMPILE_ERROR | Same tiling crash |
| -- | ts256_only | INCORRECT | max_diff=94,575 — tile_size=256 breaks quantization |
