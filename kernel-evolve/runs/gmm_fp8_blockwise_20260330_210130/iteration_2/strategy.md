## Round 2 Strategy

Focus on parameter-level changes based on Round 1 learning that Python restructuring has no effect.
User directed tiling to (512, 1024, 512). All variants remove the tile_size clamping that previously limited tiling to 128.

### Variant: tiling_512_1024_512
**Technical direction**: tiling_strategy (user-directed)
**Approach**: Tiling (512, 1024, 512) * 3 with clamping removed. tile_size=128 unchanged.
**Key changes**: Default tiling changed, removed min(t, tile_size) clamping in fwd and bwd

### Variant: tiling_512_1024_512_ts256
**Technical direction**: tiling_strategy + quantization_strategy
**Approach**: Tiling (512, 1024, 512) * 3 + tile_size=256 for quantization. Clamping removed.
**Key changes**: Larger tiles AND larger quantization blocks — 4x fewer scale computations

### Variant: tiling_256_512_256
**Technical direction**: tiling_strategy (moderate)
**Approach**: Tiling (256, 512, 256) * 3 — moderate increase. Clamping removed. tile_size=128.
**Key changes**: More conservative tile increase as control variant

### Variant: ts256_only
**Technical direction**: quantization_strategy (isolated)
**Approach**: tile_size=256 ONLY, tiling stays (128,128,128)*3. Tests quantization overhead reduction in isolation.
**Key changes**: All qpl.quantize tiled_axes use 256 instead of 128

### Variant: tiling_phased
**Technical direction**: tiling_strategy (phase-specific)
**Approach**: Different tiling per phase: fwd=(512,1024,512), bwd_gmm=(512,512,1024), bwd_tgmm=(512,512,512). Clamping removed.
**Key changes**: Phase-specific tiling tuple exploiting different matrix shapes per phase
