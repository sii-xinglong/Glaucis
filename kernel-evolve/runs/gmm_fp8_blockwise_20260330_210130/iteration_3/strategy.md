## Round 3 Strategy

Valid tiling configurations within per-group constraints (TM≤256, TK≤512, TN≤512).
All variants remove the min(t, tile_size) clamping. tile_size stays at 128.

### Variant: tiling_256_128_128
**Approach**: Only increase TM to 256 (max per-group). Isolates effect of M-tiling.

### Variant: tiling_128_256_128
**Approach**: Only increase TK to 256. Tests K-dimension tiling effect on reduction loop.

### Variant: tiling_128_128_256
**Approach**: Only increase TN to 256. Tests N-dimension tiling effect on output width.

### Variant: tiling_256_256_256
**Approach**: All dimensions doubled. Moderate increase across the board.

### Variant: tiling_256_512_512
**Approach**: Maximum valid tiling. TM=256 (max), TK=512 (max), TN=512 (max).
