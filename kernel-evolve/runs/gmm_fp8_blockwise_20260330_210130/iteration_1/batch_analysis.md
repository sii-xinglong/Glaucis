## Round 1 Batch Analysis

**Variants evaluated**: 5
**Successes**: 3 | **Failures**: 2
**Best speedup this round**: 1.0005x (hbm_compute_overlap)
**Overall best speedup**: 1.0005x (Round 1)

### Key Finding

All three successful variants compiled to **identical LLO code** (4065 VLIW bundles, 36/36 MXU ops, ~160K spills). The JAX/XLA compiler optimizes the mathematical computation graph as a whole — Python-level restructuring (deferring quantization, reordering operations, reducing residuals) is traced and compiled into the same IR. The custom_vjp defines mathematical relationships, not execution order.

**This means**: High-level Python restructuring of qwix/tokamax API calls has no effect on the compiled kernel. To actually change the TPU execution, we need to change the **mathematical parameters** (tile sizes, quantization configuration, tiling dimensions) or use different underlying API calls.

### Comparative Ranking

| Rank | Variant | Status | Speedup | Latency (ms) | Spills | VLIW Bundles | Direction |
|------|---------|--------|---------|--------------|--------|-------------|-----------|
| 1 | hbm_compute_overlap | SUCCESS | 1.0005x | 10.277 | 160,586 | 4065 | hbm_compute_overlap |
| 2 | mxu_vpu_overlap | SUCCESS | 1.0001x | 10.280 | 160,845 | 4065 | mxu_vpu_overlap |
| 3 | memory_layout | SUCCESS | 0.9614x | 10.696 | 160,845 | 4065 | memory_layout |
| -- | quantization_strategy | INCORRECT | -- | -- | -- | -- | quantization_strategy |
| -- | tiling_strategy | INCORRECT | -- | -- | -- | -- | tiling_strategy |

### Per-Variant Details

#### hbm_compute_overlap (Rank 1)
**Status**: SUCCESS | **Speedup**: 1.0005x | **Latency**: 10.277ms

| Metric | Value | vs Baseline | Assessment |
|--------|-------|-------------|------------|
| Register fills/spills | 160,586/160,586 | -259 | negligible change |
| VLIW bundles | 4065 | same | identical compiled code |
| MXU dual_ratio | 1.0 | same | unchanged |
| Compute efficiency | 6.96% | +0.28% | within noise |
| DMA transfers | 35 | same | unchanged |

**Bottleneck**: Unchanged — register pressure + scalar-heavy. The deferred lhs_t quantization was optimized away by XLA.

#### mxu_vpu_overlap (Rank 2)
**Status**: SUCCESS | **Speedup**: 1.0001x | **Latency**: 10.280ms

Metrics identical to baseline (4065 bundles, 160,845 spills, 1.0 dual_ratio). The reordering of quantization ops had zero effect on compiled output.

#### memory_layout (Rank 3)
**Status**: SUCCESS | **Speedup**: 0.9614x | **Latency**: 10.696ms

Metrics identical to baseline. The slight regression is within measurement noise.

#### quantization_strategy (INCORRECT)
**Error**: Correctness check failed with traceback
**Cause**: The aggressive changes (tile_size 128→256, channelwise-only lhs_t, shared grad quantization) produced incorrect numerical results
**Fix**: Less aggressive changes — try tile_size=256 alone without the channelwise or shared grad changes

#### tiling_strategy (INCORRECT)
**Error**: Runtime error during `kernel_fn(**shape)` call
**Cause**: Tiling (64,128,64) likely violated tokamax backend constraints — tiles may be too small for the FP8 matmul implementation
**Fix**: Try larger tiles within tokamax constraints, or only change tiling for specific phases

### Strategic Assessment

The core problem is that this kernel wraps **library APIs** (qwix quantization + tokamax GMM backend), not raw Pallas operations. The evolve block controls:
1. **Quantization parameters**: tile_size, calibration method, axis configuration
2. **Tiling passed to tokamax**: the tiling tuple for gmm/tgmm
3. **Control flow**: order of quantization and computation

Of these, only (1) and (2) change the compiled output. (3) is irrelevant since XLA traces the full graph.

**Next round should focus on**:
- Different tiling values that tokamax actually accepts (try (128,128,128), (128,256,128), etc.)
- Quantization tile_size changes (256) isolated from other changes
- Reducing the number of quantization passes (e.g., skip backward quantization)
- Using different precision settings in the gmm call
