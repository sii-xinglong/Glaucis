# Pallas Kernel Optimization Agent Knowledge

## Failure Patterns

### [F001] Python-level restructuring has no effect on qwix/tokamax compiled output
- **Symptom**: Variants with different Python code (deferred quantization, reordered ops, reduced residuals) compile to identical LLO — same VLIW bundle count (4065), same MXU ops (36/36), same register spills (~160K).
- **Root cause**: The GMM FP8 kernel wraps library APIs (qwix quantization + tokamax GMM backend). JAX traces through custom_vjp into a computation graph that XLA optimizes holistically. Python-level control flow (order of qpl.quantize calls, which tensors are stored in residuals vs recomputed) is normalized during XLA compilation. The compiler generates the same IR regardless of how the Python code structures the operations.
- **Fix**: For qwix/tokamax-based kernels, only **mathematical parameter changes** affect compiled output: tile_size, tiling dimensions, quantization axis configs, precision settings, calibration methods. Do NOT waste rounds on code restructuring (reordering, deferring, residual minimization) — these compile away.
- **First seen**: 2026-03-30, gmm_fp8_blockwise optimization (Round 1: 3/5 variants compiled to identical code)

### [F002] Tiling (64,x,64) causes runtime errors in tokamax GMM backend
- **Symptom**: `tiling_strategy` variant with (64,128,64) tiling failed with INCORRECT status and runtime error during `kernel_fn(**shape)` call
- **Root cause**: tokamax backend likely has minimum tile size constraints. The tile dimensions must be compatible with the underlying Pallas kernel's BlockSpec and FP8 matmul implementation requirements.
- **Fix**: When changing tiling for tokamax gmm/tgmm, stay at 128 minimum per dimension. Try (128,128,128), (128,256,128), (256,128,128), etc. Do not go below 128.
- **First seen**: 2026-03-30, gmm_fp8_blockwise optimization

### [F003] Simultaneous quantization parameter changes cause correctness failures
- **Symptom**: `quantization_strategy` variant applying 3 changes at once (tile_size 128→256, channelwise-only lhs_t, shared grad quantization) produced INCORRECT output
- **Root cause**: Multiple aggressive quantization changes compound accuracy loss. Each change individually might be within tolerance (rtol=1e-2, atol=1.0), but combined they exceed thresholds.
- **Fix**: Change one quantization parameter at a time. Test tile_size=256 alone first. Test channelwise-only alone. Never combine multiple quantization simplifications in a single variant.
- **First seen**: 2026-03-30, gmm_fp8_blockwise optimization

## Successful Optimizations

