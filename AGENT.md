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

### [F004] GMM tiling must respect per-group dimensions
- **Symptom**: Tiling (512,1024,512) and (256,512,256) both crash with `Check failed: limits[i] <= dim(i)` core dump in Mosaic compiler array.h
- **Root cause**: GMM is *grouped* — tiling (TM, TK, TN) operates within each group, not the full matrix. With G=32 groups, M=8192: per-group M=256. Tiling TM=512 > 256 causes the crash. Similarly, the Down projection has K=512, so TK=1024 > 512 also crashes. The valid tiling range per dimension is bounded by the **minimum per-group dimension across all shapes**.
- **Fix**: For gmm_fp8_blockwise with shapes [8192,2048]@[32,2048,512] and [8192,512]@[32,512,2048]: TM ≤ 256 (M/G), TK ≤ 512 (min K), TN ≤ 512 (min N). Maximum valid uniform tiling: (256, 512, 512). Must also remove the `min(t, tile_size)` clamping to allow tiling > 128.
- **First seen**: 2026-03-30, gmm_fp8_blockwise optimization (Round 2: 4/5 variants crashed)

### [F005] Quantization tile_size=256 completely breaks FP8 correctness
- **Symptom**: `ts256_only` variant with tile_size=256 (isolated change) produced max_diff=94,575 — catastrophically wrong output
- **Root cause**: qwix FP8 blockwise quantization with tile_size=256 produces wildly different numerical results than tile_size=128. The scaling factors computed over 256-element blocks are too coarse for the FP8 dynamic range, causing massive overflow/underflow.
- **Fix**: tile_size MUST stay at 128 for this kernel. Do NOT increase quantization tile_size. The optimization surface for this kernel is limited to tokamax tiling parameters and computation structure.
- **First seen**: 2026-03-30, gmm_fp8_blockwise optimization (Round 2)

### [F006] Tokamax only supports increasing ONE tiling dimension beyond 128 at a time
- **Symptom**: Tiling (256,256,256) and (256,512,512) both crash with core dumps. But (256,128,128), (128,256,128), and (128,128,256) all work fine.
- **Root cause**: tokamax backend's ragged_dot Pallas kernel has internal constraints that prevent multiple tile dimensions from exceeding 128 simultaneously. The exact constraint likely involves VMEM allocation or BlockSpec grid limits.
- **Fix**: Only increase ONE of TM/TK/TN beyond 128 per tiling triplet. Use per-phase tiling to get different single-dimension increases for fwd/bwd_gmm/bwd_tgmm.
- **First seen**: 2026-03-30, gmm_fp8_blockwise optimization (Round 3)

## Successful Optimizations

### [S001] Increasing single tiling dimension from 128→256 yields 1.22-1.32x speedup
- **Optimization**: Changed one dimension of the tokamax tiling from 128 to 256. Must remove the `tiling = tuple(min(t, tile_size) for t in tiling)` clamping in both _gmm_fwd and _gmm_bwd.
- **Impact**: 1.0x → 1.32x for TK=256 (128,256,128), 1.32x for TM=256 (256,128,128), 1.22x for TN=256 (128,128,256)
- **Why it works**: Larger tiles process more data per kernel invocation, increasing MXU throughput. TK=256 and TM=256 increase MXU ops from 36 to 56 per MXU. TN=256 halves register spills from 161K to 74K.
- **Applicable when**: Optimizing tokamax gmm/tgmm kernels where default tiling is (128,128,128). Always try single-dimension increases first.
- **First seen**: 2026-03-30, gmm_fp8_blockwise optimization (Round 3)

