# Pallas Kernel Optimization Agent Knowledge

## Failure Patterns

### FP1: Changing tile_size causes massive correctness failure
- **What**: Setting tile_size=256 (from 128) in QtRule causes max_diff=88604 vs atol=1.0
- **Why**: Coarser quantization granularity produces different scaling factors. Errors accumulate across ~4M output elements. Reference is locked at tile_size=128.
- **Rule**: NEVER change tile_size from 128. Quantization granularity must match the reference exactly.

### FP2: tgmm does not support N tiling > 128 (subchannel_iters)
- **What**: Setting tgmm tiling to N=256 causes `NotImplementedError: subchannel_iters != 1 not supported yet in tgmm`
- **Why**: tokamax's tgmm backend doesn't support sub-channel iterations needed when N tile exceeds the quantization tile size. This is a backend limitation.
- **Rule**: tgmm N tiling MUST stay at 128. Only M tiling can be increased for tgmm.

## Successful Optimizations

### SO1: K-only tiling clamp enables larger M tiles (1.175x speedup)
- **What**: Replace `tiling = tuple(min(t, tile_size) for t in tiling)` with K-only clamping. Use M=256, N=128, K=128 tiling for fwd/bwd_gmm phases.
- **Why**: The blanket clamp restricts M/N tiles to tile_size=128 unnecessarily. Only K needs to match quantization tile boundaries. Larger M tiles (256) halve the M-grid and improve MXU utilization.
- **Impact**: 1.175x speedup, correctness preserved (max_diff within atol=1.0)
- **Rule**: M and N tiling dimensions are independent of quantization tile_size. Only clamp K.

### SO2: Aggressive M=512, N=256 tiling (1.310x speedup)
- **What**: Push M tiling to 512, N tiling to 256 for fwd/bwd_gmm. Tgmm uses M=512, N=128.
- **Why**: Fewer, larger tiles reduce kernel launch overhead and improve MXU utilization. Kernel is fully compute-bound (compute_ratio=1.0).
- **Impact**: 1.310x speedup (8.15ms vs 10.68ms). +12% over SO1.
- **Config**: tiling = (512, 256, 128, 512, 256, 128, 512, 128, 128)

### SO3: Deferred lhs_t quantization (1.583x speedup)
- **What**: Move `lhs_t = qpl.quantize(lhs_bf16.swapaxes(0, 1), ...)` from forward to backward. Store `lhs_bf16` in residuals instead of pre-quantized `lhs_t`.
- **Why**: Forward pass is on the critical path. Removing one quantize call from forward reduces forward latency. The backward pass already has grad quantization overhead, so the deferred quantize overlaps with existing backward work.
- **Impact**: 1.583x speedup (6.32ms vs 10.00ms). +20.8% over SO2.
- **Trade-off**: Residuals store bf16 tensor (larger than fp8 quantized), increasing memory pressure slightly. Net effect is still strongly positive.
- **Rule**: Consider deferring non-critical quantization from forward to backward when forward latency is the bottleneck.

### SO4: M=1024 tiling for fwd/bwd_gmm (1.621x speedup)
- **What**: Increase M tiling from 512 to 1024 for fwd and bwd_gmm phases. Tgmm unchanged at M=512.
- **Why**: With M=8192, M=1024 halves M-grid from 16 to 8 tiles, reducing kernel launch overhead.
- **Impact**: 1.621x speedup (6.17ms vs 10.00ms). +2.4% over SO3.
- **Config**: tiling = (1024, 256, 128, 1024, 256, 128, 512, 128, 128)
- **Rule**: Larger M tiles help up to the point where VMEM is saturated.

### FP3: M=2048 + N=512 tiling causes VMEM regression
- **What**: Setting fwd tiling to (2048, 512, 128) regresses from 1.621x to 1.512x despite fewer grid tiles.
- **Why**: Tile accumulators of 2048x512xfloat32 = 4MB per tile. Multiple simultaneous accumulators cause VMEM spilling to HBM, negating the benefit of fewer tiles.
- **Rule**: M*N tile product should not exceed ~256K elements (1024*256). For this problem shape, M=1024, N=256 is the sweet spot.
