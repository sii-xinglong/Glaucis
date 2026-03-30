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
- **Why**: Forward pass is on the critical path. Removing one quantize call from forward reduces forward latency. Backward latency does not increase because `lhs_t` quantization (absmax + scale + cast, pure VPU/SFU vector ops) has no data dependency on `dlhs = gmm(dlhs_dout, rhs)` (MXU matmul), so XLA schedules them in parallel on different hardware units. The quantize is fully hidden behind the gmm compute.
- **Impact**: 1.583x speedup (6.32ms vs 10.00ms). +20.8% over SO2.
- **Trade-off**: Residuals store bf16 tensor (larger than fp8 quantized), increasing memory pressure slightly. Net effect is still strongly positive.
- **Rule**: Consider deferring non-critical quantization from forward to backward when forward latency is the bottleneck.

### SO4: M=1024 tiling for fwd/bwd_gmm (1.621x speedup)
- **What**: Increase M tiling from 512 to 1024 for fwd and bwd_gmm phases. Tgmm unchanged at M=512.
- **Why**: With M=8192, M=1024 halves M-grid from 16 to 8 tiles, reducing kernel launch overhead.
- **Impact**: 1.621x speedup (6.17ms vs 10.00ms). +2.4% over SO3.
- **Config**: tiling = (1024, 256, 128, 1024, 256, 128, 512, 128, 128)
- **Rule**: Larger M tiles help up to the point where VMEM is saturated.

### SO5: BF16 tgmm — eliminate backward weight gradient quantization (1.733x speedup)
- **What**: Skip quantizing drhs_dout and lhs_t for tgmm in backward. Pass bf16 grad and bf16 lhs_t directly to `tokamax_backend.tgmm()` instead of fp8-quantized tensors. Eliminates 2 `qpl.quantize()` calls from backward.
- **Why**: On TPU v7x, bf16 matmul throughput for tgmm is comparable to fp8 throughput. The 2 quantization calls (absmax + scale + cast on VPU/SFU) have non-trivial cost that exceeds any throughput benefit of fp8 tgmm. Weight gradients (drhs) are less precision-sensitive, so bf16 is sufficient.
- **Impact**: 1.733x speedup (5.93ms vs 10.28ms). +6.9% over SO4.
- **Config**: tiling = (1024, 256, 128, 1024, 256, 128, 1024, 128, 128) with bf16 tgmm
- **Profile**: VLIW bundles unchanged (18,172), MXU dual_ratio=1.0, compute_efficiency=12.05%
- **Trade-off**: Weight gradients computed at bf16 precision instead of fp8. For training, bf16 is MORE precise than fp8, so this is actually beneficial for convergence.
- **Rule**: When VPU quantization cost exceeds MXU throughput benefit, skip quantization and use bf16 directly. Particularly applicable to weight gradient (tgmm) computations.
- **First seen**: 2026-03-30, gmm_fp8_blockwise batch round 1

### SO6: Unconstrained bf16 tgmm tiling (1.861x speedup)
- **What**: With bf16 tgmm (SO5), use tiling (2048, 256, 128) for tgmm phase — 2x larger M and 2x larger N than the previous fp8-constrained maximum.
- **Why**: FP2 (N≤128) and FP7 (M≤1024) constraints were caused by fp8 scale tensor shape mismatches in tokamax. With bf16 inputs, there are no scale tensors, so these constraints don't apply. Larger tiles reduce the tgmm grid from 32 tiles to 8 tiles.
- **Impact**: 1.861x speedup (5.55ms vs 10.33ms). +7.4% over SO5.
- **Config**: tiling = (1024, 256, 128, 1024, 256, 128, 2048, 256, 128) with bf16 tgmm
- **Profile**: VLIW bundles unchanged (18,172), MXU dual_ratio=1.0, compute_efficiency=12.87%
- **Rule**: When using bf16 inputs for tokamax gmm/tgmm, tiling constraints from fp8 scale handling (FP2, FP7) do not apply. Explore the full tiling space.
- **First seen**: 2026-03-30, gmm_fp8_blockwise batch round 2

### SO7: Zero backward quantization — mixed bf16/fp8 bwd_gmm (1.833x speedup)
- **What**: Skip dlhs_dout quantization in backward. Pass bf16 grad directly to `tokamax_backend.gmm` with fp8 rhs (mixed precision inputs). Combined with bf16 tgmm (SO5), this eliminates ALL quantization from backward.
- **Why**: tokamax gmm accepts mixed bf16/fp8 inputs. Removing the dlhs_dout quantize call eliminates ~3,200 VLIW bundles of VPU code (18,172 → 14,937, -18%). MXU ops increase from 672 to 1,056 (+57%), indicating the kernel is more matmul-dominated.
- **Impact**: 1.833x speedup (5.79ms vs 10.61ms). +5.8% over SO5.
- **Profile**: VLIW 14,937, MXU 1,056, DMA 28. Fundamentally different compilation profile.
- **Rule**: tokamax accepts mixed precision (bf16 lhs + fp8 rhs) for gmm. Removing quantization simplifies the VLIW schedule and increases MXU utilization.
- **First seen**: 2026-03-30, gmm_fp8_blockwise batch round 2

### FP4: stage_profile xprof trace returns zero-duration window
- **What**: `stage_profile` reports `Invalid trace timing (total_time <= 0)` — xprof trace captures TPU events but the time window between the last two computation events has zero or negative duration.
- **Why**: The profiled runs (3 iterations) may not produce distinct computation events with measurable gaps on v7x. Events may be merged/grouped by the trace processor, or `tpu_trace_mode=TRACE_COMPUTE_AND_SYNC` may not generate the expected event structure on Ironwood hardware.
- **Rule**: Profile parsing logic needs to handle edge cases where computation events have overlapping or identical timestamps. Consider using the full trace window rather than just the last two events.
- **First seen**: 2026-03-30, gmm_fp8_blockwise iteration 1

### FP5: stage_profile_deep produces all-null results when dump flags set mid-process
- **What**: `stage_profile_deep` returns `ok: true` but all fields (`vliw_bundle_count`, `mxu_utilization`, `hbm_bandwidth_bytes`, `flops`, `arithmetic_intensity`) are null. No `.llo`, `.hlo`, or `.txt` dump files found.
- **Why**: `XLA_FLAGS` and `LIBTPU_INIT_ARGS` are set after JAX has already compiled the kernel in stages 1-3. `jax.clear_caches()` may not force recompilation if libtpu's internal compilation cache persists. Some LIBTPU_INIT_ARGS flags are only read at process initialization.
- **Rule**: IR dump flags must be set BEFORE any JAX compilation occurs, or deep profiling must run as a separate process/Job with flags set from startup.
- **First seen**: 2026-03-30, gmm_fp8_blockwise iteration 1

### FP3: M=2048 + N=512 tiling causes VMEM regression
- **What**: Setting fwd tiling to (2048, 512, 128) regresses from 1.621x to 1.512x despite fewer grid tiles.
- **Why**: Tile accumulators of 2048x512xfloat32 = 4MB per tile. Multiple simultaneous accumulators cause VMEM spilling to HBM, negating the benefit of fewer tiles.
- **Rule**: M*N tile product should not exceed ~256K elements (1024*256). For this problem shape, M=1024, N=256 is the sweet spot.

### [FP7] tgmm M tiling > 1024 causes FP8 scale shape mismatch
- **Symptom**: `ValueError: scales=JitTracer(float32[128,256]) cannot be broadcast to out=JitTracer(float32[128,128])` in tokamax `_scale_out_by_scale()`
- **Root cause**: tgmm with M=2048 changes internal reduction tiling in tokamax, producing FP8 scale tensors of shape [128, 256] that don't match the [128, 128] output tile. Combined with FP2 (N≤128), tgmm tiling is fully constrained to max (1024, 128, 128).
- **Fix**: NEVER set tgmm M > 1024 with FP8 blockwise quantization. tgmm is maximally tiled at (1024, 128, 128).
- **First seen**: 2026-03-30, gmm_fp8_blockwise iteration 4

### [FP6] Tile aspect ratio matters: (2048, 128) much worse than (1024, 256) at same M*N
- **Symptom**: Tiling (2048, 128, 128) regresses from 1.662x to 1.288x (-22%) despite same M*N=256K accumulator
- **Root cause**: Tall-narrow tiles (M=2048, N=128) generate more complex VLIW code (+18.5% bundles: 18,172 → 21,525) and reduce compute efficiency (48.5% → 37.1%). M=2048 creates 16 sub-tile rows × 1 column, serializing the computation within each tile. M=1024, N=256 gives 8×2 sub-tiles enabling more instruction-level parallelism.
- **Fix**: For fwd/bwd_gmm, prefer square-ish tile aspect ratios. (1024, 256) is optimal for the current shapes (M=8192, K=2048, N=512).
- **First seen**: 2026-03-30, gmm_fp8_blockwise iteration 3

### [FP8] Backward operation reordering does not improve performance
- **Symptom**: Reordering backward to execute tgmm before bwd_gmm regresses from 1.621x to 1.529x (-5.7%). Placing lhs_t quantization before dlhs_dout quantization regresses further to 1.501x (-7.4%).
- **Root cause**: XLA schedules operations based on data dependencies, not Python source order. Changing the order of independent operations in Python doesn't meaningfully affect the compiled VLIW schedule. The regression stems from different register allocation choices caused by the altered operation order. Placing lhs_t quantization early keeps the large transposed tensor alive longer, increasing register pressure during bwd_gmm.
- **Fix**: Don't attempt to influence XLA scheduling by reordering independent operations in Python. Focus on reducing total work (fewer ops) rather than operation ordering.
- **Evidence**: All orderings produce identical VLIW bundle count (18,172) and MXU dual_ratio (1.0), confirming XLA normalizes the schedule. Latency differences (0.8-1.3ms) come from subtle register allocation effects.
- **First seen**: 2026-03-30, gmm_fp8_blockwise batch round 1
- **Additional evidence**: 2026-03-30, batch round 3 (overlap_bwd_interleave: 1.501x regression)

### [FP9] bwd_gmm M=2048 causes VLIW bloat regardless of input precision
- **Symptom**: bwd_gmm with M=2048 tiling (bf16 lhs + fp8 rhs mixed precision) produces 25,048 VLIW bundles (+42% over baseline 17,558), 3,757 register spills, and regresses from 1.974x to 1.818x.
- **Root cause**: The VMEM constraint from FP3 is independent of input precision. bwd_gmm M=2048 with N=256 creates sub-tiles that serialize within each tile, adding loop overhead and register pressure. This applies whether inputs are fp8, bf16, or mixed.
- **Fix**: bwd_gmm M tiling MUST stay at 1024 regardless of precision. The constraint is VMEM capacity and sub-tile serialization, not fp8 scale handling.
- **Extends**: FP3 (VMEM regression), FP6 (tile aspect ratio)
- **First seen**: 2026-03-30, gmm_fp8_blockwise batch round 3

### SO8: Combined SO5+SO6+SO7 — zero backward quantization with unconstrained tiling (1.974x speedup)
- **What**: Combine all backward quantization eliminations in a single variant: SO7 (bf16 grad + fp8 rhs for bwd_gmm) + SO5 (bf16 tgmm) + SO6 (unconstrained bf16 tgmm tiling 2048,256,128). Forward keeps FP8 quantization.
- **Why**: Removing ALL backward quantization eliminates VPU/SFU work AND reduces register pressure (1,822 spills vs 2,777 with partial quantization). The register pressure reduction has a cascading benefit on VLIW schedule quality, producing 17,558 bundles (vs 18,172 baseline) and 896 MXU ops (vs 672 baseline). The combination compounds: SO6's tiling benefit (fewer grid iterations) + SO7's quantization elimination (cleaner schedule).
- **Impact**: 1.974x speedup (5.29ms vs 10.44ms). +6.1% over SO6 (1.861x). +7.7% over SO7 (1.833x). New overall best.
- **Config**: tiling = (1024, 256, 128, 1024, 256, 128, 2048, 256, 128), zero backward quantization, bf16 tgmm
- **Profile**: VLIW 17,558, MXU 896, DMA 21, dual_ratio=1.0, compute_efficiency=13.51%, arithmetic_intensity=12,288
- **Rule**: When optimizations reduce both compute AND register pressure, they compound. Always try combining individually-proven optimizations — the interaction effects can exceed the sum of parts.
- **First seen**: 2026-03-30, gmm_fp8_blockwise batch round 3
