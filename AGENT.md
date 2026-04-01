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

### FP5: stage_profile_deep produces all-null results on TPU v7x (FIXED)
- **What**: `stage_profile_deep` returns `ok: true` but all fields (`vliw_bundle_count`, `mxu_utilization`, `hbm_bandwidth_bytes`, etc.) are null.
- **Root causes** (three issues):
  1. `import jax` at module level triggered libtpu initialization before `_setup_dump_env()` set `XLA_FLAGS`/`LIBTPU_INIT_ARGS`. libtpu reads these flags once at load time.
  2. LLO file parser searched only `llo/` dir. On v7x, VLIW LLO lives in `mosaic/` dump dir (`post-lower-to-llo.txt`, `post-eliminate-llo-extensions.txt`). The `llo/` dir contains scheduled HLO.
  3. v7x uses MLIR LLO format (not classic `;;`-delimited VLIW). Parsers for MXU ops (`.mxu0`/`.mxu1`), VMEM (`#allocation`), DMA (`dma.*`), bundle density (`;;`), and special units didn't match MLIR patterns.
- **Fix**: Deferred `import jax` to `main()` after dump env setup. LLO parser searches `mosaic/` first. All metric parsers updated for MLIR patterns (`llo.vmatmul`, `llo.vector_load/store`, `memref<...memory_space<vmem>>`, `llo.vxpose`, `llo.vdwg`).
- **Rule**: Never import jax at module level in evaluate.py. IR dump flags must be set BEFORE jax loads. LLO parser must handle both classic VLIW and MLIR formats.
- **First seen**: 2026-03-30, gmm_fp8_blockwise iteration 1
- **Fixed**: 2026-04-01, PR #72

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

### [FP10] Forward GMM rejects mixed bf16/fp8 precision — catastrophic correctness failure
- **Symptom**: Passing bf16 lhs + fp8 rhs to `tokamax_backend.gmm()` in forward (transpose_rhs=False) produces max_diff=120,649 — catastrophically wrong output. Both `fwd_mixed_precision` and `reduce_fwd_quant` variants hit this.
- **Root cause**: tokamax's forward gmm requires matching precision for both operands when using FP8 blockwise quantization. The internal scale application path differs between forward (transpose_rhs=False) and backward (transpose_rhs=True). SO7 proved mixed precision works for backward gmm with transpose_rhs=True, but this does NOT extend to forward.
- **Fix**: Forward lhs MUST be quantized to fp8 when rhs is fp8. Mixed precision (bf16 lhs + fp8 rhs) is ONLY valid for backward gmm (transpose_rhs=True). Do NOT skip forward lhs quantization.
- **First seen**: 2026-03-30, gmm_fp8_blockwise session 2, round 1

### [FP16] emit_pipeline body function must match in_specs ref count
- **Symptom**: `TypeError: pipeline_body() missing 1 required positional argument: 'v_tile_ref'` when using `pltpu.emit_pipeline` in Pallas kernel.
- **Root cause**: `emit_pipeline` passes scratch refs to the body function based on the `in_specs` count. If `in_specs` declares N input BlockSpecs, the body function receives N positional ref arguments. A body function expecting `(i_t, k_tile_ref, v_tile_ref)` with only 1 in_spec will fail because pipeline only passes 1 ref.
- **Fix**: Ensure `pipeline_body` signature matches: one ref argument per `in_specs` entry, plus any `out_specs` refs. Use `*refs` for variable-length if needed.
- **First seen**: 2026-03-30, chunk_gla optimization round 1

### [FP17] BlockSpec dimensions must match array dimensions after layout changes
- **Symptom**: `ValueError: Block shape (= (Blocked(block_size=1), Blocked(block_size=1), Blocked(block_size=64), Blocked(block_size=128))) must have the same number of dimensions as the array shape (2, 16, 64, 64, 128)` when changing memory layout.
- **Root cause**: After reshaping an input array (e.g., from 4D to 5D by splitting a dimension), the corresponding `BlockSpec` in `in_specs` must be updated to match the new dimensionality. A 4D BlockSpec `(1, 1, 64, 128)` cannot index a 5D array `(2, 16, 64, 64, 128)`.
- **Fix**: When changing input array shapes/layouts, ALWAYS update the corresponding BlockSpec dimensions to match. Verify ndim(BlockSpec.block_shape) == ndim(array) for every in_spec/out_spec.
- **First seen**: 2026-03-30, chunk_gla optimization round 1

### [FP18] Source-level operation reordering does not affect Pallas/Mosaic VLIW scheduling
- **Symptom**: Restructuring Pallas kernel code to separate VPU and MXU operations into explicit phases produces identical compiled VLIW bundle count (8270) and identical MXU ops (4656) as baseline.
- **Root cause**: The Mosaic compiler (like XLA) schedules operations based on the dataflow graph, not source order. Reordering independent operations in Python has no effect on the compiled TPU schedule. The compiler independently determines optimal VLIW packing.
- **Fix**: Do NOT attempt to influence VLIW scheduling by reordering operations in Pallas kernel code. Focus on reducing total work, changing data dependencies, or altering block dimensions.
- **Extends**: FP8 (same principle, confirmed for Pallas/Mosaic in addition to XLA)
- **First seen**: 2026-03-30, chunk_gla optimization round 1

### [FP19] Recomputing intermediates in split kernels can INCREASE register pressure
- **Symptom**: Split backward kernel that recomputes dA in the dg sub-kernel produces 3.3M register spills (73% WORSE than 1.9M baseline), despite having fewer outputs per kernel.
- **Root cause**: The dA matrix (BT x BT = 64x64 float32 = 16KB) plus all intermediate values needed for its recomputation (q, k, g, masking) overflow registers. The recomputation adds MORE live values than it saves by eliminating one output.
- **Fix**: When splitting kernels, pass intermediate results between kernels via HBM rather than recomputing them. Only recompute if the intermediate is small AND its computation requires few additional live variables.
- **First seen**: 2026-03-30, chunk_gla optimization round 1

### [FP20] Kernel splitting reduces spills but adds launch overhead that can negate gains
- **Symptom**: Split backward kernel achieves 81% spill reduction (1.9M -> 369K) and 37% VLIW reduction (8270 -> 5226), but latency INCREASES 8.3% (8691ms -> 9410ms, 0.825x vs 0.885x baseline).
- **Root cause**: Two separate `pallas_call` invocations incur: (1) additional kernel launch overhead, (2) inter-kernel data transfer (dq passed from kernel A to kernel B via HBM), (3) 6 extra computation events (270 vs 264). The overhead exceeds the savings from simpler, spill-free kernels.
- **Fix**: Prefer optimizing WITHIN the fused kernel (reduce intermediates, simplify control flow, smaller blocks) over splitting into multiple kernels. Splitting is only worthwhile if the per-kernel improvement vastly exceeds the launch+transfer overhead.
- **First seen**: 2026-03-30, chunk_gla optimization round 1

### [FP21] Dual-MXU scheduling not triggered by register pressure reduction alone
- **Symptom**: Even with 369K spills (down from 1.9M, -81%), MXU dual_ratio remains 0.0 (mxu0=2910, mxu1=0). Compiler still places all MXU ops on a single unit.
- **Root cause**: Dual-MXU scheduling depends on matmul dimensions and data dependencies, not just available registers. For chunk_gla's matmul dimensions (128x128 with K=128 on BT=64 chunks), the compiler may determine that a single MXU is sufficient or that data dependencies prevent parallel scheduling.
- **Fix**: To enable dual-MXU, investigate: (1) matmul dimensions — ensure they're large enough to benefit from dual scheduling, (2) independent matmul pairs — the compiler needs two concurrent matmuls with no data dependency to use both MXUs, (3) operand layout — contiguous memory for both MXU ports.
- **First seen**: 2026-03-30, chunk_gla optimization round 1

### SO8: Combined SO5+SO6+SO7 — zero backward quantization with unconstrained tiling (1.974x speedup)
- **What**: Combine all backward quantization eliminations in a single variant: SO7 (bf16 grad + fp8 rhs for bwd_gmm) + SO5 (bf16 tgmm) + SO6 (unconstrained bf16 tgmm tiling 2048,256,128). Forward keeps FP8 quantization.
- **Why**: Removing ALL backward quantization eliminates VPU/SFU work AND reduces register pressure (1,822 spills vs 2,777 with partial quantization). The register pressure reduction has a cascading benefit on VLIW schedule quality, producing 17,558 bundles (vs 18,172 baseline) and 896 MXU ops (vs 672 baseline). The combination compounds: SO6's tiling benefit (fewer grid iterations) + SO7's quantization elimination (cleaner schedule).
- **Impact**: 1.974x speedup (5.29ms vs 10.44ms). +6.1% over SO6 (1.861x). +7.7% over SO7 (1.833x). New overall best.
- **Config**: tiling = (1024, 256, 128, 1024, 256, 128, 2048, 256, 128), zero backward quantization, bf16 tgmm
- **Profile**: VLIW 17,558, MXU 896, DMA 21, dual_ratio=1.0, compute_efficiency=13.51%, arithmetic_intensity=12,288
- **Rule**: When optimizations reduce both compute AND register pressure, they compound. Always try combining individually-proven optimizations — the interaction effects can exceed the sum of parts.
- **First seen**: 2026-03-30, gmm_fp8_blockwise batch round 3

### SO9: Phase-specialized tiling — TM=256 forward + TK=512 tgmm (2.294x speedup)
- **What**: Use different tiling per compute phase instead of uniform tiling. Forward gmm: (256, 256, 128), bwd_gmm: (1024, 256, 128), tgmm: (2048, 512, 128). Combined with SO8's zero backward quantization.
- **Why**: TM=256 for forward matches per-group M exactly (M=8192 / G=32 = 256), creating perfectly-sized tiles with no wasted computation. This produces 2x more MXU ops (1,792 vs 896) because the compiler generates more matmul tiles that each do a full 256x256 matmul. TK=512 for tgmm halves K-loop iterations (bf16 tgmm has no fp8 constraints on K).
- **Impact**: 2.294x speedup (4.692ms vs 10.762ms). +16.2% over SO8 (1.974x). New overall best.
- **Config**: tiling = (256, 256, 128, 1024, 256, 128, 2048, 512, 128), zero backward quantization, bf16 tgmm
- **Profile**: VLIW 23,462, MXU 1,792, DMA 21, dual_ratio=1.0, compute_efficiency=15.24%, spills=156
- **Key insight**: More VLIW bundles (23,462 vs 17,558) but FASTER because MXU throughput doubled. VLIW bundle count alone is not a reliable quality metric — MXU ops per bundle matters more.
- **Rule**: For grouped matmul, match TM to per-group M dimension for optimal tile utilization. More tiles with exact sizing beats fewer tiles with partial utilization.
- **First seen**: 2026-03-30, gmm_fp8_blockwise session 2, round 1

### SO10: Forward TK=512 + tgmm TK=512 (2.335x speedup)
- **What**: Increase forward K-tiling from 256 to 512. Tiling: (256, 512, 128, 1024, 256, 128, 2048, 512, 128).
- **Why**: Forward TK=512 halves K-loop iterations for Gate/Up (K=2048: 4 iterations instead of 8). For Down (K=512), TK=512 means single K-iteration. Combined with tgmm TK=512 from SO9.
- **Impact**: 2.335x speedup (4.403ms). +1.8% over SO9 (2.294x).
- **Profile**: VLIW 23,462, MXU 1,792, dual_ratio=1.0. Same VLIW/MXU as SO9.
- **Rule**: For forward gmm with fp8 inputs, TK can exceed tile_size=128 without issues. Maximize TK up to the minimum K dimension across shapes (512 for this kernel).

### SO11-gmm: Uniform TM=256 + bwd_gmm TK=512 compound effect (2.460x speedup)
- **What**: Use TM=256 for BOTH fwd and bwd_gmm (per-group M alignment), plus TK=512 for bwd_gmm. Tiling: (256, 256, 128, 256, 512, 128, 2048, 512, 128).
- **Why**: TM=256 matching per-group M (M=8192/G=32=256) creates exactly 1 sub-tile row, producing the simplest inner loop. TK=512 halves K-loop iterations for bwd_gmm. The combination compounds: TM=256 creates clean sub-tiles that TK=512 can efficiently iterate over. TM=1024 + TK=512 (L1_bwd_tk512) actually regressed because 1024/256=4 sub-tile rows × 4 K iterations creates complex loop nesting.
- **Impact**: 2.460x speedup (4.182ms). +5.4% over SO10 (2.335x). New overall best.
- **Profile**: VLIW 23,462, MXU 1,792, dual_ratio=1.0, compute_efficiency=17.09%, spills=234
- **Rule**: For bwd_gmm, TM=256 (per-group alignment) enables TK=512 benefits. TM=1024+TK=512 does NOT help. Always pair per-group TM with larger TK.
- **First seen**: 2026-03-31, gmm_fp8_blockwise session 2, round 3

### [FP12-gmm] Forward out_dtype=bfloat16 causes correctness failure
- **Symptom**: Setting `out_dtype=jnp.bfloat16` in forward `tokamax_backend.gmm()` produces max_diff=2065.6875 (atol=1.0)
- **Root cause**: bf16 accumulation truncates partial products during K-reduction. For K=2048, the sum of 2048 fp8×fp8 products loses significant precision at bf16. The reference uses f32 accumulation.
- **Fix**: Forward `out_dtype` MUST remain `jnp.float32`. Do NOT use bf16 accumulation for forward pass.
- **First seen**: 2026-03-31, gmm_fp8_blockwise session 2, round 3

### [FP13-gmm] tgmm TM=4096 causes VLIW complexity bloat without speedup
- **Symptom**: tgmm TM=4096 doubles VLIW bundles (47,683 vs 23,462) and MXU ops (3,584 vs 1,792) but regresses speedup from 2.335x to 2.277x (-2.5%).
- **Root cause**: TM=4096 tiles are too large for efficient compilation. The compiler generates 2x more code per tile without reducing total grid iterations proportionally. Accumulators of 4096×128×4B=2MB per tile likely cause internal compiler complexity.
- **Fix**: tgmm TM=1024 is optimal (see SO14-gmm). TM=2048 works but is suboptimal. TM=4096 is counterproductive.
- **First seen**: 2026-03-31, gmm_fp8_blockwise session 2, round 3
- **Updated**: 2026-03-31, round 6 — TM=1024 proved superior to TM=2048 (SO14-gmm)

### [FP14-gmm] bwd_gmm TK=512 regresses with TM=1024 but improves with TM=256
- **Symptom**: bwd_gmm (1024, 512, 128) regresses to 2.307x from 2.335x (-1.2%). But bwd_gmm (256, 512, 128) improves to 2.460x (+5.4%).
- **Root cause**: TM=1024 creates 1024/256=4 sub-tile rows, and TK=512 creates 512/128=4 K-iterations. The 4×4 inner loop structure generates 24,673 VLIW bundles (+5.2% over 23,462). TM=256 creates 1 sub-tile row × 4 K-iterations — much simpler loop structure.
- **Fix**: When increasing TK, ensure TM is small enough to keep sub-tile count low. Per-group TM alignment (TM=256) is the prerequisite for TK enlargement.
- **First seen**: 2026-03-31, gmm_fp8_blockwise session 2, round 3

### SO12-gmm: Optimal tiling — all TK=512, all TM=256 for gmm (2.651x speedup)
- **What**: Cross-pollinate L1 (fwd TK=512) with L2 (bwd TM=256+TK=512). Tiling: (256, 512, 128, 256, 512, 128, 2048, 512, 128).
- **Why**: All gmm phases use TM=256 (per-group M alignment) and TK=512 (halved K-loops). tgmm uses proven TM=2048, TK=512, TN=128. This is the culmination of SO9+SO10+SO11-gmm.
- **Impact**: 2.651x speedup (3.898ms). +7.8% over SO11-gmm (2.460x). New overall best.
- **Profile**: VLIW 23,462, MXU 1,792, dual_ratio=1.0, compute_efficiency=18.34%, spills=156
- **Rule**: For grouped matmul with per-group M=256: use TM=256 for ALL gmm phases (fwd + bwd), TK=512 for ALL phases, TN=128. tgmm: TM=2048, TK=512, TN=128.
- **First seen**: 2026-03-31, gmm_fp8_blockwise session 2, round 4

### SO13-gmm: Forward TN=256 — N-grid halving (2.769x speedup)
- **What**: Increase forward TN from 128 to 256. Tiling: (256, 512, 256, 256, 512, 128, 2048, 512, 128).
- **Why**: For Gate/Up (N=512), TN=256 halves N-grid from 4→2 tiles. For Down (N=2048), 16→8 tiles. Fewer grid iterations reduce kernel launch overhead. Same VLIW/MXU profile (23,462 bundles, 1,792 ops) but faster due to fewer grid iterations.
- **Impact**: 2.769x speedup (~3.80ms). +4.5% over SO12-gmm (2.651x). New overall best.
- **Profile**: VLIW 23,462, MXU 1,792, dual_ratio=1.0, compute_efficiency=19.25%, spills=156
- **Rule**: Forward TN=256 is safe and beneficial for these shapes. N=512 gives exactly 2 tiles (perfect), N=2048 gives 8 tiles.
- **First seen**: 2026-03-31, gmm_fp8_blockwise session 2, round 5

### [FP24-gmm] bf16 scale_dtype and tgmm TM=1024 are mutually exclusive optimizations
- **Symptom**: Cross-pollinating SO14-gmm (tgmm TM=1024) with SO15-gmm (bf16 scales) regresses on BOTH lineages: L2_scale_bf16 (2.682x vs 2.847x, -5.8%), L1_tgmm_tm1024 (2.688x vs 2.819x, -4.6%). Spills decrease (156→81/84) but speedup decreases.
- **Root cause**: bf16 scale dtype changes the compilation path such that the VLIW simplification from tgmm TM=1024 is negated. The two optimizations each work by changing the compiler's scheduling decisions, and these scheduling changes conflict. Fewer spills does NOT guarantee better performance when the VLIW schedule quality degrades.
- **Fix**: Do NOT combine bf16 scale_dtype with tgmm TM=1024. Keep them as separate lineage strategies. L2 uses f32 scales + TM=1024, L1 uses bf16 scales + TM=2048.
- **First seen**: 2026-03-31, gmm_fp8_blockwise session 2, round 7

### [FP25-gmm] tgmm TM<1024 causes regression — TM=1024 is the sweet spot
- **Symptom**: tgmm TM=512 (2.719x, 576 MXU, 231 spills) and TM=256 (2.276x, 576 MXU, 84 spills) both regress from TM=1024 (2.847x, 896 MXU, 156 spills).
- **Root cause**: TM=512 and TM=256 reduce MXU ops to 576 (from 896 at TM=1024) because smaller tiles produce fewer matmul operations per tile. The VLIW simplification trend (23,462→12,951) does NOT continue below TM=1024 (12,360 for TM=512, 8,497 for TM=256) — the overhead of more grid iterations dominates. TM=256 for tgmm does NOT benefit from per-group alignment the way fwd/bwd does, because tgmm's computation structure is fundamentally different (weight gradient accumulation vs. activation multiplication).
- **Fix**: tgmm TM MUST be 1024 for bf16 tgmm. The range [512, 4096] has been fully explored: TM=512 (-4.5%), TM=1024 (optimal), TM=2048 (-2.7%), TM=4096 (FP13-gmm, bloat).
- **First seen**: 2026-03-31, gmm_fp8_blockwise session 2, round 7

### [FP27-gmm] tgmm TM=1024 improvement is code-path specific (L2 only)
- **Symptom**: tgmm TM=1024 with f32 scales on L1 code path gives 2.241x — a massive regression from L1's 2.751x (TM=2048). The exact same tgmm TM=1024 on L2 gives 2.847x (+2.8% improvement).
- **Root cause**: The XLA/Mosaic compiler makes different scheduling decisions based on the full kernel code, not just the tiling parameters. L1 and L2 have identical tiling but subtly different code structure (from their independent evolutionary paths). tgmm TM=1024 simplifies the VLIW schedule in a way that benefits L2's specific compilation but hurts L1's.
- **Fix**: Do NOT assume tiling improvements are portable between code paths. Always test each optimization on each lineage independently. tgmm TM=1024 is L2-specific; L1 should stay at TM=2048.
- **First seen**: 2026-03-31, gmm_fp8_blockwise session 2, round 8

### [FP26-gmm] bwd_gmm TN<128 causes correctness failure
- **Symptom**: bwd TN=64 (with _clamp_tiling min lowered to 64) returns INCORRECT status.
- **Root cause**: tokamax's internal tiling logic assumes TN >= 128. Sub-128 N tiles break the matmul decomposition or scale application, producing incorrect results.
- **Fix**: All tiling dimensions MUST be >= 128. Do NOT lower the _clamp_tiling minimum below 128.
- **First seen**: 2026-03-31, gmm_fp8_blockwise session 2, round 7

### [FP22-gmm] Forward TK=1024 causes major regression
- **Symptom**: fwd TK=1024 with TN=256 regresses from 2.751x to 2.498x (-9.2%), increases spills from 156 to 234.
- **Root cause**: TK=1024 creates tiles that exceed the compiler's efficient scheduling range. For K=2048, TK=1024 means only 2 K-iterations — the loop body is too large (one iteration processes too much data) causing register pressure.
- **Fix**: Forward TK must stay at 512. TK=1024 exceeds the optimal point.
- **First seen**: 2026-03-31, gmm_fp8_blockwise session 2, round 6

### [FP23-gmm] Forward TN=512 regression — single-tile N inefficient
- **Symptom**: fwd TN=512 for Gate/Up (N=512) regresses from 2.769x to 2.410x (-13.0%), 231 spills.
- **Root cause**: TN=512 covers the entire N dimension for Gate/Up in a single tile, but the 256×512 tile output is too large for efficient register allocation. TN=256 (2 tiles) provides better scheduling granularity.
- **Fix**: Forward TN must stay at 256. Do NOT set TN equal to N for any shape.
- **First seen**: 2026-03-31, gmm_fp8_blockwise session 2, round 6

### [FP16b-gmm] bwd_gmm TN=256 causes catastrophic register spills
- **Symptom**: bwd_gmm TN=256 produces 9,447 register spills (60x increase from 156), regressing speedup from 2.651x to 2.295x (-13.4%).
- **Root cause**: bwd_gmm with transpose_rhs=True creates intermediate tensors proportional to TN. TN=256 doubles the intermediate buffer size, overflowing register capacity. The compiler spills 9,447 vectors to VMEM.
- **Fix**: bwd_gmm TN MUST stay at 128. Only forward gmm (transpose_rhs=False) benefits from TN=256.
- **First seen**: 2026-03-31, gmm_fp8_blockwise session 2, round 5

### [FP15-gmm] tgmm TN=256 halves MXU ops — regression like FP2
- **Symptom**: tgmm TN=256 (vs TN=128) drops MXU ops from 1,792 to 1,024 (-43%), increases VLIW bundles from 23,462 to 24,465 (+4.3%), regresses speedup from 2.460x to 2.452x.
- **Root cause**: Larger N tiles change the tgmm matmul decomposition, reducing the number of MXU operations per tile. Similar mechanism to FP2 (N>128 constraint) but for bf16 tgmm — even without fp8 scale issues, the larger N tiles produce suboptimal compilation.
- **Fix**: tgmm TN MUST stay at 128 regardless of input precision (fp8 or bf16). This extends FP2 beyond fp8 constraints.
- **First seen**: 2026-03-31, gmm_fp8_blockwise session 2, round 4

### SO14-gmm: tgmm TM=1024 — halved VLIW complexity (2.847x speedup)
- **What**: Reduce tgmm TM from 2048 to 1024. Tiling: (256, 512, 256, 256, 512, 128, 1024, 512, 128).
- **Why**: TM=1024 halves the tgmm tile work, producing dramatically simpler compiled code. VLIW bundles dropped from 23,462 to 12,951 (-44.8%), MXU ops halved from 1,792 to 896. Despite doing half the MXU ops per tile, the kernel is faster because the simpler VLIW schedule has better pipeline utilization. This is the opposite of FP13-gmm (TM=4096 bloat) — the compiler efficiency sweet spot is at TM=1024.
- **Impact**: 2.847x speedup (~3.61ms). +2.8% over SO13-gmm (2.769x). New overall best.
- **Profile**: VLIW 12,951, MXU 896, dual_ratio=1.0, compute_efficiency=19.80%, spills=156
- **Rule**: For bf16 tgmm with G=32 groups: TM=1024 is optimal. Smaller tiles = simpler VLIW = faster execution, even with fewer MXU ops. This creates 8 tgmm tiles per group (M=8192 / TM=1024 = 8).
- **First seen**: 2026-03-31, gmm_fp8_blockwise session 2, round 6

### SO15-gmm: bf16 scale_dtype — register pressure elimination (2.819x speedup)
- **What**: Use `scale_dtype=jnp.bfloat16` (instead of float32) for all FP8 quantization scale factors. Tiling unchanged: (256, 512, 256, 256, 512, 128, 2048, 512, 128).
- **Why**: bf16 scale factors halve the register footprint of scale tensors. This reduces register spills from 156 to 9 (-94%). The compiler can keep more values in registers, improving VLIW scheduling quality. bf16 scales have sufficient precision for FP8 blockwise quantization.
- **Impact**: 2.819x speedup (~3.65ms). +2.5% over L1's previous best (2.751x).
- **Profile**: VLIW 23,462, MXU 896, dual_ratio=1.0, compute_efficiency=19.59%, spills=9
- **Note**: Previously, bf16 scales interfered with L2+TK=512 compilation (R4, 2.456x vs 2.570x). On L1's code path, they work well. Code-path sensitivity persists.
- **Rule**: bf16 scale_dtype can eliminate register pressure when the kernel has significant spills. Test on each lineage's code path separately.
- **First seen**: 2026-03-31, gmm_fp8_blockwise session 2, round 6

### [FP22] jax.lax.dynamic_slice not supported in Pallas TPU lowering (Mosaic)
- **Symptom**: `NotImplementedError: Unimplemented primitive in Pallas TPU lowering for tc: dynamic_slice`
- **Root cause**: `jax.lax.dynamic_slice` is NOT implemented in Pallas TPU kernel lowering (Mosaic). The K-tiling approach used `dynamic_slice` to slice already-loaded arrays within the kernel body — this is not available on TPU.
- **Fix**: Cannot use `dynamic_slice` in Pallas kernels. Use only static indexing with compile-time constants (e.g., `array[:, 0:64]` and `array[:, 64:128]`) or add dimensions to the grid/BlockSpec to tile at the pallas_call level. Note that static manual unrolling may not help register pressure (see FP23).
- **First seen**: 2026-03-30, chunk_gla optimization round 2

### [FP23] Manual loop unrolling in Pallas kernels increases register pressure
- **Symptom**: Manually unrolling the V-dimension loop (V=128 → 2x64) caused VLIW bundles to increase 59% (8270 → 13140), register spills to triple (1.96M → 5.74M), and speedup to improve only marginally (0.825x → 0.872x).
- **Root cause**: Manual unrolling gives the compiler visibility into both iterations simultaneously. The compiler hoists all intermediate arrays from both V-halves, keeping them live concurrently. This is the OPPOSITE of the intended effect — instead of reducing peak liveness, it doubles it.
- **Fix**: Do NOT manually unroll loops in Pallas kernels to reduce register pressure. The compiler will hoist all iterations, increasing live values. To actually reduce liveness, use `jax.lax.fori_loop` (which hides iterations from the compiler) or add the tiled dimension to the pallas_call grid via BlockSpec.
- **First seen**: 2026-03-30, chunk_gla optimization round 2

### [FP24] BT (chunk block tile) < 64 critically underutilizes MXU on TPU v7x
- **Symptom**: chunk_size=32 (BT=32) drops MXU runtime utilization to 12% (from 22.5% at BT=64). Near-zero register spills but doubled grid iterations. Speedup: 0.811x (worse than 0.885x baseline).
- **Root cause**: With BT=32, matmul dimensions are [32,128]@[128,128]. The 32-wide dimension underutilizes the 128x128 MXU systolic array — only 1/4 of MXU rows are active per matmul. More grid iterations (128 vs 64 tiles) add launch overhead that exceeds the simpler-code benefit.
- **Fix**: BT (chunk block tile) must be >= 64 for reasonable MXU utilization on TPU v7x. BT=32 is below the threshold for efficient MXU use.
- **First seen**: 2026-03-30, chunk_gla optimization round 2

### [FP11] tgmm TK=256 halves MXU ops — critical performance factor
- **Symptom**: tgmm TK=256 (vs TK=512) drops speedup from 2.294x to 2.046x (-10.8%). VLIW bundles halve from 23,462 to 17,558, MXU ops halve from 1,792 to 896.
- **Root cause**: tgmm with TK=512 generates 2x more matmul tiles, each doing a full K=512 reduction. TK=256 reduces the matmul work per tile by half. The extra VLIW bundles with TK=512 are dominated by MXU ops, which is beneficial.
- **Fix**: tgmm TK MUST be 512 (not 256) for optimal performance. The K-dimension is the most impactful tiling parameter for tgmm.
- **First seen**: 2026-03-30, gmm_fp8_blockwise session 2, round 2

### SO12: Forward A fusion — recompute A inside forward output kernel (1.461x speedup)
- **What**: Recomputed the attention matrix A inside the forward output kernel `_chunk_gla_fwd_o_gk_pl` by adding k as an input and computing `b_A = (q * exp(g)) @ (k * exp(-g)).T * scale` inline. Eliminated the separate `chunk_gla_fwd_intra_gk` pallas_call from forward. Removed A from residuals (backward already recomputes it via SO11).
- **Why**: Same principle as SO11 (backward reduce_inputs): eliminating a separate pallas_call saves kernel launch overhead, DMA transfers, and HBM round-trips. The forward A kernel had its own compilation, launch, and data movement. The A tensor was also stored as a forward-to-backward residual — removing it saves additional HBM bandwidth.
- **Impact**: 1.097x → 1.461x (+33.2%). Computation events decreased from 237 to 213 (-10.1%).
- **Applicable when**: A forward pass computes an intermediate (like attention matrix A) in a separate pallas_call, and that intermediate can be cheaply recomputed inside another kernel that already has the required inputs.
- **First seen**: 2026-03-30, chunk_gla optimization round 3

### SO13: Skip dead output computation — removing discarded dg gradient (compounds with SO12)
- **What**: Removed all dg (gate gradient) computation from the backward kernel. The caller discards dg (`dq, dk, dv, _ = ...`), making all dg work dead code: 1 MXU matmul (M_upper @ dg_raw, [BT,BT]@[BT,K]), VPU mask construction, dgk_inter accumulation, dg_ref output write.
- **Why**: Pure dead-code elimination. The dg computation accounted for ~12.5% of VLIW bundles (9058 → 7930) and ~11% of MXU ops (5238 → 4656). Alone it only improved speedup +0.5% (1.097x → 1.102x), but **combined with SO12 it compounds**: 1.097x → 1.577x (+43.7%) vs SO12 alone at 1.461x (+33.2%). The compound effect is +7.9% on top of SO12.
- **Impact**: Alone: +0.5%. Combined with SO12: +43.7% total. VLIW reduced 12.5%.
- **Key insight**: **Intra-kernel simplification has minimal standalone impact but compounds with inter-kernel optimizations.** Reducing VLIW complexity allows the compiler to produce tighter schedules that benefit more from kernel launch reduction.
- **Applicable when**: A Pallas kernel produces outputs that the caller discards. Check all output Refs.
- **First seen**: 2026-03-30, chunk_gla optimization round 3

### [FP25] Mixed bf16/f32 matmul inputs not supported in Mosaic; bf16 + Precision.HIGHEST also rejected
- **Symptom**: `MosaicError: INTERNAL: Mosaic failed to compile TPU kernel: Bad lhs type` with MLIR op `tpu.matmul (vector<64x256xbf16>, vector<256x256xf32>, vector<64x256xf32>)` OR `tpu.matmul (vector<64x256xbf16>, vector<256x256xbf16>, vector<64x256xf32>)` when `precision=lax.Precision.HIGHEST` is set.
- **Root cause**: Two distinct constraints: (1) Mosaic rejects mixed-precision matmul inputs (bf16 + f32). (2) Even with BOTH operands in bf16, Mosaic rejects the matmul when `precision=lax.Precision.HIGHEST` is specified. HIGHEST forces `contract_precision<fp32>` in MLIR, which is incompatible with bf16 input types.
- **Fix**: Either (a) keep all matmul operands in float32 with HIGHEST precision, or (b) cast both operands to bf16 AND remove/lower the precision parameter to `lax.Precision.DEFAULT`. Cannot have bf16 inputs AND HIGHEST precision simultaneously.
- **First seen**: 2026-03-30, chunk_gla optimization round 4
- **Extended**: 2026-03-30, chunk_gla round 5 (L2_bf16_uniform confirmed bf16+HIGHEST constraint)
- **Extended**: 2026-03-31, chunk_gla round 10 — bf16 residual storage (q, k, do arrays stored as bf16 between forward and backward) triggers FP25 when backward kernel uses `.astype(b_v.dtype)` / `.astype(b_q.dtype)` intermediate casts that propagate bf16 to HIGHEST-precision matmuls. Fix: replace ALL intermediate casts with explicit `.astype(jnp.float32)` before `jnp.dot(..., precision=HIGHEST)`. 3 of 5 R10 variants hit this.

### [FP26] PrefetchScalarGridSpec index_map must accept all non-scalar grid dimensions
- **Symptom**: `TypeError: kqg_map() takes 3 positional arguments but 4 were given` when using `PrefetchScalarGridSpec` with grid `(B, H, 1, 1, NT)` and `num_scalar_prefetch=1`.
- **Root cause**: With `PrefetchScalarGridSpec(num_scalar_prefetch=S, grid=G)`, the index_map functions receive `len(G) - S` arguments (the non-scalar grid dimensions). For grid `(B, H, 1, 1, NT)` with `num_scalar_prefetch=1`, maps need 4 args: `(b, h, ki, vi, t)` minus 1 scalar = 4 args. Defining maps with only `(b, h, t)` (skipping the constant-1 dims) causes argument count mismatch.
- **Fix**: index_map functions must always accept `len(grid) - num_scalar_prefetch` arguments, even for grid dimensions that are 1. Include placeholder args for constant dimensions: `lambda b, h, _ki, _vi, t: ...`.
- **First seen**: 2026-03-30, chunk_gla optimization round 4

### [FP27] Replacing exp(-x) with reciprocal(exp(x)) is NOT faster on TPU v7x
- **Symptom**: L1_reduce_exp variant regressed from 1.577x to 1.498x (-5.0%) by replacing `exp(-x)` with `1.0 / exp(x)` and `exp(a-b)` with `exp(a) / exp(b)`.
- **Root cause**: TPU v7x has a dedicated hardware Exponential Unit (EUP) that handles exp() efficiently. Replacing exp() calls with ALU division operations trades cheap EUP cycles for more expensive VPU division instructions. The EUP utilization dropped as expected (2.10% → 1.37%), but ALU divisions added pipeline stalls that negated any benefit.
- **Fix**: Do NOT try to reduce exp() calls by converting to reciprocal/division form. The hardware EUP handles exp() more efficiently than the ALU handles division. Keep exp(-x) as-is.
- **First seen**: 2026-03-30, chunk_gla optimization round 4

### SO14: lax.scan → pallas_call with "arbitrary" dimension_semantics (6.831x — MASSIVE breakthrough)
- **Optimization**: Replaced `lax.scan` for backward dh state propagation (64 sequential iterations) with a single `pallas_call` using `dimension_semantics=("parallel","parallel","arbitrary","arbitrary","arbitrary")`. The time dimension is marked "arbitrary" (sequentially iterated), keeping the dh state in VMEM scratch between iterations. Input arrays are flipped before pallas_call (to implement reverse scan) and output flipped after.
- **Impact**: 1.577x → 6.831x (+333%) on shape B=2, T=4096, H=16, K=128, V=128, chunk_size=64. Latency: 4884ms → 1122ms (-77%).
- **Why it works**: `lax.scan` dispatches each of 64 iterations as a SEPARATE XLA computation with full host→device communication, kernel scheduling, and HBM allocation for intermediates. Each scan iteration cost ~59ms of dispatch overhead — far more than the actual computation. `pallas_call` with "arbitrary" dims compiles ALL 64 iterations into a single on-chip kernel loop with no per-iteration dispatch. The Pallas kernels themselves are IDENTICAL (same VLIW bundles, MXU ops, register spills, DMA) — only the dispatch mechanism changed.
- **Critical evidence**: Computation events dropped only 4.3% (207 → 198), yet latency dropped 77%. This breaks the previous "10% event reduction → 25-35% speedup" correlation because lax.scan events are VASTLY more expensive per event than pallas_call grid tile events.
- **Applicable when**: ANY `lax.scan` or `jax.lax.scan` with sequential state dependency (h[t] = f(h[t-1], x[t])) in a Pallas program. Replace with a `pallas_call` where the time dimension uses "arbitrary" semantics and state is kept in VMEM scratch. This is the single most impactful optimization pattern discovered.
- **First seen**: 2026-03-30, chunk_gla optimization round 4

### SO15: Forward kernel fusion — merge chunk_fwd_h + chunk_gla_fwd_o_gk into single pallas_call (7.845x)
- **Optimization**: Merged the two forward pallas_calls (chunk_fwd_h for inter-chunk state propagation, chunk_gla_fwd_o_gk for output combination) into a single `_chunk_fwd_combined_kernel` with grid (B, H, K/BK, V/BV, NT). Time dimension uses "arbitrary" semantics. VMEM scratch holds h state [BK, BV] between time steps. At each step: save h → compute output (recompute A inline) → update h.
- **Impact**: 6.831x → 7.845x (+14.8%). Computation events: 198 → 177 (-10.6%). The 10.6% event reduction produced 14.8% speedup, consistent with the "10% event reduction → ~25% speedup" ratio for pallas_call grid tile events.
- **Why it works**: Eliminates one pallas_call kernel launch, the h tensor HBM write-then-read cycle between the two forward kernels, and 21 computation events. The h state stays in VMEM scratch instead of round-tripping through HBM.
- **Applicable when**: Forward pass has two sequential pallas_calls where one produces state consumed by the next. The "arbitrary" dimension semantics on the time dimension enables sequential state propagation within a single kernel.
- **Relationship**: Extends SO14 (lax.scan→pallas_call) and SO12 (forward A fusion). The forward is now a single pallas_call with A recomputed inline and h state in scratch.
- **First seen**: 2026-03-30, chunk_gla optimization round 5

### [FP28] BT=128 (chunk_size doubling) counterproductive for chunk_gla — VLIW bloat exceeds tile reduction benefit
- **Symptom**: BT=128 regressed to 5.395x (-21.0% from 6.831x at BT=64). Despite halving grid iterations (NT=64→32) and improving MXU util (33.2%→45.9%), overall latency increased 26% (1122ms→1412ms).
- **Root cause**: Per-tile VLIW complexity increased +44% (7930→11449 bundles), MXU ops increased +33% (4656→6192), and register spills increased +6.1% (2.50M→2.65M). The [128,128] attention matrix is 4x larger than [64,64], creating substantial register pressure. Computation events stayed at 198 despite halving NT — the overhead per tile grew to consume the iteration savings.
- **Fix**: BT=64 is the optimal chunk size for chunk_gla on shape B=2, T=4096, H=16, K=128, V=128. Do NOT increase BT beyond 64. The per-tile VLIW complexity growth exceeds the benefit of fewer tiles.
- **Extends**: FP24 (BT<64 underutilizes MXU). Together: BT must be exactly 64 for this kernel.
- **First seen**: 2026-03-30, chunk_gla optimization round 5

### [FP29] Staged backward output writes reduce register spills but can regress MXU utilization
- **Symptom**: Staged output writes (write dv → dq → dk sequentially, deferring exp computations) reduced register spills 9.8% (2.50M→2.25M) and Vector Store util 30.7% (9.81%→6.80%), but MXU runtime util dropped 24.0% (33.20%→25.23%). Net speedup improvement was marginal (+1.8%, 6.831x→6.957x).
- **Root cause**: The staged writes introduce dependency barriers between output write groups. While these barriers help the register allocator (arrays become dead earlier), they also prevent the compiler from overlapping MXU operations across the barriers. The serialization between dv/dq/dk phases creates pipeline bubbles.
- **Fix**: Do NOT use explicit staging barriers in Pallas backward kernels to reduce register pressure. The MXU scheduling regression typically negates the spill reduction benefit. Better approaches: reduce total work (fewer intermediates) or fuse kernels (fewer pallas_calls).
- **First seen**: 2026-03-30, chunk_gla optimization round 5

### [FP30] fori_loop + lax.cond inside Pallas kernels INCREASES register pressure
- **Symptom**: Using `jax.lax.fori_loop(0, 2, body)` with `lax.cond` to split backward fused kernel into 2 sequential passes increased register spills +66% (2.5M → 4.15M), fills +83% (3.0M → 5.6M), and dropped MXU util from 33.2% to 13.5%. Speedup regressed from 7.845x to 7.731x.
- **Root cause**: The fori_loop carry tuple materializes all shared arrays in VMEM at loop boundaries. The `lax.cond` branches add conditional control flow that the compiler must handle conservatively, keeping both branches' intermediates potentially live. The carry + cond overhead exceeds any benefit from hiding iterations.
- **Fix**: Do NOT use `jax.lax.fori_loop` with `lax.cond` inside Pallas TPU kernels to reduce register pressure. The FP23 advice about fori_loop hiding iterations applies to simple loops, not to loops with conditional branches. For register pressure reduction, prefer: kernel fusion (eliminate pallas_calls), reducing total intermediate count, or adding grid dimensions via BlockSpec.
- **First seen**: 2026-03-31, chunk_gla optimization round 6

### [FP31] Precision.DEFAULT causes correctness failure for chunk_gla (max_diff=23.91)
- **Symptom**: Changing all `precision=lax.Precision.HIGHEST` to `lax.Precision.DEFAULT` produces max_diff=23.91 (atol=10.0).
- **Root cause**: DEFAULT precision allows bf16 intermediate accumulation. With BT=64 and K=128, matmul sums of 8192 products at bf16 lose significant precision. Errors accumulate across 64 chunks and the loss sum, exceeding the relaxed atol=10.0 threshold.
- **Fix**: chunk_gla REQUIRES `Precision.HIGHEST` for correctness. Do NOT reduce precision. This closes off precision reduction as an optimization direction for this kernel.
- **First seen**: 2026-03-31, chunk_gla optimization round 6

### [FP32] TPU block shape last-two-dim alignment constraint (8 and 128)
- **Symptom**: `ValueError: The Pallas TPU lowering currently requires that the last two dimensions of your block shape are divisible by 8 and 128 respectively, or be equal to the respective dimensions of the overall array.` Block spec `(1, 1, 64, 64)` for array shape `(16, 128, 64, 128)` — last dim 64 not divisible by 128.
- **Root cause**: Pallas TPU (Mosaic) lowering enforces that the last two dimensions of every BlockSpec block_shape must either: (1) be divisible by 8 (second-to-last) and 128 (last), OR (2) equal the full array dimension. This constraint applies to ALL in_specs and out_specs arrays. When V-tiling with BV_inner=64, the block shape's last dim becomes 64, which violates the 128-divisibility requirement.
- **Fix**: V-dimension sub-tiling with BV < 128 is impossible via BlockSpec when V is the last array dimension. To V-tile, either: (1) transpose the array so V is NOT the last dimension, (2) keep BV=128 (full V dim), or (3) use a different approach (e.g., recomputation within the kernel body, not BlockSpec tiling). The constraint is on the block_shape, not the grid — adding a V grid dimension doesn't help if the block still has last dim < 128.
- **First seen**: 2026-03-31, chunk_gla optimization round 7

### SO16: Backward kernel fusion — merge bwd_dh + bwd_fused into single pallas_call (8.988x)
- **Optimization**: Merged the two backward pallas_calls (chunk_bwd_dh_pallas for reverse-time dh state propagation, chunk_gla_bwd_fused for dq/dk/dv computation) into a single `_chunk_bwd_combined_kernel` with grid (B, H, K/BK, V/BV, NT). Time dimension uses "arbitrary" semantics with reverse scan. VMEM scratch holds dh state [BK, BV] between time steps. At each step: load h for this time step → compute dq/dk/dv using h and dh → update dh state. 5D BlockSpec `(1,1,1,BK,BV)` with 5-element index_map for h array `[B,NT,H,K,V]`. **KEY FIX from R6**: h BlockSpec MUST be 5D to match the 5D h array — R6 used 4D BlockSpec `(1,1,K,V)` which caused ndim mismatch.
- **Impact**: 7.845x → 8.988x (+14.6%). Computation events: 177 → 171 (-3.4%). Architecture reduced from 3 pallas_calls to 2 total.
- **Why it works**: Eliminates the dh tensor HBM round-trip (512MB for B=2,H=16,NT=64,K=128,V=128 at f32). The dh state stays in VMEM scratch instead of being written to HBM by bwd_dh and read back by bwd_fused. Same principle as SO15 (forward fusion) — merging sequential state-dependent kernels into a single pallas_call with VMEM scratch.
- **Trade-off**: VLIW bundles increased +18.8% (7930→9420), register spills increased +8.8% (2.50M→2.72M). The HBM round-trip elimination vastly outweighs the increased kernel complexity.
- **Applicable when**: Backward pass has two sequential pallas_calls where one propagates state (dh) consumed by the next. The "arbitrary" dimension semantics on the time dimension enables sequential state propagation within a single kernel. Requires 5D BlockSpec matching the 5D state array.
- **Relationship**: Mirrors SO15 (forward fusion). Combined, the kernel now has just 2 pallas_calls: 1 combined forward + 1 combined backward.
- **First seen**: 2026-03-31, chunk_gla optimization round 7

### [FP33] Manual VMEM scratch staging counterproductive for Pallas backward kernel register pressure
- **Symptom**: Adding 3 explicit VMEM scratch buffers (scratch_exp [BT,BK], scratch_A [BT,BT], scratch_dA [BT,BT]) for intermediate staging reduced register spills only 5% (2.72M→2.58M) while regressing speedup -2.7% (8.988x→8.749x).
- **Root cause**: The explicit scratch store/load instructions add more VLIW overhead than the compiler's automatic spill management. The compiler's register allocator already produces reasonably efficient spill patterns for this kernel complexity. Adding explicit VMEM staging creates ADDITIONAL load/store instructions on top of the compiler's own spills, rather than replacing them.
- **Fix**: Do NOT attempt to manually manage register pressure by adding VMEM scratch buffers for intermediate staging. The Mosaic compiler's register allocator is hard to beat with explicit staging at high kernel complexity (~9400 VLIW bundles). Prefer approaches that ELIMINATE intermediates (algebraic simplification, input elimination) rather than STAGE them.
- **Extends**: FP29 (staged writes regress MXU util), R7 L2_reduce_intermediates (scoped recomputation counterproductive)
- **First seen**: 2026-03-31, chunk_gla optimization round 8

### [FP34] Register pressure reduction saturates at high optimization levels — spill reduction ≠ speedup
- **Symptom**: Five different register pressure reduction strategies in Round 8 ALL successfully reduced spills (by 5-26%) but the maximum speedup improvement was +0.2%. The variant with the LARGEST spill reduction (-25.8%, L2_fwd_store_A) actually REGRESSED speedup by -0.8%.
- **Root cause**: At ~9x speedup (2 fused pallas_calls, ~9400 VLIW bundles), register pressure is no longer the binding performance constraint. The remaining spills occur in locations where the compiler can overlap spill/fill traffic with MXU computation. Reducing spills doesn't free up useful pipeline slots because those slots are already utilized. The binding constraint has shifted to MXU pipeline utilization efficiency (~34% of single-MXU capacity) and single-MXU structural limitation (dual_ratio=0.0).
- **Evidence**: Round 8 results across 5 variants:
  - L2_fwd_store_A: spills -25.8%, speedup -0.8% (HBM traffic for A matrix offsets savings)
  - L2_store_gated_residuals: spills -10.9%, speedup -1.9% (residual storage HBM cost > spill savings)
  - L2_algebraic_simplify: spills -7.9%, speedup -0.8% (MXU util regressed from 34.6%→29.0%)
  - L2_eliminate_gcumsum: spills -5.2%, speedup +0.2% (only winner — minimal HBM traffic change)
  - L2_extra_scratch: spills -5.0%, speedup -2.7% (VMEM staging overhead)
- **Fix**: Stop targeting register pressure reduction as primary optimization direction when the kernel is at ~9x or higher. Future optimization should focus on: (1) MXU pipeline efficiency, (2) algorithmic restructuring to reduce total matmul count, (3) reducing total VLIW bundle count through simplification. Register pressure reduction only helps when it does NOT add HBM traffic or disrupt MXU scheduling.
- **First seen**: 2026-03-31, chunk_gla optimization round 8
- **Extended**: 2026-03-31, round 9 — Phase reordering (merge_A_dA) achieved -17.7% spills and -22.0% fills but REGRESSED MXU util -6.0pp (32.1%→26.1%), confirming that even dramatic spill reduction harms performance when it disrupts MXU scheduling

### [FP35] bf16 precision for accumulating matmuls causes correctness failure (extends FP31)
- **Symptom**: Selectively using bf16 operands with `lax.Precision.DEFAULT` for the dh update matmul (`q_hat.T @ do` in Phase 6 backward) and forward h update (`k.T @ v_gated`) produces max_diff=22.28 (atol=10.0).
- **Root cause**: The dh state accumulates across 64 time steps (T/chunk_size = 4096/64 = 64). Each time step's bf16 truncation error compounds through sequential state propagation. Unlike FP31 (global Precision.DEFAULT), this applied bf16 to ONLY the state-accumulating matmuls while keeping f32/HIGHEST for all others — but the accumulation makes even selective bf16 unsafe.
- **Fix**: Any matmul whose output feeds into cross-time-step state accumulation MUST use f32 operands with Precision.HIGHEST. bf16 is ONLY safe for matmuls producing point-in-time values consumed without accumulation (see SO17). The rule: accumulated → f32; snapshot → bf16 OK.
- **Extends**: FP31 (global Precision.DEFAULT fails) — this shows that selective bf16 for accumulating matmuls also fails.
- **First seen**: 2026-03-31, chunk_gla optimization round 9

### [FP36] VPU exp() optimization by manual CSE is negligible — compiler already optimizes
- **Symptom**: Reducing exp() calls from 5 to 3 per backward time step (by deriving exp_gn_minus from existing values and reusing exp_pos) produced only -0.2% VLIW reduction (-19 bundles) and -0.3% speedup regression. Vector EUP util dropped 0.07pp.
- **Root cause**: The Mosaic compiler performs common subexpression elimination (CSE) on exp() operations. Redundant exp() calls that could be derived from existing values are already optimized by the compiler. The hardware EUP unit is utilized at <1% — exp() is not a bottleneck. Manual exp() CSE provides negligible benefit and can slightly disrupt VLIW scheduling.
- **Fix**: Do NOT manually optimize exp() call count in Pallas kernels. The EUP utilization at <1% confirms exp() is not a performance-limiting factor. Focus on MXU pipeline efficiency and HBM bandwidth instead.
- **First seen**: 2026-03-31, chunk_gla optimization round 9

### SO17: bf16 h residual storage — halve HBM bandwidth for forward-to-backward state snapshot (9.054x)
- **Optimization**: Store the forward h state residual as bf16 instead of f32. Changed `h_ref` output ShapeDtypeStruct dtype from float32 to bfloat16. Forward kernel casts h state to bf16 before writing to `h_ref`. Backward already casts h to f32 on load via `.astype(jnp.float32)`.
- **Impact**: 9.005x → 9.054x (+0.5%). h residual array [B,NT,H,K,V] = [2,64,16,128,128] reduced from 128MB (f32) to 64MB (bf16). VLIW bundles -0.9% (9332→9252), spills -2.9% (2.58M→2.50M), fills -3.0% (3.30M→3.20M).
- **Why it works**: The h state at chunk boundaries is a point-in-time snapshot used for inter-chunk gradient contributions in the backward pass. It is consumed ONCE (not accumulated), so bf16 truncation errors do not compound. This is structurally different from the dh state update (FP35), which accumulates across 64 time steps.
- **Applicable when**: A forward-to-backward residual is: (1) large enough that HBM bandwidth is meaningful, (2) consumed without accumulation in the backward pass, (3) used as an approximation (not exact value). The residual's role must be "read once, approximate OK."
- **Rule**: Forward-to-backward residuals that are point-in-time snapshots consumed without accumulation can safely use bf16 storage. Residuals that feed into sequential state accumulation MUST stay f32 (see FP35).
- **First seen**: 2026-03-31, chunk_gla optimization round 9

### [FP37] Intermediate dtype casts are normalized by Mosaic compiler — removing them has zero effect
- **Symptom**: Removing `.astype(b_v.dtype)` intermediate bf16 casts in the forward kernel (when followed by f32 casts/matmuls) produces an IDENTICAL compiled kernel: same VLIW bundles (9252), MXU ops (5430), register spills (2,503,044), fills (3,198,000).
- **Root cause**: The Mosaic compiler performs dead-cast elimination. When a bf16 cast is immediately followed by an f32 cast or consumed by an f32 matmul, the compiler removes the redundant bf16 cast entirely. The compiled TPU kernel is identical regardless of whether intermediate casts are present in the source.
- **Fix**: Do NOT waste optimization effort on removing intermediate dtype casts in Pallas kernel source. The compiler handles this automatically. Focus on changes that affect the dataflow graph (different operations, different array dimensions) rather than cosmetic cast cleanup.
- **First seen**: 2026-03-31, chunk_gla optimization round 10 (L2_fwd_skip_v_gated_cast confirmed identical output)
- **What**: Removed the `a_ref` input from the fused backward kernel and eliminated the separate `chunk_gla_fwd_intra_gk` pallas_call that computed the A matrix. Instead, recompute A inside the backward kernel as `b_a = dot(q_pos, k_neg.T) * scale` using already-available q and k tiles.
- **Why**: The separate pallas_call for A computation had its own compilation, launch overhead, DMA transfers (A tiles from HBM), and register allocation. By recomputing A inside the backward kernel (just 1 extra dot product), all of that overhead is eliminated: 27 fewer computation events (-10%), 2 fewer DMA transfers (-8%), and no HBM round-trip for A.
- **Impact**: 0.876x → 1.097x (+25.2%). First variant to beat the JAX reference. Despite register spills INCREASING 24% (1.96M → 2.44M), the kernel launch overhead savings dominate.
- **Key insight**: **Kernel launch overhead > register pressure** for multi-kernel Pallas programs. Eliminating a separate pallas_call is more impactful than reducing register spills within the fused kernel.
- **Applicable when**: A Pallas program uses multiple pallas_call invocations and one kernel's output can be cheaply recomputed from inputs already available in another kernel.
- **First seen**: 2026-03-30, chunk_gla optimization round 2

### [FP38] Outer product gating to eliminate named intermediates INCREASES register spills
- **Symptom**: Factoring gating out of matmul operands via `dot(q, k.T) * outer(exp_g, exp_neg_g)` instead of `dot(q*exp_g, (k*exp_neg_g).T)` increased register spills from 20,420 to 37,840 (+85%) and fills from 21,440 to 40,870 (+91%). Speedup regressed from 1.155x to 1.139x.
- **Root cause**: The outer product `exp_g[:, None] * exp_neg_g[None, :]` creates a [BT, BT] = [64, 64] broadcast intermediate (16KB) that the compiler must materialize in registers. Despite eliminating the named [BT, BK] intermediates (b_qg, b_kg) from the source, the compiler generates MORE live values because: (1) the broadcast expansion creates a temporary [BT, BT] tensor, (2) the subsequent element-wise multiply with the [BT, BT] raw dot result requires both to be live simultaneously, (3) the raw dot(q, k.T) operands (q [BT, BK] and k [BT, BK]) must also be live through the multiply chain. The net effect is more simultaneous live values than the original approach.
- **Evidence**: 6 variants tested on chunk_fused_kernels (BT=64, BK=128, BV=128). ALL variants compiled to identical VLIW (8270 bundles) and MXU (4656 ops). Only register allocation differed. The 4 variants using outer product gating had 47-85% more spills than baseline, while the 2 variants not using it had only 14% more spills.
- **Fix**: Do NOT factor element-wise gating out of matmul operands to "reduce named intermediates." The pre-gated operand approach `dot(q*scale, (k*scale).T)` is more register-efficient because the scaling is absorbed into the operand before the matmul, never creating a separate broadcast tensor. Source-level intermediate reduction does not guarantee compiled-level register pressure reduction.
- **Extends**: FP37 (compiler normalizes source-level changes), FP34 (register pressure optimization saturates)
- **First seen**: 2026-04-01, chunk_fused_kernels optimization round 1

### [FP39] Source-level dataflow restructuring produces identical VLIW/MXU on maximally-fused kernels
- **Symptom**: 6 different source-level optimizations (CSE, outer product gating, deferred materialization, VMEM scratch separation, intermediate elimination, complete gating factorization) ALL compiled to identical VLIW bundle count (8,270) and MXU ops (4,656 on mxu0, 0 on mxu1, dual_ratio=0.0). Only register allocation (spill/fill counts) differed.
- **Root cause**: For maximally-fused Pallas kernels (single pallas_call per pass) with fixed matmul dimensions (BT=64, BK=128, BV=128) and fixed algorithm structure (12 dot products per fwd+bwd time step), the Mosaic compiler's VLIW scheduling and MXU placement are determined entirely by the matmul dependency graph, not by how intermediates are expressed in the source. The compiler reconstructs the canonical schedule regardless of source-level intermediate management.
- **Fix**: For kernels at this complexity level (~8K VLIW bundles, ~4.6K MXU ops, 2 fused pallas_calls), do NOT attempt source-level intermediate management optimizations. The only effective approaches are: (1) reducing total matmul count (algorithmic), (2) changing matmul dimensions (block sizes), (3) changing pallas_call architecture (grid, dimension_semantics), (4) changing the algorithm itself.
- **Extends**: FP18 (source reordering), FP37 (dtype cast normalization)
- **Ultimate confirmation (Round 5)**: Extended across ALL source-level variations: Python for-loop unrolling (`for s in range(UNROLL)`), manual step-by-step unrolling, scratch buffer presence/absence (o_scratch_ref), shape-adaptive UNROLL=NT, forward-only vs both-pass unrolling, eager vs deferred b_h load — ALL compile to identical 8270 VLIW / 4656 MXU / 36100/35720 fills/spills at 4-step grid. Two completely independent lineages (L1 and L2) with different evolutionary histories produce identical compiled kernels when grid structure matches.
- **First seen**: 2026-04-01, chunk_fused_kernels optimization round 1
- **Extended**: 2026-04-01, round 5 — definitive convergence across 25+ variants and 5 rounds
- **Extended**: 2026-04-02, fused_chunk_simple_gla round 2 — Sub-tiling attention matrix [64,64] into three [32,32] sub-tiles produces identical latency (13.44ms) and spills (310K) despite +77% VLIW bundles (1652→2920) and +73% MXU ops (160→276). Compiler reconstructs the full computation from sub-tiles. Extends FP42 (K-split normalization) to M/N-split normalization.

### [FP40] bf16+DEFAULT for non-accumulating matmuls still exceeds atol=1.0 in gradient chains
- **Symptom**: Casting 4 non-accumulating matmul inputs to bf16 with `Precision.DEFAULT` (A recomputation, dA_raw, dv_intra, dA_T) produced max_diff=1.294677734375, exceeding atol=1.0. Two independent implementations (L1 and L2 lineages) produced identical max_diff, confirming it's systematic.
- **Root cause**: Even "non-accumulating" matmuls propagate truncation error through the backward gradient chain. The bf16 A recomputation (dot(qg, kg.T) in bf16) introduces error in the attention weights, which flows into dA_raw → dv_intra → final gradients. Each bf16 truncation compounds, and 4 cascaded truncations push the accumulated error past atol=1.0. The hypothesis that "snapshot" (non-accumulated) matmuls can tolerate bf16 is wrong for this kernel's error budget.
- **Fix**: Do NOT use bf16+DEFAULT for any matmuls in the gradient chain, even non-accumulating ones. The only safe precision reduction would be individual matmuls whose output is not used as input to another matmul, but in this kernel all matmuls are chained. Extends FP31 (Precision.DEFAULT causes correctness failure) and FP35 (bf16 for accumulating matmuls).
- **First seen**: 2026-04-01, chunk_fused_kernels optimization round 2

### [FP41] Structural changes to pallas_call I/O (adding/removing Refs) produce identical compiled kernels
- **Symptom**: Adding A_masked as a forward-to-backward residual (extra output Ref in forward pallas_call, extra input Ref in backward pallas_call) to eliminate 1 backward matmul recomputation produced IDENTICAL compiled code: 8270 VLIW bundles, 4656 MXU ops, 24455/23435 fills/spills, 24 DMA — all exactly matching the baseline and all Round 1 variants.
- **Root cause**: The Mosaic compiler's VLIW/MXU schedule is determined by the matmul dependency graph within the kernel body, not by the I/O structure (number of Refs, residual management). Adding a Ref to pass A_masked from forward to backward doesn't change the backward's internal computation graph — the compiler still generates the same code whether A is recomputed or loaded from a Ref, because both paths produce the same data dependencies.
- **Fix**: Do NOT add/remove pallas_call Ref arguments as an optimization strategy. The compiler normalizes I/O changes to the same VLIW schedule. This extends FP39 from "source-level dataflow" to "pallas_call I/O structure."
- **Extends**: FP39 (source-level restructuring produces identical VLIW/MXU)
- **First seen**: 2026-04-01, chunk_fused_kernels optimization round 2

### [FP42] Splitting matmul dimensions (K-split) is normalized by Mosaic compiler to identical VLIW/MXU
- **Symptom**: Splitting ALL 13 matmuls from [64,128]@[128,64] into two [64,64]@[64,64] sub-matmuls (K=128 → 2x K/2=64) produced IDENTICAL compiled kernel: 8270 VLIW bundles, 4656 mxu0 ops, dual_ratio=0.0 — same as baseline. Register spills decreased 34% (16080/16065 vs 24455/23435) but speedup was 0.997x (no improvement).
- **Root cause**: The Mosaic compiler recognizes that two consecutive partial-reduction matmuls on the same operands can be merged into a single larger matmul. The compiler reconstructs the original [64,128]@[128,64] operation from the two [64,64]@[64,64] halves. This is an extreme extension of FP39: the compiler normalizes not just intermediate management but also matmul dimension changes.
- **Fix**: Do NOT split matmul dimensions to try to change the compiled VLIW/MXU schedule. The compiler will merge them back. The only way to change the compiled schedule is to change the GRID structure (number of iterations, pallas_call count), not the operation-level structure within a single kernel body.
- **Extends**: FP39 (source-level restructuring), FP41 (I/O structure changes)
- **First seen**: 2026-04-01, chunk_fused_kernels optimization round 3

### SO18: Grid iteration reduction via multi-step unrolling (1.223x — +5.6% breakthrough)
- **Optimization**: Process 2 time steps per grid iteration instead of 1. Changed grid from (B, H, 1, 1, NT) to (B, H, 1, 1, NT//2). Input BlockSpecs load 2*BT rows per iteration. Kernel body manually processes sub-step 0 (rows [0, BT)) then sub-step 1 (rows [BT, 2*BT)) with h state updated between them. h output BlockSpec covers 2 NT slots per iteration.
- **Impact**: 1.158x → 1.223x (+5.6%). MXU util improved +4.9pp (22.8% → 27.7%). Latency: 0.0542ms → 0.0508ms.
- **Why it works**: The compiled kernel body is IDENTICAL (8270 VLIW bundles, 4656 MXU ops, dual_ratio=0.0) — but halving grid iterations reduces pipeline flush/fill overhead between iterations. At this kernel scale (0.05ms total), inter-iteration overhead is ~6% of total runtime. The speedup comes entirely from amortizing grid overhead, not from better kernel body compilation.
- **Trade-off**: Register spills increased +27-38% (30975/32435 vs 24455/23435) due to double the work per iteration. Despite more spills, the reduced iteration overhead dominates.
- **Applicable when**: A Pallas kernel with "arbitrary" time dimension has small per-iteration compute relative to grid overhead. Multi-step unrolling amortizes overhead. Requires NT % unroll_factor == 0.
- **Next step**: Try 4-step unrolling (NT/4 iterations). For T=256/BT=64, NT=4, so 4-step = 1 iteration total (eliminate ALL grid overhead).
- **First seen**: 2026-04-01, chunk_fused_kernels optimization round 3

### SO19: 4-step grid unrolling — maximum grid reduction (1.321x — +8.0% over previous best)
- **Optimization**: Extend SO18 from 2-step to 4-step unrolling. Grid changes from (B, H, 1, 1, NT//2) to (B, H, 1, 1, NT//4). For T=256/BT=64, NT=4 → NT//4=1, completely eliminating ALL grid iteration overhead. BlockSpecs load 4*BT rows. h output covers 4 NT slots.
- **Impact**: 1.223x → 1.321x (+8.0%). MXU util improved +6.2pp (27.7% → 33.9%). Latency: 0.0508ms → 0.0471ms.
- **Why it works**: Same mechanism as SO18 but pushed to the maximum. For T=256, zero grid iterations remain — the entire time dimension processes in a single kernel invocation. For T=512 (NT=8), 2 iterations remain (vs 4 at 2-step). The compiled kernel body is STILL identical (8270 VLIW, 4656 MXU, dual_ratio=0.0).
- **Critical finding — forward is the bottleneck**: Asymmetric testing proved that forward 4-step alone (backward 2-step) captures ~100% of the benefit (1.317x), while backward 4-step alone (forward 2-step) adds ~0% (1.225x vs 1.223x). The forward pass has 4 matmuls/sub-step making grid overhead a larger fraction, while the backward's 9 matmuls/sub-step already amortizes overhead well.
- **Critical finding — FP39 extends to grid-level equivalence**: L1 and L2 lineages (with completely different source-level kernel code — different VMEM scratch configs, different operation ordering) produce IDENTICAL compiled metrics at the same grid structure. All 4-step variants: 8270 VLIW, 4656 MXU, 36100/35720 fills/spills.
- **Trade-off**: Register spills increased further (36100/35720 vs 30975/32435 at 2-step), but still not the binding constraint.
- **Applicable when**: Extending SO18. Use the maximum unroll factor that evenly divides NT for all target shapes. For shapes with NT=4, 4-step is the maximum. For NT=8, up to 8-step is possible.
- **First seen**: 2026-04-01, chunk_fused_kernels optimization round 4

### [FP44] Pallas kernel function must not capture Python constants — pass via functools.partial or Ref
- **Symptom**: `ValueError: The kernel function ... captures constants [f32[]]. You should pass them as inputs.`
- **Root cause**: Pallas (Mosaic) kernel functions cannot close over Python-scope variables. A scalar like `scale` used inside the kernel body as a closure variable is detected as a captured constant and rejected. This applies to ALL captured values: scalars, arrays, and JAX tracers.
- **Fix**: Pass constants as compile-time parameters via `functools.partial(kernel_fn, scale=scale)` (for compile-time constants like scalars) or as explicit `Ref` inputs via BlockSpec (for runtime values). Never reference outer-scope variables inside Pallas kernel functions.
- **First seen**: 2026-04-01, fused_chunk_simple_gla optimization round 1 (simplify_fwd_grid variant)

### [FP45] Pallas_call count reduction and HBM bandwidth optimization have zero impact when register pressure dominates
- **Symptom**: Three successful variants — eliminate_h_precompute (3→2 pallas_calls), bf16_h_residual (halved h HBM bandwidth), bwd_2pass_no_h_hbm (eliminated h from HBM entirely) — ALL showed exactly 1.000x speedup (13.451ms = baseline).
- **Root cause**: At severe register pressure (310K fills/spills, 5.6% MXU utilization), the kernel body compute time dominates total execution time. Pallas_call launch overhead and HBM bandwidth for the h tensor are negligible fractions of the ~13.45ms total. The binding constraint is per-chunk compute (12+ matmuls per chunk, 64 chunks per sequence) within the fwd and bwd kernel bodies.
- **Fix**: When register pressure is the dominant bottleneck (>100K spills, <10% MXU util), do NOT pursue pallas_call count reduction or HBM bandwidth optimization. Focus exclusively on reducing register pressure within kernel bodies: smaller block sizes, fewer live intermediates, simpler kernel body logic.
- **Relationship**: Contrasts with SO14 (lax.scan→pallas_call) where launch overhead WAS the bottleneck. The difference: SO14's lax.scan had ~59ms PER-ITERATION dispatch overhead (host-device round-trip), while pallas_call grid iterations have negligible overhead when the kernel body itself is massive.
- **First seen**: 2026-04-01, fused_chunk_simple_gla optimization round 1
- **Extended**: 2026-04-02, fused_chunk_simple_gla session 2 round 1 — bwd_2pass_split (split backward into dv+dh and dq+dk passes for per-kernel register pressure reduction) also showed 1.000x. Splitting does NOT reduce aggregate spills (still 310K total across passes). CSE/intermediate reduction (reduce_bwd_intermediates) produced identical VLIW/MXU (FP39), and fwd_bwd_fusion (save h from fwd kernel) increased peak memory 50% without speedup. Grid unrolling (SO20 approach) crashed from implementation bugs — should be retried with correct indexing.

### SO20: Grid unrolling on high-iteration-count kernels (1.388x on fused_chunk_simple_gla)
- **Optimization**: N-step grid unrolling on forward kernel, reducing grid iterations by factor N. 2-step: 64→32 iterations. 4-step: 64→16 iterations. Extends SO18/SO19 to a much larger kernel (~13ms total execution vs 0.05ms).
- **Impact**:
  - 2-step: 1.000x → 1.222x (+22.2%). Latency: 13.45ms → 11.01ms.
  - 4-step: 1.222x → 1.388x (+16.6pp). Latency: 11.01ms → 9.69ms. Forward iterations 32→16.
- **Why it works**: Grid iteration overhead is a significant fraction of total runtime. Each doubling of the unroll factor halves forward iterations. Despite register spills increasing (310K → 4.3M at 2-step → 6.3M at 4-step), the overhead reduction dominates.
- **Key finding — spill tolerance scales**: At 2-step, spills increased 14x (310K → 4.3M). At 4-step, spills increased 20x (310K → 6.3M). Yet each step produced significant speedup, proving the spill-vs-overhead tradeoff strongly favors more aggressive unrolling.
- **Key finding — backward simplicity enables forward scaling**: L3 (simple single-step backward) achieves 1.388x with 4-step forward, while L2 (2-step backward) achieves only 1.222x with IDENTICAL 4-step forward code (same VLIW=5275, same MXU=640). At 2-step forward, backward complexity had zero impact (FP43). At 4-step forward, backward complexity INTERFERES. Simpler backward enables better total compilation.
- **Forward-only = combined at 2-step**: Round 3 confirmed that forward-only 2-step (1.221x) achieves the SAME speedup as combined fwd+bwd 2-step (1.222x). Consistent with FP43 — backward unrolling adds negligible benefit.
- **Extends**: SO18 (+5.6% at 2-step), SO19 (+8.0% at 4-step) — same mechanism, proven on a much larger kernel scale
- **First seen**: 2026-04-01, fused_chunk_simple_gla optimization round 2
- **Updated**: 2026-04-02, round 3 — forward-only achieves same speedup as combined fwd+bwd
- **Updated**: 2026-04-02, round 4 — 4-step forward achieves 1.388x; backward simplicity is critical at >=4-step
- **Updated**: 2026-04-02, round 5 — L1_fwd_4step (split backward: separate dv+dh and dq+dk passes) achieves 1.388x, IDENTICAL to L3_fwd_4step (simple two-phase backward). Compiles to same forward kernel (`_fused_chunk_fwd_4step_kernel`, iteration_bounds=[10,16,1,1,16], VLIW=5275, MXU=640). Proves 4-step technique transfer is fully architecture-independent: backward architecture choice (split vs simple vs combined) does NOT affect forward unrolling benefit. FP48 interference is about backward COMPLEXITY (unrolling), not backward architecture.
- **Updated**: 2026-04-02, round 6 — **4-step is the CEILING**. 8-step manual unrolling REGRESSES to 1.314x (-7.4pp from 1.388x). Two genuine measurements confirm (L1_fwd_8step=10.228ms, L3_fwd_8step_pyloop=10.229ms). Code bloat (VLIW 5275→10403) exceeds grid reduction benefit (16→8 iterations). See FP50. Grid unrolling optimization direction is exhausted.

### [FP43] Backward grid unrolling has negligible impact on compute-heavy backward passes
- **Symptom**: L1_bwd_only_four_step (forward=2-step, backward=4-step) achieved only 1.225x — a mere +0.16% over L1's 1.223x (forward=2-step, backward=2-step). Meanwhile L1_fwd_only_four_step (forward=4-step, backward=2-step) achieved 1.317x (+7.7%).
- **Root cause**: The backward pass has 9 matmuls per sub-step (2 for A recompute + 1 for dA + 2 for dv + 2 for dq/dk inter + 2 for dq/dk intra), making each sub-step compute-heavy. Grid iteration overhead is a negligible fraction of backward time. The forward pass has only 4 matmuls per sub-step, making grid overhead a larger fraction.
- **Fix**: When applying grid unrolling to asymmetric kernels (forward lighter than backward), prioritize forward unrolling. Backward unrolling adds code complexity and register pressure for negligible benefit.
- **Extended evidence (R4)**: L2_bwd_4step (most aggressive backward unrolling tested: 4-step backward with 2-step forward) achieved only 1.222x — identical to L2's previous best with 2-step backward. VLIW bloat was massive (9544 bundles, 2x the 4825 at 2-step) with zero speedup gain. Additionally, L3_add_bwd_2step (adding 2-step backward to L3's forward-only) achieved 1.222x — exactly matching the forward-only result. Backward unrolling is confirmed dead-end across 4 independent tests.
- **Compounding effect at >=4-step forward**: At 4-step forward, backward complexity becomes actively harmful (see SO20 update). L2 (2-step backward) achieves only 1.222x while L3 (no backward unrolling) achieves 1.388x with identical forward code.
- **First seen**: 2026-04-01, chunk_fused_kernels optimization round 4
- **Extended**: 2026-04-02, fused_chunk_simple_gla round 4 — backward 4-step and backward 2-step both confirmed zero benefit

### [FP46] BlockSpec block_shape dimensionality must exactly match out_shape for grid unrolling h_buf
- **Symptom**: `ValueError: Block shape for outputs[4] (= (1, 1, 2, 128, 128)) must have the same number of dimensions as the array shape (10, 16, 32, 2, 128, 128)` — h_buf BlockSpec has 5 dimensions but out_shape creates a 6D array.
- **Root cause**: When grid unrolling creates a multi-step h_buf with shape (B, H, NT_GROUPS, STEPS_PER_GROUP, K, V) = 6 dimensions, the BlockSpec block_shape must also be 6D. Using (1, 1, STEPS, K, V) = 5D misses the NT_GROUPS dimension. The index_map must return 6 values correspondingly: (b, h, group_idx, 0, 0, 0).
- **Fix**: For N-step grid unrolling h_buf: block_shape = (1, 1, 1, N, K, V) with 6D index_map returning (b, h, block_idx, 0, 0, 0). Count the dimensions of the out_shape array and ensure block_shape has the same count.
- **Extends**: FP17 (BlockSpec dimensions must match array dimensions). Same principle but specific to output buffers created via grid unrolling with multi-step structure.
- **First seen**: 2026-04-02, fused_chunk_simple_gla optimization round 2 (3 variants: L2_grid_unroll_2step, L2_grid_unroll_4step, L2_fwd_bwd_unroll)

### [FP47] Grid unrolling h_buf Ref indexing must use correct scalar index count for 6D blocks
- **Symptom**: `ValueError: Invalid shape for 'swap'. Ref shape: (1, 1, 1, 2, 128, 128). Expected shape: (2, 128, 128). Value shape: (128, 128).` — kernel body writes (K,V)=(128,128) value to h_buf_ref but indexing selects (STEPS,K,V)=(2,128,128) slice.
- **Root cause**: After fixing FP46 (BlockSpec dimensionality 5D→6D), the kernel body code still uses the old 3-scalar-index pattern `h_buf_ref[0, 0, 0]` which selects batch/head/group dims, leaving (STEPS, K, V) = (2, 128, 128). To get a (K, V) slice, must use 4 scalar indices: `h_buf_ref[0, 0, 0, step_idx]`.
- **Fix**: When adding a STEPS dimension to h_buf for grid unrolling, update ALL kernel body `h_buf_ref` read/write accesses to include the step index as a 4th scalar dimension. For writes: `h_buf_ref[0, 0, 0, step] = value` (not `h_buf_ref[0, 0, 0] = value`). For reads: `h_buf_ref[0, 0, 0, step]` (not `h_buf_ref[0, 0, step]`).
- **Distinction from FP46**: FP46 is about the BlockSpec/index_map declaration at the pallas_call level. FP47 is about Ref indexing within the kernel body function. Both must be updated when adding a dimension — fixing one without the other causes different errors.
- **First seen**: 2026-04-02, fused_chunk_simple_gla optimization round 3 (2 variants: L2_grid_2step_fix, L2_grid_4step_fix)

### [FP48] Backward kernel complexity interferes with forward unrolling at >=4-step
- **Symptom**: L2_fwd_4step (4-step forward + 2-step backward) achieves only 1.222x, while L3_fwd_4step (4-step forward + no backward unrolling) achieves 1.388x. Both have IDENTICAL compiled forward kernels (VLIW=5275, MXU=640).
- **Root cause**: At 4-step forward unrolling, the total kernel compilation includes both forward and backward kernels. The backward kernel's complexity (2-step unrolled backward with VLIW=4825 vs original single-step backward) affects the TOTAL execution path. At 2-step forward, backward complexity had zero impact (FP43) — the interaction only appears at higher forward unrolling levels. The mechanism is likely: (1) complex backward increases total register pressure across the fused computation, (2) XLA/Mosaic makes different scheduling decisions for the total program when backward is complex, (3) DMA/memory scheduling for the full forward+backward pipeline is affected by backward complexity.
- **Fix**: When pushing forward grid unrolling beyond 2-step, keep backward kernel as SIMPLE as possible (no backward unrolling). Use the simplest backward code path as the base for aggressive forward unrolling. This is the opposite of intuition — more backward unrolling is actively harmful at high forward unrolling.
- **Evidence**: L3 base (forward-only 2-step, no backward unrolling) → 1.388x at 4-step forward. L2 base (forward+backward 2-step) → 1.222x at 4-step forward. Same compiled forward kernel, different total latency (9.687ms vs 11.006ms).
- **First seen**: 2026-04-02, fused_chunk_simple_gla optimization round 4

### [FP49] fori_loop inside Pallas forward kernels regresses performance vs manual grid unrolling
- **Symptom**: `jax.lax.fori_loop(0, N, body, ...)` for N-step forward grid unrolling achieves only 1.19-1.21x, while manual N-step unrolling achieves 1.388x. Three variants tested: fori_loop 4-step (1.194x), fori_loop 8-step (1.194x), fori_loop 16-step (1.212x) — all significantly worse than manual 4-step (1.388x).
- **Root cause**: fori_loop compiles to a compact runtime loop rather than straight-line code. The profiler captured the backward kernel as the larger module (VLIW=2663 for backward vs smaller forward), confirming the fori_loop forward kernel is much smaller than manual unrolling (VLIW=5275). While this dramatically reduces register pressure (500K vs 6.3M spills), it prevents the Mosaic compiler from optimizing ACROSS sub-steps. Manual unrolling exposes all sub-step computations (matmuls, accumulations, gating) as a single straight-line body, enabling inter-step instruction scheduling, operand reuse, and pipeline optimization that fori_loop cannot access. The inter-step optimization benefit vastly exceeds the register pressure cost.
- **Fix**: Use manual unrolling for forward grid iteration reduction, NOT `jax.lax.fori_loop`. The FP23 suggestion to use fori_loop to "hide iterations from the compiler" is counterproductive for grid unrolling — the compiler NEEDS visibility into all sub-steps to optimize effectively. fori_loop is only appropriate when the loop body has no cross-iteration optimization opportunity (e.g., independent iterations).
- **Extends**: FP30 (fori_loop + lax.cond bad). **Contradicts FP23** (which suggested fori_loop for register pressure — this is wrong for forward grid unrolling where inter-step optimization dominates).
- **First seen**: 2026-04-02, fused_chunk_simple_gla optimization round 5

### [FP50] 8-step forward grid unrolling regresses from 4-step — 4-step is the ceiling
- **Symptom**: Manual 8-step forward grid unrolling achieves 1.314x, a -7.4pp regression from 4-step's 1.388x. Latency increases from 9.687ms to 10.228ms despite halving grid iterations (16→8). Two independent genuine measurements confirm: L1_fwd_8step (1.314x, first in batch) and L3_fwd_8step_pyloop (1.315x, unique timing tier).
- **Root cause**: At 8-step, code complexity doubles (VLIW 5,275→10,403, LLO 10K→21K lines, DMA 464→912) but grid iterations only halve (16→8). The code bloat penalty exceeds the grid reduction benefit. Paradoxically, register spills are LOWER at 8-step (5.6M vs 6.3M at 4-step), suggesting the compiler reorganizes allocation with larger windows — but MXU utilization drops dramatically (0.44%→0.16%), indicating the doubled code complexity prevents efficient MXU scheduling. 16-step (VLIW=20,659, 41K LLO lines) has leaked profiling but is almost certainly worse.
- **Fix**: Forward grid unrolling MUST stop at 4-step for this kernel. The diminishing returns curve (22.2pp at 2-step, 16.6pp at 4-step, -7.4pp at 8-step) has definitively crossed the break-even point. Do NOT try 8-step, 16-step, or higher forward unrolling.
- **Extends**: SO20 (grid unrolling progression now has upper bound), SO18/SO19 (4-step remains optimal)
- **First seen**: 2026-04-02, fused_chunk_simple_gla optimization round 6

### SO21: Python `for step in range(N)` compiles identically to manual unrolling in Pallas
- **Optimization**: Replace N manually copy-pasted sub-step blocks in Pallas kernel body with `for step in range(N): body(step)` Python loop. Tested at 4-step and 8-step levels.
- **Impact**: L3_fwd_4step_pyloop achieves identical metrics to manual L3_fwd_4step: VLIW=5,275, MXU=640, fills=6,762,360, spills=6,326,905, latency=9.687ms. LLO line count identical (10,457). L3_fwd_8step_pyloop also matches manual L3_fwd_8step (VLIW=10,403, MXU=1,280). Both confirmed at sub-millisecond timing precision (stddev <0.001ms).
- **Why it works**: Python `for` loops are unrolled at JAX trace time — JAX traces through each iteration, producing the same flat dataflow graph as manual copy-paste. This is fundamentally different from `jax.lax.fori_loop` (FP49), which compiles to a runtime loop that hides iterations from the compiler.
- **Applicable when**: Any Pallas kernel using manual copy-paste for grid unrolling sub-steps. Replace with Python `for step in range(N)` for cleaner, more maintainable code with zero performance cost. This enables higher unrolling levels (8-step, 16-step) without code maintenance burden.
- **Contradicts**: The implicit assumption that manual copy-paste might compile differently than Python loops. It does not — JAX trace-time unrolling produces identical compiled output.
- **First seen**: 2026-04-02, fused_chunk_simple_gla optimization round 6
