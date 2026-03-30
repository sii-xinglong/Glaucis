## Round 1 Batch Analysis

**Variants evaluated**: 5
**Successes**: 3 | **Failures**: 2
**Best speedup this round**: 2.294x (tiling_phase_specialization)
**Overall best speedup**: 2.294x (new — exceeds prior session best of 1.974x)

### Comparative Ranking

| Rank | Variant | Status | Speedup | Latency (ms) | VLIW Bundles | MXU Ops | Spills | Direction |
|------|---------|--------|---------|-------------|-------------|---------|--------|-----------|
| 1 | tiling_phase_specialization | SUCCESS | 2.294x | 4.692ms | 23,462 | 1,792 | 156 | tiling_specialization |
| 2 | skip_lhs_t_fwd | SUCCESS | 1.944x | 5.294ms | 17,558 | 896 | 156 | quant_reduction |
| 3 | zero_bwd_quant | SUCCESS | 1.266x | 8.356ms | 17,558 | 896 | 156 | quant_reduction |
| -- | fwd_mixed_precision | INCORRECT | -- | -- | -- | -- | -- | fwd_mixed_precision |
| -- | reduce_fwd_quant | INCORRECT | -- | -- | -- | -- | -- | minimal_quant |

### Per-Variant Details

#### tiling_phase_specialization (Rank 1) — NEW BEST

**Status**: SUCCESS
**Speedup**: 2.294x (4.692ms vs 10.762ms reference)

| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | 23,462 | Higher than skip_lhs_t_fwd (17,558) but faster — more MXU work per bundle |
| MXU ops | 1,792 (dual_ratio=1.0) | 2x more MXU ops than skip_lhs_t_fwd — double the matmul throughput |
| MXU dual ratio | 1.0 | Perfect — both MXUs equally loaded |
| Compute efficiency | 15.24% | Best of all variants (+13% over skip_lhs_t_fwd's 13.50%) |
| DMA transfers | 21 (double-buffered) | Same as other variants |
| Register fills/spills | 156/156 | Negligible — down from 160K baseline |
| HBM bandwidth | N/A | Not measured |

**Bottleneck**: Compute-bound with room for improvement. compute_efficiency=15.24% means 84.76% of peak TFLOPS is still unused.

**Key insight**: Tiling (256, 256, 128) for forward + (2048, 512, 128) for tgmm produces 2x more MXU ops than (1024, 256, 128) uniform tiling. The smaller forward tiles (TM=256 matching per-group M exactly) create more grid iterations but each has cleaner MXU scheduling. The larger tgmm TK=512 halves K-loop iterations.

**LLO observations**:
- 1,792 MXU ops (896 mxu0 + 896 mxu1) — exactly 2x more than SO8's 896
- bf16 matmul ops (vmatpush1.bf16) on both MXU ports
- 9,075 scalar spill/fill ops (smem, not VMEM vector spills)
- 0 NOPs — no pipeline bubbles
- Clean MXU scheduling with both ports active

#### skip_lhs_t_fwd (Rank 2)

**Status**: SUCCESS
**Speedup**: 1.944x (5.294ms vs 10.292ms reference)

| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | 17,558 | Same as SO8 — removing lhs_t quant doesn't change VLIW |
| MXU ops | 896 (dual_ratio=1.0) | Same as SO8 |
| Compute efficiency | 13.50% | Same as SO8 |
| Register fills/spills | 156/156 | Same as SO8 |
| HBM bandwidth | 134,217,728 bytes | 0.69% of peak |
| Arithmetic intensity | 12,288 FLOPs/byte | Very high — compute-bound |

**Bottleneck**: Identical compilation to SO8. Per F001, removing lhs_t quantization from forward doesn't change compiled output because XLA traces through the custom_vjp and optimizes holistically — the unused lhs_t quantization was already eliminated by the compiler.

#### zero_bwd_quant (Rank 3) — REGRESSION

**Status**: SUCCESS
**Speedup**: 1.266x (8.356ms vs 10.579ms reference) — significantly below SO8's 1.974x

**Root cause**: Sub-agent's `_clamp_tiling(use_fp8=True)` does `min(t, tile_size=128)` for the forward pass, capping the (1024, 256, 128) tiling to (128, 128, 128). This is the same behavior as the baseline's `min(t, tile_size)` clamping that SO8 specifically removed. The kernel compiled with the same small tiles as baseline but without backward quantization — hence only marginal improvement.

**Fix**: Replace `_clamp_tiling(use_fp8=True)` with a simple `max(t, 128)` clamp (lower bound only, no upper cap).

### Failed Variants Summary

| Variant | Error | max_diff | Cause | Fix |
|---------|-------|----------|-------|-----|
| fwd_mixed_precision | INCORRECT | 120,649 | tokamax forward gmm rejects bf16 lhs + fp8 rhs mixed precision | Forward GMM requires both operands quantized to fp8. Mixed precision only works for backward gmm (SO7). |
| reduce_fwd_quant | INCORRECT | 120,649 | Same as above — bf16 lhs + fp8 rhs in forward | Same fix — forward lhs MUST be fp8 quantized |

**New failure pattern**: Forward GMM mixed precision (bf16 lhs + fp8 rhs) produces catastrophically wrong output (max_diff=120,649). This is because tokamax's forward `gmm()` requires matching precision for both operands when using FP8. SO7's mixed precision only worked in backward because `transpose_rhs=True` changes the internal computation path. **Forward lhs quantization MUST be preserved.**

### IR Analysis (tiling_phase_specialization)

**LLO**: 24,870 lines, 23,462 VLIW bundles. Contains bf16 matmul operations (vmatpush1.bf16) on both MXU ports. 9,075 scalar register spills (to smem, not VMEM) but only 156 vector spills — register pressure is well-controlled. The 2x MXU op count (1,792 vs 896) comes from the different tiling creating more matmul tiles with the (256, 256, 128) forward tiling.

**HLO**: Only shows the reference `abs()` computation — Pallas custom calls are opaque in HLO. 0 fusions (ideal).

**Key finding**: The tiling (256, 256, 128) for forward doubles MXU ops because TM=256 (matching per-group M=256) creates more tile iterations than TM=1024, and each tile does a full 256x256 matmul. Despite more VLIW bundles (23,462 vs 17,558), the kernel is faster because MXU throughput doubled.
