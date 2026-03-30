## Round 1 Strategy

Generating 5 variants from baseline, each building on SO8 (1.974x) and exploring different quantization reduction approaches.
Variants generated in parallel via sub-agents.

Profile diagnosis: 160K register spills, 6.68% compute efficiency, scalar ALU 33x higher than MXU — quantization overhead dominates.

### Variant: zero_bwd_quant
**Technical direction**: Reproduce SO8 (proven baseline)
**Profile motivation**: 160K register spills from quantization overhead; SO8 proved eliminating backward quantization gives 1.974x
**Approach**: Zero backward quantization + optimized tiling (1024,256,128, 1024,256,128, 2048,256,128). Forward keeps 3 fp8 quantize calls (lhs, lhs_t, rhs). Backward passes bf16 directly.
**Expected impact**: ~1.974x (reproduce known best)
**Target metric improvement**: spills 160K → ~1.8K, VLIW bundles 4065 → ~17.5K, compute_efficiency 6.68% → ~13.5%
**Key changes**: Remove backward quantization, store lhs_bf16 in residuals, _clamp_tiling helper

### Variant: skip_lhs_t_fwd
**Technical direction**: Eliminate redundant forward lhs_t quantization
**Profile motivation**: Forward lhs_t quantization produces fp8 tensor never consumed (SO8 uses bf16 lhs_t in backward)
**Approach**: SO8 + remove forward qpl.quantize for lhs_t. Only 2 forward quantize calls (lhs, rhs).
**Expected impact**: >2.0x — fewer VLIW bundles and register pressure than SO8
**Target metric improvement**: VLIW bundles < 17,558, register spills < 1,822
**Key changes**: Remove lhs_t quantization from _gmm_fwd, compute swapaxes in backward

### Variant: fwd_mixed_precision
**Technical direction**: Forward mixed precision (bf16 lhs + fp8 rhs)
**Profile motivation**: SO7 proved mixed precision works for backward gmm; apply to forward too
**Approach**: SO8 + skip both lhs AND lhs_t forward quantization. Only rhs quantized. Forward gmm uses bf16 lhs + fp8 rhs.
**Expected impact**: >2.0x — eliminates 4 of 5 total quantize calls
**Target metric improvement**: Major VLIW and spill reduction from eliminating most quantization
**Key changes**: Forward gmm mixed precision, only rhs quantized in entire kernel

### Variant: tiling_phase_specialization
**Technical direction**: Shape-aware per-phase tiling
**Profile motivation**: Gate/Up and Down projections have asymmetric dimensions; uniform tiling may not be optimal
**Approach**: SO8 base + different tiling per phase: fwd (256,256,128), bwd_gmm (1024,256,128), tgmm (2048,512,128)
**Expected impact**: Explore whether per-phase tiling improves over uniform (1024,256,128)
**Target metric improvement**: Better tile utilization, potentially fewer grid iterations
**Key changes**: TM=256 for forward (matching per-group M), TK=512 for tgmm

### Variant: reduce_fwd_quant
**Technical direction**: Minimal quantization (only rhs weights)
**Profile motivation**: Maximum quantization reduction — only 1 quantize call in entire kernel
**Approach**: Only rhs quantized to fp8. All activations (lhs, grad) stay bf16 throughout. Forward uses mixed precision gmm.
**Expected impact**: >2.0x — 5 quantize calls reduced to 1
**Target metric improvement**: Dramatic VLIW and spill reduction
**Key changes**: Single qpl.quantize call for rhs weights only
