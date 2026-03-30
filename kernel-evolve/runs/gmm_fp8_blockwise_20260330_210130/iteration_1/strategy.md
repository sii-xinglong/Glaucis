## Round 1 Strategy

Generating 5 variants from baseline, each exploring a different technical direction.
Variants generated in parallel via sub-agents.

### Variant: tiling_strategy
**Technical direction**: tiling_strategy
**Profile motivation**: 160,845 register spills from too many simultaneously-live tile values with (128,128,128) tiling
**Approach**: Phase-specific tiling with smaller M/N tiles (64) while keeping K=128 for MXU throughput. Added `_phase_tiling()` helper for dimension-aware tiling. Default changed to (64,128,64)*3.
**Expected impact**: ~2.4x reduction in live tile values per phase, cutting register spills substantially
**Target metric improvement**: Register spills 160,845 → <80,000; compute_efficiency 6.68% → 10%+
**Key changes**: Added _phase_tiling() helper, changed default tiling to (64,128,64)*3

### Variant: hbm_compute_overlap
**Technical direction**: hbm_compute_overlap
**Profile motivation**: 160,845 spills from 3 quantized tensors (lhs, lhs_t, rhs) live simultaneously in forward
**Approach**: Deferred lhs_t quantization from forward to backward pass. Save lhs_bf16 in residuals instead. Backward serializes phases so fewer quantized tensors are live at once.
**Expected impact**: Forward has 2 quantized tensors live instead of 3; backward never has more than 2 live at once
**Target metric improvement**: Register spills 160,845 → <80,000; scalar ALU / MXU ratio down from 33x
**Key changes**: Removed lhs_t quantization from _gmm_fwd, added deferred quantization in _gmm_bwd

### Variant: mxu_vpu_overlap
**Technical direction**: mxu_vpu_overlap
**Profile motivation**: Scalar ALU 33x higher than MXU; VPU quantization serialized with MXU matmul
**Approach**: Deferred lhs_t quantization to backward, reordered forward (rhs first, then lhs), restructured backward for 4-step MXU/VPU pipelining
**Expected impact**: MXU/VPU overlap in backward where dlhs GMM runs on MXU while lhs_t and drhs_dout quantize on VPU
**Target metric improvement**: MXU utilization 0.002% → higher; scalar ALU/MXU ratio 33x → lower
**Key changes**: Reordered forward quantization, deferred lhs_t, restructured backward for overlap

### Variant: memory_layout
**Technical direction**: memory_layout
**Profile motivation**: 160,845 spills from carrying too many quantized tensors as custom_vjp residuals
**Approach**: Recomputation-over-materialization — eliminated lhs_q and lhs_t from residuals, save only lhs_bf16 and rhs_q. Recompute lhs_t in backward.
**Expected impact**: Residuals drop from (lhs_q, rhs_q, group_sizes, lhs_t) to (lhs_bf16, rhs_q, group_sizes)
**Target metric improvement**: Register spills 160,845 → large reduction; compute_efficiency 6.68% → higher
**Key changes**: Factored out _quantize_lhs/_quantize_lhs_t helpers, reduced residual tuple, backward recomputes lhs_t

### Variant: quantization_strategy
**Technical direction**: quantization_strategy
**Profile motivation**: Scalar ALU 33x higher than MXU from FP8 quantization overhead (absmax, scaling per tile)
**Approach**: Three simultaneous changes: (1) tile_size 128→256 for 4x fewer scale computations, (2) channelwise-only quantization for lhs_t, (3) single quantized grad reused for both backward passes
**Expected impact**: 4x fewer quantization tiles, one fewer full quantization pass in backward
**Target metric improvement**: Scalar ALU utilization down; register spills down from fewer scale tensors
**Key changes**: tile_size=256, removed tiled_axes for lhs_t, unified grad quantization in backward
