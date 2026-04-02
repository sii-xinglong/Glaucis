## Round 2 Strategy

Active lineages: 2 (L1: bwd_phase_separation 1.3882x, L2: bwd_fusion 1.3881x)
Total variants this round: 5
Variants generated in parallel via sub-agents.

### Profile-Derived Direction Rationale

Round 1 confirmed source-level restructuring is exhausted (FP39). The profile brief shows:
- **Register pressure**: 6.3M fills/spills dominate execution (primary bottleneck)
- **Single-MXU**: dual_ratio=0.0, only MXU0 used (half capacity wasted)
- **VMEM underutilized**: 0.88% of 64 MiB (63.4 MiB headroom)
- **No double buffering**: DMA and compute cannot overlap

Round 2 targets compiler-level changes: compiler directives, VMEM scratch allocation, precision changes, and backward architecture with explicit state management.

### Lineage L1 (best speedup: 1.3882x, direction: bwd_phase_separation)

#### Variant: L1_emit_pipeline
**Base kernel**: iteration_1/variants/bwd_kv_tiling/kernel.py
**Technical direction**: Compiler directive optimization
**Profile motivation**: Scalar ALU at 2.8x MXU util suggests control-flow overhead from bounds checking. 5275 VLIW bundles may include redundant bounds check instructions.
**Approach**: Added `disable_bounds_checks=True` to forward kernel CompilerParams. Also converted manual 4-step copy-paste to Python for-loop (confirmed equivalent by SO21, but cleaner code).
**Expected impact**: Reduced VLIW bundle count from eliminated bounds check instructions, potentially freeing VLIW slots for better ILP.
**Target metric improvement**: VLIW bundles 5275 → lower, scalar_alu_util reduction

#### Variant: L1_scratch_A
**Base kernel**: iteration_1/variants/bwd_kv_tiling/kernel.py
**Technical direction**: Explicit VMEM scratch for attention matrix
**Profile motivation**: 6.3M register spills with VMEM at 0.88% — massive headroom to pin intermediates in VMEM scratch instead of register allocation. The attention matrix A [BT×BT = 64×64 = 16KB] is computed and consumed within each sub-step, a prime candidate for explicit VMEM placement.
**Approach**: Added [BT,BT] VMEM scratch `scratch_A_ref` to forward kernel. Each sub-step stores A to scratch after computation and loads from scratch before use, giving the compiler explicit VMEM placement hints.
**Expected impact**: Reduced register spills by moving A from register file to VMEM scratch. WARNING: FP33 noted manual VMEM staging can be counterproductive — Mosaic may add redundant load/store ops.
**Target metric improvement**: Register spills 6.3M → lower, VMEM utilization 0.88% → higher

#### Variant: L1_bf16_h_scratch
**Base kernel**: iteration_1/variants/bwd_kv_tiling/kernel.py
**Technical direction**: Precision reduction for h residual computation
**Profile motivation**: All matmuls use f32 HIGHEST precision (FP55 confirmed mandatory for correctness at atol=10.0). However, the separate `chunk_fwd_h` call (which computes h for backward) may tolerate bf16 — it only needs approximate values since backward uses them as a starting point.
**Approach**: Changed `states_in_fp32=True` to `states_in_fp32=False` in `chunk_fwd_h` call. Single parameter change — no kernel code modification.
**Expected impact**: Halved memory for h states (float32 → bfloat16), potentially reduced DMA traffic for h in backward pass. Risk: may fail correctness if backward is sensitive to h precision.
**Target metric improvement**: Peak memory reduction, potentially reduced backward DMA

### Lineage L2 (best speedup: 1.3881x, direction: bwd_fusion)

#### Variant: L2_bwd_fused_arbitrary
**Base kernel**: iteration_1/variants/bwd_monolithic/kernel.py
**Technical direction**: Fused backward with VMEM state management
**Profile motivation**: Current backward uses 2 separate pallas_calls (dv+dh then dq+dk). FP53 confirmed splitting/merging backward has zero perf impact, but the current split requires materializing h_states [B,NT,H,BK,BV] in HBM between passes. A single backward kernel with h accessed via 5D BlockSpec could eliminate this HBM round-trip.
**Approach**: Single backward pallas_call with "arbitrary" time dimension (reverse iteration). dh accumulated in VMEM scratch. h_ref accessed via 5D BlockSpec with `(b, NT-1-t, h, 0, 0)` index map to read pre-computed h in reverse order without transpose.
**Expected impact**: Eliminated h transpose + reduced HBM traffic from fewer pallas_calls. However, FP53 suggests compiler normalization may negate this.
**Target metric improvement**: DMA count reduction, HBM bandwidth reduction

#### Variant: L2_disable_bounds_checks
**Base kernel**: iteration_1/variants/bwd_monolithic/kernel.py
**Technical direction**: Compiler directive optimization (all kernels)
**Profile motivation**: Same as L1_emit_pipeline — scalar ALU overhead from bounds checking. Applied to ALL three pallas_calls (forward, backward pass 1, backward pass 2) for maximum coverage. chunk_fwd_h already had disable_bounds_checks.
**Approach**: Added `compiler_params=pltpu.TPUCompilerParams(disable_bounds_checks=True)` to all three pallas_call invocations.
**Expected impact**: Broader bounds check elimination across forward + backward, potentially larger VLIW reduction than L1_emit_pipeline (which only targets forward).
**Target metric improvement**: VLIW bundles reduction across all kernels, scalar_alu_util reduction
