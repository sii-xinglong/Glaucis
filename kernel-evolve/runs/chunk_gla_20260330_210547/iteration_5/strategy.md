## Round 5 Strategy

Active lineages: 2
Total variants this round: 5
Variants generated in parallel via sub-agents.

### Lineage L2 (best speedup: 6.831x, direction: fold_dh_pallas)

#### Variant: L2_fuse_fwd_combined
**Base kernel**: iteration_4/variants/L2_fold_dh_pallas/kernel.py
**Technical direction**: Forward kernel fusion
**Profile motivation**: Forward pass still uses 2 pallas_calls (chunk_fwd_h + chunk_gla_fwd_o_gk). Merging eliminates one kernel launch, the h tensor HBM round-trip, and ~20 computation events. Two R4 attempts failed with fixable bugs (shape mismatch and index_map arg count).
**Approach**: Single `_chunk_fwd_combined_kernel` with grid (B, H, K/BK, V/BV, NT), time as "arbitrary". VMEM scratch for h state [BK, BV]. At each time step: save h → compute output (recompute A inline) → update h. All index_maps accept 5 args matching grid dimensions. h-update dot contracts on BT=64 (not K=128).
**Expected impact**: Eliminate 1 pallas_call, reduce events by ~20. Estimated ~5-10% speedup improvement.
**Target metric improvement**: Computation events -10%, DMA transfers -10%

#### Variant: L2_bf16_uniform
**Base kernel**: iteration_4/variants/L2_fold_dh_pallas/kernel.py
**Technical direction**: Uniform bfloat16 matmul operands for register pressure reduction
**Profile motivation**: 2.5M register spills with 9.8% Vector Store util. All intermediate arrays are float32 consuming register space. FP25 proved that BOTH matmul operands must be same dtype — this variant casts BOTH operands to bf16 consistently.
**Approach**: In backward fused kernel, backward dh kernel, and forward output kernel: cast ALL matmul operands to bfloat16 before each dot. Keep accumulators in float32 via preferred_element_type. Pre-cast gated variants (k_neg, k_decay, q_pos) to bf16. Store attention matrices in bf16.
**Expected impact**: Halve register footprint of matmul operand arrays. Target: 2.5M → ~1.5M spills.
**Target metric improvement**: Register spills -40%, Vector Store util 9.8% → ~6%

#### Variant: L2_reduce_live_bwd
**Base kernel**: iteration_4/variants/L2_fold_dh_pallas/kernel.py
**Technical direction**: Reduce peak live variable count via staged output writes
**Profile motivation**: 17+ simultaneously live arrays in backward kernel cause 2.5M spills. Current structure computes all intermediates upfront then writes all outputs at the end.
**Approach**: Restructure backward to write each output (dv, dq, dk) as soon as fully computed. Defer exp_gn_minus_g and k_decay computation until needed. Stages: compute dv → write dv_ref (freeing b_a_masked, b_dv*) → compute dq → write dq_ref → compute dk → write dk_ref. Creates genuine dependency barriers that tell compiler when arrays are dead.
**Expected impact**: Reduce peak live arrays from ~17 to ~12. Target: 30-40% spill reduction.
**Target metric improvement**: Register spills -30%, improved register allocator decisions

#### Variant: L2_bt128
**Base kernel**: iteration_4/variants/L2_fold_dh_pallas/kernel.py
**Technical direction**: Double chunk size (BT=64 → BT=128)
**Profile motivation**: NT=64 chunks drives grid size. Halving to NT=32 halves time-sequential kernel iterations. Larger [128,128] matmuls are better for MXU utilization.
**Approach**: Override chunk_size=128 in optimized_compute. All downstream code auto-adapts. T=4096/128=32 chunks.
**Expected impact**: Halve grid iterations. Larger MXU matmuls. Risk: 4x larger attention matrix [128,128] may increase VMEM pressure.
**Target metric improvement**: Computation events -50%, MXU util improvement

### Lineage L1 (best speedup: 1.577x, direction: combined_fuse_skip, stagnant: 1 round)

#### Variant: L1_fold_dh_fuse_fwd
**Base kernel**: iteration_3/variants/L1_combined_fuse_skip/kernel.py
**Technical direction**: Adopt fold_dh_pallas + forward kernel fusion
**Profile motivation**: L1 still uses lax.scan for backward dh — the proven SO14 bottleneck. Adopting fold_dh_pallas should bring it to ~6.8x. Adding forward fusion on top could push further.
**Approach**: (1) Copy chunk_bwd_dh_pallas from L2. (2) New _chunk_fwd_combined_kernel merging chunk_fwd_h + chunk_gla_fwd_o_gk into single pallas_call with time as "arbitrary" and VMEM scratch for h state.
**Expected impact**: fold_dh_pallas: ~6-7x (proven). Forward fusion: +5-10% additional. Total: ~6.5-7.5x.
**Target metric improvement**: Massive event reduction from eliminating lax.scan + forward fusion

### Round 5 Risk Assessment

| Variant | Risk | Reward | Category |
|---------|------|--------|----------|
| L2_fuse_fwd_combined | Medium | Medium | Forward kernel fusion (fixes R4 bugs) |
| L2_bf16_uniform | Low | Medium | Register pressure via bf16 operands |
| L2_reduce_live_bwd | Low | Low-Medium | Register pressure via staged writes |
| L2_bt128 | High | High | Chunk size doubling — untested territory |
| L1_fold_dh_fuse_fwd | High | Very High | Double optimization — could leapfrog L2 |

**High-value bets**: L1_fold_dh_fuse_fwd (if forward fusion works on top of fold_dh, it would be the most optimized variant yet) and L2_bt128 (untested chunk size that could halve grid iterations).

**Safe bets**: L2_bf16_uniform (fixing the bf16 approach that failed in R4) and L2_reduce_live_bwd (restructuring backward for better register allocation).

**Bug-fix hedge**: L2_fuse_fwd_combined is the corrected version of two R4 failures — moderate risk since the bugs are understood.
