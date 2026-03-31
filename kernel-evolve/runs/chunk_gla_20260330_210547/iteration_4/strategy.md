## Round 4 Strategy

Active lineages: 2
Total variants this round: 5
Variants generated in parallel via sub-agents.

### Lineage L1 (best speedup: 1.577x, direction: combined_fuse_skip)

#### Variant: L1_reduce_exp
**Base kernel**: iteration_3/variants/L1_combined_fuse_skip/kernel.py
**Technical direction**: Reduce exp() computations
**Profile motivation**: 2.5M register spills with 9.8% Vector Store util dedicated to spill traffic. Three independent [64, 128] exp() calls in backward kernel create unnecessary intermediate arrays.
**Approach**: Compute exp_pos_g = exp(b_g) once, derive exp_neg_g = 1/exp_pos_g (reciprocal) and exp_gn_minus_g = exp(b_gn)/exp_pos_g (scalar broadcast + division). Same pattern in forward output kernel. Reduces from 3+2=5 full [BT,K] exp() calls to 2 [BT,K] + 1 [K] exp() + 3 divisions.
**Expected impact**: Reduce register pressure (target: 2.5M → ~2.0M spills). May compound with event reduction.
**Target metric improvement**: Register spills -20%, Vector Store util 9.8% → ~8%

#### Variant: L1_fwd_single_kernel
**Base kernel**: iteration_3/variants/L1_combined_fuse_skip/kernel.py
**Technical direction**: Merge both forward pallas_calls into one
**Profile motivation**: Forward has 2 pallas_calls (chunk_fwd_h + chunk_gla_fwd_o_gk). Computation event reduction is the proven dominant lever (each 10% → ~25-35% speedup). Merging eliminates ~20+ events and the h tensor HBM round-trip.
**Approach**: Single pallas_call with grid (B, H, 1, 1, NT), time as "arbitrary" dimension. VMEM scratch for h state [128,128] float32 (64KB). At each time step: read h → compute output (recompute A inline) → update h. Combines chunk_fwd_h_kernel and chunk_gla_fwd_o_gk_pl bodies.
**Expected impact**: Eliminate 1 pallas_call, reduce events by ~20+ (207 → ~187). Expected ~15-25% speedup improvement.
**Target metric improvement**: Computation events -10%, DMA transfers -10%

#### Variant: L1_bf16_intermediates
**Base kernel**: iteration_3/variants/L1_combined_fuse_skip/kernel.py
**Technical direction**: Cast intermediate matrices to bfloat16
**Profile motivation**: 2.5M register spills. Intermediate [64,64] and [64,128] float32 arrays consume significant register space. Matmuls accumulate in float32 regardless of input dtype.
**Approach**: Cast b_a_masked and b_dA to bfloat16 before matmul inputs (halves [64,64] from 16KB to 8KB each). Ensure k_neg, k_decay, q_pos are explicitly bfloat16. Same pattern in forward output kernel.
**Expected impact**: Reduce register pressure (target: ~64KB savings per invocation). May compound with event reduction.
**Target metric improvement**: Register spills -15-20%

### Lineage L2 (best speedup: 1.056x, direction: PIVOT from V-tiling)

#### Variant: L2_fold_dh_pallas
**Base kernel**: iteration_3/variants/L1_combined_fuse_skip/kernel.py (PIVOT: adopts L1's clean kernel)
**Technical direction**: Replace lax.scan dh with pallas_call
**Profile motivation**: The backward lax.scan generates ~64 separate computation events (one per scan iteration). This is the largest remaining source of events after forward fusion. Replacing with a single pallas_call could eliminate ~60 events.
**Approach**: New `_chunk_bwd_dh_kernel` (mirrors `_chunk_fwd_h_kernel` structure). Grid: (B, H, K/BK, V/BV, NT) with time as "arbitrary". VMEM scratch for dh state. Reverse scan implemented by flipping input arrays before pallas_call and flipping output after.
**Expected impact**: Reduce events from ~207 to ~147 (-29%). Based on proven correlation, expect ~50-70% speedup improvement.
**Target metric improvement**: Computation events -29%, potential major speedup breakthrough

#### Variant: L2_fuse_fwd_output_h
**Base kernel**: iteration_3/variants/L1_combined_fuse_skip/kernel.py (PIVOT: adopts L1's clean kernel)
**Technical direction**: Fuse forward h propagation + output into single pallas_call
**Profile motivation**: Same as L1_fwd_single_kernel but implemented independently with (B, H, NT) grid. Forward has 2 pallas_calls that can be merged, eliminating ~20+ events and h HBM round-trip.
**Approach**: Single `_chunk_fwd_combined_kernel` with grid (B, H, NT), time as "arbitrary". VMEM scratch for h state [K, V]. At each time step: save h → compute output → update h. Uses PrefetchScalarGridSpec for g_gamma.
**Expected impact**: Eliminate 1 forward pallas_call, reduce events by ~20+. Expected ~15-25% speedup improvement.
**Target metric improvement**: Computation events -10%, eliminate h HBM round-trip

### Round 4 Risk Assessment

| Variant | Risk | Reward | Category |
|---------|------|--------|----------|
| L1_reduce_exp | Low | Low-Medium | Register pressure reduction |
| L1_fwd_single_kernel | Medium | Medium-High | Kernel fusion (event reduction) |
| L1_bf16_intermediates | Low | Low-Medium | Register pressure reduction |
| L2_fold_dh_pallas | High | Very High | Major event reduction (-29%) |
| L2_fuse_fwd_output_h | Medium | Medium-High | Kernel fusion (event reduction) |

**High-value bets**: L2_fold_dh_pallas (if successful, could be the biggest single improvement yet) and L1_fwd_single_kernel / L2_fuse_fwd_output_h (two independent implementations of forward fusion — hedges against implementation bugs).

**Safe bets**: L1_reduce_exp and L1_bf16_intermediates are conservative register pressure optimizations. Even if speedup is minimal, they demonstrate whether register pressure reduction has reached diminishing returns.
