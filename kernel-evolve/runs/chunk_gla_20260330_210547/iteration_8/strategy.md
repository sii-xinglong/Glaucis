## Round 8 Strategy

Active lineages: 2 (L1 stagnant 2 rounds, L2 active at 8.988x)
Total variants this round: 5 (all from L2 — L1 is stagnant and converged to same architecture)
Variants generated in parallel via sub-agents.

### Profile-Derived Directions

The R8 profile brief identifies **register pressure** as the primary bottleneck (2.72M spills, +42.4% vs baseline, growing trend). Secondary bottleneck is single-MXU execution (dual_ratio=0.0, structural limitation). With no more kernel fusion possible (already 2 pallas_calls), all 5 directions target register pressure reduction through different technical approaches.

### Lineage L2 (best speedup: 8.988x, direction: bwd_fusion)

#### Variant: L2_store_gated_residuals
**Base kernel**: iteration_7/variants/L2_recompute_dh_v2/kernel.py
**Technical direction**: Forward-backward asymmetry exploitation — store gating residuals from forward
**Profile motivation**: 2.72M register spills driven by backward kernel's ~30+ live arrays including 3 [BT,K] exp() intermediates
**Approach**: Forward stores q_pos (q*exp_pos) and k_neg (k*exp_neg) as residuals. Backward receives them as inputs, eliminating 3 exp() calls on [BT,K] arrays and the g_cumsum input entirely. Uses [BT] exp vectors instead of [BT,K] arrays for remaining gating.
**Expected impact**: 30-50% spill reduction by eliminating 3 large intermediate arrays from backward
**Target metric improvement**: vector_spills 2.72M → 1.4-1.9M, Vector Store util 14.6% → <10%

#### Variant: L2_extra_scratch
**Base kernel**: iteration_7/variants/L2_recompute_dh_v2/kernel.py
**Technical direction**: VMEM scratch for intermediate spill management
**Profile motivation**: 14.60% Vector Store util dominated by register spill writes; compiler's register allocator produces random access spill patterns
**Approach**: Adds 3 explicit VMEM scratch buffers (scratch_exp [BT,BK], scratch_A [BT,BT], scratch_dA [BT,BT]) for programmer-controlled intermediate staging. Moves ~24K float32 values from registers to VMEM scratch during matmul-heavy phases 3-4.
**Expected impact**: Reduce compiler-generated spills by giving it explicit VMEM staging areas
**Target metric improvement**: vector_spills 2.72M → 2.0-2.3M, MXU util maintained or improved

#### Variant: L2_algebraic_simplify
**Base kernel**: iteration_7/variants/L2_recompute_dh_v2/kernel.py
**Technical direction**: Algebraic simplification to eliminate intermediates
**Profile motivation**: 2.72M spills from ~30+ live arrays; 3 [BT,K] exp intermediates are prime candidates for elimination
**Approach**: Eliminates g_cumsum input entirely. Replaces [BT,K] exp arrays with [BT] vectors that broadcast during multiplication. Factors dq through exp_pos and dk through exp_neg using algebraic equivalences.
**Expected impact**: 3 fewer [BT,K] intermediate arrays → significant register pressure relief
**Target metric improvement**: vector_spills 2.72M → 1.5-2.0M, VLIW bundles 9420 → <9000

#### Variant: L2_fwd_store_A
**Base kernel**: iteration_7/variants/L2_recompute_dh_v2/kernel.py
**Technical direction**: Forward-backward asymmetry — store A matrix as residual
**Profile motivation**: Backward kernel has ~7 matmuls per time step; A-matrix matmul (q @ k.T) is redundantly computed in both forward and backward
**Approach**: Forward computes and stores the masked A matrix [BT,BT] as a 5D output. Backward receives it as input, skipping 1 matmul per time step. HBM cost: +33.5MB for the stored A matrices.
**Expected impact**: Eliminate 1 matmul + mask computation from backward, reducing both MXU time and live variable count
**Target metric improvement**: MXU ops 5430 → ~5000, vector_spills reduced via fewer live variables

#### Variant: L2_eliminate_gcumsum
**Base kernel**: iteration_7/variants/L2_recompute_dh_v2/kernel.py
**Technical direction**: Input elimination — remove g_cumsum from backward path
**Profile motivation**: g_cumsum is a [B,H,NT,BT,K] input (67MB) that creates HBM bandwidth pressure and exp([BT,K]) intermediates
**Approach**: Drops g_cumsum from backward entirely. Recomputes gating from the g_gamma scalar already available. Uses [BT] exp vectors (128x fewer elements per exp call) instead of [BT,K] exp arrays.
**Expected impact**: Saves 67MB HBM residual bandwidth, replaces [BT,K] exp with [BT] exp
**Target metric improvement**: vector_spills 2.72M → 2.0-2.4M, HBM bandwidth reduced
