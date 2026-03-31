## Round 3 Strategy

Active lineages: 2 (L1: 1.097x, L2: 0.872x)
Total variants this round: 5
Variants generated in parallel via sub-agents.

### Lineage L1 (best speedup: 1.097x, direction: reduce_inputs)

#### Variant: L1_fuse_fwd_A
**Base kernel**: iteration_2/variants/L1_reduce_inputs/kernel.py
**Technical direction**: fuse_fwd_A
**Profile motivation**: Forward pass has 3 separate pallas_calls. SO11 proved eliminating a pallas_call in backward gave 25% improvement. Apply same principle to forward: recompute A inside forward output kernel, eliminating chunk_gla_fwd_intra_gk pallas_call.
**Approach**: Add k as input to _chunk_gla_fwd_o_gk_pl, recompute A = q_pos @ k_neg.T * scale inside. Remove chunk_gla_fwd_intra_gk call from forward. Remove A from residuals.
**Expected impact**: Eliminate 1 forward kernel launch + HBM round-trip for A tensor. Cost: 1 extra [64,128]@[128,64] MXU dot (small vs existing [64,128]@[128,128]).
**Target metric improvement**: computation_events -10%, latency improvement ~5-15%

#### Variant: L1_skip_dg
**Base kernel**: iteration_2/variants/L1_reduce_inputs/kernel.py
**Technical direction**: skip_dg
**Profile motivation**: Backward kernel computes dg but caller discards it (`dq, dk, dv, _ = ...`). dg computation includes 1 MXU matmul (M_upper @ dg_raw), 1 BT×BT intermediate (16KB), and VPU work — all wasted.
**Approach**: Remove dg_ref from kernel signature, remove ALL dg computation, reduce out_shape/out_specs from 4 to 3.
**Expected impact**: Save 1 MXU matmul + 16KB M_upper register pressure + VPU chain. Pure dead-code elimination with zero correctness risk.
**Target metric improvement**: VLIW bundles -5-10%, register spills -10%, speedup 1.097x → ~1.15-1.20x

#### Variant: L1_combined_fuse_skip
**Base kernel**: iteration_2/variants/L1_reduce_inputs/kernel.py
**Technical direction**: fuse_fwd_A + skip_dg combined
**Profile motivation**: Compound two independent optimizations: (1) forward A fusion saves kernel launch overhead, (2) backward dg elimination saves 1 matmul + register pressure. Independent changes should compound.
**Approach**: Apply both fuse_fwd_A changes to forward AND skip_dg changes to backward in a single variant.
**Expected impact**: ~10-20% end-to-end improvement from combining forward kernel elimination + backward dead-code removal.
**Target metric improvement**: computation_events -10% (fwd) + VLIW bundles -5-10% (bwd)

### Lineage L2 (best speedup: 0.872x, direction: v_tiling)

#### Variant: L2_reduce_inputs
**Base kernel**: iteration_2/variants/L2_single_fused_k_tile/kernel.py
**Technical direction**: reduce_inputs (applied to V-tiled kernel)
**Profile motivation**: L2's V-tiled kernel still uses separate chunk_gla_fwd_intra_gk call in backward. SO11 showed this adds launch overhead. Apply same optimization.
**Approach**: Remove a_ref from V-tiled backward kernel, recompute b_a = q_pos @ k_neg.T * scale before V-loop. Remove chunk_gla_fwd_intra_gk call from bwd path.
**Expected impact**: Eliminate 1 backward kernel launch. A recomputation is V-independent so happens once before V-loop.
**Target metric improvement**: computation_events -10%, partially offset by V-tiling spill regression

#### Variant: L2_skip_dg
**Base kernel**: iteration_2/variants/L2_single_fused_k_tile/kernel.py
**Technical direction**: skip_dg (applied to V-tiled kernel)
**Profile motivation**: L2's V-tiled kernel has 5.74M register spills. dg computation adds ~48KB of live values (dgk_h_dh_acc, dg_raw, M_upper) plus 1 MXU matmul — all producing a discarded output.
**Approach**: Remove dg_ref, dgk_h_dh_acc accumulation (both V-chunks), and all post-accumulation dg computation. Reduce outputs from 4 to 3.
**Expected impact**: Meaningful spill reduction from removing 48KB of live values + 1 matmul. Pure dead-code elimination.
**Target metric improvement**: register spills -15-20% from 5.74M baseline, VLIW bundles -5%
