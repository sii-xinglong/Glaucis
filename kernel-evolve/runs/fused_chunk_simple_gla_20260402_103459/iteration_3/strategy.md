## Round 3 Strategy

Active lineages: 2 (L1 @ 1.4023x, L2 @ 1.4023x)
Total variants this round: 5
Variants generated in parallel via sub-agents.

### Lineage L1 (best speedup: 1.4023x, direction: scratch_memory)

#### Variant: L1_no_scratch_A
**Base kernel**: iteration_2/variants/L1_scratch_A/kernel.py
**Technical direction**: cleanup
**Profile motivation**: L1_scratch_A added +224 VLIW bundles (+4.2%), +32 DMA ops (+6.9%), +32KB VMEM with zero spill reduction (FP33 confirmed). Removing the counterproductive scratch_A restores the cleaner 5275 VLIW base.
**Approach**: Remove scratch_A_ref from forward kernel, use partial_A directly in dot products
**Expected impact**: Recover ~4% VLIW overhead, eliminate unnecessary DMA traffic
**Target metric improvement**: VLIW 5499 → 5275, DMA 496 → 464

#### Variant: L1_2step
**Base kernel**: iteration_2/variants/L1_scratch_A/kernel.py
**Technical direction**: register_pressure_reduction
**Profile motivation**: 6.3M register spills dominate execution. 4-step unrolling keeps 4 sets of q/k/v/A/o intermediates live. 2-step halves the peak live set per kernel body.
**Approach**: Convert 4-step to 2-step forward grid unrolling using `for step in range(2)` (SO21). Grid iterations: 16 → 32. Also removes scratch_A (cleanup).
**Expected impact**: Fewer spills per iteration vs more grid iterations — tests the tradeoff at 2-step
**Target metric improvement**: spills 6.3M → significantly lower, but grid overhead increases
**Risk**: SO20 showed 4-step > 2-step historically (1.388x vs 1.222x). This is a regression test to confirm the hierarchy holds.

#### Variant: L1_bt32
**Base kernel**: iteration_2/variants/L1_scratch_A/kernel.py
**Technical direction**: block_shape_change
**Profile motivation**: Attention matrix [64,64]=16KB dominates register pressure. [32,32]=4KB is 4x smaller per sub-step.
**Approach**: Change chunk_size from 64 to 32. Removes scratch_A. Keeps 4-step grid unrolling. NT=128, NT//4=32 iterations.
**Expected impact**: 4x smaller attention matrix per sub-step should reduce register spills dramatically
**Target metric improvement**: spills 6.3M → potentially much lower
**Risk**: FP24 warns BT=32 underutilizes MXU ([32,128] tiles use 1/4 of MXU rows). More grid iterations (32 vs 16).

#### Variant: L1_k_split_h
**Base kernel**: iteration_2/variants/L1_scratch_A/kernel.py
**Technical direction**: bf16_dh_states
**Profile motivation**: dh_states tensor [B,H,NT,K,V] at f32 is ~6.4GB — largest backward intermediate. bf16 halves HBM bandwidth for this tensor.
**Approach**: Store dh_states snapshots as bf16 (Pass 1 writes bf16, Pass 2 reads and casts to f32). Analogous to SO17 (bf16 h residual). Also removes scratch_A (cleanup).
**Expected impact**: ~3.2GB less HBM bandwidth per backward pass. dh_states are point-in-time snapshots (not accumulated), so bf16 truncation is bounded.
**Target metric improvement**: HBM bandwidth reduction; potential latency improvement from reduced memory traffic
**Risk**: Profile shows compute_ratio=1.0 — kernel is compute-bound, not memory-bound. HBM bandwidth reduction may have zero impact (FP45, FP53).

### Lineage L2 (best speedup: 1.4023x, direction: bwd_fused_arbitrary)

#### Variant: L2_bwd_2step
**Base kernel**: iteration_2/variants/L2_bwd_fused_arbitrary/kernel.py
**Technical direction**: backward_grid_reduction
**Profile motivation**: Monolithic backward with 9+ matmuls per step may benefit from grid reduction differently than split backward.
**Approach**: 2-step backward grid unrolling on L2's monolithic backward. Grid iterations NT → NT//2. Each grid iteration processes 2 time chunks using Python for-loop sub-stepping.
**Expected impact**: Halved backward grid iterations, but doubled per-iteration complexity
**Target metric improvement**: backward grid iterations 64 → 32, potential latency reduction
**Risk**: FP43 says backward grid unrolling is negligible for split backward. FP48 says backward complexity interferes with forward at >=4-step. L2's monolithic backward may respond differently.
