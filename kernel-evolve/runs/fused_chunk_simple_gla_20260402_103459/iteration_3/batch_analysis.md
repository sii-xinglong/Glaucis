## Round 3 Batch Analysis

**Variants evaluated**: 5
**Successes**: 4 | **Failures**: 1
**Best speedup this round**: 1.3889x (L2_bwd_2step)
**Overall best speedup**: 1.4023x (lineage L1, from Round 2 L1_scratch_A — no improvement this round)

### Comparative Ranking

| Rank | Variant | Status | Speedup | Latency (ms) | Compute Ratio | Bottleneck | Direction |
|------|---------|--------|---------|--------------|---------------|------------|-----------|
| 1 | L2_bwd_2step | SUCCESS | 1.3889x | 9.692 | 1.0 | register pressure + single-MXU | backward_grid_reduction |
| 2 | L1_no_scratch_A | SUCCESS | 1.3881x | 9.692 | 1.0 | register pressure + single-MXU | cleanup |
| 3 | L1_k_split_h | SUCCESS | 1.3881x | 9.692 | 1.0 | register pressure + single-MXU | bf16_dh_states |
| 4 | L1_2step | SUCCESS | 1.2221x | 11.011 | 1.0 | grid overhead + register pressure | register_pressure_reduction |
| -- | L1_bt32 | INCORRECT | -- | -- | -- | -- | block_shape_change |

### Critical Finding: Compiler Normalization

**L1_no_scratch_A, L1_k_split_h, and L2_bwd_2step compiled to IDENTICAL binaries:**
- VLIW bundles: 5,275 (all three)
- MXU ops: 640 mxu0 / 0 mxu1 (all three)
- VMEM: 589,824 bytes (all three)
- DMA: 464 (all three)
- Spills: 6,326,905 / Fills: 6,762,360 (all three)
- Kernel latency: 9.692ms (identical to 3 decimal places, all three)

Three completely different source-level changes (removing scratch_A staging, switching dh_states to bf16, adding backward 2-step grid reduction) all compiled to the same binary output. The tiny speedup difference between L2_bwd_2step (1.3889x) and the L1 variants (1.3881x) is entirely due to reference benchmark variance (13.462ms vs 13.454ms).

This is the strongest evidence yet for **FP39** (Mosaic compiler normalization): source-level restructuring of backward passes, dtype changes to intermediate tensors, and removal of unused scratch operands are all normalized away by the compiler.

### Per-Variant Details

#### L2_bwd_2step (Rank 1)

**Status**: SUCCESS
**Speedup**: 1.3889x (regression from L2's best of 1.4023x)
**Latency**: 9.692ms
**Lineage**: L2 (round 3)

| Metric | Value | Assessment |
|--------|-------|------------|
| compute_ratio | 1.0 | compute-bound |
| vliw_bundle_count | 5,275 | identical to Round 2 L1_bwd_kv_tiling baseline |
| MXU dual_ratio | 0.0 | critical — only MXU0 active |
| VMEM utilization | 0.88% of 64 MiB | extremely low — 63.4 MiB headroom |
| HBM bandwidth | 335 MB | 0.94% of peak |
| Peak memory | 1,280 MB | 0.65% of 192 GB |
| Bundle density (avg) | 3.39 ops/bundle | good ILP |
| DMA transfers | 464 (single-buffered) | no double buffering |
| Pipeline NOPs | 0 | ideal |
| Register fills/spills | 6,762,360 / 6,326,905 | severe register pressure |

**Bottleneck**: Register pressure (6.3M spills) + single-MXU scheduling (dual_ratio=0.0). Identical compiled output to L1 variants — backward changes had zero effect.

#### L1_no_scratch_A (Rank 2)

**Status**: SUCCESS
**Speedup**: 1.3881x (regression from L1's best of 1.4023x)
**Latency**: 9.692ms
**Lineage**: L1 (round 3)

Removing scratch_A successfully eliminated the +224 VLIW bundles and +32 DMA ops that L1_scratch_A had added. Metrics now match the clean Round 2 baseline exactly:
- VLIW: 5,275 (was 5,499 with scratch_A, confirming strategy.md prediction of 5499→5275)
- DMA: 464 (was 496 with scratch_A, confirming prediction of 496→464)

However, the speedup dropped from 1.4023x to 1.3881x. The 1.4023x measurement in Round 2 appears to have been a reference benchmark fluctuation rather than a genuine improvement from scratch_A.

#### L1_k_split_h (Rank 3)

**Status**: SUCCESS
**Speedup**: 1.3881x
**Latency**: 9.692ms
**Lineage**: L1 (round 3)

bf16 dh_states storage had zero impact. Compiled identically to L1_no_scratch_A. This confirms:
- **FP45/FP53**: HBM bandwidth reduction on a compute-bound kernel (compute_ratio=1.0) has zero effect
- The bf16 cast was optimized away by the compiler — dh_states access patterns are identical in the compiled output

#### L1_2step (Rank 4)

**Status**: SUCCESS
**Speedup**: 1.2221x (significant regression from 1.388x baseline)
**Latency**: 11.011ms
**Lineage**: L1 (round 3)

| Metric | Value | vs 4-step | Assessment |
|--------|-------|-----------|------------|
| vliw_bundle_count | 2,643 | 5,275 (-49.9%) | exactly half — expected |
| MXU ops | 320 | 640 (-50%) | exactly half — expected |
| MXU dual_ratio | 0.0 | 0.0 | unchanged — still single-MXU |
| VMEM bytes | 327,680 | 589,824 (-44.4%) | smaller kernel footprint |
| VMEM utilization | 0.49% | 0.88% | lower — less on-chip data |
| DMA transfers | 240 | 464 (-48.3%) | roughly half |
| Register fills | 4,122,360 | 6,762,360 (-39.0%) | reduced — fewer live values |
| Register spills | 4,275,040 | 6,326,905 (-32.4%) | reduced — fewer live values |
| Grid iterations | 32 | 16 | doubled — 2x more overhead |
| Latency | 11.011ms | 9.692ms (+13.6%) | slower despite lower spills |

**Key insight**: 2-step halves per-iteration complexity (VLIW, MXU, spills) but doubles grid iterations. The grid overhead (scratch h state read/write per iteration, DMA per iteration) wins. Per-iteration spill reduction (~35%) is not enough to offset 2x more iterations. This definitively confirms **SO20**: 4-step > 2-step hierarchy.

**LLO analysis** (5,253 lines):
- Kernel: `_fused_chunk_fwd_2step_kernel`
- iteration_bounds = [10, 16, 1, 1, 32] (B=10, H=16, NT=32 vs 16 for 4-step)
- scratch_operands = 1 (just h state — scratch_A correctly removed)
- h state initialization: 16 zero-stores to arg10 (128×128 matrix = 16 [8×128] vectors)
- MXU scheduling: all mxu_id=0, zero mxu1 ops — single-MXU pattern persists regardless of unroll factor

#### L1_bt32 (INCORRECT)

**Status**: INCORRECT
**Error**: max_diff=24.21875 (atol=10.0)
**Direction**: block_shape_change (BT=64→32)

**Cause**: BT=32 produces incorrect results. max_diff=24.22 is 2.4x the atol threshold. This is NOT a numerical precision issue — it's a fundamental algorithmic incorrectness at BT=32.

**Root cause analysis**: The kernel's attention computation assumes BT=64 tiling. With BT=32:
1. The attention matrix [32,32] changes the causal masking pattern
2. The chunk boundary alignment changes (T=4096 / 32 = 128 chunks vs 64 chunks)
3. The g_gamma decay computation may depend on chunk_size=64 for correct accumulation

This extends **FP24** (BT<64 underutilizes MXU) to also include correctness failure, not just MXU efficiency.

### Lineage Trends

#### L1 Trend (Rounds 1-3)
| Round | Variant | Speedup | VLIW | Spills | Direction |
|-------|---------|---------|------|--------|-----------|
| 1 | bwd_kv_tiling | 1.3881x | 5,275 | 6,326,905 | bwd_kv_tiling |
| 2 | L1_scratch_A | 1.4023x | 5,499 | 6,326,905 | scratch_memory |
| 3 | L1_no_scratch_A | 1.3881x | 5,275 | 6,326,905 | cleanup |

**Assessment**: L1 has **stagnated**. Round 2's improvement to 1.4023x was likely reference benchmark variance (L1_scratch_A added VLIW overhead with zero spill reduction). The true baseline is 1.388x. All source-level modifications produce the same compiled output.

#### L2 Trend (Rounds 1-3)
| Round | Variant | Speedup | VLIW | Spills | Direction |
|-------|---------|---------|------|--------|-----------|
| 1 | bwd_monolithic | 1.3881x | 5,275 | 6,326,905 | bwd_monolithic |
| 2 | L2_bwd_fused_arbitrary | 1.4023x | 5,275 | 6,326,905 | bwd_fused_arbitrary |
| 3 | L2_bwd_2step | 1.3889x | 5,275 | 6,326,905 | backward_grid_reduction |

**Assessment**: L2 has also **stagnated**. Same pattern as L1 — Round 2 had marginally higher reference latency measurement, and L2's monolithic backward compiles identically to L1's split backward.

### Stagnation Diagnosis

Both lineages are fundamentally stuck at the **same compiled kernel** (VLIW=5275, MXU=640/0, spills=6.3M). After 3 rounds and 15 total variants:

**What has been proven ineffective:**
1. Source-level backward restructuring (split vs monolithic vs fused arbitrary) — FP39
2. VMEM scratch staging (scratch_A) — FP33
3. bf16 intermediate storage — FP45/FP53
4. 2-step unrolling — SO20 confirms 4-step is superior
5. BT=32 — correctness failure (extending FP24)
6. Backward grid unrolling — FP43 (confirmed for both split and monolithic)

**Remaining avenues (untested):**
1. **BV=64 output tiling**: Process half the V dimension per grid iteration. Risk: FP54 showed BK=64 violates block shape, but BV may be different since it's the output dimension, not the reduction dimension.
2. **Attention matrix recomputation**: Instead of storing A in registers, recompute A=q·k^T on demand. Trades MXU ops for fewer live values.
3. **Forward-only optimization**: Since backward compiles identically regardless of source structure, focus exclusively on the forward pass.
4. **emit_pipeline**: Use Pallas pipelining primitives to overlap DMA with compute (currently no double buffering).
5. **Compiler hints**: VMEM at 0.88% with 6.3M spills suggests the compiler is under-allocating VMEM for register spill space. Investigate compiler flags or larger scratch allocations that hint the compiler to use more VMEM.

### IR Analysis (L1_2step — only variant with artifacts)

**LLO Structure** (5,253 lines):
- `_fused_chunk_fwd_2step_kernel` with iteration_bounds=[10,16,1,1,32]
- Single scratch operand: arg10 (128×128 h state)
- h state initialization: 16 vector_stores of zeros (conditional on first time iteration)
- MXU scheduling: exclusively mxu_id=0, confirming dual_ratio=0.0 persists at 2-step

**Comparison to 4-step LLO**:
- 2-step: 5,253 lines, 2,643 bundles, 320 MXU ops
- 4-step: ~10,500 lines (estimated), 5,275 bundles, 640 MXU ops
- Ratio is exactly 1:2, confirming the compiler generates linear code proportional to unroll factor
- No optimization from reduced register pressure at 2-step — spills still 4.1M/4.3M (not zero)

**Spill analysis**: 2-step reduced spills by ~35% (4.3M vs 6.3M) but did NOT eliminate them. With 2 sub-steps, the kernel still holds 2 sets of q/k/v/A/o intermediates plus the h state. The 128×128 h state alone is 16 [8×128] vectors = significant register pressure even at 2-step.
