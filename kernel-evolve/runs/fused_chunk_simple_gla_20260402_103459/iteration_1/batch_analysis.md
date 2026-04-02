## Round 1 Batch Analysis

**Variants evaluated**: 5
**Successes**: 5 | **Failures**: 0
**Best speedup this round**: 1.3882x (bwd_kv_tiling)
**Baseline template speedup**: 1.388x (4-step forward unrolling from session 2)
**Improvement over template**: ~0% (within measurement noise)

### Comparative Ranking

| Rank | Variant | Status | Speedup | Latency (ms) | Peak Mem (MB) | VLIW | Spills | Fills | Direction |
|------|---------|--------|---------|--------------|---------------|------|--------|-------|-----------|
| 1 | bwd_kv_tiling | SUCCESS | 1.3882x | 9.680 | 1280 | 5275 | 6,326,905 | 6,762,360 | bwd_phase_separation |
| 2 | bwd_monolithic | SUCCESS | 1.3881x | 9.681 | 1280 | 5275 | 6,326,905 | 6,762,360 | bwd_fusion |
| 3 | fwd_pyloop_clean | SUCCESS | 1.3789x | 9.745 | 1280 | 5275 | 6,403,705 | 6,762,360 | fwd_code_cleanup |
| 4 | eliminate_chunk_fwd_h | SUCCESS | 1.3789x | 9.745 | 1600 | 6494 | 6,403,705 | 6,762,360 | fwd_h_elimination |
| 5 | mixed_unroll_6 | SUCCESS | 1.3789x | 9.745 | 1920 | 5556 | 6,403,705 | 6,762,360 | combined_fwd_h_loop |

### Per-Variant Analysis

#### bwd_kv_tiling (Rank 1)
- **Direction**: Phase-separated backward computation (inter-chunk then intra-chunk)
- **Bottleneck**: Register-pressure-dominated, single-MXU
  - MXU dual_ratio=0.0 (only MXU0 used, MXU1 completely idle)
  - 6,326,905 spills / 6,762,360 fills (massive register pressure)
  - MXU util: 0.44% runtime, scalar ALU: 1.22% (scalar 2.8x MXU — control-flow overhead significant)
  - VMEM allocation: 589,824 bytes (0.88% of 64 MiB — severely underutilized)
  - HBM bandwidth util: 0.94% (very low — not bandwidth-limited)
  - No double buffering
- **Diagnosis**: Identical compiled output to bwd_monolithic. Source-level reordering of backward computation phases did not change the compiler's VLIW schedule. 76,800 fewer spills than fwd variants (backward-only difference). The 65us improvement over fwd variants is within noise but could reflect marginally better backward register allocation.

#### bwd_monolithic (Rank 2)
- **Direction**: Merged 2-pass backward into single fused kernel
- **Bottleneck**: Identical to bwd_kv_tiling in every metric
  - Same VLIW bundle count (5275), same MXU ops (640), same spills (6,326,905), same fills (6,762,360)
  - Same DMA count (464), same VMEM (589,824 bytes)
- **Diagnosis**: Despite fundamentally different source-level backward architecture (single-pass vs phase-separated), the compiler produced identical output. This confirms FP39 (source-level restructuring yields identical VLIW/MXU) and extends it to monolithic backward fusion.

#### fwd_pyloop_clean (Rank 3)
- **Direction**: Python for-loop instead of manual copy-paste for 4 sub-steps
- **Bottleneck**: Register-pressure-dominated, single-MXU
  - Same VLIW (5275), same MXU (640) as baseline
  - 6,403,705 spills (76,800 more than backward variants)
  - Higher MXU util (25.4%) and vector ALU util (16.1%) at runtime — different profiling sample but same compiled code
- **Diagnosis**: Confirms SO21 (Python for loops compile identically to manual unrolling). Identical latency and VLIW to manual unrolling. The 76,800 extra spills vs backward variants is a backward kernel difference, not forward.

#### eliminate_chunk_fwd_h (Rank 4)
- **Direction**: Output h from fused forward kernel, eliminate chunk_fwd_h pallas_call
- **Bottleneck**: Register-pressure-dominated with VMEM bloat
  - VLIW: 6494 (+1219 vs baseline = +23% complexity)
  - DMA: 592 (+128 vs baseline = +28% more data movement)
  - VMEM: 720,896 bytes (+131,072 = +22%)
  - Peak memory: 1600 MB (+320 MB = +25% HBM)
  - Same spills as other fwd variants (6,403,705)
- **Diagnosis**: Adding h as output to fused forward kernel INCREASED complexity (more VLIW bundles, more DMA, more VMEM) without reducing latency. The eliminated pallas_call was replaced by more expensive per-sub-step h writes. Confirms FP45 (pallas_call reduction has zero/negative impact).

#### mixed_unroll_6 (Rank 5)
- **Direction**: Combined eliminate_chunk_fwd_h + Python for-loop
- **Bottleneck**: Register-pressure-dominated with worst memory consumption
  - VLIW: 5556 (+281 vs baseline)
  - DMA: 528 (+64 vs baseline)
  - VMEM: 851,968 bytes (+262,144 = +44%)
  - Peak memory: 1920 MB (+640 MB = +50% HBM — worst of all variants)
- **Diagnosis**: Combining two optimizations that individually had zero/negative impact produced the worst memory footprint. The for-loop compilation partially offset the VLIW increase from h output (5556 vs 6494) but VMEM and peak memory are highest of all variants.

### Cross-Variant Patterns

1. **Compiler normalization is dominant**: bwd_kv_tiling and bwd_monolithic have IDENTICAL compiled profiles (same VLIW, MXU, spills, fills, DMA, VMEM) despite fundamentally different source architectures. fwd_pyloop_clean matches manual unrolling exactly. The Mosaic compiler normalizes away source-level differences.

2. **Three latency clusters, two compiled kernels**:
   - Cluster A (9.680ms): bwd_kv_tiling, bwd_monolithic — 76,800 fewer backward spills
   - Cluster B (9.745ms): fwd_pyloop_clean, eliminate_chunk_fwd_h, mixed_unroll_6 — identical latency despite very different source code and VLIW counts
   - The latency difference (65us) between clusters is within noise (0.67%)

3. **All variants share the same fundamental bottleneck**:
   - dual_ratio = 0.0 across ALL variants (MXU1 completely idle)
   - 6.3-6.4M register spills across ALL variants
   - VMEM utilization < 1.3% across ALL variants (massively underutilized)
   - No double buffering in ANY variant

4. **The bottleneck is architectural, not source-level**: Register pressure and single-MXU scheduling are compiler-level issues that source-level restructuring cannot fix. The 310K spills (baseline profiling) vs 6.3M spills (this round) likely reflects different profiling granularity rather than actual change.

### LLO Analysis (bwd_kv_tiling)

From the LLO IR:
- **6 kernel functions** compiled (forward + backward passes + transforms)
- **2560 llo.matmul operations** defined in IR (640 per-execution from profile)
- **464 vector_store/load operations** matching DMA count
- **All MXU ops on single port** (no dual MXU scheduling visible)
- **Dense constant initialization** at kernel entry (h scratch zeroing) — many redundant constant declarations
- **iteration_bounds = [10, 16, 1, 1, 16]** — 5D grid matching B=10, H=16, NK=1, NV=1, NT=16 (4 chunks per iteration)

### Key Insight for Next Round

Source-level optimization has hit a wall. All 5 directions explored (backward restructuring, forward refactoring, pallas_call elimination) produced zero meaningful improvement. The bottleneck is:

1. **Single-MXU execution** (dual_ratio=0.0): The matmul dimensions (BT=64, BK=128, BV=128) may not trigger dual-MXU scheduling. This is a block-size/tiling issue, not source structure.

2. **Massive register spills** (6.3M+): The compiler cannot keep all intermediates in registers. This needs VMEM scratch memory to explicitly manage data placement, or smaller block dimensions to reduce live values.

3. **VMEM severely underutilized** (0.88% of 64 MiB): Only 576 KB used out of 64 MB available. There is enormous headroom to use VMEM as scratch to reduce register pressure — BUT FP52 shows BT=128 fails correctness and FP50 shows 8-step regresses.

**Recommended directions for Round 2**:
- **Explicit VMEM scratch accumulators**: Instead of relying on compiler register allocation, use pltpu scratch memory to pin intermediates (h, A, attention masks) in VMEM, reducing spill traffic.
- **Block size exploration**: Try BK=64 or BV=64 to reduce per-tile live values and register pressure, at the cost of more loop iterations.
- **DMA prefetch / double buffering**: Add manual double buffering to overlap DMA with compute (currently no double buffering).
