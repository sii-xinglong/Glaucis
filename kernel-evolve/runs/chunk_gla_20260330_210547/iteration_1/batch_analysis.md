## Round 1 Batch Analysis

**Variants evaluated**: 5
**Successes**: 3 | **Failures**: 2 (INCORRECT)
**Best speedup this round**: 0.876x (mxu_vpu_overlap)
**Baseline speedup**: 0.885x
**All variants slower than baseline**

### Comparative Ranking

| Rank | Variant | Status | Speedup | Latency (ms) | VLIW Bundles | Spills | MXU dual | Direction |
|------|---------|--------|---------|--------------|-------------|--------|----------|-----------|
| 1 | mxu_vpu_overlap | SUCCESS | 0.876x | 8816 | 8270 | 1.96M | 0.0 | mxu_vpu_overlap |
| 2 | mxu_utilization | SUCCESS | 0.825x | 9410 | 5226 | 369K | 0.0 | mxu_utilization |
| 3 | tiling_strategy | SUCCESS | 0.823x | 9289 | 6312 | 3.30M | 0.0 | tiling_strategy |
| -- | hbm_compute_overlap | INCORRECT | -- | -- | -- | -- | -- | hbm_compute_overlap |
| -- | memory_layout | INCORRECT | -- | -- | -- | -- | -- | memory_layout |

### Per-Variant Details

#### mxu_vpu_overlap (Rank 1)

**Status**: SUCCESS | **Speedup**: 0.876x | **Latency**: 8816ms

| Metric | Value | vs Baseline | Assessment |
|--------|-------|-------------|------------|
| MXU util | 28.09% | +5.56pp | improved but still low |
| Vector ALU util | 21.58% | +4.48pp | improved |
| VLIW bundles | 8270 | 0 | **identical to baseline** |
| MXU dual ratio | 0.0 | 0 | unchanged — still single-MXU |
| MXU0 ops | 4656 | 0 | identical |
| Register fills | 2.56M | +197K | slightly worse |
| Register spills | 1.96M | +49K | slightly worse |
| DMA transfers | 24 | 0 | unchanged |

**Diagnosis**: The Python-level reordering of VPU/MXU ops had **NO effect on compiled code** — same VLIW bundle count (8270), same MXU ops (4656). The Mosaic compiler independently schedules operations regardless of source-level ordering. The measured speedup difference (0.876x vs 0.885x baseline) is within run-to-run variance. This direction is a dead end.

**Key learning**: Reordering operations in Python source does NOT influence the TPU compiler's VLIW scheduling. The compiler sees the dataflow graph, not the source order.

#### mxu_utilization (Rank 2)

**Status**: SUCCESS | **Speedup**: 0.825x | **Latency**: 9410ms

| Metric | Value | vs Baseline | Assessment |
|--------|-------|-------------|------------|
| MXU util | 11.96% | -10.57pp | **regressed** |
| Vector ALU util | 11.65% | -5.45pp | reduced |
| VLIW bundles | 5226 | -3044 (-37%) | **significant improvement** |
| MXU dual ratio | 0.0 | 0 | still single-MXU |
| MXU0 ops | 2910 | -1746 | fewer total MXU ops |
| Register fills | 369K | -1.99M (-84%) | **massive improvement** |
| Register spills | 369K | -1.54M (-81%) | **massive improvement** |
| DMA transfers | 20 | -4 | improved |
| HBM bandwidth | 134MB | N/A | measured |
| Computation events | 270 | +6 | slightly more kernel launches |

**Diagnosis**: Split backward kernel achieved its architectural goals:
- 81% spill reduction (1.9M → 369K) — the primary bottleneck is substantially mitigated
- 37% VLIW bundle reduction (8270 → 5226) — simpler kernels compile more efficiently
- BUT latency increased 8.3% (8691ms → 9410ms) — the overhead of two kernel launches + inter-kernel data transfer (passing dq) exceeds the savings

The dual-MXU scheduling still didn't activate (dual_ratio=0.0). Despite fewer live values, the compiler still puts all MXU ops on mxu0. This suggests the issue is NOT register pressure alone — the matmul dimensions or data dependencies may inherently prevent dual-MXU.

**Key insight**: The latency regression comes from 6 extra computation events (270 vs 264), confirming additional kernel launch overhead. The spill reduction is real and substantial but is offset by the operational overhead of the split.

#### tiling_strategy (Rank 3)

**Status**: SUCCESS | **Speedup**: 0.823x | **Latency**: 9289ms

| Metric | Value | vs Baseline | Assessment |
|--------|-------|-------------|------------|
| MXU util | 17.08% | -5.45pp | regressed |
| Vector ALU util | 15.63% | -1.47pp | slightly worse |
| VLIW bundles | 6312 | -1958 (-24%) | improved |
| MXU dual ratio | 0.0 | 0 | still single-MXU |
| Register fills | 3.35M | +987K (+42%) | **significantly worse** |
| Register spills | 3.30M | +1.39M (+73%) | **significantly worse** |
| DMA transfers | 20 | -4 | improved |

**Diagnosis**: This variant also split the backward kernel but the dg kernel's dA recomputation introduced MORE register pressure (3.3M spills vs 1.9M baseline). The recomputation adds intermediate values that overflow registers. The VLIW bundle reduction is modest and doesn't compensate.

**Key learning**: Recomputing dA in the dg kernel creates worse register pressure than keeping it from the fused kernel. The dA matrix (BT×BT = 64×64 float32 = 16KB) plus the additional intermediates for its recomputation exceed what the original kernel needed for dA as a reference.

### Failed Variants Summary

| Variant | Error Type | Root Cause | Fix |
|---------|-----------|------------|-----|
| hbm_compute_overlap | TypeError | `emit_pipeline` body function had wrong arity — `pipeline_body(i_t, k_tile_ref, v_tile_ref)` needed 2 ref args but pipeline passed 1 | Fix pipeline_body to accept refs as `*refs` or match emit_pipeline's in_specs count |
| memory_layout | ValueError | BlockSpec 4D `(1,1,64,128)` vs array 5D `(2,16,64,64,128)` — data layout change wasn't reflected in BlockSpec | Update BlockSpec dimensions to match new array shape |

### Key Insights for Next Round

1. **Source-level reordering is useless**: The Mosaic compiler schedules independently. Don't try to influence VLIW scheduling through Python operation order.
2. **Register pressure reduction is achievable via kernel splitting**: 81% spill reduction demonstrated (mxu_utilization). But the current split approach adds too much launch overhead.
3. **Dual-MXU is not triggered by register pressure reduction alone**: Even with 369K spills (down from 1.9M), dual_ratio remains 0.0. The issue may be in matmul dimensions (128×128 with K=128 on BT=64 chunks).
4. **dA recomputation is harmful**: It adds register pressure rather than reducing it. Pass dA between kernels or avoid recomputing it.
5. **All variants regressed vs baseline**: The baseline's fused approach, despite high spills, has less overhead than split approaches. The next round should focus on optimizing WITHIN the fused kernel rather than splitting it.
