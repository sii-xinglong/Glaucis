## Profile Brief for Round 8

### Source
- Kernel: iteration_7/variants/L2_recompute_dh_v2/kernel.py
- Speedup: 8.988x | Latency: 855.8ms
- Compute ratio: 1.0 | Memory transfer ratio: 0.0

### Delta vs Baseline
| Metric | Baseline (R0) | Current Best (R7) | Delta |
|--------|---------------|-------------------|-------|
| Speedup | 0.885x | 8.988x | +916% |
| Compute ratio | 1.0 | 1.0 | unchanged |
| VLIW bundles | 8270 | 9420 | +13.9% |
| MXU ops (mxu0) | 4656 | 5430 | +16.6% |
| MXU dual ratio | 0.0 | 0.0 | unchanged |
| Register spills | 1.91M | 2.72M | +42.4% |
| Register fills | 2.36M | 3.26M | +37.9% |
| Computation events | 264 | 171 | -35.2% |
| MXU util (runtime) | 22.5% | 34.6% | +53.6% |
| Vector Store util | 5.49% | 14.60% | +165.9% |

### Hardware Utilization Summary
| Unit | Utilization % | Assessment |
|------|--------------|------------|
| MXU  | 34.60% | medium — compute-bound but single-MXU |
| Scalar ALU | 0.04% | negligible — no control-flow bottleneck |
| Vector ALU | 18.93% | moderate — significant VPU work (exp, gating) |
| Vector Load | 0.01% | negligible |
| Vector Store | 14.60% | elevated — dominated by register spill writes |
| Vector EUP | 1.87% | low — exp() hardware utilized |
| Register fills/spills | 3,259,500 / 2,718,288 | severe register pressure — GROWING trend |

### Deep Profiling Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | 9420 | increased from R5 (7930) due to backward fusion |
| MXU dual ratio | 0.0 | poor — ONLY mxu0 active (5430 ops, mxu1=0) |
| DMA transfers | 20 | moderate |
| HLO fusions | 0 | ideal for Pallas |
| Computation events | 171 | 2 pallas_calls total (1 fwd + 1 bwd) |

### Bottleneck Diagnosis

**Primary bottleneck**: Register pressure (2.72M spills, growing trend) driving elevated Vector Store util (14.60%)

**Evidence**:
- Register spills grew +42.4% from baseline (1.91M → 2.72M) despite speedup improving 916%
- Vector Store util at 14.60% is the second-highest utilized unit (after MXU), dominated by register spill traffic
- Vector fills at 3.26M indicate constant reloading of spilled values, adding latency
- The backward fusion (R7) added +8.8% more spills vs R5 (2.50M → 2.72M) — each fusion increases register pressure

**Secondary bottleneck**: Single-MXU execution (dual_ratio=0.0)
- Only mxu0 is active (5430 ops), mxu1 has 0 ops
- Actual MXU utilization of total capacity is ~17.3% (34.6% / 2)
- This is a structural limitation of chunk_gla's matmul dimensions (128x128 with K=128, BT=64)
- FP21 confirmed: not fixable by register pressure reduction alone

### Current Kernel Architecture
- **Forward**: 1 pallas_call (combined h propagation + output computation with VMEM scratch, "arbitrary" time dim)
- **Backward**: 1 pallas_call (combined dh state propagation + dq/dk/dv computation with VMEM scratch, "arbitrary" time dim, reverse scan)
- **Total**: 2 pallas_calls (fully fused — down from 5 at baseline)
- **No more kernel fusion possible**: Both forward and backward are already single pallas_calls

### Round 7 Key Learnings (constrain Round 8 directions)
- **SO16**: Backward fusion yielded +14.6% by eliminating dh HBM round-trip (512MB). Architecture now fully fused at 2 pallas_calls.
- **FP32**: V-tiling via grid dimension impossible — block shape last dim must be divisible by 128 or equal array dim. BV_inner=64 fails.
- **R7 reduce_intermediates**: Scoped recomputation within kernel REGRESSED (-1.8%) — recomputation cost > spill savings at 7.845x baseline
- **R7 split_dv_dqdk**: Kernel splitting REGRESSED (-12.4%) — launch overhead > spill savings, confirming FP20 at higher baseline
- **FP30**: fori_loop + lax.cond INCREASES register pressure — do NOT use
- **FP23**: Manual loop unrolling INCREASES register pressure — do NOT use
- **FP29**: Staged output writes reduce spills but regress MXU util — marginal net benefit

### Optimization Priorities (derived from profile + R7 learnings)

1. **Reduce backward kernel intermediate count without recomputation**: The fused backward kernel has ~30+ simultaneously live arrays. R7's scoped recomputation approach failed because recomputing A-matrix and dA is expensive. Instead, try algebraic simplifications that ELIMINATE intermediates rather than recompute them. For example: combine `exp(g)` and `exp(-g)` into shared intermediate; merge gated products; fuse scale multiplications.

2. **VMEM scratch for intermediate spill management**: Instead of relying on the compiler's register allocator, explicitly use VMEM scratch buffers for intermediate results that are computed early but consumed late. This gives the programmer control over what stays in registers vs. VMEM, potentially reducing the random access pattern of compiler-generated spills.

3. **Reduce matmul count in backward**: The backward kernel has ~7 matmuls per time step (A computation, dv, dq, dk, dh update). If any matmuls share operands or can be combined, reducing the total matmul count would reduce both MXU time and intermediate live variables.

4. **Forward-backward asymmetry exploitation**: The forward kernel is likely much simpler (fewer intermediates). If the backward kernel contributes disproportionately to register pressure, consider whether any backward computation can be moved to forward (stored as residuals) or deferred.

5. **Alternative backward algorithm**: The current backward computes dq, dk, dv, dh all within a single time step. An alternative formulation might compute dh separately (simpler kernel) and pass it via VMEM, even within the same pallas_call using scratch with different addressing patterns.

### What NOT to try (profile evidence + R7 learnings)
- **Kernel splitting**: FP20 + R7 L2_split_dv_dqdk — launch overhead dominates at 8.988x
- **Scoped recomputation within kernel**: R7 L2_reduce_intermediates — recomputation cost > spill savings
- **fori_loop + lax.cond**: FP30 — increases register pressure
- **Manual loop unrolling**: FP23 — increases register pressure
- **V-tiling via grid dim with BV<128**: FP32 — TPU alignment constraint
- **Precision.DEFAULT**: FP31 — correctness failure
- **BT=128**: FP28 — VLIW bloat +44%
- **Dual-MXU optimization**: FP21 — structural limitation
- **Operation reordering**: FP18/FP8 — compiler ignores source order
- **exp(-x) → reciprocal(exp(x))**: FP27 — EUP is cheaper than ALU division
