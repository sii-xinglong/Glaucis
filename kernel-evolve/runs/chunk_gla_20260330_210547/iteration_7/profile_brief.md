## Profile Brief for Round 7

### Source
- Kernel: iteration_5/variants/L2_fuse_fwd_combined/kernel.py
- Speedup: 7.845x | Latency: 976ms
- Compute ratio: 1.0 | Memory transfer ratio: 0.0

### Delta vs Baseline
| Metric | Baseline (R0) | Current Best (R5) | Delta |
|--------|---------------|-------------------|-------|
| Speedup | 0.885x | 7.845x | +786% |
| Compute ratio | 1.0 | 1.0 | unchanged |
| VLIW bundles | 8270 | 7930 | -4.1% |
| MXU dual ratio | 0.0 | 0.0 | unchanged |
| Register spills | 1.96M | 2.50M | +27.5% |
| Computation events | 264 | 177 | -33.0% |
| MXU util (runtime) | 22.5% | 33.2% | +47.6% |

### Hardware Utilization Summary
| Unit | Utilization % | Assessment |
|------|--------------|------------|
| MXU  | 33.22% | medium — compute-bound but single-MXU |
| Scalar ALU | 0.04% | negligible — no control-flow bottleneck |
| Vector ALU | 20.20% | moderate — significant VPU work (exp, gating) |
| Vector Load | 0.01% | negligible |
| Vector Store | 9.82% | elevated — register spill traffic |
| Vector EUP | 2.08% | low — exp() hardware utilized |
| Register fills/spills | 3,038,841 / 2,497,509 | severe register pressure |

### Deep Profiling Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | 7930 | stable from R4 |
| MXU dual ratio | 0.0 | poor — ONLY mxu0 active (4656 ops, mxu1=0) |
| DMA transfers | 20 | moderate |
| HLO fusions | 0 | ideal for Pallas |
| Computation events | 177 | forward: 1 pallas_call, backward: 2 pallas_calls |

### Bottleneck Diagnosis

**Primary bottleneck**: Register pressure (2.5M spills) + single-MXU execution (dual_ratio=0.0)

**Evidence**:
- Vector Store util at 9.82% is dominated by register spill traffic (2.5M spills writing to VMEM)
- Vector fills at 3.0M indicate constant reloading of spilled values
- MXU dual_ratio=0.0 means only 1 of 2 MXU units is active — half the matmul capacity is unused
- Despite 33.2% MXU runtime utilization, this is on a single MXU — actual utilization of total MXU capacity is ~16.6%

**Combined patterns**:
- Register pressure + high VLIW count → compiler cannot fit all intermediates in registers, generating spill/fill traffic that slows execution
- Single-MXU + compute-bound → matmul dimensions or data dependencies prevent dual-MXU scheduling

### Current Kernel Architecture
- **Forward**: 1 pallas_call (combined h propagation + output computation with VMEM scratch)
- **Backward dh**: 1 pallas_call (reverse time scan via flipped inputs, VMEM scratch)
- **Backward fused**: 1 pallas_call (dq/dk/dv computation, grid=(H, B*NT))
- **Total**: 3 pallas_calls (down from 5 at baseline)

### Round 6 Key Learnings (negative results that constrain Round 7)
- **FP30**: fori_loop + lax.cond INCREASES register pressure (+66% spills). Do NOT use fori_loop with conditionals.
- **FP31**: Precision.DEFAULT fails correctness (max_diff=23.91 > atol=10.0). MUST keep HIGHEST.
- **R6 backward fusion attempts**: L2_bwd_fuse_dh and L2_recompute_dh both failed with FIXABLE BlockSpec rank bug (4D spec for 5D h array). The optimization approach is SOUND — needs 5D BlockSpec `(1, 1, 1, BK, BV)` with 5-element index_map.
- **R6 V-tiling attempt**: L1_bwd_grid_kv failed because V sub-tile BV_inner=64 didn't match array dim 128. Need proper index_map that selects V sub-tile range, not reduced block shape.

### Optimization Priorities (derived from profile + R6 learnings)

1. **Retry backward kernel fusion with corrected 5D BlockSpecs**: The R6 backward fusion attempts (L2_bwd_fuse_dh, L2_recompute_dh) had the RIGHT optimization approach but WRONG BlockSpec implementation. The h array is 5D `(B, NT, H, K, V)` and needs 5D BlockSpec `(1, 1, 1, BK, BV)` with `index_map` returning 5 indices `(i_b, i_nt, i_h, i_k, i_v)`. Fixing this is the highest-priority direction — forward fusion (SO15) yielded +14.8% from the same pattern.

2. **Reduce live intermediates in backward fused kernel**: The backward fused kernel has ~24 simultaneously live arrays causing 2.5M register spills. Restructuring to compute and consume intermediates in tighter scopes (e.g., compute b_gk and use it immediately before computing b_gv) might reduce peak register pressure without adding loop overhead.

3. **Split backward fused into dv-only + dq/dk passes**: Instead of computing dq, dk, dv together (all live simultaneously), split into two sequential pallas_calls: one for dv (fewer intermediates) and one for dq/dk. This trades an extra kernel launch for reduced register pressure in each.

### What NOT to try (profile evidence + R6 learnings)
- **Precision.DEFAULT**: FP31 — max_diff=23.91, fundamental correctness failure
- **fori_loop + lax.cond**: FP30 — increases register pressure, not reduces
- **BT=128 (chunk size increase)**: FP28 — VLIW bloat +44%
- **bf16 operands with Precision.HIGHEST**: FP25 — Mosaic rejects
- **Manual loop unrolling**: FP23 — increases register pressure
- **BT < 64**: FP24 — critically underutilizes MXU
- **Dual-MXU optimization**: FP21 — structural limitation of matmul dimensions
