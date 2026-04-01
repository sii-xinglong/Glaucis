## Profile Brief for Round 0

### Source
- Kernel: `examples/kernels/chunk_fused_kernels.py` (baseline)
- Speedup: 4.28x | Latency: 0.054ms (ref: 0.231ms)
- Compute ratio: 1.0 | Memory transfer ratio: 0.0

### Hardware Utilization Summary
| Unit | Utilization % | Assessment |
|------|--------------|------------|
| MXU  | 22.7% | Medium — room for improvement |
| Scalar ALU | 0.003% | Negligible — not control-flow heavy |
| Vector ALU | 14.9% | Medium — significant VPU work (exp, masking) |
| Vector Load | 0.001% | Negligible |
| Vector Store | 7.6% | Notable — likely register spill traffic |
| Vector EUP | 0.4% | Low |
| Register fills/spills | 21440/20420 | High — significant register pressure |

### Deep Profiling Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | 4 | Very low — kernel is compact |
| MXU dual ratio | N/A | Not available (no LLO) |
| Avg ops/bundle (ILP) | N/A | Not available |
| HBM bandwidth | N/A | Not available |
| Arithmetic intensity | N/A | Not available |
| Compute efficiency | N/A | Not available |
| VMEM utilization | N/A | Not available |
| HBM capacity used | 0.002% of 192 GB (4.0 MB) | Very low — minimal allocation |
| DMA transfers | N/A | Not available |
| Pipeline NOPs | N/A | Not available |
| Compile time | 9.1s | Moderate |
| Timing CV | 0.5% | Excellent stability |

### Bottleneck Diagnosis
**Primary bottleneck**: Register pressure + MXU underutilization

**Evidence**: The kernel shows 21,440 vector fills and 20,420 vector spills — very high register pressure indicating VMEM scratch and intermediate values are exceeding register file capacity. MXU utilization at 22.7% is medium, suggesting the MXU is often stalled waiting on data. The 7.6% vector store utilization is disproportionately high relative to vector load (0.001%), consistent with spill-dominated store traffic. Compute ratio of 1.0 with memory_transfer_ratio of 0.0 indicates the profiler sees this as fully compute-bound, but the register spills suggest hidden memory overhead from spill/fill cycles that drag down effective throughput.

**Combined patterns**: High spills (20K) + medium MXU (22.7%) + high vector store (7.6%) = register-pressure-limited kernel where MXU is starved by spill/fill traffic. The kernel body recomputes A (BT x BT matmul) and performs multiple BK x BV matmuls per time step, requiring many large intermediates that don't fit in the register file simultaneously.

### LLO Key Observations
LLO not available — no `llo_final.txt` artifact was collected. To get LLO data, ensure the evaluator's `--deep-profile` flag is enabled and GCS artifact upload is configured.

### HLO Key Observations
HLO not available — no `hlo_post_opt.txt` artifact was collected.
- Fusions: 0 (ideal for single Pallas kernel)
- HBM capacity: 0.002% — kernel is extremely memory-efficient

### Optimization Priorities (derived from profile)
1. **Reduce register pressure**: 20K+ spills dominate vector store traffic. The fused kernel holds multiple large intermediates (b_qg, b_kg, b_A, exp_g, exp_neg_g — all [BT, BK] or [BT, BT]) simultaneously. Reordering operations to reduce live intermediate count, or splitting the time-step body into phases that free intermediates earlier, could dramatically reduce spills and free MXU cycles.

2. **Improve MXU scheduling**: At 22.7% utilization, the MXU has significant headroom. With register pressure reduced, MXU ops should pipeline better. Consider restructuring dot products to enable dual-MXU scheduling (mxu0/mxu1 co-issue). The four Phase 3 dot products in the backward kernel are independent and could potentially overlap.

3. **Explore precision trade-offs**: All dot products currently use `Precision.HIGHEST`. For intermediates that don't affect final accuracy (e.g., the causal-masked A matrix computation), using default precision could reduce MXU cycle count and register pressure from wider intermediate types.

### What NOT to try (profile evidence)
- **Increasing block sizes**: BK=128 and BV=128 already match the full K and V dims. The kernel requires K==BK and V==BV (single-tile), so tiling these further is not applicable.
- **Adding more scratch memory / double buffering**: With 20K+ register spills, VMEM is already under pressure. Adding more scratch buffers would worsen the situation.
- **Reducing compilation overhead**: Compile time (9.1s) is a one-time cost and does not affect runtime latency.
- **HBM optimization**: At 0.002% HBM capacity and 4 MB peak memory, HBM is not a bottleneck. The kernel is already extremely memory-efficient on the HBM side.
