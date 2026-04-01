## Profile Brief for Round 0

### Source
- Kernel: `examples/kernels/chunk_fused_kernels.py` (baseline)
- Reference: `examples/kernels/chunk_fused_kernels_ref.py` (non-fused Pallas, 3 separate pallas_calls)
- Speedup: 1.15x | Latency: 0.054ms (ref: 0.063ms)
- Compute ratio: 1.0 | Memory transfer ratio: 0.0

### Hardware Utilization Summary
| Unit | Utilization % | Assessment |
|------|--------------|------------|
| MXU  | 22.7% | Medium — room for improvement |
| Scalar ALU | 0.002% | Negligible — not control-flow heavy |
| Vector ALU | 14.9% | Medium — significant VPU work (exp, masking) |
| Vector Load | 0.001% | Negligible |
| Vector Store | 7.6% | Notable — likely register spill traffic |
| Vector EUP | 0.4% | Low |
| Register fills/spills | 21440/20420 | High — significant register pressure |

### Deep Profiling Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | 8270 | Moderate kernel complexity |
| MXU dual ratio | 0.0 (4656 mxu0, 0 mxu1) | Poor — only one MXU used, mxu1 completely idle |
| Avg ops/bundle (ILP) | 0.56 (4656/8270) | Very poor — most bundles have no MXU op |
| HBM bandwidth | N/A | Not available |
| Arithmetic intensity | N/A | Not available |
| Compute efficiency | N/A | Not available |
| VMEM utilization | N/A | Not available |
| HBM capacity used | 0.002% of 192 GB (4.0 MB) | Very low — minimal allocation |
| DMA transfers | 24 (no double buffering) | Low DMA count, no overlap with compute |
| Pipeline NOPs | 0 | Ideal — no explicit pipeline bubbles |
| Compile time | 9.1s | Moderate |
| Timing CV | 0.5% | Excellent stability |

### Bottleneck Diagnosis
**Primary bottleneck**: Register pressure + single-MXU execution

**Evidence**: The kernel shows 21,440 vector fills and 20,420 vector spills — very high register pressure. MXU dual_ratio is 0.0: all 4,656 MXU operations run on mxu0 with mxu1 completely idle, meaning the kernel uses only half the available matrix compute capacity. The ILP is extremely low at 0.56 MXU ops per VLIW bundle (4656 ops across 8270 bundles), meaning 44% of bundles contain an MXU op and the rest are occupied by VPU/scalar work or stalls. The 7.6% vector store utilization is disproportionately high relative to vector load (0.001%), consistent with spill-dominated store traffic.

**Combined patterns**: High spills (20K) + zero dual_ratio + low ILP (0.56) = register-pressure-limited kernel where MXU is both underutilized (single unit) and starved by spill/fill traffic. The fused kernel body recomputes A (BT x BT matmul) and performs multiple BK x BV matmuls per time step, requiring many large intermediates (b_qg, b_kg, b_A, exp_g, exp_neg_g — all [BT, BK] or [BT, BT]) that exceed register file capacity.

### LLO Key Observations
LLO not available — no `llo_final.txt` artifact was collected.

### HLO Key Observations
HLO not available — no `hlo_post_opt.txt` artifact was collected.
- Fusions: 0 (ideal for single Pallas kernel)
- HBM capacity: 0.002% — kernel is extremely memory-efficient

### Optimization Priorities (derived from profile)
1. **Enable dual-MXU scheduling**: dual_ratio=0.0 means mxu1 is completely idle. All 4,656 MXU ops run on mxu0 only. The backward kernel has four independent dot products in Phase 3 (b_dv_intra, b_dv_inter, b_dq_inter, b_dk_inter) that could potentially be split across mxu0/mxu1. Restructuring matmul dimensions or reordering operations to enable co-issue could nearly double MXU throughput. This is the single largest headroom — going from 0% to even 50% dual_ratio would significantly improve throughput.

2. **Reduce register pressure**: 20K+ spills dominate vector store traffic. The fused kernel holds multiple large intermediates ([BT, BK]=[64,128] and [BT, BT]=[64,64]) simultaneously. Reordering operations to reduce live intermediate count, splitting the time-step body into phases that free intermediates earlier, or reducing intermediate precision could reduce spills and free MXU cycles.

3. **Improve ILP / VLIW packing**: Only 0.56 MXU ops per bundle means most VLIW bundles lack MXU work. With register pressure reduced and dual-MXU enabled, restructuring the compute graph to interleave MXU and VPU work could improve bundle density. The 14.9% vector ALU utilization suggests exp() and masking operations could potentially overlap better with MXU dot products.

### What NOT to try (profile evidence)
- **Increasing block sizes**: BK=128 and BV=128 already match the full K and V dims. The kernel requires K==BK and V==BV (single-tile), so tiling these further is not applicable.
- **Adding more scratch memory / double buffering**: With 20K+ register spills, VMEM is already under pressure. Adding more scratch buffers would worsen the situation.
- **HBM optimization**: At 0.002% HBM capacity and 4 MB peak memory, HBM is not a bottleneck. The kernel is already extremely memory-efficient on the HBM side.
- **Reducing compilation overhead**: Compile time (9.1s) is a one-time cost and does not affect runtime latency.
- **Adding more pipelining/prefetch**: compute_ratio=1.0 indicates the kernel is compute-bound, not memory-bound. More prefetch won't help.
