## Profile Brief for Round 0

### Source
- Kernel: fused_chunk_simple_gla (baseline template)
- Speedup: 1.00x | Latency: 13.45ms (template) vs 13.45ms (reference)
- Compute ratio: 1.0 | Memory transfer ratio: 0.0
- Compile time: 26.9s | Peak memory: 1280 MB

### Hardware Utilization Summary
| Unit | Utilization % | Assessment |
|------|--------------|------------|
| MXU  | 5.6% | very low — kernel is far from compute-bound |
| Scalar ALU | 3.9% | low, but comparable to MXU — suggests control-flow overhead is proportionally significant |
| Vector ALU | 3.8% | low |
| Vector Load | 0.7% | very low |
| Vector Store | 0.9% | very low |
| Vector EUP | 0.4% | very low |
| Register fills/spills | 310,256 / 310,256 | severe register pressure — high VMEM spill traffic |

### Deep Profiling Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | 5 | very low count — may indicate coarse profiling granularity |
| MXU dual ratio | n/a | not available |
| Avg ops/bundle (ILP) | n/a | not available |
| HBM bandwidth | n/a | not available |
| Arithmetic intensity | n/a | not available |
| Compute efficiency | n/a | not available |
| VMEM utilization | n/a | not available |
| HBM capacity used | 0.65% of 192 GB (1280 MB) | low — kernel footprint is small relative to device memory |
| DMA transfers | n/a | not available |
| Pipeline NOPs | n/a | not available |
| Fusion count | 0 | ideal — single Pallas kernel, no XLA fusion overhead |

### Benchmark Details
| Metric | Template | Reference |
|--------|----------|-----------|
| Median latency | 13.45ms | 13.45ms |
| Min latency | 0.006ms | 0.006ms |
| Max latency | 13.56ms | 13.56ms |
| Compile time | 26.9s | 26.6s |
| Peak memory | 1280 MB | 1280 MB |
| Timing source | xprof_clustered | xprof_clustered |

Note: The min latency (~6us) is an outlier likely from cached/no-op execution. The clustered median (13.45ms) is the reliable measurement.

### Bottleneck Diagnosis
**Primary bottleneck**: register pressure / memory-bound
**Evidence**: MXU utilization is only 5.6%, indicating the kernel spends most time on non-MXU operations. The 310,256 register fills/spills indicate severe VMEM register pressure — data is being spilled to and reloaded from HBM, consuming bandwidth that should be used for useful computation. All hardware units show very low utilization (<6%), suggesting the kernel is bottlenecked on memory traffic from spills rather than actual computation.
**Combined patterns**: Low MXU + high spills = register-pressure-dominated. The kernel's fused design (keeping h in VMEM scratch) may be exceeding available register/VMEM capacity, causing the compiler to spill aggressively.

### LLO Key Observations
LLO not available — no `llo_final.txt` artifact was downloaded from GCS.

### HLO Key Observations
HLO not available — no `hlo_post_opt.txt` artifact was downloaded from GCS.
- Fusion count: 0 (ideal for single Pallas kernel)

### Optimization Priorities (derived from profile)
1. **Reduce register pressure**: 310K fills/spills dominate execution time. Reducing block sizes (smaller BT, BK, BV tiles), splitting the kernel into smaller stages, or reducing the number of live intermediates in the inner loop could dramatically cut spill traffic. This is the single biggest lever.
2. **Improve MXU utilization**: At 5.6% MXU, the kernel is far from compute-bound. After reducing spills, focus on ensuring matmul dimensions are large enough to fill both MXUs (dual scheduling). Check that K and V tile sizes are multiples of 128 for efficient MXU usage.
3. **DMA/compute overlap**: Once spills are reduced, ensure DMA prefetching of next chunk's q/k/v tiles overlaps with current chunk's MXU computation. Double-buffering scratch memory can hide HBM latency.

### What NOT to try (profile evidence)
- **Increasing block sizes**: With 310K spills already, larger tiles will only worsen register pressure and increase spill traffic. Do not increase BT, BK, or BV until spills are near zero.
- **Adding more scratch buffers**: VMEM is already under pressure from spills. Adding more scratch allocations risks OOM or worsening spills.
- **Fusion of additional operations**: Fusion count is already 0 (ideal). The kernel is already fully fused — the bottleneck is within the fused kernel itself, not at the XLA fusion level.
