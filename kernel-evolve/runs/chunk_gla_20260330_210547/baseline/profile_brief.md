## Profile Brief for Round 0 (Baseline)

### Source
- Kernel: kernel-evolve/examples/kernels/chunk_gla.py
- Speedup: 0.885x | Latency: 8690.57ms (reference: 7687.41ms)
- Compute ratio: 1.0 | Memory transfer ratio: 0.0
- **Note**: Baseline Pallas kernel is SLOWER than pure JAX reference (which uses lax.scan + einsum)

### Hardware Utilization Summary
| Unit | Utilization % | Assessment |
|------|--------------|------------|
| MXU  | 22.53% | medium — underutilized even with only 1 MXU active |
| Scalar ALU | 0.005% | negligible — not a bottleneck |
| Vector ALU | 17.10% | medium — significant VPU work (gating, exp, masking) |
| Vector Load | 0.001% | negligible |
| Vector Store | 5.49% | low — some store traffic |
| Register fills/spills | 2,363,328 / 1,907,895 | **CRITICAL** — massive register pressure |

### Deep Profiling Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | 8270 | high — complex multi-kernel program |
| MXU dual ratio | 0.0 | **CRITICAL** — only mxu0 used, mxu1 completely idle |
| MXU0 ops | 4656 | 100% of MXU work on one unit |
| MXU1 ops | 0 | 0% — second MXU entirely wasted |
| HBM bandwidth | N/A | not measured |
| Arithmetic intensity | N/A | not measured |
| Compute efficiency | N/A | not measured |
| DMA transfers | 24 | double_buffered: no |
| DMA syncs | 0 | no explicit sync points |
| Pipeline NOPs | 0 | low — no stall bubbles |
| Xlane ops | 32 | cross-lane communication present |

### Bottleneck Diagnosis
**Primary bottleneck**: Single-MXU + register pressure
**Evidence**: MXU dual_ratio = 0.0 means only one of two matrix units fires — 50% compute capacity wasted. Simultaneously, 1.9M register spills indicate massive VMEM pressure from large intermediates in the fused backward kernel (128×128 tiles for q, k, v, g, h, A, do, dh simultaneously). The 8270 VLIW bundles are high, suggesting the compiler struggles to schedule efficiently with this many live values.
**Combined patterns**: Low MXU utilization (22.53%) on a single MXU + high register spills = the kernel has too many large intermediates that prevent the compiler from packing MXU ops densely. The backward kernel loads 8 refs and produces 4 outputs, all at 64×128 or 128×128 tile sizes.

### LLO Key Observations
LLO not available (no GCS artifact download). Analysis based on profiler counters only.

### HLO Key Observations
HLO not available (no GCS artifact download). Fusions: 0 (ideal for Pallas single-kernel).

### Optimization Priorities (derived from profile)
1. **Register pressure reduction**: With 1.9M spills, the fused backward kernel is the likely culprit — it processes 8 input refs + 4 outputs in a single kernel. Splitting the backward into smaller kernels or reducing tile sizes from 128 to 64 would dramatically reduce live values and enable better VLIW scheduling. This is the #1 priority because register pressure cascades: it forces VMEM traffic, blocks MXU scheduling, and inflates bundle count.
2. **MXU dual scheduling**: dual_ratio = 0.0 is catastrophic — half the compute silicon is idle. After reducing register pressure, the compiler may auto-schedule across both MXUs. Alternatively, restructuring dot products to use independent operands (e.g., computing dq and dk simultaneously from different inputs) could enable dual-MXU.
3. **Kernel fusion / orchestration**: The current design has 4 separate Pallas kernels + 1 lax.scan, each with its own launch overhead and HBM round-trips. Fusing the intra-chunk attention into the output kernel (they share q, g_cumsum) would reduce kernel launches and HBM traffic.

### What NOT to try (profile evidence)
- **Scalar ALU optimization**: At 0.005% utilization, scalar compute is negligible — don't simplify control flow.
- **Adding more pipelining/prefetch to the existing kernels**: With 1.9M register spills, adding pipeline state would worsen register pressure. Fix spills first.
- **Larger block sizes**: The 128×128 tiles are already causing register pressure. Don't try 256×256 — it would OOM on VMEM.
