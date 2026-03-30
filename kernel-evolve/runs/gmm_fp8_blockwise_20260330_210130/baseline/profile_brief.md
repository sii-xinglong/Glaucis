## Profile Brief for Round 0 (Baseline)

### Source
- Kernel: kernel-evolve/examples/kernels/gmm_fp8_blockwise.py
- Speedup: 1.015x | Latency: 10.698ms
- Compute ratio: 1.0 | Memory transfer ratio: 0.0

### Hardware Utilization Summary
| Unit | Utilization % | Assessment |
|------|--------------|------------|
| MXU  | 0.002% | extremely low — MXU idle nearly all the time |
| Scalar ALU | 0.063% | very low, but 33x higher than MXU — control-flow heavy |
| Vector ALU | 0.014% | very low |
| Vector Load | 0.020% | very low |
| Vector Store | 0.005% | very low |
| Register fills/spills | 160845/160845 | **CRITICAL** — massive register pressure |

### Deep Profiling Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | 4065 | high — complex kernel |
| MXU dual ratio | 1.0 | perfect — both MXUs evenly loaded (36/36 ops) |
| Avg ops/bundle (ILP) | N/A | not available |
| HBM bandwidth | N/A | not measured |
| Arithmetic intensity | N/A | not measured |
| Compute efficiency | 6.68% | of 2307 TFLOPS peak — massive headroom |
| DMA transfers | 35 (8 syncs) | double_buffered: yes |
| Pipeline NOPs | N/A | not measured separately |

### Bottleneck Diagnosis
**Primary bottleneck**: register-pressure + scalar-heavy
**Evidence**: 160,845 vector fills and 160,845 vector spills indicate catastrophic register pressure. The kernel spends the vast majority of time moving data between VMEM and HBM for spills rather than performing useful computation, resulting in only 6.68% compute efficiency. Scalar ALU utilization (0.063%) is 33x higher than MXU utilization (0.002%), suggesting FP8 quantization scale computation and address arithmetic dominate execution time.
**Combined patterns**: The combination of massive register spills with very low MXU utilization despite perfect dual_ratio=1.0 suggests the inner loop body is far too complex — the compiler cannot keep all intermediate values in registers, causing a cascade of spills that stall the pipeline.

### LLO Key Observations

**MXU Scheduling** (36 mxu0 ops, 36 mxu1 ops, dual_ratio=1.0):
```
%2568 = vmatprep.subr.f8e4m3.mxu0 %v689_v35
%2608 = vmatprep.subr.f8e4m3.mxu1 %v689_v35
%2569 = vmatpush3.f8e4m3.xpose.msra.mxu0 %v689_v35
%2612 = vmatpush3.f8e4m3.xpose.msra.mxu1 %v689_v35
%2570 = vmatprep.subr.f8e4m3.mxu0 %v690_v36
%2609 = vmatprep.subr.f8e4m3.mxu1 %v690_v36
%2571 = vmatpush3.f8e4m3.xpose.msra.mxu0 %v690_v36
%2613 = vmatpush3.f8e4m3.xpose.msra.mxu1 %v690_v36
%2576 = vmatprep.mubr.f8e4m3fn.mxu0 %v664_v39
%2577 = vmatmul.mubr.f8e4m3fn.vlgmr.msra.gmra.mrb[0].mxu0 %v665_v40
%2582 = vmatprep.mubr.f8e4m3fn.mxu1 %v666_v41
%2583 = vmatmul.mubr.f8e4m3fn.vlgmr.msra.gmra.mrb[0].mxu1 %v667_v42
```
Good: Both MXUs are co-scheduled with matching operations. FP8 matmul uses vmatprep/vmatpush3/vmatmul sequence on both units.

**Register Spill Pattern** (160,845 fills + 160,845 spills — CRITICAL):
```
%v4022_v43 = vpop.trf.xlu0
%v4023_v43 = vpseudo.spill_to_mem %v4022_v43     // immediate spill after vpop
%s3343_s12 = smov 16
%s4024_s12 = spseudo.remat %s3343_s12, 16        // rematerialization
%v4025_v43 = vpseudo.fill_from_mem %v4023_v43    // fill back immediately
%834 = vrot.lane.b32.xlu1 %v4025_v43, %s4024_s12
%v4026_v44 = vpop.trf.xlu0
%v4027_v44 = vpseudo.spill_to_mem %v4026_v44     // another immediate spill
```
The pattern shows vpop results immediately spilled to memory and then filled back for a single use. This spill-fill-use cycle dominates the inner loop.

**DMA Pattern** (35 transfers, 8 syncs):
```
%86 = dma.hbm_to_vmem [thread:$0]  /*hbm=*/%s77_s27, /*size=*/512, /*vmem=*/...
%229 = dma.hbm_to_vmem [thread:$0]  /*hbm=*/%s220_s14, /*size=*/512, /*vmem=*/...
%251 = dma.hbm_to_vmem [thread:$0]  /*hbm=*/%s3578_s23, /*size=*/256, /*vmem=*/...
...
%3212 = dma.done.wait (%p557_p10), %s564_s3, 512
%3216 = dma.done.wait (%p574_p9), %s581_s29, 2048
```
Double buffering is present but DMA transfers are interleaved with massive spill traffic, reducing effective overlap.

### HLO Key Observations
- Fusions: 0 (ideal for single Pallas kernel)
- VMEM usage: 32.00 MiB allocated, suggesting room for more aggressive VMEM utilization
- Default memory: 512.0 KiB for bf16[2048,64] shapes — intermediate buffers

### Optimization Priorities (derived from profile)
1. **Register pressure reduction**: 160,845 spills is the dominant bottleneck. The forward+backward pass with 3 separate quantization steps (lhs, lhs_t, rhs) creates too many live intermediates. Restructuring to reduce simultaneous live values — e.g., fusing quantization with consumption, or reducing the tiling parameters to need fewer simultaneously-live tiles — should drastically reduce spills.
2. **Quantization overhead reduction**: The per-block absmax calibration (reduce_max → divide → clip → cast) runs on scalar/vector units for each tile. Explore alternative calibration approaches or pre-computed scales to reduce the scalar ALU dominance.
3. **Tiling strategy**: The default (128, 128, 128) tiling for all three phases (fwd gmm, bwd gmm, bwd tgmm) may not be optimal. Different tiling per phase could reduce memory footprint and register pressure while maintaining compute throughput.

### What NOT to try (profile evidence)
- **MXU utilization / dual scheduling**: dual_ratio=1.0 already — both MXUs are perfectly balanced. No benefit from restructuring for MXU parallelism.
- **Additional double buffering**: DMA analysis already shows double_buffering=true. The bottleneck is register spills, not HBM latency hiding.
