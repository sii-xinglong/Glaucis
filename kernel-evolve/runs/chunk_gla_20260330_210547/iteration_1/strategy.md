## Round 1 Strategy

Generating 5 variants from baseline, each exploring a different technical direction.
Variants generated in parallel via sub-agents.

### Variant: tiling_strategy
**Technical direction**: tiling_strategy
**Profile motivation**: Register fills=2.4M, spills=1.9M (CRITICAL). The fused backward kernel has 8 inputs + 4 outputs causing massive register pressure.
**Approach**: Split the fused backward kernel (`_chunk_gla_bwd_fused_kernel`) into two separate Pallas kernels: `_chunk_gla_bwd_dqdv_kernel` (dq + dv) and `_chunk_gla_bwd_dkdg_kernel` (dk + dg). Each kernel has fewer live values, reducing register pressure. The dg kernel recomputes dA (compute/register trade-off).
**Expected impact**: Halving output pressure per kernel should dramatically reduce spills and enable better VLIW scheduling.
**Target metric improvement**: Register spills 1.9M -> <600K, VLIW bundles 8270 -> <6000
**Key changes**: Backward kernel split into two separate pallas_call invocations; dA recomputed in second kernel.

### Variant: hbm_compute_overlap
**Technical direction**: hbm_compute_overlap
**Profile motivation**: DMA count=24, double_buffering=false. MXU utilization at 22.53% suggests significant time waiting for memory transfers.
**Approach**: Used `pltpu.emit_pipeline` inside `_chunk_fwd_h_kernel` to overlap HBM DMA transfers with MXU compute. The time-step loop is now managed by emit_pipeline's internal scheduler, which prefetches the next k/v tile while computing the current one. Grid reduced from 5D to 4D.
**Expected impact**: HBM latency hidden behind compute should improve MXU utilization without worsening register pressure (compiler-managed buffers).
**Target metric improvement**: MXU utilization 22.53% -> 30%+, latency reduction from HBM overlap
**Key changes**: _chunk_fwd_h_kernel restructured with emit_pipeline; grid dimension for time steps moved inside pipeline.

### Variant: mxu_vpu_overlap
**Technical direction**: mxu_vpu_overlap
**Profile motivation**: MXU=22.53%, Vector ALU=17.10% — both units active but underutilized due to sequential VPU->MXU->VPU chains.
**Approach**: Restructured backward and forward output kernels into explicit phases separating VPU (exp, gating, masking) from MXU (dot products). All VPU pre-compute happens first, then MXU dot products are batched together, then VPU combination ops. This eliminates VPU/MXU alternation in the dependency chain.
**Expected impact**: Better hardware scheduler utilization of both units simultaneously, reducing idle time.
**Target metric improvement**: Combined MXU+VPU throughput increase, latency reduction 10-20%
**Key changes**: Backward kernel reorganized into 6 explicit phases (VPU->MXU->VPU->MXU batch->MXU->VPU); forward output kernel similarly restructured.

### Variant: memory_layout
**Technical direction**: memory_layout
**Profile motivation**: Register spills=1.9M, vector_store=5.49%. High spills from too many live values simultaneously.
**Approach**: Pre-compute gated q and k (`q*exp(g)`, `k*exp(-g)`) once in host and pass to kernels, eliminating duplicate exp() chains. Backward kernel recomputes A inline instead of loading a separate tile. Added VMEM scratch buffers for dq/dk accumulators. Simplified intra-chunk kernel from 3 inputs to 2.
**Expected impact**: Fewer live values in register file, reduced DMA transfers, less spill traffic.
**Target metric improvement**: Register spills 1.9M -> <1M, vector_store 5.49% -> <3%
**Key changes**: Pre-gated values computed once; A tile eliminated from backward kernel; VMEM scratch for accumulators.

### Variant: mxu_utilization
**Technical direction**: mxu_utilization
**Profile motivation**: MXU dual_ratio=0.0 (mxu0=4656 ops, mxu1=0). 50% of matrix compute capacity wasted.
**Approach**: Split the fused backward kernel into `_chunk_gla_bwd_dq_dv_kernel` and `_chunk_gla_bwd_dk_dg_kernel`. Each smaller kernel has independent dot product pairs that the compiler can distribute across both MXUs. Kernel B receives dq as additional input for dg computation.
**Expected impact**: Halved live values per kernel enables dual-MXU scheduling. Independent operand sets within each kernel promote mxu0/mxu1 co-scheduling.
**Target metric improvement**: MXU dual_ratio 0.0 -> 0.5+, MXU utilization 22.53% -> 35%+
**Key changes**: Backward kernel split into two; dq passed from Kernel A to Kernel B for dg; orchestrator function coordinates the two kernels.
