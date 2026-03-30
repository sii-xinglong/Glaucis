## Round 3 Strategy

Active lineages: 2
Total variants this round: 5
Variants generated in parallel via sub-agents.

### Lineage L1 (best speedup: 2.335x, direction: tiling_specialization)

#### Variant: L1_fwd_bf16_out
**Base kernel**: iteration_2/variants/L1_fwd_tk512/kernel.py
**Technical direction**: Forward accumulator precision reduction
**Profile motivation**: Compute efficiency is 16.24% with 92.4% non-MXU VLIW bundles. The f32 accumulator uses 2x register space per element. bf16 accumulation halves this and eliminates f32→bf16 conversion for backward.
**Tiling**: (256, 512, 128, 1024, 256, 128, 2048, 512, 128) — unchanged
**Target metric improvement**: Reduced VLIW bundles from accumulator overhead, potentially higher compute_efficiency

#### Variant: L1_bwd_tk512
**Base kernel**: iteration_2/variants/L1_fwd_tk512/kernel.py
**Technical direction**: Backward K-tile enlargement
**Profile motivation**: Forward TK=512 gave +1.8% by halving K-loop iterations. bwd_gmm TK is still 256. For Gate/Up backward (K=2048): 8→4 iterations. For Down backward (K=512): 2→1 iteration.
**Tiling**: (256, 512, 128, 1024, 512, 128, 2048, 512, 128) — bwd_gmm TK changed 256→512
**Target metric improvement**: ~1-2% improvement from reduced loop overhead in backward

#### Variant: L1_tgmm_tm4096
**Base kernel**: iteration_2/variants/L1_fwd_tk512/kernel.py
**Technical direction**: tgmm M-tile enlargement beyond SO6
**Profile motivation**: bf16 tgmm has no fp8 constraints (SO6). TM=2048 gives 4 M-tiles, TM=4096 halves to 2. Fewer tiles = less kernel launch overhead. Risk: accumulator 4096*128*4B = 2MB per tile may cause VMEM pressure.
**Tiling**: (256, 512, 128, 1024, 256, 128, 4096, 512, 128) — tgmm TM changed 2048→4096
**Target metric improvement**: Fewer grid iterations for tgmm, potentially +2-5% if VMEM allows

### Lineage L2 (best speedup: 2.290x, direction: tiling_specialization)

#### Variant: L2_scale_bf16
**Base kernel**: iteration_2/variants/L1_bwd_tm256/kernel.py
**Technical direction**: Forward quantization scale precision reduction
**Profile motivation**: 92.4% of VLIW bundles are non-MXU, dominated by 2 forward qpl.quantize() calls. Scale computation uses float32. bf16 scales reduce tensor size and may simplify VPU operations.
**Tiling**: (256, 256, 128, 256, 256, 128, 2048, 512, 128) — unchanged from L2 base
**Target metric improvement**: Fewer VLIW bundles from simplified scale computation

#### Variant: L2_bwd_tm256_tk512
**Base kernel**: iteration_2/variants/L1_bwd_tm256/kernel.py
**Technical direction**: Backward per-group alignment + K-loop halving
**Profile motivation**: SO9 showed TM=256 matching per-group M doubled MXU ops for forward. Combining TM=256 with TK=512 for bwd_gmm is untested. TK=512 halves K-loop while TM=256 aligns to per-group M.
**Tiling**: (256, 256, 128, 256, 512, 128, 2048, 512, 128) — bwd_gmm TK changed 256→512
**Target metric improvement**: Compound improvement from alignment + K-loop reduction
