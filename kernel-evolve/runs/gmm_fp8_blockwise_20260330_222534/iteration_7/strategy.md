## Round 7 Strategy

Active lineages: 2
Total variants this round: 5
Variants generated via sub-agent.

### Lineage L2 (best speedup: 2.847x, direction: tiling_specialization)

#### Variant: L2_scale_bf16
**Base kernel**: iteration_6/variants/L2_tgmm_tm1024/kernel.py
**Technical direction**: Cross-pollination SO14+SO15
**Profile motivation**: L2_tgmm_tm1024 has 156 spills; L1_scale_bf16 showed bf16 scales reduce to 9 spills. Combining tgmm TM=1024 (VLIW halving) with bf16 scales (spill elimination) should compound.
**Tiling**: (256, 512, 256, 256, 512, 128, 1024, 512, 128) + scale_dtype=bfloat16

#### Variant: L2_tgmm_tm512
**Base kernel**: iteration_6/variants/L2_tgmm_tm1024/kernel.py
**Technical direction**: Smaller tgmm TM exploration
**Profile motivation**: TM=2048→1024 gave +2.8% with 44.8% VLIW reduction. TM=512 tests if the trend continues — even simpler tgmm tiles.
**Tiling**: (256, 512, 256, 256, 512, 128, 512, 512, 128)

#### Variant: L2_tgmm_tm1024_bwd_tn64
**Base kernel**: iteration_6/variants/L2_tgmm_tm1024/kernel.py
**Technical direction**: Smaller bwd TN exploration
**Profile motivation**: bwd TN=256 caused 9,447 spills (FP16b). If LARGER TN is bad, SMALLER TN might be good. TN=64 halves intermediate buffer vs TN=128. _clamp_tiling lowered to min 64.
**Tiling**: (256, 512, 256, 256, 512, 64, 1024, 512, 128)

### Lineage L1 (best speedup: 2.819x, direction: tiling_specialization)

#### Variant: L1_tgmm_tm1024
**Base kernel**: iteration_6/variants/L1_scale_bf16/kernel.py
**Technical direction**: Cross-pollination SO14+SO15 (L1 path)
**Profile motivation**: L1_scale_bf16 (2.819x, 9 spills) + tgmm TM=1024 (VLIW halving). Same cross-pollination as L2_scale_bf16 but from L1's code path — code-path sensitivity matters (L1 vs L2 produce different results with identical tiling).
**Tiling**: (256, 512, 256, 256, 512, 128, 1024, 512, 128) + scale_dtype=bfloat16

#### Variant: L1_tgmm_tm256_bf16
**Base kernel**: iteration_6/variants/L1_scale_bf16/kernel.py
**Technical direction**: Per-group tgmm TM alignment
**Profile motivation**: TM=256 matching per-group M was the key insight for fwd/bwd (SO9). Testing if the same principle applies to tgmm — each tgmm tile processes exactly one group's worth of data. Combined with bf16 scales.
**Tiling**: (256, 512, 256, 256, 512, 128, 256, 512, 128) + scale_dtype=bfloat16
