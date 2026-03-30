## Round 4 Strategy

Active lineages: 2
Total variants this round: 5
Variants generated in parallel via sub-agents.

### Lineage L2 (best speedup: 2.460x, direction: tiling_specialization)

#### Variant: L2_fwd_tk512
**Base kernel**: iteration_3/variants/L2_bwd_tm256_tk512/kernel.py
**Technical direction**: Forward K-tile enlargement
**Profile motivation**: L1 proved fwd TK=512 gives +1.8%. L2's best still uses fwd TK=256. Applying proven fwd TK=512 to L2's winning bwd tiling.
**Tiling**: (256, 512, 128, 256, 512, 128, 2048, 512, 128) — fwd TK 256→512
**Target metric improvement**: +1-2% from halved forward K-loop

#### Variant: L2_fwd_tk512_scale_bf16
**Base kernel**: iteration_3/variants/L2_bwd_tm256_tk512/kernel.py
**Technical direction**: Forward TK=512 + bf16 scale compound
**Profile motivation**: Combining two individually-proven improvements: fwd TK=512 (+1.8% in R2) and scale_bf16 (+2.3% in R3). The combination may compound.
**Tiling**: (256, 512, 128, 256, 512, 128, 2048, 512, 128) — fwd TK 256→512 + bf16 scales
**Target metric improvement**: +3-4% from compound effect

#### Variant: L2_tgmm_tn256
**Base kernel**: iteration_3/variants/L2_bwd_tm256_tk512/kernel.py
**Technical direction**: tgmm N-tile enlargement
**Profile motivation**: tgmm TN=128 currently. bf16 tgmm (SO6) has no fp8 constraints. TN=256 halves N-grid. FP2 (N>128 with fp8) does not apply.
**Tiling**: (256, 256, 128, 256, 512, 128, 2048, 512, 256) — tgmm TN 128→256
**Target metric improvement**: Fewer tgmm grid iterations

#### Variant: L2_fwd_tn256
**Base kernel**: iteration_3/variants/L2_bwd_tm256_tk512/kernel.py
**Technical direction**: Forward N-tile enlargement
**Profile motivation**: Forward TN=128 currently. For Gate/Up (N=512), TN=256 halves N-grid from 4→2. For Down (N=2048), 16→8. Unexplored dimension.
**Tiling**: (256, 256, 256, 256, 512, 128, 2048, 512, 128) — fwd TN 128→256
**Target metric improvement**: Fewer forward grid iterations

### Lineage L1 (best speedup: 2.335x, direction: tiling_specialization, stagnant: 1)

#### Variant: L1_adopt_l2_tiling
**Base kernel**: iteration_2/variants/L1_fwd_tk512/kernel.py
**Technical direction**: Cross-pollination from L2's winning tiling
**Profile motivation**: L1 uses bwd_gmm TM=1024,TK=256. L2 proved TM=256,TK=512 is superior (+5.4%). Apply L2's winning bwd tiling to L1.
**Tiling**: (256, 512, 128, 256, 512, 128, 2048, 512, 128) — bwd TM 1024→256, TK 256→512
**Target metric improvement**: Should match or exceed L2's 2.460x since L1 has fwd TK=512
