## Round 5 Strategy

Active lineages: 2
Total variants this round: 5
Variants generated via sub-agent.

### Lineage L1 (best speedup: 2.651x, direction: tiling_specialization)

#### Variant: L1_fwd_tn256
**Base kernel**: iteration_4/variants/L1_adopt_l2_tiling/kernel.py
**Technical direction**: Forward N-tile enlargement
**Profile motivation**: L2_fwd_tn256 showed +3.1% in Round 4 (2.536x vs 2.460x). Untested on L1's more efficient kernel.
**Tiling**: (256, 512, 256, 256, 512, 128, 2048, 512, 128)

#### Variant: L1_bwd_tn256
**Base kernel**: iteration_4/variants/L1_adopt_l2_tiling/kernel.py
**Technical direction**: Backward N-tile enlargement
**Profile motivation**: bwd_gmm TN=128 is the only untested TN parameter. TN=256 halves N-grid for bwd_gmm.
**Tiling**: (256, 512, 128, 256, 512, 256, 2048, 512, 128)

#### Variant: L1_all_tn256
**Base kernel**: iteration_4/variants/L1_adopt_l2_tiling/kernel.py
**Technical direction**: Uniform TN=256 across fwd+bwd
**Profile motivation**: Test compound effect of TN=256 in both gmm phases simultaneously.
**Tiling**: (256, 512, 256, 256, 512, 256, 2048, 512, 128)

#### Variant: L1_tgmm_tk1024
**Base kernel**: iteration_4/variants/L1_adopt_l2_tiling/kernel.py
**Technical direction**: tgmm K-tile doubling
**Profile motivation**: TK=512 proved critical for tgmm (FP11). TK=1024 would further halve K-loop iterations. With bf16 tgmm, no fp8 K constraints.
**Tiling**: (256, 512, 128, 256, 512, 128, 2048, 1024, 128)

### Lineage L2 (best speedup: 2.570x, direction: tiling_specialization)

#### Variant: L2_fwd_tn256_tk512
**Base kernel**: iteration_4/variants/L2_fwd_tk512/kernel.py
**Technical direction**: Forward TN=256 on L2
**Profile motivation**: Compare L1 vs L2 with identical fwd TN=256 + TK=512 to understand code-level compilation differences.
**Tiling**: (256, 512, 256, 256, 512, 128, 2048, 512, 128)
