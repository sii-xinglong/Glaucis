## Round 2 Strategy

Active lineages: 2
Total variants this round: 5
Variants generated directly (tiling parameter changes only — per F001, only parameter changes matter).

### Lineage L1 (best speedup: 2.294x, direction: tiling_specialization)

#### Variant: L1_fwd_tk512
**Base kernel**: iteration_1/variants/tiling_phase_specialization/kernel.py
**Technical direction**: Forward K-tile enlargement
**Profile motivation**: Forward TK=256 means 8 K-iterations for Gate/Up (K=2048). TK=512 would halve this to 4, reducing loop overhead.
**Tiling**: (256, 512, 128, 1024, 256, 128, 2048, 512, 128)
**Target metric improvement**: Fewer K-loop iterations, potentially fewer VLIW bundles

#### Variant: L1_bwd_tm256
**Base kernel**: iteration_1/variants/tiling_phase_specialization/kernel.py
**Technical direction**: bwd_gmm per-group M alignment
**Profile motivation**: Forward TM=256 matching per-group M was the key insight of SO9. Apply same principle to bwd_gmm (currently TM=1024).
**Tiling**: (256, 256, 128, 256, 256, 128, 2048, 512, 128)
**Target metric improvement**: Per-group alignment for bwd_gmm, potentially more MXU ops

#### Variant: L1_tgmm_tk256
**Base kernel**: iteration_1/variants/tiling_phase_specialization/kernel.py
**Technical direction**: tgmm K-tile reduction
**Profile motivation**: Test whether tgmm TK=256 (vs current TK=512) changes performance. Smaller K tiles mean less VMEM per tile but more iterations.
**Tiling**: (256, 256, 128, 1024, 256, 128, 2048, 256, 128)
**Target metric improvement**: Explore tgmm TK sensitivity

### Lineage L2 (best speedup: 1.944x, direction: quant_reduction)

#### Variant: L2_so9_tiling
**Base kernel**: iteration_1/variants/skip_lhs_t_fwd/kernel.py
**Technical direction**: Apply SO9's winning tiling to L2
**Profile motivation**: L2 and L1 have identical code structure (both skip lhs_t quant). L2 uses (1024,256,128) uniform tiling while L1 uses (256,256,128) phase-specialized. Apply L1's winning tiling to L2 to confirm the tiling is the differentiator.
**Tiling**: (256, 256, 128, 1024, 256, 128, 2048, 512, 128)
**Target metric improvement**: Should match L1's 2.294x, confirming tiling is the key factor

#### Variant: L2_bwd_tm256
**Base kernel**: iteration_1/variants/skip_lhs_t_fwd/kernel.py
**Technical direction**: Uniform TM=256 across all phases
**Profile motivation**: Apply per-group M alignment to ALL phases including bwd_gmm.
**Tiling**: (256, 256, 128, 256, 256, 128, 2048, 512, 128)
**Target metric improvement**: Test whether bwd_gmm also benefits from TM=256
