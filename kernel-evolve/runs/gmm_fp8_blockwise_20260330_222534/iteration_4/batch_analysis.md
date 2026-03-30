## Round 4 Batch Analysis

**Variants evaluated**: 5
**Successes**: 5 | **Failures**: 0
**Best speedup this round**: 2.651x (L1_adopt_l2_tiling)
**Overall best speedup**: 2.651x (lineage L1, +7.8% over Round 3's 2.460x)

### Delta vs Baseline
| Metric | Baseline | Round 3 Best | Round 4 Best | Delta R3→R4 |
|--------|----------|-------------|-------------|-------------|
| Speedup | 1.015x | 2.460x | 2.651x | +7.8% |
| Latency | 10.698ms | 4.182ms | 3.898ms | -6.8% |

### Comparative Ranking

| Rank | Variant | Speedup | Latency (ms) | VLIW | MXU Ops | Spills | Compute Eff. | Tiling (fwd, bwd, tgmm) |
|------|---------|---------|-------------|------|---------|--------|-------------|--------------------------|
| 1 | L1_adopt_l2_tiling | 2.651x | 3.898ms | 23,462 | 1,792 | 156 | 18.34% | (256,512,128), (256,512,128), (2048,512,128) |
| 2 | L2_fwd_tk512 | 2.570x | 4.002ms | 23,462 | 1,792 | 231 | 17.86% | (256,512,128), (256,512,128), (2048,512,128) |
| 3 | L2_fwd_tn256 | 2.536x | 4.317ms | 23,462 | 1,792 | 231 | 16.56% | (256,256,256), (256,512,128), (2048,512,128) |
| 4 | L2_fwd_tk512_scale_bf16 | 2.456x | 4.187ms | 23,462 | 1,792 | 159 | 17.07% | (256,512,128), (256,512,128), (2048,512,128) + bf16 scales |
| 5 | L2_tgmm_tn256 | 2.452x | 4.341ms | 24,465 | 1,024 | 156 | 16.47% | (256,256,128), (256,512,128), (2048,512,256) |

### Key Findings

1. **Cross-pollination produces new best**: L1_adopt_l2_tiling (2.651x) combines L1's fwd TK=512 with L2's bwd TM=256+TK=512. The tiling (256,512,128, 256,512,128, 2048,512,128) is now the optimal configuration. All phases use TK=512 except tgmm which uses TK=512 already. All gmm phases use TM=256 (per-group alignment).

2. **L1 vs L2 with same tiling differ**: L1_adopt_l2_tiling (2.651x, 156 spills) and L2_fwd_tk512 (2.570x, 231 spills) use IDENTICAL tiling but produce different results. L1's kernel code path generates fewer spills (156 vs 231), resulting in 3.2% better latency. The subtle code evolution history (different docstrings/variable ordering) affects XLA compilation.

3. **bf16 scales DON'T compound with fwd TK=512**: L2_fwd_tk512_scale_bf16 (2.456x) is WORSE than L2_fwd_tk512 alone (2.570x, -4.4%). The bf16 scale computation changes the quantization code path in a way that interferes with TK=512 compilation. Despite fewer spills (159), compute efficiency drops.

4. **tgmm TN=256 halves MXU ops — regression**: L2_tgmm_tn256 (2.452x) drops MXU ops from 1,792 to 1,024 (-43%) and increases VLIW bundles from 23,462 to 24,465 (+4.3%). The larger N tiles change the tgmm compilation significantly. tgmm TN MUST stay at 128.

5. **Forward TN=256 provides moderate improvement**: L2_fwd_tn256 (2.536x) vs L2 base (2.460x) is +3.1%. Less impactful than fwd TK=512 but still positive.

### Lineage Trends

**L1** (tiling_specialization):
- Round 1: 2.294x → Round 2: 2.335x (+1.8%) → Round 3: stagnant → Round 4: 2.651x (+13.5%)
- **Assessment**: Major breakthrough via cross-pollination. Adopting L2's bwd tiling produced the biggest single-round improvement. L1 is now the clear leader.

**L2** (tiling_specialization):
- Round 1: 1.944x → Round 2: 2.290x → Round 3: 2.460x → Round 4: 2.570x (+4.5%)
- **Assessment**: Steady improvement. fwd TK=512 helped. But L1's kernel code compiles more efficiently.
