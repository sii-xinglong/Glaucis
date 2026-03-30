## Round 3 Batch Analysis

**Variants evaluated**: 5
**Successes**: 4 | **Failures**: 1
**Best speedup this round**: 2.460x (L2_bwd_tm256_tk512)
**Overall best speedup**: 2.460x (lineage L2, +5.4% over Round 2's 2.335x)

### Delta vs Baseline
| Metric | Baseline | Round 2 Best | Round 3 Best | Delta R2→R3 |
|--------|----------|-------------|-------------|-------------|
| Speedup | 1.015x | 2.335x | 2.460x | +5.4% |
| Latency | 10.698ms | 4.403ms | 4.182ms | -5.0% |

### Comparative Ranking

| Rank | Variant | Speedup | Latency (ms) | VLIW | MXU Ops | Spills | Compute Eff. | Tiling (fwd, bwd, tgmm) |
|------|---------|---------|-------------|------|---------|--------|-------------|--------------------------|
| 1 | L2_bwd_tm256_tk512 | 2.460x | 4.182ms | 23,462 | 1,792 | 234 | 17.09% | (256,256,128), (256,512,128), (2048,512,128) |
| 2 | L2_scale_bf16 | 2.342x | 4.561ms | 23,462 | 1,792 | 231 | 15.67% | (256,256,128), (256,256,128), (2048,512,128) |
| 3 | L1_bwd_tk512 | 2.307x | 4.707ms | 24,673 | 2,112 | 156 | 15.19% | (256,512,128), (1024,512,128), (2048,512,128) |
| 4 | L1_tgmm_tm4096 | 2.277x | 4.705ms | 47,683 | 3,584 | 156 | 15.19% | (256,512,128), (1024,256,128), (4096,512,128) |
| -- | L1_fwd_bf16_out | INCORRECT | -- | -- | -- | -- | -- | out_dtype=bf16 in forward |

### Per-Variant Details

#### L2_bwd_tm256_tk512 (Rank 1) — NEW OVERALL BEST

**Status**: SUCCESS
**Speedup**: 2.460x
**Latency**: 4.182ms
**Lineage**: L2

| Metric | Value | Assessment |
|--------|-------|------------|
| compute_ratio | 1.0 | Compute-bound |
| vliw_bundle_count | 23,462 | Same as Round 2 best — same VLIW complexity |
| MXU dual_ratio | 1.0 | Excellent — both MXUs perfectly utilized |
| MXU ops total | 1,792 | Same as Round 2 best |
| compute_efficiency | 17.09% | Best yet, +5.2% over Round 2 (16.24%) |
| DMA transfers | 21 (double-buffered) | Good |
| Fusions | 0 | Ideal |
| Register fills/spills | 234/234 | Slight increase from 156 but still very low |

**Bottleneck**: Compute-bound with improving compute efficiency. Same VLIW bundle count (23,462) and MXU ops (1,792) as Round 2 best, but latency is 5% lower. The bwd_gmm TK=512 reduces K-loop iterations in the backward pass, and the combination with TM=256 produces a tighter execution schedule.

**Key insight**: bwd_gmm TK=512 matters more when combined with TM=256 than with TM=1024. The L1 variant (TM=1024, TK=512) actually regressed slightly, while L2 (TM=256, TK=512) jumped +7.4% from L2's previous best. Per-group M alignment (TM=256) is the enabling factor for TK=512 to help.

#### L2_scale_bf16 (Rank 2)

**Status**: SUCCESS
**Speedup**: 2.342x
**Latency**: 4.561ms
**Lineage**: L2

| Metric | Value | Assessment |
|--------|-------|------------|
| compute_ratio | 1.0 | Compute-bound |
| vliw_bundle_count | 23,462 | Same complexity |
| MXU ops total | 1,792 | Same |
| compute_efficiency | 15.67% | Slight improvement over L2's previous (15.24%) |
| Register fills/spills | 231/231 | Slight increase |

**Assessment**: bf16 scales provided +2.3% improvement over L2's previous best (2.290x). Same VLIW/MXU profile but slightly better latency. The bf16 scale computation is marginally cheaper. Correctness preserved (within atol=1.0).

#### L1_bwd_tk512 (Rank 3)

**Status**: SUCCESS
**Speedup**: 2.307x
**Latency**: 4.707ms
**Lineage**: L1

| Metric | Value | Assessment |
|--------|-------|------------|
| compute_ratio | 1.0 | Compute-bound |
| vliw_bundle_count | 24,673 | +5.2% increase vs L1 best (23,462) |
| MXU ops total | 2,112 | +17.9% more MXU ops (vs 1,792) |
| compute_efficiency | 15.19% | Decreased from 16.24% |
| Register fills/spills | 156/156 | Same as previous |

**Assessment**: bwd_gmm TK=512 with TM=1024 REGRESSED from L1's best (2.335x → 2.307x, -1.2%). VLIW bloat (+5.2% bundles) and more MXU ops (+17.9%) without speedup indicates the compiler generated a less efficient schedule. TK=512 with TM=1024 creates sub-tiles that don't align well, causing loop overhead.

**Regression cause**: TM=1024 with TK=512 creates 1024/256=4 sub-tile rows × 512/128=4 sub-tile K iterations, generating more complex inner loop code. TM=256 with TK=512 creates exactly 1×4 sub-tiles — simpler and more efficient.

#### L1_tgmm_tm4096 (Rank 4)

**Status**: SUCCESS
**Speedup**: 2.277x
**Latency**: 4.705ms
**Lineage**: L1

| Metric | Value | Assessment |
|--------|-------|------------|
| compute_ratio | 1.0 | Compute-bound |
| vliw_bundle_count | 47,683 | **+103% increase** vs L1 best (23,462) — massive complexity bloat |
| MXU ops total | 3,584 | +100% more MXU ops (vs 1,792) |
| compute_efficiency | 15.19% | Decreased |
| Register fills/spills | 156/156 | Same |

**Assessment**: tgmm TM=4096 REGRESSED from L1's best (2.335x → 2.277x, -2.5%). Despite doubling MXU ops (3,584 vs 1,792), VLIW bundles also doubled (47,683 vs 23,462). The TM=4096 tiles are too large — the compiler generates 2x the code without any latency benefit. This is classic **complexity bloat**: more MXU ops / more VLIW bundles = same latency. The tgmm M dimension is maximally tiled at TM=2048 for bf16.

#### L1_fwd_bf16_out (INCORRECT)

**Error**: max_diff=2065.6875 (atol=1.0)
**Cause**: Forward gmm with out_dtype=jnp.bfloat16 accumulates in bf16 precision instead of f32. The reduced accumulator precision causes significant numerical errors in the matmul reduction (especially for K=2048 where many partial products are summed). The reference uses f32 accumulation.
**Fix**: Forward out_dtype MUST remain jnp.float32. bf16 accumulation is not precise enough for this kernel's matrix sizes.

### Key Findings

1. **bwd_gmm TK=512 + TM=256 is a compound winner**: L2_bwd_tm256_tk512 (2.460x) proves that TK=512 only helps when combined with TM=256 (per-group alignment). L1_bwd_tk512 (TM=1024, TK=512) actually regressed. The per-group M alignment creates cleaner sub-tile structure that benefits from K-loop halving.

2. **scale_dtype=bf16 provides marginal improvement**: L2_scale_bf16 (2.342x vs L2's 2.290x, +2.3%) shows bf16 scales are slightly cheaper without correctness impact. Worth keeping as a secondary optimization.

3. **tgmm TM=4096 causes complexity bloat**: Doubling VLIW bundles (47,683) without latency improvement. tgmm TM is maximally tiled at 2048.

4. **Forward out_dtype=bf16 fails correctness**: max_diff=2065.6875, confirms f32 accumulation is required.

5. **L2 lineage overtakes L1**: L2 now leads at 2.460x vs L1's 2.335x. The key differentiator is bwd_gmm tiling: TM=256 enables optimizations that TM=1024 cannot.

### Lineage Trends

**L1** (tiling_specialization):
- Round 1: 2.294x (SO9 phase-specialized tiling)
- Round 2: 2.335x (+1.8% — fwd TK=512)
- Round 3: 2.307x (-1.2% regression) / 2.277x (-2.5% regression)
- **Assessment**: L1 has peaked. bwd_gmm TK=512 and tgmm TM=4096 both regressed. TM=1024 for bwd_gmm creates sub-optimal sub-tile structures. Stagnant_rounds → 1.

**L2** (tiling_specialization):
- Round 1: 1.944x (skip_lhs_t_fwd)
- Round 2: 2.290x (+17.8% — adopted SO9 tiling with bwd_gmm TM=256)
- Round 3: 2.460x (+7.4% — bwd_gmm TK=512 compound effect)
- **Assessment**: L2 is the clear winner. Uniform TM=256 across fwd/bwd_gmm combined with TK=512 for bwd_gmm produced the best result. Strong improving trend.
