## Round 1 Batch Analysis

**Variants evaluated**: 5
**Successes**: 5 | **Failures**: 0
**Best speedup this round**: 9.058x (bf16_residuals)
**Baseline speedup**: 9.066x
**Session goal**: Reduce HBM memory usage while maintaining ~9x speedup

### Comparative Ranking

| Rank | Variant | Status | Speedup | Delta vs Baseline | Latency (ms) | VLIW Bundles | Spills | Direction |
|------|---------|--------|---------|-------------------|--------------|--------------|--------|-----------|
| 1 | bf16_residuals | SUCCESS | 9.058x | -0.09% | 847.1 | 9332 | 2,576,844 | bf16 residual precision |
| 2 | eliminate_flip | SUCCESS | 9.014x | -0.57% | 855.6 | 9348 | 2,576,844 | flip elimination |
| 3 | reverse_indexing | SUCCESS | 8.969x | -1.07% | 859.3 | 9348 | 2,576,844 | combined bf16 + flip |
| 4 | h_recompute | SUCCESS | 6.567x | -27.6% | 1177.2 | 9332 | 2,577,471 | activation recomputation |
| 5 | activation_checkpoint | SUCCESS | 6.389x | -29.5% | 1199.8 | 9348 | 2,577,471 | aggressive checkpoint |

### Key Finding: HBM Reduction vs Speed Tradeoff

The results clearly partition into two groups:

**Group A — Speed maintained (<1% regression):**
- bf16_residuals, eliminate_flip, reverse_indexing
- These target HBM data layout (residual precision, copy elimination) without adding computation
- Estimated HBM savings: 96MB (bf16), ~288MB (flip), ~384MB (combined)

**Group B — Significant speed regression (>27%):**
- h_recompute, activation_checkpoint
- These add forward recomputation in backward, trading compute for memory
- The 27-30% speed regression far exceeds the expected ~10% cost, suggesting recomputation triggers additional overhead (extra compilation complexity, register pressure)

### Per-Variant Details

#### bf16_residuals (Rank 1)

**Status**: SUCCESS
**Speedup**: 9.058x (vs 9.066x baseline, -0.09%)
**Latency**: 847.1ms
**HBM Savings**: ~96MB (residuals 224MB → 128MB)

| Metric | Value | vs Baseline | Assessment |
|--------|-------|-------------|------------|
| VLIW bundle count | 9332 | unchanged | No complexity change |
| MXU dual_ratio | 0.0 | unchanged | Single-MXU (inherited) |
| MXU util (runtime) | 32.12% | unchanged | Stable |
| Vector ALU util | 16.72% | unchanged | Stable |
| Scalar ALU util | 0.039% | unchanged | Minimal overhead |
| Vector fills/spills | 3,296,400/2,576,844 | unchanged | No register impact |
| DMA transfers | 18 | unchanged | No DMA change |
| HLO fusions | 0 | unchanged | Ideal for Pallas |

**Bottleneck**: Identical to baseline — kernel code is unchanged, only the _fwd/_bwd wrapper stores residuals as bf16 instead of f32. The 0.09% difference is within measurement noise.

**HBM Impact**: Saves 96MB by storing q/k/v residuals as bf16 (16 bytes/element) instead of f32 (32 bytes/element). Residuals: 3 arrays × [2,16,4096,128] × 2 bytes = 96MB (was 192MB). h remains bf16 (32MB unchanged). Total residuals: 128MB (was 224MB).

**Assessment**: Best variant. Maintains full speed with meaningful HBM savings. The bf16→f32 cast in backward is essentially free on TPU.

#### eliminate_flip (Rank 2)

**Status**: SUCCESS
**Speedup**: 9.014x (vs 9.066x baseline, -0.57%)
**Latency**: 855.6ms
**HBM Savings**: ~288MB (eliminated 5 jnp.flip() temporary copies)

| Metric | Value | vs Baseline | Assessment |
|--------|-------|-------------|------------|
| VLIW bundle count | 9348 | +16 (+0.17%) | Negligible increase |
| MXU dual_ratio | 0.0 | unchanged | Single-MXU (inherited) |
| MXU util (runtime) | 32.14% | unchanged | Stable |
| Vector fills/spills | 3,296,400/2,576,844 | unchanged | No register impact |
| DMA transfers | 18 | unchanged | No DMA change |

**Bottleneck**: The +16 VLIW bundles suggest the reversed index computation (`NT-1-t` mapping) adds minimal overhead. The 0.57% slowdown may come from slightly more complex address computation in the backward kernel.

**HBM Impact**: Eliminates 5 jnp.flip() calls that each created a full temporary copy. Estimated savings: 5 arrays × ~58MB average = ~288MB peak HBM reduction during backward pass.

**Assessment**: Good HBM savings with acceptable speed cost. The `jit__flip` is no longer present in process names (confirmed: not in event_names_sample), validating that flip elimination worked.

#### reverse_indexing (Rank 3)

**Status**: SUCCESS
**Speedup**: 8.969x (vs 9.066x baseline, -1.07%)
**Latency**: 859.3ms
**HBM Savings**: ~384MB (combined bf16 + flip elimination)

| Metric | Value | vs Baseline | Assessment |
|--------|-------|-------------|------------|
| VLIW bundle count | 9348 | +16 (+0.17%) | Same as eliminate_flip |
| MXU dual_ratio | 0.0 | unchanged | Single-MXU (inherited) |
| Vector fills/spills | 3,296,400/2,576,844 | unchanged | No register impact |

**Bottleneck**: Combines bf16_residuals and eliminate_flip, but the 1.07% slowdown is greater than either alone (-0.09% + -0.57% = -0.66% expected, got -1.07%). The interaction penalty (~0.4%) may come from bf16→f32 casts happening in the reverse-indexed backward kernel where the compiler can't optimize as effectively.

**HBM Impact**: Combined savings — bf16 residuals (-96MB) + flip elimination (-288MB) = ~384MB total. This is the largest HBM reduction that maintains acceptable speed.

**Assessment**: Most HBM savings among speed-maintaining variants, but the interaction penalty means it's slightly worse than keeping the techniques separate. Still viable.

**IR Analysis (from downloaded artifacts)**:
- HLO: Single `tpu_custom_call` with no extra fusions — ideal
- HLO inputs: f32[2,16,4096,128] × 3 + f32[16] → outputs include f32[2,64,16,128,128]
- LLO: Shows double-buffered input windows (2 buffering levels) for all input operands
- LLO main loop: start=0, step=1, limit=2050 — sequential time dimension scan
- LLO scratch: f32[128,128] = 64KB VMEM for h state propagation
- LLO output windows: 4MB each × 3 outputs = 12MB double-buffered in VMEM
- 24 loop-carried scalar phi nodes — consistent with the multi-output fused kernel

#### h_recompute (Rank 4)

**Status**: SUCCESS
**Speedup**: 6.567x (vs 9.066x baseline, -27.6%)
**Latency**: 1177.2ms
**HBM Savings**: ~64MB (h residual + h flip copy eliminated)

| Metric | Value | vs Baseline | Assessment |
|--------|-------|-------------|------------|
| VLIW bundle count | 9332 | unchanged | Same kernel, additional h-only kernel added |
| MXU util (runtime) | 32.11% | unchanged | Additional kernel runs sequentially |
| Vector fills/spills | 3,297,204/2,577,471 | +804/+627 | Slight register pressure increase |
| Trace window | 3,329ms | +1,235ms (59%) | Significantly longer |

**Bottleneck**: The `jit__flip` still appears in process names — flip copies are NOT eliminated in this variant (only h is recomputed, flips remain for q/k/v). The recomputation runs as a separate `pallas_call`, effectively doubling the forward pass cost. The 27.6% regression (not ~10% as predicted) indicates:
1. The h recomputation kernel is NOT fused with the backward kernel
2. Additional JIT compilation overhead for the extra kernel
3. Sequential execution of recompute_h + backward kernel, preventing overlap

**Assessment**: Not viable. The speed penalty far exceeds the modest HBM savings. The h recomputation approach needs fusion with the backward kernel to be competitive.

#### activation_checkpoint (Rank 5)

**Status**: SUCCESS
**Speedup**: 6.389x (vs 9.066x baseline, -29.5%)
**Latency**: 1199.8ms
**HBM Savings**: ~468MB (maximum — bf16 residuals + h recompute + flip elimination)

| Metric | Value | vs Baseline | Assessment |
|--------|-------|-------------|------------|
| VLIW bundle count | 9348 | +16 | Flip elimination overhead |
| Vector fills/spills | 3,297,204/2,577,471 | +804/+627 | Same as h_recompute |

**Bottleneck**: Inherits h_recompute's regression plus the flip elimination overhead. The combined 29.5% regression is dominated by the h recomputation cost.

**Assessment**: Not viable at current speed cost. Maximum HBM savings but unacceptable speed regression.

### Failed Variants Summary

No failures — all 5 variants compiled and passed correctness.

### Observations for Next Round

1. **bf16 residuals are essentially free**: The 0.09% cost is within noise. This should be adopted as the new baseline.

2. **Flip elimination works but has a small cost**: The 0.57% overhead is from reversed index computation. Acceptable tradeoff for ~288MB savings.

3. **Combining bf16 + flip has an interaction penalty**: The combined variant is 1.07% slower, slightly more than the sum of individual penalties (0.66%). May warrant investigation.

4. **h recomputation is too expensive without fusion**: The separate kernel approach adds 39% latency. Fusing h recomputation into the backward kernel could reduce this overhead significantly.

5. **All variants share identical kernel structure**: VLIW counts, MXU patterns, and register pressure are essentially unchanged. The HBM reductions come from wrapper-level changes (dtype, flip elimination), not kernel-level optimization.

6. **Next directions to explore**:
   - Apply bf16_residuals as the new baseline and explore further reductions on top
   - Investigate fusing h recomputation into the backward kernel (avoid the separate pallas_call)
   - Explore reducing backward temporary memory through in-place operations
   - Consider output chunking to reduce peak backward HBM
