## Profile Brief for Round 4

### Source
- Kernel: iteration_3/variants/L2_bwd_tm256_tk512/kernel.py
- Speedup: 2.460x | Latency: 4.182ms
- Compute ratio: 1.0 | Memory transfer ratio: 0.0

### Delta vs Baseline
| Metric | Baseline | Current Best | Delta |
|--------|----------|-------------|-------|
| Speedup | 1.015x | 2.460x | +142% |
| Latency | 10.698ms | 4.182ms | -60.9% |
| VLIW bundles | 4,065 | 23,462 | +477% |
| MXU ops | 72 | 1,792 | +2389% |
| Register spills | 160,845 | 234 | -99.9% |
| Compute efficiency | 6.68% | 17.09% | +156% |

### Key Observations
- Tiling: (256, 256, 128, 256, 512, 128, 2048, 512, 128)
- Forward TK=256 (while bwd and tgmm both use TK=512)
- Uniform TM=256 for fwd and bwd_gmm (per-group alignment)
- Still 2 forward quantize calls (lhs + rhs)
- scale_dtype=float32 (bf16 showed +2.3% improvement separately)

### Optimization Priorities
1. **Forward TK=512**: L1 proved fwd TK=512 gives +1.8%. L2's current best has fwd TK=256. Combining L2's bwd tiling with fwd TK=512 is the most obvious next step.
2. **Combine scale_bf16**: L2_scale_bf16 showed 2.342x (vs L2 base 2.290x). Combining bf16 scales with the bwd TK=512 winner may compound.
3. **tgmm TN=256**: Current tgmm TN=128. With bf16 tgmm, N can potentially be larger. TN=256 would halve N-grid.
4. **Forward TN=256**: Forward TN is 128. For Gate/Up (N=512), TN=256 halves N-grid from 4 to 2.

### What NOT to try
- tgmm TM=4096: FP13 — causes complexity bloat
- Forward out_dtype=bf16: FP12 — correctness failure
- bwd_gmm TM=1024: FP14 — regresses with TK=512
- Forward mixed precision: FP10 — catastrophic correctness failure
