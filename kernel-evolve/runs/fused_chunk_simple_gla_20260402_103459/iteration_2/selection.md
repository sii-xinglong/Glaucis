## Round 2 Selection

### Selection Criteria
- Top-K: 2
- Max active lineages: 4
- Metric: speedup (descending)

### Selected Lineages

| Lineage | Variant | Speedup | Direction | Rationale |
|---------|---------|---------|-----------|-----------|
| L1 | L1_scratch_A | 1.4023x | scratch_memory | Highest speedup (tied). Only variant with genuinely different compiled kernel (5499 VLIW vs 5275 baseline). The VMEM scratch A didn't reduce spills but provides a structurally different base — future mutations on this variant have a different compiler starting point. |
| L2 | L2_bwd_fused_arbitrary | 1.4023x | bwd_fused_arbitrary | Highest speedup (tied). Despite compiling identically to baseline, preserves the fused backward architecture as a different evolutionary lineage. Source-level backward fusion may respond differently to future tiling or block-size changes. |

### Not Selected

| Variant | Speedup | Direction | Reason |
|---------|---------|-----------|--------|
| L1_emit_pipeline | 1.4021x | compiler_directives | Compiled identically to baseline. disable_bounds_checks had zero effect (FP56). |
| L2_disable_bounds_checks | 1.4018x | compiler_directives | Compiled identically to baseline. disable_bounds_checks no-op on all kernels (FP56). |
| L1_bf16_h_scratch | 1.3878x | precision_reduction | Regression from L1 best (1.3882x → 1.3878x). bf16 h_states may degrade backward quality. |

### Lineage Updates

- **L1**: Updated best_speedup 1.3882 → 1.4023, best_kernel → iteration_2/variants/L1_scratch_A/kernel.py, stagnant_rounds reset to 0. Note: the +1.0% improvement may be measurement variance between batch evaluations.
- **L2**: Updated best_speedup 1.3881 → 1.4023, best_kernel → iteration_2/variants/L2_bwd_fused_arbitrary/kernel.py, stagnant_rounds reset to 0. Same measurement variance caveat.

### Selection Notes

The ~1% improvement over Round 1 is statistically ambiguous — 4 of 5 variants share identical benchmark arrays (XLA compilation caching), and the compiled kernels are largely identical. The fundamental bottleneck (6.3M register spills, dual_ratio=0.0, VMEM 0.88%) remains completely unchanged.

**New failure pattern identified**: FP56 — `disable_bounds_checks=True` is a no-op for kernels with aligned block sizes (BT=64, BK=128, BV=128 that evenly divide tensor dimensions). The compiler already optimizes away bounds checks for aligned access.

**Round 3 should pivot** to changes that genuinely alter the compiled kernel:
- Block size exploration (reduce BK or BV to lower register pressure)
- Grid dimension changes (different tiling that forces different compiler scheduling)
- Algorithm-level changes (different accumulation patterns, recomputation vs storage)
