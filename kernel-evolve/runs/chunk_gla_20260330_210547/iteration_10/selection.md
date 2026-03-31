## Round 10 Selection

### Selection Criteria
- Top-K: 2, Max active lineages: 4
- Only 1 active lineage (L2), no new lineages to add

### Results

No variant improved over R9's best (9.054x). Both successful variants produced IDENTICAL compiled kernels to R9's best:
- L2_fwd_skip_v_gated_cast: 8.937x (measurement noise, -1.3%)
- L2_bf16_v_residual: 8.907x (measurement noise, -1.6%)

3 variants hit FP25 (bf16 + Precision.HIGHEST compile error): L2_bf16_qk_residual, L2_bf16_do_residual, L2_bf16_all_residuals.

### Lineage Decisions

| Lineage | Action | Speedup | Stagnant Rounds | Rationale |
|---------|--------|---------|-----------------|-----------|
| L2 | RETAIN | 9.054x | 1 (was 0) | Only active lineage, still well above pruning threshold |

### Promoted Variants
None — no improvement over R9's best.

### Pruned Variants
All 5 R10 variants pruned (no improvement):
- L2_fwd_skip_v_gated_cast (8.937x) — identical compiled kernel, measurement noise
- L2_bf16_v_residual (8.907x) — identical compiled kernel, measurement noise
- L2_bf16_qk_residual (COMPILE_ERROR) — FP25
- L2_bf16_do_residual (COMPILE_ERROR) — FP25
- L2_bf16_all_residuals (COMPILE_ERROR) — FP25

### Best Kernel
Unchanged: `iteration_9/variants/L2_bf16_h_residual/kernel.py` (9.054x)

### Key Takeaway for Round 11
The bf16 residual frontier is partially mapped:
- h residual: WORKS (SO17, +0.5%)
- v residual: NO EFFECT (compiler normalizes intermediate casts)
- q/k/do residuals: COMPILE ERROR (FP25 — sub-agents didn't add explicit f32 casts before HIGHEST-precision matmuls)

**Round 11 priority**: Retry bf16 q/k/do residuals with explicit `.astype(jnp.float32)` casts before ALL `jnp.dot(..., precision=HIGHEST)` calls. This directly addresses the FP25 failures and could unlock ~136MB HBM savings. Also explore fundamentally different directions (algorithmic restructuring, matmul count reduction) since bf16 residuals have limited headroom.
