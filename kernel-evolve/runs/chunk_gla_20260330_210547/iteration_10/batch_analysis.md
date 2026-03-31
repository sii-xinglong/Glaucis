## Round 10 Batch Analysis

**Variants evaluated**: 5
**Successes**: 2 | **Failures**: 3 (COMPILE_ERROR — FP25 bf16+HIGHEST)
**Best speedup this round**: 8.937x (L2_fwd_skip_v_gated_cast)
**Overall best speedup**: 9.054x (lineage L2, R9 L2_bf16_h_residual)
**Previous best**: 9.054x (R9) — **no improvement, stagnant round**

### Comparative Ranking

| Rank | Variant | Status | Speedup | Latency (ms) | Compute Ratio | Bottleneck | Direction |
|------|---------|--------|---------|--------------|---------------|------------|-----------|
| 1 | L2_fwd_skip_v_gated_cast | SUCCESS | 8.937x | 860.5 | 1.0 | MXU pipeline + single-MXU | skip_cast |
| 2 | L2_bf16_v_residual | SUCCESS | 8.907x | 864.1 | 1.0 | MXU pipeline + single-MXU | bf16_v_residual |
| -- | L2_bf16_qk_residual | COMPILE_ERROR | -- | -- | -- | -- | bf16_qk_residual |
| -- | L2_bf16_do_residual | COMPILE_ERROR | -- | -- | -- | -- | bf16_do_residual |
| -- | L2_bf16_all_residuals | COMPILE_ERROR | -- | -- | -- | -- | bf16_all_residuals |

### Per-Variant Details

#### L2_fwd_skip_v_gated_cast (Rank 1)

**Status**: SUCCESS
**Speedup**: 8.937x (down from 9.054x, -1.3% regression)
**Latency**: 860.5ms
**Lineage**: L2 (round 10 variant)

| Metric | Value | vs R9 Best | Assessment |
|--------|-------|------------|------------|
| compute_ratio | 1.0 | unchanged | compute-bound |
| vliw_bundle_count | 9252 | unchanged | identical compiled kernel |
| MXU ops (mxu0) | 5430 | unchanged | same matmul work |
| MXU util (runtime) | 32.41% | -0.01pp | unchanged |
| Vector ALU util | 16.49% | unchanged | identical |
| Vector Store util | 11.84% | -0.02pp | unchanged |
| Vector fills | 3,198,000 | unchanged | identical |
| Vector spills | 2,503,044 | unchanged | identical |

**Bottleneck**: The compiled kernel is IDENTICAL to R9's best (same VLIW, spills, fills, MXU ops). Removing intermediate bf16 casts had zero effect on the compiled output — the Mosaic compiler already optimizes these away. The -1.3% speedup regression is measurement variance between evaluation runs.

**Key learning**: Confirms FP18/FP8 — the Mosaic compiler normalizes intermediate dtype casts. Removing `.astype(b_v.dtype)` calls that are immediately followed by f32 casts produces the exact same compiled kernel.

#### L2_bf16_v_residual (Rank 2)

**Status**: SUCCESS
**Speedup**: 8.907x (down from 9.054x, -1.6% regression)
**Latency**: 864.1ms
**Lineage**: L2 (round 10 variant)

| Metric | Value | vs R9 Best | Assessment |
|--------|-------|------------|------------|
| compute_ratio | 1.0 | unchanged | compute-bound |
| vliw_bundle_count | 9252 | unchanged | identical compiled kernel |
| MXU ops (mxu0) | 5430 | unchanged | same |
| MXU util (runtime) | 32.42% | unchanged | identical |
| Vector fills | 3,198,000 | unchanged | identical |
| Vector spills | 2,503,044 | unchanged | identical |

**Bottleneck**: Same compiled kernel as R9 best. The bf16 v residual storage either: (1) was cast back to f32 by the sub-agent before entering the kernel, making it equivalent to the base, or (2) the v array's bf16 storage saved some HBM traffic but added conversion overhead that cancelled it out. The -1.6% regression is likely measurement noise.

**Key learning**: bf16 v residual provides no measurable improvement. The v array is smaller than h (67MB vs 128MB), so the HBM savings are less impactful. Additionally, the backward kernel may cast v back to f32 immediately, negating any savings.

#### L2_bf16_qk_residual (COMPILE_ERROR)

**Error**: `MosaicError: INTERNAL: Mosaic failed to compile TPU kernel: Bad lhs type`
**MLIR**: `tpu.matmul (vector<64x256xbf16>, vector<256x256xbf16>, vector<64x256xf32>) -> vector<64x256xf32>` with `contract_precision<fp32>`
**Cause**: FP25 — Mosaic rejects bf16 matmul operands when `precision=lax.Precision.HIGHEST` is specified. HIGHEST maps to `contract_precision<fp32>`, which requires f32 operands. When q and k are stored as bf16, the backward kernel computes `q_pos = (b_q * exp_pos[:, None]).astype(b_q.dtype)` which stays bf16, then passes bf16 operands to `jnp.dot(..., precision=HIGHEST)`.
**Fix**: To use bf16 residuals with HIGHEST precision matmuls, ALL operands must be explicitly cast to f32 BEFORE the dot call: `q_pos.astype(jnp.float32)`.

#### L2_bf16_do_residual (COMPILE_ERROR)

**Error**: Same FP25 — `Bad lhs type` with bf16 operands + `contract_precision<fp32>`
**Cause**: Casting do to bf16 before the backward kernel causes `b_do.astype(b_v.dtype)` to stay bf16, and the subsequent `jnp.dot(b_do.astype(b_v.dtype), b_v.T, precision=HIGHEST)` fails because both operands are bf16 with HIGHEST precision.
**Fix**: Same as above — explicit f32 casts before all HIGHEST-precision matmuls.

#### L2_bf16_all_residuals (COMPILE_ERROR)

**Error**: Same FP25 — bf16 operands + HIGHEST precision.
**Cause**: All residuals as bf16 triggers the compile error on the first backward matmul encountered.
**Fix**: Same — must cast to f32 before HIGHEST-precision matmuls.

### Failed Variants Summary

| Variant | Status | Error | Root Cause |
|---------|--------|-------|------------|
| L2_bf16_qk_residual | COMPILE_ERROR | Bad lhs type | FP25: bf16 operands + Precision.HIGHEST incompatible in Mosaic |
| L2_bf16_do_residual | COMPILE_ERROR | Bad lhs type | FP25: same — bf16 do enters backward kernel, stays bf16 through matmul |
| L2_bf16_all_residuals | COMPILE_ERROR | Bad lhs type | FP25: same — all bf16 residuals trigger compile error |

**Common fix for all**: When storing residuals as bf16 to reduce HBM traffic, the backward kernel must explicitly cast ALL matmul operands to f32 BEFORE calling `jnp.dot(..., precision=HIGHEST)`. The `.astype(b_v.dtype)` / `.astype(b_q.dtype)` intermediate casts propagate bf16 to the matmul, which Mosaic rejects. Replace with `.astype(jnp.float32)`.

### Lineage Trends

#### Lineage L2

| Round | Best Variant | Speedup | VLIW | Spills | Fills | MXU Util |
|-------|-------------|---------|------|--------|-------|----------|
| R0 (baseline) | baseline | 0.885x | 8270 | 1.91M | 2.36M | 22.5% |
| R5 | L2_fuse_fwd_combined | 7.845x | 7930 | 2.50M | 3.04M | 33.2% |
| R7 | L2_recompute_dh_v2 | 8.988x | 9420 | 2.72M | 3.26M | 34.6% |
| R8 | L2_eliminate_gcumsum | 9.005x | 9332 | 2.58M | 3.30M | 32.1% |
| R9 | L2_bf16_h_residual | 9.054x | 9252 | 2.50M | 3.20M | 32.4% |
| **R10** | **none improved** | **9.054x** | -- | -- | -- | -- |

**Trend**: First stagnant round for L2 since R8→R9 rebound. The bf16 residual frontier is limited: h residual (SO17, +0.5%) works because it's large (128MB→64MB) and the kernel already handles the bf16→f32 cast. Other arrays hit FP25 (bf16+HIGHEST compile error) when sub-agents didn't add explicit f32 casts.

### Cross-Variant Insights

**FP25 is the dominant failure mode for bf16 residual optimization**: 3 of 5 variants hit the bf16+HIGHEST incompatibility. The fix is straightforward (explicit f32 casts before matmuls) but was not applied by the sub-agents. Future rounds should retry bf16 residuals with the explicit cast fix.

**The compiled kernel is deterministic**: Both successful variants produced the IDENTICAL compiled kernel (same VLIW, MXU ops, fills, spills) as R9's best, confirming that the Mosaic compiler normalizes intermediate dtype casts and the bf16 v residual had no effect on the compiled output.

**Measurement variance is ~1-2%**: The two successful variants show 8.907x and 8.937x vs R9's 9.054x. All three have identical compiled kernels, so the difference is pure measurement noise from varying reference latency and TPU thermal/scheduling state.
