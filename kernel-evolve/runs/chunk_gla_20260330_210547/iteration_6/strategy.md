## Round 6 Strategy

Active lineages: 2 (converged to same kernel)
Total variants this round: 5
Variants generated in parallel via sub-agents.

### Lineage L2 (best speedup: 7.845x, direction: fuse_fwd_combined)

#### Variant: L2_bwd_fuse_dh
**Base kernel**: iteration_5/variants/L2_fuse_fwd_combined/kernel.py
**Technical direction**: Backward kernel fusion — merge dh propagation + fused dq/dk/dv into single pallas_call
**Profile motivation**: Backward uses 2 pallas_calls with dh tensor HBM round-trip (256MB). Forward fusion (SO15) yielded +14.8% from same pattern.
**Approach**: Run entire backward in reverse time order in single pallas_call with grid (B, H, K/BK, V/BV, NT). VMEM scratch holds dh state. At each reverse-time step: read dh → compute dq/dk/dv → update dh.
**Expected impact**: Eliminate 1 pallas_call + 256MB HBM traffic. Reduce events from 177 to ~120.
**Target metric improvement**: Computation events -30%, DMA transfers reduced

#### Variant: L2_precision_default
**Base kernel**: iteration_5/variants/L2_fuse_fwd_combined/kernel.py
**Technical direction**: Remove Precision.HIGHEST, use DEFAULT precision
**Profile motivation**: 2.5M register spills. HIGHEST forces f32 accumulators for all matmuls. DEFAULT gives compiler more freedom.
**Approach**: Replace all 14 occurrences of `precision=lax.Precision.HIGHEST` with `precision=lax.Precision.DEFAULT`. No other changes.
**Expected impact**: Compiler may use more register-efficient representations. Risk: correctness (rtol=1e-2, atol=10.0 is relaxed enough).
**Target metric improvement**: Register spills 2.5M → reduced, Vector Store util 9.82% → lower

#### Variant: L2_fori_v_bwd
**Base kernel**: iteration_5/variants/L2_fuse_fwd_combined/kernel.py
**Technical direction**: Use fori_loop to split backward fused kernel into 2 sequential passes
**Profile motivation**: 2.5M register spills from ~24 simultaneously live arrays. FP23 says fori_loop "hides iterations from compiler" unlike manual unrolling.
**Approach**: fori_loop(0, 2, body) with lax.cond: pass 0 computes dv, pass 1 computes dq/dk. Arrays from each pass don't overlap in register lifetime.
**Expected impact**: Halve peak register pressure from ~24 to ~14 simultaneously live arrays.
**Target metric improvement**: Register spills 2.5M → ~1.2M, VLIW bundles reduced

#### Variant: L2_recompute_dh
**Base kernel**: iteration_5/variants/L2_fuse_fwd_combined/kernel.py
**Technical direction**: Eliminate dh HBM traffic by recomputing dh on-the-fly inside fused backward
**Profile motivation**: dh tensor (256MB) makes HBM round-trip between 2 backward kernels. Same pattern as SO15 forward fusion.
**Approach**: Single combined backward pallas_call with grid (B, H, K/BK, V/BV, NT), time as "arbitrary". VMEM scratch holds dh state. Process in reverse time: compute dq/dk/dv using current dh, then update dh state.
**Expected impact**: Reduce backward from 2 pallas_calls to 1. Total kernel from 3 to 2 pallas_calls.
**Target metric improvement**: Computation events -30%, eliminate 256MB HBM traffic

### Lineage L1 (best speedup: 7.639x, direction: fold_dh_fuse_fwd)

#### Variant: L1_bwd_grid_kv
**Base kernel**: iteration_5/variants/L1_fold_dh_fuse_fwd/kernel.py
**Technical direction**: V-only tiling of backward fused kernel grid to reduce per-tile register pressure
**Profile motivation**: 2.5M register spills from processing full V=128 in single tile. V-wide arrays (b_v, b_do, b_h, b_dh) dominate register pressure.
**Approach**: Grid becomes (H, total_NT, V//64=2). V-dependent arrays halve to [BT,64], [K,64]. K stays full (avoids A/dA cross-tile accumulation). VMEM scratch accumulates dq_inter and dk_inter across V tiles.
**Expected impact**: Halve V-dimension register pressure. b_h and b_dh drop from [128,128] to [128,64].
**Target metric improvement**: Register spills 2.5M → ~1.5M, improved VMEM allocation

### Round 6 Risk Assessment

| Variant | Risk | Reward | Category |
|---------|------|--------|----------|
| L2_bwd_fuse_dh | High | High | Backward kernel fusion — complex restructuring |
| L2_precision_default | Low | Medium | Precision reduction — simple change |
| L2_fori_v_bwd | Medium | Medium | fori_loop staging — novel approach for Pallas |
| L2_recompute_dh | High | High | Backward fusion + recompute — most ambitious |
| L1_bwd_grid_kv | Medium | Medium | V-tiling with scratch accumulation |

**High-value bets**: L2_bwd_fuse_dh and L2_recompute_dh attempt the same optimization (backward fusion) with different approaches. If either succeeds, it would be the most significant improvement since SO14.

**Safe bet**: L2_precision_default is a minimal change with low risk.

**Novel approaches**: L2_fori_v_bwd tests whether fori_loop actually reduces register pressure in Pallas (theoretical basis from FP23, never tested). L1_bwd_grid_kv tests V-dimension grid tiling.
