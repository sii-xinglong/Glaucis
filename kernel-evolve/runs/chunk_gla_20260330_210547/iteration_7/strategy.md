## Round 7 Strategy

Active lineages: 2 (both stagnant for 1 round, converged to same kernel architecture)
Total variants this round: 5
Variants generated in parallel via sub-agents.

### Lineage L2 (best speedup: 7.845x, direction: fuse_fwd_combined)

#### Variant: L2_bwd_fuse_dh_v2
**Base kernel**: iteration_5/variants/L2_fuse_fwd_combined/kernel.py
**Technical direction**: Backward kernel fusion — retry R6 L2_bwd_fuse_dh with corrected 5D BlockSpec
**Profile motivation**: Backward uses 2 pallas_calls with dh tensor HBM round-trip (512MB total). Forward fusion (SO15) yielded +14.8% from same pattern.
**Approach**: Merge chunk_bwd_dh_pallas + chunk_gla_bwd_fused into single pallas_call. Grid (B, H, K/BK, V/BV, NT) with "arbitrary" time. VMEM scratch holds dh state. Reverse scan via flipped inputs. **KEY FIX**: h BlockSpec changed from 4D `(1,1,K,V)` to 5D `(1,1,1,BK,BV)` with `h_map(b,h,ki,vi,t) -> (b,h,t,ki,vi)`. Outputs are flat 4D `(B,H,NT*BT,dim)`, flipped after call.
**Expected impact**: Eliminate 1 pallas_call + 512MB HBM traffic. Reduce from 3 to 2 total pallas_calls.
**Target metric improvement**: Computation events -30%, backward latency -15-20%

#### Variant: L2_recompute_dh_v2
**Base kernel**: iteration_5/variants/L2_fuse_fwd_combined/kernel.py
**Technical direction**: Backward fusion with 5D output layout — alternative approach to L2_bwd_fuse_dh_v2
**Profile motivation**: Same as L2_bwd_fuse_dh_v2 — eliminate dh HBM round-trip.
**Approach**: Same fusion goal but different output strategy. Outputs are 5D `(B,NT,H,BT,K)` with block `(1,NT,1,BT,BK)`. Kernel writes to `dq_ref[0, i_t, 0]` using time index directly. **KEY FIX**: h BlockSpec 5D `(1,1,1,BK,BV)` with correct 5-element index_map.
**Expected impact**: Same as L2_bwd_fuse_dh_v2 but different compilation path — the 5D output layout may produce different VLIW scheduling.
**Target metric improvement**: Computation events -30%, backward latency -15-20%

#### Variant: L2_split_dv_dqdk
**Base kernel**: iteration_5/variants/L2_fuse_fwd_combined/kernel.py
**Technical direction**: Split backward fused into dv-only + dq/dk-only kernels to reduce register pressure
**Profile motivation**: 2.5M register spills from ~24 simultaneously live arrays in fused backward kernel.
**Approach**: Replace single bwd_fused with two pallas_calls: (1) dv-only kernel with ~7 peak live arrays, (2) dq/dk kernel with ~10 peak live arrays. A matrix recomputed in both kernels (2x matmul overhead) to avoid HBM round-trip. Grid (H, total_NT) for both.
**Expected impact**: Cut register spills from 2.5M to ~1M per kernel. Extra kernel launch + A recomputation is the cost.
**Target metric improvement**: Register spills 2.5M → ~1M, Vector Store util 9.82% → lower

#### Variant: L2_reduce_intermediates
**Base kernel**: iteration_5/variants/L2_fuse_fwd_combined/kernel.py
**Technical direction**: Restructure backward fused kernel body to reduce peak live intermediates
**Profile motivation**: 2.5M register spills — all exp values and gated variants computed upfront and kept live.
**Approach**: Compute-and-consume in 3 scoped blocks: (1) compute dv, write dv_ref (frees dv intermediates), (2) recompute exp/gated values, compute dq, write dq_ref, (3) recompute again, compute dk, write dk_ref. Each scope has ~13 peak live arrays instead of ~20+. Recomputes A 2x and dA 1x extra.
**Expected impact**: Reduce peak register pressure within single kernel. Unlike FP29 (staged writes only), this genuinely eliminates arrays from later scopes.
**Target metric improvement**: Register spills 2.5M → reduced, maintains single kernel (no extra launch overhead)

### Lineage L1 (best speedup: 7.639x, direction: fold_dh_fuse_fwd)

#### Variant: L1_bwd_grid_v_v2
**Base kernel**: iteration_5/variants/L2_fuse_fwd_combined/kernel.py (L1 converged to same architecture)
**Technical direction**: V-tiling via grid dimension — retry R6 L1_bwd_grid_kv with correct BlockSpec
**Profile motivation**: 2.5M register spills. V-dimension arrays (b_v, b_do, b_dh, b_h) dominate register pressure at V=128.
**Approach**: Expand backward fused grid from (H, total_NT) to (H, total_NT, NV=2). V-dimension arrays use BV_inner=64 sub-tiles via BlockSpec index_map (not reduced block shape — fixes R6 bug). K-dimension arrays unchanged. VMEM scratch accumulates dq_inter and dk_inter across V tiles. dv written independently per tile.
**Expected impact**: Halve V-dimension register footprint. 2x V-tile iterations is the cost.
**Target metric improvement**: Register spills 2.5M → ~1.5M, improved VMEM allocation

### Round 7 Risk Assessment

| Variant | Risk | Reward | Category |
|---------|------|--------|----------|
| L2_bwd_fuse_dh_v2 | Medium | High | R6 BlockSpec fix — approach proven sound |
| L2_recompute_dh_v2 | Medium | High | Alternative output layout for same fusion |
| L2_split_dv_dqdk | Medium | Medium | Kernel splitting — FP20 risk but different context |
| L2_reduce_intermediates | Low | Medium | Intra-kernel restructure — low risk, uncertain gain |
| L1_bwd_grid_v_v2 | Medium | Medium | V-tiling fix — R6 approach corrected |

**Highest-value bets**: L2_bwd_fuse_dh_v2 and L2_recompute_dh_v2 attempt the same backward fusion that yielded the biggest breakthrough (SO14, SO15) for forward. If either compiles successfully, it could yield 10-20% improvement.

**Risk mitigation**: Two backward fusion variants with different output layout approaches — if one hits a compilation issue, the other may succeed.

**Novel approaches**: L2_split_dv_dqdk tests whether kernel splitting is now viable at 7.845x (FP20 was measured at 0.885x). L2_reduce_intermediates tests whether scoped recomputation genuinely reduces register pressure vs. FP29's staged writes.
