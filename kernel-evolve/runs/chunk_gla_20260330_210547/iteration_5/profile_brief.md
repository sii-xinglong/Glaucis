## Profile Brief for Round 5

### Source
- Kernel: iteration_4/variants/L2_fold_dh_pallas/kernel.py (best from Round 4)
- Speedup: 6.831x | Latency: 1122ms (vs 7662ms reference)
- Compute ratio: 1.0 | Memory transfer ratio: 0.0

### Delta vs Baseline
| Metric | Baseline | Current Best (L2 R4) | Delta |
|--------|----------|---------------------|-------|
| Speedup | 0.885x | 6.831x | +672% |
| Latency | 8691ms | 1122ms | -87.1% |
| Compute ratio | 1.0 | 1.0 | unchanged |
| VLIW bundles | 8270 | 7930 | -4.1% |
| MXU0 ops | 4656 | 4656 | unchanged |
| MXU dual ratio | 0.0 | 0.0 | unchanged — CRITICAL |
| DMA transfers | 24 | 20 | -16.7% |
| Computation events | 264 | 198 | -25.0% |
| Register spills | 1,907,895 | 2,498,118 | +30.9% WORSE |
| Register fills | 2,363,328 | 3,039,582 | +28.6% WORSE |
| MXU util (runtime) | 22.53% | 33.20% | +10.7pp |
| Vector ALU util | 17.10% | 20.19% | +3.1pp |
| Vector Store util | 5.49% | 9.81% | +4.3pp (spill traffic) |

### Hardware Utilization Summary
| Unit | Utilization % | Assessment |
|------|--------------|------------|
| MXU | 33.20% | medium — improved from 22.5% baseline but room for improvement |
| Scalar ALU | 0.042% | negligible |
| Vector ALU | 20.19% | medium — vector computation active |
| Vector Load | 0.006% | negligible |
| Vector Store | 9.81% | elevated — register spill traffic |
| Vector EUP | 2.08% | low — exp() hardware unit |
| Register fills/spills | 3,039,582 / 2,498,118 | **HIGH** — 30.9% worse than baseline |

### Deep Profiling Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | 7930 | -4.1% vs baseline — slightly simpler |
| MXU dual ratio | 0.0 | **CRITICAL** — mxu1 completely idle |
| MXU0 ops | 4656 | all MXU ops on single unit |
| HBM bandwidth | 134MB | low — compute-bound |
| HLO fusions | 0 | ideal for Pallas |
| DMA transfers | 20 | no double buffering |
| Computation events | 198 | -25.0% vs baseline |
| Pipeline NOPs | 0 | no pipeline bubbles |

### LLO Key Observations

**MXU Scheduling** (4656 mxu0 ops, 0 mxu1 ops, dual_ratio=0.0):
All 4656 MXU operations are scheduled on mxu0 only. mxu1 is completely idle. The backward kernel has 8 independent dot products but the compiler schedules them all sequentially on mxu0, likely due to data dependencies between the dot products (each uses outputs from VPU gating operations that feed the next dot).

**DMA Pattern** (20 transfers, 0 syncs):
20 DMA transfers with no explicit sync operations. The backward fused kernel has 7 inputs + 3 outputs = 10 HBM windows, double-buffered (2 buffering levels each per LLO allocations) = 20 DMA transfers.

**Register Pressure**:
2.5M spills indicate significant register pressure in the backward kernel. The fused backward kernel has 7 inputs, 3 outputs, and computes 8 dot products + 3 exp() arrays + 2 masks + multiple intermediate arrays. Peak live values include: q, k, v, g, h, do, dh (inputs) + exp_pos_g, exp_neg_g, exp_gn_minus_g, k_neg, k_decay, q_pos (intermediates) + b_a, b_dA, mask, b_a_masked, b_dA_masked (attention matrices) = at least 17 live [64,128] or [64,64] arrays, totaling ~1.5MB when in float32.

### Bottleneck Diagnosis

**Primary bottleneck**: Register pressure (2.5M spills) + single-MXU (dual_ratio=0.0)

**Evidence**:
- 2.5M register spills with 9.8% Vector Store util dedicated to spill traffic. At 1122ms total latency, spill traffic accounts for a meaningful fraction of execution time.
- dual_ratio=0.0: mxu1 completely idle, halving potential matmul throughput
- MXU util at 33.2% — improved but still far from theoretical max
- All 4656 MXU ops on mxu0 only

**Secondary considerations**:
- The kernel is fully compute-bound (compute_ratio=1.0), so memory bandwidth optimizations won't help.
- The forward pass uses 2 pallas_calls (chunk_fwd_h + chunk_gla_fwd_o_gk). Merging could eliminate ~20 computation events, but the benefit is proportionally smaller now that each event is cheap (pallas_call grid tiles, not lax.scan iterations).
- At 198 computation events, further event reduction has diminishing returns since the lax.scan events (the expensive ones) are already eliminated.

### What has been proven across rounds
1. **lax.scan → pallas_call is the dominant optimization** (SO14): 1.577x → 6.831x (+333%)
2. **Computation event reduction via kernel fusion** (SO11, SO12, SO13): Each 10% → ~25-35% speedup for pallas_call events
3. **Intra-kernel optimizations have diminishing returns** at current optimization level
4. **Register pressure is NOT the primary bottleneck** when event count is high, but BECOMES MORE IMPORTANT as event count decreases
5. **exp(-x) is efficient on TPU EUP** (FP27): don't replace with reciprocal
6. **Mixed bf16/f32 matmul not supported** (FP25): all matmul operands must match
7. **Source-level operation reordering has no effect** (FP18): compiler determines schedule

### Optimization Priorities (derived from profile)

1. **Forward kernel fusion**: Merge chunk_fwd_h + chunk_gla_fwd_o_gk into a single pallas_call. Both R4 attempts (L1_fwd_single_kernel, L2_fuse_fwd_output_h) had fixable implementation bugs. Successful fusion would eliminate ~20 events and one h tensor HBM round-trip. Grid: (B, H, K/BK, V/BV, NT) with time as "arbitrary". VMEM scratch for h state [BK, BV]. Key challenges: (a) the output kernel needs all of q, k, v, g while h-update only needs k, v — index_maps must handle different input/output block shapes, (b) the h-update dot contracts on BT=64 while output dot contracts on K=128.

2. **Register pressure reduction in backward kernel**: The backward fused kernel has 17+ simultaneous live arrays. Possible approaches:
   - Split the backward into two kernels (dv+dk separate from dq) — but FP20 warns this adds launch overhead
   - Reduce intermediate arrays by recomputing instead of storing (but FP19 warns recomputation can increase pressure)
   - Use bfloat16 for ALL matmul inputs consistently (both operands same dtype per FP25)
   - Reduce the number of independent intermediates by combining operations

3. **Dual-MXU scheduling**: dual_ratio=0.0 across ALL rounds. The backward kernel has 8 dot products — some are independent (dv_intra, dv_inter, dq_inter, dk_inter are all independent). If the compiler could co-schedule any two on mxu0+mxu1 simultaneously, MXU throughput could increase up to 2x. Approaches: ensure matmul dimensions are multiples of 128 (already true), try reordering operand layout, try explicitly independent matmul pairs.

4. **BT=128 (larger chunks)**: Currently BT=64 with 64 chunks. BT=128 would halve the grid (32 chunks), halving kernel launch iterations. However, FP24 warns that BT changes affect MXU utilization, and larger BT means larger attention matrices [BT, BT] which could increase register pressure.

### What NOT to try (profile evidence)
- **exp() reduction via reciprocal**: FP27 — reciprocal is slower than exp() on TPU EUP
- **V-tiling / manual unrolling**: FP23 — increases register pressure
- **Kernel splitting for register pressure**: FP20 — launch overhead negates gains
- **Source-level operation reordering**: FP18 — no effect on Mosaic scheduling
- **Mixed bf16/f32 matmul inputs**: FP25 — Mosaic rejects mixed types
- **dynamic_slice in Pallas**: FP22 — not supported
- **Smaller blocks (BT < 64)**: FP24 — underutilizes MXU
