## Round 1 Selection

### Selected (Top-2)

| Lineage | Variant | Speedup | Direction | Rationale |
|---------|---------|---------|-----------|-----------|
| L1 | tiling_phase_specialization | 2.294x | tiling_specialization | New overall best, 16.2% better than prior session best (1.974x). Phase-specialized tiling doubles MXU throughput. |
| L2 | skip_lhs_t_fwd | 1.944x | quant_reduction | Solid baseline matching SO8. Different approach (skip lhs_t quant) — though F001 shows it compiles identically, the kernel code is cleaner. |

### Not Selected

| Variant | Status | Speedup | Reason |
|---------|--------|---------|--------|
| zero_bwd_quant | SUCCESS | 1.266x | Sub-agent bug: _clamp_tiling re-capped forward tiling to 128. Below top-K. |
| fwd_mixed_precision | INCORRECT | -- | Forward mixed precision (bf16 lhs + fp8 rhs) produces wrong output. New failure pattern. |
| reduce_fwd_quant | INCORRECT | -- | Same forward mixed precision issue. |

### Lineage Updates

- **L1** (NEW): Created from tiling_phase_specialization, best_speedup=2.294x
- **L2** (NEW): Created from skip_lhs_t_fwd, best_speedup=1.944x

### Lineages.json Status

- Active lineages: 2
- Pruned lineages: 0
- Current round: 1
