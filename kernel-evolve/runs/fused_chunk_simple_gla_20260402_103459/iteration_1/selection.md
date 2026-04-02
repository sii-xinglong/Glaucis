## Round 1 Selection

### Selection Criteria
- Top-K: 2
- Max active lineages: 4
- Metric: speedup (descending)

### Selected Lineages

| Lineage | Variant | Speedup | Direction | Rationale |
|---------|---------|---------|-----------|-----------|
| L1 | bwd_kv_tiling | 1.3882x | bwd_phase_separation | Highest speedup (marginal). Identical compiled output to L2 but preserves split backward architecture which may respond differently to future optimizations. |
| L2 | bwd_monolithic | 1.3881x | bwd_fusion | Second highest. Monolithic backward architecture provides a structurally different base for future mutations despite compiling identically this round. |

### Pruned Variants

| Variant | Speedup | Direction | Reason |
|---------|---------|-----------|--------|
| fwd_pyloop_clean | 1.3789x | fwd_code_cleanup | No improvement. Compiled identically to manual unrolling (confirms SO21). |
| eliminate_chunk_fwd_h | 1.3789x | fwd_h_elimination | No improvement. +23% VLIW, +25% peak memory. Adding h output increased complexity (confirms FP45). |
| mixed_unroll_6 | 1.3789x | combined_fwd_h_loop | No improvement. Worst peak memory (+50%). Combined dead-end optimizations. |

### Selection Notes

All 5 variants are within noise of the baseline template (1.388x). No genuine improvement was achieved this round. The selected lineages preserve two structurally different backward architectures as bases for Round 2, though both compiled to identical output in this round.

**Critical observation**: Source-level restructuring has exhausted its potential. Round 2 must explore fundamentally different approaches:
- Compiler-directive-level changes (scratch memory, explicit VMEM management)
- Tiling parameter changes (block sizes)
- DMA/compute overlap (double buffering, prefetch)
