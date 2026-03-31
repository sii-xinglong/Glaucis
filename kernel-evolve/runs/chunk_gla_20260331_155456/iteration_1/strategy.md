## Round 1 Strategy

Generating 5 variants from baseline (9.066x), each exploring a different HBM memory reduction direction.
Variants generated in parallel via sub-agents.

**Session goal**: Reduce HBM memory usage while maintaining ~9x speedup.

### Variant: bf16_residuals
**Technical direction**: Reduce residual precision
**Profile motivation**: q/k/v stored as f32 residuals = 192MB; inputs are bf16
**Approach**: Store q/k/v as bf16 in _fwd residuals, cast back to f32 in _bwd
**Expected impact**: Residuals 224MB → 128MB (-43%). Minimal speed impact.
**Target metric improvement**: HBM residuals -96MB
**Key changes**: Only _fwd/_bwd wrapper modified (2 lines)

### Variant: eliminate_flip
**Technical direction**: Eliminate backward flip copies
**Profile motivation**: 5 jnp.flip() calls create ~288MB temporary HBM copies
**Approach**: Use reversed BlockSpec index_maps (NT-1-t) instead of pre-flipping arrays. Kernel writes outputs to NT-1-i_t positions.
**Expected impact**: Backward temps ~544MB → ~256MB (-53%). Should maintain speed.
**Target metric improvement**: Peak HBM -288MB from eliminated flip copies
**Key changes**: Backward launcher index_maps + kernel output write positions

### Variant: h_recompute
**Technical direction**: Activation recomputation
**Profile motivation**: h [B,NT,H,K,V] = 32MB residual + 32MB flip copy
**Approach**: Don't store h as residual. Recompute it in backward by re-running chunk_fwd_combined.
**Expected impact**: Residuals 224MB → 192MB (-14%). Adds ~50% forward compute in backward.
**Target metric improvement**: HBM residuals -32MB, backward temps -32MB (no h flip)
**Key changes**: New recompute_h function, _fwd/_bwd wrapper modified

### Variant: reverse_indexing
**Technical direction**: Combined bf16 residuals + flip elimination
**Profile motivation**: Combines two biggest HBM savings
**Approach**: bf16 residuals (saves 96MB) + reversed index_maps (saves ~288MB flip copies)
**Expected impact**: Residuals 128MB + no flip copies. Combined savings ~384MB.
**Target metric improvement**: HBM residuals -96MB, backward temps -288MB
**Key changes**: Both bf16 wrapper changes + backward index_map reversal

### Variant: activation_checkpoint
**Technical direction**: Most aggressive — bf16 + h recompute + flip elimination
**Profile motivation**: Minimizes residual storage to absolute minimum
**Approach**: Store only bf16 q/k/v (48MB total), recompute h in backward, use reversed index_maps
**Expected impact**: Residuals 224MB → 48MB (-79%). Adds forward recompute cost.
**Target metric improvement**: HBM residuals -176MB, backward temps -288MB
**Key changes**: bf16 wrapper + recompute_h + backward index_map reversal
