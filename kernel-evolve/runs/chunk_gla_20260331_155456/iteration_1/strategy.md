## Round 1 Strategy

Generating 5 variants from baseline (9.066x), each exploring a different HBM memory reduction direction derived from profile analysis.
Variants generated in parallel via sub-agents.

**Session goal**: Reduce HBM memory usage while maintaining ~9x speedup.

### Variant: bf16_residuals
**Technical direction**: Residual precision reduction
**Profile motivation**: q/k/v stored as f32 residuals (3 x 64MB = 192MB) despite bf16 inputs
**Approach**: Store q/k/v as bf16 in _fwd residuals, cast to f32 in _bwd
**Expected impact**: -96MB residuals (224MB → 128MB, -43%)
**Target metric improvement**: No speed change expected; memory only
**Key changes**: _fwd stores bf16, _bwd casts back

### Variant: eliminate_flip
**Technical direction**: Eliminate backward flip copies
**Profile motivation**: 5x jnp.flip() + 3x output flip = ~460MB temporary HBM copies
**Approach**: Reversed BlockSpec index_maps (NT-1-t) instead of flipping arrays. Kernel writes to NT-1-i_t positions.
**Expected impact**: ~460MB temporary allocation savings
**Target metric improvement**: Slight speed improvement possible (less memory traffic)
**Key changes**: Removed all flip calls, reversed index_maps in backward

### Variant: h_recompute
**Technical direction**: Activation recomputation
**Profile motivation**: h [B,NT,H,K,V] at bf16 = 32MB stored as residual
**Approach**: Don't store h; recompute it in backward via a separate h-only pallas_call
**Expected impact**: -32MB residuals (224MB → 192MB, -14%)
**Target metric improvement**: Possible slight speed regression from extra pallas_call
**Key changes**: Added _h_only_kernel + recompute_h, removed h from residuals

### Variant: reverse_indexing
**Technical direction**: Combined bf16 residuals + flip elimination
**Profile motivation**: Combines the two biggest HBM savings opportunities
**Approach**: bf16 residuals (-96MB) + reversed BlockSpec indexing (~460MB flip savings)
**Expected impact**: -96MB residuals + ~460MB temporaries = ~556MB total savings
**Target metric improvement**: No speed regression expected
**Key changes**: bf16 _fwd, reversed index_maps, no flips

### Variant: activation_checkpoint
**Technical direction**: Maximum residual reduction (bf16 + h recompute + flip elimination)
**Profile motivation**: Most aggressive — only store bf16 q/k/v (48MB total, -79% from 224MB)
**Approach**: All three: bf16 residuals + h recomputation + reversed indexing
**Expected impact**: -176MB residuals + ~460MB temporaries = ~636MB total savings
**Target metric improvement**: Possible speed regression from h recomputation pallas_call
**Key changes**: bf16 _fwd, h recompute, reversed index_maps, no flips
