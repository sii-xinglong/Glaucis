## Round 1 Strategy

Generating 5 variants from baseline (1.388x, 4-step forward unrolling), each exploring a different technical direction derived from profile analysis.
Variants generated in parallel via sub-agents.

### Variant: eliminate_chunk_fwd_h
**Technical direction**: Eliminate separate chunk_fwd_h pallas_call
**Profile motivation**: 3 pallas_calls total; fused forward already computes h in VMEM scratch but doesn't output it. chunk_fwd_h redundantly recomputes h.
**Approach**: Add h as output Ref of fused forward kernel, store h state at each sub-step boundary (BEFORE update). Eliminate chunk_fwd_h call from _fwd. h_buf BlockSpec: (1, 4, 1, BK, BV) with 4*t mapping.
**Expected impact**: Remove 1 of 3 pallas_calls. FP45 said this had zero impact at 1.0x baseline, but at 1.388x the balance may differ.
**Target metric improvement**: Reduce total computation events by ~33% (eliminate chunk_fwd_h events)
**Key changes**: Added h_ref output to forward kernel, h_out_map, spec_h_out; modified _fwd to use fused forward h; removed chunk_fwd_h call.

### Variant: bwd_monolithic
**Technical direction**: Merge backward 2-pass into single fused kernel
**Profile motivation**: Current backward has 2 pallas_calls (Pass 1: dv+dh, Pass 2: dq+dk) with dh_states intermediate tensor [B,H,NT,K,V] in HBM. Monolithic backward eliminates dh_states HBM traffic.
**Approach**: Replace 2-pass backward with single fused backward kernel (like reference _fused_chunk_bwd_kernel). Computes dq, dk, dv and propagates dh in one pass using VMEM scratch.
**Expected impact**: Eliminate 1 backward pallas_call, remove dh_states intermediate. FP53 suggests zero impact but was tested with different forward.
**Target metric improvement**: Reduce backward HBM traffic by eliminating dh_states tensor
**Key changes**: Replaced _bwd_dv_dh_kernel + _bwd_dq_dk_kernel with single _fused_chunk_bwd_kernel; simplified _fused_chunk_bwd_launcher.

### Variant: fwd_pyloop_clean
**Technical direction**: Refactor forward to Python for-loop
**Profile motivation**: 4 manually copy-pasted sub-step blocks (~250 lines repeated). SO21 proves Python for loops compile identically to manual unrolling.
**Approach**: Replace 4 manual sub-steps with `for step in range(4)` Python loop. Cleaner code, same JAX trace-time unrolling.
**Expected impact**: SO21 predicts identical performance. Different code structure may trigger different compiler heuristics.
**Target metric improvement**: Code clarity; possible marginal compilation path difference
**Key changes**: Replaced 4 copy-pasted sub-step blocks with single for loop body.

### Variant: bwd_kv_tiling
**Technical direction**: Phase-separated backward computation
**Profile motivation**: Backward has 9 matmuls per step mixing inter-chunk (h/dh-dependent) and intra-chunk (A-dependent) computations. Separating phases may change data dependency patterns.
**Approach**: Restructure backward kernel into two explicit phases within single kernel: Phase A computes all inter-chunk terms (using h, dh); Phase B computes all intra-chunk terms (using attention matrix A). Different from source-level reordering (FP39) because this changes which values are live simultaneously.
**Expected impact**: May change register allocation patterns. FP39 predicts identical VLIW/MXU but register spills may differ.
**Target metric improvement**: Reduce register spills by narrowing live-value windows
**Key changes**: Reorganized backward computation order to separate inter-chunk and intra-chunk phases.

### Variant: mixed_unroll_6
**Technical direction**: Eliminate chunk_fwd_h + Python for-loop forward (combined)
**Profile motivation**: Combines two optimizations: (1) output h from fused forward to eliminate chunk_fwd_h, (2) use Python for-loop for cleaner forward code. Tests whether the combination provides different compilation outcomes.
**Approach**: 4-step Python for-loop forward kernel that also stores h state at sub-step boundaries. Eliminates separate chunk_fwd_h pallas_call.
**Expected impact**: Combined benefit of eliminate_chunk_fwd_h + fwd_pyloop_clean. Tests interaction effects.
**Target metric improvement**: Reduce pallas_call count + cleaner compilation path
**Key changes**: Python for-loop forward with h_buf output; removed chunk_fwd_h from _fwd.
