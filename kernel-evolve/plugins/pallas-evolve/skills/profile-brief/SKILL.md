---
name: profile-brief
description: Use when generating a profile brief from TPU evaluation artifacts — reads eval_result.json, LLO/HLO/trace files, classifies bottleneck, derives optimization priorities, writes profile_brief.md
---

# Generate Profile Brief from TPU Evaluation Artifacts

Read raw profiling artifacts from a TPU evaluation and generate a structured `profile_brief.md` with bottleneck diagnosis, optimization priorities, and LLO/HLO key observations.

## Arguments

Expects arguments in this format:

```
/pallas-evolve:profile-brief <artifacts_dir> [--baseline <baseline_dir>] [--round <N>]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `artifacts_dir` | Yes | — | Directory containing `eval_result.json` and optional `llo_final.txt`, `hlo_post_opt.txt`, `trace_events.json` |
| `--baseline` | No | — | Baseline artifacts directory for delta comparison (Round 2+) |
| `--round` | No | `0` | Round number for the profile brief header |

**Parse the arguments** from the invocation. Extract:
- `ARTIFACTS_DIR`: the first positional argument (required)
- `BASELINE_DIR`: value after `--baseline` flag, default empty (no delta table)
- `ROUND_N`: value after `--round` flag, default `0`

If `ARTIFACTS_DIR` is missing, print usage and stop.

## Procedure

### Step 1: Read eval_result.json

Read `{ARTIFACTS_DIR}/eval_result.json`. Extract:
- Top-level: `speedup`, `latency_ms`, `compute_ratio`, `memory_transfer_ratio`
- `metadata.hw_utilization`: all unit utilization percentages, fills/spills
- `metadata.profile`: all deep profiling metrics (VLIW bundles, MXU dual_ratio, HBM bandwidth, arithmetic intensity, compute efficiency, bundle density, DMA analysis, etc.)

If `eval_result.json` does not exist, stop with error: "No eval_result.json found in {ARTIFACTS_DIR}."

### Step 2: Read optional LLO file

If `{ARTIFACTS_DIR}/llo_final.txt` exists, read it and extract key excerpts (max 100 lines total):
- **Inner loop body**: Find the main loop (look for loop markers or repeated MXU op sequences) and extract the body showing VLIW bundle structure
- **MXU scheduling**: Find consecutive `.mxu0`/`.mxu1` operations. Show whether both MXUs are co-scheduled in the same VLIW bundles or separated
- **DMA patterns**: Find `dma.start`/`dma.done` pairs. Show whether DMA overlaps with computation
- **Register spills**: Find `.vmem_store`/`.vmem_load` patterns that indicate register pressure
- **Pipeline bubbles**: Find sequences of `nop` instructions

If the file does not exist, note "LLO not available" for the brief.

### Step 3: Read optional HLO file

If `{ARTIFACTS_DIR}/hlo_post_opt.txt` exists, read it and note:
- Number of `fusion` blocks (0 is ideal for Pallas single-kernel)
- Any `transpose`, `copy`, or `bitcast` operations outside the main custom_call
- Shape information from `tpu_custom_call` parameters

If the file does not exist, note "HLO not available" for the brief.

### Step 4: Classify the bottleneck

Use the multi-signal table:
- `compute_ratio < 0.50` → memory-bound
- `compute_ratio >= 0.75` → compute-bound
- `dual_ratio < 0.5` → single-MXU
- `arithmetic_intensity < 10` → low arithmetic intensity
- `vector_spills > 0` → register pressure
- `vmem_utilization_pct < 30` → VMEM underutilized (room to increase block sizes)
- `vmem_utilization_pct > 90` → VMEM near capacity (OOM risk)
- `hbm_capacity_utilization_pct > 50` → high HBM allocation
- `scalar_alu_util_pct > mxu_util_pct` → scalar-heavy
- Check combined patterns (see analyze skill for full table)

### Step 5: Derive optimization priorities

Rank the top 3 optimization directions that the profile data suggests will have the most impact:
- Memory-bound → prioritize K-tiling, scratch memory, double buffering
- Compute-bound + low dual_ratio → prioritize MXU dual scheduling
- Register pressure → prioritize smaller blocks, fewer intermediates
- VMEM underutilized (<30%) + memory-bound → increase block sizes, add scratch memory to improve on-chip data reuse
- VMEM near capacity (>90%) → do not increase blocks, reduce intermediates
- High HBM allocation (>50%) → eliminate redundant buffers, use in-place updates
- Low ILP (avg_ops_per_bundle < 2) → simplify kernel for better VLIW packing
- High scalar ALU → reduce index computation, simplify control flow

### Step 6: Identify what NOT to try

Directions that won't help given the profile:
- If already compute-bound (compute_ratio > 0.8), don't add more pipelining/prefetch
- If dual_ratio > 0.9, don't focus on MXU utilization
- If no register spills and VMEM usage is low, don't reduce block sizes
- If `vmem_utilization_pct > 90`, don't increase block sizes or add scratch buffers
- If `vmem_utilization_pct < 30` and memory-bound, don't reduce block sizes (VMEM has headroom to grow)

### Step 7: Write profile_brief.md

Write the profile brief to `{ARTIFACTS_DIR}/profile_brief.md` using this template:

````markdown
## Profile Brief for Round {ROUND_N}

### Source
- Kernel: {path to source kernel}
- Speedup: {speedup}x | Latency: {latency_ms}ms
- Compute ratio: {compute_ratio} | Memory transfer ratio: {memory_transfer_ratio}

### Hardware Utilization Summary
| Unit | Utilization % | Assessment |
|------|--------------|------------|
| MXU  | {mxu_util_pct}% | {low(<15)/medium(15-40)/high(>40)} |
| Scalar ALU | {scalar_alu_util_pct}% | {high if > mxu = control-flow heavy} |
| Vector ALU | {vector_alu_util_pct}% | {assessment} |
| Vector Load | {vector_load_util_pct}% | {assessment} |
| Vector Store | {vector_store_util_pct}% | {assessment} |
| Register fills/spills | {fills}/{spills} | {0/0 = ideal, >0 = pressure} |

### Deep Profiling Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| VLIW bundle count | {count} | {comparison to baseline if Round 2+} |
| MXU dual ratio | {dual_ratio} | {poor(<0.5)/fair(0.5-0.8)/good(>0.8)} |
| Avg ops/bundle (ILP) | {avg} | {poor(<2)/fair(2-3)/good(>3)} |
| HBM bandwidth | {bytes} | {utilization_pct}% of 3690 GB/s peak |
| Arithmetic intensity | {AI} FLOPs/byte | {low(<10)/medium(10-50)/high(>50)} |
| Compute efficiency | {pct}% | of 2307 TFLOPS peak |
| VMEM utilization | {vmem_pct}% of 64 MiB | {low(<30)/medium(30-70)/good(70-90)/critical(>90)} |
| HBM capacity used | {hbm_cap_pct}% of 192 GB ({peak_memory_mb} MB) | {low/medium/high(>50)} |
| DMA transfers | {count} | {double_buffered: yes/no} |
| Pipeline NOPs | {nop_count} | {low(<10)/medium(10-50)/high(>50)} |

### Bottleneck Diagnosis
**Primary bottleneck**: {memory-bound / compute-bound / register-pressure / scalar-heavy / low-ILP}
**Evidence**: {2-3 sentences citing specific metric values}
**Combined patterns**: {any combined diagnoses from multi-signal analysis}

### LLO Key Observations

**MXU Scheduling** ({mxu0_count} mxu0 ops, {mxu1_count} mxu1 ops, dual_ratio={ratio}):
```
{10-20 lines showing MXU op placement in VLIW bundles}
```

**DMA Pattern** ({dma_count} transfers, {sync_count} syncs):
```
{10-15 lines showing dma.start/dma.done spacing and overlap with compute}
```

**{Additional section if register spills or pipeline bubbles detected}**:
```
{10-15 lines showing the issue}
```

### HLO Key Observations
- Fusions: {count} (0 = ideal for single Pallas kernel)
- {Notable patterns: transposes, copies, shape mismatches}

### Optimization Priorities (derived from profile)
1. **{Direction}**: {Why this is the top priority, citing specific metrics}
2. **{Direction}**: {Why this is second priority}
3. **{Direction}**: {Why this is third priority}

### What NOT to try (profile evidence)
- **{Direction}**: {Why profile shows this won't help, citing metrics}
````

### Step 8: Add delta table (if --baseline provided)

If `BASELINE_DIR` is set, read `{BASELINE_DIR}/eval_result.json` and compare with the current variant's metrics. **Prepend** a delta table section at the top of the profile brief (after the `## Profile Brief` heading):

```markdown
### Delta vs Baseline
| Metric | Baseline | Current Best | Delta |
|--------|----------|-------------|-------|
| Speedup | 1.00x | {x}x | +{pct}% |
| Compute ratio | {base} | {curr} | {+/-} |
| VLIW bundles | {base} | {curr} | {+/-} |
| MXU dual ratio | {base} | {curr} | {+/-} |
| Register spills | {base} | {curr} | {+/-} |
| VMEM utilization | {base}% | {curr}% | {+/-} |
| HBM capacity | {base} MB | {curr} MB | {+/-} |
```

## Deep Profiling Signal Reference

These signals are available in `eval_result.json` under `metadata.profile`:

- `vliw_bundle_count`: Total compiled VLIW bundles. Fewer bundles = simpler kernel = faster. Compare across iterations to detect complexity bloat.
- `mxu_utilization.dual_ratio`: How evenly both MXUs (matrix units) are used. 1.0 = both equally loaded. <0.5 means one MXU is idle — check matmul dimensions.
- `hbm_bandwidth_bytes`: Total HBM memory traffic per invocation. Lower = better. Pallas should keep data in VMEM to avoid HBM round-trips.
- `arithmetic_intensity` (FLOPs/byte): Higher means more compute per byte of memory traffic. Low values indicate memory-bound behavior.
- `compute_efficiency_pct`: Actual throughput vs TPU v7x peak (275 TFLOPS BF16). Shows headroom for optimization.
- `vmem_utilization_pct`: On-chip VMEM usage as % of 64 MiB capacity. Higher is better (more on-chip reuse). <30% = underutilized, >90% = near OOM.
- `hbm_capacity_utilization_pct`: Peak HBM memory usage as % of 192 GB capacity. High values mean large buffer allocations — check for redundant intermediates.

**When analyzing iteration results, check all signals — not just speedup and compute_ratio. VLIW bundle count and MXU dual_ratio are leading indicators of kernel quality.**
