---
name: analyze
description: Use when analyzing batch TPU kernel evaluation results — compares N variants, classifies bottlenecks, selects top-K, updates lineages.json, writes batch_analysis.md
---

# Analyze Batch Evaluation Results

Parse evaluation results from all variants in a batch, classify performance bottlenecks per variant, build a comparative ranking, select top-K for the next round, update lineage tracking, and write structured outputs.

## Context

Invoked by `pallas-evolve:start` after `pallas-evolve:submit` completes all variant evaluations for a round. Expects `iteration_{N}/variants/{variant_name}/eval_result.json` files to exist in the run directory, one per variant.

## Procedure

### Step 1: Read all results

Glob `iteration_{N}/variants/*/eval_result.json` within the run directory. Parse each JSON file.

Each `eval_result.json` has this structure:

```json
{
  "status": "SUCCESS|COMPILE_ERROR|INCORRECT",
  "fitness": 1.5,
  "error": "error message if failed",
  "max_diff": 0.0,
  "latency_ms": 2.3,
  "speedup": 1.5,
  "flops": 1.07e9,
  "compute_ratio": 0.85,
  "memory_transfer_ratio": 0.15,
  "metadata": {
    "reference_latency_ms": 3.45,
    "profile": {
      "vliw_bundle_count": 4302,
      "mxu_utilization": {"mxu0": 1396, "mxu1": 1383, "dual_ratio": 0.99},
      "hbm_bandwidth_bytes": 14680064,
      "arithmetic_intensity": 72.8,
      "flops": 1.07e9,
      "compute_efficiency_pct": 45.2,
      "hbm_bandwidth_utilization_pct": 2.1,
      "vmem_allocation": {"vmem_bytes": 1572864, "smem_bytes": 1024, "allocation_count": 8},
      "bundle_density": {"total_bundles": 4302, "avg_ops_per_bundle": 3.2, "max_ops_per_bundle": 7},
      "dma_analysis": {"dma_count": 42, "dma_sync_count": 20, "double_buffering": true},
      "fusion_analysis": {"fusion_count": 0, "has_cross_program_prefetch": false},
      "special_units": {"xlane_ops": 15, "eup_ops": 8, "nop_count": 3}
    }
  }
}
```

Extract the variant name from the directory path (the directory name under `variants/`).

### Step 2: Per-variant analysis

For each variant, classify status and analyze performance:

**If status is COMPILE_ERROR:**
- Record the error message
- Identify the likely cause:
  - `SyntaxError` -> Python syntax issue in kernel code
  - `TypeError` / `ValueError` -> Wrong argument types or shapes
  - `Mosaic` or `MLIR` in error -> TPU compiler issue (likely dtype or API misuse)
  - `ResourceExhausted` -> VMEM/memory overflow (block size too large)
- Note specific fix suggestion

**If status is INCORRECT:**
- Record `max_diff` (maximum absolute difference from reference)
- Compare with correctness thresholds (`rtol`, `atol` from config)
- Common causes: integer overflow, wrong accumulator dtype, incorrect tiling boundaries

**If status is SUCCESS:**
- Extract primary metrics:
  - `speedup`: ratio of reference_latency / kernel_latency. >1.0 means faster than baseline.
  - `latency_ms`: absolute kernel latency
  - `compute_ratio`: fraction of time in useful computation (0.0-1.0). Higher is better.
  - `memory_transfer_ratio`: fraction of time in SyncWait/DMA (0.0-1.0). Lower is better.

- Extract deep profiling metrics from `metadata.profile` (may be absent if profiling failed):
  - `vliw_bundle_count`: total VLIW bundles in the compiled kernel. Lower = simpler/faster.
  - `mxu_utilization.dual_ratio`: how evenly both MXUs are used (0.0-1.0). 1.0 = perfect.
  - `hbm_bandwidth_bytes`: total HBM bytes read + written per invocation.
  - `arithmetic_intensity`: FLOPs per byte of HBM traffic. Higher = more compute per byte.
  - `compute_efficiency_pct`: actual FLOPS / peak FLOPS as percentage.

- **Multi-signal bottleneck classification:**

  Extract deep profiling metrics from `metadata.profile` (may be absent if profiling failed):
  - `vliw_bundle_count`: total VLIW bundles in the compiled kernel. Lower = simpler/faster.
  - `mxu_utilization.dual_ratio`: how evenly both MXUs are used (0.0-1.0). 1.0 = perfect.
  - `hbm_bandwidth_bytes`: total HBM bytes read + written per invocation.
  - `arithmetic_intensity`: FLOPs per byte of HBM traffic. Higher = more compute per byte.
  - `compute_efficiency_pct`: actual FLOPS / peak FLOPS as percentage.
  - `hbm_bandwidth_utilization_pct`: actual HBM bandwidth / peak bandwidth as percentage. >80% = near bandwidth ceiling.
  - `vmem_allocation.vmem_bytes`: total on-chip VMEM allocated. High values indicate VMEM pressure.
  - `bundle_density.avg_ops_per_bundle`: average operations per VLIW bundle. Higher = better ILP. <2.0 = poor slot utilization.
  - `bundle_density.max_ops_per_bundle`: peak ILP achieved. TPU v7x can do up to 8 ops/bundle.
  - `dma_analysis.dma_count`: total DMA transfer operations. High count may indicate excessive data movement.
  - `dma_analysis.double_buffering`: whether the kernel uses double buffering (iteration % 2 buffer slots). False = DMA cannot overlap with compute.
  - `fusion_analysis.fusion_count`: number of XLA fused_computation blocks. More fusions = more HBM round-trips. Pallas kernels should have 0.
  - `special_units.xlane_ops`: cross-lane reduction operations (XLU). Used for max/sum reductions.
  - `special_units.eup_ops`: hardware exponential unit operations. Used for exp() via vpow2.
  - `special_units.nop_count`: empty VLIW slots. High count indicates pipeline bubbles.

  | Signal | Threshold | Diagnosis |
  |--------|-----------|-----------|
  | `compute_ratio < 0.50` | -- | Memory-bound: VPU stalling on DMA/SyncWait |
  | `compute_ratio >= 0.75` | -- | Compute-bound: VPU busy, optimize ALU ops |
  | `arithmetic_intensity < 10` | -- | Low arithmetic intensity: too much HBM traffic per FLOP |
  | `dual_ratio < 0.5` | -- | Single-MXU: only one MXU active, matmul dims may be wrong |
  | `compute_efficiency_pct < 10` | -- | Very far from peak: major pipeline stalls or register spills |
  | `vliw_bundle_count increasing` | vs prior round's best | Complexity bloat: kernel getting more complex without speedup |

- **Combined diagnosis patterns:**
  - **Low arithmetic_intensity + high compute_ratio** -> Compute-bound but under-utilizing memory bandwidth. Consider larger tiles or prefetching.
  - **Low dual_ratio + compute-bound** -> Only one MXU active. Check if matmul dimensions are multiples of 128 for dual-MXU scheduling.
  - **Growing vliw_bundle_count + flat speedup** -> Kernel complexity bloat. Simplify the algorithm.
  - **Low compute_efficiency_pct + high compute_ratio** -> Near-peak VPU util but far from peak FLOPS. Check for unnecessary recomputation or register pressure.

| Signal | Threshold | Diagnosis |
|--------|-----------|-----------|
| `compute_ratio < 0.50` | — | Memory-bound: VPU stalling on DMA/SyncWait |
| `compute_ratio >= 0.75` | — | Compute-bound: VPU busy, optimize ALU ops |
| `arithmetic_intensity < 10` | — | Low arithmetic intensity: too much HBM traffic per FLOP |
| `dual_ratio < 0.5` | — | Single-MXU: only one MXU active, matmul dims may be wrong |
| `compute_efficiency_pct < 10` | — | Very far from peak: major pipeline stalls or register spills |
| `vliw_bundle_count increasing` | vs prior iteration | Complexity bloat: kernel getting more complex without speedup |
| `avg_ops_per_bundle < 2.0` | — | Low ILP: VLIW slots underutilized, compiler couldn't parallelize |
| `double_buffering == false` | — | No double buffering: DMA and compute cannot overlap |
| `fusion_count > 3` | — | Too many fusions: excessive HBM round-trips between fusions |
| `nop_count > 50` | — | Pipeline bubbles: compiler couldn't fill VLIW slots with useful work |
| `hbm_bandwidth_utilization_pct > 80` | — | Near HBM bandwidth ceiling: memory-bound, reduce data movement |

- Determine a mutation `direction` label for this variant (e.g., `tiling_strategy`, `block_size`, `k_tiling`, `pipelining`, `mxu_scheduling`, `scratch_memory`, etc.) from the variant's strategy.md or directory name.

### Step 3: Comparative analysis

Build a comparison table ranking all variants:

```markdown
| Rank | Variant | Status | Speedup | Latency (ms) | Compute Ratio | Bottleneck | Direction |
|------|---------|--------|---------|--------------|---------------|------------|-----------|
| 1    | k_tile  | SUCCESS | 1.82   | 1.89         | 0.78          | compute    | k_tiling  |
| 2    | big_blk | SUCCESS | 1.45   | 2.38         | 0.62          | balanced   | block_size|
| --   | vec128  | COMPILE_ERROR | -- | --          | --            | --         | vectorize |
```

- Rank SUCCESS variants by speedup descending.
- Place failed variants (COMPILE_ERROR, INCORRECT) below the ranked successes, unranked.

### Step 4: Top-K selection

Select the best variants to continue as lineages:

1. Read `batch.top_k` and `batch.max_active_lineages` from the run config (config.yaml) or from `lineages.json` metadata if present. If not configured, default to `top_k = 3` and `max_active_lineages = 5`.
2. Filter to SUCCESS results only.
3. Sort by speedup descending.
4. Take the top K variants. If fewer than K successes exist, take all successes.
5. Record which variants were selected and which were pruned.

### Step 5: Update lineages.json

Read the current `lineages.json` from the run directory. If it does not exist yet, treat this as Round 1.

**Round 1** (lineages list is empty or file does not exist):

- Create a new lineage entry for each selected variant:
  ```json
  {
    "id": "L1",
    "parent": null,
    "best_speedup": 1.5,
    "best_kernel": "iteration_1/variants/tiling/kernel.py",
    "direction": "tiling_strategy",
    "history": ["iter-1-tiling"],
    "stagnant_rounds": 0
  }
  ```
- Assign sequential IDs: L1, L2, L3, ...
- Add unselected variants to the `pruned` list in lineages.json with their final metrics.
- Set `round` to 1.

**Round 2+** (lineages list is non-empty):

- For each existing active lineage, find its corresponding new variant(s) (matched by lineage ID from variant metadata or directory naming convention).
- If a lineage's new variant improved over its `best_speedup`:
  - Update `best_speedup` to the new value
  - Update `best_kernel` to the new kernel path
  - Append to `history` (e.g., `"iter-3-k_tile"`)
  - Reset `stagnant_rounds` to 0
- If the lineage's variant did not improve (or failed):
  - Increment `stagnant_rounds` by 1
- If new lineages were spawned (e.g., from exploration variants), add them with fresh entries.
- Enforce `max_active_lineages`:
  - If total active lineages exceeds the limit, prune lineages with the highest `stagnant_rounds` first (ties broken by lowest `best_speedup`).
  - Move pruned lineages to the `pruned` list.
- Increment `round`.

Write the updated `lineages.json` back to the run directory. The full structure:

```json
{
  "round": 2,
  "config": {
    "top_k": 3,
    "max_active_lineages": 5
  },
  "lineages": [
    {
      "id": "L1",
      "parent": null,
      "best_speedup": 1.82,
      "best_kernel": "iteration_2/variants/k_tile_v2/kernel.py",
      "direction": "k_tiling",
      "history": ["iter-1-k_tile", "iter-2-k_tile_v2"],
      "stagnant_rounds": 0
    }
  ],
  "pruned": [
    {
      "id": "L3",
      "best_speedup": 1.05,
      "direction": "vectorize",
      "pruned_at_round": 2,
      "reason": "stagnant_3_rounds"
    }
  ]
}
```

### Step 6: Deep IR analysis (if artifacts available)

Check if raw profile artifacts were downloaded for any successful variants. Apply this analysis to the **best variant** (highest speedup) or all successful variants if there are few enough to examine.

**HLO IR (`iteration_{N}/variants/{name}/hlo_post_opt.txt`)**

If this file exists, read it and analyze:
- **Fusion decisions**: Which ops were fused into the `tpu_custom_call`? Were any ops left unfused that could benefit from fusion?
- **Memory layout**: Are there unnecessary transposes, copies, or layout conversions? Check for `transpose`, `copy`, or `bitcast` ops outside the fused region.
- **Parameter shapes**: Verify that the tiling dimensions visible in HLO match the Pallas `BlockSpec` grid. Mismatches indicate suboptimal tiling.
- **Constant folding**: Are there constants that could be folded at compile time but weren't?
- **Redundant ops**: Look for `broadcast`, `reshape`, or `slice` chains that suggest the compiler couldn't simplify the data flow.

**LLO IR (`iteration_{N}/variants/{name}/llo_final.txt`)**

If this file exists, read it and analyze:
- **VLIW bundle density**: Are bundles densely packed (3-4 ops per `;;` block) or mostly single-op? Dense bundles mean the compiler is effectively utilizing instruction-level parallelism.
- **MXU scheduling**: Where are `.mxu0`/`.mxu1` ops placed? Long gaps between MXU ops suggest pipeline bubbles. Consecutive MXU ops on both ports in the same bundle indicate dual-MXU scheduling.
- **Pipeline stalls**: Look for `nop` instructions or `wait` barriers. Multiple `nop`s in sequence indicate the compiler couldn't fill the pipeline.
- **Register pressure**: Look for store/load patterns to VMEM (`.vmem_store` followed later by `.vmem_load` of the same address) -- these indicate register spills.
- **DMA scheduling**: Check for `dma.start` and `dma.done` pairs. Good pipelining has `dma.start` well ahead of the corresponding `dma.done`, overlapping with computation.

**Trace Events (`iteration_{N}/variants/{name}/trace_events.json`)**

If this file exists, read it (it may be large -- focus on events with `dur > 0` on the TPU device pid) and analyze:
- **Event distribution**: What types of events dominate? Group by `name` and sum durations.
- **Compute vs sync gaps**: Look for long `SyncWait` events between computation events. These represent times the TPU is idle waiting for data.
- **DMA overlap**: Are there DMA transfer events running concurrently with computation events (overlapping `ts` + `dur` ranges)?
- **Iteration consistency**: Are the 3 profiled iterations similar in timing, or is there variance suggesting cold-start effects?

Include IR-based findings in the batch_analysis.md output. When IR analysis reveals something the scalar metrics missed (e.g., register spills, missed fusions, pipeline bubbles), highlight it as a concrete optimization target with specific suggestions.

### Step 7: Trend analysis

If this is Round 2+, track per-lineage metrics across rounds. Read previous round results from earlier iteration directories.

| Metric | Good trend | Bad trend |
|--------|-----------|-----------|
| `speedup` | Increasing | Decreasing (regression) |
| `compute_ratio` | Increasing | Decreasing |
| `vliw_bundle_count` | Decreasing or stable | Increasing (complexity bloat) |
| `mxu_utilization.dual_ratio` | Increasing toward 1.0 | Decreasing |
| `arithmetic_intensity` | Increasing | Decreasing (more memory traffic) |
| `compute_efficiency_pct` | Increasing | Decreasing |

Flag per-lineage:
- **Regressions**: speedup decreased from previous round for this lineage
- **Stagnation**: lineage has not improved for 2+ rounds (`stagnant_rounds >= 2`)
- **"All-cost improvement"**: speedup improved but bundle count doubled -- likely unsustainable
- **"Diminishing returns"**: each round yields <2% improvement -- consider a different direction
- **"MXU regression"**: dual_ratio dropped after a code change -- the change broke MXU scheduling

### Step 8: Write outputs

Write two output files:

**`iteration_{N}/batch_analysis.md`** -- the main analysis report:

```markdown
## Round {N} Batch Analysis

**Variants evaluated**: {total_count}
**Successes**: {success_count} | **Failures**: {failure_count}
**Best speedup this round**: {best_speedup}x ({variant_name})
**Overall best speedup**: {overall_best}x (lineage {lineage_id})

### Comparative Ranking

| Rank | Variant | Status | Speedup | Latency (ms) | Compute Ratio | Bottleneck | Direction |
|------|---------|--------|---------|--------------|---------------|------------|-----------|
| ...  | ...     | ...    | ...     | ...          | ...           | ...        | ...       |

### Per-Variant Details

#### {variant_name} (Rank 1)

**Status**: SUCCESS
**Speedup**: {speedup}x
**Latency**: {latency_ms}ms
**Lineage**: {lineage_id} (round {round_in_lineage})

| Metric | Value | Assessment |
|--------|-------|------------|
| compute_ratio | {compute_ratio} | {memory-bound/balanced/compute-bound} |
| vliw_bundle_count | {vliw_bundle_count} | {vs previous: +/-N} |
| MXU dual_ratio | {dual_ratio} | {poor/fair/good/excellent} |
| arithmetic_intensity | {arithmetic_intensity} | {low/medium/high} |
| compute_efficiency | {compute_efficiency_pct}% | {vs peak FLOPS} |
| HBM bandwidth | {hbm_bandwidth_bytes} bytes | {comparison to optimal} |
| HBM BW utilization | {hbm_bandwidth_utilization_pct}% | {near ceiling / headroom} |
| VMEM allocated | {vmem_allocation.vmem_bytes} bytes | {pressure level} |
| Bundle density (avg) | {avg_ops_per_bundle} ops/bundle | {poor/fair/good} |
| DMA transfers | {dma_count} ({double_buffering ? "double-buffered" : "single-buffered"}) | {assessment} |
| Pipeline NOPs | {nop_count} | {low/concerning/high} |
| HLO fusions | {fusion_count} | {0 = ideal for Pallas} |

**Bottleneck**: {Multi-signal diagnosis}
**Suggestions**: {Specific optimization suggestions}

#### {variant_name} (COMPILE_ERROR)

**Error**: {error_message}
**Cause**: {diagnosed cause}
**Fix**: {suggested fix}

(Repeat for each variant)

### Failed Variants Summary

{Table or list of failed variants with error categories and suggested fixes}

### Lineage Trends (Round 2+)

{Per-lineage trend analysis across rounds -- improvements, regressions, stagnation}

### IR Analysis (if available)

{Specific findings from HLO/LLO/trace for the best variant(s)}
```

**`iteration_{N}/selection.md`** -- top-K selection rationale:

```markdown
## Round {N} Selection

### Selected (Top-K)

| Lineage | Variant | Speedup | Direction | Rationale |
|---------|---------|---------|-----------|-----------|
| L1      | k_tile  | 1.82x   | k_tiling  | Best speedup, improving trend |
| L2      | big_blk | 1.45x   | block_size| New direction, promising |

### Not Selected

| Variant | Status | Speedup | Reason |
|---------|--------|---------|--------|
| vec128  | COMPILE_ERROR | -- | Failed to compile |
| scratch | SUCCESS | 1.02x  | Below top-K threshold |

### Lineage Updates

- **L1**: Updated best_speedup 1.50 -> 1.82, reset stagnant_rounds
- **L3**: Pruned (stagnant 3 rounds, best_speedup 1.05x)

### Lineages.json Status

- Active lineages: {count}
- Pruned lineages: {count}
- Current round: {round}
```

### Optimization suggestion categories

When generating suggestions for each variant, use these categories based on the multi-signal analysis:

**Memory-bound (compute_ratio < 0.50):**
- Increase block size to process more data per tile
- Add K-tiling to reduce HBM round-trips
- Use scratch memory for accumulators
- Add pipelining to overlap compute and memory

**Compute-bound (compute_ratio >= 0.75):**
- If `dual_ratio < 0.5`: ensure matmul dimensions allow dual-MXU scheduling (multiples of 128)
- If `compute_efficiency_pct < 20`: look for unnecessary recomputation or register pressure
- Try different block aspect ratios
- Consider algorithmic improvements

**Low arithmetic intensity (< 10 FLOPs/byte):**
- Reduce HBM traffic: keep more data in VMEM via larger tiles
- Eliminate redundant reads/writes
- Use scratch memory to avoid round-trips to HBM

**Complexity bloat (vliw_bundle_count increasing without speedup gain):**
- Revert to simpler kernel structure
- Remove unnecessary conditionals or branching
- Simplify accumulator logic

**Balanced (0.50 - 0.75):**
- Try pipelining to overlap compute and memory
- Adjust block sizes to find the sweet spot
- Profile deeper: which operations are slow?

**Regression detected:**
- Compare the current kernel with the lineage's previous best
- Identify what changed and why it was slower
- Suggest reverting the specific change that caused regression
