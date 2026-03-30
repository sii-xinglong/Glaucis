---
name: analyze
description: Use when analyzing TPU kernel evaluation results — parses eval_result.json, classifies bottlenecks, compares with history, writes analysis.md
---

# Analyze Evaluation Results

Parse the evaluation result from a TPU kernel run, classify performance bottlenecks, compare with previous iterations, and write a structured analysis.

## Context

Invoked by `pallas-evolve:start` after `pallas-evolve:submit`, or standalone for debugging. Expects `iteration_{N}/eval_result.json` to exist in the run directory.

## Procedure

### Step 1: Read eval result

Read `iteration_{N}/eval_result.json`. The JSON has this structure:

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

### Step 2: Classify result

**If status is COMPILE_ERROR:**
- Report the error message
- Identify the likely cause:
  - `SyntaxError` → Python syntax issue in kernel code
  - `TypeError` / `ValueError` → Wrong argument types or shapes
  - `Mosaic` or `MLIR` in error → TPU compiler issue (likely dtype or API misuse)
  - `ResourceExhausted` → VMEM/memory overflow (block size too large)
- Suggest specific fix based on the error
- Write to analysis.md and return

**If status is INCORRECT:**
- Report `max_diff` (maximum absolute difference from reference)
- Compare with correctness thresholds (`rtol`, `atol` from config)
- Common causes: integer overflow, wrong accumulator dtype, incorrect tiling boundaries
- Write to analysis.md and return

**If status is SUCCESS:**
- Proceed to performance analysis (Step 3)

### Step 3: Performance analysis (SUCCESS only)

Extract primary metrics:
- `speedup`: ratio of reference_latency / kernel_latency. >1.0 means faster than baseline.
- `latency_ms`: absolute kernel latency
- `compute_ratio`: fraction of time in useful computation (0.0-1.0). Higher is better.
- `memory_transfer_ratio`: fraction of time in SyncWait/DMA (0.0-1.0). Lower is better.

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

**Multi-signal bottleneck classification:**

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

**Combined diagnosis patterns:**
- **Low arithmetic_intensity + high compute_ratio** → Compute-bound but under-utilizing memory bandwidth. Consider larger tiles or prefetching.
- **Low dual_ratio + compute-bound** → Only one MXU active. Check if matmul dimensions are multiples of 128 for dual-MXU scheduling.
- **Growing vliw_bundle_count + flat speedup** → Kernel complexity bloat. Simplify the algorithm.
- **Low compute_efficiency_pct + high compute_ratio** → Near-peak VPU util but far from peak FLOPS. Check for unnecessary recomputation or register pressure.

### Step 3b: Deep IR analysis (if available)

Check if raw profile artifacts were downloaded to the iteration directory. These provide much richer optimization signals than the scalar metrics alone.

**HLO IR (`iteration_{N}/hlo_post_opt.txt`)**

If this file exists, read it and analyze:

- **Fusion decisions**: Which ops were fused into the `tpu_custom_call`? Were any ops left unfused that could benefit from fusion?
- **Memory layout**: Are there unnecessary transposes, copies, or layout conversions? Check for `transpose`, `copy`, or `bitcast` ops outside the fused region.
- **Parameter shapes**: Verify that the tiling dimensions visible in HLO match the Pallas `BlockSpec` grid. Mismatches indicate suboptimal tiling.
- **Constant folding**: Are there constants that could be folded at compile time but weren't?
- **Redundant ops**: Look for `broadcast`, `reshape`, or `slice` chains that suggest the compiler couldn't simplify the data flow.

**LLO IR (`iteration_{N}/llo_final.txt`)**

If this file exists, read it and analyze:

- **VLIW bundle density**: Are bundles densely packed (3-4 ops per `;;` block) or mostly single-op? Dense bundles mean the compiler is effectively utilizing instruction-level parallelism.
- **MXU scheduling**: Where are `.mxu0`/`.mxu1` ops placed? Long gaps between MXU ops suggest pipeline bubbles. Consecutive MXU ops on both ports (`.mxu0` and `.mxu1` in the same bundle) indicate dual-MXU scheduling.
- **Pipeline stalls**: Look for `nop` instructions or `wait` barriers. Multiple `nop`s in sequence indicate the compiler couldn't fill the pipeline.
- **Register pressure**: Look for store/load patterns to VMEM (`.vmem_store` followed later by `.vmem_load` of the same address) — these indicate register spills.
- **DMA scheduling**: Check for `dma.start` and `dma.done` pairs. Good pipelining has `dma.start` well ahead of the corresponding `dma.done`, overlapping with computation.

**Trace Events (`iteration_{N}/trace_events.json`)**

If this file exists, read it (it may be large — focus on events with `dur > 0` on the TPU device pid) and analyze:

- **Event distribution**: What types of events dominate? Group by `name` and sum durations.
- **Compute vs sync gaps**: Look for long `SyncWait` events between computation events. These represent times the TPU is idle waiting for data.
- **DMA overlap**: Are there DMA transfer events running concurrently with computation events (overlapping `ts` + `dur` ranges)?
- **Iteration consistency**: Are the 3 profiled iterations similar in timing, or is there variance suggesting cold-start effects?

**Include IR-based findings in the analysis.md output.** When IR analysis reveals something the scalar metrics missed (e.g., register spills, missed fusions, pipeline bubbles), highlight it as a concrete optimization target with specific suggestions.

### Step 4: Trend analysis

Read previous iteration results (if any) from `iteration_{N-1}/eval_result.json`, `iteration_{N-2}/eval_result.json`, etc.

Track these metrics across iterations:

| Metric | Good trend | Bad trend |
|--------|-----------|-----------|
| `speedup` | Increasing | Decreasing (regression) |
| `compute_ratio` | Increasing | Decreasing |
| `vliw_bundle_count` | Decreasing or stable | Increasing (complexity bloat) |
| `mxu_utilization.dual_ratio` | Increasing toward 1.0 | Decreasing |
| `arithmetic_intensity` | Increasing | Decreasing (more memory traffic) |
| `compute_efficiency_pct` | Increasing | Decreasing |

Flag any regressions (speedup decreased from previous iteration).

Detect concerning patterns:
- **"All-cost improvement"**: speedup improved but bundle count doubled — likely unsustainable
- **"Diminishing returns"**: each iteration yields <2% improvement — consider trying a different approach
- **"MXU regression"**: dual_ratio dropped after a code change — the change broke MXU scheduling

### Step 5: Generate optimization suggestions

Based on the multi-signal analysis, suggest next steps:

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
- Compare the current kernel with the previous best
- Identify what changed and why it was slower
- Suggest reverting the specific change that caused regression

### Step 6: Write analysis

Write `iteration_{N}/analysis.md`:

```markdown
## Iteration {N} Analysis

**Status**: {SUCCESS/COMPILE_ERROR/INCORRECT}
**Speedup**: {speedup}x (best so far: {best_speedup}x)
**Latency**: {latency_ms}ms

### Performance Profile
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

### Bottleneck
{Multi-signal diagnosis — primary and secondary bottlenecks}

### Trend
{Comparison with previous iterations across all metrics}

### Suggestions
{Specific optimization suggestions based on multi-signal analysis}

### IR Analysis (if available)
{Specific findings from HLO/LLO/trace — e.g., "LLO shows 12 nop sequences averaging 4 nops each, indicating pipeline bubbles between DMA and MXU ops. Consider adding prefetch or increasing tile size to hide latency."}
```
