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
      "compute_efficiency_pct": 45.2
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

**Multi-signal bottleneck classification:**

| Signal | Threshold | Diagnosis |
|--------|-----------|-----------|
| `compute_ratio < 0.50` | — | Memory-bound: VPU stalling on DMA/SyncWait |
| `compute_ratio >= 0.75` | — | Compute-bound: VPU busy, optimize ALU ops |
| `arithmetic_intensity < 10` | — | Low arithmetic intensity: too much HBM traffic per FLOP |
| `dual_ratio < 0.5` | — | Single-MXU: only one MXU active, matmul dims may be wrong |
| `compute_efficiency_pct < 10` | — | Very far from peak: major pipeline stalls or register spills |
| `vliw_bundle_count increasing` | vs prior iteration | Complexity bloat: kernel getting more complex without speedup |

**Combined diagnosis patterns:**
- **Low arithmetic_intensity + high compute_ratio** → Compute-bound but under-utilizing memory bandwidth. Consider larger tiles or prefetching.
- **Low dual_ratio + compute-bound** → Only one MXU active. Check if matmul dimensions are multiples of 128 for dual-MXU scheduling.
- **Growing vliw_bundle_count + flat speedup** → Kernel complexity bloat. Simplify the algorithm.
- **Low compute_efficiency_pct + high compute_ratio** → Near-peak VPU util but far from peak FLOPS. Check for unnecessary recomputation or register pressure.

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

### Bottleneck
{Multi-signal diagnosis — primary and secondary bottlenecks}

### Trend
{Comparison with previous iterations across all metrics}

### Suggestions
{Specific optimization suggestions based on multi-signal analysis}
```
