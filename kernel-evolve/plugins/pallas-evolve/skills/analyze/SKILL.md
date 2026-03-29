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
  "compute_ratio": 0.85,
  "memory_transfer_ratio": 0.15,
  "metadata": {}
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

Extract key metrics:
- `speedup`: ratio of reference_latency / kernel_latency. >1.0 means faster than baseline.
- `latency_ms`: absolute kernel latency
- `compute_ratio`: fraction of time spent on useful computation (0.0-1.0). Higher is better.
- `memory_transfer_ratio`: fraction of time spent on memory transfers / sync waits (0.0-1.0). Lower is better.

**Bottleneck classification:**

| compute_ratio | Classification | Primary Bottleneck |
|---------------|---------------|-------------------|
| >= 0.75       | Compute-bound | MXU utilization, vectorization |
| 0.50 - 0.75   | Balanced      | Both compute and memory |
| < 0.50        | Memory-bound  | HBM bandwidth, data movement |
| None          | Unknown       | Profiling data not available |

### Step 4: Trend analysis

Read previous iteration results (if any) from `iteration_{N-1}/eval_result.json`, `iteration_{N-2}/eval_result.json`, etc.:

- **Speedup trend**: improving, flat, or regressing?
- **compute_ratio trend**: are we becoming more compute-efficient?
- **Strategy correlation**: which optimization approaches improved things?

Flag any regressions (speedup decreased from previous iteration).

### Step 5: Generate optimization suggestions

Based on the bottleneck classification, suggest next steps:

**Memory-bound (compute_ratio < 0.50):**
- Increase block size to process more data per tile
- Add K-tiling to reduce HBM round-trips
- Use scratch memory for accumulators
- Add pipelining to overlap compute and memory

**Compute-bound (compute_ratio >= 0.75):**
- Optimize inner loop vectorization
- Ensure dimensions are multiples of 128 for MXU
- Try different block aspect ratios
- Consider algorithmic improvements

**Balanced (0.50 - 0.75):**
- Profile deeper: which specific operations are slow?
- Try pipelining to overlap compute and memory
- Adjust block sizes to find the sweet spot

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
**Compute ratio**: {compute_ratio} ({classification})

### Bottleneck
{Description of the primary bottleneck}

### Trend
{Comparison with previous iterations}

### Suggestions
{Specific optimization suggestions for next iteration}
```
