# Benchmark Infrastructure Improvement: XProf-Based Device Timing

**Date:** 2026-03-31
**Status:** Draft
**Motivation:** Adopt best practices from [tokamax](https://github.com/openxla/tokamax) benchmarking infrastructure to improve measurement accuracy and metric richness.

## Problem

The current `stage_performance()` in `evaluate.py` uses `time.perf_counter()` wallclock timing with `block_until_ready()`. This includes Python overhead and scheduling latency, producing measurements that don't reflect true device execution time. Additionally, `stage_performance()` and `stage_profile()` run as separate stages, executing xprof traces redundantly.

## Solution

Merge `stage_performance()` and `stage_profile()` into a single `stage_benchmark()` function that:

1. Isolates and times compilation separately (lowering + compile)
2. Uses hermetic xprof (`.xplane.pb` parsing) for device-side timing
3. Tracks peak device memory
4. Reports statistical spread (min, max, stddev, coefficient of variation)
5. Eliminates redundant xprof trace sessions

## Scope

- TPU only (no GPU/CUPTI paths)
- Approach A: Minimal Merge — changes stay within `evaluate.py` and `evaluator.py`, no new files
- Backward compatible — existing `latency_ms`, `speedup`, `compute_ratio` fields preserved

## Design

### 1. BenchmarkData Dataclass (evaluator.py)

```python
@dataclass
class BenchmarkData:
    lower_time_ms: float                    # JAX -> HLO lowering time
    compile_time_ms: float                  # HLO -> executable compilation time
    evaluation_times_ms: tuple[float, ...]  # per-iteration device times from xprof
    peak_memory_mb: float | None            # peak device memory via memory_analysis()

    @property
    def median_ms(self) -> float:
        return float(np.median(self.evaluation_times_ms))

    @property
    def min_ms(self) -> float:
        return float(np.min(self.evaluation_times_ms))

    @property
    def max_ms(self) -> float:
        return float(np.max(self.evaluation_times_ms))

    @property
    def stddev_ms(self) -> float:
        return float(np.std(self.evaluation_times_ms))

    @property
    def cv(self) -> float:
        """Coefficient of variation (stddev / median)."""
        med = self.median_ms
        return self.stddev_ms / med if med > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower_time_ms": self.lower_time_ms,
            "compile_time_ms": self.compile_time_ms,
            "evaluation_times_ms": list(self.evaluation_times_ms),
            "peak_memory_mb": self.peak_memory_mb,
            "median_ms": self.median_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "stddev_ms": self.stddev_ms,
            "cv": self.cv,
        }
```

### 2. Merged stage_benchmark() (evaluate.py)

Replaces both `stage_performance()` and `stage_profile()`.

**Execution flow:**

```
stage_benchmark(exec_globals, shapes, trace_dir="/tmp/xplane_trace")
|
+- 1. Resolve kernel_fn from exec_globals
+- 2. Compile with timing isolation:
|    +- lowered = jax.jit(kernel_fn).lower(*args)     -> lower_time_ms
|    +- compiled = lowered.compile()                   -> compile_time_ms
+- 3. Peak memory:
|    +- compiled.memory_analysis().peak_memory_in_bytes -> peak_memory_mb
+- 4. Warmup (1 iteration, outside profiler):
|    +- compiled(*args).block_until_ready()
+- 5. Profiled execution (5 iterations under xprof):
|    +- jax.profiler.start_trace(trace_dir, options)
|    +- for i in range(5): compiled(*args).block_until_ready()
|    +- jax.profiler.stop_trace()
+- 6. Parse .xplane.pb:
|    +- Extract per-iteration device times -> evaluation_times_ms
|    +- Extract compute_ratio, memory_transfer_ratio
|    +- Extract hw_utilization (MXU%, Vector ALU%, etc.)
+- 7. Return BenchmarkData + profile metrics
```

**XProf options** (same as current `stage_profile`):
- `python_tracer_level = 0`
- `host_tracer_level = 2`
- `tpu_trace_mode = "TRACE_COMPUTE_AND_SYNC"`

**Compilation isolation detail:**

The current code hides compilation inside warmup iterations — the first `kernel_fn(**shape)` call triggers JIT compilation. The new approach explicitly controls this:

```python
import jax

# Build args for jit tracing
shape = shapes[0]
kernel_fn = _resolve_compute_fn(exec_globals)

# Time lowering (JAX -> HLO)
t0 = time.perf_counter()
lowered = jax.jit(kernel_fn).lower(**shape)
lower_time_ms = (time.perf_counter() - t0) * 1000

# Time compilation (HLO -> executable)
t0 = time.perf_counter()
compiled = lowered.compile()
compile_time_ms = (time.perf_counter() - t0) * 1000

# Peak memory
peak_memory_mb = None
try:
    mem = compiled.memory_analysis()
    if hasattr(mem, 'peak_memory_in_bytes'):
        peak_memory_mb = mem.peak_memory_in_bytes / (1024 * 1024)
except Exception:
    pass  # Non-fatal
```

### 3. Per-Iteration Device Time Extraction

Parse the xprof trace events to extract individual iteration durations:

```python
def _extract_iteration_times(events, tpu_pid, n_iters=5):
    """Extract per-iteration device times from xprof trace events.

    Groups computation events into iterations based on timestamp gaps.
    Returns list of per-iteration durations in milliseconds.
    """
    comp_events = sorted(
        [e for e in events
         if e.get("pid") == tpu_pid
         and "dur" in e
         and _is_computation_event(e)],
        key=lambda e: e["ts"]
    )

    if not comp_events:
        return None

    clusters = _cluster_by_gap(comp_events, n_iters)

    times_ms = []
    for cluster in clusters:
        start = cluster[0]["ts"]
        end = max(e["ts"] + e["dur"] for e in cluster)
        times_ms.append((end - start) / 1000.0)  # us -> ms

    return times_ms
```

**Gap-based clustering:** Sort events by timestamp. Compute inter-event gaps. The N-1 largest gaps are iteration boundaries. This is robust because inter-iteration gaps (including host-side Python loop overhead) are orders of magnitude larger than intra-iteration event gaps.

**Fallback chain:**
1. Gap-based clustering -> per-iteration device times (preferred)
2. Total trace window / n_iters -> average device time per iteration
3. `time.perf_counter()` wallclock -> per-iteration wallclock times (last resort)

### 4. Integration with main()

```python
# BEFORE (evaluate.py main()):
perf_result = stage_performance(compile_result["globals"], request["shapes"])
# ... later ...
profile_result = stage_profile(compile_result["globals"], request["shapes"])

# AFTER:
bench_result = stage_benchmark(compile_result["globals"], request["shapes"])
```

The same change applies to the reference kernel benchmark.

**Return value of stage_benchmark():**

```python
{
    "ok": True,
    "benchmark": BenchmarkData(...).to_dict(),
    "latency_ms": benchmark.median_ms,          # backward compat
    "compute_ratio": 0.95,                       # from xprof trace
    "memory_transfer_ratio": 0.05,               # from xprof trace
    "hw_utilization": {...},                      # from xprof trace
    "diagnostics": {...},                         # trace diagnostics
    "_trace_events_path": "...",                  # for GCS upload
}
```

**EvalResult output (backward compatible):**

```json
{
    "status": "SUCCESS",
    "latency_ms": 1.23,
    "speedup": 2.5,
    "flops": 1234567890,
    "compute_ratio": 0.95,
    "memory_transfer_ratio": 0.05,
    "benchmark": {
        "lower_time_ms": 450.2,
        "compile_time_ms": 12340.5,
        "evaluation_times_ms": [1.21, 1.23, 1.24, 1.22, 1.25],
        "peak_memory_mb": 128.5,
        "median_ms": 1.23,
        "min_ms": 1.21,
        "max_ms": 1.25,
        "stddev_ms": 0.015,
        "cv": 0.012
    },
    "metadata": {
        "reference_latency_ms": 2.46,
        "reference_benchmark": { ... },
        "hw_utilization": { ... },
        "profile_diagnostics": { ... },
        "profile": { ... }
    }
}
```

### 5. EvalResult Changes (evaluator.py)

New optional fields added to `EvalResult`:

```python
@dataclass
class EvalResult:
    # Existing fields (unchanged)
    status: EvalStatus
    fitness: float = 0.0
    error: str = ""
    max_diff: float = 0.0
    latency_ms: float = 0.0
    speedup: float = 0.0
    flops: float = 0.0
    compute_ratio: float | None = None
    memory_transfer_ratio: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # New fields (all optional with defaults for backward compat)
    lower_time_ms: float = 0.0
    compile_time_ms: float = 0.0
    peak_memory_mb: float | None = None
    eval_times_ms: list[float] = field(default_factory=list)
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_stddev_ms: float = 0.0
    latency_cv: float = 0.0
```

## Iteration Counts

| Stage | Before | After |
|-------|--------|-------|
| Performance warmup | 10 | 1 |
| Performance timed | 50 (wallclock) | 5 (xprof device time) |
| Profile warmup | 3 | (merged) |
| Profile traced | 3 | (merged) |
| **Total kernel executions** | **66** | **6** |

Fewer iterations are needed because device-side xprof timing has much lower variance than wallclock timing (no Python/scheduling noise).

## Files Changed

| File | Change |
|------|--------|
| `kernel-evolve/docker/evaluate.py` | Replace `stage_performance()` + `stage_profile()` with `stage_benchmark()`. Add `_extract_iteration_times()`, `_cluster_by_gap()`. Update `main()`. |
| `kernel-evolve/src/kernel_evolve/evaluator.py` | Add `BenchmarkData` dataclass. Add new optional fields to `EvalResult`. |
| `kernel-evolve/tests/test_docker_evaluate.py` | Update `stage_performance` tests to test `stage_benchmark`. |
| `kernel-evolve/tests/test_evaluator.py` | Add `BenchmarkData` tests, updated `EvalResult` tests. |
| `kernel-evolve/tests/test_evaluate_artifacts.py` | Merge `stage_profile` tests into `stage_benchmark` tests. |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| xprof per-iteration extraction fails on edge cases | Three-level fallback chain (gap clustering -> average -> wallclock) |
| `memory_analysis()` unavailable on some JAX versions | Non-fatal: `peak_memory_mb = None` on failure |
| `jax.jit(f).lower(**shape)` API differences | kwargs passed as dict via `**shape` matches current usage pattern |
| Fewer iterations (5 vs 50) reduces statistical confidence | Device-side timing has ~10x lower variance than wallclock; 5 iterations with xprof is comparable to 50 with wallclock |

## Non-Goals

- GPU support (CUPTI timer, GPU memory analysis)
- Standalone benchmark CLI tool
- google_benchmark integration
- TensorBoard result export
- Autotuning benchmark infrastructure
- `standardize_function` (tokamax pattern for normalizing arbitrary function signatures)
