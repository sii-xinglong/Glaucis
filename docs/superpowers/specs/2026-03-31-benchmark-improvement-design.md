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
    timing_source: str = "xprof_clustered"  # "xprof_clustered" | "xprof_average" | "wallclock"

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
            "timing_source": self.timing_source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkData:
        return cls(
            lower_time_ms=data.get("lower_time_ms", 0.0),
            compile_time_ms=data.get("compile_time_ms", 0.0),
            evaluation_times_ms=tuple(data.get("evaluation_times_ms", ())),
            peak_memory_mb=data.get("peak_memory_mb"),
            timing_source=data.get("timing_source", "xprof_clustered"),
        )
```

### 2. Merged stage_benchmark() (evaluate.py)

Replaces both `stage_performance()` and `stage_profile()`.

**Execution flow:**

```
stage_benchmark(exec_globals, shapes, trace_dir="/tmp/xplane_trace")
|
+- 1. Resolve kernel_fn from exec_globals
+- 2. Compile with timing isolation:
|    +- jitted = jax.jit(kernel_fn, static_argnames=list(shape.keys()))
|    +- lowered = jitted.lower(**shape)                -> lower_time_ms
|    +- compiled = lowered.compile()                   -> compile_time_ms
+- 3. Peak memory:
|    +- compiled.memory_analysis().peak_memory_in_bytes -> peak_memory_mb
+- 4. Warmup (3 iterations, outside profiler):
|    +- for _ in range(3): compiled().block_until_ready()
+- 5. Profiled execution (5 iterations under xprof):
|    +- jax.profiler.start_trace(trace_dir, options)
|    +- try:
|    |    for i in range(5): compiled().block_until_ready()
|    +- finally:
|    |    jax.profiler.stop_trace()
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

**Compilation isolation — handling integer kwargs:**

The kernel functions (e.g., `optimized_compute(M=1024, N=1024, K=1024)`) take integer dimension parameters as kwargs and create JAX arrays internally. All kwargs must be declared `static_argnames` so `jax.jit().lower()` traces through the function correctly:

```python
import jax

shape = shapes[0]  # e.g., {"M": 1024, "K": 512, "N": 256}
kernel_fn = _resolve_compute_fn(exec_globals)

# All kwargs are integer dimension params — mark as static for tracing
static_names = list(shape.keys())
jitted = jax.jit(kernel_fn, static_argnames=static_names)

# Time lowering (JAX -> HLO)
t0 = time.perf_counter()
lowered = jitted.lower(**shape)
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

# Execution: compiled() takes NO args since all are static
# Warmup (3 iterations — enough to warm TPU memory caches)
for _ in range(3):
    out = compiled()
    if hasattr(out, 'block_until_ready'):
        out.block_until_ready()

# Profiled runs (5 iterations)
# ... use compiled() in the profiled loop ...
```

Since all kwargs are static, they are baked into HLO at lowering time. The `compiled()` callable takes no arguments — it runs with the static values embedded in the compiled program.

### 3. Per-Iteration Device Time Extraction

Parse the xprof trace events to extract individual iteration durations.

**`_is_computation_event(event)`** — extracted from existing `stage_profile()` logic:

```python
def _is_computation_event(event):
    """Check if a trace event represents a TPU computation."""
    name = event.get("name") or ""
    return (
        "jit_computation" in name
        or "jit(" in name
        or "pallas" in name.lower()
    )
```

**`_cluster_by_gap(events, n_clusters)`** — gap-based event clustering:

```python
def _cluster_by_gap(events, n_clusters):
    """Cluster sorted events into n_clusters groups using largest gaps.

    Algorithm:
    1. Compute gaps between consecutive events: gap[i] = events[i+1].ts - (events[i].ts + events[i].dur)
    2. Find the (n_clusters - 1) largest gaps — these are iteration boundaries.
    3. Split events at these boundaries.

    Returns list of event lists. If fewer than n_clusters groups found
    (e.g., events can't be cleanly separated), returns None to trigger fallback.
    """
    if len(events) < n_clusters:
        return None

    # Compute inter-event gaps
    gaps = []
    for i in range(len(events) - 1):
        gap = events[i + 1]["ts"] - (events[i]["ts"] + events[i]["dur"])
        gaps.append((gap, i))

    # Find the (n_clusters - 1) largest gaps
    gaps.sort(key=lambda x: x[0], reverse=True)
    boundary_indices = sorted(g[1] for g in gaps[:n_clusters - 1])

    # Split events at boundaries
    clusters = []
    prev = 0
    for idx in boundary_indices:
        clusters.append(events[prev:idx + 1])
        prev = idx + 1
    clusters.append(events[prev:])

    # Validate: should have exactly n_clusters groups with events in each
    if len(clusters) != n_clusters or any(len(c) == 0 for c in clusters):
        return None

    return clusters
```

**`_extract_iteration_times(events, tpu_pid, n_iters)`** — the main extraction function:

```python
def _extract_iteration_times(events, tpu_pid, n_iters=5):
    """Extract per-iteration device times from xprof trace events.

    xprof trace events use microsecond units for `ts` and `dur` fields
    (Chrome Trace Format convention). This function returns milliseconds.

    Returns list of per-iteration durations in milliseconds, or None on failure.
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
    if clusters is None:
        return None

    times_ms = []
    for cluster in clusters:
        start = cluster[0]["ts"]
        end = max(e["ts"] + e["dur"] for e in cluster)
        times_ms.append((end - start) / 1000.0)  # us -> ms

    return times_ms
```

**Fallback chain** (engaged within `stage_benchmark()`):

1. **Gap-based clustering** -> per-iteration device times (preferred)
2. **Total trace window / n_iters** -> average device time per iteration (if clustering returns None)
3. **Wallclock timing** -> per-iteration wallclock times (last resort, if xprof trace parsing fails entirely)

For fallback level 3: wallclock timing is collected proactively alongside xprof. During the profiled loop, wallclock times are recorded in parallel:

```python
wallclock_times = []
jax.profiler.start_trace(trace_dir, profiler_options=options)
try:
    for _ in range(n_iters):
        t0 = time.perf_counter()
        out = compiled()
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
        wallclock_times.append((time.perf_counter() - t0) * 1000)
finally:
    jax.profiler.stop_trace()

# Try xprof extraction first, fall back to wallclock if needed
xprof_times = _extract_iteration_times(events, tpu_pid, n_iters)
timing_source = "xprof_clustered"
if xprof_times is None:
    # Fallback 2: average from trace window
    if total_time > 0:
        xprof_times = [total_time / n_iters / 1000.0] * n_iters
        timing_source = "xprof_average"
    else:
        # Fallback 3: wallclock
        xprof_times = wallclock_times
        timing_source = "wallclock"
```

### 4. Integration with main()

```python
# BEFORE (evaluate.py main()):
perf_result = stage_performance(compile_result["globals"], request["shapes"])
# ... later ...
profile_result = stage_profile(compile_result["globals"], request["shapes"])

# AFTER:
bench_result = stage_benchmark(compile_result["globals"], request["shapes"])
```

The same change applies to the reference kernel benchmark. Each call to `stage_benchmark()` uses a **distinct trace directory** to avoid stale `.xplane.pb` files:

```python
bench_result = stage_benchmark(compile_result["globals"], request["shapes"],
                                trace_dir="/tmp/xplane_trace/kernel")
# ...
ref_bench = stage_benchmark(ref_compile.get("globals", {}), request["shapes"],
                             trace_dir="/tmp/xplane_trace/reference")
```

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

### 5. stage_profile_deep() Interaction

`stage_profile_deep()` remains unchanged. It parses XLA/LIBTPU dump files generated during compilation, which are controlled by `_setup_dump_env()` (sets `XLA_FLAGS` and `LIBTPU_INIT_ARGS` before JAX init).

The new explicit `jax.jit().lower().compile()` path still triggers XLA compilation, which generates the same dump files that `_setup_dump_env()` enables. The dump directory is set at process startup and is independent of how compilation is invoked. Therefore `stage_profile_deep()` continues to work correctly with no changes.

Ordering in `main()`:
1. `stage_compile()` — exec kernel code (no compilation yet)
2. `stage_correctness()` — first real execution, triggers JIT compilation and generates dumps
3. `stage_benchmark()` — explicit lower/compile (may generate additional dumps, but `stage_profile_deep` looks for any matching files)
4. `stage_profile_deep()` — parses dump files from steps 2-3

### 6. EvalResult Changes (evaluator.py)

`BenchmarkData` is stored only in the `metadata` dict (under `metadata.benchmark`), avoiding field duplication. No new top-level fields are added to `EvalResult`. This keeps the data model clean — `BenchmarkData` is the single source of truth for benchmark metrics.

The `latency_ms` top-level field is still populated from `benchmark.median_ms` for backward compatibility.

**`to_dict()` and `from_dict()`** — no changes needed since `BenchmarkData` lives inside `metadata` which is already a free-form dict.

**EvalResult output format (backward compatible):**

```json
{
    "status": "SUCCESS",
    "latency_ms": 1.23,
    "speedup": 2.5,
    "flops": 1234567890,
    "compute_ratio": 0.95,
    "memory_transfer_ratio": 0.05,
    "metadata": {
        "reference_latency_ms": 2.46,
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
        "reference_benchmark": {
            "lower_time_ms": 200.1,
            "compile_time_ms": 5600.3,
            "evaluation_times_ms": [2.44, 2.46, 2.48, 2.45, 2.47],
            "peak_memory_mb": 96.0,
            "median_ms": 2.46,
            "min_ms": 2.44,
            "max_ms": 2.48,
            "stddev_ms": 0.015,
            "cv": 0.006
        },
        "hw_utilization": { "..." : "..." },
        "profile_diagnostics": { "..." : "..." },
        "profile": { "..." : "..." }
    }
}
```

## Iteration Counts

| Stage | Before | After |
|-------|--------|-------|
| Performance warmup | 10 | 3 |
| Performance timed | 50 (wallclock) | 5 (xprof device time) |
| Profile warmup | 3 | (merged) |
| Profile traced | 3 | (merged) |
| **Total kernel executions** | **66** | **8** |

Fewer iterations are needed because device-side xprof timing has much lower variance than wallclock timing (no Python/scheduling noise). 3 warmup iterations (instead of 1) are used to ensure TPU memory hierarchy and scheduling caches are warm for complex kernels.

## Files Changed

| File | Change |
|------|--------|
| `kernel-evolve/docker/evaluate.py` | Replace `stage_performance()` + `stage_profile()` with `stage_benchmark()`. Add `_is_computation_event()`, `_cluster_by_gap()`, `_extract_iteration_times()`. Update `main()`. Keep `parse_hw_utilization()` (used by `stage_benchmark`). Remove old `stage_performance()` and `stage_profile()`. |
| `kernel-evolve/src/kernel_evolve/evaluator.py` | Add `BenchmarkData` dataclass with `to_dict()` and `from_dict()`. No changes to `EvalResult` (benchmark data goes in `metadata`). |
| `kernel-evolve/tests/test_docker_evaluate.py` | Update `stage_performance` tests to test `stage_benchmark`. Add tests for `_extract_iteration_times`, `_cluster_by_gap`, `_is_computation_event`. |
| `kernel-evolve/tests/test_evaluator.py` | Add `BenchmarkData` tests (construction, to_dict, from_dict, properties). |
| `kernel-evolve/tests/test_evaluate_artifacts.py` | Merge `stage_profile` tests into `stage_benchmark` tests. |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| xprof per-iteration extraction fails on edge cases | Three-level fallback chain with proactive wallclock collection |
| `memory_analysis()` unavailable on some JAX versions | Non-fatal: `peak_memory_mb = None` on failure |
| Kernel kwargs aren't all static (e.g., JAX arrays as kwargs) | Should not happen for current kernels, but if it does, fallback to calling `kernel_fn(**shape)` directly without compilation isolation |
| Fewer iterations (5 vs 50) reduces statistical confidence | Device-side timing has ~10x lower variance than wallclock; CV is reported to flag high-variance results |
| `jax.profiler.stop_trace()` not called on exception | `try/finally` around profiled loop ensures stop_trace is always called |
| Stale `.xplane.pb` from candidate kernel bleeds into reference | Distinct `trace_dir` per call: `/tmp/xplane_trace/kernel` vs `/tmp/xplane_trace/reference` |
| `stage_profile_deep()` breaks with new compilation path | Dump files are generated by XLA flags set at process startup, independent of how compilation is invoked. Verified that `.lower().compile()` triggers the same dump generation. |

## Non-Goals

- GPU support (CUPTI timer, GPU memory analysis)
- Standalone benchmark CLI tool
- google_benchmark integration
- TensorBoard result export
- Autotuning benchmark infrastructure
- `standardize_function` (tokamax pattern for normalizing arbitrary function signatures)
