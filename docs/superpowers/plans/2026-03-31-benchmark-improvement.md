# Benchmark Improvement Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `stage_performance()` + `stage_profile()` into a single `stage_benchmark()` with hermetic xprof device timing, compilation isolation, peak memory tracking, and statistical reporting.

**Architecture:** Replace two separate evaluation stages with one that uses `jax.jit().lower().compile()` for compilation isolation, then runs 5 iterations under xprof to extract per-iteration device times via gap-based clustering. `BenchmarkData` dataclass captures all metrics. Fallback chain ensures timing always succeeds.

**Tech Stack:** JAX, xprof (`raw_to_tool_data`), numpy, pytest

**Spec:** `docs/superpowers/specs/2026-03-31-benchmark-improvement-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `kernel-evolve/src/kernel_evolve/evaluator.py` | `BenchmarkData` dataclass (lines 1-174) | Add dataclass after `EvalStatus` (line 16) |
| `kernel-evolve/docker/evaluate.py` | `stage_benchmark()`, helpers, `main()` (lines 1-871) | Replace `stage_performance` (113-133) + `stage_profile` (187-360) with `stage_benchmark()` + helpers; update `main()` (703-835) |
| `kernel-evolve/tests/test_evaluator.py` | `BenchmarkData` unit tests | Add tests |
| `kernel-evolve/tests/test_docker_evaluate.py` | `stage_benchmark` + helper tests | Replace `test_stage_performance_*` with `stage_benchmark` tests |
| `kernel-evolve/tests/test_evaluate_artifacts.py` | `stage_profile_deep` + artifact tests (unchanged) | Minor update to `test_main_collects_artifacts_for_upload` |

---

## Chunk 1: BenchmarkData Dataclass

### Task 1: Add BenchmarkData to evaluator.py

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/evaluator.py:1-16`
- Test: `kernel-evolve/tests/test_evaluator.py`

- [ ] **Step 1: Write failing tests for BenchmarkData**

Add to `kernel-evolve/tests/test_evaluator.py` after the existing imports:

```python
from kernel_evolve.evaluator import (
  BatchEvalRequest,
  BatchEvalResult,
  BenchmarkData,  # NEW
  EvalRequest,
  EvalResult,
  EvalStatus,
)
```

Then add these tests at the end of the file:

```python
def test_benchmark_data_basic():
  bd = BenchmarkData(
    lower_time_ms=100.0,
    compile_time_ms=5000.0,
    evaluation_times_ms=(1.1, 1.3, 1.2, 1.4, 1.0),
    peak_memory_mb=128.5,
  )
  assert bd.median_ms == 1.2
  assert bd.min_ms == 1.0
  assert bd.max_ms == 1.4
  assert bd.timing_source == "xprof_clustered"


def test_benchmark_data_cv():
  bd = BenchmarkData(
    lower_time_ms=0.0,
    compile_time_ms=0.0,
    evaluation_times_ms=(2.0, 2.0, 2.0),
    peak_memory_mb=None,
  )
  assert bd.cv == 0.0  # zero variance


def test_benchmark_data_to_dict():
  bd = BenchmarkData(
    lower_time_ms=100.0,
    compile_time_ms=5000.0,
    evaluation_times_ms=(1.0, 2.0, 3.0),
    peak_memory_mb=64.0,
    timing_source="wallclock",
  )
  d = bd.to_dict()
  assert d["lower_time_ms"] == 100.0
  assert d["compile_time_ms"] == 5000.0
  assert d["evaluation_times_ms"] == [1.0, 2.0, 3.0]
  assert d["peak_memory_mb"] == 64.0
  assert d["median_ms"] == 2.0
  assert d["timing_source"] == "wallclock"


def test_benchmark_data_from_dict():
  d = {
    "lower_time_ms": 50.0,
    "compile_time_ms": 2000.0,
    "evaluation_times_ms": [1.5, 1.6],
    "peak_memory_mb": None,
    "timing_source": "xprof_average",
  }
  bd = BenchmarkData.from_dict(d)
  assert bd.lower_time_ms == 50.0
  assert bd.evaluation_times_ms == (1.5, 1.6)
  assert bd.peak_memory_mb is None
  assert bd.timing_source == "xprof_average"


def test_benchmark_data_roundtrip():
  original = BenchmarkData(
    lower_time_ms=200.0,
    compile_time_ms=8000.0,
    evaluation_times_ms=(0.5, 0.6, 0.7, 0.8, 0.9),
    peak_memory_mb=256.0,
    timing_source="xprof_clustered",
  )
  restored = BenchmarkData.from_dict(original.to_dict())
  assert restored.lower_time_ms == original.lower_time_ms
  assert restored.compile_time_ms == original.compile_time_ms
  assert restored.evaluation_times_ms == original.evaluation_times_ms
  assert restored.peak_memory_mb == original.peak_memory_mb
  assert restored.timing_source == original.timing_source
  assert restored.median_ms == original.median_ms


def test_benchmark_data_from_dict_defaults():
  """from_dict should handle missing keys gracefully."""
  bd = BenchmarkData.from_dict({})
  assert bd.lower_time_ms == 0.0
  assert bd.compile_time_ms == 0.0
  assert bd.evaluation_times_ms == ()
  assert bd.peak_memory_mb is None
  assert bd.timing_source == "xprof_clustered"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/xl/Code/Glaucis/.claude/worktrees/humming-enchanting-lagoon && python -m pytest kernel-evolve/tests/test_evaluator.py -v -k "benchmark_data" 2>&1 | tail -20`

Expected: FAIL with `ImportError: cannot import name 'BenchmarkData'`

- [ ] **Step 3: Implement BenchmarkData dataclass**

Add to `kernel-evolve/src/kernel_evolve/evaluator.py` after line 5 (`from typing import Any`), add:

```python
import numpy as np
```

Then after `EvalStatus` class (after line 16), add:

```python
@dataclass
class BenchmarkData:
  lower_time_ms: float
  compile_time_ms: float
  evaluation_times_ms: tuple[float, ...]
  peak_memory_mb: float | None
  timing_source: str = "xprof_clustered"

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

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/xl/Code/Glaucis/.claude/worktrees/humming-enchanting-lagoon && python -m pytest kernel-evolve/tests/test_evaluator.py -v 2>&1 | tail -30`

Expected: All tests PASS, including the 7 new `benchmark_data` tests.

- [ ] **Step 5: Commit**

```bash
cd /Users/xl/Code/Glaucis/.claude/worktrees/humming-enchanting-lagoon
git add kernel-evolve/src/kernel_evolve/evaluator.py kernel-evolve/tests/test_evaluator.py
git commit -m "feat(evaluator): add BenchmarkData dataclass with statistical properties

TDD: tests for construction, to_dict, from_dict, roundtrip, defaults.
Dataclass captures lower/compile time, per-iteration device times,
peak memory, and timing source (xprof_clustered/xprof_average/wallclock).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Chunk 2: XProf Extraction Helpers

### Task 2: Add _is_computation_event, _cluster_by_gap, _extract_iteration_times

**Files:**
- Modify: `kernel-evolve/docker/evaluate.py:113-133` (add new functions before `stage_performance`)
- Test: `kernel-evolve/tests/test_docker_evaluate.py`

- [ ] **Step 1: Write failing tests for the three helpers**

Add to `kernel-evolve/tests/test_docker_evaluate.py` at the end of the file:

```python
# ── xprof extraction helper tests ──


def test_is_computation_event_matches():
  evaluate = _load_evaluate_module()
  assert evaluate._is_computation_event({"name": "jit_computation.123"}) is True
  assert evaluate._is_computation_event({"name": "jit(fn)"}) is True
  assert evaluate._is_computation_event({"name": "pallas_call"}) is True
  assert evaluate._is_computation_event({"name": "PALLAS_tpu"}) is True


def test_is_computation_event_rejects():
  evaluate = _load_evaluate_module()
  assert evaluate._is_computation_event({"name": "SyncWait"}) is False
  assert evaluate._is_computation_event({"name": "idle"}) is False
  assert evaluate._is_computation_event({}) is False
  assert evaluate._is_computation_event({"name": ""}) is False


def test_cluster_by_gap_basic():
  """5 events with 4 clear gaps should produce 5 clusters of 1 event each."""
  evaluate = _load_evaluate_module()
  events = [
    {"ts": 0, "dur": 10},
    {"ts": 1000, "dur": 10},
    {"ts": 2000, "dur": 10},
    {"ts": 3000, "dur": 10},
    {"ts": 4000, "dur": 10},
  ]
  clusters = evaluate._cluster_by_gap(events, 5)
  assert clusters is not None
  assert len(clusters) == 5
  assert all(len(c) == 1 for c in clusters)


def test_cluster_by_gap_groups_close_events():
  """Events close together should be grouped; large gaps separate iterations."""
  evaluate = _load_evaluate_module()
  # 2 iterations, each with 3 closely-spaced events, big gap between iterations
  events = [
    {"ts": 0, "dur": 5},
    {"ts": 10, "dur": 5},
    {"ts": 20, "dur": 5},
    # gap of 9975
    {"ts": 10000, "dur": 5},
    {"ts": 10010, "dur": 5},
    {"ts": 10020, "dur": 5},
  ]
  clusters = evaluate._cluster_by_gap(events, 2)
  assert clusters is not None
  assert len(clusters) == 2
  assert len(clusters[0]) == 3
  assert len(clusters[1]) == 3


def test_cluster_by_gap_too_few_events():
  evaluate = _load_evaluate_module()
  events = [{"ts": 0, "dur": 10}]
  assert evaluate._cluster_by_gap(events, 5) is None


def test_cluster_by_gap_empty():
  evaluate = _load_evaluate_module()
  assert evaluate._cluster_by_gap([], 3) is None


def test_extract_iteration_times_basic():
  """Should extract per-iteration times from well-separated computation events."""
  evaluate = _load_evaluate_module()
  tpu_pid = 42
  events = []
  for i in range(3):
    # Each iteration: one computation event of 500us, separated by 10000us gaps
    events.append({
      "pid": tpu_pid,
      "ts": i * 10000,
      "dur": 500,
      "name": "jit_computation.fn",
    })
  times = evaluate._extract_iteration_times(events, tpu_pid, n_iters=3)
  assert times is not None
  assert len(times) == 3
  assert all(abs(t - 0.5) < 0.01 for t in times)  # 500us = 0.5ms


def test_extract_iteration_times_wrong_pid():
  evaluate = _load_evaluate_module()
  events = [{"pid": 99, "ts": 0, "dur": 100, "name": "jit_computation.fn"}]
  assert evaluate._extract_iteration_times(events, tpu_pid=42, n_iters=1) is None


def test_extract_iteration_times_no_computation_events():
  evaluate = _load_evaluate_module()
  events = [{"pid": 42, "ts": 0, "dur": 100, "name": "SyncWait"}]
  assert evaluate._extract_iteration_times(events, tpu_pid=42, n_iters=1) is None


def test_extract_iteration_times_multi_event_iterations():
  """Each iteration may have multiple computation events."""
  evaluate = _load_evaluate_module()
  tpu_pid = 1
  # 2 iterations, each with 2 events, gap between iterations is large
  events = [
    # Iteration 1
    {"pid": tpu_pid, "ts": 0, "dur": 200, "name": "jit_computation.a"},
    {"pid": tpu_pid, "ts": 300, "dur": 200, "name": "pallas_call"},
    # Iteration 2 (gap of 9500)
    {"pid": tpu_pid, "ts": 10000, "dur": 200, "name": "jit_computation.a"},
    {"pid": tpu_pid, "ts": 10300, "dur": 200, "name": "pallas_call"},
  ]
  times = evaluate._extract_iteration_times(events, tpu_pid, n_iters=2)
  assert times is not None
  assert len(times) == 2
  # Each iteration spans from ts=0,dur=200 to ts=300+200=500 => 500us = 0.5ms
  assert abs(times[0] - 0.5) < 0.01
  assert abs(times[1] - 0.5) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/xl/Code/Glaucis/.claude/worktrees/humming-enchanting-lagoon && python -m pytest kernel-evolve/tests/test_docker_evaluate.py -v -k "is_computation_event or cluster_by_gap or extract_iteration" 2>&1 | tail -20`

Expected: FAIL with `AttributeError: module has no attribute '_is_computation_event'`

- [ ] **Step 3: Implement the three helper functions**

Add to `kernel-evolve/docker/evaluate.py` after `stage_compile()` (after line 88), before `stage_correctness()`:

```python
def _is_computation_event(event):
  """Check if a trace event represents a TPU computation."""
  name = event.get("name") or ""
  return (
    "jit_computation" in name
    or "jit(" in name
    or "pallas" in name.lower()
  )


def _cluster_by_gap(events, n_clusters):
  """Cluster sorted events into n_clusters groups using largest gaps.

  Computes gaps between consecutive events, finds the (n_clusters - 1)
  largest gaps as iteration boundaries, and splits events at those points.

  Returns list of event lists, or None if clustering fails.
  """
  if len(events) < n_clusters:
    return None

  gaps = []
  for i in range(len(events) - 1):
    gap = events[i + 1]["ts"] - (events[i]["ts"] + events[i]["dur"])
    gaps.append((gap, i))

  gaps.sort(key=lambda x: x[0], reverse=True)
  boundary_indices = sorted(g[1] for g in gaps[:n_clusters - 1])

  clusters = []
  prev = 0
  for idx in boundary_indices:
    clusters.append(events[prev:idx + 1])
    prev = idx + 1
  clusters.append(events[prev:])

  if len(clusters) != n_clusters or any(len(c) == 0 for c in clusters):
    return None

  return clusters


def _extract_iteration_times(events, tpu_pid, n_iters=5):
  """Extract per-iteration device times from xprof trace events.

  xprof trace events use microsecond units for ts and dur fields
  (Chrome Trace Format convention). Returns milliseconds, or None on failure.
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
    times_ms.append((end - start) / 1000.0)
  return times_ms
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/xl/Code/Glaucis/.claude/worktrees/humming-enchanting-lagoon && python -m pytest kernel-evolve/tests/test_docker_evaluate.py -v -k "is_computation_event or cluster_by_gap or extract_iteration" 2>&1 | tail -20`

Expected: All 10 new tests PASS.

- [ ] **Step 5: Run all existing tests to ensure no regressions**

Run: `cd /Users/xl/Code/Glaucis/.claude/worktrees/humming-enchanting-lagoon && python -m pytest kernel-evolve/tests/test_docker_evaluate.py kernel-evolve/tests/test_evaluate_artifacts.py -v 2>&1 | tail -30`

Expected: All existing tests still PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/xl/Code/Glaucis/.claude/worktrees/humming-enchanting-lagoon
git add kernel-evolve/docker/evaluate.py kernel-evolve/tests/test_docker_evaluate.py
git commit -m "feat(evaluate): add xprof extraction helpers for per-iteration device timing

TDD: _is_computation_event, _cluster_by_gap, _extract_iteration_times.
Gap-based clustering separates xprof trace events into per-iteration
groups using the N-1 largest inter-event gaps as boundaries.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Chunk 3: stage_benchmark() Implementation

### Task 3: Implement stage_benchmark() replacing stage_performance + stage_profile

**Files:**
- Modify: `kernel-evolve/docker/evaluate.py:113-360` (replace `stage_performance` + `stage_profile`)
- Test: `kernel-evolve/tests/test_docker_evaluate.py`

**Critical design decision — module-level imports for testability:**

The existing `stage_profile()` uses local `import jax` inside the function body. This makes mocking impossible with standard `patch()`. The new `stage_benchmark()` must use **module-level** imports so tests can patch `evaluate.jax` and `evaluate.raw_to_tool_data`. Add these at the top of `evaluate.py` (after `import numpy as np`):

```python
try:
  import jax
except ImportError:
  jax = None

try:
  from xprof.convert import raw_to_tool_data
except ImportError:
  raw_to_tool_data = None
```

- [ ] **Step 1: Add MagicMock import and write failing tests for stage_benchmark**

First, update the imports in `kernel-evolve/tests/test_docker_evaluate.py` (line 11):

```python
from unittest.mock import MagicMock, patch
```

Then add to `kernel-evolve/tests/test_docker_evaluate.py` at the end:

```python
def test_stage_benchmark_returns_benchmark_data(tmp_path):
  """stage_benchmark should return BenchmarkData with timing and profile metrics."""
  evaluate = _load_evaluate_module()

  globals_dict = {"optimized_compute": lambda M=1: M}

  # Mock jax.jit -> lower -> compile chain
  mock_lowered = MagicMock()
  mock_compiled = MagicMock()
  mock_compiled.return_value = 42
  mock_mem = MagicMock()
  mock_mem.peak_memory_in_bytes = 128 * 1024 * 1024  # 128 MB
  mock_compiled.memory_analysis.return_value = mock_mem
  mock_lowered.compile.return_value = mock_compiled
  mock_jitted = MagicMock()
  mock_jitted.lower.return_value = mock_lowered

  # Prepare xprof trace events that simulate 5 iterations
  trace_events = []
  for i in range(5):
    trace_events.append({
      "pid": 1, "ts": i * 10000, "dur": 500 + i * 10,
      "name": "jit_computation.fn",
    })
  trace_events.append({"args": {"name": "/device:TPU:0"}, "pid": 1})
  trace_events.append({
    "pid": 1, "tid": 4294967295, "name": "MXU", "dur": 100,
    "args": {"% util": "50.0"},
  })

  mock_xprof = MagicMock()
  mock_xprof.xspace_to_tool_data.return_value = (
    json.dumps({"traceEvents": trace_events}), None
  )

  # Create dummy .xplane.pb so os.walk finds it and proceeds to xprof parsing
  (tmp_path / "trace.xplane.pb").write_bytes(b"")

  # Patch module-level attributes on the loaded evaluate module
  with patch.object(evaluate, "jax") as mock_jax, \
       patch.object(evaluate, "raw_to_tool_data", mock_xprof):
    mock_jax.jit.return_value = mock_jitted
    mock_jax.profiler.ProfileOptions.return_value = MagicMock()

    result = evaluate.stage_benchmark(
      globals_dict,
      shapes=[{"M": 4}],
      trace_dir=str(tmp_path),
    )

  assert result["ok"] is True
  assert "benchmark" in result
  assert result["latency_ms"] > 0
  assert result["benchmark"]["peak_memory_mb"] == 128.0
  assert result["benchmark"]["timing_source"] == "xprof_clustered"
  assert len(result["benchmark"]["evaluation_times_ms"]) == 5
  assert "compute_ratio" in result
  assert "hw_utilization" in result


def test_stage_benchmark_wallclock_fallback(tmp_path):
  """stage_benchmark should fall back to wallclock timing when xprof fails."""
  evaluate = _load_evaluate_module()

  globals_dict = {"optimized_compute": lambda M=1: M}

  mock_lowered = MagicMock()
  mock_compiled = MagicMock()
  mock_compiled.return_value = 42
  mock_compiled.memory_analysis.side_effect = Exception("no memory_analysis")
  mock_lowered.compile.return_value = mock_compiled
  mock_jitted = MagicMock()
  mock_jitted.lower.return_value = mock_lowered

  with patch.object(evaluate, "jax") as mock_jax, \
       patch.object(evaluate, "raw_to_tool_data", None):
    mock_jax.jit.return_value = mock_jitted
    mock_jax.profiler.ProfileOptions.return_value = MagicMock()

    result = evaluate.stage_benchmark(
      globals_dict,
      shapes=[{"M": 4}],
      trace_dir=str(tmp_path),
    )

  assert result["ok"] is True
  assert result["benchmark"]["timing_source"] == "wallclock"
  assert result["benchmark"]["peak_memory_mb"] is None
  assert result["latency_ms"] > 0


def test_stage_benchmark_xprof_average_fallback(tmp_path):
  """stage_benchmark should fall back to xprof_average when clustering fails."""
  evaluate = _load_evaluate_module()

  globals_dict = {"optimized_compute": lambda M=1: M}

  mock_lowered = MagicMock()
  mock_compiled = MagicMock()
  mock_compiled.return_value = 42
  mock_compiled.memory_analysis.side_effect = Exception("no memory")
  mock_lowered.compile.return_value = mock_compiled
  mock_jitted = MagicMock()
  mock_jitted.lower.return_value = mock_lowered

  # Events that can't be clustered into 5 groups (only 2 events, need 5 iters)
  trace_events = [
    {"pid": 1, "ts": 0, "dur": 2500, "name": "jit_computation.fn"},
    {"pid": 1, "ts": 3000, "dur": 2500, "name": "jit_computation.fn"},
    {"args": {"name": "/device:TPU:0"}, "pid": 1},
  ]

  mock_xprof = MagicMock()
  mock_xprof.xspace_to_tool_data.return_value = (
    json.dumps({"traceEvents": trace_events}), None
  )

  # Create dummy .xplane.pb so os.walk finds it and proceeds to xprof parsing
  (tmp_path / "trace.xplane.pb").write_bytes(b"")

  with patch.object(evaluate, "jax") as mock_jax, \
       patch.object(evaluate, "raw_to_tool_data", mock_xprof):
    mock_jax.jit.return_value = mock_jitted
    mock_jax.profiler.ProfileOptions.return_value = MagicMock()

    result = evaluate.stage_benchmark(
      globals_dict,
      shapes=[{"M": 4}],
      trace_dir=str(tmp_path),
      n_iters=5,
    )

  assert result["ok"] is True
  assert result["benchmark"]["timing_source"] == "xprof_average"


def test_stage_benchmark_no_compute_fn():
  """stage_benchmark should return error when no compute function found."""
  evaluate = _load_evaluate_module()
  result = evaluate.stage_benchmark({}, shapes=[{"M": 4}])
  assert result["ok"] is False
  assert "No compute function" in result["error"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/xl/Code/Glaucis/.claude/worktrees/humming-enchanting-lagoon && python -m pytest kernel-evolve/tests/test_docker_evaluate.py -v -k "stage_benchmark" 2>&1 | tail -20`

Expected: FAIL with `AttributeError: module has no attribute 'stage_benchmark'`

- [ ] **Step 3: Move jax/xprof imports to module level and implement stage_benchmark()**

First, add module-level imports in `kernel-evolve/docker/evaluate.py` after `import numpy as np` (line 17):

```python
try:
  import jax
except ImportError:
  jax = None

try:
  from xprof.convert import raw_to_tool_data
except ImportError:
  raw_to_tool_data = None
```

Then add `stage_benchmark()` after `stage_compile()` (after line 88), before `stage_correctness()`. Keep `parse_hw_utilization()` as-is — it's called by `stage_benchmark()`.

The function uses the module-level `jax` and `raw_to_tool_data` references (not local imports), making it testable with `patch.object(evaluate, "jax")`.

```python
def stage_benchmark(exec_globals, shapes, trace_dir="/tmp/xplane_trace", warmup=3, n_iters=5):
  """Merged performance + profile stage using hermetic xprof timing.

  Execution flow:
  1. Resolve kernel_fn
  2. Compile with timing isolation (jax.jit().lower().compile())
  3. Peak memory via memory_analysis()
  4. Warmup (warmup iterations, outside profiler)
  5. Profiled execution (n_iters iterations under xprof)
  6. Parse .xplane.pb for per-iteration device times + hw utilization
  7. Return BenchmarkData + profile metrics

  Falls back to wallclock timing if xprof fails.
  """
  try:
    kernel_fn = _resolve_compute_fn(exec_globals, allow_reference=True)
    if kernel_fn is None:
      return {"ok": False, "error": "No compute function found"}

    shape = shapes[0]

    # ── Step 2: Compilation isolation ──
    static_names = list(shape.keys())
    jitted = jax.jit(kernel_fn, static_argnames=static_names)

    t0 = time.perf_counter()
    lowered = jitted.lower(**shape)
    lower_time_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    compiled = lowered.compile()
    compile_time_ms = (time.perf_counter() - t0) * 1000

    # ── Step 3: Peak memory ──
    peak_memory_mb = None
    try:
      mem = compiled.memory_analysis()
      if hasattr(mem, "peak_memory_in_bytes"):
        peak_memory_mb = mem.peak_memory_in_bytes / (1024 * 1024)
    except Exception:
      pass

    # ── Step 4: Warmup ──
    for _ in range(warmup):
      out = compiled()
      if hasattr(out, "block_until_ready"):
        out.block_until_ready()

    # ── Step 5: Profiled execution with wallclock fallback ──
    from pathlib import Path
    Path(trace_dir).mkdir(parents=True, exist_ok=True)

    wallclock_times = []
    xprof_ok = False
    events = []

    if raw_to_tool_data is not None:  # module-level import, patchable
      try:
        options = jax.profiler.ProfileOptions()
        options.python_tracer_level = 0
        options.host_tracer_level = 2
        options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}

        jax.profiler.start_trace(trace_dir, profiler_options=options)
        try:
          for _ in range(n_iters):
            t0 = time.perf_counter()
            out = compiled()
            if hasattr(out, "block_until_ready"):
              out.block_until_ready()
            wallclock_times.append((time.perf_counter() - t0) * 1000)
        finally:
          jax.profiler.stop_trace()
        xprof_ok = True
      except Exception as e:
        print(f"xprof profiling failed: {e}", file=sys.stderr)

    # Wallclock-only fallback if xprof profiling failed
    if not xprof_ok:
      for _ in range(n_iters):
        t0 = time.perf_counter()
        out = compiled()
        if hasattr(out, "block_until_ready"):
          out.block_until_ready()
        wallclock_times.append((time.perf_counter() - t0) * 1000)

    # ── Step 6: Parse xprof trace ──
    compute_ratio = None
    memory_transfer_ratio = None
    hw_utilization = None
    profile_diag = {}
    trace_events_path = None
    timing_source = "wallclock"
    eval_times = wallclock_times

    if xprof_ok:
      xplane_path = None
      for root, _dirs, files in os.walk(trace_dir):
        for f in files:
          if f.endswith(".xplane.pb"):
            xplane_path = os.path.join(root, f)
            break
        if xplane_path:
          break

      if xplane_path is not None:
        try:
          tool_data_result, _ = raw_to_tool_data.xspace_to_tool_data(
            [xplane_path], "trace_viewer", {}
          )
          trace_data = json.loads(tool_data_result)
          events = trace_data.get("traceEvents", [])

          trace_events_path = os.path.join(trace_dir, "trace_events.json")
          with open(trace_events_path, "w") as f:
            json.dump(events, f)

          # Find TPU device pid
          tpu_pid = None
          for event in events:
            if "args" in event and event["args"].get("name") == "/device:TPU:0":
              tpu_pid = event.get("pid")
              break
          if tpu_pid is None:
            for event in events:
              if "args" in event:
                name = event["args"].get("name", "")
                if name.startswith("/device:TPU:"):
                  tpu_pid = event.get("pid")
                  break

          if tpu_pid is not None:
            # Extract per-iteration device times
            xprof_times = _extract_iteration_times(events, tpu_pid, n_iters)
            if xprof_times is not None:
              eval_times = xprof_times
              timing_source = "xprof_clustered"
            else:
              # Fallback 2: average from trace window
              comp_events = [
                e for e in events
                if e.get("pid") == tpu_pid and "dur" in e and _is_computation_event(e)
              ]
              if comp_events:
                trace_start = comp_events[0]["ts"]
                trace_end = comp_events[-1]["ts"] + comp_events[-1]["dur"]
                total_time = trace_end - trace_start
                if total_time > 0:
                  eval_times = [total_time / n_iters / 1000.0] * n_iters
                  timing_source = "xprof_average"

            # Extract compute_ratio from sync/idle events
            events_for_tpu = [e for e in events if e.get("pid") == tpu_pid]
            computation_events = [
              e for e in events_for_tpu
              if "dur" in e and _is_computation_event(e)
            ]
            if len(computation_events) >= 2:
              trace_start = computation_events[0]["ts"]
              trace_end = computation_events[-1]["ts"] + computation_events[-1]["dur"]
              sync_wait_total = 0
              for event in events_for_tpu:
                if "dur" not in event:
                  continue
                evt_start = event["ts"]
                evt_end = evt_start + event["dur"]
                if evt_start >= trace_start and evt_end <= trace_end:
                  name = event.get("name") or ""
                  if "SyncWait" in name or "idle" in name.lower():
                    sync_wait_total += event["dur"]
              total_time = trace_end - trace_start
              if total_time > 0:
                ratio = sync_wait_total / total_time
                compute_ratio = 1.0 - ratio
                memory_transfer_ratio = ratio

            hw_utilization = parse_hw_utilization(events, tpu_pid)

            # Diagnostics
            process_names = {}
            for event in events:
              if "args" in event and "name" in event["args"]:
                pid_val = event.get("pid")
                if pid_val not in process_names:
                  process_names[pid_val] = []
                name_val = event["args"]["name"]
                if name_val not in process_names[pid_val]:
                  process_names[pid_val].append(name_val)

            profile_diag = {
              "process_names": {str(k): v for k, v in process_names.items()},
              "selected_pid": tpu_pid,
              "total_events": len(events),
              "tpu_events": len(events_for_tpu),
              "computation_events": len(computation_events),
            }
        except Exception as e:
          print(f"xprof trace parsing failed: {e}", file=sys.stderr)
          traceback.print_exc(file=sys.stderr)

    # ── Step 7: Build result ──
    from kernel_evolve.evaluator import BenchmarkData

    benchmark = BenchmarkData(
      lower_time_ms=lower_time_ms,
      compile_time_ms=compile_time_ms,
      evaluation_times_ms=tuple(eval_times),
      peak_memory_mb=peak_memory_mb,
      timing_source=timing_source,
    )

    return {
      "ok": True,
      "benchmark": benchmark.to_dict(),
      "latency_ms": benchmark.median_ms,
      "compute_ratio": compute_ratio,
      "memory_transfer_ratio": memory_transfer_ratio,
      "hw_utilization": hw_utilization,
      "diagnostics": profile_diag,
      "_trace_events_path": trace_events_path,
    }
  except Exception:
    return {"ok": False, "error": f"Benchmark error: {traceback.format_exc()}"}
```

**Important:** Do NOT delete `stage_performance()` or `stage_profile()` yet — they're still referenced by `main()` and tests. We'll remove them in Task 5.

- [ ] **Step 4: Fix test mocks as needed and run tests**

The tests may need adjustment based on exact mock setup. Run:

Run: `cd /Users/xl/Code/Glaucis/.claude/worktrees/humming-enchanting-lagoon && python -m pytest kernel-evolve/tests/test_docker_evaluate.py -v -k "stage_benchmark" 2>&1 | tail -30`

Expected: All 3 `stage_benchmark` tests PASS.

- [ ] **Step 5: Run all tests to verify no regressions**

Run: `cd /Users/xl/Code/Glaucis/.claude/worktrees/humming-enchanting-lagoon && python -m pytest kernel-evolve/tests/ -v 2>&1 | tail -40`

Expected: All tests PASS (old tests still pass since old functions still exist).

- [ ] **Step 6: Commit**

```bash
cd /Users/xl/Code/Glaucis/.claude/worktrees/humming-enchanting-lagoon
git add kernel-evolve/docker/evaluate.py kernel-evolve/tests/test_docker_evaluate.py
git commit -m "feat(evaluate): add stage_benchmark with xprof device timing

Merged stage_performance + stage_profile into stage_benchmark().
Uses jax.jit().lower().compile() for compilation isolation,
hermetic xprof for device-side timing with 3-level fallback chain.
Old functions preserved for backward compat during migration.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Chunk 4: main() Integration and Cleanup

### Task 4: Update main() to use stage_benchmark()

**Files:**
- Modify: `kernel-evolve/docker/evaluate.py:703-835` (main function)

- [ ] **Step 1: Update main() to replace stage_performance + stage_profile calls**

In `kernel-evolve/docker/evaluate.py`, modify `main()` (starting at line 703):

Replace the performance + profile section (lines 749-773) with:

```python
  # Stage 3+4 merged: Benchmark (performance + profile)
  bench_result = stage_benchmark(
    compile_result["globals"], request["shapes"],
    trace_dir="/tmp/xplane_trace/kernel",
  )
  if not bench_result["ok"]:
    err = {"status": "COMPILE_ERROR", "variant_id": job_name, "error": bench_result["error"]}
    print(f"EVAL_RESULT:{json.dumps(err)}")
    sys.exit(0)

  ref_compile = stage_compile(request["reference_code"])
  ref_bench = stage_benchmark(
    ref_compile.get("globals", {}), request["shapes"],
    trace_dir="/tmp/xplane_trace/reference",
  )
  ref_latency = ref_bench.get("latency_ms", bench_result["latency_ms"])
  speedup = ref_latency / bench_result["latency_ms"] if bench_result["latency_ms"] > 0 else 0.0

  compute_ratio = bench_result.get("compute_ratio")
  memory_transfer_ratio = bench_result.get("memory_transfer_ratio")
  hw_utilization = bench_result.get("hw_utilization")
  profile_diag = bench_result.get("diagnostics", {})
```

Update the deep profile + artifact section (lines 776-812) — change references from `profile_result` to `bench_result` and from `perf_result` to `bench_result`:

```python
  # Stage 5: Deep profile (non-fatal) — parse dumps generated by earlier stages
  deep_profile = stage_profile_deep(compile_result["globals"], request["shapes"])
  if not deep_profile["ok"]:
    print(
      f"Deep profile skipped: {deep_profile.get('error', 'unknown')}",
      file=sys.stderr,
    )

  # Compute efficiency from deep profile FLOPs and measured latency
  peak_flops = float(os.environ.get("PEAK_FLOPS", 2307e12))
  if deep_profile.get("flops") and bench_result["latency_ms"] > 0:
    actual_fps = deep_profile["flops"] / (bench_result["latency_ms"] / 1000.0)
    deep_profile["compute_efficiency_pct"] = (actual_fps / peak_flops) * 100.0

  peak_hbm_bw = float(os.environ.get("PEAK_HBM_BW", 3690e9))
  if deep_profile.get("hbm_bandwidth_bytes") and bench_result["latency_ms"] > 0:
    actual_bw = deep_profile["hbm_bandwidth_bytes"] / (bench_result["latency_ms"] / 1000.0)
    deep_profile["hbm_bandwidth_utilization_pct"] = (actual_bw / peak_hbm_bw) * 100.0

  # ── Upload profile artifacts to GCS (non-fatal) ──
  artifacts = {}
  if bench_result.get("ok"):
    trace_path = bench_result.get("_trace_events_path")
    if trace_path:
      artifacts["trace_events.json"] = trace_path
  if deep_profile.get("ok"):
    hlo_path = deep_profile.get("_hlo_file")
    llo_path = deep_profile.get("_llo_file")
    if hlo_path:
      artifacts["hlo_post_opt.txt"] = hlo_path
    if llo_path:
      artifacts["llo_final.txt"] = llo_path

  gcs_result = upload_to_gcs(job_name, artifacts) if artifacts else {"ok": False, "uploaded": [], "gcs_prefix": ""}
  if gcs_result["ok"]:
    print(f"Uploaded artifacts: {gcs_result['uploaded']} to {gcs_result['gcs_prefix']}", file=sys.stderr)

  clean_deep_profile = {k: v for k, v in deep_profile.items() if not k.startswith("_")}

  result = {
    "status": "SUCCESS",
    "variant_id": job_name,
    "fitness": speedup,
    "latency_ms": bench_result["latency_ms"],
    "speedup": speedup,
    "flops": clean_deep_profile.get("flops", 0.0) or 0.0,
    "compute_ratio": compute_ratio,
    "memory_transfer_ratio": memory_transfer_ratio,
    "metadata": {
      "reference_latency_ms": ref_latency,
      "reference_perf_ok": ref_bench.get("ok", False),
      "benchmark": bench_result.get("benchmark"),
      "reference_benchmark": ref_bench.get("benchmark"),
      "profile_diagnostics": profile_diag,
      "hw_utilization": hw_utilization,
      "profile": clean_deep_profile,
      **({"artifacts_gcs_prefix": gcs_result["gcs_prefix"]} if gcs_result.get("ok") else {}),
    },
  }
  print(f"EVAL_RESULT:{json.dumps(result)}")
```

- [ ] **Step 2: Update test_main_collects_artifacts_for_upload**

In `kernel-evolve/tests/test_evaluate_artifacts.py`, update `test_main_collects_artifacts_for_upload` (line 264) — change `profile_result` key name to `bench_result` to match the new main() variable name:

```python
def test_main_collects_artifacts_for_upload():
  """Verify the artifact collection logic from stage results."""
  bench_result = {
    "ok": True,
    "compute_ratio": 0.85,
    "memory_transfer_ratio": 0.15,
    "diagnostics": {},
    "_trace_events_path": "/tmp/xplane_trace/trace_events.json",
  }
  deep_profile = {
    "ok": True,
    "vliw_bundle_count": 100,
    "_hlo_file": "/tmp/ir_dumps/hlo/module.after.txt",
    "_llo_file": "/tmp/ir_dumps/llo/module.pass_79.llo",
  }

  artifacts = {}
  if bench_result.get("ok"):
    trace_path = bench_result.get("_trace_events_path")
    if trace_path:
      artifacts["trace_events.json"] = trace_path
  if deep_profile.get("ok"):
    hlo_path = deep_profile.get("_hlo_file")
    llo_path = deep_profile.get("_llo_file")
    if hlo_path:
      artifacts["hlo_post_opt.txt"] = hlo_path
    if llo_path:
      artifacts["llo_final.txt"] = llo_path

  assert artifacts == {
    "trace_events.json": "/tmp/xplane_trace/trace_events.json",
    "hlo_post_opt.txt": "/tmp/ir_dumps/hlo/module.after.txt",
    "llo_final.txt": "/tmp/ir_dumps/llo/module.pass_79.llo",
  }
```

- [ ] **Step 3: Run all tests**

Run: `cd /Users/xl/Code/Glaucis/.claude/worktrees/humming-enchanting-lagoon && python -m pytest kernel-evolve/tests/ -v 2>&1 | tail -40`

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
cd /Users/xl/Code/Glaucis/.claude/worktrees/humming-enchanting-lagoon
git add kernel-evolve/docker/evaluate.py kernel-evolve/tests/test_evaluate_artifacts.py
git commit -m "feat(evaluate): update main() to use stage_benchmark

Replace stage_performance + stage_profile calls with stage_benchmark.
Distinct trace dirs for kernel vs reference. BenchmarkData added to
metadata for both kernel and reference results.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

### Task 5: Remove old stage_performance and stage_profile

**Files:**
- Modify: `kernel-evolve/docker/evaluate.py`
- Modify: `kernel-evolve/tests/test_docker_evaluate.py`

- [ ] **Step 1: Remove old stage_performance function**

Delete `stage_performance()` from `kernel-evolve/docker/evaluate.py` (the function that uses `time.perf_counter()` wallclock timing).

- [ ] **Step 2: Remove old stage_profile function**

Delete `stage_profile()` from `kernel-evolve/docker/evaluate.py` (the function with local `import jax` and `from xprof.convert import raw_to_tool_data`). Also remove the local `import jax` from `_has_tpu()` and add an explicit guard:

```python
def _has_tpu() -> bool:
  if jax is None:
    return False
  try:
    devices = jax.devices()
    print(f"JAX devices: {devices}", file=sys.stderr)
    return any(d.platform == "tpu" for d in devices)
  except Exception as e:
    print(f"TPU detection error: {e}", file=sys.stderr)
    return False
```

- [ ] **Step 3: Update test_stage_performance_accepts_reference_entrypoints**

In `kernel-evolve/tests/test_docker_evaluate.py`, replace `test_stage_performance_accepts_reference_entrypoints` (lines 23-39) to test `stage_benchmark` instead:

```python
def test_stage_benchmark_accepts_reference_entrypoints(tmp_path):
  evaluate = _load_evaluate_module()

  globals_dict = {
    "simple_compute": lambda M=1: M,
    "reference_fn": lambda **kwargs: kwargs["M"],
  }

  mock_lowered = MagicMock()
  mock_compiled = MagicMock()
  mock_compiled.return_value = 42
  mock_compiled.memory_analysis.side_effect = Exception("no TPU")
  mock_lowered.compile.return_value = mock_compiled
  mock_jitted = MagicMock()
  mock_jitted.lower.return_value = mock_lowered

  with patch.object(evaluate, "jax") as mock_jax, \
       patch.object(evaluate, "raw_to_tool_data", None):
    mock_jax.jit.return_value = mock_jitted
    mock_jax.profiler.ProfileOptions.return_value = MagicMock()

    result = evaluate.stage_benchmark(
      globals_dict,
      shapes=[{"M": 4}],
      trace_dir=str(tmp_path),
      warmup=1,
      n_iters=2,
    )

  assert result["ok"] is True
  assert result["latency_ms"] > 0
```

- [ ] **Step 4: Run all tests**

Run: `cd /Users/xl/Code/Glaucis/.claude/worktrees/humming-enchanting-lagoon && python -m pytest kernel-evolve/tests/ -v 2>&1 | tail -40`

Expected: All tests PASS. No test references `stage_performance` or `stage_profile` (except `stage_profile_deep` tests which are unchanged).

- [ ] **Step 5: Commit**

```bash
cd /Users/xl/Code/Glaucis/.claude/worktrees/humming-enchanting-lagoon
git add kernel-evolve/docker/evaluate.py kernel-evolve/tests/test_docker_evaluate.py
git commit -m "refactor(evaluate): remove old stage_performance and stage_profile

Both are fully replaced by stage_benchmark(). Updated test to use
stage_benchmark with mocked JAX/xprof. stage_profile_deep unchanged.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

### Task 6: Final verification

- [ ] **Step 1: Run the full test suite**

Run: `cd /Users/xl/Code/Glaucis/.claude/worktrees/humming-enchanting-lagoon && python -m pytest kernel-evolve/tests/ -v 2>&1`

Expected: All tests PASS with 0 failures.

- [ ] **Step 2: Verify no references to removed functions**

Run: `cd /Users/xl/Code/Glaucis/.claude/worktrees/humming-enchanting-lagoon && grep -rn "stage_performance\|stage_profile[^_]" kernel-evolve/ --include="*.py" | grep -v "stage_profile_deep" | grep -v "__pycache__"`

Expected: No output (no remaining references to the old functions).

- [ ] **Step 3: Review the diff**

Run: `cd /Users/xl/Code/Glaucis/.claude/worktrees/humming-enchanting-lagoon && git diff main --stat`

Verify the changes are limited to the 4 files in the spec.
