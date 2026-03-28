# Kernel Profile Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add XPlane trace-based profiling to kernel-evolve's evaluation pipeline and use `compute_ratio` as a MAP-Elites behavioral descriptor.

**Architecture:** New `profiler.py` module captures JAX profiler traces and parses them with `xprof`. `evaluate.py` gains a 4th stage (`stage_profile`). `compute_ratio` becomes a 4th MAP-Elites descriptor axis with 4 buckets.

**Tech Stack:** `xprof` (trace conversion), `jax.profiler` (trace capture), existing `kernel_evolve` framework

---

### Task 1: Add `compute_ratio` and `memory_transfer_ratio` to EvalResult

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/evaluator.py`
- Test: `kernel-evolve/tests/test_evaluator.py`

**Step 1: Write the failing test**

Add to `kernel-evolve/tests/test_evaluator.py`:

```python
def test_eval_result_success_with_profile():
  r = EvalResult.success(latency_ms=1.5, speedup=2.3, flops=1e12, compute_ratio=0.85, memory_transfer_ratio=0.15)
  assert r.compute_ratio == 0.85
  assert r.memory_transfer_ratio == 0.15
  d = r.to_dict()
  assert d["compute_ratio"] == 0.85
  assert d["memory_transfer_ratio"] == 0.15


def test_eval_result_success_without_profile():
  r = EvalResult.success(latency_ms=1.5, speedup=2.3)
  assert r.compute_ratio is None
  assert r.memory_transfer_ratio is None
  d = r.to_dict()
  assert d["compute_ratio"] is None


def test_eval_result_roundtrip_with_profile():
  r = EvalResult.success(latency_ms=1.0, speedup=1.5, compute_ratio=0.7, memory_transfer_ratio=0.3)
  d = r.to_dict()
  restored = EvalResult.from_dict(d)
  assert restored.compute_ratio == 0.7
  assert restored.memory_transfer_ratio == 0.3
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/xl/Code/Glaucis/kernel-evolve && python -m pytest tests/test_evaluator.py -v -k "profile"`
Expected: FAIL — `success()` doesn't accept `compute_ratio`

**Step 3: Write minimal implementation**

In `kernel-evolve/src/kernel_evolve/evaluator.py`, modify `EvalResult`:

```python
@dataclass
class EvalResult:
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

  @classmethod
  def success(cls, latency_ms: float, speedup: float, flops: float = 0.0,
              compute_ratio: float | None = None, memory_transfer_ratio: float | None = None) -> EvalResult:
    return cls(status=EvalStatus.SUCCESS, fitness=speedup, latency_ms=latency_ms,
               speedup=speedup, flops=flops, compute_ratio=compute_ratio,
               memory_transfer_ratio=memory_transfer_ratio)

  def to_dict(self) -> dict[str, Any]:
    return {
      "status": self.status.name,
      "fitness": self.fitness,
      "error": self.error,
      "max_diff": self.max_diff,
      "latency_ms": self.latency_ms,
      "speedup": self.speedup,
      "flops": self.flops,
      "compute_ratio": self.compute_ratio,
      "memory_transfer_ratio": self.memory_transfer_ratio,
      "metadata": self.metadata,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> EvalResult:
    return cls(
      status=EvalStatus[data["status"]],
      fitness=data.get("fitness", 0.0),
      error=data.get("error", ""),
      max_diff=data.get("max_diff", 0.0),
      latency_ms=data.get("latency_ms", 0.0),
      speedup=data.get("speedup", 0.0),
      flops=data.get("flops", 0.0),
      compute_ratio=data.get("compute_ratio"),
      memory_transfer_ratio=data.get("memory_transfer_ratio"),
      metadata=data.get("metadata", {}),
    )
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/xl/Code/Glaucis/kernel-evolve && python -m pytest tests/test_evaluator.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add kernel-evolve/src/kernel_evolve/evaluator.py kernel-evolve/tests/test_evaluator.py
git commit -m "feat(evaluator): add compute_ratio and memory_transfer_ratio to EvalResult"
```

---

### Task 2: Add `compute_profile` descriptor to MAP-Elites population

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/population.py`
- Modify: `kernel-evolve/tests/test_population.py`

**Step 1: Write the failing tests**

Add to `kernel-evolve/tests/test_population.py`:

```python
def test_compute_profile_descriptor():
  desc = BehaviorDescriptor(block_size=128, pipeline_stages=2, memory_strategy="scratch", compute_profile="high")
  assert desc.cell_key() == (128, 2, "scratch", "high")


def test_compute_profile_default():
  desc = BehaviorDescriptor(block_size=128, pipeline_stages=2, memory_strategy="scratch")
  assert desc.compute_profile == "medium"
  assert desc.cell_key() == (128, 2, "scratch", "medium")


def test_archive_capacity_with_compute_profile():
  archive = Archive()
  assert archive.capacity == 4 * 4 * 3 * 4  # 192 cells


def test_ratio_to_profile_bucket():
  from kernel_evolve.population import ratio_to_compute_profile
  assert ratio_to_compute_profile(0.1) == "very_low"
  assert ratio_to_compute_profile(0.3) == "low"
  assert ratio_to_compute_profile(0.6) == "medium"
  assert ratio_to_compute_profile(0.9) == "high"
  assert ratio_to_compute_profile(None) == "medium"
  assert ratio_to_compute_profile(0.0) == "very_low"
  assert ratio_to_compute_profile(1.0) == "high"
  assert ratio_to_compute_profile(0.25) == "low"
  assert ratio_to_compute_profile(0.75) == "high"
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/xl/Code/Glaucis/kernel-evolve && python -m pytest tests/test_population.py -v -k "compute_profile or ratio_to"`
Expected: FAIL

**Step 3: Write minimal implementation**

In `kernel-evolve/src/kernel_evolve/population.py`:

Add the `ratio_to_compute_profile` function:

```python
def ratio_to_compute_profile(compute_ratio: float | None) -> str:
  if compute_ratio is None:
    return "medium"
  if compute_ratio < 0.25:
    return "very_low"
  if compute_ratio < 0.50:
    return "low"
  if compute_ratio < 0.75:
    return "medium"
  return "high"
```

Modify `BehaviorDescriptor`:

```python
@dataclass
class BehaviorDescriptor:
  """Behavioral descriptors that define a cell in the MAP-Elites grid."""

  block_size: int = 128
  pipeline_stages: int = 1
  memory_strategy: str = "scratch"
  compute_profile: str = "medium"

  def cell_key(self) -> tuple:
    return (self.block_size, self.pipeline_stages, self.memory_strategy, self.compute_profile)
```

Modify `Archive.__init__` default axes:

```python
self._descriptor_axes = descriptor_axes or {
  "block_size": [64, 128, 256, 512],
  "pipeline_stages": [1, 2, 3, 4],
  "memory_strategy": ["scratch", "hbm", "rmw"],
  "compute_profile": ["very_low", "low", "medium", "high"],
}
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/xl/Code/Glaucis/kernel-evolve && python -m pytest tests/test_population.py -v`
Expected: ALL PASS (update the `archive` fixture to include `compute_profile` axis, and fix `test_archive_empty_on_creation` capacity assertion from `48` to `192`)

Note: The existing `archive` fixture in `test_population.py` hard-codes 3 axes without `compute_profile`. Update it:

```python
@pytest.fixture
def archive():
  return Archive(
    descriptor_axes={
      "block_size": [64, 128, 256, 512],
      "pipeline_stages": [1, 2, 3, 4],
      "memory_strategy": ["scratch", "hbm", "rmw"],
      "compute_profile": ["very_low", "low", "medium", "high"],
    }
  )
```

And fix `test_archive_empty_on_creation`:

```python
def test_archive_empty_on_creation(archive):
  assert archive.size == 0
  assert archive.best is None
  assert archive.capacity == 4 * 4 * 3 * 4  # 192 cells
```

Also update existing test variants that create `BehaviorDescriptor` without `compute_profile` — they should still work since `compute_profile` defaults to `"medium"`.

**Step 5: Commit**

```bash
git add kernel-evolve/src/kernel_evolve/population.py kernel-evolve/tests/test_population.py
git commit -m "feat(population): add compute_profile descriptor axis for MAP-Elites"
```

---

### Task 3: Create profiler module with `analyze_trace`

**Files:**
- Create: `kernel-evolve/src/kernel_evolve/profiler.py`
- Create: `kernel-evolve/tests/test_profiler.py`

**Step 1: Write the failing tests**

Create `kernel-evolve/tests/test_profiler.py`:

```python
"""Tests for the XPlane trace profiler module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from kernel_evolve.profiler import analyze_trace, capture_trace, stage_profile


# Minimal Chrome trace JSON matching the structure produced by xprof trace_viewer
MOCK_TRACE_JSON = json.dumps({
  "traceEvents": [
    # Process metadata event identifying TPU:0
    {"pid": 1, "tid": 0, "ph": "M", "name": "process_name", "args": {"name": "/device:TPU:0"}},
    # Three jit_computation events (3 runs in the profiling loop)
    {"pid": 1, "tid": 1, "ph": "X", "name": "jit_computation.1", "ts": 100, "dur": 50},
    {"pid": 1, "tid": 1, "ph": "X", "name": "jit_computation.2", "ts": 200, "dur": 50},
    {"pid": 1, "tid": 1, "ph": "X", "name": "jit_computation.3", "ts": 300, "dur": 50},
    # SyncWait event in the last iteration window (250-350)
    {"pid": 1, "tid": 1, "ph": "X", "name": "SyncWait:hbm_transfer", "ts": 260, "dur": 20},
    # SyncWait outside the window (should be excluded)
    {"pid": 1, "tid": 1, "ph": "X", "name": "SyncWait:old", "ts": 100, "dur": 10},
    # Non-TPU:0 event (should be excluded)
    {"pid": 2, "tid": 0, "ph": "M", "name": "process_name", "args": {"name": "/host:CPU"}},
    {"pid": 2, "tid": 1, "ph": "X", "name": "SyncWait:cpu", "ts": 260, "dur": 30},
  ]
})


@patch("kernel_evolve.profiler.raw_to_tool_data")
def test_analyze_trace_computes_ratios(mock_xprof):
  mock_xprof.xspace_to_tool_data.return_value = (MOCK_TRACE_JSON, None)
  result = analyze_trace("/fake/trace.xplane.pb")
  # Window: ts=250 (end of 2nd jit_computation) to ts=350 (end of 3rd)
  # Total = 100, SyncWait = 20, ratio = 0.2
  assert result["memory_transfer_ratio"] == pytest.approx(0.2)
  assert result["compute_ratio"] == pytest.approx(0.8)


@patch("kernel_evolve.profiler.raw_to_tool_data")
def test_analyze_trace_no_tpu_events(mock_xprof):
  mock_xprof.xspace_to_tool_data.return_value = (json.dumps({"traceEvents": []}), None)
  result = analyze_trace("/fake/empty.xplane.pb")
  assert result is None


@patch("kernel_evolve.profiler.raw_to_tool_data")
def test_analyze_trace_no_syncwait(mock_xprof):
  trace = json.dumps({
    "traceEvents": [
      {"pid": 1, "tid": 0, "ph": "M", "name": "process_name", "args": {"name": "/device:TPU:0"}},
      {"pid": 1, "tid": 1, "ph": "X", "name": "jit_computation.1", "ts": 100, "dur": 50},
      {"pid": 1, "tid": 1, "ph": "X", "name": "jit_computation.2", "ts": 200, "dur": 50},
      {"pid": 1, "tid": 1, "ph": "X", "name": "jit_computation.3", "ts": 300, "dur": 50},
    ]
  })
  mock_xprof.xspace_to_tool_data.return_value = (trace, None)
  result = analyze_trace("/fake/trace.xplane.pb")
  assert result["memory_transfer_ratio"] == pytest.approx(0.0)
  assert result["compute_ratio"] == pytest.approx(1.0)
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/xl/Code/Glaucis/kernel-evolve && python -m pytest tests/test_profiler.py -v -k "analyze_trace"`
Expected: FAIL — `kernel_evolve.profiler` doesn't exist

**Step 3: Write minimal implementation**

Create `kernel-evolve/src/kernel_evolve/profiler.py`:

```python
"""XPlane trace-based Pallas kernel profiler.

Captures JAX profiler traces and analyzes them using xprof to extract
compute_ratio and memory_transfer_ratio for MAP-Elites fitness signals.

Reference: accelerator-agents/MaxKernel/.../analyze_profile.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from xprof.convert import raw_to_tool_data


def analyze_trace(xplane_path: str) -> dict[str, float] | None:
  """Parse an .xplane.pb file and extract compute vs memory transfer ratio.

  Returns {"compute_ratio": float, "memory_transfer_ratio": float} or None on failure.
  """
  tool_data_result, _ = raw_to_tool_data.xspace_to_tool_data([xplane_path], "trace_viewer", {})
  trace_data = json.loads(tool_data_result)
  events = trace_data.get("traceEvents", [])

  # Find the pid for /device:TPU:0
  pid = None
  for event in events:
    if "args" in event and event["args"].get("name") == "/device:TPU:0":
      pid = event.get("pid")
      break

  if pid is None:
    return None

  # Collect TPU:0 events and jit_computation events
  events_for_tpu_0 = []
  jit_computation_events = []
  for event in events:
    if event.get("pid") != pid:
      continue
    events_for_tpu_0.append(event)
    name = event.get("name") or ""
    if "jit_computation" in name:
      jit_computation_events.append(event)

  if len(jit_computation_events) < 2:
    return None

  # Focus on the last iteration window (between end of 2nd-to-last and end of last jit_computation)
  start_last = jit_computation_events[-2]["ts"] + jit_computation_events[-2]["dur"]
  end_last = jit_computation_events[-1]["ts"] + jit_computation_events[-1]["dur"]

  # Sum SyncWait durations within the window
  sync_wait_total = 0
  for event in events_for_tpu_0:
    if "dur" not in event:
      continue
    evt_start = event["ts"]
    evt_end = evt_start + event["dur"]
    if evt_start >= start_last and evt_end <= end_last:
      name = event.get("name") or ""
      if "SyncWait" in name:
        sync_wait_total += event["dur"]

  total_time = end_last - start_last
  if total_time <= 0:
    return None

  ratio = sync_wait_total / total_time
  return {
    "compute_ratio": 1.0 - ratio,
    "memory_transfer_ratio": ratio,
  }


def capture_trace(kernel_fn, shapes: dict[str, Any], trace_dir: str,
                  warmup: int = 3, runs: int = 3) -> str | None:
  """Run kernel_fn under jax.profiler and return the path to the .xplane.pb file."""
  import jax

  trace_path = Path(trace_dir)
  trace_path.mkdir(parents=True, exist_ok=True)

  # Warmup runs (outside profiler)
  for _ in range(warmup):
    out = kernel_fn(**shapes)
    if hasattr(out, "block_until_ready"):
      out.block_until_ready()

  # Profiled runs
  options = jax.profiler.ProfileOptions()
  options.python_tracer_level = 0
  options.host_tracer_level = 2
  options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}

  jax.profiler.start_trace(trace_dir, profiler_options=options)
  for _ in range(runs):
    out = kernel_fn(**shapes)
    if hasattr(out, "block_until_ready"):
      out.block_until_ready()
  jax.profiler.stop_trace()

  # Search for .xplane.pb file
  for root, _dirs, files in os.walk(trace_dir):
    for f in files:
      if f.endswith(".xplane.pb"):
        return os.path.join(root, f)

  return None


def stage_profile(exec_globals: dict[str, Any], shapes: list[dict[str, Any]],
                  trace_dir: str = "/tmp/xplane_trace") -> dict[str, Any]:
  """Stage 4: Profile kernel using JAX profiler and xprof trace analysis.

  Non-fatal: returns ok=False on failure without stopping the evaluation pipeline.
  """
  try:
    kernel_fn = exec_globals.get("optimized_compute") or exec_globals.get("kernel_fn")
    if kernel_fn is None:
      return {"ok": False, "error": "No kernel_fn found for profiling"}

    xplane_path = capture_trace(kernel_fn, shapes[0], trace_dir)
    if xplane_path is None:
      return {"ok": False, "error": "No .xplane.pb file generated"}

    result = analyze_trace(xplane_path)
    if result is None:
      return {"ok": False, "error": "Could not parse trace (no TPU:0 or jit_computation events)"}

    return {
      "ok": True,
      "compute_ratio": result["compute_ratio"],
      "memory_transfer_ratio": result["memory_transfer_ratio"],
    }
  except Exception:
    return {"ok": False, "error": f"Profile error: {traceback.format_exc()}"}
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/xl/Code/Glaucis/kernel-evolve && python -m pytest tests/test_profiler.py -v -k "analyze_trace"`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add kernel-evolve/src/kernel_evolve/profiler.py kernel-evolve/tests/test_profiler.py
git commit -m "feat(profiler): add XPlane trace analysis module with analyze_trace"
```

---

### Task 4: Add tests for `capture_trace` and `stage_profile`

**Files:**
- Modify: `kernel-evolve/tests/test_profiler.py`

**Step 1: Write the tests**

Add to `kernel-evolve/tests/test_profiler.py`:

```python
@patch("kernel_evolve.profiler.raw_to_tool_data")
def test_capture_trace_finds_xplane(mock_xprof, tmp_path):
  # Create a fake .xplane.pb file in the expected nested directory structure
  profile_dir = tmp_path / "plugins" / "profile" / "2026_01_01_00_00_00"
  profile_dir.mkdir(parents=True)
  xplane_file = profile_dir / "worker-0.xplane.pb"
  xplane_file.write_bytes(b"fake")

  mock_kernel = MagicMock(return_value=MagicMock(block_until_ready=MagicMock()))

  with patch("kernel_evolve.profiler.jax") as mock_jax:
    mock_jax.profiler.ProfileOptions.return_value = MagicMock()
    result = capture_trace(mock_kernel, {"M": 64, "N": 64}, str(tmp_path), warmup=1, runs=1)

  assert result is not None
  assert result.endswith(".xplane.pb")


def test_capture_trace_returns_none_when_no_xplane(tmp_path):
  mock_kernel = MagicMock(return_value=MagicMock(block_until_ready=MagicMock()))

  with patch("kernel_evolve.profiler.jax") as mock_jax:
    mock_jax.profiler.ProfileOptions.return_value = MagicMock()
    result = capture_trace(mock_kernel, {"M": 64}, str(tmp_path), warmup=1, runs=1)

  assert result is None


@patch("kernel_evolve.profiler.analyze_trace")
@patch("kernel_evolve.profiler.capture_trace")
def test_stage_profile_success(mock_capture, mock_analyze):
  mock_capture.return_value = "/tmp/trace.xplane.pb"
  mock_analyze.return_value = {"compute_ratio": 0.85, "memory_transfer_ratio": 0.15}

  exec_globals = {"optimized_compute": lambda **kw: None}
  result = stage_profile(exec_globals, [{"M": 64, "N": 64}])

  assert result["ok"] is True
  assert result["compute_ratio"] == 0.85
  assert result["memory_transfer_ratio"] == 0.15


@patch("kernel_evolve.profiler.capture_trace")
def test_stage_profile_no_xplane(mock_capture):
  mock_capture.return_value = None

  exec_globals = {"optimized_compute": lambda **kw: None}
  result = stage_profile(exec_globals, [{"M": 64}])

  assert result["ok"] is False
  assert "No .xplane.pb" in result["error"]


def test_stage_profile_no_kernel_fn():
  result = stage_profile({}, [{"M": 64}])
  assert result["ok"] is False
  assert "No kernel_fn" in result["error"]
```

**Step 2: Run tests to verify they pass**

Run: `cd /Users/xl/Code/Glaucis/kernel-evolve && python -m pytest tests/test_profiler.py -v`
Expected: ALL PASS

Note: The `capture_trace` test mocking needs to handle `import jax` inside the function. The implementation uses a late import (`import jax` inside the function body), so we patch `kernel_evolve.profiler.jax` module-level after it's first imported. Adjust the import in `profiler.py` to be at module level but guarded:

Actually, looking at the implementation more carefully, `capture_trace` does `import jax` inside the function body. For testing, we need to patch it differently. Let's restructure: move `import jax` to the top of `capture_trace` but since `jax` may not be installed in the test env, we'll lazy-import it. The tests will mock `kernel_evolve.profiler.jax` — but for that to work, `jax` needs to be patchable at module level.

Simplest approach: have `capture_trace` do `import jax` at function level, and in tests, insert a mock `jax` into `sys.modules` before importing profiler, OR use `@patch` on the `jax` module attribute within `kernel_evolve.profiler`.

Let's keep the lazy import and adjust test approach — the test for `capture_trace` will inject `jax` into the profiler module's namespace:

```python
import kernel_evolve.profiler as profiler_mod

def test_capture_trace_finds_xplane(tmp_path):
  profile_dir = tmp_path / "plugins" / "profile" / "2026_01_01"
  profile_dir.mkdir(parents=True)
  (profile_dir / "w-0.xplane.pb").write_bytes(b"fake")

  mock_kernel = MagicMock(return_value=MagicMock(block_until_ready=MagicMock()))
  mock_jax = MagicMock()

  with patch.dict("sys.modules", {"jax": mock_jax, "jax.profiler": mock_jax.profiler}):
    result = capture_trace(mock_kernel, {"M": 64}, str(tmp_path), warmup=1, runs=1)

  assert result is not None
  assert result.endswith(".xplane.pb")
```

**Step 3: Commit**

```bash
git add kernel-evolve/tests/test_profiler.py
git commit -m "test(profiler): add tests for capture_trace and stage_profile"
```

---

### Task 5: Integrate `stage_profile` into evaluate.py

**Files:**
- Modify: `kernel-evolve/docker/evaluate.py`

**Step 1: Modify evaluate.py**

Add the import at the top of `evaluate.py`:

```python
from kernel_evolve.profiler import stage_profile
```

Wait — `evaluate.py` runs inside a K8s pod and imports `kernel_evolve.profiler` which imports `xprof`. But the pod clones the repo and runs `evaluate.py` directly. The import path may not be set up. Looking at the K8s job template, it runs:

```bash
python kernel-evolve/docker/evaluate.py --eval-payload "$EVAL_PAYLOAD"
```

So `kernel_evolve` package is NOT installed. `evaluate.py` is a standalone script. We should NOT import from `kernel_evolve` in evaluate.py. Instead, we need to either:
- (a) Inline the profiling code in evaluate.py, or
- (b) Add the profiler as a separate file that evaluate.py can import via sys.path manipulation, or
- (c) Install kernel_evolve in the K8s job.

The simplest approach that matches the existing pattern: **copy the core profiling functions into evaluate.py** as local functions since evaluate.py is a self-contained script. The `profiler.py` module in `kernel_evolve` package serves as the canonical, tested implementation, and `evaluate.py` contains a slimmed-down inline version.

Actually, better: put the profiling functions in a separate file `kernel-evolve/docker/profiler.py` that evaluate.py can import (same directory), AND keep `kernel-evolve/src/kernel_evolve/profiler.py` as the package version. But this duplicates code.

Best approach: Add `sys.path` to include the kernel_evolve src directory, since the repo is already cloned. Change the K8s job command to also install the package, or just add to sys.path.

Looking at the K8s job: it already clones the repo to `/workspace`. So we can add:

```bash
cd /workspace && pip install -e kernel-evolve/
```

Or simpler: just add `sys.path.insert(0, "/workspace/kernel-evolve/src")` in evaluate.py. But this couples evaluate.py to the pod layout.

**Decision: Keep evaluate.py self-contained.** Add `stage_profile` directly in evaluate.py as an inline function that imports `xprof` and `jax.profiler`. This matches the existing pattern where all stage functions are in evaluate.py.

Add after `stage_performance`:

```python
def stage_profile(exec_globals, shapes, trace_dir="/tmp/xplane_trace"):
  try:
    import jax
    kernel_fn = exec_globals.get("optimized_compute") or exec_globals.get("kernel_fn")
    if kernel_fn is None:
      return {"ok": False, "error": "No kernel_fn found for profiling"}

    # Warmup
    shape = shapes[0]
    for _ in range(3):
      out = kernel_fn(**shape)
      if hasattr(out, "block_until_ready"):
        out.block_until_ready()

    # Capture trace
    import os
    os.makedirs(trace_dir, exist_ok=True)
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 2
    options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}

    jax.profiler.start_trace(trace_dir, profiler_options=options)
    for _ in range(3):
      out = kernel_fn(**shape)
      if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    jax.profiler.stop_trace()

    # Find .xplane.pb
    xplane_path = None
    for root, _dirs, files in os.walk(trace_dir):
      for f in files:
        if f.endswith(".xplane.pb"):
          xplane_path = os.path.join(root, f)
          break
      if xplane_path:
        break

    if xplane_path is None:
      return {"ok": False, "error": "No .xplane.pb file generated"}

    # Analyze trace
    from xprof.convert import raw_to_tool_data
    import json as _json
    tool_data_result, _ = raw_to_tool_data.xspace_to_tool_data([xplane_path], "trace_viewer", {})
    trace_data = _json.loads(tool_data_result)
    events = trace_data.get("traceEvents", [])

    pid = None
    for event in events:
      if "args" in event and event["args"].get("name") == "/device:TPU:0":
        pid = event.get("pid")
        break

    if pid is None:
      return {"ok": False, "error": "No /device:TPU:0 in trace"}

    jit_events = []
    tpu_events = []
    for event in events:
      if event.get("pid") != pid:
        continue
      tpu_events.append(event)
      name = event.get("name") or ""
      if "jit_computation" in name:
        jit_events.append(event)

    if len(jit_events) < 2:
      return {"ok": False, "error": f"Only {len(jit_events)} jit_computation events found"}

    start_last = jit_events[-2]["ts"] + jit_events[-2]["dur"]
    end_last = jit_events[-1]["ts"] + jit_events[-1]["dur"]

    sync_wait_total = 0
    for event in tpu_events:
      if "dur" not in event:
        continue
      if event["ts"] >= start_last and (event["ts"] + event["dur"]) <= end_last:
        if "SyncWait" in (event.get("name") or ""):
          sync_wait_total += event["dur"]

    total_time = end_last - start_last
    if total_time <= 0:
      return {"ok": False, "error": "Zero total computation time in trace"}

    ratio = sync_wait_total / total_time
    return {
      "ok": True,
      "compute_ratio": round(1.0 - ratio, 4),
      "memory_transfer_ratio": round(ratio, 4),
    }
  except Exception:
    return {"ok": False, "error": f"Profile error: {traceback.format_exc()}"}
```

Modify `main()` to call `stage_profile` after performance and include results:

```python
  # After perf_result and speedup calculation, before final result print:

  # Stage 4: Profile (non-fatal)
  profile_result = stage_profile(compile_result["globals"], request["shapes"])
  compute_ratio = None
  memory_transfer_ratio = None
  if profile_result["ok"]:
    compute_ratio = profile_result["compute_ratio"]
    memory_transfer_ratio = profile_result["memory_transfer_ratio"]
    print(f"Profile: compute_ratio={compute_ratio}, memory_transfer_ratio={memory_transfer_ratio}", file=sys.stderr)
  else:
    print(f"Profile skipped: {profile_result.get('error', 'unknown')}", file=sys.stderr)

  result = {
    "status": "SUCCESS",
    "fitness": speedup,
    "latency_ms": perf_result["latency_ms"],
    "speedup": speedup,
    "flops": 0.0,
    "compute_ratio": compute_ratio,
    "memory_transfer_ratio": memory_transfer_ratio,
  }
  print(f"EVAL_RESULT:{json.dumps(result)}")
```

**Step 2: Verify evaluate.py is syntactically valid**

Run: `python -c "import ast; ast.parse(open('/Users/xl/Code/Glaucis/kernel-evolve/docker/evaluate.py').read()); print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add kernel-evolve/docker/evaluate.py
git commit -m "feat(evaluate): add stage_profile for XPlane trace analysis"
```

---

### Task 6: Update CI dispatcher to extract profile metrics

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/ci_dispatcher.py`
- Modify: `kernel-evolve/src/kernel_evolve/engine.py`

**Step 1: Update `_collect_result` in ci_dispatcher.py**

The `_collect_result` method calls `EvalResult.from_dict()` which already handles `compute_ratio`/`memory_transfer_ratio` from Task 1. No change needed in ci_dispatcher.py itself.

**Step 2: Update engine.py to use compute_profile descriptor**

In `engine.py`, the `_evolve_one` method builds a `BehaviorDescriptor` from the LLM's `suggested_descriptor`. After evaluation, we now also have `compute_ratio` from the eval result. We need to map it to `compute_profile`.

Modify `engine.py` line ~162 (descriptor construction):

```python
from kernel_evolve.population import ratio_to_compute_profile

# After eval result is available (line ~188):
descriptor = BehaviorDescriptor(
  block_size=desc_data.get("block_size", 128),
  pipeline_stages=desc_data.get("pipeline_stages", 1),
  memory_strategy=desc_data.get("memory_strategy", "scratch"),
  compute_profile=ratio_to_compute_profile(result.compute_ratio),
)
```

The descriptor must be set AFTER evaluation since `compute_ratio` comes from the eval result. Currently, the descriptor is set before eval and the variant is created before eval. We need to restructure `_evolve_one` slightly: create the variant after evaluation, or update the descriptor after eval.

Looking at the current flow:
1. Line 161-166: Create descriptor from LLM suggestion
2. Line 168-176: Create variant with that descriptor
3. Line 178-186: Run eval
4. Line 188-189: Set fitness and insert into archive

We need to set `compute_profile` AFTER eval but BEFORE archive insert. Simplest: update variant.descriptor after eval:

```python
    result = await self._evaluator.evaluate(eval_request)

    variant.fitness = result.fitness
    variant.descriptor.compute_profile = ratio_to_compute_profile(result.compute_ratio)
    island.insert(variant)
```

**Step 3: Write tests**

Add to `kernel-evolve/tests/test_engine.py`:

```python
@pytest.mark.asyncio
async def test_engine_sets_compute_profile(minimal_config, mock_provider, tmp_path):
  minimal_config.logging.output_dir = str(tmp_path / "run")
  eval_result = EvalResult.success(latency_ms=1.0, speedup=2.0, compute_ratio=0.9, memory_transfer_ratio=0.1)
  mock_evaluator = AsyncMock()
  mock_evaluator.evaluate.return_value = eval_result

  engine = EvolutionEngine(
    config=minimal_config,
    provider=mock_provider,
    evaluator=mock_evaluator,
    template_code="# EVOLVE-BLOCK-START\npass\n# EVOLVE-BLOCK-END",
    reference_code="def ref(): pass",
  )
  await engine.run_generation()
  best = engine.best
  assert best is not None
  assert best.descriptor.compute_profile == "high"
```

**Step 4: Run tests**

Run: `cd /Users/xl/Code/Glaucis/kernel-evolve && python -m pytest tests/test_engine.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add kernel-evolve/src/kernel_evolve/engine.py kernel-evolve/tests/test_engine.py
git commit -m "feat(engine): set compute_profile descriptor from eval result"
```

---

### Task 7: Add `xprof` dependency to K8s job and pyproject.toml

**Files:**
- Modify: `.github/ci/kernel-eval-job.yaml`
- Modify: `kernel-evolve/pyproject.toml`

**Step 1: Update K8s job template**

In `.github/ci/kernel-eval-job.yaml`, add `xprof` to the `uv pip install` command on line 33:

```yaml
          uv pip install --system --prerelease=allow --no-cache \
            jax jaxlib libtpu numpy xprof \
            --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
```

**Step 2: Update pyproject.toml**

Add `xprof` as an optional dependency:

```toml
[project.optional-dependencies]
anthropic = ["anthropic>=0.40"]
google = ["google-genai>=1.0"]
openai = ["openai>=1.0"]
charts = ["matplotlib>=3.7"]
profile = ["xprof"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "ruff>=0.4"]
```

**Step 3: Commit**

```bash
git add .github/ci/kernel-eval-job.yaml kernel-evolve/pyproject.toml
git commit -m "build: add xprof dependency to K8s job and pyproject.toml"
```

---

### Task 8: Create E2E verification script

**Files:**
- Create: `kernel-evolve/scripts/verify_profile.py`

**Step 1: Write the verification script**

This script can be run on a real TPU to validate the full profiling pipeline end-to-end. It uses the matmul kernel from examples.

```python
#!/usr/bin/env python3
"""End-to-end verification of the kernel profiling pipeline.

Run on a TPU to validate: matmul kernel -> jax.profiler trace -> .xplane.pb -> analyze_trace -> metrics.

Usage:
  python kernel-evolve/scripts/verify_profile.py [--trace-dir /tmp/xplane_trace]
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
  parser = argparse.ArgumentParser(description="Verify kernel profiling pipeline")
  parser.add_argument("--trace-dir", default="/tmp/xplane_verify", help="Directory for trace output")
  args = parser.parse_args()

  print("=== Kernel Profile E2E Verification ===\n")

  # Step 1: Check JAX and TPU
  print("[1/5] Checking JAX and TPU...")
  import jax
  devices = jax.devices()
  print(f"  JAX devices: {devices}")
  has_tpu = any(d.platform == "tpu" for d in devices)
  print(f"  TPU available: {has_tpu}")
  if not has_tpu:
    print("  WARNING: No TPU detected. Profiling will proceed but trace may not contain TPU events.")

  # Step 2: Run matmul kernel
  print("\n[2/5] Running matmul kernel...")
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  def matmul_kernel(x_ref, y_ref, o_ref):
    acc = jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)
    o_ref[...] = acc.astype(o_ref.dtype)

  M, N, K = 1024, 1024, 1024
  BLOCK_M, BLOCK_N = 128, 128

  def optimized_compute(M=M, N=N, K=K):
    x = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=jnp.bfloat16)
    y = jax.random.normal(jax.random.PRNGKey(1), (K, N), dtype=jnp.bfloat16)
    return pl.pallas_call(
      matmul_kernel,
      out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
      grid=(M // BLOCK_M, N // BLOCK_N),
      in_specs=[
        pl.BlockSpec((BLOCK_M, K), lambda i, j: (i, 0)),
        pl.BlockSpec((K, BLOCK_N), lambda i, j: (0, j)),
      ],
      out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),
    )(x, y)

  # Warmup
  out = jax.block_until_ready(optimized_compute())
  print(f"  Output shape: {out.shape}, dtype: {out.dtype}")

  # Step 3: Capture trace
  print(f"\n[3/5] Capturing profiler trace to {args.trace_dir}...")
  os.makedirs(args.trace_dir, exist_ok=True)
  options = jax.profiler.ProfileOptions()
  options.python_tracer_level = 0
  options.host_tracer_level = 2
  options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}

  jax.profiler.start_trace(args.trace_dir, profiler_options=options)
  for i in range(3):
    jax.block_until_ready(optimized_compute())
  jax.profiler.stop_trace()
  print("  Trace captured.")

  # Step 4: Find .xplane.pb
  print("\n[4/5] Searching for .xplane.pb file...")
  xplane_path = None
  for root, _dirs, files in os.walk(args.trace_dir):
    for f in files:
      if f.endswith(".xplane.pb"):
        xplane_path = os.path.join(root, f)
        break
    if xplane_path:
      break

  if xplane_path is None:
    print("  ERROR: No .xplane.pb file found!")
    # List what was actually generated
    for root, _dirs, files in os.walk(args.trace_dir):
      for f in files:
        print(f"  Found: {os.path.join(root, f)}")
    sys.exit(1)

  file_size = os.path.getsize(xplane_path)
  print(f"  Found: {xplane_path} ({file_size} bytes)")

  # Step 5: Analyze trace
  print("\n[5/5] Analyzing trace with xprof...")
  from xprof.convert import raw_to_tool_data
  tool_data_result, _ = raw_to_tool_data.xspace_to_tool_data([xplane_path], "trace_viewer", {})
  trace_data = json.loads(tool_data_result)
  events = trace_data.get("traceEvents", [])
  print(f"  Total trace events: {len(events)}")

  # Find TPU:0
  pid = None
  for event in events:
    if "args" in event and event["args"].get("name") == "/device:TPU:0":
      pid = event.get("pid")
      break

  if pid is None:
    print("  ERROR: No /device:TPU:0 process found in trace!")
    process_names = set()
    for e in events:
      if "args" in e and "name" in e.get("args", {}):
        process_names.add(e["args"]["name"])
    print(f"  Available processes: {process_names}")
    sys.exit(1)

  jit_events = [e for e in events if e.get("pid") == pid and "jit_computation" in (e.get("name") or "")]
  print(f"  jit_computation events: {len(jit_events)}")

  if len(jit_events) < 2:
    print("  ERROR: Not enough jit_computation events for analysis!")
    sys.exit(1)

  start_last = jit_events[-2]["ts"] + jit_events[-2]["dur"]
  end_last = jit_events[-1]["ts"] + jit_events[-1]["dur"]
  total_time = end_last - start_last

  sync_wait = 0
  tpu_events = [e for e in events if e.get("pid") == pid]
  for e in tpu_events:
    if "dur" not in e:
      continue
    if e["ts"] >= start_last and (e["ts"] + e["dur"]) <= end_last:
      if "SyncWait" in (e.get("name") or ""):
        sync_wait += e["dur"]

  ratio = sync_wait / total_time if total_time > 0 else 0
  compute_ratio = 1.0 - ratio
  memory_transfer_ratio = ratio

  print(f"\n{'='*50}")
  print(f"  compute_ratio:          {compute_ratio:.4f}")
  print(f"  memory_transfer_ratio:  {memory_transfer_ratio:.4f}")
  print(f"  total_time (us):        {total_time}")
  print(f"  sync_wait (us):         {sync_wait}")
  print(f"{'='*50}")

  from kernel_evolve.population import ratio_to_compute_profile
  bucket = ratio_to_compute_profile(compute_ratio)
  print(f"  MAP-Elites bucket:      {bucket}")
  print("\nVERIFICATION PASSED")


if __name__ == "__main__":
  main()
```

**Step 2: Commit**

```bash
git add kernel-evolve/scripts/verify_profile.py
git commit -m "feat(scripts): add E2E profile verification script for TPU"
```

---

### Task 9: Run full test suite and verify

**Step 1: Run all tests**

Run: `cd /Users/xl/Code/Glaucis/kernel-evolve && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 2: Run ruff lint**

Run: `cd /Users/xl/Code/Glaucis/kernel-evolve && python -m ruff check src/ tests/ docker/ scripts/`
Expected: No errors

**Step 3: Fix any issues found**

Address any lint or test failures.

**Step 4: Final commit if needed**

```bash
git add -A && git commit -m "fix: address lint and test issues"
```
