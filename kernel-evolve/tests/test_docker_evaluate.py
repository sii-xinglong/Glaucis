"""Tests for the TPU pod evaluator helpers."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch


def _load_evaluate_module():
  module_path = Path(__file__).resolve().parents[1] / "docker" / "evaluate.py"
  spec = importlib.util.spec_from_file_location("kernel_evolve_docker_evaluate", module_path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


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


def test_stage_correctness_uses_reference_entrypoint_names():
  evaluate = _load_evaluate_module()

  kernel_code = """
def optimized_compute(M=1):
  return M + 1
"""
  reference_code = """
def simple_compute(M=1):
  return M + 1
"""
  compiled = evaluate.stage_compile(kernel_code)

  result = evaluate.stage_correctness(
    kernel_code,
    reference_code,
    shapes=[{"M": 3}],
    rtol=1e-2,
    atol=0.0,
    exec_globals=compiled["globals"],
  )

  assert result["ok"] is True
  assert result["max_diff"] == 0.0


def test_decode_batch_request():
  """decode_request should handle batch payload format."""
  from kernel_evolve.docker_evaluate_helpers import decode_request

  payload = {
    "batch": True,
    "reference_code": "def ref(): pass",
    "shapes": [{"M": 64}],
    "variants": [
      {"variant_id": "v1", "kernel_code": "def k1(): pass"},
    ],
  }
  b64 = base64.b64encode(json.dumps(payload).encode()).decode()
  result = decode_request(b64)
  assert result["batch"] is True
  assert len(result["variants"]) == 1


def test_batch_dispatch_calls_subprocess():
  """batch_dispatch should spawn subprocesses and collect results."""
  from kernel_evolve.docker_evaluate_helpers import batch_dispatch

  payload = {
    "batch": True,
    "reference_code": "def ref(): pass",
    "shapes": [{"M": 64}],
    "rtol": 0.01,
    "atol": 1.0,
    "variants": [
      {"variant_id": "v1-tiling", "kernel_code": "def k(): pass"},
      {"variant_id": "v2-pipe", "kernel_code": "def k(): pass"},
    ],
  }

  fake_result_json = json.dumps({"status": "SUCCESS", "speedup": 1.5, "latency_ms": 0.5, "fitness": 1.5})
  fake_stdout = f"some output\nEVAL_RESULT:{fake_result_json}\n"
  mock_completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=fake_stdout, stderr="")

  with patch("kernel_evolve.docker_evaluate_helpers.subprocess.run", return_value=mock_completed) as mock_run:
    results = batch_dispatch(payload, evaluator_script="/fake/evaluate.py", per_variant_timeout=60)

  assert mock_run.call_count == 2
  assert len(results) == 2
  assert all("EVAL_RESULT:" in line for line in results)


def test_batch_dispatch_handles_subprocess_crash():
  """batch_dispatch should emit COMPILE_ERROR for crashed subprocesses."""
  from kernel_evolve.docker_evaluate_helpers import batch_dispatch

  payload = {
    "batch": True,
    "reference_code": "def ref(): pass",
    "shapes": [{"M": 64}],
    "variants": [
      {"variant_id": "v1-crash", "kernel_code": "raise Exception('boom')"},
    ],
  }

  mock_completed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="Segmentation fault")

  with patch("kernel_evolve.docker_evaluate_helpers.subprocess.run", return_value=mock_completed):
    results = batch_dispatch(payload, evaluator_script="/fake/evaluate.py", per_variant_timeout=60)

  assert len(results) == 1
  parsed = json.loads(results[0].split("EVAL_RESULT:", 1)[1])
  assert parsed["status"] == "COMPILE_ERROR"
  assert parsed["variant_id"] == "v1-crash"


def test_setup_dump_env_uses_variant_id():
  """_get_dump_dir should return variant-specific path when VARIANT_ID is set."""
  evaluate = _load_evaluate_module()

  # Without VARIANT_ID
  with patch.dict(os.environ, {}, clear=True):
    os.environ.pop("VARIANT_ID", None)
    assert evaluate._get_dump_dir() == "/tmp/ir_dumps"

  # With VARIANT_ID
  with patch.dict(os.environ, {"VARIANT_ID": "v1-tiling"}):
    assert evaluate._get_dump_dir() == "/tmp/ir_dumps/v1-tiling"


def test_batch_dispatch_handles_timeout():
  """batch_dispatch should handle subprocess timeout gracefully."""
  from kernel_evolve.docker_evaluate_helpers import batch_dispatch

  payload = {
    "batch": True,
    "reference_code": "def ref(): pass",
    "shapes": [{"M": 64}],
    "variants": [
      {"variant_id": "v1-slow", "kernel_code": "import time; time.sleep(999)"},
    ],
  }

  with patch(
    "kernel_evolve.docker_evaluate_helpers.subprocess.run",
    side_effect=subprocess.TimeoutExpired(cmd="python", timeout=60),
  ):
    results = batch_dispatch(payload, evaluator_script="/fake/evaluate.py", per_variant_timeout=60)

  assert len(results) == 1
  parsed = json.loads(results[0].split("EVAL_RESULT:", 1)[1])
  assert parsed["status"] == "COMPILE_ERROR"
  assert "timeout" in parsed["error"].lower()


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

  # Create dummy .xplane.pb so os.walk finds it
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
