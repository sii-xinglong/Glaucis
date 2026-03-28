"""Tests for the XPlane trace profiler module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from kernel_evolve.profiler import analyze_trace, capture_trace, stage_profile

# Minimal Chrome trace JSON matching the structure produced by xprof trace_viewer
MOCK_TRACE_JSON = json.dumps({
  "traceEvents": [
    {"pid": 1, "tid": 0, "ph": "M", "name": "process_name", "args": {"name": "/device:TPU:0"}},
    {"pid": 1, "tid": 1, "ph": "X", "name": "jit_computation.1", "ts": 100, "dur": 50},
    {"pid": 1, "tid": 1, "ph": "X", "name": "jit_computation.2", "ts": 200, "dur": 50},
    {"pid": 1, "tid": 1, "ph": "X", "name": "jit_computation.3", "ts": 300, "dur": 50},
    {"pid": 1, "tid": 1, "ph": "X", "name": "SyncWait:hbm_transfer", "ts": 260, "dur": 20},
    {"pid": 1, "tid": 1, "ph": "X", "name": "SyncWait:old", "ts": 100, "dur": 10},
    {"pid": 2, "tid": 0, "ph": "M", "name": "process_name", "args": {"name": "/host:CPU"}},
    {"pid": 2, "tid": 1, "ph": "X", "name": "SyncWait:cpu", "ts": 260, "dur": 30},
  ]
})


@patch("kernel_evolve.profiler.raw_to_tool_data")
def test_analyze_trace_computes_ratios(mock_xprof):
  mock_xprof.xspace_to_tool_data.return_value = (MOCK_TRACE_JSON, None)
  result = analyze_trace("/fake/trace.xplane.pb")
  # Window: ts=250 (end of jit_computation.2) to ts=350 (end of jit_computation.3)
  # Total = 100, SyncWait in window = 20 (hbm_transfer at 260-280), ratio = 0.2
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


def test_capture_trace_finds_xplane(tmp_path):
  profile_dir = tmp_path / "plugins" / "profile" / "2026_01_01"
  profile_dir.mkdir(parents=True)
  (profile_dir / "w-0.xplane.pb").write_bytes(b"fake")

  mock_kernel = MagicMock(return_value=MagicMock(block_until_ready=MagicMock()))

  with patch.dict("sys.modules", {"jax": MagicMock(), "jax.profiler": MagicMock()}):
    result = capture_trace(mock_kernel, {"M": 64}, str(tmp_path), warmup=1, runs=1)

  assert result is not None
  assert result.endswith(".xplane.pb")


def test_capture_trace_returns_none_when_no_xplane(tmp_path):
  mock_kernel = MagicMock(return_value=MagicMock(block_until_ready=MagicMock()))

  with patch.dict("sys.modules", {"jax": MagicMock(), "jax.profiler": MagicMock()}):
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
