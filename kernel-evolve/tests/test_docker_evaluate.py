"""Tests for the TPU pod evaluator helpers."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch


def _load_evaluate_module():
  module_path = Path(__file__).resolve().parents[1] / "docker" / "evaluate.py"
  spec = importlib.util.spec_from_file_location("kernel_evolve_docker_evaluate", module_path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def test_stage_performance_accepts_reference_entrypoints():
  evaluate = _load_evaluate_module()

  globals_dict = {
    "simple_compute": lambda M=1: M,
    "reference_fn": lambda **kwargs: kwargs["M"],
  }

  result = evaluate.stage_performance(
    globals_dict,
    shapes=[{"M": 4}],
    warmup=1,
    iters=2,
  )

  assert result["ok"] is True
  assert result["latency_ms"] >= 0.0


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
