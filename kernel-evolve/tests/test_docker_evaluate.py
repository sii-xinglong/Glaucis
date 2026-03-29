"""Tests for the TPU pod evaluator helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_evaluate_module():
  module_path = (
    Path(__file__).resolve().parents[1] / "docker" / "evaluate.py"
  )
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
