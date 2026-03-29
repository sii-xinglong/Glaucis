"""Tests for the three-stage evaluator pipeline."""

from kernel_evolve.evaluator import (
  EvalRequest,
  EvalResult,
  EvalStatus,
)


def test_eval_status_ordering():
  assert EvalStatus.COMPILE_ERROR.value < EvalStatus.INCORRECT.value < EvalStatus.SUCCESS.value


def test_eval_result_compile_error():
  r = EvalResult.compile_error("undefined variable 'x'")
  assert r.status == EvalStatus.COMPILE_ERROR
  assert r.fitness == 0.0
  assert "undefined" in r.error


def test_eval_result_incorrect():
  r = EvalResult.incorrect(max_diff=0.5)
  assert r.status == EvalStatus.INCORRECT
  assert r.fitness == 0.0
  assert r.max_diff == 0.5


def test_eval_result_success():
  r = EvalResult.success(latency_ms=1.5, speedup=2.3, flops=1e12)
  assert r.status == EvalStatus.SUCCESS
  assert r.fitness == 2.3
  assert r.latency_ms == 1.5


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


def test_eval_request_serialization():
  req = EvalRequest(
    variant_id="v001",
    kernel_code="def k(): pass",
    reference_code="def r(): pass",
    shapes=[{"M": 1024, "N": 1024}],
    rtol=1e-2,
    atol=1e-2,
  )
  data = req.to_dict()
  restored = EvalRequest.from_dict(data)
  assert restored.variant_id == "v001"
  assert restored.kernel_code == "def k(): pass"
  assert len(restored.shapes) == 1
