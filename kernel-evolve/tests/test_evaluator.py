"""Tests for the three-stage evaluator pipeline."""

import base64
import json

from kernel_evolve.evaluator import (
  BatchEvalRequest,
  BatchEvalResult,
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


def test_batch_eval_request_creation():
  req = BatchEvalRequest(
    reference_code="def ref(): pass",
    shapes=[{"M": 1024}],
    variants=[
      {"variant_id": "v1-tiling", "kernel_code": "def k1(): pass"},
      {"variant_id": "v2-pipeline", "kernel_code": "def k2(): pass"},
    ],
  )
  assert len(req.variants) == 2
  assert req.variants[0]["variant_id"] == "v1-tiling"
  assert req.reference_code == "def ref(): pass"


def test_batch_eval_request_to_dict():
  req = BatchEvalRequest(
    reference_code="def ref(): pass",
    shapes=[{"M": 1024}],
    variants=[
      {"variant_id": "v1", "kernel_code": "def k1(): pass"},
    ],
    rtol=0.01,
    atol=1.0,
  )
  d = req.to_dict()
  assert d["batch"] is True
  assert d["reference_code"] == "def ref(): pass"
  assert len(d["variants"]) == 1
  assert d["rtol"] == 0.01


def test_batch_eval_request_encode_decode_b64():
  req = BatchEvalRequest(
    reference_code="def ref(): pass",
    shapes=[{"M": 64}],
    variants=[
      {"variant_id": "v1", "kernel_code": "def k1(): pass"},
      {"variant_id": "v2", "kernel_code": "def k2(): pass"},
    ],
  )
  encoded = req.encode_b64()
  decoded = json.loads(base64.b64decode(encoded).decode())
  assert decoded["batch"] is True
  assert len(decoded["variants"]) == 2


def test_batch_eval_request_to_single_requests():
  req = BatchEvalRequest(
    reference_code="def ref(): pass",
    shapes=[{"M": 64}],
    variants=[
      {"variant_id": "v1", "kernel_code": "def k1(): pass"},
      {"variant_id": "v2", "kernel_code": "def k2(): pass"},
    ],
    rtol=0.05,
    atol=0.5,
  )
  singles = req.to_single_requests()
  assert len(singles) == 2
  assert isinstance(singles[0], EvalRequest)
  assert singles[0].variant_id == "v1"
  assert singles[0].kernel_code == "def k1(): pass"
  assert singles[0].reference_code == "def ref(): pass"
  assert singles[0].rtol == 0.05
  assert singles[1].variant_id == "v2"


def test_batch_eval_result():
  r1 = EvalResult.success(latency_ms=1.0, speedup=1.5)
  r2 = EvalResult.compile_error("bad code")
  batch = BatchEvalResult(results={"v1": r1, "v2": r2})
  assert batch.results["v1"].speedup == 1.5
  assert batch.results["v2"].status == EvalStatus.COMPILE_ERROR
  assert batch.best() == r1
  assert batch.ranked() == [("v1", r1)]


def test_batch_eval_result_best_returns_none_when_all_failed():
  r1 = EvalResult.compile_error("err1")
  r2 = EvalResult.compile_error("err2")
  batch = BatchEvalResult(results={"v1": r1, "v2": r2})
  assert batch.best() is None
  assert batch.ranked() == []
