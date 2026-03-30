"""Integration test for batch evolution data flow: config -> request -> dispatch -> result."""

import base64
import json

from kernel_evolve.config import EvolveConfig
from kernel_evolve.evaluator import (
  BatchEvalRequest,
  BatchEvalResult,
  EvalResult,
)


def test_full_batch_data_flow():
  """Config -> BatchEvalRequest -> payload -> parse results -> BatchEvalResult -> top-K."""
  cfg = EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64, "N": 64, "K": 64}],
    tpu={"cluster": "c", "zone": "z"},
    batch={"variants_per_round": 3, "top_k": 2},
  )

  # Build batch request using config
  batch_req = BatchEvalRequest(
    reference_code="def reference_fn(**kw): return 0",
    shapes=cfg.shapes,
    variants=[
      {"variant_id": "iter-1-tiling", "kernel_code": "def kernel_fn(**kw): return 0"},
      {"variant_id": "iter-1-pipeline", "kernel_code": "def kernel_fn(**kw): return 0"},
      {"variant_id": "iter-1-memory", "kernel_code": "def kernel_fn(**kw): return 0"},
    ],
    rtol=cfg.correctness.rtol,
    atol=cfg.correctness.atol,
  )

  # Verify payload format
  payload_dict = batch_req.to_dict()
  assert payload_dict["batch"] is True
  assert len(payload_dict["variants"]) == 3

  # Verify single request decomposition
  singles = batch_req.to_single_requests()
  assert len(singles) == 3
  assert all(s.reference_code == batch_req.reference_code for s in singles)

  # Verify base64 roundtrip
  b64 = batch_req.encode_b64()
  decoded = json.loads(base64.b64decode(b64).decode())
  assert decoded["batch"] is True
  assert len(decoded["variants"]) == 3

  # Simulate batch results (as would come from Pod logs)
  batch_result = BatchEvalResult(
    results={
      "iter-1-tiling": EvalResult.success(latency_ms=0.8, speedup=1.5),
      "iter-1-pipeline": EvalResult.compile_error("bad code"),
      "iter-1-memory": EvalResult.success(latency_ms=1.0, speedup=1.2),
    }
  )

  # Top-K selection
  ranked = batch_result.ranked()
  assert len(ranked) == 2  # only successes
  assert ranked[0][0] == "iter-1-tiling"  # highest speedup first
  assert ranked[1][0] == "iter-1-memory"

  # Apply top_k from config
  top_k = ranked[: cfg.batch.top_k]
  assert len(top_k) == 2
  assert top_k[0][1].speedup == 1.5

  # Best
  assert batch_result.best().speedup == 1.5
