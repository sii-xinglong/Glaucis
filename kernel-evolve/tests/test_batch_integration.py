"""Integration test for batch evolution data flow: config -> request -> dispatch -> result."""

import base64
import json

from kernel_evolve.config import EvolveConfig
from kernel_evolve.evaluator import (
  BatchEvalRequest,
  BatchEvalResult,
  EvalResult,
)
from kernel_evolve.tuning import TuningConfig, TuningParam, expand_variants


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


def test_tuning_expansion_through_batch_pipeline():
  """End-to-end: expand variants with tuning configs, build batch request, verify metadata."""
  # Define tuning config
  tc = TuningConfig(
    params={
      "BLOCK_K": TuningParam(values=[64, 128]),
      "BLOCK_V": TuningParam(values=[32, 64]),
    },
    constraints=["BLOCK_K >= BLOCK_V"],
  )

  # Two LLM-generated code variants
  v1_code = "BLOCK_K = 256\nBLOCK_V = 256\ndef kernel_v1(): pass\n"
  v2_code = "BLOCK_K = 512\nBLOCK_V = 128\ndef kernel_v2(): pass\n"

  # Expand
  expanded = expand_variants([v1_code, v2_code], tc)
  # 4 configs (all pass constraint: 64>=32, 64>=64, 128>=32, 128>=64) * 2 variants = 8
  assert len(expanded) == 8

  # Build BatchEvalRequest with metadata
  variants_for_batch = []
  code_variants = [("L1_v1", v1_code), ("L2_v2", v2_code)]
  for code_variant_id, original_code in code_variants:
    variant_expanded = expand_variants([original_code], tc)
    for idx, (modified_code, cfg) in enumerate(variant_expanded):
      variants_for_batch.append({
        "variant_id": f"{code_variant_id}_t{idx}",
        "kernel_code": modified_code,
        "metadata": {
          "tuning_config": cfg,
          "code_variant_id": code_variant_id,
        },
      })

  batch = BatchEvalRequest(
    reference_code="ref",
    shapes=[{"M": 1024}],
    variants=variants_for_batch,
  )

  # Verify batch decomposes correctly
  singles = batch.to_single_requests()
  assert len(singles) == 8

  # Check metadata flows through
  for req in singles:
    assert "tuning_config" in req.metadata
    assert "code_variant_id" in req.metadata

  # Check specific variant: metadata and code substitution
  t0 = next(r for r in singles if r.variant_id == "L1_v1_t0")
  assert t0.metadata["code_variant_id"] == "L1_v1"
  assert t0.metadata["tuning_config"] == {"BLOCK_K": 64, "BLOCK_V": 32}

  # Verify kernel code was actually modified by apply_config
  assert "BLOCK_K = 64" in t0.kernel_code
  assert "BLOCK_V = 32" in t0.kernel_code
  assert "def kernel_v1" in t0.kernel_code  # original code structure preserved

  # Verify payload roundtrip
  encoded = batch.encode_b64()
  decoded = json.loads(base64.b64decode(encoded).decode())
  assert decoded["variants"][0]["metadata"]["tuning_config"] is not None
