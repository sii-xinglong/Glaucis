"""Batch dispatch helpers for evaluate.py. Importable for testing."""

import base64
import json
import os
import shutil
import subprocess
import sys
from typing import Any


def decode_request(b64_payload: str) -> dict[str, Any]:
  return json.loads(base64.b64decode(b64_payload).decode())


def _cleanup_variant_artifacts(variant_id: str) -> None:
  """Remove XLA dumps and trace data between variants to prevent storage exhaustion."""
  for path in [
    f"/tmp/ir_dumps/{variant_id}",
    "/tmp/ir_dumps",
    "/tmp/xplane_trace",
  ]:
    if os.path.exists(path):
      shutil.rmtree(path, ignore_errors=True)


def batch_dispatch(
  payload: dict[str, Any],
  evaluator_script: str,
  per_variant_timeout: int = 300,
) -> list[str]:
  """Dispatch each variant as a subprocess, return list of EVAL_RESULT: lines."""
  reference_code = payload["reference_code"]
  shapes = payload["shapes"]
  rtol = payload.get("rtol", 1e-2)
  atol = payload.get("atol", 1e-2)
  results: list[str] = []

  for variant in payload["variants"]:
    variant_id = variant["variant_id"]
    single_payload = {
      "variant_id": variant_id,
      "kernel_code": variant["kernel_code"],
      "reference_code": reference_code,
      "shapes": shapes,
      "rtol": rtol,
      "atol": atol,
    }
    b64 = base64.b64encode(json.dumps(single_payload).encode()).decode()

    env = os.environ.copy()
    env["VARIANT_ID"] = variant_id

    try:
      proc = subprocess.run(
        [sys.executable, evaluator_script, "--eval-payload", b64],
        capture_output=True,
        text=True,
        timeout=per_variant_timeout,
        env=env,
      )
      found = False
      if proc.returncode != 0:
        print(
          f"[batch] variant {variant_id} subprocess exited with code {proc.returncode}",
          file=sys.stderr,
        )
      for line in proc.stdout.split("\n"):
        if "EVAL_RESULT:" in line:
          json_str = line.split("EVAL_RESULT:", 1)[1].strip()
          result_data = json.loads(json_str)
          result_data.setdefault("variant_id", variant_id)
          results.append(f"EVAL_RESULT:{json.dumps(result_data)}")
          found = True
          break
      if not found:
        error = proc.stderr[-500:] if proc.stderr else "No EVAL_RESULT in output"
        fallback = {"variant_id": variant_id, "status": "COMPILE_ERROR", "error": error}
        results.append(f"EVAL_RESULT:{json.dumps(fallback)}")

    except subprocess.TimeoutExpired:
      fallback = {
        "variant_id": variant_id,
        "status": "COMPILE_ERROR",
        "error": f"Subprocess timeout after {per_variant_timeout}s",
      }
      results.append(f"EVAL_RESULT:{json.dumps(fallback)}")

    finally:
      _cleanup_variant_artifacts(variant_id)

  return results
