#!/usr/bin/env python3
"""Submit a matmul kernel evaluation via KubeEvaluator and verify profiling data.

Usage:
  python3 kernel-evolve/scripts/run_matmul_eval.py

Requires:
  - kubectl configured with access to the GKE cluster
  - feat/kernel-profile branch pushed (contains profiling stage in evaluate.py)
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kernel_evolve.evaluator import EvalRequest
from kernel_evolve.kube_evaluator import KubeConfig, KubeEvaluator
from kernel_evolve.population import ratio_to_compute_profile


def main():
  # Read kernel and reference code
  examples_dir = Path(__file__).resolve().parent.parent / "examples" / "kernels"
  kernel_code = (examples_dir / "matmul.py").read_text()
  reference_code = (examples_dir / "matmul_ref.py").read_text()

  # Use only the 1024x1024 shape for a faster eval
  shapes = [{"M": 1024, "N": 1024, "K": 1024}]

  request = EvalRequest(
    variant_id="matmul-profile-verify",
    kernel_code=kernel_code,
    reference_code=reference_code,
    shapes=shapes,
    rtol=1e-2,
    atol=1.0,
  )

  config = KubeConfig(
    namespace="default",
    job_template=".github/ci/kernel-eval-job.yaml",
    repo="sii-xinglong/Glaucis",
    branch="feat/kernel-profile",
    poll_interval=15,
    timeout=600,
  )

  evaluator = KubeEvaluator(config)

  print("=" * 60)
  print("Matmul Kernel Evaluation with Profiling Verification")
  print("=" * 60)
  print(f"Variant ID:  {request.variant_id}")
  print(f"Branch:      {config.branch}")
  print(f"Shapes:      {shapes}")
  print(f"Timeout:     {config.timeout}s")
  print()

  print("Submitting K8s job via KubeEvaluator...")
  result = asyncio.run(evaluator.evaluate(request))

  print()
  print("=" * 60)
  print("EVALUATION RESULT")
  print("=" * 60)
  print(json.dumps(result.to_dict(), indent=2))
  print()

  # Verify profiling data
  print("-" * 60)
  print("PROFILING VERIFICATION")
  print("-" * 60)

  if result.status.name == "SUCCESS":
    print(f"Status:                 SUCCESS")
    print(f"Latency:                {result.latency_ms:.3f} ms")
    print(f"Speedup:                {result.speedup:.3f}x")

    if result.compute_ratio is not None:
      bucket = ratio_to_compute_profile(result.compute_ratio)
      print(f"Compute Ratio:          {result.compute_ratio:.4f}")
      print(f"Memory Transfer Ratio:  {result.memory_transfer_ratio:.4f}")
      print(f"MAP-Elites Bucket:      {bucket}")
      print()

      # Display trace diagnostics if available
      diag = result.metadata.get("profile_diagnostics", {})
      if diag:
        print("-" * 60)
        print("TRACE DIAGNOSTICS")
        print("-" * 60)
        print(f"Process names:         {diag.get('process_names', {})}")
        print(f"Selected PID:          {diag.get('selected_pid')}")
        print(f"Total trace events:    {diag.get('total_events')}")
        print(f"TPU events:            {diag.get('tpu_events')}")
        print(f"Computation events:    {diag.get('computation_events')}")
        print(f"SyncWait events:       {diag.get('sync_wait_events')}")
        print(f"Window (us):           {diag.get('window_us')}")
        print(f"SyncWait total (us):   {diag.get('sync_wait_us')}")
        event_names = diag.get("event_names_sample", [])
        print(f"Event names:           {event_names}")
        print()

      print("PROFILING: PASS - compute_ratio and memory_transfer_ratio present")
    else:
      print("Compute Ratio:          None")
      print("Memory Transfer Ratio:  None")
      print()
      print("PROFILING: FAIL - profiling data not returned")
      sys.exit(1)
  else:
    print(f"Status:  {result.status.name}")
    print(f"Error:   {result.error}")
    print()
    print("EVALUATION: FAIL - kernel did not succeed")
    sys.exit(1)


if __name__ == "__main__":
  main()
