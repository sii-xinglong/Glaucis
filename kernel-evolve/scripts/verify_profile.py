#!/usr/bin/env python3
"""End-to-end verification of the kernel profiling pipeline.

Run on a TPU to validate: matmul kernel -> jax.profiler trace -> .xplane.pb -> analyze_trace -> metrics.

Usage:
  python kernel-evolve/scripts/verify_profile.py [--trace-dir /tmp/xplane_trace]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
  parser = argparse.ArgumentParser(description="Verify kernel profiling pipeline")
  parser.add_argument("--trace-dir", default="/tmp/xplane_verify", help="Directory for trace output")
  args = parser.parse_args()

  print("=== Kernel Profile E2E Verification ===\n")

  # Step 1: Check JAX and TPU
  print("[1/5] Checking JAX and TPU...")
  import jax
  devices = jax.devices()
  print(f"  JAX devices: {devices}")
  has_tpu = any(d.platform == "tpu" for d in devices)
  print(f"  TPU available: {has_tpu}")
  if not has_tpu:
    print("  WARNING: No TPU detected. Profiling will proceed but trace may not contain TPU events.")

  # Step 2: Run matmul kernel
  print("\n[2/5] Running matmul kernel...")
  import jax.numpy as jnp
  from jax.experimental import pallas as pl

  def matmul_kernel(x_ref, y_ref, o_ref):
    acc = jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)
    o_ref[...] = acc.astype(o_ref.dtype)

  M, N, K = 1024, 1024, 1024
  BLOCK_M, BLOCK_N = 128, 128

  def optimized_compute(M=M, N=N, K=K):
    x = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=jnp.bfloat16)
    y = jax.random.normal(jax.random.PRNGKey(1), (K, N), dtype=jnp.bfloat16)
    return pl.pallas_call(
      matmul_kernel,
      out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
      grid=(M // BLOCK_M, N // BLOCK_N),
      in_specs=[
        pl.BlockSpec((BLOCK_M, K), lambda i, j: (i, 0)),
        pl.BlockSpec((K, BLOCK_N), lambda i, j: (0, j)),
      ],
      out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),
    )(x, y)

  # Warmup
  out = jax.block_until_ready(optimized_compute())
  print(f"  Output shape: {out.shape}, dtype: {out.dtype}")

  # Step 3: Capture trace
  print(f"\n[3/5] Capturing profiler trace to {args.trace_dir}...")
  os.makedirs(args.trace_dir, exist_ok=True)
  options = jax.profiler.ProfileOptions()
  options.python_tracer_level = 0
  options.host_tracer_level = 2
  options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}

  jax.profiler.start_trace(args.trace_dir, profiler_options=options)
  for i in range(3):
    jax.block_until_ready(optimized_compute())
  jax.profiler.stop_trace()
  print("  Trace captured.")

  # Step 4: Find .xplane.pb
  print("\n[4/5] Searching for .xplane.pb file...")
  xplane_path = None
  for root, _dirs, files in os.walk(args.trace_dir):
    for f in files:
      if f.endswith(".xplane.pb"):
        xplane_path = os.path.join(root, f)
        break
    if xplane_path:
      break

  if xplane_path is None:
    print("  ERROR: No .xplane.pb file found!")
    for root, _dirs, files in os.walk(args.trace_dir):
      for f in files:
        print(f"  Found: {os.path.join(root, f)}")
    sys.exit(1)

  file_size = os.path.getsize(xplane_path)
  print(f"  Found: {xplane_path} ({file_size} bytes)")

  # Step 5: Analyze trace
  print("\n[5/5] Analyzing trace with xprof...")
  from xprof.convert import raw_to_tool_data
  tool_data_result, _ = raw_to_tool_data.xspace_to_tool_data([xplane_path], "trace_viewer", {})
  trace_data = json.loads(tool_data_result)
  events = trace_data.get("traceEvents", [])
  print(f"  Total trace events: {len(events)}")

  # Find TPU:0
  pid = None
  for event in events:
    if "args" in event and event["args"].get("name") == "/device:TPU:0":
      pid = event.get("pid")
      break

  if pid is None:
    print("  ERROR: No /device:TPU:0 process found in trace!")
    process_names = set()
    for e in events:
      if "args" in e and "name" in e.get("args", {}):
        process_names.add(e["args"]["name"])
    print(f"  Available processes: {process_names}")
    sys.exit(1)

  jit_events = [e for e in events if e.get("pid") == pid and "jit_computation" in (e.get("name") or "")]
  print(f"  jit_computation events: {len(jit_events)}")

  if len(jit_events) < 2:
    print("  ERROR: Not enough jit_computation events for analysis!")
    sys.exit(1)

  start_last = jit_events[-2]["ts"] + jit_events[-2]["dur"]
  end_last = jit_events[-1]["ts"] + jit_events[-1]["dur"]
  total_time = end_last - start_last

  sync_wait = 0
  tpu_events = [e for e in events if e.get("pid") == pid]
  for e in tpu_events:
    if "dur" not in e:
      continue
    if e["ts"] >= start_last and (e["ts"] + e["dur"]) <= end_last:
      if "SyncWait" in (e.get("name") or ""):
        sync_wait += e["dur"]

  ratio = sync_wait / total_time if total_time > 0 else 0
  compute_ratio = 1.0 - ratio
  memory_transfer_ratio = ratio

  print(f"\n{'='*50}")
  print(f"  compute_ratio:          {compute_ratio:.4f}")
  print(f"  memory_transfer_ratio:  {memory_transfer_ratio:.4f}")
  print(f"  total_time (us):        {total_time}")
  print(f"  sync_wait (us):         {sync_wait}")
  print(f"{'='*50}")

  from kernel_evolve.population import ratio_to_compute_profile
  bucket = ratio_to_compute_profile(compute_ratio)
  print(f"  MAP-Elites bucket:      {bucket}")
  print("\nVERIFICATION PASSED")


if __name__ == "__main__":
  main()
