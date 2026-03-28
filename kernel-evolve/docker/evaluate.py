"""Four-stage kernel evaluator for TPU. Runs inside a K8s Pod."""

import argparse
import base64
import json
import os
import sys
import time
import traceback
from typing import Any

import numpy as np


def decode_request(b64_payload: str) -> dict[str, Any]:
  return json.loads(base64.b64decode(b64_payload).decode())


def _has_tpu() -> bool:
  try:
    import jax
    devices = jax.devices()
    print(f"JAX devices: {devices}", file=sys.stderr)
    has = any(d.platform == "tpu" for d in devices)
    if not has:
      print(f"JAX default backend: {jax.default_backend()}", file=sys.stderr)
      try:
        tpu_devices = jax.devices("tpu")
        print(f"TPU devices (explicit): {tpu_devices}", file=sys.stderr)
        has = len(tpu_devices) > 0
      except RuntimeError as e:
        print(f"TPU backend error: {e}", file=sys.stderr)
    return has
  except Exception as e:
    print(f"TPU detection error: {e}", file=sys.stderr)
    return False


def stage_compile(kernel_code: str) -> dict[str, Any]:
  try:
    exec_globals = {}
    exec(kernel_code, exec_globals)
    return {"ok": True, "globals": exec_globals}
  except Exception:
    return {"ok": False, "error": f"Compilation error: {traceback.format_exc()}"}


def stage_correctness(kernel_code, reference_code, shapes, rtol, atol, exec_globals):
  try:
    ref_globals = {}
    exec(reference_code, ref_globals)
    kernel_fn = exec_globals.get("optimized_compute") or exec_globals.get("kernel_fn")
    ref_fn = ref_globals.get("simple_compute") or ref_globals.get("reference_fn")
    if kernel_fn is None or ref_fn is None:
      return {"ok": False, "error": "Could not find kernel_fn/reference_fn", "max_diff": 0.0}
    for shape in shapes:
      kernel_out = kernel_fn(**shape)
      ref_out = ref_fn(**shape)
      max_diff = float(np.max(np.abs(np.array(kernel_out) - np.array(ref_out))))
      if max_diff > atol:
        return {"ok": False, "error": f"Correctness failed for shape {shape}: max_diff={max_diff}", "max_diff": max_diff}
    return {"ok": True, "max_diff": 0.0}
  except Exception:
    return {"ok": False, "error": f"Correctness error: {traceback.format_exc()}", "max_diff": 0.0}


def stage_performance(exec_globals, shapes, warmup=10, iters=50):
  try:
    import jax
    kernel_fn = exec_globals.get("optimized_compute") or exec_globals.get("kernel_fn")
    if kernel_fn is None:
      return {"ok": False, "error": "No kernel_fn found"}
    shape = shapes[0]
    for _ in range(warmup):
      out = kernel_fn(**shape)
      if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    times = []
    for _ in range(iters):
      start = time.perf_counter()
      out = kernel_fn(**shape)
      if hasattr(out, "block_until_ready"):
        out.block_until_ready()
      times.append(time.perf_counter() - start)
    latency_ms = float(np.median(times)) * 1000
    return {"ok": True, "latency_ms": latency_ms}
  except Exception:
    return {"ok": False, "error": f"Performance error: {traceback.format_exc()}"}


def stage_profile(exec_globals, shapes, trace_dir="/tmp/xplane_trace"):
  """Stage 4: Profile kernel using JAX profiler and xprof trace analysis.

  Self-contained — does NOT import from kernel_evolve.profiler.
  Non-fatal: returns ok=False on failure without stopping the evaluation pipeline.
  """
  try:
    import jax
    from pathlib import Path

    try:
      from xprof.convert import raw_to_tool_data
    except ImportError:
      return {"ok": False, "error": "xprof package not available"}

    kernel_fn = exec_globals.get("optimized_compute") or exec_globals.get("kernel_fn")
    if kernel_fn is None:
      return {"ok": False, "error": "No kernel_fn found for profiling"}

    shape = shapes[0]
    Path(trace_dir).mkdir(parents=True, exist_ok=True)

    # Warmup runs (outside profiler)
    for _ in range(3):
      out = kernel_fn(**shape)
      if hasattr(out, "block_until_ready"):
        out.block_until_ready()

    # Profiled runs
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 2
    options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}

    jax.profiler.start_trace(trace_dir, profiler_options=options)
    for _ in range(3):
      out = kernel_fn(**shape)
      if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    jax.profiler.stop_trace()

    # Find the .xplane.pb file
    xplane_path = None
    for root, _dirs, files in os.walk(trace_dir):
      for f in files:
        if f.endswith(".xplane.pb"):
          xplane_path = os.path.join(root, f)
          break
      if xplane_path:
        break

    if xplane_path is None:
      return {"ok": False, "error": "No .xplane.pb file generated"}

    # Parse trace using xprof
    tool_data_result, _ = raw_to_tool_data.xspace_to_tool_data(
      [xplane_path], "trace_viewer", {}
    )
    trace_data = json.loads(tool_data_result)
    events = trace_data.get("traceEvents", [])

    # Find the pid for /device:TPU:0
    pid = None
    for event in events:
      if "args" in event and event["args"].get("name") == "/device:TPU:0":
        pid = event.get("pid")
        break

    if pid is None:
      return {"ok": False, "error": "No /device:TPU:0 found in trace"}

    # Collect TPU:0 events and jit_computation events
    events_for_tpu_0 = []
    jit_computation_events = []
    for event in events:
      if event.get("pid") != pid:
        continue
      events_for_tpu_0.append(event)
      name = event.get("name") or ""
      if "jit_computation" in name:
        jit_computation_events.append(event)

    if len(jit_computation_events) < 2:
      return {"ok": False, "error": "Not enough jit_computation events in trace"}

    # Focus on the last iteration window
    start_last = jit_computation_events[-2]["ts"] + jit_computation_events[-2]["dur"]
    end_last = jit_computation_events[-1]["ts"] + jit_computation_events[-1]["dur"]

    # Sum SyncWait durations within the window
    sync_wait_total = 0
    for event in events_for_tpu_0:
      if "dur" not in event:
        continue
      evt_start = event["ts"]
      evt_end = evt_start + event["dur"]
      if evt_start >= start_last and evt_end <= end_last:
        name = event.get("name") or ""
        if "SyncWait" in name:
          sync_wait_total += event["dur"]

    total_time = end_last - start_last
    if total_time <= 0:
      return {"ok": False, "error": "Invalid trace timing (total_time <= 0)"}

    ratio = sync_wait_total / total_time
    return {
      "ok": True,
      "compute_ratio": 1.0 - ratio,
      "memory_transfer_ratio": ratio,
    }
  except Exception:
    return {"ok": False, "error": f"Profile error: {traceback.format_exc()}"}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--eval-payload", required=True)
  args = parser.parse_args()
  request = decode_request(args.eval_payload)

  if not _has_tpu():
    print("ERROR: No TPU detected. This evaluator requires a TPU device.", file=sys.stderr)
    sys.exit(1)

  compile_result = stage_compile(request["kernel_code"])
  if not compile_result["ok"]:
    print(f'EVAL_RESULT:{json.dumps({"status": "COMPILE_ERROR", "error": compile_result["error"]})}')
    sys.exit(0)

  correct_result = stage_correctness(
    request["kernel_code"], request["reference_code"], request["shapes"],
    request.get("rtol", 1e-2), request.get("atol", 1e-2), compile_result["globals"],
  )
  if not correct_result["ok"]:
    print(f'EVAL_RESULT:{json.dumps({"status": "INCORRECT", "error": correct_result["error"], "max_diff": correct_result["max_diff"]})}')
    sys.exit(0)

  perf_result = stage_performance(compile_result["globals"], request["shapes"])
  if not perf_result["ok"]:
    print(f'EVAL_RESULT:{json.dumps({"status": "COMPILE_ERROR", "error": perf_result["error"]})}')
    sys.exit(0)

  ref_compile = stage_compile(request["reference_code"])
  ref_perf = stage_performance(ref_compile.get("globals", {}), request["shapes"])
  ref_latency = ref_perf.get("latency_ms", perf_result["latency_ms"])
  speedup = ref_latency / perf_result["latency_ms"] if perf_result["latency_ms"] > 0 else 0.0

  # Stage 4: Profile (non-fatal)
  profile_result = stage_profile(compile_result["globals"], request["shapes"])
  compute_ratio = None
  memory_transfer_ratio = None
  if profile_result["ok"]:
    compute_ratio = profile_result["compute_ratio"]
    memory_transfer_ratio = profile_result["memory_transfer_ratio"]
    print(f"Profile: compute_ratio={compute_ratio}, memory_transfer_ratio={memory_transfer_ratio}", file=sys.stderr)
  else:
    print(f"Profile skipped: {profile_result.get('error', 'unknown')}", file=sys.stderr)

  result = {
    "status": "SUCCESS",
    "fitness": speedup,
    "latency_ms": perf_result["latency_ms"],
    "speedup": speedup,
    "flops": 0.0,
    "compute_ratio": compute_ratio,
    "memory_transfer_ratio": memory_transfer_ratio,
  }
  print(f"EVAL_RESULT:{json.dumps(result)}")


if __name__ == "__main__":
  main()
