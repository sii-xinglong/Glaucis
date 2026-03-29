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
        return {
          "ok": False,
          "error": f"Correctness failed for shape {shape}: max_diff={max_diff}",
          "max_diff": max_diff,
        }
    return {"ok": True, "max_diff": 0.0}
  except Exception:
    return {"ok": False, "error": f"Correctness error: {traceback.format_exc()}", "max_diff": 0.0}


def stage_performance(exec_globals, shapes, warmup=10, iters=50):
  try:
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
    from pathlib import Path

    import jax

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

    # Dump all device/process names for diagnostics (collect all names per pid)
    process_names: dict[int, list[str]] = {}
    for event in events:
      if "args" in event and "name" in event["args"]:
        pid_val = event.get("pid")
        if pid_val not in process_names:
          process_names[pid_val] = []
        name_val = event["args"]["name"]
        if name_val not in process_names[pid_val]:
          process_names[pid_val].append(name_val)
    print(f"Trace processes: {process_names}", file=sys.stderr)
    print(f"Total trace events: {len(events)}", file=sys.stderr)

    # Find the pid for TPU device — try /device:TPU:0 first, then any TPU device
    pid = None
    for event in events:
      if "args" in event and event["args"].get("name") == "/device:TPU:0":
        pid = event.get("pid")
        break
    if pid is None:
      # Fallback: find any /device:TPU:* process
      for event in events:
        if "args" in event:
          name = event["args"].get("name", "")
          if name.startswith("/device:TPU:"):
            pid = event.get("pid")
            print(f"Using fallback TPU device: {name} (pid={pid})", file=sys.stderr)
            break

    if pid is None:
      return {"ok": False, "error": f"No TPU device in trace. Processes: {process_names}"}

    # Collect TPU events and computation events (try multiple patterns)
    events_for_tpu = []
    computation_events = []
    all_event_names = set()
    for event in events:
      if event.get("pid") != pid:
        continue
      events_for_tpu.append(event)
      name = event.get("name") or ""
      if name and "dur" in event:
        all_event_names.add(name)
      # Match jit_computation, pallas_call, or any XLA computation
      if ("jit_computation" in name or "jit(" in name
          or "pallas" in name.lower()):
        if "dur" in event:
          computation_events.append(event)

    print(
      f"TPU event names (sample): {sorted(all_event_names)[:30]}",
      file=sys.stderr,
    )
    print(
      f"Computation events found: {len(computation_events)}",
      file=sys.stderr,
    )

    # Check for SyncWait/idle events across ALL pids
    sync_events_global = []
    for event in events:
      name = event.get("name") or ""
      if "SyncWait" in name or "idle" in name.lower():
        sync_events_global.append(
          (event.get("pid"), name, event.get("dur", 0))
        )
    print(
      f"SyncWait/idle events (all pids): {len(sync_events_global)}",
      file=sys.stderr,
    )
    if sync_events_global:
      print(
        f"  Sample: {sync_events_global[:5]}", file=sys.stderr,
      )

    if len(computation_events) < 2:
      # Last resort: use any duration event on TPU as computation marker
      dur_events = [e for e in events_for_tpu if "dur" in e and e["dur"] > 0]
      if len(dur_events) >= 2:
        print(
          f"Fallback: using {len(dur_events)} duration events as computation markers",
          file=sys.stderr,
        )
        computation_events = dur_events
      else:
        return {
          "ok": False,
          "error": (
            f"Not enough computation events in trace. "
            f"Found {len(computation_events)}. "
            f"Event names: {sorted(all_event_names)[:30]}"
          ),
        }

    # Focus on the last iteration window
    start_last = computation_events[-2]["ts"] + computation_events[-2]["dur"]
    end_last = computation_events[-1]["ts"] + computation_events[-1]["dur"]

    # Sum SyncWait/idle durations within the window
    sync_wait_total = 0
    for event in events_for_tpu:
      if "dur" not in event:
        continue
      evt_start = event["ts"]
      evt_end = evt_start + event["dur"]
      if evt_start >= start_last and evt_end <= end_last:
        name = event.get("name") or ""
        if "SyncWait" in name or "idle" in name.lower():
          sync_wait_total += event["dur"]

    total_time = end_last - start_last
    if total_time <= 0:
      return {"ok": False, "error": "Invalid trace timing (total_time <= 0)"}

    ratio = sync_wait_total / total_time
    diag = {
      "process_names": {str(k): v for k, v in process_names.items()},
      "selected_pid": pid,
      "total_events": len(events),
      "tpu_events": len(events_for_tpu),
      "computation_events": len(computation_events),
      "sync_wait_events": len(sync_events_global),
      "event_names_sample": sorted(all_event_names)[:30],
      "window_us": total_time,
      "sync_wait_us": sync_wait_total,
    }
    return {
      "ok": True,
      "compute_ratio": 1.0 - ratio,
      "memory_transfer_ratio": ratio,
      "diagnostics": diag,
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
    result_data = {
      "status": "INCORRECT",
      "error": correct_result["error"],
      "max_diff": correct_result["max_diff"],
    }
    print(f"EVAL_RESULT:{json.dumps(result_data)}")
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
  profile_diag = {}
  if profile_result["ok"]:
    compute_ratio = profile_result["compute_ratio"]
    memory_transfer_ratio = profile_result["memory_transfer_ratio"]
    profile_diag = profile_result.get("diagnostics", {})
    print(f"Profile: compute_ratio={compute_ratio}, memory_transfer_ratio={memory_transfer_ratio}", file=sys.stderr)
  else:
    profile_diag = {"error": profile_result.get("error", "unknown")}
    print(f"Profile skipped: {profile_result.get('error', 'unknown')}", file=sys.stderr)

  result = {
    "status": "SUCCESS",
    "fitness": speedup,
    "latency_ms": perf_result["latency_ms"],
    "speedup": speedup,
    "flops": 0.0,
    "compute_ratio": compute_ratio,
    "memory_transfer_ratio": memory_transfer_ratio,
    "metadata": {"profile_diagnostics": profile_diag},
  }
  print(f"EVAL_RESULT:{json.dumps(result)}")


if __name__ == "__main__":
  main()
