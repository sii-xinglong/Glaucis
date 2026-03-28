"""XPlane trace-based Pallas kernel profiler.

Captures JAX profiler traces and analyzes them using xprof to extract
compute_ratio and memory_transfer_ratio for MAP-Elites fitness signals.

Reference: accelerator-agents/MaxKernel/.../analyze_profile.py
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any

try:
  from xprof.convert import raw_to_tool_data
except ImportError:
  raw_to_tool_data = None  # type: ignore[assignment]


def analyze_trace(xplane_path: str) -> dict[str, float] | None:
  """Parse an .xplane.pb file and extract compute vs memory transfer ratio.

  Returns {"compute_ratio": float, "memory_transfer_ratio": float} or None on failure.
  """
  if raw_to_tool_data is None:
    raise ImportError("xprof package is required for profiling. Install with: pip install xprof")

  tool_data_result, _ = raw_to_tool_data.xspace_to_tool_data([xplane_path], "trace_viewer", {})
  trace_data = json.loads(tool_data_result)
  events = trace_data.get("traceEvents", [])

  # Find the pid for /device:TPU:0
  pid = None
  for event in events:
    if "args" in event and event["args"].get("name") == "/device:TPU:0":
      pid = event.get("pid")
      break

  if pid is None:
    return None

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
    return None

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
    return None

  ratio = sync_wait_total / total_time
  return {
    "compute_ratio": 1.0 - ratio,
    "memory_transfer_ratio": ratio,
  }


def capture_trace(
  kernel_fn,
  shapes: dict[str, Any],
  trace_dir: str,
  warmup: int = 3,
  runs: int = 3,
) -> str | None:
  """Run kernel_fn under jax.profiler and return the path to the .xplane.pb file."""
  import jax

  Path(trace_dir).mkdir(parents=True, exist_ok=True)

  # Warmup runs (outside profiler)
  for _ in range(warmup):
    out = kernel_fn(**shapes)
    if hasattr(out, "block_until_ready"):
      out.block_until_ready()

  # Profiled runs
  options = jax.profiler.ProfileOptions()
  options.python_tracer_level = 0
  options.host_tracer_level = 2
  options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}

  jax.profiler.start_trace(trace_dir, profiler_options=options)
  for _ in range(runs):
    out = kernel_fn(**shapes)
    if hasattr(out, "block_until_ready"):
      out.block_until_ready()
  jax.profiler.stop_trace()

  # Search for .xplane.pb file
  for root, _dirs, files in os.walk(trace_dir):
    for f in files:
      if f.endswith(".xplane.pb"):
        return os.path.join(root, f)

  return None


def stage_profile(
  exec_globals: dict[str, Any],
  shapes: list[dict[str, Any]],
  trace_dir: str = "/tmp/xplane_trace",
) -> dict[str, Any]:
  """Stage 4: Profile kernel using JAX profiler and xprof trace analysis.

  Non-fatal: returns ok=False on failure without stopping the evaluation pipeline.
  """
  try:
    kernel_fn = exec_globals.get("optimized_compute") or exec_globals.get("kernel_fn")
    if kernel_fn is None:
      return {"ok": False, "error": "No kernel_fn found for profiling"}

    xplane_path = capture_trace(kernel_fn, shapes[0], trace_dir)
    if xplane_path is None:
      return {"ok": False, "error": "No .xplane.pb file generated"}

    result = analyze_trace(xplane_path)
    if result is None:
      return {"ok": False, "error": "Could not parse trace (no TPU:0 or jit_computation events)"}

    return {
      "ok": True,
      "compute_ratio": result["compute_ratio"],
      "memory_transfer_ratio": result["memory_transfer_ratio"],
    }
  except Exception:
    return {"ok": False, "error": f"Profile error: {traceback.format_exc()}"}
