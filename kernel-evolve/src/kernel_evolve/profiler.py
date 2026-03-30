"""XPlane trace-based Pallas kernel profiler.

Captures JAX profiler traces and analyzes them using xprof to extract
compute_ratio and memory_transfer_ratio for MAP-Elites fitness signals.

Reference: accelerator-agents/MaxKernel/.../analyze_profile.py
"""

from __future__ import annotations

import json
import os
import re
import traceback
from functools import reduce
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

  # Find the pid for TPU device — try /device:TPU:0 first, then any TPU
  pid = None
  for event in events:
    if "args" in event and event["args"].get("name") == "/device:TPU:0":
      pid = event.get("pid")
      break
  if pid is None:
    for event in events:
      if "args" in event:
        name = event["args"].get("name", "")
        if name.startswith("/device:TPU:"):
          pid = event.get("pid")
          break

  if pid is None:
    return None

  # Collect TPU events and computation events (multiple name patterns)
  events_for_tpu = []
  computation_events = []
  for event in events:
    if event.get("pid") != pid:
      continue
    events_for_tpu.append(event)
    name = event.get("name") or ""
    if ("jit_computation" in name or "jit(" in name
        or "pallas" in name.lower()):
      if "dur" in event:
        computation_events.append(event)

  if len(computation_events) < 2:
    # Fallback: use any duration event on TPU
    dur_events = [e for e in events_for_tpu if "dur" in e and e["dur"] > 0]
    if len(dur_events) >= 2:
      computation_events = dur_events
    else:
      return None

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


# ---------------------------------------------------------------------------
# IR text parsing functions (no TPU/JAX dependency)
# ---------------------------------------------------------------------------

_DTYPE_BYTES: dict[str, int] = {
  "pred": 1,
  "s8": 1,
  "u8": 1,
  "s16": 2,
  "u16": 2,
  "f16": 2,
  "bf16": 2,
  "s32": 4,
  "u32": 4,
  "f32": 4,
  "s64": 8,
  "u64": 8,
  "f64": 8,
}


def _shape_bytes(shape_str: str) -> int | None:
  """Parse a shape like 'bf16[8,2048,128]' and return total bytes."""
  m = re.match(r"(\w+)\[([^\]]+)\]", shape_str.strip())
  if m is None:
    return None
  dtype = m.group(1)
  if dtype not in _DTYPE_BYTES:
    return None
  dims = [int(d.strip()) for d in m.group(2).split(",")]
  return _DTYPE_BYTES[dtype] * reduce(lambda a, b: a * b, dims)


def count_vliw_bundles(llo_text: str) -> int | None:
  """Count VLIW bundles in LLO text IR.

  Bundles are separated by ';;' in LLO output. Returns None if no bundles found.
  """
  if not llo_text:
    return None
  count = llo_text.count(";;")
  return count if count > 0 else None


def parse_mxu_distribution(llo_text: str) -> dict | None:
  """Count .mxu0 vs .mxu1 operations in LLO text.

  Returns {"mxu0": int, "mxu1": int, "dual_ratio": float} where dual_ratio = min/max.
  Returns None if no MXU ops found.
  """
  mxu0 = len(re.findall(r"\.mxu0\b", llo_text))
  mxu1 = len(re.findall(r"\.mxu1\b", llo_text))
  if mxu0 == 0 and mxu1 == 0:
    return None
  hi = max(mxu0, mxu1)
  lo = min(mxu0, mxu1)
  return {
    "mxu0": mxu0,
    "mxu1": mxu1,
    "dual_ratio": lo / hi if hi > 0 else 0.0,
  }


def estimate_hbm_bandwidth(hlo_text: str) -> int | None:
  """Parse first tpu_custom_call in HLO text, sum input + output bytes from shapes.

  Returns total bytes transferred or None if no tpu_custom_call found.
  """
  # Find the first custom-call line targeting tpu_custom_call
  cc_match = re.search(
    r"%\S+\s*=\s*(\w+\[[^\]]+\])\s+custom-call\(([^)]*)\).*?custom_call_target=\"tpu_custom_call\"",
    hlo_text,
  )
  if cc_match is None:
    return None

  output_shape_str = cc_match.group(1)
  args_str = cc_match.group(2)

  total = 0

  # Sum output bytes
  out_bytes = _shape_bytes(output_shape_str)
  if out_bytes is not None:
    total += out_bytes

  # Sum input bytes — resolve each %arg to its parameter shape
  arg_names = [a.strip() for a in args_str.split(",") if a.strip()]
  for arg_name in arg_names:
    # Find the parameter definition for this arg
    param_pattern = re.escape(arg_name) + r"\s*=\s*(\w+\[[^\]]+\])"
    param_match = re.search(param_pattern, hlo_text)
    if param_match:
      b = _shape_bytes(param_match.group(1))
      if b is not None:
        total += b

  return total if total > 0 else None


def count_flops_from_hlo(hlo_text: str) -> float | None:
  """Parse dot operations in HLO and compute FLOPs.

  FLOPs = 2 * product(output_dims) * product(contracting_dims).
  Returns None if no dot operations found.
  """
  # Pattern: %name = dtype[dims] dot(%a, %b), lhs_contracting_dims={d1}, rhs_contracting_dims={d2}
  dot_pattern = re.compile(
    r"%\S+\s*=\s*\w+\[([^\]]+)\]\s+dot\([^)]+\)"
    r".*?lhs_contracting_dims=\{([^}]*)\}"
    r".*?rhs_contracting_dims=\{([^}]*)\}"
  )

  total_flops = 0.0
  found = False

  for m in dot_pattern.finditer(hlo_text):
    found = True
    output_dims = [int(d.strip()) for d in m.group(1).split(",")]
    lhs_contracting = [int(d.strip()) for d in m.group(2).split(",") if d.strip()]

    # We need contracting dim sizes. Look up from the lhs operand.
    # Find the dot line to extract the lhs operand name
    dot_line_match = re.search(
      r"(%\S+)\s*=\s*\w+\[" + re.escape(m.group(1)) + r"\]\s+dot\((%\S+),",
      hlo_text,
    )
    if dot_line_match is None:
      continue

    lhs_name = dot_line_match.group(2)
    # Find the shape of lhs operand
    lhs_shape_match = re.search(re.escape(lhs_name) + r"\s*=\s*\w+\[([^\]]+)\]", hlo_text)
    if lhs_shape_match is None:
      continue

    lhs_dims = [int(d.strip()) for d in lhs_shape_match.group(1).split(",")]
    contracting_sizes = [lhs_dims[i] for i in lhs_contracting if i < len(lhs_dims)]

    output_product = reduce(lambda a, b: a * b, output_dims)
    contracting_product = reduce(lambda a, b: a * b, contracting_sizes) if contracting_sizes else 1
    total_flops += 2.0 * output_product * contracting_product

  return total_flops if found else None


def compute_derived_metrics(
  flops: float | None,
  hbm_bytes: int | None,
  latency_ms: float,
  peak_flops_per_sec: float = 275e12,
) -> dict:
  """Compute arithmetic intensity and compute efficiency.

  Returns {"arithmetic_intensity": float|None, "compute_efficiency_pct": float|None}.
  """
  arithmetic_intensity = None
  compute_efficiency_pct = None

  if flops is not None and hbm_bytes is not None and hbm_bytes > 0:
    arithmetic_intensity = flops / hbm_bytes

  if flops is not None and latency_ms > 0 and peak_flops_per_sec > 0:
    actual_flops_per_sec = flops / (latency_ms / 1000.0)
    compute_efficiency_pct = (actual_flops_per_sec / peak_flops_per_sec) * 100.0

  return {
    "arithmetic_intensity": arithmetic_intensity,
    "compute_efficiency_pct": compute_efficiency_pct,
  }
