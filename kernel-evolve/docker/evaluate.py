"""Four-stage kernel evaluator for TPU. Runs inside a K8s Pod."""

import argparse
import base64
import json
import os
import sys
import time
import traceback
from typing import Any

try:
  from google.cloud import storage
except ImportError:
  storage = None

import numpy as np


def upload_to_gcs(
  job_name: str,
  artifacts: dict[str, str],
  bucket_name: str = "glaucis-profiles",
) -> dict[str, Any]:
  """Upload profile artifacts to GCS. Non-fatal — never raises."""
  prefix = f"gs://{bucket_name}/{job_name}"
  uploaded: list[str] = []
  if not artifacts:
    return {"ok": False, "uploaded": uploaded, "gcs_prefix": prefix}
  if storage is None:
    print("google-cloud-storage not installed, skipping GCS upload", file=sys.stderr)
    return {"ok": False, "uploaded": uploaded, "gcs_prefix": prefix}
  try:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for name, local_path in artifacts.items():
      if not os.path.exists(local_path):
        continue
      try:
        blob = bucket.blob(f"{job_name}/{name}")
        blob.upload_from_filename(local_path)
        uploaded.append(name)
      except Exception as e:
        print(f"GCS upload failed for {name}: {e}", file=sys.stderr)
  except Exception as e:
    print(f"GCS client init failed: {e}", file=sys.stderr)
  return {"ok": len(uploaded) > 0, "uploaded": uploaded, "gcs_prefix": prefix}


def decode_request(b64_payload: str) -> dict[str, Any]:
  return json.loads(base64.b64decode(b64_payload).decode())


def _resolve_compute_fn(
  exec_globals: dict[str, Any],
  *,
  allow_reference: bool = False,
):
  names = ["optimized_compute", "kernel_fn"]
  if allow_reference:
    names.extend(["simple_compute", "reference_fn"])
  for name in names:
    fn = exec_globals.get(name)
    if fn is not None:
      return fn
  return None


def _has_tpu() -> bool:
  try:
    import jax

    devices = jax.devices()
    print(f"JAX devices: {devices}", file=sys.stderr)
    return any(d.platform == "tpu" for d in devices)
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
    kernel_fn = _resolve_compute_fn(exec_globals)
    ref_fn = _resolve_compute_fn(ref_globals, allow_reference=True)
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
    kernel_fn = _resolve_compute_fn(exec_globals, allow_reference=True)
    if kernel_fn is None:
      return {"ok": False, "error": "No compute function found"}
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


def parse_hw_utilization(events, tpu_pid):
  """Extract per-hardware-unit utilization from xprof counter events.

  The _counters_ thread (tid=0xFFFFFFFF) contains time-windowed utilization
  percentages for MXU, Scalar ALU, Vector ALU, Vector Load/Store, Vector EUP,
  plus absolute fill/spill counts. Returns a dict of time-weighted averages
  or None if no counter events are found.
  """
  COUNTER_TID = 4294967295  # 0xFFFFFFFF, _counters_ thread
  UTIL_UNITS = {
    "MXU": "mxu_util_pct",
    "Vector ALU": "vector_alu_util_pct",
    "Scalar ALU": "scalar_alu_util_pct",
    "Vector Load": "vector_load_util_pct",
    "Vector Store": "vector_store_util_pct",
    "Vector EUP": "vector_eup_util_pct",
  }
  COUNTER_UNITS = {
    "Vector Fills": ("vector_fills", "fills"),
    "Vector Spills": ("vector_spills", "spills"),
  }

  counter_events = [
    e for e in events
    if e.get("pid") == tpu_pid and e.get("tid") == COUNTER_TID and "dur" in e
  ]
  if not counter_events:
    return None

  result = {}
  for xprof_name, key in UTIL_UNITS.items():
    weighted_sum = 0.0
    total_dur = 0.0
    for e in counter_events:
      if e.get("name") == xprof_name:
        dur = e["dur"]
        util = float(e.get("args", {}).get("% util", 0))
        weighted_sum += util * dur
        total_dur += dur
    result[key] = weighted_sum / total_dur if total_dur > 0 else 0.0

  for xprof_name, (key, arg_key) in COUNTER_UNITS.items():
    total = 0
    for e in counter_events:
      if e.get("name") == xprof_name:
        total += int(e.get("args", {}).get(arg_key, 0))
    result[key] = total

  return result


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
    tool_data_result, _ = raw_to_tool_data.xspace_to_tool_data([xplane_path], "trace_viewer", {})
    trace_data = json.loads(tool_data_result)
    events = trace_data.get("traceEvents", [])

    # Save trace events for GCS upload
    trace_events_path = os.path.join(trace_dir, "trace_events.json")
    with open(trace_events_path, "w") as f:
      json.dump(events, f)

    # Collect process names per pid for diagnostics
    process_names: dict[int, list[str]] = {}
    for event in events:
      if "args" in event and "name" in event["args"]:
        pid_val = event.get("pid")
        if pid_val not in process_names:
          process_names[pid_val] = []
        name_val = event["args"]["name"]
        if name_val not in process_names[pid_val]:
          process_names[pid_val].append(name_val)

    # Find the pid for TPU device — try /device:TPU:0 first, then any TPU device
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
      return {"ok": False, "error": f"No TPU device in trace. Processes: {process_names}"}

    # Collect TPU events and computation events (multiple name patterns)
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
      if "jit_computation" in name or "jit(" in name or "pallas" in name.lower():
        if "dur" in event:
          computation_events.append(event)

    # Count SyncWait/idle events across all pids
    sync_events_count = 0
    for event in events:
      name = event.get("name") or ""
      if "SyncWait" in name or "idle" in name.lower():
        sync_events_count += 1

    if len(computation_events) < 2:
      # Last resort: use any duration event on TPU as computation marker
      dur_events = [e for e in events_for_tpu if "dur" in e and e["dur"] > 0]
      if len(dur_events) >= 2:
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

    # Use full trace span from first to last computation event
    trace_start = computation_events[0]["ts"]
    trace_end = computation_events[-1]["ts"] + computation_events[-1]["dur"]

    # Sum SyncWait/idle durations within the full trace window
    sync_wait_total = 0
    for event in events_for_tpu:
      if "dur" not in event:
        continue
      evt_start = event["ts"]
      evt_end = evt_start + event["dur"]
      if evt_start >= trace_start and evt_end <= trace_end:
        name = event.get("name") or ""
        if "SyncWait" in name or "idle" in name.lower():
          sync_wait_total += event["dur"]

    total_time = trace_end - trace_start
    if total_time <= 0:
      return {"ok": False, "error": "Invalid trace timing (total_time <= 0)"}

    ratio = sync_wait_total / total_time
    hw_utilization = parse_hw_utilization(events, pid)
    diag = {
      "process_names": {str(k): v for k, v in process_names.items()},
      "selected_pid": pid,
      "total_events": len(events),
      "tpu_events": len(events_for_tpu),
      "computation_events": len(computation_events),
      "sync_wait_events": sync_events_count,
      "event_names_sample": sorted(all_event_names)[:30],
      "window_us": total_time,
      "sync_wait_us": sync_wait_total,
    }
    return {
      "ok": True,
      "compute_ratio": 1.0 - ratio,
      "memory_transfer_ratio": ratio,
      "hw_utilization": hw_utilization,
      "diagnostics": diag,
      "_trace_events_path": trace_events_path,
    }
  except Exception:
    return {"ok": False, "error": f"Profile error: {traceback.format_exc()}"}


def stage_profile_deep(exec_globals, shapes, dump_dir=None):
  """Stage 4b: Deep IR profiling via HLO/LLO/Mosaic dumps.

  Dump flags are set at process startup (before JAX init) via
  _setup_dump_env(). This function only parses the dump files that were
  generated during earlier stages.

  For FLOPs: Pallas GMM matmuls are inside tpu_custom_call and invisible
  to HLO dot-pattern parsing. FLOPs are computed from the eval shapes
  directly (fwd + bwd_gmm + tgmm = 3 matmuls per shape).

  Self-contained — does NOT import from kernel_evolve.profiler.
  Non-fatal: returns ok=False on failure without stopping the evaluation pipeline.
  """
  try:
    if dump_dir is None:
      dump_dir = _get_dump_dir()
    import glob
    import re
    from pathlib import Path

    _DTYPE_BYTES = {
      "f32": 4,
      "float32": 4,
      "f16": 2,
      "float16": 2,
      "bf16": 2,
      "bfloat16": 2,
      "f8e5m2": 1,
      "f8e4m3fn": 1,
      "s32": 4,
      "int32": 4,
      "s16": 2,
      "int16": 2,
      "s8": 1,
      "int8": 1,
    }

    dump_path = Path(dump_dir)
    hlo_dir = dump_path / "hlo"
    llo_dir = dump_path / "llo"

    # ── Compute FLOPs from eval shapes ──────────────────────────────
    # GMM fwd+bwd has 3 matmul passes per shape:
    #   fwd:      lhs[M,K] @ rhs[G,K,N] -> [M,N]   => 2*M*K*N*G
    #   bwd_gmm:  dlhs_dout[M,N] @ rhs[G,K,N]^T     => 2*M*N*K*G
    #   tgmm:     lhs_t[K,M] @ drhs_dout[M,N]        => 2*K*M*N*G
    # Total per shape: 6 * M * K * N * G
    # Only the first shape is benchmarked for performance.
    flops = None
    if shapes:
      s = shapes[0]
      M = s.get("M", 0)
      K = s.get("K", 0)
      N = s.get("N", 0)
      G = s.get("G", 1)
      if M and K and N:
        flops = 6.0 * M * K * N * G

    # Log dump file counts
    for label, d in [("hlo", hlo_dir), ("llo", llo_dir), ("mosaic", dump_path / "mosaic")]:
      d_path = Path(d)
      count = sum(1 for _ in d_path.rglob("*") if _.is_file()) if d_path.exists() else 0
      print(f"Dump dir [{label}]: {count} files", file=sys.stderr)

    # ── Parse LLO dumps ──────────────────────────────────────────────
    # LLO files are named: {hash}-{op_name}-{pass_num}-{pass_name}.txt
    # The actual kernel body is in files with "pallas" in op_name (not
    # "custom-call" which is just the wrapper). We want the final pass
    # with the most instructions (largest file).
    vliw_bundle_count = None
    mxu_utilization = None
    mxu0_count = 0
    mxu1_count = 0

    llo_files = glob.glob(str(llo_dir / "**" / "*.txt"), recursive=True)
    llo_files += glob.glob(str(llo_dir / "**" / "*.llo"), recursive=True)

    # Parse filename pattern: {hash}-{op_name}-{pass_num}-{pass_name}.txt
    # Also accepts .llo extension and dotted naming conventions.
    file_re = re.compile(r"^\d+-(.+?)-(\d+)-(.+)\.(?:txt|llo)$")

    # Strategy: find the largest file among Pallas kernel final passes.
    # Pallas ops are named pallas_tpu_*. Final passes include
    # "final_bundles", "packed-bundles", "schedule-analysis".
    pallas_candidates = []
    other_candidates = []
    for f in llo_files:
      basename = os.path.basename(f)
      m = file_re.match(basename)
      if not m:
        # Fallback: any LLO file is a candidate (sorted by size later)
        other_candidates.append((f, basename, 0, ""))
        continue
      op_name = m.group(1)
      pass_num = int(m.group(2))
      pass_name = m.group(3)
      entry = (f, op_name, pass_num, pass_name)
      if "pallas" in op_name.lower():
        pallas_candidates.append(entry)
      elif "custom-call" in op_name or "custom_call" in op_name:
        other_candidates.append(entry)

    # Pick the largest Pallas file (most likely to contain full LLO body)
    best_file = None
    if pallas_candidates:
      pallas_candidates.sort(key=lambda e: os.path.getsize(e[0]), reverse=True)
      best_file = pallas_candidates[0][0]
    elif other_candidates:
      other_candidates.sort(key=lambda e: os.path.getsize(e[0]), reverse=True)
      best_file = other_candidates[0][0]

    if best_file is not None:
      with open(best_file) as fh:
        llo_text = fh.read()
      # Count VLIW bundles — multiple formats:
      #   Classic: separated by `;;`
      #   v7x libtpu: numbered entries like `  N  :  { ... }`
      #   v7x alt: `bundle N` markers
      bundle_count = llo_text.count(";;")
      if bundle_count == 0:
        bundle_count = len(re.findall(r"^\s*\d+\s*:\s*\{", llo_text, re.MULTILINE))
      if bundle_count == 0:
        bundle_count = len(re.findall(r"\bbundle\s+\d+", llo_text, re.IGNORECASE))
      if bundle_count == 0:
        # Last resort: count instruction-like lines (register assignments)
        bundle_count = len(
          re.findall(
            r"^\s*%\w+\s*=",
            llo_text,
            re.MULTILINE,
          )
        )
      if bundle_count > 0:
        vliw_bundle_count = bundle_count
      # Count MXU operations — multiple naming conventions
      mxu0_count = len(re.findall(r"\.mxu0\b", llo_text))
      mxu1_count = len(re.findall(r"\.mxu1\b", llo_text))
      # v7x may use different MXU names: matmul, dot, mxu, tc (tensor core)
      if mxu0_count + mxu1_count == 0:
        mxu0_count = len(re.findall(r"\bmatmul\b", llo_text, re.IGNORECASE))
        mxu1_count = len(re.findall(r"\bmxu\b", llo_text, re.IGNORECASE))

    total_mxu = mxu0_count + mxu1_count
    if total_mxu > 0:
      max_mxu = max(mxu0_count, mxu1_count)
      min_mxu = min(mxu0_count, mxu1_count)
      mxu_utilization = {
        "mxu0": mxu0_count,
        "mxu1": mxu1_count,
        "total": total_mxu,
        "dual_ratio": min_mxu / max_mxu if max_mxu > 0 else 0.0,
        "distribution": {
          "mxu0_pct": mxu0_count / total_mxu * 100,
          "mxu1_pct": mxu1_count / total_mxu * 100,
        },
      }

    # ── New LLO metrics ─────────────────────────────────────────────
    vmem_allocation = None
    bundle_density = None
    dma_analysis = None
    special_units = None

    if best_file is not None:
      # VMEM allocations
      alloc_re = re.compile(
        r"#allocation\d+\s*=\s*\w+\[[^\]]*\]"
        r"(?:,\s*space=(\w+))?"
        r",\s*size=0x([0-9a-fA-F]+)"
      )
      vmem_b = 0
      smem_b = 0
      alloc_count = 0
      for am in alloc_re.finditer(llo_text):
        space = am.group(1)
        sz = int(am.group(2), 16)
        if space == "smem":
          smem_b += sz
        else:
          vmem_b += sz
        alloc_count += 1
      if alloc_count > 0:
        vmem_allocation = {
          "vmem_bytes": vmem_b,
          "smem_bytes": smem_b,
          "allocation_count": alloc_count,
        }

      # Bundle density (ILP)
      segments = llo_text.split(";;")
      if len(segments) > 1:
        op_re = re.compile(r"^\s*%\w+\s*=", re.MULTILINE)
        ops_list = []
        for seg in segments[:-1]:
          cnt = len(op_re.findall(seg))
          if cnt > 0:
            ops_list.append(cnt)
        if ops_list:
          bundle_density = {
            "total_bundles": len(ops_list),
            "avg_ops_per_bundle": sum(ops_list) / len(ops_list),
            "max_ops_per_bundle": max(ops_list),
          }

      # DMA analysis
      dma_cnt = len(re.findall(r"\bdma\.\w+", llo_text))
      if dma_cnt > 0:
        dma_analysis = {
          "dma_count": dma_cnt,
          "dma_sync_count": len(re.findall(r"\bdma\.done\.wait\b", llo_text)),
          "double_buffering": bool(re.search(r"\bsand\.u32\s+1\b", llo_text)),
        }

      # Special hardware units
      xlane = len(re.findall(r"\b\w+\.xlane\b", llo_text))
      eup = len(re.findall(r"\bvpow2\b", llo_text)) + len(re.findall(r"\bvpop\.eup\b", llo_text))
      nops = len(re.findall(r"^\s*nop\b", llo_text, re.MULTILINE))
      if xlane + eup + nops > 0:
        special_units = {
          "xlane_ops": xlane,
          "eup_ops": eup,
          "nop_count": nops,
        }

    # ── Parse HLO dumps (for HBM bandwidth) ────────────────────────
    hbm_bytes = None

    hlo_files = glob.glob(str(hlo_dir / "**" / "*.txt"), recursive=True)
    hlo_files += glob.glob(str(hlo_dir / "**" / "*.hlo"), recursive=True)

    # Prefer files with 'after' in name (optimized HLO)
    after_files = [f for f in hlo_files if "after" in os.path.basename(f).lower()]
    chosen_hlo = after_files[0] if after_files else (hlo_files[0] if hlo_files else None)

    if chosen_hlo is not None:
      with open(chosen_hlo) as fh:
        hlo_text = fh.read()

      # Extract HBM bandwidth from custom_call with tpu_custom_call
      shape_re = re.compile(r"(\w+)\[([\d,]+)\]")
      cc_pat = re.compile(
        r"(%\S+)\s*=\s*(\S+)\s+custom-call\(([^)]+)\)"
        r"[^\"]*custom_call_target=\"tpu_custom_call\""
      )
      cc_match = cc_pat.search(hlo_text)
      if cc_match:
        hbm_bytes = 0
        # Parse output shape
        out_shape = cc_match.group(2)  # e.g., "bf16[8,2048,128]"
        out_m = shape_re.match(out_shape)
        if out_m and out_m.group(1) in _DTYPE_BYTES:
          elems = 1
          for d in out_m.group(2).split(","):
            elems *= int(d)
          hbm_bytes += elems * _DTYPE_BYTES[out_m.group(1)]

        # Parse input operand names and look up their shapes
        input_names = re.findall(r"(%\S+)", cc_match.group(3))
        for name in input_names:
          # Find parameter definition: %name = dtype[dims] parameter(N)
          param_pat = re.compile(re.escape(name) + r"\s*=\s*(\w+\[\d[\d,]*\])")
          param_m = param_pat.search(hlo_text)
          if param_m:
            inp_m = shape_re.match(param_m.group(1))
            if inp_m and inp_m.group(1) in _DTYPE_BYTES:
              elems = 1
              for d in inp_m.group(2).split(","):
                elems *= int(d)
              hbm_bytes += elems * _DTYPE_BYTES[inp_m.group(1)]

    # HLO fusion analysis
    fusion_analysis = None
    if chosen_hlo is not None:
      fc = len(re.findall(r"^fused_computation", hlo_text, re.MULTILINE))
      cpf = bool(re.search(r"cross_program_prefetch_index=", hlo_text))
      fusion_analysis = {
        "fusion_count": fc,
        "has_cross_program_prefetch": cpf,
      }

    arithmetic_intensity = (
      flops / hbm_bytes if flops and hbm_bytes else None
    )

    return {
      "ok": True,
      "vliw_bundle_count": vliw_bundle_count,
      "mxu_utilization": mxu_utilization,
      "hbm_bandwidth_bytes": hbm_bytes,
      "flops": flops,
      "arithmetic_intensity": arithmetic_intensity,
      "vmem_allocation": vmem_allocation,
      "bundle_density": bundle_density,
      "dma_analysis": dma_analysis,
      "fusion_analysis": fusion_analysis,
      "special_units": special_units,
      "_hlo_file": chosen_hlo,
      "_llo_file": best_file,
    }
  except Exception:
    return {"ok": False, "error": f"Deep profile error: {traceback.format_exc()}"}


def _get_dump_dir() -> str:
  """Return variant-specific dump dir when VARIANT_ID is set, else default."""
  variant_id = os.environ.get("VARIANT_ID", "")
  if variant_id:
    return f"/tmp/ir_dumps/{variant_id}"
  return "/tmp/ir_dumps"


def _setup_dump_env():
  """Set XLA/LIBTPU dump flags before JAX init so libtpu reads them.

  Must be called before any JAX import or compilation. Dump files are
  generated during all subsequent compilations and parsed later by
  stage_profile_deep().

  When VARIANT_ID env var is set (batch subprocess mode), dumps go to
  a variant-specific subdirectory to avoid cross-variant file conflicts.
  """
  dump_dir = _get_dump_dir()
  os.makedirs(f"{dump_dir}/hlo", exist_ok=True)
  os.makedirs(f"{dump_dir}/llo", exist_ok=True)
  os.makedirs(f"{dump_dir}/mosaic", exist_ok=True)

  os.environ["XLA_FLAGS"] = f"--xla_dump_hlo_as_text --xla_dump_to={dump_dir}/hlo " + os.environ.get("XLA_FLAGS", "")
  os.environ["LIBTPU_INIT_ARGS"] = (
    f"--xla_jf_dump_to={dump_dir}/llo "
    "--xla_jf_dump_hlo_text=true "
    "--xla_jf_dump_llo_text=true "
    "--xla_jf_emit_annotations=true "
    f"--xla_mosaic_dump_to={dump_dir}/mosaic "
    "--xla_mosaic_enable_llo_source_annotations=true "
    "--xla_enable_custom_call_region_trace=true "
    "--xla_xprof_register_llo_debug_info=true " + os.environ.get("LIBTPU_INIT_ARGS", "")
  )


def main():
  _setup_dump_env()

  parser = argparse.ArgumentParser()
  parser.add_argument("--eval-payload", required=False)
  parser.add_argument("--eval-payload-file", required=False,
                      help="Path to file containing base64 payload")
  args = parser.parse_args()
  if args.eval_payload_file:
    with open(args.eval_payload_file) as f:
      b64_payload = f.read().strip()
  elif args.eval_payload:
    b64_payload = args.eval_payload
  else:
    parser.error("--eval-payload or --eval-payload-file is required")
  request = decode_request(b64_payload)
  request_metadata = request.get("metadata", {})
  job_name = request.get("variant_id", "unknown")

  if not _has_tpu():
    print("ERROR: No TPU detected. This evaluator requires a TPU device.", file=sys.stderr)
    sys.exit(1)

  compile_result = stage_compile(request["kernel_code"])
  if not compile_result["ok"]:
    err = {"status": "COMPILE_ERROR", "variant_id": job_name, "error": compile_result["error"], "metadata": request_metadata}
    print(f"EVAL_RESULT:{json.dumps(err)}")
    sys.exit(0)

  correct_result = stage_correctness(
    request["kernel_code"],
    request["reference_code"],
    request["shapes"],
    request.get("rtol", 1e-2),
    request.get("atol", 1e-2),
    compile_result["globals"],
  )
  if not correct_result["ok"]:
    result_data = {
      "status": "INCORRECT",
      "variant_id": job_name,
      "error": correct_result["error"],
      "max_diff": correct_result["max_diff"],
      "metadata": request_metadata,
    }
    print(f"EVAL_RESULT:{json.dumps(result_data)}")
    sys.exit(0)

  perf_result = stage_performance(compile_result["globals"], request["shapes"])
  if not perf_result["ok"]:
    err = {"status": "COMPILE_ERROR", "variant_id": job_name, "error": perf_result["error"], "metadata": request_metadata}
    print(f"EVAL_RESULT:{json.dumps(err)}")
    sys.exit(0)

  ref_compile = stage_compile(request["reference_code"])
  ref_perf = stage_performance(ref_compile.get("globals", {}), request["shapes"])
  ref_latency = ref_perf.get("latency_ms", perf_result["latency_ms"])
  speedup = ref_latency / perf_result["latency_ms"] if perf_result["latency_ms"] > 0 else 0.0

  # Stage 4: Profile (non-fatal)
  profile_result = stage_profile(compile_result["globals"], request["shapes"])

  compute_ratio = None
  memory_transfer_ratio = None
  hw_utilization = None
  profile_diag = {}
  if profile_result["ok"]:
    compute_ratio = profile_result["compute_ratio"]
    memory_transfer_ratio = profile_result["memory_transfer_ratio"]
    hw_utilization = profile_result.get("hw_utilization")
    profile_diag = profile_result.get("diagnostics", {})
  else:
    profile_diag = {"error": profile_result.get("error", "unknown")}

  # Stage 5: Deep profile (non-fatal) — parse dumps generated by earlier stages
  deep_profile = stage_profile_deep(compile_result["globals"], request["shapes"])
  if not deep_profile["ok"]:
    print(
      f"Deep profile skipped: {deep_profile.get('error', 'unknown')}",
      file=sys.stderr,
    )

  # Compute efficiency from deep profile FLOPs and measured latency
  # TPU v7x per-chip peak: 2307 TFLOPS
  peak_flops = float(os.environ.get("PEAK_FLOPS", 2307e12))
  if deep_profile.get("flops") and perf_result["latency_ms"] > 0:
    actual_fps = deep_profile["flops"] / (perf_result["latency_ms"] / 1000.0)
    deep_profile["compute_efficiency_pct"] = (actual_fps / peak_flops) * 100.0

  # HBM bandwidth utilization (TPU v7x peak: 3690 GB/s)
  peak_hbm_bw = float(os.environ.get("PEAK_HBM_BW", 3690e9))
  if deep_profile.get("hbm_bandwidth_bytes") and perf_result["latency_ms"] > 0:
    actual_bw = deep_profile["hbm_bandwidth_bytes"] / (perf_result["latency_ms"] / 1000.0)
    deep_profile["hbm_bandwidth_utilization_pct"] = (actual_bw / peak_hbm_bw) * 100.0

  # ── Upload profile artifacts to GCS (non-fatal) ──
  artifacts = {}
  if profile_result.get("ok"):
    trace_path = profile_result.get("_trace_events_path")
    if trace_path:
      artifacts["trace_events.json"] = trace_path
  if deep_profile.get("ok"):
    hlo_path = deep_profile.get("_hlo_file")
    llo_path = deep_profile.get("_llo_file")
    if hlo_path:
      artifacts["hlo_post_opt.txt"] = hlo_path
    if llo_path:
      artifacts["llo_final.txt"] = llo_path

  gcs_result = upload_to_gcs(job_name, artifacts) if artifacts else {"ok": False, "uploaded": [], "gcs_prefix": ""}
  if gcs_result["ok"]:
    print(f"Uploaded artifacts: {gcs_result['uploaded']} to {gcs_result['gcs_prefix']}", file=sys.stderr)

  # Strip internal fields from deep_profile before including in result
  clean_deep_profile = {k: v for k, v in deep_profile.items() if not k.startswith("_")}

  result = {
    "status": "SUCCESS",
    "variant_id": job_name,
    "fitness": speedup,
    "latency_ms": perf_result["latency_ms"],
    "speedup": speedup,
    "flops": clean_deep_profile.get("flops", 0.0) or 0.0,
    "compute_ratio": compute_ratio,
    "memory_transfer_ratio": memory_transfer_ratio,
    "metadata": {
      **request_metadata,
      "reference_latency_ms": ref_latency,
      "reference_perf_ok": ref_perf.get("ok", False),
      "profile_diagnostics": profile_diag,
      "hw_utilization": hw_utilization,
      "profile": clean_deep_profile,
      **({"artifacts_gcs_prefix": gcs_result["gcs_prefix"]} if gcs_result.get("ok") else {}),
    },
  }
  print(f"EVAL_RESULT:{json.dumps(result)}")


if __name__ == "__main__":
  # Parse payload to check for batch mode BEFORE setting up dump env.
  # In batch mode, subprocesses each set their own dump env.
  _parser = argparse.ArgumentParser()
  _parser.add_argument("--eval-payload", required=False)
  _parser.add_argument("--eval-payload-file", required=False,
                        help="Path to file containing base64 payload")
  _args = _parser.parse_args()
  if _args.eval_payload_file:
    with open(_args.eval_payload_file) as f:
      _b64 = f.read().strip()
  elif _args.eval_payload:
    _b64 = _args.eval_payload
  else:
    _parser.error("--eval-payload or --eval-payload-file is required")
  _request = decode_request(_b64)

  if _request.get("batch"):
    # Batch mode: skip TPU check in parent — each subprocess checks
    # independently. Calling _has_tpu() here locks libtpu for PID 1,
    # preventing subprocesses from accessing the TPU.
    from kernel_evolve.docker_evaluate_helpers import batch_dispatch

    result_lines = batch_dispatch(
      _request,
      evaluator_script=__file__,
      per_variant_timeout=1500,
    )
    for line in result_lines:
      print(line)
  else:
    # Single-variant mode: original path
    main()
