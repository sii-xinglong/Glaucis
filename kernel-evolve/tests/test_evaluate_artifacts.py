"""Tests for artifact path propagation in evaluate.py stages.

These tests call the real stage_profile_deep() function with mocked
kernel execution and pre-created dump files, verifying that the
file-finding and parsing logic works correctly end-to-end.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the docker dir to path so we can import evaluate
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "docker"))


def test_stage_profile_deep_returns_file_paths(tmp_path):
  """stage_profile_deep should find dump files and return their paths in the result."""
  from evaluate import stage_profile_deep

  hlo_dir = tmp_path / "hlo"
  llo_dir = tmp_path / "llo"
  mosaic_dir = tmp_path / "mosaic"
  hlo_dir.mkdir(parents=True)
  llo_dir.mkdir(parents=True)
  mosaic_dir.mkdir(parents=True)

  # Pre-create dump files that would normally be generated during kernel execution
  hlo_file = hlo_dir / "module.after_optimizations.txt"
  hlo_file.write_text(
    '%p0 = bf16[1024,1024] parameter(0)\n'
    '%result = bf16[1024,1024] custom-call(%p0)'
    ' custom_call_target="tpu_custom_call"'
  )
  llo_file = llo_dir / "12345-pallas_tpu_kernel-79-final_bundles.txt"
  llo_file.write_text(";; bundle1\n.mxu0 op1\n;; bundle2\n.mxu1 op2\n;; end")

  # stage_profile_deep only reads dump files; exec_globals is unused
  result = stage_profile_deep({}, [{"M": 1024}], dump_dir=str(tmp_path))

  assert result["ok"] is True
  assert result["_hlo_file"] == str(hlo_file)
  assert result["_llo_file"] is not None
  # 3 ;; separators in llo file content
  assert result["vliw_bundle_count"] == 3


def test_stage_profile_deep_none_when_no_dumps(tmp_path):
  """stage_profile_deep should return None for _hlo_file/_llo_file when no dumps exist."""
  from evaluate import stage_profile_deep

  result = stage_profile_deep({}, [{"M": 1024}], dump_dir=str(tmp_path))

  assert result["ok"] is True
  assert result["_hlo_file"] is None
  assert result["_llo_file"] is None
  assert result["vliw_bundle_count"] is None


def test_stage_profile_deep_mxu_utilization(tmp_path):
  """stage_profile_deep should parse MXU utilization from LLO dumps."""
  from evaluate import stage_profile_deep

  llo_dir = tmp_path / "llo"
  llo_dir.mkdir(parents=True)

  # Create LLO with known MXU operation counts
  llo_file = llo_dir / "12345-pallas_tpu_kernel-10-post_mxu_assigner.txt"
  llo_file.write_text(
    ";; bundle0\n"
    ".mxu0 matmul_start\n"
    ".mxu1 matmul_start\n"
    ";; bundle1\n"
    ".mxu0 matmul_done\n"
    ";; end\n"
  )

  result = stage_profile_deep({}, [{"M": 512}], dump_dir=str(tmp_path))

  assert result["ok"] is True
  assert result["mxu_utilization"] is not None
  assert result["mxu_utilization"]["mxu0"] == 2
  assert result["mxu_utilization"]["mxu1"] == 1
  assert result["mxu_utilization"]["total"] == 3
  # dual_ratio = min/max = 1/2 = 0.5
  assert result["mxu_utilization"]["dual_ratio"] == 0.5


def test_stage_profile_deep_hlo_bandwidth_parsing(tmp_path):
  """stage_profile_deep should extract HBM bandwidth from HLO custom_call."""
  from evaluate import stage_profile_deep

  hlo_dir = tmp_path / "hlo"
  hlo_dir.mkdir(parents=True)

  # HLO with a custom_call that has parseable shapes:
  # Output: bf16[8,2048,128] = 8*2048*128 = 2097152 elements * 2 bytes = 4194304
  # Input: bf16[8,2048,128] parameter(0) = same = 4194304
  # Total: 8388608 bytes
  hlo_file = hlo_dir / "module.after_optimizations.txt"
  hlo_file.write_text(
    '%p0 = bf16[8,2048,128] parameter(0)\n'
    '%result = bf16[8,2048,128] custom-call(%p0)'
    ' custom_call_target="tpu_custom_call"\n'
  )

  result = stage_profile_deep({}, [{"M": 1024}], dump_dir=str(tmp_path))

  assert result["ok"] is True
  assert result["hbm_bandwidth_bytes"] == 8 * 2048 * 128 * 2 * 2  # output + input, bf16=2 bytes


def test_stage_profile_deep_picks_highest_pass_llo(tmp_path):
  """stage_profile_deep should select the largest LLO file (most complete body)."""
  from evaluate import stage_profile_deep

  llo_dir = tmp_path / "llo"
  llo_dir.mkdir(parents=True)

  # Create multiple LLO files with different pass numbers — code picks the largest file
  (llo_dir / "12345-pallas_tpu_kernel-10-early.txt").write_text(";; early\n.mxu0 op1\n;; end")
  (llo_dir / "12345-pallas_tpu_kernel-99-final_bundles.txt").write_text(";; late\n.mxu0 op1\n.mxu0 op2\n.mxu1 op3\n;; end")
  (llo_dir / "12345-pallas_tpu_kernel-50-mid.txt").write_text(";; mid\n;; end")

  result = stage_profile_deep({}, [{"M": 256}], dump_dir=str(tmp_path))

  assert result["ok"] is True
  # Should pick the largest pallas file (pass_99 has most content)
  assert "pallas_tpu_kernel-99" in result["_llo_file"]
  # pass_99 has 2 mxu0 ops and 1 mxu1 op
  assert result["mxu_utilization"]["mxu0"] == 2
  assert result["mxu_utilization"]["mxu1"] == 1


def test_stage_profile_deep_restores_env_vars(tmp_path):
  """stage_profile_deep should restore XLA_FLAGS and LIBTPU_INIT_ARGS after execution."""
  from evaluate import stage_profile_deep

  mock_out = MagicMock()
  mock_out.block_until_ready = MagicMock()
  mock_kernel_fn = MagicMock(return_value=mock_out)
  exec_globals = {"optimized_compute": mock_kernel_fn}

  # Set known env vars before calling
  os.environ["XLA_FLAGS"] = "--original-flag"
  os.environ["LIBTPU_INIT_ARGS"] = "--original-libtpu"

  try:
    stage_profile_deep(exec_globals, [{"M": 1024}], dump_dir=str(tmp_path))

    # Verify env vars are restored
    assert os.environ.get("XLA_FLAGS") == "--original-flag"
    assert os.environ.get("LIBTPU_INIT_ARGS") == "--original-libtpu"
  finally:
    # Clean up
    os.environ.pop("XLA_FLAGS", None)
    os.environ.pop("LIBTPU_INIT_ARGS", None)


def test_stage_profile_deep_no_dumps_returns_ok_true(tmp_path):
  """stage_profile_deep returns ok=True with null metrics when no dumps exist.

  The function only reads dump files; it does not use exec_globals.
  Missing dumps are not an error — they just mean no IR-level metrics are available.
  """
  from evaluate import stage_profile_deep

  result = stage_profile_deep({}, [{"M": 1024}], dump_dir=str(tmp_path))

  assert result["ok"] is True
  assert result["_hlo_file"] is None
  assert result["_llo_file"] is None
  assert result["vliw_bundle_count"] is None
  assert result["mxu_utilization"] is None


def test_trace_events_json_roundtrip(tmp_path):
  """Verify trace events JSON write/read roundtrip (format validation)."""
  events = [
    {"pid": 1, "ts": 0, "dur": 100, "name": "op1"},
    {"pid": 1, "ts": 100, "dur": 50, "name": "SyncWait"},
  ]
  path = tmp_path / "trace_events.json"
  with open(path, "w") as f:
    json.dump(events, f)
  loaded = json.loads(path.read_text())
  assert len(loaded) == 2
  assert loaded[0]["name"] == "op1"


def test_upload_to_gcs_success(tmp_path):
    """upload_to_gcs uploads files and returns uploaded list."""
    from evaluate import upload_to_gcs

    hlo = tmp_path / "hlo.txt"
    hlo.write_text("HLO content")
    llo = tmp_path / "llo.txt"
    llo.write_text("LLO content")

    mock_blob = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    with patch("evaluate.storage") as mock_storage:
        mock_storage.Client.return_value = mock_client
        result = upload_to_gcs("test-job", {
            "hlo_post_opt.txt": str(hlo),
            "llo_final.txt": str(llo),
        })

    assert result["ok"] is True
    assert set(result["uploaded"]) == {"hlo_post_opt.txt", "llo_final.txt"}
    assert result["gcs_prefix"] == "gs://glaucis-profiles/test-job"
    assert mock_blob.upload_from_filename.call_count == 2


def test_upload_to_gcs_missing_file(tmp_path):
    """upload_to_gcs skips files that don't exist."""
    from evaluate import upload_to_gcs

    existing = tmp_path / "exists.txt"
    existing.write_text("content")

    mock_blob = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    with patch("evaluate.storage") as mock_storage:
        mock_storage.Client.return_value = mock_client
        result = upload_to_gcs("test-job", {
            "exists.txt": str(existing),
            "missing.txt": "/nonexistent/path.txt",
        })

    assert result["ok"] is True
    assert result["uploaded"] == ["exists.txt"]
    assert mock_blob.upload_from_filename.call_count == 1


def test_upload_to_gcs_all_fail():
    """upload_to_gcs returns ok=False when all uploads fail."""
    from evaluate import upload_to_gcs
    result = upload_to_gcs("test-job", {
        "a.txt": "/nonexistent/a.txt",
        "b.txt": "/nonexistent/b.txt",
    })
    assert result["ok"] is False
    assert result["uploaded"] == []


def test_upload_to_gcs_empty_artifacts():
    """upload_to_gcs handles empty artifact dict."""
    from evaluate import upload_to_gcs
    result = upload_to_gcs("test-job", {})
    assert result["ok"] is False
    assert result["uploaded"] == []


def test_main_collects_artifacts_for_upload():
  """Verify the artifact collection logic from stage results."""
  profile_result = {
    "ok": True,
    "compute_ratio": 0.85,
    "memory_transfer_ratio": 0.15,
    "diagnostics": {},
    "_trace_events_path": "/tmp/xplane_trace/trace_events.json",
  }
  deep_profile = {
    "ok": True,
    "vliw_bundle_count": 100,
    "_hlo_file": "/tmp/ir_dumps/hlo/module.after.txt",
    "_llo_file": "/tmp/ir_dumps/llo/module.pass_79.llo",
  }

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

  assert artifacts == {
    "trace_events.json": "/tmp/xplane_trace/trace_events.json",
    "hlo_post_opt.txt": "/tmp/ir_dumps/hlo/module.after.txt",
    "llo_final.txt": "/tmp/ir_dumps/llo/module.pass_79.llo",
  }


def test_internal_fields_stripped_from_eval_result():
  """Internal _fields must not appear in EVAL_RESULT JSON."""
  deep_profile = {
    "ok": True,
    "vliw_bundle_count": 100,
    "_hlo_file": "/tmp/some/path.txt",
    "_llo_file": "/tmp/other/path.llo",
  }
  clean = {k: v for k, v in deep_profile.items() if not k.startswith("_")}
  assert "_hlo_file" not in clean
  assert "_llo_file" not in clean
  assert "vliw_bundle_count" in clean
  assert "ok" in clean


def test_stage_profile_deep_extended_metrics(tmp_path):
    """stage_profile_deep should return new extended metrics."""
    from evaluate import stage_profile_deep

    llo_dir = tmp_path / "llo"
    llo_dir.mkdir(parents=True)
    llo_text = """\
#allocation0 = f32[262144], size=0x100000
#allocation1 = u8[512], space=smem, size=0x200
;;
%v0 = vmatmul.mubr.bf16.gmra.mxu0 %r0
%v1 = vmax.xlane.f32.xlu0 %r1
;;
%v2 = dma.hbm_to_vmem %r2
%s3 = sand.u32 1, %s0
dma.done.wait %flag0
;;
%v4 = vpow2.f32 %r3
nop
;;
"""
    llo_file = llo_dir / "12345-pallas_tpu_kernel-79-final_bundles.txt"
    llo_file.write_text(llo_text)

    hlo_dir = tmp_path / "hlo"
    hlo_dir.mkdir(parents=True)
    hlo_text = """\
HloModule test
fused_computation.1 { ROOT %r = f32[1] parameter(0) }
ENTRY main {
  %p0 = bf16[8,2048,128] parameter(0)
  ROOT %out = bf16[8,2048,128] custom-call(%p0), custom_call_target="tpu_custom_call"
}
"""
    (hlo_dir / "module.after_optimizations.txt").write_text(hlo_text)

    mock_out = MagicMock()
    mock_out.block_until_ready = MagicMock()
    mock_kernel_fn = MagicMock(return_value=mock_out)
    exec_globals = {"optimized_compute": mock_kernel_fn}

    result = stage_profile_deep(exec_globals, [{"M": 1024}], dump_dir=str(tmp_path))

    assert result["ok"] is True

    # VMEM allocation
    assert result["vmem_allocation"] is not None
    assert result["vmem_allocation"]["vmem_bytes"] == 0x100000
    assert result["vmem_allocation"]["smem_bytes"] == 0x200

    # Bundle density
    assert result["bundle_density"] is not None
    assert result["bundle_density"]["max_ops_per_bundle"] >= 2

    # DMA analysis
    assert result["dma_analysis"] is not None
    assert result["dma_analysis"]["dma_count"] >= 1
    assert result["dma_analysis"]["double_buffering"] is True
    assert result["dma_analysis"]["dma_sync_count"] == 1

    # Fusion analysis
    assert result["fusion_analysis"] is not None
    assert result["fusion_analysis"]["fusion_count"] == 1

    # Special units
    assert result["special_units"] is not None
    assert result["special_units"]["xlane_ops"] == 1
    assert result["special_units"]["eup_ops"] == 1
    assert result["special_units"]["nop_count"] == 1


# ── parse_hw_utilization tests ──


def _make_counter_event(pid, name, dur, args):
  """Helper: build a Chrome trace counter event."""
  return {"pid": pid, "tid": 4294967295, "ph": "X", "name": name, "dur": dur, "args": args}


def test_parse_hw_utilization_basic():
  """Weighted average across two equal-duration MXU windows, plus fills/spills."""
  from evaluate import parse_hw_utilization

  TPU_PID = 3
  events = [
    # Two MXU windows with equal duration
    _make_counter_event(TPU_PID, "MXU", 100, {"% util": "20.0"}),
    _make_counter_event(TPU_PID, "MXU", 100, {"% util": "40.0"}),
    # One Vector ALU window
    _make_counter_event(TPU_PID, "Vector ALU", 100, {"% util": "10.0"}),
    # Fills and spills
    _make_counter_event(TPU_PID, "Vector Fills", 100, {"fills": "5"}),
    _make_counter_event(TPU_PID, "Vector Fills", 100, {"fills": "3"}),
    _make_counter_event(TPU_PID, "Vector Spills", 100, {"spills": "0"}),
    # Non-counter event (different tid) — should be ignored
    {"pid": TPU_PID, "tid": 8, "ph": "X", "name": "MXU", "dur": 100, "args": {"% util": "99"}},
  ]
  result = parse_hw_utilization(events, TPU_PID)
  assert result is not None
  assert result["mxu_util_pct"] == 30.0
  assert result["vector_alu_util_pct"] == 10.0
  assert result["scalar_alu_util_pct"] == 0.0
  assert result["vector_fills"] == 8
  assert result["vector_spills"] == 0


def test_parse_hw_utilization_no_counters():
  """Returns None when no counter events present."""
  from evaluate import parse_hw_utilization

  events = [
    {"pid": 3, "tid": 8, "ph": "X", "name": "tpu_custom_call", "dur": 100},
  ]
  assert parse_hw_utilization(events, 3) is None


def test_parse_hw_utilization_ignores_other_pids():
  """Only events matching tpu_pid are used."""
  from evaluate import parse_hw_utilization

  events = [
    _make_counter_event(999, "MXU", 100, {"% util": "90.0"}),  # wrong pid
    _make_counter_event(1, "MXU", 100, {"% util": "10.0"}),    # correct pid
  ]
  result = parse_hw_utilization(events, tpu_pid=1)
  assert result is not None
  assert result["mxu_util_pct"] == 10.0


def test_parse_hw_utilization_weighted_average():
  """Unequal durations produce correct time-weighted average, not simple mean."""
  from evaluate import parse_hw_utilization

  events = [
    _make_counter_event(3, "MXU", 50, {"% util": "80.0"}),
    _make_counter_event(3, "MXU", 150, {"% util": "20.0"}),
  ]
  result = parse_hw_utilization(events, 3)
  # weighted: (80*50 + 20*150) / 200 = 35.0, not simple mean 50.0
  assert result["mxu_util_pct"] == 35.0
