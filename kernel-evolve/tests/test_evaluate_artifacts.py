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
  llo_file = llo_dir / "module.pass_79.llo"
  llo_file.write_text(";; bundle1\n.mxu0 op1\n;; bundle2\n.mxu1 op2\n;; end")

  # Mock kernel_fn that does nothing (dump files already exist)
  mock_out = MagicMock()
  mock_out.block_until_ready = MagicMock()
  mock_kernel_fn = MagicMock(return_value=mock_out)
  exec_globals = {"optimized_compute": mock_kernel_fn}

  result = stage_profile_deep(exec_globals, [{"M": 1024}], dump_dir=str(tmp_path))

  assert result["ok"] is True
  assert result["_hlo_file"] == str(hlo_file)
  assert result["_llo_file"] == str(llo_file)
  # 3 ;; separators in llo file content
  assert result["vliw_bundle_count"] == 3
  # Verify kernel was actually called
  mock_kernel_fn.assert_called_once_with(M=1024)


def test_stage_profile_deep_none_when_no_dumps(tmp_path):
  """stage_profile_deep should return None for _hlo_file/_llo_file when no dumps exist."""
  from evaluate import stage_profile_deep

  mock_out = MagicMock()
  mock_out.block_until_ready = MagicMock()
  mock_kernel_fn = MagicMock(return_value=mock_out)
  exec_globals = {"optimized_compute": mock_kernel_fn}

  result = stage_profile_deep(exec_globals, [{"M": 1024}], dump_dir=str(tmp_path))

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
  llo_file = llo_dir / "module.pass_10.llo"
  llo_file.write_text(
    ";; bundle0\n"
    ".mxu0 matmul_start\n"
    ".mxu1 matmul_start\n"
    ";; bundle1\n"
    ".mxu0 matmul_done\n"
    ";; end\n"
  )

  mock_out = MagicMock()
  mock_out.block_until_ready = MagicMock()
  mock_kernel_fn = MagicMock(return_value=mock_out)
  exec_globals = {"optimized_compute": mock_kernel_fn}

  result = stage_profile_deep(exec_globals, [{"M": 512}], dump_dir=str(tmp_path))

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

  mock_out = MagicMock()
  mock_out.block_until_ready = MagicMock()
  mock_kernel_fn = MagicMock(return_value=mock_out)
  exec_globals = {"optimized_compute": mock_kernel_fn}

  result = stage_profile_deep(exec_globals, [{"M": 1024}], dump_dir=str(tmp_path))

  assert result["ok"] is True
  assert result["hbm_bandwidth_bytes"] == 8 * 2048 * 128 * 2 * 2  # output + input, bf16=2 bytes


def test_stage_profile_deep_picks_highest_pass_llo(tmp_path):
  """stage_profile_deep should select the LLO file with the highest pass number."""
  from evaluate import stage_profile_deep

  llo_dir = tmp_path / "llo"
  llo_dir.mkdir(parents=True)

  # Create multiple LLO files with different pass numbers
  (llo_dir / "module.pass_10.llo").write_text(";; early\n.mxu0 op1\n;; end")
  (llo_dir / "module.pass_99.llo").write_text(";; late\n.mxu0 op1\n.mxu0 op2\n.mxu1 op3\n;; end")
  (llo_dir / "module.pass_50.llo").write_text(";; mid\n;; end")

  mock_out = MagicMock()
  mock_out.block_until_ready = MagicMock()
  mock_kernel_fn = MagicMock(return_value=mock_out)
  exec_globals = {"optimized_compute": mock_kernel_fn}

  result = stage_profile_deep(exec_globals, [{"M": 256}], dump_dir=str(tmp_path))

  assert result["ok"] is True
  # Should pick pass_99 (highest pass number)
  assert result["_llo_file"] == str(llo_dir / "module.pass_99.llo")
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


def test_stage_profile_deep_no_kernel_fn(tmp_path):
  """stage_profile_deep should return ok=False when no kernel function is found."""
  from evaluate import stage_profile_deep

  result = stage_profile_deep({}, [{"M": 1024}], dump_dir=str(tmp_path))

  assert result["ok"] is False
  assert "No kernel_fn" in result["error"]


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
