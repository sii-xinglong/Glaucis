"""Tests for artifact path propagation in evaluate.py stages."""

import json
import os
import tempfile
from pathlib import Path


def test_stage_profile_returns_trace_events_path(tmp_path):
    """stage_profile should write trace_events.json and return its path."""
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    events = [{"pid": 1, "ts": 0, "dur": 100, "name": "op"}]
    trace_events_path = str(trace_dir / "trace_events.json")
    with open(trace_events_path, "w") as f:
        json.dump(events, f)
    assert Path(trace_events_path).exists()
    loaded = json.loads(Path(trace_events_path).read_text())
    assert len(loaded) == 1
    assert loaded[0]["name"] == "op"


def test_stage_profile_deep_returns_file_paths(tmp_path):
    """stage_profile_deep result should include _hlo_file and _llo_file paths."""
    hlo_dir = tmp_path / "hlo"
    llo_dir = tmp_path / "llo"
    hlo_dir.mkdir()
    llo_dir.mkdir()
    hlo_file = hlo_dir / "module.after_optimizations.txt"
    hlo_file.write_text("HLO { custom-call(...) }")
    llo_file = llo_dir / "module.pass_79.llo"
    llo_file.write_text(";; bundle1\n.mxu0 op1\n;; bundle2\n.mxu1 op2")
    result = {
        "ok": True,
        "_hlo_file": str(hlo_file),
        "_llo_file": str(llo_file),
        "vliw_bundle_count": 2,
    }
    assert Path(result["_hlo_file"]).exists()
    assert Path(result["_llo_file"]).exists()
