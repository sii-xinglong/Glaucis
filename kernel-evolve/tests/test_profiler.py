"""Tests for the XPlane trace profiler module."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from kernel_evolve.profiler import (
  analyze_bundle_density,
  analyze_dma_ops,
  analyze_ir_dumps,
  analyze_special_units,
  analyze_trace,
  capture_ir_dumps,
  capture_trace,
  compute_derived_metrics,
  count_flops_from_hlo,
  count_hlo_fusions,
  count_vliw_bundles,
  estimate_hbm_bandwidth,
  find_final_llo_file,
  find_hlo_file,
  parse_mxu_distribution,
  parse_vmem_allocations,
  stage_profile,
)

# Minimal Chrome trace JSON matching the structure produced by xprof trace_viewer
MOCK_TRACE_JSON = json.dumps({
  "traceEvents": [
    {"pid": 1, "tid": 0, "ph": "M", "name": "process_name", "args": {"name": "/device:TPU:0"}},
    {"pid": 1, "tid": 1, "ph": "X", "name": "jit_computation.1", "ts": 100, "dur": 50},
    {"pid": 1, "tid": 1, "ph": "X", "name": "jit_computation.2", "ts": 200, "dur": 50},
    {"pid": 1, "tid": 1, "ph": "X", "name": "jit_computation.3", "ts": 300, "dur": 50},
    {"pid": 1, "tid": 1, "ph": "X", "name": "SyncWait:hbm_transfer", "ts": 260, "dur": 20},
    {"pid": 1, "tid": 1, "ph": "X", "name": "SyncWait:old", "ts": 100, "dur": 10},
    {"pid": 2, "tid": 0, "ph": "M", "name": "process_name", "args": {"name": "/host:CPU"}},
    {"pid": 2, "tid": 1, "ph": "X", "name": "SyncWait:cpu", "ts": 260, "dur": 30},
  ]
})


@patch("kernel_evolve.profiler.raw_to_tool_data")
def test_analyze_trace_computes_ratios(mock_xprof):
  mock_xprof.xspace_to_tool_data.return_value = (MOCK_TRACE_JSON, None)
  result = analyze_trace("/fake/trace.xplane.pb")
  # Window: ts=250 (end of jit_computation.2) to ts=350 (end of jit_computation.3)
  # Total = 100, SyncWait in window = 20 (hbm_transfer at 260-280), ratio = 0.2
  assert result["memory_transfer_ratio"] == pytest.approx(0.2)
  assert result["compute_ratio"] == pytest.approx(0.8)


@patch("kernel_evolve.profiler.raw_to_tool_data")
def test_analyze_trace_no_tpu_events(mock_xprof):
  mock_xprof.xspace_to_tool_data.return_value = (json.dumps({"traceEvents": []}), None)
  result = analyze_trace("/fake/empty.xplane.pb")
  assert result is None


@patch("kernel_evolve.profiler.raw_to_tool_data")
def test_analyze_trace_no_syncwait(mock_xprof):
  trace = json.dumps({
    "traceEvents": [
      {"pid": 1, "tid": 0, "ph": "M", "name": "process_name", "args": {"name": "/device:TPU:0"}},
      {"pid": 1, "tid": 1, "ph": "X", "name": "jit_computation.1", "ts": 100, "dur": 50},
      {"pid": 1, "tid": 1, "ph": "X", "name": "jit_computation.2", "ts": 200, "dur": 50},
      {"pid": 1, "tid": 1, "ph": "X", "name": "jit_computation.3", "ts": 300, "dur": 50},
    ]
  })
  mock_xprof.xspace_to_tool_data.return_value = (trace, None)
  result = analyze_trace("/fake/trace.xplane.pb")
  assert result["memory_transfer_ratio"] == pytest.approx(0.0)
  assert result["compute_ratio"] == pytest.approx(1.0)


def test_capture_trace_finds_xplane(tmp_path):
  profile_dir = tmp_path / "plugins" / "profile" / "2026_01_01"
  profile_dir.mkdir(parents=True)
  (profile_dir / "w-0.xplane.pb").write_bytes(b"fake")

  mock_kernel = MagicMock(return_value=MagicMock(block_until_ready=MagicMock()))

  with patch.dict("sys.modules", {"jax": MagicMock(), "jax.profiler": MagicMock()}):
    result = capture_trace(mock_kernel, {"M": 64}, str(tmp_path), warmup=1, runs=1)

  assert result is not None
  assert result.endswith(".xplane.pb")


def test_capture_trace_returns_none_when_no_xplane(tmp_path):
  mock_kernel = MagicMock(return_value=MagicMock(block_until_ready=MagicMock()))

  with patch.dict("sys.modules", {"jax": MagicMock(), "jax.profiler": MagicMock()}):
    result = capture_trace(mock_kernel, {"M": 64}, str(tmp_path), warmup=1, runs=1)

  assert result is None


@patch("kernel_evolve.profiler.analyze_trace")
@patch("kernel_evolve.profiler.capture_trace")
def test_stage_profile_success(mock_capture, mock_analyze):
  mock_capture.return_value = "/tmp/trace.xplane.pb"
  mock_analyze.return_value = {"compute_ratio": 0.85, "memory_transfer_ratio": 0.15}

  exec_globals = {"optimized_compute": lambda **kw: None}
  result = stage_profile(exec_globals, [{"M": 64, "N": 64}])

  assert result["ok"] is True
  assert result["compute_ratio"] == 0.85
  assert result["memory_transfer_ratio"] == 0.15


@patch("kernel_evolve.profiler.capture_trace")
def test_stage_profile_no_xplane(mock_capture):
  mock_capture.return_value = None

  exec_globals = {"optimized_compute": lambda **kw: None}
  result = stage_profile(exec_globals, [{"M": 64}])

  assert result["ok"] is False
  assert "No .xplane.pb" in result["error"]


def test_stage_profile_no_kernel_fn():
  result = stage_profile({}, [{"M": 64}])
  assert result["ok"] is False
  assert "No kernel_fn" in result["error"]


# ---------------------------------------------------------------------------
# Fixtures for IR parsing functions
# ---------------------------------------------------------------------------

LLO_TEXT_FIXTURE = """\
# Module: jit_optimized_compute
# Pass 79: final

%s0 = scalar_const 0
%s1 = scalar_const 1
;;
%v0 = vmatprep.subr.mxu0 %r0, %r1
%v1 = vmatpush1.bf16.xpose.msra.mxu1 %r2
%v2 = vmax.xlane.f32.xlu0 %r3, %r4
;;
%v3 = vmatmul.mubr.bf16.gmra.mxu0 %r5
%v4 = vpow2.f32 %r6
;;
%v5 = dma.hbm_to_vmem %r7
%v6 = vsel.bf16 %r8, %r9
;;
"""

HLO_TEXT_FIXTURE = """\
HloModule jit_optimized_compute

ENTRY main {
  %p0 = bf16[8,2048,128] parameter(0)
  %p1 = bf16[8,2048,128] parameter(1)
  %p2 = bf16[8,2048,128] parameter(2)
  %custom-call = bf16[8,2048,128] custom-call(%p0, %p1, %p2), custom_call_target="tpu_custom_call", backend_config="..."
  ROOT %tuple = (bf16[8,2048,128]) tuple(%custom-call)
}
"""

HLO_DOT_FIXTURE = """\
HloModule jit_reference

ENTRY main {
  %p0 = f32[512,256] parameter(0)
  %p1 = f32[256,512] parameter(1)
  %dot.1 = f32[512,512] dot(%p0, %p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT %tuple = (f32[512,512]) tuple(%dot.1)
}
"""


# ---------------------------------------------------------------------------
# Tests: count_vliw_bundles
# ---------------------------------------------------------------------------


def test_count_vliw_bundles():
  result = count_vliw_bundles(LLO_TEXT_FIXTURE)
  assert result == 4


def test_count_vliw_bundles_empty():
  result = count_vliw_bundles("")
  assert result is None


def test_count_vliw_bundles_no_bundles():
  result = count_vliw_bundles("some text without double semicolons")
  assert result is None


# ---------------------------------------------------------------------------
# Tests: parse_mxu_distribution
# ---------------------------------------------------------------------------


def test_parse_mxu_distribution():
  result = parse_mxu_distribution(LLO_TEXT_FIXTURE)
  assert result is not None
  assert result["mxu0"] == 2
  assert result["mxu1"] == 1
  assert result["dual_ratio"] == pytest.approx(0.5)


def test_parse_mxu_distribution_no_ops():
  result = parse_mxu_distribution("no mxu operations here")
  assert result is None


# ---------------------------------------------------------------------------
# Tests: estimate_hbm_bandwidth
# ---------------------------------------------------------------------------


def test_estimate_hbm_bandwidth():
  result = estimate_hbm_bandwidth(HLO_TEXT_FIXTURE)
  # 3 inputs bf16[8,2048,128] + 1 output bf16[8,2048,128] = 4 * 2 * 8 * 2048 * 128 = 16777216
  assert result == 16777216


def test_estimate_hbm_bandwidth_no_custom_call():
  hlo = """\
HloModule simple

ENTRY main {
  %p0 = f32[64] parameter(0)
  ROOT %add = f32[64] add(%p0, %p0)
}
"""
  result = estimate_hbm_bandwidth(hlo)
  assert result is None


# ---------------------------------------------------------------------------
# Tests: count_flops_from_hlo
# ---------------------------------------------------------------------------


def test_count_flops_from_hlo():
  result = count_flops_from_hlo(HLO_DOT_FIXTURE)
  # 2 * 512 * 256 * 512 = 134217728
  assert result == 134217728


def test_count_flops_from_hlo_no_dots():
  hlo = """\
HloModule simple

ENTRY main {
  %p0 = f32[64] parameter(0)
  ROOT %add = f32[64] add(%p0, %p0)
}
"""
  result = count_flops_from_hlo(hlo)
  assert result is None


# ---------------------------------------------------------------------------
# Tests: compute_derived_metrics
# ---------------------------------------------------------------------------


def test_compute_derived_metrics():
  flops = 134217728.0
  hbm_bytes = 16777216
  latency_ms = 0.5
  result = compute_derived_metrics(flops, hbm_bytes, latency_ms)
  assert result["arithmetic_intensity"] == pytest.approx(flops / hbm_bytes)
  assert result["compute_efficiency_pct"] is not None
  assert result["compute_efficiency_pct"] > 0
  assert result["compute_efficiency_pct"] < 100


def test_compute_derived_metrics_missing_data():
  result = compute_derived_metrics(None, None, 0.5)
  assert result["arithmetic_intensity"] is None
  assert result["compute_efficiency_pct"] is None


# ---------------------------------------------------------------------------
# Tests: find_final_llo_file
# ---------------------------------------------------------------------------


def test_find_final_llo_file(tmp_path):
  (tmp_path / "module.pass_02.llo").write_text("early pass")
  (tmp_path / "module.pass_45.llo").write_text("middle pass")
  (tmp_path / "module.pass_79.llo").write_text("final pass content")
  result = find_final_llo_file(str(tmp_path))
  assert result is not None
  assert "pass_79" in result


def test_find_final_llo_file_empty_dir(tmp_path):
  assert find_final_llo_file(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Tests: find_hlo_file
# ---------------------------------------------------------------------------


def test_find_hlo_file(tmp_path):
  (tmp_path / "module.after_all_optimizations.txt").write_text("HLO content")
  (tmp_path / "module.before_optimizations.txt").write_text("early HLO")
  result = find_hlo_file(str(tmp_path))
  assert result is not None


def test_find_hlo_file_empty_dir(tmp_path):
  assert find_hlo_file(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Tests: analyze_ir_dumps
# ---------------------------------------------------------------------------


def test_analyze_ir_dumps(tmp_path):
  llo_dir = tmp_path / "llo"
  llo_dir.mkdir()
  (llo_dir / "module.pass_79.llo").write_text(LLO_TEXT_FIXTURE)
  hlo_dir = tmp_path / "hlo"
  hlo_dir.mkdir()
  (hlo_dir / "module.after_all_optimizations.txt").write_text(HLO_TEXT_FIXTURE)
  result = analyze_ir_dumps(str(hlo_dir), str(llo_dir))
  assert result["vliw_bundle_count"] == 4
  assert result["mxu_utilization"]["mxu0"] == 2
  assert result["hbm_bandwidth_bytes"] == 16777216


def test_analyze_ir_dumps_empty_dirs(tmp_path):
  result = analyze_ir_dumps(str(tmp_path / "hlo"), str(tmp_path / "llo"))
  assert result["vliw_bundle_count"] is None
  assert result["mxu_utilization"] is None
  assert result["hbm_bandwidth_bytes"] is None


# ---------------------------------------------------------------------------
# Tests: capture_ir_dumps
# ---------------------------------------------------------------------------


@patch.dict("os.environ", {}, clear=False)
def test_capture_ir_dumps_sets_env_and_runs(tmp_path):
  mock_kernel = MagicMock(return_value=MagicMock(block_until_ready=MagicMock()))
  dump_dir = str(tmp_path / "dumps")
  capture_ir_dumps(mock_kernel, {"M": 64}, dump_dir)
  assert mock_kernel.called
  assert os.path.isdir(dump_dir)


# ---------------------------------------------------------------------------
# Fixtures for extended IR metrics
# ---------------------------------------------------------------------------

VMEM_ALLOCATION_FIXTURE = """\
#allocation0 = u8[524288], size=0x80000
#allocation1 = f32[262144], size=0x100000
#allocation2 = bf16[4096], size=0x2000
#allocation3 = u8[512], space=smem, size=0x200
"""

HLO_FUSION_FIXTURE = """\
HloModule jit_fused

fused_computation.1 {
  %p0 = f32[1024] parameter(0)
  ROOT %add = f32[1024] add(%p0, %p0)
}

fused_computation.2 {
  %p0 = f32[1024] parameter(0)
  ROOT %mul = f32[1024] multiply(%p0, %p0)
}

fused_computation.3 {
  %p0 = f32[1024] parameter(0)
  ROOT %neg = f32[1024] negate(%p0)
}

ENTRY main {
  %p0 = f32[1024] parameter(0), cross_program_prefetch_index=0
  %fusion.1 = f32[1024] fusion(%p0), calls=fused_computation.1
  %fusion.2 = f32[1024] fusion(%fusion.1), calls=fused_computation.2
  ROOT %fusion.3 = f32[1024] fusion(%fusion.2), calls=fused_computation.3
}
"""

LLO_DOUBLE_BUFFER_FIXTURE = """\
%v0 = dma.hbm_to_vmem %r0
%v1 = dma.vmem_to_hbm %r1
%v2 = dma.done.wait %r2
%s0 = sand.u32 1
;;
"""

LLO_SPECIAL_UNITS_FIXTURE = """\
  nop
%v0 = vmax.xlane.f32.xlu0 %r0, %r1
%v1 = vadd.xlane.f32 %r2, %r3
%v2 = vpow2.f32 %r6
%v3 = vpop.eup %r7
  nop
;;
"""


# ---------------------------------------------------------------------------
# Tests: parse_vmem_allocations
# ---------------------------------------------------------------------------


def test_parse_vmem_allocations():
  result = parse_vmem_allocations(VMEM_ALLOCATION_FIXTURE)
  assert result is not None
  # alloc0: 0x80000 = 524288, alloc1: 0x100000 = 1048576, alloc2: 0x2000 = 8192 -> vmem
  # alloc3: 0x200 = 512 -> smem
  assert result["vmem_bytes"] == 524288 + 1048576 + 8192
  assert result["smem_bytes"] == 512
  assert result["allocation_count"] == 4


def test_parse_vmem_allocations_no_allocs():
  result = parse_vmem_allocations("no allocations here")
  assert result is None


# ---------------------------------------------------------------------------
# Tests: analyze_bundle_density
# ---------------------------------------------------------------------------


def test_analyze_bundle_density():
  result = analyze_bundle_density(LLO_TEXT_FIXTURE)
  assert result is not None
  # 4 ;; separators -> 4 bundles (last segment after final ;; is skipped)
  # Bundle 1: %s0, %s1 -> 2 ops
  # Bundle 2: %v0, %v1, %v2 -> 3 ops
  # Bundle 3: %v3, %v4 -> 2 ops
  # Bundle 4: %v5, %v6 -> 2 ops
  assert result["total_bundles"] == 4
  assert result["avg_ops_per_bundle"] == pytest.approx(2.25)
  assert result["max_ops_per_bundle"] == 3


def test_analyze_bundle_density_empty():
  result = analyze_bundle_density("")
  assert result is None


def test_analyze_bundle_density_single_bundle():
  llo = """\
%v0 = vmatprep.subr.mxu0 %r0, %r1
%v1 = vmatpush1.bf16.xpose.msra.mxu1 %r2
;;
"""
  result = analyze_bundle_density(llo)
  assert result is not None
  assert result["total_bundles"] == 1
  assert result["avg_ops_per_bundle"] == pytest.approx(2.0)
  assert result["max_ops_per_bundle"] == 2


# ---------------------------------------------------------------------------
# Tests: analyze_dma_ops
# ---------------------------------------------------------------------------


def test_analyze_dma_ops():
  result = analyze_dma_ops(LLO_TEXT_FIXTURE)
  assert result is not None
  assert result["dma_count"] == 1  # dma.hbm_to_vmem
  assert result["dma_sync_count"] == 0
  assert result["double_buffering"] is False


def test_analyze_dma_ops_double_buffering():
  result = analyze_dma_ops(LLO_DOUBLE_BUFFER_FIXTURE)
  assert result is not None
  assert result["dma_count"] == 3  # dma.hbm_to_vmem, dma.vmem_to_hbm, dma.done.wait
  assert result["dma_sync_count"] == 1  # dma.done.wait
  assert result["double_buffering"] is True


def test_analyze_dma_ops_no_dma():
  result = analyze_dma_ops("no dma operations here")
  assert result is None


# ---------------------------------------------------------------------------
# Tests: count_hlo_fusions
# ---------------------------------------------------------------------------


def test_count_hlo_fusions():
  result = count_hlo_fusions(HLO_FUSION_FIXTURE)
  assert result is not None
  assert result["fusion_count"] == 3
  assert result["has_cross_program_prefetch"] is True


def test_count_hlo_fusions_no_fusions():
  result = count_hlo_fusions(HLO_TEXT_FIXTURE)
  assert result is not None
  assert result["fusion_count"] == 0
  assert result["has_cross_program_prefetch"] is False


def test_count_hlo_fusions_empty():
  result = count_hlo_fusions("")
  assert result is None


# ---------------------------------------------------------------------------
# Tests: analyze_special_units
# ---------------------------------------------------------------------------


def test_analyze_special_units():
  result = analyze_special_units(LLO_TEXT_FIXTURE)
  assert result is not None
  assert result["xlane_ops"] == 1  # vmax.xlane
  assert result["eup_ops"] == 1   # vpow2
  assert result["nop_count"] == 0


def test_analyze_special_units_multiple():
  result = analyze_special_units(LLO_SPECIAL_UNITS_FIXTURE)
  assert result is not None
  assert result["xlane_ops"] == 2   # vmax.xlane, vadd.xlane
  assert result["eup_ops"] == 2    # vpow2, vpop.eup
  assert result["nop_count"] == 2


def test_analyze_special_units_no_special():
  result = analyze_special_units("plain instructions only")
  assert result is None
