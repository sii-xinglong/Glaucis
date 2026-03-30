# Deep Profiling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enhance kernel-evolve's profiling pipeline with HLO/LLO IR analysis, VLIW bundle counting, MXU utilization, HBM bandwidth estimation, and FLOP counting — feeding richer signals into both the LLM optimizer and human analysis.

**Architecture:** Add IR dump capture + parsing functions to `profiler.py` and `evaluate.py`. A single kernel invocation with XLA/LIBTPU dump flags generates HLO text, LLO text, and Mosaic IR. Parsers extract VLIW bundle count, MXU distribution, HBM bandwidth, and FLOPs from these dumps. Derived metrics (arithmetic intensity, compute efficiency) are calculated from the parsed data. The enhanced EVAL_RESULT flows into an upgraded analyze skill that does multi-signal bottleneck classification.

**Tech Stack:** Python 3.10+, JAX, xprof, pytest, regex for IR parsing

**Design doc:** `docs/plans/2026-03-30-deep-profiling-design.md`

---

### Task 1: Add IR text parsing functions to profiler.py

These are pure functions that parse IR text strings. No TPU or JAX dependency — fully testable locally.

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/profiler.py`
- Test: `kernel-evolve/tests/test_profiler.py`

**Step 1: Write failing tests for `count_vliw_bundles`**

Add to `kernel-evolve/tests/test_profiler.py`:

```python
from kernel_evolve.profiler import count_vliw_bundles


# Realistic LLO text fixture — bundles separated by ";;" in VLIW IR
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


def test_count_vliw_bundles():
  result = count_vliw_bundles(LLO_TEXT_FIXTURE)
  assert result == 4


def test_count_vliw_bundles_empty():
  assert count_vliw_bundles("") is None


def test_count_vliw_bundles_no_bundles():
  assert count_vliw_bundles("# just a comment\nsome text\n") is None
```

**Step 2: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_count_vliw_bundles tests/test_profiler.py::test_count_vliw_bundles_empty tests/test_profiler.py::test_count_vliw_bundles_no_bundles -v`
Expected: FAIL — `ImportError: cannot import name 'count_vliw_bundles'`

**Step 3: Implement `count_vliw_bundles`**

Add to `kernel-evolve/src/kernel_evolve/profiler.py`:

```python
import re

def count_vliw_bundles(llo_text: str) -> int | None:
  """Count VLIW bundles in LLO text IR.

  Bundles are separated by ';;' in LLO output. Returns None if no bundles found.
  """
  if not llo_text.strip():
    return None
  # Split on ';;' — each separator marks a bundle boundary
  bundles = re.split(r"^\s*;;\s*$", llo_text, flags=re.MULTILINE)
  # Filter out empty/comment-only segments
  real_bundles = []
  for b in bundles:
    lines = [l.strip() for l in b.strip().splitlines() if l.strip() and not l.strip().startswith("#")]
    if lines:
      real_bundles.append(lines)
  return len(real_bundles) if real_bundles else None
```

**Step 4: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_count_vliw_bundles tests/test_profiler.py::test_count_vliw_bundles_empty tests/test_profiler.py::test_count_vliw_bundles_no_bundles -v`
Expected: PASS

**Step 5: Write failing tests for `parse_mxu_distribution`**

Add to tests:

```python
from kernel_evolve.profiler import parse_mxu_distribution


def test_parse_mxu_distribution():
  result = parse_mxu_distribution(LLO_TEXT_FIXTURE)
  # LLO_TEXT_FIXTURE has: 1 mxu0 op (vmatprep), 1 mxu1 op (vmatpush1), 1 mxu0 op (vmatmul)
  assert result["mxu0"] == 2
  assert result["mxu1"] == 1
  assert result["dual_ratio"] == pytest.approx(1 / 2)  # min/max = 1/2


def test_parse_mxu_distribution_no_ops():
  text = "%v0 = vadd.f32 %r0, %r1\n;;\n"
  result = parse_mxu_distribution(text)
  assert result is None
```

**Step 6: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_parse_mxu_distribution tests/test_profiler.py::test_parse_mxu_distribution_no_ops -v`
Expected: FAIL

**Step 7: Implement `parse_mxu_distribution`**

```python
def parse_mxu_distribution(llo_text: str) -> dict[str, Any] | None:
  """Count MXU0 vs MXU1 operations in LLO text and compute dual-MXU utilization.

  Returns {"mxu0": int, "mxu1": int, "dual_ratio": float} or None.
  dual_ratio = min(mxu0, mxu1) / max(mxu0, mxu1). 1.0 = perfect dual-MXU use.
  """
  mxu0 = len(re.findall(r"\.mxu0\b", llo_text))
  mxu1 = len(re.findall(r"\.mxu1\b", llo_text))
  if mxu0 == 0 and mxu1 == 0:
    return None
  max_mxu = max(mxu0, mxu1)
  min_mxu = min(mxu0, mxu1)
  return {
    "mxu0": mxu0,
    "mxu1": mxu1,
    "dual_ratio": min_mxu / max_mxu if max_mxu > 0 else 0.0,
  }
```

**Step 8: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_parse_mxu_distribution tests/test_profiler.py::test_parse_mxu_distribution_no_ops -v`
Expected: PASS

**Step 9: Write failing tests for `estimate_hbm_bandwidth`**

```python
from kernel_evolve.profiler import estimate_hbm_bandwidth


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


def test_estimate_hbm_bandwidth():
  result = estimate_hbm_bandwidth(HLO_TEXT_FIXTURE)
  # 3 inputs bf16[8,2048,128] = 3 * 8*2048*128 * 2 bytes = 3 * 4MB = 12MB
  # 1 output bf16[8,2048,128] = 1 * 8*2048*128 * 2 bytes = 4MB
  # Total = 16MB = 16777216 bytes
  assert result == 16777216


def test_estimate_hbm_bandwidth_no_custom_call():
  hlo = "HloModule test\nENTRY main { ROOT %c = f32[] constant(0) }\n"
  result = estimate_hbm_bandwidth(hlo)
  assert result is None
```

**Step 10: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_estimate_hbm_bandwidth tests/test_profiler.py::test_estimate_hbm_bandwidth_no_custom_call -v`
Expected: FAIL

**Step 11: Implement `estimate_hbm_bandwidth`**

```python
_DTYPE_BYTES = {
  "pred": 1, "s8": 1, "u8": 1,
  "bf16": 2, "f16": 2, "s16": 2, "u16": 2,
  "f32": 4, "s32": 4, "u32": 4,
  "f64": 8, "s64": 8, "u64": 8,
}


def _shape_bytes(shape_str: str) -> int | None:
  """Parse 'bf16[8,2048,128]' into total bytes."""
  m = re.match(r"(\w+)\[([\d,]+)\]", shape_str.strip())
  if not m:
    return None
  dtype, dims = m.group(1), m.group(2)
  byte_size = _DTYPE_BYTES.get(dtype)
  if byte_size is None:
    return None
  total_elements = 1
  for d in dims.split(","):
    total_elements *= int(d)
  return total_elements * byte_size


def estimate_hbm_bandwidth(hlo_text: str) -> int | None:
  """Estimate HBM bytes read+written from HLO custom_call operand shapes.

  Parses the first tpu_custom_call in the HLO text, sums input and output bytes.
  Returns total bytes or None if no custom_call found.
  """
  # Find custom_call with tpu_custom_call target
  # Pattern: %name = <output_shape> custom-call(<input_shapes>), custom_call_target="tpu_custom_call"
  pattern = r"(%\S+)\s*=\s*(\S+)\s+custom-call\(([^)]+)\)[^\"]*custom_call_target=\"tpu_custom_call\""
  match = re.search(pattern, hlo_text)
  if not match:
    return None

  output_shape_str = match.group(2)
  inputs_str = match.group(3)

  total_bytes = 0

  # Parse output shape
  out_bytes = _shape_bytes(output_shape_str)
  if out_bytes:
    total_bytes += out_bytes

  # Parse input shapes (comma-separated, each is "dtype[dims] %name")
  for inp in re.findall(r"(\w+\[[\d,]+\])", inputs_str):
    inp_bytes = _shape_bytes(inp)
    if inp_bytes:
      total_bytes += inp_bytes

  return total_bytes if total_bytes > 0 else None
```

**Step 12: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_estimate_hbm_bandwidth tests/test_profiler.py::test_estimate_hbm_bandwidth_no_custom_call -v`
Expected: PASS

**Step 13: Write failing tests for `count_flops_from_hlo`**

```python
from kernel_evolve.profiler import count_flops_from_hlo


HLO_DOT_FIXTURE = """\
HloModule jit_reference

ENTRY main {
  %p0 = f32[512,256] parameter(0)
  %p1 = f32[256,512] parameter(1)
  %dot.1 = f32[512,512] dot(%p0, %p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT %tuple = (f32[512,512]) tuple(%dot.1)
}
"""


def test_count_flops_from_hlo():
  result = count_flops_from_hlo(HLO_DOT_FIXTURE)
  # dot(f32[512,256], f32[256,512]) → 2 * 512 * 256 * 512 = 134217728
  assert result == 2 * 512 * 256 * 512


def test_count_flops_from_hlo_no_dots():
  hlo = "HloModule test\nENTRY main { ROOT %c = f32[] constant(0) }\n"
  result = count_flops_from_hlo(hlo)
  assert result is None
```

**Step 14: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_count_flops_from_hlo tests/test_profiler.py::test_count_flops_from_hlo_no_dots -v`
Expected: FAIL

**Step 15: Implement `count_flops_from_hlo`**

```python
def count_flops_from_hlo(hlo_text: str) -> float | None:
  """Estimate total FLOPs from dot operations in HLO text.

  For dot(dtype[M,K], dtype[K,N]) → FLOPs = 2*M*K*N (multiply-accumulate).
  Sums across all dot operations found.
  """
  # Pattern: dot(dtype[dims] %name, dtype[dims] %name), lhs_contracting_dims={N}, rhs_contracting_dims={N}
  dot_pattern = r"dot\(\s*\w+\[([\d,]+)\]\s*%\S+\s*,\s*\w+\[([\d,]+)\]\s*%\S+\s*\)\s*,\s*lhs_contracting_dims=\{(\d+)\}"
  total_flops = 0.0
  for match in re.finditer(dot_pattern, hlo_text):
    lhs_dims = [int(d) for d in match.group(1).split(",")]
    rhs_dims = [int(d) for d in match.group(2).split(",")]
    contract_dim = int(match.group(3))
    # FLOPs = 2 * product(all output dims) * product(contracting dims)
    # For dot(M×K, K×N) with lhs_contracting={1}: output is M×N, K is contracted
    k = lhs_dims[contract_dim]
    output_dims = [d for i, d in enumerate(lhs_dims) if i != contract_dim]
    output_dims.extend([d for i, d in enumerate(rhs_dims) if d != k or i != 0])
    # Simpler: total elements = product(lhs) * product(rhs) / K
    # FLOPs for matmul: 2 * M * N * K
    total_elements = 1
    for d in lhs_dims:
      total_elements *= d
    for d in rhs_dims:
      total_elements *= d
    total_elements //= k  # divide by K (counted in both)
    total_flops += 2.0 * total_elements / k  # divide by K again (was counted twice)

  return total_flops if total_flops > 0 else None
```

Wait — let me simplify the FLOP calculation. For `dot(f32[M,K], f32[K,N])` with `lhs_contracting_dims={1}`:
- Output shape: [M, N]
- FLOPs = 2 * M * K * N

Let me rewrite:

```python
def count_flops_from_hlo(hlo_text: str) -> float | None:
  """Estimate total FLOPs from dot operations in HLO text.

  For dot(dtype[...M,K], dtype[K,N...]) → FLOPs = 2 * product(batch_dims) * M * K * N.
  Sums across all dot operations found. Returns None if no dots found.
  """
  # Match: %name = dtype[out_dims] dot(dtype[lhs_dims] %x, dtype[rhs_dims] %y), lhs_contracting_dims={d}
  dot_pattern = (
    r"\w+\[([\d,]+)\]\s+"  # output shape
    r"dot\(\s*\w+\[([\d,]+)\]\s+%\S+\s*,"  # lhs shape
    r"\s*\w+\[([\d,]+)\]\s+%\S+\s*\)"  # rhs shape
    r"\s*,\s*lhs_contracting_dims=\{([\d,]+)\}"  # contracting dims
  )
  total_flops = 0.0
  for match in re.finditer(dot_pattern, hlo_text):
    out_dims = [int(d) for d in match.group(1).split(",")]
    lhs_dims = [int(d) for d in match.group(2).split(",")]
    contract_indices = [int(d) for d in match.group(4).split(",")]
    # FLOPs = 2 * product(output_dims) * product(contracting_dims)
    out_size = 1
    for d in out_dims:
      out_size *= d
    contract_size = 1
    for idx in contract_indices:
      contract_size *= lhs_dims[idx]
    total_flops += 2.0 * out_size * contract_size

  return total_flops if total_flops > 0 else None
```

**Step 16: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_count_flops_from_hlo tests/test_profiler.py::test_count_flops_from_hlo_no_dots -v`
Expected: PASS

**Step 17: Write failing tests for `compute_derived_metrics`**

```python
from kernel_evolve.profiler import compute_derived_metrics


def test_compute_derived_metrics():
  result = compute_derived_metrics(
    flops=1.07e9,
    hbm_bytes=14680064,
    latency_ms=0.45,
    peak_flops_per_sec=275e12,  # TPU v7x BF16 peak ~275 TFLOPS
  )
  assert result["arithmetic_intensity"] == pytest.approx(1.07e9 / 14680064, rel=1e-3)
  assert result["compute_efficiency_pct"] > 0
  assert result["compute_efficiency_pct"] < 100


def test_compute_derived_metrics_missing_data():
  result = compute_derived_metrics(flops=None, hbm_bytes=None, latency_ms=0.45)
  assert result["arithmetic_intensity"] is None
  assert result["compute_efficiency_pct"] is None
```

**Step 18: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_compute_derived_metrics tests/test_profiler.py::test_compute_derived_metrics_missing_data -v`
Expected: FAIL

**Step 19: Implement `compute_derived_metrics`**

```python
def compute_derived_metrics(
  flops: float | None,
  hbm_bytes: int | None,
  latency_ms: float,
  peak_flops_per_sec: float = 275e12,  # TPU v7x BF16 peak
) -> dict[str, float | None]:
  """Compute arithmetic intensity and compute efficiency from raw metrics."""
  arith_intensity = None
  if flops is not None and hbm_bytes is not None and hbm_bytes > 0:
    arith_intensity = flops / hbm_bytes

  compute_eff = None
  if flops is not None and latency_ms > 0:
    actual_flops_per_sec = flops / (latency_ms / 1000.0)
    compute_eff = (actual_flops_per_sec / peak_flops_per_sec) * 100.0

  return {
    "arithmetic_intensity": arith_intensity,
    "compute_efficiency_pct": compute_eff,
  }
```

**Step 20: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_compute_derived_metrics tests/test_profiler.py::test_compute_derived_metrics_missing_data -v`
Expected: PASS

**Step 21: Run ALL profiler tests**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py -v`
Expected: ALL PASS (both old and new tests)

**Step 22: Commit**

```bash
git add kernel-evolve/src/kernel_evolve/profiler.py kernel-evolve/tests/test_profiler.py
git commit -m "feat(profiler): add IR parsing functions for VLIW bundles, MXU, HBM bandwidth, FLOPs"
```

---

### Task 2: Add IR dump capture + analysis to profiler.py

Add functions to set dump flags, run a kernel invocation, find dump files, and orchestrate the full deep profile analysis.

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/profiler.py`
- Test: `kernel-evolve/tests/test_profiler.py`

**Step 1: Write failing tests for `find_final_llo_file` and `find_hlo_file`**

```python
from kernel_evolve.profiler import find_final_llo_file, find_hlo_file


def test_find_final_llo_file(tmp_path):
  # Create mock LLO dump files with pass numbers
  (tmp_path / "module.pass_02.llo").write_text("early pass")
  (tmp_path / "module.pass_45.llo").write_text("middle pass")
  (tmp_path / "module.pass_79.llo").write_text("final pass content")
  result = find_final_llo_file(str(tmp_path))
  assert result is not None
  assert "pass_79" in result


def test_find_final_llo_file_empty_dir(tmp_path):
  assert find_final_llo_file(str(tmp_path)) is None


def test_find_hlo_file(tmp_path):
  (tmp_path / "module.after_all_optimizations.txt").write_text("HLO content")
  (tmp_path / "module.before_optimizations.txt").write_text("early HLO")
  result = find_hlo_file(str(tmp_path))
  assert result is not None


def test_find_hlo_file_empty_dir(tmp_path):
  assert find_hlo_file(str(tmp_path)) is None
```

**Step 2: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_find_final_llo_file tests/test_profiler.py::test_find_hlo_file -v`
Expected: FAIL

**Step 3: Implement `find_final_llo_file` and `find_hlo_file`**

```python
import glob


def find_final_llo_file(dump_dir: str) -> str | None:
  """Find the highest-numbered LLO pass file in a dump directory.

  LLO files are named like 'module.pass_NN.llo' or similar patterns.
  Returns the path to the final (highest pass number) file, or None.
  """
  llo_files = []
  for pattern in ["*.llo", "*.llo.txt"]:
    llo_files.extend(glob.glob(os.path.join(dump_dir, "**", pattern), recursive=True))
  if not llo_files:
    return None
  # Extract pass numbers and sort
  def pass_number(path: str) -> int:
    m = re.search(r"pass[_.]?(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1
  llo_files.sort(key=pass_number)
  return llo_files[-1] if pass_number(llo_files[-1]) >= 0 else llo_files[-1]


def find_hlo_file(dump_dir: str) -> str | None:
  """Find an HLO text file in the dump directory.

  Prefers files with 'after' or 'final' or 'optimizations' in the name.
  Returns the path or None.
  """
  hlo_files = glob.glob(os.path.join(dump_dir, "**", "*.txt"), recursive=True)
  hlo_files.extend(glob.glob(os.path.join(dump_dir, "**", "*.hlo"), recursive=True))
  if not hlo_files:
    return None
  # Prefer files with 'after' in name (post-optimization HLO)
  for f in hlo_files:
    base = os.path.basename(f).lower()
    if "after" in base and "optimiz" in base:
      return f
  # Fall back to any HLO file (pick the largest — likely the most complete)
  return max(hlo_files, key=os.path.getsize)
```

**Step 4: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_find_final_llo_file tests/test_profiler.py::test_find_hlo_file -v`
Expected: PASS

**Step 5: Write failing test for `analyze_ir_dumps`**

```python
from kernel_evolve.profiler import analyze_ir_dumps


def test_analyze_ir_dumps(tmp_path):
  # Set up mock dump directories
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
```

**Step 6: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_analyze_ir_dumps -v`
Expected: FAIL

**Step 7: Implement `analyze_ir_dumps`**

```python
def analyze_ir_dumps(hlo_dir: str, llo_dir: str) -> dict[str, Any]:
  """Analyze HLO and LLO dump files to extract performance metrics.

  Returns dict with vliw_bundle_count, mxu_utilization, hbm_bandwidth_bytes, flops.
  Each field is None if the corresponding data could not be parsed.
  """
  result: dict[str, Any] = {
    "vliw_bundle_count": None,
    "mxu_utilization": None,
    "hbm_bandwidth_bytes": None,
    "flops": None,
  }

  # Parse LLO dumps
  llo_path = find_final_llo_file(llo_dir)
  if llo_path:
    llo_text = Path(llo_path).read_text()
    result["vliw_bundle_count"] = count_vliw_bundles(llo_text)
    result["mxu_utilization"] = parse_mxu_distribution(llo_text)

  # Parse HLO dumps
  hlo_path = find_hlo_file(hlo_dir)
  if hlo_path:
    hlo_text = Path(hlo_path).read_text()
    result["hbm_bandwidth_bytes"] = estimate_hbm_bandwidth(hlo_text)
    result["flops"] = count_flops_from_hlo(hlo_text)

  return result
```

**Step 8: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_analyze_ir_dumps tests/test_profiler.py::test_analyze_ir_dumps_empty_dirs -v`
Expected: PASS

**Step 9: Write failing test for `capture_ir_dumps`**

This function requires JAX so we mock it:

```python
@patch.dict("os.environ", {}, clear=False)
def test_capture_ir_dumps_sets_env_and_runs(tmp_path):
  mock_kernel = MagicMock(return_value=MagicMock(block_until_ready=MagicMock()))
  dump_dir = str(tmp_path / "dumps")

  with patch.dict("sys.modules", {"jax": MagicMock()}):
    from kernel_evolve.profiler import capture_ir_dumps
    capture_ir_dumps(mock_kernel, {"M": 64}, dump_dir)

  # Verify kernel was called at least once
  assert mock_kernel.called
  # Verify env vars were set during capture
  # (They get cleaned up after, so we check the dump_dir was created)
  assert os.path.isdir(dump_dir)
```

**Step 10: Run test to verify it fails**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_capture_ir_dumps_sets_env_and_runs -v`
Expected: FAIL

**Step 11: Implement `capture_ir_dumps`**

```python
def capture_ir_dumps(
  kernel_fn,
  shapes: dict[str, Any],
  dump_dir: str,
) -> tuple[str, str]:
  """Run kernel once with XLA/LIBTPU dump flags to capture HLO and LLO IR.

  Returns (hlo_dir, llo_dir) paths. The caller should parse these with analyze_ir_dumps().
  Cleans up env vars after capture.
  """
  hlo_dir = os.path.join(dump_dir, "hlo")
  llo_dir = os.path.join(dump_dir, "llo")
  mosaic_dir = os.path.join(dump_dir, "mosaic")
  Path(hlo_dir).mkdir(parents=True, exist_ok=True)
  Path(llo_dir).mkdir(parents=True, exist_ok=True)
  Path(mosaic_dir).mkdir(parents=True, exist_ok=True)

  # Save original env
  orig_xla = os.environ.get("XLA_FLAGS", "")
  orig_libtpu = os.environ.get("LIBTPU_INIT_ARGS", "")

  try:
    os.environ["XLA_FLAGS"] = (
      f"--xla_dump_hlo_as_text --xla_dump_to={hlo_dir} "
      + orig_xla
    )
    os.environ["LIBTPU_INIT_ARGS"] = (
      f"--xla_jf_dump_to={llo_dir} "
      f"--xla_jf_dump_hlo_text=true "
      f"--xla_jf_dump_llo_text=true "
      f"--xla_jf_emit_annotations=true "
      f"--xla_mosaic_dump_to={mosaic_dir} "
      f"--xla_mosaic_enable_llo_source_annotations=true "
      + orig_libtpu
    )
    # Single invocation to trigger compilation with dumps
    out = kernel_fn(**shapes)
    if hasattr(out, "block_until_ready"):
      out.block_until_ready()
  finally:
    # Restore env
    os.environ["XLA_FLAGS"] = orig_xla
    os.environ["LIBTPU_INIT_ARGS"] = orig_libtpu

  return hlo_dir, llo_dir
```

**Step 12: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_capture_ir_dumps_sets_env_and_runs -v`
Expected: PASS

**Step 13: Run ALL profiler tests**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py -v`
Expected: ALL PASS

**Step 14: Commit**

```bash
git add kernel-evolve/src/kernel_evolve/profiler.py kernel-evolve/tests/test_profiler.py
git commit -m "feat(profiler): add IR dump capture and analysis (VLIW, MXU, HBM, FLOP)"
```

---

### Task 3: Integrate deep profiling into evaluate.py

Add `stage_profile_deep()` to the Docker evaluator. This is self-contained (no import from profiler.py) — a copy of the parsing logic for the containerized environment. Update `main()` to call it and merge results.

**Files:**
- Modify: `kernel-evolve/docker/evaluate.py`

**Step 1: Add `stage_profile_deep()` function**

Add after `stage_profile()` in `evaluate.py`:

```python
def stage_profile_deep(exec_globals, shapes, dump_dir="/tmp/ir_dumps"):
  """Stage 4b: Capture and analyze HLO/LLO IR dumps for deep profiling.

  Self-contained — includes all parsing logic inline.
  Non-fatal: returns ok=False on failure without stopping the pipeline.
  """
  try:
    import glob
    import re
    from pathlib import Path

    import jax  # noqa: F811 — reimport ok in self-contained function

    kernel_fn = exec_globals.get("optimized_compute") or exec_globals.get("kernel_fn")
    if kernel_fn is None:
      return {"ok": False, "error": "No kernel_fn found for IR dump profiling"}

    shape = shapes[0]
    hlo_dir = os.path.join(dump_dir, "hlo")
    llo_dir = os.path.join(dump_dir, "llo")
    mosaic_dir = os.path.join(dump_dir, "mosaic")
    Path(hlo_dir).mkdir(parents=True, exist_ok=True)
    Path(llo_dir).mkdir(parents=True, exist_ok=True)
    Path(mosaic_dir).mkdir(parents=True, exist_ok=True)

    orig_xla = os.environ.get("XLA_FLAGS", "")
    orig_libtpu = os.environ.get("LIBTPU_INIT_ARGS", "")

    try:
      os.environ["XLA_FLAGS"] = (
        f"--xla_dump_hlo_as_text --xla_dump_to={hlo_dir} " + orig_xla
      )
      os.environ["LIBTPU_INIT_ARGS"] = (
        f"--xla_jf_dump_to={llo_dir} "
        f"--xla_jf_dump_hlo_text=true "
        f"--xla_jf_dump_llo_text=true "
        f"--xla_jf_emit_annotations=true "
        f"--xla_mosaic_dump_to={mosaic_dir} "
        f"--xla_mosaic_enable_llo_source_annotations=true "
        + orig_libtpu
      )
      # Force recompilation by clearing JAX cache
      jax.clear_caches()
      out = kernel_fn(**shape)
      if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    finally:
      os.environ["XLA_FLAGS"] = orig_xla
      os.environ["LIBTPU_INIT_ARGS"] = orig_libtpu

    # --- Parse LLO dumps ---
    DTYPE_BYTES = {
      "pred": 1, "s8": 1, "u8": 1,
      "bf16": 2, "f16": 2, "s16": 2, "u16": 2,
      "f32": 4, "s32": 4, "u32": 4,
      "f64": 8, "s64": 8, "u64": 8,
    }

    vliw_bundle_count = None
    mxu_util = None

    llo_files = glob.glob(os.path.join(llo_dir, "**", "*.llo"), recursive=True)
    llo_files.extend(glob.glob(os.path.join(llo_dir, "**", "*.llo.txt"), recursive=True))
    if llo_files:
      def pass_num(p):
        m = re.search(r"pass[_.]?(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1
      llo_files.sort(key=pass_num)
      llo_text = Path(llo_files[-1]).read_text()

      # Count VLIW bundles (separated by ';;')
      bundles = re.split(r"^\s*;;\s*$", llo_text, flags=re.MULTILINE)
      real = [b for b in bundles if any(
        l.strip() and not l.strip().startswith("#")
        for l in b.strip().splitlines()
      )]
      if real:
        vliw_bundle_count = len(real)

      # MXU distribution
      mxu0 = len(re.findall(r"\.mxu0\b", llo_text))
      mxu1 = len(re.findall(r"\.mxu1\b", llo_text))
      if mxu0 > 0 or mxu1 > 0:
        max_m = max(mxu0, mxu1)
        mxu_util = {
          "mxu0": mxu0, "mxu1": mxu1,
          "dual_ratio": min(mxu0, mxu1) / max_m if max_m else 0.0,
        }

    # --- Parse HLO dumps ---
    hbm_bytes = None
    flops = None

    hlo_files = glob.glob(os.path.join(hlo_dir, "**", "*.txt"), recursive=True)
    hlo_files.extend(glob.glob(os.path.join(hlo_dir, "**", "*.hlo"), recursive=True))
    if hlo_files:
      # Prefer post-optimization HLO
      hlo_path = None
      for f in hlo_files:
        if "after" in os.path.basename(f).lower():
          hlo_path = f
          break
      if hlo_path is None:
        hlo_path = max(hlo_files, key=os.path.getsize)
      hlo_text = Path(hlo_path).read_text()

      # HBM bandwidth from custom_call shapes
      cc = re.search(
        r"(%\S+)\s*=\s*(\S+)\s+custom-call\(([^)]+)\)[^\"]*custom_call_target=\"tpu_custom_call\"",
        hlo_text,
      )
      if cc:
        total_b = 0
        out_m = re.match(r"(\w+)\[([\d,]+)\]", cc.group(2))
        if out_m:
          dtype_b = DTYPE_BYTES.get(out_m.group(1), 0)
          elems = 1
          for d in out_m.group(2).split(","):
            elems *= int(d)
          total_b += elems * dtype_b
        for inp in re.findall(r"(\w+\[[\d,]+\])", cc.group(3)):
          inp_m = re.match(r"(\w+)\[([\d,]+)\]", inp)
          if inp_m:
            dtype_b = DTYPE_BYTES.get(inp_m.group(1), 0)
            elems = 1
            for d in inp_m.group(2).split(","):
              elems *= int(d)
            total_b += elems * dtype_b
        if total_b > 0:
          hbm_bytes = total_b

      # FLOPs from dot operations
      dot_pat = (
        r"\w+\[([\d,]+)\]\s+dot\(\s*\w+\[([\d,]+)\]\s+%\S+\s*,"
        r"\s*\w+\[([\d,]+)\]\s+%\S+\s*\)\s*,\s*lhs_contracting_dims=\{([\d,]+)\}"
      )
      total_flops = 0.0
      for dm in re.finditer(dot_pat, hlo_text):
        out_dims = [int(d) for d in dm.group(1).split(",")]
        lhs_dims = [int(d) for d in dm.group(2).split(",")]
        ci = [int(d) for d in dm.group(4).split(",")]
        out_sz = 1
        for d in out_dims:
          out_sz *= d
        k_sz = 1
        for i in ci:
          k_sz *= lhs_dims[i]
        total_flops += 2.0 * out_sz * k_sz
      if total_flops > 0:
        flops = total_flops

    # --- Derived metrics ---
    arith_intensity = None
    if flops and hbm_bytes and hbm_bytes > 0:
      arith_intensity = flops / hbm_bytes

    return {
      "ok": True,
      "vliw_bundle_count": vliw_bundle_count,
      "mxu_utilization": mxu_util,
      "hbm_bandwidth_bytes": hbm_bytes,
      "flops": flops,
      "arithmetic_intensity": arith_intensity,
    }
  except Exception:
    return {"ok": False, "error": f"Deep profile error: {traceback.format_exc()}"}
```

**Step 2: Update `main()` to add xprof custom call flags and call `stage_profile_deep()`**

In `main()`, before the existing `stage_profile()` call, set the enhanced profiling flags. After `stage_profile()`, call `stage_profile_deep()`. Merge results into the final output.

Replace the profile section in `main()` (lines 318-337) with:

```python
  # Stage 4a: XPlane trace profile (enhanced with custom call region tracing)
  orig_libtpu = os.environ.get("LIBTPU_INIT_ARGS", "")
  os.environ["LIBTPU_INIT_ARGS"] = (
    "--xla_enable_custom_call_region_trace=true "
    "--xla_xprof_register_llo_debug_info=true "
    + orig_libtpu
  )
  profile_result = stage_profile(compile_result["globals"], request["shapes"])
  os.environ["LIBTPU_INIT_ARGS"] = orig_libtpu

  compute_ratio = None
  memory_transfer_ratio = None
  profile_diag = {}
  if profile_result["ok"]:
    compute_ratio = profile_result["compute_ratio"]
    memory_transfer_ratio = profile_result["memory_transfer_ratio"]
    profile_diag = profile_result.get("diagnostics", {})
    print(
      f"Profile: compute_ratio={compute_ratio}, "
      f"memory_transfer_ratio={memory_transfer_ratio}",
      file=sys.stderr,
    )
  else:
    profile_diag = {"error": profile_result.get("error", "unknown")}
    print(f"Profile skipped: {profile_result.get('error', 'unknown')}", file=sys.stderr)

  # Stage 4b: Deep profile — IR dump analysis (non-fatal)
  deep_profile = {}
  deep_result = stage_profile_deep(compile_result["globals"], request["shapes"])
  if deep_result["ok"]:
    deep_profile = {
      "vliw_bundle_count": deep_result.get("vliw_bundle_count"),
      "mxu_utilization": deep_result.get("mxu_utilization"),
      "hbm_bandwidth_bytes": deep_result.get("hbm_bandwidth_bytes"),
      "flops": deep_result.get("flops"),
      "arithmetic_intensity": deep_result.get("arithmetic_intensity"),
    }
    print(f"Deep profile: {json.dumps(deep_profile, default=str)}", file=sys.stderr)
  else:
    deep_profile = {"error": deep_result.get("error", "unknown")}
    print(f"Deep profile skipped: {deep_result.get('error', 'unknown')}", file=sys.stderr)

  # Stage 4c: Compute efficiency (if we have FLOPS and latency)
  compute_efficiency_pct = None
  if deep_profile.get("flops") and perf_result["latency_ms"] > 0:
    actual_fps = deep_profile["flops"] / (perf_result["latency_ms"] / 1000.0)
    compute_efficiency_pct = (actual_fps / 275e12) * 100.0  # TPU v7x BF16 peak
    deep_profile["compute_efficiency_pct"] = compute_efficiency_pct
```

**Step 3: Update the result dict in `main()` to include deep profile data**

Replace the result dict (lines 339-353) with:

```python
  result = {
    "status": "SUCCESS",
    "fitness": speedup,
    "latency_ms": perf_result["latency_ms"],
    "speedup": speedup,
    "flops": deep_profile.get("flops", 0.0) or 0.0,
    "compute_ratio": compute_ratio,
    "memory_transfer_ratio": memory_transfer_ratio,
    "metadata": {
      "reference_latency_ms": ref_latency,
      "reference_perf_ok": ref_perf.get("ok", False),
      "profile_diagnostics": profile_diag,
      "profile": deep_profile,
    },
  }
  print(f"EVAL_RESULT:{json.dumps(result)}")
```

**Step 4: Commit**

```bash
git add kernel-evolve/docker/evaluate.py
git commit -m "feat(eval): add stage_profile_deep for HLO/LLO IR dump analysis"
```

---

### Task 4: Update K8s Job template with profiling env vars

Add `LIBTPU_INIT_ARGS` to the K8s Job template so xprof custom call tracing flags are always available.

**Files:**
- Modify: `.github/ci/kernel-eval-job.yaml`

**Step 1: Add LIBTPU_INIT_ARGS env var**

Add after the `EVAL_PAYLOAD` env var block in `kernel-eval-job.yaml`:

```yaml
        - name: LIBTPU_INIT_ARGS
          value: "--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"
```

**Step 2: Commit**

```bash
git add .github/ci/kernel-eval-job.yaml
git commit -m "feat(ci): add LIBTPU_INIT_ARGS for xprof custom call profiling"
```

---

### Task 5: Enhance analyze skill with multi-signal analysis

Replace the simple compute_ratio-based bottleneck classification with multi-signal analysis using all the new profiling data.

**Files:**
- Modify: `kernel-evolve/plugins/pallas-evolve/skills/analyze/SKILL.md`

**Step 1: Update eval_result.json structure docs**

Replace the JSON example in Step 1 with the extended structure:

```json
{
  "status": "SUCCESS|COMPILE_ERROR|INCORRECT",
  "fitness": 1.5,
  "error": "error message if failed",
  "max_diff": 0.0,
  "latency_ms": 2.3,
  "speedup": 1.5,
  "flops": 1.07e9,
  "compute_ratio": 0.85,
  "memory_transfer_ratio": 0.15,
  "metadata": {
    "reference_latency_ms": 3.45,
    "profile": {
      "vliw_bundle_count": 4302,
      "mxu_utilization": {"mxu0": 1396, "mxu1": 1383, "dual_ratio": 0.99},
      "hbm_bandwidth_bytes": 14680064,
      "arithmetic_intensity": 72.8,
      "flops": 1.07e9,
      "compute_efficiency_pct": 45.2
    }
  }
}
```

**Step 2: Replace Step 3 (Performance analysis) with multi-signal analysis**

Replace the existing Step 3 content with:

```markdown
### Step 3: Performance analysis (SUCCESS only)

Extract primary metrics:
- `speedup`: ratio of reference_latency / kernel_latency. >1.0 means faster than baseline.
- `latency_ms`: absolute kernel latency
- `compute_ratio`: fraction of time in useful computation (0.0-1.0). Higher is better.
- `memory_transfer_ratio`: fraction of time in SyncWait/DMA (0.0-1.0). Lower is better.

Extract deep profiling metrics from `metadata.profile` (may be absent if profiling failed):
- `vliw_bundle_count`: total VLIW bundles in the compiled kernel. Lower = simpler/faster.
- `mxu_utilization.dual_ratio`: how evenly both MXUs are used (0.0-1.0). 1.0 = perfect.
- `hbm_bandwidth_bytes`: total HBM bytes read + written per invocation.
- `arithmetic_intensity`: FLOPs per byte of HBM traffic. Higher = more compute per byte.
- `compute_efficiency_pct`: actual FLOPS / peak FLOPS as percentage.

**Multi-signal bottleneck classification:**

| Signal | Threshold | Diagnosis |
|--------|-----------|-----------|
| `compute_ratio < 0.50` | — | Memory-bound: VPU stalling on DMA/SyncWait |
| `compute_ratio >= 0.75` | — | Compute-bound: VPU busy, optimize ALU ops |
| `arithmetic_intensity < 10` | — | Low arithmetic intensity: too much HBM traffic per FLOP |
| `dual_ratio < 0.5` | — | Single-MXU: only one MXU active, matmul dims may be wrong |
| `compute_efficiency_pct < 10` | — | Very far from peak: major pipeline stalls or register spills |
| `vliw_bundle_count increasing` | vs prior iteration | Complexity bloat: kernel getting more complex without speedup |

**Combined diagnosis patterns:**
- **Low arithmetic_intensity + high compute_ratio** → Compute-bound but under-utilizing memory bandwidth. Consider larger tiles or prefetching.
- **Low dual_ratio + compute-bound** → Only one MXU active. Check if matmul dimensions are multiples of 128 for dual-MXU scheduling.
- **Growing vliw_bundle_count + flat speedup** → Kernel complexity bloat. Simplify the algorithm.
- **Low compute_efficiency_pct + high compute_ratio** → Near-peak VPU util but far from peak FLOPS. Check for unnecessary recomputation or register pressure.
```

**Step 3: Update Step 4 (Trend analysis) to track new metrics**

Replace the existing Step 4 content with:

```markdown
### Step 4: Trend analysis

Read previous iteration results (if any) from `iteration_{N-1}/eval_result.json`, `iteration_{N-2}/eval_result.json`, etc.

Track these metrics across iterations:

| Metric | Good trend | Bad trend |
|--------|-----------|-----------|
| `speedup` | Increasing | Decreasing (regression) |
| `compute_ratio` | Increasing | Decreasing |
| `vliw_bundle_count` | Decreasing or stable | Increasing (complexity bloat) |
| `mxu_utilization.dual_ratio` | Increasing toward 1.0 | Decreasing |
| `arithmetic_intensity` | Increasing | Decreasing (more memory traffic) |
| `compute_efficiency_pct` | Increasing | Decreasing |

Flag any regressions (speedup decreased from previous iteration).

Detect concerning patterns:
- **"All-cost improvement"**: speedup improved but bundle count doubled — likely unsustainable
- **"Diminishing returns"**: each iteration yields <2% improvement — consider trying a different approach
- **"MXU regression"**: dual_ratio dropped after a code change — the change broke MXU scheduling
```

**Step 4: Update Step 5 (Optimization suggestions) with new signals**

Replace the existing Step 5 content with:

```markdown
### Step 5: Generate optimization suggestions

Based on the multi-signal analysis, suggest next steps:

**Memory-bound (compute_ratio < 0.50):**
- Increase block size to process more data per tile
- Add K-tiling to reduce HBM round-trips
- Use scratch memory for accumulators
- Add pipelining to overlap compute and memory

**Compute-bound (compute_ratio >= 0.75):**
- If `dual_ratio < 0.5`: ensure matmul dimensions allow dual-MXU scheduling (multiples of 128)
- If `compute_efficiency_pct < 20`: look for unnecessary recomputation or register pressure
- Try different block aspect ratios
- Consider algorithmic improvements

**Low arithmetic intensity (< 10 FLOPs/byte):**
- Reduce HBM traffic: keep more data in VMEM via larger tiles
- Eliminate redundant reads/writes
- Use scratch memory to avoid round-trips to HBM

**Complexity bloat (vliw_bundle_count increasing without speedup gain):**
- Revert to simpler kernel structure
- Remove unnecessary conditionals or branching
- Simplify accumulator logic

**Balanced (0.50 - 0.75):**
- Try pipelining to overlap compute and memory
- Adjust block sizes to find the sweet spot
- Profile deeper: which operations are slow?

**Regression detected:**
- Compare the current kernel with the previous best
- Identify what changed and why it was slower
- Suggest reverting the specific change that caused regression
```

**Step 5: Update Step 6 (Write analysis) template**

Replace the analysis.md template with:

```markdown
### Step 6: Write analysis

Write `iteration_{N}/analysis.md`:

\`\`\`markdown
## Iteration {N} Analysis

**Status**: {SUCCESS/COMPILE_ERROR/INCORRECT}
**Speedup**: {speedup}x (best so far: {best_speedup}x)
**Latency**: {latency_ms}ms

### Performance Profile
| Metric | Value | Assessment |
|--------|-------|------------|
| compute_ratio | {compute_ratio} | {memory-bound/balanced/compute-bound} |
| vliw_bundle_count | {vliw_bundle_count} | {vs previous: +/-N} |
| MXU dual_ratio | {dual_ratio} | {poor/fair/good/excellent} |
| arithmetic_intensity | {arithmetic_intensity} | {low/medium/high} |
| compute_efficiency | {compute_efficiency_pct}% | {vs peak FLOPS} |
| HBM bandwidth | {hbm_bandwidth_bytes} bytes | {comparison to optimal} |

### Bottleneck
{Multi-signal diagnosis — primary and secondary bottlenecks}

### Trend
{Comparison with previous iterations across all metrics}

### Suggestions
{Specific optimization suggestions based on multi-signal analysis}
\`\`\`
```

**Step 6: Commit**

```bash
git add kernel-evolve/plugins/pallas-evolve/skills/analyze/SKILL.md
git commit -m "feat(analyze): upgrade to multi-signal bottleneck analysis with deep profiling"
```

---

### Task 6: Update start skill with deep profiling knowledge

Add the new profiling signals to the TPU v7x optimization knowledge section.

**Files:**
- Modify: `kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md`

**Step 1: Add deep profiling knowledge to the optimization knowledge section**

After the existing "Common pitfalls:" section at the end of `start/SKILL.md`, add:

```markdown

**Deep profiling signals (from eval_result.json → metadata.profile):**
- `vliw_bundle_count`: Total compiled VLIW bundles. Fewer bundles = simpler kernel = faster. Compare across iterations to detect complexity bloat.
- `mxu_utilization.dual_ratio`: How evenly both MXUs (matrix units) are used. 1.0 = both equally loaded. <0.5 means one MXU is idle — check matmul dimensions.
- `hbm_bandwidth_bytes`: Total HBM memory traffic per invocation. Lower = better. Pallas should keep data in VMEM to avoid HBM round-trips.
- `arithmetic_intensity` (FLOPs/byte): Higher means more compute per byte of memory traffic. Low values indicate memory-bound behavior.
- `compute_efficiency_pct`: Actual throughput vs TPU v7x peak (275 TFLOPS BF16). Shows headroom for optimization.

**When analyzing iteration results, check all signals — not just speedup and compute_ratio. VLIW bundle count and MXU dual_ratio are leading indicators of kernel quality.**
```

**Step 2: Commit**

```bash
git add kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md
git commit -m "feat(start): add deep profiling signals to TPU optimization knowledge"
```

---

### Task 7: Update EvalResult dataclass (optional — backwards compatible)

The existing `EvalResult.metadata` dict already carries arbitrary data, so the deep profile data flows through without dataclass changes. However, we should update the docstring and the `flops` field to reflect that it's now populated.

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/evaluator.py`

**Step 1: Update the docstring**

Change the module docstring from:
```python
"""Three-stage evaluation pipeline: compile -> correctness -> performance."""
```
to:
```python
"""Four-stage evaluation pipeline: compile -> correctness -> performance -> profile."""
```

**Step 2: Commit**

```bash
git add kernel-evolve/src/kernel_evolve/evaluator.py
git commit -m "docs(evaluator): update docstring for four-stage pipeline"
```

---

### Verification

After all tasks are complete, run the full test suite:

```bash
cd kernel-evolve && python -m pytest tests/ -v
```

Expected: ALL PASS

Then lint:

```bash
cd kernel-evolve && ruff check src/ tests/ docker/
```

Expected: No errors
