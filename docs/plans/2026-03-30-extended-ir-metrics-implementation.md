# Extended IR Metrics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 6 new IR analysis metrics (VMEM allocation, bundle density, DMA analysis, HLO fusion count, special hardware units, HBM bandwidth utilization) to both `profiler.py` and `evaluate.py`, with full TDD and analyze skill updates.

**Architecture:** Each metric is a standalone parsing function in `profiler.py` (testable library) with a self-contained equivalent inlined in `evaluate.py`'s `stage_profile_deep()`. All new fields are additive (None when unavailable). Tests use string fixtures — no TPU/JAX required.

**Tech Stack:** Python, regex, pytest. No new dependencies.

**Reference:** [Design doc](2026-03-30-extended-ir-metrics-design.md)

---

### Task 1: VMEM Allocation Tracking (profiler.py)

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/profiler.py:234` (after `parse_mxu_distribution`)
- Modify: `kernel-evolve/tests/test_profiler.py`

**Step 1: Add LLO fixture with allocation entries to test_profiler.py**

After the existing `LLO_TEXT_FIXTURE` (line 147), add a new fixture:

```python
LLO_ALLOC_FIXTURE = """\
# Allocations:
#allocation0 = u8[524288], size=0x80000
#allocation1 = f32[262144], size=0x100000
#allocation3 = u8[512], space=smem, size=0x200
#allocation4 = u8[512], space=smem, size=0x200
;;
%v0 = vmatmul.mubr.bf16.gmra.mxu0 %r0
;;
"""
```

**Step 2: Write failing tests for `parse_vmem_allocations`**

```python
from kernel_evolve.profiler import (
    # ... existing imports ...
    parse_vmem_allocations,
)


# Tests: parse_vmem_allocations
def test_parse_vmem_allocations():
    result = parse_vmem_allocations(LLO_ALLOC_FIXTURE)
    assert result is not None
    assert result["vmem_bytes"] == 0x80000 + 0x100000  # 524288 + 1048576
    assert result["smem_bytes"] == 0x200 + 0x200  # 512 + 512
    assert result["allocation_count"] == 4


def test_parse_vmem_allocations_no_allocs():
    result = parse_vmem_allocations("no allocations here\n;;\n")
    assert result is None
```

**Step 3: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_parse_vmem_allocations tests/test_profiler.py::test_parse_vmem_allocations_no_allocs -v`
Expected: FAIL — `ImportError: cannot import name 'parse_vmem_allocations'`

**Step 4: Implement `parse_vmem_allocations` in profiler.py**

Insert after `parse_mxu_distribution` (after line 234):

```python
def parse_vmem_allocations(llo_text: str) -> dict | None:
  """Parse memory allocations from LLO text.

  Matches lines like: #allocationN = type[dims], size=0xHEX
  Optional: space=smem tag separates SMEM from VMEM.
  """
  alloc_re = re.compile(
    r"#allocation\d+\s*=\s*\w+\[[^\]]*\]"
    r"(?:,\s*space=(\w+))?"
    r",\s*size=0x([0-9a-fA-F]+)"
  )
  vmem_bytes = 0
  smem_bytes = 0
  count = 0
  for m in alloc_re.finditer(llo_text):
    space = m.group(1)
    size = int(m.group(2), 16)
    if space == "smem":
      smem_bytes += size
    else:
      vmem_bytes += size
    count += 1
  if count == 0:
    return None
  return {
    "vmem_bytes": vmem_bytes,
    "smem_bytes": smem_bytes,
    "allocation_count": count,
  }
```

**Step 5: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_parse_vmem_allocations tests/test_profiler.py::test_parse_vmem_allocations_no_allocs -v`
Expected: PASS

**Step 6: Commit**

```bash
cd kernel-evolve && git add src/kernel_evolve/profiler.py tests/test_profiler.py
git commit -m "feat(profiler): add VMEM allocation parsing from LLO"
```

---

### Task 2: Bundle Density / ILP (profiler.py)

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/profiler.py` (after `parse_vmem_allocations`)
- Modify: `kernel-evolve/tests/test_profiler.py`

**Step 1: Write failing tests for `analyze_bundle_density`**

```python
from kernel_evolve.profiler import (
    # ... existing imports ...
    analyze_bundle_density,
)


def test_analyze_bundle_density():
    # LLO_TEXT_FIXTURE has 4 ";;" separators creating 4 bundles
    # Bundle 0 (before first ;;): 2 ops (%s0, %s1)
    # Bundle 1: 3 ops (%v0, %v1, %v2)
    # Bundle 2: 2 ops (%v3, %v4)
    # Bundle 3: 2 ops (%v5, %v6)
    result = analyze_bundle_density(LLO_TEXT_FIXTURE)
    assert result is not None
    assert result["total_bundles"] == 4
    assert result["max_ops_per_bundle"] == 3
    assert result["avg_ops_per_bundle"] == pytest.approx(2.25)


def test_analyze_bundle_density_empty():
    result = analyze_bundle_density("")
    assert result is None


def test_analyze_bundle_density_single_bundle():
    llo = "%v0 = vmatmul.mubr.bf16.gmra.mxu0 %r0\n;;\n"
    result = analyze_bundle_density(llo)
    assert result is not None
    assert result["total_bundles"] == 1
    assert result["max_ops_per_bundle"] == 1
    assert result["avg_ops_per_bundle"] == 1.0
```

**Step 2: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_analyze_bundle_density tests/test_profiler.py::test_analyze_bundle_density_empty tests/test_profiler.py::test_analyze_bundle_density_single_bundle -v`
Expected: FAIL

**Step 3: Implement `analyze_bundle_density`**

```python
def analyze_bundle_density(llo_text: str) -> dict | None:
  """Analyze instruction-level parallelism per VLIW bundle.

  Splits LLO text by ';;' and counts instruction lines (lines with '=')
  in each bundle segment.
  """
  if not llo_text:
    return None
  segments = llo_text.split(";;")
  if len(segments) <= 1:
    return None
  # Last segment after final ;; is usually empty or epilogue — skip it
  bundles = segments[:-1]
  op_re = re.compile(r"^\s*%\w+\s*=", re.MULTILINE)
  ops_per_bundle = []
  for seg in bundles:
    count = len(op_re.findall(seg))
    if count > 0:
      ops_per_bundle.append(count)
  if not ops_per_bundle:
    return None
  return {
    "total_bundles": len(ops_per_bundle),
    "avg_ops_per_bundle": sum(ops_per_bundle) / len(ops_per_bundle),
    "max_ops_per_bundle": max(ops_per_bundle),
  }
```

**Step 4: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_analyze_bundle_density tests/test_profiler.py::test_analyze_bundle_density_empty tests/test_profiler.py::test_analyze_bundle_density_single_bundle -v`
Expected: PASS

**Step 5: Commit**

```bash
cd kernel-evolve && git add src/kernel_evolve/profiler.py tests/test_profiler.py
git commit -m "feat(profiler): add VLIW bundle density / ILP analysis"
```

---

### Task 3: DMA Analysis (profiler.py)

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/profiler.py`
- Modify: `kernel-evolve/tests/test_profiler.py`

**Step 1: Write failing tests for `analyze_dma_ops`**

```python
from kernel_evolve.profiler import (
    # ... existing imports ...
    analyze_dma_ops,
)


def test_analyze_dma_ops():
    result = analyze_dma_ops(LLO_TEXT_FIXTURE)
    assert result is not None
    # LLO_TEXT_FIXTURE has 1 dma.hbm_to_vmem op
    assert result["dma_count"] == 1
    assert result["dma_sync_count"] == 0
    assert result["double_buffering"] is False


def test_analyze_dma_ops_with_double_buffering():
    llo = """\
%v0 = dma.hbm_to_vmem %r0
;;
%s1 = sand.u32 1, %s0
%v1 = dma.vmem_to_hbm %r1
dma.done.wait %flag0
;;
%v2 = dma.hbm_to_vmem %r2
dma.done.wait %flag1
;;
"""
    result = analyze_dma_ops(llo)
    assert result is not None
    assert result["dma_count"] == 3
    assert result["dma_sync_count"] == 2
    assert result["double_buffering"] is True


def test_analyze_dma_ops_no_dma():
    result = analyze_dma_ops("no dma here\n;;\n")
    assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_analyze_dma_ops tests/test_profiler.py::test_analyze_dma_ops_with_double_buffering tests/test_profiler.py::test_analyze_dma_ops_no_dma -v`
Expected: FAIL

**Step 3: Implement `analyze_dma_ops`**

```python
def analyze_dma_ops(llo_text: str) -> dict | None:
  """Analyze DMA operations and detect double buffering in LLO text.

  Counts dma.* operations, dma.done.wait sync points, and detects
  the double-buffering pattern (sand.u32 1 = iteration % 2 for buffer slot).
  """
  if not llo_text:
    return None
  dma_count = len(re.findall(r"\bdma\.\w+", llo_text))
  if dma_count == 0:
    return None
  dma_sync_count = len(re.findall(r"\bdma\.done\.wait\b", llo_text))
  # Double buffering: sand.u32 1 computes slot = iteration % 2
  double_buffering = bool(re.search(r"\bsand\.u32\s+1\b", llo_text))
  return {
    "dma_count": dma_count,
    "dma_sync_count": dma_sync_count,
    "double_buffering": double_buffering,
  }
```

**Step 4: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_analyze_dma_ops tests/test_profiler.py::test_analyze_dma_ops_with_double_buffering tests/test_profiler.py::test_analyze_dma_ops_no_dma -v`
Expected: PASS

**Step 5: Commit**

```bash
cd kernel-evolve && git add src/kernel_evolve/profiler.py tests/test_profiler.py
git commit -m "feat(profiler): add DMA analysis with double-buffering detection"
```

---

### Task 4: HLO Fusion Count (profiler.py)

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/profiler.py`
- Modify: `kernel-evolve/tests/test_profiler.py`

**Step 1: Add HLO fixture with fusions to test_profiler.py**

```python
HLO_FUSION_FIXTURE = """\
HloModule jit_attention

fused_computation.1 {
  %p0 = f32[8,2048,2048] parameter(0)
  %p1 = f32[8,2048] parameter(1)
  ROOT %sub = f32[8,2048,2048] subtract(%p0, %p1)
}

fused_computation.2 {
  %p0 = f32[8,2048,2048] parameter(0)
  ROOT %exp = f32[8,2048,2048] exponential(%p0)
}

fused_computation {
  %p0 = f32[8,2048,128] parameter(0)
  ROOT %out = f32[8,2048,128] multiply(%p0, %p0)
}

ENTRY main {
  %p0 = bf16[8,2048,128] parameter(0)
  %copy-start = bf16[8,2048,128] copy-start(%p0), cross_program_prefetch_index=0
  ROOT %out = bf16[8,2048,128] custom-call(%p0), custom_call_target="tpu_custom_call"
}
"""
```

**Step 2: Write failing tests**

```python
from kernel_evolve.profiler import (
    # ... existing imports ...
    count_hlo_fusions,
)


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
```

**Step 3: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_count_hlo_fusions tests/test_profiler.py::test_count_hlo_fusions_no_fusions tests/test_profiler.py::test_count_hlo_fusions_empty -v`
Expected: FAIL

**Step 4: Implement `count_hlo_fusions`**

```python
def count_hlo_fusions(hlo_text: str) -> dict | None:
  """Count fused_computation blocks and detect cross-program prefetch in HLO.

  More fusions = more HBM round-trips between them. Pallas kernels
  typically have 0 fusions (everything inside tpu_custom_call).
  """
  if not hlo_text:
    return None
  fusion_count = len(re.findall(r"\bfused_computation\b", hlo_text))
  has_prefetch = bool(re.search(r"cross_program_prefetch_index=", hlo_text))
  return {
    "fusion_count": fusion_count,
    "has_cross_program_prefetch": has_prefetch,
  }
```

**Step 5: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_count_hlo_fusions tests/test_profiler.py::test_count_hlo_fusions_no_fusions tests/test_profiler.py::test_count_hlo_fusions_empty -v`
Expected: PASS

**Step 6: Commit**

```bash
cd kernel-evolve && git add src/kernel_evolve/profiler.py tests/test_profiler.py
git commit -m "feat(profiler): add HLO fusion count and cross-program prefetch detection"
```

---

### Task 5: Special Hardware Unit Usage (profiler.py)

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/profiler.py`
- Modify: `kernel-evolve/tests/test_profiler.py`

**Step 1: Write failing tests for `analyze_special_units`**

```python
from kernel_evolve.profiler import (
    # ... existing imports ...
    analyze_special_units,
)


def test_analyze_special_units():
    # LLO_TEXT_FIXTURE has: 1 xlane op (vmax.xlane), 1 eup op (vpow2), 0 nops
    result = analyze_special_units(LLO_TEXT_FIXTURE)
    assert result is not None
    assert result["xlane_ops"] == 1
    assert result["eup_ops"] == 1
    assert result["nop_count"] == 0


def test_analyze_special_units_with_nops():
    llo = """\
%v0 = vmax.xlane.f32.xlu0 %r0
;;
nop
nop
nop
;;
%v1 = vpow2.f32 %r1
%v2 = vpop.eup %e0
%v3 = vpop.xlane.xlu0 %r2
;;
"""
    result = analyze_special_units(llo)
    assert result is not None
    assert result["xlane_ops"] == 2  # vmax.xlane + vpop.xlane
    assert result["eup_ops"] == 2   # vpow2 + vpop.eup
    assert result["nop_count"] == 3


def test_analyze_special_units_no_special():
    result = analyze_special_units("%v0 = vadd.f32 %r0, %r1\n;;\n")
    assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_analyze_special_units tests/test_profiler.py::test_analyze_special_units_with_nops tests/test_profiler.py::test_analyze_special_units_no_special -v`
Expected: FAIL

**Step 3: Implement `analyze_special_units`**

```python
def analyze_special_units(llo_text: str) -> dict | None:
  """Count cross-lane (XLU), exponent (EUP), and nop operations in LLO.

  - XLU: vmax.xlane, vpop.xlane — cross-lane reductions
  - EUP: vpow2, vpop.eup — hardware exponential unit
  - nop: pipeline bubbles where compiler couldn't schedule useful work
  """
  if not llo_text:
    return None
  xlane_ops = len(re.findall(r"\b\w+\.xlane\b", llo_text))
  eup_ops = len(re.findall(r"\bvpow2\b", llo_text)) + len(re.findall(r"\bvpop\.eup\b", llo_text))
  nop_count = len(re.findall(r"^\s*nop\b", llo_text, re.MULTILINE))
  if xlane_ops == 0 and eup_ops == 0 and nop_count == 0:
    return None
  return {
    "xlane_ops": xlane_ops,
    "eup_ops": eup_ops,
    "nop_count": nop_count,
  }
```

**Step 4: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_analyze_special_units tests/test_profiler.py::test_analyze_special_units_with_nops tests/test_profiler.py::test_analyze_special_units_no_special -v`
Expected: PASS

**Step 5: Commit**

```bash
cd kernel-evolve && git add src/kernel_evolve/profiler.py tests/test_profiler.py
git commit -m "feat(profiler): add XLU/EUP/nop special hardware unit analysis"
```

---

### Task 6: HBM Bandwidth Utilization + update compute_derived_metrics (profiler.py)

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/profiler.py:320-343` (`compute_derived_metrics`)
- Modify: `kernel-evolve/tests/test_profiler.py`

**Step 1: Write failing test for `hbm_bandwidth_utilization_pct`**

```python
def test_compute_derived_metrics_hbm_bw_util():
    flops = 134217728.0
    hbm_bytes = 16777216  # 16 MB
    latency_ms = 0.5      # 0.5 ms
    result = compute_derived_metrics(flops, hbm_bytes, latency_ms)
    # actual_bw = 16777216 / 0.0005 = 33554432000 B/s = ~33.5 GB/s
    # hbm_bw_util_pct = 33554432000 / 3690e9 * 100 = ~0.91%
    assert result["hbm_bandwidth_utilization_pct"] is not None
    assert result["hbm_bandwidth_utilization_pct"] == pytest.approx(
        (16777216 / 0.0005) / 3690e9 * 100.0
    )


def test_compute_derived_metrics_hbm_bw_util_missing():
    result = compute_derived_metrics(None, None, 0.5)
    assert result["hbm_bandwidth_utilization_pct"] is None
```

**Step 2: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_compute_derived_metrics_hbm_bw_util tests/test_profiler.py::test_compute_derived_metrics_hbm_bw_util_missing -v`
Expected: FAIL — `KeyError: 'hbm_bandwidth_utilization_pct'`

**Step 3: Update `compute_derived_metrics` in profiler.py**

Replace the function at line 320-343:

```python
def compute_derived_metrics(
  flops: float | None,
  hbm_bytes: int | None,
  latency_ms: float,
  peak_flops_per_sec: float = 2307e12,
  peak_hbm_bw_bytes_per_sec: float = 3690e9,
) -> dict:
  """Compute arithmetic intensity, compute efficiency, and HBM BW utilization.

  Returns dict with arithmetic_intensity, compute_efficiency_pct,
  and hbm_bandwidth_utilization_pct — each None when input data is missing.
  """
  arithmetic_intensity = None
  compute_efficiency_pct = None
  hbm_bandwidth_utilization_pct = None

  if flops is not None and hbm_bytes is not None and hbm_bytes > 0:
    arithmetic_intensity = flops / hbm_bytes

  if flops is not None and latency_ms > 0 and peak_flops_per_sec > 0:
    actual_flops_per_sec = flops / (latency_ms / 1000.0)
    compute_efficiency_pct = (actual_flops_per_sec / peak_flops_per_sec) * 100.0

  if hbm_bytes is not None and latency_ms > 0 and peak_hbm_bw_bytes_per_sec > 0:
    actual_bw = hbm_bytes / (latency_ms / 1000.0)
    hbm_bandwidth_utilization_pct = (actual_bw / peak_hbm_bw_bytes_per_sec) * 100.0

  return {
    "arithmetic_intensity": arithmetic_intensity,
    "compute_efficiency_pct": compute_efficiency_pct,
    "hbm_bandwidth_utilization_pct": hbm_bandwidth_utilization_pct,
  }
```

**Step 4: Run all compute_derived_metrics tests**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py -k "compute_derived" -v`
Expected: PASS (including existing tests — they don't check for extra keys)

**Step 5: Commit**

```bash
cd kernel-evolve && git add src/kernel_evolve/profiler.py tests/test_profiler.py
git commit -m "feat(profiler): add HBM bandwidth utilization metric (3690 GB/s peak)"
```

---

### Task 7: Update `analyze_ir_dumps` to include new metrics (profiler.py)

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/profiler.py:395-427` (`analyze_ir_dumps`)
- Modify: `kernel-evolve/tests/test_profiler.py`

**Step 1: Update test for `analyze_ir_dumps`**

Update the existing `test_analyze_ir_dumps` and add a new test:

```python
def test_analyze_ir_dumps_extended_metrics(tmp_path):
    """analyze_ir_dumps returns new metrics alongside existing ones."""
    llo_dir = tmp_path / "llo"
    llo_dir.mkdir()
    llo_text = """\
#allocation0 = f32[262144], size=0x100000
#allocation1 = u8[512], space=smem, size=0x200
;;
%v0 = vmatmul.mubr.bf16.gmra.mxu0 %r0
%v1 = vmax.xlane.f32.xlu0 %r1
;;
%v2 = dma.hbm_to_vmem %r2
%v3 = vpow2.f32 %r3
;;
"""
    (llo_dir / "module.pass_79.llo").write_text(llo_text)
    hlo_dir = tmp_path / "hlo"
    hlo_dir.mkdir()
    hlo_text = """\
HloModule test
fused_computation.1 { ROOT %r = f32[1] parameter(0) }
fused_computation.2 { ROOT %r = f32[1] parameter(0) }
ENTRY main {
  %p0 = bf16[8,2048,128] parameter(0)
  %copy-start = bf16[8,2048,128] copy-start(%p0), cross_program_prefetch_index=0
  ROOT %out = bf16[8,2048,128] custom-call(%p0), custom_call_target="tpu_custom_call"
}
"""
    (hlo_dir / "module.after_all_optimizations.txt").write_text(hlo_text)
    result = analyze_ir_dumps(str(hlo_dir), str(llo_dir))

    # Existing metrics still present
    assert result["vliw_bundle_count"] == 3
    assert result["mxu_utilization"] is not None

    # New metrics
    assert result["vmem_allocation"]["vmem_bytes"] == 0x100000
    assert result["vmem_allocation"]["smem_bytes"] == 0x200
    assert result["bundle_density"]["total_bundles"] == 3
    assert result["bundle_density"]["max_ops_per_bundle"] == 2
    assert result["dma_analysis"]["dma_count"] == 1
    assert result["dma_analysis"]["double_buffering"] is False
    assert result["fusion_analysis"]["fusion_count"] == 2
    assert result["fusion_analysis"]["has_cross_program_prefetch"] is True
    assert result["special_units"]["xlane_ops"] == 1
    assert result["special_units"]["eup_ops"] == 1
    assert result["special_units"]["nop_count"] == 0
```

**Step 2: Run test to verify it fails**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py::test_analyze_ir_dumps_extended_metrics -v`
Expected: FAIL — `KeyError: 'vmem_allocation'`

**Step 3: Update `analyze_ir_dumps` in profiler.py**

Replace the function body (lines 395-427):

```python
def analyze_ir_dumps(hlo_dir: str, llo_dir: str) -> dict[str, Any]:
  """Orchestrate IR analysis from HLO and LLO dump directories.

  Returns a dict with all parsed metrics — each None when the
  corresponding file is missing or parsing fails.
  """
  vliw: int | None = None
  mxu: dict | None = None
  hbm: int | None = None
  flops: float | None = None
  vmem: dict | None = None
  density: dict | None = None
  dma: dict | None = None
  special: dict | None = None
  fusion: dict | None = None

  llo_path = find_final_llo_file(llo_dir) if os.path.isdir(llo_dir) else None
  if llo_path is not None:
    llo_text = Path(llo_path).read_text()
    vliw = count_vliw_bundles(llo_text)
    mxu = parse_mxu_distribution(llo_text)
    vmem = parse_vmem_allocations(llo_text)
    density = analyze_bundle_density(llo_text)
    dma = analyze_dma_ops(llo_text)
    special = analyze_special_units(llo_text)

  hlo_path = find_hlo_file(hlo_dir)
  if hlo_path is not None:
    hlo_text = Path(hlo_path).read_text()
    hbm = estimate_hbm_bandwidth(hlo_text)
    flops = count_flops_from_hlo(hlo_text)
    fusion = count_hlo_fusions(hlo_text)

  return {
    "vliw_bundle_count": vliw,
    "mxu_utilization": mxu,
    "hbm_bandwidth_bytes": hbm,
    "flops": flops,
    "vmem_allocation": vmem,
    "bundle_density": density,
    "dma_analysis": dma,
    "special_units": special,
    "fusion_analysis": fusion,
  }
```

**Step 4: Run all analyze_ir_dumps tests**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py -k "analyze_ir_dumps" -v`
Expected: PASS (both existing and new tests)

**Step 5: Commit**

```bash
cd kernel-evolve && git add src/kernel_evolve/profiler.py tests/test_profiler.py
git commit -m "feat(profiler): integrate new metrics into analyze_ir_dumps"
```

---

### Task 8: Add new metrics to evaluate.py stage_profile_deep

**Files:**
- Modify: `kernel-evolve/docker/evaluate.py:312-514` (`stage_profile_deep`)
- Modify: `kernel-evolve/tests/test_evaluate_artifacts.py`

**Step 1: Write failing test**

Add to `test_evaluate_artifacts.py`:

```python
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
    assert result["bundle_density"]["total_bundles"] == 4
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
```

**Step 2: Run test to verify it fails**

Run: `cd kernel-evolve && python -m pytest tests/test_evaluate_artifacts.py::test_stage_profile_deep_extended_metrics -v`
Expected: FAIL — `KeyError: 'vmem_allocation'`

**Step 3: Add inline parsing to `stage_profile_deep` in evaluate.py**

After the existing MXU parsing block (after line 451), before the HLO section (line 453), add the new LLO-based metrics. After the HLO section (after line 498), add fusion analysis. Then add the new fields to the return dict.

In evaluate.py `stage_profile_deep`, after the MXU utilization block (after the `if total_mxu > 0:` block ending at line 451), add:

```python
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
```

After the HLO parsing block (after `arithmetic_intensity` at line 501), add:

```python
    # HLO fusion analysis
    fusion_analysis = None
    if chosen_hlo is not None:
      fc = len(re.findall(r"\bfused_computation\b", hlo_text))
      cpf = bool(re.search(r"cross_program_prefetch_index=", hlo_text))
      fusion_analysis = {
        "fusion_count": fc,
        "has_cross_program_prefetch": cpf,
      }
```

Update the return dict (line 503-512) to include new fields:

```python
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
```

**Step 4: Run all evaluate artifact tests**

Run: `cd kernel-evolve && python -m pytest tests/test_evaluate_artifacts.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd kernel-evolve && git add kernel-evolve/docker/evaluate.py kernel-evolve/tests/test_evaluate_artifacts.py
git commit -m "feat(evaluate): add extended IR metrics to stage_profile_deep"
```

---

### Task 9: Add HBM bandwidth utilization to evaluate.py main()

**Files:**
- Modify: `kernel-evolve/docker/evaluate.py:610-615` (main function, after compute_efficiency_pct)

**Step 1: Add HBM BW utilization computation after line 615**

```python
  # HBM bandwidth utilization (TPU v7x peak: 3690 GB/s)
  peak_hbm_bw = float(os.environ.get("PEAK_HBM_BW", 3690e9))
  if deep_profile.get("hbm_bandwidth_bytes") and perf_result["latency_ms"] > 0:
    actual_bw = deep_profile["hbm_bandwidth_bytes"] / (perf_result["latency_ms"] / 1000.0)
    deep_profile["hbm_bandwidth_utilization_pct"] = (actual_bw / peak_hbm_bw) * 100.0
```

**Step 2: Run existing tests to verify nothing broke**

Run: `cd kernel-evolve && python -m pytest tests/test_evaluate_artifacts.py -v`
Expected: PASS

**Step 3: Commit**

```bash
cd kernel-evolve && git add kernel-evolve/docker/evaluate.py
git commit -m "feat(evaluate): add HBM bandwidth utilization metric"
```

---

### Task 10: Update analyze skill SKILL.md

**Files:**
- Modify: `kernel-evolve/plugins/pallas-evolve/skills/analyze/SKILL.md`

**Step 1: Update the eval_result JSON example (around line 22)**

Add new fields to the `metadata.profile` section of the example JSON:

```json
    "profile": {
      "vliw_bundle_count": 4302,
      "mxu_utilization": {"mxu0": 1396, "mxu1": 1383, "dual_ratio": 0.99},
      "hbm_bandwidth_bytes": 14680064,
      "arithmetic_intensity": 72.8,
      "flops": 1.07e9,
      "compute_efficiency_pct": 45.2,
      "hbm_bandwidth_utilization_pct": 2.1,
      "vmem_allocation": {"vmem_bytes": 1572864, "smem_bytes": 1024, "allocation_count": 8},
      "bundle_density": {"total_bundles": 4302, "avg_ops_per_bundle": 3.2, "max_ops_per_bundle": 7},
      "dma_analysis": {"dma_count": 42, "dma_sync_count": 20, "double_buffering": true},
      "fusion_analysis": {"fusion_count": 0, "has_cross_program_prefetch": false},
      "special_units": {"xlane_ops": 15, "eup_ops": 8, "nop_count": 3}
    }
```

**Step 2: Add new metrics to the "Extract deep profiling metrics" section (after line 79)**

Add these bullet points:

```markdown
- `hbm_bandwidth_utilization_pct`: actual HBM bandwidth / peak bandwidth as percentage. >80% = near bandwidth ceiling.
- `vmem_allocation.vmem_bytes`: total on-chip VMEM allocated. High values indicate VMEM pressure.
- `bundle_density.avg_ops_per_bundle`: average operations per VLIW bundle. Higher = better ILP. <2.0 = poor slot utilization.
- `bundle_density.max_ops_per_bundle`: peak ILP achieved. TPU v7x can do up to 8 ops/bundle.
- `dma_analysis.dma_count`: total DMA transfer operations. High count may indicate excessive data movement.
- `dma_analysis.double_buffering`: whether the kernel uses double buffering (iteration % 2 buffer slots). False = DMA cannot overlap with compute.
- `fusion_analysis.fusion_count`: number of XLA fused_computation blocks. More fusions = more HBM round-trips. Pallas kernels should have 0.
- `special_units.xlane_ops`: cross-lane reduction operations (XLU). Used for max/sum reductions.
- `special_units.eup_ops`: hardware exponential unit operations. Used for exp() via vpow2.
- `special_units.nop_count`: empty VLIW slots. High count indicates pipeline bubbles.
```

**Step 3: Add new rows to the bottleneck classification table (after line 90)**

```markdown
| `avg_ops_per_bundle < 2.0` | — | Low ILP: VLIW slots underutilized, compiler couldn't parallelize |
| `double_buffering == false` | — | No double buffering: DMA and compute cannot overlap |
| `fusion_count > 3` | — | Too many fusions: excessive HBM round-trips between fusions |
| `nop_count > 50` | — | Pipeline bubbles: compiler couldn't fill VLIW slots with useful work |
| `hbm_bandwidth_utilization_pct > 80` | — | Near HBM bandwidth ceiling: memory-bound, reduce data movement |
```

**Step 4: Update the Performance Profile table in the analysis template (around line 210)**

Add rows:

```markdown
| HBM BW utilization | {hbm_bandwidth_utilization_pct}% | {near ceiling / headroom} |
| VMEM allocated | {vmem_allocation.vmem_bytes} bytes | {pressure level} |
| Bundle density (avg) | {avg_ops_per_bundle} ops/bundle | {poor/fair/good} |
| DMA transfers | {dma_count} ({double_buffering ? "double-buffered" : "single-buffered"}) | {assessment} |
| Pipeline NOPs | {nop_count} | {low/concerning/high} |
| HLO fusions | {fusion_count} | {0 = ideal for Pallas} |
```

**Step 5: Commit**

```bash
cd kernel-evolve && git add kernel-evolve/plugins/pallas-evolve/skills/analyze/SKILL.md
git commit -m "docs(analyze skill): add extended IR metrics to bottleneck classification"
```

---

### Task 11: Update profiler.py exports and run full test suite

**Files:**
- Modify: `kernel-evolve/tests/test_profiler.py` (import line)

**Step 1: Update the import block in test_profiler.py to include all new functions**

```python
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
```

**Step 2: Run the full test suite**

Run: `cd kernel-evolve && python -m pytest tests/test_profiler.py tests/test_evaluate_artifacts.py -v`
Expected: ALL PASS

**Step 3: Commit peak value updates (from earlier)**

```bash
cd kernel-evolve && git add src/kernel_evolve/profiler.py docker/evaluate.py
git commit -m "fix(profiler): update TPU v7x peak to 2307 TFLOPS / 3690 GB/s"
```

---

### Task 12: Final integration test

**Step 1: Run full test suite from project root**

Run: `cd kernel-evolve && python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 2: Verify no import errors**

Run: `cd kernel-evolve && python -c "from kernel_evolve.profiler import parse_vmem_allocations, analyze_bundle_density, analyze_dma_ops, count_hlo_fusions, analyze_special_units; print('All imports OK')"`
Expected: `All imports OK`
