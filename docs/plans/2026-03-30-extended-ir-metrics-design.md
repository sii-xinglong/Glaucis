# Extended IR Metrics Design

**Goal:** Add 6 new IR analysis metrics to the profiling pipeline, plus an HBM bandwidth utilization derived metric. These metrics are informed by the analysis in [When XLA Isn't Enough: From Pallas to VLIW Bundles](https://patricktoulme.substack.com/p/when-xla-isnt-enough-from-pallas) and fill gaps in the current profiling output.

**Approach:** Incremental — add new parsing functions to `profiler.py` (testable library) and self-contained equivalents to `evaluate.py` (Docker evaluator). No refactoring of existing code. All new fields are additive and optional (None when data is unavailable).

## Reference: TPU v7x Specs

- Peak compute: 2307 TFLOPS
- Peak HBM bandwidth: 3690 GB/s

## New Metrics

### 1. VMEM Allocation Tracking

**Source:** LLO text — `#allocation` entries with `size=0x...` and optional `space=smem`.

**Function:** `parse_vmem_allocations(llo_text: str) -> dict | None`

**Output:**
```python
{
  "vmem_bytes": int,        # Total VMEM allocation (excludes SMEM)
  "smem_bytes": int,        # Total SMEM allocation
  "allocation_count": int,  # Number of distinct allocations
}
```

**Why:** VMEM pressure directly affects double buffering capability and DMA/compute overlap. High VMEM usage can cause register spills visible as store/load patterns.

### 2. Bundle Density (ILP)

**Source:** LLO text — split by `;;` separators, count operation lines per segment.

**Function:** `analyze_bundle_density(llo_text: str) -> dict | None`

**Output:**
```python
{
  "total_bundles": int,
  "avg_ops_per_bundle": float,
  "max_ops_per_bundle": int,
}
```

**Why:** Total bundle count alone doesn't capture instruction-level parallelism. Splash attention achieves 7 ops/bundle vs reference's 5 — this metric captures that difference. Low avg_ops_per_bundle (< 2) indicates the compiler couldn't fill VLIW slots.

### 3. DMA Analysis

**Source:** LLO text — `dma.` prefixed operations, `dma.done.wait`, `sand.u32 1` pattern.

**Function:** `analyze_dma_ops(llo_text: str) -> dict | None`

**Output:**
```python
{
  "dma_count": int,           # Total DMA operations (dma.*)
  "dma_sync_count": int,      # dma.done.wait operations
  "double_buffering": bool,   # Detected sand.u32 1 (iteration % 2) pattern
}
```

**Why:** DMA count reveals data movement overhead. Missing double buffering means DMA and compute cannot overlap, leaving the VPU idle during transfers.

### 4. HLO Fusion Count

**Source:** HLO text — `fused_computation` definitions and `cross_program_prefetch_index`.

**Function:** `count_hlo_fusions(hlo_text: str) -> dict | None`

**Output:**
```python
{
  "fusion_count": int,                 # Number of fused_computation blocks
  "has_cross_program_prefetch": bool,  # copy-start with cross_program_prefetch
}
```

**Why:** Each separate fusion implies an HBM round-trip between fusions. Pallas kernels eliminate these by keeping intermediates in VMEM. Fusion count > 3 is a red flag for memory-bound workloads. Cross-program prefetch indicates XLA is pre-staging data.

### 5. Special Hardware Unit Usage

**Source:** LLO text — `.xlane` ops (XLU), `vpow2`/`vpop.eup` (EUP), `nop` instructions.

**Function:** `analyze_special_units(llo_text: str) -> dict | None`

**Output:**
```python
{
  "xlane_ops": int,   # Cross-lane reduction ops (vmax.xlane, vpop.xlane)
  "eup_ops": int,     # Exponent unit ops (vpow2, vpop.eup)
  "nop_count": int,   # Pipeline bubble indicator
}
```

**Why:** XLU and EUP usage reveals whether the kernel exploits specialized hardware for reductions and exponentials. High nop count indicates pipeline bubbles where the compiler couldn't schedule useful work.

### 6. HBM Bandwidth Utilization

**Source:** Derived from existing `hbm_bandwidth_bytes` and `latency_ms`.

**Computed in:** `compute_derived_metrics()` and `stage_profile_deep()`.

**Output:** `hbm_bandwidth_utilization_pct: float | None`

```python
actual_bw = hbm_bytes / (latency_ms / 1000.0)
hbm_bw_util_pct = (actual_bw / 3690e9) * 100.0
```

**Why:** Completes the roofline model — `compute_efficiency_pct` measures compute axis, this measures memory axis. Together they classify whether a kernel is compute-bound or memory-bound.

## Integration Points

### evaluate.py `stage_profile_deep` return value

New fields added to the return dict (all None when unavailable):

```python
{
  # existing fields unchanged
  "vliw_bundle_count": ...,
  "mxu_utilization": ...,
  "hbm_bandwidth_bytes": ...,
  "flops": ...,
  "arithmetic_intensity": ...,
  "compute_efficiency_pct": ...,
  # new fields
  "vmem_allocation": {"vmem_bytes": ..., "smem_bytes": ..., "allocation_count": ...},
  "bundle_density": {"avg_ops_per_bundle": ..., "max_ops_per_bundle": ...},
  "dma_analysis": {"dma_count": ..., "dma_sync_count": ..., "double_buffering": ...},
  "fusion_analysis": {"fusion_count": ..., "has_cross_program_prefetch": ...},
  "special_units": {"xlane_ops": ..., "eup_ops": ..., "nop_count": ...},
  "hbm_bandwidth_utilization_pct": ...,
}
```

### EVAL_RESULT JSON

New fields flow into `metadata.profile` in the final JSON output. The analyze skill reads them for bottleneck classification.

### analyze skill (SKILL.md)

New bottleneck classification rows:

| Signal | Threshold | Diagnosis |
|--------|-----------|-----------|
| `nop_count > 50` | — | Pipeline bubbles: compiler couldn't fill VLIW slots |
| `double_buffering == False` | — | No double buffering, DMA cannot overlap with compute |
| `fusion_count > 3` | — | Too many fusions causing HBM round-trips |
| `avg_ops_per_bundle < 2` | — | Low ILP: VLIW slots underutilized |
| `vmem_bytes > threshold` | — | High VMEM pressure, risk of register spills |
| `hbm_bandwidth_utilization_pct > 80` | — | Near HBM bandwidth ceiling, memory-bound |

## Implementation Strategy

- **profiler.py**: Add 5 new standalone parsing functions + update `compute_derived_metrics` + update `analyze_ir_dumps`.
- **evaluate.py**: Add equivalent inline parsing in `stage_profile_deep` + add `hbm_bandwidth_utilization_pct` computation.
- **Tests**: TDD — write failing tests in `test_profiler.py` and `test_evaluate_artifacts.py` first, then implement.
- **analyze skill**: Update SKILL.md with new metrics in the bottleneck table and analysis template.

All changes are backward-compatible. Missing metrics default to None.
