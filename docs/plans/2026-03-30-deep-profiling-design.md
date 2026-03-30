# Deep Profiling for kernel-evolve

**Date**: 2026-03-30
**Status**: Approved

## Goal

Enhance the kernel-evolve profiling pipeline to provide richer performance signals for both the LLM optimizer (analyze skill) and human developers. Currently, the profile stage only captures `compute_ratio` and `memory_transfer_ratio` from XPlane traces. This design adds HLO/LLO IR analysis, VLIW bundle counting, MXU utilization, HBM bandwidth estimation, and FLOP counting.

## References

- [When XLA Isn't Enough: From Pallas to VLIW Bundles](https://patricktoulme.substack.com/p/when-xla-isnt-enough-from-pallas) — HLO/LLO/Mosaic IR dump methodology, VLIW bundle analysis, MXU distribution, HBM bandwidth calculation
- [xprof Custom Call Profiling](https://github.com/openxla/xprof/blob/master/docs/custom_call_profiling.md) — `--xla_enable_custom_call_region_trace` and `--xla_xprof_register_llo_debug_info` flags for per-kernel LLO utilization in traces

## Design

### Enhanced Profile Stage (evaluate.py)

The existing `stage_profile` is enhanced with three sub-stages:

#### Stage 4a: XPlane Trace (existing, enhanced)

Existing compute_ratio/memory_transfer_ratio from XPlane. Add xprof custom call region tracing flags to `LIBTPU_INIT_ARGS`:

```
--xla_enable_custom_call_region_trace=true
--xla_xprof_register_llo_debug_info=true
```

This enables LLO utilization data in the trace, providing per-core hardware utilization breakdown.

#### Stage 4b: IR Dump + Analysis (new)

Set environment variables for key IR dumps:

```python
XLA_FLAGS = (
    "--xla_dump_hlo_as_text "
    "--xla_dump_to={hlo_dir} "
)
LIBTPU_INIT_ARGS += (
    " --xla_jf_dump_to={llo_dir} "
    "--xla_jf_dump_hlo_text=true "
    "--xla_jf_dump_llo_text=true "
    "--xla_jf_emit_annotations=true "
    "--xla_mosaic_dump_to={mosaic_dir} "
    "--xla_mosaic_enable_llo_source_annotations=true"
)
```

Run a single kernel invocation with dump flags (separate from perf measurement). Parse key files:

- **VLIW bundle count**: Parse final LLO pass for bundle markers → total count
- **MXU distribution**: Count `mxu0` / `mxu1` operations → dual-MXU utilization ratio
- **HBM bandwidth**: Extract operand shapes from HLO custom_call → calculate bytes read/written
- **FLOP count**: From HLO dot/convolution dimensions → actual FLOPs

#### Stage 4c: Derived Metrics (new)

- **Arithmetic intensity**: FLOPs / HBM bytes (roofline positioning)
- **Compute efficiency**: actual_time / (FLOPs / peak_FLOPS) as percentage

### Extended EVAL_RESULT

```json
{
  "status": "SUCCESS",
  "fitness": 1.62,
  "latency_ms": 0.45,
  "speedup": 1.62,
  "flops": 1.07e9,
  "compute_ratio": 0.82,
  "memory_transfer_ratio": 0.18,
  "metadata": {
    "reference_latency_ms": 0.73,
    "profile_diagnostics": {},
    "profile": {
      "vliw_bundle_count": 4302,
      "mxu_utilization": {"mxu0": 1396, "mxu1": 1383, "dual_ratio": 0.99},
      "hbm_bandwidth_bytes": 14680064,
      "arithmetic_intensity": 72.8,
      "flops": 1.07e9,
      "compute_efficiency_pct": 45.2,
      "llo_utilization": {}
    }
  }
}
```

### Enhanced Analyze Skill

Replace single-axis bottleneck classification with multi-signal analysis:

| Signal | Metric | What it tells the LLM |
|--------|--------|----------------------|
| `compute_ratio` | 0.0-1.0 | VPU idle time (SyncWait/DMA stalls) |
| `arithmetic_intensity` | FLOPs/byte | Position on roofline |
| `dual_ratio` (MXU) | 0.0-1.0 | Whether both MXUs are utilized |
| `vliw_bundle_count` | integer | Kernel complexity |
| `compute_efficiency_pct` | 0-100% | How close to peak FLOPS |

Optimization signal mapping:

- **Low arithmetic intensity + high compute_ratio** → Consider larger tiles or prefetching
- **Low dual_ratio** → Check if matmul dimensions allow dual-MXU scheduling
- **Growing vliw_bundle_count** → Kernel complexity bloat
- **Low compute_efficiency_pct** → Pipeline stalls, register spills, unnecessary recomputation

Trend analysis tracks all new metrics across iterations.

## Files Changed

1. `kernel-evolve/docker/evaluate.py` — Enhanced stage_profile + new IR parsing helpers
2. `kernel-evolve/src/kernel_evolve/profiler.py` — Library-side profiling functions
3. `kernel-evolve/src/kernel_evolve/evaluator.py` — Extended EvalResult fields
4. `.github/ci/kernel-eval-job.yaml` — Add LIBTPU_INIT_ARGS env var
5. `kernel-evolve/plugins/pallas-evolve/skills/analyze/SKILL.md` — Multi-signal analysis
6. `kernel-evolve/plugins/pallas-evolve/skills/start/SKILL.md` — Reference new profiling signals
