# Kernel Profile: XPlane Trace-based Pallas Kernel Profiling

**Date:** 2026-03-28
**Issue:** [#7](https://github.com/sii-xinglong/Glaucis/issues/7)
**Status:** Approved

## Goal

Integrate Pallas kernel profiling (via XPlane trace) into `kernel-evolve`'s evaluation pipeline, adding `compute_ratio` as a MAP-Elites behavioral descriptor for richer evolutionary fitness signals.

## Architecture

### New Module: `kernel_evolve/profiler.py`

Three functions with clean separation:

**`capture_trace(kernel_fn, shapes, trace_dir, warmup=3, runs=3) -> str`**
- Configures `jax.profiler.ProfileOptions`: `python_tracer_level=0`, `host_tracer_level=2`, `tpu_trace_mode=TRACE_COMPUTE_AND_SYNC`
- Runs `kernel_fn` with profiler tracing enabled
- Searches `trace_dir` for generated `.xplane.pb` file
- Returns path to `.xplane.pb`

**`analyze_trace(xplane_path) -> dict`**
- Uses `xprof.convert.raw_to_tool_data.xspace_to_tool_data([path], "trace_viewer", {})` to convert XPlane protobuf to Chrome trace JSON
- Finds `/device:TPU:0` process, locates `jit_computation` events
- Computes `SyncWait / total_time` ratio from last execution window
- Returns `{"compute_ratio": float, "memory_transfer_ratio": float}`

**`stage_profile(exec_globals, shapes, trace_dir="/tmp/xplane_trace") -> dict`**
- Orchestrator for evaluate.py integration
- Extracts kernel function, calls `capture_trace` then `analyze_trace`
- Returns `{"ok": True/False, "compute_ratio": ..., "memory_transfer_ratio": ...}`

### evaluate.py Changes

Pipeline becomes 4 stages: `compile -> correctness -> performance -> profile`

Profile stage is non-fatal: if profiling fails, EVAL_RESULT still succeeds with `compute_ratio: null`.

Updated EVAL_RESULT format:
```json
{
  "status": "SUCCESS",
  "fitness": 1.5,
  "latency_ms": 2.3,
  "speedup": 1.5,
  "flops": 0.0,
  "compute_ratio": 0.85,
  "memory_transfer_ratio": 0.15
}
```

### MAP-Elites Descriptor: `compute_profile`

New 4th dimension in the behavioral descriptor grid:

| Bucket | compute_ratio range | Label |
|--------|-------------------|-------|
| 0 | [0, 0.25) | `very_low` |
| 1 | [0.25, 0.50) | `low` |
| 2 | [0.50, 0.75) | `medium` |
| 3 | [0.75, 1.0] | `high` |

Grid expansion: 48 cells -> 192 cells per island (4 x 4 x 3 x 4).

When `compute_ratio` is null (profiling failed), defaults to `medium` bucket.

### Docker/CI Dependencies

Add `xprof` to pip install in:
- `kernel-eval-job.yaml` container install step
- Docker image build (if applicable)

## Files Changed

| File | Change |
|------|--------|
| `kernel-evolve/src/kernel_evolve/profiler.py` | **New** - profiling module |
| `kernel-evolve/docker/evaluate.py` | Add `stage_profile` call after performance |
| `kernel-evolve/src/kernel_evolve/population.py` | Add `compute_profile` descriptor dimension |
| `kernel-evolve/src/kernel_evolve/evaluator.py` | Add `compute_ratio`/`memory_transfer_ratio` to EvalResult |
| `kernel-evolve/src/kernel_evolve/ci_dispatcher.py` | Map `compute_ratio` to descriptor |
| `.github/ci/kernel-eval-job.yaml` | Add `xprof` dependency |
| `kernel-evolve/tests/test_profiler.py` | **New** - profiler unit tests |
| `kernel-evolve/tests/test_population.py` | Update for new descriptor |

## Testing

### Unit tests
- Mock `xprof.convert.raw_to_tool_data` with known Chrome trace JSON
- Test all 4 compute_profile buckets + null case
- Test graceful failure paths

### E2E verification
- Standalone script to run matmul on real TPU, capture trace, parse metrics
- Abstraction layer for remote TPU execution

## Reference

Ported from `accelerator-agents/MaxKernel/tpu_kernel_gen/agents/kernel_gen_agent/analyze_profile.py`.
