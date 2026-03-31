# Tiling Parameter Sweeper Design

**Date:** 2026-03-31
**Status:** Approved
**Context:** Inspired by tokamax's auto-tuning mechanism (`_get_autotuning_configs` + `Autotuner`), this design adds algorithmic grid search for tiling parameters to Glaucis's LLM-driven kernel optimization loop.

## Problem

Glaucis currently relies entirely on the LLM agent to select tiling parameters (block sizes, pipeline stages, chunk sizes). While the LLM is effective at structural code transformations (kernel fusion, scan-to-pallas conversion), tiling exploration is a numeric search problem better suited to systematic enumeration. The gmm_fp8_blockwise optimization spent multiple rounds manually trying tiling combinations that a grid search would have found faster.

## Solution

Add a **post-mutation tiling sweeper** that automatically expands each LLM-generated code variant into multiple variants with different tiling configurations. The sweeper uses grid search over a YAML-defined parameter space, filtered by constraints. Expanded variants are submitted as additional entries in the existing batch evaluation pipeline.

## Architecture

### Flow

```
THINK (LLM generates code variants)
  → expand_with_tuning(variants, tuning_config)
  → [v1_t0.py, v1_t1.py, ..., v2_t0.py, v2_t1.py, ...]
  → SUBMIT (batch to TPU)
  → ANALYZE (group by code variant, report best tiling per variant)
```

The LLM focuses on code structure; the sweeper handles numeric tuning. They are orthogonal.

## YAML Config Schema

The `EvolveConfig` gains an optional `tuning_params` section:

```yaml
tuning_params:
  params:
    BLOCK_K:
      values: [64, 128, 256]
      marker: "BLOCK_K ="        # regex to find the assignment in code
    BLOCK_V:
      values: [64, 128, 256]
    CHUNK_SIZE:
      values: [32, 64, 128]
  constraints:
    - "BLOCK_K * BLOCK_V <= 65536"   # Python expr, evaluated with params in scope
  max_configs: 32                     # cap grid size; uniform sample if exceeded
  enabled: true                       # toggle per-kernel
```

### Param Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `values` | `list[int\|float]` | Yes | Explicit list of values to try |
| `marker` | `str` | No | Regex pattern to find the assignment. Defaults to `{PARAM_NAME} =` |

### Top-level Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `constraints` | `list[str]` | `[]` | Python expressions; config is valid iff all evaluate to `True` |
| `max_configs` | `int` | `32` | Maximum configs to evaluate; deterministic uniform sampling if exceeded |
| `enabled` | `bool` | `True` | Disable tiling sweep without removing the config |

## Grid Generation Module

**New file:** `kernel_evolve/src/kernel_evolve/tuning.py`

### Data Models

```python
class TuningParam(BaseModel):
    name: str
    values: list[int | float]
    marker: str | None = None

class TuningConfig(BaseModel):
    params: list[TuningParam]
    constraints: list[str] = []
    max_configs: int = 32
    enabled: bool = True
```

### Functions

**`generate_configs(tuning_config: TuningConfig) -> list[dict[str, int | float]]`**

1. Compute cartesian product of all `param.values`.
2. Filter by constraints: for each config dict, evaluate each constraint string with the config as locals. Keep configs where all constraints return `True`.
3. If result exceeds `max_configs`, take a deterministic uniform sample: sort by hash of config repr, take every `N/max_configs`-th entry.
4. Return the list of valid config dicts.

**`apply_config(kernel_code: str, config: dict[str, int | float], params: list[TuningParam]) -> str`**

For each param in the config:
1. Determine marker: `param.marker` or `f"{param.name} ="`.
2. Use regex `r"({marker}\s*)(\S+)"` to find the assignment line.
3. Replace the value portion with the new value.
4. If marker not found: log warning, skip this param.
5. If marker found multiple times: raise error (ambiguous).
6. After all substitutions, validate syntax with `ast.parse()`.
7. Return modified code.

**`expand_variants(variants: list[str], tuning_config: TuningConfig, params: list[TuningParam]) -> list[tuple[str, dict]]`**

For each variant code string, apply each config from `generate_configs()`, returning `(modified_code, config_dict)` pairs. Total output size = `len(variants) * len(configs)`.

## Integration with Optimization Loop

### Variant Naming

Expanded variants use the naming convention: `{lineage}_{variant}_t{config_idx}`

Examples: `L1_v1_t0.py`, `L1_v1_t1.py`, `L2_v3_t5.py`

### Metadata Propagation

Each expanded variant carries:
- `code_variant_id`: identifies the LLM-generated code (e.g., `L1_v1`)
- `tuning_config`: the tiling config dict (e.g., `{"BLOCK_K": 256, "BLOCK_V": 128}`)

This metadata is:
1. Written as `_tuning_config.json` alongside the kernel code in the submission payload.
2. Read by `docker/evaluate.py` and included in the `EVAL_RESULT:` JSON output.
3. Used by the ANALYZE phase to group results.

### Analysis Grouping

The ANALYZE phase:
1. Groups `EvalResult`s by `code_variant_id`.
2. For each code variant, identifies the tiling config with the best speedup.
3. Reports **tiling sensitivity**: max speedup / min speedup across configs (indicates how much tiling matters for this variant).
4. Scores each code variant by its **best** tiling result, not its default.
5. Records the winning (code, tiling) pair in lineage tracking.

### Lineage Tracking

`lineages.json` gains a `best_tuning_config` field per lineage:

```json
{
  "L1": {
    "best_speedup": 9.054,
    "kernel_path": "round_10/L1_v2_t3.py",
    "best_tuning_config": {"BLOCK_K": 256, "BLOCK_V": 128, "CHUNK_SIZE": 64},
    ...
  }
}
```

The next round's THINK phase sees both the winning code and its best tiling config, informing the LLM's next mutations.

### Budget

Total batch size per round = `num_lineages * num_code_variants_per_lineage * min(grid_size, max_configs)`

Example: 2 lineages * 3 code variants * 12 tiling configs = 72 variants per batch.

## Changes to Existing Components

### `config.py`

Add `tuning_params: TuningConfig | None = None` to `EvolveConfig`.

### `evaluator.py`

Add optional `tuning_config: dict[str, int | float] | None = None` to both `EvalRequest` and `EvalResult`.

### `kube_evaluator.py`

No changes to submission logic. Expanded variants are additional entries in `BatchEvalRequest`. The variant payload includes a `_tuning_config.json` sidecar when tuning is active.

### `docker/evaluate.py`

Read `_tuning_config.json` if present. Include the config dict in the `EVAL_RESULT:` JSON output line.

### `mutation.py`

No changes. EVOLVE-BLOCK mutation and tiling substitution are orthogonal.

### Pallas-evolve Skills

| Skill | Change |
|-------|--------|
| `start` | Read `tuning_params` from config; pass to expansion step |
| `submit` | No changes (already handles batch variants) |
| `analyze` | Group results by code variant; report best tiling per variant; report tiling sensitivity |
| `reflect` | Record tiling insights (e.g., "BLOCK_K=256 consistently best") |

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Marker not found in code | Skip param, log warning. LLM may have renamed/removed the variable. |
| All configs filtered by constraints | Use original code as-is (no expansion). Log warning. |
| Syntax error after substitution | Skip that config, log error. Other configs proceed. |
| Grid exceeds `max_configs` | Deterministic uniform sampling. Log total vs. sampled size. |
| `tuning_params` absent or `enabled: false` | Variants pass through unchanged. Fully backward compatible. |

## Testing

### New: `test_tuning.py`

- `generate_configs`: cartesian product correctness, constraint filtering, max_configs sampling, empty grid edge case
- `apply_config`: single marker replacement, missing marker warning, multiple marker error, syntax validation after substitution
- `expand_variants`: end-to-end expansion of N variants * M configs

### Extended: `test_config.py`

- Validate `EvolveConfig` with `tuning_params` section
- Validate missing/optional `tuning_params`
- Validate constraint syntax errors

### Extended: `test_evaluator.py`

- `EvalRequest`/`EvalResult` serialization with `tuning_config` field
- Backward compatibility without `tuning_config`

### Integration Test

End-to-end expansion of a 2-variant * 3-config scenario with mock kernel code, verifying correct substitution and metadata propagation.

## Tokamax Reference

This design is inspired by tokamax's auto-tuning architecture:

| Tokamax Concept | Glaucis Equivalent |
|----------------|-------------------|
| `_get_autotuning_configs(ba)` | `generate_configs(tuning_config)` |
| `_get_heuristics_config(ba)` | LLM-selected default tiling (existing behavior) |
| `Autotuner.autotune()` | Post-mutation expansion + batch submission |
| `ArgSpec` (benchmark shapes) | Fixed shapes in YAML config (existing `shapes` section) |
| `ConfigBase` per op | `TuningParam` list in YAML |
| Per-device cache | `lineages.json` `best_tuning_config` field |

Key difference: tokamax sweeps configs for a fixed kernel implementation. Glaucis sweeps configs for each LLM-mutated kernel variant, combining structural exploration (LLM) with numeric exploration (grid search).
