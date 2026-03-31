# Tiling Parameter Sweeper Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add post-mutation tiling parameter grid search that expands LLM-generated kernel variants into multiple tiling configurations for batch evaluation on TPU.

**Architecture:** New `tuning.py` module handles grid generation and code substitution. Metadata flows through `EvalRequest.metadata` (new field) → `batch_dispatch` → `docker/evaluate.py` → `EvalResult.metadata` (existing field). The expansion happens between THINK and SUBMIT phases.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, itertools, hashlib, re, ast

**Spec:** `docs/superpowers/specs/2026-03-31-tiling-parameter-sweeper-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `kernel-evolve/src/kernel_evolve/tuning.py` | Create | Pydantic models (`TuningParam`, `TuningConfig`), `generate_configs()`, `apply_config()`, `expand_variants()` |
| `kernel-evolve/src/kernel_evolve/config.py` | Modify | Add `tuning_params: TuningConfig \| None` to `EvolveConfig` |
| `kernel-evolve/src/kernel_evolve/evaluator.py` | Modify | Add `metadata` field to `EvalRequest`, update serialization, change `BatchEvalRequest.variants` type |
| `kernel-evolve/src/kernel_evolve/docker_evaluate_helpers.py` | Modify | Forward `metadata` from variant dict to single-variant payload |
| `kernel-evolve/docker/evaluate.py` | Modify | Merge request metadata into result metadata |
| `kernel-evolve/tests/test_tuning.py` | Create | Tests for all tuning module functions |
| `kernel-evolve/tests/test_config.py` | Modify | Add tests for `tuning_params` config parsing |
| `kernel-evolve/tests/test_evaluator.py` | Modify | Add tests for `EvalRequest.metadata` and `BatchEvalRequest` type change |
| `kernel-evolve/tests/test_batch_integration.py` | Modify | Add test for metadata propagation through batch pipeline |

---

## Chunk 1: Core Tuning Module

### Task 1: TuningParam and TuningConfig Models + Config Integration

**Files:**
- Create: `kernel-evolve/src/kernel_evolve/tuning.py`
- Modify: `kernel-evolve/src/kernel_evolve/config.py:53-63`
- Test: `kernel-evolve/tests/test_tuning.py`
- Test: `kernel-evolve/tests/test_config.py`

- [ ] **Step 1: Write failing tests for TuningParam/TuningConfig models**

Create `kernel-evolve/tests/test_tuning.py`:

```python
"""Tests for the tuning grid generation module."""

import pytest

from kernel_evolve.tuning import TuningConfig, TuningParam


class TestTuningModels:
    def test_tuning_param_basic(self):
        p = TuningParam(values=[64, 128, 256])
        assert p.values == [64, 128, 256]
        assert p.marker is None

    def test_tuning_param_with_marker(self):
        p = TuningParam(values=[1, 2], marker="MY_BLOCK =")
        assert p.marker == "MY_BLOCK ="

    def test_tuning_config_basic(self):
        tc = TuningConfig(
            params={
                "BLOCK_K": TuningParam(values=[64, 128]),
                "BLOCK_V": TuningParam(values=[64, 128]),
            }
        )
        assert len(tc.params) == 2
        assert tc.constraints == []
        assert tc.max_configs == 32
        assert tc.enabled is True

    def test_tuning_config_disabled(self):
        tc = TuningConfig(params={}, enabled=False)
        assert tc.enabled is False

    def test_tuning_config_with_constraints(self):
        tc = TuningConfig(
            params={"X": TuningParam(values=[1])},
            constraints=["X > 0"],
            max_configs=16,
        )
        assert tc.constraints == ["X > 0"]
        assert tc.max_configs == 16
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_tuning.py::TestTuningModels -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kernel_evolve.tuning'`

- [ ] **Step 3: Write minimal TuningParam and TuningConfig models**

Create `kernel-evolve/src/kernel_evolve/tuning.py`:

```python
"""Tiling parameter grid search: generate configs and apply to kernel code."""

from __future__ import annotations

from pydantic import BaseModel


class TuningParam(BaseModel):
    values: list[int | float]
    marker: str | None = None


class TuningConfig(BaseModel):
    params: dict[str, TuningParam]
    constraints: list[str] = []
    max_configs: int = 32
    enabled: bool = True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_tuning.py::TestTuningModels -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Write failing test for EvolveConfig with tuning_params**

Add to `kernel-evolve/tests/test_config.py` (using dict-style construction to match existing test patterns):

```python
def test_evolve_config_with_tuning_params():
    """EvolveConfig accepts optional tuning_params."""
    config = EvolveConfig(
        kernel={"name": "test", "template": "t.py", "reference": "r.py"},
        shapes=[{"M": 1024}],
        tpu={"cluster": "c", "zone": "z"},
        tuning_params={
            "params": {"BLOCK_K": {"values": [64, 128]}},
        },
    )
    assert config.tuning_params is not None
    assert "BLOCK_K" in config.tuning_params.params


def test_evolve_config_without_tuning_params():
    """EvolveConfig works without tuning_params (backward compatible)."""
    config = EvolveConfig(
        kernel={"name": "test", "template": "t.py", "reference": "r.py"},
        shapes=[{"M": 1024}],
        tpu={"cluster": "c", "zone": "z"},
    )
    assert config.tuning_params is None


def test_load_config_with_tuning_params(tmp_path):
    """YAML with tuning_params section parses correctly."""
    yaml_str = """
kernel:
  name: test
  template: t.py
  reference: r.py
shapes:
  - M: 1024
tpu:
  cluster: c
  zone: z
tuning_params:
  params:
    BLOCK_K:
      values: [64, 128, 256]
      marker: "BLOCK_K ="
    BLOCK_V:
      values: [64, 128]
  constraints:
    - "BLOCK_K * BLOCK_V <= 32768"
  max_configs: 16
  enabled: true
"""
    p = tmp_path / "test.yaml"
    p.write_text(yaml_str)
    config = load_config(p)
    assert config.tuning_params is not None
    assert config.tuning_params.max_configs == 16
    assert len(config.tuning_params.params) == 2
    assert config.tuning_params.constraints == ["BLOCK_K * BLOCK_V <= 32768"]
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_config.py::test_evolve_config_with_tuning_params tests/test_config.py::test_evolve_config_without_tuning_params tests/test_config.py::test_load_config_with_tuning_params -v`
Expected: FAIL — `EvolveConfig` rejects `tuning_params` because `extra = "forbid"`

- [ ] **Step 7: Add tuning_params to EvolveConfig**

Modify `kernel-evolve/src/kernel_evolve/config.py`:

Add import at top:
```python
from kernel_evolve.tuning import TuningConfig
```

Add field to `EvolveConfig` class (after `batch`):
```python
  tuning_params: TuningConfig | None = None
```

- [ ] **Step 8: Run all config and model tests**

Run: `cd kernel-evolve && python -m pytest tests/test_config.py tests/test_tuning.py -v`
Expected: All tests PASS

- [ ] **Step 9: Commit**

```bash
git add kernel-evolve/src/kernel_evolve/tuning.py kernel-evolve/src/kernel_evolve/config.py kernel-evolve/tests/test_tuning.py kernel-evolve/tests/test_config.py
git commit -m "feat(tuning): add TuningParam/TuningConfig models and EvolveConfig integration"
```

---

### Task 2: generate_configs

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/tuning.py`
- Test: `kernel-evolve/tests/test_tuning.py`

- [ ] **Step 1: Write failing tests for generate_configs**

Add to `kernel-evolve/tests/test_tuning.py`:

```python
from kernel_evolve.tuning import generate_configs


class TestGenerateConfigs:
    def test_single_param(self):
        tc = TuningConfig(params={"X": TuningParam(values=[1, 2, 3])})
        configs = generate_configs(tc)
        assert configs == [{"X": 1}, {"X": 2}, {"X": 3}]

    def test_cartesian_product(self):
        tc = TuningConfig(
            params={
                "A": TuningParam(values=[1, 2]),
                "B": TuningParam(values=[10, 20]),
            }
        )
        configs = generate_configs(tc)
        assert len(configs) == 4
        assert {"A": 1, "B": 10} in configs
        assert {"A": 1, "B": 20} in configs
        assert {"A": 2, "B": 10} in configs
        assert {"A": 2, "B": 20} in configs

    def test_constraint_filters(self):
        tc = TuningConfig(
            params={
                "X": TuningParam(values=[1, 2, 3, 4]),
                "Y": TuningParam(values=[1, 2, 3, 4]),
            },
            constraints=["X * Y <= 6"],
        )
        configs = generate_configs(tc)
        for cfg in configs:
            assert cfg["X"] * cfg["Y"] <= 6
        # Valid pairs: (1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(2,3),(3,1),(3,2),(4,1)
        assert len(configs) == 10

    def test_multiple_constraints(self):
        tc = TuningConfig(
            params={
                "X": TuningParam(values=[1, 2, 3]),
                "Y": TuningParam(values=[1, 2, 3]),
            },
            constraints=["X + Y <= 4", "X != Y"],
        )
        configs = generate_configs(tc)
        for cfg in configs:
            assert cfg["X"] + cfg["Y"] <= 4
            assert cfg["X"] != cfg["Y"]
        # Valid: (1,2),(1,3),(2,1),(3,1)
        assert len(configs) == 4

    def test_all_filtered_returns_empty(self):
        tc = TuningConfig(
            params={"X": TuningParam(values=[1, 2])},
            constraints=["X > 100"],
        )
        configs = generate_configs(tc)
        assert configs == []

    def test_max_configs_caps(self):
        tc = TuningConfig(
            params={"X": TuningParam(values=list(range(50)))},
            max_configs=10,
        )
        configs = generate_configs(tc)
        assert len(configs) == 10

    def test_max_configs_deterministic(self):
        """Same input always produces same sample."""
        tc = TuningConfig(
            params={"X": TuningParam(values=list(range(50)))},
            max_configs=10,
        )
        configs_a = generate_configs(tc)
        configs_b = generate_configs(tc)
        assert configs_a == configs_b

    def test_empty_params(self):
        tc = TuningConfig(params={})
        configs = generate_configs(tc)
        # Cartesian product of empty set = one empty dict
        assert configs == [{}]

    def test_disabled_returns_empty(self):
        tc = TuningConfig(
            params={"X": TuningParam(values=[1, 2])},
            enabled=False,
        )
        configs = generate_configs(tc)
        assert configs == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_tuning.py::TestGenerateConfigs -v`
Expected: FAIL with `ImportError: cannot import name 'generate_configs'`

- [ ] **Step 3: Implement generate_configs**

Add to `kernel-evolve/src/kernel_evolve/tuning.py`:

```python
import hashlib
import itertools


def generate_configs(tuning_config: TuningConfig) -> list[dict[str, int | float]]:
    """Generate all valid tiling configs via cartesian product, filtered by constraints."""
    if not tuning_config.enabled:
        return []

    param_names = list(tuning_config.params.keys())
    param_values = [tuning_config.params[name].values for name in param_names]

    configs = []
    for combo in itertools.product(*param_values):
        cfg = dict(zip(param_names, combo))
        if _satisfies_constraints(cfg, tuning_config.constraints):
            configs.append(cfg)

    if len(configs) > tuning_config.max_configs:
        configs = _deterministic_sample(configs, tuning_config.max_configs)

    return configs


def _satisfies_constraints(
    config: dict[str, int | float], constraints: list[str]
) -> bool:
    for constraint in constraints:
        if not eval(constraint, {"__builtins__": {}}, config):  # noqa: S307
            return False
    return True


def _deterministic_sample(
    configs: list[dict[str, int | float]], n: int
) -> list[dict[str, int | float]]:
    """Uniform deterministic sample: sort by sha256 hash, take every k-th."""
    keyed = sorted(configs, key=lambda c: hashlib.sha256(repr(c).encode()).hexdigest())
    step = len(keyed) / n
    return [keyed[int(i * step)] for i in range(n)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_tuning.py::TestGenerateConfigs -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add kernel-evolve/src/kernel_evolve/tuning.py kernel-evolve/tests/test_tuning.py
git commit -m "feat(tuning): implement generate_configs with constraint filtering and deterministic sampling"
```

---

### Task 3: apply_config

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/tuning.py`
- Test: `kernel-evolve/tests/test_tuning.py`

- [ ] **Step 1: Write failing tests for apply_config**

Add to `kernel-evolve/tests/test_tuning.py`:

```python
import logging

from kernel_evolve.tuning import apply_config


SAMPLE_KERNEL = """\
import jax
import jax.numpy as jnp

BLOCK_K = 128
BLOCK_V = 64
NUM_STAGES = 2

def kernel(x):
    return x + 1
"""


class TestApplyConfig:
    def test_single_param_replacement(self):
        params = {"BLOCK_K": TuningParam(values=[64, 128, 256])}
        result = apply_config(SAMPLE_KERNEL, {"BLOCK_K": 256}, params)
        assert "BLOCK_K = 256" in result
        assert "BLOCK_V = 64" in result  # unchanged

    def test_multiple_param_replacement(self):
        params = {
            "BLOCK_K": TuningParam(values=[64, 256]),
            "BLOCK_V": TuningParam(values=[32, 128]),
        }
        result = apply_config(
            SAMPLE_KERNEL, {"BLOCK_K": 256, "BLOCK_V": 128}, params
        )
        assert "BLOCK_K = 256" in result
        assert "BLOCK_V = 128" in result
        assert "NUM_STAGES = 2" in result  # unchanged

    def test_custom_marker(self):
        code = "my_block_size = 64\n"
        params = {"X": TuningParam(values=[128], marker="my_block_size =")}
        result = apply_config(code, {"X": 128}, params)
        assert "my_block_size = 128" in result

    def test_missing_marker_logs_warning(self, caplog):
        params = {"NONEXISTENT": TuningParam(values=[1])}
        with caplog.at_level(logging.WARNING):
            result = apply_config(SAMPLE_KERNEL, {"NONEXISTENT": 1}, params)
        assert "NONEXISTENT" in caplog.text
        assert result == SAMPLE_KERNEL  # unchanged

    def test_ambiguous_marker_raises(self):
        code = "X = 1\nX = 2\n"
        params = {"X": TuningParam(values=[3])}
        with pytest.raises(ValueError, match="ambiguous"):
            apply_config(code, {"X": 3}, params)

    def test_syntax_validation(self):
        code = "BLOCK_K = 128\ndef f(:\n"  # already invalid
        params = {"BLOCK_K": TuningParam(values=[256])}
        with pytest.raises(SyntaxError):
            apply_config(code, {"BLOCK_K": 256}, params)

    def test_float_value(self):
        code = "SCALE = 1.0\n"
        params = {"SCALE": TuningParam(values=[0.5, 2.0])}
        result = apply_config(code, {"SCALE": 0.5}, params)
        assert "SCALE = 0.5" in result

    def test_preserves_surrounding_code(self):
        params = {"BLOCK_K": TuningParam(values=[256])}
        result = apply_config(SAMPLE_KERNEL, {"BLOCK_K": 256}, params)
        assert "import jax" in result
        assert "def kernel(x):" in result
        assert "return x + 1" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_tuning.py::TestApplyConfig -v`
Expected: FAIL with `ImportError: cannot import name 'apply_config'`

- [ ] **Step 3: Implement apply_config**

Add to `kernel-evolve/src/kernel_evolve/tuning.py`:

```python
import ast
import logging
import re

logger = logging.getLogger(__name__)


def apply_config(
    kernel_code: str,
    config: dict[str, int | float],
    params: dict[str, TuningParam],
) -> str:
    """Replace tiling parameter values in kernel code. Validates syntax after substitution."""
    result = kernel_code
    for name, value in config.items():
        param = params.get(name)
        marker = param.marker if param and param.marker else f"{name} ="
        escaped = re.escape(marker)
        pattern = rf"({escaped}\s*)(\d+(?:\.\d+)?)"
        matches = list(re.finditer(pattern, result))

        if len(matches) == 0:
            logger.warning(f"Marker for '{name}' not found in kernel code, skipping")
            continue
        if len(matches) > 1:
            raise ValueError(
                f"Marker for '{name}' is ambiguous: found {len(matches)} matches"
            )

        match = matches[0]
        replacement = f"{match.group(1)}{value}"
        result = result[: match.start()] + replacement + result[match.end() :]

    ast.parse(result)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_tuning.py::TestApplyConfig -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add kernel-evolve/src/kernel_evolve/tuning.py kernel-evolve/tests/test_tuning.py
git commit -m "feat(tuning): implement apply_config with marker-based substitution"
```

---

### Task 4: expand_variants

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/tuning.py`
- Test: `kernel-evolve/tests/test_tuning.py`

- [ ] **Step 1: Write failing tests for expand_variants**

Add to `kernel-evolve/tests/test_tuning.py`:

```python
from kernel_evolve.tuning import expand_variants


class TestExpandVariants:
    def test_basic_expansion(self):
        tc = TuningConfig(
            params={
                "BLOCK_K": TuningParam(values=[64, 128]),
            }
        )
        code = "BLOCK_K = 256\ndef f(): pass\n"
        expanded = expand_variants([code], tc)
        assert len(expanded) == 2
        assert "BLOCK_K = 64" in expanded[0][0]
        assert expanded[0][1] == {"BLOCK_K": 64}
        assert "BLOCK_K = 128" in expanded[1][0]
        assert expanded[1][1] == {"BLOCK_K": 128}

    def test_multiple_variants(self):
        tc = TuningConfig(
            params={"BLOCK_K": TuningParam(values=[64, 128])},
        )
        v1 = "BLOCK_K = 256\ndef f(): pass\n"
        v2 = "BLOCK_K = 512\ndef g(): pass\n"
        expanded = expand_variants([v1, v2], tc)
        # 2 variants * 2 configs = 4
        assert len(expanded) == 4

    def test_disabled_returns_originals(self):
        tc = TuningConfig(
            params={"BLOCK_K": TuningParam(values=[64, 128])},
            enabled=False,
        )
        code = "BLOCK_K = 256\ndef f(): pass\n"
        expanded = expand_variants([code], tc)
        # Disabled: returns original code with empty config
        assert len(expanded) == 1
        assert expanded[0] == (code, {})

    def test_empty_grid_returns_originals(self):
        tc = TuningConfig(
            params={"X": TuningParam(values=[1, 2])},
            constraints=["X > 100"],  # filters everything
        )
        code = "X = 50\ndef f(): pass\n"
        expanded = expand_variants([code], tc)
        assert len(expanded) == 1
        assert expanded[0] == (code, {})

    def test_skips_failed_substitutions(self):
        """Variants where apply_config fails (missing marker) are skipped."""
        tc = TuningConfig(
            params={"NONEXISTENT": TuningParam(values=[1, 2])},
        )
        code = "BLOCK_K = 128\ndef f(): pass\n"
        expanded = expand_variants([code], tc)
        # Marker not found => warning logged, code returned unchanged
        assert len(expanded) == 2
        # Code is unchanged because marker not found (warning only)
        assert expanded[0][0] == code
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_tuning.py::TestExpandVariants -v`
Expected: FAIL with `ImportError: cannot import name 'expand_variants'`

- [ ] **Step 3: Implement expand_variants**

Add to `kernel-evolve/src/kernel_evolve/tuning.py`:

```python
def expand_variants(
    variants: list[str], tuning_config: TuningConfig
) -> list[tuple[str, dict[str, int | float]]]:
    """Expand each variant code with all valid tiling configs.

    Returns list of (modified_code, config_dict) pairs.
    If tuning is disabled or grid is empty, returns originals with empty config.
    """
    configs = generate_configs(tuning_config)
    if not configs:
        return [(v, {}) for v in variants]

    expanded = []
    for variant_code in variants:
        for cfg in configs:
            try:
                modified = apply_config(variant_code, cfg, tuning_config.params)
                expanded.append((modified, cfg))
            except (ValueError, SyntaxError):
                logger.warning(f"Skipping config {cfg}: substitution failed")
                continue
    return expanded
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_tuning.py::TestExpandVariants -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Run all tuning tests together**

Run: `cd kernel-evolve && python -m pytest tests/test_tuning.py -v`
Expected: All tests PASS (5 model + 9 generate + 8 apply + 5 expand = 27 tests)

- [ ] **Step 6: Commit**

```bash
git add kernel-evolve/src/kernel_evolve/tuning.py kernel-evolve/tests/test_tuning.py
git commit -m "feat(tuning): implement expand_variants for post-mutation tiling sweep"
```

---

## Chunk 2: Evaluator Metadata Pipeline

### Task 5: Add metadata field to EvalRequest

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/evaluator.py:88-117`
- Test: `kernel-evolve/tests/test_evaluator.py`

- [ ] **Step 1: Write failing tests for EvalRequest.metadata**

Add to `kernel-evolve/tests/test_evaluator.py`:

```python
def test_eval_request_metadata_default():
    """EvalRequest has empty metadata by default."""
    req = EvalRequest(
        variant_id="v1",
        kernel_code="code",
        reference_code="ref",
        shapes=[{"M": 1024}],
    )
    assert req.metadata == {}


def test_eval_request_metadata_roundtrip():
    """EvalRequest metadata survives to_dict/from_dict roundtrip."""
    req = EvalRequest(
        variant_id="v1",
        kernel_code="code",
        reference_code="ref",
        shapes=[{"M": 1024}],
        metadata={
            "tuning_config": {"BLOCK_K": 256, "BLOCK_V": 128},
            "code_variant_id": "L1_v1",
        },
    )
    d = req.to_dict()
    assert d["metadata"]["tuning_config"] == {"BLOCK_K": 256, "BLOCK_V": 128}
    assert d["metadata"]["code_variant_id"] == "L1_v1"

    restored = EvalRequest.from_dict(d)
    assert restored.metadata == req.metadata


def test_eval_request_metadata_b64_roundtrip():
    """EvalRequest metadata survives encode_b64/decode_b64 roundtrip."""
    req = EvalRequest(
        variant_id="v1",
        kernel_code="code",
        reference_code="ref",
        shapes=[{"M": 1024}],
        metadata={"tuning_config": {"BLOCK_K": 64}},
    )
    encoded = req.encode_b64()
    restored = EvalRequest.decode_b64(encoded)
    assert restored.metadata == {"tuning_config": {"BLOCK_K": 64}}


def test_eval_request_backward_compat():
    """EvalRequest.from_dict works without metadata key (backward compat)."""
    data = {
        "variant_id": "v1",
        "kernel_code": "code",
        "reference_code": "ref",
        "shapes": [{"M": 1024}],
    }
    req = EvalRequest.from_dict(data)
    assert req.metadata == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_evaluator.py::test_eval_request_metadata_default tests/test_evaluator.py::test_eval_request_metadata_roundtrip tests/test_evaluator.py::test_eval_request_metadata_b64_roundtrip tests/test_evaluator.py::test_eval_request_backward_compat -v`
Expected: FAIL — `EvalRequest` does not accept `metadata` argument

- [ ] **Step 3: Add metadata field to EvalRequest**

Modify `kernel-evolve/src/kernel_evolve/evaluator.py`:

Add to `EvalRequest` dataclass (after `atol` field):
```python
  metadata: dict[str, Any] = field(default_factory=dict)
```

Update `to_dict()` method — add after `"atol"` line:
```python
      "metadata": self.metadata,
```

Update `from_dict()` — change from `cls(**data)` to explicit construction:
```python
  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> EvalRequest:
    return cls(
      variant_id=data["variant_id"],
      kernel_code=data["kernel_code"],
      reference_code=data["reference_code"],
      shapes=data["shapes"],
      rtol=data.get("rtol", 1e-2),
      atol=data.get("atol", 1e-2),
      metadata=data.get("metadata", {}),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_evaluator.py -v`
Expected: All tests PASS (existing + 4 new)

- [ ] **Step 5: Commit**

```bash
git add kernel-evolve/src/kernel_evolve/evaluator.py kernel-evolve/tests/test_evaluator.py
git commit -m "feat(evaluator): add metadata field to EvalRequest with serialization support"
```

---

### Task 6: Update BatchEvalRequest to support metadata

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/evaluator.py:120-152`
- Test: `kernel-evolve/tests/test_evaluator.py`

- [ ] **Step 1: Write failing tests for BatchEvalRequest metadata forwarding**

Add to `kernel-evolve/tests/test_evaluator.py`:

```python
def test_batch_eval_request_variants_with_metadata():
    """BatchEvalRequest.to_single_requests forwards variant metadata."""
    batch = BatchEvalRequest(
        reference_code="ref",
        shapes=[{"M": 1024}],
        variants=[
            {
                "variant_id": "L1_v1_t0",
                "kernel_code": "code1",
                "metadata": {
                    "tuning_config": {"BLOCK_K": 64},
                    "code_variant_id": "L1_v1",
                },
            },
            {
                "variant_id": "L1_v1_t1",
                "kernel_code": "code2",
                "metadata": {
                    "tuning_config": {"BLOCK_K": 128},
                    "code_variant_id": "L1_v1",
                },
            },
        ],
    )
    singles = batch.to_single_requests()
    assert len(singles) == 2
    assert singles[0].metadata["tuning_config"] == {"BLOCK_K": 64}
    assert singles[1].metadata["tuning_config"] == {"BLOCK_K": 128}


def test_batch_eval_request_variants_without_metadata():
    """BatchEvalRequest.to_single_requests works without metadata (backward compat)."""
    batch = BatchEvalRequest(
        reference_code="ref",
        shapes=[{"M": 1024}],
        variants=[
            {"variant_id": "v1", "kernel_code": "code1"},
        ],
    )
    singles = batch.to_single_requests()
    assert singles[0].metadata == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd kernel-evolve && python -m pytest tests/test_evaluator.py::test_batch_eval_request_variants_with_metadata tests/test_evaluator.py::test_batch_eval_request_variants_without_metadata -v`
Expected: FAIL — `to_single_requests` doesn't pass `metadata`

- [ ] **Step 3: Update BatchEvalRequest**

Modify `kernel-evolve/src/kernel_evolve/evaluator.py`:

Change type annotation:
```python
  variants: list[dict[str, Any]]
```

Update `to_single_requests()`:
```python
  def to_single_requests(self) -> list[EvalRequest]:
    return [
      EvalRequest(
        variant_id=v["variant_id"],
        kernel_code=v["kernel_code"],
        reference_code=self.reference_code,
        shapes=self.shapes,
        rtol=self.rtol,
        atol=self.atol,
        metadata=v.get("metadata", {}),
      )
      for v in self.variants
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd kernel-evolve && python -m pytest tests/test_evaluator.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add kernel-evolve/src/kernel_evolve/evaluator.py kernel-evolve/tests/test_evaluator.py
git commit -m "feat(evaluator): update BatchEvalRequest to forward variant metadata"
```

---

### Task 7: Update batch_dispatch to forward metadata

**Files:**
- Modify: `kernel-evolve/src/kernel_evolve/docker_evaluate_helpers.py:39-48`
- Test: `kernel-evolve/tests/test_docker_evaluate.py`

- [ ] **Step 1: Write failing test for metadata forwarding in batch_dispatch**

First, read `kernel-evolve/tests/test_docker_evaluate.py` to understand existing test patterns. Then add:

```python
def test_batch_dispatch_forwards_metadata(tmp_path):
    """batch_dispatch includes variant metadata in single-variant payloads."""
    # Create a mock evaluator script that echoes the decoded payload's metadata
    script = tmp_path / "echo_eval.py"
    script.write_text('''\
import sys, json, base64
payload = json.loads(base64.b64decode(sys.argv[2]).decode())
meta = payload.get("metadata", {})
result = {"status": "SUCCESS", "variant_id": payload["variant_id"],
          "fitness": 1.0, "latency_ms": 1.0, "speedup": 1.0,
          "metadata": {"forwarded_meta": meta}}
print(f"EVAL_RESULT:{json.dumps(result)}")
''')

    payload = {
        "reference_code": "ref",
        "shapes": [{"M": 1024}],
        "variants": [
            {
                "variant_id": "L1_v1_t0",
                "kernel_code": "code",
                "metadata": {
                    "tuning_config": {"BLOCK_K": 64},
                    "code_variant_id": "L1_v1",
                },
            },
        ],
    }
    results = batch_dispatch(payload, str(script), per_variant_timeout=10)
    assert len(results) == 1
    result_data = json.loads(results[0].split("EVAL_RESULT:", 1)[1])
    forwarded = result_data["metadata"]["forwarded_meta"]
    assert forwarded["tuning_config"] == {"BLOCK_K": 64}
    assert forwarded["code_variant_id"] == "L1_v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd kernel-evolve && python -m pytest tests/test_docker_evaluate.py::test_batch_dispatch_forwards_metadata -v`
Expected: FAIL — metadata not included in the single-variant payload

- [ ] **Step 3: Update batch_dispatch to forward metadata**

Modify `kernel-evolve/src/kernel_evolve/docker_evaluate_helpers.py`, in `batch_dispatch()` at the `single_payload` construction (lines 41-48):

```python
    single_payload = {
      "variant_id": variant_id,
      "kernel_code": variant["kernel_code"],
      "reference_code": reference_code,
      "shapes": shapes,
      "rtol": rtol,
      "atol": atol,
      "metadata": variant.get("metadata", {}),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd kernel-evolve && python -m pytest tests/test_docker_evaluate.py::test_batch_dispatch_forwards_metadata -v`
Expected: PASS

- [ ] **Step 5: Run all docker evaluate tests**

Run: `cd kernel-evolve && python -m pytest tests/test_docker_evaluate.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add kernel-evolve/src/kernel_evolve/docker_evaluate_helpers.py kernel-evolve/tests/test_docker_evaluate.py
git commit -m "feat(evaluator): forward variant metadata through batch_dispatch"
```

---

### Task 8: Update docker/evaluate.py to merge request metadata into result

**Files:**
- Modify: `kernel-evolve/docker/evaluate.py:817-835`

- [ ] **Step 1: Identify the metadata merge point**

In `kernel-evolve/docker/evaluate.py`, the SUCCESS result is constructed at lines 817-835. The `metadata` dict is built at lines 826-834. We need to merge any request metadata into this dict.

The request is available as `request` (decoded at line 718 in `main()`). We need to:
1. Read `request.get("metadata", {})` early.
2. Merge it into the result metadata.

- [ ] **Step 2: Add request metadata forwarding**

At the top of `main()` (after line 718 where `request` is decoded), add:

```python
  request_metadata = request.get("metadata", {})
```

Then at the SUCCESS result construction (line 826), merge:

```python
    "metadata": {
        **request_metadata,
        "reference_latency_ms": ref_latency,
        ...existing fields...
    },
```

Also add `request_metadata` to the COMPILE_ERROR and INCORRECT result dicts so metadata always propagates:

For COMPILE_ERROR (line 727-729):
```python
    err = {
        "status": "COMPILE_ERROR",
        "variant_id": job_name,
        "error": compile_result["error"],
        "metadata": request_metadata,
    }
```

For INCORRECT (lines 740-747), add `"metadata": request_metadata` to the dict.

For the perf failure COMPILE_ERROR (lines 752-753), add `"metadata": request_metadata` to the dict.

- [ ] **Step 3: Run existing docker evaluate tests**

Run: `cd kernel-evolve && python -m pytest tests/test_docker_evaluate.py -v`
Expected: All tests PASS (no behavior change for payloads without metadata)

- [ ] **Step 4: Commit**

```bash
git add kernel-evolve/docker/evaluate.py
git commit -m "feat(evaluator): merge request metadata into all EVAL_RESULT outputs"
```

---

## Chunk 3: Integration Test

### Task 9: End-to-end integration test

**Files:**
- Modify: `kernel-evolve/tests/test_batch_integration.py`

- [ ] **Step 1: Write integration test for tuning expansion through batch pipeline**

Add to `kernel-evolve/tests/test_batch_integration.py` (note: `base64`, `json`, and `BatchEvalRequest` are already imported at the top of this file):

```python
from kernel_evolve.tuning import TuningConfig, TuningParam, expand_variants


def test_tuning_expansion_through_batch_pipeline():
    """End-to-end: expand variants with tuning configs, build batch request, verify metadata."""
    # Define tuning config
    tc = TuningConfig(
        params={
            "BLOCK_K": TuningParam(values=[64, 128]),
            "BLOCK_V": TuningParam(values=[32, 64]),
        },
        constraints=["BLOCK_K >= BLOCK_V"],
    )

    # Two LLM-generated code variants
    v1_code = "BLOCK_K = 256\nBLOCK_V = 256\ndef kernel_v1(): pass\n"
    v2_code = "BLOCK_K = 512\nBLOCK_V = 128\ndef kernel_v2(): pass\n"

    # Expand
    expanded = expand_variants([v1_code, v2_code], tc)
    # 4 configs (all pass constraint: 64>=32, 64>=64, 128>=32, 128>=64) * 2 variants = 8
    assert len(expanded) == 8

    # Build BatchEvalRequest with metadata
    variants_for_batch = []
    code_variants = [("L1_v1", v1_code), ("L2_v2", v2_code)]
    for code_variant_id, original_code in code_variants:
        variant_expanded = expand_variants([original_code], tc)
        for idx, (modified_code, cfg) in enumerate(variant_expanded):
            variants_for_batch.append({
                "variant_id": f"{code_variant_id}_t{idx}",
                "kernel_code": modified_code,
                "metadata": {
                    "tuning_config": cfg,
                    "code_variant_id": code_variant_id,
                },
            })

    batch = BatchEvalRequest(
        reference_code="ref",
        shapes=[{"M": 1024}],
        variants=variants_for_batch,
    )

    # Verify batch decomposes correctly
    singles = batch.to_single_requests()
    assert len(singles) == 8

    # Check metadata flows through
    for req in singles:
        assert "tuning_config" in req.metadata
        assert "code_variant_id" in req.metadata

    # Check specific variant: metadata and code substitution
    t0 = next(r for r in singles if r.variant_id == "L1_v1_t0")
    assert t0.metadata["code_variant_id"] == "L1_v1"
    assert t0.metadata["tuning_config"] == {"BLOCK_K": 64, "BLOCK_V": 32}

    # Verify kernel code was actually modified by apply_config
    assert "BLOCK_K = 64" in t0.kernel_code
    assert "BLOCK_V = 32" in t0.kernel_code
    assert "def kernel_v1" in t0.kernel_code  # original code structure preserved

    # Verify payload roundtrip
    encoded = batch.encode_b64()
    decoded = json.loads(base64.b64decode(encoded).decode())
    assert decoded["variants"][0]["metadata"]["tuning_config"] is not None
```

- [ ] **Step 2: Run integration test**

Run: `cd kernel-evolve && python -m pytest tests/test_batch_integration.py::test_tuning_expansion_through_batch_pipeline -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `cd kernel-evolve && python -m pytest tests/ -v --ignore=tests/standalone_chunk_gla_test.py --ignore=tests/standalone_gmm_fp8_blockwise_test.py`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add kernel-evolve/tests/test_batch_integration.py
git commit -m "test(tuning): add end-to-end integration test for tuning expansion pipeline"
```

---

## Summary

| Task | Component | New/Modify | Tests |
|------|-----------|------------|-------|
| 1 | TuningParam/TuningConfig models + config | New `tuning.py`, Modify `config.py` | 5 model + 3 config |
| 2 | `generate_configs` | Modify `tuning.py` | 9 tests |
| 3 | `apply_config` | Modify `tuning.py` | 8 tests |
| 4 | `expand_variants` | Modify `tuning.py` | 5 tests |
| 5 | `EvalRequest.metadata` | Modify `evaluator.py` | 4 tests |
| 6 | `BatchEvalRequest` metadata | Modify `evaluator.py` | 2 tests |
| 7 | `batch_dispatch` metadata | Modify `docker_evaluate_helpers.py` | 1 test |
| 8 | `docker/evaluate.py` metadata | Modify `docker/evaluate.py` | 0 (covered by existing) |
| 9 | Integration test | Modify `test_batch_integration.py` | 1 test |
| **Total** | | **1 new file, 6 modified** | **38 tests** |
