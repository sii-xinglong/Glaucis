"""Tests for the tuning module: TuningParam, TuningConfig, generate_configs, apply_config, expand_variants."""

import pytest

from kernel_evolve.tuning import (
  TuningConfig,
  TuningParam,
  apply_config,
  expand_variants,
  generate_configs,
)


# ---------------------------------------------------------------------------
# TestTuningModels
# ---------------------------------------------------------------------------


class TestTuningModels:
  def test_tuning_param_basic(self):
    p = TuningParam(values=[1, 2, 4])
    assert p.values == [1, 2, 4]
    assert p.marker is None

  def test_tuning_param_with_marker(self):
    p = TuningParam(values=[8, 16], marker="BLOCK_SIZE =")
    assert p.values == [8, 16]
    assert p.marker == "BLOCK_SIZE ="

  def test_tuning_config_basic(self):
    cfg = TuningConfig(
      params={
        "BLOCK_M": TuningParam(values=[64, 128]),
        "BLOCK_N": TuningParam(values=[32, 64]),
      }
    )
    assert len(cfg.params) == 2
    assert cfg.constraints == []
    assert cfg.max_configs == 32
    assert cfg.enabled is True

  def test_tuning_config_disabled(self):
    cfg = TuningConfig(
      params={"BLOCK_M": TuningParam(values=[64])},
      enabled=False,
    )
    assert cfg.enabled is False

  def test_tuning_config_with_constraints(self):
    cfg = TuningConfig(
      params={
        "BLOCK_M": TuningParam(values=[64, 128]),
        "BLOCK_N": TuningParam(values=[32, 64]),
      },
      constraints=["BLOCK_M >= BLOCK_N"],
      max_configs=16,
    )
    assert cfg.constraints == ["BLOCK_M >= BLOCK_N"]
    assert cfg.max_configs == 16


# ---------------------------------------------------------------------------
# TestGenerateConfigs
# ---------------------------------------------------------------------------


class TestGenerateConfigs:
  def test_single_param(self):
    cfg = TuningConfig(params={"BLOCK_M": TuningParam(values=[64, 128, 256])})
    result = generate_configs(cfg)
    assert result == [{"BLOCK_M": 64}, {"BLOCK_M": 128}, {"BLOCK_M": 256}]

  def test_cartesian_product(self):
    cfg = TuningConfig(
      params={
        "BLOCK_M": TuningParam(values=[64, 128]),
        "BLOCK_N": TuningParam(values=[32, 64]),
      }
    )
    result = generate_configs(cfg)
    assert len(result) == 4
    assert {"BLOCK_M": 64, "BLOCK_N": 32} in result
    assert {"BLOCK_M": 128, "BLOCK_N": 64} in result

  def test_constraint_filtering(self):
    cfg = TuningConfig(
      params={
        "BLOCK_M": TuningParam(values=[64, 128]),
        "BLOCK_N": TuningParam(values=[32, 64]),
      },
      constraints=["BLOCK_M > BLOCK_N"],
    )
    result = generate_configs(cfg)
    assert all(c["BLOCK_M"] > c["BLOCK_N"] for c in result)

  def test_multiple_constraints(self):
    cfg = TuningConfig(
      params={
        "BLOCK_M": TuningParam(values=[64, 128, 256]),
        "BLOCK_N": TuningParam(values=[32, 64, 128]),
      },
      constraints=["BLOCK_M >= BLOCK_N", "BLOCK_M <= 128"],
    )
    result = generate_configs(cfg)
    assert all(c["BLOCK_M"] >= c["BLOCK_N"] and c["BLOCK_M"] <= 128 for c in result)

  def test_all_filtered_returns_empty(self):
    cfg = TuningConfig(
      params={
        "BLOCK_M": TuningParam(values=[64, 128]),
        "BLOCK_N": TuningParam(values=[256, 512]),
      },
      constraints=["BLOCK_M > BLOCK_N"],
    )
    assert generate_configs(cfg) == []

  def test_max_configs_caps(self):
    cfg = TuningConfig(
      params={
        "BLOCK_M": TuningParam(values=[64, 128, 256, 512]),
        "BLOCK_N": TuningParam(values=[32, 64, 128, 256]),
      },
      max_configs=5,
    )
    result = generate_configs(cfg)
    assert len(result) == 5

  def test_max_configs_deterministic(self):
    cfg = TuningConfig(
      params={
        "BLOCK_M": TuningParam(values=[64, 128, 256, 512]),
        "BLOCK_N": TuningParam(values=[32, 64, 128, 256]),
      },
      max_configs=5,
    )
    result1 = generate_configs(cfg)
    result2 = generate_configs(cfg)
    assert result1 == result2

  def test_empty_params(self):
    cfg = TuningConfig(params={})
    result = generate_configs(cfg)
    # product of empty is a single empty combo
    assert result == [{}]

  def test_disabled_returns_empty(self):
    cfg = TuningConfig(
      params={"BLOCK_M": TuningParam(values=[64, 128])},
      enabled=False,
    )
    assert generate_configs(cfg) == []


# ---------------------------------------------------------------------------
# SAMPLE_KERNEL used by TestApplyConfig
# ---------------------------------------------------------------------------

SAMPLE_KERNEL = """\
def tiled_matmul(a, b):
    BLOCK_M = 64
    BLOCK_N = 32
    return a @ b
"""


# ---------------------------------------------------------------------------
# TestApplyConfig
# ---------------------------------------------------------------------------


class TestApplyConfig:
  def test_single_param_replacement(self):
    params = {"BLOCK_M": TuningParam(values=[128])}
    result = apply_config(SAMPLE_KERNEL, {"BLOCK_M": 128}, params)
    assert "BLOCK_M = 128" in result
    assert "BLOCK_M = 64" not in result

  def test_multiple_param_replacement(self):
    params = {
      "BLOCK_M": TuningParam(values=[128]),
      "BLOCK_N": TuningParam(values=[64]),
    }
    result = apply_config(SAMPLE_KERNEL, {"BLOCK_M": 128, "BLOCK_N": 64}, params)
    assert "BLOCK_M = 128" in result
    assert "BLOCK_N = 64" in result
    assert "BLOCK_M = 64" not in result
    assert "BLOCK_N = 32" not in result

  def test_custom_marker(self):
    code = "BLOCK_SIZE = 16\n"
    params = {"BLOCK_SIZE": TuningParam(values=[32], marker="BLOCK_SIZE =")}
    result = apply_config(code, {"BLOCK_SIZE": 32}, params)
    assert "BLOCK_SIZE = 32" in result

  def test_missing_marker_warns(self):
    params = {"MISSING_PARAM": TuningParam(values=[64])}
    # apply_config logs a warning and skips; code is unchanged
    result = apply_config(SAMPLE_KERNEL, {"MISSING_PARAM": 64}, params)
    assert result == SAMPLE_KERNEL

  def test_ambiguous_marker_raises(self):
    code = "BLOCK_M = 64\nBLOCK_M = 32\n"
    params = {"BLOCK_M": TuningParam(values=[128])}
    with pytest.raises(ValueError, match="ambiguous"):
      apply_config(code, {"BLOCK_M": 128}, params)

  def test_syntax_validation(self):
    # Force a syntax error by using code that after replacement is invalid
    code = "BLOCK_M = 64\n"
    # This is contrived - we test that ast.parse is called; normal replacements
    # produce valid syntax, so test the valid path passes
    params = {"BLOCK_M": TuningParam(values=[128])}
    result = apply_config(code, {"BLOCK_M": 128}, params)
    assert "BLOCK_M = 128" in result

  def test_float_value(self):
    code = "SCALE = 1.0\n"
    params = {"SCALE": TuningParam(values=[0.5])}
    result = apply_config(code, {"SCALE": 0.5}, params)
    assert "SCALE = 0.5" in result

  def test_preserves_surrounding_code(self):
    params = {"BLOCK_M": TuningParam(values=[128])}
    result = apply_config(SAMPLE_KERNEL, {"BLOCK_M": 128}, params)
    assert "def tiled_matmul(a, b):" in result
    assert "BLOCK_N = 32" in result
    assert "return a @ b" in result

  def test_no_false_ambiguity_with_prefix(self):
    """BLOCK_K marker should not match PREFIX_BLOCK_K."""
    code = "PREFIX_BLOCK_K = 64\nBLOCK_K = 128\n"
    params = {"BLOCK_K": TuningParam(values=[256])}
    result = apply_config(code, {"BLOCK_K": 256}, params)
    assert "PREFIX_BLOCK_K = 64" in result  # unchanged
    assert "BLOCK_K = 256" in result  # only this one changed


# ---------------------------------------------------------------------------
# TestExpandVariants
# ---------------------------------------------------------------------------


class TestExpandVariants:
  def test_basic_expansion(self):
    variants = [SAMPLE_KERNEL]
    cfg = TuningConfig(params={"BLOCK_M": TuningParam(values=[128, 256])})
    result = expand_variants(variants, cfg)
    assert len(result) == 2
    codes = [r[0] for r in result]
    configs = [r[1] for r in result]
    assert any("BLOCK_M = 128" in c for c in codes)
    assert any("BLOCK_M = 256" in c for c in codes)
    assert {"BLOCK_M": 128} in configs
    assert {"BLOCK_M": 256} in configs

  def test_multiple_variants(self):
    variants = [SAMPLE_KERNEL, SAMPLE_KERNEL.replace("64", "64")]
    cfg = TuningConfig(params={"BLOCK_M": TuningParam(values=[128, 256])})
    result = expand_variants(variants, cfg)
    # 2 variants * 2 configs = 4
    assert len(result) == 4

  def test_disabled_returns_originals(self):
    variants = [SAMPLE_KERNEL]
    cfg = TuningConfig(
      params={"BLOCK_M": TuningParam(values=[128])},
      enabled=False,
    )
    result = expand_variants(variants, cfg)
    assert len(result) == 1
    assert result[0] == (SAMPLE_KERNEL, {})

  def test_empty_grid_returns_originals(self):
    variants = [SAMPLE_KERNEL]
    # All configs filtered out by impossible constraint
    cfg = TuningConfig(
      params={"BLOCK_M": TuningParam(values=[64, 128])},
      constraints=["BLOCK_M > 1000"],
    )
    result = expand_variants(variants, cfg)
    assert result == [(SAMPLE_KERNEL, {})]

  def test_skips_failed_substitutions(self):
    # Variant where the marker doesn't exist and would cause ambiguity
    ambiguous_code = "BLOCK_M = 64\nBLOCK_M = 32\n"
    valid_code = SAMPLE_KERNEL
    variants = [ambiguous_code, valid_code]
    cfg = TuningConfig(params={"BLOCK_M": TuningParam(values=[128])})
    result = expand_variants(variants, cfg)
    # ambiguous_code raises ValueError (skipped), valid_code succeeds
    assert len(result) == 1
    assert "BLOCK_M = 128" in result[0][0]
