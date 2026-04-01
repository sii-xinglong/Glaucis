"""Tiling parameter grid search: generate configs and apply to kernel code."""

from __future__ import annotations

import ast
import hashlib
import itertools
import logging
import re

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TuningParam(BaseModel):
  values: list[int | float]
  marker: str | None = None


class TuningConfig(BaseModel):
  params: dict[str, TuningParam]
  constraints: list[str] = []
  max_configs: int = 32
  enabled: bool = True


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


def _satisfies_constraints(config: dict[str, int | float], constraints: list[str]) -> bool:
  for constraint in constraints:
    if not eval(constraint, {"__builtins__": {}}, config):  # noqa: S307
      return False
  return True


def _deterministic_sample(configs: list[dict[str, int | float]], n: int) -> list[dict[str, int | float]]:
  """Uniform deterministic sample: sort by sha256 hash, take every k-th."""
  keyed = sorted(configs, key=lambda c: hashlib.sha256(repr(c).encode()).hexdigest())
  step = len(keyed) / n
  return [keyed[int(i * step)] for i in range(n)]


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
    pattern = rf"(?<![A-Za-z0-9_])({escaped}\s*)(\d+(?:\.\d+)?)"
    matches = list(re.finditer(pattern, result))

    if len(matches) == 0:
      logger.warning(f"Marker for '{name}' not found in kernel code, skipping")
      continue
    if len(matches) > 1:
      raise ValueError(f"Marker for '{name}' is ambiguous: found {len(matches)} matches")

    match = matches[0]
    replacement = f"{match.group(1)}{value}"
    result = result[: match.start()] + replacement + result[match.end() :]

  ast.parse(result)
  return result


def expand_variants(variants: list[str], tuning_config: TuningConfig) -> list[tuple[str, dict[str, int | float]]]:
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
