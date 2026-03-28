"""Tests for YAML config parsing and validation."""

from pathlib import Path

import pytest

from kernel_evolve.config import EvolveConfig, load_config


@pytest.fixture
def example_config_path():
  return Path(__file__).parent.parent / "examples" / "matmul.yaml"


def test_load_config_from_yaml(example_config_path):
  cfg = load_config(example_config_path)
  assert cfg.kernel.name == "tiled_matmul"
  assert cfg.kernel.evolve_markers.start == "# EVOLVE-BLOCK-START"
  assert len(cfg.shapes) == 2
  assert cfg.shapes[0]["M"] == 1024
  assert cfg.correctness.method == "allclose"
  assert cfg.correctness.rtol == pytest.approx(1e-2)
  assert cfg.evolution.population_size == 25
  assert cfg.evolution.num_islands == 3
  assert cfg.llm.provider == "anthropic"
  assert cfg.tpu.tpu_type == "v7x"
  assert cfg.logging.perf_log is True


def test_config_defaults():
  cfg = EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64, "N": 64, "K": 64}],
    llm={"provider": "anthropic", "model": "claude-opus-4-6"},
    tpu={"cluster": "c", "zone": "z", "tpu_type": "v4-8", "image": "img"},
  )
  assert cfg.evolution.population_size == 25
  assert cfg.evolution.max_generations == 50
  assert cfg.evolution.stagnation_limit == 10
  assert cfg.correctness.method == "allclose"
  assert cfg.logging.output_dir == "runs/default"


def test_config_invalid_provider():
  with pytest.raises(ValueError):
    EvolveConfig(
      kernel={"name": "t", "template": "k.py", "reference": "r.py"},
      shapes=[{"M": 64}],
      llm={"provider": "invalid_provider", "model": "m"},
      tpu={"cluster": "c", "zone": "z", "tpu_type": "v4-8", "image": "i"},
    )


def test_config_with_kube_evaluator():
  cfg = EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64}],
    llm={"provider": "anthropic", "model": "claude-opus-4-6"},
    tpu={"cluster": "c", "zone": "z", "tpu_type": "v4-8", "image": "img"},
    evaluator={
      "type": "kube",
      "namespace": "default",
      "job_template": ".github/ci/kernel-eval-job.yaml",
      "repo": "sii-xinglong/Glaucis",
      "branch": "main",
      "poll_interval": 15,
      "timeout": 600,
    },
  )
  assert cfg.evaluator.type.value == "kube"
  assert cfg.evaluator.namespace == "default"
  assert cfg.evaluator.poll_interval == 15


def test_config_evaluator_defaults():
  cfg = EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64}],
    llm={"provider": "anthropic", "model": "claude-opus-4-6"},
    tpu={"cluster": "c", "zone": "z", "tpu_type": "v4-8", "image": "img"},
  )
  assert cfg.evaluator.type.value == "kube"
  assert cfg.evaluator.poll_interval == 15
  assert cfg.evaluator.timeout == 600
