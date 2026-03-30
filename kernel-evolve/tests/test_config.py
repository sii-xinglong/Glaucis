"""Tests for simplified YAML config parsing and validation."""

from pathlib import Path

import pytest

from kernel_evolve.config import BatchConfig, EvolveConfig, SessionConfig, load_config


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
  assert cfg.tpu.cluster == "tpu7x-cluster"
  assert cfg.session.max_iterations == 20
  assert cfg.session.output_dir == "runs/matmul"


def test_config_defaults():
  cfg = EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64, "N": 64, "K": 64}],
    tpu={"cluster": "c", "zone": "z"},
  )
  assert cfg.correctness.method == "allclose"
  assert cfg.evaluator.namespace == "default"
  assert cfg.evaluator.poll_interval == 15
  assert cfg.evaluator.timeout == 600
  assert cfg.session.max_iterations == 20
  assert cfg.session.output_dir == "runs/default"


def test_config_with_evaluator():
  cfg = EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64}],
    tpu={"cluster": "c", "zone": "z"},
    evaluator={
      "namespace": "custom-ns",
      "job_template": "custom-template.yaml",
      "repo": "user/repo",
      "branch": "dev",
      "poll_interval": 30,
      "timeout": 1200,
    },
  )
  assert cfg.evaluator.namespace == "custom-ns"
  assert cfg.evaluator.repo == "user/repo"
  assert cfg.evaluator.poll_interval == 30


def test_config_with_session():
  cfg = EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64}],
    tpu={"cluster": "c", "zone": "z"},
    session={"max_iterations": 50, "output_dir": "runs/custom"},
  )
  assert cfg.session.max_iterations == 50
  assert cfg.session.output_dir == "runs/custom"


def test_session_config_defaults():
  s = SessionConfig()
  assert s.max_iterations == 20
  assert s.output_dir == "runs/default"


def test_config_no_evolution_or_llm_fields():
  """Verify old fields (evolution, llm, logging) are not accepted."""
  with pytest.raises(ValueError):
    EvolveConfig(
      kernel={"name": "t", "template": "k.py", "reference": "r.py"},
      shapes=[{"M": 64}],
      tpu={"cluster": "c", "zone": "z"},
      evolution={"population_size": 25},
    )


def test_config_with_batch():
  cfg = EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64}],
    tpu={"cluster": "c", "zone": "z"},
    batch={
      "variants_per_round": 5,
      "top_k": 2,
      "max_active_lineages": 4,
      "diversity_directions": ["tiling_strategy", "pipeline_depth"],
    },
  )
  assert cfg.batch.variants_per_round == 5
  assert cfg.batch.top_k == 2
  assert cfg.batch.max_active_lineages == 4
  assert cfg.batch.diversity_directions == ["tiling_strategy", "pipeline_depth"]


def test_batch_config_defaults():
  cfg = EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64}],
    tpu={"cluster": "c", "zone": "z"},
  )
  assert cfg.batch.variants_per_round == 1
  assert cfg.batch.top_k == 1
  assert cfg.batch.max_active_lineages == 4
  assert cfg.batch.diversity_directions == []


def test_load_matmul_config_has_batch(example_config_path):
  cfg = load_config(example_config_path)
  assert cfg.batch.variants_per_round == 5
  assert cfg.batch.top_k == 2


def test_batch_config_loaded_from_yaml(tmp_path):
  yaml_content = """
kernel:
  name: "test"
  template: "k.py"
  reference: "r.py"
shapes:
  - { M: 64 }
tpu:
  cluster: "c"
  zone: "z"
batch:
  variants_per_round: 3
  top_k: 2
"""
  f = tmp_path / "config.yaml"
  f.write_text(yaml_content)
  cfg = load_config(f)
  assert cfg.batch.variants_per_round == 3
  assert cfg.batch.top_k == 2
