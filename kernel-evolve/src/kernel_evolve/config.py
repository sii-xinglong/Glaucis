"""YAML config parsing and validation with Pydantic."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from kernel_evolve.tuning import TuningConfig


class EvolveMarkers(BaseModel):
  start: str = "# EVOLVE-BLOCK-START"
  end: str = "# EVOLVE-BLOCK-END"


class KernelConfig(BaseModel):
  name: str
  template: str
  reference: str
  evolve_markers: EvolveMarkers = Field(default_factory=EvolveMarkers)


class CorrectnessConfig(BaseModel):
  method: str = "allclose"
  rtol: float = 1e-2
  atol: float = 1e-2


class EvaluatorConfig(BaseModel):
  namespace: str = "default"
  job_template: str = ".github/ci/kernel-eval-job.yaml"
  repo: str = ""
  branch: str = "main"
  poll_interval: int = 15
  timeout: int = 600


class TPUConfig(BaseModel):
  cluster: str
  zone: str


class SessionConfig(BaseModel):
  max_iterations: int = 20
  output_dir: str = "runs/default"


class BatchConfig(BaseModel):
  variants_per_round: int = Field(default=1, ge=1)
  top_k: int = Field(default=1, ge=1)
  max_active_lineages: int = Field(default=4, ge=1)


class EvolveConfig(BaseModel):
  model_config = {"extra": "forbid"}

  kernel: KernelConfig
  shapes: list[dict[str, Any]]
  correctness: CorrectnessConfig = Field(default_factory=CorrectnessConfig)
  evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
  tpu: TPUConfig
  session: SessionConfig = Field(default_factory=SessionConfig)
  batch: BatchConfig = Field(default_factory=BatchConfig)
  tuning_params: TuningConfig | None = None


def load_config(path: str | Path) -> EvolveConfig:
  """Load and validate an EvolveConfig from a YAML file."""
  with open(path) as f:
    data = yaml.safe_load(f)
  return EvolveConfig(**data)
