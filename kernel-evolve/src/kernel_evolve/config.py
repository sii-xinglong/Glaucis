"""YAML config parsing and validation with Pydantic."""

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


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


class EvolutionConfig(BaseModel):
  population_size: int = 25
  num_islands: int = 3
  max_generations: int = 50
  stagnation_limit: int = 10
  fitness: str = "speedup"
  migration_interval: int = 5


class LLMProvider(str, Enum):
  anthropic = "anthropic"
  google = "google"
  openai = "openai"


class EvaluatorType(str, Enum):
  kube = "kube"
  ci = "ci"


class EvaluatorConfig(BaseModel):
  type: EvaluatorType = EvaluatorType.kube
  namespace: str = "default"
  job_template: str = ".github/ci/kernel-eval-job.yaml"
  repo: str = ""
  branch: str = "main"
  poll_interval: int = 15
  timeout: int = 600


class LLMConfig(BaseModel):
  provider: LLMProvider
  model: str
  temperature: float = 0.7


class TPUConfig(BaseModel):
  cluster: str
  zone: str
  tpu_type: str
  namespace: str = "default"
  image: str
  timeout: int = 300


class LoggingConfig(BaseModel):
  output_dir: str = "runs/default"
  perf_log: bool = True
  charts: bool = True


class EvolveConfig(BaseModel):
  kernel: KernelConfig
  shapes: list[dict[str, Any]]
  correctness: CorrectnessConfig = Field(default_factory=CorrectnessConfig)
  evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
  llm: LLMConfig
  tpu: TPUConfig
  evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
  logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: str | Path) -> EvolveConfig:
  """Load and validate an EvolveConfig from a YAML file."""
  with open(path) as f:
    data = yaml.safe_load(f)
  return EvolveConfig(**data)
