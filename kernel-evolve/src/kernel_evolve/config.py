"""YAML config parsing and validation with Pydantic."""

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


class HardwareConfig(BaseModel):
  """Target TPU hardware specs for roofline analysis."""
  chip: str = "v7x"
  peak_flops_tflops: float = 2307.0       # BF16 peak TFLOPS
  fp8_flops_tflops: float | None = None   # FP8 peak (None = 2x bf16)
  peak_hbm_bw_gbps: float = 3690.0        # HBM bandwidth GB/s
  hbm_capacity_gb: float = 192.0          # HBM capacity GB
  vmem_capacity_mib: float = 64.0         # VMEM per chip MiB
  num_mxu: int = 2                        # MXU units (for dual-issue target)


class TilingConstraints(BaseModel):
  """Hard constraints on tile dimensions."""
  tm_must_divide: str | None = None       # e.g. "group_size" — tm must divide this
  tk_must_divide: str | None = None       # e.g. "block_size"
  min_tile: int = 64                      # minimum tile dimension
  max_tile: int = 1024                    # maximum tile dimension
  tk_multiple_of_block_size: bool = True  # tk must be a multiple of block_size


class ConstraintsConfig(BaseModel):
  """Optimization constraints that persist across rounds."""
  vmem_budget_mib: float | None = None       # max VMEM usage (None = no limit)
  compile_time_budget_s: float | None = None # max XLA compile time
  tiling: TilingConstraints = Field(default_factory=TilingConstraints)
  hard_rules: list[str] = Field(default_factory=list)
  performance_targets: dict[str, float] = Field(default_factory=dict)


class EvolveConfig(BaseModel):
  model_config = {"extra": "forbid"}

  kernel: KernelConfig
  shapes: list[dict[str, Any]]
  correctness: CorrectnessConfig = Field(default_factory=CorrectnessConfig)
  evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
  tpu: TPUConfig
  session: SessionConfig = Field(default_factory=SessionConfig)
  batch: BatchConfig = Field(default_factory=BatchConfig)
  hardware: HardwareConfig = Field(default_factory=HardwareConfig)
  constraints: ConstraintsConfig = Field(default_factory=ConstraintsConfig)


def load_config(path: str | Path) -> EvolveConfig:
  """Load and validate an EvolveConfig from a YAML file."""
  with open(path) as f:
    data = yaml.safe_load(f)
  return EvolveConfig(**data)
