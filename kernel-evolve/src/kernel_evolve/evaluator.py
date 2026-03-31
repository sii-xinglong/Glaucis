"""Four-stage evaluation pipeline: compile -> correctness -> performance -> profile."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


class EvalStatus(IntEnum):
  COMPILE_ERROR = 0
  INCORRECT = 1
  SUCCESS = 2


@dataclass
class BenchmarkData:
  lower_time_ms: float
  compile_time_ms: float
  evaluation_times_ms: tuple[float, ...]
  peak_memory_mb: float | None
  timing_source: str = "xprof_clustered"

  @property
  def median_ms(self) -> float:
    return float(np.median(self.evaluation_times_ms))

  @property
  def min_ms(self) -> float:
    return float(np.min(self.evaluation_times_ms))

  @property
  def max_ms(self) -> float:
    return float(np.max(self.evaluation_times_ms))

  @property
  def stddev_ms(self) -> float:
    return float(np.std(self.evaluation_times_ms))

  @property
  def cv(self) -> float:
    """Coefficient of variation (stddev / median)."""
    med = self.median_ms
    return self.stddev_ms / med if med > 0 else 0.0

  def to_dict(self) -> dict[str, Any]:
    return {
      "lower_time_ms": self.lower_time_ms,
      "compile_time_ms": self.compile_time_ms,
      "evaluation_times_ms": list(self.evaluation_times_ms),
      "peak_memory_mb": self.peak_memory_mb,
      "median_ms": self.median_ms,
      "min_ms": self.min_ms,
      "max_ms": self.max_ms,
      "stddev_ms": self.stddev_ms,
      "cv": self.cv,
      "timing_source": self.timing_source,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> BenchmarkData:
    return cls(
      lower_time_ms=data.get("lower_time_ms", 0.0),
      compile_time_ms=data.get("compile_time_ms", 0.0),
      evaluation_times_ms=tuple(data.get("evaluation_times_ms", ())),
      peak_memory_mb=data.get("peak_memory_mb"),
      timing_source=data.get("timing_source", "xprof_clustered"),
    )


@dataclass
class EvalResult:
  status: EvalStatus
  fitness: float = 0.0
  error: str = ""
  max_diff: float = 0.0
  latency_ms: float = 0.0
  speedup: float = 0.0
  flops: float = 0.0
  compute_ratio: float | None = None
  memory_transfer_ratio: float | None = None
  metadata: dict[str, Any] = field(default_factory=dict)

  @classmethod
  def compile_error(cls, error: str) -> EvalResult:
    return cls(status=EvalStatus.COMPILE_ERROR, error=error)

  @classmethod
  def incorrect(cls, max_diff: float, error: str = "") -> EvalResult:
    return cls(status=EvalStatus.INCORRECT, max_diff=max_diff, error=error)

  @classmethod
  def success(
    cls,
    latency_ms: float,
    speedup: float,
    flops: float = 0.0,
    compute_ratio: float | None = None,
    memory_transfer_ratio: float | None = None,
  ) -> EvalResult:
    return cls(
      status=EvalStatus.SUCCESS,
      fitness=speedup,
      latency_ms=latency_ms,
      speedup=speedup,
      flops=flops,
      compute_ratio=compute_ratio,
      memory_transfer_ratio=memory_transfer_ratio,
    )

  def to_dict(self) -> dict[str, Any]:
    return {
      "status": self.status.name,
      "fitness": self.fitness,
      "error": self.error,
      "max_diff": self.max_diff,
      "latency_ms": self.latency_ms,
      "speedup": self.speedup,
      "flops": self.flops,
      "compute_ratio": self.compute_ratio,
      "memory_transfer_ratio": self.memory_transfer_ratio,
      "metadata": self.metadata,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> EvalResult:
    return cls(
      status=EvalStatus[data["status"]],
      fitness=data.get("fitness", 0.0),
      error=data.get("error", ""),
      max_diff=data.get("max_diff", 0.0),
      latency_ms=data.get("latency_ms", 0.0),
      speedup=data.get("speedup", 0.0),
      flops=data.get("flops", 0.0),
      compute_ratio=data.get("compute_ratio"),
      memory_transfer_ratio=data.get("memory_transfer_ratio"),
      metadata=data.get("metadata", {}),
    )


@dataclass
class EvalRequest:
  variant_id: str
  kernel_code: str
  reference_code: str
  shapes: list[dict[str, Any]]
  rtol: float = 1e-2
  atol: float = 1e-2

  def to_dict(self) -> dict[str, Any]:
    return {
      "variant_id": self.variant_id,
      "kernel_code": self.kernel_code,
      "reference_code": self.reference_code,
      "shapes": self.shapes,
      "rtol": self.rtol,
      "atol": self.atol,
    }

  def encode_b64(self) -> str:
    return base64.b64encode(json.dumps(self.to_dict()).encode()).decode()

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> EvalRequest:
    return cls(**data)

  @classmethod
  def decode_b64(cls, encoded: str) -> EvalRequest:
    data = json.loads(base64.b64decode(encoded).decode())
    return cls.from_dict(data)


@dataclass
class BatchEvalRequest:
  reference_code: str
  shapes: list[dict[str, Any]]
  variants: list[dict[str, str]]
  rtol: float = 1e-2
  atol: float = 1e-2

  def to_dict(self) -> dict[str, Any]:
    return {
      "batch": True,
      "reference_code": self.reference_code,
      "shapes": self.shapes,
      "variants": self.variants,
      "rtol": self.rtol,
      "atol": self.atol,
    }

  def encode_b64(self) -> str:
    return base64.b64encode(json.dumps(self.to_dict()).encode()).decode()

  def to_single_requests(self) -> list[EvalRequest]:
    return [
      EvalRequest(
        variant_id=v["variant_id"],
        kernel_code=v["kernel_code"],
        reference_code=self.reference_code,
        shapes=self.shapes,
        rtol=self.rtol,
        atol=self.atol,
      )
      for v in self.variants
    ]


@dataclass
class BatchEvalResult:
  results: dict[str, EvalResult]

  def best(self) -> EvalResult | None:
    successes = [r for r in self.results.values() if r.status == EvalStatus.SUCCESS]
    return max(successes, key=lambda r: r.speedup) if successes else None

  def ranked(self) -> list[tuple[str, EvalResult]]:
    return sorted(
      [(vid, r) for vid, r in self.results.items() if r.status == EvalStatus.SUCCESS],
      key=lambda x: x[1].speedup,
      reverse=True,
    )


class Evaluator:
  async def evaluate(self, request: EvalRequest) -> EvalResult:
    raise NotImplementedError("Subclass must implement evaluate()")
