"""Three-stage evaluation pipeline: compile -> correctness -> performance."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class EvalStatus(IntEnum):
  COMPILE_ERROR = 0
  INCORRECT = 1
  SUCCESS = 2


@dataclass
class EvalResult:
  status: EvalStatus
  fitness: float = 0.0
  error: str = ""
  max_diff: float = 0.0
  latency_ms: float = 0.0
  speedup: float = 0.0
  flops: float = 0.0
  metadata: dict[str, Any] = field(default_factory=dict)

  @classmethod
  def compile_error(cls, error: str) -> EvalResult:
    return cls(status=EvalStatus.COMPILE_ERROR, error=error)

  @classmethod
  def incorrect(cls, max_diff: float, error: str = "") -> EvalResult:
    return cls(status=EvalStatus.INCORRECT, max_diff=max_diff, error=error)

  @classmethod
  def success(cls, latency_ms: float, speedup: float, flops: float = 0.0) -> EvalResult:
    return cls(status=EvalStatus.SUCCESS, fitness=speedup, latency_ms=latency_ms, speedup=speedup, flops=flops)

  def to_dict(self) -> dict[str, Any]:
    return {
      "status": self.status.name,
      "fitness": self.fitness,
      "error": self.error,
      "max_diff": self.max_diff,
      "latency_ms": self.latency_ms,
      "speedup": self.speedup,
      "flops": self.flops,
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


class Evaluator:
  async def evaluate(self, request: EvalRequest) -> EvalResult:
    raise NotImplementedError("Subclass must implement evaluate()")
