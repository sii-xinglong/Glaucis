"""Four-stage evaluation pipeline: compile -> correctness -> performance -> profile."""

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
  metadata: dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    return {
      "variant_id": self.variant_id,
      "kernel_code": self.kernel_code,
      "reference_code": self.reference_code,
      "shapes": self.shapes,
      "rtol": self.rtol,
      "atol": self.atol,
      "metadata": self.metadata,
    }

  def encode_b64(self) -> str:
    return base64.b64encode(json.dumps(self.to_dict()).encode()).decode()

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

  @classmethod
  def decode_b64(cls, encoded: str) -> EvalRequest:
    data = json.loads(base64.b64decode(encoded).decode())
    return cls.from_dict(data)


@dataclass
class BatchEvalRequest:
  reference_code: str
  shapes: list[dict[str, Any]]
  variants: list[dict[str, Any]]
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
        metadata=v.get("metadata", {}),
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
