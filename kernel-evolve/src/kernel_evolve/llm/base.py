"""Abstract LLM provider interface for kernel mutation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MutationRequest:
  parent_code: str
  fitness: float
  generation: int
  optimization_history: list[str] = field(default_factory=list)
  failed_attempts: list[str] = field(default_factory=list)
  focus_hint: str | None = None


@dataclass
class MutationResponse:
  mutated_code: str
  explanation: str = ""
  suggested_descriptor: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
  @abstractmethod
  async def mutate(self, request: MutationRequest) -> MutationResponse: ...
