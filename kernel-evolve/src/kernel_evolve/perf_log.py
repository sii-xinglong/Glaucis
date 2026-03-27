"""Markdown performance log writer inspired by sparse-mask-attention's perf_log.md."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kernel_evolve.evaluator import EvalResult, EvalStatus
from kernel_evolve.population import BehaviorDescriptor


class PerfLog:
  def __init__(self, path: str | Path, kernel_name: str):
    self.path = Path(path)
    self._kernel_name = kernel_name
    self._cumulative_best_id: str | None = None
    self._cumulative_best_fitness: float = 0.0

  def write_header(self) -> None:
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self.path.write_text(f"# Performance Log: {self._kernel_name}\n\n")

  def log_generation(
    self,
    generation: int,
    entries: list[dict[str, Any]],
    best_id: str,
    best_fitness: float,
  ) -> None:
    if best_fitness > self._cumulative_best_fitness:
      self._cumulative_best_id = best_id
      self._cumulative_best_fitness = best_fitness

    lines = [f"## Generation {generation}\n"]
    lines.append("| Variant | Block | Pipes | Mem | Speedup | Status | Notes |")
    lines.append("|---------|-------|-------|-----|---------|--------|-------|")

    for entry in entries:
      vid = entry["variant_id"]
      desc: BehaviorDescriptor = entry["descriptor"]
      result: EvalResult = entry["result"]
      explanation = entry.get("explanation", "")

      speedup = f"{result.speedup:.2f}x" if result.status == EvalStatus.SUCCESS else "-"
      status = result.status.name.lower()
      notes = explanation[:50] if explanation else (result.error[:50] if result.error else "")

      lines.append(
        f"| {vid} | {desc.block_size} | {desc.pipeline_stages} | {desc.memory_strategy} "
        f"| {speedup} | {status} | {notes} |"
      )

    lines.append(f"\nBest this gen: {best_id} ({best_fitness:.1f}x)")
    lines.append(f"Cumulative best: {self._cumulative_best_id} ({self._cumulative_best_fitness:.1f}x)\n")

    with open(self.path, "a") as f:
      f.write("\n".join(lines) + "\n")
