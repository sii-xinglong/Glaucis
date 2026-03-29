"""Evolution engine: MAP-Elites loop with island model and stagnation detection."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from kernel_evolve.config import EvolveConfig
from kernel_evolve.evaluator import EvalRequest, EvalResult, EvalStatus, Evaluator
from kernel_evolve.llm.base import LLMProvider, MutationRequest
from kernel_evolve.mutation import extract_evolve_block, inject_evolve_block, validate_syntax
from kernel_evolve.perf_log import PerfLog
from kernel_evolve.population import Archive, BehaviorDescriptor, Variant

FOCUS_AREAS = ["tiling", "memory_access", "compute_optimization", "vectorization", "pipelining"]


class EvolutionEngine:
  def __init__(
    self,
    config: EvolveConfig,
    provider: LLMProvider,
    evaluator: Evaluator,
    template_code: str,
    reference_code: str,
  ):
    self._config = config
    self._provider = provider
    self._evaluator = evaluator
    self._template_code = template_code
    self._reference_code = reference_code
    self._generation = 0
    self._stagnation_count = 0
    self._best_fitness = 0.0
    self._optimization_history: list[str] = []
    self._failed_attempts: list[str] = []
    self._base_temperature = config.llm.temperature

    self._islands: list[Archive] = [Archive() for _ in range(config.evolution.num_islands)]

    self._output_dir = Path(config.logging.output_dir)
    self._output_dir.mkdir(parents=True, exist_ok=True)
    (self._output_dir / "population" / "variants").mkdir(parents=True, exist_ok=True)
    (self._output_dir / "best").mkdir(parents=True, exist_ok=True)
    (self._output_dir / "failed").mkdir(parents=True, exist_ok=True)

    self._perf_log = PerfLog(self._output_dir / "perf_log.md", config.kernel.name)
    if config.logging.perf_log:
      self._perf_log.write_header()

  @property
  def best(self) -> Variant | None:
    candidates = [island.best for island in self._islands if island.best is not None]
    return max(candidates, key=lambda v: v.fitness) if candidates else None

  async def run(self) -> Variant | None:
    for _ in range(self._config.evolution.max_generations):
      await self.run_generation()
    return self.best

  async def run_generation(self) -> None:
    self._generation += 1
    gen_entries: list[dict[str, Any]] = []
    gen_best_id = ""
    gen_best_fitness = 0.0

    for island_idx, island in enumerate(self._islands):
      variant, result, explanation = await self._evolve_one(island)
      if variant is None:
        continue

      entry = {
        "variant_id": variant.id,
        "descriptor": variant.descriptor,
        "result": result,
        "explanation": explanation,
      }
      gen_entries.append(entry)

      variant_path = self._output_dir / "population" / "variants" / f"{variant.id}.py"
      variant_path.write_text(variant.code)

      if result.status == EvalStatus.SUCCESS and result.speedup > gen_best_fitness:
        gen_best_id = variant.id
        gen_best_fitness = result.speedup

      status_str = result.status.name.lower()
      self._optimization_history.append(
        f"Gen {self._generation}: {variant.id} ({variant.descriptor.block_size}/"
        f"{variant.descriptor.pipeline_stages}/{variant.descriptor.memory_strategy}) "
        f"-> {status_str} {f'{result.speedup:.2f}x' if result.status == EvalStatus.SUCCESS else result.error[:30]}"
      )

      if result.status != EvalStatus.SUCCESS:
        self._failed_attempts.append(f"{explanation}: {result.error[:80]}")
        error_path = self._output_dir / "failed" / f"{variant.id}.log"
        error_path.write_text(f"Status: {result.status.name}\nError: {result.error}\n")

    if self._config.logging.perf_log and gen_entries:
      self._perf_log.log_generation(self._generation, gen_entries, gen_best_id or "none", gen_best_fitness)

    if gen_best_fitness > self._best_fitness:
      self._best_fitness = gen_best_fitness
      self._stagnation_count = 0
      best = self.best
      if best:
        (self._output_dir / "best" / "kernel.py").write_text(best.code)
    else:
      self._stagnation_count += 1

    if self._stagnation_count >= self._config.evolution.stagnation_limit:
      self._handle_stagnation()

    if len(self._islands) > 1 and self._generation % self._config.evolution.migration_interval == 0:
      self._migrate()

    for i, island in enumerate(self._islands):
      island.save(self._output_dir / "population" / f"archive_island_{i}.json")

  async def _evolve_one(self, island: Archive) -> tuple[Variant | None, EvalResult, str]:
    if island.size == 0:
      parent_code = self._template_code
      parent_fitness = 0.0
    else:
      parent = island.tournament_select(k=3)
      parent_code = parent.code
      parent_fitness = parent.fitness

    focus_hint = None
    if self._stagnation_count > 0:
      focus_idx = self._stagnation_count % len(FOCUS_AREAS)
      focus_hint = FOCUS_AREAS[focus_idx]

    try:
      evolve_block = extract_evolve_block(parent_code)
    except ValueError:
      evolve_block = parent_code

    request = MutationRequest(
      parent_code=evolve_block,
      fitness=parent_fitness,
      generation=self._generation,
      optimization_history=self._optimization_history[-10:],
      failed_attempts=self._failed_attempts[-5:],
      focus_hint=focus_hint,
    )

    try:
      response = await self._provider.mutate(request)
    except Exception as e:
      return None, EvalResult.compile_error(f"LLM error: {e}"), ""

    try:
      full_code = inject_evolve_block(self._template_code, response.mutated_code)
    except ValueError:
      full_code = response.mutated_code

    if not validate_syntax(full_code):
      result = EvalResult.compile_error("Invalid Python syntax")
      return None, result, response.explanation

    desc_data = response.suggested_descriptor
    descriptor = BehaviorDescriptor(
      block_size=desc_data.get("block_size", 128),
      pipeline_stages=desc_data.get("pipeline_stages", 1),
      memory_strategy=desc_data.get("memory_strategy", "scratch"),
    )

    variant_id = f"v{self._generation:03d}_{uuid.uuid4().hex[:6]}"
    variant = Variant(
      id=variant_id,
      code=full_code,
      descriptor=descriptor,
      fitness=0.0,
      generation=self._generation,
      parent_id=None,
    )

    eval_request = EvalRequest(
      variant_id=variant_id,
      kernel_code=full_code,
      reference_code=self._reference_code,
      shapes=self._config.shapes,
      rtol=self._config.correctness.rtol,
      atol=self._config.correctness.atol,
    )
    result = await self._evaluator.evaluate(eval_request)

    variant.fitness = result.fitness
    island.insert(variant)

    return variant, result, response.explanation

  def _handle_stagnation(self) -> None:
    # Placeholder for future stagnation-breaking strategies (e.g., increased
    # temperature, population reset, or archive pruning).  The counter is NOT
    # reset here so callers can observe cumulative stagnation.
    pass

  def _migrate(self) -> None:
    tops = [island.best for island in self._islands if island.best is not None]
    for island in self._islands:
      for top in tops:
        if top is not None:
          island.insert(top)
