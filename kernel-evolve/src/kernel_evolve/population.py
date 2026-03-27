"""MAP-Elites population archive with tournament selection."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BehaviorDescriptor:
  """Behavioral descriptors that define a cell in the MAP-Elites grid."""

  block_size: int = 128
  pipeline_stages: int = 1
  memory_strategy: str = "scratch"

  def cell_key(self) -> tuple:
    return (self.block_size, self.pipeline_stages, self.memory_strategy)


@dataclass
class Variant:
  """A single kernel variant in the population."""

  id: str
  code: str
  descriptor: BehaviorDescriptor
  fitness: float
  generation: int
  parent_id: str | None
  metadata: dict[str, Any] = field(default_factory=dict)


class Archive:
  """MAP-Elites archive keyed by behavioral descriptors."""

  def __init__(self, descriptor_axes: dict[str, list] | None = None):
    self._descriptor_axes = descriptor_axes or {
      "block_size": [64, 128, 256, 512],
      "pipeline_stages": [1, 2, 3, 4],
      "memory_strategy": ["scratch", "hbm", "rmw"],
    }
    self._grid: dict[tuple, Variant] = {}

  @property
  def capacity(self) -> int:
    result = 1
    for values in self._descriptor_axes.values():
      result *= len(values)
    return result

  @property
  def size(self) -> int:
    return len(self._grid)

  @property
  def best(self) -> Variant | None:
    if not self._grid:
      return None
    return max(self._grid.values(), key=lambda v: v.fitness)

  def insert(self, variant: Variant) -> bool:
    key = variant.descriptor.cell_key()
    existing = self._grid.get(key)
    if existing is None or variant.fitness > existing.fitness:
      self._grid[key] = variant
      return True
    return False

  def tournament_select(self, k: int = 3, rng_seed: int | None = None) -> Variant:
    rng = random.Random(rng_seed)
    candidates = rng.sample(list(self._grid.values()), min(k, len(self._grid)))
    return max(candidates, key=lambda v: v.fitness)

  def all_variants(self) -> list[Variant]:
    return list(self._grid.values())

  def save(self, path: str | Path) -> None:
    path = Path(path)
    data = {
      "descriptor_axes": self._descriptor_axes,
      "variants": [
        {
          "id": v.id,
          "code": v.code,
          "descriptor": asdict(v.descriptor),
          "fitness": v.fitness,
          "generation": v.generation,
          "parent_id": v.parent_id,
          "metadata": v.metadata,
        }
        for v in self._grid.values()
      ],
    }
    path.write_text(json.dumps(data, indent=2))

  @classmethod
  def load(cls, path: str | Path) -> Archive:
    path = Path(path)
    data = json.loads(path.read_text())
    archive = cls(descriptor_axes=data["descriptor_axes"])
    for vd in data["variants"]:
      desc = BehaviorDescriptor(**vd["descriptor"])
      variant = Variant(
        id=vd["id"],
        code=vd["code"],
        descriptor=desc,
        fitness=vd["fitness"],
        generation=vd["generation"],
        parent_id=vd["parent_id"],
        metadata=vd.get("metadata", {}),
      )
      archive.insert(variant)
    return archive
