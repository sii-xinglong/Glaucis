"""Tests for the main evolution engine loop."""

from unittest.mock import AsyncMock

import pytest

from kernel_evolve.config import EvolveConfig
from kernel_evolve.engine import EvolutionEngine
from kernel_evolve.evaluator import EvalResult
from kernel_evolve.llm.base import MutationResponse


@pytest.fixture
def minimal_config():
  return EvolveConfig(
    kernel={"name": "test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64, "N": 64, "K": 64}],
    evolution={"population_size": 4, "num_islands": 1, "max_generations": 3, "stagnation_limit": 2},
    llm={"provider": "anthropic", "model": "claude-opus-4-6"},
    tpu={"cluster": "c", "zone": "z", "tpu_type": "v4-8", "image": "img"},
    logging={"output_dir": "/tmp/test_run"},
  )


@pytest.fixture
def mock_provider():
  provider = AsyncMock()
  provider.mutate.return_value = MutationResponse(
    mutated_code="def kernel(): optimized()",
    explanation="optimized the loop",
    suggested_descriptor={"block_size": 128, "pipeline_stages": 2, "memory_strategy": "scratch"},
  )
  return provider


@pytest.fixture
def mock_evaluator():
  evaluator = AsyncMock()
  evaluator.evaluate.return_value = EvalResult.success(latency_ms=1.0, speedup=2.0, flops=1e12)
  return evaluator


def test_engine_creation(minimal_config, mock_provider, mock_evaluator, tmp_path):
  minimal_config.logging.output_dir = str(tmp_path / "run")
  engine = EvolutionEngine(
    config=minimal_config,
    provider=mock_provider,
    evaluator=mock_evaluator,
    template_code="# EVOLVE-BLOCK-START\npass\n# EVOLVE-BLOCK-END",
    reference_code="def ref(): pass",
  )
  assert engine._generation == 0
  assert len(engine._islands) == 1


@pytest.mark.asyncio
async def test_engine_single_generation(minimal_config, mock_provider, mock_evaluator, tmp_path):
  minimal_config.logging.output_dir = str(tmp_path / "run")
  engine = EvolutionEngine(
    config=minimal_config,
    provider=mock_provider,
    evaluator=mock_evaluator,
    template_code="# EVOLVE-BLOCK-START\npass\n# EVOLVE-BLOCK-END",
    reference_code="def ref(): pass",
  )
  await engine.run_generation()
  assert engine._generation == 1
  assert mock_provider.mutate.call_count >= 1
  assert mock_evaluator.evaluate.call_count >= 1


@pytest.mark.asyncio
async def test_engine_stagnation_detection(minimal_config, mock_provider, mock_evaluator, tmp_path):
  minimal_config.logging.output_dir = str(tmp_path / "run")
  minimal_config.evolution.stagnation_limit = 2

  engine = EvolutionEngine(
    config=minimal_config,
    provider=mock_provider,
    evaluator=mock_evaluator,
    template_code="# EVOLVE-BLOCK-START\npass\n# EVOLVE-BLOCK-END",
    reference_code="def ref(): pass",
  )
  # Run 3 generations with same fitness -> stagnation should be detected
  for _ in range(3):
    await engine.run_generation()
  assert engine._stagnation_count >= 2
