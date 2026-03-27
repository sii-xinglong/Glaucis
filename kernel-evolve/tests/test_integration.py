"""Integration test: full evolution loop with mocked LLM and evaluator."""

from unittest.mock import AsyncMock

import pytest

from kernel_evolve.config import EvolveConfig
from kernel_evolve.engine import EvolutionEngine
from kernel_evolve.evaluator import EvalResult
from kernel_evolve.llm.base import MutationResponse

TEMPLATE = """\
def kernel():
  # EVOLVE-BLOCK-START
  x = 1 + 1
  # EVOLVE-BLOCK-END
  return x
"""

REFERENCE = "def simple_compute(M=64): return 42"

call_count = 0


@pytest.fixture
def config(tmp_path):
  return EvolveConfig(
    kernel={"name": "integration_test", "template": "k.py", "reference": "r.py"},
    shapes=[{"M": 64}],
    evolution={
      "population_size": 4,
      "num_islands": 1,
      "max_generations": 3,
      "stagnation_limit": 10,
    },
    llm={"provider": "anthropic", "model": "test"},
    tpu={"cluster": "c", "zone": "z", "tpu_type": "v4-8", "image": "img"},
    logging={"output_dir": str(tmp_path / "run")},
  )


@pytest.mark.asyncio
async def test_full_evolution_loop(config, tmp_path):
  provider = AsyncMock()
  global call_count
  call_count = 0

  async def mock_mutate(req):
    global call_count
    call_count += 1
    return MutationResponse(
      mutated_code=f"  x = {call_count} * 2",
      explanation=f"optimization attempt {call_count}",
      suggested_descriptor={
        "block_size": 128,
        "pipeline_stages": call_count % 4 + 1,
        "memory_strategy": "scratch",
      },
    )

  provider.mutate = mock_mutate

  evaluator = AsyncMock()
  evaluator.evaluate.return_value = EvalResult.success(latency_ms=1.0, speedup=2.0, flops=1e12)

  engine = EvolutionEngine(
    config=config,
    provider=provider,
    evaluator=evaluator,
    template_code=TEMPLATE,
    reference_code=REFERENCE,
  )

  best = await engine.run()
  assert best is not None
  assert best.fitness > 0

  # Check outputs were created
  run_dir = tmp_path / "run"
  assert (run_dir / "perf_log.md").exists()
  assert (run_dir / "best" / "kernel.py").exists()

  perf_log = (run_dir / "perf_log.md").read_text()
  assert "Generation 1" in perf_log
  assert "Generation 3" in perf_log
