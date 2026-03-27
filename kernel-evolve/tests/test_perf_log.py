"""Tests for sparse-mask-attention-style performance logging."""

import pytest

from kernel_evolve.evaluator import EvalResult
from kernel_evolve.perf_log import PerfLog
from kernel_evolve.population import BehaviorDescriptor


@pytest.fixture
def perf_log(tmp_path):
  return PerfLog(tmp_path / "perf_log.md", kernel_name="test_matmul")


def test_write_header(perf_log):
  perf_log.write_header()
  content = perf_log.path.read_text()
  assert "# Performance Log: test_matmul" in content


def test_log_generation(perf_log):
  perf_log.write_header()

  entries = [
    {
      "variant_id": "v001",
      "descriptor": BehaviorDescriptor(block_size=128, pipeline_stages=2, memory_strategy="scratch"),
      "result": EvalResult.success(latency_ms=1.5, speedup=2.3, flops=1e12),
      "explanation": "Larger tiles",
    },
    {
      "variant_id": "v002",
      "descriptor": BehaviorDescriptor(block_size=256, pipeline_stages=1, memory_strategy="hbm"),
      "result": EvalResult.compile_error("invalid grid"),
      "explanation": "Tried 256 blocks",
    },
  ]
  perf_log.log_generation(generation=1, entries=entries, best_id="v001", best_fitness=2.3)

  content = perf_log.path.read_text()
  assert "## Generation 1" in content
  assert "v001" in content
  assert "2.3" in content
  assert "compile_error" in content.lower() or "COMPILE_ERROR" in content
  assert "Best this gen: v001" in content


def test_cumulative_best_tracking(perf_log):
  perf_log.write_header()

  entries1 = [
    {
      "variant_id": "v001",
      "descriptor": BehaviorDescriptor(),
      "result": EvalResult.success(latency_ms=2.0, speedup=1.5, flops=1e12),
      "explanation": "first",
    }
  ]
  perf_log.log_generation(1, entries1, "v001", 1.5)

  entries2 = [
    {
      "variant_id": "v002",
      "descriptor": BehaviorDescriptor(block_size=256),
      "result": EvalResult.success(latency_ms=1.0, speedup=3.0, flops=1e12),
      "explanation": "better",
    }
  ]
  perf_log.log_generation(2, entries2, "v002", 3.0)

  content = perf_log.path.read_text()
  assert "Cumulative best: v002 (3.0x)" in content
