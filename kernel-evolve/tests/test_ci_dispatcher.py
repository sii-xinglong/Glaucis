"""Tests for CI-based evaluation dispatcher."""

from unittest.mock import AsyncMock

import pytest

from kernel_evolve.ci_dispatcher import CIConfig, CIDispatcher
from kernel_evolve.evaluator import EvalRequest, EvalResult, EvalStatus


@pytest.fixture
def ci_config():
  return CIConfig(
    repo="sii-xinglong/Glaucis",
    workflow="kernel-eval.yaml",
    branch="main",
    poll_interval=1,
    timeout=10,
  )


def test_ci_config_defaults():
  cfg = CIConfig(repo="org/repo", workflow="eval.yaml")
  assert cfg.branch == "main"
  assert cfg.poll_interval == 30
  assert cfg.timeout == 600


@pytest.mark.asyncio
async def test_dispatch_and_collect(ci_config):
  dispatcher = CIDispatcher(ci_config)

  mock_trigger = AsyncMock(return_value="12345")
  mock_wait = AsyncMock(return_value="completed")
  mock_collect = AsyncMock(return_value=EvalResult.success(latency_ms=1.0, speedup=2.5, flops=1e12))

  dispatcher._trigger_workflow = mock_trigger
  dispatcher._wait_for_completion = mock_wait
  dispatcher._collect_result = mock_collect

  req = EvalRequest(
    variant_id="v001",
    kernel_code="def k(): pass",
    reference_code="def r(): pass",
    shapes=[{"M": 1024}],
  )
  result = await dispatcher.evaluate(req)
  assert result.status == EvalStatus.SUCCESS
  assert result.speedup == 2.5
  mock_trigger.assert_called_once()


@pytest.mark.asyncio
async def test_dispatch_timeout(ci_config):
  dispatcher = CIDispatcher(ci_config)
  dispatcher._trigger_workflow = AsyncMock(return_value="12345")
  dispatcher._wait_for_completion = AsyncMock(return_value="timed_out")
  dispatcher._collect_result = AsyncMock()

  req = EvalRequest(variant_id="v001", kernel_code="x", reference_code="r", shapes=[])
  result = await dispatcher.evaluate(req)
  assert result.status == EvalStatus.COMPILE_ERROR
  assert "timed out" in result.error.lower()
  dispatcher._collect_result.assert_not_called()
