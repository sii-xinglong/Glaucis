"""Tests for LLM provider interface."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kernel_evolve.llm.anthropic_provider import AnthropicProvider
from kernel_evolve.llm.base import LLMProvider, MutationRequest, MutationResponse


def test_mutation_request_format():
  req = MutationRequest(
    parent_code="def kernel(): pass",
    fitness=1.5,
    generation=3,
    optimization_history=["tried larger blocks", "tried more pipelines"],
    failed_attempts=["syntax error with approach X"],
    focus_hint="tiling",
  )
  assert "def kernel(): pass" in req.parent_code
  assert req.fitness == 1.5
  assert len(req.optimization_history) == 2


def test_mutation_response_parse():
  resp = MutationResponse(
    mutated_code="def kernel_v2(): pass",
    explanation="Changed block size from 128 to 256",
    suggested_descriptor={"block_size": 256, "pipeline_stages": 1, "memory_strategy": "scratch"},
  )
  assert "kernel_v2" in resp.mutated_code
  assert resp.suggested_descriptor["block_size"] == 256


def test_provider_interface():
  """LLMProvider is abstract and cannot be instantiated."""
  with pytest.raises(TypeError):
    LLMProvider()


@pytest.mark.asyncio
async def test_anthropic_provider_mutate():
  mock_client = MagicMock()
  mock_message = MagicMock()
  mock_text = (
    "```python\ndef optimized(): pass\n```\n"
    '{"block_size": 128, "pipeline_stages": 2, "memory_strategy": "scratch"}\n'
    "Explanation: optimized loop"
  )
  mock_message.content = [MagicMock(text=mock_text)]
  mock_client.messages.create = AsyncMock(return_value=mock_message)

  with patch("kernel_evolve.llm.anthropic_provider.AsyncAnthropic", return_value=mock_client):
    provider = AnthropicProvider(model="claude-sonnet-4-6", temperature=0.7)
    req = MutationRequest(parent_code="def kernel(): pass", fitness=1.0, generation=1)
    resp = await provider.mutate(req)
    assert "optimized" in resp.mutated_code
    mock_client.messages.create.assert_called_once()
