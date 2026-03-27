"""Tests for Google and OpenAI LLM providers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kernel_evolve.llm.base import MutationRequest
from kernel_evolve.llm.google_provider import GoogleProvider
from kernel_evolve.llm.openai_provider import OpenAIProvider


@pytest.mark.asyncio
async def test_google_provider_mutate():
  mock_client = MagicMock()
  mock_response = MagicMock()
  mock_response.text = (
    "```python\ndef optimized(): pass\n```\n"
    '{"block_size": 256, "pipeline_stages": 1, "memory_strategy": "hbm"}\n'
    "Used larger blocks"
  )
  mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

  with patch("kernel_evolve.llm.google_provider.genai.Client", return_value=mock_client):
    provider = GoogleProvider(model="gemini-2.5-pro", temperature=0.7)
    req = MutationRequest(parent_code="def kernel(): pass", fitness=1.0, generation=1)
    resp = await provider.mutate(req)
    assert "optimized" in resp.mutated_code


@pytest.mark.asyncio
async def test_openai_provider_mutate():
  mock_client = MagicMock()
  mock_response = MagicMock()
  mock_choice = MagicMock()
  mock_choice.message.content = (
    "```python\ndef optimized(): pass\n```\n"
    '{"block_size": 128, "pipeline_stages": 3, "memory_strategy": "scratch"}\n'
    "Added pipelining"
  )
  mock_response.choices = [mock_choice]
  mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

  with patch("kernel_evolve.llm.openai_provider.AsyncOpenAI", return_value=mock_client):
    provider = OpenAIProvider(model="o3", temperature=0.7)
    req = MutationRequest(parent_code="def kernel(): pass", fitness=1.0, generation=1)
    resp = await provider.mutate(req)
    assert "optimized" in resp.mutated_code
