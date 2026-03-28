"""Anthropic Claude provider for kernel mutation."""

from __future__ import annotations

import json
import os
import re

from kernel_evolve.llm.base import LLMProvider, MutationRequest, MutationResponse

try:
  from anthropic import AsyncAnthropic
except ImportError:
  AsyncAnthropic = None


MUTATION_SYSTEM_PROMPT = """\
You are an expert TPU kernel optimizer specializing in JAX Pallas.
Given a kernel and its performance data, produce an improved variant.

Rules:
- Only modify code within the EVOLVE-BLOCK markers
- Maintain the same function signature
- Ensure the kernel remains correct
- Focus on one optimization strategy per mutation

Output format:
1. The mutated kernel code in a ```python``` block
2. A JSON object with suggested behavioral descriptors:
   {"block_size": N, "pipeline_stages": N, "memory_strategy": "scratch|hbm|rmw"}
3. A one-line explanation of the change
"""


class AnthropicProvider(LLMProvider):
  def __init__(self, model: str = "claude-opus-4-6", temperature: float = 0.7):
    self._model = model
    self._temperature = temperature
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    self._client = AsyncAnthropic(api_key=api_key, base_url=base_url)

  async def mutate(self, request: MutationRequest) -> MutationResponse:
    user_prompt = self._build_prompt(request)
    message = await self._client.messages.create(
      model=self._model,
      max_tokens=4096,
      temperature=self._temperature,
      system=MUTATION_SYSTEM_PROMPT,
      messages=[{"role": "user", "content": user_prompt}],
    )
    return self._parse_response(message.content[0].text)

  def _build_prompt(self, request: MutationRequest) -> str:
    parts = [
      f"## Parent Kernel (fitness: {request.fitness}x speedup, generation {request.generation})\n",
      f"```python\n{request.parent_code}\n```\n",
    ]
    if request.optimization_history:
      parts.append("## Optimization History\n")
      for entry in request.optimization_history[-10:]:
        parts.append(f"- {entry}\n")
    if request.failed_attempts:
      parts.append("\n## Failed Attempts (do NOT repeat these)\n")
      for entry in request.failed_attempts[-5:]:
        parts.append(f"- {entry}\n")
    if request.focus_hint:
      parts.append(f"\n## Focus Area\nPrioritize: {request.focus_hint}\n")
    return "".join(parts)

  def _parse_response(self, text: str) -> MutationResponse:
    code_match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    mutated_code = code_match.group(1).strip() if code_match else text

    desc = {}
    json_match = re.search(r"\{[^{}]*\"block_size\"[^{}]*\}", text)
    if json_match:
      try:
        desc = json.loads(json_match.group())
      except json.JSONDecodeError:
        pass

    explanation_lines = [
      line
      for line in text.split("\n")
      if line.strip() and not line.strip().startswith("```") and not line.strip().startswith("{")
    ]
    explanation = explanation_lines[-1] if explanation_lines else ""

    return MutationResponse(mutated_code=mutated_code, explanation=explanation, suggested_descriptor=desc)
