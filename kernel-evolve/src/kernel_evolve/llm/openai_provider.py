"""OpenAI provider for kernel mutation."""

from __future__ import annotations

import json
import re

from kernel_evolve.llm.base import LLMProvider, MutationRequest, MutationResponse
from kernel_evolve.llm.anthropic_provider import MUTATION_SYSTEM_PROMPT

try:
  from openai import AsyncOpenAI
except ImportError:
  AsyncOpenAI = None


class OpenAIProvider(LLMProvider):
  def __init__(self, model: str = "o3", temperature: float = 0.7):
    self._model = model
    self._temperature = temperature
    self._client = AsyncOpenAI()

  async def mutate(self, request: MutationRequest) -> MutationResponse:
    user_prompt = self._build_prompt(request)
    response = await self._client.chat.completions.create(
      model=self._model,
      temperature=self._temperature,
      max_tokens=4096,
      messages=[
        {"role": "system", "content": MUTATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
      ],
    )
    return self._parse_response(response.choices[0].message.content)

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
      line for line in text.split("\n")
      if line.strip() and not line.strip().startswith("```") and not line.strip().startswith("{")
    ]
    explanation = explanation_lines[-1] if explanation_lines else ""

    return MutationResponse(mutated_code=mutated_code, explanation=explanation, suggested_descriptor=desc)
