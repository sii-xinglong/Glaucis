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
You are an expert TPU kernel optimizer specializing in JAX Pallas on TPU v7x (Ironwood).
Given a kernel and its performance data, produce an improved variant that runs faster.

The code you receive includes both the kernel function and the pallas_call configuration.
You MUST optimize BOTH the kernel body AND the pallas_call parameters together.

Available imports (already provided, do NOT include in your output):
  import jax, jax.numpy as jnp
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

CRITICAL API RULES:
- For compiler_params, ALWAYS use: compiler_params=pltpu.CompilerParams(dimension_semantics=(...))
  NEVER use: compiler_params=dict(mosaic=dict(...))  # CRASHES with unhashable type error
  NEVER use: pltpu.TPUCompilerParams  # WRONG NAME, does not exist
- For 2D grids: compiler_params is OPTIONAL, omit it entirely
- For 3D grids (K-tiling): compiler_params=pltpu.TPUCompilerParams(dimension_semantics=("parallel","parallel","arbitrary")) is REQUIRED

Key optimization levers (explore aggressively):
- Block sizes (BLOCK_M, BLOCK_N): try 64, 128, 256, 512, 1024
- Tiled K-reduction: split K into BLOCK_K chunks using a 3D grid
  - Use scratch_shapes=[pltpu.VMEM((BLOCK_M, BLOCK_N), dtype=jnp.float32)] for the accumulator
  - Initialize accumulator with @pl.when(pl.program_id(2) == 0), store with @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
  - The kernel function gets an extra acc_ref parameter from scratch_shapes
- Grid dimensions: 2D (M//BM, N//BN) for simple, 3D (M//BM, N//BN, K//BK) for K-tiling

Constraints:
- Function signature `optimized_compute(M=1024, N=1024, K=1024)` MUST stay unchanged
- Input generation (jax.random.normal with same PRNGKey(0), PRNGKey(1), same dtypes) MUST stay unchanged
- The kernel must return correct results (allclose with atol=1.0)
- Use bfloat16 (NOT float16) — TPU Mosaic requires bfloat16
- Do NOT include import statements — they are provided by the template

TPU v7x Pallas API:
- Ref indexing: x_ref[...], x_ref[pl.ds(start, size)] — NOT pl.load/pl.store
- Scratch memory: pltpu.VMEM((shape), dtype=dtype)
- jnp.dot inside kernels compiles to MXU hardware dot

Output format:
1. The mutated code in a ```python``` block (kernel function + optimized_compute, NO imports)
2. A JSON object: {"block_size": N, "pipeline_stages": N, "memory_strategy": "scratch|hbm|rmw"}
3. A one-line explanation of what you changed and WHY it should be faster
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
