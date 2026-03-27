"""LLM providers for kernel mutation."""

from kernel_evolve.llm.base import LLMProvider, MutationRequest, MutationResponse


def create_provider(provider: str, model: str, temperature: float = 0.7) -> LLMProvider:
  """Factory to create an LLM provider by name."""
  if provider == "anthropic":
    from kernel_evolve.llm.anthropic_provider import AnthropicProvider

    return AnthropicProvider(model=model, temperature=temperature)
  elif provider == "google":
    from kernel_evolve.llm.google_provider import GoogleProvider

    return GoogleProvider(model=model, temperature=temperature)
  elif provider == "openai":
    from kernel_evolve.llm.openai_provider import OpenAIProvider

    return OpenAIProvider(model=model, temperature=temperature)
  else:
    raise ValueError(f"Unknown LLM provider: {provider}")


__all__ = ["LLMProvider", "MutationRequest", "MutationResponse", "create_provider"]
