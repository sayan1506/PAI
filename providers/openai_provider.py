"""
PAI OpenAI Provider (stub)

Placeholder for future OpenAI integration.
Implement in Phase 2.
"""

from providers.base import LLMProvider, LLMResponse, Message
from core.exceptions import ProviderError


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider — not yet implemented."""

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    def generate(self, messages: list[Message]) -> LLMResponse:
        raise ProviderError("OpenAIProvider is not yet implemented.")

    def health_check(self) -> bool:
        raise ProviderError("OpenAIProvider is not yet implemented.")
