"""
PAI Anthropic Provider (stub)

Placeholder for future Anthropic Claude integration.
Implement in Phase 2.
"""

from providers.base import LLMProvider, LLMResponse, Message
from core.exceptions import ProviderError


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider — not yet implemented."""

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    def generate(self, messages: list[Message]) -> LLMResponse:
        raise ProviderError("AnthropicProvider is not yet implemented.")

    def health_check(self) -> bool:
        raise ProviderError("AnthropicProvider is not yet implemented.")
