"""
PAI GitHub Models Provider (stub)

Placeholder for future GitHub Models integration.
Implement in Phase 2.
"""

from providers.base import LLMProvider, LLMResponse, Message
from core.exceptions import ProviderError


class GitHubModelsProvider(LLMProvider):
    """GitHub Models LLM provider — not yet implemented."""

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    def generate(self, messages: list[Message]) -> LLMResponse:
        raise ProviderError("GitHubModelsProvider is not yet implemented.")

    def health_check(self) -> bool:
        raise ProviderError("GitHubModelsProvider is not yet implemented.")
