"""
PAI Providers Module

Provides the get_provider factory function for instantiating LLM providers
based on configuration or explicit name.
"""

import config
from core.exceptions import ConfigError
from providers.base import LLMProvider

# Valid provider names
VALID_PROVIDERS = {"gemini", "ollama"}


def get_provider(name: str | None = None) -> LLMProvider:
    """Factory function to create an LLM provider instance.

    Args:
        name: The provider name to instantiate. If None, uses
              config.LLM_PROVIDER as the default.

    Returns:
        An instance of the requested LLMProvider implementation.

    Raises:
        ConfigError: If the provider name is not recognized.
    """
    provider_name = name if name is not None else config.LLM_PROVIDER

    if provider_name == "gemini":
        from providers.gemini import GeminiProvider

        return GeminiProvider()

    if provider_name == "ollama":
        from providers.ollama import OllamaProvider

        return OllamaProvider()

    raise ConfigError(
        f"Unknown provider: '{provider_name}'. "
        f"Valid providers: {', '.join(sorted(VALID_PROVIDERS))}"
    )
