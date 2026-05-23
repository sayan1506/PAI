"""
PAI Providers Module

get_provider() factory — instantiates the LLM provider selected in config.
Supported provider names: gemini | ollama | github | anthropic | openai
"""

import config
from core.exceptions import ConfigError
from providers.base import LLMProvider

VALID_PROVIDERS = {"gemini", "ollama", "github", "anthropic", "openai"}


def get_provider(name: str | None = None) -> LLMProvider:
    """
    Instantiate and return an LLM provider.

    Args:
        name: Provider name. If None, uses config.LLM_PROVIDER.

    Returns:
        An LLMProvider instance ready for use.

    Raises:
        ConfigError: If the provider name is not in VALID_PROVIDERS.
    """
    provider_name = name if name is not None else config.LLM_PROVIDER

    if provider_name == "gemini":
        from providers.gemini import GeminiProvider
        return GeminiProvider()

    if provider_name == "ollama":
        from providers.ollama import OllamaProvider
        return OllamaProvider()

    if provider_name == "github":
        from providers.github_models import GitHubModelsProvider
        return GitHubModelsProvider()

    if provider_name == "anthropic":
        from providers.anthropic import AnthropicProvider
        return AnthropicProvider()

    if provider_name == "openai":
        from providers.openai_provider import OpenAIProvider
        return OpenAIProvider()

    raise ConfigError(
        f"Unknown provider: '{provider_name}'. "
        f"Valid providers: {', '.join(sorted(VALID_PROVIDERS))}"
    )
