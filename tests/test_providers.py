"""
Property-based tests for the provider factory.

Feature: pai-phase1-foundation
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from core.exceptions import ConfigError
from providers import get_provider, VALID_PROVIDERS


class TestProviderFactoryInvalidNames:
    """Property 3: Invalid provider names raise ConfigError

    For any string that is not in the set of valid provider names
    ("gemini", "ollama"), calling get_provider() with that string
    SHALL raise a ConfigError.

    Validates: Requirements 8.3
    """

    @given(name=st.text().filter(lambda s: s not in VALID_PROVIDERS))
    @settings(max_examples=200)
    def test_invalid_provider_names_raise_config_error(self, name: str) -> None:
        """Property 3: Invalid provider names raise ConfigError"""
        with pytest.raises(ConfigError):
            get_provider(name)


class TestProviderFactoryValidNames:
    """Unit tests for provider factory with valid provider names.

    Validates: Requirements 11.3, 11.4
    """

    def test_get_provider_gemini_returns_gemini_provider(self, monkeypatch):
        """get_provider('gemini') returns a GeminiProvider instance."""
        from unittest.mock import MagicMock, patch

        mock_gemini_module = MagicMock()
        mock_gemini_provider_class = MagicMock()
        mock_gemini_instance = MagicMock()
        mock_gemini_provider_class.return_value = mock_gemini_instance
        mock_gemini_module.GeminiProvider = mock_gemini_provider_class

        with patch.dict(
            "sys.modules", {"providers.gemini": mock_gemini_module}
        ):
            result = get_provider("gemini")

        assert result is mock_gemini_instance
        mock_gemini_provider_class.assert_called_once()

    def test_get_provider_ollama_returns_ollama_provider(self, monkeypatch):
        """get_provider('ollama') returns an OllamaProvider instance."""
        from unittest.mock import MagicMock, patch

        mock_ollama_module = MagicMock()
        mock_ollama_provider_class = MagicMock()
        mock_ollama_instance = MagicMock()
        mock_ollama_provider_class.return_value = mock_ollama_instance
        mock_ollama_module.OllamaProvider = mock_ollama_provider_class

        with patch.dict(
            "sys.modules", {"providers.ollama": mock_ollama_module}
        ):
            result = get_provider("ollama")

        assert result is mock_ollama_instance
        mock_ollama_provider_class.assert_called_once()

    def test_get_provider_no_args_uses_default_provider(self, monkeypatch):
        """get_provider() with no args uses config.LLM_PROVIDER."""
        from unittest.mock import MagicMock, patch
        import config

        # Set the default provider to "ollama"
        monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")

        mock_ollama_module = MagicMock()
        mock_ollama_provider_class = MagicMock()
        mock_ollama_instance = MagicMock()
        mock_ollama_provider_class.return_value = mock_ollama_instance
        mock_ollama_module.OllamaProvider = mock_ollama_provider_class

        with patch.dict(
            "sys.modules", {"providers.ollama": mock_ollama_module}
        ):
            result = get_provider()

        assert result is mock_ollama_instance
        mock_ollama_provider_class.assert_called_once()

    def test_get_provider_no_args_uses_gemini_default(self, monkeypatch):
        """get_provider() with no args defaults to gemini when config says so."""
        from unittest.mock import MagicMock, patch
        import config

        # Set the default provider to "gemini"
        monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")

        mock_gemini_module = MagicMock()
        mock_gemini_provider_class = MagicMock()
        mock_gemini_instance = MagicMock()
        mock_gemini_provider_class.return_value = mock_gemini_instance
        mock_gemini_module.GeminiProvider = mock_gemini_provider_class

        with patch.dict(
            "sys.modules", {"providers.gemini": mock_gemini_module}
        ):
            result = get_provider()

        assert result is mock_gemini_instance
        mock_gemini_provider_class.assert_called_once()

    def test_get_provider_github_returns_github_provider(self):
        """get_provider('github') returns a GitHubModelsProvider instance."""
        from unittest.mock import MagicMock, patch

        mock_github_module = MagicMock()
        mock_github_provider_class = MagicMock()
        mock_github_instance = MagicMock()
        mock_github_provider_class.return_value = mock_github_instance
        mock_github_module.GitHubModelsProvider = mock_github_provider_class

        with patch.dict(
            "sys.modules", {"providers.github_models": mock_github_module}
        ):
            result = get_provider("github")

        assert result is mock_github_instance
        mock_github_provider_class.assert_called_once()

    def test_get_provider_anthropic_returns_anthropic_provider(self):
        """get_provider('anthropic') returns an AnthropicProvider instance."""
        from unittest.mock import MagicMock, patch

        mock_anthropic_module = MagicMock()
        mock_anthropic_provider_class = MagicMock()
        mock_anthropic_instance = MagicMock()
        mock_anthropic_provider_class.return_value = mock_anthropic_instance
        mock_anthropic_module.AnthropicProvider = mock_anthropic_provider_class

        with patch.dict(
            "sys.modules", {"providers.anthropic": mock_anthropic_module}
        ):
            result = get_provider("anthropic")

        assert result is mock_anthropic_instance
        mock_anthropic_provider_class.assert_called_once()

    def test_get_provider_openai_returns_openai_provider(self):
        """get_provider('openai') returns an OpenAIProvider instance."""
        from unittest.mock import MagicMock, patch

        mock_openai_module = MagicMock()
        mock_openai_provider_class = MagicMock()
        mock_openai_instance = MagicMock()
        mock_openai_provider_class.return_value = mock_openai_instance
        mock_openai_module.OpenAIProvider = mock_openai_provider_class

        with patch.dict(
            "sys.modules", {"providers.openai_provider": mock_openai_module}
        ):
            result = get_provider("openai")

        assert result is mock_openai_instance
        mock_openai_provider_class.assert_called_once()
