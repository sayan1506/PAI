"""
Unit tests for the PAI configuration system.

Tests verify:
- Default values are applied when environment variables are absent
- validate_config() raises ConfigError when required config is missing
- validate_config() passes when all required values are present

Requirements: 11.1, 11.2
"""

import importlib
import pytest

import config
from core.exceptions import ConfigError


class TestConfigDefaults:
    """Test that default values are applied when env vars are absent."""

    def test_default_llm_provider(self, monkeypatch):
        """LLM_PROVIDER defaults to 'gemini' when not set."""
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        importlib.reload(config)
        assert config.LLM_PROVIDER == "gemini"

    def test_default_ollama_host(self, monkeypatch):
        """OLLAMA_HOST defaults to 'http://localhost:11434' when not set."""
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        importlib.reload(config)
        assert config.OLLAMA_HOST == "http://localhost:11434"

    def test_default_ollama_model(self, monkeypatch):
        """OLLAMA_MODEL defaults to 'llama3' when not set."""
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        importlib.reload(config)
        assert config.OLLAMA_MODEL == "llama3"

    def test_default_log_level(self, monkeypatch):
        """LOG_LEVEL defaults to 'INFO' when not set."""
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        importlib.reload(config)
        assert config.LOG_LEVEL == "INFO"

    def test_default_agent_name(self, monkeypatch):
        """AGENT_NAME defaults to 'PAI' when not set."""
        monkeypatch.delenv("AGENT_NAME", raising=False)
        importlib.reload(config)
        assert config.AGENT_NAME == "PAI"

    def test_default_max_session_turns(self, monkeypatch):
        """MAX_SESSION_TURNS defaults to 20 when not set."""
        monkeypatch.delenv("MAX_SESSION_TURNS", raising=False)
        importlib.reload(config)
        assert config.MAX_SESSION_TURNS == 20

    def test_default_gemini_api_key_empty(self, monkeypatch):
        """GEMINI_API_KEY defaults to empty string when not set."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        importlib.reload(config)
        assert config.GEMINI_API_KEY == ""


class TestValidateConfigRaises:
    """Test that validate_config() raises ConfigError for invalid config."""

    def test_raises_when_gemini_key_missing(self, monkeypatch):
        """validate_config() raises ConfigError when provider is gemini and key is absent."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        importlib.reload(config)

        with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
            config.validate_config()

    def test_raises_when_provider_unknown(self, monkeypatch):
        """validate_config() raises ConfigError for an unsupported provider name."""
        monkeypatch.setenv("LLM_PROVIDER", "unknown_provider")
        importlib.reload(config)

        with pytest.raises(ConfigError, match="Unknown LLM provider"):
            config.validate_config()


class TestValidateConfigPasses:
    """Test that validate_config() passes when config is valid."""

    def test_passes_with_gemini_key_set(self, monkeypatch):
        """validate_config() does not raise when gemini key is provided."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "test-api-key-123")
        importlib.reload(config)

        # Should not raise
        config.validate_config()

    def test_passes_with_ollama_provider(self, monkeypatch):
        """validate_config() does not raise when provider is ollama (no key needed)."""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        importlib.reload(config)

        # Should not raise
        config.validate_config()
