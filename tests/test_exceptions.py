"""
Tests for the PAI custom exception hierarchy.

Verifies inheritance relationships, message storage, and that all exceptions
can be caught by the base PAIError class.
"""

import pytest

from core.exceptions import (
    PAIError,
    ConfigError,
    ProviderError,
    RateLimitError,
    AgentError,
    ToolError,
    AudioError,
)


class TestExceptionHierarchy:
    """Verify that all exceptions inherit from PAIError."""

    def test_config_error_inherits_from_pai_error(self):
        assert issubclass(ConfigError, PAIError)

    def test_provider_error_inherits_from_pai_error(self):
        assert issubclass(ProviderError, PAIError)

    def test_rate_limit_error_inherits_from_provider_error(self):
        assert issubclass(RateLimitError, ProviderError)

    def test_rate_limit_error_inherits_from_pai_error(self):
        assert issubclass(RateLimitError, PAIError)

    def test_agent_error_inherits_from_pai_error(self):
        assert issubclass(AgentError, PAIError)

    def test_tool_error_inherits_from_pai_error(self):
        assert issubclass(ToolError, PAIError)

    def test_audio_error_inherits_from_pai_error(self):
        assert issubclass(AudioError, PAIError)

    def test_pai_error_inherits_from_exception(self):
        assert issubclass(PAIError, Exception)


class TestExceptionMessages:
    """Verify that exceptions store and expose descriptive messages."""

    def test_pai_error_stores_message(self):
        err = PAIError("something went wrong")
        assert err.message == "something went wrong"
        assert str(err) == "something went wrong"

    def test_config_error_stores_message(self):
        err = ConfigError("missing API key")
        assert err.message == "missing API key"
        assert str(err) == "missing API key"

    def test_provider_error_stores_message(self):
        err = ProviderError("connection refused")
        assert err.message == "connection refused"
        assert str(err) == "connection refused"

    def test_agent_error_stores_message(self):
        err = AgentError("history overflow")
        assert err.message == "history overflow"
        assert str(err) == "history overflow"

    def test_rate_limit_error_stores_message(self):
        err = RateLimitError("429 Too Many Requests")
        assert err.message == "429 Too Many Requests"
        assert str(err) == "429 Too Many Requests"

    def test_tool_error_stores_message(self):
        err = ToolError("tool not found")
        assert err.message == "tool not found"
        assert str(err) == "tool not found"

    def test_audio_error_stores_message(self):
        err = AudioError("microphone unavailable")
        assert err.message == "microphone unavailable"
        assert str(err) == "microphone unavailable"

    def test_pai_error_default_empty_message(self):
        err = PAIError()
        assert err.message == ""
        assert str(err) == ""


class TestExceptionCatching:
    """Verify that exceptions can be caught by parent classes."""

    def test_config_error_caught_by_pai_error(self):
        with pytest.raises(PAIError):
            raise ConfigError("test")

    def test_provider_error_caught_by_pai_error(self):
        with pytest.raises(PAIError):
            raise ProviderError("test")

    def test_rate_limit_error_caught_by_provider_error(self):
        with pytest.raises(ProviderError):
            raise RateLimitError("test")

    def test_agent_error_caught_by_pai_error(self):
        with pytest.raises(PAIError):
            raise AgentError("test")

    def test_tool_error_caught_by_pai_error(self):
        with pytest.raises(PAIError):
            raise ToolError("test")

    def test_audio_error_caught_by_pai_error(self):
        with pytest.raises(PAIError):
            raise AudioError("test")
