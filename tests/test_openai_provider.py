"""
Unit tests for OpenAIProvider.

All OpenAI client calls are mocked — no real network requests.
"""

from unittest.mock import MagicMock, patch

import pytest
import openai

import config
from core.exceptions import ProviderError, RateLimitError
from providers.base import Message
from providers.openai_provider import OpenAIProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_response(content="Hello", tool_calls=None):
    """Build a minimal mock ChatCompletion response."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


def _fake_tool_call(id="1", name="browser", arguments='{"action": "open_url", "url": "https://example.com"}'):
    """Build a mock tool call object matching OpenAI's response shape."""
    tc = MagicMock()
    tc.id = id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    """Unit tests for OpenAIProvider."""

    def test_openai_provider_model_name(self, monkeypatch):
        """model_name returns config.OPENAI_MODEL."""
        monkeypatch.setattr(config, "OPENAI_MODEL", "gpt-4o-test")
        provider = OpenAIProvider()
        assert provider.model_name == "gpt-4o-test"

    def test_openai_make_client_uses_openai_api_key(self, monkeypatch):
        """_make_client passes api_key from config and does NOT set base_url."""
        monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")

        with patch("providers.openai_provider.OpenAI") as mock_openai:
            provider = OpenAIProvider()
            provider._make_client()

            mock_openai.assert_called_once_with(api_key="sk-test")

    def test_openai_generate_returns_response(self):
        """generate() returns LLMResponse with correct content."""
        provider = OpenAIProvider()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response(
            content="Hello from GPT"
        )

        with patch.object(provider, "_make_client", return_value=mock_client):
            result = provider.generate([Message(role="user", content="Hi")])

        assert result.content == "Hello from GPT"

    def test_openai_generate_with_tools_no_tool_calls(self):
        """generate_with_tools() with no tool calls returns empty list."""
        provider = OpenAIProvider()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response(
            content="Done", tool_calls=None
        )

        with patch.object(provider, "_make_client", return_value=mock_client):
            result = provider.generate_with_tools(
                [Message(role="user", content="Hi")],
                tools=[{"name": "browser", "description": "Browse web", "parameters": {}}],
            )

        assert result.tool_calls == []
        assert result.content == "Done"

    def test_openai_generate_with_tools_with_tool_call(self):
        """generate_with_tools() parses tool calls correctly."""
        provider = OpenAIProvider()

        fake_tc = _fake_tool_call(
            id="tc1",
            name="browser",
            arguments='{"action": "open_url", "url": "https://example.com"}',
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response(
            content="", tool_calls=[fake_tc]
        )

        with patch.object(provider, "_make_client", return_value=mock_client):
            result = provider.generate_with_tools(
                [Message(role="user", content="Open example.com")],
                tools=[{"name": "browser", "description": "Browse web", "parameters": {}}],
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "browser"
        assert result.tool_calls[0].arguments == {
            "action": "open_url",
            "url": "https://example.com",
        }

    def test_openai_rate_limit_raises_pai_rate_limit_error(self):
        """generate() maps openai.RateLimitError to PAI RateLimitError."""
        provider = OpenAIProvider()

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="rate limited",
            response=mock_response,
            body=None,
        )

        with patch.object(provider, "_make_client", return_value=mock_client):
            with pytest.raises(RateLimitError):
                provider.generate([Message(role="user", content="Hi")])

    def test_openai_api_error_raises_provider_error(self):
        """generate() maps openai.APIError to PAI ProviderError."""
        provider = OpenAIProvider()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = openai.APIError(
            message="server error",
            request=MagicMock(),
            body=None,
        )

        with patch.object(provider, "_make_client", return_value=mock_client):
            with pytest.raises(ProviderError):
                provider.generate([Message(role="user", content="Hi")])

    def test_openai_health_check_passes(self):
        """health_check() returns True on successful response."""
        provider = OpenAIProvider()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response(content="p")

        with patch.object(provider, "_make_client", return_value=mock_client):
            assert provider.health_check() is True

    def test_openai_health_check_rate_limited_returns_true(self):
        """health_check() returns True when rate-limited (still reachable)."""
        provider = OpenAIProvider()

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="rate limited",
            response=mock_response,
            body=None,
        )

        with patch.object(provider, "_make_client", return_value=mock_client):
            assert provider.health_check() is True
