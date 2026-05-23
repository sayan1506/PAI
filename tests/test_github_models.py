"""
Unit tests for GitHubModelsProvider.

All OpenAI client calls are mocked — no real network requests.
"""

from unittest.mock import MagicMock, patch

import pytest
import openai

import config
from core.exceptions import ProviderError, RateLimitError
from providers.base import Message, ToolCall
from providers.github_models import GitHubModelsProvider


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


def _fake_tool_call(id="1", name="weather", arguments='{"location": "London"}'):
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


class TestGitHubModelsProvider:
    """Unit tests for GitHubModelsProvider."""

    def test_github_models_provider_model_name(self, monkeypatch):
        """model_name returns config.GITHUB_MODEL."""
        monkeypatch.setattr(config, "GITHUB_MODEL", "test-model-xyz")
        provider = GitHubModelsProvider()
        assert provider.model_name == "test-model-xyz"

    def test_github_models_make_client_uses_github_base_url(self, monkeypatch):
        """_make_client passes the correct base_url and api_key."""
        monkeypatch.setattr(config, "GITHUB_TOKEN", "test-token")

        with patch("providers.github_models.OpenAI") as mock_openai:
            provider = GitHubModelsProvider()
            provider._make_client()

            mock_openai.assert_called_once_with(
                base_url="https://models.inference.ai.azure.com",
                api_key="test-token",
            )

    def test_github_models_generate_returns_response(self):
        """generate() returns LLMResponse with correct content."""
        provider = GitHubModelsProvider()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response(content="Hello")

        with patch.object(provider, "_make_client", return_value=mock_client):
            result = provider.generate([Message(role="user", content="Hi")])

        assert result.content == "Hello"

    def test_github_models_generate_with_tools_no_tool_calls(self):
        """generate_with_tools() with no tool calls returns empty list."""
        provider = GitHubModelsProvider()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response(
            content="Done", tool_calls=None
        )

        with patch.object(provider, "_make_client", return_value=mock_client):
            result = provider.generate_with_tools(
                [Message(role="user", content="Hi")],
                tools=[{"name": "weather", "description": "Get weather", "parameters": {}}],
            )

        assert result.tool_calls == []
        assert result.content == "Done"

    def test_github_models_generate_with_tools_with_tool_call(self):
        """generate_with_tools() parses tool calls correctly."""
        provider = GitHubModelsProvider()

        fake_tc = _fake_tool_call(id="1", name="weather", arguments='{"location": "London"}')
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response(
            content="", tool_calls=[fake_tc]
        )

        with patch.object(provider, "_make_client", return_value=mock_client):
            result = provider.generate_with_tools(
                [Message(role="user", content="Weather?")],
                tools=[{"name": "weather", "description": "Get weather", "parameters": {}}],
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "1"
        assert result.tool_calls[0].name == "weather"
        assert result.tool_calls[0].arguments == {"location": "London"}

    def test_github_models_generate_with_tools_malformed_json_args(self):
        """Malformed JSON in tool arguments falls back to empty dict."""
        provider = GitHubModelsProvider()

        fake_tc = _fake_tool_call(id="2", name="search", arguments="not-json")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response(
            content="", tool_calls=[fake_tc]
        )

        with patch.object(provider, "_make_client", return_value=mock_client):
            result = provider.generate_with_tools(
                [Message(role="user", content="Search")],
                tools=[{"name": "search", "description": "Search", "parameters": {}}],
            )

        assert result.tool_calls[0].arguments == {}

    def test_github_models_rate_limit_raises_pai_rate_limit_error(self):
        """generate() maps openai.RateLimitError to PAI RateLimitError."""
        provider = GitHubModelsProvider()

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

    def test_github_models_api_error_raises_provider_error(self):
        """generate() maps openai.APIError to PAI ProviderError."""
        provider = GitHubModelsProvider()

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}
        mock_client.chat.completions.create.side_effect = openai.APIError(
            message="server error",
            request=MagicMock(),
            body=None,
        )

        with patch.object(provider, "_make_client", return_value=mock_client):
            with pytest.raises(ProviderError):
                provider.generate([Message(role="user", content="Hi")])

    def test_github_models_health_check_passes(self):
        """health_check() returns True on successful response."""
        provider = GitHubModelsProvider()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response(content="p")

        with patch.object(provider, "_make_client", return_value=mock_client):
            assert provider.health_check() is True

    def test_github_models_health_check_rate_limited_returns_true(self):
        """health_check() returns True when rate-limited (still reachable)."""
        provider = GitHubModelsProvider()

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

    def test_github_models_health_check_api_error_raises_provider_error(self):
        """health_check() raises ProviderError on API failure."""
        provider = GitHubModelsProvider()

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}
        mock_client.chat.completions.create.side_effect = openai.APIError(
            message="server error",
            request=MagicMock(),
            body=None,
        )

        with patch.object(provider, "_make_client", return_value=mock_client):
            with pytest.raises(ProviderError):
                provider.health_check()

    def test_build_messages_maps_roles_correctly(self):
        """_build_messages maps user/assistant/tool roles correctly."""
        provider = GitHubModelsProvider()

        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
            Message(role="tool", content='{"result": 42}', tool_call_id="tc-1"),
        ]

        result = provider._build_messages(messages)

        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi there"}
        assert result[2] == {"role": "tool", "content": '{"result": 42}', "tool_call_id": "tc-1"}

    def test_build_tools_wraps_in_function_envelope(self):
        """_build_tools wraps specs in {"type": "function", "function": spec}."""
        provider = GitHubModelsProvider()

        spec = {"name": "weather", "description": "Get weather", "parameters": {}}
        result = provider._build_tools([spec])

        assert result[0] == {"type": "function", "function": spec}
