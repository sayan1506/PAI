"""
Unit tests for AnthropicProvider.

All Anthropic SDK calls are mocked — no real network requests.
"""

from unittest.mock import MagicMock, patch

import pytest

import config
from core.exceptions import ProviderError, RateLimitError
from providers.base import Message, ToolCall
from providers.anthropic import AnthropicProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_text_block(text="Hello"):
    """Build a mock text content block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _fake_tool_use_block(id="t1", name="weather", input_data=None):
    """Build a mock tool_use content block."""
    block = MagicMock()
    block.type = "tool_use"
    block.id = id
    block.name = name
    block.input = input_data if input_data is not None else {}
    return block


def _fake_response(content_blocks=None):
    """Build a minimal mock Anthropic Messages response."""
    response = MagicMock()
    response.content = content_blocks if content_blocks is not None else [_fake_text_block()]
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    """Unit tests for AnthropicProvider."""

    @pytest.fixture(autouse=True)
    def _patch_client(self):
        """Patch the Anthropic client so no real API calls are made."""
        with patch("providers.anthropic._anthropic.Anthropic") as mock_cls:
            self.mock_client = MagicMock()
            mock_cls.return_value = self.mock_client
            yield

    # 1. model_name
    def test_anthropic_provider_model_name(self, monkeypatch):
        """model_name returns config.ANTHROPIC_MODEL."""
        monkeypatch.setattr(config, "ANTHROPIC_MODEL", "claude-sonnet-4-5")
        provider = AnthropicProvider()
        assert provider.model_name == "claude-sonnet-4-5"

    # 2. _extract_system_and_messages — system extraction
    def test_extract_system_takes_first_message_content(self):
        """First message content becomes system; priming skipped; rest converted."""
        provider = AnthropicProvider()
        messages = [
            Message("user", "System prompt"),
            Message("assistant", "Ok"),
            Message("user", "Hello"),
        ]
        system, anthropic_msgs = provider._extract_system_and_messages(messages)
        assert system == "System prompt"
        assert anthropic_msgs == [{"role": "user", "content": "Hello"}]

    # 3. _extract_system_and_messages — priming skip
    def test_extract_system_skips_priming_assistant_response(self):
        """The priming assistant message (index 1) is NOT in returned messages."""
        provider = AnthropicProvider()
        messages = [
            Message("user", "System prompt"),
            Message("assistant", "Understood, I am ready."),
            Message("user", "What is the weather?"),
        ]
        system, anthropic_msgs = provider._extract_system_and_messages(messages)
        # Priming message should not appear
        for msg in anthropic_msgs:
            if isinstance(msg.get("content"), str):
                assert "Understood" not in msg["content"]

    # 4. _extract_system_and_messages — tool result batching
    def test_extract_system_batches_consecutive_tool_results(self):
        """Consecutive tool messages are batched into ONE user turn."""
        provider = AnthropicProvider()
        messages = [
            Message("user", "System prompt"),
            Message("assistant", "Ok"),
            Message("user", "Do stuff"),
            Message("tool", "result1", tool_call_id="tc1"),
            Message("tool", "result2", tool_call_id="tc2"),
        ]
        _, anthropic_msgs = provider._extract_system_and_messages(messages)
        # First is the user message, second is the batched tool results
        assert anthropic_msgs[0] == {"role": "user", "content": "Do stuff"}
        batched = anthropic_msgs[1]
        assert batched["role"] == "user"
        assert len(batched["content"]) == 2
        assert batched["content"][0] == {
            "type": "tool_result",
            "tool_use_id": "tc1",
            "content": "result1",
        }
        assert batched["content"][1] == {
            "type": "tool_result",
            "tool_use_id": "tc2",
            "content": "result2",
        }

    # 5. _extract_system_and_messages — assistant with tool_calls
    def test_extract_system_assistant_with_tool_calls(self):
        """Assistant message with tool_calls emits tool_use blocks."""
        provider = AnthropicProvider()
        messages = [
            Message("user", "System prompt"),
            Message("assistant", "Ok"),
            Message(
                "assistant",
                "",
                tool_calls=[ToolCall(id="t1", name="weather", arguments={})],
            ),
        ]
        _, anthropic_msgs = provider._extract_system_and_messages(messages)
        assert len(anthropic_msgs) == 1
        assistant_msg = anthropic_msgs[0]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == [
            {"type": "tool_use", "id": "t1", "name": "weather", "input": {}}
        ]

    # 6. _build_anthropic_tools — renames parameters to input_schema
    def test_build_anthropic_tools_renames_parameters_to_input_schema(self):
        """parameters key is renamed to input_schema; parameters key absent."""
        provider = AnthropicProvider()
        spec = [{"name": "weather", "description": "Get weather", "parameters": {"type": "object"}}]
        result = provider._build_anthropic_tools(spec)
        assert result[0]["input_schema"] == {"type": "object"}
        assert "parameters" not in result[0]

    # 7. _parse_response — text block
    def test_parse_response_text_block(self):
        """Text block is parsed into content; tool_calls is empty."""
        provider = AnthropicProvider()
        response = _fake_response([_fake_text_block("Hello")])
        result = provider._parse_response(response)
        assert result.content == "Hello"
        assert result.tool_calls == []

    # 8. _parse_response — tool_use block
    def test_parse_response_tool_use_block(self):
        """tool_use block is parsed into a ToolCall."""
        provider = AnthropicProvider()
        response = _fake_response([
            _fake_tool_use_block(id="t1", name="weather", input_data={"location": "Tokyo"})
        ])
        result = provider._parse_response(response)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "t1"
        assert result.tool_calls[0].name == "weather"
        assert result.tool_calls[0].arguments == {"location": "Tokyo"}

    # 9. _parse_response — mixed text and tool
    def test_parse_response_mixed_text_and_tool(self):
        """Mixed response has both content and tool_calls."""
        provider = AnthropicProvider()
        response = _fake_response([
            _fake_text_block("Checking weather"),
            _fake_tool_use_block(id="t1", name="weather", input_data={"location": "NYC"}),
        ])
        result = provider._parse_response(response)
        assert result.content != ""
        assert len(result.tool_calls) == 1

    # 10. generate passes system kwarg
    def test_anthropic_generate_passes_system_kwarg(self):
        """generate() passes system= keyword argument to messages.create."""
        provider = AnthropicProvider()
        self.mock_client.messages.create.return_value = _fake_response()

        messages = [
            Message("user", "You are helpful"),
            Message("assistant", "Ok"),
            Message("user", "Hi"),
        ]
        provider.generate(messages)

        call_kwargs = self.mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful"

    # 11. generate_with_tools passes tools kwarg
    def test_anthropic_generate_with_tools_passes_tools_kwarg(self):
        """generate_with_tools() passes tools= with input_schema format."""
        provider = AnthropicProvider()
        self.mock_client.messages.create.return_value = _fake_response()

        tool_specs = [{"name": "x", "description": "y", "parameters": {"type": "object"}}]
        messages = [
            Message("user", "System"),
            Message("assistant", "Ok"),
            Message("user", "Do it"),
        ]
        provider.generate_with_tools(messages, tools=tool_specs)

        call_kwargs = self.mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == [{"name": "x", "description": "y", "input_schema": {"type": "object"}}]

    # 12. generate_with_tools with no tools omits tools kwarg
    def test_anthropic_generate_with_no_tools_omits_tools_kwarg(self):
        """generate_with_tools() with empty tools list omits tools kwarg."""
        provider = AnthropicProvider()
        self.mock_client.messages.create.return_value = _fake_response()

        messages = [
            Message("user", "System"),
            Message("assistant", "Ok"),
            Message("user", "Hello"),
        ]
        provider.generate_with_tools(messages, tools=[])

        call_kwargs = self.mock_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs

    # 13. rate limit mapping
    def test_anthropic_rate_limit_raises_pai_rate_limit_error(self):
        """generate() maps anthropic.RateLimitError to PAI RateLimitError."""
        import anthropic as _anthropic

        provider = AnthropicProvider()

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_response.json.return_value = {"error": {"message": "rate limited"}}

        self.mock_client.messages.create.side_effect = _anthropic.RateLimitError(
            message="rate limited",
            response=mock_response,
            body=None,
        )

        with pytest.raises(RateLimitError):
            provider.generate([
                Message("user", "System"),
                Message("assistant", "Ok"),
                Message("user", "Hi"),
            ])

    # 14. API error mapping
    def test_anthropic_api_error_raises_provider_error(self):
        """generate() maps anthropic.APIError to PAI ProviderError."""
        import anthropic as _anthropic

        provider = AnthropicProvider()

        mock_request = MagicMock()
        self.mock_client.messages.create.side_effect = _anthropic.APIError(
            message="server error",
            request=mock_request,
            body=None,
        )

        with pytest.raises(ProviderError):
            provider.generate([
                Message("user", "System"),
                Message("assistant", "Ok"),
                Message("user", "Hi"),
            ])

    # 15. health_check passes
    def test_anthropic_health_check_passes(self):
        """health_check() returns True on successful response."""
        provider = AnthropicProvider()
        self.mock_client.messages.create.return_value = _fake_response()
        assert provider.health_check() is True

    # 16. health_check rate-limited returns True
    def test_anthropic_health_check_rate_limited_returns_true(self):
        """health_check() returns True when rate-limited (still reachable)."""
        import anthropic as _anthropic

        provider = AnthropicProvider()

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_response.json.return_value = {"error": {"message": "rate limited"}}

        self.mock_client.messages.create.side_effect = _anthropic.RateLimitError(
            message="rate limited",
            response=mock_response,
            body=None,
        )

        assert provider.health_check() is True

    # 17. generate returns content
    def test_anthropic_generate_returns_content(self):
        """generate() returns LLMResponse with correct content."""
        provider = AnthropicProvider()
        self.mock_client.messages.create.return_value = _fake_response(
            [_fake_text_block("Hello from Claude")]
        )

        messages = [
            Message("user", "System"),
            Message("assistant", "Ok"),
            Message("user", "Hi"),
        ]
        result = provider.generate(messages)
        assert result.content == "Hello from Claude"
