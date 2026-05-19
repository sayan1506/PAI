"""
Property-based tests for Ollama message formatting.

Feature: pai-phase1-foundation
"""

from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from providers.base import Message


# Strategy for generating valid message roles
roles_strategy = st.sampled_from(["user", "assistant", "system"])

# Strategy for generating a single Message with non-empty content
message_strategy = st.builds(
    Message,
    role=roles_strategy,
    content=st.text(min_size=1, max_size=200),
)

# Strategy for generating a non-empty list of Messages
message_list_strategy = st.lists(message_strategy, min_size=1, max_size=20)


class TestOllamaMessageFormatting:
    """Property 2: Ollama message formatting preserves content

    For any list of Message objects, formatting them as the Ollama JSON payload
    SHALL produce a messages array where each entry contains the original role
    and content, and the payload includes stream: false.

    Validates: Requirements 7.2
    """

    @given(messages=message_list_strategy)
    @settings(max_examples=200, deadline=None)
    def test_ollama_formatting_preserves_content(self, messages: list[Message]) -> None:
        """Property 2: Ollama message formatting preserves content

        **Validates: Requirements 7.2**
        """
        # Mock requests.post to capture the JSON payload sent to Ollama
        with patch("providers.ollama.requests.post") as mock_post:
            # Set up the mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "message": {"role": "assistant", "content": "mocked response"}
            }
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            # Import and instantiate the provider with mocked config
            from providers.ollama import OllamaProvider

            provider = OllamaProvider()

            # Call generate to trigger the formatting
            provider.generate(messages)

            # Capture the JSON payload passed to requests.post
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

            # Verify: payload includes "stream": False
            assert payload["stream"] is False

            # Verify: payload includes "model" field
            assert "model" in payload
            assert isinstance(payload["model"], str)
            assert len(payload["model"]) > 0

            # Verify: same number of messages in payload
            assert len(payload["messages"]) == len(messages)

            # Verify: each message preserves original role and content
            for original, formatted in zip(messages, payload["messages"]):
                assert formatted["role"] == original.role
                assert formatted["content"] == original.content
