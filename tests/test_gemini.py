"""
Property-based tests for Gemini message conversion.

Feature: Core Foundation
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


class TestGeminiMessageConversion:
    """Property 1: Gemini message conversion preserves content

    For any list of Message objects, converting them to Gemini API format
    SHALL preserve all message content strings and correctly map roles
    ("user" → "user", "assistant" → "model").

    Validates: Requirements 6.2
    """

    @given(messages=message_list_strategy)
    @settings(max_examples=200, deadline=None)
    def test_gemini_conversion_preserves_content(self, messages: list[Message]) -> None:
        """Property 1: Gemini message conversion preserves content

        **Validates: Requirements 6.2**
        """
        # Mock the google.generativeai module to avoid needing a real API key
        with patch("providers.gemini.genai") as mock_genai:
            # Set up the mock model and its generate_content method
            mock_model = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model

            mock_response = MagicMock()
            mock_response.text = "mocked response"
            mock_model.generate_content.return_value = mock_response

            # Import and instantiate the provider with mocked SDK
            from providers.gemini import GeminiProvider

            provider = GeminiProvider()

            # Call generate to trigger the conversion
            provider.generate(messages)

            # Capture the contents argument passed to generate_content
            mock_model.generate_content.assert_called_once()
            contents = mock_model.generate_content.call_args[0][0]

            # Verify: same number of messages
            assert len(contents) == len(messages)

            # Verify: content preserved and roles mapped correctly
            for original, converted in zip(messages, contents):
                # Content is preserved in the parts list
                assert converted["parts"] == [original.content]

                # Role mapping: "assistant" → "model", everything else → "user"
                expected_role = "model" if original.role == "assistant" else "user"
                assert converted["role"] == expected_role
