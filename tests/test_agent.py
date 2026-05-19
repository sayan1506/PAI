"""
Property-based tests for Agent conversation invariants.

Feature: pai-phase1-foundation, Property 4: Agent chat preserves conversation invariants
Feature: pai-phase1-foundation, Property 5: History trimming preserves system prompt and respects bounds
"""

from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from core.agent import Agent
from providers.base import LLMProvider, LLMResponse, Message
import config


def make_mock_provider(response_content: str) -> LLMProvider:
    """Create a mock LLMProvider that returns a controlled response."""
    provider = MagicMock(spec=LLMProvider)
    provider.model_name = "mock-model"
    provider.generate.return_value = LLMResponse(content=response_content)
    return provider


class TestAgentConversationInvariants:
    """Property 4: Agent chat preserves conversation invariants

    For any user message string, calling agent.chat(message) SHALL:
    (a) increase the conversation history length by exactly 2,
    (b) append the user message and the provider's response to history, and
    (c) return the provider's response content as the return value.

    **Validates: Requirements 9.2, 9.3**
    """

    @given(
        user_message=st.text(min_size=1),
        response_content=st.text(min_size=1),
    )
    @settings(max_examples=200)
    def test_chat_history_grows_by_two(
        self, user_message: str, response_content: str
    ):
        """After chat(), history length increases by exactly 2 (user + assistant).

        **Validates: Requirements 9.2, 9.3**
        """
        # Skip messages that become empty after stripping (Agent returns early)
        if not user_message.strip():
            return

        provider = make_mock_provider(response_content)
        agent = Agent(provider)
        initial_len = len(agent.history)

        agent.chat(user_message)

        assert len(agent.history) == initial_len + 2

    @given(
        user_message=st.text(min_size=1),
        response_content=st.text(min_size=1),
    )
    @settings(max_examples=200)
    def test_chat_appends_correct_messages(
        self, user_message: str, response_content: str
    ):
        """The last two messages after chat() are the user message and provider response.

        **Validates: Requirements 9.2, 9.3**
        """
        if not user_message.strip():
            return

        provider = make_mock_provider(response_content)
        agent = Agent(provider)

        agent.chat(user_message)

        # Last two messages should be user input (stripped) and assistant response (stripped)
        user_msg = agent.history[-2]
        assistant_msg = agent.history[-1]

        assert user_msg.role == "user"
        assert user_msg.content == user_message.strip()
        assert assistant_msg.role == "assistant"
        assert assistant_msg.content == response_content.strip()

    @given(
        user_message=st.text(min_size=1),
        response_content=st.text(min_size=1),
    )
    @settings(max_examples=200)
    def test_chat_return_value_matches_provider_response(
        self, user_message: str, response_content: str
    ):
        """The return value of chat() equals the provider's response content (stripped).

        **Validates: Requirements 9.2, 9.3**
        """
        if not user_message.strip():
            return

        provider = make_mock_provider(response_content)
        agent = Agent(provider)

        result = agent.chat(user_message)

        assert result == response_content.strip()


class TestHistoryTrimmingProperty:
    """Property 5: History trimming preserves system prompt and respects bounds

    For any conversation history that exceeds MAX_SESSION_TURNS * 2 messages,
    after trimming: (a) the first two messages (system prompt pair) SHALL be
    preserved unchanged, and (b) the total history length SHALL not exceed
    MAX_SESSION_TURNS * 2 + 2.

    **Validates: Requirements 9.4**
    """

    @given(
        num_turns=st.integers(min_value=6, max_value=30),
    )
    @settings(max_examples=200)
    def test_history_trimming_preserves_system_prompt_and_respects_bounds(
        self, num_turns: int
    ):
        """After exceeding MAX_SESSION_TURNS, system prompt is preserved and
        history length stays within bounds.

        **Validates: Requirements 9.4**
        """
        # Use a small MAX_SESSION_TURNS so trimming triggers quickly
        max_turns = 4
        original_max = config.MAX_SESSION_TURNS
        config.MAX_SESSION_TURNS = max_turns

        try:
            provider = make_mock_provider("response")
            agent = Agent(provider)

            # Capture the original system prompt messages before any chat
            original_system_msg_0 = agent.history[0]
            original_system_msg_1 = agent.history[1]

            # Send enough messages to exceed the limit and trigger trimming
            for i in range(num_turns):
                agent.chat(f"message {i}")

            # (a) The first two messages (system prompt pair) are preserved unchanged
            assert len(agent.history) >= 2, "History must have at least the system prompt pair"
            assert agent.history[0].role == original_system_msg_0.role
            assert agent.history[0].content == original_system_msg_0.content
            assert agent.history[1].role == original_system_msg_1.role
            assert agent.history[1].content == original_system_msg_1.content

            # (b) Total history length does not exceed MAX_SESSION_TURNS * 2 + 2
            max_allowed = max_turns * 2 + 2
            assert len(agent.history) <= max_allowed, (
                f"History length {len(agent.history)} exceeds max allowed {max_allowed}"
            )
        finally:
            config.MAX_SESSION_TURNS = original_max


class TestAgentResetProperty:
    """Property 6: Agent reset restores initial state

    For any conversation state (regardless of history length or content),
    calling agent.reset() SHALL result in a history containing only the
    system prompt messages, equivalent to a freshly initialized Agent.

    **Validates: Requirements 9.5**
    """

    @given(
        num_turns=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=200)
    def test_reset_restores_initial_state(self, num_turns: int):
        """After sending num_turns messages and calling reset(), history
        equals the history of a freshly initialized Agent (just the 2 system
        prompt messages).

        **Validates: Requirements 9.5**
        """
        provider = make_mock_provider("response")
        agent = Agent(provider)

        # Send random number of messages to build up conversation state
        for i in range(num_turns):
            agent.chat(f"message {i}")

        # Verify history has grown beyond initial state
        assert len(agent.history) > 2

        # Call reset
        agent.reset()

        # Create a fresh agent to compare against
        fresh_agent = Agent(provider)

        # Verify history after reset equals a freshly initialized Agent
        assert len(agent.history) == len(fresh_agent.history) == 2
        assert agent.history[0].role == fresh_agent.history[0].role
        assert agent.history[0].content == fresh_agent.history[0].content
        assert agent.history[1].role == fresh_agent.history[1].role
        assert agent.history[1].content == fresh_agent.history[1].content
