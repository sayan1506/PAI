"""
Unit tests for the Agent's agentic tool loop (core/agent.py).

Tests verify that the Agent correctly:
- Falls back to plain generate() when no tools are available
- Executes tool calls and loops back to the LLM with results
- Maintains correct history after tool execution
- Handles multiple tool calls in a single turn
- Respects MAX_TOOL_ITERATIONS and returns a fallback message
- Handles ToolError gracefully and continues the loop
- Does not split tool-call blocks during history trimming
"""

from unittest.mock import patch, MagicMock

import pytest

import config
from core.agent import Agent
from core.exceptions import ToolError
from providers.base import LLMProvider, LLMResponse, Message, ToolCall
from tools.base import ToolResult


def make_mock_provider() -> MagicMock:
    """Create a mock LLMProvider with generate and generate_with_tools."""
    provider = MagicMock(spec=LLMProvider)
    provider.model_name = "mock-model"
    return provider


class TestAgentToolLoop:
    """Tests for the agentic tool loop in Agent.chat()."""

    def test_no_tools_falls_back_to_generate(self):
        """When get_tools() returns empty list, plain generate() is used."""
        provider = make_mock_provider()
        provider.generate.return_value = LLMResponse(content="Hello there!")

        with patch("core.agent.get_tools", return_value=[]):
            agent = Agent(provider)
            result = agent.chat("Hi")

        provider.generate.assert_called_once()
        provider.generate_with_tools.assert_not_called()
        assert result == "Hello there!"

    def test_tool_call_executes_and_loops(self):
        """LLM returns tool_call, tool runs, LLM called again with result, final text returned."""
        provider = make_mock_provider()

        # First call: LLM requests a tool call
        tool_call = ToolCall(id="tc1", name="file_system", arguments={"operation": "list", "path": "."})
        first_response = LLMResponse(content="", tool_calls=[tool_call])

        # Second call: LLM returns final text
        second_response = LLMResponse(content="Here are your files.")

        provider.generate_with_tools.side_effect = [first_response, second_response]

        mock_tool = MagicMock()
        mock_tool.name = "file_system"
        mock_tool.to_function_spec.return_value = {"name": "file_system", "description": "FS", "parameters": {}}

        tool_result = ToolResult(success=True, output="file1.txt\nfile2.txt")

        with patch("core.agent.get_tools", return_value=[mock_tool]), \
             patch("core.agent.dispatch", return_value=tool_result) as mock_dispatch:
            agent = Agent(provider)
            result = agent.chat("List my files")

        mock_dispatch.assert_called_once_with("file_system", {"operation": "list", "path": "."})
        assert provider.generate_with_tools.call_count == 2
        assert result == "Here are your files."

    def test_tool_call_result_in_history(self):
        """History contains tool result message after execution."""
        provider = make_mock_provider()

        tool_call = ToolCall(id="tc1", name="file_system", arguments={"operation": "read", "path": "x.txt"})
        first_response = LLMResponse(content="", tool_calls=[tool_call])
        second_response = LLMResponse(content="File content is hello.")

        provider.generate_with_tools.side_effect = [first_response, second_response]

        mock_tool = MagicMock()
        mock_tool.name = "file_system"
        mock_tool.to_function_spec.return_value = {"name": "file_system", "description": "FS", "parameters": {}}

        tool_result = ToolResult(success=True, output="hello")

        with patch("core.agent.get_tools", return_value=[mock_tool]), \
             patch("core.agent.dispatch", return_value=tool_result):
            agent = Agent(provider)
            agent.chat("Read x.txt")

        # Find tool messages in history
        tool_messages = [m for m in agent.history if m.role == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0].content == "hello"
        assert tool_messages[0].tool_call_id == "tc1"

    def test_multi_tool_call_single_turn(self):
        """Two tool_calls in one response both get executed."""
        provider = make_mock_provider()

        tc1 = ToolCall(id="tc1", name="file_system", arguments={"operation": "read", "path": "a.txt"})
        tc2 = ToolCall(id="tc2", name="file_system", arguments={"operation": "read", "path": "b.txt"})
        first_response = LLMResponse(content="", tool_calls=[tc1, tc2])
        second_response = LLMResponse(content="Both files read.")

        provider.generate_with_tools.side_effect = [first_response, second_response]

        mock_tool = MagicMock()
        mock_tool.name = "file_system"
        mock_tool.to_function_spec.return_value = {"name": "file_system", "description": "FS", "parameters": {}}

        results = [
            ToolResult(success=True, output="content_a"),
            ToolResult(success=True, output="content_b"),
        ]

        with patch("core.agent.get_tools", return_value=[mock_tool]), \
             patch("core.agent.dispatch", side_effect=results) as mock_dispatch:
            agent = Agent(provider)
            result = agent.chat("Read both files")

        assert mock_dispatch.call_count == 2
        assert result == "Both files read."

        # Verify both tool results are in history
        tool_messages = [m for m in agent.history if m.role == "tool"]
        assert len(tool_messages) == 2
        assert tool_messages[0].content == "content_a"
        assert tool_messages[0].tool_call_id == "tc1"
        assert tool_messages[1].content == "content_b"
        assert tool_messages[1].tool_call_id == "tc2"

    def test_max_iterations_returns_fallback(self, monkeypatch):
        """When LLM always returns tool_call, loop hits max iterations and returns fallback."""
        monkeypatch.setattr(config, "MAX_TOOL_ITERATIONS", 3)

        provider = make_mock_provider()

        # Always return a tool call — never a final text response
        tool_call = ToolCall(id="tc1", name="file_system", arguments={"operation": "list", "path": "."})
        tool_response = LLMResponse(content="", tool_calls=[tool_call])
        provider.generate_with_tools.return_value = tool_response

        mock_tool = MagicMock()
        mock_tool.name = "file_system"
        mock_tool.to_function_spec.return_value = {"name": "file_system", "description": "FS", "parameters": {}}

        tool_result = ToolResult(success=True, output="files")

        with patch("core.agent.get_tools", return_value=[mock_tool]), \
             patch("core.agent.dispatch", return_value=tool_result):
            agent = Agent(provider)
            result = agent.chat("Do something complex")

        assert provider.generate_with_tools.call_count == 3
        assert "wasn't able to complete" in result.lower() or "simpler request" in result.lower()

    def test_tool_error_handled_gracefully(self):
        """dispatch raises ToolError → tool message contains error, loop continues."""
        provider = make_mock_provider()

        tool_call = ToolCall(id="tc1", name="file_system", arguments={"operation": "delete", "path": "/bad"})
        first_response = LLMResponse(content="", tool_calls=[tool_call])
        second_response = LLMResponse(content="Sorry, that file doesn't exist.")

        provider.generate_with_tools.side_effect = [first_response, second_response]

        mock_tool = MagicMock()
        mock_tool.name = "file_system"
        mock_tool.to_function_spec.return_value = {"name": "file_system", "description": "FS", "parameters": {}}

        with patch("core.agent.get_tools", return_value=[mock_tool]), \
             patch("core.agent.dispatch", side_effect=ToolError("File not found")):
            agent = Agent(provider)
            result = agent.chat("Delete /bad")

        # The loop should have continued and returned the second response
        assert result == "Sorry, that file doesn't exist."

        # Verify the tool error message is in history
        tool_messages = [m for m in agent.history if m.role == "tool"]
        assert len(tool_messages) == 1
        assert "error" in tool_messages[0].content.lower() or "Tool error" in tool_messages[0].content

    def test_history_not_split_on_tool_block(self, monkeypatch):
        """Trim doesn't cut mid tool-call block (assistant + tool result pairs stay together)."""
        # Use a very small MAX_SESSION_TURNS to force trimming
        monkeypatch.setattr(config, "MAX_TOOL_ITERATIONS", 5)
        monkeypatch.setattr(config, "MAX_SESSION_TURNS", 3)

        provider = make_mock_provider()

        mock_tool = MagicMock()
        mock_tool.name = "file_system"
        mock_tool.to_function_spec.return_value = {"name": "file_system", "description": "FS", "parameters": {}}

        # Build up history with tool calls to trigger trimming
        # We'll do multiple turns: some with tool calls, some without
        call_count = [0]

        def generate_side_effect(messages, tools):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                # Odd calls: return a tool call
                tc = ToolCall(id=f"tc{call_count[0]}", name="file_system", arguments={"operation": "list", "path": "."})
                return LLMResponse(content="", tool_calls=[tc])
            else:
                # Even calls: return final text
                return LLMResponse(content=f"Response {call_count[0]}")

        provider.generate_with_tools.side_effect = generate_side_effect

        tool_result = ToolResult(success=True, output="result")

        with patch("core.agent.get_tools", return_value=[mock_tool]), \
             patch("core.agent.dispatch", return_value=tool_result):
            agent = Agent(provider)

            # Send enough messages to trigger trimming
            for i in range(8):
                call_count[0] = 0  # Reset for each chat call
                agent.chat(f"Message {i}")

        # After trimming, verify no tool message exists without its preceding
        # assistant message with tool_calls
        history = agent.history
        for i, msg in enumerate(history):
            if msg.role == "tool":
                # There must be a preceding assistant message with tool_calls
                preceding_assistant = None
                for j in range(i - 1, -1, -1):
                    if history[j].role == "assistant" and history[j].tool_calls:
                        preceding_assistant = history[j]
                        break
                    elif history[j].role == "user":
                        # Hit a user message without finding the assistant+tool_calls
                        break
                assert preceding_assistant is not None, (
                    f"Tool message at index {i} has no preceding assistant message with tool_calls"
                )
