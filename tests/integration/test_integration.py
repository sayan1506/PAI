"""
Integration tests for PAI.

These tests verify end-to-end flows across multiple modules with external
calls mocked at the boundary.
"""

from unittest.mock import MagicMock, patch

import pytest

from core.agent import Agent
from core.health import run_health_check
from providers.base import LLMProvider, LLMResponse, ToolCall
from tools.base import ToolResult
from tools.file_system import FileSystemTool
import config


def _make_mock_provider(response: LLMResponse) -> LLMProvider:
    """Create a mock LLMProvider returning a fixed response."""
    provider = MagicMock(spec=LLMProvider)
    provider.model_name = "mock-model"
    provider.generate.return_value = response
    provider.generate_with_tools.return_value = response
    return provider


class TestAgentSingleTurnNoTools:
    """Agent returns LLM text directly when no tool calls are made."""

    def test_agent_single_turn_no_tools(self):
        response = LLMResponse(content="Hello!", tool_calls=[])
        provider = _make_mock_provider(response)
        agent = Agent(provider=provider)

        result = agent.chat("hi")

        assert result == "Hello!"


class TestAgentSingleTurnWithToolCall:
    """Agent executes a tool call and returns the final LLM response."""

    def test_agent_single_turn_with_tool_call(self):
        # First LLM call returns a tool call
        tool_call_response = LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="t1", name="file_system", arguments={"operation": "list", "path": "."})
            ],
        )
        # Second LLM call returns final text
        final_response = LLMResponse(content="Done", tool_calls=[])

        provider = MagicMock(spec=LLMProvider)
        provider.model_name = "mock-model"
        provider.generate_with_tools.side_effect = [tool_call_response, final_response]

        # Mock the dispatch function so no real tool runs
        with patch("core.agent.dispatch") as mock_dispatch:
            mock_dispatch.return_value = ToolResult(success=True, output="ok")
            agent = Agent(provider=provider)
            result = agent.chat("create a file")

        assert result == "Done"
        mock_dispatch.assert_called_once_with("file_system", {"operation": "list", "path": "."})


class TestHealthReportCriticalFailure:
    """Health check reports critical failure when API key is missing."""

    def test_health_report_critical_failure_on_bad_api_key(self, monkeypatch):
        monkeypatch.setattr(config, "GEMINI_API_KEY", "")
        monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
        monkeypatch.setattr(config, "SKIP_HEALTH_CHECK", False)

        health = run_health_check()

        assert health.critical_failure is True


class TestConfirmationBlocksDeleteOnNo:
    """Confirmation gate blocks file deletion when user says no."""

    def test_confirmation_gate_blocks_file_delete_on_no(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "CONFIRM_DESTRUCTIVE", True)
        monkeypatch.setattr(config, "VOICE_ENABLED", False)
        monkeypatch.setattr(config, "ENABLE_FILE_OPS", True)

        # Mock input() to return "n" and sys.stdin.isatty() to return True
        monkeypatch.setattr("builtins.input", lambda _: "n")
        import sys
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        tmp_file = tmp_path / "deleteme.txt"
        tmp_file.write_text("important data")

        tool = FileSystemTool()
        result = tool.execute(operation="delete", path=str(tmp_file))

        assert result.success is False
        assert "cancelled" in result.error.lower()
        assert tmp_file.exists()


class TestConfirmationAllowsDeleteOnYes:
    """Confirmation gate allows file deletion when user says yes."""

    def test_confirmation_gate_allows_file_delete_on_yes(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "CONFIRM_DESTRUCTIVE", True)
        monkeypatch.setattr(config, "VOICE_ENABLED", False)
        monkeypatch.setattr(config, "ENABLE_FILE_OPS", True)

        # Mock input() to return "y" and sys.stdin.isatty() to return True
        monkeypatch.setattr("builtins.input", lambda _: "y")
        import sys
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        tmp_file = tmp_path / "deleteme.txt"
        tmp_file.write_text("important data")

        tool = FileSystemTool()
        result = tool.execute(operation="delete", path=str(tmp_file))

        assert result.success is True
        assert not tmp_file.exists()
