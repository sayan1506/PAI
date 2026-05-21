"""
Unit tests for the TerminalTool.

Tests verify command execution, denylist enforcement, timeout handling,
and that the tool respects the ENABLE_TERMINAL config flag.

Uses pytest with monkeypatch and mocks for subprocess isolation.
"""

import subprocess
from unittest.mock import patch, MagicMock

import pytest

import config
from tools.terminal import TerminalTool
from tools.base import ToolResult


@pytest.fixture
def tool():
    """Return a fresh TerminalTool instance."""
    return TerminalTool()


@pytest.fixture(autouse=True)
def enable_terminal(monkeypatch):
    """Enable terminal by default for tests (override per-test as needed)."""
    monkeypatch.setattr(config, "ENABLE_TERMINAL", True)


class TestDisabledConfig:
    """Tests for when ENABLE_TERMINAL is False."""

    def test_disabled_by_default(self, tool, monkeypatch):
        """ENABLE_TERMINAL=False → success=False with disabled message."""
        monkeypatch.setattr(config, "ENABLE_TERMINAL", False)

        result = tool.execute(command="echo hello")

        assert result.success is False
        assert "disabled" in result.error.lower()
        assert result.output == ""


class TestCommandExecution:
    """Tests for successful and failed command execution."""

    @patch("tools.terminal.subprocess.run")
    def test_run_command_success(self, mock_run, tool):
        """subprocess returns rc=0 → output captured, success=True."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="hello world",
            stderr="",
        )

        result = tool.execute(command="echo hello world")

        assert result.success is True
        assert "hello world" in result.output
        assert result.error == ""
        mock_run.assert_called_once()
        # On Windows, uses PowerShell; on Linux, uses shell=True
        call_args = mock_run.call_args
        assert call_args[1]["timeout"] == 30
        assert call_args[1]["capture_output"] is True

    @patch("tools.terminal.subprocess.run")
    def test_run_command_nonzero(self, mock_run, tool):
        """subprocess returns rc=1 → success=False with exit code in error."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="command not found",
        )

        result = tool.execute(command="badcommand")

        assert result.success is False
        assert "command not found" in result.output
        assert "exit code" in result.error.lower()


class TestDenylist:
    """Tests for the denylist pattern matching."""

    @patch("tools.terminal.subprocess.run")
    def test_denied_pattern_rm_rf(self, mock_run, tool):
        """'rm -rf /' is refused and subprocess is never called."""
        result = tool.execute(command="rm -rf /")

        assert result.success is False
        assert "refused" in result.error.lower() or "blocked" in result.error.lower()
        mock_run.assert_not_called()

    @patch("tools.terminal.subprocess.run")
    def test_denied_pattern_shutdown(self, mock_run, tool):
        """'shutdown now' is refused and subprocess is never called."""
        result = tool.execute(command="shutdown now")

        assert result.success is False
        assert "refused" in result.error.lower() or "blocked" in result.error.lower()
        mock_run.assert_not_called()


class TestTimeout:
    """Tests for timeout handling."""

    @patch("tools.terminal.subprocess.run")
    def test_timeout(self, mock_run, tool):
        """TimeoutExpired → success=False with clear timeout message."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 60", timeout=30)

        result = tool.execute(command="sleep 60")

        assert result.success is False
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_command(self, tool):
        """Empty command string → success=False with error message."""
        result = tool.execute(command="")

        assert result.success is False
        assert result.output == ""
        assert "no command" in result.error.lower()
