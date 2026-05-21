"""
Unit tests for the tool registry (tools/__init__.py).

Tests verify that get_tools() respects config flags and that dispatch()
routes to the correct tool or raises ToolError for unknown names.

Uses pytest with monkeypatch to control config flags.
"""

from unittest.mock import patch, MagicMock

import pytest

import config
from tools import get_tools, dispatch
from tools.file_system import FileSystemTool
from tools.app_launcher import AppLauncherTool
from tools.terminal import TerminalTool
from tools.base import ToolResult
from core.exceptions import ToolError


class TestGetTools:
    """Tests for get_tools() returning enabled tool instances based on config."""

    def test_file_ops_enabled(self, monkeypatch):
        """ENABLE_FILE_OPS=True → FileSystemTool in list."""
        monkeypatch.setattr(config, "ENABLE_FILE_OPS", True)
        monkeypatch.setattr(config, "ENABLE_APP_LAUNCHER", False)
        monkeypatch.setattr(config, "ENABLE_TERMINAL", False)

        tools = get_tools()

        assert any(isinstance(t, FileSystemTool) for t in tools)

    def test_file_ops_disabled(self, monkeypatch):
        """ENABLE_FILE_OPS=False → FileSystemTool not in list."""
        monkeypatch.setattr(config, "ENABLE_FILE_OPS", False)
        monkeypatch.setattr(config, "ENABLE_APP_LAUNCHER", False)
        monkeypatch.setattr(config, "ENABLE_TERMINAL", False)

        tools = get_tools()

        assert not any(isinstance(t, FileSystemTool) for t in tools)

    def test_terminal_disabled_default(self, monkeypatch):
        """ENABLE_TERMINAL=False → TerminalTool not in list."""
        monkeypatch.setattr(config, "ENABLE_FILE_OPS", False)
        monkeypatch.setattr(config, "ENABLE_APP_LAUNCHER", False)
        monkeypatch.setattr(config, "ENABLE_TERMINAL", False)

        tools = get_tools()

        assert not any(isinstance(t, TerminalTool) for t in tools)

    def test_terminal_enabled(self, monkeypatch):
        """ENABLE_TERMINAL=True → TerminalTool in list."""
        monkeypatch.setattr(config, "ENABLE_FILE_OPS", False)
        monkeypatch.setattr(config, "ENABLE_APP_LAUNCHER", False)
        monkeypatch.setattr(config, "ENABLE_TERMINAL", True)

        tools = get_tools()

        assert any(isinstance(t, TerminalTool) for t in tools)

    def test_all_enabled(self, monkeypatch):
        """All flags True → 6 tools returned."""
        monkeypatch.setattr(config, "ENABLE_FILE_OPS", True)
        monkeypatch.setattr(config, "ENABLE_APP_LAUNCHER", True)
        monkeypatch.setattr(config, "ENABLE_TERMINAL", True)
        monkeypatch.setattr(config, "BROWSER_ENABLED", True)
        monkeypatch.setattr(config, "VISION_ENABLED", True)
        monkeypatch.setattr(config, "MEMORY_ENABLED", True)

        tools = get_tools()

        assert len(tools) == 6
        assert any(isinstance(t, FileSystemTool) for t in tools)
        assert any(isinstance(t, AppLauncherTool) for t in tools)
        assert any(isinstance(t, TerminalTool) for t in tools)


class TestDispatch:
    """Tests for dispatch() routing and error handling."""

    def test_dispatch_calls_correct_tool(self, monkeypatch):
        """dispatch("file_system", {...}) calls FileSystemTool."""
        monkeypatch.setattr(config, "ENABLE_FILE_OPS", True)
        monkeypatch.setattr(config, "ENABLE_APP_LAUNCHER", False)
        monkeypatch.setattr(config, "ENABLE_TERMINAL", False)

        with patch.object(FileSystemTool, "execute", return_value=ToolResult(success=True, output="ok")) as mock_exec:
            result = dispatch("file_system", {"operation": "list", "path": "."})

        mock_exec.assert_called_once_with(operation="list", path=".")
        assert result.success is True
        assert result.output == "ok"

    def test_dispatch_unknown_tool_raises(self, monkeypatch):
        """ToolError raised for unknown tool name."""
        monkeypatch.setattr(config, "ENABLE_FILE_OPS", True)
        monkeypatch.setattr(config, "ENABLE_APP_LAUNCHER", True)
        monkeypatch.setattr(config, "ENABLE_TERMINAL", True)

        with pytest.raises(ToolError):
            dispatch("nonexistent_tool", {})

    def test_dispatch_passes_arguments(self, monkeypatch):
        """kwargs forwarded correctly to the tool's execute method."""
        monkeypatch.setattr(config, "ENABLE_FILE_OPS", False)
        monkeypatch.setattr(config, "ENABLE_APP_LAUNCHER", False)
        monkeypatch.setattr(config, "ENABLE_TERMINAL", True)

        with patch.object(TerminalTool, "execute", return_value=ToolResult(success=True, output="done")) as mock_exec:
            result = dispatch("terminal", {"command": "echo hi"})

        mock_exec.assert_called_once_with(command="echo hi")
        assert result.success is True
        assert result.output == "done"
