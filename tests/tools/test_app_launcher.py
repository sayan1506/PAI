"""
Unit tests for the AppLauncherTool.

Tests verify all app launcher operations (launch, close, list_running, is_running)
work correctly, and that the tool respects the ENABLE_APP_LAUNCHER config flag.

Uses pytest with monkeypatch and mocks for subprocess/psutil isolation.
"""

from unittest.mock import patch, MagicMock

import pytest

import config
from tools.app_launcher import AppLauncherTool
from tools.base import ToolResult


@pytest.fixture
def tool():
    """Return a fresh AppLauncherTool instance."""
    return AppLauncherTool()


@pytest.fixture(autouse=True)
def enable_app_launcher(monkeypatch):
    """Ensure app launcher is enabled by default for all tests."""
    monkeypatch.setattr(config, "ENABLE_APP_LAUNCHER", True)


class TestLaunch:
    """Tests for the launch operation."""

    @patch("tools.app_launcher.find_app_path", return_value="C:\\Windows\\notepad.exe")
    @patch("tools.app_launcher.subprocess.Popen")
    @patch("tools.app_launcher.get_platform", return_value="windows")
    def test_launch_calls_popen(self, mock_platform, mock_popen, mock_find, tool):
        """Popen is called with the resolved path for the app."""
        mock_popen.return_value = MagicMock()

        result = tool.execute(operation="launch", app_name="notepad")

        assert result.success is True
        assert "notepad" in result.output.lower()
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert "notepad.exe" in call_args[0]

    def test_launch_disabled(self, tool, monkeypatch):
        """ENABLE_APP_LAUNCHER=False returns ToolResult with success=False."""
        monkeypatch.setattr(config, "ENABLE_APP_LAUNCHER", False)

        result = tool.execute(operation="launch", app_name="notepad")

        assert result.success is False
        assert "disabled" in result.error.lower()


class TestClose:
    """Tests for the close operation."""

    @patch("tools.app_launcher.psutil.process_iter")
    @patch("tools.app_launcher.get_platform", return_value="windows")
    def test_close_kills_matching_procs(self, mock_platform, mock_process_iter, tool):
        """Close terminates processes matching the resolved app name."""
        mock_proc = MagicMock()
        mock_proc.info = {"name": "notepad.exe"}
        mock_proc.terminate = MagicMock()
        mock_process_iter.return_value = [mock_proc]

        result = tool.execute(operation="close", app_name="notepad")

        assert result.success is True
        assert "1" in result.output
        mock_proc.terminate.assert_called_once()

    @patch("tools.app_launcher.psutil.process_iter")
    @patch("tools.app_launcher.get_platform", return_value="windows")
    def test_close_no_match(self, mock_platform, mock_process_iter, tool):
        """Close with no matching process returns success=False with 'not running' message."""
        mock_proc = MagicMock()
        mock_proc.info = {"name": "chrome.exe"}
        mock_process_iter.return_value = [mock_proc]

        result = tool.execute(operation="close", app_name="notepad")

        assert result.success is False
        assert "not running" in result.error.lower()


class TestListRunning:
    """Tests for the list_running operation."""

    @patch("tools.app_launcher.psutil.process_iter")
    def test_list_running_returns_names(self, mock_process_iter, tool):
        """list_running returns deduplicated sorted process names."""
        proc1 = MagicMock()
        proc1.info = {"name": "chrome.exe"}
        proc2 = MagicMock()
        proc2.info = {"name": "notepad.exe"}
        proc3 = MagicMock()
        proc3.info = {"name": "chrome.exe"}  # duplicate
        mock_process_iter.return_value = [proc1, proc2, proc3]

        result = tool.execute(operation="list_running")

        assert result.success is True
        assert "chrome.exe" in result.output
        assert "notepad.exe" in result.output
        # Deduplicated: chrome.exe should appear only once
        assert result.output.count("chrome.exe") == 1


class TestIsRunning:
    """Tests for the is_running operation."""

    @patch("tools.app_launcher.psutil.process_iter")
    @patch("tools.app_launcher.get_platform", return_value="windows")
    def test_is_running_true(self, mock_platform, mock_process_iter, tool):
        """is_running returns 'running' when process is found."""
        mock_proc = MagicMock()
        mock_proc.info = {"name": "notepad.exe"}
        mock_process_iter.return_value = [mock_proc]

        result = tool.execute(operation="is_running", app_name="notepad")

        assert result.success is True
        assert "running" in result.output.lower()
        assert "not running" not in result.output.lower()

    @patch("tools.app_launcher.psutil.process_iter")
    @patch("tools.app_launcher.get_platform", return_value="windows")
    def test_is_running_false(self, mock_platform, mock_process_iter, tool):
        """is_running returns 'not running' when no matching process found."""
        mock_proc = MagicMock()
        mock_proc.info = {"name": "chrome.exe"}
        mock_process_iter.return_value = [mock_proc]

        result = tool.execute(operation="is_running", app_name="notepad")

        assert result.success is True
        assert "not running" in result.output.lower()
