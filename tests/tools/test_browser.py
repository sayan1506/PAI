"""
tests/tools/test_browser.py

Comprehensive unit tests for BrowserTool, CDPClient, and ChromeLauncher.
All external dependencies (subprocess, pyautogui, websocket, requests, time.sleep)
are mocked — no real browser is launched.
"""

import json
from unittest.mock import Mock, patch

import pytest
import requests as requests_lib

# Import directly from submodules to avoid tools/__init__.py which pulls in
# unrelated dependencies (psutil, etc.)
from tools.base import BaseTool, ToolResult
from tools.browser import BrowserTool
from tools.browser_cdp import CDPClient
from tools.browser_launcher import ChromeLauncher


# ═══════════════════════════════════════════════════════════════════════════════
# BrowserTool tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestBrowserToolMetadata:
    """Tests for BrowserTool name and description properties."""

    def test_browser_tool_name_and_description(self):
        tool = BrowserTool()
        assert tool.name == "browser"
        assert "Chrome" in tool.description
        assert "action=" in tool.description


class TestBrowserToolExecute:
    """Tests for BrowserTool.execute routing and error handling."""

    def test_execute_unknown_action_returns_failure(self):
        tool = BrowserTool()
        # Patch _ensure_loaded so it doesn't try to launch Chrome
        with patch.object(tool, "_ensure_loaded"):
            result = tool.execute(action="fly")
        assert result.success is False
        assert "Unknown action" in result.error

    def test_execute_wraps_exception_as_toolresult(self):
        tool = BrowserTool()
        with patch.object(tool, "_ensure_loaded", side_effect=RuntimeError("boom")):
            result = tool.execute(action="read_page")
        assert result.success is False
        assert "boom" in result.error


class TestBrowserToolOpenUrl:
    """Tests for BrowserTool._open_url."""

    @patch("tools.browser.time.sleep")
    @patch("tools.browser.pyautogui.press")
    @patch("tools.browser.pyautogui.typewrite")
    @patch("tools.browser.pyautogui.hotkey")
    def test_open_url_normalises_bare_domain(
        self, mock_hotkey, mock_typewrite, mock_press, mock_sleep
    ):
        tool = BrowserTool()
        tool._loaded = True  # skip _ensure_loaded
        tool._open_url("youtube.com")
        # typewrite should be called with the normalised URL
        mock_typewrite.assert_called_once()
        assert mock_typewrite.call_args[0][0] == "https://youtube.com"

    @patch("tools.browser.time.sleep")
    @patch("tools.browser.pyautogui.press")
    @patch("tools.browser.pyautogui.typewrite")
    @patch("tools.browser.pyautogui.hotkey")
    def test_open_url_preserves_existing_scheme(
        self, mock_hotkey, mock_typewrite, mock_press, mock_sleep
    ):
        tool = BrowserTool()
        tool._loaded = True
        tool._open_url("https://github.com/search")
        mock_typewrite.assert_called_once()
        assert mock_typewrite.call_args[0][0] == "https://github.com/search"

    @patch("tools.browser.time.sleep")
    @patch("tools.browser.pyautogui.press")
    @patch("tools.browser.pyautogui.typewrite")
    @patch("tools.browser.pyautogui.hotkey")
    def test_open_url_returns_success_with_url(
        self, mock_hotkey, mock_typewrite, mock_press, mock_sleep
    ):
        tool = BrowserTool()
        tool._loaded = True
        result = tool._open_url("youtube.com")
        assert result.success is True
        assert "youtube.com" in result.output


class TestBrowserToolSearch:
    """Tests for BrowserTool._search."""

    def test_search_builds_google_url(self):
        tool = BrowserTool()
        tool._loaded = True
        captured_url = {}

        def fake_open_url(url):
            captured_url["url"] = url
            return ToolResult(success=True, output="ok")

        with patch.object(tool, "_open_url", side_effect=fake_open_url):
            tool._search("python tutorials")

        url = captured_url["url"]
        assert "google.com/search" in url
        assert "python+tutorials" in url

    def test_search_returns_success(self):
        tool = BrowserTool()
        tool._loaded = True
        with patch.object(
            tool, "_open_url", return_value=ToolResult(success=True, output="ok")
        ):
            result = tool._search("hello world")
        assert result.success is True


class TestBrowserToolScroll:
    """Tests for BrowserTool._scroll."""

    @patch("tools.browser.pyautogui.scroll")
    @patch("tools.browser.pyautogui.moveTo")
    @patch("tools.browser.pyautogui.size", return_value=(1920, 1080))
    def test_scroll_down_calls_pyautogui(self, mock_size, mock_moveTo, mock_scroll):
        tool = BrowserTool()
        tool._loaded = True
        tool._scroll("down", 3)
        mock_scroll.assert_called_once_with(-3)

    @patch("tools.browser.pyautogui.scroll")
    @patch("tools.browser.pyautogui.moveTo")
    @patch("tools.browser.pyautogui.size", return_value=(1920, 1080))
    def test_scroll_up_calls_pyautogui(self, mock_size, mock_moveTo, mock_scroll):
        tool = BrowserTool()
        tool._loaded = True
        tool._scroll("up", 5)
        mock_scroll.assert_called_once_with(5)


class TestBrowserToolReadPage:
    """Tests for BrowserTool._read_page."""

    def test_read_page_returns_text(self):
        tool = BrowserTool()
        tool._loaded = True
        tool._launcher = Mock()
        tool._launcher.cdp.get_page_text.return_value = "Hello world page content"
        result = tool._read_page()
        assert result.success is True
        assert result.output == "Hello world page content"

    def test_read_page_truncates_long_content(self):
        tool = BrowserTool()
        tool._loaded = True
        tool._launcher = Mock()
        tool._launcher.cdp.get_page_text.return_value = "x" * 10_000
        result = tool._read_page()
        assert len(result.output) <= 8100  # 8000 chars + truncation note
        assert "[truncated" in result.output

    def test_read_page_cdp_failure_returns_error(self):
        tool = BrowserTool()
        tool._loaded = True
        tool._launcher = Mock()
        tool._launcher.cdp.get_page_text.side_effect = RuntimeError("connection lost")
        # Use execute so the exception is caught by the try/except wrapper
        with patch.object(tool, "_ensure_loaded"):
            result = tool.execute(action="read_page")
        assert result.success is False
        assert "connection lost" in result.error


class TestBrowserToolClose:
    """Tests for BrowserTool._close."""

    def test_close_resets_loaded_flag(self):
        tool = BrowserTool()
        tool._loaded = True
        tool._launcher = Mock()
        result = tool._close()
        assert tool._loaded is False
        assert result.success is True


# ═══════════════════════════════════════════════════════════════════════════════
# CDPClient tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCDPClientReachability:
    """Tests for CDPClient.is_reachable."""

    @patch("tools.browser_cdp.requests.get")
    def test_cdp_is_reachable_true_on_200(self, mock_get):
        mock_get.return_value = Mock(status_code=200)
        client = CDPClient()
        assert client.is_reachable() is True

    @patch("tools.browser_cdp.requests.get")
    def test_cdp_is_reachable_false_on_connection_error(self, mock_get):
        mock_get.side_effect = requests_lib.ConnectionError("refused")
        client = CDPClient()
        assert client.is_reachable() is False


class TestCDPClientCommands:
    """Tests for CDPClient.send and get_page_text."""

    def test_cdp_get_page_text_strips_excess_whitespace(self):
        client = CDPClient()
        mock_ws = Mock()
        client._ws = mock_ws

        # Mock recv to return a response with excess whitespace in value
        response = {
            "id": 1,
            "result": {"result": {"value": "  Hello\n\n\n\nWorld  "}},
        }
        mock_ws.recv.return_value = json.dumps(response)

        text = client.get_page_text()
        # Should strip leading/trailing whitespace and collapse 3+ newlines to 2
        assert text == "Hello\n\nWorld"

    def test_cdp_send_raises_on_cdp_error(self):
        client = CDPClient()
        mock_ws = Mock()
        client._ws = mock_ws

        # Mock recv to return an error response
        error_response = {
            "id": 1,
            "error": {"message": "Target closed"},
        }
        mock_ws.recv.return_value = json.dumps(error_response)

        with pytest.raises(RuntimeError, match="Target closed"):
            client.send("Page.navigate", {"url": "x"})


# ═══════════════════════════════════════════════════════════════════════════════
# ChromeLauncher tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestChromeLauncherEnsureRunning:
    """Tests for ChromeLauncher.ensure_running."""

    @patch("tools.browser_launcher.subprocess.Popen")
    @patch.object(CDPClient, "connect")
    @patch.object(CDPClient, "is_reachable", return_value=True)
    def test_launcher_attaches_without_launching_if_port_reachable(
        self, mock_reachable, mock_connect, mock_popen
    ):
        launcher = ChromeLauncher()
        launcher.ensure_running()
        mock_popen.assert_not_called()
        assert launcher._proc is None

    @patch.object(CDPClient, "connect")
    @patch.object(ChromeLauncher, "_get_chrome_path", return_value=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    @patch("tools.browser_launcher.subprocess.Popen")
    @patch.object(CDPClient, "is_reachable")
    def test_launcher_launches_chrome_if_port_not_reachable(
        self, mock_reachable, mock_popen, mock_get_path, mock_connect
    ):
        # First call: not reachable (triggers launch)
        # Second call: reachable (in _wait_for_port)
        mock_reachable.side_effect = [False, True]
        mock_popen.return_value = Mock()

        launcher = ChromeLauncher()
        launcher.ensure_running()

        mock_popen.assert_called_once()
        call_args_str = str(mock_popen.call_args)
        assert "--remote-debugging-port" in call_args_str

    @patch.object(ChromeLauncher, "_get_chrome_path", side_effect=FileNotFoundError("not found"))
    @patch.object(CDPClient, "is_reachable", return_value=False)
    def test_launcher_raises_if_chrome_binary_not_found(
        self, mock_reachable, mock_get_path
    ):
        launcher = ChromeLauncher()
        with pytest.raises(RuntimeError):
            launcher.ensure_running()


class TestChromeLauncherClose:
    """Tests for ChromeLauncher.close."""

    def test_launcher_close_terminates_owned_process(self):
        launcher = ChromeLauncher()
        mock_proc = Mock()
        launcher._proc = mock_proc
        launcher.cdp = Mock()

        launcher.close()

        mock_proc.terminate.assert_called_once()
        launcher.cdp.disconnect.assert_called_once()

    def test_launcher_close_only_disconnects_if_not_owner(self):
        launcher = ChromeLauncher()
        launcher._proc = None  # attached, not launched
        launcher.cdp = Mock()

        launcher.close()

        launcher.cdp.disconnect.assert_called_once()
        # No terminate call since _proc is None — nothing to terminate


class TestChromeLauncherGetChromePath:
    """Tests for ChromeLauncher._get_chrome_path."""

    def test_get_chrome_path_returns_config_binary_when_set(self, monkeypatch):
        monkeypatch.setattr("config.BROWSER_BINARY", r"C:\custom\chrome.exe")
        launcher = ChromeLauncher()
        result = launcher._get_chrome_path()
        assert result == r"C:\custom\chrome.exe"

    def test_get_chrome_path_falls_through_candidates_on_linux(self, monkeypatch):
        monkeypatch.setattr("config.BROWSER_BINARY", "")
        monkeypatch.setattr("tools.browser_launcher.sys.platform", "linux")

        # Import the candidate list to know what paths to mock
        from tools.browser_launcher import _CHROME_CANDIDATES_LINUX

        original_candidates = list(_CHROME_CANDIDATES_LINUX)

        # We need to mock Path(path_str).exists() — the module uses
        # `from pathlib import Path`, so we patch the Path class in that module.
        class FakePath:
            def __init__(self, path_str):
                self._path = path_str

            def exists(self):
                if self._path == original_candidates[0]:
                    return False  # first candidate doesn't exist
                elif self._path == original_candidates[1]:
                    return True  # second candidate exists
                return False

        with patch("tools.browser_launcher.Path", FakePath):
            launcher = ChromeLauncher()
            result = launcher._get_chrome_path()
        assert result == original_candidates[1]
