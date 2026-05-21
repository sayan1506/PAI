"""Tests for BrowserTool."""
from unittest.mock import patch, MagicMock
import pytest
import config
from tools.browser import BrowserTool


@pytest.fixture
def tool():
    return BrowserTool()


@pytest.fixture(autouse=True)
def enable_browser(monkeypatch):
    monkeypatch.setattr(config, "BROWSER_ENABLED", True)
    monkeypatch.setattr(config, "BROWSER_APP", "chrome")
    monkeypatch.setattr(config, "BROWSER_REMOTE_PORT", 9222)
    monkeypatch.setattr(config, "TYPING_SPEED", 0.01)


class TestBrowserDisabled:
    def test_disabled_by_default(self, tool, monkeypatch):
        monkeypatch.setattr(config, "BROWSER_ENABLED", False)
        result = tool.execute(operation="open_url", url="https://google.com")
        assert result.success is False
        assert "disabled" in result.error.lower()


class TestSearch:
    @patch.object(BrowserTool, "_op_open_url")
    @patch.object(BrowserTool, "_ensure_browser_running")
    def test_search_builds_correct_url(self, mock_ensure, mock_open_url, tool):
        mock_ensure.return_value = MagicMock(success=True)
        mock_open_url.return_value = MagicMock(success=True, output="Navigated")

        tool.execute(operation="search", query="lo-fi music")

        call_kwargs = mock_open_url.call_args[1]
        assert "lo-fi+music" in call_kwargs.get("url", "")


class TestOpenUrl:
    @patch("tools.browser.pyautogui")
    @patch.object(BrowserTool, "_focus_browser")
    @patch.object(BrowserTool, "_ensure_browser_running")
    def test_open_url_adds_https_prefix(self, mock_ensure, mock_focus, mock_pyautogui, tool):
        mock_ensure.return_value = MagicMock(success=True)

        result = tool.execute(operation="open_url", url="youtube.com")

        assert result.success is True
        assert "https://youtube.com" in result.output


class TestReadPage:
    @patch("tools.browser.cdp_active_tab")
    @patch.object(BrowserTool, "_ensure_browser_running")
    def test_read_page_returns_title_and_url(self, mock_ensure, mock_cdp, tool):
        mock_ensure.return_value = MagicMock(success=True)
        mock_cdp.return_value = {"title": "YouTube", "url": "https://youtube.com", "type": "page"}

        result = tool.execute(operation="read_page")

        assert result.success is True
        assert "YouTube" in result.output
        assert "https://youtube.com" in result.output


class TestBrowserLaunch:
    @patch("tools.browser.psutil.process_iter", return_value=[])
    @patch("tools.browser.wait_for_cdp", return_value=True)
    @patch("tools.browser.subprocess.Popen")
    @patch("tools.browser.find_app_path", return_value="C:\\chrome.exe")
    @patch("tools.browser.chrome_profile_path")
    @patch("tools.browser.cdp_active_tab", return_value=None)
    def test_launches_with_real_profile_path(self, mock_cdp, mock_profile, mock_find, mock_popen, mock_wait, mock_psutil, tool, tmp_path):
        mock_profile.return_value = tmp_path
        mock_popen.return_value = MagicMock()

        result = tool._ensure_browser_running()

        assert result.success is True
        cmd = mock_popen.call_args[0][0]
        assert any(str(tmp_path) in arg for arg in cmd)

    @patch("tools.browser.psutil.process_iter", return_value=[])
    @patch("tools.browser.wait_for_cdp", return_value=False)
    @patch("tools.browser.subprocess.Popen")
    @patch("tools.browser.find_app_path", return_value="C:\\chrome.exe")
    @patch("tools.browser.chrome_profile_path", return_value=None)
    @patch("tools.browser.cdp_active_tab", return_value=None)
    def test_cdp_timeout(self, mock_cdp, mock_profile, mock_find, mock_popen, mock_wait, mock_psutil, tool):
        mock_popen.return_value = MagicMock()

        result = tool._ensure_browser_running()

        # CDP not available but browser launched — still success (PyAutoGUI mode)
        assert result.success is True


class TestScroll:
    @patch("tools.browser.pyautogui")
    @patch.object(BrowserTool, "_focus_browser")
    @patch.object(BrowserTool, "_ensure_browser_running")
    def test_scroll_down(self, mock_ensure, mock_focus, mock_pyautogui, tool):
        mock_ensure.return_value = MagicMock(success=True)

        result = tool.execute(operation="scroll", direction="down", amount=3)

        assert result.success is True
        mock_pyautogui.scroll.assert_called_once_with(-3)

    @patch("tools.browser.pyautogui")
    @patch.object(BrowserTool, "_focus_browser")
    @patch.object(BrowserTool, "_ensure_browser_running")
    def test_scroll_up(self, mock_ensure, mock_focus, mock_pyautogui, tool):
        mock_ensure.return_value = MagicMock(success=True)

        result = tool.execute(operation="scroll", direction="up", amount=5)

        assert result.success is True
        mock_pyautogui.scroll.assert_called_once_with(5)
