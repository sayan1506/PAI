"""
tools/browser.py

Browser control tool for PAI.
Launches Chrome/Brave with the real user profile so existing sessions
(YouTube, Gmail, etc.) are already logged in.

Strategy:
  - subprocess + --remote-debugging-port  → launch with real profile
  - PyAutoGUI                             → type text, focus window
  - CDP (urllib, no Playwright)           → read page URL and title
"""

from __future__ import annotations
import time
import subprocess
import psutil
import pyautogui
import config
from tools.base import BaseTool, ToolResult
from utils.platform_utils import find_app_path, get_platform
from utils.browser_utils import (
    chrome_profile_path,
    brave_profile_path,
    browser_executable,
    wait_for_cdp,
    cdp_active_tab,
)
from core.logger import logger

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


class BrowserTool(BaseTool):
    """Control a real browser session."""

    @property
    def name(self) -> str:
        return "browser"

    @property
    def description(self) -> str:
        return (
            "Control a web browser. Can open URLs, search the web, type in the "
            "address bar, scroll, and read the current page title and URL. "
            "The browser opens with the user's real profile so they are already "
            "logged in to their accounts."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["open_url", "search", "search_youtube", "type_text", "scroll", "read_page", "get_url"],
                    "description": (
                        "open_url: navigate to a URL. "
                        "search: search Google for a query. "
                        "search_youtube: open YouTube and search for a query in YouTube's search bar. "
                        "type_text: type text into the currently focused input field and press Enter. "
                        "scroll: scroll the page up or down. "
                        "read_page: return the current page title and URL. "
                        "get_url: return just the current URL."
                    ),
                },
                "url": {
                    "type": "string",
                    "description": "Full URL to open (required for open_url).",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (required for search and search_youtube).",
                },
                "text": {
                    "type": "string",
                    "description": "Text to type into the current input field (required for type_text).",
                },
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Scroll direction (required for scroll).",
                },
                "amount": {
                    "type": "integer",
                    "description": "Number of scroll clicks (default 3).",
                },
            },
            "required": ["operation"],
        }

    def execute(self, **kwargs) -> ToolResult:
        if not config.BROWSER_ENABLED:
            return ToolResult(
                success=False,
                error="Browser tool is disabled. Set BROWSER_ENABLED=true in config.",
            )

        operation = kwargs.get("operation")
        handler = getattr(self, f"_op_{operation}", None)
        if not handler:
            return ToolResult(success=False, error=f"Unknown browser operation: {operation}")

        # Ensure browser is running before any operation
        launch_result = self._ensure_browser_running()
        if not launch_result.success:
            return launch_result

        return handler(**kwargs)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_profile_path(self) -> str | None:
        browser = config.BROWSER_APP.lower()
        path = chrome_profile_path() if browser == "chrome" else brave_profile_path()
        return str(path) if path else None

    def _ensure_browser_running(self) -> ToolResult:
        """Launch the browser if CDP is not already available."""
        if cdp_active_tab(config.BROWSER_REMOTE_PORT):
            self._cdp_available = True
            return ToolResult(success=True, output="Browser already running.")

        browser = config.BROWSER_APP.lower()
        exe = browser_executable(browser)
        if not exe:
            return ToolResult(success=False, error=f"Unknown browser: {browser}")

        full_path = find_app_path(exe)
        if not full_path:
            return ToolResult(
                success=False,
                error=f"{browser.capitalize()} not found. Is it installed?",
            )

        profile = self._get_profile_path()

        # Check if ANY browser is already running — use OS URL opening
        browser_names = ["chrome.exe", "brave.exe", "firefox.exe", "msedge.exe",
                         "google-chrome", "brave-browser"]
        browser_running = any(
            (p.info.get("name") or "").lower() in browser_names
            for p in psutil.process_iter(["name"])
        )

        if browser_running:
            logger.info("Browser already running — using OS URL opening mode")
            self._cdp_available = False
            return ToolResult(success=True, output="Browser already running.")

        cmd = [
            full_path,
            f"--remote-debugging-port={config.BROWSER_REMOTE_PORT}",
            "--no-first-run",
            "--no-default-browser-check",
        ]
        if profile:
            cmd.append(f"--user-data-dir={profile}")

        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info(f"Browser launched: {browser} (profile={profile})")

            if wait_for_cdp(config.BROWSER_REMOTE_PORT, timeout=15.0):
                self._cdp_available = True
            else:
                self._cdp_available = False
                logger.warning("CDP not available — using OS URL opening mode")

            return ToolResult(success=True, output="Browser launched.")

        except Exception as e:
            return ToolResult(success=False, error=f"Failed to launch browser: {e}")

    def _focus_browser(self):
        """Bring the browser window to the foreground."""
        if get_platform() == "windows":
            try:
                import pygetwindow as gw
                # Search for browser windows by common title patterns
                browser_keywords = ["brave", "chrome", "firefox", "edge", "mozilla"]
                all_wins = gw.getAllWindows()
                for keyword in browser_keywords:
                    wins = [w for w in all_wins if keyword in w.title.lower() and w.title.strip()]
                    if wins:
                        try:
                            wins[0].activate()
                        except Exception:
                            # If activate fails, try minimize then restore
                            wins[0].minimize()
                            time.sleep(0.1)
                            wins[0].restore()
                        time.sleep(0.5)
                        return
                # Fallback: Alt+Tab to switch to last window
                pyautogui.hotkey("alt", "tab")
                time.sleep(0.5)
            except Exception:
                pyautogui.hotkey("alt", "tab")
                time.sleep(0.5)
        else:
            subprocess.run(
                ["xdotool", "search", "--name", config.BROWSER_APP,
                 "windowactivate", "--sync"],
                capture_output=True,
            )

    # ── Operations ─────────────────────────────────────────────────────────────

    def _op_open_url(self, **kwargs) -> ToolResult:
        url = kwargs.get("url", "").strip()
        if not url:
            return ToolResult(success=False, error="No URL provided.")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            # If we have CDP, use PyAutoGUI for precise control
            if getattr(self, "_cdp_available", False):
                self._focus_browser()
                time.sleep(0.4)
                pyautogui.hotkey("ctrl", "l")
                time.sleep(0.3)
                pyautogui.hotkey("ctrl", "a")
                pyautogui.typewrite(url, interval=config.TYPING_SPEED)
                pyautogui.press("enter")
                time.sleep(1.5)
            else:
                # No CDP — use OS to open URL in default/configured browser
                if get_platform() == "windows":
                    import os
                    os.startfile(url)
                else:
                    subprocess.Popen(
                        ["xdg-open", url],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                time.sleep(2.0)

            return ToolResult(success=True, output=f"Navigated to: {url}")
        except Exception as e:
            return ToolResult(success=False, error=f"open_url failed: {e}")

    def _op_search(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "").strip()
        if not query:
            return ToolResult(success=False, error="No search query provided.")

        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        return self._op_open_url(url=search_url)

    def _op_search_youtube(self, **kwargs) -> ToolResult:
        """Open YouTube and search using YouTube's search bar."""
        query = kwargs.get("query", "").strip()
        if not query:
            return ToolResult(success=False, error="No search query provided.")

        # Use YouTube's search URL directly — this searches within YouTube
        yt_search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        return self._op_open_url(url=yt_search_url)

    def _op_type_text(self, **kwargs) -> ToolResult:
        """Type text into the currently focused input field and press Enter."""
        text = kwargs.get("text", "").strip()
        if not text:
            return ToolResult(success=False, error="No text provided.")

        try:
            self._focus_browser()
            time.sleep(0.3)
            pyautogui.typewrite(text, interval=config.TYPING_SPEED)
            pyautogui.press("enter")
            time.sleep(1.0)
            return ToolResult(success=True, output=f"Typed and submitted: {text}")
        except Exception as e:
            return ToolResult(success=False, error=f"type_text failed: {e}")

    def _op_scroll(self, **kwargs) -> ToolResult:
        direction = kwargs.get("direction", "down")
        amount = int(kwargs.get("amount", 3))
        clicks = -amount if direction == "down" else amount

        try:
            self._focus_browser()
            time.sleep(0.3)
            pyautogui.scroll(clicks)
            return ToolResult(success=True, output=f"Scrolled {direction} {amount} clicks.")
        except Exception as e:
            return ToolResult(success=False, error=f"scroll failed: {e}")

    def _op_read_page(self, **kwargs) -> ToolResult:
        tab = cdp_active_tab(config.BROWSER_REMOTE_PORT)
        if not tab:
            return ToolResult(
                success=True,
                output="Cannot read page info — browser is running without CDP. Use screen_reader to see what's on screen.",
            )
        return ToolResult(
            success=True,
            output=f"Title: {tab.get('title', 'N/A')}\nURL: {tab.get('url', 'N/A')}",
        )

    def _op_get_url(self, **kwargs) -> ToolResult:
        tab = cdp_active_tab(config.BROWSER_REMOTE_PORT)
        if not tab:
            return ToolResult(
                success=True,
                output="Cannot read URL — browser is running without CDP.",
            )
        return ToolResult(success=True, output=tab.get("url", ""))
