"""
tools/browser.py

BrowserTool — gives the LLM real, visible control over Chrome.

Operations (passed as `action` argument):
  open_url    — navigate to a URL
  search      — open default search engine and type a query
  click       — click by visible text label or x,y coordinates
  scroll      — scroll the page up or down
  read_page   — return the page's visible text to the LLM
  close       — close the browser

All input actions (typing, clicking) use PyAutoGUI so they are
visible on screen. Page reading uses CDP for speed and accuracy.
"""

import json
import time
import urllib.parse

import pyautogui

import config
from tools.base import BaseTool, ToolResult
from tools.browser_launcher import ChromeLauncher
from core.logger import logger


# Search URL templates — suffix with urllib.parse.quote_plus(query)
_SEARCH_ENGINES = {
    "google": "https://www.google.com/search?q=",
    "bing": "https://www.bing.com/search?q=",
    "duckduckgo": "https://duckduckgo.com/?q=",
}
_DEFAULT_SEARCH_ENGINE = "google"


class BrowserTool(BaseTool):
    """Tool that gives the LLM visible control over a real Chrome browser.

    Routes a single ``action`` argument to handlers for navigating, web
    searching, clicking, scrolling, reading page text, and closing the
    browser. Input actions (typing, clicking, scrolling) use PyAutoGUI so
    they are visible on screen, while page reading goes through the Chrome
    DevTools Protocol for speed and accuracy. Chrome is launched or attached
    lazily on first use via ``ChromeLauncher``.

    Attributes:
        _launcher: ``ChromeLauncher`` owning the browser process and CDP
            client.
        _loaded: Whether Chrome has been launched/attached this session.
    """

    def __init__(self):
        """Initialize the tool without launching the browser.

        Creates the ``ChromeLauncher`` and marks the browser as not yet
        loaded; the actual launch/attach is deferred until first use.
        """
        self._launcher = ChromeLauncher()
        self._loaded = False

    @property
    def name(self) -> str:
        """Return the tool's unique identifier."""
        return "browser"

    @property
    def description(self) -> str:
        """Return the LLM-facing description of this tool."""
        return (
            "Controls the user's real Chrome browser. "
            "Use this tool to open websites, search the web, click links, "
            "scroll pages, and read the text content of any web page. "
            "All actions are visible on screen. "
            "The browser uses the user's real profile, so they are already "
            "logged in to all their accounts. "
            "Use action='open_url' to navigate to a URL. "
            "Use action='search' to search the web for a query. "
            "Use action='click' to click a link or button by its text label. "
            "Use action='scroll' to scroll the page up or down. "
            "Use action='read_page' to get the visible text of the current page. "
            "Use action='close' to close the browser."
        )

    @property
    def parameters(self) -> dict:
        """Return the JSON Schema for this tool's arguments."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["open_url", "search", "click", "scroll",
                             "read_page", "close"],
                    "description": "The browser action to perform.",
                },
                "url": {
                    "type": "string",
                    "description": "URL to navigate to. Required for action='open_url'.",
                },
                "query": {
                    "type": "string",
                    "description": "Search query. Required for action='search'.",
                },
                "target": {
                    "type": "string",
                    "description": (
                        "For action='click': visible text of the link or button to click. "
                        "For action='scroll': direction — 'up' or 'down'."
                    ),
                },
                "amount": {
                    "type": "integer",
                    "description": "For action='scroll': number of scroll clicks (default 3).",
                },
            },
            "required": ["action"],
        }

    def execute(self, **kwargs) -> ToolResult:
        """Route a browser action to its handler.

        Ensures Chrome is running, then dispatches on the ``action`` argument.
        All exceptions are caught and returned as failed results rather than
        raised.

        Args:
            **kwargs: Expects ``action`` plus action-specific arguments such
                as ``url``, ``query``, ``target``, and ``amount``.

        Returns:
            The ``ToolResult`` from the selected handler, or an error result
            for an unknown action or any raised exception.
        """
        action = kwargs.get("action", "")
        try:
            self._ensure_loaded()
            if action == "open_url":
                return self._open_url(kwargs.get("url", ""))
            elif action == "search":
                return self._search(kwargs.get("query", ""))
            elif action == "click":
                return self._click(kwargs.get("target", ""))
            elif action == "scroll":
                return self._scroll(
                    kwargs.get("target", "down"),
                    kwargs.get("amount", 3),
                )
            elif action == "read_page":
                return self._read_page()
            elif action == "close":
                return self._close()
            else:
                return ToolResult(success=False, error=f"Unknown action: {action!r}")
        except Exception as e:
            logger.error(f"BrowserTool error ({action}): {e}")
            return ToolResult(success=False, error=str(e))

    def _ensure_loaded(self) -> None:
        """Launch or attach to Chrome on first use.

        No-op after the first successful call. Delegates to
        ``ChromeLauncher.ensure_running`` and records that the browser is
        loaded.

        Side Effects:
            May launch a Chrome process and open a CDP connection.
        """
        if not self._loaded:
            self._launcher.ensure_running()
            self._loaded = True

    def _open_url(self, url: str) -> ToolResult:
        """Navigate to a URL by typing it into the address bar.

        Normalises the URL (prepending ``https://`` when no scheme is
        present), focuses the address bar with Ctrl+L, selects existing
        text, types the URL visibly, presses Enter, and waits briefly for the
        page to begin loading.

        Args:
            url: The destination URL or bare host.

        Returns:
            A ``ToolResult`` confirming navigation, or an error if no URL was
            provided.

        Side Effects:
            Drives the keyboard via PyAutoGUI to control the visible browser.
        """
        if not url:
            return ToolResult(success=False, error="No URL provided for open_url action")

        # Normalise URL — prepend https:// if no scheme present
        if "://" not in url:
            url = "https://" + url

        # Focus the address bar
        pyautogui.hotkey("ctrl", "l")
        time.sleep(0.3)

        # Select all existing text in the address bar
        pyautogui.hotkey("ctrl", "a")

        # Type the URL visibly
        pyautogui.typewrite(url, interval=config.BROWSER_TYPE_DELAY)

        # Press Enter to navigate
        pyautogui.press("enter")

        # Wait for page to start loading
        time.sleep(1.5)

        return ToolResult(success=True, output=f"Navigated to {url}")

    def _search(self, query: str) -> ToolResult:
        """Search the web for a query using the default search engine.

        Builds the full search URL from the configured default engine and
        navigates to it via ``_open_url``.

        Args:
            query: The search terms.

        Returns:
            A ``ToolResult`` from the navigation, or an error if the query is
            empty.
        """
        if not query:
            return ToolResult(success=False, error="No query provided for search action")

        search_url = _SEARCH_ENGINES[_DEFAULT_SEARCH_ENGINE] + urllib.parse.quote_plus(query)
        return self._open_url(search_url)

    def _click(self, target: str) -> ToolResult:
        """Click an element whose visible text matches ``target``.

        Runs JavaScript via CDP to locate the first clickable element whose
        inner text contains ``target`` (case-insensitive), reads its bounding
        rect, scales the centre by ``config.SCREEN_SCALE_FACTOR``, then issues
        a visible PyAutoGUI click there.

        Args:
            target: The visible text of the link or button to click.

        Returns:
            A ``ToolResult`` confirming the click, or an error if ``target``
            is empty or no matching element was found.

        Side Effects:
            Evaluates JavaScript in the page and moves/clicks the mouse.
        """
        if not target:
            return ToolResult(success=False, error="No target provided for click action")

        target_lower = target.lower().replace('"', '\\"')

        # JavaScript to find the first matching clickable element
        js = (
            f'const target = "{target_lower}";'
            'const elements = document.querySelectorAll("a, button, input, [role=\'button\'], [onclick]");'
            'for (const el of elements) {'
            '    if (el.innerText && el.innerText.trim().toLowerCase().includes(target)) {'
            '        const rect = el.getBoundingClientRect();'
            '        return JSON.stringify({x: rect.x, y: rect.y, width: rect.width, height: rect.height});'
            '    }'
            '}'
            'return null;'
        )

        # Wrap in an IIFE so 'return' is valid
        js_wrapped = f"(function() {{ {js} }})()"

        result = self._launcher.cdp.send(
            "Runtime.evaluate",
            {"expression": js_wrapped, "returnByValue": True}
        )

        value = result.get("result", {}).get("value")

        if value is None:
            return ToolResult(success=False, error=f"No element found with text: {target}")

        # Parse the bounding rect
        rect = json.loads(value)
        cx = (rect["x"] + rect["width"] / 2) * config.SCREEN_SCALE_FACTOR
        cy = (rect["y"] + rect["height"] / 2) * config.SCREEN_SCALE_FACTOR

        # Click at the computed centre
        pyautogui.click(int(cx), int(cy))
        time.sleep(0.5)

        return ToolResult(success=True, output=f"Clicked element: {target}")

    def _scroll(self, direction: str, amount: int) -> ToolResult:
        """Scroll the page up or down.

        Moves the mouse to the screen centre, then scrolls by ``amount``
        clicks in the given direction (any direction other than ``"up"`` is
        treated as down).

        Args:
            direction: ``"up"`` to scroll up, otherwise scrolls down.
            amount: Number of scroll clicks.

        Returns:
            A ``ToolResult`` describing the scroll.

        Side Effects:
            Moves the mouse and scrolls the active window via PyAutoGUI.
        """
        width, height = pyautogui.size()
        pyautogui.moveTo(width // 2, height // 2)

        if direction == "up":
            pyautogui.scroll(amount)
        else:
            pyautogui.scroll(-amount)

        return ToolResult(success=True, output=f"Scrolled {direction} by {amount} clicks")

    def _read_page(self) -> ToolResult:
        """Return the visible text of the current page via CDP.

        Truncates output to 8000 characters, appending a marker when the page
        has more content.

        Returns:
            A ``ToolResult`` whose ``output`` is the (possibly truncated) page
            text.
        """
        text = self._launcher.cdp.get_page_text()

        if len(text) > 8000:
            text = text[:8000] + "\n\n[truncated — page has more content]"

        return ToolResult(success=True, output=text)

    def _close(self) -> ToolResult:
        """Close the browser and reset loaded state.

        Returns:
            A ``ToolResult`` confirming the browser was closed.

        Side Effects:
            Terminates or detaches the browser via ``ChromeLauncher.close``.
        """
        self._launcher.close()
        self._loaded = False
        return ToolResult(success=True, output="Browser closed")
