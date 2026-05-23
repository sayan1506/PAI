"""
tools/browser_cdp.py

Low-level Chrome DevTools Protocol client.
Connects to a running Chrome instance via the HTTP JSON API and
the WebSocket target. Used by BrowserTool and ChromeLauncher.
"""

import json
import re
import time

import requests
import websocket  # websocket-client

import config
from core.logger import logger


class CDPClient:
    """
    Connects to Chrome's remote debugging port and sends CDP commands.

    Usage:
        client = CDPClient()
        client.connect()             # attach to the first available tab
        text = client.get_page_text()
        client.navigate("https://example.com")
        client.disconnect()
    """

    def __init__(self, port: int = None):
        self._port = port or config.BROWSER_REMOTE_PORT
        self._ws = None
        self._msg_id = 0

    # ── Connection ────────────────────────────────────────────────────────

    def is_reachable(self) -> bool:
        """Return True if Chrome's debugging port is accepting connections."""
        try:
            resp = requests.get(
                f"http://localhost:{self._port}/json", timeout=2
            )
            return resp.status_code == 200
        except Exception:
            return False

    def connect(self) -> None:
        """
        Attach to the first available (non-devtools) Chrome tab via WebSocket.

        Steps:
          1. GET http://localhost:{port}/json → list of tab descriptors
          2. Pick the first tab whose type == "page"
          3. Open a websocket to tab["webSocketDebuggerUrl"]

        Raises:
            ConnectionError if no tabs are found or websocket fails.
        """
        try:
            resp = requests.get(
                f"http://localhost:{self._port}/json", timeout=5
            )
            tabs = resp.json()
        except Exception as e:
            raise ConnectionError(
                f"Failed to query Chrome tabs on port {self._port}: {e}"
            )

        # Find the first page-type tab
        page_tab = None
        for tab in tabs:
            if tab.get("type") == "page":
                page_tab = tab
                break

        if page_tab is None:
            raise ConnectionError(
                f"No page tabs found on Chrome port {self._port}"
            )

        ws_url = page_tab.get("webSocketDebuggerUrl")
        if not ws_url:
            raise ConnectionError(
                "Tab does not expose a webSocketDebuggerUrl"
            )

        try:
            self._ws = websocket.create_connection(
                ws_url, timeout=config.BROWSER_CDP_TIMEOUT
            )
            logger.debug(f"CDP connected to {ws_url}")
        except Exception as e:
            raise ConnectionError(
                f"Failed to open WebSocket to Chrome tab: {e}"
            )

    def disconnect(self) -> None:
        """Close the WebSocket connection if open. Set self._ws = None."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
            logger.debug("CDP disconnected")

    # ── Commands ──────────────────────────────────────────────────────────

    def send(self, method: str, params: dict = None) -> dict:
        """
        Send a CDP command and wait for its response.

        Args:
            method: CDP method string e.g. "Runtime.evaluate"
            params: CDP params dict

        Returns:
            The "result" field from the CDP response dict.

        Raises:
            RuntimeError if CDP returns an error field.
            RuntimeError if WebSocket is not connected.
        """
        if self._ws is None:
            raise RuntimeError("WebSocket is not connected. Call connect() first.")

        msg_id = self._next_id()
        payload = {"id": msg_id, "method": method}
        if params:
            payload["params"] = params

        self._ws.send(json.dumps(payload))
        logger.debug(f"CDP send: {method} (id={msg_id})")

        response = self._recv_until(msg_id, timeout=config.BROWSER_CDP_TIMEOUT)

        if "error" in response:
            error_info = response["error"]
            raise RuntimeError(
                f"CDP error on {method}: {error_info.get('message', error_info)}"
            )

        return response.get("result", {})

    def navigate(self, url: str) -> None:
        """Navigate the current tab to url using Page.navigate."""
        self.send("Page.navigate", {"url": url})

    def get_page_text(self) -> str:
        """
        Return the visible text content of the current page.

        Uses Runtime.evaluate to call document.body.innerText.
        Strips leading/trailing whitespace and collapses runs of
        blank lines to at most two newlines.
        Returns empty string if the page has no body or result is None.
        """
        result = self.send(
            "Runtime.evaluate",
            {"expression": "document.body ? document.body.innerText : ''"}
        )

        value = result.get("result", {}).get("value")
        if value is None:
            return ""

        text = str(value).strip()
        # Collapse runs of blank lines to at most two newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def get_page_url(self) -> str:
        """Return the current tab's URL via Runtime.evaluate on window.location.href."""
        result = self.send(
            "Runtime.evaluate",
            {"expression": "window.location.href"}
        )
        return result.get("result", {}).get("value", "")

    def get_page_title(self) -> str:
        """Return document.title of the current tab via Runtime.evaluate."""
        result = self.send(
            "Runtime.evaluate",
            {"expression": "document.title"}
        )
        return result.get("result", {}).get("value", "")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _next_id(self) -> int:
        """Increment and return the next message ID."""
        self._msg_id += 1
        return self._msg_id

    def _recv_until(self, target_id: int, timeout: float = 10.0) -> dict:
        """
        Read WebSocket messages until we receive one whose "id" matches
        target_id. Discard unmatched events/messages. Raise TimeoutError if
        timeout is exceeded.
        """
        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed >= timeout:
                raise TimeoutError(
                    f"CDP response for id={target_id} not received "
                    f"within {timeout}s"
                )

            remaining = timeout - elapsed
            self._ws.settimeout(remaining)

            try:
                raw = self._ws.recv()
            except websocket.WebSocketTimeoutException:
                raise TimeoutError(
                    f"CDP response for id={target_id} not received "
                    f"within {timeout}s"
                )

            if not raw:
                continue

            msg = json.loads(raw)

            # Check if this message matches our target id
            if msg.get("id") == target_id:
                return msg
            # Otherwise discard (it's an event or response to another command)
