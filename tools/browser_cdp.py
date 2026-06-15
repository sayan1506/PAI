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
    """Low-level client for Chrome's remote debugging (DevTools) protocol.

    Connects to a running Chrome instance over its HTTP JSON endpoint to
    discover tabs, then opens a WebSocket to a page tab to send CDP commands
    and read results. Used by ``BrowserTool`` and ``ChromeLauncher``.

    Usage:
        client = CDPClient()
        client.connect()             # attach to the first available tab
        text = client.get_page_text()
        client.navigate("https://example.com")
        client.disconnect()

    Attributes:
        _port: Remote debugging port to connect to.
        _ws: Active WebSocket connection, or None when disconnected.
        _msg_id: Monotonically increasing CDP message id counter.
    """

    def __init__(self, port: int = None):
        """Initialize the client without connecting.

        Args:
            port: Remote debugging port. Falls back to
                ``config.BROWSER_REMOTE_PORT`` when not provided.
        """
        self._port = port or config.BROWSER_REMOTE_PORT
        self._ws = None
        self._msg_id = 0

    # ── Connection ────────────────────────────────────────────────────────

    def is_reachable(self) -> bool:
        """Check whether Chrome's debugging port is accepting connections.

        Returns:
            True if a GET to the ``/json`` endpoint returns HTTP 200, False on
            any error or non-200 response.
        """
        try:
            resp = requests.get(
                f"http://localhost:{self._port}/json", timeout=2
            )
            return resp.status_code == 200
        except Exception:
            return False

    def connect(self) -> None:
        """Attach to the first available page tab via WebSocket.

        Queries the ``/json`` endpoint for tab descriptors, picks the first
        tab whose type is ``"page"``, and opens a WebSocket to its
        ``webSocketDebuggerUrl``.

        Raises:
            ConnectionError: If tabs cannot be queried, no page tab exists,
                the tab exposes no WebSocket URL, or the WebSocket fails to
                open.

        Side Effects:
            Opens a WebSocket connection stored on ``self._ws``.
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
        """Close the WebSocket connection if open.

        Safe to call when already disconnected. Always leaves ``self._ws`` set
        to None.

        Side Effects:
            Closes the active WebSocket connection.
        """
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
            logger.debug("CDP disconnected")

    # ── Commands ──────────────────────────────────────────────────────────

    def send(self, method: str, params: dict = None) -> dict:
        """Send a CDP command and wait for its matching response.

        Args:
            method: CDP method string, e.g. ``"Runtime.evaluate"``.
            params: Optional CDP parameters dict.

        Returns:
            The ``result`` field of the CDP response.

        Raises:
            RuntimeError: If the WebSocket is not connected, or if the CDP
                response contains an ``error`` field.
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
        """Navigate the current tab to a URL.

        Args:
            url: Destination URL, sent via the ``Page.navigate`` command.
        """
        self.send("Page.navigate", {"url": url})

    def get_page_text(self) -> str:
        """Return the visible text content of the current page.

        Evaluates ``document.body.innerText`` via ``Runtime.evaluate``, strips
        surrounding whitespace, and collapses runs of blank lines to at most
        two newlines.

        Returns:
            The page's visible text, or an empty string if the page has no
            body or the result is None.
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
        """Return the current tab's URL.

        Returns:
            The value of ``window.location.href``, or an empty string if
            unavailable.
        """
        result = self.send(
            "Runtime.evaluate",
            {"expression": "window.location.href"}
        )
        return result.get("result", {}).get("value", "")

    def get_page_title(self) -> str:
        """Return the current tab's document title.

        Returns:
            The value of ``document.title``, or an empty string if
            unavailable.
        """
        result = self.send(
            "Runtime.evaluate",
            {"expression": "document.title"}
        )
        return result.get("result", {}).get("value", "")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _next_id(self) -> int:
        """Increment and return the next CDP message id.

        Returns:
            The new monotonically increasing message id.
        """
        self._msg_id += 1
        return self._msg_id

    def _recv_until(self, target_id: int, timeout: float = 10.0) -> dict:
        """Read WebSocket messages until the response for ``target_id`` arrives.

        Unmatched messages (asynchronous events or responses to other
        commands) are discarded.

        Args:
            target_id: The CDP message id whose response is awaited.
            timeout: Maximum seconds to wait before giving up.

        Returns:
            The decoded message whose ``id`` matches ``target_id``.

        Raises:
            TimeoutError: If no matching response arrives within ``timeout``.
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
