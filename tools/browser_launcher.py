"""
tools/browser_launcher.py

Launches a Chromium-based browser (Chrome, Brave, Edge) with remote
debugging enabled, or attaches to an already-running instance.
Owns the subprocess handle so the tool can cleanly close the browser.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import config
from core.logger import logger
from tools.browser_cdp import CDPClient


# Ordered candidate binary paths per platform and browser
_CANDIDATES = {
    "chrome": {
        "win32": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        ],
        "linux": [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium",
            "/snap/bin/chromium",
        ],
    },
    "brave": {
        "win32": [
            r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
            r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"),
        ],
        "linux": [
            "/usr/bin/brave-browser",
            "/usr/bin/brave",
            "/snap/bin/brave",
        ],
    },
    "edge": {
        "win32": [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ],
        "linux": [
            "/usr/bin/microsoft-edge",
            "/usr/bin/microsoft-edge-stable",
        ],
    },
}

# Keep backward compat for tests
_CHROME_CANDIDATES_WINDOWS = _CANDIDATES["chrome"]["win32"]
_CHROME_CANDIDATES_LINUX = _CANDIDATES["chrome"]["linux"]


class ChromeLauncher:
    """Manages a Chromium-based browser process with remote debugging.

    Supports Chrome, Brave, and Edge based on ``config.BROWSER_APP``. If a
    browser is already listening on the configured remote debugging port, it
    attaches without launching a new process. Otherwise it launches the
    browser with a dedicated debug profile so it does not conflict with the
    user's existing windows. Owns both the subprocess handle and the
    ``CDPClient`` used to talk to the browser.

    Attributes:
        _proc: Handle to the launched browser process, or None when attached
            to a pre-existing instance.
        cdp: The ``CDPClient`` used to send DevTools commands.
    """

    def __init__(self):
        """Initialize the launcher without starting a browser.

        Creates an unconnected ``CDPClient`` and leaves the process handle
        empty until ``ensure_running`` is called.
        """
        self._proc = None
        self.cdp = CDPClient()

    def ensure_running(self) -> None:
        """Ensure the browser is running and the CDP client is connected.

        If the debugging port is already reachable, simply connects the CDP
        client. Otherwise locates the browser binary, launches it with remote
        debugging enabled, waits for the port to open, and then connects.

        Raises:
            RuntimeError: If the browser binary cannot be found or the
                debugging port never opens within the timeout.

        Side Effects:
            May spawn a browser subprocess and open a CDP connection.
        """
        if self.cdp.is_reachable():
            logger.info(
                f"Browser already listening on port {config.BROWSER_REMOTE_PORT}, attaching"
            )
            self.cdp.connect()
            return

        # Need to launch browser ourselves
        try:
            binary = self._get_chrome_path()
        except FileNotFoundError as e:
            raise RuntimeError(str(e))

        args = self._build_launch_args(binary)
        logger.info(f"Launching browser: {' '.join(args)}")

        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            self._wait_for_port(config.BROWSER_CDP_TIMEOUT)
        except TimeoutError as e:
            # Clean up the process we just started
            self._proc.terminate()
            self._proc = None
            raise RuntimeError(str(e))

        self.cdp.connect()
        logger.info("Browser launched and CDP connected")

    def close(self) -> None:
        """Close the browser and disconnect the CDP client.

        Always disconnects the CDP WebSocket. If PAI launched the browser
        (``_proc`` is set), the process is terminated (and killed if it does
        not exit within 5 seconds). If PAI merely attached to a pre-existing
        browser, the user's process is left running.

        Side Effects:
            May terminate or kill the launched browser process.
        """
        self.cdp.disconnect()

        if self._proc is not None:
            logger.info("Terminating browser process launched by PAI")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def _get_chrome_path(self) -> str:
        """Return the path to the browser binary to launch.

        Prefers ``config.BROWSER_BINARY`` when set; otherwise checks the
        candidate paths for ``config.BROWSER_APP`` on the current platform,
        falling back to Chrome candidates for unknown browser names.

        Returns:
            The first existing candidate binary path.

        Raises:
            FileNotFoundError: If no candidate binary exists on disk.
        """
        if config.BROWSER_BINARY:
            return config.BROWSER_BINARY

        browser = config.BROWSER_APP.lower()
        platform = "win32" if sys.platform == "win32" else "linux"

        candidates = _CANDIDATES.get(browser, {}).get(platform, [])
        if not candidates:
            # Fallback to chrome if unknown browser name
            candidates = _CANDIDATES["chrome"].get(platform, [])

        for path_str in candidates:
            if Path(path_str).exists():
                return path_str

        raise FileNotFoundError(
            f"{browser.title()} binary not found. Tried: {candidates}. "
            f"Set BROWSER_BINARY in your .env to specify the path manually."
        )

    def _build_launch_args(self, binary: str) -> list[str]:
        """Build the subprocess argument list for launching the browser.

        Uses a dedicated ``user-data-dir`` under ``~/.pai`` so that, if the
        browser is already running without debugging, this launch creates a
        separate process that binds to the remote debugging port. Does not
        enable headless or incognito mode.

        Args:
            binary: Path to the browser executable.

        Returns:
            The full argument list (binary plus flags) for ``subprocess.Popen``.
        """
        data_dir = os.path.expanduser("~/.pai/browser-debug-profile")
        return [
            binary,
            f"--remote-debugging-port={config.BROWSER_REMOTE_PORT}",
            f"--user-data-dir={data_dir}",
            "--profile-directory=Default",
            "--no-first-run",
            "--no-default-browser-check",
        ]

    def _wait_for_port(self, timeout: int) -> None:
        """Poll until the remote debugging port becomes reachable.

        Checks ``cdp.is_reachable()`` every 0.5 seconds until it succeeds or
        the timeout elapses.

        Args:
            timeout: Maximum seconds to wait for the port to open.

        Raises:
            TimeoutError: If the port does not open within ``timeout`` seconds.
        """
        start = time.time()
        while time.time() - start < timeout:
            if self.cdp.is_reachable():
                return
            time.sleep(0.5)

        raise TimeoutError(
            f"Browser debugging port {config.BROWSER_REMOTE_PORT} did not open "
            f"within {timeout} seconds"
        )
