"""Browser detection and Chrome DevTools Protocol (CDP) helpers.

Locates default Chrome/Brave user-data directories and executables per OS,
and provides thin CDP client helpers to enumerate tabs, find the active
page, and wait for a debugging endpoint to come online. All OS-specific
branching is centralised here.
"""

import time
import json
import urllib.request
from pathlib import Path
from utils.platform_utils import get_platform


# ── Profile path detection ────────────────────────────────────────────────────

def chrome_profile_path() -> Path | None:
    """Return the default Chrome user-data directory for this OS.

    Returns:
        The first existing Chrome (or Chromium, on Linux) user-data path, or
        None if none of the known locations exist.
    """
    home = Path.home()
    candidates = {
        "windows": [
            home / "AppData" / "Local" / "Google" / "Chrome" / "User Data",
        ],
        "linux": [
            home / ".config" / "google-chrome",
            home / ".config" / "chromium",
        ],
    }
    for path in candidates.get(get_platform(), []):
        if path.exists():
            return path
    return None


def brave_profile_path() -> Path | None:
    """Return the default Brave user-data directory for this OS.

    Returns:
        The first existing Brave user-data path, or None if none of the
        known locations exist.
    """
    home = Path.home()
    candidates = {
        "windows": [
            home / "AppData" / "Local" / "BraveSoftware" / "Brave-Browser" / "User Data",
        ],
        "linux": [
            home / ".config" / "BraveSoftware" / "Brave-Browser",
        ],
    }
    for path in candidates.get(get_platform(), []):
        if path.exists():
            return path
    return None


def browser_executable(browser: str) -> str | None:
    """Return the executable name for the requested browser on this OS.

    Args:
        browser: Browser identifier, ``"chrome"`` or ``"brave"``
            (case-insensitive).

    Returns:
        The platform-specific executable name (e.g. ``"chrome.exe"`` or
        ``"google-chrome"``), or None if the browser is unknown.
    """
    exes = {
        "windows": {
            "chrome": "chrome.exe",
            "brave": "brave.exe",
        },
        "linux": {
            "chrome": "google-chrome",
            "brave": "brave-browser",
        },
    }
    return exes.get(get_platform(), {}).get(browser.lower())


# ── CDP helpers ───────────────────────────────────────────────────────────────

def cdp_get_tabs(port: int = 9222) -> list[dict]:
    """Fetch the list of open targets from a CDP-enabled browser.

    Args:
        port: The remote-debugging port to query. Defaults to 9222.

    Returns:
        A list of target dicts as returned by the CDP ``/json`` endpoint, or
        an empty list if the endpoint is unreachable or errors.
    """
    try:
        url = f"http://localhost:{port}/json"
        with urllib.request.urlopen(url, timeout=3) as resp:
            return json.loads(resp.read())
    except Exception:
        return []


def cdp_active_tab(port: int = 9222) -> dict | None:
    """Return the first page-type tab from CDP.

    Args:
        port: The remote-debugging port to query. Defaults to 9222.

    Returns:
        The first target whose ``type`` is ``"page"``, or None if there are
        no page tabs.
    """
    for tab in cdp_get_tabs(port):
        if tab.get("type") == "page":
            return tab
    return None


def wait_for_cdp(port: int = 9222, timeout: float = 10.0) -> bool:
    """Block until the CDP endpoint responds or the timeout elapses.

    Polls :func:`cdp_get_tabs` every 0.5 seconds until it returns a
    non-empty result or the deadline passes.

    Args:
        port: The remote-debugging port to poll. Defaults to 9222.
        timeout: Maximum seconds to wait. Defaults to 10.0.

    Returns:
        True if CDP became available within the timeout, False otherwise.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cdp_get_tabs(port):
            return True
        time.sleep(0.5)
    return False
