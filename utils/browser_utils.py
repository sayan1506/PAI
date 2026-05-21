"""
utils/browser_utils.py

Browser detection and Chrome DevTools Protocol (CDP) helpers.
All platform branching lives here.
"""

import time
import json
import urllib.request
from pathlib import Path
from utils.platform_utils import get_platform


# ── Profile path detection ────────────────────────────────────────────────────

def chrome_profile_path() -> Path | None:
    """Return the default Chrome user-data-dir for this OS."""
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
    """Return the default Brave user-data-dir for this OS."""
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
    """Return the executable name for the requested browser."""
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
    """Return the list of open tabs from a CDP-connected browser."""
    try:
        url = f"http://localhost:{port}/json"
        with urllib.request.urlopen(url, timeout=3) as resp:
            return json.loads(resp.read())
    except Exception:
        return []


def cdp_active_tab(port: int = 9222) -> dict | None:
    """Return the first page-type tab from CDP."""
    for tab in cdp_get_tabs(port):
        if tab.get("type") == "page":
            return tab
    return None


def wait_for_cdp(port: int = 9222, timeout: float = 10.0) -> bool:
    """Block until CDP is available or timeout. Returns True on success."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cdp_get_tabs(port):
            return True
        time.sleep(0.5)
    return False
