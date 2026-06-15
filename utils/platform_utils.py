"""Cross-platform helpers for system operations.

All OS-specific branching is centralised here so the rest of the codebase
(tools, agents) can stay platform-agnostic. Covers platform detection,
launching and killing applications, resolving installed apps to executable
paths (Windows registry / Linux ``.desktop`` files), and locating well-known
user directories including OneDrive-backed folders on Windows.
"""

import sys
import os
import shutil
import subprocess
import configparser
from pathlib import Path


def get_platform() -> str:
    """Return the current platform family.

    Returns:
        ``"windows"`` on Win32, otherwise ``"linux"``.
    """
    return "windows" if sys.platform == "win32" else "linux"


def open_app_command(app_name: str) -> list[str]:
    """Build the subprocess command to launch an application by name.

    Args:
        app_name: The application or document to open.

    Returns:
        A command list suitable for :func:`subprocess.run`/``Popen``.
        Windows uses ``['cmd', '/c', 'start', '', app_name]``; Linux uses
        ``['xdg-open', app_name]``.
    """
    if get_platform() == "windows":
        return ["cmd", "/c", "start", "", app_name]
    return ["xdg-open", app_name]


def kill_app_command(process_name: str) -> list[str]:
    """Build the subprocess command to kill a named process.

    Args:
        process_name: The process/image name to terminate.

    Returns:
        A command list: ``taskkill`` on Windows, ``pkill`` on Linux.
    """
    if get_platform() == "windows":
        return ["taskkill", "/F", "/IM", process_name]
    return ["pkill", "-f", process_name]


def shell() -> str:
    """Return the shell to use for terminal commands.

    Returns:
        ``"powershell"`` on Windows, ``"bash"`` on Linux.
    """
    return "powershell" if get_platform() == "windows" else "bash"


def home_dir() -> Path:
    """Return the user's home directory.

    Returns:
        The home directory as a :class:`pathlib.Path`.
    """
    return Path.home()


def desktop_dir() -> Path:
    """Return the user's Desktop directory, best-effort.

    Returns:
        The ``Desktop`` path under home if it exists, otherwise the home
        directory itself.
    """
    home = home_dir()
    candidate = home / "Desktop"
    return candidate if candidate.exists() else home


# ── Windows App Resolution ────────────────────────────────────────────────────

def _find_app_windows(app_name: str) -> str | None:
    """Resolve an application name to a full executable path on Windows.

    Search order:
        1. Registry ``App Paths`` keys under HKLM and HKCU, including the
           WOW6432Node mirror (Steam, Discord, Spotify, Epic register here).
        2. Common install directories (Program Files, ``LocalAppData\\Programs``).
        3. Top-level ``AppData\\Local`` app folders (Discord, WhatsApp, Postman
           install directly here), matched by name to avoid a slow full scan.

    Args:
        app_name: Application name, with or without a ``.exe`` suffix.

    Returns:
        The resolved executable path string, or None if not found.
    """
    import winreg

    exe_name = app_name if app_name.endswith(".exe") else f"{app_name}.exe"

    # 1. Registry: HKLM and HKCU App Paths
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for sub in (
            rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe_name}",
            rf"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths\{exe_name}",
        ):
            try:
                with winreg.OpenKey(hive, sub) as key:
                    path, _ = winreg.QueryValueEx(key, "")
                    if path and Path(path).exists():
                        return str(Path(path))
            except (FileNotFoundError, OSError):
                continue

    # 2. Common install directories
    localappdata = Path(os.environ.get("LOCALAPPDATA", ""))
    search_roots = [
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")),
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")),
        localappdata / "Programs",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for match in root.rglob(exe_name):
            if match.is_file():
                return str(match)

    # 3. AppData\Local — Discord, WhatsApp, Postman install here directly
    # Only search top-level app folders (not full rglob which is too slow)
    if localappdata.exists():
        app_base = exe_name.replace(".exe", "")
        for folder in localappdata.iterdir():
            if not folder.is_dir():
                continue
            if app_base.lower() in folder.name.lower():
                for match in folder.rglob(exe_name):
                    if match.is_file():
                        return str(match)

    return None


# ── Linux App Resolution ──────────────────────────────────────────────────────

_DESKTOP_DIRS = [
    Path("/usr/share/applications"),
    Path("/usr/local/share/applications"),
    Path("/var/lib/snapd/desktop/applications"),
    Path("/var/lib/flatpak/exports/share/applications"),
    Path.home() / ".local/share/applications",
]


def _find_app_linux(app_name: str) -> str | None:
    """Resolve an application name to a launch command on Linux.

    Search order:
        1. ``shutil.which`` — the binary is already on PATH.
        2. ``.desktop`` file scan across system, Snap, Flatpak, and per-user
           application directories, matching on the entry ``Name`` or the file
           stem and extracting the ``Exec`` binary.
        3. Common binary directories (``/opt``, ``/usr/local/bin``, ``/usr/games``).

    Args:
        app_name: The application name to resolve.

    Returns:
        A binary path or launch command string, or None if not found.
    """
    # 1. which — fastest check
    found = shutil.which(app_name)
    if found:
        return found

    # 2. .desktop file scan
    query = app_name.lower()
    for desktop_dir in _DESKTOP_DIRS:
        if not desktop_dir.exists():
            continue
        for desktop_file in desktop_dir.glob("*.desktop"):
            try:
                parser = configparser.ConfigParser(
                    strict=False, interpolation=None
                )
                parser.read(desktop_file, encoding="utf-8")
                if "Desktop Entry" not in parser:
                    continue
                entry = parser["Desktop Entry"]
                name = entry.get("Name", "").lower()
                exec_val = entry.get("Exec", "")
                if not exec_val:
                    continue
                if query in name or query in desktop_file.stem.lower():
                    binary = exec_val.split()[0].strip()
                    binary = binary.split("%")[0].strip()
                    if binary:
                        return binary
            except Exception:
                continue

    # 3. Common binary locations
    for base in ("/opt", "/usr/local/bin", "/usr/games"):
        candidate = Path(base) / app_name
        if candidate.exists():
            return str(candidate)

    return None


# ── Public API ────────────────────────────────────────────────────────────────

def find_app_path(app_name: str) -> str | None:
    """Resolve an application name to a full path or launch command.

    Dispatches to the Windows or Linux resolver based on the current platform.

    Args:
        app_name: The application name to resolve.

    Returns:
        The resolved path/command string if found, or None. Callers should
        fall back to :func:`open_app_command` when None is returned.
    """
    if get_platform() == "windows":
        return _find_app_windows(app_name)
    return _find_app_linux(app_name)


def find_uwp_app_id(app_name: str) -> str | None:
    """Find the AppUserModelId for a Windows Store / UWP app by name.

    UWP apps cannot be launched by running their ``.exe`` directly; they must
    be launched via ``explorer.exe shell:AppsFolder\\<AppId>``. This queries
    PowerShell's ``Get-StartApps`` for the first name match.

    Args:
        app_name: A substring of the app's display name to match.

    Returns:
        The AppUserModelId string if found, or None (also None on non-Windows
        platforms, on timeout, or on any query failure).
    """
    if get_platform() != "windows":
        return None

    try:
        import subprocess as sp
        # Use PowerShell Get-StartApps to find the app
        result = sp.run(
            ["powershell", "-NoProfile", "-Command",
             f"Get-StartApps | Where-Object {{ $_.Name -like '*{app_name}*' }} | Select-Object -First 1 -ExpandProperty AppID"],
            capture_output=True, text=True, timeout=10,
        )
        app_id = result.stdout.strip()
        if app_id and not app_id.startswith("Error"):
            return app_id
    except Exception:
        pass

    return None


def special_dirs() -> dict[str, Path]:
    """Return a mapping of well-known user directories.

    On Windows, OneDrive-backed copies of these folders (e.g.
    ``~/OneDrive/Desktop``) take precedence when present. Any directory that
    cannot be located falls back to the home directory.

    Returns:
        A dict with keys ``home``, ``desktop``, ``documents``, and
        ``downloads`` mapping to resolved :class:`pathlib.Path` objects.
    """
    home = Path.home()

    def _resolve(name: str) -> Path:
        """Resolve a named user folder, preferring a OneDrive-backed copy.

        Args:
            name: The folder name (e.g. ``"Documents"``).

        Returns:
            The OneDrive path if present, else the home-relative path if it
            exists, else the home directory.
        """
        # Prefer the OneDrive-backed copy when Windows backup is enabled.
        onedrive = home / "OneDrive" / name
        if onedrive.exists():
            return onedrive
        candidate = home / name
        return candidate if candidate.exists() else home

    return {
        "home":      home,
        "desktop":   _resolve("Desktop"),
        "documents": _resolve("Documents"),
        "downloads": _resolve("Downloads"),
    }
