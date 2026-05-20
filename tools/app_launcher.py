"""
tools/app_launcher.py

Application launcher tool for PAI.
Handles launching, closing, and querying running applications.
"""

import subprocess

import psutil

import config
from core.logger import logger
from tools.base import BaseTool, ToolResult
from utils.platform_utils import get_platform, open_app_command, find_app_path, find_uwp_app_id


class AppLauncherTool(BaseTool):
    """Application launcher and manager tool."""

    # Cross-platform app name aliases
    APP_ALIASES = {
        "windows": {
            # Browsers
            "chrome": "chrome.exe",
            "google chrome": "chrome.exe",
            "brave": "brave.exe",
            "firefox": "firefox.exe",
            "edge": "msedge.exe",
            # System
            "notepad": "notepad.exe",
            "explorer": "explorer.exe",
            "file explorer": "explorer.exe",
            "vscode": "Code.exe",
            "vs code": "Code.exe",
            "visual studio code": "Code.exe",
            "terminal": "wt.exe",
            "windows terminal": "wt.exe",
            "cmd": "cmd.exe",
            "powershell": "powershell.exe",
            "calculator": "calc.exe",
            "paint": "mspaint.exe",
            "task manager": "taskmgr.exe",
            # Gaming / launchers
            "steam": "steam.exe",
            "epic": "EpicGamesLauncher.exe",
            "epic games": "EpicGamesLauncher.exe",
            "epic games launcher": "EpicGamesLauncher.exe",
            "gog": "GalaxyClient.exe",
            "gog galaxy": "GalaxyClient.exe",
            # Common apps
            "spotify": "Spotify.exe",
            "discord": "Discord.exe",
            "slack": "slack.exe",
            "zoom": "Zoom.exe",
            "vlc": "vlc.exe",
            "obs": "obs64.exe",
            "obs studio": "obs64.exe",
            "7zip": "7zFM.exe",
            "7-zip": "7zFM.exe",
            "whatsapp": "WhatsApp.exe",
            "postman": "Postman.exe",
        },
        "linux": {
            # Browsers
            "chrome": "google-chrome",
            "google chrome": "google-chrome",
            "brave": "brave-browser",
            "firefox": "firefox",
            "edge": "microsoft-edge",
            # System
            "notepad": "gedit",
            "explorer": "nautilus",
            "file manager": "nautilus",
            "vscode": "code",
            "vs code": "code",
            "visual studio code": "code",
            "terminal": "gnome-terminal",
            "calculator": "gnome-calculator",
            "paint": "gimp",
            # Gaming / launchers
            "steam": "steam",
            "lutris": "lutris",
            "heroic": "heroic",
            "epic": "heroic",
            "epic games": "heroic",
            # Common apps
            "spotify": "spotify",
            "discord": "discord",
            "slack": "slack",
            "zoom": "zoom",
            "vlc": "vlc",
            "obs": "obs",
            "obs studio": "obs",
        },
    }

    @property
    def name(self) -> str:
        return "app_launcher"

    @property
    def description(self) -> str:
        return (
            "Launch, close, or check the status of desktop applications. "
            "Use 'launch' to open an app by name, 'close' to quit it, "
            "'list_running' to see what is open, or 'is_running' to check "
            "if a specific app is currently running."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["launch", "close", "list_running", "is_running"],
                    "description": "The operation to perform.",
                },
                "app_name": {
                    "type": "string",
                    "description": "Name of the application (e.g. 'notepad', 'chrome').",
                },
            },
            "required": ["operation"],
        }

    def execute(self, **kwargs) -> ToolResult:
        """Execute an app launcher operation."""
        if not config.ENABLE_APP_LAUNCHER:
            return ToolResult(success=False, output="", error="App launcher is disabled.")

        operation = kwargs.get("operation")
        try:
            handler = getattr(self, f"_op_{operation}", None)
            if handler is None:
                return ToolResult(success=False, output="", error=f"Unknown operation: {operation}")
            return handler(**kwargs)
        except Exception as e:
            logger.error(f"AppLauncherTool error: {e}")
            return ToolResult(success=False, output="", error=str(e))

    def _resolve_app_name(self, app_name: str) -> str:
        """Resolve an app name using platform-specific aliases."""
        platform = get_platform()
        aliases = self.APP_ALIASES.get(platform, {})
        return aliases.get(app_name.lower(), app_name)

    # Known UWP/Store apps that need special launch via shell protocol
    UWP_PROTOCOLS = {
        "WhatsApp.exe": "whatsapp:",
        "whatsapp": "whatsapp:",
    }

    def _op_launch(self, **kwargs) -> ToolResult:
        """Launch an application by name."""
        app_name = kwargs.get("app_name", "")
        if not app_name:
            return ToolResult(success=False, error="No app_name provided.")

        resolved = self._resolve_app_name(app_name)

        try:
            # Try full path resolution first
            full_path = find_app_path(resolved)

            if full_path:
                # Check if it's a WindowsApps path (UWP) — can't launch directly
                if "WindowsApps" in full_path:
                    # Must use shell activation for UWP apps
                    app_id = find_uwp_app_id(app_name)
                    if app_id:
                        subprocess.Popen(
                            ["explorer.exe", f"shell:AppsFolder\\{app_id}"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        logger.info(f"Launched UWP app: {app_name} (id={app_id})")
                        return ToolResult(success=True, output=f"Launched: {app_name}")

                # Regular app — launch directly
                subprocess.Popen(
                    [full_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info(f"Launched app: {app_name} (path={full_path})")
                return ToolResult(success=True, output=f"Launched: {app_name}")

            # Check if it's a UWP app not found via path but available via shell
            app_id = find_uwp_app_id(app_name)
            if app_id:
                subprocess.Popen(
                    ["explorer.exe", f"shell:AppsFolder\\{app_id}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info(f"Launched UWP app: {app_name} (id={app_id})")
                return ToolResult(success=True, output=f"Launched: {app_name}")

            # No path found — report failure
            logger.info(f"App not found: {app_name} (resolved={resolved})")
            return ToolResult(
                success=False,
                error=(
                    f"Could not find '{app_name}' installed on this computer. "
                    "It may not be installed, or it may be a web-based service."
                ),
            )

        except FileNotFoundError:
            return ToolResult(
                success=False,
                error=(
                    f"Could not find '{app_name}'. "
                    "Make sure it is installed on this computer."
                ),
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to launch {app_name}: {e}")

    def _op_close(self, **kwargs) -> ToolResult:
        """Close an application by killing matching processes."""
        app_name = kwargs.get("app_name", "")
        if not app_name:
            return ToolResult(success=False, output="", error="No app_name provided.")

        resolved = self._resolve_app_name(app_name)
        killed = 0

        for proc in psutil.process_iter(["name"]):
            try:
                if proc.info["name"] and resolved.lower() in proc.info["name"].lower():
                    proc.terminate()
                    killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if killed == 0:
            return ToolResult(success=False, output="", error=f"{app_name} is not running.")

        return ToolResult(success=True, output=f"Closed {killed} process(es) matching '{app_name}'.")

    def _op_list_running(self, **kwargs) -> ToolResult:
        """List names of all running processes (deduplicated)."""
        names = set()
        for proc in psutil.process_iter(["name"]):
            try:
                if proc.info["name"]:
                    names.add(proc.info["name"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        sorted_names = sorted(names)
        output = "\n".join(sorted_names) if sorted_names else "(no processes found)"
        return ToolResult(success=True, output=output)

    def _op_is_running(self, **kwargs) -> ToolResult:
        """Check if a specific application is running."""
        app_name = kwargs.get("app_name", "")
        if not app_name:
            return ToolResult(success=False, output="", error="No app_name provided.")

        resolved = self._resolve_app_name(app_name)

        for proc in psutil.process_iter(["name"]):
            try:
                if proc.info["name"] and resolved.lower() in proc.info["name"].lower():
                    return ToolResult(success=True, output=f"{app_name} is running.")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return ToolResult(success=True, output=f"{app_name} is not running.")
