"""
tools/terminal.py

Terminal/shell command execution tool for PAI.
Disabled by default for safety. Includes a denylist of dangerous patterns.
"""

import re
import subprocess

import config
from core.logger import logger
from tools.base import BaseTool, ToolResult


# Patterns that are always refused regardless of config
DENIED_PATTERNS = [
    r"rm\s+-rf\s+[/~]",        # rm -rf / or ~/
    r"format\s+[a-zA-Z]:",     # Windows format drive
    r"del\s+/[fFsS].*\s+/[qQ]",# Windows del /f /s /q
    r"mkfs\.",                  # Linux disk format
    r"dd\s+if=",               # dd overwrites
    r":\(\){.*};:",             # fork bomb
    r"shutdown",
    r"reboot",
    r"halt",
]


class TerminalTool(BaseTool):
    """Shell command execution tool."""

    @property
    def name(self) -> str:
        return "terminal"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command and return its output. "
            "Use for developer tasks: checking versions, running scripts, "
            "querying system info, or any task requiring CLI access. "
            "ONLY call this when no other tool covers the task. "
            "This tool is powerful — prefer specific tools for file or app tasks."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
            },
            "required": ["command"],
        }

    def execute(self, **kwargs) -> ToolResult:
        """Execute a shell command."""
        if not config.ENABLE_TERMINAL:
            return ToolResult(
                success=False,
                output="",
                error="Terminal tool is disabled. Set ENABLE_TERMINAL=true to enable.",
            )

        command = kwargs.get("command", "").strip()
        if not command:
            return ToolResult(success=False, output="", error="No command provided.")

        # Denylist check
        for pattern in DENIED_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                logger.warning(f"TerminalTool blocked denied command: {command}")
                return ToolResult(
                    success=False,
                    output="",
                    error="Command refused: matches a blocked pattern for safety.",
                )

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = (result.stdout + result.stderr).strip()
            success = result.returncode == 0
            logger.info(f"Terminal: [{result.returncode}] {command[:80]}")
            return ToolResult(success=success, output=output or "(no output)", error="" if success else f"Exit code: {result.returncode}")
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Command timed out after 30 seconds.")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
