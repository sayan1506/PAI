"""
core/confirmation.py

Destructive action confirmation gate.

Defines which operations require explicit user confirmation before executing,
and provides request_confirmation() to prompt the user.

Two scopes:
  File system — "delete" and "move" (move can silently overwrite an existing
                target on some platforms).
  Terminal    — commands that match risky patterns but are NOT in the absolute
                denylist in tools/terminal.py. The denylist always blocks;
                this confirmation list asks first.

Voice mode: request_confirmation() auto-denies and logs a clear message.
Voice confirmation (speak "yes"/"no") is a planned future feature.

Config key: CONFIRM_DESTRUCTIVE (default True). Set False to bypass all gates.
"""

import re
import sys

import config
from core.logger import logger


# ── File system ───────────────────────────────────────────────────────────────

DESTRUCTIVE_FS_OPS: frozenset[str] = frozenset({"delete", "move"})


# ── Terminal ──────────────────────────────────────────────────────────────────
# These patterns are LESS severe than tools/terminal.py's DENIED_PATTERNS
# (which always block). These prompt the user and let them decide.

_RISKY_TERMINAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"\brm\b",         re.IGNORECASE),  # any rm
    re.compile(r"\bdel\b",        re.IGNORECASE),  # Windows del
    re.compile(r"\brmdir\b",      re.IGNORECASE),  # remove directory
    re.compile(r"\brd\s+/s\b",    re.IGNORECASE),  # Windows rd /s
    re.compile(r"\btaskkill\b",   re.IGNORECASE),  # Windows process kill
    re.compile(r"\bpkill\b",      re.IGNORECASE),  # Linux process kill
    re.compile(r"\bkill\s+-9\b",  re.IGNORECASE),  # force kill
    re.compile(r"\btruncate\b",   re.IGNORECASE),  # truncate file
]


# ── Public API ────────────────────────────────────────────────────────────────

def needs_file_confirmation(operation: str) -> bool:
    """
    Return True if the file system operation requires confirmation.

    Args:
        operation: FileSystemTool operation string (e.g. "delete", "move").

    Returns:
        True if CONFIRM_DESTRUCTIVE is enabled and op is in DESTRUCTIVE_FS_OPS.
    """
    return config.CONFIRM_DESTRUCTIVE and operation in DESTRUCTIVE_FS_OPS


def is_risky_terminal_command(command: str) -> bool:
    """
    Return True if the command matches any risky terminal pattern.

    Does NOT overlap with tools/terminal.py's DENIED_PATTERNS (those always
    block; this function only checks the softer confirmation list).

    Args:
        command: Shell command string to check.

    Returns:
        True if the command matches any pattern in _RISKY_TERMINAL_PATTERNS.
    """
    return any(p.search(command) for p in _RISKY_TERMINAL_PATTERNS)


def needs_terminal_confirmation(command: str) -> bool:
    """
    Return True if the terminal command requires confirmation.

    Args:
        command: Shell command string.

    Returns:
        True if CONFIRM_DESTRUCTIVE is enabled and command is risky.
    """
    return config.CONFIRM_DESTRUCTIVE and is_risky_terminal_command(command)


def request_confirmation(prompt: str) -> bool:
    """
    Display a Y/N confirmation prompt and return the user's decision.

    Text mode: prints the prompt and reads from stdin.
    Voice mode: auto-denies for safety and explains why in the log.
      (Voice Y/N confirmation is a future feature.)

    Args:
        prompt: Human-readable description of the action being confirmed.
                Example: "Delete file /home/user/important.txt"

    Returns:
        True  if the user typed 'y' or 'yes' (case-insensitive).
        False in all other cases (including voice mode and interrupted input).
    """
    if not config.CONFIRM_DESTRUCTIVE:
        return True  # guard not active — always proceed

    logger.warning(f"Confirmation required: {prompt}")

    # Voice mode safety: cannot interleave blocking input() with the audio
    # pipeline. Auto-deny and tell user to use text mode.
    if config.VOICE_ENABLED and not _is_interactive_tty():
        logger.warning(
            "Destructive action requested in voice mode — auto-denied for safety. "
            "Use text mode (python main.py without --voice) to confirm "
            "destructive operations."
        )
        return False

    try:
        print(f"\n⚠  {prompt}")
        response = input("   Confirm? [y/N] ").strip().lower()
        confirmed = response in ("y", "yes")
        action = "confirmed" if confirmed else "denied"
        logger.info(f"User {action} destructive action: {prompt}")
        return confirmed
    except (EOFError, KeyboardInterrupt):
        logger.info(f"Confirmation interrupted — treating as denied: {prompt}")
        return False


# ── Internal ──────────────────────────────────────────────────────────────────

def _is_interactive_tty() -> bool:
    """Return True if stdin is an interactive terminal (not piped)."""
    return sys.stdin.isatty()
