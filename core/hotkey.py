"""
core/hotkey.py

HotkeyListener — listens globally for a keyboard shortcut and fires a callback.

Default combo: Ctrl+Shift+J (config.HOTKEY_COMBO).
Uses pynput.keyboard.GlobalHotKeys — works user-space on Windows and Linux.
"""

import threading
from collections.abc import Callable

import config
from core.logger import logger


class HotkeyListener:
    """
    Background daemon thread that detects a global keyboard shortcut.

    Args:
        callback: Zero-argument callable invoked on each hotkey activation.
        combo:    Key combo in pynput format. Defaults to config.HOTKEY_COMBO.
    """

    def __init__(
        self,
        callback: Callable[[], None],
        combo: str | None = None,
    ) -> None:
        self._callback = callback
        self._combo = combo or config.HOTKEY_COMBO
        self._thread: threading.Thread | None = None
        self._pynput_listener = None

    def start(self) -> None:
        """Start the listener in a background daemon thread."""
        if not config.HOTKEY_ENABLED:
            logger.debug("HotkeyListener: disabled (HOTKEY_ENABLED=false)")
            return

        if self._thread and self._thread.is_alive():
            logger.debug("HotkeyListener: already running")
            return

        self._thread = threading.Thread(
            target=self._run,
            name="HotkeyListener",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"HotkeyListener started — combo: {self._combo}")

    def stop(self) -> None:
        """Stop the listener. Safe to call if never started."""
        if self._pynput_listener is not None:
            try:
                self._pynput_listener.stop()
            except Exception:
                pass
            self._pynput_listener = None
            logger.info("HotkeyListener stopped")

    def is_alive(self) -> bool:
        """Return True if the listener thread is currently running."""
        return self._thread is not None and self._thread.is_alive()

    def _on_activate(self) -> None:
        """Called by pynput when the configured combo is pressed."""
        logger.info(f"Hotkey activated ({self._combo})")
        try:
            self._callback()
        except Exception as e:
            logger.error(f"HotkeyListener callback raised: {e}")

    def _run(self) -> None:
        """Thread target: block in pynput's event loop until stopped."""
        try:
            from pynput import keyboard

            self._pynput_listener = keyboard.GlobalHotKeys(
                {self._combo: self._on_activate}
            )
            self._pynput_listener.run()
        except ImportError:
            logger.error(
                "HotkeyListener: pynput not installed. "
                "Run: pip install pynput>=1.7.6"
            )
        except Exception as e:
            logger.error(f"HotkeyListener thread crashed: {e}")
