"""
tests/test_hotkey.py

Unit tests for core.hotkey.HotkeyListener.
pynput is mocked — no real keyboard interaction.
"""

import threading
from unittest.mock import Mock, patch

import pytest

import config
from core.hotkey import HotkeyListener


class TestHotkeyDisabled:
    """Tests for when HOTKEY_ENABLED is False."""

    def test_hotkey_start_does_nothing_when_disabled(self, monkeypatch):
        """When config.HOTKEY_ENABLED is False, start() should not spawn a thread."""
        monkeypatch.setattr(config, "HOTKEY_ENABLED", False)
        cb = Mock()
        listener = HotkeyListener(callback=cb)
        listener.start()
        assert listener.is_alive() is False


class TestHotkeyAlive:
    """Tests for is_alive() after start."""

    def test_hotkey_is_alive_after_start(self, monkeypatch):
        """After start() with HOTKEY_ENABLED=True, is_alive() should be True."""
        monkeypatch.setattr(config, "HOTKEY_ENABLED", True)

        # Use a threading.Event to keep the mock GlobalHotKeys.run() blocking
        block_event = threading.Event()

        class FakeGlobalHotKeys:
            def __init__(self, hotkeys):
                pass

            def run(self):
                # Block until the event is set (simulates pynput event loop)
                block_event.wait()

            def stop(self):
                block_event.set()

        with patch("pynput.keyboard.GlobalHotKeys", FakeGlobalHotKeys):
            cb = Mock()
            listener = HotkeyListener(callback=cb)
            listener.start()
            # Give the thread a moment to start
            import time
            time.sleep(0.1)
            try:
                assert listener.is_alive() is True
            finally:
                # Clean up: unblock the thread so it exits
                block_event.set()
                listener._thread.join(timeout=1)


class TestHotkeyCallback:
    """Tests for _on_activate() callback behavior."""

    def test_hotkey_callback_fires_on_activate(self):
        """Calling _on_activate() directly should invoke the callback exactly once."""
        cb = Mock()
        listener = HotkeyListener(callback=cb)
        listener._on_activate()
        cb.assert_called_once()

    def test_hotkey_callback_exception_does_not_crash_listener(self):
        """If the callback raises, _on_activate() should not propagate the exception."""
        cb = Mock(side_effect=RuntimeError("boom"))
        listener = HotkeyListener(callback=cb)
        # Should not raise
        listener._on_activate()
        cb.assert_called_once()


class TestHotkeyStop:
    """Tests for stop() behavior."""

    def test_hotkey_stop_calls_pynput_stop(self):
        """stop() should call _pynput_listener.stop() and set it to None."""
        cb = Mock()
        listener = HotkeyListener(callback=cb)
        # Simulate that a pynput listener was created
        mock_pynput = Mock()
        listener._pynput_listener = mock_pynput

        listener.stop()

        mock_pynput.stop.assert_called_once()
        assert listener._pynput_listener is None
