"""
tests/ui/test_notifier.py

Unit tests for the desktop notification wrapper (ui/notifier.py).
All external dependencies (plyer) are mocked — no real notification
backend needed.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock plyer so tests run without it installed
if "plyer" not in sys.modules:
    sys.modules["plyer"] = MagicMock()
    sys.modules["plyer.notification"] = MagicMock()


from ui.notifier import notify


class TestNotifyEnabled:
    """Tests for notify() when NOTIFICATIONS_ENABLED=true."""

    def test_calls_plyer_notify(self):
        """notify() calls plyer.notification.notify when enabled."""
        with patch("ui.notifier.config") as mock_config:
            mock_config.NOTIFICATIONS_ENABLED = True
            mock_config.AGENT_NAME = "PAI"

            mock_notif_module = MagicMock()
            with patch.dict(sys.modules, {"plyer": MagicMock(notification=mock_notif_module),
                                          "plyer.notification": mock_notif_module}):
                notify("Hello from PAI")

                mock_notif_module.notify.assert_called_once_with(
                    title="PAI",
                    message="Hello from PAI",
                    app_name="PAI",
                    timeout=5,
                )


class TestNotifyDisabled:
    """Tests for notify() when NOTIFICATIONS_ENABLED=false."""

    def test_does_not_call_plyer(self):
        """notify() does NOT call plyer when disabled."""
        with patch("ui.notifier.config") as mock_config:
            mock_config.NOTIFICATIONS_ENABLED = False

            mock_notif_module = MagicMock()
            with patch.dict(sys.modules, {"plyer": MagicMock(notification=mock_notif_module),
                                          "plyer.notification": mock_notif_module}):
                notify("Should not fire")

                mock_notif_module.notify.assert_not_called()


class TestExceptionSwallowing:
    """Tests that notify() swallows all exceptions."""

    def test_no_exception_propagates_on_plyer_error(self):
        """When plyer raises, no exception propagates (debug log only)."""
        with patch("ui.notifier.config") as mock_config, \
             patch("ui.notifier.logger") as mock_logger:
            mock_config.NOTIFICATIONS_ENABLED = True
            mock_config.AGENT_NAME = "PAI"

            mock_notif_module = MagicMock()
            mock_notif_module.notify.side_effect = RuntimeError("No notification backend")
            with patch.dict(sys.modules, {"plyer": MagicMock(notification=mock_notif_module),
                                          "plyer.notification": mock_notif_module}):
                # Should not raise
                notify("This will fail silently")

                # Debug log should have been called
                mock_logger.debug.assert_called_once()

    def test_import_error_is_swallowed(self):
        """When plyer import fails, no exception propagates."""
        with patch("ui.notifier.config") as mock_config, \
             patch("ui.notifier.logger") as mock_logger:
            mock_config.NOTIFICATIONS_ENABLED = True
            mock_config.AGENT_NAME = "PAI"

            # Remove plyer from sys.modules and make import fail
            with patch.dict(sys.modules, {"plyer": None}):
                # Should not raise even if plyer can't be imported
                notify("Import will fail")

                mock_logger.debug.assert_called_once()


class TestAppName:
    """Tests that config.AGENT_NAME is passed as app_name."""

    def test_agent_name_passed_as_app_name(self):
        """config.AGENT_NAME is passed as both title and app_name to plyer."""
        with patch("ui.notifier.config") as mock_config:
            mock_config.NOTIFICATIONS_ENABLED = True
            mock_config.AGENT_NAME = "CustomAgent"

            mock_notif_module = MagicMock()
            with patch.dict(sys.modules, {"plyer": MagicMock(notification=mock_notif_module),
                                          "plyer.notification": mock_notif_module}):
                notify("test message")

                call_kwargs = mock_notif_module.notify.call_args[1]
                assert call_kwargs["app_name"] == "CustomAgent"
                assert call_kwargs["title"] == "CustomAgent"


class TestTimeout:
    """Tests that custom timeout is passed through to plyer."""

    def test_default_timeout_is_5(self):
        """Default timeout of 5 seconds is passed to plyer."""
        with patch("ui.notifier.config") as mock_config:
            mock_config.NOTIFICATIONS_ENABLED = True
            mock_config.AGENT_NAME = "PAI"

            mock_notif_module = MagicMock()
            with patch.dict(sys.modules, {"plyer": MagicMock(notification=mock_notif_module),
                                          "plyer.notification": mock_notif_module}):
                notify("test")

                call_kwargs = mock_notif_module.notify.call_args[1]
                assert call_kwargs["timeout"] == 5

    def test_custom_timeout_passed_through(self):
        """Custom timeout value is passed through to plyer."""
        with patch("ui.notifier.config") as mock_config:
            mock_config.NOTIFICATIONS_ENABLED = True
            mock_config.AGENT_NAME = "PAI"

            mock_notif_module = MagicMock()
            with patch.dict(sys.modules, {"plyer": MagicMock(notification=mock_notif_module),
                                          "plyer.notification": mock_notif_module}):
                notify("test", timeout=10)

                call_kwargs = mock_notif_module.notify.call_args[1]
                assert call_kwargs["timeout"] == 10


class TestMessageTruncation:
    """Tests that messages longer than 80 chars are truncated."""

    def test_long_message_truncated_to_80_chars(self):
        """Messages longer than 80 characters are truncated."""
        with patch("ui.notifier.config") as mock_config:
            mock_config.NOTIFICATIONS_ENABLED = True
            mock_config.AGENT_NAME = "PAI"

            mock_notif_module = MagicMock()
            with patch.dict(sys.modules, {"plyer": MagicMock(notification=mock_notif_module),
                                          "plyer.notification": mock_notif_module}):
                long_message = "A" * 120
                notify(long_message)

                call_kwargs = mock_notif_module.notify.call_args[1]
                assert len(call_kwargs["message"]) == 80
                assert call_kwargs["message"] == "A" * 80

    def test_short_message_not_truncated(self):
        """Messages shorter than 80 characters are not truncated."""
        with patch("ui.notifier.config") as mock_config:
            mock_config.NOTIFICATIONS_ENABLED = True
            mock_config.AGENT_NAME = "PAI"

            mock_notif_module = MagicMock()
            with patch.dict(sys.modules, {"plyer": MagicMock(notification=mock_notif_module),
                                          "plyer.notification": mock_notif_module}):
                short_message = "Hello PAI"
                notify(short_message)

                call_kwargs = mock_notif_module.notify.call_args[1]
                assert call_kwargs["message"] == "Hello PAI"
