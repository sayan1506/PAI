"""
Desktop notification wrapper using plyer.

Provides a best-effort notify() function that is gated by
NOTIFICATIONS_ENABLED and swallows all exceptions (debug log only).
"""

import logging

import config

logger = logging.getLogger(__name__)


def notify(message: str, timeout: int = 5) -> None:
    """Send a desktop toast notification.

    Args:
        message: The notification body text (truncated to 80 chars).
        timeout: How long the notification stays visible, in seconds.

    Notes:
        - Returns immediately if NOTIFICATIONS_ENABLED is false.
        - All exceptions are caught and logged at DEBUG level.
        - plyer is imported inside the function to avoid import-time
          failures on systems without notification backends.
    """
    if not config.NOTIFICATIONS_ENABLED:
        return

    try:
        from plyer import notification  # noqa: PLC0415

        truncated = message[:80]
        notification.notify(
            title=config.AGENT_NAME,
            message=truncated,
            app_name=config.AGENT_NAME,
            timeout=timeout,
        )
    except Exception:
        logger.debug("Notification failed", exc_info=True)
