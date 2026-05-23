"""
core/reminder_scheduler.py

Background daemon thread that fires past-due reminders as desktop
notifications. Polls the reminders table every REMINDER_POLL_INTERVAL
seconds. Stops cleanly when stop() is called or when the PAI process exits.
"""

import threading
import time

import config
from core.logger import logger
from ui.notifier import notify


class ReminderScheduler:
    """
    Polls SQLite every REMINDER_POLL_INTERVAL seconds for due reminders,
    fires each as a desktop notification, and marks it fired.

    Usage:
        scheduler = ReminderScheduler()
        scheduler.start()     # call once at PAI startup
        ...
        scheduler.stop()      # call in finally block at shutdown
    """

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """
        Start the background polling thread.

        The thread is a daemon so it will not prevent Python from exiting
        if stop() is never called (e.g. on Ctrl+C without finally block).
        Calling start() when already running is a no-op.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.debug("ReminderScheduler already running, ignoring start()")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="reminder-scheduler",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"ReminderScheduler started "
            f"(poll interval: {config.REMINDER_POLL_INTERVAL}s)"
        )

    def stop(self) -> None:
        """
        Signal the polling thread to stop and wait for it to exit.

        Safe to call even if start() was never called.
        Times out after (REMINDER_POLL_INTERVAL + 2) seconds to avoid
        blocking shutdown indefinitely.
        """
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            timeout = config.REMINDER_POLL_INTERVAL + 2
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    "ReminderScheduler thread did not exit within timeout"
                )
        self._thread = None
        logger.info("ReminderScheduler stopped")

    def _run(self) -> None:
        """
        Main polling loop.

        Runs until _stop_event is set. On each iteration:
          1. Import pending_reminders and mark_reminder_fired locally
             to avoid a circular import at module level.
          2. Call pending_reminders() to get all past-due, unfired rows.
          3. For each reminder, call _fire() then mark_reminder_fired().
          4. Sleep REMINDER_POLL_INTERVAL seconds (interruptible by stop_event).
        """
        from memory.store import pending_reminders, mark_reminder_fired

        logger.debug("ReminderScheduler loop started")
        while not self._stop_event.is_set():
            try:
                due = pending_reminders()
                for reminder in due:
                    self._fire(reminder)
                    mark_reminder_fired(reminder["id"])
            except Exception as e:
                logger.error(f"ReminderScheduler error during poll: {e}")

            # Use wait() instead of sleep() so stop() wakes us immediately
            self._stop_event.wait(timeout=config.REMINDER_POLL_INTERVAL)

        logger.debug("ReminderScheduler loop exited")

    def _fire(self, reminder: dict) -> None:
        """
        Fire a single reminder as a desktop notification.

        Args:
            reminder: Dict with keys: id, message, fire_at, created_at.

        Calls ui.notifier.notify() with title="PAI Reminder" and the
        reminder's message as the body. Logs the event. Does not raise.
        """
        msg = reminder.get("message", "(no message)")
        logger.info(f"Firing reminder #{reminder['id']}: {msg}")
        try:
            notify(msg, title="PAI Reminder")
        except Exception as e:
            logger.error(
                f"ReminderScheduler: notify() failed for #{reminder['id']}: {e}"
            )
