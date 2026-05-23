"""
tests/core/test_reminder_scheduler.py

Unit tests for ReminderScheduler — all threading, store, and notify calls mocked.
"""

import sqlite3
import threading
from unittest.mock import patch, MagicMock, call

import pytest

from core.reminder_scheduler import ReminderScheduler


@pytest.fixture
def scheduler():
    """Create a fresh ReminderScheduler for each test."""
    s = ReminderScheduler()
    yield s
    # Ensure cleanup even if test forgets to stop
    s._stop_event.set()
    if s._thread and s._thread.is_alive():
        s._thread.join(timeout=2)


# ─── 1. Thread lifecycle ─────────────────────────────────────────────────────


def test_scheduler_start_creates_daemon_thread(scheduler):
    """start() creates a daemon thread that is alive."""
    with patch.object(scheduler, "_run", side_effect=lambda: scheduler._stop_event.wait()):
        scheduler.start()
        assert scheduler._thread is not None
        assert scheduler._thread.daemon is True
        assert scheduler._thread.is_alive() is True
        scheduler.stop()


def test_scheduler_start_idempotent(scheduler):
    """Calling start() twice does not create a second thread."""
    with patch.object(scheduler, "_run", side_effect=lambda: scheduler._stop_event.wait()):
        scheduler.start()
        first_thread = scheduler._thread
        scheduler.start()
        assert scheduler._thread is first_thread
        # Verify only one thread with our name exists
        reminder_threads = [
            t for t in threading.enumerate()
            if t.name == "reminder-scheduler"
        ]
        assert len(reminder_threads) == 1
        scheduler.stop()


def test_scheduler_stop_joins_thread(scheduler):
    """stop() joins the thread and sets _thread to None."""
    with patch.object(scheduler, "_run", side_effect=lambda: scheduler._stop_event.wait()):
        scheduler.start()
        assert scheduler._thread is not None
        scheduler.stop()
        assert scheduler._thread is None


# ─── 2. Reminder firing ──────────────────────────────────────────────────────


@patch("core.reminder_scheduler.notify")
@patch("memory.store.mark_reminder_fired")
@patch("memory.store.pending_reminders")
def test_scheduler_fires_due_reminders(mock_pending, mock_mark, mock_notify, scheduler):
    """_run() fires due reminders via notify and marks them fired."""
    mock_pending.return_value = [
        {"id": 1, "message": "eat", "fire_at": "2024-01-15T12:00:00Z"}
    ]

    # Set stop event so _run exits after one iteration
    def stop_after_poll(*args, **kwargs):
        scheduler._stop_event.set()
        return True  # wait returns True when event is set

    with patch.object(scheduler._stop_event, "wait", side_effect=stop_after_poll):
        scheduler._run()

    mock_notify.assert_called_once_with("eat", title="PAI Reminder")
    mock_mark.assert_called_once_with(1)


@patch("core.reminder_scheduler.notify")
@patch("memory.store.mark_reminder_fired")
@patch("memory.store.pending_reminders")
def test_scheduler_marks_reminder_fired_after_notify(mock_pending, mock_mark, mock_notify, scheduler):
    """mark_reminder_fired is called AFTER notify (order matters)."""
    mock_pending.return_value = [
        {"id": 2, "message": "stretch", "fire_at": "2024-01-15T12:00:00Z"}
    ]

    call_order = []
    mock_notify.side_effect = lambda *a, **kw: call_order.append("notify")
    mock_mark.side_effect = lambda *a, **kw: call_order.append("mark_fired")

    def stop_after_poll(*args, **kwargs):
        scheduler._stop_event.set()
        return True

    with patch.object(scheduler._stop_event, "wait", side_effect=stop_after_poll):
        scheduler._run()

    assert call_order == ["notify", "mark_fired"]


# ─── 3. Error handling ───────────────────────────────────────────────────────


@patch("core.reminder_scheduler.notify")
@patch("memory.store.mark_reminder_fired")
@patch("memory.store.pending_reminders")
def test_scheduler_handles_notify_exception_gracefully(mock_pending, mock_mark, mock_notify, scheduler):
    """If notify raises, _run() does not propagate and still marks fired."""
    mock_pending.return_value = [
        {"id": 5, "message": "test", "fire_at": "2024-01-15T12:00:00Z"}
    ]
    mock_notify.side_effect = RuntimeError("display error")

    def stop_after_poll(*args, **kwargs):
        scheduler._stop_event.set()
        return True

    with patch.object(scheduler._stop_event, "wait", side_effect=stop_after_poll):
        scheduler._run()  # Should not raise

    mock_mark.assert_called_once_with(5)


@patch("core.reminder_scheduler.notify")
@patch("memory.store.pending_reminders")
def test_scheduler_handles_db_exception_gracefully(mock_pending, mock_notify, scheduler):
    """If pending_reminders raises a DB error, _run() logs and continues."""
    mock_pending.side_effect = sqlite3.OperationalError("database is locked")

    def stop_after_poll(*args, **kwargs):
        scheduler._stop_event.set()
        return True

    with patch.object(scheduler._stop_event, "wait", side_effect=stop_after_poll):
        scheduler._run()  # Should not raise

    # notify should never have been called since we couldn't get reminders
    mock_notify.assert_not_called()


# ─── 4. Notify integration ───────────────────────────────────────────────────


@patch("core.reminder_scheduler.notify")
def test_fire_calls_notify_with_correct_message(mock_notify, scheduler):
    """_fire() calls notify with the reminder's message."""
    scheduler._fire({"id": 3, "message": "drink water", "fire_at": "2024-01-15T12:00:00Z"})
    mock_notify.assert_called_once_with("drink water", title="PAI Reminder")


@patch("core.reminder_scheduler.notify")
def test_fire_uses_title_pai_reminder(mock_notify, scheduler):
    """_fire() passes title='PAI Reminder' as keyword arg to notify."""
    scheduler._fire({"id": 1, "message": "test", "fire_at": "2024-01-15T12:00:00Z"})
    mock_notify.assert_called_once()
    _, kwargs = mock_notify.call_args
    assert kwargs["title"] == "PAI Reminder"
