"""
tests/tools/test_reminder.py

Unit tests for ReminderTool — all store calls and datetime.now() mocked.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from tools.reminder import ReminderTool
from tools.base import ToolResult


@pytest.fixture
def tool():
    return ReminderTool()


# ─── 1. Metadata ─────────────────────────────────────────────────────────────

def test_reminder_tool_name_and_description(tool):
    assert tool.name == "reminder"
    assert "set" in tool.description.lower()
    assert "list" in tool.description.lower()
    assert "cancel" in tool.description.lower()


# ─── 2. Unknown action ───────────────────────────────────────────────────────

def test_execute_unknown_action_returns_failure(tool):
    result = tool.execute(action="snooze")
    assert result.success is False
    assert "Unknown action" in result.error


# ─── 3–9. _parse_when tests ──────────────────────────────────────────────────

def test_parse_when_minutes(tool):
    now = datetime.now(timezone.utc)
    result = tool._parse_when("10m")
    expected = now + timedelta(minutes=10)
    assert abs((result - expected).total_seconds()) < 2


def test_parse_when_hours(tool):
    now = datetime.now(timezone.utc)
    result = tool._parse_when("3h")
    expected = now + timedelta(hours=3)
    assert abs((result - expected).total_seconds()) < 2


def test_parse_when_seconds(tool):
    now = datetime.now(timezone.utc)
    result = tool._parse_when("45s")
    expected = now + timedelta(seconds=45)
    assert abs((result - expected).total_seconds()) < 2


def test_parse_when_iso_utc_z(tool):
    result = tool._parse_when("2024-06-15T12:00:00Z")
    assert result == datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def test_parse_when_iso_utc_offset(tool):
    result = tool._parse_when("2024-06-15T12:00:00+00:00")
    assert result == datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def test_parse_when_invalid_raises_value_error(tool):
    with pytest.raises(ValueError) as exc_info:
        tool._parse_when("next Tuesday")
    assert "Cannot parse" in str(exc_info.value)


def test_parse_when_zero_duration_raises_value_error(tool):
    with pytest.raises(ValueError):
        tool._parse_when("0m")


# ─── 10–13. _set tests ───────────────────────────────────────────────────────

def test_set_missing_when_returns_error(tool):
    result = tool.execute(action="set", message="stretch")
    assert result.success is False
    assert "'when' is required" in result.error


def test_set_missing_message_returns_error(tool):
    result = tool.execute(action="set", when="10m")
    assert result.success is False
    assert "'message' is required" in result.error


def test_set_invalid_when_returns_error(tool):
    result = tool.execute(action="set", when="ASAP", message="call mom")
    assert result.success is False
    assert "Cannot parse" in result.error


@patch("tools.reminder.datetime")
@patch("tools.reminder.store")
def test_set_stores_reminder_and_returns_confirmation(mock_store, mock_datetime, tool):
    fixed_now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    mock_datetime.now.return_value = fixed_now
    mock_datetime.fromisoformat = datetime.fromisoformat
    # timedelta must still work — patch only datetime class attributes
    mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
    mock_store.add_reminder.return_value = 7

    # We need _parse_when to work correctly, so we patch datetime.now inside it
    # The tool calls datetime.now(timezone.utc) in _parse_when and _set
    with patch.object(tool, '_parse_when', return_value=fixed_now + timedelta(minutes=5)):
        result = tool.execute(action="set", when="5m", message="take medicine")

    mock_store.add_reminder.assert_called_once()
    assert result.success is True
    assert "#7" in result.output
    assert "take medicine" in result.output


# ─── 14–15. _list tests ──────────────────────────────────────────────────────

@patch("tools.reminder.store")
def test_list_empty_returns_no_reminders_message(mock_store, tool):
    mock_store.all_reminders.return_value = []
    result = tool.execute(action="list")
    assert result.success is True
    assert "no pending reminders" in result.output.lower()


@patch("tools.reminder.store")
def test_list_returns_formatted_entries(mock_store, tool):
    mock_store.all_reminders.return_value = [
        {
            "id": 1,
            "message": "stretch",
            "fire_at": "2024-01-15T14:00:00Z",
            "fired": 0,
            "created_at": "2024-01-15T10:00:00Z",
        }
    ]
    result = tool.execute(action="list")
    assert result.success is True
    assert "#1" in result.output
    assert "stretch" in result.output


# ─── 16–20. _cancel tests ────────────────────────────────────────────────────

@patch("tools.reminder.store")
def test_cancel_by_id_success(mock_store, tool):
    mock_store.delete_reminder.return_value = True
    result = tool.execute(action="cancel", id=3)
    assert result.success is True
    assert "#3" in result.output


@patch("tools.reminder.store")
def test_cancel_by_id_not_found(mock_store, tool):
    mock_store.delete_reminder.return_value = False
    result = tool.execute(action="cancel", id=99)
    assert result.success is False
    assert "99" in result.error


@patch("tools.reminder.store")
def test_cancel_by_message_fuzzy_match(mock_store, tool):
    mock_store.all_reminders.return_value = [
        {
            "id": 2,
            "message": "take medicine",
            "fire_at": "2024-01-15T14:00:00Z",
            "fired": 0,
            "created_at": "2024-01-15T10:00:00Z",
        }
    ]
    mock_store.delete_reminder.return_value = True
    result = tool.execute(action="cancel", message="medicine")
    mock_store.delete_reminder.assert_called_with(2)
    assert result.success is True


@patch("tools.reminder.store")
def test_cancel_by_message_no_match(mock_store, tool):
    mock_store.all_reminders.return_value = []
    result = tool.execute(action="cancel", message="dentist")
    assert result.success is False
    assert "dentist" in result.error


def test_cancel_neither_id_nor_message_returns_error(tool):
    result = tool.execute(action="cancel")
    assert result.success is False
    assert "id" in result.error.lower() or "message" in result.error.lower()
