"""
Unit tests for core/rate_tracker.py

Tests verify:
- get_count returns zero when no requests recorded
- record_request increments the count correctly
- Multiple providers are tracked independently
- Day reset: old date rows don't affect today's count
- is_near_limit returns False below threshold, True at threshold
- is_at_limit returns False below limit, True at limit
- record_request is a noop when MEMORY_ENABLED is False
"""

import sqlite3
from datetime import date, timedelta
from unittest.mock import patch

import pytest

import config
import memory.store
from core import rate_tracker


@pytest.fixture(autouse=True)
def _setup_test_db(tmp_path, monkeypatch):
    """Point memory store at a temporary SQLite DB and initialise tables."""
    db_file = tmp_path / "test_memory.db"
    monkeypatch.setattr(config, "MEMORY_DB_PATH", str(db_file))
    monkeypatch.setattr(config, "MEMORY_ENABLED", True)
    memory.store.init_db()
    yield


def _insert_rate_row(provider: str, day: str, count: int) -> None:
    """Helper: directly insert a rate_limits row for testing."""
    conn = sqlite3.connect(config.MEMORY_DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO rate_limits (provider, date, count, updated_at) "
        "VALUES (?, ?, ?, '2024-01-01T00:00:00+00:00')",
        (provider, day, count),
    )
    conn.commit()
    conn.close()


class TestGetCount:
    """Tests for rate_tracker.get_count()."""

    def test_get_count_returns_zero_on_first_call(self):
        """Fresh DB with no rows returns 0 for any provider."""
        assert rate_tracker.get_count("gemini") == 0


class TestRecordRequest:
    """Tests for rate_tracker.record_request()."""

    def test_record_request_increments_count(self):
        """Calling record_request three times yields count == 3."""
        rate_tracker.record_request("gemini")
        rate_tracker.record_request("gemini")
        rate_tracker.record_request("gemini")
        assert rate_tracker.get_count("gemini") == 3

    def test_record_request_multiple_providers_independent(self):
        """Different providers are tracked independently."""
        rate_tracker.record_request("gemini")
        rate_tracker.record_request("gemini")
        rate_tracker.record_request("openai")

        assert rate_tracker.get_count("gemini") == 2
        assert rate_tracker.get_count("openai") == 1

    def test_record_request_resets_on_new_day(self):
        """Yesterday's count does not carry over; today starts fresh."""
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        _insert_rate_row("gemini", yesterday, 999)

        # Record one request for today
        rate_tracker.record_request("gemini")

        assert rate_tracker.get_count("gemini") == 1

    def test_record_request_noop_when_memory_disabled(self, monkeypatch):
        """When MEMORY_ENABLED is False, record_request does nothing."""
        monkeypatch.setattr(config, "MEMORY_ENABLED", False)

        with patch("memory.store.increment_daily_rate_count") as mock_inc:
            rate_tracker.record_request("gemini")
            mock_inc.assert_not_called()


class TestIsNearLimit:
    """Tests for rate_tracker.is_near_limit()."""

    def test_is_near_limit_false_below_threshold(self, monkeypatch):
        """Returns False when count is below GEMINI_WARN_AT."""
        monkeypatch.setattr(config, "GEMINI_WARN_AT", 800)
        today = date.today().isoformat()
        _insert_rate_row("gemini", today, 799)

        assert rate_tracker.is_near_limit("gemini") is False

    def test_is_near_limit_true_at_threshold(self, monkeypatch):
        """Returns True when count equals GEMINI_WARN_AT."""
        monkeypatch.setattr(config, "GEMINI_WARN_AT", 800)
        today = date.today().isoformat()
        _insert_rate_row("gemini", today, 800)

        assert rate_tracker.is_near_limit("gemini") is True


class TestIsAtLimit:
    """Tests for rate_tracker.is_at_limit()."""

    def test_is_at_limit_false_below_limit(self, monkeypatch):
        """Returns False when count is below GEMINI_DAILY_LIMIT."""
        monkeypatch.setattr(config, "GEMINI_DAILY_LIMIT", 1000)
        today = date.today().isoformat()
        _insert_rate_row("gemini", today, 999)

        assert rate_tracker.is_at_limit("gemini") is False

    def test_is_at_limit_true_at_limit(self, monkeypatch):
        """Returns True when count equals GEMINI_DAILY_LIMIT."""
        monkeypatch.setattr(config, "GEMINI_DAILY_LIMIT", 1000)
        today = date.today().isoformat()
        _insert_rate_row("gemini", today, 1000)

        assert rate_tracker.is_at_limit("gemini") is True
