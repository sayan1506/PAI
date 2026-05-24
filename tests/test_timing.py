"""
tests/test_timing.py

Unit tests for core/timing.py — Timer context manager and TurnTimer class.

Tests verify:
- Timer logs stage name at DEBUG level
- Timer measures nonzero elapsed time
- TurnTimer.stage() appends entries to stages list
- Multiple stages are recorded in order
- log_summary() emits an INFO-level line with expected format
- Debug warning is emitted when turn exceeds 2500ms budget
"""

import time
from unittest.mock import patch, call

import pytest

from core.timing import Timer, TurnTimer


class TestTimer:
    """Tests for the Timer context manager."""

    @patch("core.timing.logger")
    def test_timer_logs_stage_name(self, mock_logger):
        """Timer logs the stage name at DEBUG level."""
        with Timer("stt_test"):
            pass

        mock_logger.debug.assert_called_once()
        log_msg = mock_logger.debug.call_args[0][0]
        assert "stt_test" in log_msg

    @patch("core.timing.logger")
    def test_timer_measures_nonzero_elapsed(self, mock_logger):
        """Timer logs a nonzero elapsed time after a short sleep."""
        with Timer("sleep_test"):
            time.sleep(0.01)

        mock_logger.debug.assert_called_once()
        log_msg = mock_logger.debug.call_args[0][0]
        # Extract the numeric portion — format is "[TIMING] sleep_test: X.Xms"
        # The number after the colon should be > 0
        parts = log_msg.split(":")
        ms_part = parts[-1].strip().replace("ms", "")
        assert float(ms_part) > 0


class TestTurnTimer:
    """Tests for the TurnTimer class."""

    def test_turn_timer_stage_appends_to_stages_list(self):
        """TurnTimer.stage() appends one entry with the correct name."""
        tt = TurnTimer()
        with tt.stage("llm"):
            pass

        stages = tt.stages()
        assert len(stages) == 1
        assert stages[0][0] == "llm"

    def test_turn_timer_multiple_stages_in_order(self):
        """Multiple stages are recorded in the order they were entered."""
        tt = TurnTimer()
        with tt.stage("stt"):
            pass
        with tt.stage("llm"):
            pass

        names = [name for name, _ in tt.stages()]
        assert names == ["stt", "llm"]

    @patch("core.timing.logger")
    def test_turn_timer_log_summary_emits_info_line(self, mock_logger):
        """log_summary() emits an INFO line containing [TURN], stage name, and total."""
        tt = TurnTimer()
        with tt.stage("llm"):
            pass
        tt.log_summary()

        mock_logger.info.assert_called_once()
        log_msg = mock_logger.info.call_args[0][0]
        assert "[TURN]" in log_msg
        assert "llm=" in log_msg
        assert "total=" in log_msg

    @patch("core.timing.logger")
    @patch("time.perf_counter")
    def test_turn_timer_debug_warning_on_slow_turn(self, mock_perf, mock_logger):
        """Debug warning is emitted when total turn time exceeds 2500ms budget."""
        # We need perf_counter to return values that make total > 2500ms
        # TurnTimer.__init__ calls perf_counter once (start)
        # log_summary calls perf_counter once (end)
        # We set start=0.0, end=3.0 (3000ms total)
        mock_perf.side_effect = [0.0, 3.0]

        tt = TurnTimer()
        # Manually inject a stage so the warning branch is triggered
        tt._stages = [("llm", 3000.0)]
        tt.log_summary()

        # Should have both an info call and a debug call
        assert mock_logger.info.called
        assert mock_logger.debug.called
        debug_msg = mock_logger.debug.call_args[0][0]
        assert "2500ms" in debug_msg or "budget" in debug_msg
