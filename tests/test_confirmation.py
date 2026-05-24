"""
Unit tests for core/confirmation.py — destructive action confirmation gate.

Tests verify:
- needs_file_confirmation() correctly identifies destructive file ops
- is_risky_terminal_command() detects risky shell commands
- request_confirmation() handles user input and voice mode auto-deny
"""

from unittest.mock import patch

import pytest

import config
from core.confirmation import (
    is_risky_terminal_command,
    needs_file_confirmation,
    request_confirmation,
)


class TestNeedsFileConfirmation:
    """Tests for needs_file_confirmation()."""

    def test_needs_file_confirmation_true_for_delete(self, monkeypatch):
        """Delete operations require confirmation when CONFIRM_DESTRUCTIVE is True."""
        monkeypatch.setattr(config, "CONFIRM_DESTRUCTIVE", True)
        assert needs_file_confirmation("delete") is True

    def test_needs_file_confirmation_true_for_move(self, monkeypatch):
        """Move operations require confirmation when CONFIRM_DESTRUCTIVE is True."""
        monkeypatch.setattr(config, "CONFIRM_DESTRUCTIVE", True)
        assert needs_file_confirmation("move") is True

    def test_needs_file_confirmation_false_for_create(self, monkeypatch):
        """Create operations never require confirmation."""
        monkeypatch.setattr(config, "CONFIRM_DESTRUCTIVE", True)
        assert needs_file_confirmation("create") is False

    def test_needs_file_confirmation_false_when_config_off(self, monkeypatch):
        """No confirmation needed when CONFIRM_DESTRUCTIVE is False."""
        monkeypatch.setattr(config, "CONFIRM_DESTRUCTIVE", False)
        assert needs_file_confirmation("delete") is False


class TestIsRiskyTerminalCommand:
    """Tests for is_risky_terminal_command()."""

    def test_is_risky_terminal_command_detects_rm(self):
        """rm commands are detected as risky."""
        assert is_risky_terminal_command("rm old_logs.txt") is True

    def test_is_risky_terminal_command_detects_del(self):
        """Windows del commands are detected as risky."""
        assert is_risky_terminal_command("del /q important.docx") is True

    def test_is_risky_terminal_command_detects_taskkill(self):
        """taskkill commands are detected as risky."""
        assert is_risky_terminal_command("taskkill /F /IM python.exe") is True

    def test_is_risky_terminal_command_false_for_safe_command(self):
        """Safe commands are not flagged as risky."""
        assert is_risky_terminal_command("python --version") is False
        assert is_risky_terminal_command("ls -la") is False
        assert is_risky_terminal_command("Get-Volume") is False


class TestRequestConfirmation:
    """Tests for request_confirmation()."""

    @patch("core.confirmation._is_interactive_tty", return_value=True)
    @patch("builtins.input", return_value="y")
    def test_request_confirmation_returns_true_on_yes(
        self, mock_input, mock_tty, monkeypatch
    ):
        """User typing 'y' confirms the action."""
        monkeypatch.setattr(config, "CONFIRM_DESTRUCTIVE", True)
        monkeypatch.setattr(config, "VOICE_ENABLED", False)
        assert request_confirmation("Delete test.txt") is True

    @patch("core.confirmation._is_interactive_tty", return_value=True)
    @patch("builtins.input", return_value="n")
    def test_request_confirmation_returns_false_on_no(
        self, mock_input, mock_tty, monkeypatch
    ):
        """User typing 'n' denies the action."""
        monkeypatch.setattr(config, "CONFIRM_DESTRUCTIVE", True)
        monkeypatch.setattr(config, "VOICE_ENABLED", False)
        assert request_confirmation("Delete test.txt") is False

    @patch("core.confirmation._is_interactive_tty", return_value=True)
    @patch("builtins.input", return_value="")
    def test_request_confirmation_returns_false_on_empty_input(
        self, mock_input, mock_tty, monkeypatch
    ):
        """Empty input (just pressing Enter) denies the action."""
        monkeypatch.setattr(config, "CONFIRM_DESTRUCTIVE", True)
        monkeypatch.setattr(config, "VOICE_ENABLED", False)
        assert request_confirmation("Delete test.txt") is False

    @patch("core.confirmation._is_interactive_tty", return_value=False)
    @patch("builtins.input")
    def test_request_confirmation_auto_denies_in_voice_mode(
        self, mock_input, mock_tty, monkeypatch
    ):
        """Voice mode auto-denies without calling input()."""
        monkeypatch.setattr(config, "CONFIRM_DESTRUCTIVE", True)
        monkeypatch.setattr(config, "VOICE_ENABLED", True)
        assert request_confirmation("Delete test.txt") is False
        mock_input.assert_not_called()
