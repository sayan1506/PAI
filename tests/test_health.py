"""
Unit tests for the PAI startup health checker (core/health.py).

Tests verify:
- All components report OK with valid config
- Critical failure on missing LLM key
- Memory DB missing on unwritable path
- Text-only mode when no microphone detected
- TTS degraded when kokoro missing but pyttsx3 present
- TTS missing when both engines absent
- Browser degraded on missing binary
- Browser OK when binary override set
- Summary contains expected component names
- Health check skipped when SKIP_HEALTH_CHECK is True

All external dependencies are mocked — no real hardware required.
"""

import sys
from unittest.mock import patch, MagicMock

import pytest

import config
from core.health import (
    run_health_check,
    ComponentStatus,
)


class TestHealthCheckOK:
    """Test that a fully valid config produces a healthy report."""

    def test_health_check_ok_with_valid_config(self, monkeypatch):
        """Mock all checks to produce OK components — no critical failure, no text_only_mode."""
        monkeypatch.setattr(config, "SKIP_HEALTH_CHECK", False)
        monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
        monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key-123")
        monkeypatch.setattr(config, "MEMORY_DB_PATH", "~/.pai/test_memory.db")
        monkeypatch.setattr(config, "TTS_ENGINE", "kokoro")
        monkeypatch.setattr(config, "BROWSER_BINARY", "")
        monkeypatch.setattr(config, "BROWSER_APP", "chrome")

        # Mock sounddevice to return a device with input channels
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [{"max_input_channels": 2, "name": "Test Mic"}]
        monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)

        # Mock faster_whisper as importable
        mock_fw = MagicMock()
        monkeypatch.setitem(sys.modules, "faster_whisper", mock_fw)

        # Mock kokoro as importable
        mock_kokoro = MagicMock()
        monkeypatch.setitem(sys.modules, "kokoro", mock_kokoro)

        # Mock shutil.which to find a browser
        with patch("shutil.which", return_value="/usr/bin/google-chrome"):
            report = run_health_check()

        assert report.critical_failure is False
        assert report.text_only_mode is False


class TestHealthCriticalFailure:
    """Test critical failure detection."""

    def test_health_critical_failure_on_missing_llm_key(self, monkeypatch):
        """Missing GEMINI_API_KEY with gemini provider triggers critical_failure."""
        monkeypatch.setattr(config, "SKIP_HEALTH_CHECK", False)
        monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
        monkeypatch.setattr(config, "GEMINI_API_KEY", "")
        monkeypatch.setattr(config, "MEMORY_DB_PATH", "~/.pai/test_memory.db")
        monkeypatch.setattr(config, "TTS_ENGINE", "kokoro")
        monkeypatch.setattr(config, "BROWSER_BINARY", "")
        monkeypatch.setattr(config, "BROWSER_APP", "chrome")

        # Mock other dependencies to avoid side effects
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [{"max_input_channels": 1, "name": "Mic"}]
        monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)
        mock_fw = MagicMock()
        monkeypatch.setitem(sys.modules, "faster_whisper", mock_fw)
        mock_kokoro = MagicMock()
        monkeypatch.setitem(sys.modules, "kokoro", mock_kokoro)

        with patch("shutil.which", return_value="/usr/bin/chrome"):
            report = run_health_check()

        assert report.critical_failure is True


class TestHealthMemoryDB:
    """Test memory DB component checks."""

    def test_health_memory_db_missing_on_unwritable_path(self, monkeypatch):
        """Invalid/unwritable MEMORY_DB_PATH results in memory_db MISSING."""
        monkeypatch.setattr(config, "SKIP_HEALTH_CHECK", False)
        monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
        monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
        # Use a path that will fail — null bytes are invalid in paths
        monkeypatch.setattr(config, "MEMORY_DB_PATH", "/nonexistent\x00/invalid/path/memory.db")
        monkeypatch.setattr(config, "TTS_ENGINE", "kokoro")
        monkeypatch.setattr(config, "BROWSER_BINARY", "")
        monkeypatch.setattr(config, "BROWSER_APP", "chrome")

        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [{"max_input_channels": 1, "name": "Mic"}]
        monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)
        mock_fw = MagicMock()
        monkeypatch.setitem(sys.modules, "faster_whisper", mock_fw)
        mock_kokoro = MagicMock()
        monkeypatch.setitem(sys.modules, "kokoro", mock_kokoro)

        with patch("shutil.which", return_value="/usr/bin/chrome"):
            report = run_health_check()

        memory_component = next(c for c in report.components if c.name == "memory_db")
        assert memory_component.status == ComponentStatus.MISSING


class TestHealthMicrophone:
    """Test microphone detection and text_only_mode."""

    def test_health_text_only_mode_on_no_microphone(self, monkeypatch):
        """No input devices detected triggers text_only_mode."""
        monkeypatch.setattr(config, "SKIP_HEALTH_CHECK", False)
        monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
        monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(config, "MEMORY_DB_PATH", "~/.pai/test_memory.db")
        monkeypatch.setattr(config, "TTS_ENGINE", "kokoro")
        monkeypatch.setattr(config, "BROWSER_BINARY", "")
        monkeypatch.setattr(config, "BROWSER_APP", "chrome")

        # Mock sounddevice to return empty list (no devices)
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = []
        monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)

        mock_fw = MagicMock()
        monkeypatch.setitem(sys.modules, "faster_whisper", mock_fw)
        mock_kokoro = MagicMock()
        monkeypatch.setitem(sys.modules, "kokoro", mock_kokoro)

        with patch("shutil.which", return_value="/usr/bin/chrome"):
            report = run_health_check()

        assert report.text_only_mode is True
        mic_component = next(c for c in report.components if c.name == "microphone")
        assert mic_component.status == ComponentStatus.MISSING


class TestHealthTTS:
    """Test TTS engine detection."""

    def test_health_tts_degraded_when_kokoro_missing_pyttsx3_present(self, monkeypatch):
        """kokoro import fails but pyttsx3 succeeds → tts_engine DEGRADED."""
        monkeypatch.setattr(config, "SKIP_HEALTH_CHECK", False)
        monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
        monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(config, "MEMORY_DB_PATH", "~/.pai/test_memory.db")
        monkeypatch.setattr(config, "TTS_ENGINE", "kokoro")
        monkeypatch.setattr(config, "BROWSER_BINARY", "")
        monkeypatch.setattr(config, "BROWSER_APP", "chrome")

        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [{"max_input_channels": 1, "name": "Mic"}]
        monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)
        mock_fw = MagicMock()
        monkeypatch.setitem(sys.modules, "faster_whisper", mock_fw)

        # kokoro fails to import, pyttsx3 succeeds
        monkeypatch.setitem(sys.modules, "kokoro", None)  # will cause ImportError on import
        mock_pyttsx3 = MagicMock()
        monkeypatch.setitem(sys.modules, "pyttsx3", mock_pyttsx3)

        with patch("shutil.which", return_value="/usr/bin/chrome"):
            with patch("builtins.__import__", wraps=_import_side_effect(
                block={"kokoro"}, allow={"pyttsx3": MagicMock()}
            )):
                report = run_health_check()

        tts_component = next(c for c in report.components if c.name == "tts_engine")
        assert tts_component.status == ComponentStatus.DEGRADED

    def test_health_tts_missing_when_both_engines_absent(self, monkeypatch):
        """Both kokoro and pyttsx3 fail to import → tts_engine MISSING."""
        monkeypatch.setattr(config, "SKIP_HEALTH_CHECK", False)
        monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
        monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(config, "MEMORY_DB_PATH", "~/.pai/test_memory.db")
        monkeypatch.setattr(config, "TTS_ENGINE", "kokoro")
        monkeypatch.setattr(config, "BROWSER_BINARY", "")
        monkeypatch.setattr(config, "BROWSER_APP", "chrome")

        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [{"max_input_channels": 1, "name": "Mic"}]
        monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)
        mock_fw = MagicMock()
        monkeypatch.setitem(sys.modules, "faster_whisper", mock_fw)

        with patch("shutil.which", return_value="/usr/bin/chrome"):
            with patch("builtins.__import__", wraps=_import_side_effect(
                block={"kokoro", "pyttsx3"}, allow={}
            )):
                report = run_health_check()

        tts_component = next(c for c in report.components if c.name == "tts_engine")
        assert tts_component.status == ComponentStatus.MISSING


class TestHealthBrowser:
    """Test browser detection."""

    def test_health_browser_degraded_on_missing_binary(self, monkeypatch):
        """No browser found on PATH and no override → browser DEGRADED, not critical."""
        monkeypatch.setattr(config, "SKIP_HEALTH_CHECK", False)
        monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
        monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(config, "MEMORY_DB_PATH", "~/.pai/test_memory.db")
        monkeypatch.setattr(config, "TTS_ENGINE", "kokoro")
        monkeypatch.setattr(config, "BROWSER_BINARY", "")
        monkeypatch.setattr(config, "BROWSER_APP", "chrome")

        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [{"max_input_channels": 1, "name": "Mic"}]
        monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)
        mock_fw = MagicMock()
        monkeypatch.setitem(sys.modules, "faster_whisper", mock_fw)
        mock_kokoro = MagicMock()
        monkeypatch.setitem(sys.modules, "kokoro", mock_kokoro)

        # shutil.which returns None for all candidates
        with patch("shutil.which", return_value=None):
            report = run_health_check()

        browser_component = next(c for c in report.components if c.name == "browser")
        assert browser_component.status == ComponentStatus.DEGRADED
        assert report.critical_failure is False

    def test_health_browser_ok_when_binary_override_set(self, monkeypatch):
        """BROWSER_BINARY set to a valid file → browser OK."""
        monkeypatch.setattr(config, "SKIP_HEALTH_CHECK", False)
        monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
        monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(config, "MEMORY_DB_PATH", "~/.pai/test_memory.db")
        monkeypatch.setattr(config, "TTS_ENGINE", "kokoro")
        monkeypatch.setattr(config, "BROWSER_BINARY", "/usr/bin/google-chrome")
        monkeypatch.setattr(config, "BROWSER_APP", "chrome")

        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [{"max_input_channels": 1, "name": "Mic"}]
        monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)
        mock_fw = MagicMock()
        monkeypatch.setitem(sys.modules, "faster_whisper", mock_fw)
        mock_kokoro = MagicMock()
        monkeypatch.setitem(sys.modules, "kokoro", mock_kokoro)

        with patch("os.path.isfile", return_value=True):
            report = run_health_check()

        browser_component = next(c for c in report.components if c.name == "browser")
        assert browser_component.status == ComponentStatus.OK


class TestHealthSummary:
    """Test summary output content."""

    def test_health_summary_contains_all_components(self, monkeypatch):
        """Summary string contains header and key component names."""
        monkeypatch.setattr(config, "SKIP_HEALTH_CHECK", False)
        monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
        monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(config, "MEMORY_DB_PATH", "~/.pai/test_memory.db")
        monkeypatch.setattr(config, "TTS_ENGINE", "kokoro")
        monkeypatch.setattr(config, "BROWSER_BINARY", "")
        monkeypatch.setattr(config, "BROWSER_APP", "chrome")

        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [{"max_input_channels": 1, "name": "Mic"}]
        monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)
        mock_fw = MagicMock()
        monkeypatch.setitem(sys.modules, "faster_whisper", mock_fw)
        mock_kokoro = MagicMock()
        monkeypatch.setitem(sys.modules, "kokoro", mock_kokoro)

        with patch("shutil.which", return_value="/usr/bin/chrome"):
            report = run_health_check()

        summary = report.summary()
        assert "PAI Component Health Check:" in summary
        assert "llm_provider" in summary
        assert "memory_db" in summary


class TestHealthSkip:
    """Test skip configuration."""

    def test_health_skipped_when_skip_config_true(self, monkeypatch):
        """SKIP_HEALTH_CHECK=True → empty components, no critical failure."""
        monkeypatch.setattr(config, "SKIP_HEALTH_CHECK", True)

        report = run_health_check()

        assert report.components == []
        assert report.critical_failure is False


# ─── Helper for mocking imports ───────────────────────────────────────────────

def _import_side_effect(block: set, allow: dict):
    """
    Return a custom __import__ function that raises ImportError for modules
    in `block` and returns mocks for modules in `allow`.
    Falls through to the real __import__ for everything else.
    """
    real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def custom_import(name, *args, **kwargs):
        if name in block:
            raise ImportError(f"Mocked: {name} not installed")
        if name in allow:
            return allow[name]
        return real_import(name, *args, **kwargs)

    return custom_import
