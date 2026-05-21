"""
tests/voice/test_tts.py

Unit tests for the TTS module (voice/tts.py).
All external dependencies (kokoro, pyttsx3, sounddevice) are mocked —
no real audio packages needed.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock heavy third-party modules that may not be installed in test env
for mod_name in [
    "openwakeword", "openwakeword.model",
    "torch", "faster_whisper",
    "sounddevice",
    "kokoro",
    "pyttsx3",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from voice.tts import KokoroTTS, Pyttsx3TTS, get_tts


# ── get_tts() factory ─────────────────────────────────────────────────────────


class TestGetTtsFactory:
    """Tests for the get_tts() factory function."""

    def test_returns_none_when_tts_disabled(self):
        """get_tts() returns None when TTS_ENABLED=false."""
        with patch("voice.tts.config") as mock_config:
            mock_config.TTS_ENABLED = False

            result = get_tts()
            assert result is None

    def test_returns_kokoro_when_engine_kokoro_and_load_succeeds(self):
        """get_tts() returns KokoroTTS when TTS_ENGINE=kokoro and load succeeds."""
        with patch("voice.tts.config") as mock_config, \
             patch("voice.tts.KokoroTTS") as mock_kokoro_cls:
            mock_config.TTS_ENABLED = True
            mock_config.TTS_ENGINE = "kokoro"
            mock_instance = MagicMock()
            mock_kokoro_cls.return_value = mock_instance

            result = get_tts()
            assert result is mock_instance
            mock_instance.load.assert_called_once()

    def test_falls_back_to_pyttsx3_when_kokoro_fails(self):
        """get_tts() falls back to Pyttsx3TTS when kokoro fails to load."""
        with patch("voice.tts.config") as mock_config, \
             patch("voice.tts.KokoroTTS") as mock_kokoro_cls, \
             patch("voice.tts.Pyttsx3TTS") as mock_pyttsx3_cls:
            mock_config.TTS_ENABLED = True
            mock_config.TTS_ENGINE = "kokoro"

            # KokoroTTS.load() raises
            mock_kokoro_instance = MagicMock()
            mock_kokoro_instance.load.side_effect = ImportError("kokoro not installed")
            mock_kokoro_cls.return_value = mock_kokoro_instance

            # Pyttsx3TTS succeeds
            mock_pyttsx3_instance = MagicMock()
            mock_pyttsx3_cls.return_value = mock_pyttsx3_instance

            result = get_tts()
            assert result is mock_pyttsx3_instance
            mock_pyttsx3_instance.load.assert_called_once()

    def test_returns_pyttsx3_when_engine_pyttsx3(self):
        """get_tts() returns Pyttsx3TTS when TTS_ENGINE=pyttsx3."""
        with patch("voice.tts.config") as mock_config, \
             patch("voice.tts.Pyttsx3TTS") as mock_pyttsx3_cls:
            mock_config.TTS_ENABLED = True
            mock_config.TTS_ENGINE = "pyttsx3"

            mock_pyttsx3_instance = MagicMock()
            mock_pyttsx3_cls.return_value = mock_pyttsx3_instance

            result = get_tts()
            assert result is mock_pyttsx3_instance
            mock_pyttsx3_instance.load.assert_called_once()

    def test_returns_none_when_all_engines_fail(self):
        """get_tts() returns None when all engines fail to load."""
        with patch("voice.tts.config") as mock_config, \
             patch("voice.tts.KokoroTTS") as mock_kokoro_cls, \
             patch("voice.tts.Pyttsx3TTS") as mock_pyttsx3_cls:
            mock_config.TTS_ENABLED = True
            mock_config.TTS_ENGINE = "kokoro"

            # Both engines fail
            mock_kokoro_instance = MagicMock()
            mock_kokoro_instance.load.side_effect = ImportError("kokoro not installed")
            mock_kokoro_cls.return_value = mock_kokoro_instance

            mock_pyttsx3_instance = MagicMock()
            mock_pyttsx3_instance.load.side_effect = RuntimeError("pyttsx3 broken")
            mock_pyttsx3_cls.return_value = mock_pyttsx3_instance

            result = get_tts()
            assert result is None


# ── KokoroTTS ─────────────────────────────────────────────────────────────────


class TestKokoroTTS:
    """Tests for the KokoroTTS class."""

    def test_speak_plays_audio_chunks(self):
        """speak() plays audio chunks from the pipeline generator."""
        with patch("voice.tts.sd") as mock_sd, \
             patch("voice.tts.config") as mock_config:
            mock_config.TTS_VOICE = "af_heart"
            mock_config.TTS_SPEED = 1.0

            tts = KokoroTTS()

            # Set up a mock pipeline that yields two audio chunks
            chunk1 = np.ones(1000, dtype=np.float32)
            chunk2 = np.ones(500, dtype=np.float32)
            result1 = MagicMock()
            result1.audio = chunk1
            result2 = MagicMock()
            result2.audio = chunk2

            mock_pipeline = MagicMock()
            mock_pipeline.return_value = iter([result1, result2])
            tts._pipeline = mock_pipeline

            tts.speak("hello world")

            # sd.play should have been called twice (once per chunk)
            assert mock_sd.play.call_count == 2
            assert mock_sd.wait.call_count == 2

    def test_stop_increments_epoch_and_calls_sd_stop(self):
        """stop() increments epoch and calls sd.stop()."""
        with patch("voice.tts.sd") as mock_sd:
            tts = KokoroTTS()
            initial_epoch = tts._epoch

            tts.stop()

            assert tts._epoch == initial_epoch + 1
            mock_sd.stop.assert_called_once()

    def test_speak_exits_early_when_epoch_changes(self):
        """speak() exits early when epoch changes (barge-in simulation)."""
        with patch("voice.tts.sd") as mock_sd, \
             patch("voice.tts.config") as mock_config:
            mock_config.TTS_VOICE = "af_heart"
            mock_config.TTS_SPEED = 1.0

            tts = KokoroTTS()

            # Create chunks — the second one should NOT be played
            chunk1 = np.ones(1000, dtype=np.float32)
            chunk2 = np.ones(500, dtype=np.float32)
            result1 = MagicMock()
            result1.audio = chunk1
            result2 = MagicMock()
            result2.audio = chunk2

            # After playing the first chunk, simulate barge-in by incrementing epoch
            def increment_epoch_on_wait():
                tts._epoch += 1

            mock_sd.wait.side_effect = increment_epoch_on_wait

            mock_pipeline = MagicMock()
            mock_pipeline.return_value = iter([result1, result2])
            tts._pipeline = mock_pipeline

            tts.speak("hello world")

            # Only the first chunk should have been played
            assert mock_sd.play.call_count == 1

    def test_is_speaking_true_during_speak_false_after(self):
        """is_speaking is True during speak() and False after."""
        with patch("voice.tts.sd") as mock_sd, \
             patch("voice.tts.config") as mock_config:
            mock_config.TTS_VOICE = "af_heart"
            mock_config.TTS_SPEED = 1.0

            tts = KokoroTTS()

            # Track is_speaking state during playback
            speaking_during_play = []

            chunk = np.ones(1000, dtype=np.float32)
            result = MagicMock()
            result.audio = chunk

            def capture_speaking():
                speaking_during_play.append(tts.is_speaking)

            mock_sd.wait.side_effect = capture_speaking

            mock_pipeline = MagicMock()
            mock_pipeline.return_value = iter([result])
            tts._pipeline = mock_pipeline

            assert tts.is_speaking is False
            tts.speak("hello")
            assert speaking_during_play == [True]
            assert tts.is_speaking is False


# ── Pyttsx3TTS ────────────────────────────────────────────────────────────────


class TestPyttsx3TTS:
    """Tests for the Pyttsx3TTS class."""

    def test_speak_calls_engine_say_and_run_and_wait(self):
        """speak() calls engine.say() and engine.runAndWait()."""
        tts = Pyttsx3TTS()
        mock_engine = MagicMock()
        tts._engine = mock_engine

        tts.speak("hello world")

        mock_engine.say.assert_called_once_with("hello world")
        mock_engine.runAndWait.assert_called_once()

    def test_load_logs_warning_when_barge_in_enabled(self):
        """load() logs warning when BARGE_IN_ENABLED=true."""
        with patch("voice.tts.config") as mock_config, \
             patch("voice.tts.logger") as mock_logger:
            mock_config.TTS_SPEED = 1.0
            mock_config.BARGE_IN_ENABLED = True

            # Mock pyttsx3 import inside load()
            mock_pyttsx3 = MagicMock()
            with patch.dict(sys.modules, {"pyttsx3": mock_pyttsx3}):
                tts = Pyttsx3TTS()
                tts.load()

                # Check that a warning was logged about barge-in
                mock_logger.warning.assert_called_once()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "Barge-in" in warning_msg or "barge-in" in warning_msg.lower()

    def test_stop_is_noop(self):
        """stop() is a no-op (does not raise or change state)."""
        tts = Pyttsx3TTS()
        # Should not raise
        tts.stop()
