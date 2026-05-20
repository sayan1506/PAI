"""
tests/test_voice.py

Unit tests for the Phase 2 voice input pipeline.
All hardware (mic, models) is mocked — no real audio device needed.
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
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from voice.mic import MicCapture
from core.exceptions import AudioError


# ── MicCapture ────────────────────────────────────────────────────────────────

class TestMicCapture:
    def test_start_opens_stream(self):
        with patch("sounddevice.InputStream") as mock_stream:
            mic = MicCapture()
            mic.start()
            mock_stream.assert_called_once()

    def test_read_returns_flat_array(self):
        with patch("sounddevice.InputStream") as mock_stream:
            mock_instance = MagicMock()
            mock_instance.read.return_value = (np.zeros((512, 1), dtype="float32"), None)
            mock_stream.return_value = mock_instance
            mic = MicCapture()
            mic.start()
            data = mic.read()
            assert data.ndim == 1
            assert len(data) == 512

    def test_stop_closes_stream(self):
        with patch("sounddevice.InputStream") as mock_stream:
            mock_instance = MagicMock()
            mock_stream.return_value = mock_instance
            mic = MicCapture()
            mic.start()
            mic.stop()
            mock_instance.stop.assert_called_once()
            mock_instance.close.assert_called_once()

    def test_read_before_start_raises_audio_error(self):
        mic = MicCapture()
        with pytest.raises(AudioError):
            mic.read()

    def test_context_manager(self):
        with patch("sounddevice.InputStream"):
            with MicCapture() as mic:
                assert mic._stream is not None


# ── VoiceActivityDetector ─────────────────────────────────────────────────────

class TestVAD:
    def test_is_speech_returns_true_above_threshold(self):
        with patch("torch.hub.load") as mock_load:
            mock_model = MagicMock()
            mock_model.return_value = MagicMock(item=lambda: 0.9)
            mock_load.return_value = (mock_model, MagicMock())
            from voice.vad import VoiceActivityDetector
            vad = VoiceActivityDetector()
            vad.load()
            chunk = np.random.randn(512).astype(np.float32)
            assert vad.is_speech(chunk) is True

    def test_is_speech_returns_false_below_threshold(self):
        with patch("torch.hub.load") as mock_load:
            mock_model = MagicMock()
            mock_model.return_value = MagicMock(item=lambda: 0.1)
            mock_load.return_value = (mock_model, MagicMock())
            from voice.vad import VoiceActivityDetector
            vad = VoiceActivityDetector()
            vad.load()
            chunk = np.zeros(512, dtype=np.float32)
            assert vad.is_speech(chunk) is False

    def test_is_speech_before_load_raises_audio_error(self):
        from voice.vad import VoiceActivityDetector
        vad = VoiceActivityDetector()
        with pytest.raises(AudioError):
            vad.is_speech(np.zeros(512, dtype=np.float32))


# ── SpeechToText ──────────────────────────────────────────────────────────────

class TestSTT:
    def test_transcribe_returns_text(self):
        with patch("voice.stt.WhisperModel") as mock_model_cls:
            mock_seg = MagicMock()
            mock_seg.text = " hello world"
            mock_model = MagicMock()
            mock_model.transcribe.return_value = ([mock_seg], MagicMock())
            mock_model_cls.return_value = mock_model
            from voice.stt import SpeechToText
            stt = SpeechToText()
            stt.load()
            result = stt.transcribe(np.zeros(16000, dtype=np.float32))
            assert result == "hello world"

    def test_transcribe_before_load_raises_audio_error(self):
        from voice.stt import SpeechToText
        stt = SpeechToText()
        with pytest.raises(AudioError):
            stt.transcribe(np.zeros(16000, dtype=np.float32))

    def test_transcribe_empty_segments_returns_empty_string(self):
        with patch("voice.stt.WhisperModel") as mock_model_cls:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = ([], MagicMock())
            mock_model_cls.return_value = mock_model
            from voice.stt import SpeechToText
            stt = SpeechToText()
            stt.load()
            result = stt.transcribe(np.zeros(16000, dtype=np.float32))
            assert result == ""


# ── UtteranceListener ─────────────────────────────────────────────────────────

class TestUtteranceListener:
    def test_collects_audio_until_silence(self):
        mock_vad = MagicMock()
        # Need enough speech chunks to meet minimum (16) and enough silence (1200ms / 32ms = 37)
        speech_chunks = 20
        silence_chunks_needed = 38  # 1200ms silence at 32ms/chunk
        mock_vad.is_speech.side_effect = (
            [True] * speech_chunks + [False] * silence_chunks_needed
        )
        chunk = np.ones(512, dtype=np.float32)
        mock_mic = MagicMock()
        mock_mic.read.return_value = chunk

        from voice.listener import UtteranceListener
        listener = UtteranceListener(mock_vad)
        result = listener.listen(mock_mic)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_discards_leading_silence(self):
        mock_vad = MagicMock()
        speech_chunks = 20
        # Use enough silence to exceed any reasonable threshold
        silence_chunks_needed = 50
        # Lots of silence before speech starts
        mock_vad.is_speech.side_effect = (
            [False] * 10 + [True] * speech_chunks + [False] * silence_chunks_needed
        )
        chunk = np.ones(512, dtype=np.float32)
        mock_mic = MagicMock()
        mock_mic.read.return_value = chunk

        from voice.listener import UtteranceListener
        listener = UtteranceListener(mock_vad)
        result = listener.listen(mock_mic)

        # Leading silence (10 chunks) should NOT be in the buffer
        # Buffer should contain speech + some trailing silence (up to threshold)
        # but NOT the 10 leading silence chunks
        assert len(result) < (10 + speech_chunks + silence_chunks_needed) * 512
        assert len(result) >= speech_chunks * 512



# ── VoicePipeline ─────────────────────────────────────────────────────────────

class TestVoicePipeline:
    def test_listen_once_returns_transcription(self):
        with patch("voice.pipeline.MicCapture") as mock_mic_cls, \
             patch("voice.pipeline.WakeWordDetector") as mock_ww_cls, \
             patch("voice.pipeline.VoiceActivityDetector") as mock_vad_cls, \
             patch("voice.pipeline.UtteranceListener") as mock_listener_cls, \
             patch("voice.pipeline.SpeechToText") as mock_stt_cls:

            mock_stt = MagicMock()
            mock_stt.transcribe.return_value = "open notepad"
            mock_stt_cls.return_value = mock_stt

            mock_listener = MagicMock()
            mock_listener.listen.return_value = np.zeros(16000, dtype=np.float32)
            mock_listener_cls.return_value = mock_listener

            mock_mic = MagicMock()
            mock_mic_cls.return_value = mock_mic
            mock_mic.__enter__ = MagicMock(return_value=mock_mic)
            mock_mic.__exit__ = MagicMock(return_value=False)

            from voice.pipeline import VoicePipeline
            pipeline = VoicePipeline()
            pipeline.load()
            result = pipeline.listen_once()

            assert result == "open notepad"
