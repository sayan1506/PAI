"""
voice/stt.py

Faster-Whisper speech-to-text wrapper.
Accepts a complete audio buffer and returns the transcribed string.
"""

import numpy as np
from faster_whisper import WhisperModel
from core.logger import logger
from core.exceptions import AudioError
import config


class SpeechToText:
    """
    Wraps Faster-Whisper for local, offline transcription.

    The model is downloaded on first use and cached locally.
    Supported sizes: tiny, base, small, medium, large-v3
    """

    def __init__(self):
        self._model = None
        self.model_size = config.STT_SIZE

    def load(self):
        """Load the Whisper model (downloads on first use).

        Raises:
            AudioError: If the model fails to load.
        """
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
            )
            logger.info("Whisper STT ready")
        except Exception as e:
            raise AudioError(f"Failed to load Whisper model '{self.model_size}': {e}")

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a complete audio buffer to text.

        Args:
            audio: float32 numpy array at 16kHz.

        Returns:
            Transcribed text string. Empty string if nothing detected.

        Raises:
            AudioError: If model not loaded or transcription fails.
        """
        if self._model is None:
            raise AudioError("STT not loaded. Call load() first.")
        try:
            segments, _ = self._model.transcribe(
                audio,
                language="en",
                vad_filter=True,
                beam_size=5,
            )
            text = " ".join(seg.text for seg in segments).strip()
            logger.info(f"Transcribed: '{text}'")
            return text
        except Exception as e:
            raise AudioError(f"Transcription failed: {e}")
