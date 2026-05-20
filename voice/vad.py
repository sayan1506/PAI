"""
voice/vad.py

Silero VAD wrapper. Detects whether a given audio chunk contains speech.
Used to determine when the user has started and stopped speaking.
"""

import numpy as np
import torch
from core.logger import logger
from core.exceptions import AudioError
import config


class VoiceActivityDetector:
    """
    Wraps Silero VAD for real-time speech detection.

    Usage:
        vad = VoiceActivityDetector()
        vad.load()
        is_speech = vad.is_speech(audio_chunk)
    """

    SAMPLE_RATE = 16000  # Silero VAD requires 16kHz

    def __init__(self):
        self._model = None
        self._utils = None

    def load(self):
        """Download (first time) and load the Silero VAD model."""
        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            self._utils = utils
            logger.info("Silero VAD loaded")
        except Exception as e:
            raise AudioError(f"Failed to load Silero VAD: {e}")

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Return True if the chunk contains speech above threshold.

        Args:
            audio_chunk: float32 numpy array at 16kHz sample rate.

        Returns:
            True if speech detected, False if silence.
        """
        if self._model is None:
            raise AudioError("VAD not loaded. Call load() first.")
        tensor = torch.FloatTensor(audio_chunk)
        confidence = self._model(tensor, self.SAMPLE_RATE).item()
        return confidence > 0.5

    def reset(self):
        """Reset VAD internal state between utterances."""
        if self._model:
            self._model.reset_states()
