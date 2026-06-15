"""Silero voice activity detection wrapper.

Provides :class:`VoiceActivityDetector`, which loads the Silero VAD
model via ``torch.hub`` and classifies individual audio chunks as speech
or silence. Used by the utterance listener to decide when the user has
started and stopped speaking. The model is downloaded and cached on
first load and expects 16 kHz float32 audio.
"""

import numpy as np
import torch
from core.logger import logger
from core.exceptions import AudioError
import config


class VoiceActivityDetector:
    """Wraps Silero VAD for real-time speech detection.

    Holds the loaded Silero model and its utility callables. The model
    is stateful across calls, so :meth:`reset` should be invoked between
    distinct utterances to clear carried-over context.

    Attributes:
        SAMPLE_RATE: Required input sample rate in Hz (Silero needs 16k).

    Usage:
        vad = VoiceActivityDetector()
        vad.load()
        is_speech = vad.is_speech(audio_chunk)
    """

    SAMPLE_RATE = 16000  # Silero VAD requires 16kHz

    def __init__(self):
        """Initialise with no model loaded; call :meth:`load` first."""
        self._model = None
        self._utils = None

    def load(self):
        """Download (first time) and load the Silero VAD model.

        Side effects:
            Fetches the model from ``torch.hub`` (cached after the first
            call) and stores the model and its utilities on the instance.

        Raises:
            AudioError: If the model cannot be downloaded or loaded.
        """
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
        """Return True if the chunk contains speech above threshold.

        Runs the Silero model on the chunk and compares its confidence
        score against a fixed 0.5 threshold.

        Args:
            audio_chunk: float32 numpy array at 16kHz sample rate.

        Returns:
            True if speech is detected, False if the chunk is silence.

        Raises:
            AudioError: If the model has not been loaded.
        """
        if self._model is None:
            raise AudioError("VAD not loaded. Call load() first.")
        tensor = torch.FloatTensor(audio_chunk)
        confidence = self._model(tensor, self.SAMPLE_RATE).item()
        return confidence > 0.5

    def reset(self):
        """Reset the model's internal state between utterances.

        Clears the recurrent state Silero carries across calls so a new
        utterance is not influenced by the previous one. No-op if the
        model is not loaded.
        """
        if self._model:
            self._model.reset_states()
