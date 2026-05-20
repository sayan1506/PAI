"""
voice/wake_word.py

OpenWakeWord-based wake word detector.
Listens continuously and fires when the configured wake word is detected.
"""

import numpy as np
from openwakeword.model import Model
from core.logger import logger
from core.exceptions import AudioError
import config


class WakeWordDetector:
    """
    Listens for the wake word using OpenWakeWord.

    Supported built-in wake words (no training required):
        "hey jarvis", "hey mycroft", "alexa", "ok google", and others.

    Usage:
        detector = WakeWordDetector()
        detector.load()
        detector.wait_for_wake_word(mic)   # blocks until triggered
    """

    CHUNK_SIZE = 1280   # OpenWakeWord expects 80ms chunks at 16kHz

    def __init__(self):
        self._model = None
        self.wake_word = config.WAKE_WORD
        self.sensitivity = config.WAKE_WORD_SENSITIVITY

    def load(self):
        """Load the wake word model."""
        try:
            # Normalise config value → model name (e.g. "hey jarvis" → "hey_jarvis")
            model_name = self.wake_word.lower().replace(" ", "_")
            self._model = Model(
                wakeword_models=[model_name],
                inference_framework="onnx",
            )
            logger.info(f"Wake word detector ready: '{self.wake_word}'")
        except Exception as e:
            raise AudioError(f"Failed to load wake word model: {e}")

    def wait_for_wake_word(self, mic) -> None:
        """
        Block until the wake word is detected.

        Args:
            mic: MicCapture instance (must be started).
        """
        logger.info(f"Listening for wake word: '{self.wake_word}'...")
        buffer = np.array([], dtype=np.float32)

        while True:
            chunk = mic.read()
            buffer = np.concatenate([buffer, chunk])

            if len(buffer) >= self.CHUNK_SIZE:
                window = buffer[:self.CHUNK_SIZE]
                buffer = buffer[self.CHUNK_SIZE:]

                audio_int16 = (window * 32767).astype(np.int16)
                prediction = self._model.predict(audio_int16)

                for word, score in prediction.items():
                    if score >= self.sensitivity:
                        logger.info(f"Wake word detected! (score: {score:.2f})")
                        self._model.reset()
                        return
