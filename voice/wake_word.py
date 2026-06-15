"""OpenWakeWord-based wake word detection.

Provides :class:`WakeWordDetector`, which loads an OpenWakeWord ONNX
model and continuously scans microphone audio until the configured wake
word is heard. Audio is buffered into the 80 ms / 1280-sample windows
the model expects at 16 kHz. The wake word and detection sensitivity are
read from project configuration.
"""

import numpy as np
from openwakeword.model import Model
from core.logger import logger
from core.exceptions import AudioError
import config


class WakeWordDetector:
    """Listens for the wake word using OpenWakeWord.

    Holds the loaded OpenWakeWord model and the configured wake word and
    sensitivity. :meth:`wait_for_wake_word` blocks, reading from the
    microphone and accumulating audio into fixed windows until a model
    score meets the sensitivity threshold.

    Supported built-in wake words (no training required):
        "hey jarvis", "hey mycroft", "alexa", "ok google", and others.

    Attributes:
        CHUNK_SIZE: Number of samples per inference window (80 ms @ 16k).
        wake_word: The configured wake phrase to listen for.
        sensitivity: Minimum detection score required to trigger.

    Usage:
        detector = WakeWordDetector()
        detector.load()
        detector.wait_for_wake_word(mic)   # blocks until triggered
    """

    CHUNK_SIZE = 1280   # OpenWakeWord expects 80ms chunks at 16kHz

    def __init__(self):
        """Initialise with wake word and sensitivity from configuration.

        Does not load the model; call :meth:`load` before detecting.
        """
        self._model = None
        self.wake_word = config.WAKE_WORD
        self.sensitivity = config.WAKE_WORD_SENSITIVITY

    def load(self):
        """Load the OpenWakeWord model for the configured wake word.

        Normalises the configured phrase to the model name format
        (e.g. "hey jarvis" -> "hey_jarvis") and loads it using the ONNX
        inference framework.

        Side effects:
            Stores the loaded model on the instance.

        Raises:
            AudioError: If the wake word model cannot be loaded.
        """
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
        """Block until the wake word is detected.

        Continuously reads audio from the microphone, accumulating it
        into ``CHUNK_SIZE`` windows. Each window is converted to int16
        and scored by the model; the call returns once any wake word
        score meets the configured sensitivity. The model state is reset
        before returning so the next call starts cleanly.

        Args:
            mic: A started MicCapture instance to read audio from.

        Side effects:
            Consumes audio from the microphone and resets model state on
            detection.
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
