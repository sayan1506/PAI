"""Text-to-speech engines for spoken responses.

Defines the :class:`BaseTTS` abstract interface and two concrete
implementations: :class:`KokoroTTS`, a neural engine that streams audio
in chunks at 24 kHz with epoch-based barge-in support, and
:class:`Pyttsx3TTS`, a synchronous OS-engine fallback without
interruption support. The :func:`get_tts` factory selects and loads an
engine from configuration, falling back gracefully on failure.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import sounddevice as sd

import config
from core.logger import logger


class BaseTTS(ABC):
    """
    Abstract base class for text-to-speech engines.

    Subclasses must implement load(), speak(), stop(), and is_speaking.
    """

    @abstractmethod
    def load(self) -> None:
        """Load models and initialise the engine. Called once at startup."""
        ...

    @abstractmethod
    def speak(self, text: str) -> None:
        """Synthesise and play audio. Blocks until done or interrupted."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Interrupt playback immediately."""
        ...

    @property
    @abstractmethod
    def is_speaking(self) -> bool:
        """True while audio is actively playing."""
        ...


class KokoroTTS(BaseTTS):
    """Neural TTS engine using Kokoro KPipeline.

    Streams audio in chunks at 24kHz. Uses an epoch counter for
    instant barge-in interruption: stop() increments the epoch and
    calls sd.stop(), causing the speak() loop to exit on the next
    chunk boundary.

    Attributes:
        SAMPLE_RATE: Output sample rate in Hz for synthesised audio.
    """

    SAMPLE_RATE: int = 24_000

    def __init__(self) -> None:
        """Initialise an unloaded Kokoro engine.

        Sets up the epoch counter and speaking flag used for barge-in;
        the pipeline is built lazily in :meth:`load`.
        """
        self._pipeline = None
        self._epoch: int = 0
        self._speaking: bool = False

    def load(self) -> None:
        """Load the Kokoro pipeline with the configured voice."""
        from kokoro import KPipeline

        self._pipeline = KPipeline(lang_code="a")
        logger.info(
            "KokoroTTS loaded (voice=%s, speed=%.2f)",
            config.TTS_VOICE,
            config.TTS_SPEED,
        )

    def speak(self, text: str) -> None:
        """
        Synthesise *text* and stream audio chunk-by-chunk.

        Between each chunk the current epoch is compared to the epoch
        captured at entry. If stop() was called in the meantime the
        epoch will have advanced and playback exits immediately.
        """
        if self._pipeline is None:
            logger.warning("KokoroTTS.speak() called before load()")
            return

        start_epoch = self._epoch
        self._speaking = True

        try:
            generator = self._pipeline(
                text,
                voice=config.TTS_VOICE,
                speed=config.TTS_SPEED,
            )

            for result in generator:
                # Epoch mismatch means stop() was called — exit immediately
                if self._epoch != start_epoch:
                    break

                audio = result.audio
                if audio is None:
                    continue

                # Ensure audio is a numpy float32 array for sounddevice
                if not isinstance(audio, np.ndarray):
                    audio = np.array(audio, dtype=np.float32)
                elif audio.dtype != np.float32:
                    audio = audio.astype(np.float32)

                sd.play(audio, samplerate=self.SAMPLE_RATE)
                sd.wait()

                # Check epoch again after playback of this chunk
                if self._epoch != start_epoch:
                    break
        finally:
            self._speaking = False

    def stop(self) -> None:
        """Interrupt playback by advancing the epoch and stopping audio."""
        self._epoch += 1
        sd.stop()

    @property
    def is_speaking(self) -> bool:
        """True while speak() is actively playing audio chunks."""
        return self._speaking


class Pyttsx3TTS(BaseTTS):
    """
    Synchronous TTS fallback using the OS speech engine via pyttsx3.

    Plays utterances to completion; barge-in is not supported.
    A warning is logged at load time when BARGE_IN_ENABLED is true.
    """

    def __init__(self) -> None:
        """Initialise an unloaded pyttsx3 engine."""
        self._engine = None
        self._speaking: bool = False

    def load(self) -> None:
        """Initialise the pyttsx3 engine and set speech rate."""
        import pyttsx3

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", int(175 * config.TTS_SPEED))

        if config.BARGE_IN_ENABLED:
            logger.warning(
                "Barge-in is not supported with pyttsx3 TTS engine; "
                "BARGE_IN_ENABLED will be ignored."
            )

        logger.info(
            "Pyttsx3TTS loaded (rate={}, speed={:.2f})",
            int(175 * config.TTS_SPEED),
            config.TTS_SPEED,
        )

    def speak(self, text: str) -> None:
        """Synthesise and play *text* synchronously (blocks until complete)."""
        if self._engine is None:
            logger.warning("Pyttsx3TTS.speak() called before load()")
            return

        self._speaking = True
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except RuntimeError:
            # pyttsx3 can get into a bad state; reinitialise
            logger.debug("Pyttsx3TTS: reinitialising engine after RuntimeError")
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", int(175 * config.TTS_SPEED))
            self._engine.say(text)
            self._engine.runAndWait()
        finally:
            self._speaking = False

    def stop(self) -> None:
        """No-op — pyttsx3 is synchronous and cannot be interrupted."""
        pass

    @property
    def is_speaking(self) -> bool:
        """True while speak() is actively playing."""
        return self._speaking


def get_tts() -> Optional[BaseTTS]:
    """
    Factory function that returns the configured TTS engine.

    Returns None if TTS_ENABLED is false. If TTS_ENGINE is "kokoro",
    attempts to load KokoroTTS and falls back to Pyttsx3TTS on failure.
    If all engines fail, returns None with a warning.
    """
    if not config.TTS_ENABLED:
        logger.info("TTS disabled (TTS_ENABLED=false)")
        return None

    engine_name = config.TTS_ENGINE.lower()

    if engine_name == "kokoro":
        try:
            tts = KokoroTTS()
            tts.load()
            return tts
        except Exception as exc:
            logger.warning(
                "KokoroTTS failed to load (%s); falling back to pyttsx3", exc
            )
            # Fall through to pyttsx3 fallback
            try:
                tts = Pyttsx3TTS()
                tts.load()
                return tts
            except Exception as fallback_exc:
                logger.warning(
                    "Pyttsx3TTS fallback also failed (%s); TTS unavailable",
                    fallback_exc,
                )
                return None

    if engine_name == "pyttsx3":
        try:
            tts = Pyttsx3TTS()
            tts.load()
            return tts
        except Exception as exc:
            logger.warning(
                "Pyttsx3TTS failed to load (%s); TTS unavailable", exc
            )
            return None

    # Unknown engine name
    logger.warning(
        "Unknown TTS_ENGINE=%r; falling back to pyttsx3", config.TTS_ENGINE
    )
    try:
        tts = Pyttsx3TTS()
        tts.load()
        return tts
    except Exception as exc:
        logger.warning(
            "Pyttsx3TTS fallback failed (%s); TTS unavailable", exc
        )
        return None
