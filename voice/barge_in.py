"""
voice/barge_in.py

Barge-in monitor for interrupting TTS playback when the user speaks.

Uses a sounddevice callback-based InputStream (separate from MicCapture)
to detect speech via Silero VAD during TTS playback. When speech is
detected, tts.stop() is called immediately for near-instant interruption.
"""

import numpy as np
import sounddevice as sd
import torch

import config
from core.logger import logger
from voice.tts import BaseTTS


class BargeInMonitor:
    """
    Monitors microphone input during TTS playback for barge-in detection.

    Opens a separate sounddevice InputStream with a callback at 16kHz,
    blocksize=512 (~32ms). The callback runs Silero VAD on each audio
    frame; on speech detection it sets the triggered flag and calls
    tts.stop() to interrupt playback immediately.

    Usage:
        monitor = BargeInMonitor(tts)
        monitor.start()
        tts.speak(text)
        monitor.stop()
        if monitor.triggered:
            # User interrupted — handle follow-up
    """

    SAMPLE_RATE: int = 16_000
    BLOCK_SIZE: int = 512  # ~32ms at 16kHz

    def __init__(self, tts: BaseTTS) -> None:
        """
        Initialise the barge-in monitor.

        Args:
            tts: Reference to the active TTS engine. Its stop() method
                 will be called when speech is detected.
        """
        self._tts = tts
        self._stream: sd.InputStream | None = None
        self._triggered: bool = False
        self._vad_model = None
        self._load_vad()

    def _load_vad(self) -> None:
        """Load the Silero VAD model for speech detection."""
        try:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._vad_model = model
            logger.debug("BargeInMonitor: Silero VAD loaded")
        except Exception as exc:
            logger.error("BargeInMonitor: Failed to load Silero VAD: %s", exc)
            self._vad_model = None

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info, status
    ) -> None:
        """
        Sounddevice callback invoked on the audio driver thread.

        Runs VAD on the incoming audio frame. If speech is detected
        and barge-in has not already been triggered, sets the triggered
        flag and calls tts.stop().
        """
        if status:
            logger.debug("BargeInMonitor: stream status: %s", status)

        if self._triggered or self._vad_model is None:
            return

        # Convert to mono float32 (indata shape is (frames, channels))
        audio = indata[:, 0].astype(np.float32)
        tensor = torch.FloatTensor(audio)

        try:
            confidence = self._vad_model(tensor, self.SAMPLE_RATE).item()
        except Exception:
            return

        if confidence > 0.5:
            self._triggered = True
            logger.info("BargeInMonitor: Speech detected — interrupting TTS")
            self._tts.stop()

    def start(self) -> None:
        """
        Begin monitoring for barge-in.

        Opens the InputStream and resets the triggered flag.
        No-op when BARGE_IN_ENABLED is false or VAD failed to load.
        """
        if not config.BARGE_IN_ENABLED:
            return

        if self._vad_model is None:
            logger.warning(
                "BargeInMonitor: Cannot start — VAD model not loaded"
            )
            return

        self._triggered = False

        # Reset VAD internal state for a fresh detection window
        self._vad_model.reset_states()

        try:
            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                blocksize=self.BLOCK_SIZE,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.debug("BargeInMonitor: Monitoring started")
        except Exception as exc:
            logger.error("BargeInMonitor: Failed to open stream: %s", exc)
            self._stream = None

    def stop(self) -> None:
        """
        Stop monitoring for barge-in.

        Closes the InputStream. Safe to call even if start() was not
        called or if monitoring is already stopped.
        """
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                logger.debug("BargeInMonitor: Error closing stream: %s", exc)
            finally:
                self._stream = None
                logger.debug("BargeInMonitor: Monitoring stopped")

    @property
    def triggered(self) -> bool:
        """True if speech was detected during the monitoring window."""
        return self._triggered
