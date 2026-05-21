"""
voice/pipeline.py

Full voice input pipeline: wake word → listen → transcribe → return text.

Usage:
    pipeline = VoicePipeline()
    pipeline.load()
    text = pipeline.listen_once()   # blocks until one utterance is complete
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from voice.mic import MicCapture
from voice.wake_word import WakeWordDetector
from voice.vad import VoiceActivityDetector
from voice.listener import UtteranceListener
from voice.stt import SpeechToText
from voice.barge_in import BargeInMonitor
from core.logger import logger
from core.exceptions import AudioError
import config

if TYPE_CHECKING:
    from voice.tts import BaseTTS


class VoicePipeline:
    """Orchestrates the full voice input chain."""

    def __init__(self):
        self.mic = MicCapture()
        self.wake_word = WakeWordDetector()
        self.vad = VoiceActivityDetector()
        self.listener = None   # created after VAD loads
        self.stt = SpeechToText()

    def load(self):
        """Load all models. Call once at startup."""
        logger.info("Loading voice pipeline...")
        self.wake_word.load()
        self.vad.load()
        self.listener = UtteranceListener(self.vad)
        self.stt.load()
        logger.info("Voice pipeline ready")

    def listen_once(self) -> str:
        """
        Run one full voice interaction cycle.

        Blocks until:
          1. Wake word is detected
          2. User speaks and pauses
          3. Speech is transcribed

        Returns:
            Transcribed text string. Empty string if nothing was heard.
        """
        with self.mic:
            self.wake_word.wait_for_wake_word(self.mic)
            # Drain mic buffer to flush wake word tail audio
            for _ in range(15):  # ~480ms at 32ms/chunk
                self.mic.read()
            self.vad.reset()
            print("\n[Listening...]")
            audio = self.listener.listen(self.mic)
            text = self.stt.transcribe(audio)
            return text

    def run_forever(self, on_transcription, tts: "BaseTTS | None" = None):
        """
        Run the voice pipeline in a loop, calling on_transcription
        with each transcribed utterance.

        Args:
            on_transcription: Callable that receives a text string.
            tts: Optional TTS engine instance. When provided and
                 BARGE_IN_ENABLED is true, a BargeInMonitor wraps
                 the on_transcription callback to detect interruptions.
        """
        with self.mic:
            while True:
                try:
                    self.wake_word.wait_for_wake_word(self.mic)
                    # Drain mic buffer to flush wake word tail audio
                    for _ in range(15):  # ~480ms at 32ms/chunk
                        self.mic.read()
                    self.vad.reset()
                    print("\n[Listening...]")
                    audio = self.listener.listen(self.mic)
                    text = self.stt.transcribe(audio)
                    if text:
                        self._process_transcription(text, on_transcription, tts)
                    # Flush mic buffer after processing to discard stale audio
                    # that accumulated while the LLM was generating a response
                    for _ in range(30):  # ~960ms flush
                        self.mic.read()
                except AudioError as e:
                    logger.error(f"Voice pipeline error: {e}")
                except KeyboardInterrupt:
                    logger.info("Voice pipeline stopped by user")
                    break

    def _process_transcription(
        self, text: str, on_transcription, tts: "BaseTTS | None"
    ) -> None:
        """
        Process a transcription with optional barge-in monitoring.

        Starts a BargeInMonitor before calling on_transcription and
        stops it after. If the monitor was triggered (user interrupted),
        enters a follow-up listen cycle without requiring the wake word.
        """
        monitor = self._start_monitor(tts)
        try:
            on_transcription(text)
        finally:
            if monitor is not None:
                monitor.stop()

        # If barge-in was triggered, listen for follow-up without wake word
        if monitor is not None and monitor.triggered:
            self._handle_followup(on_transcription, tts)

    def _start_monitor(self, tts: "BaseTTS | None") -> "BargeInMonitor | None":
        """Create and start a BargeInMonitor if barge-in is enabled.

        Skips monitoring for engines that don't support interruption
        (e.g. Pyttsx3TTS) to avoid false triggers from speaker bleed.
        """
        if tts is not None and config.BARGE_IN_ENABLED:
            # Only monitor if the engine actually supports barge-in
            from voice.tts import Pyttsx3TTS
            if isinstance(tts, Pyttsx3TTS):
                return None
            monitor = BargeInMonitor(tts)
            monitor.start()
            return monitor
        return None

    def _handle_followup(self, on_transcription, tts: "BaseTTS | None") -> None:
        """
        Handle a post-interruption follow-up cycle.

        Listens directly (no wake word required), transcribes, and
        processes the follow-up with a fresh BargeInMonitor.
        """
        logger.info("Barge-in detected — listening for follow-up")
        # Drain mic buffer to discard any residual TTS audio
        for _ in range(15):  # ~480ms flush
            self.mic.read()
        self.vad.reset()
        print("\n[Listening (follow-up)...]")
        audio = self.listener.listen(self.mic)
        followup_text = self.stt.transcribe(audio)
        if followup_text:
            # Wrap follow-up with a fresh monitor
            self._process_transcription(followup_text, on_transcription, tts)
