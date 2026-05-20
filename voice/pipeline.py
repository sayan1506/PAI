"""
voice/pipeline.py

Full voice input pipeline: wake word → listen → transcribe → return text.

Usage:
    pipeline = VoicePipeline()
    pipeline.load()
    text = pipeline.listen_once()   # blocks until one utterance is complete
"""

from voice.mic import MicCapture
from voice.wake_word import WakeWordDetector
from voice.vad import VoiceActivityDetector
from voice.listener import UtteranceListener
from voice.stt import SpeechToText
from core.logger import logger
from core.exceptions import AudioError
import config


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

    def run_forever(self, on_transcription):
        """
        Run the voice pipeline in a loop, calling on_transcription
        with each transcribed utterance.

        Args:
            on_transcription: Callable that receives a text string.
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
                        on_transcription(text)
                except AudioError as e:
                    logger.error(f"Voice pipeline error: {e}")
                except KeyboardInterrupt:
                    logger.info("Voice pipeline stopped by user")
                    break
