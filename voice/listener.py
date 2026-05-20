"""
voice/listener.py

Utterance collector using Silero VAD.
Records from the mic until the user stops speaking, then returns
the full audio buffer as a numpy array ready for STT.
"""

import numpy as np
from voice.vad import VoiceActivityDetector
from core.logger import logger
import config


class UtteranceListener:
    """
    Collects a complete speech utterance from the microphone.

    Uses Silero VAD to detect when speech starts and ends.
    Returns the full audio buffer once silence is detected.
    """

    def __init__(self, vad: VoiceActivityDetector):
        self.vad = vad
        # How many consecutive silent chunks = end of speech
        # Derived from actual config values (512 samples per chunk)
        _ms_per_chunk = (512 * 1000) // config.MIC_SAMPLE_RATE  # 32 ms at 16kHz
        self._silence_chunks = config.VAD_SILENCE_MS // _ms_per_chunk

    def listen(self, mic) -> np.ndarray:
        """
        Record until the user stops speaking.

        Args:
            mic: MicCapture instance (must be started).

        Returns:
            numpy float32 array of the complete utterance at 16kHz.
        """
        logger.info("Listening for speech...")
        audio_buffer = []
        silent_chunks = 0
        speech_chunks = 0
        speech_started = False
        # Require at least ~500ms of speech before allowing silence to end
        _ms_per_chunk = (512 * 1000) // config.MIC_SAMPLE_RATE
        min_speech_chunks = 500 // _ms_per_chunk  # ~16 chunks

        while True:
            chunk = mic.read()
            is_speech = self.vad.is_speech(chunk)

            if is_speech:
                speech_started = True
                silent_chunks = 0
                speech_chunks += 1
                audio_buffer.append(chunk)
            elif speech_started:
                # Speech already started — count silence
                audio_buffer.append(chunk)
                silent_chunks += 1
                if silent_chunks >= self._silence_chunks and speech_chunks >= min_speech_chunks:
                    logger.info(f"Speech ended ({len(audio_buffer)} chunks collected, {speech_chunks} speech)")
                    self.vad.reset()
                    return np.concatenate(audio_buffer)
            # If speech hasn't started yet, discard the chunk
