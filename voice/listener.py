"""Utterance collector driven by Silero VAD.

Provides :class:`UtteranceListener`, which records from the microphone
until the user stops speaking, then returns the full utterance as a
numpy array ready for speech-to-text. Speech start and end are detected
with the shared :class:`VoiceActivityDetector`; a trailing-silence
threshold and a minimum-speech-length requirement guard against
premature cutoffs and spurious triggers.
"""

import numpy as np
from voice.vad import VoiceActivityDetector
from core.logger import logger
import config


class UtteranceListener:
    """Collects a complete speech utterance from the microphone.

    Uses Silero VAD to detect when speech starts and ends, returning the
    buffered audio once enough trailing silence is observed. The silence
    threshold is derived from ``VAD_SILENCE_MS`` and the per-chunk
    duration implied by the mic sample rate.

    Attributes:
        vad: The voice activity detector used to classify each chunk.
    """

    def __init__(self, vad: VoiceActivityDetector):
        """Initialise the listener and derive the silence threshold.

        Computes how many consecutive silent chunks mark the end of
        speech, based on ``VAD_SILENCE_MS`` and the chunk duration
        implied by ``MIC_SAMPLE_RATE`` (512 samples per chunk).

        Args:
            vad: A loaded :class:`VoiceActivityDetector` instance.
        """
        self.vad = vad
        # How many consecutive silent chunks = end of speech
        # Derived from actual config values (512 samples per chunk)
        _ms_per_chunk = (512 * 1000) // config.MIC_SAMPLE_RATE  # 32 ms at 16kHz
        self._silence_chunks = config.VAD_SILENCE_MS // _ms_per_chunk

    def listen(self, mic) -> np.ndarray:
        """Record until the user stops speaking.

        Reads chunks from the microphone, discarding leading silence
        until speech begins. Once speech has started, chunks are buffered
        (including short internal pauses) until enough consecutive silent
        chunks are seen and a minimum amount of speech has accumulated,
        at which point the buffered utterance is returned.

        Args:
            mic: A started MicCapture instance to read audio from.

        Returns:
            numpy float32 array of the complete utterance at 16kHz.

        Side effects:
            Consumes audio from the microphone and resets VAD state once
            the utterance ends.
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
