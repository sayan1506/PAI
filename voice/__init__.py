"""Voice subsystem for the Personal AI Assistant.

Provides the end-to-end voice input/output stack: microphone capture,
wake-word detection, voice activity detection, utterance collection,
speech-to-text, text-to-speech, and barge-in interruption. The public
entry point is :class:`VoicePipeline`, which orchestrates the full
chain from wake word to transcribed text.
"""

from voice.pipeline import VoicePipeline

__all__ = ["VoicePipeline"]
