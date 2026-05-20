"""
voice/mic.py

Low-level microphone capture using sounddevice.
Returns a blocking stream of audio chunks as numpy float32 arrays.
"""

import numpy as np
import sounddevice as sd
from core.logger import logger
from core.exceptions import AudioError
import config


class MicCapture:
    """Captures audio from the default microphone in blocking mode."""

    def __init__(self):
        self.sample_rate = config.MIC_SAMPLE_RATE
        self.channels = config.MIC_CHANNELS
        self.chunk_size = 512  # frames per read — small for low latency
        self._stream = None
        # Parse device: empty string = default, otherwise int index
        self.device = int(config.MIC_DEVICE) if config.MIC_DEVICE.strip() else None

    def start(self):
        """Open the audio stream."""
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.chunk_size,
                device=self.device,
            )
            self._stream.start()
            device_name = self.device if self.device is not None else "default"
            logger.info(f"Mic opened: {self.sample_rate}Hz, {self.channels}ch, device={device_name}")
        except Exception as e:
            raise AudioError(f"Cannot open microphone: {e}")

    def read(self) -> np.ndarray:
        """Read one chunk from the mic. Blocks until data is available."""
        if self._stream is None:
            raise AudioError("MicCapture not started. Call start() first.")
        data, _ = self._stream.read(self.chunk_size)
        return data.flatten()

    def stop(self):
        """Close the audio stream."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Mic closed")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
