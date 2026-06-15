"""Low-level microphone capture using sounddevice.

Provides :class:`MicCapture`, a thin blocking wrapper around a
``sounddevice.InputStream`` that yields fixed-size audio chunks as
numpy float32 arrays. Sample rate, channel count, and input device are
read from the project configuration. The small chunk size keeps read
latency low for real-time wake-word and VAD processing.
"""

import numpy as np
import sounddevice as sd
from core.logger import logger
from core.exceptions import AudioError
import config


class MicCapture:
    """Captures audio from the default microphone in blocking mode.

    Wraps a ``sounddevice.InputStream`` opened in blocking mode and
    exposes a simple start/read/stop lifecycle. Also supports use as a
    context manager so the stream is opened on entry and closed on exit.

    Attributes:
        sample_rate: Capture sample rate in Hz (from config).
        channels: Number of input channels (from config).
        chunk_size: Frames returned per :meth:`read` call.
        device: Input device index, or None for the system default.
    """

    def __init__(self):
        """Initialise capture settings from configuration.

        Resolves the configured device string into an integer index, or
        None when blank (meaning the system default device). Does not
        open the audio stream; call :meth:`start` for that.
        """
        self.sample_rate = config.MIC_SAMPLE_RATE
        self.channels = config.MIC_CHANNELS
        self.chunk_size = 512  # frames per read — small for low latency
        self._stream = None
        # Parse device: empty string = default, otherwise int index
        self.device = int(config.MIC_DEVICE) if config.MIC_DEVICE.strip() else None

    def start(self):
        """Open and start the audio input stream.

        Side effects:
            Opens a sounddevice ``InputStream`` and begins capture,
            storing it on the instance.

        Raises:
            AudioError: If the microphone or stream cannot be opened.
        """
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
        """Read one chunk of audio from the microphone.

        Blocks until ``chunk_size`` frames are available.

        Returns:
            A flattened float32 numpy array of length ``chunk_size``.

        Raises:
            AudioError: If called before :meth:`start`.
        """
        if self._stream is None:
            raise AudioError("MicCapture not started. Call start() first.")
        data, _ = self._stream.read(self.chunk_size)
        return data.flatten()

    def stop(self):
        """Stop and close the audio input stream.

        Safe to call when the stream is already closed. Clears the
        stored stream reference after closing.
        """
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Mic closed")

    def __enter__(self):
        """Open the stream on context entry and return self."""
        self.start()
        return self

    def __exit__(self, *_):
        """Close the stream on context exit."""
        self.stop()
