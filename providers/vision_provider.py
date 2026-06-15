"""
providers/vision_provider.py

Vision model abstraction for screen reading.
Supports Gemini (cloud) and Ollama (local) backends.
"""

from __future__ import annotations
import base64
import json
import urllib.request
from abc import ABC, abstractmethod
from PIL import Image
import io
import config
from core.logger import logger


class VisionProvider(ABC):
    """Abstract base for vision model providers.

    Subclasses send raw image bytes plus a text prompt to a vision-capable
    model and return the model's textual description.
    """

    @abstractmethod
    def describe(self, image_bytes: bytes, prompt: str) -> str:
        """Send image bytes to the vision model and return its text response.

        Args:
            image_bytes: Raw encoded image data (e.g. PNG or JPEG bytes).
            prompt: Instruction describing what to extract from the image.

        Returns:
            The model's textual description of the image.
        """
        ...


class GeminiVisionProvider(VisionProvider):
    """Uses Gemini Flash for vision (already authenticated via GEMINI_API_KEY)."""

    def describe(self, image_bytes: bytes, prompt: str) -> str:
        """Describe an image using the Gemini Flash vision model.

        Configures the google-generativeai SDK with ``config.GEMINI_API_KEY``,
        decodes the bytes into a PIL image, and sends it with the prompt.

        Args:
            image_bytes: Raw encoded image data.
            prompt: Instruction describing what to extract from the image.

        Returns:
            The stripped text content of the model's response.
        """
        import google.generativeai as genai
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
        image = Image.open(io.BytesIO(image_bytes))
        response = model.generate_content([prompt, image])
        return response.text.strip()


class OllamaVisionProvider(VisionProvider):
    """
    Uses a local Ollama vision model (e.g. gemma3:4b, minicpm-v, llava).
    No data leaves the machine.
    """

    def describe(self, image_bytes: bytes, prompt: str) -> str:
        """Describe an image using a local Ollama vision model.

        Base64-encodes the image, POSTs it with the prompt to the Ollama
        ``/api/generate`` endpoint with streaming disabled, and returns the
        response text. All processing stays on the local machine.

        Args:
            image_bytes: Raw encoded image data.
            prompt: Instruction describing what to extract from the image.

        Returns:
            The stripped ``response`` field from the Ollama reply, or an empty
            string if absent.
        """
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = json.dumps({
            "model": config.VISION_MODEL,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{config.OLLAMA_HOST}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()


def get_vision_provider() -> VisionProvider:
    """Return the configured vision provider instance.

    Selects the backend from ``config.VISION_PROVIDER``: "ollama" yields a
    local ``OllamaVisionProvider``; any other value falls back to the
    cloud ``GeminiVisionProvider``.

    Returns:
        An instantiated ``VisionProvider`` for the configured backend.
    """
    provider = config.VISION_PROVIDER.lower()
    if provider == "ollama":
        logger.info(f"Vision: using Ollama ({config.VISION_MODEL}) — local/private")
        return OllamaVisionProvider()
    logger.info("Vision: using Gemini Flash — cloud")
    return GeminiVisionProvider()
