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
    """Abstract base for vision model providers."""

    @abstractmethod
    def describe(self, image_bytes: bytes, prompt: str) -> str:
        """Send image bytes to the vision model and return its text response."""
        ...


class GeminiVisionProvider(VisionProvider):
    """Uses Gemini Flash for vision (already authenticated via GEMINI_API_KEY)."""

    def describe(self, image_bytes: bytes, prompt: str) -> str:
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
    """Return the configured vision provider instance."""
    provider = config.VISION_PROVIDER.lower()
    if provider == "ollama":
        logger.info(f"Vision: using Ollama ({config.VISION_MODEL}) — local/private")
        return OllamaVisionProvider()
    logger.info("Vision: using Gemini Flash — cloud")
    return GeminiVisionProvider()
