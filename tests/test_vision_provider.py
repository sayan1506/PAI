"""Tests for vision provider module."""
import base64
import json
from unittest.mock import patch, MagicMock
import pytest
import config
from providers.vision_provider import (
    OllamaVisionProvider,
    GeminiVisionProvider,
    get_vision_provider,
)


class TestOllamaVisionProvider:
    @patch("providers.vision_provider.urllib.request.urlopen")
    def test_sends_base64_image(self, mock_urlopen, monkeypatch):
        monkeypatch.setattr(config, "VISION_MODEL", "gemma3:4b")
        monkeypatch.setattr(config, "OLLAMA_HOST", "http://localhost:11434")

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"response": "A screen with code"}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        provider = OllamaVisionProvider()
        result = provider.describe(b"fake_image_data", "What is on screen?")

        assert result == "A screen with code"
        # Verify the request body contains base64 encoded image
        call_args = mock_urlopen.call_args[0][0]
        body = json.loads(call_args.data)
        expected_b64 = base64.b64encode(b"fake_image_data").decode()
        assert body["images"] == [expected_b64]
        assert body["model"] == "gemma3:4b"
        assert body["prompt"] == "What is on screen?"


class TestGetVisionProvider:
    def test_returns_ollama_when_configured(self, monkeypatch):
        monkeypatch.setattr(config, "VISION_PROVIDER", "ollama")
        provider = get_vision_provider()
        assert isinstance(provider, OllamaVisionProvider)

    def test_returns_gemini_when_configured(self, monkeypatch):
        monkeypatch.setattr(config, "VISION_PROVIDER", "gemini")
        provider = get_vision_provider()
        assert isinstance(provider, GeminiVisionProvider)

    def test_defaults_to_gemini(self, monkeypatch):
        monkeypatch.setattr(config, "VISION_PROVIDER", "unknown")
        provider = get_vision_provider()
        assert isinstance(provider, GeminiVisionProvider)
