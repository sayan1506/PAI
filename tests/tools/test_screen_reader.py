"""Tests for ScreenReaderTool."""
from unittest.mock import patch, MagicMock
import pytest
import config
from tools.screen_reader import ScreenReaderTool


@pytest.fixture
def tool():
    return ScreenReaderTool()


class TestScreenReader:
    def test_disabled_by_default(self, tool, monkeypatch):
        monkeypatch.setattr(config, "VISION_ENABLED", False)
        result = tool.execute(prompt="what's on screen?")
        assert result.success is False
        assert "disabled" in result.error.lower()

    @patch("providers.vision_provider.get_vision_provider")
    @patch("utils.vision_utils.capture_screenshot")
    def test_captures_and_describes(self, mock_capture, mock_provider_fn, tool, monkeypatch):
        monkeypatch.setattr(config, "VISION_ENABLED", True)
        mock_capture.return_value = b"fake_png_bytes"
        mock_provider = MagicMock()
        mock_provider.describe.return_value = "A desktop with VS Code open"
        mock_provider_fn.return_value = mock_provider

        result = tool.execute()

        assert result.success is True
        assert "VS Code" in result.output
        mock_capture.assert_called_once()
        mock_provider.describe.assert_called_once()

    @patch("providers.vision_provider.get_vision_provider")
    @patch("utils.vision_utils.capture_screenshot")
    def test_uses_custom_prompt(self, mock_capture, mock_provider_fn, tool, monkeypatch):
        monkeypatch.setattr(config, "VISION_ENABLED", True)
        mock_capture.return_value = b"png"
        mock_provider = MagicMock()
        mock_provider.describe.return_value = "Error dialog visible"
        mock_provider_fn.return_value = mock_provider

        result = tool.execute(prompt="What error is shown?")

        assert result.success is True
        mock_provider.describe.assert_called_once_with(b"png", "What error is shown?")

    @patch("providers.vision_provider.get_vision_provider")
    @patch("utils.vision_utils.capture_screenshot")
    def test_handles_provider_failure(self, mock_capture, mock_provider_fn, tool, monkeypatch):
        monkeypatch.setattr(config, "VISION_ENABLED", True)
        mock_capture.return_value = b"png"
        mock_provider = MagicMock()
        mock_provider.describe.side_effect = RuntimeError("Model crashed")
        mock_provider_fn.return_value = mock_provider

        result = tool.execute()

        assert result.success is False
        assert "failed" in result.error.lower()
