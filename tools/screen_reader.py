"""
tools/screen_reader.py

Screen reader tool for PAI.
Captures a screenshot and sends it to a vision model for analysis.
"""

from __future__ import annotations
import config
from tools.base import BaseTool, ToolResult
from core.logger import logger


class ScreenReaderTool(BaseTool):
    """Capture and analyse the current screen using a vision model."""

    @property
    def name(self) -> str:
        return "screen_reader"

    @property
    def description(self) -> str:
        return (
            "Capture a screenshot of the screen and describe what is visible. "
            "Use this when the user asks 'what's on my screen', 'what does this say', "
            "'what is open', or any question that requires seeing the screen."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "What to look for or ask about the screenshot. "
                        "Default: 'Describe everything visible on the screen in detail.'"
                    ),
                },
            },
            "required": [],
        }

    def execute(self, **kwargs) -> ToolResult:
        if not config.VISION_ENABLED:
            return ToolResult(
                success=False,
                error="Screen reader is disabled. Set VISION_ENABLED=true in config.",
            )

        prompt = kwargs.get(
            "prompt",
            "Describe everything visible on the screen in detail. "
            "Include open windows, text, buttons, and any notable content.",
        )

        try:
            from utils.vision_utils import capture_screenshot
            from providers.vision_provider import get_vision_provider

            image_bytes = capture_screenshot()
            provider = get_vision_provider()
            description = provider.describe(image_bytes, prompt)

            logger.info(f"Screen described: {len(description)} chars")
            return ToolResult(success=True, output=description)

        except Exception as e:
            logger.error(f"Screen reader failed: {e}")
            return ToolResult(success=False, error=f"Screen reader failed: {e}")
