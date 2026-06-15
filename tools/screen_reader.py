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
    """Tool that captures the screen and describes it using a vision model.

    Takes a screenshot and sends it to the configured vision provider for
    analysis, returning a natural-language description. Gated by
    ``config.VISION_ENABLED``; depends on ``utils.vision_utils`` for capture
    and ``providers.vision_provider`` for the model.
    """

    @property
    def name(self) -> str:
        """Return the tool's unique identifier."""
        return "screen_reader"

    @property
    def description(self) -> str:
        """Return the LLM-facing description of this tool."""
        return (
            "Capture a screenshot of the screen and describe what is visible. "
            "Use this when the user asks 'what's on my screen', 'what does this say', "
            "'what is open', or any question that requires seeing the screen."
        )

    @property
    def parameters(self) -> dict:
        """Return the JSON Schema for this tool's arguments."""
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
        """Capture the screen and return a vision model's description.

        Takes a screenshot, sends it to the configured vision provider along
        with an optional ``prompt``, and returns the resulting description.

        Args:
            **kwargs: Optionally ``prompt`` describing what to look for;
                defaults to a general full-screen description request.

        Returns:
            A ``ToolResult`` whose ``output`` is the description, or an error
            if vision is disabled or capture/analysis fails.

        Side Effects:
            Captures a screenshot and makes a request to the vision provider.
        """
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
