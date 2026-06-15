"""Screen capture helpers for the vision pipeline.

Uses the ``mss`` library for fast, cross-platform screenshot capture without
depending on the X11 or Win32 APIs directly. Output is PNG bytes ready to
hand to a vision-capable LLM provider.
"""

import time
import config
from core.logger import logger


def capture_screenshot(monitor: int = 1) -> bytes:
    """Capture a monitor and return the image as raw PNG bytes.

    Sleeps for ``config.SCREENSHOT_DELAY`` seconds before grabbing, giving
    any triggering UI time to dismiss, then encodes the grab to PNG and logs
    its dimensions and size.

    Args:
        monitor: Which monitor to capture. ``1`` is the primary monitor (the
            default); ``0`` captures all monitors combined.

    Returns:
        PNG-encoded image bytes suitable for passing to a vision provider.

    Side Effects:
        Blocks for the configured screenshot delay and logs an info line.
    """
    import mss
    import mss.tools

    time.sleep(config.SCREENSHOT_DELAY)

    with mss.mss() as sct:
        mon = sct.monitors[monitor]
        screenshot = sct.grab(mon)
        png_bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)
        logger.info(
            f"Screenshot captured: {screenshot.width}x{screenshot.height}px, "
            f"{len(png_bytes) // 1024}KB"
        )
        return png_bytes
