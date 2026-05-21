"""
utils/vision_utils.py

Screenshot capture using mss (cross-platform, no dependencies on X11/Win32 API).
"""

import time
import config
from core.logger import logger


def capture_screenshot(monitor: int = 1) -> bytes:
    """
    Capture the primary monitor and return raw PNG bytes.

    Args:
        monitor: 1 = primary monitor, 0 = all monitors combined

    Returns:
        PNG bytes ready to pass to a vision provider.
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
