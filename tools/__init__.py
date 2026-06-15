"""
tools/__init__.py

Tool registry and dispatcher.
get_tools() returns the list of enabled tools based on current config.
dispatch() executes a named tool with given arguments.
"""

import config
from tools.base import BaseTool, ToolResult
from tools.file_system import FileSystemTool
from tools.app_launcher import AppLauncherTool
from tools.terminal import TerminalTool
from core.exceptions import ToolError
from core.logger import logger


def get_tools() -> list[BaseTool]:
    """Return instances of all currently enabled tools.

    Reads the feature flags in ``config`` and constructs one instance per
    enabled tool. Optional tools are imported lazily so that disabling a
    feature also avoids importing its (sometimes heavy) dependencies.

    Returns:
        A list of ready-to-use ``BaseTool`` instances reflecting the current
        configuration.
    """
    tools = []
    if config.ENABLE_FILE_OPS:
        tools.append(FileSystemTool())
    if config.ENABLE_APP_LAUNCHER:
        tools.append(AppLauncherTool())
    if config.ENABLE_TERMINAL:
        tools.append(TerminalTool())
    if config.BROWSER_ENABLED:
        from tools.browser import BrowserTool
        tools.append(BrowserTool())
    if config.VISION_ENABLED:
        from tools.screen_reader import ScreenReaderTool
        tools.append(ScreenReaderTool())
    if config.MEMORY_ENABLED:
        from tools.memory_tool import MemoryTool
        tools.append(MemoryTool())
    if config.WEATHER_ENABLED:
        from tools.weather import WeatherTool
        tools.append(WeatherTool())
    if config.REMINDERS_ENABLED:
        from tools.reminder import ReminderTool
        tools.append(ReminderTool())
    return tools


def dispatch(name: str, arguments: dict) -> ToolResult:
    """Execute a registered tool by name.

    Looks up the enabled tool whose ``name`` matches and invokes its
    ``execute`` method with the provided arguments, logging the outcome.

    Args:
        name: Tool name as returned by ``BaseTool.name``.
        arguments: Keyword arguments forwarded to ``tool.execute``.

    Returns:
        The ``ToolResult`` produced by the tool.

    Raises:
        ToolError: If no enabled tool matches ``name``.
    """
    for tool in get_tools():
        if tool.name == name:
            logger.info(f"Dispatching tool: {name}({arguments})")
            result = tool.execute(**arguments)
            if result.success:
                logger.info(f"Tool result: success=True, output={result.output[:100]}")
            else:
                logger.info(f"Tool result: success=False, error={result.error}")
            return result
    raise ToolError(f"No tool named '{name}' is registered or enabled.")
