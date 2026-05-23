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
    """Return all currently enabled tool instances."""
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
    """
    Find and execute a tool by name.

    Args:
        name: Tool name as returned by BaseTool.name.
        arguments: Dict of arguments to pass to tool.execute().

    Returns:
        ToolResult from the tool.

    Raises:
        ToolError: If no tool with the given name is registered or enabled.
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
