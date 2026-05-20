"""Quick smoke test for AppLauncherTool import and interface."""
from tools.app_launcher import AppLauncherTool
from tools.base import BaseTool, ToolResult

t = AppLauncherTool()

# Verify it's a BaseTool
assert isinstance(t, BaseTool), "Must be a BaseTool instance"

# Verify properties
assert t.name == "app_launcher"
assert "launch" in t.description.lower()
assert "operation" in t.parameters["properties"]
assert "app_name" in t.parameters["properties"]
assert t.parameters["properties"]["operation"]["enum"] == ["launch", "close", "list_running", "is_running"]

# Verify to_function_spec
spec = t.to_function_spec()
assert spec["name"] == "app_launcher"
assert "description" in spec
assert "parameters" in spec

# Verify disabled check
import config
original = config.ENABLE_APP_LAUNCHER
config.ENABLE_APP_LAUNCHER = False
result = t.execute(operation="list_running")
assert result.success is False
assert "disabled" in result.error.lower()
config.ENABLE_APP_LAUNCHER = original

# Verify unknown operation
result = t.execute(operation="unknown_op")
assert result.success is False
assert "Unknown operation" in result.error

# Verify is_running works (just check it returns a ToolResult)
result = t.execute(operation="is_running", app_name="nonexistent_app_xyz")
assert isinstance(result, ToolResult)
assert result.success is True
assert "not running" in result.output.lower()

# Verify list_running works
result = t.execute(operation="list_running")
assert isinstance(result, ToolResult)
assert result.success is True

print("All checks passed!")
