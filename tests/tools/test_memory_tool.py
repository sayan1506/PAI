"""Tests for MemoryTool."""
from unittest.mock import patch, MagicMock
import pytest
import config
from tools.memory_tool import MemoryTool


@pytest.fixture
def tool():
    return MemoryTool()


@pytest.fixture(autouse=True)
def enable_memory(monkeypatch):
    monkeypatch.setattr(config, "MEMORY_ENABLED", True)


class TestMemoryToolDisabled:
    def test_disabled_returns_error(self, tool, monkeypatch):
        monkeypatch.setattr(config, "MEMORY_ENABLED", False)
        result = tool.execute(operation="remember_fact", key="k", value="v")
        assert result.success is False
        assert "disabled" in result.error.lower()


class TestUnknownOperation:
    def test_unknown_returns_error(self, tool):
        result = tool.execute(operation="fly_to_moon")
        assert result.success is False
        assert "Unknown operation" in result.error


class TestRememberFact:
    @patch("tools.memory_tool.store.upsert_fact")
    def test_stores_and_confirms(self, mock_upsert, tool):
        result = tool.execute(operation="remember_fact", key="user name", value="Sayan")
        assert result.success is True
        assert "Sayan" in result.output
        mock_upsert.assert_called_once()

    def test_missing_key_returns_error(self, tool):
        result = tool.execute(operation="remember_fact", value="Sayan")
        assert result.success is False
        assert "'key' is required" in result.error


class TestForgetFact:
    @patch("tools.memory_tool.store.delete_fact", return_value=True)
    def test_success(self, mock_delete, tool):
        result = tool.execute(operation="forget_fact", key="user_name")
        assert result.success is True
        assert "Forgotten" in result.output

    @patch("tools.memory_tool.store.delete_fact", return_value=False)
    def test_not_found(self, mock_delete, tool):
        result = tool.execute(operation="forget_fact", key="ghost")
        assert result.success is False
        assert "No fact" in result.error


class TestListFacts:
    @patch("tools.memory_tool.store.all_facts", return_value=[])
    def test_empty(self, mock_all, tool):
        result = tool.execute(operation="list_facts")
        assert result.success is True
        assert "No facts" in result.output

    @patch("tools.memory_tool.store.all_facts", return_value=[{"key": "user_name", "value": "Sayan", "source": "user"}])
    def test_with_data(self, mock_all, tool):
        result = tool.execute(operation="list_facts")
        assert result.success is True
        assert "user name: Sayan" in result.output


class TestDefineShortcut:
    @patch("tools.memory_tool.shortcut_manager.define", return_value="Shortcut 'work mode' saved.")
    def test_success(self, mock_define, tool):
        result = tool.execute(operation="define_shortcut", name="work mode", description="open VS Code and Spotify")
        assert result.success is True
        assert "saved" in result.output

    def test_missing_name(self, tool):
        result = tool.execute(operation="define_shortcut", description="open VS Code")
        assert result.success is False
        assert "'name' is required" in result.error


class TestDeleteShortcut:
    @patch("tools.memory_tool.shortcut_manager.remove", return_value="Shortcut 'work mode' deleted.")
    def test_success(self, mock_remove, tool):
        result = tool.execute(operation="delete_shortcut", name="work mode")
        assert result.success is True


class TestListShortcuts:
    @patch("tools.memory_tool.shortcut_manager.list_all", return_value=[])
    def test_empty(self, mock_list, tool):
        result = tool.execute(operation="list_shortcuts")
        assert result.success is True
        assert "No shortcuts" in result.output

    @patch("tools.memory_tool.shortcut_manager.list_all", return_value=[{"name": "work mode", "description": "open VS Code"}])
    def test_with_data(self, mock_list, tool):
        result = tool.execute(operation="list_shortcuts")
        assert result.success is True
        assert '"work mode": open VS Code' in result.output
