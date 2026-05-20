"""
Unit tests for the FileSystemTool.

Tests verify all file system operations (create, read, rename, delete,
move, copy, list, search) work correctly, and that the tool respects
the ENABLE_FILE_OPS config flag.

Uses pytest with monkeypatch and tmp_path for isolation.
"""

import importlib
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import config
from tools.file_system import FileSystemTool
from tools.base import ToolResult


@pytest.fixture
def tool():
    """Return a fresh FileSystemTool instance."""
    return FileSystemTool()


@pytest.fixture(autouse=True)
def enable_file_ops(monkeypatch):
    """Ensure file ops are enabled by default for all tests."""
    monkeypatch.setattr(config, "ENABLE_FILE_OPS", True)


class TestCreateFile:
    """Tests for the create operation."""

    def test_create_file_success(self, tool, tmp_path):
        """Create a file and verify output contains the filename."""
        target = tmp_path / "hello.txt"
        result = tool.execute(operation="create", path=str(target), content="hello world")

        assert result.success is True
        assert "hello.txt" in result.output
        assert target.read_text(encoding="utf-8") == "hello world"

    def test_create_file_disabled(self, tool, monkeypatch):
        """ENABLE_FILE_OPS=False returns ToolResult with success=False."""
        monkeypatch.setattr(config, "ENABLE_FILE_OPS", False)

        result = tool.execute(operation="create", path="/tmp/test.txt", content="data")

        assert result.success is False
        assert "disabled" in result.error.lower()

    def test_create_file_creates_parent_dirs(self, tool, tmp_path):
        """Create operation creates parent directories if they don't exist."""
        target = tmp_path / "sub" / "dir" / "file.txt"
        result = tool.execute(operation="create", path=str(target), content="nested")

        assert result.success is True
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "nested"

    def test_create_file_no_path(self, tool):
        """Create with empty path returns error."""
        result = tool.execute(operation="create", path="")

        assert result.success is False
        assert "no path" in result.error.lower()


class TestReadFile:
    """Tests for the read operation."""

    def test_read_file_returns_content(self, tool, tmp_path):
        """Read operation returns file content."""
        target = tmp_path / "data.txt"
        target.write_text("file content here", encoding="utf-8")

        result = tool.execute(operation="read", path=str(target))

        assert result.success is True
        assert result.output == "file content here"

    def test_read_file_not_found(self, tool, tmp_path):
        """Read of non-existent file returns success=False."""
        target = tmp_path / "nonexistent.txt"

        result = tool.execute(operation="read", path=str(target))

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_read_file_no_path(self, tool):
        """Read with empty path returns error."""
        result = tool.execute(operation="read", path="")

        assert result.success is False


class TestRenameFile:
    """Tests for the rename operation."""

    def test_rename_file_success(self, tool, tmp_path):
        """Rename a file and verify the new name exists."""
        original = tmp_path / "old_name.txt"
        original.write_text("content", encoding="utf-8")

        result = tool.execute(operation="rename", path=str(original), new_name="new_name.txt")

        assert result.success is True
        assert "new_name.txt" in result.output
        assert (tmp_path / "new_name.txt").exists()
        assert not original.exists()

    def test_rename_file_not_found(self, tool, tmp_path):
        """Rename of non-existent file returns error."""
        target = tmp_path / "ghost.txt"

        result = tool.execute(operation="rename", path=str(target), new_name="new.txt")

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_rename_file_no_new_name(self, tool, tmp_path):
        """Rename without new_name returns error."""
        target = tmp_path / "file.txt"
        target.write_text("x", encoding="utf-8")

        result = tool.execute(operation="rename", path=str(target), new_name="")

        assert result.success is False
        assert "new_name" in result.error.lower()


class TestDeleteFile:
    """Tests for the delete operation."""

    def test_delete_file_success(self, tool, tmp_path):
        """Delete a file and verify it no longer exists."""
        target = tmp_path / "to_delete.txt"
        target.write_text("bye", encoding="utf-8")

        result = tool.execute(operation="delete", path=str(target))

        assert result.success is True
        assert "deleted" in result.output.lower()
        assert not target.exists()

    def test_delete_file_not_found(self, tool, tmp_path):
        """Delete of non-existent file returns error."""
        target = tmp_path / "nope.txt"

        result = tool.execute(operation="delete", path=str(target))

        assert result.success is False
        assert "not found" in result.error.lower()


class TestListDirectory:
    """Tests for the list operation."""

    def test_list_directory(self, tool, tmp_path):
        """List directory returns names of entries."""
        (tmp_path / "alpha.txt").write_text("a", encoding="utf-8")
        (tmp_path / "beta.txt").write_text("b", encoding="utf-8")
        (tmp_path / "subdir").mkdir()

        result = tool.execute(operation="list", path=str(tmp_path))

        assert result.success is True
        assert "alpha.txt" in result.output
        assert "beta.txt" in result.output
        assert "subdir" in result.output

    def test_list_directory_not_found(self, tool, tmp_path):
        """List of non-existent directory returns error."""
        target = tmp_path / "no_such_dir"

        result = tool.execute(operation="list", path=str(target))

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_list_empty_directory(self, tool, tmp_path):
        """List of empty directory returns empty indicator."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = tool.execute(operation="list", path=str(empty_dir))

        assert result.success is True
        assert "empty" in result.output.lower()


class TestSearchFiles:
    """Tests for the search operation."""

    def test_search_files(self, tool, tmp_path):
        """Search with glob pattern returns matching files."""
        (tmp_path / "report.txt").write_text("r", encoding="utf-8")
        (tmp_path / "data.txt").write_text("d", encoding="utf-8")
        (tmp_path / "image.png").write_text("i", encoding="utf-8")

        result = tool.execute(operation="search", directory=str(tmp_path), pattern="*.txt")

        assert result.success is True
        assert "report.txt" in result.output
        assert "data.txt" in result.output
        assert "image.png" not in result.output

    def test_search_no_matches(self, tool, tmp_path):
        """Search with no matches returns informative message."""
        (tmp_path / "file.txt").write_text("x", encoding="utf-8")

        result = tool.execute(operation="search", directory=str(tmp_path), pattern="*.csv")

        assert result.success is True
        assert "no files found" in result.output.lower()

    def test_search_no_directory(self, tool):
        """Search without directory returns error."""
        result = tool.execute(operation="search", path="", pattern="*.txt")

        assert result.success is False
        assert "no path" in result.error.lower()


class TestMoveFile:
    """Tests for the move operation."""

    def test_move_file(self, tool, tmp_path):
        """Move a file to a new location."""
        source = tmp_path / "source.txt"
        source.write_text("moving", encoding="utf-8")
        dest = tmp_path / "dest" / "moved.txt"
        dest.parent.mkdir(parents=True, exist_ok=True)

        result = tool.execute(operation="move", path=str(source), destination=str(dest))

        assert result.success is True
        assert "moved" in result.output.lower()
        assert not source.exists()
        assert dest.read_text(encoding="utf-8") == "moving"

    def test_move_file_not_found(self, tool, tmp_path):
        """Move of non-existent file returns error."""
        source = tmp_path / "ghost.txt"
        dest = tmp_path / "dest.txt"

        result = tool.execute(operation="move", path=str(source), destination=str(dest))

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_move_file_no_destination(self, tool, tmp_path):
        """Move without destination returns error."""
        source = tmp_path / "file.txt"
        source.write_text("x", encoding="utf-8")

        result = tool.execute(operation="move", path=str(source), destination="")

        assert result.success is False
        assert "no destination" in result.error.lower()


class TestCopyFile:
    """Tests for the copy operation."""

    def test_copy_file(self, tool, tmp_path):
        """Copy a file to a new location."""
        source = tmp_path / "original.txt"
        source.write_text("copy me", encoding="utf-8")
        dest = tmp_path / "copy.txt"

        result = tool.execute(operation="copy", path=str(source), destination=str(dest))

        assert result.success is True
        assert "copied" in result.output.lower()
        assert source.exists()  # Original still exists
        assert dest.read_text(encoding="utf-8") == "copy me"

    def test_copy_file_not_found(self, tool, tmp_path):
        """Copy of non-existent file returns error."""
        source = tmp_path / "nope.txt"
        dest = tmp_path / "dest.txt"

        result = tool.execute(operation="copy", path=str(source), destination=str(dest))

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_copy_file_no_destination(self, tool, tmp_path):
        """Copy without destination returns error."""
        source = tmp_path / "file.txt"
        source.write_text("x", encoding="utf-8")

        result = tool.execute(operation="copy", path=str(source), destination="")

        assert result.success is False
        assert "no destination" in result.error.lower()


class TestUnknownOperation:
    """Tests for unknown/invalid operations."""

    def test_unknown_operation(self, tool):
        """Unknown operation returns success=False with descriptive error."""
        result = tool.execute(operation="explode")

        assert result.success is False
        assert "unknown operation" in result.error.lower()
        assert "explode" in result.error.lower()
