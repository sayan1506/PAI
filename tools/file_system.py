"""
tools/file_system.py

File system operations tool for PAI.
Handles create, read, rename, delete, move, copy, list, and search operations.
"""

import shutil
from pathlib import Path

import config
from core.logger import logger
from tools.base import BaseTool, ToolResult


class FileSystemTool(BaseTool):
    """Tool for managing files and folders on the local file system.

    Dispatches a single ``operation`` argument to a matching ``_op_*``
    handler, covering create, read, rename, delete, move, copy, list,
    search, open, and mkdir. The whole tool is gated by
    ``config.ENABLE_FILE_OPS``, and destructive operations (delete, move)
    may additionally require interactive user confirmation. Operations that
    touch the disk have side effects on the local file system.
    """

    @property
    def name(self) -> str:
        """Return the tool's unique identifier."""
        return "file_system"

    @property
    def description(self) -> str:
        """Return the LLM-facing description of this tool."""
        return (
            "Manage files and folders on the local file system. "
            "Use for creating, reading, renaming, deleting, moving, copying, "
            "listing, or searching files and directories. "
            "All paths may be absolute or relative to the user's home directory."
        )

    @property
    def parameters(self) -> dict:
        """Return the JSON Schema for this tool's arguments."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["create", "read", "rename", "delete", "move", "copy", "list", "search", "open", "mkdir"],
                    "description": "The file operation to perform.",
                },
                "path": {
                    "type": "string",
                    "description": "REQUIRED for all operations. The full absolute path to the file or folder. Always use full paths like C:\\Users\\sayan\\Desktop\\file.txt",
                },
                "new_name": {
                    "type": "string",
                    "description": "New filename for rename operations.",
                },
                "destination": {
                    "type": "string",
                    "description": "Destination path for move or copy.",
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write when creating a file. Use empty string for empty file.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern for search (e.g. '*.txt').",
                },
            },
            "required": ["operation", "path"],
        }

    def execute(self, **kwargs) -> ToolResult:
        """Execute a file system operation.

        Reads the ``operation`` argument and dispatches to the matching
        ``_op_<operation>`` handler. If the LLM supplied ``directory`` but
        not ``path``, ``directory`` is used as the path for convenience.

        Args:
            **kwargs: Operation arguments. ``operation`` selects the action;
                remaining keys (``path``, ``new_name``, ``destination``,
                ``content``, ``pattern``) depend on the operation.

        Returns:
            A ``ToolResult``. Failure is returned (not raised) when file
            operations are disabled, the operation is unknown, or the handler
            raises an exception.
        """
        if not config.ENABLE_FILE_OPS:
            return ToolResult(success=False, output="", error="File operations are disabled.")

        operation = kwargs.get("operation")

        # Normalize: if LLM passed 'directory' but not 'path', use directory as path
        if not kwargs.get("path") and kwargs.get("directory"):
            kwargs["path"] = kwargs["directory"]

        try:
            handler = getattr(self, f"_op_{operation}", None)
            if handler is None:
                return ToolResult(success=False, output="", error=f"Unknown operation: {operation}")
            return handler(**kwargs)
        except Exception as e:
            logger.error(f"FileSystemTool error: {e}")
            return ToolResult(success=False, output="", error=str(e))

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve a path string to an absolute, normalized ``Path``.

        Expands ``~``, treats relative paths as relative to the user's home
        directory, and resolves the result to an absolute path.

        Args:
            path_str: The raw path string from the caller.

        Returns:
            The resolved absolute ``Path``.
        """
        p = Path(path_str).expanduser()
        if not p.is_absolute():
            p = Path.home() / p
        return p.resolve()

    def _op_create(self, **kwargs) -> ToolResult:
        """Create a file, writing optional text content.

        Creates any missing parent directories, then writes ``content``
        (default empty) to the file as UTF-8.

        Args:
            **kwargs: Expects ``path`` and optionally ``content``.

        Returns:
            A ``ToolResult`` describing the created file, or an error if no
            path was provided.
        """
        path_str = kwargs.get("path", "")
        if not path_str:
            return ToolResult(success=False, output="", error="No path provided.")

        path = self._resolve_path(path_str)
        content = kwargs.get("content", "")

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        return ToolResult(success=True, output=f"Created file: {path}")

    def _op_read(self, **kwargs) -> ToolResult:
        """Read and return a text file's contents.

        Only regular files up to 8 KB are read; larger or non-file paths are
        rejected. Content is decoded as UTF-8.

        Args:
            **kwargs: Expects ``path``.

        Returns:
            A ``ToolResult`` whose ``output`` holds the file text, or an error
            if the path is missing, not a file, or too large.
        """
        path_str = kwargs.get("path", "")
        if not path_str:
            return ToolResult(success=False, output="", error="No path provided.")

        path = self._resolve_path(path_str)

        if not path.exists():
            return ToolResult(success=False, output="", error=f"File not found: {path}")
        if not path.is_file():
            return ToolResult(success=False, output="", error=f"Not a file: {path}")

        # Limit to 8KB
        if path.stat().st_size > 8192:
            return ToolResult(success=False, output="", error="File too large (max 8KB for reading).")

        content = path.read_text(encoding="utf-8")
        return ToolResult(success=True, output=content)

    def _op_rename(self, **kwargs) -> ToolResult:
        """Rename a file or folder in place.

        The target keeps its parent directory and takes ``new_name`` as its
        new name.

        Args:
            **kwargs: Expects ``path`` and ``new_name``.

        Returns:
            A ``ToolResult`` with the new path, or an error if arguments are
            missing or the source does not exist.
        """
        path_str = kwargs.get("path", "")
        new_name = kwargs.get("new_name", "")
        if not path_str:
            return ToolResult(success=False, output="", error="No path provided.")
        if not new_name:
            return ToolResult(success=False, output="", error="No new_name provided.")

        path = self._resolve_path(path_str)
        if not path.exists():
            return ToolResult(success=False, output="", error=f"Path not found: {path}")

        new_path = path.parent / new_name
        path.rename(new_path)
        return ToolResult(success=True, output=f"Renamed to: {new_path}")

    def _op_delete(self, **kwargs) -> ToolResult:
        """Delete a file or an empty folder.

        May prompt for interactive confirmation when the confirmation policy
        requires it; directories are only removed if empty.

        Args:
            **kwargs: Expects ``path``.

        Returns:
            A ``ToolResult`` confirming deletion, or an error if the path is
            missing, not found, the user declined, or the target cannot be
            deleted.

        Side Effects:
            Permanently removes the file or empty directory from disk.
        """
        path_str = kwargs.get("path", "")
        if not path_str:
            return ToolResult(success=False, output="", error="No path provided.")

        path = self._resolve_path(path_str)
        if not path.exists():
            return ToolResult(success=False, output="", error=f"Path not found: {path}")

        from core.confirmation import needs_file_confirmation, request_confirmation
        if needs_file_confirmation("delete"):
            kind = "folder" if path.is_dir() else "file"
            if not request_confirmation(f"Delete {kind}: {path}"):
                return ToolResult(
                    success=False, output="",
                    error="Deletion cancelled — user did not confirm.",
                )

        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()  # Only removes empty directories
        else:
            return ToolResult(success=False, output="", error=f"Cannot delete: {path}")

        return ToolResult(success=True, output=f"Deleted: {path}")

    def _op_move(self, **kwargs) -> ToolResult:
        """Move a file or folder to a new location.

        May prompt for interactive confirmation when the confirmation policy
        requires it.

        Args:
            **kwargs: Expects ``path`` and ``destination``.

        Returns:
            A ``ToolResult`` confirming the move, or an error if arguments are
            missing, the source does not exist, or the user declined.

        Side Effects:
            Relocates the source on disk via ``shutil.move``.
        """
        path_str = kwargs.get("path", "")
        destination = kwargs.get("destination", "")
        if not path_str:
            return ToolResult(success=False, output="", error="No path provided.")
        if not destination:
            return ToolResult(success=False, output="", error="No destination provided.")

        path = self._resolve_path(path_str)
        dest = self._resolve_path(destination)

        if not path.exists():
            return ToolResult(success=False, output="", error=f"Path not found: {path}")

        from core.confirmation import needs_file_confirmation, request_confirmation
        if needs_file_confirmation("move"):
            if not request_confirmation(f"Move {path} → {dest}"):
                return ToolResult(
                    success=False, output="",
                    error="Move cancelled — user did not confirm.",
                )

        shutil.move(str(path), str(dest))
        return ToolResult(success=True, output=f"Moved {path} to {dest}")

    def _op_copy(self, **kwargs) -> ToolResult:
        """Copy a file to a new location.

        Only regular files are supported; missing destination parent
        directories are created. Metadata is preserved via ``shutil.copy2``.

        Args:
            **kwargs: Expects ``path`` and ``destination``.

        Returns:
            A ``ToolResult`` confirming the copy, or an error if arguments are
            missing, the source does not exist, or it is not a file.
        """
        path_str = kwargs.get("path", "")
        destination = kwargs.get("destination", "")
        if not path_str:
            return ToolResult(success=False, output="", error="No path provided.")
        if not destination:
            return ToolResult(success=False, output="", error="No destination provided.")

        path = self._resolve_path(path_str)
        dest = self._resolve_path(destination)

        if not path.exists():
            return ToolResult(success=False, output="", error=f"Path not found: {path}")
        if not path.is_file():
            return ToolResult(success=False, output="", error="Copy only supports files.")

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(path), str(dest))
        return ToolResult(success=True, output=f"Copied {path} to {dest}")

    def _op_list(self, **kwargs) -> ToolResult:
        """List the entries of a directory.

        Entries are sorted and directories are prefixed with ``[DIR]``.

        Args:
            **kwargs: Expects ``path`` pointing to a directory.

        Returns:
            A ``ToolResult`` whose ``output`` is the newline-joined listing
            (or ``(empty directory)``), or an error if the path is missing,
            not found, or not a directory.
        """
        path_str = kwargs.get("path", "")
        if not path_str:
            return ToolResult(success=False, output="", error="No path provided.")

        path = self._resolve_path(path_str)
        if not path.exists():
            return ToolResult(success=False, output="", error=f"Directory not found: {path}")
        if not path.is_dir():
            return ToolResult(success=False, output="", error=f"Not a directory: {path}")

        entries = sorted(path.iterdir())
        names = [f"{'[DIR] ' if e.is_dir() else ''}{e.name}" for e in entries]
        output = "\n".join(names) if names else "(empty directory)"
        return ToolResult(success=True, output=output)

    def _op_search(self, **kwargs) -> ToolResult:
        """Search a directory for files matching a glob pattern.

        Results are capped at 50 entries.

        Args:
            **kwargs: Expects a directory (``path`` or ``directory``) and a
                glob ``pattern`` (e.g. ``'*.txt'``).

        Returns:
            A ``ToolResult`` listing matched paths (or a no-match message), or
            an error if the directory or pattern is missing or invalid.
        """
        directory = kwargs.get("path", "") or kwargs.get("directory", "")
        pattern = kwargs.get("pattern", "")
        if not directory:
            return ToolResult(success=False, output="", error="No path provided.")
        if not pattern:
            return ToolResult(success=False, output="", error="No pattern provided.")

        path = self._resolve_path(directory)
        if not path.exists() or not path.is_dir():
            return ToolResult(success=False, output="", error=f"Directory not found: {path}")

        matches = list(path.glob(pattern))
        if not matches:
            return ToolResult(success=True, output="No files found matching the pattern.")

        output = "\n".join(str(m) for m in matches[:50])  # Limit to 50 results
        return ToolResult(success=True, output=output)

    def _op_open(self, **kwargs) -> ToolResult:
        """Open a file or folder in its default application.

        Uses ``os.startfile`` on Windows and ``xdg-open`` elsewhere.

        Args:
            **kwargs: Expects ``path``.

        Returns:
            A ``ToolResult`` confirming the open, or an error if the path is
            missing, not found, or the launch failed.

        Side Effects:
            Spawns an external application to handle the path.
        """
        import subprocess as sp
        from utils.platform_utils import get_platform

        path_str = kwargs.get("path", "")
        if not path_str:
            return ToolResult(success=False, error="No path provided.")

        path = self._resolve_path(path_str)
        if not path.exists():
            return ToolResult(success=False, error=f"Path not found: {path}")

        try:
            if get_platform() == "windows":
                import os
                os.startfile(str(path))
            else:
                sp.Popen(
                    ["xdg-open", str(path)],
                    stdout=sp.DEVNULL,
                    stderr=sp.DEVNULL,
                )
            return ToolResult(success=True, output=f"Opened: {path}")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to open {path}: {e}")

    def _op_mkdir(self, **kwargs) -> ToolResult:
        """Create a folder, including any missing parent folders.

        Treats an already-existing directory as success; an existing file at
        the same path is an error.

        Args:
            **kwargs: Expects ``path``.

        Returns:
            A ``ToolResult`` confirming creation (or prior existence), or an
            error if the path is missing or a file occupies it.
        """
        path_str = kwargs.get("path", "")
        if not path_str:
            return ToolResult(success=False, error="No path provided.")

        path = self._resolve_path(path_str)

        if path.exists():
            if path.is_dir():
                return ToolResult(
                    success=True,
                    output=f"Folder already exists: {path}",
                )
            return ToolResult(
                success=False,
                error=f"A file already exists at that path: {path}",
            )

        path.mkdir(parents=True, exist_ok=True)
        return ToolResult(success=True, output=f"Created folder: {path}")
