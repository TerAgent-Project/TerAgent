"""teragent.tools.builtin.file — File operation tools

Provides four file system tools:
  - ReadFileTool: Read file contents (READ_ONLY, concurrency safe)
  - WriteFileTool: Write file contents with atomic write (SAFE_WRITE, not concurrency safe)
  - ListDirectoryTool: List directory contents (READ_ONLY, concurrency safe)
  - SearchFilesTool: Search for patterns in files (READ_ONLY, concurrency safe)

All tools extend BaseTool and follow the safety/concurrency model.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from teragent.core.types import ToolSafety
from teragent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

__all__ = [
    "ReadFileTool",
    "WriteFileTool",
    "ListDirectoryTool",
    "SearchFilesTool",
]


class ReadFileTool(BaseTool):
    """Read file contents — READ_ONLY, concurrency safe.

    Supports reading the entire file or a specific line range.
    Returns file content as a string in the ToolResult data.

    Usage::

        tool = ReadFileTool()
        result = await tool.execute({"path": "/path/to/file.py"})
        result = await tool.execute({"path": "/path/to/file.py", "start_line": 10, "end_line": 50})
    """

    name = "read_file"
    description = "Read the contents of a file. Supports reading specific line ranges."
    _safety = ToolSafety.READ_ONLY
    _concurrency_safe = True

    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read",
            },
            "start_line": {
                "type": "integer",
                "description": "Starting line number (1-based, inclusive). If omitted, starts from the beginning.",
            },
            "end_line": {
                "type": "integer",
                "description": "Ending line number (1-based, inclusive). If omitted, reads to the end.",
            },
        },
        "required": ["path"],
    }

    async def execute(
        self,
        params: dict,
        progress_callback: Any | None = None,
    ) -> ToolResult:
        """Read file contents, optionally limited to a line range.

        Args:
            params: Must include 'path'. Optional 'start_line' and 'end_line'.
            progress_callback: Not used.

        Returns:
            ToolResult with file content in data['content'] and line count in data['lines'].
        """
        path = params.get("path", "")
        start_line = params.get("start_line")
        end_line = params.get("end_line")

        if not path:
            return ToolResult(success=False, error="Parameter 'path' is required")

        try:
            file_path = Path(path).expanduser().resolve()

            if not file_path.exists():
                return ToolResult(
                    success=False,
                    error=f"File not found: {file_path}",
                    safety=self._safety,
                )

            if not file_path.is_file():
                return ToolResult(
                    success=False,
                    error=f"Path is not a file: {file_path}",
                    safety=self._safety,
                )

            # Read file content
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = file_path.read_text(encoding="latin-1")

            lines = content.splitlines()

            # Apply line range if specified
            if start_line is not None or end_line is not None:
                start = (start_line or 1) - 1  # Convert to 0-based
                end = end_line or len(lines)
                selected_lines = lines[start:end]
                # Add line numbers for clarity
                numbered = []
                for i, line in enumerate(selected_lines, start=start + 1):
                    numbered.append(f"{i:6d}\t{line}")
                result_content = "\n".join(numbered)
                total_lines = len(lines)
                shown_lines = len(selected_lines)
            else:
                result_content = content
                total_lines = len(lines)
                shown_lines = total_lines

            # Truncate very large files
            max_chars = 100_000
            truncated = False
            if len(result_content) > max_chars:
                result_content = result_content[:max_chars] + "\n... [truncated]"
                truncated = True

            data = {
                "content": result_content,
                "path": str(file_path),
                "lines": total_lines,
                "shown_lines": shown_lines,
            }
            if truncated:
                data["truncated"] = True

            return ToolResult(
                success=True,
                data=data,
                safety=self._safety,
            )

        except PermissionError:
            return ToolResult(
                success=False,
                error=f"Permission denied: {path}",
                safety=self._safety,
            )
        except Exception as e:
            logger.error(f"ReadFileTool failed for {path}: {e}")
            return ToolResult(
                success=False,
                error=f"Failed to read file: {e}",
                safety=self._safety,
            )

    def describe_usage(self, params: dict) -> str:
        path = params.get("path", "?")
        start = params.get("start_line")
        end = params.get("end_line")
        if start or end:
            return f"读取 {path} (行 {start or '1'}-{end or '末尾'})"
        return f"读取 {path}"


class WriteFileTool(BaseTool):
    """Write file contents with atomic write — SAFE_WRITE, not concurrency safe.

    Uses tempfile + os.replace for atomic writes, ensuring that
    the file is either fully written or not modified at all.

    Usage::

        tool = WriteFileTool()
        result = await tool.execute({"path": "/path/to/file.py", "content": "print('hello')"})
    """

    name = "write_file"
    description = "Write content to a file. Creates the file if it does not exist, overwrites if it does. Uses atomic write for safety."
    _safety = ToolSafety.SAFE_WRITE
    _concurrency_safe = False

    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
            "append": {
                "type": "boolean",
                "description": "If true, append to the file instead of overwriting",
                "default": False,
            },
        },
        "required": ["path", "content"],
    }

    async def execute(
        self,
        params: dict,
        progress_callback: Any | None = None,
    ) -> ToolResult:
        """Write content to a file using atomic write.

        Args:
            params: Must include 'path' and 'content'. Optional 'append' (default False).
            progress_callback: Not used.

        Returns:
            ToolResult indicating success or failure.
        """
        path = params.get("path", "")
        content = params.get("content", "")
        append = params.get("append", False)

        if not path:
            return ToolResult(success=False, error="Parameter 'path' is required")
        if "content" not in params:
            return ToolResult(success=False, error="Parameter 'content' is required")

        try:
            file_path = Path(path).expanduser().resolve()

            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if append:
                # Append mode — not atomic, but straightforward
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(content)
            else:
                # Atomic write using tempfile + os.replace
                # This ensures the file is either fully written or not modified
                dir_path = file_path.parent
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=dir_path,
                    prefix=".teragent_write_",
                    suffix=".tmp",
                    delete=False,
                ) as tmp_file:
                    tmp_file.write(content)
                    tmp_path = tmp_file.name

                # Atomic replace
                os.replace(tmp_path, file_path)

            # Get file info for result
            file_size = file_path.stat().st_size

            return ToolResult(
                success=True,
                data={
                    "path": str(file_path),
                    "size": file_size,
                    "append": append,
                },
                safety=self._safety,
            )

        except PermissionError:
            return ToolResult(
                success=False,
                error=f"Permission denied: {path}",
                safety=self._safety,
            )
        except OSError as e:
            # Clean up temp file if it exists
            if "tmp_path" in locals():
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            logger.error(f"WriteFileTool failed for {path}: {e}")
            return ToolResult(
                success=False,
                error=f"Failed to write file: {e}",
                safety=self._safety,
            )
        except Exception as e:
            logger.error(f"WriteFileTool failed for {path}: {e}")
            return ToolResult(
                success=False,
                error=f"Failed to write file: {e}",
                safety=self._safety,
            )

    def describe_usage(self, params: dict) -> str:
        path = params.get("path", "?")
        content_len = len(params.get("content", ""))
        return f"写入 {path} ({content_len} 字符)"


class ListDirectoryTool(BaseTool):
    """List directory contents — READ_ONLY, concurrency safe.

    Returns a list of files and directories in the specified path,
    with file sizes and types.

    Usage::

        tool = ListDirectoryTool()
        result = await tool.execute({"path": "/path/to/directory"})
    """

    name = "list_directory"
    description = "List the contents of a directory. Returns file names, sizes, and types."
    _safety = ToolSafety.READ_ONLY
    _concurrency_safe = True

    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the directory to list",
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern to filter results (e.g., '*.py')",
            },
            "recursive": {
                "type": "boolean",
                "description": "If true, list files recursively",
                "default": False,
            },
        },
        "required": ["path"],
    }

    async def execute(
        self,
        params: dict,
        progress_callback: Any | None = None,
    ) -> ToolResult:
        """List directory contents.

        Args:
            params: Must include 'path'. Optional 'pattern' and 'recursive'.
            progress_callback: Not used.

        Returns:
            ToolResult with directory listing in data['entries'].
        """
        path = params.get("path", "")
        pattern = params.get("pattern", "*")
        recursive = params.get("recursive", False)

        if not path:
            return ToolResult(success=False, error="Parameter 'path' is required")

        try:
            dir_path = Path(path).expanduser().resolve()

            if not dir_path.exists():
                return ToolResult(
                    success=False,
                    error=f"Directory not found: {dir_path}",
                    safety=self._safety,
                )

            if not dir_path.is_dir():
                return ToolResult(
                    success=False,
                    error=f"Path is not a directory: {dir_path}",
                    safety=self._safety,
                )

            entries = []
            if recursive:
                glob_result = dir_path.rglob(pattern)
            else:
                glob_result = dir_path.glob(pattern)

            for entry in sorted(glob_result):
                try:
                    rel_path = entry.relative_to(dir_path)
                    is_dir = entry.is_dir()
                    size = 0
                    if not is_dir:
                        try:
                            size = entry.stat().st_size
                        except OSError:
                            pass

                    entries.append({
                        "name": entry.name,
                        "path": str(rel_path),
                        "type": "directory" if is_dir else "file",
                        "size": size,
                    })
                except (OSError, ValueError):
                    continue

                # Limit number of entries
                if len(entries) >= 1000:
                    break

            return ToolResult(
                success=True,
                data={
                    "path": str(dir_path),
                    "entries": entries,
                    "total": len(entries),
                    "truncated": len(entries) >= 1000,
                },
                safety=self._safety,
            )

        except PermissionError:
            return ToolResult(
                success=False,
                error=f"Permission denied: {path}",
                safety=self._safety,
            )
        except Exception as e:
            logger.error(f"ListDirectoryTool failed for {path}: {e}")
            return ToolResult(
                success=False,
                error=f"Failed to list directory: {e}",
                safety=self._safety,
            )

    def describe_usage(self, params: dict) -> str:
        path = params.get("path", "?")
        return f"列出目录 {path}"


class SearchFilesTool(BaseTool):
    """Search for patterns in files — READ_ONLY, concurrency safe.

    Uses ripgrep (rg) if available for fast searching, falls back
    to a Python-based grep implementation otherwise.

    Usage::

        tool = SearchFilesTool()
        result = await tool.execute({"path": "/path/to/search", "pattern": "TODO"})
    """

    name = "search_files"
    description = "Search for a text pattern in files. Uses ripgrep if available, falls back to Python grep."
    _safety = ToolSafety.READ_ONLY
    _concurrency_safe = True

    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory or file path to search in",
            },
            "pattern": {
                "type": "string",
                "description": "Search pattern (regular expression supported)",
            },
            "file_pattern": {
                "type": "string",
                "description": "File glob pattern to filter (e.g., '*.py')",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 50,
            },
        },
        "required": ["path", "pattern"],
    }

    def __init__(self) -> None:
        super().__init__()
        self._has_rg = self._check_ripgrep()

    @staticmethod
    def _check_ripgrep() -> bool:
        """Check if ripgrep (rg) is available on the system."""
        try:
            import shutil
            return shutil.which("rg") is not None
        except Exception:
            return False

    async def execute(
        self,
        params: dict,
        progress_callback: Any | None = None,
    ) -> ToolResult:
        """Search for a pattern in files.

        Args:
            params: Must include 'path' and 'pattern'. Optional 'file_pattern' and 'max_results'.
            progress_callback: Not used.

        Returns:
            ToolResult with matching lines in data['matches'].
        """
        path = params.get("path", "")
        pattern = params.get("pattern", "")
        file_pattern = params.get("file_pattern")
        max_results = params.get("max_results", 50)

        if not path:
            return ToolResult(success=False, error="Parameter 'path' is required")
        if not pattern:
            return ToolResult(success=False, error="Parameter 'pattern' is required")

        try:
            search_path = Path(path).expanduser().resolve()

            if not search_path.exists():
                return ToolResult(
                    success=False,
                    error=f"Path not found: {search_path}",
                    safety=self._safety,
                )

            # Try ripgrep first, fall back to Python implementation
            if self._has_rg:
                result = await self._search_ripgrep(
                    search_path, pattern, file_pattern, max_results
                )
            else:
                result = await self._search_python(
                    search_path, pattern, file_pattern, max_results
                )

            return result

        except Exception as e:
            logger.error(f"SearchFilesTool failed for {path}: {e}")
            return ToolResult(
                success=False,
                error=f"Failed to search files: {e}",
                safety=self._safety,
            )

    async def _search_ripgrep(
        self,
        search_path: Path,
        pattern: str,
        file_pattern: str | None,
        max_results: int,
    ) -> ToolResult:
        """Search using ripgrep (rg) subprocess."""
        cmd = [
            "rg",
            "--no-heading",
            "--line-number",
            "--color=never",
            "--max-count", str(max_results),
            pattern,
            str(search_path),
        ]

        if file_pattern:
            cmd.extend(["--glob", file_pattern])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=30.0
            )

            # rg returns exit code 1 when no matches found (not an error)
            if proc.returncode not in (0, 1):
                # Fall back to Python search on rg error
                return await self._search_python(
                    search_path, pattern, file_pattern, max_results
                )

            output = stdout.decode("utf-8", errors="replace")
            matches = self._parse_rg_output(output, max_results)

            return ToolResult(
                success=True,
                data={
                    "matches": matches,
                    "total": len(matches),
                    "pattern": pattern,
                    "path": str(search_path),
                    "engine": "ripgrep",
                },
                safety=self._safety,
            )

        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error="Search timed out (30s)",
                safety=self._safety,
            )
        except FileNotFoundError:
            # rg not found, fall back
            self._has_rg = False
            return await self._search_python(
                search_path, pattern, file_pattern, max_results
            )

    async def _search_python(
        self,
        search_path: Path,
        pattern: str,
        file_pattern: str | None,
        max_results: int,
    ) -> ToolResult:
        """Search using Python re module (fallback when rg is not available)."""
        import re
        import fnmatch

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return ToolResult(
                success=False,
                error=f"Invalid regex pattern: {e}",
                safety=self._safety,
            )

        matches = []
        paths_to_search = [search_path]

        if search_path.is_file():
            paths_to_search = [search_path]
        else:
            paths_to_search = list(search_path.rglob("*"))

        for file_path in paths_to_search:
            if len(matches) >= max_results:
                break

            if not file_path.is_file():
                continue

            # Apply file pattern filter
            if file_pattern and not fnmatch.fnmatch(file_path.name, file_pattern):
                continue

            # Skip binary files and common non-text directories
            if any(part.startswith(".") for part in file_path.parts):
                continue
            if any(part in ("node_modules", "__pycache__", ".git", "venv") for part in file_path.parts):
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(content.splitlines(), 1):
                    if len(matches) >= max_results:
                        break
                    if regex.search(line):
                        matches.append({
                            "file": str(file_path),
                            "line": i,
                            "text": line[:200],  # Truncate long lines
                        })
            except (OSError, UnicodeDecodeError):
                continue

        return ToolResult(
            success=True,
            data={
                "matches": matches,
                "total": len(matches),
                "pattern": pattern,
                "path": str(search_path),
                "engine": "python",
            },
            safety=self._safety,
        )

    @staticmethod
    def _parse_rg_output(output: str, max_results: int) -> list[dict]:
        """Parse ripgrep output into structured match list."""
        matches = []
        for line in output.splitlines()[:max_results]:
            # rg format: file:line:content
            parts = line.split(":", 2)
            if len(parts) >= 3:
                matches.append({
                    "file": parts[0],
                    "line": int(parts[1]) if parts[1].isdigit() else 0,
                    "text": parts[2][:200],
                })
            elif len(parts) == 2:
                matches.append({
                    "file": parts[0],
                    "line": 0,
                    "text": parts[1][:200],
                })
        return matches

    def describe_usage(self, params: dict) -> str:
        pattern = params.get("pattern", "?")
        path = params.get("path", "?")
        return f"搜索 {path} 中的 '{pattern}'"
