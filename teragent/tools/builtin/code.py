"""teragent.tools.builtin.code — Code execution tool

Provides a tool for executing Python code snippets in a subprocess
with timeout control. This is a DESTRUCTIVE tool (highest safety level)
requiring the highest permission level.

Design reference:
    - Claude-Code: CodeExecutionTool with sandbox
    - OpenAI Agents SDK: CodeInterpreter tool
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Any

from teragent.core.types import ToolSafety
from teragent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

__all__ = [
    "CodeExecutionTool",
]


class CodeExecutionTool(BaseTool):
    """Execute Python code snippet and return output — DESTRUCTIVE safety level.

    Runs Python code in a subprocess with configurable timeout.
    Returns stdout, stderr, and return code.

    .. warning::
        This tool executes arbitrary code. It requires the highest
        permission level (AUTO, level >= 3) due to its DESTRUCTIVE
        safety classification.

    Usage::

        tool = CodeExecutionTool()
        result = await tool.execute({"code": "print('hello world')"})
        # result.data = {"stdout": "hello world\\n", "stderr": "", "returncode": 0}
    """

    name = "execute_code"
    description = (
        "Execute Python code snippet and return output. "
        "Use with caution — this tool can modify system state. "
        "Requires AUTO permission level."
    )
    _safety = ToolSafety.DESTRUCTIVE
    _concurrency_safe = False

    parameters_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 30)",
                "default": 30,
            },
        },
        "required": ["code"],
    }

    async def execute(
        self,
        params: dict,
        progress_callback: Any | None = None,
    ) -> ToolResult:
        """Execute Python code in a subprocess.

        The code is written to a temporary file and executed with
        ``python3 -c`` or as a script file for more complex code.

        Args:
            params: Must include 'code'. Optional 'timeout' (default 30).
            progress_callback: Not used.

        Returns:
            ToolResult with stdout, stderr, and returncode in data.
        """
        code = params.get("code", "")
        timeout = params.get("timeout", 30)

        if not code or not code.strip():
            return ToolResult(
                success=False,
                error="Parameter 'code' is required and must not be empty",
                safety=self._safety,
            )

        # Validate timeout
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            timeout = 30

        proc = None
        try:
            # Use subprocess execution for isolation
            # Write code to a temp file for complex code that doesn't fit in -c
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                prefix="teragent_exec_",
                delete=False,
                encoding="utf-8",
            ) as tmp_file:
                tmp_file.write(code)
                tmp_path = tmp_file.name

            proc = await asyncio.create_subprocess_exec(
                "python3",
                tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Prevent environment pollution
                env={
                    **os.environ,
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONUNBUFFERED": "1",
                },
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            return ToolResult(
                success=proc.returncode == 0,
                data={
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "returncode": proc.returncode,
                },
                error=stderr_str if proc.returncode != 0 else "",
                safety=self._safety,
            )

        except asyncio.TimeoutError:
            # Kill the process on timeout
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass

            return ToolResult(
                success=False,
                error=f"Code execution timed out after {timeout}s",
                safety=self._safety,
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="python3 interpreter not found",
                safety=self._safety,
            )
        except Exception as e:
            logger.error(f"CodeExecutionTool failed: {e}")
            return ToolResult(
                success=False,
                error=f"Code execution failed: {e}",
                safety=self._safety,
            )
        finally:
            # Clean up temp file
            if "tmp_path" in locals():
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def describe_usage(self, params: dict) -> str:
        code = params.get("code", "")
        preview = code[:40].replace("\n", "\\n")
        if len(code) > 40:
            preview += "..."
        return f"执行代码: {preview}"
