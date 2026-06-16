"""内置工具集 — 开箱即用的常用工具

Provides ready-to-use built-in tools for common operations:
  - File tools: read, write, list directory, search
  - Code execution: Python code runner
  - Web tools: search and scrape
  - Analysis tools: code structure and semantic search

Usage::

    from teragent.tools.builtin import all_builtin_tools

    # Get all built-in tool instances
    tools = all_builtin_tools()

    # Or import individual tools
    from teragent.tools.builtin import ReadFileTool, WriteFileTool
"""

from __future__ import annotations

from teragent.tools.builtin.analysis import AnalyzeCodeTool, SearchCodeSemanticTool
from teragent.tools.builtin.code import CodeExecutionTool
from teragent.tools.builtin.file import (
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from teragent.tools.builtin.web import WebScrapeTool, WebSearchTool

from teragent.tools.base import BaseTool


def all_builtin_tools() -> list[BaseTool]:
    """返回所有内置工具实例

    Returns:
        List of all built-in tool instances, ready for registration
        with a ToolRegistry.
    """
    return [
        ReadFileTool(),
        WriteFileTool(),
        ListDirectoryTool(),
        SearchFilesTool(),
        CodeExecutionTool(),
        WebSearchTool(),
        WebScrapeTool(),
        AnalyzeCodeTool(),
        SearchCodeSemanticTool(),
    ]


__all__ = [
    "ReadFileTool",
    "WriteFileTool",
    "ListDirectoryTool",
    "SearchFilesTool",
    "CodeExecutionTool",
    "WebSearchTool",
    "WebScrapeTool",
    "AnalyzeCodeTool",
    "SearchCodeSemanticTool",
    "all_builtin_tools",
]
