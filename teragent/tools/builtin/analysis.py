"""teragent.tools.builtin.analysis — Code analysis tools

Provides two code analysis tools:
  - AnalyzeCodeTool: Analyze code structure and quality (READ_ONLY)
  - SearchCodeSemanticTool: Semantic code search (READ_ONLY)

Both tools are lightweight implementations with optional dependencies.
When optional dependencies (tree-sitter, etc.) are not installed,
they gracefully degrade to simpler text-based analysis.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

from teragent.core.types import ToolSafety
from teragent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

__all__ = [
    "AnalyzeCodeTool",
    "SearchCodeSemanticTool",
]


class AnalyzeCodeTool(BaseTool):
    """Analyze code structure and quality — READ_ONLY, concurrency safe.

    Provides code analysis capabilities including:
      - Function/class counting and listing
      - Import analysis
      - Complexity metrics (simple)
      - Style issue detection (basic)

    Uses Python's ast module for Python files and regex-based
    analysis for other languages. Falls back gracefully for
    binary or unreadable files.

    Usage::

        tool = AnalyzeCodeTool()
        result = await tool.execute({"path": "/path/to/code.py"})
    """

    name = "analyze_code"
    description = (
        "Analyze code structure, quality, and metrics. "
        "Supports Python (full AST analysis) and other languages (basic analysis). "
        "Returns function/class listings, imports, and basic complexity metrics."
    )
    _safety = ToolSafety.READ_ONLY
    _concurrency_safe = True

    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file or directory to analyze",
            },
            "analysis_type": {
                "type": "string",
                "description": "Type of analysis: 'structure' (default), 'quality', or 'full'",
                "default": "structure",
            },
        },
        "required": ["path"],
    }

    async def execute(
        self,
        params: dict,
        progress_callback: Any | None = None,
    ) -> ToolResult:
        """Analyze code structure and quality.

        Args:
            params: Must include 'path'. Optional 'analysis_type'.
            progress_callback: Not used.

        Returns:
            ToolResult with analysis results in data.
        """
        path = params.get("path", "")
        analysis_type = params.get("analysis_type", "structure")

        if not path:
            return ToolResult(
                success=False,
                error="Parameter 'path' is required",
                safety=self._safety,
            )

        try:
            file_path = Path(path).expanduser().resolve()

            if not file_path.exists():
                return ToolResult(
                    success=False,
                    error=f"Path not found: {file_path}",
                    safety=self._safety,
                )

            if file_path.is_dir():
                return await self._analyze_directory(file_path, analysis_type)
            elif file_path.is_file():
                return await self._analyze_file(file_path, analysis_type)
            else:
                return ToolResult(
                    success=False,
                    error=f"Path is neither file nor directory: {file_path}",
                    safety=self._safety,
                )

        except Exception as e:
            logger.error(f"AnalyzeCodeTool failed for {path}: {e}")
            return ToolResult(
                success=False,
                error=f"Code analysis failed: {e}",
                safety=self._safety,
            )

    async def _analyze_file(
        self,
        file_path: Path,
        analysis_type: str,
    ) -> ToolResult:
        """Analyze a single code file."""
        # Check if it's a Python file for AST analysis
        if file_path.suffix == ".py":
            return await self._analyze_python_file(file_path, analysis_type)
        else:
            return await self._analyze_generic_file(file_path, analysis_type)

    async def _analyze_python_file(
        self,
        file_path: Path,
        analysis_type: str,
    ) -> ToolResult:
        """Analyze a Python file using AST."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return ToolResult(
                success=False,
                error=f"Cannot read file: {e}",
                safety=self._safety,
            )

        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            return ToolResult(
                success=False,
                error=f"Syntax error in {file_path}: {e}",
                safety=self._safety,
            )

        # Extract structure
        functions = []
        classes = []
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                functions.append({
                    "name": node.name,
                    "line": node.lineno,
                    "args": [arg.arg for arg in node.args.args],
                    "decorators": [
                        d.id if isinstance(d, ast.Name) else str(d)
                        for d in node.decorator_list
                    ],
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                })
            elif isinstance(node, ast.ClassDef):
                methods = [
                    n.name for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                classes.append({
                    "name": node.name,
                    "line": node.lineno,
                    "methods": methods,
                    "bases": [
                        b.id if isinstance(b, ast.Name) else str(b)
                        for b in node.bases
                    ],
                })
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)

        lines = content.splitlines()
        total_lines = len(lines)
        code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))
        comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
        blank_lines = total_lines - code_lines - comment_lines

        data = {
            "path": str(file_path),
            "language": "python",
            "analysis_type": analysis_type,
            "structure": {
                "functions": functions,
                "classes": classes,
                "imports": imports,
            },
            "metrics": {
                "total_lines": total_lines,
                "code_lines": code_lines,
                "comment_lines": comment_lines,
                "blank_lines": blank_lines,
                "function_count": len(functions),
                "class_count": len(classes),
                "import_count": len(imports),
            },
        }

        # Add quality analysis if requested
        if analysis_type in ("quality", "full"):
            issues = self._check_python_quality(tree, content)
            data["quality"] = {
                "issues": issues,
                "issue_count": len(issues),
            }

        return ToolResult(
            success=True,
            data=data,
            safety=self._safety,
        )

    @staticmethod
    def _check_python_quality(tree: ast.AST, content: str) -> list[dict]:
        """Basic Python quality checks."""
        issues = []

        # Check for bare except clauses
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append({
                        "type": "bare_except",
                        "line": node.lineno,
                        "message": "Bare 'except:' catches all exceptions including SystemExit and KeyboardInterrupt",
                    })

        # Check for overly long lines
        for i, line in enumerate(content.splitlines(), 1):
            if len(line) > 120:
                issues.append({
                    "type": "long_line",
                    "line": i,
                    "message": f"Line exceeds 120 characters ({len(line)} chars)",
                })

        # Check for TODO/FIXME/HACK comments
        for i, line in enumerate(content.splitlines(), 1):
            stripped = line.upper()
            for marker in ("TODO", "FIXME", "HACK", "XXX"):
                if marker in stripped and "#" in line:
                    issues.append({
                        "type": "todo_marker",
                        "line": i,
                        "message": f"Found {marker} marker in comment",
                    })
                    break

        return issues

    async def _analyze_generic_file(
        self,
        file_path: Path,
        analysis_type: str,
    ) -> ToolResult:
        """Analyze a non-Python file using basic heuristics."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return ToolResult(
                success=False,
                error=f"Cannot read file: {e}",
                safety=self._safety,
            )

        # Determine language from extension
        ext_to_lang = {
            ".js": "javascript", ".ts": "typescript", ".tsx": "typescript",
            ".jsx": "javascript", ".java": "java", ".go": "go",
            ".rs": "rust", ".rb": "ruby", ".cpp": "cpp", ".c": "c",
            ".h": "c_header", ".hpp": "cpp_header", ".cs": "csharp",
            ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
            ".r": "r", ".R": "r", ".sql": "sql", ".sh": "shell",
            ".bash": "shell", ".zsh": "shell", ".yaml": "yaml",
            ".yml": "yaml", ".toml": "toml", ".json": "json",
            ".xml": "xml", ".html": "html", ".css": "css",
            ".md": "markdown", ".rst": "restructuredtext",
        }
        language = ext_to_lang.get(file_path.suffix, "unknown")

        lines = content.splitlines()
        total_lines = len(lines)
        blank_lines = sum(1 for line in lines if not line.strip())

        # Basic regex-based function/class detection
        func_pattern = re.compile(
            r"^\s*(function\s+\w+|def\s+\w+|fn\s+\w+|func\s+\w+|pub\s+fn\s+\w+|"
            r"public\s+\w+\s+\w+\s*\(|class\s+\w+|struct\s+\w+|interface\s+\w+|"
            r"type\s+\w+\s+struct)",
            re.MULTILINE,
        )
        functions = [
            {"match": m.group(0).strip(), "line": content[:m.start()].count("\n") + 1}
            for m in func_pattern.finditer(content)
        ]

        data = {
            "path": str(file_path),
            "language": language,
            "analysis_type": analysis_type,
            "structure": {
                "functions": functions,
            },
            "metrics": {
                "total_lines": total_lines,
                "blank_lines": blank_lines,
                "non_blank_lines": total_lines - blank_lines,
                "size_bytes": file_path.stat().st_size,
            },
        }

        return ToolResult(
            success=True,
            data=data,
            safety=self._safety,
        )

    async def _analyze_directory(
        self,
        dir_path: Path,
        analysis_type: str,
    ) -> ToolResult:
        """Analyze a directory of code files."""
        # Analyze up to 20 files
        code_extensions = {
            ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go",
            ".rs", ".rb", ".cpp", ".c", ".h", ".hpp", ".cs",
            ".swift", ".kt", ".scala", ".sql", ".sh", ".bash",
        }

        files = []
        for f in sorted(dir_path.rglob("*")):
            if f.is_file() and f.suffix in code_extensions:
                files.append(f)
            if len(files) >= 20:
                break

        if not files:
            return ToolResult(
                success=True,
                data={
                    "path": str(dir_path),
                    "files_analyzed": 0,
                    "message": "No code files found in directory",
                },
                safety=self._safety,
            )

        summaries = []
        for f in files:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
                summaries.append({
                    "file": str(f.relative_to(dir_path)),
                    "lines": len(lines),
                    "language": f.suffix.lstrip("."),
                })
            except OSError:
                continue

        total_lines = sum(s["lines"] for s in summaries)

        return ToolResult(
            success=True,
            data={
                "path": str(dir_path),
                "files_analyzed": len(summaries),
                "total_lines": total_lines,
                "file_summaries": summaries,
            },
            safety=self._safety,
        )

    def describe_usage(self, params: dict) -> str:
        path = params.get("path", "?")
        return f"分析代码: {path}"


class SearchCodeSemanticTool(BaseTool):
    """Semantic code search — READ_ONLY, concurrency safe.

    Provides semantic code search capabilities:
      - Symbol-based search (functions, classes, variables)
      - Pattern-based search with context
      - Optional integration with CodeIndexer (tree-sitter) for
        deeper semantic understanding

    Falls back to text-based search when optional dependencies
    are not available.

    Usage::

        tool = SearchCodeSemanticTool()
        result = await tool.execute({"path": "/path/to/project", "symbol": "MyClass"})
    """

    name = "search_code_semantic"
    description = (
        "Search for code symbols (functions, classes, variables) with semantic understanding. "
        "Falls back to text-based search when advanced indexing is unavailable."
    )
    _safety = ToolSafety.READ_ONLY
    _concurrency_safe = True

    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory or file path to search in",
            },
            "symbol": {
                "type": "string",
                "description": "Symbol name to search for (function, class, variable name)",
            },
            "search_type": {
                "type": "string",
                "description": "Search type: 'definition' (find definitions), 'usage' (find usages), or 'any' (default)",
                "default": "any",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results",
                "default": 20,
            },
        },
        "required": ["path", "symbol"],
    }

    async def execute(
        self,
        params: dict,
        progress_callback: Any | None = None,
    ) -> ToolResult:
        """Search for code symbols semantically.

        Args:
            params: Must include 'path' and 'symbol'. Optional 'search_type' and 'max_results'.
            progress_callback: Not used.

        Returns:
            ToolResult with search results in data['results'].
        """
        path = params.get("path", "")
        symbol = params.get("symbol", "")
        search_type = params.get("search_type", "any")
        max_results = params.get("max_results", 20)

        if not path:
            return ToolResult(
                success=False,
                error="Parameter 'path' is required",
                safety=self._safety,
            )
        if not symbol:
            return ToolResult(
                success=False,
                error="Parameter 'symbol' is required",
                safety=self._safety,
            )

        try:
            search_path = Path(path).expanduser().resolve()

            if not search_path.exists():
                return ToolResult(
                    success=False,
                    error=f"Path not found: {search_path}",
                    safety=self._safety,
                )

            # Try CodeIndexer (tree-sitter) if available
            try:
                from teragent.context.code_indexer import CodeIndexer

                indexer = CodeIndexer(str(search_path))
                results = await self._search_with_indexer(
                    indexer, symbol, search_type, max_results
                )
                if results is not None:
                    return results
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"CodeIndexer not available, falling back: {e}")

            # Fall back to AST + regex based search
            return await self._search_fallback(
                search_path, symbol, search_type, max_results
            )

        except Exception as e:
            logger.error(f"SearchCodeSemanticTool failed: {e}")
            return ToolResult(
                success=False,
                error=f"Semantic search failed: {e}",
                safety=self._safety,
            )

    async def _search_with_indexer(
        self,
        indexer: Any,
        symbol: str,
        search_type: str,
        max_results: int,
    ) -> ToolResult | None:
        """Search using CodeIndexer (tree-sitter based)."""
        try:
            # CodeIndexer may have different API; this is a reasonable interface
            if hasattr(indexer, "search_symbol"):
                raw_results = indexer.search_symbol(symbol, limit=max_results)
                results = []
                for r in raw_results[:max_results]:
                    results.append({
                        "file": r.get("file", ""),
                        "line": r.get("line", 0),
                        "symbol": r.get("name", symbol),
                        "kind": r.get("kind", "unknown"),
                        "context": r.get("context", ""),
                    })

                return ToolResult(
                    success=True,
                    data={
                        "results": results,
                        "symbol": symbol,
                        "search_type": search_type,
                        "engine": "code_indexer",
                    },
                    safety=self._safety,
                )
        except Exception as e:
            logger.debug(f"CodeIndexer search failed: {e}")

        return None

    async def _search_fallback(
        self,
        search_path: Path,
        symbol: str,
        search_type: str,
        max_results: int,
    ) -> ToolResult:
        """Search using AST + regex as a fallback."""
        import ast as ast_module

        results = []
        code_extensions = {".py", ".js", ".ts", ".tsx", ".java", ".go", ".rs", ".rb"}

        paths = [search_path] if search_path.is_file() else list(search_path.rglob("*"))

        for file_path in paths:
            if len(results) >= max_results:
                break

            if not file_path.is_file():
                continue
            if file_path.suffix not in code_extensions:
                continue
            # Skip hidden and common non-code directories
            if any(part.startswith(".") for part in file_path.parts):
                continue
            if any(part in ("node_modules", "__pycache__", ".git", "venv", "dist", "build") for part in file_path.parts):
                continue

            # Python: use AST for precise symbol search
            if file_path.suffix == ".py":
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    tree = ast_module.parse(content, filename=str(file_path))

                    for node in ast_module.walk(tree):
                        if len(results) >= max_results:
                            break

                        name = None
                        kind = None
                        if isinstance(node, (ast_module.FunctionDef, ast_module.AsyncFunctionDef)):
                            name = node.name
                            kind = "function"
                        elif isinstance(node, ast_module.ClassDef):
                            name = node.name
                            kind = "class"

                        if name and symbol in name:
                            is_definition = True
                            if search_type == "usage" and kind:
                                is_definition = False
                            elif search_type == "definition" and kind:
                                is_definition = True
                            elif search_type == "any":
                                is_definition = True

                            if is_definition:
                                # Get line context
                                lines = content.splitlines()
                                line_idx = node.lineno - 1
                                context = lines[line_idx].strip() if line_idx < len(lines) else ""

                                results.append({
                                    "file": str(file_path),
                                    "line": node.lineno,
                                    "symbol": name,
                                    "kind": kind or "unknown",
                                    "context": context[:200],
                                })

                except (SyntaxError, OSError):
                    continue
            else:
                # Non-Python: regex-based search
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    # Search for definition-like patterns
                    if search_type in ("any", "definition"):
                        def_pattern = re.compile(
                            rf"\b(function|class|def|fn|func|struct|interface|type|const|let|var)\s+{re.escape(symbol)}\b",
                            re.IGNORECASE,
                        )
                    else:
                        def_pattern = None

                    # Search for any usage
                    usage_pattern = re.compile(rf"\b{re.escape(symbol)}\b")

                    lines = content.splitlines()
                    for i, line in enumerate(lines, 1):
                        if len(results) >= max_results:
                            break

                        if search_type in ("any", "definition") and def_pattern and def_pattern.search(line):
                            results.append({
                                "file": str(file_path),
                                "line": i,
                                "symbol": symbol,
                                "kind": "definition",
                                "context": line.strip()[:200],
                            })
                        elif search_type == "usage" and usage_pattern.search(line):
                            results.append({
                                "file": str(file_path),
                                "line": i,
                                "symbol": symbol,
                                "kind": "usage",
                                "context": line.strip()[:200],
                            })
                        elif search_type == "any" and usage_pattern.search(line):
                            results.append({
                                "file": str(file_path),
                                "line": i,
                                "symbol": symbol,
                                "kind": "reference",
                                "context": line.strip()[:200],
                            })

                except (OSError, re.error):
                    continue

        return ToolResult(
            success=True,
            data={
                "results": results,
                "symbol": symbol,
                "search_type": search_type,
                "total": len(results),
                "engine": "ast_regex",
            },
            safety=self._safety,
        )

    def describe_usage(self, params: dict) -> str:
        symbol = params.get("symbol", "?")
        path = params.get("path", "?")
        return f"语义搜索: '{symbol}' 在 {path}"
