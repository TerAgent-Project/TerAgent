# teragent/context/code_indexer.py
"""代码索引器 — 解析 Python 源文件，索引符号、调用关系和外部依赖

Phase 4 改进:
  - P-04: 集成 watchdog 自动增量索引
  - P-05: 重构主键（自增 ID + UNIQUE 约束，消除重复记录）
  - P-06: 递归索引嵌套定义（类方法、嵌套函数）
  - P-08: 扩展调用图（追踪跨文件和外部调用）
"""
import asyncio
import functools
import glob
import hashlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import aiosqlite
import tree_sitter_python as tsp
from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 外部调用分类常量
# ---------------------------------------------------------------------------

_STDLIB_MODULES: frozenset[str] = frozenset({
    "os", "sys", "json", "re", "asyncio", "logging", "pathlib",
    "collections", "functools", "itertools", "typing", "dataclasses",
    "abc", "io", "contextlib", "copy", "datetime", "enum", "hashlib",
    "importlib", "inspect", "math", "operator", "random", "shutil",
    "signal", "socket", "sqlite3", "string", "struct", "subprocess",
    "tempfile", "textwrap", "threading", "time", "traceback", "unittest",
    "urllib", "uuid", "warnings", "weakref", "xml", "zipfile",
    "argparse", "configparser", "heapq", "multiprocessing",
    "pickle", "platform", "pprint", "queue", "secrets", "concurrent",
})

_IMPLICIT_RECEIVERS: frozenset[str] = frozenset({"self", "cls", "super"})

_THIRD_PARTY_MODULES: frozenset[str] = frozenset({
    "httpx", "rich", "aiosqlite", "networkx", "grpcio", "protobuf",
    "lancedb", "tree_sitter", "watchdog", "numpy", "pandas", "requests",
    "flask", "django", "fastapi", "pydantic", "sqlalchemy", "pytest",
    "click", "typer", "jinja2", "yaml", "toml", "PIL", "cv2",
    "tiktoken", "openai", "anthropic", "aiohttp", "botocore",
})


class CodeIndexer:
    """Parse Python source files with tree-sitter and index symbols + call edges.

    The indexer maintains a SQLite database of symbols, call edges, and
    external calls discovered during parsing. Supports watchdog-based
    automatic incremental re-indexing.

    Phase 4 improvements:
      - Recursive definition collection (class methods, nested functions)
      - External call tracking (stdlib / third_party / unknown)
      - watchdog file watching for auto re-index
      - Incremental indexing with file hash comparison
      - Batch INSERT for performance
    """

    def __init__(self, db_path: str = ".agent/index.db") -> None:
        self.db_path = db_path
        # 限制最多 2 个线程做 tree-sitter 解析，防止 CPU 飙升
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._db: aiosqlite.Connection | None = None
        # watchdog 相关
        self._observer: Any | None = None
        self._workspace_root: str | None = None

    # ------------------------------------------------------------------
    # Database lifecycle
    # ------------------------------------------------------------------

    async def init_db(self) -> None:
        """必须在使用前调用，建立异步连接和表结构"""
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")

        # P-05: 自增 ID + UNIQUE 约束，避免重复记录
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line_number INT,
                signature TEXT,
                parent_scope TEXT DEFAULT '',
                UNIQUE(name, file_path, parent_scope, line_number)
            )
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS call_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                caller TEXT NOT NULL,
                callee TEXT NOT NULL,
                file_path TEXT NOT NULL,
                UNIQUE(caller, callee, file_path)
            )
        """)
        # P-08: 外部调用表
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS external_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                caller_name TEXT NOT NULL,
                caller_file TEXT NOT NULL,
                callee_name TEXT NOT NULL,
                callee_type TEXT NOT NULL,
                line_number INT,
                UNIQUE(caller_name, caller_file, callee_name, line_number)
            )
        """)
        # P-04: 索引元数据表（增量索引）
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS index_metadata (
                file_path TEXT PRIMARY KEY,
                last_indexed_at REAL NOT NULL,
                file_hash TEXT NOT NULL
            )
        """)
        # 创建索引加速查询
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbols_parent ON symbols(parent_scope)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_call_edges_file ON call_edges(file_path)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_ext_calls_caller ON external_calls(caller_name)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_ext_calls_callee ON external_calls(callee_name)"
        )
        await self._db.commit()

    async def close(self) -> None:
        """Close the database connection, stop watching, and shutdown the thread executor."""
        self.stop_watching()
        if self._db:
            await self._db.close()
            self._db = None
        # Run blocking shutdown in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._executor.shutdown(wait=True))

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index_file(self, file_path: str) -> dict:
        """Parse a single file and persist symbols + call edges + external calls.

        Uses incremental indexing: if the file hash hasn't changed since the
        last index, the operation is skipped.

        Returns:
            A dict with keys ``file``, ``symbol_count``, ``edge_count``,
            ``external_call_count``, ``skipped``.
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        # P-04: 增量索引 — 检查文件哈希
        file_hash = self._compute_file_hash(file_path)
        if file_hash is not None:
            cursor = await self._db.execute(
                "SELECT file_hash FROM index_metadata WHERE file_path = ?",
                (file_path,),
            )
            row = await cursor.fetchone()
            if row and row[0] == file_hash:
                return {
                    "file": file_path,
                    "symbol_count": 0,
                    "edge_count": 0,
                    "external_call_count": 0,
                    "skipped": True,
                }

        loop = asyncio.get_running_loop()
        # 严格隔离：tree-sitter 的 C 解析必须在子线程
        symbols, call_edges, external_calls = await loop.run_in_executor(
            self._executor, self._sync_parse_file, file_path
        )

        # 异步写入数据库 — 先删除旧数据
        await self._db.execute("DELETE FROM symbols WHERE file_path = ?", (file_path,))
        await self._db.execute("DELETE FROM call_edges WHERE file_path = ?", (file_path,))
        await self._db.execute("DELETE FROM external_calls WHERE caller_file = ?", (file_path,))

        # 批量写入 symbols (executemany)
        if symbols:
            symbol_rows = [
                (s["name"], s["type"], s["file_path"], s["line_number"],
                 s["signature"], s.get("parent_scope", ""))
                for s in symbols
            ]
            await self._db.executemany(
                "INSERT OR IGNORE INTO symbols "
                "(name, type, file_path, line_number, signature, parent_scope) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                symbol_rows,
            )

        # 批量写入 call_edges
        if call_edges:
            edge_rows = [(caller, callee, file_path) for caller, callee in call_edges]
            await self._db.executemany(
                "INSERT OR IGNORE INTO call_edges (caller, callee, file_path) "
                "VALUES (?, ?, ?)",
                edge_rows,
            )

        # P-08: 批量写入 external_calls
        if external_calls:
            ext_rows = [
                (ec["caller_name"], ec["caller_file"], ec["callee_name"],
                 ec["callee_type"], ec["line_number"])
                for ec in external_calls
            ]
            await self._db.executemany(
                "INSERT OR IGNORE INTO external_calls "
                "(caller_name, caller_file, callee_name, callee_type, line_number) "
                "VALUES (?, ?, ?, ?, ?)",
                ext_rows,
            )

        # 更新索引元数据
        await self._db.execute(
            "INSERT OR REPLACE INTO index_metadata (file_path, last_indexed_at, file_hash) "
            "VALUES (?, ?, ?)",
            (file_path, time.time(), file_hash or ""),
        )

        await self._db.commit()
        return {
            "file": file_path,
            "symbol_count": len(symbols),
            "edge_count": len(call_edges),
            "external_call_count": len(external_calls),
            "skipped": False,
        }

    async def reindex_file(self, file_path: str) -> dict:
        """Re-index a file that has changed.

        Clears the file hash cache to force re-indexing.
        """
        if not self._db:
            raise RuntimeError("Database not initialized")
        # 删除元数据以强制重新索引
        await self._db.execute(
            "DELETE FROM index_metadata WHERE file_path = ?",
            (file_path,),
        )
        return await self.index_file(file_path)

    async def index_workspace(self, workspace_root: str) -> dict:
        """Batch-index all Python files under *workspace_root*.

        Args:
            workspace_root: The root directory to scan for ``*.py`` files.

        Returns:
            A dict with keys ``total_files``, ``total_symbols``, ``total_edges``,
            ``total_external_calls``, ``skipped_files``.
        """
        self._workspace_root = workspace_root
        loop = asyncio.get_running_loop()
        pattern = os.path.join(workspace_root, "**", "*.py")
        # 阻塞 I/O：glob 文件系统扫描，包装为 run_in_executor
        py_files = await loop.run_in_executor(
            self._executor, functools.partial(glob.glob, pattern, recursive=True)
        )
        # Skip hidden / virtualenv directories
        py_files = [
            f for f in py_files
            if "/.git/" not in f and "/.agent/" not in f
            and "/site-packages/" not in f
            and "/__pycache__/" not in f
        ]

        total_symbols = 0
        total_edges = 0
        total_external_calls = 0
        skipped_files = 0
        for fpath in py_files:
            try:
                result = await self.index_file(fpath)
                if result.get("skipped"):
                    skipped_files += 1
                total_symbols += result["symbol_count"]
                total_edges += result["edge_count"]
                total_external_calls += result.get("external_call_count", 0)
            except Exception as e:
                logger.error(f"Failed to index {fpath}: {e}")

        logger.info(
            f"Workspace indexed: {len(py_files)} files "
            f"({skipped_files} skipped, unchanged), "
            f"{total_symbols} symbols, {total_edges} call edges, "
            f"{total_external_calls} external calls"
        )
        return {
            "total_files": len(py_files),
            "total_symbols": total_symbols,
            "total_edges": total_edges,
            "total_external_calls": total_external_calls,
            "skipped_files": skipped_files,
        }

    # ------------------------------------------------------------------
    # File watching (P-04: watchdog 自动增量索引)
    # ------------------------------------------------------------------

    def start_watching(self, workspace_root: str | None = None) -> None:
        """启动文件监听，自动触发增量索引

        Args:
            workspace_root: 监听的工作区根目录。
                如果为 None，使用上次 index_workspace 的根目录。
        """
        root = workspace_root or self._workspace_root
        if not root:
            logger.warning("Cannot start watching: no workspace root specified")
            return

        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            logger.warning(
                "watchdog not installed — file watching disabled. "
                "Install with: pip install watchdog"
            )
            return

        self._workspace_root = root

        # Capture the event loop reference for thread-safe scheduling
        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            event_loop = None

        class _CodeIndexerWatchdog(FileSystemEventHandler):
            """监听文件变更，自动触发增量索引"""

            def __init__(self, indexer: CodeIndexer, event_loop: asyncio.AbstractEventLoop | None = None) -> None:
                self.indexer = indexer
                self._event_loop = event_loop
                self._debounce: dict[str, float] = {}
                self._debounce_interval = 1.0  # 秒

            def on_modified(self, event: Any) -> None:
                if event.src_path.endswith('.py'):
                    self._schedule_reindex(event.src_path)

            def on_created(self, event: Any) -> None:
                if event.src_path.endswith('.py'):
                    self._schedule_reindex(event.src_path)

            def on_deleted(self, event: Any) -> None:
                if event.src_path.endswith('.py'):
                    self._remove_file(event.src_path)

            def _schedule_reindex(self, file_path: str) -> None:
                """防抖：1秒内同一文件的多次事件只触发一次索引"""
                now = time.time()
                last = self._debounce.get(file_path, 0)
                if now - last < self._debounce_interval:
                    return
                self._debounce[file_path] = now
                # 在事件循环中调度异步任务（线程安全）
                if self._event_loop is not None:
                    self._event_loop.call_soon_threadsafe(
                        lambda: self._event_loop.create_task(self._safe_reindex(file_path))
                    )
                else:
                    logger.debug(f"No event loop, skipping reindex for {file_path}")

            async def _safe_reindex(self, file_path: str) -> None:
                try:
                    result = await self.indexer.reindex_file(file_path)
                    logger.info(
                        f"Auto-reindexed: {file_path} — "
                        f"{result['symbol_count']} symbols, "
                        f"{result['edge_count']} edges"
                    )
                except Exception as e:
                    logger.error(f"Auto-reindex failed for {file_path}: {e}")

            def _remove_file(self, file_path: str) -> None:
                """从索引中移除已删除的文件"""
                if self._event_loop is not None:
                    self._event_loop.call_soon_threadsafe(
                        lambda: self._event_loop.create_task(self._safe_remove(file_path))
                    )
                else:
                    logger.debug(f"No event loop, skipping remove for {file_path}")

            async def _safe_remove(self, file_path: str) -> None:
                try:
                    await self.indexer.remove_file(file_path)
                except Exception as e:
                    logger.error(f"Auto-remove failed for {file_path}: {e}")

        handler = _CodeIndexerWatchdog(self, event_loop=event_loop)
        self._observer = Observer()
        self._observer.schedule(handler, root, recursive=True)
        self._observer.daemon = True
        self._observer.start()
        logger.info(f"File watching started for: {root}")

    def stop_watching(self) -> None:
        """停止文件监听"""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None
            logger.info("File watching stopped")

    async def remove_file(self, file_path: str) -> None:
        """从索引中移除文件的所有记录"""
        if not self._db:
            return
        await self._db.execute("DELETE FROM symbols WHERE file_path = ?", (file_path,))
        await self._db.execute("DELETE FROM call_edges WHERE file_path = ?", (file_path,))
        await self._db.execute(
            "DELETE FROM external_calls WHERE caller_file = ?", (file_path,)
        )
        await self._db.execute(
            "DELETE FROM index_metadata WHERE file_path = ?", (file_path,)
        )
        await self._db.commit()
        logger.info(f"Removed from index: {file_path}")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def find_symbols_by_file(self, file_path: str) -> list[dict]:
        """Return all symbols defined in the given file."""
        if not self._db:
            return []
        cursor = await self._db.execute(
            "SELECT name, type, signature, parent_scope, line_number FROM symbols WHERE file_path = ?",
            (file_path,)
        )
        rows = await cursor.fetchall()
        return [
            {"name": r[0], "type": r[1], "signature": r[2], "parent_scope": r[3], "line_number": r[4]}
            for r in rows
        ]

    async def find_symbols_by_name(self, name: str) -> list[dict]:
        """Global lookup: return all symbols matching the given name across all files."""
        if not self._db:
            return []
        cursor = await self._db.execute(
            "SELECT name, type, file_path, line_number, signature, parent_scope "
            "FROM symbols WHERE name = ?",
            (name,)
        )
        rows = await cursor.fetchall()
        return [
            {
                "name": r[0], "type": r[1], "file_path": r[2],
                "line_number": r[3], "signature": r[4], "parent_scope": r[5],
            }
            for r in rows
        ]

    async def find_symbols_by_parent(self, parent_scope: str) -> list[dict]:
        """Return all symbols within a given parent scope (e.g., class methods)."""
        if not self._db:
            return []
        cursor = await self._db.execute(
            "SELECT name, type, file_path, line_number, signature, parent_scope "
            "FROM symbols WHERE parent_scope = ?",
            (parent_scope,)
        )
        rows = await cursor.fetchall()
        return [
            {
                "name": r[0], "type": r[1], "file_path": r[2],
                "line_number": r[3], "signature": r[4], "parent_scope": r[5],
            }
            for r in rows
        ]

    async def get_all_files(self) -> list[str]:
        """Return a list of all file paths that have been indexed."""
        if not self._db:
            return []
        cursor = await self._db.execute("SELECT DISTINCT file_path FROM symbols")
        rows = await cursor.fetchall()
        return [r[0] for r in rows]

    async def get_call_edges(self, file_path: str | None = None) -> list[tuple[str, str]]:
        """Return call edges, optionally filtered by file path."""
        if not self._db:
            return []
        if file_path:
            cursor = await self._db.execute(
                "SELECT caller, callee FROM call_edges WHERE file_path = ?",
                (file_path,)
            )
        else:
            cursor = await self._db.execute("SELECT caller, callee FROM call_edges")
        rows = await cursor.fetchall()
        return [(r[0], r[1]) for r in rows]

    async def get_external_calls(
        self, file_path: str | None = None, callee_type: str | None = None
    ) -> list[dict]:
        """Return external calls, optionally filtered by file or callee type.

        Args:
            file_path: If provided, only return calls from this file.
            callee_type: If provided, filter by type ("stdlib" / "third_party" / "unknown").

        Returns:
            A list of dicts with caller_name, caller_file, callee_name,
            callee_type, line_number.
        """
        if not self._db:
            return []
        query = "SELECT caller_name, caller_file, callee_name, callee_type, line_number FROM external_calls"
        conditions: list[str] = []
        params: list[Any] = []
        if file_path:
            conditions.append("caller_file = ?")
            params.append(file_path)
        if callee_type:
            conditions.append("callee_type = ?")
            params.append(callee_type)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        return [
            {
                "caller_name": r[0], "caller_file": r[1],
                "callee_name": r[2], "callee_type": r[3],
                "line_number": r[4],
            }
            for r in rows
        ]

    async def get_index_metadata(self, file_path: str | None = None) -> list[dict]:
        """Return index metadata, optionally filtered by file path."""
        if not self._db:
            return []
        if file_path:
            cursor = await self._db.execute(
                "SELECT file_path, last_indexed_at, file_hash FROM index_metadata WHERE file_path = ?",
                (file_path,)
            )
        else:
            cursor = await self._db.execute(
                "SELECT file_path, last_indexed_at, file_hash FROM index_metadata"
            )
        rows = await cursor.fetchall()
        return [
            {"file_path": r[0], "last_indexed_at": r[1], "file_hash": r[2]}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Sync parsing (runs in thread executor)
    # ------------------------------------------------------------------

    def _sync_parse_file(
        self, file_path: str
    ) -> tuple[list[dict], list[tuple[str, str]], list[dict]]:
        """同步解析逻辑，在子线程中运行。

        Returns:
            A tuple of (symbols, call_edges, external_calls) where:
            - symbols: list of symbol dicts
            - call_edges: list of (caller, callee) tuples for project-internal calls
            - external_calls: list of dicts for external dependency calls
        """
        symbols: list[dict] = []
        call_edges: list[tuple[str, str]] = []
        external_calls: list[dict] = []
        try:
            with open(file_path, "rb") as f:
                code = f.read()
            parser = Parser(Language(tsp.language()))
            tree = parser.parse(code)

            # P-06: 递归收集所有定义（包括类方法、嵌套函数）
            defined_names: set[str] = set()
            definitions = self._collect_definitions(tree.root_node, file_path, "")
            for defn in definitions:
                defined_names.add(defn["name"])
                symbols.append(defn)

            # 第二遍：为每个函数/方法提取调用关系（内部 + 外部）
            for defn in definitions:
                if defn["type"] in ("function", "method"):
                    # 找到对应的 AST 节点
                    caller_name = defn["name"]
                    caller_scope = defn.get("parent_scope", "")
                    qualified_caller = (
                        f"{caller_scope}.{caller_name}" if caller_scope else caller_name
                    )
                    # 在 AST 中查找对应的函数节点
                    func_node = self._find_function_node(
                        tree.root_node, caller_name, caller_scope
                    )
                    if func_node:
                        self._extract_calls(
                            func_node, qualified_caller, file_path,
                            defined_names, call_edges, external_calls,
                        )

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
        return symbols, call_edges, external_calls

    def _collect_definitions(
        self,
        node: Any,
        file_path: str,
        parent_scope: str = "",
        definitions: list[dict] | None = None,
    ) -> list[dict]:
        """P-06: 递归收集所有函数/类定义（包括嵌套定义）

        Args:
            node: tree-sitter AST 节点
            file_path: 文件路径
            parent_scope: 父作用域（如类名）
            definitions: 累积的定义列表

        Returns:
            所有定义的列表
        """
        if definitions is None:
            definitions = []

        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = name_node.text.decode()
                scope = f"{parent_scope}.{name}" if parent_scope else name
                sym_type = "method" if parent_scope else "function"
                definitions.append({
                    "name": name,
                    "type": sym_type,
                    "file_path": file_path,
                    "line_number": node.start_point[0] + 1,
                    "signature": node.text.decode()[:200],
                    "parent_scope": parent_scope,
                })
                # 递归搜索嵌套定义（嵌套函数、装饰器内的定义等）
                for child in node.children:
                    self._collect_definitions(child, file_path, scope, definitions)

        elif node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = name_node.text.decode()
                scope = name  # 类作为新的父作用域
                definitions.append({
                    "name": name,
                    "type": "class",
                    "file_path": file_path,
                    "line_number": node.start_point[0] + 1,
                    "signature": node.text.decode()[:200],
                    "parent_scope": parent_scope,
                })
                # 递归搜索类成员（方法、嵌套类等）
                for child in node.children:
                    self._collect_definitions(child, file_path, scope, definitions)

        else:
            # 非定义节点，继续递归搜索子节点
            for child in node.children:
                self._collect_definitions(child, file_path, parent_scope, definitions)

        return definitions

    def _find_function_node(
        self, root: Any, func_name: str, parent_scope: str = ""
    ) -> Any | None:
        """在 AST 中查找指定函数名和父作用域的节点

        Supports nested classes: if parent_scope contains dots (e.g.,
        "OuterClass.InnerClass"), recursively descends into nested
        class definitions to find the target function.

        Args:
            root: AST 根节点
            func_name: 函数名
            parent_scope: 父作用域（如类名，可以是 "Outer.Inner" 形式）

        Returns:
            匹配的 AST 节点，或 None
        """
        # Handle nested class scopes like "OuterClass.InnerClass"
        if parent_scope:
            scope_parts = parent_scope.split(".", 1)
            first_scope = scope_parts[0]
            remaining_scope = scope_parts[1] if len(scope_parts) > 1 else ""

            for child in root.children:
                if child.type == "class_definition":
                    class_name_node = child.child_by_field_name("name")
                    if class_name_node and class_name_node.text.decode() == first_scope:
                        if remaining_scope:
                            # Recurse into deeper nested scopes
                            result = self._find_function_node(child, func_name, remaining_scope)
                            if result:
                                return result
                        else:
                            # This is the target class — search for the method
                            for member in child.children:
                                if member.type == "function_definition":
                                    name_node = member.child_by_field_name("name")
                                    if name_node and name_node.text.decode() == func_name:
                                        return member
                                elif member.type == "block":
                                    for block_child in member.children:
                                        if block_child.type == "function_definition":
                                            name_node = block_child.child_by_field_name("name")
                                            if name_node and name_node.text.decode() == func_name:
                                                return block_child
            return None

        # 顶层函数
        for child in root.children:
            if child.type == "function_definition":
                name_node = child.child_by_field_name("name")
                if name_node and name_node.text.decode() == func_name:
                    return child
        return None

    def _extract_calls(
        self,
        node: Any,
        caller_name: str,
        caller_file: str,
        defined_names: set[str],
        call_edges: list[tuple[str, str]],
        external_calls: list[dict],
    ) -> None:
        """P-08: 递归提取调用关系，同时追踪外部调用

        Args:
            node: AST 节点
            caller_name: 调用者名称（可能是 ClassName.method 形式）
            caller_file: 调用者所在文件
            defined_names: 当前文件内定义的符号名集合
            call_edges: 内部调用边列表（累积）
            external_calls: 外部调用列表（累积）
        """
        if node.type == "call":
            func_node = node.child_by_field_name("function")
            if func_node:
                callee_name = func_node.text.decode()
                # 提取根名称（处理 obj.method() 形式的调用）
                root_name = callee_name.split(".")[0]
                if root_name in _IMPLICIT_RECEIVERS:
                    # Can't resolve statically — classify as internal call edge
                    call_edges.append((caller_name, callee_name))
                elif root_name in defined_names or callee_name in defined_names:
                    # 项目内部调用
                    call_edges.append((caller_name, callee_name))
                else:
                    # 外部调用 — 分类
                    callee_type = self._classify_external_call(root_name)
                    external_calls.append({
                        "caller_name": caller_name,
                        "caller_file": caller_file,
                        "callee_name": callee_name,
                        "callee_type": callee_type,
                        "line_number": node.start_point[0] + 1,
                    })
        for child in node.children:
            self._extract_calls(
                child, caller_name, caller_file,
                defined_names, call_edges, external_calls,
            )

    @staticmethod
    def _classify_external_call(name: str) -> str:
        """P-08: 分类外部调用

        Args:
            name: 调用的根名称（如 "httpx"、"os"）

        Returns:
            "stdlib" / "third_party" / "unknown"
        """
        root_module = name.split(".")[0].strip()
        if root_module in _STDLIB_MODULES:
            return "stdlib"
        if root_module in _THIRD_PARTY_MODULES:
            return "third_party"
        return "unknown"

    @staticmethod
    def _compute_file_hash(file_path: str) -> str | None:
        """计算文件哈希用于增量索引判断

        Returns:
            SHA-256 哈希字符串，或 None（文件不存在或读取失败）
        """
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except (OSError, IOError):
            return None
