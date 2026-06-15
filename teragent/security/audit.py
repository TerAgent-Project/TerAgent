# teragent/security/audit.py
"""审计日志模块

改动:
  - 全局 `_db_path` 已消除
  - 新增 AuditLogger 类，通过构造器注入 db_path
  - 模块级函数保留为向后兼容别名（标记 deprecated），委托给默认实例
  - 所有消费者（permission.py, file_writer.py, audit_hook.py）可选择性接受 AuditLogger 注入

设计原则:
  - AuditLogger 实例可安全地被多个组件共享
  - 不再依赖全局变量，可同时创建多个独立审计实例
"""

import asyncio
import functools
import logging
import os
import threading
import time
from typing import Any, Awaitable, Callable

__all__ = [
    "AuditLogger",
    "DEFAULT_DB_PATH",
    "audit_log",
    "cleanup_old_entries",
    "get_audit_stats",
    "get_db_path",
    "init_audit_db",
    "log_audit",
    "query_by_action",
    "query_by_request_id",
    "query_by_time_range",
    "rotate_audit_db",
    "set_db_path",
]

import aiosqlite

from teragent.utils.tracing import get_request_id

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(".agent", "audit.db")  # 使用 os.path.join 确保跨平台路径分隔符


class AuditLogger:
    """审计日志记录器 — Phase 0.3 核心类

    替代全局 _db_path + 模块级函数的模式。
    每个 AuditLogger 实例拥有独立的 db_path，可安全并发使用。

    Usage::

        # 创建实例
        audit = AuditLogger(db_path=".agent/audit.db")

        # 初始化数据库
        await audit.init_db()

        # 记录审计日志
        await audit.log_audit("file_write", "Atomic write: main.py")

        # 查询
        records = await audit.query_by_action("file_write")
        records = await audit.query_by_time_range(start_time=1000.0)
        stats = await audit.get_audit_stats()
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        """初始化审计日志记录器

        Args:
            db_path: 审计数据库文件路径
        """
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    @property
    def db_path(self) -> str:
        """获取当前审计数据库路径"""
        return self._db_path

    async def _get_db(self) -> aiosqlite.Connection:
        """Get or create a persistent database connection with reconnect logic."""
        if self._db is None:
            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
            self._db = await aiosqlite.connect(self._db_path)
            await self._db.execute("PRAGMA journal_mode=WAL")
            # Auto-create table to prevent silent data loss when init_db() was not called
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    request_id TEXT,
                    action TEXT,
                    details TEXT
                )
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_action
                ON audit_log(action)
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                ON audit_log(timestamp)
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_request_id
                ON audit_log(request_id)
            """)
            await self._db.commit()
        return self._db

    async def init_db(self) -> None:
        """Initialize the audit database, creating the table if it doesn't exist."""
        db = await self._get_db()
        await db.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                request_id TEXT,
                action TEXT,
                details TEXT
            )
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_action
            ON audit_log(action)
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp
            ON audit_log(timestamp)
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_request_id
            ON audit_log(request_id)
        """)
        await db.commit()

    async def log_audit(self, action: str, details: str = "") -> None:
        """Write an audit log entry using persistent connection."""
        async with self._lock:
            try:
                db = await self._get_db()
                await db.execute(
                    "INSERT INTO audit_log (timestamp, request_id, action, details) VALUES (?, ?, ?, ?)",
                    (time.time(), get_request_id(), action, details)
                )
                await db.commit()
            except Exception as e:
                # Connection may have gone stale — close & reset so next call reconnects
                if self._db is not None:
                    try:
                        await self._db.close()
                    except Exception:
                        pass
                self._db = None
                logger.error(f"Failed to write audit log: {e}")

    async def query_by_action(self, action: str, limit: int = 100) -> list[dict]:
        """查询指定操作类型的审计记录。"""
        async with self._lock:
            try:
                db = await self._get_db()
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM audit_log WHERE action = ? ORDER BY timestamp DESC LIMIT ?",
                    (action, limit)
                )
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]
            except Exception as e:
                if self._db is not None:
                    try:
                        await self._db.close()
                    except Exception:
                        pass
                self._db = None
                logger.error(f"Failed to query audit by action: {e}")
                return []

    async def query_by_time_range(
        self,
        start_time: float,
        end_time: float | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """查询指定时间范围内的审计记录。"""
        if end_time is None:
            end_time = time.time()
        async with self._lock:
            try:
                db = await self._get_db()
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM audit_log WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp DESC LIMIT ?",
                    (start_time, end_time, limit)
                )
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]
            except Exception as e:
                if self._db is not None:
                    try:
                        await self._db.close()
                    except Exception:
                        pass
                self._db = None
                logger.error(f"Failed to query audit by time range: {e}")
                return []

    async def query_by_request_id(self, request_id: str) -> list[dict]:
        """查询指定请求 ID 的所有审计记录。"""
        async with self._lock:
            try:
                db = await self._get_db()
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM audit_log WHERE request_id = ? ORDER BY timestamp ASC",
                    (request_id,)
                )
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]
            except Exception as e:
                if self._db is not None:
                    try:
                        await self._db.close()
                    except Exception:
                        pass
                self._db = None
                logger.error(f"Failed to query audit by request_id: {e}")
                return []

    async def cleanup_old_entries(self, max_age_days: int = 90) -> int:
        """删除超过指定天数的审计记录。"""
        async with self._lock:
            cutoff = time.time() - (max_age_days * 86400)
            try:
                db = await self._get_db()
                cursor = await db.execute(
                    "DELETE FROM audit_log WHERE timestamp < ?", (cutoff,)
                )
                await db.commit()
                deleted = cursor.rowcount
                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} audit entries older than {max_age_days} days")
                return deleted
            except Exception as e:
                if self._db is not None:
                    try:
                        await self._db.close()
                    except Exception:
                        pass
                self._db = None
                logger.error(f"Failed to cleanup old audit entries: {e}")
                return 0

    async def rotate_audit_db(self, backup_path: str | None = None, max_age_days: int = 90) -> str | None:
        """Rotate the audit DB: create a backup, then clean up old entries."""
        if backup_path is None:
            backup_path = self._db_path + ".bak"
        try:
            db = await self._get_db()
            async with aiosqlite.connect(backup_path) as dst:
                await db.backup(dst)
            logger.info(f"Audit DB backed up to {backup_path}")
            await self.cleanup_old_entries(max_age_days)
            return backup_path
        except Exception as e:
            if self._db is not None:
                try:
                    await self._db.close()
                except Exception:
                    pass
            self._db = None
            logger.error(f"Failed to rotate audit DB: {e}")
            return None

    async def get_audit_stats(self) -> dict:
        """获取审计日志统计摘要。"""
        try:
            db = await self._get_db()
            cursor = await db.execute("SELECT COUNT(*) FROM audit_log")
            row = await cursor.fetchone()
            total = row[0] if row else 0

            cursor = await db.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM audit_log"
            )
            row = await cursor.fetchone()
            oldest = row[0] if row and row[0] is not None else None
            newest = row[1] if row and row[1] is not None else None

            cursor = await db.execute(
                "SELECT action, COUNT(*) as cnt FROM audit_log GROUP BY action ORDER BY cnt DESC"
            )
            action_rows = await cursor.fetchall()
            action_counts = {r[0]: r[1] for r in action_rows}

            return {
                "total_entries": total,
                "oldest_timestamp": oldest,
                "newest_timestamp": newest,
                "action_counts": action_counts,
            }
        except Exception as e:
            if self._db is not None:
                try:
                    await self._db.close()
                except Exception:
                    pass
            self._db = None
            logger.error(f"Failed to get audit stats: {e}")
            return {"total_entries": 0, "oldest_timestamp": None, "newest_timestamp": None, "action_counts": {}}

    async def close(self) -> None:
        """Close the persistent database connection."""
        if self._db:
            await self._db.close()
            self._db = None


# ===== 向后兼容：模块级默认实例和函数（deprecated） =====

_default_audit_logger = AuditLogger()
_db_path_lock = threading.Lock()


def set_db_path(path: str) -> None:
    """Configure the audit database path.

    DEPRECATED (Phase 0.3): Use AuditLogger(db_path=...) instead.
    """
    global _default_audit_logger
    with _db_path_lock:
        old = _default_audit_logger
        _default_audit_logger = AuditLogger(db_path=path)
        # Best-effort close old connection (safe async wrapper)
        if old is not None and hasattr(old, '_db') and old._db is not None:
            try:
                loop = asyncio.get_running_loop()
                async def _safe_close(db):
                    try:
                        await db.close()
                    except Exception:
                        pass
                loop.create_task(_safe_close(old._db))
            except Exception:
                pass


def get_db_path() -> str:
    """Return the current audit database path.

    DEPRECATED (Phase 0.3): Use AuditLogger.db_path property instead.
    """
    with _db_path_lock:
        return _default_audit_logger.db_path


async def init_audit_db() -> None:
    """Initialize the audit database.

    DEPRECATED (Phase 0.3): Use AuditLogger.init_db() instead.
    """
    with _db_path_lock:
        logger_instance = _default_audit_logger
    await logger_instance.init_db()


async def log_audit(action: str, details: str = "") -> None:
    """Write an audit log entry.

    DEPRECATED (Phase 0.3): Use AuditLogger.log_audit() instead.
    """
    with _db_path_lock:
        logger_instance = _default_audit_logger
    await logger_instance.log_audit(action, details)


def audit_log(action_name: str) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """装饰器：记录关键操作的审计日志

    DEPRECATED (Phase 0.3): Use AuditLogger.log_audit() instead.
    """
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                with _db_path_lock:
                    inst = _default_audit_logger
                await inst.log_audit(action_name, f"Success in {duration:.2f}s. Args: {str(args)[:100]}")
                return result
            except Exception as e:
                duration = time.time() - start_time
                with _db_path_lock:
                    inst = _default_audit_logger
                await inst.log_audit(action_name, f"Failed in {duration:.2f}s. Error: {str(e)}")
                raise
        return wrapper
    return decorator


async def query_by_action(action: str, limit: int = 100) -> list[dict]:
    """查询指定操作类型的审计记录。DEPRECATED: Use AuditLogger.query_by_action()"""
    with _db_path_lock:
        logger_instance = _default_audit_logger
    return await logger_instance.query_by_action(action, limit)


async def query_by_time_range(
    start_time: float,
    end_time: float | None = None,
    limit: int = 100,
) -> list[dict]:
    """查询指定时间范围内的审计记录。DEPRECATED: Use AuditLogger.query_by_time_range()"""
    with _db_path_lock:
        logger_instance = _default_audit_logger
    return await logger_instance.query_by_time_range(start_time, end_time, limit)


async def query_by_request_id(request_id: str) -> list[dict]:
    """查询指定请求 ID 的所有审计记录。DEPRECATED: Use AuditLogger.query_by_request_id()"""
    with _db_path_lock:
        logger_instance = _default_audit_logger
    return await logger_instance.query_by_request_id(request_id)


async def cleanup_old_entries(max_age_days: int = 90) -> int:
    """删除超过指定天数的审计记录。DEPRECATED: Use AuditLogger.cleanup_old_entries()"""
    with _db_path_lock:
        logger_instance = _default_audit_logger
    return await logger_instance.cleanup_old_entries(max_age_days)


async def rotate_audit_db(backup_path: str | None = None, max_age_days: int = 90) -> str | None:
    """Rotate the audit DB. DEPRECATED: Use AuditLogger.rotate_audit_db()"""
    with _db_path_lock:
        logger_instance = _default_audit_logger
    return await logger_instance.rotate_audit_db(backup_path, max_age_days)


async def get_audit_stats() -> dict:
    """获取审计日志统计摘要。DEPRECATED: Use AuditLogger.get_audit_stats()"""
    with _db_path_lock:
        logger_instance = _default_audit_logger
    return await logger_instance.get_audit_stats()
