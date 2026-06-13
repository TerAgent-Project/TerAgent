# tests/test_audit.py
"""审计日志模块单元测试

覆盖:
  - AuditLogger: 初始化、log_audit 写入、_get_db 持久连接、重连、close 方法
  - 查询操作: query_by_action, query_by_time_range, query_by_request_id
  - 清理操作: cleanup_old_entries
  - 统计: get_audit_stats
"""
import time

import pytest

from teragent.security.audit import AuditLogger

# ===== 初始化与连接管理 =====

class TestAuditLoggerInit:
    """初始化与连接管理"""

    def test_default_db_path(self):
        """默认数据库路径"""
        logger = AuditLogger()
        assert logger.db_path == ".agent/audit.db"

    def test_custom_db_path(self, tmp_path):
        """自定义数据库路径"""
        db_path = str(tmp_path / "custom.db")
        logger = AuditLogger(db_path=db_path)
        assert logger.db_path == db_path

    @pytest.mark.asyncio
    async def test_get_db_creates_connection(self, tmp_path):
        """_get_db 创建持久连接"""
        db_path = str(tmp_path / "test.db")
        logger = AuditLogger(db_path=db_path)
        db = await logger._get_db()
        assert db is not None
        assert logger._db is db  # 连接被复用
        await logger.close()

    @pytest.mark.asyncio
    async def test_get_db_reuses_connection(self, tmp_path):
        """_get_db 复用已有连接"""
        db_path = str(tmp_path / "test.db")
        logger = AuditLogger(db_path=db_path)
        db1 = await logger._get_db()
        db2 = await logger._get_db()
        assert db1 is db2
        await logger.close()

    @pytest.mark.asyncio
    async def test_close_sets_db_to_none(self, tmp_path):
        """close() 将 _db 设为 None"""
        db_path = str(tmp_path / "test.db")
        logger = AuditLogger(db_path=db_path)
        await logger._get_db()
        assert logger._db is not None
        await logger.close()
        assert logger._db is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self, tmp_path):
        """close() 可多次调用"""
        db_path = str(tmp_path / "test.db")
        logger = AuditLogger(db_path=db_path)
        await logger._get_db()
        await logger.close()
        await logger.close()  # 不应报错
        assert logger._db is None

    @pytest.mark.asyncio
    async def test_reconnect_on_error(self, tmp_path):
        """连接出错后重连"""
        db_path = str(tmp_path / "test.db")
        logger = AuditLogger(db_path=db_path)
        await logger.init_db()
        # 写入一条
        await logger.log_audit("test_action", "before break")
        # 模拟连接失效
        await logger._db.close()
        # 下一次 log_audit 应重连
        await logger.log_audit("test_action", "after reconnect")
        records = await logger.query_by_action("test_action")
        assert len(records) >= 1
        await logger.close()


# ===== 写入操作 =====

class TestAuditLogWrite:
    """审计日志写入"""

    @pytest.mark.asyncio
    async def test_log_audit_writes_entry(self, tmp_path):
        """log_audit 写入审计记录"""
        db_path = str(tmp_path / "test.db")
        logger = AuditLogger(db_path=db_path)
        await logger.init_db()
        await logger.log_audit("file_write", "Atomic write: main.py")
        records = await logger.query_by_action("file_write")
        assert len(records) == 1
        assert records[0]["action"] == "file_write"
        assert records[0]["details"] == "Atomic write: main.py"
        await logger.close()

    @pytest.mark.asyncio
    async def test_log_audit_multiple_entries(self, tmp_path):
        """多条审计记录"""
        db_path = str(tmp_path / "test.db")
        logger = AuditLogger(db_path=db_path)
        await logger.init_db()
        await logger.log_audit("action_a", "detail_a")
        await logger.log_audit("action_b", "detail_b")
        await logger.log_audit("action_a", "detail_a2")
        records_a = await logger.query_by_action("action_a")
        assert len(records_a) == 2
        records_b = await logger.query_by_action("action_b")
        assert len(records_b) == 1
        await logger.close()


# ===== 查询操作 =====

class TestAuditQueries:
    """审计查询"""

    @pytest.mark.asyncio
    async def test_query_by_action(self, tmp_path):
        """按操作类型查询"""
        db_path = str(tmp_path / "test.db")
        logger = AuditLogger(db_path=db_path)
        await logger.init_db()
        await logger.log_audit("file_write", "write main.py")
        await logger.log_audit("command_blocked", "rm -rf /")
        records = await logger.query_by_action("file_write")
        assert all(r["action"] == "file_write" for r in records)
        await logger.close()

    @pytest.mark.asyncio
    async def test_query_by_time_range(self, tmp_path):
        """按时间范围查询"""
        db_path = str(tmp_path / "test.db")
        logger = AuditLogger(db_path=db_path)
        await logger.init_db()
        before = time.time()
        await logger.log_audit("timed_action", "within range")
        after = time.time()
        records = await logger.query_by_time_range(before, after + 1)
        assert any(r["action"] == "timed_action" for r in records)
        await logger.close()

    @pytest.mark.asyncio
    async def test_query_by_request_id(self, tmp_path):
        """按请求 ID 查询"""
        db_path = str(tmp_path / "test.db")
        logger = AuditLogger(db_path=db_path)
        await logger.init_db()
        await logger.log_audit("req_action", "test")
        records = await logger.query_by_request_id("unknown_id")
        # 可能返回空（因为 request_id 不匹配）
        assert isinstance(records, list)
        await logger.close()


# ===== 清理与统计 =====

class TestAuditCleanup:
    """清理与统计"""

    @pytest.mark.asyncio
    async def test_cleanup_old_entries(self, tmp_path):
        """清理旧记录"""
        db_path = str(tmp_path / "test.db")
        logger = AuditLogger(db_path=db_path)
        await logger.init_db()
        await logger.log_audit("old_action", "old detail")
        # 清理超过 0 天的记录（即全部删除）
        deleted = await logger.cleanup_old_entries(max_age_days=0)
        assert deleted >= 1
        records = await logger.query_by_action("old_action")
        assert len(records) == 0
        await logger.close()

    @pytest.mark.asyncio
    async def test_get_audit_stats(self, tmp_path):
        """获取统计摘要"""
        db_path = str(tmp_path / "test.db")
        logger = AuditLogger(db_path=db_path)
        await logger.init_db()
        await logger.log_audit("stats_action", "detail")
        stats = await logger.get_audit_stats()
        assert "total_entries" in stats
        assert "action_counts" in stats
        assert stats["total_entries"] >= 1
        assert "stats_action" in stats["action_counts"]
        await logger.close()
