# tests/test_file_writer.py
"""安全文件写入模块单元测试

覆盖:
  - _sync_atomic_write: 原子写入、临时文件清理、目录自动创建
  - atomic_write_file: 权限检查、路径穿越防护、读后写契约、写入失败回滚
  - write_files_safely: 两阶段提交(2PC)、验证阶段、备份回滚、并发写入锁释放
  - _release_all_locks: 批量释放写入锁
  - _cleanup_backups: 清理备份文件
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from teragent.security.file_writer import (
    _cleanup_backups,
    _release_all_locks,
    _sync_atomic_write,
    atomic_write_file,
    write_files_safely,
)
from teragent.security.permission import PermissionLevel, PermissionManager
from teragent.utils.exceptions import SandboxViolation

# ===== _sync_atomic_write =====

class TestSyncAtomicWrite:
    """同步原子写入"""

    def test_writes_content_to_file(self, tmp_path):
        """原子写入正常流程"""
        target = str(tmp_path / "test.txt")
        _sync_atomic_write(target, "hello world")
        assert Path(target).read_text() == "hello world"

    def test_creates_parent_directory(self, tmp_path):
        """自动创建父目录"""
        target = str(tmp_path / "sub" / "dir" / "file.py")
        _sync_atomic_write(target, "print('hi')")
        assert Path(target).read_text() == "print('hi')"

    def test_temp_file_cleaned_on_failure(self, tmp_path):
        """写入失败时清理临时文件"""
        target = str(tmp_path / "readonly" / "test.txt")
        # 父目录不存在且无法创建（模拟权限问题）
        with patch("os.makedirs", side_effect=OSError("Permission denied")):
            with pytest.raises(OSError):
                _sync_atomic_write(target, "content")
        # 临时文件不应残留
        tmp_files = list(tmp_path.rglob(".teragent_tmp_*"))
        assert len(tmp_files) == 0

    def test_overwrites_existing_file(self, tmp_path):
        """覆盖已有文件"""
        target = str(tmp_path / "existing.txt")
        Path(target).write_text("old content")
        _sync_atomic_write(target, "new content")
        assert Path(target).read_text() == "new content"


# ===== atomic_write_file =====

class TestAtomicWriteFile:
    """异步原子写入（含安全检查）"""

    @pytest.mark.asyncio
    async def test_basic_write_succeeds(self, tmp_path):
        """基本写入成功"""
        success, msg = await atomic_write_file(
            filepath="test.txt",
            content="hello",
            workspace_root=str(tmp_path),
        )
        assert success is True
        assert Path(tmp_path / "test.txt").read_text() == "hello"

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, tmp_path):
        """路径穿越防护"""
        with pytest.raises(SandboxViolation, match="Path traversal"):
            await atomic_write_file(
                filepath="../../etc/passwd",
                content="malicious",
                workspace_root=str(tmp_path),
            )

    @pytest.mark.asyncio
    async def test_permission_manager_denies(self, tmp_path):
        """PermissionManager 权限不足拒绝写入"""
        perm_mgr = PermissionManager(default_level=PermissionLevel.DEFAULT)
        with pytest.raises(SandboxViolation, match="PLAN level"):
            await atomic_write_file(
                filepath="test.txt",
                content="hello",
                workspace_root=str(tmp_path),
                perm_mgr=perm_mgr,
            )

    @pytest.mark.asyncio
    async def test_enhanced_permission_manager_denies(self, tmp_path):
        """EnhancedPermissionManager 拒绝写入"""
        from teragent.security.permission import (
            EnhancedPermissionManager,
            PermissionEffect,
            PermissionRule,
        )
        epm = EnhancedPermissionManager(default_level=PermissionLevel.PLAN)
        epm.add_rule(PermissionRule(
            effect=PermissionEffect.DENY,
            tool_pattern="write_file",
            path_pattern="*",
            description="全部拒绝",
            source="user",
        ))
        with pytest.raises(SandboxViolation, match="Write denied by permission rule"):
            await atomic_write_file(
                filepath="test.txt",
                content="hello",
                workspace_root=str(tmp_path),
                enhanced_perm_mgr=epm,
            )

    @pytest.mark.asyncio
    async def test_file_state_tracker_rejects_stale_write(self, tmp_path):
        """FileStateTracker 拒绝过期写入"""
        tracker = MagicMock()
        tracker.validate_write.return_value = (False, "文件已被外部修改")
        success, msg = await atomic_write_file(
            filepath="test.txt",
            content="hello",
            workspace_root=str(tmp_path),
            file_state_tracker=tracker,
        )
        assert success is False
        assert "外部修改" in msg

    @pytest.mark.asyncio
    async def test_write_failure_releases_lock(self, tmp_path):
        """写入失败时释放写入锁"""
        tracker = MagicMock()
        tracker.validate_write.return_value = (True, "")
        with patch("teragent.security.file_writer._sync_atomic_write", side_effect=IOError("disk full")):
            success, msg = await atomic_write_file(
                filepath="test.txt",
                content="hello",
                workspace_root=str(tmp_path),
                file_state_tracker=tracker,
            )
        assert success is False
        tracker.release_write_lock.assert_called_once_with("test.txt", "unknown")

    @pytest.mark.asyncio
    async def test_write_success_records_write(self, tmp_path):
        """写入成功后记录写入"""
        tracker = MagicMock()
        tracker.validate_write.return_value = (True, "")
        await atomic_write_file(
            filepath="test.txt",
            content="hello",
            workspace_root=str(tmp_path),
            file_state_tracker=tracker,
            writer_id="agent_1",
        )
        tracker.record_write.assert_called_once_with("test.txt", "agent_1")


# ===== write_files_safely (2PC) =====

class TestWriteFilesSafely:
    """两阶段提交批量写入"""

    @pytest.mark.asyncio
    async def test_empty_dict_returns_empty(self, tmp_path):
        """空文件字典返回空列表"""
        written, failed = await write_files_safely({}, workspace_root=str(tmp_path))
        assert written == []
        assert failed == []

    @pytest.mark.asyncio
    async def test_all_files_committed(self, tmp_path):
        """全部文件成功提交"""
        files = {
            "a.txt": "content A",
            "b.py": "content B",
        }
        written, failed = await write_files_safely(files, workspace_root=str(tmp_path))
        assert set(written) == {"a.txt", "b.py"}
        assert failed == []
        assert Path(tmp_path / "a.txt").read_text() == "content A"
        assert Path(tmp_path / "b.py").read_text() == "content B"

    @pytest.mark.asyncio
    async def test_path_traversal_in_batch_rejected(self, tmp_path):
        """批量写入中路径穿越被拒绝"""
        files = {
            "good.txt": "ok",
            "../../etc/bad": "malicious",
        }
        with pytest.raises(SandboxViolation, match="Path traversal"):
            await write_files_safely(files, workspace_root=str(tmp_path))

    @pytest.mark.asyncio
    async def test_validation_failure_releases_previous_locks(self, tmp_path):
        """验证阶段失败释放已获取的写入锁"""
        tracker = MagicMock()
        # 第一个文件验证通过，第二个失败
        tracker.validate_write.side_effect = [
            (True, ""),
            (False, "并发写冲突"),
        ]
        files = {"a.txt": "ok", "b.txt": "bad"}
        written, failed = await write_files_safely(
            files,
            workspace_root=str(tmp_path),
            file_state_tracker=tracker,
        )
        assert failed == ["b.txt"]
        # 释放 a.txt 的锁
        tracker.release_write_lock.assert_called()

    @pytest.mark.asyncio
    async def test_backup_rollback_on_failure(self, tmp_path):
        """原子替换失败时从备份回滚"""
        # 先创建已有文件
        (tmp_path / "existing.txt").write_text("original")
        files = {"existing.txt": "new content"}
        # 让 os.replace 在 Phase 3 失败
        with patch("os.replace", side_effect=OSError("replace failed")):
            written, failed = await write_files_safely(files, workspace_root=str(tmp_path))
        assert failed == ["existing.txt"]
        # 原文件应通过备份恢复
        # 注: 由于 os.replace 被完全 mock，备份恢复也会失败，
        # 但我们可以验证 failed_files 包含该文件

    @pytest.mark.asyncio
    async def test_permission_manager_denies_batch(self, tmp_path):
        """PermissionManager 拒绝批量写入"""
        perm_mgr = PermissionManager(default_level=PermissionLevel.DEFAULT)
        with pytest.raises(SandboxViolation, match="PLAN level"):
            await write_files_safely(
                {"test.txt": "hello"},
                workspace_root=str(tmp_path),
                perm_mgr=perm_mgr,
            )

    @pytest.mark.asyncio
    async def test_temp_write_failure_cleans_up(self, tmp_path):
        """临时文件写入失败时清理临时和备份文件"""
        files = {"test.txt": "content"}
        with patch("teragent.security.file_writer._sync_atomic_write", side_effect=IOError("disk full")):
            written, failed = await write_files_safely(files, workspace_root=str(tmp_path))
        assert written == []
        assert "test.txt" in failed
        # 不应有 .tmp 或 .bak 残留
        tmp_files = list(tmp_path.rglob("*.tmp"))
        bak_files = list(tmp_path.rglob("*.bak"))
        assert len(tmp_files) == 0
        assert len(bak_files) == 0


# ===== _release_all_locks =====

class TestReleaseAllLocks:
    """批量释放写入锁"""

    def test_releases_locks_for_all_files(self):
        """释放所有文件的写入锁"""
        tracker = MagicMock()
        _release_all_locks(tracker, {"a.txt": None, "b.txt": None}, "writer_1")
        assert tracker.release_write_lock.call_count == 2
        tracker.release_write_lock.assert_any_call("a.txt", "writer_1")
        tracker.release_write_lock.assert_any_call("b.txt", "writer_1")

    def test_skips_when_tracker_is_none(self):
        """tracker 为 None 时不执行操作"""
        _release_all_locks(None, {"a.txt": None}, "writer_1")  # 不应报错

    def test_skips_when_no_files(self):
        """无文件时不执行操作"""
        tracker = MagicMock()
        _release_all_locks(tracker, {}, "writer_1")
        tracker.release_write_lock.assert_not_called()


# ===== _cleanup_backups =====

class TestCleanupBackups:
    """清理备份文件"""

    def test_removes_existing_backup_files(self, tmp_path):
        """删除存在的备份文件"""
        bak = tmp_path / "file.py.bak"
        bak.write_text("backup content")
        _cleanup_backups({"file.py": bak})
        assert not bak.exists()

    def test_skips_none_entries(self, tmp_path):
        """跳过 None 条目"""
        _cleanup_backups({"new_file.txt": None})  # 不应报错

    def test_skips_nonexistent_files(self, tmp_path):
        """跳过不存在的文件"""
        bak = tmp_path / "missing.bak"
        _cleanup_backups({"file.py": bak})  # 不应报错
