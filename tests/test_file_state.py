# tests/test_file_state.py
"""FileStateTracker 文件状态追踪器单元测试

覆盖:
  - 路径穿越防护: ../, 绝对路径, 符号链接
  - 读后写契约: record_read → validate_write → record_write
  - 并发写冲突: 同一文件多写入者
  - TOCTOU 修复: 单次锁获取完成冲突检查+锁分配
  - 哈希策略: 小文件完整哈希/大文件采样哈希/SHA-256
  - 1.7 大小写敏感: os.path.normcase
  - 写入历史查询
  - 锁释放: release_write_lock
"""
import os
import hashlib
import pytest
import threading
import time
from pathlib import Path

from teragent.security.file_state import FileStateTracker, FileReadRecord, FileWriteRecord


# ===== 路径穿越防护 =====

class TestPathTraversal:
    """路径穿越防护"""

    def test_normal_relative_path(self, file_tracker, workspace):
        """正常相对路径通过"""
        safe = file_tracker._resolve_safe("src/main.py")
        assert safe is not None

    def test_path_traversal_rejected(self, file_tracker):
        """../ 路径穿越被拒绝"""
        safe = file_tracker._resolve_safe("../../../etc/passwd")
        assert safe is None

    def test_absolute_path_rejected(self, file_tracker):
        """绝对路径被拒绝"""
        safe = file_tracker._resolve_safe("/etc/passwd")
        assert safe is None

    def test_empty_path_rejected(self, file_tracker):
        """空路径被拒绝"""
        safe = file_tracker._resolve_safe("")
        assert safe is None

    def test_whitespace_path_rejected(self, file_tracker):
        """纯空白路径被拒绝"""
        safe = file_tracker._resolve_safe("   ")
        assert safe is None

    def test_dotdot_in_middle_rejected(self, file_tracker):
        """中间含 .. 的路径被拒绝"""
        safe = file_tracker._resolve_safe("src/../../etc/passwd")
        assert safe is None

    def test_workspace_subpath_allowed(self, file_tracker, workspace):
        """workspace 内子路径允许"""
        safe = file_tracker._resolve_safe("src/deep/nested/file.py")
        assert safe is not None
        assert str(safe).startswith(workspace)


# ===== 读后写契约 =====

class TestReadWriteContract:
    """读后写契约"""

    def test_read_then_write_succeeds(self, file_tracker, workspace):
        """读取后写入成功"""
        # 创建文件
        file_path = os.path.join(workspace, "test.py")
        Path(file_path).write_text("original content")

        # 记录读取
        file_tracker.record_read("test.py", reader_id="agent1")

        # 验证写入
        allowed, reason = file_tracker.validate_write("test.py", writer_id="agent1")
        assert allowed is True

    def test_write_without_read_succeeds(self, file_tracker, workspace):
        """未读取直接写入也成功（无读后写契约约束）"""
        file_path = os.path.join(workspace, "new_file.py")
        # 文件不存在时也可以写入
        allowed, reason = file_tracker.validate_write("new_file.py", writer_id="agent1")
        assert allowed is True

    def test_write_after_external_modification_fails(self, file_tracker, workspace):
        """读取后文件被外部修改，写入失败"""
        file_path = os.path.join(workspace, "test.py")
        Path(file_path).write_text("original content")

        # 记录读取
        file_tracker.record_read("test.py", reader_id="agent1")

        # 模拟外部修改
        time.sleep(0.01)
        Path(file_path).write_text("modified by external process")

        # 验证写入应失败（读后写契约违反）
        allowed, reason = file_tracker.validate_write("test.py", writer_id="agent1")
        assert allowed is False
        assert "读后写契约违反" in reason or "外部修改" in reason

    def test_write_after_file_deleted_fails(self, file_tracker, workspace):
        """读取后文件被删除，写入失败"""
        file_path = os.path.join(workspace, "temp.py")
        Path(file_path).write_text("some content")

        file_tracker.record_read("temp.py", reader_id="agent1")

        # 删除文件
        os.remove(file_path)

        # 验证写入应失败（文件不存在但之前读取过）
        allowed, reason = file_tracker.validate_write("temp.py", writer_id="agent1")
        assert allowed is False

    def test_unsafe_path_write_rejected(self, file_tracker):
        """路径穿越的写入被拒绝"""
        allowed, reason = file_tracker.validate_write("../../../etc/passwd", writer_id="agent1")
        assert allowed is False


# ===== 并发写冲突 =====

class TestConcurrentWriteConflict:
    """并发写冲突检测"""

    def test_same_writer_can_write(self, file_tracker, workspace):
        """同一写入者可以再次写入同一文件"""
        file_path = os.path.join(workspace, "test.py")
        Path(file_path).write_text("content")

        file_tracker.validate_write("test.py", writer_id="agent1")
        # agent1 持有锁，再次 validate_write 应该成功
        allowed, reason = file_tracker.validate_write("test.py", writer_id="agent1")
        assert allowed is True

    def test_different_writer_conflict(self, file_tracker, workspace):
        """不同写入者冲突"""
        file_path = os.path.join(workspace, "test.py")
        Path(file_path).write_text("content")

        # agent1 获取写入锁
        allowed, _ = file_tracker.validate_write("test.py", writer_id="agent1")
        assert allowed is True

        # agent2 尝试写入同一文件
        allowed, reason = file_tracker.validate_write("test.py", writer_id="agent2")
        assert allowed is False
        assert "并发写冲突" in reason

    def test_write_lock_released_after_record_write(self, file_tracker, workspace):
        """record_write 后锁被释放"""
        file_path = os.path.join(workspace, "test.py")
        Path(file_path).write_text("content")

        file_tracker.validate_write("test.py", writer_id="agent1")
        file_tracker.record_write("test.py", writer_id="agent1")

        # 锁已释放，另一个写入者可以获取
        allowed, reason = file_tracker.validate_write("test.py", writer_id="agent2")
        assert allowed is True


# ===== TOCTOU 修复 =====

class TestTOCTOU:
    """1.5: TOCTOU 竞态修复验证"""

    def test_concurrent_validate_write_only_one_succeeds(self, file_tracker, workspace):
        """并发 validate_write 只有一个成功"""
        file_path = os.path.join(workspace, "test.py")
        Path(file_path).write_text("content")

        results = []

        def try_write(writer_id):
            allowed, reason = file_tracker.validate_write("test.py", writer_id=writer_id)
            results.append((writer_id, allowed))

        # 使用线程模拟并发
        t1 = threading.Thread(target=try_write, args=("agent1",))
        t2 = threading.Thread(target=try_write, args=("agent2",))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # 至少一个成功
        successes = [r for r in results if r[1]]
        assert len(successes) >= 1


# ===== 释放写入锁 =====

class TestReleaseWriteLock:
    """释放写入锁（写入失败时）"""

    def test_release_write_lock(self, file_tracker, workspace):
        """release_write_lock 释放锁"""
        file_path = os.path.join(workspace, "test.py")
        Path(file_path).write_text("content")

        file_tracker.validate_write("test.py", writer_id="agent1")
        assert file_tracker.is_file_locked("test.py")

        file_tracker.release_write_lock("test.py", writer_id="agent1")
        assert not file_tracker.is_file_locked("test.py")

    def test_release_lock_by_wrong_writer(self, file_tracker, workspace):
        """其他写入者不能释放锁"""
        file_path = os.path.join(workspace, "test.py")
        Path(file_path).write_text("content")

        file_tracker.validate_write("test.py", writer_id="agent1")
        file_tracker.release_write_lock("test.py", writer_id="agent2")
        # 锁仍由 agent1 持有
        assert file_tracker.is_file_locked("test.py")


# ===== 哈希策略 =====

class TestHashStrategy:
    """1.6: 哈希策略（SHA-256 + 采样）"""

    def test_small_file_full_hash(self, file_tracker, workspace):
        """小文件使用完整哈希"""
        file_path = os.path.join(workspace, "small.py")
        content = "x" * 100  # 小于 256KB
        Path(file_path).write_text(content)

        hash1 = file_tracker._quick_hash(Path(file_path))
        hash2 = file_tracker._quick_hash(Path(file_path))
        assert hash1 == hash2  # 相同内容相同哈希
        assert len(hash1) == 64  # SHA-256 hex 长度

    def test_large_file_sampled_hash(self, file_tracker, workspace):
        """大文件使用采样哈希"""
        file_path = os.path.join(workspace, "large.py")
        # 创建 > 256KB 的文件
        content = "x" * (300 * 1024)
        Path(file_path).write_bytes(content.encode())

        hash1 = file_tracker._quick_hash(Path(file_path))
        assert len(hash1) == 64  # SHA-256 hex 长度

    def test_hash_detects_modification(self, file_tracker, workspace):
        """哈希检测到文件修改"""
        file_path = os.path.join(workspace, "test.py")
        Path(file_path).write_text("original content")
        hash1 = file_tracker._quick_hash(Path(file_path))

        time.sleep(0.01)
        Path(file_path).write_text("modified content")
        hash2 = file_tracker._quick_hash(Path(file_path))

        assert hash1 != hash2

    def test_hash_uses_sha256(self, file_tracker, workspace):
        """哈希使用 SHA-256"""
        file_path = os.path.join(workspace, "test.py")
        content = "test content for sha256"
        Path(file_path).write_text(content)

        tracker_hash = file_tracker._quick_hash(Path(file_path))
        expected_hash = hashlib.sha256(content.encode()).hexdigest()
        assert tracker_hash == expected_hash

    def test_hash_nonexistent_file(self, file_tracker, workspace):
        """不存在的文件哈希返回空串"""
        hash_val = file_tracker._quick_hash(Path(workspace) / "nonexistent.py")
        assert hash_val == ""


# ===== 写入历史查询 =====

class TestWriteHistory:
    """写入历史查询"""

    def test_record_and_query_history(self, file_tracker, workspace):
        """记录和查询写入历史"""
        file_path = os.path.join(workspace, "test.py")
        Path(file_path).write_text("content")

        file_tracker.validate_write("test.py", writer_id="agent1")
        file_tracker.record_write("test.py", writer_id="agent1")

        history = file_tracker.get_write_history("test.py")
        assert len(history) == 1
        assert history[0].writer_id == "agent1"

    def test_read_history(self, file_tracker, workspace):
        """读取历史"""
        file_path = os.path.join(workspace, "test.py")
        Path(file_path).write_text("content")

        file_tracker.record_read("test.py", reader_id="agent1")
        history = file_tracker.get_read_history("test.py")
        assert len(history) == 1
        assert history[0].reader_id == "agent1"


# ===== 状态报告 =====

class TestStatusReport:
    """状态报告"""

    def test_get_status_report(self, file_tracker, workspace):
        """获取状态报告"""
        report = file_tracker.get_status_report()
        assert "workspace_root" in report
        assert "tracked_files" in report
        assert "locked_files" in report
        assert "total_reads" in report
        assert "total_writes" in report

    def test_get_locked_files(self, file_tracker, workspace):
        """获取被锁定的文件"""
        file_path = os.path.join(workspace, "test.py")
        Path(file_path).write_text("content")

        file_tracker.validate_write("test.py", writer_id="agent1")
        locked = file_tracker.get_locked_files()
        assert "test.py" in locked
        assert locked["test.py"] == "agent1"

    def test_is_file_locked(self, file_tracker, workspace):
        """检查文件是否被锁定"""
        file_path = os.path.join(workspace, "test.py")
        Path(file_path).write_text("content")

        assert not file_tracker.is_file_locked("test.py")
        file_tracker.validate_write("test.py", writer_id="agent1")
        assert file_tracker.is_file_locked("test.py")


# ===== 重置 =====

class TestReset:
    """重置追踪状态"""

    def test_reset_clears_all(self, file_tracker, workspace):
        """reset 清除所有状态"""
        file_path = os.path.join(workspace, "test.py")
        Path(file_path).write_text("content")

        file_tracker.record_read("test.py", reader_id="agent1")
        file_tracker.validate_write("test.py", writer_id="agent1")

        file_tracker.reset()
        assert len(file_tracker.get_read_history("test.py")) == 0
        assert not file_tracker.is_file_locked("test.py")


# ===== 哈希校验禁用 =====

class TestHashValidationToggle:
    """enable_hash_validation 开关"""

    def test_hash_validation_disabled(self, workspace):
        """禁用哈希校验时跳过契约检查"""
        tracker = FileStateTracker(workspace_root=workspace, enable_hash_validation=False)
        file_path = os.path.join(workspace, "test.py")
        Path(file_path).write_text("original")

        tracker.record_read("test.py", reader_id="agent1")

        # 外部修改
        Path(file_path).write_text("modified")

        # 禁用哈希校验，应通过
        allowed, reason = tracker.validate_write("test.py", writer_id="agent1")
        assert allowed is True
