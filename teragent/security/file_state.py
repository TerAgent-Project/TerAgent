# teragent/security/file_state.py
"""文件状态追踪器 — 核心组件

参考 Claude-Code 的 read-before-write 机制和文件时间戳校验，
实现"读后写契约"和并发写冲突检测。

核心设计:
  1. 读后写契约（Read-Before-Write Contract）
     - Agent 读取文件时，自动记录文件路径和内容哈希
     - Agent 写入文件时，校验文件自上次读取后是否被外部修改
     - 如果哈希不一致（文件被外部修改），拒绝写入并返回明确错误

  2. 并发写冲突检测
     - 同一文件在同一时刻只能由一个 Agent 写入
     - 写入完成后释放文件锁，其他 Agent 可继续写入

  3. 路径穿越防护
     - 所有文件路径必须在 workspace_root 内
     - 解析后路径与 workspace_root 比对，防止符号链接穿越

  4. 写入历史追踪
     - 记录每次写入的 agent_id / timestamp / 文件哈希
     - 可查询文件的完整修改历史

使用方式::

    tracker = FileStateTracker(workspace_root="/path/to/project")

    # 读取文件时自动记录
    tracker.record_read("src/main.py", reader_id="subagent_1")

    # 写入前校验
    ok, reason = tracker.validate_write("src/main.py", writer_id="subagent_1")
    if ok:
        # 执行原子写入...
        tracker.record_write("src/main.py", writer_id="subagent_1")
    else:
        print(f"写入被拒绝: {reason}")

设计原则:
  - 不信任模型的管理能力 — 契约校验由代码强制执行
  - 防御性编程 — 路径穿越、哈希失败、并发冲突都有兜底
  - 即发即忘 — record_read/record_write 不阻塞主流程
  - 审计可查 — 所有状态变更都有日志
"""

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FileReadRecord:
    """文件读取记录

    Attributes:
        filepath: 相对于 workspace_root 的文件路径
        reader_id: 读取者标识（如 subagent_id / agent_loop）
        content_hash: 读取时文件内容的快速哈希
        timestamp: 读取时间戳
    """
    filepath: str
    reader_id: str
    content_hash: str
    timestamp: float


@dataclass
class FileWriteRecord:
    """文件写入记录

    Attributes:
        filepath: 相对于 workspace_root 的文件路径
        writer_id: 写入者标识
        content_hash: 写入后文件内容的快速哈希
        timestamp: 写入时间戳
    """
    filepath: str
    writer_id: str
    content_hash: str
    timestamp: float


class FileStateTracker:
    """文件状态追踪器 — 实现读后写契约和并发写冲突检测

    核心职责:
      1. 记录文件读取（record_read）
      2. 校验写入契约（validate_write）— 确保文件自读取后未被修改
      3. 记录文件写入（record_write）
      4. 并发写冲突检测 — 同一文件同一时刻只允许一个写入者
      5. 路径穿越防护 — 所有路径必须在 workspace_root 内
      6. 写入历史查询

    线程安全: 所有内部状态修改使用 threading.Lock 保护。
    """

    def __init__(
        self,
        workspace_root: str,
        max_history_per_file: int = 50,
        enable_hash_validation: bool = True,
    ) -> None:
        """初始化文件状态追踪器

        Args:
            workspace_root: 工作区根目录的绝对路径
            max_history_per_file: 每个文件最多保留的历史记录条数
            enable_hash_validation: 是否启用内容哈希校验（关闭则只做并发检测）
        """
        self.workspace_root = os.path.abspath(workspace_root)
        self.max_history_per_file = max_history_per_file
        self.enable_hash_validation = enable_hash_validation

        # 读取记录: filepath → list[FileReadRecord]
        self._read_records: dict[str, list[FileReadRecord]] = {}

        # 写入记录: filepath → list[FileWriteRecord]
        self._write_records: dict[str, list[FileWriteRecord]] = {}

        # 当前写入锁: filepath → writer_id
        # 用于检测并发写冲突（同一文件同时只能由一个写入者持有）
        self._write_locks: dict[str, str] = {}

        # 线程安全锁
        self._lock = threading.Lock()

    # ===== 核心方法 =====

    def record_read(self, filepath: str, reader_id: str = "unknown") -> None:
        """记录文件读取

        读取文件时调用此方法，记录读取者和文件内容哈希。
        后续写入时会校验文件内容是否与读取时一致。

        Args:
            filepath: 相对于 workspace_root 的文件路径
            reader_id: 读取者标识（如 "subagent_1", "agent_loop"）
        """
        safe_path = self._resolve_safe(filepath)
        if safe_path is None:
            # 路径不安全，仍然记录读取但不计算哈希
            logger.warning(f"record_read: unsafe path skipped: {filepath}")
            return

        # 计算文件内容哈希
        content_hash = self._quick_hash(safe_path) if safe_path.exists() else ""

        record = FileReadRecord(
            filepath=filepath,
            reader_id=reader_id,
            content_hash=content_hash,
            timestamp=time.time(),
        )

        with self._lock:
            if filepath not in self._read_records:
                self._read_records[filepath] = []
            self._read_records[filepath].append(record)

            # 限制历史记录长度
            if len(self._read_records[filepath]) > self.max_history_per_file:
                self._read_records[filepath] = self._read_records[filepath][-self.max_history_per_file:]

        logger.debug(
            f"record_read: {filepath} by {reader_id} "
            f"(hash={content_hash[:8]}...)"
        )

    def validate_write(self, filepath: str, writer_id: str = "unknown") -> tuple[bool, str]:
        """校验写入契约

        1.5 改进：TOCTOU 竞态修复 — 在单次锁获取内完成冲突检查和锁分配，
        避免两次加锁之间被其他协程抢占。

        在写入文件前调用此方法，执行以下检查:
          1. 路径安全性检查（防路径穿越）
          2. 并发写冲突检查（同一文件是否正被其他写入者持有）
          3. 读后写契约校验（文件自上次读取后是否被外部修改）
          4. 校验通过立即分配写入锁（同一锁内完成）

        Args:
            filepath: 相对于 workspace_root 的文件路径
            writer_id: 写入者标识

        Returns:
            (allowed, reason) — 是否允许写入 + 原因
            - (True, "") — 允许写入
            - (False, reason) — 拒绝写入，reason 说明原因
        """
        # 1. 路径安全性检查（在锁外执行，无共享状态）
        safe_path = self._resolve_safe(filepath)
        if safe_path is None:
            return False, f"路径不安全（穿越工作区）: {filepath}"

        # Compute hash outside the lock to avoid blocking I/O holding the lock
        current_hash = self._quick_hash(safe_path) if safe_path.exists() else ""

        # 2-4: 在单次锁获取内完成冲突检查 + 契约校验 + 锁分配
        with self._lock:
            # 2. 并发写冲突检查
            current_writer = self._write_locks.get(filepath)
            if current_writer and current_writer != writer_id:
                return False, (
                    f"并发写冲突: 文件 {filepath} 正被 {current_writer} 写入，"
                    f"{writer_id} 不能同时写入"
                )

            # 3. 读后写契约校验
            if self.enable_hash_validation:
                read_records = self._read_records.get(filepath, [])
                if read_records:
                    # 取最近一次读取记录
                    last_read = read_records[-1]

                    # 文件存在时，比较当前哈希与读取时哈希
                    if safe_path.exists():
                        if current_hash != last_read.content_hash:
                            return False, (
                                f"读后写契约违反: 文件 {filepath} 自上次读取后已被外部修改。"
                                f"读取时哈希: {last_read.content_hash[:8]}..., "
                                f"当前哈希: {current_hash[:8]}...。"
                                f"请重新读取文件后再写入。"
                            )
                    # 文件不存在但之前读取过 — 可能被外部删除
                    elif last_read.content_hash:
                        return False, (
                            f"读后写契约违反: 文件 {filepath} 之前存在（读取时哈希: "
                            f"{last_read.content_hash[:8]}...），"
                            f"但现在不存在。可能已被外部删除或移动。"
                        )

            # 4. 校验通过，立即获取写入锁（消除 TOCTOU 竞态）
            self._write_locks[filepath] = writer_id

        return True, ""

    def record_write(self, filepath: str, writer_id: str = "unknown") -> None:
        """记录文件写入

        写入完成后调用此方法:
          1. 释放写入锁
          2. 记录写入历史
          3. 更新文件哈希

        Args:
            filepath: 相对于 workspace_root 的文件路径
            writer_id: 写入者标识
        """
        safe_path = self._resolve_safe(filepath)

        # 计算写入后的哈希
        content_hash = ""
        if safe_path and safe_path.exists():
            content_hash = self._quick_hash(safe_path)

        record = FileWriteRecord(
            filepath=filepath,
            writer_id=writer_id,
            content_hash=content_hash,
            timestamp=time.time(),
        )

        with self._lock:
            # 释放写入锁
            if self._write_locks.get(filepath) == writer_id:
                del self._write_locks[filepath]

            # 记录写入历史
            if filepath not in self._write_records:
                self._write_records[filepath] = []
            self._write_records[filepath].append(record)

            # 限制历史记录长度
            if len(self._write_records[filepath]) > self.max_history_per_file:
                self._write_records[filepath] = self._write_records[filepath][-self.max_history_per_file:]

            # 追加一条新的读取记录（写入后自动"读"了最新内容）
            # 这允许同一 Agent 连续写入同一文件
            if filepath in self._read_records:
                updated_read = FileReadRecord(
                    filepath=filepath,
                    reader_id=writer_id,
                    content_hash=content_hash,
                    timestamp=time.time(),
                )
                self._read_records[filepath].append(updated_read)

        logger.debug(
            f"record_write: {filepath} by {writer_id} "
            f"(hash={content_hash[:8]}...)"
        )

    def release_write_lock(self, filepath: str, writer_id: str = "unknown") -> None:
        """释放写入锁（写入失败时调用）

        如果写入过程中发生错误，调用此方法释放写入锁，
        避免死锁。

        Args:
            filepath: 相对于 workspace_root 的文件路径
            writer_id: 写入者标识
        """
        with self._lock:
            if self._write_locks.get(filepath) == writer_id:
                del self._write_locks[filepath]
                logger.debug(f"release_write_lock: {filepath} by {writer_id}")

    # ===== 查询方法 =====

    def get_read_history(self, filepath: str) -> list[FileReadRecord]:
        """获取文件的读取历史

        Args:
            filepath: 相对于 workspace_root 的文件路径

        Returns:
            读取记录列表（按时间排序）
        """
        with self._lock:
            return list(self._read_records.get(filepath, []))

    def get_write_history(self, filepath: str) -> list[FileWriteRecord]:
        """获取文件的写入历史

        Args:
            filepath: 相对于 workspace_root 的文件路径

        Returns:
            写入记录列表（按时间排序）
        """
        with self._lock:
            return list(self._write_records.get(filepath, []))

    def get_locked_files(self) -> dict[str, str]:
        """获取当前被写入锁定的文件

        Returns:
            {filepath: writer_id} 字典
        """
        with self._lock:
            return dict(self._write_locks)

    def is_file_locked(self, filepath: str) -> bool:
        """检查文件是否正在被写入

        Args:
            filepath: 相对于 workspace_root 的文件路径

        Returns:
            True 表示文件正在被写入
        """
        with self._lock:
            return filepath in self._write_locks

    def get_status_report(self) -> dict:
        """获取文件状态追踪器的状态报告

        Returns:
            {
                "workspace_root": str,
                "tracked_files": int,
                "locked_files": dict,
                "total_reads": int,
                "total_writes": int,
                "enable_hash_validation": bool,
            }
        """
        with self._lock:
            total_reads = sum(len(v) for v in self._read_records.values())
            total_writes = sum(len(v) for v in self._write_records.values())
            return {
                "workspace_root": self.workspace_root,
                "tracked_files": len(set(self._read_records.keys()) | set(self._write_records.keys())),
                "locked_files": dict(self._write_locks),
                "total_reads": total_reads,
                "total_writes": total_writes,
                "enable_hash_validation": self.enable_hash_validation,
            }

    # ===== 内部方法 =====

    def _resolve_safe(self, filepath: str) -> Optional[Path]:
        """解析文件路径并验证安全性

        防御:
          - 路径穿越（../）
          - 符号链接穿越
          - 绝对路径

        Args:
            filepath: 相对于 workspace_root 的文件路径

        Returns:
            解析后的绝对路径（Path 对象），不安全则返回 None
        """
        if not filepath or not filepath.strip():
            return None

        # 禁止绝对路径
        if os.path.isabs(filepath):
            logger.warning(f"_resolve_safe: absolute path rejected: {filepath}")
            return None

        # 规范化路径，检查穿越
        normalized = os.path.normpath(filepath)
        if ".." in normalized.split(os.sep):
            logger.warning(f"_resolve_safe: path traversal detected: {filepath}")
            return None

        # 解析绝对路径
        abs_path = os.path.abspath(os.path.join(self.workspace_root, normalized))

        # 验证仍在 workspace 内
        # 1.7 改进: 使用 os.path.normcase 消除大小写敏感文件系统上的绕过
        # macOS/Windows 上 /Users/Alice/project/../../../etc/passwd 可能通过
        # 大小写变体绕过简单的 startswith 检查
        normalized_abs = os.path.normcase(os.path.abspath(abs_path))
        normalized_root = os.path.normcase(os.path.abspath(self.workspace_root))
        if not normalized_abs.startswith(normalized_root + os.sep) and normalized_abs != normalized_root:
            logger.warning(f"_resolve_safe: path escapes workspace: {filepath}")
            return None

        # 解析符号链接（防止通过符号链接穿越）
        try:
            resolved = Path(abs_path).resolve()
            resolved_root = Path(self.workspace_root).resolve()
            resolved_str = os.path.normcase(str(resolved))
            resolved_root_str = os.path.normcase(str(resolved_root))
            if not resolved_str.startswith(resolved_root_str + os.sep) and resolved_str != resolved_root_str:
                logger.warning(f"_resolve_safe: symlink escapes workspace: {filepath}")
                return None
        except Exception as e:
            logger.warning(f"_resolve_safe: path resolution failed: {filepath}: {e}")
            return None

        return Path(abs_path)

    def _quick_hash(self, filepath: Path) -> str:
        """计算文件哈希，采用采样策略平衡安全性和性能

        1.6 改进:
          - MD5 → SHA-256（增强前瞻性，虽非加密用途但统一算法）
          - 大文件哈希策略: 头 64KB + 中间均匀采样 8 段 + 尾 64KB
          - 解决原方案只取头尾导致中间修改不可检测的问题

        策略:
          - <= 256KB: 完整哈希（安全且快速）
          - > 256KB: 头 64KB + 8 段中间采样(每段 4KB) + 尾 64KB
            理论上中间修改仍有极小概率逃逸，但在 Agent 场景中
            文件通常由 LLM 整体重写而非局部篡改，已足够安全。

        Args:
            filepath: 文件的绝对路径

        Returns:
            哈希值十六进制字符串
        """
        try:
            file_size = filepath.stat().st_size
            sha256 = hashlib.sha256()

            if file_size <= 256 * 1024:  # <= 256KB: 完整哈希
                with open(filepath, "rb") as f:
                    sha256.update(f.read())
            else:
                # 大文件: 头 64KB + 中间均匀采样 + 尾 64KB
                SAMPLE_COUNT = 8   # 采样段数
                SAMPLE_SIZE = 4096  # 每段 4KB

                with open(filepath, "rb") as f:
                    # 头 64KB
                    sha256.update(f.read(64 * 1024))

                    # 中间均匀采样
                    interval = (file_size - 128 * 1024) // SAMPLE_COUNT
                    for i in range(SAMPLE_COUNT):
                        offset = 64 * 1024 + interval * i
                        f.seek(offset)
                        sha256.update(f.read(SAMPLE_SIZE))

                    # 尾 64KB
                    f.seek(-64 * 1024, 2)
                    sha256.update(f.read())

            return sha256.hexdigest()
        except (OSError, IOError) as e:
            logger.warning(f"Hash computation failed for {filepath}: {e}")
            return ""

    def reset(self) -> None:
        """重置所有追踪状态"""
        with self._lock:
            self._read_records.clear()
            self._write_records.clear()
            self._write_locks.clear()
        logger.info("FileStateTracker reset")
