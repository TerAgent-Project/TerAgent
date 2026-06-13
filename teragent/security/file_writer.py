# teragent/security/file_writer.py
"""安全文件写入 — 增强版 + 两阶段提交

核心能力:
  1. 原子写入（Atomic Write）
     - 写入临时文件 → fsync → os.replace 原子替换
     - 避免写入过程中崩溃导致文件损坏
     - 避免并发读看到半写状态

  2. 读后写契约校验（Read-Before-Write Contract）
     - 集成 FileStateTracker，写入前校验文件自读取后是否被修改
     - 过时写入被拒绝，返回明确错误信息

  3. 路径穿越防护
     - 所有文件路径必须在 workspace_root 内
     - 防止 ../ 和符号链接穿越

  4. 批量写入两阶段提交（1.8 新增）
     - Phase 1: 验证所有文件可写入
     - Phase 2: 写入所有临时文件
     - Phase 3: 原子替换（全部成功才提交）
     - Phase 4: 失败时从备份回滚

原子写入流程:
  1. 验证路径安全（路径穿越防护）
  2. FileStateTracker.validate_write() 读后写契约校验
  3. tempfile.mkstemp() 创建临时文件
  4. 写入内容 + f.flush() + os.fsync()
  5. os.replace() 原子替换
  6. FileStateTracker.record_write() 记录写入
"""

import asyncio
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from teragent.security.permission import PermissionLevel, PermissionManager
from teragent.utils.exceptions import SandboxViolation

# Phase 9.4: Enhanced permission support
if TYPE_CHECKING:
    from teragent.security.file_state import FileStateTracker
    from teragent.security.permission import EnhancedPermissionManager

logger = logging.getLogger(__name__)


def _sync_atomic_write(path: str, content: str) -> None:
    """原子写入文件 — 临时文件 + fsync + os.replace

    流程:
      1. 在同目录创建临时文件
      2. 写入内容 + flush + fsync
      3. os.replace() 原子替换目标文件

    优势:
      - 崩溃安全: 写入过程中崩溃不会损坏原文件
      - 并发安全: 读者永远不会看到半写状态
      - POSIX 原子: os.replace (rename) 在同一文件系统上是原子操作
      - Windows 非原子: os.replace 在 NTFS 上不是崩溃安全的原子操作

    Args:
        path: 目标文件绝对路径
        content: 文件内容

    Raises:
        IOError: 写入失败时抛出
    """
    dir_name = os.path.dirname(path)

    # 确保目录存在
    os.makedirs(dir_name, exist_ok=True)

    # 在同目录创建临时文件（确保同一文件系统，os.replace 才能原子替换）
    fd, tmp_path = tempfile.mkstemp(
        prefix=".teragent_tmp_",
        suffix=".tmp",
        dir=dir_name,
    )

    try:
        # 写入内容
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())  # 强制落盘

        # 原子替换
        os.replace(tmp_path, path)
        # 在某些系统上，还需要 fsync 目录以确保替换持久化
        try:
            dir_fd = os.open(dir_name, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass  # 目录 fsync 是 best-effort

    except BaseException:
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


async def atomic_write_file(
    filepath: str,
    content: str,
    workspace_root: str,
    file_state_tracker: "FileStateTracker | None" = None,
    writer_id: str = "unknown",
    perm_mgr: PermissionManager | None = None,
    enhanced_perm_mgr: "EnhancedPermissionManager | None" = None,
) -> tuple[bool, str]:
    """原子写入单个文件 -- 含完整安全检查

    完整流程:
      1. 权限检查（EnhancedPermissionManager 优先，PermissionManager 回退）
      2. 路径穿越防护
      3. 读后写契约校验（FileStateTracker）
      4. 原子写入（临时文件 + os.replace）
      5. 记录写入（FileStateTracker）

    Args:
        filepath: 相对于 workspace_root 的文件路径
        content: 文件内容
        workspace_root: 工作区根目录的绝对路径
        file_state_tracker: 文件状态追踪器（可选，None 则跳过契约校验）
        writer_id: 写入者标识
        perm_mgr: 权限管理器实例（Phase 5.3 兼容）
        enhanced_perm_mgr: [Phase 9.4] 增强权限管理器实例

    Returns:
        (success, message) -- 是否成功 + 消息
    """
    # 1. 权限检查（Phase 9.4: EnhancedPermissionManager 优先）
    if enhanced_perm_mgr:
        allowed, reason = enhanced_perm_mgr.check("write_file", path=filepath)
        if not allowed:
            raise SandboxViolation(f"Write denied by permission rule: {reason}")
    elif perm_mgr and not perm_mgr.check_level(PermissionLevel.PLAN):
        raise SandboxViolation("Write operation requires PLAN level permission")

    abs_root = os.path.realpath(workspace_root)
    abs_path = os.path.realpath(os.path.join(abs_root, filepath))

    # 2. 路径穿越防护
    if not (abs_path.startswith(abs_root + os.sep) or abs_path == abs_root):
        raise SandboxViolation(f"Path traversal detected: {filepath}")

    # 3. 读后写契约校验
    if file_state_tracker:
        allowed, reason = file_state_tracker.validate_write(filepath, writer_id)
        if not allowed:
            logger.warning(f"atomic_write_file: write rejected for {filepath}: {reason}")
            return False, reason

    # 4. 原子写入
    try:
        await asyncio.get_running_loop().run_in_executor(
            None, _sync_atomic_write, abs_path, content
        )
        logger.info(f"atomic_write_file: successfully wrote {filepath}")
    except IOError as e:
        error_str = str(e)
        # 写入失败，释放 FileStateTracker 写入锁
        if file_state_tracker:
            file_state_tracker.release_write_lock(filepath, writer_id)

        if "Permission denied" in error_str or "Errno 13" in error_str:
            logger.error(
                f"Permission denied writing {filepath}: {e}. "
                f"Check directory permissions and write access."
            )
        else:
            logger.error(f"Failed to write {filepath}: {e}")
        return False, f"写入失败: {error_str}"
    except Exception as e:
        # 写入失败，释放 FileStateTracker 写入锁
        if file_state_tracker:
            file_state_tracker.release_write_lock(filepath, writer_id)
        logger.error(f"Unexpected error writing {filepath}: {e}")
        return False, f"写入失败: {str(e)}"

    # 5. 记录写入
    if file_state_tracker:
        file_state_tracker.record_write(filepath, writer_id)

    # 审计日志
    try:
        from teragent.security.audit import log_audit
        await log_audit("file_write", f"Atomic write: {filepath} by {writer_id}")
    except Exception as e:
        logger.debug(f"Audit logging failed (non-blocking): {e}")

    return True, f"成功写入 {filepath}"


async def write_files_safely(
    files_dict: dict[str, str],
    workspace_root: str,
    perm_mgr: PermissionManager | None = None,
    file_state_tracker: "FileStateTracker | None" = None,
    writer_id: str = "unknown",
    enhanced_perm_mgr: "EnhancedPermissionManager | None" = None,
) -> tuple[list[str], list[str]]:
    """安全批量写入文件 — 1.8 两阶段提交增强版

    改进（1.8 两阶段提交）:
      - Phase 1: 验证所有文件可写入（权限 + 路径 + 契约）
      - Phase 2: 写入所有临时文件（.tmp 后缀）
      - Phase 3: 原子替换（os.replace），全部成功才提交
      - Phase 4: 如果有替换失败，从备份（.bak）回滚

    旧行为: 逐文件写入，部分失败时项目处于不一致状态
    新行为: 要么全部成功，要么回滚到写入前状态

    向后兼容: 返回值仍为 (written_files, failed_files)，但失败时
    所有文件都会被回滚。

    Args:
        files_dict: 相对路径到文件内容的映射
        workspace_root: 工作区根目录的绝对路径
        perm_mgr: 权限管理器实例（Phase 5.3 兼容）
        file_state_tracker: 文件状态追踪器（Phase 5.3）
        writer_id: 写入者标识（Phase 5.3）
        enhanced_perm_mgr: [Phase 9.4] 增强权限管理器实例

    Returns:
        (成功写入的文件列表, 失败的文件列表)
    """
    if not files_dict:
        return [], []

    # 审计日志
    try:
        from teragent.security.audit import log_audit
        await log_audit("file_write", f"Batch write (2PC): {len(files_dict)} files to {workspace_root}")
    except Exception as e:
        logger.debug(f"Audit logging failed (non-blocking): {e}")

    # 权限检查（Phase 9.4: EnhancedPermissionManager 优先）
    if enhanced_perm_mgr:
        for rel_path in files_dict:
            allowed, reason = enhanced_perm_mgr.check("write_file", path=rel_path)
            if not allowed:
                raise SandboxViolation(f"Write denied by permission rule for {rel_path}: {reason}")
    elif perm_mgr and not perm_mgr.check_level(PermissionLevel.PLAN):
        raise SandboxViolation("Write operation requires PLAN level permission")

    abs_root = os.path.realpath(workspace_root)

    # ===== Phase 1: 验证所有文件可写入 =====
    temp_files: dict[str, Path] = {}
    backup_files: dict[str, Path | None] = {}

    for rel_path in files_dict:
        # 1a. 路径穿越防护
        abs_path = os.path.realpath(os.path.join(abs_root, rel_path))
        if not (abs_path.startswith(abs_root + os.sep) or abs_path == abs_root):
            # 已验证失败的文件，释放之前获取的写入锁
            _release_all_locks(file_state_tracker, temp_files, writer_id)
            raise SandboxViolation(f"Path traversal detected: {rel_path}")

        # 1b. 读后写契约校验（获取写入锁）
        if file_state_tracker:
            allowed, reason = file_state_tracker.validate_write(rel_path, writer_id)
            if not allowed:
                logger.warning(f"write_files_safely: validation failed for {rel_path}: {reason}")
                # 释放之前已获取的写入锁
                _release_all_locks(file_state_tracker, temp_files, writer_id)
                return [], [rel_path]

        temp_files[rel_path] = Path(abs_path)

    # ===== Phase 2: 写入所有临时文件 =====
    processed_contents = files_dict

    # 写入临时文件并备份已有文件
    tmp_paths: dict[str, Path] = {}
    write_errors: list[str] = []

    for rel_path, abs_path in temp_files.items():
        content = processed_contents[rel_path]
        tmp_path = abs_path.with_suffix(abs_path.suffix + f'.tmp_{int(time.monotonic_ns())}')

        try:
            # 备份已有文件
            if abs_path.exists():
                bak_path = abs_path.with_suffix(abs_path.suffix + f'.bak_{int(time.monotonic_ns())}')
                shutil.copy2(str(abs_path), str(bak_path))
                backup_files[rel_path] = bak_path
            else:
                backup_files[rel_path] = None

            # 确保目录存在
            abs_path.parent.mkdir(parents=True, exist_ok=True)

            # 写入临时文件
            await asyncio.get_running_loop().run_in_executor(
                None, _sync_atomic_write, str(tmp_path), content
            )
            tmp_paths[rel_path] = tmp_path

        except Exception as e:
            logger.error(f"Failed to write temp file for {rel_path}: {e}")
            write_errors.append(rel_path)
            break  # 任一文件写入失败，停止后续写入

    # 如果临时文件写入失败，回滚
    if write_errors:
        # 清理临时文件
        for tmp_path in tmp_paths.values():
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        # 清理备份文件
        for bak_path in backup_files.values():
            if bak_path and bak_path.exists():
                try:
                    bak_path.unlink(missing_ok=True)
                except OSError:
                    pass
        # 释放写入锁
        _release_all_locks(file_state_tracker, temp_files, writer_id)
        return [], write_errors

    # ===== Phase 3: 原子替换（全部成功才提交） =====
    committed_files: list[str] = []
    failed_files: list[str] = []

    for rel_path, tmp_path in tmp_paths.items():
        abs_path = temp_files[rel_path]
        try:
            os.replace(str(tmp_path), str(abs_path))
            committed_files.append(rel_path)
        except OSError as e:
            logger.error(f"Atomic replace failed: {tmp_path} -> {abs_path}: {e}")
            failed_files.append(rel_path)

    # ===== Phase 4: 如果有替换失败，从备份回滚 =====
    if failed_files:
        logger.warning(
            f"write_files_safely: {len(failed_files)} files failed to commit, "
            f"rolling back {len(committed_files)} committed files"
        )
        # 回滚已提交的文件
        for rel_path in committed_files:
            bak_path = backup_files.get(rel_path)
            abs_path = temp_files[rel_path]
            if bak_path and bak_path.exists():
                try:
                    os.replace(str(bak_path), str(abs_path))
                    logger.info(f"Rolled back: {rel_path}")
                except OSError as e:
                    logger.error(f"Rollback failed for {rel_path}: {e}")
            elif bak_path is None:
                # 文件原本不存在，删除新创建的文件
                try:
                    abs_path.unlink(missing_ok=True)
                    logger.info(f"Rolled back (deleted new file): {rel_path}")
                except OSError as e:
                    logger.error(f"Rollback delete failed for {rel_path}: {e}")

        # 清理所有备份文件
        _cleanup_backups(backup_files)

        # 释放写入锁
        _release_all_locks(file_state_tracker, temp_files, writer_id)

        return [], failed_files

    # 全部成功 — 清理备份文件
    _cleanup_backups(backup_files)

    # 记录写入（FileStateTracker）
    if file_state_tracker:
        for rel_path in committed_files:
            file_state_tracker.record_write(rel_path, writer_id)

    logger.info(f"write_files_safely: committed {len(committed_files)} files successfully")
    return committed_files, []


def _release_all_locks(
    file_state_tracker: "FileStateTracker | None",
    file_paths: dict[str, Any],
    writer_id: str,
) -> None:
    """释放所有已获取的写入锁"""
    if not file_state_tracker:
        return
    for rel_path in file_paths:
        file_state_tracker.release_write_lock(rel_path, writer_id)


def _cleanup_backups(backup_files: dict[str, Path | None]) -> None:
    """清理所有备份文件"""
    for bak_path in backup_files.values():
        if bak_path and bak_path.exists():
            try:
                bak_path.unlink(missing_ok=True)
            except OSError:
                pass
