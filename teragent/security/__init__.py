"""teragent/security/ — 安全架构模块（权限 + 沙箱 + 文件安全 + 审计）

所有安全功能完整保留：
  - 7 层权限解析
  - 3 级沙箱（增强黑名单+进程隔离 / Docker / Firecracker）
  - 原子写入 + 读后写契约
  - 审计日志
  - AI 权限分类器
"""

from teragent.security.permission import (
    PermissionManager,
    PermissionLevel,
    EnhancedPermissionManager,
    PermissionRule,
    PermissionEffect,
)
from teragent.security.file_state import FileStateTracker
from teragent.security.file_writer import atomic_write_file, write_files_safely
from teragent.security.ai_permission_classifier import AIPermissionClassifier
from teragent.security.sandbox import execute_in_sandbox, check_command_safety
from teragent.security.audit import AuditLogger

__all__ = [
    "PermissionManager",
    "PermissionLevel",
    "EnhancedPermissionManager",
    "PermissionRule",
    "PermissionEffect",
    "AIPermissionClassifier",
    "FileStateTracker",
    "atomic_write_file",
    "write_files_safely",
    "execute_in_sandbox",
    "check_command_safety",
    "AuditLogger",
]
