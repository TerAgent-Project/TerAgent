# teragent/hooks/builtin/__init__.py
"""内置 Hook 集合 -- Phase 9.2

提供常用的内置 Hook:
  - AuditHook: 审计 Hook，记录所有工具调用到审计日志
  - DangerousCommandHook: 危险命令拦截 Hook，阻止 rm -rf / sudo 等命令
"""
from teragent.hooks.builtin.audit_hook import AuditHook
from teragent.hooks.builtin.dangerous_command_hook import DangerousCommandHook

__all__ = [
    "AuditHook",
    "DangerousCommandHook",
]
