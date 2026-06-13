# teragent/hooks/__init__.py
"""Hook 系统 -- Phase 9.2

参考 Claude-Code 的 PreToolUse / PostToolUse hooks 机制，
允许外部程序拦截、修改、允许/拒绝工具调用和模型调用。

核心组件:
  - HookEvent: Hook 事件类型枚举
  - HookDecision: Hook 决策枚举
  - HookContext: Hook 执行上下文
  - HookResult: Hook 返回结果
  - Hook: Hook 基类
  - ShellHook: Shell 命令 Hook（外部脚本）
  - PythonHook: Python 函数 Hook（内置逻辑）
  - HookManager: Hook 管理器（注册、执行、链式决策）

使用方式::

    from teragent.hooks import HookManager, PythonHook, HookEvent, HookDecision

    manager = HookManager()

    # 注册内置 Python Hook
    async def my_handler(context):
        if context.tool_name == "execute_subtask":
            return HookResult(decision=HookDecision.ALLOW)
        return HookResult(decision=HookDecision.PASSTHROUGH)

    manager.register(PythonHook("my_hook", HookEvent.PRE_TOOL_USE, my_handler))

    # 执行 hooks
    result = await manager.run_hooks(HookEvent.PRE_TOOL_USE, context)
    if result.decision == HookDecision.DENY:
        print(f"Blocked: {result.reason}")
"""
from teragent.hooks.manager import (
    Hook,
    HookContext,
    HookDecision,
    HookEvent,
    HookManager,
    HookResult,
    PythonHook,
    ShellHook,
)

__all__ = [
    "HookEvent",
    "HookDecision",
    "HookContext",
    "HookResult",
    "Hook",
    "ShellHook",
    "PythonHook",
    "HookManager",
]
