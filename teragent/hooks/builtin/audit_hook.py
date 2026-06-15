# teragent/hooks/builtin/audit_hook.py
"""审计 Hook -- Phase 9.2

记录所有工具调用到审计日志系统（teragent.security.audit）。

功能:
  - PRE_TOOL_USE: 记录工具调用请求（工具名 + 参数）
  - POST_TOOL_USE: 记录工具执行结果（成功/失败 + 耗时）
  - 不拦截任何调用，始终返回 ALLOW

配置示例 (agent.toml)::
    [hooks]
    # 通过配置加载
    pre_tool_use = [
        { type = "python", name = "audit", module = "teragent.hooks.builtin.audit_hook", function = "create_audit_hook" }
    ]
"""
import logging
import time

__all__ = [
    "AuditHook",
    "create_audit_hook",
]

from teragent.hooks.manager import (
    HookContext,
    HookDecision,
    HookEvent,
    HookResult,
    PythonHook,
)

logger = logging.getLogger(__name__)


def create_audit_hook() -> "AuditHook":
    """工厂函数：创建审计 Hook 实例

    供配置文件通过 type="python" 加载使用。

    Returns:
        AuditHook 实例
    """
    return AuditHook()


class AuditHook(PythonHook):
    """审计 Hook -- 记录所有工具调用

    功能:
      - PRE_TOOL_USE: 记录工具调用请求
      - POST_TOOL_USE: 记录工具执行结果
      - 不拦截任何调用，始终返回 ALLOW

    与现有 audit.py 的关系:
      - 复用 log_audit() 写入审计数据库
      - 如果审计数据库未初始化，降级为 logger.info()

    使用示例::

        from teragent.hooks.builtin.audit_hook import AuditHook

        hook = AuditHook()
        manager.register(hook)

        # AuditHook 同时注册了 PRE_TOOL_USE 和 POST_TOOL_USE
        # pre_hook 和 post_hook 都会自动被注册
    """

    def __init__(self) -> None:
        # 审计 Hook 需要同时注册 PRE 和 POST 两个事件
        # 由于 PythonHook 只能绑定一个事件，这里创建两个独立的 Hook
        # 但对外表现为一个逻辑单元
        self._call_times: dict[str, float] = {}  # call_id -> start_time

        async def pre_handler(context: HookContext) -> HookResult:
            return await self._pre_tool_use(context)

        async def post_handler(context: HookContext) -> HookResult:
            return await self._post_tool_use(context)

        # 主 Hook 是 PRE_TOOL_USE
        super().__init__(
            name="audit",
            event=HookEvent.PRE_TOOL_USE,
            handler=pre_handler,
        )

        # POST_TOOL_USE Hook 作为附加属性
        self.post_hook = PythonHook(
            name="audit_post",
            event=HookEvent.POST_TOOL_USE,
            handler=post_handler,
        )

    async def _pre_tool_use(self, context: HookContext) -> HookResult:
        """PRE_TOOL_USE: 记录工具调用请求"""
        call_id = context.metadata.get("call_id", "")
        self._call_times[call_id] = time.time()

        # Evict stale entries older than 5 minutes to prevent unbounded growth
        now = time.time()
        stale = [k for k, v in self._call_times.items() if now - v > 300]
        for k in stale:
            del self._call_times[k]

        # 尝试写入审计数据库
        await self._log_audit(
            action=f"tool.{context.tool_name}.call",
            details=(
                f"Tool: {context.tool_name}, "
                f"Params: {self._truncate_params(context.params)}, "
                f"CallID: {call_id}"
            ),
        )

        return HookResult(decision=HookDecision.ALLOW)

    async def _post_tool_use(self, context: HookContext) -> HookResult:
        """POST_TOOL_USE: 记录工具执行结果"""
        call_id = context.metadata.get("call_id", "")
        start_time = self._call_times.pop(call_id, None)
        duration = (
            f"{time.time() - start_time:.3f}s"
            if start_time is not None
            else "unknown"
        )

        success = False
        result_info = ""
        if context.result is not None:
            success = context.result.get("success", False)
            if success:
                result_info = "Success"
            else:
                error = context.result.get("error", "")
                result_info = f"Failed: {error[:100]}"

        await self._log_audit(
            action=f"tool.{context.tool_name}.result",
            details=(
                f"Tool: {context.tool_name}, "
                f"Result: {result_info}, "
                f"Duration: {duration}, "
                f"CallID: {call_id}"
            ),
        )

        return HookResult(decision=HookDecision.ALLOW)

    @staticmethod
    async def _log_audit(action: str, details: str) -> None:
        """写入审计日志

        Phase 6.1: 优先使用 teragent.security.audit.log_audit()，
        如果不可用则降级为 logger.info()
        """
        try:
            from teragent.security.audit import log_audit
            await log_audit(action, details)
        except Exception as e:
            logger.info(f"[AUDIT] {action}: {details} (db_error: {e})")

    def register_to(self, manager) -> None:
        """Register both pre and post hooks to a HookManager"""
        manager.register(self)
        manager.register(self.post_hook)

    @staticmethod
    def _truncate_params(params: dict, max_len: int = 200) -> str:
        """截断参数字典的字符串表示

        Args:
            params: 参数字典
            max_len: 最大长度

        Returns:
            截断后的字符串
        """
        import json
        try:
            text = json.dumps(params, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(params)
        if len(text) > max_len:
            return text[:max_len] + "..."
        return text
