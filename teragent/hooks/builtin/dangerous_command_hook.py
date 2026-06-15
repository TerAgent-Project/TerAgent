# teragent/hooks/builtin/dangerous_command_hook.py
"""危险命令拦截 Hook -- Phase 9.2

检测并拦截 execute_subtask 工具中的危险命令模式，
防止 Agent 执行可能造成不可逆损害的操作。

功能:
  - 拦截 rm -rf / sudo / mkfs / dd if= 等危险命令
  - 对 pip install / npm install 等安装命令发出警告但允许执行
  - 委托 sandbox.classify_command_risk() 进行统一风险评估
  - 仅对 execute_subtask 工具生效，其他工具直接放行
  - 拦截时返回 DENY + 具体原因

注意: extra_patterns / extra_warning_patterns 参数当前未生效，
因为风险评估已委托给 sandbox.classify_command_risk()。
如需自定义模式，请修改 sandbox 模块的黑名单。

配置示例 (agent.toml)::
    [hooks]
    pre_tool_use = [
        { type = "python", name = "dangerous_command", module = "teragent.hooks.builtin.dangerous_command_hook", function = "create_dangerous_command_hook" }
    ]

注意: extra_patterns / extra_warning_patterns 参数当前未生效，
因为风险评估已委托给 sandbox.classify_command_risk()。
如需自定义模式，请修改 sandbox 模块的黑名单。
"""
import logging

__all__ = [
    "DangerousCommandHook",
    "create_dangerous_command_hook",
]

from teragent.hooks.manager import (
    HookContext,
    HookDecision,
    HookEvent,
    HookResult,
    PythonHook,
)

logger = logging.getLogger(__name__)

# Note: DEFAULT_DANGEROUS_PATTERNS and WARNING_PATTERNS were previously defined
# here as module-level constants but were never referenced — all risk classification
# is delegated to sandbox.classify_command_risk(). They have been removed to avoid
# dead code. If you need the pattern lists, find them in teragent.security.sandbox.


def create_dangerous_command_hook() -> "DangerousCommandHook":
    """工厂函数：创建危险命令拦截 Hook 实例

    供配置文件通过 type="python" 加载使用。

    Returns:
        DangerousCommandHook 实例
    """
    return DangerousCommandHook()


class DangerousCommandHook(PythonHook):
    """危险命令拦截 Hook

    检测 execute_subtask 工具中的危险命令模式并拦截。
    风险评估委托给 sandbox.classify_command_risk()，确保与沙箱模块一致。

    功能:
      - 仅对 execute_subtask 工具生效
      - CRITICAL 级别命令返回 DENY
      - WARNING 级别命令返回 ALLOW 但记录警告日志
      - SAFE 级别命令直接放行

    注意:
      extra_patterns 和 extra_warning_patterns 参数当前未生效，
      因为风险评估已委托给 sandbox.classify_command_risk()。
      如需自定义危险模式，请修改 sandbox 模块的黑名单规则。

    使用示例::

        from teragent.hooks.builtin.dangerous_command_hook import DangerousCommandHook

        # 使用默认模式
        hook = DangerousCommandHook()
        manager.register(hook)
    """

    def __init__(
        self,
        extra_patterns: list[str] | None = None,
        extra_warning_patterns: list[str] | None = None,
    ) -> None:
        # Note: DEFAULT_DANGEROUS_PATTERNS and WARNING_PATTERNS are NOT stored
        # locally because all risk classification is delegated to
        # sandbox.classify_command_risk(). Only user-provided extra patterns
        # are checked locally before delegating to the sandbox.

        # Validate and store user-provided extra patterns (regex)
        import re as _re
        self._extra_dangerous_patterns: list[str] = []
        for p in (extra_patterns or []):
            try:
                _re.compile(p)  # validate regex
                self._extra_dangerous_patterns.append(p)
            except _re.error as e:
                logger.warning(f"Invalid regex pattern '{p}': {e}")

        self._extra_warning_patterns: list[str] = []
        for p in (extra_warning_patterns or []):
            try:
                _re.compile(p)  # validate regex
                self._extra_warning_patterns.append(p)
            except _re.error as e:
                logger.warning(f"Invalid regex pattern '{p}': {e}")

        async def handler(context: HookContext) -> HookResult:
            return await self._check_command(context)

        super().__init__(
            name="dangerous_command",
            event=HookEvent.PRE_TOOL_USE,
            handler=handler,
        )

    async def _check_command(self, context: HookContext) -> HookResult:
        """检查命令是否包含危险模式

        Args:
            context: Hook 执行上下文

        Returns:
            HookResult 决策结果
        """
        # 仅对 execute_subtask 工具生效
        if context.tool_name != "execute_subtask":
            return HookResult(decision=HookDecision.PASSTHROUGH)

        # 提取命令
        command = str(context.params.get("command", ""))
        if not command:
            return HookResult(decision=HookDecision.ALLOW)

        # 先检查用户自定义的额外危险模式（非默认模式，默认模式由 classify_command_risk 处理）
        import re
        for pattern in self._extra_dangerous_patterns:
            if re.search(pattern, command):
                return HookResult(
                    decision=HookDecision.DENY,
                    reason=f"Matched extra dangerous pattern: {pattern}",
                )

        # Use unified risk classification from sandbox module
        # ALWAYS check sandbox classification — it cannot be bypassed by warning patterns
        from teragent.security.sandbox import CommandRiskLevel, classify_command_risk

        risk_level, reason = classify_command_risk(command)

        if risk_level in (CommandRiskLevel.CRITICAL, CommandRiskLevel.DANGEROUS):
            full_reason = (
                f"[CRITICAL] {reason}. "
                f"Command: {self._truncate(command, 100)}"
            )
            logger.warning(f"DangerousCommandHook: {full_reason}")
            return HookResult(
                decision=HookDecision.DENY,
                reason=full_reason,
            )
        elif risk_level == CommandRiskLevel.WARNING:
            full_reason = (
                f"[WARN] {reason}. "
                f"Command: {self._truncate(command, 100)}. "
                f"Allowed but logged for audit."
            )
            logger.warning(f"DangerousCommandHook: {full_reason}")
            return HookResult(decision=HookDecision.ALLOW, reason=full_reason)

        # SAFE level — also check extra warning patterns before allowing
        for pattern in self._extra_warning_patterns:
            if re.search(pattern, command):
                return HookResult(
                    decision=HookDecision.ALLOW,
                    reason=f"Matched extra warning pattern: {pattern}",
                )

        return HookResult(decision=HookDecision.ALLOW)

    @staticmethod
    def _truncate(text: str, max_len: int = 100) -> str:
        """截断文本

        Args:
            text: 原始文本
            max_len: 最大长度

        Returns:
            截断后的文本
        """
        if len(text) > max_len:
            return text[:max_len] + "..."
        return text
