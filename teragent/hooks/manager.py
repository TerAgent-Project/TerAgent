# teragent/hooks/manager.py
"""Hook 管理器 -- Phase 9.2 核心组件

参考 Claude-Code 的 PreToolUse / PostToolUse hooks 机制，
实现可扩展的 Hook 框架，支持外部脚本和内置 Python 函数两种 Hook 类型。

核心设计:
  - HookEvent: 4 种事件类型 (PRE_TOOL_USE / POST_TOOL_USE / PRE_MODEL_CALL / POST_MODEL_CALL)
  - HookDecision: 4 种决策 (ALLOW / DENY / MODIFY / PASSTHROUGH)
  - 链式执行: 按 Hook 注册顺序执行，任一 DENY 则终止，MODIFY 修改参数后继续
  - Shell 超时: ShellHook 默认 10 秒超时，超时返回 PASSTHROUGH
  - 错误容忍: Hook 执行异常不中断主流程，记录日志后继续

使用示例::

    from teragent.hooks.manager import HookManager, PythonHook, HookEvent, HookDecision, HookContext

    # 1. 创建管理器
    manager = HookManager()

    # 2. 注册 Python Hook
    async def deny_rm_rf(context: HookContext) -> HookResult:
        if context.tool_name == "execute_subtask":
            cmd = str(context.params.get("command", ""))
            if "rm -rf" in cmd:
                return HookResult(decision=HookDecision.DENY, reason="rm -rf not allowed")
        return HookResult(decision=HookDecision.PASSTHROUGH)

    manager.register(PythonHook("deny_rm_rf", HookEvent.PRE_TOOL_USE, deny_rm_rf))

    # 3. 注册 Shell Hook
    manager.register(ShellHook(
        name="my_shell_hook",
        event=HookEvent.PRE_TOOL_USE,
        command="python /path/to/hook_script.py",
    ))

    # 4. 执行 hooks
    context = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="execute_subtask", params={"command": "rm -rf /"})
    result = await manager.run_hooks(HookEvent.PRE_TOOL_USE, context)
    if result.decision == HookDecision.DENY:
        print(f"Blocked: {result.reason}")
"""
import asyncio
import json
import logging
import shlex
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

__all__ = [
    "Hook",
    "HookContext",
    "HookDecision",
    "HookEvent",
    "HookManager",
    "HookResult",
    "PythonHook",
    "ShellHook",
]

logger = logging.getLogger(__name__)


# ===== 枚举类型 =====

class HookEvent(Enum):
    """Hook 事件类型"""
    PRE_TOOL_USE = "pre_tool_use"       # 工具执行前
    POST_TOOL_USE = "post_tool_use"     # 工具执行后
    PRE_MODEL_CALL = "pre_model_call"   # 模型调用前
    POST_MODEL_CALL = "post_model_call" # 模型调用后


class HookDecision(Enum):
    """Hook 决策类型"""
    ALLOW = "allow"             # 允许继续
    DENY = "deny"               # 拒绝执行
    MODIFY = "modify"           # 修改参数后继续
    PASSTHROUGH = "passthrough"  # 不决策，交给下一个 hook


# ===== 数据类 =====

@dataclass
class HookContext:
    """Hook 执行上下文

    携带当前事件的全部信息，供 Hook 读取和决策。
    """
    event: HookEvent
    tool_name: str = ""
    params: dict = field(default_factory=dict)
    # POST_TOOL_USE 时有值：工具执行结果
    result: Optional[dict] = None
    # PRE_MODEL_CALL / POST_MODEL_CALL 时有值
    model_messages: Optional[list[dict]] = None
    # POST_MODEL_CALL 时有值：模型响应
    model_response: Optional[dict] = None
    # 元数据（时间戳、来源等）
    metadata: dict = field(default_factory=dict)


@dataclass
class HookResult:
    """Hook 返回结果

    由 Hook.execute() 返回，决定后续流程。
    """
    decision: HookDecision
    reason: str = ""
    # MODIFY 时的修改后参数（仅用于 PRE_TOOL_USE / PRE_MODEL_CALL）
    modified_params: Optional[dict] = None
    # MODIFY 时的修改后消息（仅用于 PRE_MODEL_CALL）
    modified_messages: Optional[list[dict]] = None


# ===== Hook 基类 =====

class Hook:
    """Hook 基类

    所有 Hook 必须实现 execute() 方法。
    子类需设置 name 和 event 属性。
    """
    name: str = ""
    event: HookEvent = HookEvent.PRE_TOOL_USE

    async def execute(self, context: HookContext) -> HookResult:
        """执行 Hook 逻辑

        Args:
            context: Hook 执行上下文

        Returns:
            HookResult 决策结果
        """
        raise NotImplementedError(
            f"Hook {self.name} must implement execute()"
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} event={self.event.value!r}>"


# ===== ShellHook =====

class ShellHook(Hook):
    """Shell 命令 Hook -- 参考 Claude-Code 的外部 hook 机制

    执行外部命令，通过 stdin/stdout 传递 JSON 数据。

    输入格式 (stdin JSON)::
        {
            "event": "pre_tool_use",
            "tool_name": "execute_subtask",
            "params": {"command": "rm -rf /"},
            "result": null,
            "metadata": {}
        }

    输出格式 (stdout JSON)::
        {
            "decision": "deny",       // allow | deny | modify | passthrough
            "reason": "dangerous",
            "modified_params": null    // only when decision == "modify"
        }

    约定:
      - 外部命令 exit code 0 表示成功，非 0 表示 Hook 失败（等同于 PASSTHROUGH）
      - 超时未返回视为 PASSTHROUGH
      - stdout 非 JSON 视为 PASSTHROUGH
    """

    # Shell hook 默认超时时间（秒）
    DEFAULT_TIMEOUT = 10.0

    def __init__(
        self,
        name: str,
        event: HookEvent,
        command: str,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.name = name
        self.event = event
        self.command = command
        self.timeout = timeout

    async def execute(self, context: HookContext) -> HookResult:
        """执行外部 Hook 命令

        通过 stdin 传递 JSON 格式的上下文，解析 stdout 的 JSON 结果。

        Args:
            context: Hook 执行上下文

        Returns:
            HookResult 决策结果
        """
        # 构建输入数据
        input_data: dict[str, Any] = {
            "event": context.event.value,
            "tool_name": context.tool_name,
            "params": context.params,
        }
        if context.result is not None:
            input_data["result"] = context.result
        if context.model_messages is not None:
            input_data["model_messages"] = context.model_messages
        if context.model_response is not None:
            input_data["model_response"] = context.model_response
        if context.metadata:
            input_data["metadata"] = context.metadata

        try:
            # 使用 shlex 分割命令（处理引号等）
            cmd_parts = shlex.split(self.command, posix=not sys.platform.startswith("win"))
            proc = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=json.dumps(input_data, ensure_ascii=False).encode()),
                timeout=self.timeout,
            )

            if proc.returncode != 0:
                stderr_text = stderr.decode(errors="replace")[:200]
                logger.warning(
                    f"ShellHook {self.name} exited with code {proc.returncode}: {stderr_text}"
                )
                return HookResult(decision=HookDecision.PASSTHROUGH)

            # 解析输出
            output_text = stdout.decode(errors="replace").strip()
            if not output_text:
                # 无输出 = 允许
                return HookResult(decision=HookDecision.ALLOW)

            output = json.loads(output_text)
            decision_str = output.get("decision", "allow")

            # 验证决策值
            try:
                decision = HookDecision(decision_str)
            except ValueError:
                logger.warning(
                    f"ShellHook {self.name} returned invalid decision: {decision_str}"
                )
                return HookResult(decision=HookDecision.PASSTHROUGH)

            return HookResult(
                decision=decision,
                reason=output.get("reason", ""),
                modified_params=output.get("modified_params"),
                modified_messages=output.get("modified_messages"),
            )

        except asyncio.TimeoutError:
            logger.warning(
                f"ShellHook {self.name} timed out after {self.timeout}s"
            )
            return HookResult(decision=HookDecision.PASSTHROUGH)
        except json.JSONDecodeError as e:
            logger.warning(
                f"ShellHook {self.name} returned invalid JSON: {e}"
            )
            return HookResult(decision=HookDecision.PASSTHROUGH)
        except Exception as e:
            logger.error(
                f"ShellHook {self.name} error: {e}", exc_info=True
            )
            return HookResult(decision=HookDecision.PASSTHROUGH)


# ===== PythonHook =====

class PythonHook(Hook):
    """Python 函数 Hook -- 内置 Hook 使用

    适用于审计、危险命令拦截等内置逻辑，
    不需要启动外部进程，执行效率更高。

    使用示例::

        async def audit_handler(context: HookContext) -> HookResult:
            logger.info(f"Tool {context.tool_name} called with {context.params}")
            return HookResult(decision=HookDecision.ALLOW)

        hook = PythonHook("audit", HookEvent.PRE_TOOL_USE, audit_handler)
        manager.register(hook)
    """

    def __init__(
        self,
        name: str,
        event: HookEvent,
        handler: Callable[[HookContext], Awaitable[HookResult]],
    ) -> None:
        self.name = name
        self.event = event
        self._handler = handler

    async def execute(self, context: HookContext) -> HookResult:
        """执行 Python Hook 函数

        Args:
            context: Hook 执行上下文

        Returns:
            HookResult 决策结果
        """
        try:
            return await self._handler(context)
        except Exception as e:
            logger.error(
                f"PythonHook {self.name} error: {e}", exc_info=True
            )
            return HookResult(decision=HookDecision.PASSTHROUGH)


# ===== HookManager =====

class HookManager:
    """Hook 管理器

    负责注册和执行 Hook 链。按注册顺序执行，支持链式决策:
      - DENY: 立即终止，返回拒绝结果
      - MODIFY: 修改上下文参数后继续执行后续 Hook
      - ALLOW: 标记为允许，继续执行后续 Hook
      - PASSTHROUGH: 不做决策，交给下一个 Hook

    如果所有 Hook 都未返回 DENY，最终结果为 ALLOW。
    如果没有注册任何 Hook，直接返回 ALLOW。

    使用示例::

        manager = HookManager()
        manager.register(PythonHook(...))
        manager.register(ShellHook(...))

        context = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="read_file", params={...})
        result = await manager.run_hooks(HookEvent.PRE_TOOL_USE, context)
        if result.decision == HookDecision.DENY:
            # 拒绝执行
            return ToolResult(success=False, data={}, error=f"Hook denied: {result.reason}")
    """

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[Hook]] = {
            event: [] for event in HookEvent
        }
        # Hook 执行统计
        self._stats: dict[str, int] = {
            "total_calls": 0,
            "allow_count": 0,
            "deny_count": 0,
            "modify_count": 0,
            "error_count": 0,
        }

    def register(self, hook: Hook) -> None:
        """注册 Hook

        Hook 按注册顺序执行。同一事件可以注册多个 Hook。
        If the hook has a companion post_hook attribute (e.g., AuditHook),
        it is automatically registered as well.

        Args:
            hook: Hook 实例（PythonHook / ShellHook / 自定义 Hook）
        """
        if not hook.name:
            logger.warning(f"Registering unnamed hook of type {type(hook).__name__}")
        self._hooks[hook.event].append(hook)
        logger.info(
            f"HookManager: registered {hook.name!r} for {hook.event.value}"
        )
        # Auto-register companion hooks (e.g., AuditHook.post_hook)
        if hasattr(hook, 'post_hook') and isinstance(hook.post_hook, Hook):
            self._hooks[hook.post_hook.event].append(hook.post_hook)
            logger.info(
                f"HookManager: auto-registered companion {hook.post_hook.name!r} "
                f"for {hook.post_hook.event.value}"
            )

    def unregister(self, name: str) -> bool:
        """按名称取消注册 Hook

        Args:
            name: Hook 名称

        Returns:
            是否成功取消注册
        """
        for event in HookEvent:
            hooks = self._hooks[event]
            for i, hook in enumerate(hooks):
                if hook.name == name:
                    hooks.pop(i)
                    logger.info(
                        f"HookManager: unregistered {name!r} from {event.value}"
                    )
                    return True
        return False

    async def run_hooks(
        self, event: HookEvent, context: HookContext
    ) -> HookResult:
        """运行指定事件的所有 Hook

        按注册顺序执行，任一 Hook 返回 DENY 则终止。
        任一 Hook 返回 MODIFY 则修改上下文参数后继续。
        Hook 执行异常不中断主流程，记录日志后继续。

        Args:
            event: 事件类型
            context: Hook 执行上下文（可能被 MODIFY 修改）

        Returns:
            最终决策结果
                - DENY: 任一 Hook 拒绝
                - MODIFY: 任一 Hook 修改了参数（使用最后修改的参数）
                - ALLOW: 所有 Hook 允许或无 Hook
        """
        self._stats["total_calls"] += 1
        hooks = self._hooks.get(event, [])

        if not hooks:
            self._stats["allow_count"] += 1
            return HookResult(decision=HookDecision.ALLOW)

        last_modified_params: Optional[dict] = None
        last_modified_messages: Optional[list[dict]] = None
        last_reason = ""

        for hook in hooks:
            try:
                result = await hook.execute(context)

                if result.decision == HookDecision.DENY:
                    # 拒绝: 立即终止
                    self._stats["deny_count"] += 1
                    logger.info(
                        f"HookManager: {hook.name} DENIED for "
                        f"{context.tool_name}: {result.reason}"
                    )
                    return result

                if result.decision == HookDecision.MODIFY:
                    # 修改: 更新上下文参数后继续
                    if result.modified_params is not None:
                        context.params = result.modified_params
                        last_modified_params = result.modified_params
                    if result.modified_messages is not None:
                        context.model_messages = result.modified_messages
                        last_modified_messages = result.modified_messages
                    last_reason = result.reason
                    logger.info(
                        f"HookManager: {hook.name} MODIFIED params for "
                        f"{context.tool_name}: {result.reason}"
                    )

            except Exception as e:
                self._stats["error_count"] += 1
                logger.error(
                    f"HookManager: hook {hook.name} raised exception: {e}",
                    exc_info=True,
                )
                continue

        # 如果有 MODIFY，返回修改后的结果
        if last_modified_params is not None or last_modified_messages is not None:
            self._stats["modify_count"] += 1
            return HookResult(
                decision=HookDecision.MODIFY,
                reason=last_reason,
                modified_params=last_modified_params,
                modified_messages=last_modified_messages,
            )

        # 所有 Hook 都允许
        self._stats["allow_count"] += 1
        return HookResult(decision=HookDecision.ALLOW)

    def get_hooks(self, event: HookEvent | None = None) -> list[Hook]:
        """获取已注册的 Hook 列表

        Args:
            event: 事件类型。None 则返回所有 Hook。

        Returns:
            Hook 实例列表
        """
        if event is not None:
            return list(self._hooks.get(event, []))
        result: list[Hook] = []
        for hooks in self._hooks.values():
            result.extend(hooks)
        return result

    def get_status_report(self) -> dict:
        """获取 Hook 管理器状态报告

        Returns:
            状态报告字典，包含已注册 Hook 列表和执行统计
        """
        hooks_info = {}
        for event in HookEvent:
            hooks_info[event.value] = [
                {
                    "name": h.name,
                    "type": type(h).__name__,
                }
                for h in self._hooks[event]
            ]
        return {
            "hooks": hooks_info,
            "stats": dict(self._stats),
        }

    def load_from_config(self, config: dict) -> None:
        """从配置字典加载 Hook

        配置格式（agent.toml [hooks] 段）::

            [hooks]
            pre_tool_use = [
                { type = "shell", name = "my_hook", command = "python hook.py" }
            ]
            post_tool_use = []
            pre_model_call = []
            post_model_call = []

        Args:
            config: 配置字典，键为事件类型，值为 Hook 配置列表
        """
        event_map = {
            "pre_tool_use": HookEvent.PRE_TOOL_USE,
            "post_tool_use": HookEvent.POST_TOOL_USE,
            "pre_model_call": HookEvent.PRE_MODEL_CALL,
            "post_model_call": HookEvent.POST_MODEL_CALL,
        }

        for config_key, event in event_map.items():
            hook_list = config.get(config_key, [])
            if not isinstance(hook_list, list):
                logger.warning(
                    f"HookManager: invalid config for {config_key}, expected list"
                )
                continue

            for hook_cfg in hook_list:
                if not isinstance(hook_cfg, dict):
                    logger.warning(
                        f"HookManager: invalid hook config in {config_key}: {hook_cfg}"
                    )
                    continue

                hook_type = hook_cfg.get("type", "").lower()
                hook_name = hook_cfg.get("name", f"config_{config_key}_{id(hook_cfg)}")

                if hook_type == "shell":
                    command = hook_cfg.get("command", "")
                    if not command:
                        logger.warning(
                            f"HookManager: shell hook {hook_name} missing 'command'"
                        )
                        continue
                    timeout = float(hook_cfg.get("timeout", ShellHook.DEFAULT_TIMEOUT))
                    self.register(ShellHook(hook_name, event, command, timeout))

                elif hook_type == "python":
                    # Python Hook 需要在代码中注册，配置文件只能声明
                    module_path = hook_cfg.get("module", "")
                    func_name = hook_cfg.get("function", "")
                    if module_path and func_name:
                        try:
                            import importlib
                            module = importlib.import_module(module_path)
                            func = getattr(module, func_name)
                            # Check if func is a factory that returns a Hook instance
                            # (e.g., create_audit_hook, create_dangerous_command_hook)
                            try:
                                result = func()
                                if isinstance(result, Hook):
                                    self.register(result)
                                else:
                                    # It's a raw handler function — wrap in PythonHook
                                    self.register(PythonHook(hook_name, event, func))
                            except TypeError:
                                # Factory requires no args but got some, or vice versa
                                # — treat as a raw handler function
                                self.register(PythonHook(hook_name, event, func))
                        except (ImportError, AttributeError) as e:
                            logger.warning(
                                f"HookManager: failed to load python hook "
                                f"{hook_name} from {module_path}.{func_name}: {e}"
                            )
                    else:
                        logger.warning(
                            f"HookManager: python hook {hook_name} missing "
                            f"'module' or 'function' in config"
                        )
                else:
                    logger.warning(
                        f"HookManager: unknown hook type {hook_type!r} for {hook_name}"
                    )

    def clear(self) -> None:
        """清除所有已注册的 Hook"""
        for event in HookEvent:
            self._hooks[event] = []
        logger.info("HookManager: all hooks cleared")
