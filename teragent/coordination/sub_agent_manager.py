# teragent/coordination/sub_agent_manager.py
"""SubAgentManager -- 子 Agent 管理器

核心职责:
  - 创建和管理子 Agent 生命周期
  - 支持三种执行模式: SYNC (同步阻塞) / ASYNC (异步后台) / FORK (共享前缀)
  - 步数预算控制: 防止子 Agent 无限循环
  - 并发限制: 防止子 Agent 数量失控
  - 工具白名单: 限制子 Agent 可使用的工具
  - 结果传递: 通过 MessageBus 将结果返回给父 Agent

设计原则:
  - 不信任子 Agent 的自控能力 -- 由系统强制步数预算
  - 渐进开放 -- 子 Agent 只能访问白名单内的工具
  - 信号驱动 -- 通过 EventBus 通知生命周期事件
  - KV cache 优化由模型侧完成, FORK 模式仅做语义标记
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "AgentMode",
    "SubAgentInfo",
    "SubAgentManager",
    "SubAgentStatus",
]

from teragent.coordination.message_bus import AgentMessage, AgentMessageBus
from teragent.core.prompts import get_system_prompt_for_intent
from teragent.core.provider import ModelProvider
from teragent.event_bus import EventBus
from teragent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _detect_compiler_type(provider: ModelProvider) -> str:
    """Detect the compiler type from a ModelProvider instance.

    For teragent.core.provider.ModelProvider, uses the .compiler attribute
    to determine the compiler type string for prompt selection.

    Args:
        provider: A ModelProvider instance

    Returns:
        Compiler type string: "default" | "glm" | "anthropic" | "deepseek"
    """
    compiler = getattr(provider, "compiler", None)
    if compiler is not None:
        compiler_name = type(compiler).__name__
        name_map = {
            "DefaultCompiler": "default",
            "GLMCompiler": "glm",
            "AnthropicCompiler": "anthropic",
            "DeepSeekCompiler": "deepseek",
        }
        return name_map.get(compiler_name, "default")

    # Try model_name heuristics
    model_name = getattr(provider, "model_name", "") or getattr(provider, "model", "")
    model_lower = model_name.lower() if model_name else ""
    if "glm" in model_lower:
        return "glm"
    if "claude" in model_lower or "anthropic" in model_lower:
        return "anthropic"
    if "deepseek" in model_lower:
        return "deepseek"

    return "default"


# 子 Agent 系统提示前缀
# DEPRECATED: 此提示词已迁移至 teragent/core/prompts/sub_agent.py SUB_AGENT_PROMPT_*
# 新代码应通过 TAP 编译获取：TAPRequest(meta={"intent": "sub_agent"}, ...)
SUB_AGENT_SYSTEM_PROMPT_PREFIX = """\
你是 TerAgent 的子 Agent, 负责执行父 Agent 分配的特定任务。

【硬性约束】
1. 只能使用被允许的工具, 禁止调用未授权的工具
2. 严格按照任务描述执行, 不要偏离目标
3. 完成任务后立即返回结果, 不要进行额外操作
4. 如果无法完成任务, 明确说明失败原因
5. 步数有限, 不要重复执行相同的操作
"""


class AgentMode(Enum):
    """子 Agent 执行模式

    SYNC: 阻塞父 Agent, 直到子 Agent 完成
    ASYNC: 后台运行, 完成后通知父 Agent
    FORK: 共享系统提示前缀 (KV cache 优化)
    """

    SYNC = "sync"
    ASYNC = "async"
    FORK = "fork"


class SubAgentStatus(Enum):
    """子 Agent 状态"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass
class SubAgentInfo:
    """子 Agent 信息

    属性:
        agent_id: 子 Agent 唯一标识 (格式: "sub_agent_{counter}")
        parent_id: 父 Agent ID
        mode: 执行模式
        status: 当前状态
        task: 分配的任务描述
        allowed_tools: 允许使用的工具名称列表
        created_at: 创建时间戳
        completed_at: 完成时间戳 (None 表示未完成)
        result: 执行结果 (None 表示未完成)
        error: 错误信息 (None 表示无错误)
        steps_taken: 已执行的步骤数
    """

    agent_id: str
    parent_id: str
    mode: AgentMode
    status: SubAgentStatus
    task: str
    allowed_tools: list[str] = field(default_factory=list)
    created_at: float = 0.0
    completed_at: float | None = None
    result: str | None = None
    error: str | None = None
    steps_taken: int = 0

    def __post_init__(self) -> None:
        """自动填充创建时间"""
        if self.created_at == 0.0:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        """转换为可序列化字典"""
        return {
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "mode": self.mode.value,
            "status": self.status.value,
            "task": self.task,
            "allowed_tools": self.allowed_tools,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "steps_taken": self.steps_taken,
        }


class SubAgentManager:
    """子 Agent 管理器

    创建和管理子 Agent 的生命周期, 支持同步/异步/FORK 三种模式。

    使用方式::

        manager = SubAgentManager(event_bus, model, tool_registry, message_bus)

        # 同步模式: 阻塞等待结果
        result = await manager.spawn("分析代码质量", mode=AgentMode.SYNC)

        # 异步模式: 立即返回 agent_id, 完成后通过消息总线通知
        agent_id = await manager.spawn("后台重构", mode=AgentMode.ASYNC)

        # FORK 模式: 共享前缀优化
        result = await manager.spawn("快速查询", mode=AgentMode.FORK)
    """

    # 类常量: 最大子 Agent 步数
    MAX_SUB_AGENT_STEPS: int = 15

    # 类常量: 最大并发子 Agent 数
    MAX_CONCURRENT_SUB_AGENTS: int = 5

    def __init__(
        self,
        event_bus: EventBus,
        model_provider: ModelProvider,
        tool_registry: ToolRegistry,
        message_bus: AgentMessageBus,
    ) -> None:
        self._event_bus = event_bus
        self._model_provider = model_provider
        self._tool_registry = tool_registry
        self._message_bus = message_bus

        # Per-instance counter for generating unique agent IDs within this manager
        self._agent_counter: int = 0
        self._counter_lock = asyncio.Lock()

        # 活跃子 Agent 注册表
        self._active_agents: dict[str, SubAgentInfo] = {}

        # 异步任务的引用 (防止被 GC)
        self._async_tasks: dict[str, asyncio.Task] = {}

    async def spawn(
        self,
        task: str,
        mode: AgentMode = AgentMode.SYNC,
        allowed_tools: list[str] | None = None,
        parent_id: str = "main",
    ) -> str:
        """创建并启动子 Agent

        根据模式决定执行方式:
          - SYNC: 阻塞当前协程, 等待子 Agent 完成后返回结果
          - ASYNC: 创建后台任务, 立即返回 agent_id
          - FORK: 类似 SYNC, 但标记共享前缀 (KV cache 优化)

        Args:
            task: 任务描述
            mode: 执行模式
            allowed_tools: 允许使用的工具名称列表 (None 表示使用所有工具)
            parent_id: 父 Agent ID

        Returns:
            SYNC/FORK: 子 Agent 的执行结果文本
            ASYNC: 子 Agent ID (格式: "sub_agent_{counter}")

        Raises:
            RuntimeError: 超过最大并发子 Agent 数
        """
        # 递增计数器, 生成 agent_id
        async with self._counter_lock:
            self._agent_counter += 1
            agent_id = f"sub_agent_{self._agent_counter}"

        # 确定允许的工具列表
        if allowed_tools is None:
            allowed_tools = self._tool_registry.list_tool_names()

        # 创建子 Agent 信息
        agent_info = SubAgentInfo(
            agent_id=agent_id,
            parent_id=parent_id,
            mode=mode,
            status=SubAgentStatus.PENDING,
            task=task,
            allowed_tools=allowed_tools,
        )
        self._active_agents[agent_id] = agent_info

        # 注册到消息总线
        self._message_bus.register_agent(
            agent_id,
            metadata={
                "parent_id": parent_id,
                "mode": mode.value,
                "task": task,
            },
        )

        # 构建系统提示
        system_prompt = self._build_system_prompt(allowed_tools)

        # 获取允许的工具定义
        tools = self._get_tool_definitions(allowed_tools)

        # 先标记为运行中，再检查并发限制（原子操作：无 await 间隔）
        agent_info.status = SubAgentStatus.RUNNING

        # 检查并发限制 — 在状态设置后立即检查，避免竞态条件
        running_count = sum(
            1 for info in self._active_agents.values()
            if info.status == SubAgentStatus.RUNNING
        )
        if running_count > self.MAX_CONCURRENT_SUB_AGENTS:
            # 超限回滚
            agent_info.status = SubAgentStatus.PENDING
            del self._active_agents[agent_id]
            self._message_bus.unregister_agent(agent_id)
            raise RuntimeError(
                f"并发子 Agent 数已达上限 ({self.MAX_CONCURRENT_SUB_AGENTS}), "
                f"无法创建新的子 Agent"
            )

        # 发射子 Agent 创建事件
        await self._event_bus.emit(
            "sub_agent_spawned",
            agent_info.to_dict(),
        )

        logger.info(
            f"子 Agent 已创建: id={agent_id}, mode={mode.value}, "
            f"parent={parent_id}, task={task[:50]}"
        )

        # 根据模式执行
        if mode == AgentMode.SYNC:
            result = await self._run_sync(
                agent_id, system_prompt, task, tools
            )
            return result
        elif mode == AgentMode.ASYNC:
            async_task = asyncio.create_task(
                self._run_async(
                    agent_id, system_prompt, task, tools, parent_id
                )
            )
            self._async_tasks[agent_id] = async_task
            return agent_id
        elif mode == AgentMode.FORK:
            result = await self._run_fork(
                agent_id, system_prompt, task, tools, parent_id
            )
            return result
        else:
            # 不应到达这里
            agent_info.status = SubAgentStatus.FAILED
            agent_info.error = f"未知的执行模式: {mode}"
            self._cleanup_agent(agent_id)
            return f"[ERROR] 未知的执行模式: {mode}"

    async def _run_sync(
        self,
        agent_id: str,
        system_prompt: str,
        task: str,
        tools: list[dict],
    ) -> str:
        """同步执行子 Agent

        简单的工具循环: 调用模型 -> 处理工具调用 -> 循环,
        直到模型不再调用工具或步数预算耗尽。

        Args:
            agent_id: 子 Agent ID
            system_prompt: 系统提示
            task: 任务描述
            tools: 允许的工具定义列表

        Returns:
            子 Agent 的最终文本输出
        """
        agent_info = self._active_agents.get(agent_id)
        if agent_info is None:
            return "[ERROR] 子 Agent 信息未找到"

        # 构建对话消息
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        try:
            while agent_info.steps_taken < self.MAX_SUB_AGENT_STEPS:
                # 调用模型
                try:
                    response = await self._model_provider.chat(
                        messages=messages,
                        tools=tools if tools else None,
                    )
                except Exception as e:
                    logger.error(
                        f"子 Agent '{agent_id}' 模型调用失败: {e}"
                    )
                    agent_info.status = SubAgentStatus.FAILED
                    agent_info.error = str(e)
                    agent_info.completed_at = time.time()
                    self._cleanup_agent(agent_id)
                    return f"[ERROR] 模型调用失败: {e}"

                content = response.get("content", "")
                tool_calls = response.get("tool_calls", [])

                # 没有工具调用: 子 Agent 完成
                if not tool_calls:
                    agent_info.status = SubAgentStatus.COMPLETED
                    agent_info.result = content
                    agent_info.completed_at = time.time()
                    self._cleanup_agent(agent_id)
                    logger.info(
                        f"子 Agent '{agent_id}' 完成: "
                        f"steps={agent_info.steps_taken}"
                    )
                    return content

                # 有工具调用: 处理每个工具调用
                agent_info.steps_taken += 1

                # 将 assistant 消息加入对话历史
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                })

                # 执行每个工具调用
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    tool_call_id = tc.get("id", "")

                    # 解析工具参数
                    func_args = func.get("arguments", {})
                    if isinstance(func_args, str):
                        import json
                        try:
                            func_args = json.loads(func_args)
                        except json.JSONDecodeError:
                            func_args = {}

                    # 查找工具
                    tool = self._tool_registry.get(tool_name)
                    if tool is None:
                        tool_result_str = f"错误: 工具 '{tool_name}' 未注册"
                    elif tool_name not in agent_info.allowed_tools:
                        tool_result_str = (
                            f"错误: 工具 '{tool_name}' 未被授权使用, "
                            f"允许的工具: {', '.join(agent_info.allowed_tools)}"
                        )
                    else:
                        # 执行工具
                        try:
                            tool_result = await tool.execute(func_args)
                            if tool_result.success:
                                tool_result_str = str(tool_result.data) if tool_result.data else ""
                            else:
                                tool_result_str = f"工具执行失败: {tool_result.error}"
                        except Exception as e:
                            tool_result_str = f"工具执行异常: {e}"

                    # 将工具结果加入对话历史
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_result_str,
                    })

            # 步数预算耗尽
            agent_info.status = SubAgentStatus.BUDGET_EXHAUSTED
            agent_info.error = "子 Agent 步数预算耗尽"
            agent_info.completed_at = time.time()
            self._cleanup_agent(agent_id)
            logger.warning(
                f"子 Agent '{agent_id}' 步数预算耗尽: "
                f"steps={agent_info.steps_taken}/{self.MAX_SUB_AGENT_STEPS}"
            )
            return "子 Agent 步数预算耗尽"

        except Exception as e:
            logger.error(
                f"子 Agent '{agent_id}' 执行异常: {e}", exc_info=True
            )
            agent_info.status = SubAgentStatus.FAILED
            agent_info.error = str(e)
            agent_info.completed_at = time.time()
            self._cleanup_agent(agent_id)
            return f"[ERROR] 子 Agent 执行异常: {e}"

    async def _run_async(
        self,
        agent_id: str,
        system_prompt: str,
        task: str,
        tools: list[dict],
        parent_id: str,
    ) -> None:
        """异步执行子 Agent

        执行完成后通过 MessageBus 将结果发送给父 Agent,
        并发射 sub_agent_completed 事件。

        Args:
            agent_id: 子 Agent ID
            system_prompt: 系统提示
            task: 任务描述
            tools: 允许的工具定义列表
            parent_id: 父 Agent ID
        """
        try:
            result = await self._run_sync(
                agent_id, system_prompt, task, tools
            )

            # 通过消息总线发送结果给父 Agent
            agent_info = self._active_agents.get(agent_id)
            result_msg = AgentMessage(
                from_agent=agent_id,
                to_agent=parent_id,
                message_type="result",
                content=result,
                metadata={
                    "status": agent_info.status.value if agent_info else "unknown",
                    "steps_taken": agent_info.steps_taken if agent_info else 0,
                },
            )
            await self._message_bus.send(result_msg)

            # 发射完成事件
            await self._event_bus.emit(
                "sub_agent_completed",
                agent_info.to_dict() if agent_info else {"agent_id": agent_id},
            )

            logger.info(
                f"子 Agent '{agent_id}' 异步执行完成, "
                f"结果已发送给父 Agent '{parent_id}'"
            )

        except Exception as e:
            logger.error(
                f"子 Agent '{agent_id}' 异步执行异常: {e}", exc_info=True
            )
            # 尝试通知父 Agent
            try:
                error_msg = AgentMessage(
                    from_agent=agent_id,
                    to_agent=parent_id,
                    message_type="result",
                    content=f"[ERROR] 子 Agent 异步执行异常: {e}",
                    metadata={"status": "failed", "error": str(e)},
                )
                await self._message_bus.send(error_msg)
            except Exception:
                logger.error(
                    f"子 Agent '{agent_id}' 无法通知父 Agent: "
                    f"消息发送失败"
                )
        finally:
            # 清理异步任务引用
            self._async_tasks.pop(agent_id, None)

    async def _run_fork(
        self,
        agent_id: str,
        system_prompt: str,
        task: str,
        tools: list[dict],
        parent_id: str,
    ) -> str:
        """FORK 模式执行子 Agent

        类似 SYNC 模式, 但共享系统提示前缀以优化 KV cache。
        KV cache 优化由模型侧完成, 此处仅做语义标记。

        Args:
            agent_id: 子 Agent ID
            system_prompt: 系统提示
            task: 任务描述
            tools: 允许的工具定义列表
            parent_id: 父 Agent ID

        Returns:
            子 Agent 的最终文本输出
        """
        logger.info(
            f"子 Agent '{agent_id}' 使用 FORK 模式, "
            f"共享系统提示前缀 (KV cache 优化)"
        )
        # FORK 模式: 委托给 _run_sync
        # KV cache 优化是模型侧行为, 不需要代码层面特殊处理
        return await self._run_sync(agent_id, system_prompt, task, tools)

    def get_status(self, agent_id: str) -> dict | None:
        """获取子 Agent 状态

        Args:
            agent_id: 子 Agent ID

        Returns:
            SubAgentInfo.to_dict() 或 None (Agent 不存在)
        """
        info = self._active_agents.get(agent_id)
        if info is None:
            return None
        return info.to_dict()

    def list_active_agents(self) -> list[dict]:
        """列出所有活跃子 Agent 的状态

        Returns:
            SubAgentInfo.to_dict() 列表
        """
        return [
            info.to_dict() for info in self._active_agents.values()
        ]

    async def stop(self, agent_id: str) -> None:
        """停止指定的子 Agent

        将子 Agent 标记为 STOPPED, 并从消息总线注销。
        如果有异步任务正在运行, 取消该任务。

        Args:
            agent_id: 子 Agent ID
        """
        info = self._active_agents.get(agent_id)
        if info is None:
            logger.warning(f"子 Agent '{agent_id}' 未找到, 无法停止")
            return

        info.status = SubAgentStatus.STOPPED
        info.completed_at = time.time()
        self._cleanup_agent(agent_id)

        # 取消异步任务 (如果存在)
        async_task = self._async_tasks.pop(agent_id, None)
        if async_task and not async_task.done():
            async_task.cancel()
            logger.info(f"已取消子 Agent '{agent_id}' 的异步任务")

        logger.info(f"子 Agent '{agent_id}' 已停止")

    async def stop_all(self) -> None:
        """停止所有活跃子 Agent"""
        agent_ids = list(self._active_agents.keys())
        for agent_id in agent_ids:
            await self.stop(agent_id)
        logger.info(f"已停止所有子 Agent, 共 {len(agent_ids)} 个")

    def get_status_report(self) -> dict:
        """返回子 Agent 管理器状态摘要 (供调试和 TUI 使用)

        Returns:
            {
                "total_agents": int,
                "by_status": {"running": int, "completed": int, ...},
                "agents": [SubAgentInfo.to_dict(), ...],
            }
        """
        by_status: dict[str, int] = {}
        for info in self._active_agents.values():
            status_key = info.status.value
            by_status[status_key] = by_status.get(status_key, 0) + 1

        return {
            "total_agents": len(self._active_agents),
            "by_status": by_status,
            "agents": self.list_active_agents(),
        }

    # ===== 内部方法 =====

    def _build_system_prompt(self, allowed_tools: list[str]) -> str:
        """构建子 Agent 系统提示

        Use intent-based prompt from teragent/core/prompts/
        Falls back to old constant (deprecated) if prompt selector
        returns empty string.

        Args:
            allowed_tools: 允许的工具名称列表

        Returns:
            完整的系统提示文本
        """
        # Use intent-based prompt from teragent/core/prompts/
        # Detect compiler type from model provider for prompt selection
        compiler_type = _detect_compiler_type(self._model_provider)
        base_prompt = get_system_prompt_for_intent("sub_agent", compiler_type=compiler_type)
        if not base_prompt:
            # Fallback to old constant (deprecated)
            base_prompt = SUB_AGENT_SYSTEM_PROMPT_PREFIX

        parts = [base_prompt]

        if allowed_tools:
            parts.append(f"\n允许使用的工具: {', '.join(allowed_tools)}")

        parts.append(f"\n步数预算: {self.MAX_SUB_AGENT_STEPS} 步")

        return "\n".join(parts)

    def _get_tool_definitions(self, allowed_tools: list[str]) -> list[dict]:
        """获取允许的工具定义列表

        从 ToolRegistry 中提取允许工具的 function calling 定义。

        Args:
            allowed_tools: 允许的工具名称列表

        Returns:
            符合 OpenAI tools 格式的字典列表
        """
        definitions: list[dict] = []
        for name in allowed_tools:
            tool = self._tool_registry.get(name)
            if tool is not None:
                definitions.append(tool.to_function_definition())
            else:
                logger.warning(
                    f"工具 '{name}' 在注册表中未找到, 跳过"
                )
        return definitions

    def _cleanup_agent(self, agent_id: str) -> None:
        """清理子 Agent 资源

        从消息总线注销。活跃 Agent 信息保留在 _active_agents 中
        供状态查询, 不立即删除。

        Args:
            agent_id: 子 Agent ID
        """
        self._message_bus.unregister_agent(agent_id)

        # 清理过多的已完成 agent（保留最近 100 个）
        completed_agents = [
            (aid, info) for aid, info in self._active_agents.items()
            if info.status in (SubAgentStatus.COMPLETED, SubAgentStatus.FAILED, SubAgentStatus.STOPPED, SubAgentStatus.BUDGET_EXHAUSTED)
        ]
        if len(completed_agents) > 100:
            completed_agents.sort(key=lambda x: x[1].created_at)
            for aid, _ in completed_agents[:-100]:
                del self._active_agents[aid]
