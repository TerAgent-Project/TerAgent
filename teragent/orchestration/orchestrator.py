# teragent/orchestration/orchestrator.py
"""多 Agent 编排器

统一入口，根据 mode 委托给对应的 Pattern 实现。
参考 LangGraph 的 CompiledGraph, CrewAI 的 Crew, AutoGen 的 GroupChat。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, TYPE_CHECKING

from teragent.orchestration.cancellation import CancellationToken
from teragent.orchestration.shared_state import SharedState
from teragent.orchestration.run_context import RunContext, UsageTracker
from teragent.orchestration.patterns.base import OrchestrationResult
from teragent.orchestration.patterns.sequential import SequentialPattern
from teragent.orchestration.patterns.swarm import SwarmPattern

if TYPE_CHECKING:
    from teragent.orchestration.agent import Agent
    from teragent.config.orchestration_config import OrchestrationConfig
    from teragent.event_bus import EventBus
    from teragent.core.tap import TAPRequest

logger = logging.getLogger(__name__)

__all__ = [
    "Orchestrator",
    "OrchestrationMode",
    "OrchestrationConfig",
    "RuntimeOrchestratorConfig",
    "OrchestrationResult",
    "OrchestrationEvent",
    "OrchestrationEventType",
]


class OrchestrationMode(Enum):
    """编排模式枚举

    支持的编排模式:
      - SEQUENTIAL: 顺序编排（A→B→C）
      - SWARM: 去中心化 Swarm 编排（Agent 自主转交控制权）
      - PARALLEL: 并行扇出/扇入（Phase 2）
      - CONDITIONAL: 条件路由（Phase 2）
      - LOOP: 循环迭代（Phase 2）
    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"       # Phase 2
    CONDITIONAL = "conditional"  # Phase 2
    LOOP = "loop"               # Phase 2
    SWARM = "swarm"


class RuntimeOrchestratorConfig:
    """编排器运行时配置

    与 config/orchestration_config.py 中的 OrchestrationConfig（文件映射配置）不同，
    本类是编排器运行时使用的配置，包含 max_turns、timeout 等运行参数。

    Attributes:
        mode: 编排模式
        max_turns: 最大轮次
        max_handoffs: 最大 Handoff 次数（Swarm 模式）
        timeout: 超时时间（秒）
        verbose: 是否输出详细日志
    """

    def __init__(
        self,
        mode: OrchestrationMode = OrchestrationMode.SEQUENTIAL,
        max_turns: int = 20,
        max_handoffs: int = 10,
        timeout: float = 300.0,
        verbose: bool = False,
    ):
        self.mode = mode
        self.max_turns = max_turns
        self.max_handoffs = max_handoffs
        self.timeout = timeout
        self.verbose = verbose


# 向后兼容别名
# 注意：config/orchestration_config.py 中也有一个同名类，用于文件映射配置
# 此处保留别名避免已有代码中断，新代码应使用 RuntimeOrchestratorConfig
OrchestrationConfig = RuntimeOrchestratorConfig


class Orchestrator:
    """多 Agent 编排器

    统一入口，根据 mode 委托给对应的 Pattern 实现。
    参考 LangGraph 的 CompiledGraph, CrewAI 的 Crew, AutoGen 的 GroupChat。

    使用方式:
        orchestrator = Orchestrator(
            agents=[agent_a, agent_b, agent_c],
            mode=OrchestrationMode.SEQUENTIAL,
        )
        result = await orchestrator.run("分析这份报告")

    Attributes:
        agents: 参与编排的 Agent 列表
        mode: 编排模式
        shared_state: 跨 Agent 共享状态
    """

    def __init__(
        self,
        agents: list[Agent],
        mode: OrchestrationMode = OrchestrationMode.SEQUENTIAL,
        config: OrchestrationConfig | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._agents = agents
        self._mode = mode
        self._config = config
        self._event_bus = event_bus
        self._shared_state = SharedState()
        self._pattern = self._create_pattern(mode)

    @property
    def agents(self) -> list[Agent]:
        """参与编排的 Agent 列表"""
        return self._agents

    @property
    def mode(self) -> OrchestrationMode:
        """编排模式"""
        return self._mode

    @property
    def shared_state(self) -> SharedState:
        """跨 Agent 共享状态"""
        return self._shared_state

    async def run(
        self,
        task: str | TAPRequest,
        context: RunContext | None = None,
        max_turns: int | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> OrchestrationResult:
        """执行编排

        根据配置的 mode 委托给对应的 Pattern 实现，
        并处理取消、异常等边界情况。

        max_turns 优先级：参数 > config > 默认值(20)

        Args:
            task: 任务描述或 TAPRequest
            context: 运行上下文，None 则自动创建
            max_turns: 最大轮次，None 则使用 config 或默认值
            cancellation_token: 取消令牌

        Returns:
            OrchestrationResult 编排执行结果
        """
        # 确定 max_turns：参数 > config > 默认值
        effective_max_turns = max_turns
        if effective_max_turns is None and self._config:
            effective_max_turns = self._config.max_turns
        if effective_max_turns is None:
            effective_max_turns = 20

        if context is None:
            context = RunContext(
                shared_state=self._shared_state,
                usage=UsageTracker(),
                current_agent=self._agents[0].name if self._agents else "",
                turn=0,
                max_turns=effective_max_turns,
                cancellation_token=cancellation_token or CancellationToken(),
                event_bus=self._event_bus,
            )

        # 发射开始事件
        if self._event_bus:
            await self._event_bus.emit(
                "orchestration_started",
                mode=self._mode.value,
                agents=[a.name for a in self._agents],
            )

        try:
            # 获取超时配置
            timeout = None
            if self._config and hasattr(self._config, 'timeout'):
                timeout = self._config.timeout

            if timeout is not None and timeout > 0:
                result = await asyncio.wait_for(
                    self._pattern.run(
                        task=task,
                        agents=self._agents,
                        shared_state=self._shared_state,
                        context=context,
                        config=self._config,
                    ),
                    timeout=timeout,
                )
            else:
                result = await self._pattern.run(
                    task=task,
                    agents=self._agents,
                    shared_state=self._shared_state,
                    context=context,
                    config=self._config,
                )

            # 发射完成事件
            if self._event_bus:
                await self._event_bus.emit(
                    "orchestration_completed",
                    mode=self._mode.value,
                    total_turns=result.total_turns,
                )

            return result

        except asyncio.CancelledError:
            logger.info("Orchestration cancelled")
            return OrchestrationResult(
                final_output="",
                metadata={"cancelled": True},
            )
        except asyncio.TimeoutError:
            logger.warning(f"Orchestration timed out after {self._config.timeout if self._config else 'N/A'}s")
            return OrchestrationResult(
                final_output="",
                metadata={"timeout": True, "timeout_seconds": self._config.timeout if self._config else None},
            )
        except Exception as e:
            logger.error(f"Orchestration failed: {e}", exc_info=True)
            return OrchestrationResult(
                final_output="",
                metadata={"error": str(e), "error_type": type(e).__name__},
            )

    def _create_pattern(self, mode: OrchestrationMode) -> Any:
        """创建编排模式实例

        根据 mode 创建对应的 Pattern 实例。
        不支持的 mode 会抛出 ValueError。
        Phase 2+ 模式使用延迟导入，避免缺失依赖导致整个包无法加载。

        Args:
            mode: 编排模式

        Returns:
            OrchestrationPattern 实例

        Raises:
            ValueError: 不支持的编排模式
        """
        # Phase 1 模式：直接导入
        if mode == OrchestrationMode.SEQUENTIAL:
            return SequentialPattern()
        if mode == OrchestrationMode.SWARM:
            return SwarmPattern()

        # Phase 2+ 模式：延迟导入，避免依赖问题
        if mode == OrchestrationMode.PARALLEL:
            from teragent.orchestration.patterns.parallel import ParallelPattern
            return ParallelPattern()
        if mode == OrchestrationMode.CONDITIONAL:
            from teragent.orchestration.patterns.conditional import ConditionalPattern
            return ConditionalPattern()
        if mode == OrchestrationMode.LOOP:
            from teragent.orchestration.patterns.loop import LoopPattern
            return LoopPattern()

        raise ValueError(
            f"Orchestration mode '{mode.value}' not yet implemented. "
            f"Available modes: {[m.value for m in OrchestrationMode]}"
        )

    async def run_stream(
        self,
        task: str | TAPRequest,
        context: RunContext | None = None,
        max_turns: int | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncIterator[OrchestrationEvent]:
        """流式执行编排，逐步产出事件

        与 run() 一次性返回最终结果不同，run_stream() 在编排过程中
        逐步产出 OrchestrationEvent，允许调用方实时监控编排进度。

        事件类型:
          - STARTED: 编排开始
          - AGENT_STARTED: 某个 Agent 开始执行
          - AGENT_COMPLETED: 某个 Agent 执行完成
          - HANDOFF: Agent 转交控制权（Swarm 模式）
          - GUARDRAIL_TRIGGERED: 守卫被触发
          - STEP_COMPLETED: 一个编排步骤完成
          - COMPLETED: 编排完成
          - ERROR: 编排出错
          - CANCELLED: 编排被取消

        使用方式::

            orchestrator = Orchestrator(agents=[a, b, c], mode=OrchestrationMode.SEQUENTIAL)
            async for event in orchestrator.run_stream("分析报告"):
                if event.type == OrchestrationEventType.AGENT_COMPLETED:
                    print(f"Agent {event.data['agent_name']} completed")
                elif event.type == OrchestrationEventType.COMPLETED:
                    print(f"Final output: {event.data['final_output']}")

        Args:
            task: 任务描述或 TAPRequest
            context: 运行上下文，None 则自动创建
            max_turns: 最大轮次
            cancellation_token: 取消令牌

        Yields:
            OrchestrationEvent 编排事件
        """
        # 确定 max_turns
        effective_max_turns = max_turns
        if effective_max_turns is None and self._config:
            effective_max_turns = self._config.max_turns
        if effective_max_turns is None:
            effective_max_turns = 20

        if context is None:
            context = RunContext(
                shared_state=self._shared_state,
                usage=UsageTracker(),
                current_agent=self._agents[0].name if self._agents else "",
                turn=0,
                max_turns=effective_max_turns,
                cancellation_token=cancellation_token or CancellationToken(),
                event_bus=self._event_bus,
            )

        # 1. 产出 STARTED 事件
        yield OrchestrationEvent(
            type=OrchestrationEventType.STARTED,
            data={
                "mode": self._mode.value,
                "agents": [a.name for a in self._agents],
            },
        )

        # 2. 使用事件桥接：让 EventBus 的事件转为 yield
        # 创建桥接 EventBus，将编排步骤事件转发到队列
        event_queue: asyncio.Queue[OrchestrationEvent | None] = asyncio.Queue()

        original_event_bus = self._event_bus
        bridge_bus = _BridgeEventBus(event_queue)

        # 临时将 context 的 event_bus 设为桥接总线
        # 同时保留原始总线用于并行发射
        context.event_bus = bridge_bus

        # 3. 在后台启动编排，同时消费事件
        async def _run_orchestration() -> OrchestrationResult:
            """在后台执行编排"""
            try:
                timeout = None
                if self._config and hasattr(self._config, 'timeout'):
                    timeout = self._config.timeout

                if timeout is not None and timeout > 0:
                    result = await asyncio.wait_for(
                        self._pattern.run(
                            task=task,
                            agents=self._agents,
                            shared_state=self._shared_state,
                            context=context,
                            config=self._config,
                        ),
                        timeout=timeout,
                    )
                else:
                    result = await self._pattern.run(
                        task=task,
                        agents=self._agents,
                        shared_state=self._shared_state,
                        context=context,
                        config=self._config,
                    )
                return result
            except asyncio.CancelledError:
                return OrchestrationResult(final_output="", metadata={"cancelled": True})
            except asyncio.TimeoutError:
                timeout_val = self._config.timeout if self._config else None
                return OrchestrationResult(
                    final_output="",
                    metadata={"timeout": True, "timeout_seconds": timeout_val},
                )
            except Exception as e:
                logger.error(f"Stream orchestration failed: {e}", exc_info=True)
                return OrchestrationResult(
                    final_output="",
                    metadata={"error": str(e), "error_type": type(e).__name__},
                )

        orchestration_task = asyncio.create_task(_run_orchestration())

        # 4. 消费桥接事件，同时等待编排完成
        try:
            while True:
                # 尝试从队列获取事件，带超时以检查编排是否完成
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    # 检查编排是否完成
                    if orchestration_task.done():
                        break
                    continue

                if event is None:
                    # 哨兵值，表示结束
                    break

                yield event

                # 如果是终态事件，停止消费
                if event.type in (
                    OrchestrationEventType.COMPLETED,
                    OrchestrationEventType.ERROR,
                    OrchestrationEventType.CANCELLED,
                ):
                    break

            # 等待编排任务完成
            result = await orchestration_task

            # 如果还没有产出终态事件，产出 COMPLETED
            # （当编排完成但没有通过 event_bus 发射完成事件时）
            if not orchestration_task.cancelled():
                final_event_data = {
                    "final_output": result.final_output,
                    "last_agent": result.last_agent,
                    "total_turns": result.total_turns,
                    "total_prompt_tokens": result.total_prompt_tokens,
                    "total_completion_tokens": result.total_completion_tokens,
                }
                if result.metadata.get("cancelled"):
                    yield OrchestrationEvent(
                        type=OrchestrationEventType.CANCELLED,
                        data=final_event_data,
                    )
                elif result.metadata.get("error"):
                    yield OrchestrationEvent(
                        type=OrchestrationEventType.ERROR,
                        data={**final_event_data, "error": result.metadata.get("error", "")},
                    )
                elif result.metadata.get("timeout"):
                    yield OrchestrationEvent(
                        type=OrchestrationEventType.ERROR,
                        data={**final_event_data, "error": "Orchestration timed out"},
                    )
                else:
                    # 产出最终的 COMPLETED 事件（如果 pattern 未产出）
                    yield OrchestrationEvent(
                        type=OrchestrationEventType.COMPLETED,
                        data=final_event_data,
                    )

        finally:
            # 恢复原始 event_bus
            context.event_bus = original_event_bus
            if not orchestration_task.done():
                orchestration_task.cancel()
                try:
                    await orchestration_task
                except asyncio.CancelledError:
                    pass

    def as_tool(
        self,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> OrchestratorTool:
        """将编排器封装为工具（嵌套编排模式）

        允许将此编排器作为工具嵌入到另一个编排器的 Agent 中，
        实现嵌套编排。外部 Agent 调用此工具时，内部编排器
        执行完整的多 Agent 编排流程，结果作为 ToolResult 返回。

        使用方式::

            # 内部编排
            inner = Orchestrator(
                agents=[researcher, writer, editor],
                mode=OrchestrationMode.SEQUENTIAL,
            )

            # 将内部编排器封装为工具
            inner_tool = inner.as_tool()

            # 添加到协调员的工具集
            coordinator.tools.append(inner_tool)

            # 外部编排
            outer = Orchestrator(
                agents=[coordinator],
                mode=OrchestrationMode.SEQUENTIAL,
            )

        Args:
            tool_name: 自定义工具名称（默认: "use_orchestrator_{mode}"）
            tool_description: 自定义工具描述（默认: 根据编排器配置自动生成）

        Returns:
            OrchestratorTool 实例
        """
        from teragent.tools.orchestrator_tool import OrchestratorTool

        return OrchestratorTool(
            orchestrator=self,
            tool_name=tool_name,
            tool_description=tool_description,
        )

    def __repr__(self) -> str:
        agent_names = [a.name for a in self._agents]
        return f"Orchestrator(agents={agent_names}, mode={self._mode.value})"


# ======================================================================
# 流式编排事件类型
# ======================================================================

class OrchestrationEventType(Enum):
    """编排事件类型枚举

    用于 run_stream() 方法产出的事件类型标识。

    事件流顺序:
      STARTED → (AGENT_STARTED → [STEP_COMPLETED | HANDOFF | GUARDRAIL_TRIGGERED] → AGENT_COMPLETED)* → COMPLETED

    特殊事件:
      - ERROR: 编排出错
      - CANCELLED: 编排被取消
    """

    STARTED = "started"                      # 编排开始
    AGENT_STARTED = "agent_started"          # Agent 开始执行
    AGENT_COMPLETED = "agent_completed"      # Agent 执行完成
    STEP_COMPLETED = "step_completed"        # 编排步骤完成
    HANDOFF = "handoff"                      # Agent 转交控制权
    GUARDRAIL_TRIGGERED = "guardrail_triggered"  # 守卫被触发
    COMPLETED = "completed"                  # 编排完成
    ERROR = "error"                          # 编排出错
    CANCELLED = "cancelled"                  # 编排被取消


@dataclass
class OrchestrationEvent:
    """编排事件

    run_stream() 产出的每个事件都是此类型，包含事件类型和相关数据。

    Attributes:
        type: 事件类型
        data: 事件相关数据（如 agent_name、output 等）
        timestamp: 事件时间戳
    """

    type: OrchestrationEventType
    data: dict = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# ======================================================================
# 内部桥接 EventBus — 将 EventBus 事件转为 OrchestrationEvent
# ======================================================================

# EventBus 事件名到 OrchestrationEventType 的映射
_EVENT_NAME_MAP: dict[str, OrchestrationEventType] = {
    "orchestration_step_completed": OrchestrationEventType.STEP_COMPLETED,
    "orchestration_completed": OrchestrationEventType.COMPLETED,
    "orchestration_started": OrchestrationEventType.STARTED,
    "agent_started": OrchestrationEventType.AGENT_STARTED,
    "agent_completed": OrchestrationEventType.AGENT_COMPLETED,
    "handoff": OrchestrationEventType.HANDOFF,
    "guardrail_triggered": OrchestrationEventType.GUARDRAIL_TRIGGERED,
}


class _BridgeEventBus:
    """桥接 EventBus — 将 EventBus 事件转为 OrchestrationEvent 放入队列

    内部使用，由 run_stream() 创建。将 Pattern 通过 EventBus 发射的事件
    转换为 OrchestrationEvent 并放入 asyncio.Queue，供 run_stream() 的
    async for 循环消费。

    同时也转发事件到原始 EventBus（如果存在），保持兼容性。
    """

    def __init__(
        self,
        queue: asyncio.Queue[OrchestrationEvent | None],
        original_bus: EventBus | None = None,
    ) -> None:
        self._queue = queue
        self._original_bus = original_bus

    async def emit(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        """发射事件 — 同时放入队列和转发到原始总线"""
        # 映射事件名到 OrchestrationEventType
        event_type = _EVENT_NAME_MAP.get(event_name)
        if event_type is not None:
            event = OrchestrationEvent(
                type=event_type,
                data=kwargs if kwargs else ({"args": args} if args else {}),
            )
            await self._queue.put(event)

        # 转发到原始 EventBus（保持兼容性）
        if self._original_bus is not None:
            await self._original_bus.emit(event_name, *args, **kwargs)

    def on(self, event_name: str, handler: Any) -> None:
        """注册事件处理器 — 委托到原始总线"""
        if self._original_bus is not None:
            self._original_bus.on(event_name, handler)
