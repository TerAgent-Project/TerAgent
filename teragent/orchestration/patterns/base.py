# teragent/orchestration/patterns/base.py
"""编排模式基类与统一结果类型

所有编排模式实现 OrchestrationPattern 接口。
OrchestrationResult 是所有编排模式共用的返回类型。

参考 LangGraph 的 Graph 抽象、CrewAI 的 Process 抽象。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from teragent.orchestration.agent import Agent
    from teragent.orchestration.shared_state import SharedState
    from teragent.orchestration.run_context import RunContext
    from teragent.core.tap import TAPRequest

__all__ = [
    "OrchestrationPattern",
    "OrchestrationResult",
]


@dataclass
class OrchestrationResult:
    """编排执行结果

    所有编排模式的 run() 方法返回此类型，提供统一的输出接口。

    Attributes:
        final_output: 最终输出文本
        last_agent: 最后一个执行的 Agent 名称
        agent_outputs: 各 Agent 的输出（output_key → value）
        total_turns: 总执行轮次
        total_prompt_tokens: 总 prompt token 数
        total_completion_tokens: 总 completion token 数
        metadata: 额外元数据（如错误信息、取消标记等）
    """

    final_output: str = ""
    last_agent: str = ""
    agent_outputs: dict[str, Any] = field(default_factory=dict)
    total_turns: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    metadata: dict = field(default_factory=dict)


class OrchestrationPattern(ABC):
    """编排模式基类

    所有编排模式实现此接口。
    参考 LangGraph 的 Graph 抽象、CrewAI 的 Process 抽象。

    子类必须实现:
      - run(): 执行编排
      - get_next_agent(): 获取下一个要执行的 Agent
      - get_execution_plan(): 获取执行计划（用于可视化/调试）
    """

    @abstractmethod
    async def run(
        self,
        task: str | TAPRequest,
        agents: list[Agent],
        shared_state: SharedState,
        context: RunContext,
        **kwargs,
    ) -> OrchestrationResult:
        """执行编排

        Args:
            task: 任务描述或 TAPRequest
            agents: 参与编排的 Agent 列表
            shared_state: 跨 Agent 共享状态
            context: 运行时上下文
            **kwargs: 模式特定的额外参数

        Returns:
            OrchestrationResult 编排执行结果
        """
        ...

    @abstractmethod
    def get_next_agent(self, current: Agent | None, result: Any) -> Agent | None:
        """获取下一个要执行的 Agent

        Args:
            current: 当前 Agent（None 表示尚未开始）
            result: 当前 Agent 的执行结果

        Returns:
            下一个 Agent，None 表示无后续 Agent
        """
        ...

    @abstractmethod
    def get_execution_plan(self) -> list[dict]:
        """获取执行计划（用于可视化/调试）

        Returns:
            执行步骤列表，每个步骤是一个 dict 描述
        """
        ...
