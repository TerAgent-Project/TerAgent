# teragent/orchestration/run_context.py
"""运行时上下文与 Token 使用追踪

RunContext — 跨 Agent 共享的依赖注入容器
UsageTracker — Token 使用追踪

参考 OpenAI Agents SDK 的 RunContext, Google ADK 的 InvocationContext。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from teragent.orchestration.shared_state import SharedState
    from teragent.orchestration.cancellation import CancellationToken
    from teragent.event_bus import EventBus

__all__ = [
    "RunContext",
    "UsageTracker",
]


@dataclass
class UsageTracker:
    """Token 使用追踪

    跨 Agent 累积追踪 prompt/completion token 使用量。
    支持按 Agent 维度的使用统计。

    Attributes:
        total_prompt_tokens: 总 prompt token 数
        total_completion_tokens: 总 completion token 数
        by_agent: 按 Agent 维度的使用统计
    """

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    by_agent: dict[str, dict] = field(default_factory=dict)

    def record(self, agent_name: str, prompt_tokens: int, completion_tokens: int) -> None:
        """记录一次模型调用的 token 使用

        Args:
            agent_name: Agent 名称
            prompt_tokens: prompt token 数
            completion_tokens: completion token 数
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        if agent_name not in self.by_agent:
            self.by_agent[agent_name] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "calls": 0,
            }

        self.by_agent[agent_name]["prompt_tokens"] += prompt_tokens
        self.by_agent[agent_name]["completion_tokens"] += completion_tokens
        self.by_agent[agent_name]["calls"] += 1

    def get_summary(self) -> dict:
        """获取使用摘要

        Returns:
            包含总量和按 Agent 统计的摘要字典
        """
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "by_agent": dict(self.by_agent),
        }

    @property
    def total_tokens(self) -> int:
        """总 token 数"""
        return self.total_prompt_tokens + self.total_completion_tokens


@dataclass
class RunContext:
    """运行时上下文 — 跨 Agent 共享的依赖注入容器

    参考 OpenAI Agents SDK 的 RunContext, Google ADK 的 InvocationContext。

    每次编排运行创建一个 RunContext，在 Agent 间共享。

    Attributes:
        shared_state: 跨 Agent 共享状态
        usage: Token 使用追踪器
        current_agent: 当前执行的 Agent 名称
        turn: 当前轮次
        max_turns: 最大轮次
        cancellation_token: 取消令牌
        event_bus: 事件总线
        metadata: 额外元数据
    """

    shared_state: SharedState
    usage: UsageTracker
    current_agent: str
    turn: int
    max_turns: int
    cancellation_token: CancellationToken | None = None
    event_bus: EventBus | None = None
    metadata: dict = field(default_factory=dict)

    def with_agent(self, agent_name: str, turn: int | None = None) -> RunContext:
        """创建切换 Agent 的新上下文

        创建一个新的 RunContext，将 current_agent 切换为指定 Agent。
        共享的 shared_state、usage、event_bus 等保持引用。

        Args:
            agent_name: 新的 Agent 名称
            turn: 新的轮次，None 则保持当前轮次

        Returns:
            新的 RunContext 实例
        """
        return RunContext(
            shared_state=self.shared_state,
            usage=self.usage,
            current_agent=agent_name,
            turn=turn if turn is not None else self.turn,
            max_turns=self.max_turns,
            cancellation_token=self.cancellation_token,
            event_bus=self.event_bus,
            metadata=copy.deepcopy(self.metadata),
        )
