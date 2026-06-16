# teragent/orchestration/patterns/conditional.py
"""条件路由编排模式

一种受限的 Swarm 模式，只有第一个 Agent（路由器）拥有 Handoff 权利。
路由器 Agent 决定将控制权转交给哪个 Agent，被选中的 Agent 执行完毕后
编排即结束。

执行流程:
  1. 路由器 Agent（agents[0]）执行，拥有到所有其他 Agent 的 Handoff 工具
  2. 路由器通过 Handoff 选择目标 Agent
  3. 目标 Agent 执行，如果也有 handoffs，继续跟随（类似 Swarm）
  4. 当没有更多 Handoff 时，编排结束

参考: OpenAI Agents SDK 的 Triage Agent, LangGraph 的 ConditionalGraph

与 Swarm 模式的区别:
  - 只有路由器 Agent 自动获得到其他 Agent 的 Handoff 工具
  - 非 router Agent 通常没有 Handoff（除非显式配置）
  - 编排结构更可控：单入口 → 路由决策 → 目标执行
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from teragent.orchestration.patterns.base import OrchestrationPattern, OrchestrationResult
from teragent.orchestration.handoff import Handoff
from teragent.orchestration.patterns.swarm import SwarmPattern
from teragent.core.tap import TAPRequest
from teragent.tools.base import ToolResult

if TYPE_CHECKING:
    from teragent.orchestration.agent import Agent
    from teragent.orchestration.shared_state import SharedState
    from teragent.orchestration.run_context import RunContext

logger = logging.getLogger(__name__)

__all__ = [
    "ConditionalPattern",
]


class ConditionalPattern(OrchestrationPattern):
    """条件路由编排模式

    一种受限的 Swarm 模式，只有路由器 Agent 拥有 Handoff 权利。
    路由器决定将控制权转交给哪个 Agent，被选中的 Agent 执行后编排结束。

    参考: OpenAI Agents SDK 的 Triage Agent, LangGraph 的 ConditionalGraph

    执行流程:
    1. 路由器 Agent（agents[0]）执行，拥有到所有其他 Agent 的 Handoff 工具
    2. 路由器通过 Handoff 选择目标 Agent
    3. 目标 Agent 执行（如果也有 handoffs，继续跟随）
    4. 当没有更多 Handoff 时，编排结束
    """

    def __init__(self) -> None:
        self._swarm = SwarmPattern()

    async def run(
        self,
        task: str | TAPRequest,
        agents: list[Agent],
        shared_state: SharedState,
        context: RunContext,
        **kwargs,
    ) -> OrchestrationResult:
        """执行条件路由编排

        只有第一个 Agent（路由器）自动获得到其他 Agent 的 Handoff。
        路由器决定转交给哪个 Agent，目标 Agent 执行完毕后编排结束。

        Args:
            task: 任务描述或 TAPRequest
            agents: 参与编排的 Agent 列表，agents[0] 为路由器
            shared_state: 跨 Agent 共享状态
            context: 运行时上下文

        Returns:
            OrchestrationResult 包含最终输出和各 Agent 结果
        """
        if not agents:
            return OrchestrationResult(final_output="", total_turns=0)

        # 构建增强的 Agent 列表：为路由器自动添加 Handoff
        enhanced_agents = self._build_conditional_agents(agents)

        # 发射条件路由开始事件
        if context.event_bus:
            await context.event_bus.emit(
                "conditional_routing_started",
                router_agent=enhanced_agents[0].name,
                target_agents=[a.name for a in enhanced_agents[1:]],
            )

        # 使用 Swarm 模式执行（路由器已有 Handoff，可转交给目标 Agent）
        result = await self._swarm.run(
            task=task,
            agents=enhanced_agents,
            shared_state=shared_state,
            context=context,
            **kwargs,
        )

        # 添加条件路由特有的元数据
        result.metadata["conditional_routing"] = True
        result.metadata["router_agent"] = agents[0].name

        # 发射条件路由完成事件
        if context.event_bus:
            await context.event_bus.emit(
                "conditional_routing_completed",
                router_agent=agents[0].name,
                final_agent=result.last_agent,
            )

        return result

    def get_next_agent(self, current: Agent | None, result: Any) -> Agent | None:
        """条件路由模式通过路由器 Handoff 决定下一个 Agent

        此方法始终返回 None，路由决策由路由器 Agent 的
        Handoff 工具调用动态决定。
        """
        return None

    def get_execution_plan(self) -> list[dict]:
        """获取执行计划"""
        return [
            {"type": "conditional", "description": "Router-driven conditional handoff pattern"},
        ]

    # ===== 内部方法 =====

    def _build_conditional_agents(self, agents: list[Agent]) -> list[Agent]:
        """构建条件路由的 Agent 列表

        为路由器 Agent（agents[0]）自动添加到所有其他 Agent 的 Handoff。
        如果路由器已有某些 Agent 的 Handoff，则不会重复添加。

        非 router Agent 的 Handoff 保持不变（允许级联路由）。

        **重要**: 此方法会创建路由器 Agent 的浅拷贝，避免修改原始 Agent 对象。

        Args:
            agents: 原始 Agent 列表

        Returns:
            增强后的 Agent 列表（路由器增加了 Handoff，其余 Agent 不变）
        """
        if len(agents) <= 1:
            return list(agents)

        router = agents[0]
        targets = agents[1:]

        # 收集路由器已有的 Handoff 目标
        existing_handoff_targets = {h.target_agent.name for h in router.handoffs}

        # 为路由器添加到所有目标 Agent 的 Handoff
        new_handoffs = list(router.handoffs)
        for target in targets:
            if target.name not in existing_handoff_targets:
                new_handoffs.append(Handoff(
                    target_agent=target,
                    description=target.description or f"Route to {target.name}",
                ))

        # 创建增强的路由器 Agent（浅拷贝，避免修改原始对象）
        from dataclasses import replace as _dc_replace
        enhanced_router = _dc_replace(router, handoffs=new_handoffs)

        logger.info(
            f"Conditional router '{enhanced_router.name}' configured with handoffs to: "
            f"{[h.target_agent.name for h in new_handoffs]}"
        )

        return [enhanced_router] + list(agents[1:])
