"""teragent.orchestration.handoff — Handoff 机制

Handoff 定义了 Agent 间控制权转交的规则，是 Swarm 编排模式的核心。

核心类:
1. Handoff — 转交定义，描述从源 Agent 到目标 Agent 的转交规则
2. HandoffTool — 将转交暴露为 BaseTool，LLM 通过调用 transfer_to_{agent_name} 触发转交
3. HandoffInputFilter — 控制从源 Agent 传递到目标 Agent 的历史消息

参考:
- OpenAI Agents SDK: Handoff 和 handoff_input_filter
- AutoGen: GroupChat 的 Speaker 选举机制
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

from teragent.tools.base import BaseTool, ToolResult
from teragent.core.types import ToolSafety

if TYPE_CHECKING:
    from teragent.orchestration.agent import Agent
    from teragent.orchestration.run_context import RunContext


@dataclass
class HandoffInputFilter:
    """Handoff 输入过滤器

    控制从源 Agent 传递到目标 Agent 的历史消息。
    参考 OpenAI Agents SDK 的 handoff_input_filter。

    过滤规则按顺序应用:
    1. keep_roles: 只保留特定角色的消息
    2. keep_recent: 只保留最近 N 条消息
    3. custom_filter: 自定义过滤函数

    Attributes:
        keep_recent: 保留最近 N 条消息（0 表示不限制）
        keep_roles: 只保留特定角色的消息（空列表表示不限制）
        custom_filter: 自定义过滤函数，接收消息列表，返回过滤后的列表
    """

    keep_recent: int = 0
    keep_roles: list[str] = field(default_factory=list)
    custom_filter: Callable[[list[dict]], list[dict]] | None = None

    def apply(self, messages: list[dict]) -> list[dict]:
        """应用过滤规则

        按 keep_roles → keep_recent → custom_filter 的顺序依次过滤。

        Args:
            messages: 原始消息列表

        Returns:
            过滤后的消息列表
        """
        result = messages

        # 1. 按角色过滤
        if self.keep_roles:
            result = [m for m in result if isinstance(m, dict) and m.get("role") in self.keep_roles]

        # 2. 保留最近 N 条
        if self.keep_recent > 0:
            result = result[-self.keep_recent:]

        # 3. 自定义过滤
        if self.custom_filter is not None:
            result = self.custom_filter(result)

        return result

    def __repr__(self) -> str:
        parts = []
        if self.keep_recent:
            parts.append(f"keep_recent={self.keep_recent}")
        if self.keep_roles:
            parts.append(f"keep_roles={self.keep_roles}")
        if self.custom_filter is not None:
            parts.append("custom_filter=<func>")
        return f"HandoffInputFilter({', '.join(parts) or 'no filters'})"


@dataclass
class Handoff:
    """Agent 转交定义

    定义从一个 Agent 转交控制权到另一个 Agent 的规则。
    Handoff 会被转换为 HandoffTool（继承 BaseTool），
    LLM 通过调用 transfer_to_{agent_name} 工具触发转交。

    Attributes:
        target_agent: 转交目标 Agent
        description: 转交描述（供 LLM 理解何时应转交）
        input_filter: 输入过滤器（控制传递的历史消息）
    """

    target_agent: Agent
    description: str = ""
    input_filter: HandoffInputFilter | None = None

    def __post_init__(self) -> None:
        """自动生成默认描述"""
        if not self.description:
            self.description = f"Transfer control to {self.target_agent.name}"

    def to_tool(self) -> HandoffTool:
        """将 Handoff 转换为工具

        Returns:
            HandoffTool 实例，可作为 BaseTool 使用
        """
        return HandoffTool(handoff=self)

    def __repr__(self) -> str:
        return (
            f"Handoff(target={self.target_agent.name!r}, "
            f"description={self.description!r})"
        )


class HandoffTool(BaseTool):
    """Handoff 工具实现

    LLM 调用 transfer_to_{agent_name} 时触发。
    返回特殊的 ToolResult，编排器识别 __handoff__ 标记后切换 Agent。

    设计要点:
    - 工具名格式为 transfer_to_{agent_name}，便于 LLM 理解
    - 参数包含可选的 reason 字段，供 LLM 说明转交原因
    - 返回的 ToolResult.data 中包含 __handoff__ 标记，编排器据此识别转交信号
    - 安全级别为 READ_ONLY（转交不产生副作用）
    - 并发安全（转交是纯信号操作）
    """

    _safety = ToolSafety.READ_ONLY
    _concurrency_safe = True

    def __init__(self, handoff: Handoff):
        """初始化 HandoffTool

        Args:
            handoff: Handoff 定义
        """
        self._handoff = handoff
        self.name = f"transfer_to_{handoff.target_agent.name}"
        self.description = handoff.description
        self.parameters_schema = {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": f"Reason for transferring to {handoff.target_agent.name}",
                },
            },
            "required": [],
        }

    async def execute(
        self,
        params: dict,
        progress_callback=None,
    ) -> ToolResult:
        """执行 Handoff

        返回包含 __handoff__ 标记的 ToolResult，
        编排器检测到后切换到目标 Agent。

        Args:
            params: 工具参数（包含可选的 reason）
            progress_callback: 进度回调（未使用）

        Returns:
            包含 __handoff__ 标记的 ToolResult
        """
        return ToolResult(
            success=True,
            data={
                "__handoff__": True,
                "target_agent": self._handoff.target_agent.name,
                "input_data": params,
            },
            metadata={"handoff": True},
        )

    def __repr__(self) -> str:
        return (
            f"HandoffTool(name={self.name!r}, "
            f"target={self._handoff.target_agent.name!r})"
        )
