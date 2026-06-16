"""teragent.orchestration.agent_hooks — Agent 生命周期钩子

AgentHooks 提供 Agent 执行过程中的观察和干预点。
参考 OpenAI Agents SDK 的 AgentHooksBase，采用 observe-or-override 模式：
- 返回 None 继续（observe）
- 返回值覆盖默认行为（override）

所有方法都有默认空实现，子类只需覆盖感兴趣的钩子。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from teragent.orchestration.run_context import RunContext
    from teragent.orchestration.agent import Agent
    from teragent.tools.base import BaseTool, ToolResult
    from teragent.core.tap import TAPResponse


class AgentHooks:
    """Agent 生命周期钩子

    参考 OpenAI Agents SDK 的 AgentHooksBase。
    observe-or-override 模式：返回 None 继续，返回值覆盖。
    所有方法都有默认空实现，子类只需覆盖感兴趣的钩子。

    用法:
        class MyHooks(AgentHooks):
            async def on_start(self, ctx, agent):
                print(f"Agent {agent.name} started")

            async def on_tool_end(self, ctx, agent, tool, result):
                if not result.success:
                    print(f"Tool {tool.name} failed: {result.error}")

    钩子执行时机:
        on_start  → Agent 开始执行
        on_model_start → 模型调用开始（发送请求前）
        on_model_end → 模型调用结束（收到响应后）
        on_tool_start → 工具开始执行
        on_tool_end → 工具执行结束
        on_handoff → Agent 收到 Handoff
        on_end → Agent 执行结束
    """

    async def on_start(self, ctx: RunContext, agent: Agent) -> None:
        """Agent 开始执行

        在 Agent 开始执行前调用。可用于初始化状态、记录日志等。

        Args:
            ctx: 运行上下文
            agent: 即将执行的 Agent
        """
        pass

    async def on_end(self, ctx: RunContext, agent: Agent, output: str) -> None:
        """Agent 执行结束

        在 Agent 执行完成后调用。可用于清理资源、记录结果等。

        Args:
            ctx: 运行上下文
            agent: 执行完毕的 Agent
            output: Agent 的最终输出
        """
        pass

    async def on_handoff(self, ctx: RunContext, agent: Agent, source: Agent) -> None:
        """Agent 收到从 source 的 Handoff

        当控制权从 source Agent 转交到当前 agent 时调用。

        Args:
            ctx: 运行上下文
            agent: 接收控制权的 Agent（当前 Agent）
            source: 发起 Handoff 的源 Agent
        """
        pass

    async def on_tool_start(self, ctx: RunContext, agent: Agent, tool: BaseTool) -> None:
        """工具开始执行

        在工具执行前调用。可用于权限检查、参数修改等。

        Args:
            ctx: 运行上下文
            agent: 调用工具的 Agent
            tool: 即将执行的工具
        """
        pass

    async def on_tool_end(
        self, ctx: RunContext, agent: Agent, tool: BaseTool, result: ToolResult
    ) -> None:
        """工具执行结束

        在工具执行完成后调用。可用于结果记录、错误处理等。

        Args:
            ctx: 运行上下文
            agent: 调用工具的 Agent
            tool: 执行完毕的工具
            result: 工具执行结果
        """
        pass

    async def on_model_start(
        self, ctx: RunContext, agent: Agent, system_prompt: str
    ) -> None:
        """模型调用开始

        在发送模型请求前调用。可用于修改系统提示、记录日志等。

        Args:
            ctx: 运行上下文
            agent: 调用模型的 Agent
            system_prompt: 即将发送的系统提示
        """
        pass

    async def on_model_end(
        self, ctx: RunContext, agent: Agent, response: TAPResponse
    ) -> None:
        """模型调用结束

        在收到模型响应后调用。可用于记录响应、成本追踪等。

        Args:
            ctx: 运行上下文
            agent: 调用模型的 Agent
            response: 模型响应（TAPResponse）
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
