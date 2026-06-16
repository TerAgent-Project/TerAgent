"""teragent.tools.orchestrator_tool — Orchestrator-as-Tool implementation

Wraps an Orchestrator as a BaseTool, enabling nested orchestration.
When a parent agent calls an OrchestratorTool, the inner orchestrator's
run() method is executed, and control returns to the parent agent.

This enables hierarchical composition of multi-agent workflows:
an outer orchestrator can delegate complex sub-tasks to an inner
orchestrator that manages its own team of agents.

Design reference:
    - OpenAI Agents SDK: Agent-as-Tool pattern (extended to orchestrators)
    - Google ADK: Agent.as_tool() (hierarchical delegation)
    - plan.md: "编排器本身可以作为 Agent 的一部分，实现嵌套编排"

Usage::

    from teragent.orchestration import Orchestrator, OrchestrationMode
    from teragent.tools.orchestrator_tool import OrchestratorTool

    # Create an inner orchestrator
    inner = Orchestrator(
        agents=[researcher, writer, editor],
        mode=OrchestrationMode.SEQUENTIAL,
    )

    # Wrap as a tool for a coordinator agent
    inner_tool = inner.as_tool()

    # Or create directly
    inner_tool = OrchestratorTool(orchestrator=inner)

    # Add to coordinator's tools
    coordinator.tools.append(inner_tool)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from teragent.core.types import ToolSafety
from teragent.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from teragent.orchestration.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

__all__ = [
    "OrchestratorTool",
]


class OrchestratorTool(BaseTool):
    """将编排器封装为工具（嵌套编排模式）

    允许将一个编排器作为工具嵌入到另一个编排器的 Agent 中，
    实现嵌套编排。调用时运行内部编排器的 run() 方法，
    并将编排结果作为 ToolResult 返回给调用者。

    与 AgentTool 的区别:
      - AgentTool: 包装单个 Agent，调用时执行该 Agent 的 TAP 编译链路
      - OrchestratorTool: 包装整个编排器，调用时执行完整的多 Agent 编排流程

    使用场景:
      - 外层编排器协调多个子团队
      - 每个子团队由内部编排器管理（如: 研究团队、写作团队、审核团队）
      - 外层 Agent 通过调用 OrchestratorTool 来委派子任务给子团队

    Example::

        # 内部编排: 研究→写作→审核
        inner = Orchestrator(
            agents=[researcher, writer, editor],
            mode=OrchestrationMode.SEQUENTIAL,
        )

        # 将内部编排器封装为工具
        inner_tool = inner.as_tool(
            tool_name="research_and_write",
            tool_description="Research a topic, write an article, and edit it",
        )

        # 外部编排: 协调员使用嵌套编排工具
        coordinator = Agent(
            name="coordinator",
            tools=[inner_tool],
        )
        outer = Orchestrator(
            agents=[coordinator],
            mode=OrchestrationMode.SEQUENTIAL,
        )
    """

    _safety = ToolSafety.SAFE_WRITE
    _concurrency_safe = False

    def __init__(
        self,
        orchestrator: Orchestrator,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> None:
        """Initialize OrchestratorTool with an Orchestrator instance.

        Args:
            orchestrator: The Orchestrator to wrap as a tool
            tool_name: Override tool name (default: "use_orchestrator_{mode}")
            tool_description: Override tool description (default: auto-generated
                from orchestrator's agents and mode)
        """
        self._orchestrator = orchestrator
        self._mode_name = orchestrator.mode.value

        self.name = tool_name or f"use_orchestrator_{self._mode_name}"
        self.description = tool_description or self._default_description()
        self.parameters_schema = self._default_schema()

    def _default_description(self) -> str:
        """Generate a default description based on the orchestrator's configuration.

        Returns:
            A human-readable description of the inner orchestration
        """
        agent_names = [a.name for a in self._orchestrator.agents]
        agents_str = ", ".join(agent_names)
        return (
            f"Nested orchestration ({self._mode_name} mode) "
            f"with agents: [{agents_str}]. "
            f"Delegates a task to the inner orchestrator for multi-agent processing."
        )

    def _default_schema(self) -> dict:
        """Generate the default parameters schema.

        The schema includes:
          - task (required): The task to delegate to the inner orchestrator
          - context (optional): Additional context for the inner orchestrator

        Returns:
            JSON Schema dict for the tool parameters
        """
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        f"Task to delegate to the inner {self._mode_name} orchestrator "
                        f"(agents: {[a.name for a in self._orchestrator.agents]})"
                    ),
                },
                "context": {
                    "type": "string",
                    "description": "Additional context for the inner orchestrator",
                },
            },
            "required": ["task"],
        }

    async def execute(
        self,
        params: dict,
        progress_callback: Any | None = None,
    ) -> ToolResult:
        """执行嵌套编排

        调用内部编排器的 run() 方法，将参数中的 task 传入，
        并将编排结果封装为 ToolResult 返回。

        支持从外部事件循环调用（自动检测是否在异步上下文中）。

        Args:
            params: Tool parameters, must include 'task'
            progress_callback: Optional progress callback (not forwarded,
                as the inner orchestrator has its own event bus / hooks)

        Returns:
            ToolResult with the inner orchestration's output
        """
        task = params.get("task", "")
        context_str = params.get("context", "")

        # 构建完整任务指令
        instruction = task
        if context_str:
            instruction = f"{task}\n\nContext: {context_str}"

        logger.info(
            "OrchestratorTool '%s': starting nested orchestration "
            "(mode=%s, agents=%s)",
            self.name,
            self._mode_name,
            [a.name for a in self._orchestrator.agents],
        )

        try:
            # 执行内部编排器
            result = await self._orchestrator.run(task=instruction)

            # 检查编排结果中是否有错误/取消
            is_cancelled = result.metadata.get("cancelled", False)
            is_timeout = result.metadata.get("timeout", False)
            has_error = "error" in result.metadata

            if is_cancelled:
                logger.warning(
                    "OrchestratorTool '%s': nested orchestration was cancelled",
                    self.name,
                )
                return ToolResult(
                    success=False,
                    error="Nested orchestration was cancelled",
                    metadata={
                        "orchestrator_mode": self._mode_name,
                        "cancelled": True,
                        "total_turns": result.total_turns,
                    },
                    safety=self._safety,
                )

            if is_timeout:
                logger.warning(
                    "OrchestratorTool '%s': nested orchestration timed out",
                    self.name,
                )
                return ToolResult(
                    success=False,
                    error="Nested orchestration timed out",
                    metadata={
                        "orchestrator_mode": self._mode_name,
                        "timeout": True,
                        "timeout_seconds": result.metadata.get("timeout_seconds"),
                        "total_turns": result.total_turns,
                    },
                    safety=self._safety,
                )

            if has_error:
                error_msg = result.metadata.get("error", "Unknown error")
                logger.error(
                    "OrchestratorTool '%s': nested orchestration failed: %s",
                    self.name,
                    error_msg,
                )
                return ToolResult(
                    success=False,
                    error=f"Nested orchestration failed: {error_msg}",
                    metadata={
                        "orchestrator_mode": self._mode_name,
                        "error_type": result.metadata.get("error_type", "Unknown"),
                        "total_turns": result.total_turns,
                    },
                    safety=self._safety,
                )

            # 成功完成
            logger.info(
                "OrchestratorTool '%s': nested orchestration completed "
                "(turns=%d, last_agent=%s)",
                self.name,
                result.total_turns,
                result.last_agent,
            )

            return ToolResult(
                success=True,
                data={
                    "output": result.final_output,
                    "last_agent": result.last_agent,
                    "agent_outputs": result.agent_outputs,
                },
                metadata={
                    "orchestrator_mode": self._mode_name,
                    "total_turns": result.total_turns,
                    "total_prompt_tokens": result.total_prompt_tokens,
                    "total_completion_tokens": result.total_completion_tokens,
                },
                safety=self._safety,
            )

        except Exception as e:
            logger.error(
                "OrchestratorTool '%s' execution failed: %s",
                self.name,
                e,
                exc_info=True,
            )
            return ToolResult(
                success=False,
                error=f"Nested orchestration execution failed: {e}",
                safety=self._safety,
            )

    def validate_input(self, params: dict) -> list[str]:
        """Validate that required 'task' parameter is present and non-empty.

        Args:
            params: Tool parameters to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = super().validate_input(params)
        # Additional validation: task must be a non-empty string
        task = params.get("task", "")
        if isinstance(task, str) and not task.strip():
            errors.append("task must be a non-empty string")
        return errors

    def describe_usage(self, params: dict) -> str:
        """动态描述当前工具调用（供 TUI 展示）

        Args:
            params: Tool parameters

        Returns:
            Human-readable description of this nested orchestration invocation
        """
        task = params.get("task", "")
        task_preview = task[:50] + "..." if len(task) > 50 else task
        return f"Nested orchestration ({self._mode_name}): {task_preview}"

    def __repr__(self) -> str:
        agent_names = [a.name for a in self._orchestrator.agents]
        return (
            f"OrchestratorTool(name={self.name!r}, "
            f"mode={self._mode_name!r}, "
            f"agents={agent_names!r})"
        )
