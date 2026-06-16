"""teragent.orchestration.agent — Agent 基类

Agent 是多 Agent 编排的核心抽象。设计要点:
- Agent 拥有独立的 provider: ModelProvider（TAP IR 正交性核心）
- Agent 拥有独立的 tools: list[BaseTool]
- Agent 拥有 handoffs: list[Handoff] 用于 Swarm 编排
- Agent 拥有 output_key: str 用于 SharedState 写入
- Agent 拥有 system_prompt 支持静态字符串和动态生成函数
- Agent 拥有 hooks: AgentHooks 用于生命周期回调
- Agent 拥有 mcp_servers: list[MCPToolset] 用于 MCP 远程工具发现

参考:
- OpenAI Agents SDK: Agent 类
- AutoGen: ConversableAgent
- Google ADK: Agent / LlmAgent
- CrewAI: Agent
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from teragent.core.provider import ModelProvider
    from teragent.core.types import ToolSafety
    from teragent.tools.base import BaseTool
    from teragent.orchestration.handoff import Handoff
    from teragent.orchestration.agent_hooks import AgentHooks
    from teragent.orchestration.run_context import RunContext
    from teragent.orchestration.guardrail import Guardrail
    from teragent.tools.agent_tool import AgentTool
    from teragent.config.teragent_config import TerAgentConfig
    from teragent.tools.mcp_toolset import MCPToolset

logger = logging.getLogger(__name__)

__all__ = [
    "Agent",
]


@dataclass
class Agent:
    """Agent 基类 — 多Agent编排的核心抽象

    Agent 拥有独立的 provider（ModelProvider）、工具集（tools）、
    转交定义（handoffs）和系统提示（system_prompt），
    是 TAP IR 正交性设计中的核心差异化单元。

    Attributes:
        name: Agent 唯一标识名称
        description: Agent 功能描述（供 LLM 理解用途）
        instructions: 系统提示静态字符串（兼容旧接口）

        provider: 独立的 ModelProvider 实例（TAP IR 正交性核心）
        compiler_name: 延迟创建 provider 时使用的编译器名称
        adapter_name: 延迟创建 provider 时使用的适配器名称
        model: 延迟创建 provider 时使用的模型名称

        tools: Agent 拥有的工具列表
        allowed_tool_safety: 允许使用的工具安全级别列表
        max_steps: 最大执行步骤数

        handoffs: 转交定义列表，用于 Swarm 编排

        output_key: 输出写入 SharedState 的键名

        system_prompt: 系统提示，支持静态字符串或动态生成函数
            优先级：system_prompt（Callable）> system_prompt（str）> instructions
        hooks: Agent 生命周期钩子

        metadata: 额外元数据
    """

    name: str
    description: str = ""
    instructions: str = ""

    # TAP IR 独立组合 — 核心差异化
    provider: ModelProvider | None = None
    compiler_name: str = "default"
    adapter_name: str = "openai_compatible"
    model: str = ""

    # 工具集
    tools: list[BaseTool] = field(default_factory=list)
    allowed_tool_safety: list[Any] = field(default_factory=lambda: [
        "read_only", "safe_write"
    ])
    max_steps: int = 15

    # 编排 — Handoff 机制
    handoffs: list[Handoff] = field(default_factory=list)

    # 记忆与状态
    output_key: str = ""

    # 系统提示 — 支持静态和动态
    system_prompt: str | Callable[[RunContext, Agent], str] | None = None

    # 钩子
    hooks: AgentHooks | None = None

    # 守卫 — 输入/输出检查
    input_guardrails: list[Guardrail] = field(default_factory=list)
    output_guardrails: list[Guardrail] = field(default_factory=list)

    # MCP 远程工具集
    mcp_servers: list[Any] = field(default_factory=list)

    # 额外元数据
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """初始化后处理：解析 allowed_tool_safety 中的字符串为 ToolSafety 枚举"""
        from teragent.core.types import ToolSafety

        resolved = []
        for item in self.allowed_tool_safety:
            if isinstance(item, str):
                try:
                    resolved.append(ToolSafety(item))
                except ValueError:
                    logger.warning(
                        f"Agent '{self.name}': ignoring invalid ToolSafety value '{item}' "
                        f"in allowed_tool_safety. Valid values: {[s.value for s in ToolSafety]}"
                    )
            elif isinstance(item, ToolSafety):
                resolved.append(item)
        self.allowed_tool_safety = resolved

    def as_tool(
        self,
        tool_name: str | None = None,
        tool_description: str | None = None,
        output_extractor: Any | None = None,
    ) -> AgentTool:
        """将自身封装为工具（Agent-as-Tool 模式）

        当父 Agent 调用此工具时，子 Agent 的 TAP 链被执行，
        控制权随后返回给父 Agent。

        Args:
            tool_name: 自定义工具名称，默认为 use_{agent.name}
            tool_description: 自定义工具描述，默认为 Agent 的 description
            output_extractor: 可选的输出提取/转换函数

        Returns:
            AgentTool 实例
        """
        from teragent.tools.agent_tool import AgentTool
        return AgentTool(
            agent=self,
            tool_name=tool_name or f"use_{self.name}",
            tool_description=tool_description or self.description,
            output_extractor=output_extractor,
        )

    def get_handoff_tools(self) -> list[BaseTool]:
        """获取所有 handoff 转换成的工具

        每个 Handoff 会被转换为 HandoffTool（继承 BaseTool），
        LLM 通过调用 transfer_to_{agent_name} 工具触发转交。

        Returns:
            HandoffTool 实例列表
        """
        return [h.to_tool() for h in self.handoffs]

    def resolve_provider(self, config: TerAgentConfig | None = None) -> ModelProvider:
        """解析/创建 ModelProvider

        优先使用 self.provider，否则根据 compiler_name/adapter_name/model
        通过 TerAgent 配置系统延迟创建。

        Args:
            config: 可选的 TerAgentConfig 实例，用于查找预配置的 provider

        Returns:
            ModelProvider 实例

        Raises:
            ValueError: 既没有 provider 也没有 model 配置时抛出
        """
        if self.provider is not None:
            return self.provider

        if not self.model:
            raise ValueError(
                f"Agent '{self.name}' has no provider configured. "
                f"Either set agent.provider or set agent.model (with optional "
                f"compiler_name/adapter_name) to enable lazy provider creation."
            )

        # 延迟创建逻辑 — 使用 TerAgent 配置
        from teragent.config.loader import create_provider_from_config
        from teragent.config.driver_config import DriverConfig

        driver_cfg = DriverConfig(
            adapter=self.adapter_name,
            compiler=self.compiler_name,
            model=self.model,
        )
        return create_provider_from_config(driver_cfg)

    def get_system_prompt(self, ctx: RunContext | None = None) -> str:
        """获取系统提示 — 支持静态和动态

        优先级：system_prompt（Callable）> system_prompt（str）> instructions

        当 system_prompt 为可调用对象时，使用 RunContext 和 Agent 自身
        动态生成系统提示。当 RunContext 为 None 且 system_prompt 为可调用对象时，
        回退到 instructions。

        Args:
            ctx: 可选的运行上下文

        Returns:
            系统提示字符串
        """
        if self.system_prompt is not None:
            if callable(self.system_prompt):
                if ctx is not None:
                    return self.system_prompt(ctx, self)
                # 无法动态生成，回退到 instructions
                return self.instructions
            return self.system_prompt
        return self.instructions

    def get_mcp_tools(self) -> list[BaseTool]:
        """从所有 MCP 服务器获取已发现的工具

        遍历 mcp_servers 中的 MCPToolset 实例，收集其缓存的工具代理。
        仅返回已连接且已发现工具的 MCPToolset 的工具。

        注意: 此方法不会触发 MCP 连接。需要先手动调用
        MCPToolset.connect() 或使用 MCPToolset 作为异步上下文管理器。

        Returns:
            MCPTool 代理实例列表
        """
        mcp_tools: list[BaseTool] = []
        for server in self.mcp_servers:
            # 鸭子类型判断：具有 to_base_tools 和 is_connected 属性
            if hasattr(server, "to_base_tools") and hasattr(server, "is_connected"):
                if server.is_connected:
                    mcp_tools.extend(server.to_base_tools())
                else:
                    logger.debug(
                        f"Agent '{self.name}' 的 MCP 服务器 "
                        f"'{getattr(server, 'name', '<unnamed>')}' "
                        f"未连接，跳过工具发现"
                    )
            else:
                logger.warning(
                    f"Agent '{self.name}' 的 mcp_servers 包含"
                    f"非 MCPToolset 对象: {type(server).__name__}"
                )
        return mcp_tools

    def all_tools(self) -> list[BaseTool]:
        """获取所有可用工具（自身工具 + handoff 工具 + MCP 工具）

        Returns:
            合并后的工具列表
        """
        return self.tools + self.get_handoff_tools() + self.get_mcp_tools()

    def __repr__(self) -> str:
        return (
            f"Agent(name={self.name!r}, "
            f"tools={len(self.tools)}, "
            f"handoffs={len(self.handoffs)}, "
            f"mcp_servers={len(self.mcp_servers)}, "
            f"max_steps={self.max_steps}, "
            f"output_key={self.output_key!r}, "
            f"has_provider={self.provider is not None})"
        )
