# teragent/config/agent_config.py
"""Agent 配置 — 映射到 [agents.{name}] 段

Phase 2 增强:
  - driver: 驱动器名称（替代 compiler + adapter 组合）
  - mcp_servers: MCP 服务器列表
  - input_guardrails / output_guardrails: 输入/输出护栏
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "AgentConfig",
]


@dataclass(frozen=True)
class AgentConfig:
    """Agent 配置 — 映射到 [agents.{name}] 段

    描述一个 Agent 的身份、能力、模型配置等。

    Attributes:
        name: Agent 唯一名称
        driver: 驱动器名称（Phase 2，替代 compiler + adapter 组合）
        role: Agent 角色（如 researcher, coder, reviewer）
        description: Agent 描述（供 LLM 理解用途）
        tools: 工具名称列表
        output_key: 输出到 SharedState 的键名
        max_steps: 工具调用循环最大步数
        handoffs: Handoff 目标 Agent 名称列表
        mcp_servers: MCP 服务器名称列表（Phase 2）
        input_guardrails: 输入护栏名称列表（Phase 2）
        output_guardrails: 输出护栏名称列表（Phase 2）
    """

    name: str = ""
    driver: str = ""
    role: str = ""
    description: str = ""
    tools: list[str] = field(default_factory=list)
    output_key: str = ""
    max_steps: int = 15
    handoffs: list[str] = field(default_factory=list)
    mcp_servers: list[str] = field(default_factory=list)
    input_guardrails: list[str] = field(default_factory=list)
    output_guardrails: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, name: str, data: dict) -> AgentConfig:
        """从 dict 创建配置

        Phase 2 签名变更: 接受 name 作为第一个参数，
        而非从 data 中提取。这允许更灵活的配置加载。

        Args:
            name: Agent 名称（作为 [agents.{name}] 段的键名）
            data: 配置字典

        Returns:
            AgentConfig 实例
        """
        return cls(
            name=name,
            driver=data.get("driver", ""),
            role=data.get("role", ""),
            description=data.get("description", ""),
            tools=data.get("tools", []),
            output_key=data.get("output_key", ""),
            max_steps=data.get("max_steps", 15),
            handoffs=data.get("handoffs", []),
            mcp_servers=data.get("mcp_servers", []),
            input_guardrails=data.get("input_guardrails", []),
            output_guardrails=data.get("output_guardrails", []),
        )
