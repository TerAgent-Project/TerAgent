# teragent/config/orchestration_config.py
"""编排配置 — 映射到 [orchestration] 段

支持顺序、并行、条件、循环、Swarm 五种编排模式的配置。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "OrchestrationConfig",
]


@dataclass(frozen=True)
class OrchestrationConfig:
    """编排配置 — 映射到 [orchestration] 段

    支持顺序、并行、条件、循环、Swarm 五种编排模式的配置。

    Attributes:
        mode: 编排模式（sequential, parallel, conditional, loop, swarm）
        sequence: 顺序编排的 Agent 名称列表
        fan_out: 并行扇出的 Agent 名称列表（Phase 2）
        fan_in: 并行扇入的 Agent 名称（Phase 2）
        router_agent: 条件路由的 Agent 名称（Phase 2）
        loop_agents: 循环迭代的 Agent 名称列表（Phase 2）
        max_iterations: 循环最大迭代次数（Phase 2）
        exit_condition: 循环退出条件表达式（Phase 2）
        coordinator: 协调者 Agent 名称
        checkpoint_enabled: 是否启用检查点
        human_in_the_loop: 是否启用人机协作
    """

    mode: str = "sequential"

    # Sequential
    sequence: list[str] = field(default_factory=list)

    # Parallel (Phase 2)
    fan_out: list[str] = field(default_factory=list)
    fan_in: str = ""

    # Conditional (Phase 2)
    router_agent: str = ""

    # Loop (Phase 2)
    loop_agents: list[str] = field(default_factory=list)
    max_iterations: int = 5
    exit_condition: str = ""

    # Common
    coordinator: str = ""
    checkpoint_enabled: bool = False
    human_in_the_loop: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> OrchestrationConfig:
        """从 dict 创建配置

        Args:
            data: 配置字典

        Returns:
            OrchestrationConfig 实例
        """
        return cls(
            mode=data.get("mode", "sequential"),
            sequence=data.get("sequence", []),
            fan_out=data.get("fan_out", []),
            fan_in=data.get("fan_in", ""),
            router_agent=data.get("router_agent", ""),
            loop_agents=data.get("loop_agents", []),
            max_iterations=data.get("max_iterations", 5),
            exit_condition=data.get("exit_condition", ""),
            coordinator=data.get("coordinator", ""),
            checkpoint_enabled=data.get("checkpoint_enabled", False),
            human_in_the_loop=data.get("human_in_the_loop", False),
        )
