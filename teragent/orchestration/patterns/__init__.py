"""teragent.orchestration.patterns — 编排模式包

提供多种编排模式的实现：
- Sequential: 顺序执行（A→B→C）
- Swarm: Agent 自主转交控制权
- Parallel: 并行扇出/扇入
- Conditional: 条件路由
- Loop: 循环迭代
"""
from __future__ import annotations

from teragent.orchestration.patterns.base import OrchestrationPattern, OrchestrationResult
from teragent.orchestration.patterns.sequential import SequentialPattern
from teragent.orchestration.patterns.swarm import SwarmPattern
from teragent.orchestration.patterns.parallel import ParallelPattern
from teragent.orchestration.patterns.conditional import ConditionalPattern
from teragent.orchestration.patterns.loop import LoopPattern

__all__ = [
    "OrchestrationPattern",
    "OrchestrationResult",
    "SequentialPattern",
    "SwarmPattern",
    "ParallelPattern",
    "ConditionalPattern",
    "LoopPattern",
]
