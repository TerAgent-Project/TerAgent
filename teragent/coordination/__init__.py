"""teragent.coordination — 多Agent协调模块（已废弃）

.. deprecated::
    ``teragent.coordination`` 已废弃，将在未来版本中移除。
    请使用 :mod:`teragent.orchestration` 替代。

迁移指南:
    - SubAgentManager → Orchestrator(mode=OrchestrationMode.SEQUENTIAL)
    - AgentMessageBus → EventBus + SharedState
    - AgentMode.SYNC/ASYNC/FORK → OrchestrationMode.SEQUENTIAL/PARALLEL/SWARM
    - SubAgentInfo → Agent

详细迁移说明请参考 docs/migration_guide.md
"""

from __future__ import annotations

import warnings

warnings.warn(
    "teragent.coordination is deprecated and will be removed in a future version. "
    "Use teragent.orchestration instead. "
    "See migration guide: SubAgentManager → Orchestrator, "
    "AgentMessageBus → EventBus + SharedState, "
    "AgentMode → OrchestrationMode.",
    DeprecationWarning,
    stacklevel=2,
)

# 保留空模块以避免导入错误，所有功能已迁移到 orchestration
__all__: list[str] = []
