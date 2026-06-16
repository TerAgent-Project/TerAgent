"""teragent.orchestration — 多Agent编排包

提供 Agent 抽象、Handoff 机制、编排器和编排模式。

核心类:
    Agent         — Agent 基类，拥有独立的 provider、工具集、handoffs
    Handoff       — Agent 转交定义
    HandoffInputFilter — 转交输入过滤器
    HandoffTool   — 转交工具（BaseTool 子类）
    Orchestrator  — 编排器主类
    OrchestrationConfig — 编排配置
    OrchestrationMode  — 编排模式枚举
    OrchestrationResult — 编排执行结果
    SharedState   — 跨 Agent 共享状态
    RunContext    — 运行上下文
    CancellationToken  — 取消令牌
    AgentHooks    — Agent 生命周期钩子
    Guardrail     — Agent 守卫（输入/输出检查）
    GuardrailResult — 守卫检查结果
    GuardrailTripwireTriggered — 守卫跳闸异常
"""
from __future__ import annotations

from teragent.orchestration.agent import Agent
from teragent.orchestration.agent_hooks import AgentHooks
from teragent.orchestration.handoff import Handoff, HandoffInputFilter, HandoffTool
from teragent.orchestration.cancellation import CancellationToken
from teragent.orchestration.guardrail import (
    Guardrail,
    GuardrailResult,
    GuardrailTripwireTriggered,
)
from teragent.orchestration.orchestrator import (
    OrchestrationConfig,
    OrchestrationMode,
    Orchestrator,
    RuntimeOrchestratorConfig,
    OrchestrationEvent,
    OrchestrationEventType,
)
from teragent.orchestration.patterns.base import OrchestrationResult
from teragent.orchestration.run_context import RunContext, UsageTracker
from teragent.orchestration.shared_state import ScopedState, SharedState, StateWrite
from teragent.orchestration.rwlock import AsyncRWLock, ReadLockContext, WriteLockContext
from teragent.orchestration.approval import ApprovalGate, ApprovalResult
from teragent.orchestration.checkpoint import OrchestrationCheckpoint
from teragent.tools.orchestrator_tool import OrchestratorTool

__all__ = [
    "Agent",
    "AgentHooks",
    "Handoff", "HandoffInputFilter", "HandoffTool",
    "Orchestrator", "OrchestrationConfig", "RuntimeOrchestratorConfig", "OrchestrationMode", "OrchestrationResult",
    "OrchestrationEvent", "OrchestrationEventType",
    "SharedState", "ScopedState", "StateWrite",
    "AsyncRWLock", "ReadLockContext", "WriteLockContext",
    "RunContext", "UsageTracker",
    "CancellationToken",
    "Guardrail", "GuardrailResult", "GuardrailTripwireTriggered",
    "ApprovalGate", "ApprovalResult",
    "OrchestrationCheckpoint",
    "OrchestratorTool",
]
