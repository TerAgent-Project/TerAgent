"""teragent.pipeline — Library-level pipeline primitives

These are generic, reusable building blocks for AI code generation workflows.
They are decoupled from specific personas, EventBus orchestration, and Plan objects.

Library primitives:
    - extractor: File extraction from LLM responses (3-level degradation)
    - prompt_builder: Prompt construction with template externalization + token validation
    - subagent_worker: TAP request assembly → execution → file extraction → safe write
    - checklist: Deterministic code checks (AST + runnability) with TaskInfo dataclass
    - retry: Generic retry with exponential backoff + optional result validation
    - tracing: TAP trace recording + DPO preference pair generation (Phase 10)

Reference implementations (in examples/full_agent/):
    - DesignGenerator, PlanGenerator, Reviewer, ChecklistGenerator
    - These combine library primitives with specific personas and EventBus orchestration.
"""

# Phase 10: TAP tracing + DPO pair generation
from teragent.pipeline.tracing import (
    TAPTracer,
    TraceRecord,
    DPOPair,
    DataConstitution,
    TraceStats,
)
# Sub-agent worker
from teragent.pipeline.subagent_worker import SubAgentWorker

__all__ = [
    "TAPTracer",
    "TraceRecord",
    "DPOPair",
    "DataConstitution",
    "TraceStats",
    "SubAgentWorker",
]
