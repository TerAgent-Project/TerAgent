"""teragent.router — Intelligent model routing for multi-model collaboration

Exports all public names from the router module.

Components:
    - ModelRouter: Intelligent model router based on intent, multimodal,
      context length, long-horizon, cost, and degradation
    - RoutingDecision: Record of a routing decision with reason tracking
    - RoutingTable: Configurable routing table with override rules
    - PipelineProfile: Named pipeline configuration for dynamic allocation
    - PipelineManager: Runtime pipeline switching and per-stage model assignment
"""

from teragent.router.model_router import (
    ModelRouter,
    PipelineManager,
    PipelineProfile,
    RoutingDecision,
    RoutingReason,
    RoutingTable,
)

__all__ = [
    "RoutingDecision",
    "RoutingReason",
    "RoutingTable",
    "ModelRouter",
    "PipelineProfile",
    "PipelineManager",
]
