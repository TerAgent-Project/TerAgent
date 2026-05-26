"""teragent.reliability — Circuit breaker, budget tracking, and error recovery

Exports all public names from circuit_breaker, budget, and recovery modules.

Components:
    - CircuitBreakerManager: 4-type circuit breaker (budget, failure, latency, progress)
    - StepBudget: Step-level budget tracking
    - RecoveryManager: Error recovery + model degradation strategies
"""

from teragent.reliability.circuit_breaker import (
    BudgetCheckResult,
    BreakerState,
    CostBudgetConfig,
    CostBudgetTracker,
    ConsecutiveFailureBreaker,
    LatencyBreaker,
    ProgressDetector,
    CircuitBreakerManager,
)

from teragent.reliability.budget import (
    DEFAULT_MAX_STEPS,
    StepBudget,
)

from teragent.reliability.recovery import (
    RecoveryType,
    RecoveryStats,
    RecoveryManagerConfig,
    RecoveryManager,
    is_context_overflow_error,
    is_retryable_error,
)

__all__ = [
    # circuit_breaker
    "BudgetCheckResult",
    "BreakerState",
    "CostBudgetConfig",
    "CostBudgetTracker",
    "ConsecutiveFailureBreaker",
    "LatencyBreaker",
    "ProgressDetector",
    "CircuitBreakerManager",
    # budget
    "DEFAULT_MAX_STEPS",
    "StepBudget",
    # recovery
    "RecoveryType",
    "RecoveryStats",
    "RecoveryManagerConfig",
    "RecoveryManager",
    "is_context_overflow_error",
    "is_retryable_error",
]
