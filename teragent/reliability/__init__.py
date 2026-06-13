"""teragent.reliability — Circuit breaker, budget tracking, and error recovery

Exports all public names from circuit_breaker, budget, and recovery modules.

Components:
    - CircuitBreakerManager: 4-type circuit breaker (budget, failure, latency, progress)
    - ModelCircuitBreakerManager: Per-model circuit breakers with cross-model degradation (P4-3)
    - StepBudget: Step-level budget tracking
    - CrossModelCostTracker: Multi-model cost tracking with monthly budget control
    - CostRecord: Individual cost record with cache savings
    - MonthlyBudgetConfig: Monthly budget configuration
    - RecoveryManager: Error recovery + model degradation strategies
    - DegradationChain: Cross-model degradation chain with health awareness (P4-3)
    - LongHorizonRecoveryManager: Long-horizon task fault recovery (P4-3)
    - RateLimitHandler: Unified rate limiting across models (P4-3)
"""

from teragent.reliability.budget import (
    DEFAULT_MAX_STEPS,
    CostRecord,
    CrossModelCostTracker,
    MonthlyBudgetConfig,
    StepBudget,
)
from teragent.reliability.circuit_breaker import (
    BreakerState,
    BudgetCheckResult,
    CircuitBreakerManager,
    ConsecutiveFailureBreaker,
    CostBudgetConfig,
    CostBudgetTracker,
    LatencyBreaker,
    # P4-3: Per-model circuit breaker
    ModelBreakerConfig,
    ModelBreakerState,
    ModelCircuitBreakerManager,
    ProgressDetector,
)
from teragent.reliability.recovery import (
    # P4-3: Fault recovery enhancement
    DegradationChain,
    LongHorizonRecoveryManager,
    RateLimitHandler,
    RateLimitInfo,
    RecoveryManager,
    RecoveryManagerConfig,
    RecoveryStats,
    RecoveryType,
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
    # circuit_breaker (P4-3)
    "ModelBreakerConfig",
    "ModelBreakerState",
    "ModelCircuitBreakerManager",
    # budget
    "DEFAULT_MAX_STEPS",
    "StepBudget",
    "CostRecord",
    "MonthlyBudgetConfig",
    "CrossModelCostTracker",
    # recovery
    "RecoveryType",
    "RecoveryStats",
    "RecoveryManagerConfig",
    "RecoveryManager",
    "is_context_overflow_error",
    "is_retryable_error",
    # recovery (P4-3)
    "DegradationChain",
    "LongHorizonRecoveryManager",
    "RateLimitInfo",
    "RateLimitHandler",
]
