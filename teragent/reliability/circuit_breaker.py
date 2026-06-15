# teragent/reliability/circuit_breaker.py
"""Circuit Breaker & Cost Budget Tracking System

Part of the teragent library — centralized circuit breaker and cost budget tracking.

Design philosophy (ADVISORY-FIRST, RELAXED):
  - Generous defaults: 10M tokens/session, no hard cost cap by default
  - Warnings and suggestions rather than hard blocks
  - Easy override via config
  - Clear observability of cost/budget via /cost command

Components:
  1. CostBudgetTracker   — Session-level token/cost budget tracker (advisory)
  2. ConsecutiveFailureBreaker — Circuit breaker for consecutive model/API failures
  3. LatencyBreaker      — Circuit breaker for high latency model calls (warn only)
  4. ProgressDetector    — Detect if AgentLoop is stuck (no meaningful progress)
  5. CircuitBreakerManager — Facade that manages all breakers and emits unified events

Events emitted by CircuitBreakerManager:
  - budget_warning     : approaching budget limit (advisory)
  - budget_critical    : critical budget level (advisory, suggest pause)
  - budget_exhausted   : budget fully consumed (only if hard_limit enabled)
  - circuit_open       : too many consecutive failures, pausing
  - latency_warning    : consistently slow model calls
  - progress_stalled   : AgentLoop appears stuck
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BreakerState",
    "BudgetCheckResult",
    "CircuitBreakerManager",
    "ConsecutiveFailureBreaker",
    "CostBudgetConfig",
    "CostBudgetTracker",
    "FourModelCircuitBreakerManager",
    "FOUR_MODEL_BREAKER_CONFIGS",
    "LatencyBreaker",
    "ModelBreakerConfig",
    "ModelBreakerState",
    "ModelCircuitBreakerConfig",
    "ModelCircuitBreakerManager",
    "ProgressDetector",
]

# Typed circuit breaker config support
from teragent.config.circuit_breaker_config import CircuitBreakerConfig as TypedCircuitBreakerConfig
from teragent.event_bus import EventBus

logger = logging.getLogger(__name__)


# ===== Data Classes =====

@dataclass
class BudgetCheckResult:
    """Result of a budget check operation.

    Attributes:
        level: Budget status level — "ok" | "warning" | "critical" | "exhausted"
        message: Human-readable description of the budget state
        utilization: Fraction of budget used (0.0 - 1.0)
        prompt_tokens_used: Total prompt tokens consumed in this session
        completion_tokens_used: Total completion tokens consumed in this session
        total_tokens_used: Total tokens consumed (prompt + completion)
        max_tokens: Maximum token budget for the session
        estimated_cost: Estimated cost in USD (0.0 if pricing not configured)
    """

    level: str  # "ok" | "warning" | "critical" | "exhausted"
    message: str
    utilization: float
    prompt_tokens_used: int
    completion_tokens_used: int
    total_tokens_used: int
    max_tokens: int
    estimated_cost: float


@dataclass
class BreakerState:
    """State of a circuit breaker.

    Attributes:
        name: Breaker state name — "closed" | "open" | "half_open"
        consecutive_failures: Current streak of consecutive failures
        total_failures: Total failures recorded
        total_successes: Total successes recorded
        last_error: Most recent error message
        last_failure_time: Timestamp of the most recent failure
        can_retry: True if enough cooldown time has passed (for half-open state)
    """

    name: str  # "closed" | "open" | "half_open"
    consecutive_failures: int
    total_failures: int
    total_successes: int
    last_error: str
    last_failure_time: float
    can_retry: bool


@dataclass
class CostBudgetConfig:
    """Cost budget configuration — generous defaults, advisory-first.

    All thresholds are configurable. The defaults are deliberately generous
    so they won't interfere with normal usage.

    Attributes:
        max_session_tokens: Maximum tokens per session (10M = very generous)
        warning_threshold: Fraction at which to emit a warning (0.7 = 70%)
        critical_threshold: Fraction at which to suggest a pause (0.9 = 90%)
        cost_per_million_input: USD per 1M input tokens (0 = cost tracking disabled)
        cost_per_million_output: USD per 1M output tokens (0 = cost tracking disabled)
        enable_hard_limit: If True, block calls at 100% budget (default: OFF)
        auto_downgrade_model: If set, switch to this model at critical threshold
    """

    max_session_tokens: int = 10_000_000  # 10M tokens per session (very generous)
    warning_threshold: float = 0.7  # Warn at 70% of budget
    critical_threshold: float = 0.9  # Critical at 90%, suggest pause
    cost_per_million_input: float = 0.0  # $ per 1M input tokens (0 = disabled)
    cost_per_million_output: float = 0.0  # $ per 1M output tokens (0 = disabled)
    enable_hard_limit: bool = False  # Hard stop at 100% (default: OFF)
    auto_downgrade_model: str = ""  # If set, switch to this model at critical threshold


# ===== Helper =====

def _safe_emit(bus: EventBus | None, event_name: str, **kwargs: Any) -> None:
    """Safely emit an event on the bus (fire-and-forget).

    If no bus is provided, or if no event loop is running, the emission
    is silently skipped. This follows the project's "signal-driven,
    fire-and-forget, never block the main loop" design principle.
    """
    if bus is None:
        return
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(bus.emit(event_name, **kwargs))
        task.add_done_callback(lambda t: logger.error(f"Event emit failed: {t.exception()}") if not t.cancelled() and t.exception() else None)
    except RuntimeError:
        # No running event loop — skip emission
        logger.debug(f"No running event loop; skipped event '{event_name}'")


# ===== 1. CostBudgetTracker =====

class CostBudgetTracker:
    """Session-level cost and token budget tracker.

    Design principles:
      - Advisory-first: warn, don't block (unless enable_hard_limit=True)
      - Generous defaults that won't interfere with normal usage
      - Clear observability via /cost command
      - Per-stage breakdown (design, plan, execute, review, agentloop, intent)

    Usage::

        tracker = CostBudgetTracker()
        result = tracker.record_usage(prompt_tokens=500, completion_tokens=200, stage="plan")
        if result.level == "warning":
            print(result.message)
    """

    def __init__(
        self, config: CostBudgetConfig | None = None, bus: EventBus | None = None
    ) -> None:
        self._config = config or CostBudgetConfig()
        self._bus = bus

        # Token counters
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0

        # Per-stage tracking
        self._stage_usage: dict[str, dict[str, int]] = {}

        # Track whether we've already emitted warning/critical events
        # to avoid spamming the same event on every call
        self._warning_emitted: bool = False
        self._critical_emitted: bool = False

    # ----- Core methods -----

    def record_usage(
        self, prompt_tokens: int, completion_tokens: int, stage: str = "unknown"
    ) -> BudgetCheckResult:
        """Record token usage for a model call and check the budget.

        Args:
            prompt_tokens: Number of prompt/input tokens in this call
            completion_tokens: Number of completion/output tokens in this call
            stage: Pipeline stage that generated this usage
                (e.g., "design", "plan", "execute", "review", "agentloop", "intent")

        Returns:
            BudgetCheckResult indicating the current budget state after this call
        """
        self._prompt_tokens += prompt_tokens
        self._completion_tokens += completion_tokens

        # Per-stage accounting
        if stage not in self._stage_usage:
            self._stage_usage[stage] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "calls": 0,
            }
        self._stage_usage[stage]["prompt_tokens"] += prompt_tokens
        self._stage_usage[stage]["completion_tokens"] += completion_tokens
        self._stage_usage[stage]["calls"] += 1

        result = self.check_budget()

        # Emit events on threshold crossings (only once per threshold)
        if result.level == "warning" and not self._warning_emitted:
            self._warning_emitted = True
            logger.warning(f"Budget warning: {result.message}")
            _safe_emit(
                self._bus,
                "budget_warning",
                utilization=result.utilization,
                total_tokens=result.total_tokens_used,
                max_tokens=result.max_tokens,
                message=result.message,
            )
        elif result.level == "critical" and not self._critical_emitted:
            self._critical_emitted = True
            logger.warning(f"Budget critical: {result.message}")
            _safe_emit(
                self._bus,
                "budget_critical",
                utilization=result.utilization,
                total_tokens=result.total_tokens_used,
                max_tokens=result.max_tokens,
                message=result.message,
                auto_downgrade_model=self._config.auto_downgrade_model,
            )
        elif result.level == "exhausted":
            logger.error(f"Budget exhausted: {result.message}")
            _safe_emit(
                self._bus,
                "budget_exhausted",
                utilization=result.utilization,
                total_tokens=result.total_tokens_used,
                max_tokens=result.max_tokens,
                message=result.message,
            )

        return result

    def check_budget(self) -> BudgetCheckResult:
        """Check the current budget state without recording any usage.

        Returns:
            BudgetCheckResult with current budget status
        """
        total = self._prompt_tokens + self._completion_tokens
        util = total / self._config.max_session_tokens if self._config.max_session_tokens > 0 else 0.0
        util = min(util, 1.0)  # Cap at 1.0

        cost = self._estimate_cost(self._prompt_tokens, self._completion_tokens)

        if self._config.enable_hard_limit and util >= 1.0:
            level = "exhausted"
            message = (
                f"预算耗尽: {total:,} / {self._config.max_session_tokens:,} Token ({util:.1%})。硬限制已启用，后续调用将被阻止。"
            )
        elif util >= self._config.critical_threshold:
            level = "critical"
            suggestion = ""
            if self._config.auto_downgrade_model:
                suggestion = f" 建议切换到 {self._config.auto_downgrade_model}。"
            message = (
                f"预算严重不足: {total:,} / {self._config.max_session_tokens:,} Token ({util:.1%})。建议暂停或减少使用。{suggestion}"
            )
        elif util >= self._config.warning_threshold:
            level = "warning"
            message = (
                f"预算警告: {total:,} / {self._config.max_session_tokens:,} Token ({util:.1%})。您正在接近会话预算。"
            )
        else:
            level = "ok"
            message = (
                f"预算正常: {total:,} / {self._config.max_session_tokens:,} Token ({util:.1%})."
            )

        return BudgetCheckResult(
            level=level,
            message=message,
            utilization=util,
            prompt_tokens_used=self._prompt_tokens,
            completion_tokens_used=self._completion_tokens,
            total_tokens_used=total,
            max_tokens=self._config.max_session_tokens,
            estimated_cost=cost,
        )

    def get_session_summary(self) -> dict:
        """Get a full session summary for the /cost command.

        Returns:
            Dictionary with total usage, cost, stage breakdown, and budget state
        """
        result = self.check_budget()
        return {
            "total_tokens": result.total_tokens_used,
            "prompt_tokens": result.prompt_tokens_used,
            "completion_tokens": result.completion_tokens_used,
            "max_tokens": result.max_tokens,
            "utilization": f"{result.utilization:.1%}",
            "estimated_cost": f"${result.estimated_cost:.4f}",
            "level": result.level,
            "message": result.message,
            "stages": self.get_stage_breakdown(),
            "config": {
                "max_session_tokens": self._config.max_session_tokens,
                "warning_threshold": f"{self._config.warning_threshold:.0%}",
                "critical_threshold": f"{self._config.critical_threshold:.0%}",
                "enable_hard_limit": self._config.enable_hard_limit,
                "cost_per_million_input": self._config.cost_per_million_input,
                "cost_per_million_output": self._config.cost_per_million_output,
                "auto_downgrade_model": self._config.auto_downgrade_model or "(none)",
            },
        }

    def get_stage_breakdown(self) -> dict[str, dict]:
        """Get per-stage cost/token breakdown.

        Returns:
            Dictionary mapping stage names to their usage stats
        """
        breakdown: dict[str, dict] = {}
        for stage, usage in self._stage_usage.items():
            total_stage = usage["prompt_tokens"] + usage["completion_tokens"]
            stage_cost = self._estimate_cost(
                usage["prompt_tokens"], usage["completion_tokens"]
            )
            breakdown[stage] = {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": total_stage,
                "calls": usage["calls"],
                "estimated_cost": f"${stage_cost:.4f}",
            }
        return breakdown

    def estimate_call_cost(self, prompt_tokens: int, model: str = "") -> float:
        """Estimate the cost of a model call before making it.

        This uses the configured pricing. If pricing is not configured (default),
        returns 0.0.

        Args:
            prompt_tokens: Estimated prompt tokens for the call
            model: Model name (reserved for future per-model pricing)

        Returns:
            Estimated cost in USD for the prompt portion of the call
        """
        if self._config.cost_per_million_input <= 0:
            return 0.0
        return (prompt_tokens / 1_000_000) * self._config.cost_per_million_input

    def reset(self) -> None:
        """Reset all tracking state for a new session."""
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._stage_usage.clear()
        self._warning_emitted = False
        self._critical_emitted = False

    # ----- Properties -----

    @property
    def utilization(self) -> float:
        """Current budget utilization as a fraction (0.0 - 1.0)."""
        total = self._prompt_tokens + self._completion_tokens
        if self._config.max_session_tokens <= 0:
            return 0.0
        return min(total / self._config.max_session_tokens, 1.0)

    @property
    def is_warning(self) -> bool:
        """True if budget utilization >= warning threshold."""
        return self.utilization >= self._config.warning_threshold

    @property
    def is_critical(self) -> bool:
        """True if budget utilization >= critical threshold."""
        return self.utilization >= self._config.critical_threshold

    @property
    def is_exhausted(self) -> bool:
        """True if budget utilization >= 1.0 and hard limit is enabled."""
        return self._config.enable_hard_limit and self.utilization >= 1.0

    # ----- Internal helpers -----

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate total cost from token counts using configured pricing.

        Args:
            prompt_tokens: Total prompt tokens
            completion_tokens: Total completion tokens

        Returns:
            Estimated cost in USD (0.0 if pricing not configured)
        """
        cost = 0.0
        if self._config.cost_per_million_input > 0:
            cost += (prompt_tokens / 1_000_000) * self._config.cost_per_million_input
        if self._config.cost_per_million_output > 0:
            cost += (completion_tokens / 1_000_000) * self._config.cost_per_million_output
        return cost


# ===== 2. ConsecutiveFailureBreaker =====

class ConsecutiveFailureBreaker:
    """Circuit breaker for consecutive model/API failures.

    If N consecutive calls fail, the circuit opens and requests user
    intervention before continuing. This prevents burning tokens on
    repeated failing calls.

    States:
      - closed:   Normal operation. Failures are tracked but don't block.
      - open:     Too many consecutive failures. Calls should be paused.
      - half_open: Cooldown has passed. One retry is allowed.

    Default threshold: 5 consecutive failures within a 300-second window.

    Usage::

        breaker = ConsecutiveFailureBreaker(max_consecutive=5)
        state = breaker.record_failure("API timeout")
        if breaker.is_open:
            print("Circuit is open — too many failures!")
    """

    def __init__(
        self,
        max_consecutive: int = 5,
        window_seconds: float = 300.0,
        bus: EventBus | None = None,
    ) -> None:
        self._max_consecutive = max_consecutive
        self._window_seconds = window_seconds
        self._bus = bus

        # Failure tracking
        self._consecutive_failures: int = 0
        self._total_failures: int = 0
        self._total_successes: int = 0
        self._last_error: str = ""
        self._last_failure_time: float = 0.0

        # Circuit state
        self._state_name: str = "closed"  # "closed" | "open" | "half_open"
        self._failure_timestamps: list[float] = []

    # ----- Core methods -----

    def record_success(self) -> None:
        """Record a successful model call.

        Resets the consecutive failure counter and transitions
        the circuit from half_open or open back to closed.
        """
        self._total_successes += 1
        self._failure_timestamps.clear()
        self._consecutive_failures = 0
        if self._state_name != "closed":
            logger.info(
                f"Circuit breaker recovering: {self._state_name} → closed "
                f"(after {self._total_successes} total successes)"
            )
            self._state_name = "closed"

    def record_failure(self, error: str) -> BreakerState:
        """Record a failed model call.

        Increments the consecutive failure counter. If the threshold
        is reached within the time window, opens the circuit.

        Args:
            error: Description of the failure

        Returns:
            Current BreakerState after recording the failure
        """
        self._total_failures += 1
        self._last_error = error
        now = time.time()
        self._last_failure_time = now
        self._failure_timestamps.append(now)

        # Prune timestamps outside the window
        cutoff = now - self._window_seconds
        self._failure_timestamps = [
            t for t in self._failure_timestamps if t >= cutoff
        ]

        # Sync _consecutive_failures with actual windowed failure count
        self._consecutive_failures = len(self._failure_timestamps)

        if self._state_name == "half_open":
            # Failed during half-open — go back to open
            self._state_name = "open"
            logger.warning(
                "Circuit breaker re-opened: retry failed during half_open"
            )
            _safe_emit(
                self._bus,
                "circuit_open",
                consecutive_failures=self._consecutive_failures,
                total_failures=self._total_failures,
                last_error=error,
            )
        elif (
            self._state_name == "closed"
            and len(self._failure_timestamps) >= self._max_consecutive
        ):
            # Threshold reached within the time window — open the circuit
            self._state_name = "open"
            logger.warning(
                f"Circuit breaker opened: {len(self._failure_timestamps)} "
                f"failures within {self._window_seconds}s window "
                f"(threshold: {self._max_consecutive})"
            )
            _safe_emit(
                self._bus,
                "circuit_open",
                consecutive_failures=self._consecutive_failures,
                total_failures=self._total_failures,
                last_error=error,
            )

        return self.get_state()

    def get_state(self) -> BreakerState:
        """Get the current state of the circuit breaker.

        This is a query-only method — it does NOT transition state.
        Use try_half_open() to explicitly attempt the open → half_open transition.

        Returns:
            Current BreakerState with can_retry indicating if a transition
            to half_open is possible.
        """
        can_retry = False

        if self._state_name == "open":
            # Check if cooldown has elapsed (without transitioning)
            elapsed = time.time() - self._last_failure_time
            can_retry = elapsed >= self._window_seconds
        elif self._state_name == "half_open":
            can_retry = True

        return BreakerState(
            name=self._state_name,
            consecutive_failures=self._consecutive_failures,
            total_failures=self._total_failures,
            total_successes=self._total_successes,
            last_error=self._last_error,
            last_failure_time=self._last_failure_time,
            can_retry=can_retry,
        )

    def try_half_open(self) -> bool:
        """Explicitly attempt transition from open to half_open.

        Returns True if the transition was made, False otherwise.
        This should be called before attempting a retry, not by
        query methods like get_state() or is_open.
        """
        if self._state_name == "open":
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self._window_seconds:
                self._state_name = "half_open"
                logger.info(
                    f"Circuit breaker: open → half_open "
                    f"(cooldown elapsed after {elapsed:.0f}s)"
                )
                return True
        return False

    def reset(self) -> None:
        """Reset the circuit breaker to its initial closed state."""
        self._consecutive_failures = 0
        self._total_failures = 0
        self._total_successes = 0
        self._last_error = ""
        self._last_failure_time = 0.0
        self._failure_timestamps.clear()
        self._state_name = "closed"

    # ----- Properties -----

    @property
    def is_open(self) -> bool:
        """True if the circuit is open (too many consecutive failures).

        Note: This is a query-only property. Use try_half_open() to
        explicitly transition from open → half_open before retrying.
        """
        return self._state_name == "open"


# ===== 3. LatencyBreaker =====

class LatencyBreaker:
    """Circuit breaker for high latency model calls.

    Tracks rolling average latency and warns if consistently slow.
    Does NOT block — only warns and suggests. This is purely advisory.

    Usage::

        breaker = LatencyBreaker(warn_latency_ms=30000)
        breaker.record_latency(45000)  # 45 seconds
        if breaker.is_slow:
            print("Model calls are consistently slow!")
    """

    def __init__(
        self,
        warn_latency_ms: float = 30_000.0,
        avg_window: int = 10,
        bus: EventBus | None = None,
    ) -> None:
        self._warn_latency_ms = warn_latency_ms
        self._avg_window = avg_window
        self._bus = bus

        # Rolling latency window
        self._latencies: deque[float] = deque(maxlen=avg_window)

        # Tracking
        self._total_calls: int = 0
        self._slow_calls: int = 0  # Calls exceeding warn threshold
        self._peak_latency_ms: float = 0.0
        self._slow_warning_emitted: bool = False

    def record_latency(self, latency_ms: float) -> None:
        """Record the latency of a model call.

        Args:
            latency_ms: Latency in milliseconds
        """
        self._latencies.append(latency_ms)
        self._total_calls += 1

        if latency_ms > self._peak_latency_ms:
            self._peak_latency_ms = latency_ms

        if latency_ms >= self._warn_latency_ms:
            self._slow_calls += 1

        # Emit warning if rolling average exceeds threshold
        if self.is_slow and not self._slow_warning_emitted:
            self._slow_warning_emitted = True
            avg = self.get_avg_latency()
            logger.warning(
                f"Latency warning: rolling average {avg:.0f}ms "
                f"exceeds threshold {self._warn_latency_ms:.0f}ms"
            )
            _safe_emit(
                self._bus,
                "latency_warning",
                avg_latency_ms=avg,
                warn_threshold_ms=self._warn_latency_ms,
                total_calls=self._total_calls,
                slow_calls=self._slow_calls,
            )
        elif not self.is_slow:
            # Reset the flag so we can warn again if it goes slow later
            self._slow_warning_emitted = False

    def get_avg_latency(self) -> float:
        """Get the rolling average latency.

        Returns:
            Average latency in milliseconds over the rolling window,
            or 0.0 if no calls have been recorded
        """
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    def get_state(self) -> dict:
        """Get the current state of the latency breaker.

        Returns:
            Dictionary with latency statistics and configuration
        """
        return {
            "avg_latency_ms": self.get_avg_latency(),
            "peak_latency_ms": self._peak_latency_ms,
            "total_calls": self._total_calls,
            "slow_calls": self._slow_calls,
            "warn_threshold_ms": self._warn_latency_ms,
            "is_slow": self.is_slow,
            "window_size": self._avg_window,
            "current_window_samples": len(self._latencies),
        }

    def reset(self) -> None:
        """重置延迟追踪状态"""
        self._latencies.clear()
        self._total_calls = 0
        self._slow_calls = 0
        self._peak_latency_ms = 0.0
        self._slow_warning_emitted = False

    @property
    def is_slow(self) -> bool:
        """True if the recent average latency exceeds the warn threshold.

        Requires at least 3 samples in the window to avoid false positives
        from a single slow call.
        """
        if len(self._latencies) < 3:
            return False
        return self.get_avg_latency() > self._warn_latency_ms


# ===== 4. ProgressDetector =====

class ProgressDetector:
    """Detect if AgentLoop is stuck (no meaningful progress).

    Tracks:
      - Conversation length growth
      - Unique tool calls
      - Whether tool calls had effects

    If no progress is detected after N steps, emits a warning.
    Like LatencyBreaker, this is advisory — it does NOT block.

    The stall score is computed from the ratio of ineffective steps
    (steps where had_effect=False) over the recent window. A score
    of 1.0 means all recent steps were ineffective.

    Usage::

        detector = ProgressDetector(stall_threshold=10)
        detector.record_step("read_file", had_effect=True)
        detector.record_step("read_file", had_effect=False)
        if detector.is_stalled():
            print("AgentLoop appears stuck!")
    """

    def __init__(
        self,
        stall_threshold: int = 10,
        bus: EventBus | None = None,
    ) -> None:
        self._stall_threshold = stall_threshold
        self._bus = bus

        # Step tracking
        self._steps: list[dict[str, Any]] = []  # {"tool": str, "had_effect": bool}
        self._unique_tools: set[str] = set()

        # Stall warning tracking
        self._stall_warning_emitted: bool = False

    def record_step(self, tool_name: str, had_effect: bool) -> None:
        """Record an AgentLoop step.

        Args:
            tool_name: Name of the tool that was called
            had_effect: Whether the tool call produced a meaningful change
                (e.g., file was written, code was modified, useful info retrieved)
        """
        self._steps.append({"tool": tool_name, "had_effect": had_effect})
        self._unique_tools.add(tool_name)

        # Cap _steps to prevent unbounded growth — keep only 2× stall_threshold
        max_keep = self._stall_threshold * 2
        if len(self._steps) > max_keep:
            self._steps = self._steps[-max_keep:]

        # Check for stall after recording
        if self.is_stalled() and not self._stall_warning_emitted:
            self._stall_warning_emitted = True
            score = self.get_stall_score()
            logger.warning(
                f"Progress stall detected: stall_score={score:.2f}, "
                f"steps={len(self._steps)}, unique_tools={len(self._unique_tools)}"
            )
            _safe_emit(
                self._bus,
                "progress_stalled",
                stall_score=score,
                total_steps=len(self._steps),
                unique_tools=len(self._unique_tools),
                recent_tools=[
                    s["tool"] for s in self._steps[-self._stall_threshold :]
                ],
            )

    def is_stalled(self) -> bool:
        """Check if the AgentLoop appears stuck.

        Stall is detected when:
          - At least stall_threshold steps have been recorded, AND
          - The stall score exceeds 0.8 (80% of recent steps had no effect)

        Returns:
            True if the loop appears stuck
        """
        if len(self._steps) < self._stall_threshold:
            return False
        return self.get_stall_score() >= 0.8

    def get_stall_score(self) -> float:
        """Get a stall score from 0.0 to 1.0.

        0.0 = all recent steps had effects (good progress)
        1.0 = no recent steps had effects (completely stuck)

        The score is computed from the last `stall_threshold` steps.

        Returns:
            Stall score between 0.0 and 1.0
        """
        window = self._steps[-self._stall_threshold :]
        if not window:
            return 0.0
        ineffective = sum(1 for s in window if not s["had_effect"])
        return ineffective / len(window)

    def reset(self) -> None:
        """Reset the progress detector for a new session."""
        self._steps.clear()
        self._unique_tools.clear()
        self._stall_warning_emitted = False


# ===== 5. CircuitBreakerManager =====

class CircuitBreakerManager:
    """Unified circuit breaker manager.

    Manages: CostBudgetTracker, ConsecutiveFailureBreaker, LatencyBreaker,
    ProgressDetector.

    This is the main entry point for the rest of the system. All circuit
    breaker interactions should go through this manager for consistent
    event emission and unified status reporting.

    Events emitted:
      - budget_warning    : approaching budget limit (advisory)
      - budget_critical   : critical budget level (advisory, suggest pause)
      - budget_exhausted  : budget fully consumed (only if hard_limit enabled)
      - circuit_open      : too many consecutive failures, pausing
      - latency_warning   : consistently slow model calls
      - progress_stalled  : AgentLoop appears stuck

    Usage::

        manager = CircuitBreakerManager(bus=event_bus)
        result = manager.record_model_call(
            prompt_tokens=500, completion_tokens=200,
            stage="plan", latency_ms=3500
        )
        if result.level != "ok":
            print(result.message)
    """

    def __init__(
        self,
        config: dict | TypedCircuitBreakerConfig | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self._bus = bus

        # Accept both dict and typed config
        if isinstance(config, TypedCircuitBreakerConfig):
            self._typed_config = config
            self._config = {}  # empty dict for backward compat
        else:
            self._typed_config = None
            self._config = config or {}

        # Build sub-components from config
        if self._typed_config:
            # Typed config path — use typed attributes directly
            budget_config = CostBudgetConfig(
                max_session_tokens=self._typed_config.budget.max_session_tokens,
                warning_threshold=self._typed_config.budget.warning_threshold,
                critical_threshold=self._typed_config.budget.critical_threshold,
                cost_per_million_input=self._typed_config.budget.cost_per_million_input,
                cost_per_million_output=self._typed_config.budget.cost_per_million_output,
                enable_hard_limit=self._typed_config.budget.enable_hard_limit,
                auto_downgrade_model=self._typed_config.budget.auto_downgrade_model,
            )
            self._cost_tracker = CostBudgetTracker(config=budget_config, bus=bus)

            self._failure_breaker = ConsecutiveFailureBreaker(
                max_consecutive=self._typed_config.failure_breaker.max_consecutive,
                window_seconds=self._typed_config.failure_breaker.window_seconds,
                bus=bus,
            )

            self._latency_breaker = LatencyBreaker(
                warn_latency_ms=self._typed_config.latency_breaker.warn_latency_ms,
                avg_window=self._typed_config.latency_breaker.avg_window,
                bus=bus,
            )

            self._progress_detector = ProgressDetector(
                stall_threshold=self._typed_config.progress_detector.stall_threshold,
                bus=bus,
            )
        else:
            # Legacy dict-based init (backward compat)
            budget_config = self._build_budget_config()
            self._cost_tracker = CostBudgetTracker(config=budget_config, bus=bus)

            failure_config = self._config.get("failure_breaker", {})
            self._failure_breaker = ConsecutiveFailureBreaker(
                max_consecutive=failure_config.get("max_consecutive", 5),
                window_seconds=failure_config.get("window_seconds", 300.0),
                bus=bus,
            )

            latency_config = self._config.get("latency_breaker", {})
            self._latency_breaker = LatencyBreaker(
                warn_latency_ms=latency_config.get("warn_latency_ms", 30_000.0),
                avg_window=latency_config.get("avg_window", 10),
                bus=bus,
            )

            progress_config = self._config.get("progress_detector", {})
            self._progress_detector = ProgressDetector(
                stall_threshold=progress_config.get("stall_threshold", 10),
                bus=bus,
            )

    def _build_budget_config(self) -> CostBudgetConfig:
        """Build CostBudgetConfig from the manager's config.

        Supports both dict and TypedCircuitBreakerConfig.
        When typed config is available, this method is not used during __init__,
        but is kept for backward compat and any external callers.

        Returns:
            CostBudgetConfig with values from config or generous defaults
        """
        if self._typed_config:
            return CostBudgetConfig(
                max_session_tokens=self._typed_config.budget.max_session_tokens,
                warning_threshold=self._typed_config.budget.warning_threshold,
                critical_threshold=self._typed_config.budget.critical_threshold,
                cost_per_million_input=self._typed_config.budget.cost_per_million_input,
                cost_per_million_output=self._typed_config.budget.cost_per_million_output,
                enable_hard_limit=self._typed_config.budget.enable_hard_limit,
                auto_downgrade_model=self._typed_config.budget.auto_downgrade_model,
            )
        # Legacy dict-based path
        budget = self._config.get("budget", {})
        return CostBudgetConfig(
            max_session_tokens=budget.get("max_session_tokens", 10_000_000),
            warning_threshold=budget.get("warning_threshold", 0.7),
            critical_threshold=budget.get("critical_threshold", 0.9),
            cost_per_million_input=budget.get("cost_per_million_input", 0.0),
            cost_per_million_output=budget.get("cost_per_million_output", 0.0),
            enable_hard_limit=budget.get("enable_hard_limit", False),
            auto_downgrade_model=budget.get("auto_downgrade_model", ""),
        )

    # ----- Cost tracking -----

    def record_model_call(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        stage: str,
        latency_ms: float,
    ) -> BudgetCheckResult:
        """Record a model call with full tracking.

        This is the primary method for tracking model usage. It:
        1. Records token usage and checks the budget
        2. Records the call latency
        3. Returns the budget check result

        Args:
            prompt_tokens: Number of prompt/input tokens
            completion_tokens: Number of completion/output tokens
            stage: Pipeline stage (e.g., "design", "plan", "execute")
            latency_ms: Call latency in milliseconds

        Returns:
            BudgetCheckResult with current budget state
        """
        # Track cost/tokens
        result = self._cost_tracker.record_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            stage=stage,
        )

        # Track latency
        self._latency_breaker.record_latency(latency_ms)

        return result

    def check_before_call(self, estimated_prompt_tokens: int) -> BudgetCheckResult:
        """Check the budget before making a model call.

        Use this to decide whether to proceed with a call. If the result
        level is "exhausted" and hard_limit is enabled, the call should
        not be made.

        This method does NOT record any usage — it's a pre-flight check.

        Args:
            estimated_prompt_tokens: Estimated tokens for the upcoming call

        Returns:
            BudgetCheckResult with current budget state
        """
        result = self._cost_tracker.check_budget()

        # Pre-flight estimation: if estimated tokens are provided and current
        # level is "ok", project whether this call would push us into
        # warning/critical territory
        if estimated_prompt_tokens > 0 and result.level == "ok":
            projected_total = (
                self._cost_tracker._prompt_tokens
                + self._cost_tracker._completion_tokens
                + estimated_prompt_tokens
            )
            projected_util = projected_total / self._cost_tracker._config.max_session_tokens if self._cost_tracker._config.max_session_tokens > 0 else 0.0
            projected_util = min(projected_util, 1.0)
            projected_cost = self._cost_tracker._estimate_cost(
                self._cost_tracker._prompt_tokens + estimated_prompt_tokens,
                self._cost_tracker._completion_tokens,
            )

            if projected_util >= self._cost_tracker._config.critical_threshold:
                return BudgetCheckResult(
                    level="critical",
                    message=f"Projected cost after this call: {projected_util:.1%}",
                    utilization=projected_util,
                    prompt_tokens_used=self._cost_tracker._prompt_tokens,
                    completion_tokens_used=self._cost_tracker._completion_tokens,
                    total_tokens_used=projected_total,
                    max_tokens=self._cost_tracker._config.max_session_tokens,
                    estimated_cost=projected_cost,
                )
            elif projected_util >= self._cost_tracker._config.warning_threshold:
                return BudgetCheckResult(
                    level="warning",
                    message=f"Projected cost after this call: {projected_util:.1%}",
                    utilization=projected_util,
                    prompt_tokens_used=self._cost_tracker._prompt_tokens,
                    completion_tokens_used=self._cost_tracker._completion_tokens,
                    total_tokens_used=projected_total,
                    max_tokens=self._cost_tracker._config.max_session_tokens,
                    estimated_cost=projected_cost,
                )

        return result

    # ----- Failure tracking -----

    def record_success(self) -> None:
        """Record a successful model call.

        Resets the consecutive failure counter in the failure breaker.
        """
        self._failure_breaker.record_success()

    def record_failure(self, error: str) -> BreakerState:
        """Record a failed model call.

        Args:
            error: Description of the failure

        Returns:
            Current BreakerState after recording the failure
        """
        return self._failure_breaker.record_failure(error)

    # ----- Progress tracking -----

    def record_agent_step(self, tool_name: str, had_effect: bool) -> None:
        """Record an AgentLoop step for progress tracking.

        Args:
            tool_name: Name of the tool that was called
            had_effect: Whether the tool call produced a meaningful change
        """
        self._progress_detector.record_step(tool_name, had_effect)

    # ----- Status queries -----

    def get_status(self) -> dict:
        """Get the full status of all circuit breakers.

        This is used by the /status and /cost commands to display
        comprehensive system health information.

        Returns:
            Dictionary with status of all sub-components
        """
        budget = self._cost_tracker.check_budget()
        failure = self._failure_breaker.get_state()

        return {
            "budget": {
                "level": budget.level,
                "utilization": f"{budget.utilization:.1%}",
                "total_tokens": budget.total_tokens_used,
                "max_tokens": budget.max_tokens,
                "estimated_cost": f"${budget.estimated_cost:.4f}",
                "message": budget.message,
                "hard_limit_enabled": self._cost_tracker._config.enable_hard_limit,
            },
            "failure_breaker": {
                "state": failure.name,
                "consecutive_failures": failure.consecutive_failures,
                "total_failures": failure.total_failures,
                "total_successes": failure.total_successes,
                "can_retry": failure.can_retry,
                "last_error": failure.last_error,
            },
            "latency": self._latency_breaker.get_state(),
            "progress": {
                "total_steps": len(self._progress_detector._steps),
                "unique_tools": len(self._progress_detector._unique_tools),
                "stall_score": f"{self._progress_detector.get_stall_score():.2f}",
                "is_stalled": self._progress_detector.is_stalled(),
            },
        }

    def get_budget_summary(self) -> dict:
        """Get the budget summary for the /cost command.

        Returns:
            Dictionary with detailed cost/budget information
        """
        return self._cost_tracker.get_session_summary()

    def reset_all(self) -> None:
        """重置所有熔断器状态"""
        if hasattr(self, '_cost_tracker') and self._cost_tracker:
            self._cost_tracker.reset()
        if hasattr(self, '_failure_breaker') and self._failure_breaker:
            self._failure_breaker.reset()
        if hasattr(self, '_latency_breaker') and self._latency_breaker:
            self._latency_breaker.reset()
        if hasattr(self, '_progress_detector') and self._progress_detector:
            self._progress_detector.reset()

    # ----- Properties -----

    @property
    def cost_tracker(self) -> CostBudgetTracker:
        """Access the underlying CostBudgetTracker."""
        return self._cost_tracker

    @property
    def failure_breaker(self) -> ConsecutiveFailureBreaker:
        """Access the underlying ConsecutiveFailureBreaker."""
        return self._failure_breaker

    @property
    def latency_breaker(self) -> LatencyBreaker:
        """Access the underlying LatencyBreaker."""
        return self._latency_breaker

    @property
    def progress_detector(self) -> ProgressDetector:
        """Access the underlying ProgressDetector."""
        return self._progress_detector


# ===== Per-Model Circuit Breaker (P4-3) =====


@dataclass
class ModelBreakerConfig:
    """Per-model circuit breaker configuration.

    Each model (V4-Flash, V4-Pro, M3, GLM-5) can have its own breaker
    parameters tuned to the model's reliability characteristics.

    Attributes:
        model_name: Identifier for the model (e.g., "deepseek_v4_pro", "minimax_m3", "glm_5")
        max_consecutive_failures: Number of consecutive failures before opening the breaker
        window_seconds: Time window in seconds for failure-rate calculation
        cooldown_seconds: Seconds to wait before transitioning from open to half_open
        failure_threshold_percent: Open circuit if failure rate exceeds this fraction in window
        half_open_max_calls: Number of test calls allowed in half_open state
    """

    model_name: str  # e.g., "deepseek_v4_pro", "minimax_m3", "glm_5"
    max_consecutive_failures: int = 5
    window_seconds: float = 300.0
    cooldown_seconds: float = 60.0
    failure_threshold_percent: float = 0.5  # Open circuit if >50% failures in window
    half_open_max_calls: int = 3  # Allow 3 test calls in half-open state


@dataclass
class ModelBreakerState:
    """Runtime state for a model's circuit breaker.

    Attributes:
        model_name: Identifier for the model
        state: Current breaker state — "closed", "open", or "half_open"
        failure_count: Number of consecutive failures
        success_count: Number of consecutive successes (in half_open)
        last_failure_time: Timestamp of the most recent failure
        last_state_change: Timestamp of the most recent state transition
        window_failures: List of (timestamp, error_msg) tuples within the sliding window
        window_total_calls: Total calls (successes + failures) within the sliding window
    """

    model_name: str
    state: str  # "closed", "open", "half_open"
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_state_change: float = 0.0
    window_failures: list[tuple[float, str]] = field(default_factory=list)  # (timestamp, error_msg) tuples
    window_total_calls: int = 0  # total calls in the sliding window


# Default degradation chain for the three-model architecture
_DEFAULT_DEGRADATION_CHAIN: list[list[str]] = [
    ["deepseek_v4_pro", "glm_5", "deepseek_v4_flash"],
    ["minimax_m3", "deepseek_v4_pro"],
]


class ModelCircuitBreakerManager:
    """Manages per-model circuit breakers with cross-model degradation support.

    Each model (V4-Flash, V4-Pro, M3, GLM-5) has its own circuit breaker.
    When a model's breaker opens, the degradation chain is consulted for fallback.

    Usage::

        manager = ModelCircuitBreakerManager()
        fallback = manager.record_failure("deepseek_v4_pro", "API timeout")
        if fallback:
            print(f"Breaker opened! Falling back to {fallback}")

        if manager.can_call("deepseek_v4_pro"):
            # Safe to call the model
            ...

        # Check all model states
        states = manager.get_all_states()
    """

    def __init__(
        self,
        configs: list[ModelBreakerConfig] | None = None,
        degradation_chain: list[list[str]] | None = None,
    ) -> None:
        """Initialize ModelCircuitBreakerManager.

        Args:
            configs: Per-model breaker configurations. If None, default configs
                are created for the standard three-model architecture.
            degradation_chain: Ordered fallback chains. Each inner list is a
                priority-ordered sequence of model names. If None, the default
                degradation chain is used.
        """
        now = time.time()

        # Build per-model configs
        if configs:
            self._configs: dict[str, ModelBreakerConfig] = {
                c.model_name: c for c in configs
            }
        else:
            # Default configs for the three-model architecture
            default_models = [
                "deepseek_v4_pro",
                "deepseek_v4_flash",
                "minimax_m3",
                "glm_5",
            ]
            self._configs = {
                name: ModelBreakerConfig(model_name=name) for name in default_models
            }

        # Initialize per-model states
        self._states: dict[str, ModelBreakerState] = {
            name: ModelBreakerState(
                model_name=name,
                state="closed",
                last_state_change=now,
            )
            for name in self._configs
        }

        # Degradation chain: maps model_name → ordered fallback list
        self._degradation_chain: list[list[str]] = (
            degradation_chain if degradation_chain is not None
            else _DEFAULT_DEGRADATION_CHAIN
        )

        # Half-open call tracking
        self._half_open_calls: dict[str, int] = {}

    # ----- Core methods -----

    def record_success(self, model_name: str) -> None:
        """Record a successful call for a model.

        In closed state: resets consecutive failure count and records in window.
        In half_open state: increments success count; closes breaker if
        enough successful test calls have been made.

        Args:
            model_name: The model that succeeded
        """
        state = self._get_or_create_state(model_name)
        config = self._get_or_create_config(model_name)

        # Record in the sliding window (success doesn't have an error message)
        if model_name in self._states:
            state.window_total_calls += 1
            self._prune_window(model_name)

        if state.state == "half_open":
            state.success_count += 1
            if state.success_count >= config.half_open_max_calls:
                self._transition(model_name, "closed")
                logger.info(
                    f"Model circuit breaker CLOSED for {model_name} "
                    f"after {state.success_count} successful test calls"
                )
        elif state.state == "closed":
            state.failure_count = 0

        logger.debug(
            f"Model breaker success recorded for {model_name}: "
            f"state={state.state}, failure_count={state.failure_count}"
        )

    def record_failure(self, model_name: str, error: str = "") -> str | None:
        """Record a failure for a model.

        Returns the fallback model name if the breaker just opened,
        or None if the breaker was already open or remains closed.

        Args:
            model_name: The model that failed
            error: Error message describing the failure

        Returns:
            Fallback model name if breaker just opened, None otherwise
        """
        state = self._get_or_create_state(model_name)
        config = self._get_or_create_config(model_name)
        now = time.time()

        # Record the failure in the sliding window
        state.window_failures.append((now, error))
        state.window_total_calls += 1
        self._prune_window(model_name)

        if state.state == "half_open":
            # Any failure in half_open reopens the breaker
            self._transition(model_name, "open")
            fallback = self.get_fallback(model_name)
            logger.warning(
                f"Model circuit breaker re-OPENED for {model_name} "
                f"after failure in half_open: {error}"
            )
            return fallback

        if state.state == "closed":
            state.failure_count += 1
            state.last_failure_time = now

            # Check consecutive failure threshold
            if state.failure_count >= config.max_consecutive_failures:
                self._transition(model_name, "open")
                fallback = self.get_fallback(model_name)
                logger.warning(
                    f"Model circuit breaker OPENED for {model_name} "
                    f"after {state.failure_count} consecutive failures"
                )
                return fallback

            # Check failure rate threshold in the sliding window
            # Only check when we have enough total calls for a meaningful rate
            if state.window_total_calls >= 10:
                window_failure_rate = self._get_failure_rate(model_name)
                if window_failure_rate > config.failure_threshold_percent:
                    self._transition(model_name, "open")
                    fallback = self.get_fallback(model_name)
                    logger.warning(
                        f"Model circuit breaker OPENED for {model_name} "
                        f"due to failure rate {window_failure_rate:.1%} "
                        f"in last {config.window_seconds:.0f}s "
                        f"({len(state.window_failures)}/{state.window_total_calls} calls)"
                    )
                    return fallback

        # state is "open" — already open, no new fallback needed
        return None

    def get_state(self, model_name: str) -> str:
        """Get the current breaker state for a model.

        Also handles automatic transition from "open" to "half_open"
        if the cooldown period has elapsed.

        Args:
            model_name: The model to check

        Returns:
            "closed", "open", or "half_open"
        """
        state = self._get_or_create_state(model_name)
        config = self._get_or_create_config(model_name)

        # Auto-transition from open to half_open after cooldown
        if state.state == "open":
            elapsed = time.time() - state.last_state_change
            if elapsed >= config.cooldown_seconds:
                self._transition(model_name, "half_open")
                logger.info(
                    f"Model circuit breaker HALF_OPEN for {model_name} "
                    f"after {elapsed:.1f}s cooldown"
                )

        return self._states[model_name].state

    def can_call(self, model_name: str) -> bool:
        """Check if a call can be made to the specified model.

        Args:
            model_name: The model to check

        Returns:
            True if the model is callable (closed or half_open with remaining
            test calls), False if the breaker is open
        """
        current_state = self.get_state(model_name)

        if current_state == "closed":
            return True
        elif current_state == "half_open":
            config = self._get_or_create_config(model_name)
            calls_made = self._half_open_calls.get(model_name, 0)
            return calls_made < config.half_open_max_calls
        else:  # open
            return False

    def get_fallback(self, model_name: str) -> str | None:
        """Get the best available fallback model for the given model.

        Consults the degradation chain and skips models whose breakers
        are also open.

        Args:
            model_name: The model that needs a fallback

        Returns:
            Name of the first available fallback model, or None if
            all fallbacks are unavailable
        """
        for chain in self._degradation_chain:
            if model_name not in chain:
                continue
            idx = chain.index(model_name)
            for candidate in chain[idx + 1:]:
                if self.get_state(candidate) != "open":
                    return candidate
        return None

    def get_all_states(self) -> dict[str, str]:
        """Get the breaker state for all known models.

        Returns:
            Dict mapping model_name → state string
        """
        return {name: self.get_state(name) for name in self._states}

    def reset(self, model_name: str | None = None) -> None:
        """Reset breaker state.

        Args:
            model_name: Specific model to reset, or None to reset all
        """
        now = time.time()
        if model_name is None:
            for name in self._states:
                self._states[name].state = "closed"
                self._states[name].failure_count = 0
                self._states[name].success_count = 0
                self._states[name].last_failure_time = 0.0
                self._states[name].last_state_change = now
                self._states[name].window_failures.clear()
                self._half_open_calls.pop(name, None)
        elif model_name in self._states:
            state = self._states[model_name]
            state.state = "closed"
            state.failure_count = 0
            state.success_count = 0
            state.last_failure_time = 0.0
            state.last_state_change = now
            state.window_failures.clear()
            self._half_open_calls.pop(model_name, None)

    # ----- Internal helpers -----

    def _transition(self, model_name: str, new_state: str) -> None:
        """Transition a model's breaker to a new state.

        Args:
            model_name: The model whose breaker state is changing
            new_state: The new state ("closed", "open", "half_open")
        """
        state = self._states[model_name]
        old_state = state.state
        state.state = new_state
        state.last_state_change = time.time()

        if new_state == "half_open":
            state.success_count = 0
            state.failure_count = 0
            self._half_open_calls[model_name] = 0
        elif new_state == "closed":
            state.success_count = 0
            state.failure_count = 0
            self._half_open_calls.pop(model_name, None)
        elif new_state == "open":
            self._half_open_calls.pop(model_name, None)

        logger.debug(
            f"Model breaker transition: {model_name} {old_state} → {new_state}"
        )

    def _prune_window(self, model_name: str) -> None:
        """Remove failures outside the sliding window and recalculate total calls.

        The window_total_calls counter is approximate — it is incremented on
        every call and only fully recalculated here. For most workloads this
        is sufficient because prune is called on every call and stale entries
        are removed regularly.

        Args:
            model_name: The model whose window to prune
        """
        state = self._states[model_name]
        config = self._configs[model_name]
        cutoff = time.time() - config.window_seconds
        old_failure_count = len(state.window_failures)
        state.window_failures = [
            (ts, msg) for ts, msg in state.window_failures if ts > cutoff
        ]
        # Adjust window_total_calls by the number of pruned failures.
        # Note: successes that fell outside the window can't be precisely
        # tracked, so window_total_calls may slightly overcount. This is
        # acceptable for rate-limiting purposes.
        pruned = old_failure_count - len(state.window_failures)
        if pruned > 0:
            state.window_total_calls = max(
                len(state.window_failures),
                state.window_total_calls - pruned,
            )

    def _get_failure_rate(self, model_name: str) -> float:
        """Calculate the failure rate within the sliding window.

        Uses window_total_calls as the denominator and window_failures
        count as the numerator.

        Args:
            model_name: The model to calculate rate for

        Returns:
            Failure rate as a float between 0.0 and 1.0
        """
        state = self._states[model_name]
        if state.window_total_calls == 0:
            return 0.0
        return len(state.window_failures) / state.window_total_calls

    def _get_or_create_state(self, model_name: str) -> ModelBreakerState:
        """Get existing state or create default state for a model.

        Args:
            model_name: The model name

        Returns:
            ModelBreakerState for the model
        """
        if model_name not in self._states:
            now = time.time()
            self._states[model_name] = ModelBreakerState(
                model_name=model_name,
                state="closed",
                last_state_change=now,
                window_total_calls=0,
            )
        return self._states[model_name]

    def _get_or_create_config(self, model_name: str) -> ModelBreakerConfig:
        """Get existing config or create default config for a model.

        Args:
            model_name: The model name

        Returns:
            ModelBreakerConfig for the model
        """
        if model_name not in self._configs:
            self._configs[model_name] = ModelBreakerConfig(model_name=model_name)
        return self._configs[model_name]


# ===== Four-Model Circuit Breaker Configuration (P4-3) =====


@dataclass
class ModelCircuitBreakerConfig:
    """Per-model circuit breaker configuration for the four-model architecture.

    Each model (V4-Flash, V4-Pro, M3, GLM-5, GLM-5.2) can have its own breaker
    parameters tuned to the model's reliability characteristics and context needs.

    Attributes:
        model_name: Identifier for the model (e.g., "deepseek_v4_flash", "glm_52")
        failure_threshold: Number of consecutive failures before opening the breaker
        recovery_timeout: Seconds to wait before transitioning from open to half_open
        half_open_max: Number of test calls allowed in half_open state
        window_seconds: Sliding window for failure-rate calculation (default: 300s)
        failure_threshold_percent: Open circuit if failure rate exceeds this fraction
    """

    model_name: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max: int = 2
    window_seconds: float = 300.0
    failure_threshold_percent: float = 0.5


# Pre-configured breaker settings for each model in the four-model architecture
FOUR_MODEL_BREAKER_CONFIGS: dict[str, ModelCircuitBreakerConfig] = {
    "deepseek_v4_flash": ModelCircuitBreakerConfig(
        model_name="deepseek_v4_flash",
        failure_threshold=5,
        recovery_timeout=30.0,
        half_open_max=2,
        window_seconds=300.0,
    ),
    "deepseek_v4_pro": ModelCircuitBreakerConfig(
        model_name="deepseek_v4_pro",
        failure_threshold=3,
        recovery_timeout=60.0,
        half_open_max=1,
        window_seconds=300.0,
    ),
    "minimax_m3": ModelCircuitBreakerConfig(
        model_name="minimax_m3",
        failure_threshold=5,
        recovery_timeout=30.0,
        half_open_max=2,
        window_seconds=300.0,
    ),
    "glm_5": ModelCircuitBreakerConfig(
        model_name="glm_5",
        failure_threshold=4,
        recovery_timeout=45.0,
        half_open_max=2,
        window_seconds=300.0,
    ),
    "glm_52": ModelCircuitBreakerConfig(
        model_name="glm_52",
        failure_threshold=3,
        recovery_timeout=90.0,   # 1M context needs longer recovery
        half_open_max=1,
        window_seconds=300.0,
    ),
}


class FourModelCircuitBreakerManager:
    """Manages per-model circuit breakers for the four-model architecture.

    Uses FOUR_MODEL_BREAKER_CONFIGS to create breakers with model-specific
    parameters. Supports health-aware fallback via degradation chains.

    Supported models:
      - deepseek_v4_flash (V4-Flash)
      - deepseek_v4_pro   (V4-Pro)
      - minimax_m3        (MiniMax M3)
      - glm_5             (GLM-5)
      - glm_52            (GLM-5.2)

    Usage::

        from teragent.reliability.circuit_breaker import FourModelCircuitBreakerManager

        mgr = FourModelCircuitBreakerManager()
        fallback = mgr.record_failure("deepseek_v4_pro", "API timeout")
        if fallback:
            print(f"Breaker opened! Falling back to {fallback}")

        if mgr.can_call("glm_52"):
            # Safe to call GLM-5.2
            ...

        # Check all model states
        states = mgr.get_all_states()
    """

    def __init__(
        self,
        configs: dict[str, ModelCircuitBreakerConfig] | None = None,
        degradation_chains: dict[str, list[str]] | None = None,
    ) -> None:
        """Initialize FourModelCircuitBreakerManager.

        Args:
            configs: Per-model breaker configurations. If None, uses
                FOUR_MODEL_BREAKER_CONFIGS defaults.
            degradation_chains: Named degradation chains mapping chain_name
                to an ordered list of model names. If None, default chains
                for the four-model architecture are used.
        """
        now = time.time()

        # Use provided configs or the standard four-model defaults
        self._configs: dict[str, ModelCircuitBreakerConfig] = (
            configs if configs is not None else dict(FOUR_MODEL_BREAKER_CONFIGS)
        )

        # Initialize per-model breaker states
        self._states: dict[str, ModelBreakerState] = {
            name: ModelBreakerState(
                model_name=name,
                state="closed",
                last_state_change=now,
            )
            for name in self._configs
        }

        # Default degradation chains for four-model architecture
        self._degradation_chains: dict[str, list[str]] = (
            degradation_chains if degradation_chains is not None
            else {
                "primary": ["deepseek_v4_pro", "glm_52", "glm_5", "deepseek_v4_flash"],
                "multimodal": ["minimax_m3", "deepseek_v4_pro"],
                "long_horizon": ["glm_52", "glm_5", "deepseek_v4_flash"],
                "1m_context": ["glm_52", "deepseek_v4_pro", "minimax_m3"],
            }
        )

        # Half-open call tracking
        self._half_open_calls: dict[str, int] = {}

    # ----- Core methods -----

    def record_success(self, model_name: str) -> None:
        """Record a successful call for a model.

        In closed state: resets consecutive failure count.
        In half_open state: increments success count; closes breaker if
        enough successful test calls have been made.

        Args:
            model_name: The model that succeeded
        """
        state = self._get_or_create_state(model_name)
        config = self._get_config(model_name)

        # Record in sliding window
        if model_name in self._states:
            state.window_total_calls += 1
            self._prune_window(model_name)

        if state.state == "half_open":
            state.success_count += 1
            if state.success_count >= config.half_open_max:
                self._transition(model_name, "closed")
                logger.info(
                    f"Four-model circuit breaker CLOSED for {model_name} "
                    f"after {state.success_count} successful test calls"
                )
        elif state.state == "closed":
            state.failure_count = 0

        logger.debug(
            f"Four-model breaker success recorded for {model_name}: "
            f"state={state.state}, failure_count={state.failure_count}"
        )

    def record_failure(self, model_name: str, error: str = "") -> str | None:
        """Record a failure for a model.

        Returns the fallback model name if the breaker just opened,
        or None if the breaker was already open or remains closed.

        Args:
            model_name: The model that failed
            error: Error message describing the failure

        Returns:
            Fallback model name if breaker just opened, None otherwise
        """
        state = self._get_or_create_state(model_name)
        config = self._get_config(model_name)
        now = time.time()

        # Record in sliding window
        state.window_failures.append((now, error))
        state.window_total_calls += 1
        self._prune_window(model_name)

        if state.state == "half_open":
            # Any failure in half_open reopens the breaker
            self._transition(model_name, "open")
            fallback = self.get_fallback(model_name)
            logger.warning(
                f"Four-model circuit breaker re-OPENED for {model_name} "
                f"after failure in half_open: {error}"
            )
            return fallback

        if state.state == "closed":
            state.failure_count += 1
            state.last_failure_time = now

            # Check consecutive failure threshold (model-specific)
            if state.failure_count >= config.failure_threshold:
                self._transition(model_name, "open")
                fallback = self.get_fallback(model_name)
                logger.warning(
                    f"Four-model circuit breaker OPENED for {model_name} "
                    f"after {state.failure_count} consecutive failures "
                    f"(threshold: {config.failure_threshold})"
                )
                return fallback

            # Check failure rate threshold in the sliding window
            if state.window_total_calls >= 10:
                window_failure_rate = self._get_failure_rate(model_name)
                if window_failure_rate > config.failure_threshold_percent:
                    self._transition(model_name, "open")
                    fallback = self.get_fallback(model_name)
                    logger.warning(
                        f"Four-model circuit breaker OPENED for {model_name} "
                        f"due to failure rate {window_failure_rate:.1%} "
                        f"in last {config.window_seconds:.0f}s "
                        f"({len(state.window_failures)}/{state.window_total_calls} calls)"
                    )
                    return fallback

        # state is "open" — already open, no new fallback needed
        return None

    def get_state(self, model_name: str) -> str:
        """Get the current breaker state for a model.

        Also handles automatic transition from "open" to "half_open"
        if the model-specific recovery_timeout has elapsed.

        Args:
            model_name: The model to check

        Returns:
            "closed", "open", or "half_open"
        """
        state = self._get_or_create_state(model_name)
        config = self._get_config(model_name)

        # Auto-transition from open to half_open after recovery_timeout
        if state.state == "open":
            elapsed = time.time() - state.last_state_change
            if elapsed >= config.recovery_timeout:
                self._transition(model_name, "half_open")
                logger.info(
                    f"Four-model circuit breaker HALF_OPEN for {model_name} "
                    f"after {elapsed:.1f}s recovery timeout "
                    f"(configured: {config.recovery_timeout:.0f}s)"
                )

        return self._states[model_name].state

    def can_call(self, model_name: str) -> bool:
        """Check if a call can be made to the specified model.

        Args:
            model_name: The model to check

        Returns:
            True if the model is callable (closed or half_open with remaining
            test calls), False if the breaker is open
        """
        current_state = self.get_state(model_name)

        if current_state == "closed":
            return True
        elif current_state == "half_open":
            config = self._get_config(model_name)
            calls_made = self._half_open_calls.get(model_name, 0)
            return calls_made < config.half_open_max
        else:  # open
            return False

    def get_fallback(self, model_name: str) -> str | None:
        """Get the best available fallback model for the given model.

        Consults all degradation chains and skips models whose breakers
        are also open.

        Args:
            model_name: The model that needs a fallback

        Returns:
            Name of the first available fallback model, or None if
            all fallbacks are unavailable
        """
        for chain_name, chain in self._degradation_chains.items():
            if model_name not in chain:
                continue
            idx = chain.index(model_name)
            for candidate in chain[idx + 1:]:
                if self.get_state(candidate) != "open":
                    logger.debug(
                        f"Fallback for {model_name} via chain '{chain_name}': "
                        f"→ {candidate}"
                    )
                    return candidate
        return None

    def get_all_states(self) -> dict[str, str]:
        """Get the breaker state for all known models.

        Returns:
            Dict mapping model_name → state string
        """
        return {name: self.get_state(name) for name in self._states}

    def get_degradation_chains(self) -> dict[str, list[str]]:
        """Get all configured degradation chains.

        Returns:
            Dict mapping chain_name → ordered list of model names
        """
        return {k: list(v) for k, v in self._degradation_chains.items()}

    def reset(self, model_name: str | None = None) -> None:
        """Reset breaker state.

        Args:
            model_name: Specific model to reset, or None to reset all
        """
        now = time.time()
        if model_name is None:
            for name in self._states:
                self._states[name].state = "closed"
                self._states[name].failure_count = 0
                self._states[name].success_count = 0
                self._states[name].last_failure_time = 0.0
                self._states[name].last_state_change = now
                self._states[name].window_failures.clear()
                self._half_open_calls.pop(name, None)
        elif model_name in self._states:
            state = self._states[model_name]
            state.state = "closed"
            state.failure_count = 0
            state.success_count = 0
            state.last_failure_time = 0.0
            state.last_state_change = now
            state.window_failures.clear()
            self._half_open_calls.pop(model_name, None)

    # ----- Internal helpers -----

    def _transition(self, model_name: str, new_state: str) -> None:
        """Transition a model's breaker to a new state.

        Args:
            model_name: The model whose breaker state is changing
            new_state: The new state ("closed", "open", "half_open")
        """
        state = self._states[model_name]
        old_state = state.state
        state.state = new_state
        state.last_state_change = time.time()

        if new_state == "half_open":
            state.success_count = 0
            state.failure_count = 0
            self._half_open_calls[model_name] = 0
        elif new_state == "closed":
            state.success_count = 0
            state.failure_count = 0
            self._half_open_calls.pop(model_name, None)
        elif new_state == "open":
            self._half_open_calls.pop(model_name, None)

        logger.debug(
            f"Four-model breaker transition: {model_name} {old_state} → {new_state}"
        )

    def _prune_window(self, model_name: str) -> None:
        """Remove failures outside the sliding window.

        Args:
            model_name: The model whose window to prune
        """
        if model_name not in self._states or model_name not in self._configs:
            return
        state = self._states[model_name]
        config = self._configs[model_name]
        cutoff = time.time() - config.window_seconds
        old_failure_count = len(state.window_failures)
        state.window_failures = [
            (ts, msg) for ts, msg in state.window_failures if ts > cutoff
        ]
        pruned = old_failure_count - len(state.window_failures)
        if pruned > 0:
            state.window_total_calls = max(
                len(state.window_failures),
                state.window_total_calls - pruned,
            )

    def _get_failure_rate(self, model_name: str) -> float:
        """Calculate the failure rate within the sliding window.

        Args:
            model_name: The model to calculate rate for

        Returns:
            Failure rate as a float between 0.0 and 1.0
        """
        state = self._states[model_name]
        if state.window_total_calls == 0:
            return 0.0
        return len(state.window_failures) / state.window_total_calls

    def _get_or_create_state(self, model_name: str) -> ModelBreakerState:
        """Get existing state or create default state for a model.

        Args:
            model_name: The model name

        Returns:
            ModelBreakerState for the model
        """
        if model_name not in self._states:
            now = time.time()
            self._states[model_name] = ModelBreakerState(
                model_name=model_name,
                state="closed",
                last_state_change=now,
                window_total_calls=0,
            )
        return self._states[model_name]

    def _get_config(self, model_name: str) -> ModelCircuitBreakerConfig:
        """Get config for a model, falling back to defaults.

        Args:
            model_name: The model name

        Returns:
            ModelCircuitBreakerConfig for the model

        Raises:
            KeyError: If the model has no config and no default available
        """
        if model_name in self._configs:
            return self._configs[model_name]
        # Try to find a default from FOUR_MODEL_BREAKER_CONFIGS
        if model_name in FOUR_MODEL_BREAKER_CONFIGS:
            config = FOUR_MODEL_BREAKER_CONFIGS[model_name]
            self._configs[model_name] = config
            return config
        # Last resort: create a generic config
        config = ModelCircuitBreakerConfig(model_name=model_name)
        self._configs[model_name] = config
        return config
