"""teragent.config.circuit_breaker_config — Circuit breaker typed configuration (Phase 5)

Replaces raw dict.get() access to [circuit_breaker] section in agent.toml.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BudgetConfig:
    """Cost budget configuration — generous defaults, advisory-first.

    Maps to [circuit_breaker.budget] in agent.toml.
    """
    max_session_tokens: int = 10_000_000
    warning_threshold: float = 0.7
    critical_threshold: float = 0.9
    cost_per_million_input: float = 0.0
    cost_per_million_output: float = 0.0
    enable_hard_limit: bool = False
    auto_downgrade_model: str = ""


@dataclass(frozen=True)
class FailureBreakerConfig:
    """Consecutive failure circuit breaker configuration.

    Maps to [circuit_breaker.failure_breaker] in agent.toml.
    """
    max_consecutive: int = 5
    window_seconds: float = 300.0


@dataclass(frozen=True)
class LatencyBreakerConfig:
    """Latency circuit breaker configuration.

    Maps to [circuit_breaker.latency_breaker] in agent.toml.
    """
    warn_latency_ms: float = 30_000.0
    avg_window: int = 10


@dataclass(frozen=True)
class ProgressDetectorConfig:
    """Progress stall detector configuration.

    Maps to [circuit_breaker.progress_detector] in agent.toml.
    """
    stall_threshold: int = 10


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Top-level circuit breaker configuration.

    Maps to [circuit_breaker] section in agent.toml.
    Combines all 4 sub-breaker configurations.
    """
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    failure_breaker: FailureBreakerConfig = field(default_factory=FailureBreakerConfig)
    latency_breaker: LatencyBreakerConfig = field(default_factory=LatencyBreakerConfig)
    progress_detector: ProgressDetectorConfig = field(default_factory=ProgressDetectorConfig)

    @classmethod
    def from_dict(cls, data: dict) -> CircuitBreakerConfig:
        """Create CircuitBreakerConfig from a raw TOML dict.

        Args:
            data: The [circuit_breaker] section dict from agent.toml

        Returns:
            Typed CircuitBreakerConfig instance
        """
        budget_data = data.get("budget", {})
        failure_data = data.get("failure_breaker", {})
        latency_data = data.get("latency_breaker", {})
        progress_data = data.get("progress_detector", {})

        return cls(
            budget=BudgetConfig(
                max_session_tokens=budget_data.get("max_session_tokens", 10_000_000),
                warning_threshold=budget_data.get("warning_threshold", 0.7),
                critical_threshold=budget_data.get("critical_threshold", 0.9),
                cost_per_million_input=budget_data.get("cost_per_million_input", 0.0),
                cost_per_million_output=budget_data.get("cost_per_million_output", 0.0),
                enable_hard_limit=budget_data.get("enable_hard_limit", False),
                auto_downgrade_model=budget_data.get("auto_downgrade_model", ""),
            ),
            failure_breaker=FailureBreakerConfig(
                max_consecutive=failure_data.get("max_consecutive", 5),
                window_seconds=failure_data.get("window_seconds", 300.0),
            ),
            latency_breaker=LatencyBreakerConfig(
                warn_latency_ms=latency_data.get("warn_latency_ms", 30_000.0),
                avg_window=latency_data.get("avg_window", 10),
            ),
            progress_detector=ProgressDetectorConfig(
                stall_threshold=progress_data.get("stall_threshold", 10),
            ),
        )
