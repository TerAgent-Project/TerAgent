"""teragent.reliability.budget — StepBudget + CrossModelCostTracker

Part of the teragent library.

Components:
    - StepBudget: Session-level step budget (prevents infinite tool loops)
    - CrossModelCostTracker: Multi-model cost tracking with monthly budget control
      and per-model/intent/date cost reports
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# Module-level default — replaces the former MAX_TOOL_STEPS import.
# The value 30 matches the original AgentLoopConfig.max_tool_steps default.
DEFAULT_MAX_STEPS: int = 30

logger = logging.getLogger(__name__)


@dataclass
class StepBudget:
    """会话级步数预算

    追踪 AgentLoop 的工具调用步数，防止无限循环。
    当步数耗尽时暂停，等待用户确认后可追加步数。
    """

    max_steps: int = DEFAULT_MAX_STEPS
    current_steps: int = 0
    _paused: bool = False
    _initial_max_steps: int = 0

    def __post_init__(self):
        self._initial_max_steps = self.max_steps

    def consume(self) -> bool:
        """消耗一步预算

        Returns:
            True 表示仍有预算，False 表示耗尽
        """
        if self._paused:
            return False
        if self.current_steps >= self.max_steps:
            self._paused = True
            logger.warning(
                f"AgentLoop step budget exhausted: {self.current_steps}/{self.max_steps}"
            )
            return False
        self.current_steps += 1
        return True

    def resume(self, extra_steps: int = 10) -> None:
        """用户确认后，追加额外步数"""
        self.max_steps += extra_steps
        self._paused = False
        logger.info(
            f"AgentLoop step budget resumed: {self.current_steps}/{self.max_steps} "
            f"(+{extra_steps} extra)"
        )

    @property
    def exhausted(self) -> bool:
        return self._paused

    @property
    def remaining(self) -> int:
        return max(0, self.max_steps - self.current_steps)

    def reset(self) -> None:
        """重置步数预算"""
        self.current_steps = 0
        self._paused = False
        self.max_steps = self._initial_max_steps


# ===== Cross-Model Cost Tracker (P3-3) =====


@dataclass
class CostRecord:
    """A single cost record for cross-model tracking

    Attributes:
        timestamp: Epoch seconds when the cost was recorded
        driver_name: Full driver name (e.g., "openai_compatible.deepseek_v4_pro")
        compiler: Compiler name (e.g., "deepseek_v4")
        model: Model name (e.g., "deepseek-v4-pro")
        intent: Task intent (e.g., "design", "execute")
        prompt_tokens: Prompt tokens consumed
        completion_tokens: Completion tokens generated
        cache_hit_tokens: Tokens served from cache (V4 specific)
        cache_miss_tokens: Tokens NOT served from cache
        cost_cny: Calculated cost in CNY
        cost_saved_cny: Cost saved by cache hits (difference between cache_miss and cache_hit pricing)
        success: Whether the call succeeded
        latency_ms: Request latency in milliseconds
    """

    timestamp: float = 0.0
    driver_name: str = ""
    compiler: str = ""
    model: str = ""
    intent: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    cost_cny: float = 0.0
    cost_saved_cny: float = 0.0
    success: bool = True
    latency_ms: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    @property
    def date_str(self) -> str:
        """Date string in YYYY-MM-DD format for date-dimension reporting"""
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class MonthlyBudgetConfig:
    """Monthly budget configuration

    Attributes:
        limit_cny: Monthly budget limit in CNY (0 = no limit)
        warning_threshold: Fraction at which to emit warning (0.8 = 80%)
        critical_threshold: Fraction at which to auto-downgrade (1.0 = at 100%)
        auto_downgrade_driver: Driver to downgrade to when budget exhausted
            (default: V4-Flash for maximum cost savings)
        notify_on_warning: Whether to log/emitting events on budget warning
    """

    limit_cny: float = 0.0
    warning_threshold: float = 0.8
    critical_threshold: float = 1.0
    auto_downgrade_driver: str = "openai_compatible.deepseek_v4_flash"
    notify_on_warning: bool = True


class CrossModelCostTracker:
    """Cross-model cost tracker with monthly budget control and reporting

    Tracks costs across V4/M3/GLM-5 models with:
      - Per-model cost statistics
      - Per-intent cost breakdown
      - Per-date cost aggregation
      - Monthly budget control (warning, critical, auto-downgrade)
      - Cache savings tracking
      - Cost report generation

    Usage::

        tracker = CrossModelCostTracker()
        tracker.set_monthly_budget(MonthlyBudgetConfig(limit_cny=500.0))

        # Record a cost
        tracker.record(CostRecord(
            driver_name="openai_compatible.deepseek_v4_pro",
            compiler="deepseek_v4",
            model="deepseek-v4-pro",
            intent="design",
            prompt_tokens=5000,
            completion_tokens=2000,
            cache_hit_tokens=3000,
            cost_cny=0.052,
            cost_saved_cny=0.012,
        ))

        # Check budget
        if tracker.is_budget_warning:
            print("Budget warning!")

        # Generate report
        report = tracker.generate_report()
    """

    def __init__(self, budget_config: MonthlyBudgetConfig | None = None) -> None:
        self._records: list[CostRecord] = []
        self._lock = threading.Lock()
        self._budget_config = budget_config or MonthlyBudgetConfig()

        # Budget state
        self._total_cost_cny: float = 0.0
        self._total_saved_cny: float = 0.0

        # Budget event tracking (avoid duplicate warnings)
        self._warning_emitted: bool = False
        self._critical_emitted: bool = False

    # ===== Core recording =====

    def record(self, record: CostRecord) -> dict[str, Any]:
        """Record a cost entry and check budget

        Args:
            record: CostRecord with cost details

        Returns:
            Dict with budget status: {
                "level": "ok" | "warning" | "critical" | "exhausted",
                "utilization": float,
                "message": str,
                "auto_downgrade": bool,
                "downgrade_driver": str,
            }
        """
        with self._lock:
            self._records.append(record)
            self._total_cost_cny += record.cost_cny
            self._total_saved_cny += record.cost_saved_cny

        budget_status = self.check_budget()

        # Emit budget events
        if budget_status["level"] == "warning" and not self._warning_emitted:
            self._warning_emitted = True
            logger.warning(
                f"Monthly budget warning: ¥{self._total_cost_cny:.2f} / "
                f"¥{self._budget_config.limit_cny:.2f} "
                f"({budget_status['utilization']:.1%})"
            )
        elif budget_status["level"] in ("critical", "exhausted") and not self._critical_emitted:
            self._critical_emitted = True
            logger.error(
                f"Monthly budget critical: ¥{self._total_cost_cny:.2f} / "
                f"¥{self._budget_config.limit_cny:.2f} "
                f"({budget_status['utilization']:.1%}). "
                f"Auto-downgrade to {self._budget_config.auto_downgrade_driver}"
            )

        return budget_status

    def record_from_tap_response(
        self,
        driver_name: str,
        compiler: str,
        model: str,
        intent: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_hit_tokens: int = 0,
        latency_ms: float = 0.0,
        success: bool = True,
        pricing: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Record a cost from TAP response data

        Convenience method that calculates cost from pricing data.

        Args:
            driver_name: Full driver name
            compiler: Compiler name
            model: Model name
            intent: Task intent
            prompt_tokens: Prompt tokens consumed
            completion_tokens: Completion tokens generated
            cache_hit_tokens: Tokens served from cache
            latency_ms: Request latency
            success: Whether the call succeeded
            pricing: Pricing dict with prompt_per_million, completion_per_million,
                     cache_hit_per_million, cache_miss_per_million

        Returns:
            Budget status dict
        """
        pricing = pricing or {}
        cache_miss_tokens = max(0, prompt_tokens - cache_hit_tokens)

        # Calculate cost
        cost_cny = 0.0
        cost_saved_cny = 0.0

        if pricing:
            # Calculate prompt cost with cache-aware pricing if available
            if "cache_hit_per_million" in pricing and "cache_miss_per_million" in pricing:
                prompt_cost = (
                    cache_hit_tokens * pricing["cache_hit_per_million"] / 1_000_000
                    + cache_miss_tokens * pricing["cache_miss_per_million"] / 1_000_000
                )
                # Calculate what it would have cost without cache
                full_prompt_cost = prompt_tokens * pricing.get("prompt_per_million", pricing["cache_miss_per_million"]) / 1_000_000
                cost_saved_cny = max(0.0, full_prompt_cost - prompt_cost)
            else:
                prompt_cost = prompt_tokens * pricing.get("prompt_per_million", 0.0) / 1_000_000

            completion_cost = completion_tokens * pricing.get("completion_per_million", 0.0) / 1_000_000
            cost_cny = prompt_cost + completion_cost

        record = CostRecord(
            driver_name=driver_name,
            compiler=compiler,
            model=model,
            intent=intent,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_hit_tokens=cache_hit_tokens,
            cache_miss_tokens=cache_miss_tokens,
            cost_cny=cost_cny,
            cost_saved_cny=cost_saved_cny,
            success=success,
            latency_ms=latency_ms,
        )

        return self.record(record)

    # ===== Budget management =====

    def check_budget(self) -> dict[str, Any]:
        """Check current budget status

        Returns:
            Dict with budget status information
        """
        if self._budget_config.limit_cny <= 0:
            return {
                "level": "ok",
                "utilization": 0.0,
                "message": "No monthly budget configured",
                "auto_downgrade": False,
                "downgrade_driver": "",
            }

        utilization = self._total_cost_cny / self._budget_config.limit_cny

        if utilization >= self._budget_config.critical_threshold:
            level = "exhausted" if utilization >= 1.0 else "critical"
            return {
                "level": level,
                "utilization": utilization,
                "message": (
                    f"Monthly budget {'exhausted' if utilization >= 1.0 else 'critical'}: "
                    f"¥{self._total_cost_cny:.2f} / ¥{self._budget_config.limit_cny:.2f} "
                    f"({utilization:.1%})"
                ),
                "auto_downgrade": True,
                "downgrade_driver": self._budget_config.auto_downgrade_driver,
            }
        elif utilization >= self._budget_config.warning_threshold:
            return {
                "level": "warning",
                "utilization": utilization,
                "message": (
                    f"Monthly budget warning: "
                    f"¥{self._total_cost_cny:.2f} / ¥{self._budget_config.limit_cny:.2f} "
                    f"({utilization:.1%})"
                ),
                "auto_downgrade": False,
                "downgrade_driver": "",
            }
        else:
            return {
                "level": "ok",
                "utilization": utilization,
                "message": f"Budget ok: ¥{self._total_cost_cny:.2f} / ¥{self._budget_config.limit_cny:.2f}",
                "auto_downgrade": False,
                "downgrade_driver": "",
            }

    def set_monthly_budget(self, config: MonthlyBudgetConfig) -> None:
        """Configure monthly budget

        Args:
            config: MonthlyBudgetConfig with limit and thresholds
        """
        self._budget_config = config
        logger.info(
            f"Monthly budget set: ¥{config.limit_cny:.2f} "
            f"(warning at {config.warning_threshold:.0%}, "
            f"critical at {config.critical_threshold:.0%}, "
            f"auto_downgrade={config.auto_downgrade_driver})"
        )

    @property
    def is_budget_warning(self) -> bool:
        """Whether the budget is in warning state"""
        status = self.check_budget()
        return status["level"] == "warning"

    @property
    def is_budget_exhausted(self) -> bool:
        """Whether the budget is exhausted"""
        status = self.check_budget()
        return status["level"] in ("critical", "exhausted")

    @property
    def should_auto_downgrade(self) -> bool:
        """Whether to auto-downgrade to a cheaper model"""
        status = self.check_budget()
        return status.get("auto_downgrade", False)

    # ===== Report generation =====

    def generate_report(
        self,
        group_by: str = "model",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Generate a cost report grouped by the specified dimension

        Args:
            group_by: Grouping dimension — "model", "intent", "date", or "driver"
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)

        Returns:
            Cost report dict with grouped statistics
        """
        with self._lock:
            records = list(self._records)

        # Apply date filters
        if start_date:
            records = [r for r in records if r.date_str >= start_date]
        if end_date:
            records = [r for r in records if r.date_str <= end_date]

        # Group records
        groups: dict[str, list[CostRecord]] = defaultdict(list)
        for r in records:
            if group_by == "model":
                key = r.driver_name
            elif group_by == "intent":
                key = r.intent
            elif group_by == "date":
                key = r.date_str
            elif group_by == "driver":
                key = r.driver_name
            else:
                key = r.driver_name
            groups[key].append(r)

        # Calculate per-group stats
        grouped_stats: dict[str, dict[str, Any]] = {}
        for key, group_records in groups.items():
            total_cost = sum(r.cost_cny for r in group_records)
            total_saved = sum(r.cost_saved_cny for r in group_records)
            total_prompt = sum(r.prompt_tokens for r in group_records)
            total_completion = sum(r.completion_tokens for r in group_records)
            total_cache_hit = sum(r.cache_hit_tokens for r in group_records)
            total_cache_miss = sum(r.cache_miss_tokens for r in group_records)
            successful_calls = sum(1 for r in group_records if r.success)
            failed_calls = sum(1 for r in group_records if not r.success)

            grouped_stats[key] = {
                "total_cost_cny": round(total_cost, 4),
                "total_saved_cny": round(total_saved, 4),
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_cache_hit_tokens": total_cache_hit,
                "total_cache_miss_tokens": total_cache_miss,
                "cache_hit_rate": (
                    round(total_cache_hit / (total_cache_hit + total_cache_miss), 4)
                    if (total_cache_hit + total_cache_miss) > 0
                    else 0.0
                ),
                "successful_calls": successful_calls,
                "failed_calls": failed_calls,
                "total_calls": successful_calls + failed_calls,
                "avg_latency_ms": (
                    round(sum(r.latency_ms for r in group_records) / len(group_records), 1)
                    if group_records
                    else 0.0
                ),
            }

        # Sort groups by cost (descending)
        sorted_groups = dict(
            sorted(grouped_stats.items(), key=lambda x: x[1]["total_cost_cny"], reverse=True)
        )

        # Calculate totals
        total_cost = sum(r.cost_cny for r in records)
        total_saved = sum(r.cost_saved_cny for r in records)
        total_prompt = sum(r.prompt_tokens for r in records)
        total_completion = sum(r.completion_tokens for r in records)
        total_cache_hit = sum(r.cache_hit_tokens for r in records)
        total_cache_miss = sum(r.cache_miss_tokens for r in records)

        budget_status = self.check_budget()

        return {
            "report_type": f"cost_by_{group_by}",
            "period": {
                "start": start_date or "all",
                "end": end_date or "all",
            },
            "summary": {
                "total_cost_cny": round(total_cost, 4),
                "total_saved_cny": round(total_saved, 4),
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
                "total_cache_hit_tokens": total_cache_hit,
                "total_cache_miss_tokens": total_cache_miss,
                "overall_cache_hit_rate": (
                    round(total_cache_hit / (total_cache_hit + total_cache_miss), 4)
                    if (total_cache_hit + total_cache_miss) > 0
                    else 0.0
                ),
                "total_calls": len(records),
                "successful_calls": sum(1 for r in records if r.success),
                "failed_calls": sum(1 for r in records if not r.success),
            },
            "budget": budget_status,
            "groups": sorted_groups,
        }

    # ===== Per-model statistics =====

    def get_model_stats(self, driver_name: str) -> dict[str, Any]:
        """Get cost statistics for a specific model

        Args:
            driver_name: Full driver name

        Returns:
            Dict with model-specific cost statistics
        """
        with self._lock:
            records = [r for r in self._records if r.driver_name == driver_name]

        if not records:
            return {
                "driver_name": driver_name,
                "total_cost_cny": 0.0,
                "total_calls": 0,
            }

        return {
            "driver_name": driver_name,
            "total_cost_cny": round(sum(r.cost_cny for r in records), 4),
            "total_saved_cny": round(sum(r.cost_saved_cny for r in records), 4),
            "total_prompt_tokens": sum(r.prompt_tokens for r in records),
            "total_completion_tokens": sum(r.completion_tokens for r in records),
            "total_cache_hit_tokens": sum(r.cache_hit_tokens for r in records),
            "cache_hit_rate": (
                round(
                    sum(r.cache_hit_tokens for r in records)
                    / max(1, sum(r.prompt_tokens for r in records)),
                    4,
                )
            ),
            "total_calls": len(records),
            "successful_calls": sum(1 for r in records if r.success),
            "avg_latency_ms": round(
                sum(r.latency_ms for r in records) / len(records), 1
            ),
        }

    def get_all_model_stats(self) -> dict[str, dict[str, Any]]:
        """Get cost statistics for all models

        Returns:
            Dict mapping driver_name → model stats
        """
        with self._lock:
            driver_names = set(r.driver_name for r in self._records)

        return {name: self.get_model_stats(name) for name in sorted(driver_names)}

    # ===== Cache savings tracking =====

    def get_cache_savings(self) -> dict[str, Any]:
        """Get cache savings statistics

        Returns:
            Dict with cache hit/miss/savings data
        """
        with self._lock:
            records = list(self._records)

        total_cache_hit = sum(r.cache_hit_tokens for r in records)
        total_cache_miss = sum(r.cache_miss_tokens for r in records)
        total_saved = sum(r.cost_saved_cny for r in records)

        # Per-model cache stats
        per_model: dict[str, dict[str, Any]] = {}
        model_groups: dict[str, list[CostRecord]] = defaultdict(list)
        for r in records:
            model_groups[r.driver_name].append(r)

        for driver_name, group in model_groups.items():
            hit = sum(r.cache_hit_tokens for r in group)
            miss = sum(r.cache_miss_tokens for r in group)
            saved = sum(r.cost_saved_cny for r in group)
            if hit + miss > 0:
                per_model[driver_name] = {
                    "cache_hit_tokens": hit,
                    "cache_miss_tokens": miss,
                    "cache_hit_rate": round(hit / (hit + miss), 4),
                    "cost_saved_cny": round(saved, 4),
                }

        return {
            "total_cache_hit_tokens": total_cache_hit,
            "total_cache_miss_tokens": total_cache_miss,
            "overall_cache_hit_rate": (
                round(total_cache_hit / (total_cache_hit + total_cache_miss), 4)
                if (total_cache_hit + total_cache_miss) > 0
                else 0.0
            ),
            "total_cost_saved_cny": round(total_saved, 4),
            "per_model": per_model,
        }

    # ===== Summary =====

    def get_total_cost(self) -> float:
        """Get total cost across all models"""
        with self._lock:
            return round(self._total_cost_cny, 4)

    def get_total_saved(self) -> float:
        """Get total cost saved by cache hits"""
        with self._lock:
            return round(self._total_saved_cny, 4)

    @property
    def total_records(self) -> int:
        """Number of cost records"""
        with self._lock:
            return len(self._records)

    def reset(self) -> None:
        """Reset all tracking state"""
        with self._lock:
            self._records.clear()
            self._total_cost_cny = 0.0
            self._total_saved_cny = 0.0
            self._warning_emitted = False
            self._critical_emitted = False

    def __repr__(self) -> str:
        return (
            f"CrossModelCostTracker("
            f"records={len(self._records)}, "
            f"total=¥{self._total_cost_cny:.2f}, "
            f"saved=¥{self._total_saved_cny:.2f}, "
            f"budget=¥{self._budget_config.limit_cny:.2f})"
        )
