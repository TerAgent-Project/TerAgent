"""teragent.reliability.recovery — Error recovery and model degradation

Provides reusable recovery strategies for model API calls:

1. **Output token truncation recovery**: When a model response is truncated
   (finish_reason="length"), provide the decision logic for whether to continue.
   The actual continuation request is sent by AgentLoop._tool_loop().

2. **Context overflow recovery**: When the input context exceeds the model's
   token limit, automatically compact the context and retry.

3. **Model degradation/fallback**: When the primary model fails, automatically
   fall back to a configured backup model.

4. **Recovery statistics**: Track all recovery events for monitoring and
   debugging.

These patterns are provided as library-level utilities so that any caller
can use them independently.

Design doc reference: §2.1 — teragent/reliability/recovery.py
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recovery type enum
# ---------------------------------------------------------------------------

class RecoveryType(str, Enum):
    """Types of recovery actions."""

    LENGTH = "length"                # Output token truncation
    CONTEXT_OVERFLOW = "context_overflow"  # Input context too long
    FALLBACK = "fallback"            # Model degradation to fallback
    STREAMING_RETRY = "streaming_retry"  # Streaming call retry
    TOOL_REPAIR = "tool_repair"      # Tool execution repair retry


# ---------------------------------------------------------------------------
# Recovery statistics
# ---------------------------------------------------------------------------

@dataclass
class RecoveryStats:
    """Statistics about recovery events.

    Tracks counts of each recovery type and timing information.
    Thread-safe via simple attribute assignment (no concurrent mutation
    expected in typical single-agent usage).
    """

    length_recoveries: int = 0
    overflow_recoveries: int = 0
    fallback_uses: int = 0
    streaming_retries: int = 0
    tool_repairs: int = 0
    last_recovery_type: str | None = None
    last_recovery_time: float | None = None
    fallback_model: str = ""

    def record(self, recovery_type: RecoveryType, fallback_model: str = "") -> None:
        """Record a recovery event.

        Args:
            recovery_type: The type of recovery that occurred
            fallback_model: Name of the fallback model (if applicable)
        """
        if recovery_type == RecoveryType.LENGTH:
            self.length_recoveries += 1
        elif recovery_type == RecoveryType.CONTEXT_OVERFLOW:
            self.overflow_recoveries += 1
        elif recovery_type == RecoveryType.FALLBACK:
            self.fallback_uses += 1
        elif recovery_type == RecoveryType.STREAMING_RETRY:
            self.streaming_retries += 1
        elif recovery_type == RecoveryType.TOOL_REPAIR:
            self.tool_repairs += 1

        self.last_recovery_type = recovery_type.value
        self.last_recovery_time = time.time()

        if fallback_model:
            self.fallback_model = fallback_model

    def to_dict(self) -> dict[str, Any]:
        """Export stats as a plain dict."""
        return {
            "length_recoveries": self.length_recoveries,
            "overflow_recoveries": self.overflow_recoveries,
            "fallback_uses": self.fallback_uses,
            "streaming_retries": self.streaming_retries,
            "tool_repairs": self.tool_repairs,
            "last_recovery_type": self.last_recovery_type,
            "last_recovery_time": self.last_recovery_time,
            "fallback_model": self.fallback_model,
        }


# ---------------------------------------------------------------------------
# Error classification helpers
# ---------------------------------------------------------------------------

# Patterns that indicate context overflow / token limit errors
_CONTEXT_OVERFLOW_PATTERNS: frozenset[str] = frozenset({
    "context_length_exceeded",
    "maximum context length",
    "prompt is too long",
    "token exceeds",
    "上下文长度",
    "413",
    "context length",
    "token limit",
    "max_tokens",
    "request too large",
    "too many tokens",
    "input is too long",
})


def is_context_overflow_error(error: Exception | str) -> bool:
    """Check if an error indicates context overflow / token limit.

    Args:
        error: Exception object or error message string

    Returns:
        True if the error is a context overflow error

    Examples:
        >>> is_context_overflow_error("context_length_exceeded")
        True
        >>> is_context_overflow_error(ValueError("request too large for model"))
        True
        >>> is_context_overflow_error("connection timeout")
        False
    """
    error_str = str(error).lower()
    return any(pattern in error_str for pattern in _CONTEXT_OVERFLOW_PATTERNS)


# Patterns that indicate rate limiting or temporary API errors
_RETRYABLE_PATTERNS: frozenset[str] = frozenset({
    "rate_limit",
    "rate limit",
    "429",
    "503",
    "server error",
    "internal server error",
    "service unavailable",
    "temporarily unavailable",
    "too many requests",
    "capacity",
})


def is_retryable_error(error: Exception | str) -> bool:
    """Check if an error is transient and worth retrying.

    Args:
        error: Exception object or error message string

    Returns:
        True if the error is likely transient

    Examples:
        >>> is_retryable_error("rate_limit exceeded")
        True
        >>> is_retryable_error(ValueError("429 too many requests"))
        True
        >>> is_retryable_error("invalid api key")
        False
    """
    error_str = str(error).lower()
    return any(pattern in error_str for pattern in _RETRYABLE_PATTERNS)


# ---------------------------------------------------------------------------
# Recovery configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecoveryManagerConfig:
    """Configuration for RecoveryManager behavior.

    Extends the basic RecoveryConfig from teragent.config with additional
    fields for streaming retry and fallback control.

    Attributes:
        max_output_tokens_recovery: Max retry attempts for truncated output
        max_context_overflow_recovery: Max retry attempts after context compaction
        max_streaming_retries: Max retry attempts for streaming failures
        enable_fallback: Whether to enable model degradation
    """

    max_output_tokens_recovery: int = 3
    max_context_overflow_recovery: int = 2
    max_streaming_retries: int = 2
    enable_fallback: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> RecoveryManagerConfig:
        """Create RecoveryManagerConfig from a raw TOML dict.

        Args:
            data: The [recovery] section dict from agent.toml

        Returns:
            Typed RecoveryManagerConfig instance
        """
        return cls(
            max_output_tokens_recovery=data.get("max_output_tokens_recovery", 3),
            max_context_overflow_recovery=data.get("max_context_overflow_recovery", 2),
            max_streaming_retries=data.get("max_streaming_retries", 2),
            enable_fallback=data.get("enable_fallback", True),
        )


# ---------------------------------------------------------------------------
# Recovery manager
# ---------------------------------------------------------------------------

class RecoveryManager:
    """Manages error recovery strategies for model API calls.

    This class provides a unified interface for all recovery patterns:
    - Output truncation continuation
    - Context overflow compaction + retry
    - Model fallback/degradation
    - Streaming retry

    Usage::

        from teragent.reliability.recovery import RecoveryManager, RecoveryManagerConfig

        config = RecoveryManagerConfig(max_output_tokens_recovery=3)
        manager = RecoveryManager(config=config)

        # Check if an error is recoverable
        if manager.is_context_overflow(error):
            # Compact context and retry
            ...

        # Get recovery statistics
        stats = manager.get_stats()
    """

    def __init__(
        self,
        config: RecoveryManagerConfig | None = None,
        fallback_provider: Any | None = None,
    ) -> None:
        """Initialize RecoveryManager.

        Args:
            config: Recovery configuration (uses defaults if None)
            fallback_provider: Optional fallback ModelProvider for degradation
        """
        self.config = config or RecoveryManagerConfig()
        self._fallback_provider = fallback_provider
        self._stats = RecoveryStats()

    @property
    def fallback_provider(self) -> Any | None:
        """The fallback model provider (if configured)."""
        return self._fallback_provider

    @fallback_provider.setter
    def fallback_provider(self, provider: Any) -> None:
        """Set the fallback provider."""
        self._fallback_provider = provider

    @property
    def has_fallback(self) -> bool:
        """Whether a fallback model is configured."""
        return self._fallback_provider is not None

    def is_context_overflow(self, error: Exception | str) -> bool:
        """Check if an error indicates context overflow."""
        return is_context_overflow_error(error)

    def is_retryable(self, error: Exception | str) -> bool:
        """Check if an error is transient and worth retrying."""
        return is_retryable_error(error)

    def should_continue_after_truncation(
        self,
        finish_reason: str | None,
        attempt: int,
    ) -> bool:
        """Check if output truncation recovery should continue.

        Args:
            finish_reason: The model's finish_reason field
            attempt: Current attempt number (0-indexed)

        Returns:
            True if a continuation request should be sent
        """
        return (
            finish_reason == "length"
            and attempt < self.config.max_output_tokens_recovery
        )

    def should_retry_context_overflow(self, attempt: int) -> bool:
        """Check if context overflow recovery should continue.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            True if context compaction + retry should be attempted
        """
        return attempt < self.config.max_context_overflow_recovery

    def should_retry_streaming(self, attempt: int) -> bool:
        """Check if streaming retry should continue.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            True if streaming should be retried
        """
        return attempt < self.config.max_streaming_retries

    def record_recovery(self, recovery_type: RecoveryType) -> None:
        """Record a recovery event.

        Args:
            recovery_type: The type of recovery that occurred
        """
        fallback_name = ""
        if recovery_type == RecoveryType.FALLBACK and self._fallback_provider:
            fallback_name = getattr(self._fallback_provider, "model", "unknown")

        self._stats.record(recovery_type, fallback_model=fallback_name)
        logger.info(
            f"Recovery event: {recovery_type.value}"
            + (f" → fallback to {fallback_name}" if fallback_name else "")
        )

    def get_stats(self) -> dict[str, Any]:
        """Get recovery statistics.

        Returns:
            Dict with recovery counts and timing information
        """
        stats = self._stats.to_dict()
        stats["max_output_tokens_recovery"] = self.config.max_output_tokens_recovery
        stats["max_context_overflow_recovery"] = self.config.max_context_overflow_recovery
        stats["max_streaming_retries"] = self.config.max_streaming_retries
        stats["has_fallback"] = self.has_fallback
        if self.has_fallback and not stats["fallback_model"]:
            stats["fallback_model"] = getattr(
                self._fallback_provider, "model", "unknown"
            )
        return stats

    def reset_stats(self) -> None:
        """Reset all recovery statistics."""
        self._stats = RecoveryStats()


# ---------------------------------------------------------------------------
# Cross-Model Degradation Chain (P4-3)
# ---------------------------------------------------------------------------

# Default degradation chains for the three-model architecture
_DEFAULT_DEGRADATION_CHAINS: dict[str, list[str]] = {
    "heavy": ["deepseek_v4_pro", "glm_5", "deepseek_v4_flash"],
    "multimodal": ["minimax_m3", "deepseek_v4_pro"],
    "default": ["deepseek_v4_pro", "glm_5", "deepseek_v4_flash"],
}


class DegradationChain:
    """Ordered model fallback chain with health awareness.

    Default chains:
      - Heavy tasks: V4-Pro → GLM-5 → V4-Flash
      - Multimodal: M3 → V4-Pro (degrades to text-only)
      - Default: V4-Pro → GLM-5 → V4-Flash

    The chain consults ModelCircuitBreakerManager to skip unavailable models.

    Usage::

        from teragent.reliability.circuit_breaker import ModelCircuitBreakerManager
        from teragent.reliability.recovery import DegradationChain

        breaker_mgr = ModelCircuitBreakerManager()
        chain = DegradationChain(breaker_manager=breaker_mgr)

        fallback = chain.get_fallback("deepseek_v4_pro", task_type="heavy")
        full_chain = chain.get_full_chain("heavy")
    """

    def __init__(
        self,
        chains: dict[str, list[str]] | None = None,
        breaker_manager: Any | None = None,
    ) -> None:
        """Initialize DegradationChain.

        Args:
            chains: Custom degradation chains mapping task_type → ordered model list.
                If None, default chains are used.
            breaker_manager: Optional ModelCircuitBreakerManager instance for
                health-aware fallback selection.
        """
        self._chains: dict[str, list[str]] = (
            chains if chains is not None else dict(_DEFAULT_DEGRADATION_CHAINS)
        )
        self._breaker_manager = breaker_manager

    def get_fallback(
        self, current_model: str, task_type: str = "default"
    ) -> str | None:
        """Get the next available fallback model.

        Walks the degradation chain for the given task type, starting from
        the model after current_model, and returns the first model whose
        circuit breaker is not open (or the first candidate if no breaker
        manager is configured).

        Args:
            current_model: The model that needs a fallback
            task_type: The task type to select the appropriate chain

        Returns:
            Name of the first available fallback model, or None
        """
        chain = self._chains.get(task_type, self._chains.get("default", []))
        if current_model not in chain:
            # Model not in chain — try all models in order
            for candidate in chain:
                if candidate != current_model and self._is_available(candidate):
                    return candidate
            return None

        idx = chain.index(current_model)
        for candidate in chain[idx + 1:]:
            if self._is_available(candidate):
                return candidate
        return None

    def get_full_chain(self, task_type: str = "default") -> list[str]:
        """Get the full degradation chain for a task type.

        Args:
            task_type: The task type to look up

        Returns:
            Ordered list of model names for the chain
        """
        return list(self._chains.get(task_type, self._chains.get("default", [])))

    def add_chain(self, task_type: str, models: list[str]) -> None:
        """Add or replace a degradation chain for a task type.

        Args:
            task_type: The task type key
            models: Ordered list of model names (highest to lowest priority)
        """
        self._chains[task_type] = list(models)

    def _is_available(self, model_name: str) -> bool:
        """Check if a model is available via the breaker manager.

        Args:
            model_name: The model to check

        Returns:
            True if the model is available (breaker not open), or True
            if no breaker manager is configured
        """
        if self._breaker_manager is None:
            return True
        try:
            return self._breaker_manager.get_state(model_name) != "open"
        except Exception:
            return True


# ---------------------------------------------------------------------------
# Long-Horizon Task Fault Recovery (P4-3)
# ---------------------------------------------------------------------------

class LongHorizonRecoveryManager:
    """Recovery manager for long-horizon tasks.

    Handles:
    - Task interruption recovery (from latest checkpoint)
    - API reconnection with exponential backoff
    - Progress-preserving retry
    - Strategy downgrade (e.g., long-horizon → standard)

    Usage::

        from teragent.reliability.recovery import LongHorizonRecoveryManager
        from teragent.long_horizon.checkpoint import CheckpointStore

        store = CheckpointStore()
        recovery_mgr = LongHorizonRecoveryManager(checkpoint_store=store)

        # Attempt recovery
        success = await recovery_mgr.recover_from_checkpoint(task_manager)

        # Check if should downgrade
        if recovery_mgr.should_downgrade_to_standard(attempts=3, elapsed=1200):
            print("Switching to standard mode")
    """

    def __init__(
        self,
        checkpoint_store: Any | None = None,
        max_reconnection_attempts: int = 5,
        reconnection_base_delay: float = 2.0,
    ) -> None:
        """Initialize LongHorizonRecoveryManager.

        Args:
            checkpoint_store: CheckpointStore instance for loading checkpoints.
                If None, checkpoint recovery will not be available.
            max_reconnection_attempts: Maximum number of reconnection attempts
            reconnection_base_delay: Base delay in seconds for exponential backoff
        """
        self._checkpoint_store = checkpoint_store
        self._max_reconnection_attempts = max_reconnection_attempts
        self._reconnection_base_delay = reconnection_base_delay

        # Recovery tracking
        self._recovery_attempts: int = 0
        self._recovery_successes: int = 0
        self._recovery_failures: int = 0
        self._last_recovery_time: float = 0.0

        # Downgrade thresholds
        self._downgrade_max_attempts: int = 3
        self._downgrade_max_elapsed: float = 1800.0  # 30 minutes

    async def recover_from_checkpoint(self, task_manager: Any) -> bool:
        """Recover a long-horizon task from its latest checkpoint.

        Loads the most recent checkpoint for the task and restores the
        task manager's state so execution can resume from that point.

        Args:
            task_manager: LongHorizonTaskManager instance to recover

        Returns:
            True if recovery was successful, False otherwise
        """
        if self._checkpoint_store is None:
            logger.warning("No checkpoint store configured; cannot recover")
            return False

        task_id = getattr(task_manager, "task_id", None)
        if not task_id:
            logger.warning("Task manager has no task_id; cannot recover")
            return False

        try:
            checkpoint = await self._checkpoint_store.load_latest(task_id)
            if checkpoint is None:
                logger.warning(f"No checkpoint found for task {task_id}")
                return False

            # Restore task manager state from checkpoint
            task_manager.completed_sub_goals = list(checkpoint.completed_sub_goals)
            task_manager.current_sub_goal = checkpoint.current_sub_goal
            task_manager.steps_completed = checkpoint.steps_completed
            task_manager.elapsed_minutes = checkpoint.elapsed_minutes
            task_manager.strategy_switches = checkpoint.strategy_switches

            # Restore phase if the task manager supports it
            if hasattr(task_manager, "current_phase"):
                task_manager.current_phase = checkpoint.phase

            # Restore arbitrary state data
            if hasattr(task_manager, "state_data"):
                task_manager.state_data = dict(checkpoint.state_data)

            self._recovery_attempts += 1
            self._recovery_successes += 1
            self._last_recovery_time = time.time()

            logger.info(
                f"Recovered task {task_id} from checkpoint "
                f"(phase={checkpoint.phase}, steps={checkpoint.steps_completed})"
            )
            return True

        except Exception as e:
            self._recovery_attempts += 1
            self._recovery_failures += 1
            logger.error(f"Failed to recover task {task_id}: {e}")
            return False

    def should_downgrade_to_standard(
        self, recovery_attempts: int, elapsed_time: float
    ) -> bool:
        """Determine if a long-horizon task should be downgraded to standard mode.

        Downgrade is recommended when:
        - Multiple recovery attempts have failed, OR
        - The elapsed time exceeds the downgrade threshold

        Args:
            recovery_attempts: Number of recovery attempts made so far
            elapsed_time: Time elapsed since the task started (seconds)

        Returns:
            True if the task should be downgraded to standard mode
        """
        if recovery_attempts >= self._downgrade_max_attempts:
            logger.info(
                f"Recommending downgrade: {recovery_attempts} recovery attempts "
                f"(max={self._downgrade_max_attempts})"
            )
            return True

        if elapsed_time >= self._downgrade_max_elapsed:
            logger.info(
                f"Recommending downgrade: {elapsed_time:.0f}s elapsed "
                f"(max={self._downgrade_max_elapsed:.0f}s)"
            )
            return True

        return False

    def get_reconnection_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay for reconnection attempts.

        Uses the formula: base_delay * 2^attempt with jitter.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds before the next reconnection attempt
        """
        import random

        if attempt >= self._max_reconnection_attempts:
            return 0.0  # No more attempts

        delay = self._reconnection_base_delay * (2 ** attempt)
        # Add jitter (±25%) to avoid thundering herd
        jitter = delay * 0.25 * (random.random() * 2 - 1)
        return max(0.1, delay + jitter)

    def record_recovery_attempt(self, success: bool) -> None:
        """Record the outcome of a recovery attempt.

        Args:
            success: Whether the recovery attempt succeeded
        """
        self._recovery_attempts += 1
        if success:
            self._recovery_successes += 1
        else:
            self._recovery_failures += 1
        self._last_recovery_time = time.time()

    @property
    def recovery_stats(self) -> dict[str, Any]:
        """Get recovery statistics.

        Returns:
            Dict with recovery attempt counts and timing
        """
        return {
            "total_attempts": self._recovery_attempts,
            "successes": self._recovery_successes,
            "failures": self._recovery_failures,
            "last_recovery_time": self._last_recovery_time,
            "max_reconnection_attempts": self._max_reconnection_attempts,
        }

    @property
    def has_checkpoint_store(self) -> bool:
        """Whether a checkpoint store is configured."""
        return self._checkpoint_store is not None


# ---------------------------------------------------------------------------
# Unified Rate Limiting Handler (P4-3)
# ---------------------------------------------------------------------------


@dataclass
class RateLimitInfo:
    """Unified rate limit information across models.

    Normalizes the different rate limit response formats from each model
    provider into a consistent structure.

    Attributes:
        model_name: The model that returned the rate limit response
        requests_remaining: Remaining requests in current window (None if unknown)
        tokens_remaining: Remaining tokens in current window (None if unknown)
        reset_time: Unix timestamp when the rate limit window resets (None if unknown)
        retry_after: Seconds to wait before retrying (None if unknown)
    """

    model_name: str
    requests_remaining: int | None = None
    tokens_remaining: int | None = None
    reset_time: float | None = None  # Unix timestamp
    retry_after: float | None = None  # Seconds to wait


class RateLimitHandler:
    """Handles rate limiting for all three models.

    Each model has different rate limit response formats:
    - DeepSeek V4: Standard OpenAI 429 with Retry-After header
    - MiniMax M3: X-RateLimit-* headers
    - GLM-5: Custom 429 response body

    This handler normalizes all formats and provides unified backoff.

    Usage::

        from teragent.reliability.recovery import RateLimitHandler

        handler = RateLimitHandler()

        info = handler.parse_rate_limit_response(
            model_name="deepseek_v4_pro",
            status_code=429,
            headers={"Retry-After": "30"},
            body=None,
        )

        if handler.should_retry("deepseek_v4_pro", info):
            delay = handler.get_backoff_delay("deepseek_v4_pro", attempt=1, rate_limit_info=info)
    """

    # Maximum retries for rate-limited requests
    _MAX_RATE_LIMIT_RETRIES: int = 5
    # Base backoff delay in seconds
    _BASE_BACKOFF: float = 1.0
    # Maximum backoff cap in seconds
    _MAX_BACKOFF: float = 120.0

    def __init__(self, breaker_manager: Any | None = None) -> None:
        """Initialize RateLimitHandler.

        Args:
            breaker_manager: Optional ModelCircuitBreakerManager for
                circuit breaker integration.
        """
        self._breaker_manager = breaker_manager

    def parse_rate_limit_response(
        self,
        model_name: str,
        status_code: int,
        headers: dict[str, str],
        body: dict | None = None,
    ) -> RateLimitInfo:
        """Parse a rate limit response into a unified format.

        Handles the different response formats from each model provider:
        - DeepSeek V4: Standard OpenAI 429 with Retry-After header
        - MiniMax M3: X-RateLimit-* headers
        - GLM-5: Custom 429 response body with retry_after field

        Args:
            model_name: The model that returned the response
            status_code: HTTP status code
            headers: Response headers (case-insensitive keys)
            body: Response body as a dict (if available)

        Returns:
            RateLimitInfo with normalized rate limit information
        """
        info = RateLimitInfo(model_name=model_name)

        if status_code != 429:
            return info

        # Normalize headers to lowercase for case-insensitive lookup
        lower_headers = {k.lower(): v for k, v in headers.items()}

        # Parse Retry-After header (DeepSeek V4 / standard OpenAI format)
        retry_after_str = lower_headers.get("retry-after")
        if retry_after_str:
            try:
                info.retry_after = float(retry_after_str)
            except (ValueError, TypeError):
                pass

        # Parse X-RateLimit-* headers (MiniMax M3 format)
        requests_remaining_str = lower_headers.get("x-ratelimit-remaining-requests")
        if requests_remaining_str:
            try:
                info.requests_remaining = int(requests_remaining_str)
            except (ValueError, TypeError):
                pass

        tokens_remaining_str = lower_headers.get("x-ratelimit-remaining-tokens")
        if tokens_remaining_str:
            try:
                info.tokens_remaining = int(tokens_remaining_str)
            except (ValueError, TypeError):
                pass

        reset_time_str = lower_headers.get("x-ratelimit-reset")
        if reset_time_str:
            try:
                info.reset_time = float(reset_time_str)
            except (ValueError, TypeError):
                pass

        # Parse response body (GLM-5 custom format)
        if body:
            if "retry_after" in body:
                try:
                    info.retry_after = float(body["retry_after"])
                except (ValueError, TypeError):
                    pass

            if "requests_remaining" in body:
                try:
                    info.requests_remaining = int(body["requests_remaining"])
                except (ValueError, TypeError):
                    pass

            if "tokens_remaining" in body:
                try:
                    info.tokens_remaining = int(body["tokens_remaining"])
                except (ValueError, TypeError):
                    pass

            if "reset_time" in body:
                try:
                    info.reset_time = float(body["reset_time"])
                except (ValueError, TypeError):
                    pass

        return info

    def should_retry(
        self, model_name: str, rate_limit_info: RateLimitInfo
    ) -> bool:
        """Determine if a rate-limited request should be retried.

        A request should be retried if:
        - The model's circuit breaker allows calls, AND
        - There are remaining requests/tokens or a reset time is known

        Args:
            model_name: The model to check
            rate_limit_info: Parsed rate limit information

        Returns:
            True if the request should be retried
        """
        # Check circuit breaker
        if self._breaker_manager is not None:
            try:
                if not self._breaker_manager.can_call(model_name):
                    return False
            except Exception:
                pass

        # If we have remaining requests/tokens, definitely retry
        if rate_limit_info.requests_remaining is not None:
            return rate_limit_info.requests_remaining > 0

        if rate_limit_info.tokens_remaining is not None:
            return rate_limit_info.tokens_remaining > 0

        # If we have retry_after or reset_time, we can wait and retry
        if rate_limit_info.retry_after is not None:
            return rate_limit_info.retry_after <= self._MAX_BACKOFF

        if rate_limit_info.reset_time is not None:
            wait_time = rate_limit_info.reset_time - time.time()
            return 0 < wait_time <= self._MAX_BACKOFF

        # Default: allow retry (conservative approach)
        return True

    def get_backoff_delay(
        self,
        model_name: str,
        attempt: int,
        rate_limit_info: RateLimitInfo | None = None,
    ) -> float:
        """Calculate the backoff delay for a rate-limited retry.

        Uses the following priority:
        1. retry_after from the rate limit response (if available)
        2. Time until reset_time (if available)
        3. Exponential backoff with jitter

        Args:
            model_name: The model being retried
            attempt: Current attempt number (0-indexed)
            rate_limit_info: Optional parsed rate limit information

        Returns:
            Delay in seconds before the next retry
        """
        import random

        if attempt >= self._MAX_RATE_LIMIT_RETRIES:
            return 0.0  # No more retries

        # Priority 1: Use server-provided retry_after
        if rate_limit_info and rate_limit_info.retry_after is not None:
            delay = min(rate_limit_info.retry_after, self._MAX_BACKOFF)
            # Add small jitter (±10%)
            jitter = delay * 0.1 * (random.random() * 2 - 1)
            return max(0.1, delay + jitter)

        # Priority 2: Calculate wait time from reset_time
        if rate_limit_info and rate_limit_info.reset_time is not None:
            wait_time = rate_limit_info.reset_time - time.time()
            if 0 < wait_time <= self._MAX_BACKOFF:
                # Add small jitter
                jitter = wait_time * 0.1 * (random.random() * 2 - 1)
                return max(0.1, wait_time + jitter)

        # Priority 3: Exponential backoff with jitter
        delay = self._BASE_BACKOFF * (2 ** attempt)
        delay = min(delay, self._MAX_BACKOFF)
        # Add jitter (±25%)
        jitter = delay * 0.25 * (random.random() * 2 - 1)
        return max(0.1, delay + jitter)
