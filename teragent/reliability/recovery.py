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
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "CheckpointData",
    "ContextStabilityMetrics",
    "CrossModelDegradationChain",
    "DegradationChain",
    "DegradationEvent",
    "FourModelRateLimitHandler",
    "GLM52ContextStabilityRecovery",
    "LongHorizonRecovery",
    "LongHorizonRecoveryManager",
    "ModelRateLimitState",
    "PartialResult",
    "ProviderRateLimitConfig",
    "RateLimitHandler",
    "RateLimitInfo",
    "RecoveryManager",
    "RecoveryManagerConfig",
    "RecoveryStats",
    "RecoveryType",
    "is_context_overflow_error",
    "is_retryable_error",
]

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
    "413 ",       # HTTP 413 Payload Too Large (trailing space to avoid matching "41357")
    "413\n",      # HTTP 413 at end of line
    "413:",       # HTTP 413 with colon (e.g., "413: Payload Too Large")
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


# ---------------------------------------------------------------------------
# Cross-Model Degradation Chain for Four-Model Architecture (P4-3)
# ---------------------------------------------------------------------------


@dataclass
class DegradationEvent:
    """Record of a model degradation event.

    Attributes:
        timestamp: When the degradation occurred
        from_model: The model that degraded
        to_model: The fallback model chosen
        reason: Why the degradation occurred
        chain_name: Which degradation chain was used
    """

    timestamp: float
    from_model: str
    to_model: str
    reason: str
    chain_name: str


class CrossModelDegradationChain:
    """Cross-model degradation chain for the four-model architecture.

    Defines four specialized degradation chains:
      - Primary chain:    V4-Pro → GLM-5.2 → GLM-5 → V4-Flash
      - Multimodal chain: M3 → V4-Pro (text fallback)
      - Long-horizon chain: GLM-5.2 → GLM-5 → V4-Flash
      - 1M context chain:  GLM-5.2 → V4-Pro → M3

    Each chain is designed for a specific failure scenario, considering
    the capabilities of each fallback model.

    Usage::

        from teragent.reliability.recovery import CrossModelDegradationChain

        chain = CrossModelDegradationChain()
        fallback = chain.degrade("deepseek_v4_pro", "API timeout")
        full_chain = chain.get_chain("deepseek_v4_pro")
        events = chain.get_degradation_events()
    """

    # Named degradation chains for the four-model architecture
    _DEFAULT_CHAINS: dict[str, list[str]] = {
        "primary": ["deepseek_v4_pro", "glm_52", "glm_5", "deepseek_v4_flash"],
        "multimodal": ["minimax_m3", "deepseek_v4_pro"],
        "long_horizon": ["glm_52", "glm_5", "deepseek_v4_flash"],
        "1m_context": ["glm_52", "deepseek_v4_pro", "minimax_m3"],
    }

    # Reverse map: model_name → list of chain names that include it
    # (used by get_chain to find which chain a model belongs to)

    def __init__(
        self,
        chains: dict[str, list[str]] | None = None,
        breaker_manager: Any | None = None,
    ) -> None:
        """Initialize CrossModelDegradationChain.

        Args:
            chains: Custom degradation chains. If None, default four-model
                chains are used.
            breaker_manager: Optional FourModelCircuitBreakerManager or
                ModelCircuitBreakerManager for health-aware fallback.
        """
        self._chains: dict[str, list[str]] = (
            chains if chains is not None else dict(self._DEFAULT_CHAINS)
        )
        self._breaker_manager = breaker_manager

        # Degradation event log
        self._events: list[DegradationEvent] = []

        # Build reverse map: model_name → chain names
        self._model_to_chains: dict[str, list[str]] = {}
        for chain_name, chain in self._chains.items():
            for model in chain:
                if model not in self._model_to_chains:
                    self._model_to_chains[model] = []
                self._model_to_chains[model].append(chain_name)

    def degrade(self, model_name: str, reason: str = "") -> str | None:
        """Degrade from the given model to the next available fallback.

        Searches all chains that include the model and returns the first
        available fallback, skipping models whose circuit breakers are open.

        Args:
            model_name: The model that needs to be degraded
            reason: Why the degradation is needed (e.g., "API timeout", "circuit open")

        Returns:
            Name of the next available fallback model, or None if all
            fallbacks are unavailable
        """
        # Find all chains that include this model
        chain_names = self._model_to_chains.get(model_name, [])

        if not chain_names:
            # Model not in any chain — try all chains in order
            for chain_name, chain in self._chains.items():
                for candidate in chain:
                    if candidate != model_name and self._is_available(candidate):
                        self._log_degradation(
                            model_name, candidate, reason, chain_name
                        )
                        return candidate
            logger.warning(
                f"No degradation path found for {model_name}: "
                f"not in any chain and no available fallbacks"
            )
            return None

        # Try each chain that includes this model
        for chain_name in chain_names:
            chain = self._chains[chain_name]
            if model_name not in chain:
                continue
            idx = chain.index(model_name)
            for candidate in chain[idx + 1:]:
                if self._is_available(candidate):
                    self._log_degradation(
                        model_name, candidate, reason, chain_name
                    )
                    return candidate

        logger.warning(
            f"No available fallback for {model_name} in chains: {chain_names}"
        )
        return None

    def get_chain(self, model_name: str) -> list[str]:
        """Get the full degradation chain for a model.

        Returns the chain that the model belongs to (preferring the first
        chain found). If the model is in multiple chains, returns the
        first one.

        Args:
            model_name: The model to look up

        Returns:
            Ordered list of model names in the chain, or empty list if
            the model is not in any chain
        """
        chain_names = self._model_to_chains.get(model_name, [])
        if not chain_names:
            return []
        # Return the first chain that includes the model
        return list(self._chains.get(chain_names[0], []))

    def get_all_chains(self) -> dict[str, list[str]]:
        """Get all configured degradation chains.

        Returns:
            Dict mapping chain_name → ordered list of model names
        """
        return {k: list(v) for k, v in self._chains.items()}

    def get_chains_for_model(self, model_name: str) -> dict[str, list[str]]:
        """Get all chains that include a specific model.

        Args:
            model_name: The model to look up

        Returns:
            Dict mapping chain_name → ordered list of model names
        """
        result: dict[str, list[str]] = {}
        for chain_name in self._model_to_chains.get(model_name, []):
            result[chain_name] = list(self._chains[chain_name])
        return result

    def get_degradation_events(
        self, model_name: str | None = None, limit: int = 100
    ) -> list[DegradationEvent]:
        """Get recent degradation events.

        Args:
            model_name: Filter by source model (None for all)
            limit: Maximum number of events to return

        Returns:
            List of DegradationEvent instances, most recent first
        """
        events = self._events
        if model_name is not None:
            events = [e for e in events if e.from_model == model_name]
        return list(reversed(events[-limit:]))

    def add_chain(self, chain_name: str, models: list[str]) -> None:
        """Add or replace a degradation chain.

        Args:
            chain_name: Name for the chain
            models: Ordered list of model names (highest to lowest priority)
        """
        self._chains[chain_name] = list(models)
        # Rebuild reverse map
        self._model_to_chains.clear()
        for cn, chain in self._chains.items():
            for model in chain:
                if model not in self._model_to_chains:
                    self._model_to_chains[model] = []
                self._model_to_chains[model].append(cn)

    def _is_available(self, model_name: str) -> bool:
        """Check if a model is available via the breaker manager.

        Args:
            model_name: The model to check

        Returns:
            True if available, or True if no breaker manager configured
        """
        if self._breaker_manager is None:
            return True
        try:
            return self._breaker_manager.get_state(model_name) != "open"
        except Exception:
            return True

    def _log_degradation(
        self, from_model: str, to_model: str, reason: str, chain_name: str
    ) -> None:
        """Log a degradation event.

        Args:
            from_model: The model being degraded from
            to_model: The fallback model
            reason: Why the degradation occurred
            chain_name: Which chain was used
        """
        event = DegradationEvent(
            timestamp=time.time(),
            from_model=from_model,
            to_model=to_model,
            reason=reason,
            chain_name=chain_name,
        )
        self._events.append(event)

        logger.info(
            f"Model degradation: {from_model} → {to_model} "
            f"(chain: {chain_name}, reason: {reason or 'unspecified'})"
        )


# ---------------------------------------------------------------------------
# Long-Horizon Task Fault Recovery (P4-3 — Four-Model Enhanced)
# ---------------------------------------------------------------------------


@dataclass
class CheckpointData:
    """Simplified checkpoint data for long-horizon task recovery.

    Attributes:
        task_id: The task identifier
        phase: Current execution phase
        completed_steps: Number of steps completed
        total_steps: Total expected steps (0 if unknown)
        context_snapshot: Serialized context/state for resumption
        timestamp: When this checkpoint was created
        model_name: The model being used when checkpoint was created
    """

    task_id: str
    phase: str
    completed_steps: int
    total_steps: int
    context_snapshot: dict[str, Any]
    timestamp: float = 0.0
    model_name: str = ""

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class PartialResult:
    """Partial result from an interrupted long-horizon task.

    Attributes:
        task_id: The task identifier
        completed_content: Content produced before interruption
        completed_steps: Steps completed before interruption
        total_steps: Total expected steps
        interruption_reason: Why the task was interrupted
        can_resume: Whether the task can be resumed from this point
    """

    task_id: str
    completed_content: str
    completed_steps: int
    total_steps: int
    interruption_reason: str
    can_resume: bool = True


class LongHorizonRecovery:
    """Recovery handler for long-horizon tasks in the four-model architecture.

    Provides:
      - Checkpoint-based recovery (resume from latest checkpoint)
      - API reconnection with exponential backoff
      - Partial completion handling (salvage partial results)

    Unlike LongHorizonRecoveryManager (which integrates with LongHorizonTaskManager),
    this class provides standalone recovery primitives that can be used by any
    long-running task handler.

    Usage::

        from teragent.reliability.recovery import LongHorizonRecovery

        recovery = LongHorizonRecovery()
        result = recovery.recover_from_checkpoint(task_id="task_123", checkpoint=cp)
        delay = recovery.handle_api_reconnect(task_id="task_123", max_retries=3)
        partial = recovery.handle_partial_completion(task_id, partial_result)
    """

    def __init__(
        self,
        max_reconnect_retries: int = 3,
        reconnect_base_delay: float = 2.0,
        reconnect_max_delay: float = 60.0,
    ) -> None:
        """Initialize LongHorizonRecovery.

        Args:
            max_reconnect_retries: Maximum reconnection attempts
            reconnect_base_delay: Base delay in seconds for exponential backoff
            reconnect_max_delay: Maximum delay cap in seconds
        """
        self._max_reconnect_retries = max_reconnect_retries
        self._reconnect_base_delay = reconnect_base_delay
        self._reconnect_max_delay = reconnect_max_delay

        # Per-task checkpoint storage
        self._checkpoints: dict[str, list[CheckpointData]] = {}

        # Per-task reconnection tracking
        self._reconnect_attempts: dict[str, int] = {}

        # Per-task partial results
        self._partial_results: dict[str, PartialResult] = {}

        # Statistics
        self._recovery_count: int = 0
        self._reconnect_count: int = 0
        self._partial_save_count: int = 0

    def recover_from_checkpoint(
        self, task_id: str, checkpoint: CheckpointData
    ) -> dict[str, Any]:
        """Resume a task from a checkpoint.

        Restores the task's state from the provided checkpoint data
        and returns the information needed to continue execution.

        Args:
            task_id: The task to recover
            checkpoint: CheckpointData with the resumption state

        Returns:
            Dict with recovery result:
              - "recovered": bool — whether recovery succeeded
              - "task_id": str — the task identifier
              - "resume_from_step": int — step to resume from
              - "phase": str — the phase to resume in
              - "context": dict — the restored context snapshot
              - "model_name": str — model to use for resumption
              - "message": str — human-readable result description
        """
        try:
            # Store the checkpoint for this task
            if task_id not in self._checkpoints:
                self._checkpoints[task_id] = []
            self._checkpoints[task_id].append(checkpoint)

            self._recovery_count += 1

            result = {
                "recovered": True,
                "task_id": task_id,
                "resume_from_step": checkpoint.completed_steps,
                "phase": checkpoint.phase,
                "context": dict(checkpoint.context_snapshot),
                "model_name": checkpoint.model_name,
                "message": (
                    f"Recovered task {task_id} from checkpoint: "
                    f"phase={checkpoint.phase}, "
                    f"step={checkpoint.completed_steps}/{checkpoint.total_steps or '?'}"
                ),
            }

            logger.info(result["message"])
            return result

        except Exception as e:
            logger.error(f"Failed to recover task {task_id} from checkpoint: {e}")
            return {
                "recovered": False,
                "task_id": task_id,
                "resume_from_step": 0,
                "phase": "",
                "context": {},
                "model_name": "",
                "message": f"Recovery failed: {e}",
            }

    def handle_api_reconnect(
        self, task_id: str, max_retries: int = 3
    ) -> dict[str, Any]:
        """Handle API reconnection with exponential backoff.

        Tracks reconnection attempts per task and calculates the
        appropriate backoff delay.

        Args:
            task_id: The task needing reconnection
            max_retries: Maximum number of retries (overrides constructor default)

        Returns:
            Dict with reconnection information:
              - "should_retry": bool — whether to attempt reconnection
              - "delay_seconds": float — seconds to wait before retrying
              - "attempt": int — current attempt number (0-indexed)
              - "remaining_attempts": int — retries left after this one
              - "task_id": str — the task identifier
        """
        current_attempt = self._reconnect_attempts.get(task_id, 0)
        effective_max = max_retries or self._max_reconnect_retries

        if current_attempt >= effective_max:
            logger.warning(
                f"API reconnection exhausted for task {task_id}: "
                f"{current_attempt} attempts (max: {effective_max})"
            )
            return {
                "should_retry": False,
                "delay_seconds": 0.0,
                "attempt": current_attempt,
                "remaining_attempts": 0,
                "task_id": task_id,
            }

        # Calculate exponential backoff with jitter
        import random
        delay = self._reconnect_base_delay * (2 ** current_attempt)
        delay = min(delay, self._reconnect_max_delay)
        # Add jitter (±25%)
        jitter = delay * 0.25 * (random.random() * 2 - 1)
        delay = max(0.1, delay + jitter)

        # Increment attempt counter
        self._reconnect_attempts[task_id] = current_attempt + 1
        self._reconnect_count += 1

        remaining = effective_max - current_attempt - 1

        logger.info(
            f"API reconnection for task {task_id}: "
            f"attempt {current_attempt + 1}/{effective_max}, "
            f"delay={delay:.1f}s, remaining={remaining}"
        )

        return {
            "should_retry": True,
            "delay_seconds": delay,
            "attempt": current_attempt,
            "remaining_attempts": remaining,
            "task_id": task_id,
        }

    def handle_partial_completion(
        self, task_id: str, partial_result: PartialResult
    ) -> dict[str, Any]:
        """Handle partial results from an interrupted task.

        Saves the partial result and provides guidance on whether the
        task can be resumed or should be considered complete with
        partial output.

        Args:
            task_id: The task with a partial result
            partial_result: PartialResult with the incomplete output

        Returns:
            Dict with handling result:
              - "saved": bool — whether the partial result was saved
              - "task_id": str — the task identifier
              - "can_resume": bool — whether the task can be resumed
              - "completed_fraction": float — fraction of steps completed
              - "recommendation": str — "resume" | "accept_partial" | "retry_full"
              - "message": str — human-readable description
        """
        try:
            # Save the partial result
            self._partial_results[task_id] = partial_result
            self._partial_save_count += 1

            # Calculate completion fraction
            if partial_result.total_steps > 0:
                fraction = partial_result.completed_steps / partial_result.total_steps
            else:
                fraction = 0.0

            # Determine recommendation
            if fraction >= 0.9 and partial_result.can_resume:
                recommendation = "resume"  # Almost done, worth completing
            elif fraction >= 0.5 and partial_result.can_resume:
                recommendation = "resume"  # Significant progress, resume
            elif fraction >= 0.3:
                recommendation = "accept_partial"  # Some progress, may not be worth retrying
            else:
                recommendation = "retry_full"  # Little progress, start over

            result = {
                "saved": True,
                "task_id": task_id,
                "can_resume": partial_result.can_resume,
                "completed_fraction": round(fraction, 3),
                "recommendation": recommendation,
                "message": (
                    f"Partial result saved for task {task_id}: "
                    f"{partial_result.completed_steps}/{partial_result.total_steps} steps "
                    f"({fraction:.0%}), recommendation: {recommendation}"
                ),
            }

            logger.info(result["message"])
            return result

        except Exception as e:
            logger.error(f"Failed to handle partial completion for task {task_id}: {e}")
            return {
                "saved": False,
                "task_id": task_id,
                "can_resume": False,
                "completed_fraction": 0.0,
                "recommendation": "retry_full",
                "message": f"Partial result handling failed: {e}",
            }

    def save_checkpoint(self, checkpoint: CheckpointData) -> None:
        """Save a checkpoint for a task.

        Args:
            checkpoint: CheckpointData to save
        """
        if checkpoint.task_id not in self._checkpoints:
            self._checkpoints[checkpoint.task_id] = []
        self._checkpoints[checkpoint.task_id].append(checkpoint)
        logger.debug(
            f"Checkpoint saved for task {checkpoint.task_id}: "
            f"step {checkpoint.completed_steps}, phase={checkpoint.phase}"
        )

    def get_latest_checkpoint(self, task_id: str) -> CheckpointData | None:
        """Get the latest checkpoint for a task.

        Args:
            task_id: The task to look up

        Returns:
            Latest CheckpointData, or None if no checkpoints exist
        """
        checkpoints = self._checkpoints.get(task_id, [])
        return checkpoints[-1] if checkpoints else None

    def reset_reconnect(self, task_id: str) -> None:
        """Reset reconnection attempt counter for a task.

        Args:
            task_id: The task to reset
        """
        self._reconnect_attempts.pop(task_id, None)

    @property
    def stats(self) -> dict[str, Any]:
        """Get recovery statistics.

        Returns:
            Dict with counts of recoveries, reconnections, and partial saves
        """
        return {
            "checkpoint_recoveries": self._recovery_count,
            "reconnection_attempts": self._reconnect_count,
            "partial_result_saves": self._partial_save_count,
            "tasks_with_checkpoints": len(self._checkpoints),
            "tasks_with_partial_results": len(self._partial_results),
        }


# ---------------------------------------------------------------------------
# Four-Model Rate Limit Handler (P4-3 — Enhanced)
# ---------------------------------------------------------------------------


@dataclass
class ProviderRateLimitConfig:
    """Rate limit configuration for a specific provider.

    Attributes:
        provider_name: Provider identifier (e.g., "deepseek", "minimax", "glm")
        requests_per_minute: Maximum requests per minute (0 = unlimited/unknown)
        tokens_per_minute: Maximum tokens per minute (0 = unlimited/unknown)
        throttle_threshold: Fraction at which to start proactive throttling (0.8 = 80%)
        base_backoff: Base backoff delay in seconds for 429 responses
        max_backoff: Maximum backoff cap in seconds
        max_retries: Maximum retries for rate-limited requests
    """

    provider_name: str
    requests_per_minute: int = 0
    tokens_per_minute: int = 0
    throttle_threshold: float = 0.8
    base_backoff: float = 1.0
    max_backoff: float = 120.0
    max_retries: int = 5


@dataclass
class ModelRateLimitState:
    """Runtime rate limit state for a model.

    Attributes:
        model_name: Model identifier
        provider: Provider name
        requests_in_window: Requests made in the current minute window
        tokens_in_window: Tokens consumed in the current minute window
        window_start_time: Start of the current rate limit window
        last_429_time: Timestamp of the last 429 response
        total_429s: Total 429 responses received
        throttled_requests: Number of requests proactively throttled
    """

    model_name: str
    provider: str
    requests_in_window: int = 0
    tokens_in_window: int = 0
    window_start_time: float = 0.0
    last_429_time: float = 0.0
    total_429s: int = 0
    throttled_requests: int = 0

    def __post_init__(self):
        if self.window_start_time == 0.0:
            self.window_start_time = time.time()


# Default rate limit configs for each provider
_DEFAULT_PROVIDER_CONFIGS: dict[str, ProviderRateLimitConfig] = {
    "deepseek": ProviderRateLimitConfig(
        provider_name="deepseek",
        requests_per_minute=60,
        tokens_per_minute=1_000_000,
        throttle_threshold=0.8,
        base_backoff=1.0,
        max_backoff=60.0,
        max_retries=5,
    ),
    "minimax": ProviderRateLimitConfig(
        provider_name="minimax",
        requests_per_minute=30,
        tokens_per_minute=500_000,
        throttle_threshold=0.8,
        base_backoff=2.0,
        max_backoff=90.0,
        max_retries=3,
    ),
    "glm": ProviderRateLimitConfig(
        provider_name="glm",
        requests_per_minute=60,
        tokens_per_minute=800_000,
        throttle_threshold=0.8,
        base_backoff=1.5,
        max_backoff=120.0,
        max_retries=5,
    ),
}

# Model name → provider mapping
_MODEL_PROVIDER_MAP: dict[str, str] = {
    "deepseek_v4_flash": "deepseek",
    "deepseek_v4_pro": "deepseek",
    "minimax_m3": "minimax",
    "glm_5": "glm",
    "glm_52": "glm",
}


class FourModelRateLimitHandler:
    """Enhanced rate limit handler for the four-model architecture.

    Features:
      - Per-model rate limit tracking (requests/min, tokens/min)
      - Different rate limit configs for each provider (DeepSeek, MiniMax, GLM)
      - Automatic retry with exponential backoff on 429 responses
      - Rate limit header parsing (X-RateLimit-Remaining, Retry-After)
      - Proactive throttling when approaching rate limits

    Usage::

        from teragent.reliability.recovery import FourModelRateLimitHandler

        handler = FourModelRateLimitHandler()

        # Check before making a call
        if handler.should_proactive_throttle("deepseek_v4_pro", estimated_tokens=50000):
            delay = handler.get_proactive_delay("deepseek_v4_pro")
            await asyncio.sleep(delay)

        # Record a successful call
        handler.record_request("deepseek_v4_pro", tokens_used=5000)

        # Handle a 429 response
        info = handler.parse_429_response("glm_52", headers={...}, body={...})
        if handler.should_retry("glm_52", info):
            delay = handler.get_backoff_delay("glm_52", attempt=0, rate_limit_info=info)
    """

    def __init__(
        self,
        provider_configs: dict[str, ProviderRateLimitConfig] | None = None,
        model_provider_map: dict[str, str] | None = None,
        breaker_manager: Any | None = None,
    ) -> None:
        """Initialize FourModelRateLimitHandler.

        Args:
            provider_configs: Per-provider rate limit configs. If None,
                default configs for DeepSeek/MiniMax/GLM are used.
            model_provider_map: Mapping of model_name → provider_name.
                If None, the default mapping is used.
            breaker_manager: Optional circuit breaker manager for integration.
        """
        self._provider_configs: dict[str, ProviderRateLimitConfig] = (
            provider_configs if provider_configs is not None
            else dict(_DEFAULT_PROVIDER_CONFIGS)
        )
        self._model_provider_map: dict[str, str] = (
            model_provider_map if model_provider_map is not None
            else dict(_MODEL_PROVIDER_MAP)
        )
        self._breaker_manager = breaker_manager

        # Per-model rate limit state
        self._model_states: dict[str, ModelRateLimitState] = {}

        # Initialize state for known models
        for model_name, provider in self._model_provider_map.items():
            self._model_states[model_name] = ModelRateLimitState(
                model_name=model_name,
                provider=provider,
            )

    # ----- Proactive throttling -----

    def should_proactive_throttle(
        self, model_name: str, estimated_tokens: int = 0
    ) -> bool:
        """Check if a request should be proactively throttled.

        Proactive throttling engages when the model is approaching its
        rate limit (above throttle_threshold), helping avoid 429 errors.

        Args:
            model_name: The model to check
            estimated_tokens: Estimated tokens for the upcoming request

        Returns:
            True if the request should be delayed
        """
        state = self._get_or_create_state(model_name)
        config = self._get_provider_config(state.provider)

        if config.requests_per_minute <= 0 and config.tokens_per_minute <= 0:
            return False  # No limits configured

        # Check request rate
        self._refresh_window(state)
        if config.requests_per_minute > 0:
            request_fraction = state.requests_in_window / config.requests_per_minute
            if request_fraction >= config.throttle_threshold:
                logger.info(
                    f"Proactive throttle for {model_name}: "
                    f"request rate {request_fraction:.1%} "
                    f"(threshold: {config.throttle_threshold:.0%})"
                )
                return True

        # Check token rate
        if config.tokens_per_minute > 0 and estimated_tokens > 0:
            projected_tokens = state.tokens_in_window + estimated_tokens
            token_fraction = projected_tokens / config.tokens_per_minute
            if token_fraction >= config.throttle_threshold:
                logger.info(
                    f"Proactive throttle for {model_name}: "
                    f"token rate {token_fraction:.1%} "
                    f"(threshold: {config.throttle_threshold:.0%})"
                )
                return True

        return False

    def get_proactive_delay(self, model_name: str) -> float:
        """Calculate the delay for proactive throttling.

        Returns the time until the current rate limit window resets,
        or a short delay if the window is about to reset.

        Args:
            model_name: The model to calculate delay for

        Returns:
            Delay in seconds (0.0 if no throttling needed)
        """
        state = self._get_or_create_state(model_name)
        now = time.time()

        # Time until window resets (1-minute window)
        window_elapsed = now - state.window_start_time
        window_remaining = max(0.0, 60.0 - window_elapsed)

        if window_remaining <= 0:
            return 0.0  # Window already reset

        # Return a fraction of the remaining time (don't wait the full window)
        return max(0.1, window_remaining * 0.5)

    # ----- Request tracking -----

    def record_request(
        self, model_name: str, tokens_used: int = 0, success: bool = True
    ) -> None:
        """Record a model request for rate limit tracking.

        Args:
            model_name: The model that was called
            tokens_used: Total tokens consumed by the request
            success: Whether the request succeeded (not a 429)
        """
        state = self._get_or_create_state(model_name)
        self._refresh_window(state)

        state.requests_in_window += 1
        state.tokens_in_window += tokens_used

    def record_429(
        self,
        model_name: str,
        headers: dict[str, str] | None = None,
        body: dict | None = None,
    ) -> dict[str, Any]:
        """Record a 429 rate limit response.

        Parses rate limit headers and body, updates state, and returns
        retry guidance.

        Args:
            model_name: The model that returned 429
            headers: Response headers (case-insensitive)
            body: Response body as a dict (if available)

        Returns:
            Dict with retry guidance:
              - "should_retry": bool
              - "delay_seconds": float
              - "remaining_requests": int | None
              - "remaining_tokens": int | None
              - "retry_after": float | None
        """
        state = self._get_or_create_state(model_name)
        config = self._get_provider_config(state.provider)

        state.last_429_time = time.time()
        state.total_429s += 1

        # Parse rate limit information from headers and body
        rate_info = self.parse_429_response(model_name, headers or {}, body)

        # Calculate backoff
        delay = self.get_backoff_delay(
            model_name,
            attempt=state.total_429s - 1,
            rate_limit_info=rate_info,
        )

        should_retry = state.total_429s <= config.max_retries
        if self._breaker_manager is not None:
            try:
                if not self._breaker_manager.can_call(model_name):
                    should_retry = False
            except Exception:
                pass

        logger.warning(
            f"429 rate limit for {model_name}: "
            f"attempt {state.total_429s}/{config.max_retries}, "
            f"delay={delay:.1f}s, retry={should_retry}"
        )

        return {
            "should_retry": should_retry,
            "delay_seconds": delay,
            "remaining_requests": rate_info.requests_remaining,
            "remaining_tokens": rate_info.tokens_remaining,
            "retry_after": rate_info.retry_after,
        }

    # ----- 429 response parsing -----

    def parse_429_response(
        self,
        model_name: str,
        headers: dict[str, str],
        body: dict | None = None,
    ) -> RateLimitInfo:
        """Parse a 429 rate limit response into unified format.

        Handles different response formats from each provider:
        - DeepSeek V4: Standard OpenAI 429 with Retry-After header
        - MiniMax M3: X-RateLimit-* headers
        - GLM-5/5.2: Custom 429 response body with retry_after field

        Args:
            model_name: The model that returned the response
            headers: Response headers
            body: Response body as a dict (if available)

        Returns:
            RateLimitInfo with normalized rate limit information
        """
        info = RateLimitInfo(model_name=model_name)

        # Normalize headers to lowercase
        lower_headers = {k.lower(): v for k, v in headers.items()}

        # Parse Retry-After header (DeepSeek V4 / standard OpenAI)
        retry_after_str = lower_headers.get("retry-after")
        if retry_after_str:
            try:
                info.retry_after = float(retry_after_str)
            except (ValueError, TypeError):
                pass

        # Parse X-RateLimit-Remaining header
        remaining_str = lower_headers.get("x-ratelimit-remaining")
        if remaining_str:
            try:
                info.requests_remaining = int(remaining_str)
            except (ValueError, TypeError):
                pass

        # Parse X-RateLimit-Remaining-Requests (MiniMax format)
        remaining_requests_str = lower_headers.get("x-ratelimit-remaining-requests")
        if remaining_requests_str:
            try:
                info.requests_remaining = int(remaining_requests_str)
            except (ValueError, TypeError):
                pass

        # Parse X-RateLimit-Remaining-Tokens
        tokens_remaining_str = lower_headers.get("x-ratelimit-remaining-tokens")
        if tokens_remaining_str:
            try:
                info.tokens_remaining = int(tokens_remaining_str)
            except (ValueError, TypeError):
                pass

        # Parse X-RateLimit-Reset
        reset_str = lower_headers.get("x-ratelimit-reset")
        if reset_str:
            try:
                info.reset_time = float(reset_str)
            except (ValueError, TypeError):
                pass

        # Parse response body (GLM-5/5.2 custom format)
        if body:
            for key, attr in [
                ("retry_after", "retry_after"),
                ("requests_remaining", "requests_remaining"),
                ("tokens_remaining", "tokens_remaining"),
                ("reset_time", "reset_time"),
            ]:
                if key in body:
                    try:
                        if attr == "requests_remaining" or attr == "tokens_remaining":
                            setattr(info, attr, int(body[key]))
                        else:
                            setattr(info, attr, float(body[key]))
                    except (ValueError, TypeError):
                        pass

        return info

    # ----- Retry logic -----

    def should_retry(
        self, model_name: str, rate_limit_info: RateLimitInfo
    ) -> bool:
        """Determine if a rate-limited request should be retried.

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

        # Check per-model 429 count against max retries
        state = self._get_or_create_state(model_name)
        config = self._get_provider_config(state.provider)
        if state.total_429s >= config.max_retries:
            return False

        # If we have remaining requests/tokens, retry
        if rate_limit_info.requests_remaining is not None:
            return rate_limit_info.requests_remaining > 0
        if rate_limit_info.tokens_remaining is not None:
            return rate_limit_info.tokens_remaining > 0

        # If we have retry_after, check if it's reasonable
        if rate_limit_info.retry_after is not None:
            return rate_limit_info.retry_after <= config.max_backoff

        # If we have reset_time, check if it's reasonable
        if rate_limit_info.reset_time is not None:
            wait_time = rate_limit_info.reset_time - time.time()
            return 0 < wait_time <= config.max_backoff

        return True  # Default: allow retry

    def get_backoff_delay(
        self,
        model_name: str,
        attempt: int,
        rate_limit_info: RateLimitInfo | None = None,
    ) -> float:
        """Calculate the backoff delay for a rate-limited retry.

        Priority:
        1. retry_after from the rate limit response
        2. Time until reset_time
        3. Provider-specific exponential backoff with jitter

        Args:
            model_name: The model being retried
            attempt: Current attempt number (0-indexed)
            rate_limit_info: Optional parsed rate limit information

        Returns:
            Delay in seconds before the next retry
        """
        import random

        state = self._get_or_create_state(model_name)
        config = self._get_provider_config(state.provider)

        if attempt >= config.max_retries:
            return 0.0

        # Priority 1: Use server-provided retry_after
        if rate_limit_info and rate_limit_info.retry_after is not None:
            delay = min(rate_limit_info.retry_after, config.max_backoff)
            jitter = delay * 0.1 * (random.random() * 2 - 1)
            return max(0.1, delay + jitter)

        # Priority 2: Calculate wait time from reset_time
        if rate_limit_info and rate_limit_info.reset_time is not None:
            wait_time = rate_limit_info.reset_time - time.time()
            if 0 < wait_time <= config.max_backoff:
                jitter = wait_time * 0.1 * (random.random() * 2 - 1)
                return max(0.1, wait_time + jitter)

        # Priority 3: Provider-specific exponential backoff
        delay = config.base_backoff * (2 ** attempt)
        delay = min(delay, config.max_backoff)
        jitter = delay * 0.25 * (random.random() * 2 - 1)
        return max(0.1, delay + jitter)

    # ----- State queries -----

    def get_model_state(self, model_name: str) -> dict[str, Any]:
        """Get rate limit state for a model.

        Args:
            model_name: The model to query

        Returns:
            Dict with rate limit state information
        """
        state = self._get_or_create_state(model_name)
        config = self._get_provider_config(state.provider)
        self._refresh_window(state)

        return {
            "model_name": model_name,
            "provider": state.provider,
            "requests_in_window": state.requests_in_window,
            "tokens_in_window": state.tokens_in_window,
            "requests_per_minute_limit": config.requests_per_minute,
            "tokens_per_minute_limit": config.tokens_per_minute,
            "request_utilization": (
                state.requests_in_window / config.requests_per_minute
                if config.requests_per_minute > 0 else 0.0
            ),
            "token_utilization": (
                state.tokens_in_window / config.tokens_per_minute
                if config.tokens_per_minute > 0 else 0.0
            ),
            "total_429s": state.total_429s,
            "throttled_requests": state.throttled_requests,
            "is_throttled": self.should_proactive_throttle(model_name),
        }

    def get_all_states(self) -> dict[str, dict[str, Any]]:
        """Get rate limit state for all known models.

        Returns:
            Dict mapping model_name → state dict
        """
        return {name: self.get_model_state(name) for name in self._model_states}

    # ----- Internal helpers -----

    def _get_or_create_state(self, model_name: str) -> ModelRateLimitState:
        """Get or create rate limit state for a model.

        Args:
            model_name: The model name

        Returns:
            ModelRateLimitState for the model
        """
        if model_name not in self._model_states:
            provider = self._model_provider_map.get(model_name, "unknown")
            self._model_states[model_name] = ModelRateLimitState(
                model_name=model_name,
                provider=provider,
            )
        return self._model_states[model_name]

    def _get_provider_config(self, provider_name: str) -> ProviderRateLimitConfig:
        """Get rate limit config for a provider.

        Args:
            provider_name: Provider name

        Returns:
            ProviderRateLimitConfig for the provider
        """
        if provider_name in self._provider_configs:
            return self._provider_configs[provider_name]
        # Return a default config
        return ProviderRateLimitConfig(provider_name=provider_name)

    def _refresh_window(self, state: ModelRateLimitState) -> None:
        """Refresh the rate limit window if it has expired.

        The window is 60 seconds (1 minute). If the current time is
        more than 60 seconds past the window start, the counters
        are reset.

        Args:
            state: The ModelRateLimitState to refresh
        """
        now = time.time()
        if now - state.window_start_time >= 60.0:
            state.requests_in_window = 0
            state.tokens_in_window = 0
            state.window_start_time = now


# ---------------------------------------------------------------------------
# GLM-5.2 1M Context Stability Recovery (P4-3)
# ---------------------------------------------------------------------------


@dataclass
class ContextStabilityMetrics:
    """Stability metrics for GLM-5.2 1M context requests.

    Attributes:
        total_1m_requests: Total 1M context requests attempted
        successful_1m_requests: Successful 1M context requests
        failed_1m_requests: Failed 1M context requests
        failure_rate: Current failure rate for 1M requests (0.0 - 1.0)
        avg_latency_ms: Average latency for 1M context requests
        current_context_mode: Current context mode ("1m", "500k", "200k", "128k")
        auto_downgrade_count: Number of times auto-downgraded
        last_downgrade_time: Timestamp of most recent auto-downgrade
    """

    total_1m_requests: int = 0
    successful_1m_requests: int = 0
    failed_1m_requests: int = 0
    failure_rate: float = 0.0
    avg_latency_ms: float = 0.0
    current_context_mode: str = "1m"
    auto_downgrade_count: int = 0
    last_downgrade_time: float = 0.0


# Context size negotiation steps
_CONTEXT_NEGOTIATION_STEPS: list[tuple[str, int]] = [
    ("1m", 1_000_000),
    ("500k", 500_000),
    ("200k", 200_000),
    ("128k", 128_000),
]


class GLM52ContextStabilityRecovery:
    """Stability recovery for GLM-5.2 1M context requests.

    Monitors the success rate of 1M context requests and automatically
    downgrades to smaller context sizes when the failure rate exceeds
    the threshold (default: 10%).

    Context size negotiation tries: 1M → 500K → 200K → 128K

    Usage::

        from teragent.reliability.recovery import GLM52ContextStabilityRecovery

        recovery = GLM52ContextStabilityRecovery()

        # Before making a 1M context request
        mode = recovery.get_current_context_mode()
        max_tokens = recovery.get_max_context_tokens()

        # After a request completes
        recovery.record_request(success=True, latency_ms=35000)

        # Check stability
        metrics = recovery.get_metrics()
        if metrics.failure_rate > 0.1:
            print(f"High failure rate: {metrics.failure_rate:.1%}")
    """

    def __init__(
        self,
        failure_rate_threshold: float = 0.1,
        window_size: int = 20,
        auto_recovery_threshold: float = 0.05,
    ) -> None:
        """Initialize GLM52ContextStabilityRecovery.

        Args:
            failure_rate_threshold: Failure rate that triggers auto-downgrade (0.1 = 10%)
            window_size: Number of recent requests to consider for failure rate
            auto_recovery_threshold: Failure rate below which to try upgrading
                back to a higher context mode (0.05 = 5%)
        """
        self._failure_rate_threshold = failure_rate_threshold
        self._window_size = window_size
        self._auto_recovery_threshold = auto_recovery_threshold

        # Stability metrics
        self._metrics = ContextStabilityMetrics()

        # Rolling window of recent request outcomes (True=success, False=failure)
        self._recent_outcomes: list[bool] = []
        # Rolling window of recent latencies
        self._recent_latencies: list[float] = []

        # Current context mode index into _CONTEXT_NEGOTIATION_STEPS
        self._current_mode_idx: int = 0  # Start at 1M

    # ----- Core methods -----

    def record_request(self, success: bool, latency_ms: float = 0.0) -> None:
        """Record the outcome of a GLM-5.2 context request.

        Updates failure rate tracking and may trigger auto-downgrade
        if the failure rate exceeds the threshold.

        Args:
            success: Whether the request succeeded
            latency_ms: Request latency in milliseconds
        """
        self._metrics.total_1m_requests += 1

        if success:
            self._metrics.successful_1m_requests += 1
        else:
            self._metrics.failed_1m_requests += 1

        # Update rolling window
        self._recent_outcomes.append(success)
        if len(self._recent_outcomes) > self._window_size:
            self._recent_outcomes = self._recent_outcomes[-self._window_size:]

        # Update latency tracking
        if latency_ms > 0:
            self._recent_latencies.append(latency_ms)
            if len(self._recent_latencies) > self._window_size:
                self._recent_latencies = self._recent_latencies[-self._window_size:]

        # Recalculate failure rate from rolling window
        self._update_metrics()

        # Check if auto-downgrade is needed
        if (
            not success
            and self._metrics.failure_rate > self._failure_rate_threshold
            and self._metrics.current_context_mode == "1m"
        ):
            self._auto_downgrade("failure_rate_exceeded")

        logger.debug(
            f"GLM-5.2 context request recorded: success={success}, "
            f"latency={latency_ms:.0f}ms, "
            f"failure_rate={self._metrics.failure_rate:.1%}, "
            f"mode={self._metrics.current_context_mode}"
        )

    def get_current_context_mode(self) -> str:
        """Get the current context mode.

        Returns:
            Context mode string: "1m", "500k", "200k", or "128k"
        """
        return self._metrics.current_context_mode

    def get_max_context_tokens(self) -> int:
        """Get the maximum context tokens for the current mode.

        Returns:
            Maximum context token count
        """
        mode_name, token_count = _CONTEXT_NEGOTIATION_STEPS[self._current_mode_idx]
        return token_count

    def negotiate_context_size(
        self, requested_tokens: int
    ) -> tuple[str, int]:
        """Negotiate the appropriate context size for a request.

        Starts from 1M and steps down through 500K → 200K → 128K
        until finding a context size that is both:
        - At or above the requested token count, AND
        - At or below the current stable context mode

        Args:
            requested_tokens: The number of tokens needed for the request

        Returns:
            Tuple of (mode_name, max_tokens) for the negotiated context size
        """
        # Start from the current stable mode (can't go higher)
        for i in range(self._current_mode_idx, len(_CONTEXT_NEGOTIATION_STEPS)):
            mode_name, max_tokens = _CONTEXT_NEGOTIATION_STEPS[i]
            if max_tokens >= requested_tokens:
                return mode_name, max_tokens

        # Even the smallest mode can't fit — return the smallest
        return _CONTEXT_NEGOTIATION_STEPS[-1]

    def force_context_mode(self, mode: str) -> bool:
        """Force a specific context mode.

        Useful for manual override when the operator knows a specific
        mode is more appropriate.

        Args:
            mode: Context mode string ("1m", "500k", "200k", "128k")

        Returns:
            True if the mode was set, False if the mode is invalid
        """
        for idx, (mode_name, _) in enumerate(_CONTEXT_NEGOTIATION_STEPS):
            if mode_name == mode:
                self._current_mode_idx = idx
                self._metrics.current_context_mode = mode
                logger.info(
                    f"GLM-5.2 context mode forced to {mode}"
                )
                return True
        logger.warning(f"Invalid context mode: {mode}")
        return False

    def get_metrics(self) -> ContextStabilityMetrics:
        """Get current stability metrics.

        Returns:
            ContextStabilityMetrics with current state
        """
        return self._metrics

    def check_upgrade_eligible(self) -> bool:
        """Check if the context mode can be upgraded.

        The mode can be upgraded if:
        - The current failure rate is below the auto_recovery_threshold
        - There's a higher mode available

        Returns:
            True if upgrade is possible
        """
        if self._current_mode_idx == 0:
            return False  # Already at highest mode (1M)

        return self._metrics.failure_rate < self._auto_recovery_threshold

    def try_upgrade(self) -> bool:
        """Attempt to upgrade to a higher context mode.

        Only upgrades if the failure rate is below the auto_recovery_threshold.

        Returns:
            True if the upgrade was successful
        """
        if not self.check_upgrade_eligible():
            return False

        old_mode = self._metrics.current_context_mode
        self._current_mode_idx -= 1
        new_mode_name, _ = _CONTEXT_NEGOTIATION_STEPS[self._current_mode_idx]
        self._metrics.current_context_mode = new_mode_name

        logger.info(
            f"GLM-5.2 context mode upgraded: {old_mode} → {new_mode_name} "
            f"(failure_rate={self._metrics.failure_rate:.1%})"
        )
        return True

    # ----- Internal helpers -----

    def _auto_downgrade(self, reason: str) -> None:
        """Auto-downgrade to a smaller context mode.

        Args:
            reason: Why the downgrade is happening
        """
        if self._current_mode_idx >= len(_CONTEXT_NEGOTIATION_STEPS) - 1:
            logger.warning(
                f"GLM-5.2 cannot downgrade further: already at "
                f"{self._metrics.current_context_mode}"
            )
            return

        old_mode = self._metrics.current_context_mode
        self._current_mode_idx += 1
        new_mode_name, _ = _CONTEXT_NEGOTIATION_STEPS[self._current_mode_idx]
        self._metrics.current_context_mode = new_mode_name
        self._metrics.auto_downgrade_count += 1
        self._metrics.last_downgrade_time = time.time()

        logger.warning(
            f"GLM-5.2 auto-downgraded: {old_mode} → {new_mode_name} "
            f"(reason: {reason}, failure_rate={self._metrics.failure_rate:.1%})"
        )

    def _update_metrics(self) -> None:
        """Update stability metrics from the rolling window."""
        if self._recent_outcomes:
            failures = sum(1 for ok in self._recent_outcomes if not ok)
            self._metrics.failure_rate = failures / len(self._recent_outcomes)
        else:
            self._metrics.failure_rate = 0.0

        if self._recent_latencies:
            self._metrics.avg_latency_ms = (
                sum(self._recent_latencies) / len(self._recent_latencies)
            )
