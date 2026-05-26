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
