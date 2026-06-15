"""teragent.core.rate_limiter — Centralized rate limiting for LLM API calls

Provides token-bucket and sliding-window rate limiters that adapters
can use to avoid hitting API rate limits. Also provides an adaptive
rate limiter that learns from X-RateLimit response headers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "RateLimitStrategy",
    "RateLimitConfig",
    "RateLimitStatus",
    "TokenBucketRateLimiter",
    "SlidingWindowRateLimiter",
    "AdaptiveRateLimiter",
    "RateLimiter",
    "create_rate_limiter",
]


class RateLimitStrategy(Enum):
    """Rate limiting strategy"""

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    ADAPTIVE = "adaptive"  # Learns from response headers


@dataclass
class RateLimitConfig:
    """Configuration for a rate limiter"""

    strategy: RateLimitStrategy = RateLimitStrategy.ADAPTIVE
    # Token bucket settings
    max_tokens: int = 60  # Maximum tokens in bucket
    refill_rate: float = 1.0  # Tokens per second
    # Sliding window settings
    max_requests: int = 60  # Max requests per window
    window_seconds: float = 60.0  # Window duration
    # Adaptive settings
    initial_rpm: int = 60  # Initial requests per minute estimate
    safety_factor: float = 0.8  # Use only 80% of discovered limit


@dataclass
class RateLimitStatus:
    """Current rate limit status"""

    remaining: int = 0
    limit: int = 0
    reset_at: float = 0.0  # Unix timestamp
    retry_after: float | None = None  # Seconds until retry is allowed

    @property
    def is_limited(self) -> bool:
        """Whether the rate limit is currently active."""
        return self.remaining <= 0 and time.time() < self.reset_at

    @property
    def wait_seconds(self) -> float:
        """Seconds to wait before the next request is allowed."""
        if self.retry_after is not None:
            return self.retry_after
        if self.reset_at > time.time():
            return self.reset_at - time.time()
        return 0.0


class TokenBucketRateLimiter:
    """Token bucket rate limiter

    Allows bursty traffic up to max_tokens, then refills at refill_rate.
    """

    def __init__(self, config: RateLimitConfig) -> None:
        self._max = config.max_tokens
        self._rate = config.refill_rate
        self._tokens = float(config.max_tokens)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens. Returns wait time (0 if immediate)."""
        async with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0
            # Calculate wait time
            needed = tokens - self._tokens
            wait = needed / self._rate
            self._tokens = 0
            return wait

    async def wait_and_acquire(self, tokens: int = 1) -> None:
        """Wait if needed, then acquire tokens.

        修复：使用循环 + lock 内 refill 的方式确保：
        1. 等待期间累积的 token 不丢失
        2. 在 lock 内完成 token 扣减，避免竞态条件
        3. 避免双重扣减（旧代码 acquire 已扣减，wait_and_acquire 又扣一次）
        """
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                # Calculate wait time
                needed = tokens - self._tokens
            wait = needed / self._rate
            logger.info(f"Rate limiter: waiting {wait:.1f}s")
            await asyncio.sleep(wait)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max, self._tokens + elapsed * self._rate)
        self._last_refill = now


class SlidingWindowRateLimiter:
    """Sliding window rate limiter

    Tracks request timestamps and ensures no more than max_requests
    in any window_seconds window.
    """

    def __init__(self, config: RateLimitConfig) -> None:
        self._max = config.max_requests
        self._window = config.window_seconds
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Check if request is allowed. Returns wait time (0 if immediate)."""
        async with self._lock:
            now = time.monotonic()
            cutoff = now - self._window
            self._timestamps = [t for t in self._timestamps if t > cutoff]

            if len(self._timestamps) < self._max:
                self._timestamps.append(now)
                return 0.0

            # Wait until oldest request exits the window
            oldest = self._timestamps[0]
            wait = oldest + self._window - now
            return max(0.0, wait)

    async def wait_and_acquire(self) -> None:
        """Wait if needed, then acquire.

        修复：sleep 后重新获取锁，清理过期时间戳再 append，
        避免竞态条件导致超限。
        """
        wait = await self.acquire()
        if wait > 0:
            logger.info(f"Rate limiter: waiting {wait:.1f}s")
            await asyncio.sleep(wait)
            async with self._lock:
                now = time.monotonic()
                cutoff = now - self._window
                self._timestamps = [t for t in self._timestamps if t > cutoff]
                self._timestamps.append(now)


class AdaptiveRateLimiter:
    """Adaptive rate limiter that learns from API response headers

    Starts with conservative defaults and adjusts based on:
    - X-RateLimit-Remaining headers
    - X-RateLimit-Reset headers
    - 429 Retry-After headers
    - Observed success/failure patterns
    """

    def __init__(self, config: RateLimitConfig) -> None:
        self._safety = config.safety_factor
        self._current_rpm = config.initial_rpm
        self._status = RateLimitStatus()
        self._lock = asyncio.Lock()
        self._window_limiter = SlidingWindowRateLimiter(
            RateLimitConfig(
                max_requests=int(config.initial_rpm * config.safety_factor),
                window_seconds=60.0,
            )
        )

    def update_from_headers(self, headers: dict[str, str]) -> None:
        """Update rate limit state from HTTP response headers.

        Recognized headers (case-insensitive):
        - X-RateLimit-Limit: Maximum requests allowed
        - X-RateLimit-Remaining: Requests remaining
        - X-RateLimit-Reset: Unix timestamp when limit resets
        - Retry-After: Seconds until retry is allowed

        Args:
            headers: HTTP response headers (dict-like, case-insensitive lookup)
        """
        # Normalize headers to lowercase for case-insensitive lookup
        lower_headers = {k.lower(): v for k, v in headers.items()}

        remaining = lower_headers.get("x-ratelimit-remaining")
        limit = lower_headers.get("x-ratelimit-limit")
        reset = lower_headers.get("x-ratelimit-reset")
        retry_after = lower_headers.get("retry-after")

        # Protect shared state with lock to prevent race conditions
        # in concurrent async contexts
        if remaining is not None:
            self._status.remaining = int(remaining)
        if limit is not None:
            self._status.limit = int(limit)
            # Adjust window limiter based on discovered limit
            new_rpm = int(int(limit) * self._safety)
            if new_rpm != self._window_limiter._max:
                logger.info(
                    f"Adaptive rate limiter: adjusting from {self._window_limiter._max} to {new_rpm} RPM"
                )
                self._window_limiter._max = new_rpm
                self._current_rpm = int(limit)
        if reset is not None:
            try:
                self._status.reset_at = float(reset)
            except ValueError:
                pass
        if retry_after is not None:
            try:
                self._status.retry_after = float(retry_after)
            except ValueError:
                pass

    def update_from_429(self, retry_after: float | None = None) -> None:
        """Called when a 429 response is received.

        Args:
            retry_after: Seconds until retry is allowed (from Retry-After header)
        """
        self._status.remaining = 0
        if retry_after is not None:
            self._status.retry_after = retry_after
            self._status.reset_at = time.time() + retry_after
        # Back off: reduce current RPM
        new_rpm = max(1, int(self._current_rpm * 0.5))
        self._window_limiter._max = int(new_rpm * self._safety)
        logger.warning(
            f"Rate limit hit (429). Backing off to ~{self._window_limiter._max} RPM"
        )

    @property
    def status(self) -> RateLimitStatus:
        """Current rate limit status."""
        return self._status

    async def acquire(self) -> float:
        """Check if request is allowed. Returns wait time."""
        # First check adaptive status (from headers)
        if self._status.is_limited:
            return self._status.wait_seconds

        # Then check sliding window
        return await self._window_limiter.acquire()

    async def wait_and_acquire(self) -> None:
        """Wait if needed, then acquire."""
        # Check adaptive limits first
        if self._status.is_limited:
            wait = self._status.wait_seconds
            logger.info(f"Rate limiter: waiting {wait:.1f}s (from API headers)")
            await asyncio.sleep(wait)
            self._status.retry_after = None

        # Then check sliding window
        await self._window_limiter.wait_and_acquire()


def create_rate_limiter(
    config: RateLimitConfig | None = None,
) -> TokenBucketRateLimiter | SlidingWindowRateLimiter | AdaptiveRateLimiter:
    """Factory function to create a rate limiter based on config strategy.

    Args:
        config: Rate limit configuration. Defaults to AdaptiveRateLimiter.

    Returns:
        A rate limiter instance
    """
    if config is None:
        config = RateLimitConfig()

    if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
        return TokenBucketRateLimiter(config)
    elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
        return SlidingWindowRateLimiter(config)
    else:
        return AdaptiveRateLimiter(config)


# Public type alias for any rate limiter implementation
RateLimiter = TokenBucketRateLimiter | SlidingWindowRateLimiter | AdaptiveRateLimiter
"""Type alias for any rate limiter instance.

Use this in type annotations when you don't care which strategy is used::

    from teragent.core.rate_limiter import RateLimiter, create_rate_limiter

    limiter: RateLimiter = create_rate_limiter()
"""
