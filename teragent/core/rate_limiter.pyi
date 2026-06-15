"""Type stubs for teragent.core.rate_limiter — Centralized rate limiting for LLM API calls"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

class RateLimitStrategy(Enum):
    """Rate limiting strategy."""

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    ADAPTIVE = "adaptive"

@dataclass
class RateLimitConfig:
    """Configuration for a rate limiter."""

    strategy: RateLimitStrategy
    max_tokens: int
    refill_rate: float
    max_requests: int
    window_seconds: float
    initial_rpm: int
    safety_factor: float

@dataclass
class RateLimitStatus:
    """Current rate limit status."""

    remaining: int
    limit: int
    reset_at: float
    retry_after: float | None

    @property
    def is_limited(self) -> bool: ...
    @property
    def wait_seconds(self) -> float: ...

class TokenBucketRateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, config: RateLimitConfig) -> None: ...
    async def acquire(self, tokens: int = ...) -> float: ...
    async def wait_and_acquire(self, tokens: int = ...) -> None: ...

class SlidingWindowRateLimiter:
    """Sliding window rate limiter."""

    def __init__(self, config: RateLimitConfig) -> None: ...
    async def acquire(self) -> float: ...
    async def wait_and_acquire(self) -> None: ...

class AdaptiveRateLimiter:
    """Adaptive rate limiter that learns from API response headers."""

    def __init__(self, config: RateLimitConfig) -> None: ...
    def update_from_headers(self, headers: dict[str, str]) -> None: ...
    def update_from_429(self, retry_after: float | None = ...) -> None: ...

    @property
    def status(self) -> RateLimitStatus: ...

    async def acquire(self) -> float: ...
    async def wait_and_acquire(self) -> None: ...

def create_rate_limiter(
    config: RateLimitConfig | None = ...,
) -> TokenBucketRateLimiter | SlidingWindowRateLimiter | AdaptiveRateLimiter: ...

# Public type alias for any rate limiter implementation
RateLimiter = TokenBucketRateLimiter | SlidingWindowRateLimiter | AdaptiveRateLimiter
