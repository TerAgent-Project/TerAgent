"""teragent.pipeline.retry — Generic retry with exponential backoff

Extracted from design_generator.py and plan_generator.py, which had
highly similar retry+backoff logic. This is a universal primitive that
any async pipeline stage needs.

Usage:
    from teragent.pipeline.retry import retry_with_backoff

    result = await retry_with_backoff(
        fn=lambda: model.chat(messages=messages),
        max_retries=3,
        backoff_base=10.0,
        validate=lambda r: _validate_sections(r),
        on_retry=lambda attempt, err: logger.warning(f"Retry {attempt}: {err}"),
    )
"""
import asyncio
import logging
from typing import Awaitable, Callable, TypeVar

__all__ = [
    "RetryValidationError",
    "retry_with_backoff",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryValidationError(ValueError):
    """Raised when validation fails on the last retry attempt.

    This distinguishes "function ran but output was invalid" from
    "function crashed with an exception", allowing callers to handle
    the two failure modes differently.
    """

    def __init__(self, errors: list[str] | str) -> None:
        if isinstance(errors, list):
            error_msg = f"Validation failed: {errors}"
        else:
            error_msg = errors
        super().__init__(error_msg)
        self.validation_errors = errors


async def retry_with_backoff(
    fn: Callable[..., Awaitable[T]],
    max_retries: int = 3,
    backoff_base: float = 10.0,
    validate: Callable[[T], list[str]] | None = None,
    on_retry: Callable[[int, str], None] | None = None,
) -> T:
    """Generic retry with exponential backoff and optional result validation.

    Args:
        fn: Async function to execute
        max_retries: Maximum number of retries (total attempts = max_retries + 1)
        backoff_base: Base delay in seconds (actual = base * 2^attempt)
        validate: Optional result validation function; returns error list
            (empty = pass). If validation fails on last attempt, raises RetryValidationError.
        on_retry: Optional callback invoked before each retry: (attempt, error_msg)

    Returns:
        The return value of fn()

    Raises:
        RetryValidationError: If validation fails on the last attempt
        Exception: The last exception from fn() if all retries exhausted
    """
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            result = await fn()
            if validate:
                errors = validate(result)
                if errors:
                    last_error = f"Validation failed: {errors}"
                    if attempt < max_retries:
                        if on_retry:
                            on_retry(attempt, last_error)
                        await asyncio.sleep(backoff_base * (2 ** attempt))
                        continue
                    # Last attempt validation failure → raise RetryValidationError
                    # (distinct from function crash, which re-raises the original exception)
                    raise RetryValidationError(errors)
            return result
        except Exception as e:
            last_error = str(e)
            if attempt >= max_retries:
                raise
            if on_retry:
                on_retry(attempt, last_error)
            await asyncio.sleep(backoff_base * (2 ** attempt))
    # Unreachable: all paths in the loop either return or raise
    raise RuntimeError(f"All retries exhausted: {last_error}")  # pragma: no cover
