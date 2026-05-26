"""Distributed tracing utilities for the TerAgent framework.

NOTE: The span-based tracing API (start_span, end_span, get_all_spans,
get_span_summary) is marked @experimental — it is a planned feature
for distributed tracing and its interface may change in future versions.
"""

import uuid
import time
import contextvars
import functools
import logging
import warnings
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def experimental(func):
    """Decorator to mark a function as experimental.

    Experimental functions are part of a planned feature and their
    interface may change in future versions. A UserWarning is issued
    on the first call.
    """
    _warned = False

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal _warned
        if not _warned:
            warnings.warn(
                f"{func.__name__} is experimental and its interface may change.",
                UserWarning,
                stacklevel=2,
            )
            _warned = True
        return func(*args, **kwargs)

    return wrapper

_current_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)
_current_span_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "span_id", default=""
)


@dataclass
class Span:
    """A tracing span that records an operation's timing and hierarchy.

    Spans form a parent-child tree: each span stores the ID of its
    parent, and the _current_span_id context variable tracks the
    deepest active span so that start_span() can auto-link children.
    """
    span_id: str          # Unique identifier for this span (8-char UUID)
    operation: str        # Human-readable operation name
    start_time: float = 0.0   # Unix timestamp when the span started
    end_time: float = 0.0     # Unix timestamp when the span ended
    parent_id: str = ""       # ID of the parent span (empty for root)
    request_id: str = ""      # ID of the enclosing request trace
    attributes: dict = field(default_factory=dict)  # Arbitrary key-value metadata
    status: str = "ok"        # "ok" | "error"

    @property
    def duration_ms(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "operation": self.operation,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "parent_id": self.parent_id,
            "request_id": self.request_id,
            "attributes": self.attributes,
            "status": self.status,
        }


# In-memory span store for the current request
_active_spans: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "active_spans", default=None
)


def get_request_id() -> str:
    """Return the trace ID for the current request, or empty string if unset."""
    return _current_request_id.get()


def set_request_id(req_id: str | None = None) -> str:
    """Set the trace ID for the current request, auto-generating a UUID if not provided."""
    rid = req_id or str(uuid.uuid4())
    _current_request_id.set(rid)
    _active_spans.set({})
    return rid


def reset_request_id() -> None:
    """Reset the current request trace ID and clear all active spans."""
    _current_request_id.set("")
    _active_spans.set({})


@experimental
def start_span(operation: str, attributes: dict | None = None) -> Span:
    """Create and start a new span, auto-linking it as a child of the current span.

    The parent-child relationship is maintained via the _current_span_id
    context variable: we read the current span ID to use as parent_id,
    then push the new span's ID so that subsequent start_span() calls
    will nest beneath this one.
    """
    span_id = uuid.uuid4().hex[:12]
    parent_id = _current_span_id.get()
    span = Span(
        span_id=span_id,
        operation=operation,
        start_time=time.time(),
        parent_id=parent_id,
        request_id=_current_request_id.get(),
        attributes=attributes or {},
    )
    spans = {**(_active_spans.get() or {}), span_id: span}
    _active_spans.set(spans)
    # Push: make this span the new "current" so children will link to it
    _current_span_id.set(span_id)
    return span


@experimental
def end_span(span: Span, status: str = "ok") -> None:
    """End a span, recording its finish time and status.

    After recording, the _current_span_id is restored to the span's
    parent_id ONLY if the ended span is the current span. This prevents
    stack corruption when spans are ended out-of-order.
    """
    span.end_time = time.time()
    span.status = status
    # Only pop if this span is the current one — prevents stack corruption
    if _current_span_id.get() == span.span_id:
        _current_span_id.set(span.parent_id)
    else:
        logger.debug(
            f"Span {span.operation} ({span.span_id}) ended out of order. "
            f"Current span is {_current_span_id.get()}. Stack not modified."
        )
    logger.debug(
        f"Span {span.operation} completed in {span.duration_ms:.1f}ms (status={status})"
    )


@experimental
def get_all_spans() -> list[dict]:
    """Return all spans for the current request as a list of dicts."""
    spans = _active_spans.get() or {}
    return [s.to_dict() for s in spans.values()]


@experimental
def get_span_summary() -> dict:
    """Return a summary dict for the current request's spans.

    Returns:
        dict with keys: request_id, total_spans, total_duration_ms, error_count
    """
    spans = _active_spans.get() or {}
    total_duration = sum(s.duration_ms for s in spans.values() if s.end_time > 0)
    error_count = sum(1 for s in spans.values() if s.status == "error")
    return {
        "request_id": _current_request_id.get(),
        "total_spans": len(spans),
        "total_duration_ms": total_duration,
        "error_count": error_count,
    }
