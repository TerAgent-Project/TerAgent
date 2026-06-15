"""Type stubs for teragent.core.provider — ModelProvider"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator

__all__ = ["ModelProvider"]

if TYPE_CHECKING:
    from teragent.core.adapter import TAPAdapter
    from teragent.core.compiler import TAPCompiler
    from teragent.core.tap import (
        CompiledPrompt,
        CostTracker,
        TAPCostRecord,
        TAPRequest,
        TAPResponse,
    )
    from teragent.pipeline.tracing import TAPTracer

class ModelProvider:
    """Model provider — composes Compiler + Adapter."""

    compiler: TAPCompiler
    adapter: TAPAdapter
    model: str

    def __init__(
        self,
        compiler: TAPCompiler,
        adapter: TAPAdapter,
        model: str,
        fallback: ModelProvider | None = ...,
        circuit_breaker: Any | None = ...,
        tracer: TAPTracer | None = ...,
    ) -> None: ...

    # ===== Core TAP interface =====

    async def execute_tap(self, request: TAPRequest) -> TAPResponse: ...
    async def stream_tap(self, request: TAPRequest) -> AsyncIterator[str]: ...
    async def chat(
        self, messages: list[dict], tools: list[dict] | None = ...
    ) -> dict: ...

    def _validate_compiled_mode(self, compiled: CompiledPrompt) -> None: ...

    # ===== TAP with retry =====

    async def execute_tap_with_retry(
        self,
        request: TAPRequest,
        max_retries: int = ...,
        retry_delay: float = ...,
    ) -> TAPResponse: ...

    # ===== Chat with fallback =====

    async def chat_with_fallback(
        self, messages: list[dict], tools: list[dict] | None = ...
    ) -> dict: ...

    # ===== Tracer integration =====

    @property
    def tracer(self) -> TAPTracer | None: ...
    def set_tracer(self, tracer: TAPTracer | None) -> None: ...

    # ===== Fallback =====

    @property
    def fallback_provider(self) -> ModelProvider | None: ...
    @property
    def has_fallback(self) -> bool: ...
    def set_fallback(self, fallback: ModelProvider | None) -> None: ...

    # ===== Cost tracking =====

    @property
    def cost_records(self) -> list[TAPCostRecord]: ...
    def get_cost_summary(self) -> dict: ...

    # ===== Capabilities =====

    @property
    def capabilities(self) -> dict: ...

    # ===== Resource cleanup =====

    async def close(self) -> None: ...

    # ===== Async context manager =====

    async def __aenter__(self) -> ModelProvider: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> bool: ...

    def __repr__(self) -> str: ...
