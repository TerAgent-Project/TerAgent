"""teragent.core.provider — ModelProvider (combines Compiler + Adapter)

The core entry point for the teragent library. A ModelProvider composes
a TAPCompiler (how to compile prompts) with a TAPAdapter (how to send HTTP),
plus a model identifier.

This design enables orthogonal composition:
  - Same Adapter with different Compilers (e.g., OpenAI adapter + GLM compiler)
  - Same Compiler with different Adapters (e.g., Anthropic compiler + OpenRouter adapter)

Usage:
    from teragent import create_provider, TAPRequest

    provider = create_provider(
        compiler="glm",
        adapter="openai_compatible",
        model="glm-5.1",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="GLM_API_KEY",
    )

    response = await provider.execute_tap(TAPRequest(
        meta={"task_id": "1.1", "intent": "code_generation"},
        instruction="实现用户登录模块",
        constraints=["Python 3.10+"],
        output_format_hint="<file path='...'>完整代码</file>",
    ))
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, TYPE_CHECKING

from teragent.core.tap import (
    TAPRequest,
    TAPResponse,
    TAPCostRecord,
    CompiledPrompt,
    CostTracker,
)
from teragent.core.compiler import TAPCompiler
from teragent.core.adapter import TAPAdapter

if TYPE_CHECKING:
    from teragent.pipeline.tracing import TAPTracer

logger = logging.getLogger(__name__)


class ModelProvider:
    """Model provider — composes Compiler + Adapter

    The central class of the teragent library. Users specify a compiler
    (prompt compilation strategy) and an adapter (HTTP protocol), and
    the provider automatically combines them.

    Attributes:
        compiler: TAPCompiler instance for prompt compilation
        adapter: TAPAdapter instance for HTTP sending
        model: Model identifier string
        fallback_provider: Optional fallback ModelProvider (None if not configured)
        has_fallback: Whether a fallback provider is configured
        tracer: Optional TAPTracer for auto-tracing TAP calls (Phase 10)
        cost_records: All cost records from TAP calls
    """

    def __init__(
        self,
        compiler: TAPCompiler,
        adapter: TAPAdapter,
        model: str,
        fallback: ModelProvider | None = None,
        circuit_breaker: Any | None = None,
        tracer: TAPTracer | None = None,
    ) -> None:
        self.compiler = compiler
        self.adapter = adapter
        self.model = model
        self._fallback = fallback
        self._circuit_breaker = circuit_breaker
        self._tracer = tracer
        self._cost_tracker = CostTracker()

    # ===== Core TAP interface =====

    async def execute_tap(self, request: TAPRequest) -> TAPResponse:
        """Execute a TAP request: compile → send

        If a TAPTracer is attached (Phase 10), the request and response
        are automatically recorded for DPO pair generation.

        Args:
            request: The TAP request to execute

        Returns:
            TAPResponse with the model's output
        """
        # Phase 10: Auto-trace TAP request if tracer is attached
        trace_id = ""
        if self._tracer is not None:
            trace_id = await self._tracer.record_request(request)

        compiled = self.compiler.compile(request)

        # Validate CompiledPrompt mode compatibility with adapter
        self._validate_compiled_mode(compiled)

        response = await self.adapter.send(compiled, self.model)

        # Phase 10: Auto-trace TAP response if tracer is attached
        if self._tracer is not None:
            task_id = request.meta.get("task_id", "unknown")
            intent = request.meta.get("intent", "unknown")
            await self._tracer.record_response(
                response, task_id=task_id, trace_id=trace_id, intent=intent
            )

        return response

    async def stream_tap(self, request: TAPRequest) -> AsyncIterator[str]:
        """Stream a TAP request: compile → stream

        Note: Streaming is NOT traced by TAPTracer (partial chunks don't
        form meaningful DPO pairs). Use execute_tap() for traced calls.

        Args:
            request: The TAP request to stream

        Yields:
            Text chunks as they arrive from the model
        """
        compiled = self.compiler.compile(request)
        async for chunk in self.adapter.stream(compiled, self.model):
            yield chunk

    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Simple chat interface — bypasses Compiler optimization

        Directly wraps messages in a CompiledPrompt and sends via the adapter.
        Useful for quick conversational exchanges that don't need TAP compilation.

        Args:
            messages: Chat message list in OpenAI format
            tools: Optional tool definitions for function calling

        Returns:
            {"content": str, "tool_calls": list, "usage": dict, "finish_reason": str}
        """
        compiled = CompiledPrompt(messages=messages, tools=tools or [])
        response = await self.adapter.send(compiled, self.model)
        result = {"content": response.raw_text or ""}
        # Propagate tool_calls from TAPResponse
        result["tool_calls"] = response.tool_calls if response.tool_calls else []
        result.setdefault("usage", response.usage if response.usage else {})
        result.setdefault("finish_reason", response.finish_reason or "stop")
        return result

    def _validate_compiled_mode(self, compiled: CompiledPrompt) -> None:
        """Validate that the compiled prompt mode matches the adapter's expectation.

        Logs an error if the compiled prompt is empty, or a warning if there
        is a mode mismatch that could result in an empty API call (e.g., Mode A
        compiler output with Mode B adapter).

        Args:
            compiled: The compiled prompt to validate
        """
        required = self.adapter.required_mode
        if required == "any":
            return

        actual = compiled.mode
        if actual == "empty":
            logger.error(
                "CompiledPrompt is empty — no messages or system/user prompts. "
                "This will likely result in an API error."
            )
        elif required != actual:
            logger.warning(
                f"Mode mismatch: adapter requires '{required}' "
                f"but compiler produced '{actual}'. "
                f"This may result in an empty or incorrect API call."
            )

    # ===== TAP with retry =====

    async def execute_tap_with_retry(
        self,
        request: TAPRequest,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> TAPResponse:
        """Execute TAP with retry, cost tracking, and circuit breaker integration

        Args:
            request: The TAP request to execute
            max_retries: Maximum number of retries on failure
            retry_delay: Base delay for exponential backoff (seconds)

        Returns:
            TAPResponse with the model's output

        Raises:
            Exception: If all retries fail
        """
        task_id = request.meta.get("task_id", "unknown")
        intent = request.meta.get("intent", "unknown")
        provider_name = f"{self.compiler.__class__.__name__}+{self.adapter.__class__.__name__}"

        # Pre-flight budget check
        if self._circuit_breaker is not None:
            pre_check = self._circuit_breaker.check_before_call(
                estimated_prompt_tokens=request.estimate_prompt_tokens()
            )
            if pre_check.level == "exhausted" and pre_check.max_tokens > 0:
                logger.error(f"TAP call blocked by budget: {pre_check.message}")
                raise RuntimeError(f"Budget exhausted: {pre_check.message}")

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            start = time.time()
            try:
                response = await self.execute_tap(request)
                latency = (time.time() - start) * 1000

                # Record cost
                record = TAPCostRecord(
                    task_id=task_id,
                    intent=intent,
                    provider=provider_name,
                    model=self.model,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    latency_ms=latency,
                    success=True,
                )
                self._cost_tracker.append(record)

                # Record success in circuit breaker
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_model_call(
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        stage=intent,
                        latency_ms=latency,
                    )
                    self._circuit_breaker.record_success()

                logger.info(
                    f"TAP call succeeded: task={task_id} intent={intent} "
                    f"tokens={response.total_tokens} latency={latency:.0f}ms attempt={attempt + 1}"
                )
                return response

            except Exception as e:
                latency = (time.time() - start) * 1000
                last_error = e

                record = TAPCostRecord(
                    task_id=task_id,
                    intent=intent,
                    provider=provider_name,
                    model=self.model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=latency,
                    success=False,
                    error=str(e),
                )
                self._cost_tracker.append(record)

                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_failure(str(e))

                logger.warning(
                    f"TAP call failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2 ** attempt))

        if last_error is None:
            raise RuntimeError(
                "Unexpected state: no error recorded after all retries failed"
            )
        raise last_error

    # ===== Chat with fallback =====

    async def chat_with_fallback(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Chat with fallback provider on failure

        Args:
            messages: Chat message list
            tools: Optional tool definitions for function calling

        Returns:
            {"content": str, "tool_calls": list, "usage": dict, "finish_reason": str}
              from primary or fallback provider

        Raises:
            Exception: If both primary and fallback fail
        """
        try:
            return await self.chat(messages, tools=tools)
        except Exception as e:
            if self._fallback is not None:
                logger.warning(
                    f"Primary provider failed, switching to fallback: {e}"
                )
                return await self._fallback.chat(messages, tools=tools)
            raise

    # ===== Tracer integration (Phase 10) =====

    @property
    def tracer(self) -> TAPTracer | None:
        """Get the attached TAPTracer (Phase 10)"""
        return self._tracer

    def set_tracer(self, tracer: TAPTracer | None) -> None:
        """Attach or detach a TAPTracer for auto-tracing TAP calls (Phase 10)

        When a tracer is attached, execute_tap() automatically records
        request and response traces. This is the easiest way to enable
        self-RL data collection.

        Args:
            tracer: TAPTracer instance, or None to disable tracing
        """
        self._tracer = tracer

    @property
    def fallback_provider(self) -> ModelProvider | None:
        """Get the fallback provider"""
        return self._fallback

    @property
    def has_fallback(self) -> bool:
        """Whether a fallback provider is configured"""
        return self._fallback is not None

    def set_fallback(self, fallback: ModelProvider | None) -> None:
        """Set the fallback provider

        Args:
            fallback: Fallback ModelProvider, or None to clear
        """
        self._fallback = fallback

    @property
    def cost_records(self) -> list[TAPCostRecord]:
        """Get all cost records"""
        return self._cost_tracker.get_all()

    def get_cost_summary(self) -> dict:
        """Get aggregated cost summary by provider"""
        records = self._cost_tracker.get_all()
        total_prompt = sum(r.prompt_tokens for r in records)
        total_completion = sum(r.completion_tokens for r in records)
        by_provider: dict[str, dict] = {}
        for r in records:
            if r.provider not in by_provider:
                by_provider[r.provider] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "errors": 0,
                }
            by_provider[r.provider]["calls"] += 1
            by_provider[r.provider]["prompt_tokens"] += r.prompt_tokens
            by_provider[r.provider]["completion_tokens"] += r.completion_tokens
            if not r.success:
                by_provider[r.provider]["errors"] += 1
        return {
            "total_calls": len(records),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "by_provider": by_provider,
        }

    @property
    def capabilities(self) -> dict:
        """Get adapter capabilities (delegates to self.adapter.capabilities)"""
        return self.adapter.capabilities

    async def close(self) -> None:
        """Close the adapter's persistent resources"""
        await self.adapter.close()
        if self._fallback is not None:
            await self._fallback.close()

    def __repr__(self) -> str:
        tracer_info = f", tracer={self._tracer!r}" if self._tracer else ""
        return (
            f"ModelProvider("
            f"compiler={self.compiler.__class__.__name__}, "
            f"adapter={self.adapter.__class__.__name__}, "
            f"model={self.model!r}{tracer_info})"
        )
