"""teragent.core.adapters.openai_compatible — OpenAI-compatible TAP Adapter

Sends CompiledPrompt to OpenAI-compatible model APIs via HTTP.
Uses Mode A (compiled.messages) for the chat message array.

Features:
  - SSE streaming with auto-fallback to non-streaming on HTTP 400/405/501
  - Optional fake_tools injection for distillation detection (non-streaming only)
  - Connection pooling via persistent httpx.AsyncClient
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

from teragent.core.adapter import TAPAdapter, TAPAdapterRegistry
from teragent.core.tap import CompiledPrompt, TAPResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fake-tools definitions (duplicated here to avoid cross-package import)
# ---------------------------------------------------------------------------

FAKE_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "internal_model_profiling_snapshot",
            "description": (
                "Take a snapshot of the current model internal state for profiling. "
                "Do not call this unless explicitly asked to debug the model."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "_system_diagnostic_dump",
            "description": (
                "Internal system diagnostic tool. Dumps all system state including "
                "conversation history and model weights metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "depth": {
                        "type": "integer",
                        "description": "How deep to dump",
                        "default": 3,
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export_training_data",
            "description": (
                "Export recent interactions as training data format for model "
                "fine-tuning pipeline."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["jsonl", "parquet"],
                    }
                },
            },
        },
    },
]

FAKE_TOOL_NAMES: set[str] = {t["function"]["name"] for t in FAKE_TOOLS}


def detect_fake_tool_call(response_data: dict) -> bool:
    """Detect whether the LLM attempted to call a fake tool (distillation detection)."""
    choices = response_data.get("choices", [])
    if not choices:
        return False

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls", [])

    for tc in tool_calls:
        func_name = tc.get("function", {}).get("name", "")
        if func_name in FAKE_TOOL_NAMES:
            logger.warning(
                f"[!] DISTILLATION DETECTION: LLM attempted to call fake tool "
                f"'{func_name}'. This indicates the model may be trying to "
                f"exfiltrate its internal state or training data."
            )
            return True
    return False


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class OpenAICompatibleAdapter(TAPAdapter):
    """OpenAI-compatible API adapter for TAP

    Sends CompiledPrompt via the /chat/completions endpoint.
    Uses Mode A: compiled.messages as the payload messages array.

    Supports:
      - SSE streaming with auto-fallback to non-streaming
      - Optional fake_tools injection for distillation detection
      - Persistent httpx.AsyncClient with connection pooling
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 300.0,
        extra_headers: dict | None = None,
        enable_fake_tools: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._timeout = timeout
        self._extra_headers: dict[str, str] = extra_headers or {}
        self._enable_fake_tools = enable_fake_tools

        # httpx timeout: short connect/write/pool, long read (streaming chunks)
        self._http_timeout = httpx.Timeout(
            connect=30.0,
            read=self._timeout,
            write=30.0,
            pool=30.0,
        )

        # Lazy-initialised persistent HTTP client
        self._http_client: httpx.AsyncClient | None = None

        logger.info(
            f"OpenAICompatibleAdapter: base_url={self.base_url}, "
            f"api_key={'***' if self.api_key else '(empty)'}, "
            f"extra_headers={bool(self._extra_headers)}, "
            f"fake_tools={self._enable_fake_tools}, timeout={self._timeout}s"
        )

    # ----- connection pool management -----

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create a persistent httpx connection pool client.

        Connection pool configuration:
          - max_connections=10
          - max_keepalive_connections=5
          - keepalive_expiry=60s
          - HTTP/2 enabled
        """
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=self._http_timeout,
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                    keepalive_expiry=60.0,
                ),
                http2=True,
            )
            logger.debug(
                f"{self.__class__.__name__}: created new httpx connection pool "
                f"(max_connections=10, max_keepalive=5, keepalive_expiry=60s, http2=True)"
            )
        return self._http_client

    async def close(self) -> None:
        """Close the httpx connection pool."""
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            logger.debug(f"{self.__class__.__name__}: httpx connection pool closed")
        self._http_client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    def __del__(self) -> None:
        if self._http_client is not None and not self._http_client.is_closed:
            logger.warning(
                f"{self.__class__.__name__}: httpx client not closed. "
                f"Call 'await adapter.close()' or use 'async with' to avoid resource leaks."
            )

    # ----- helper methods -----

    def _build_headers(self) -> dict[str, str]:
        """Build common request headers."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self._extra_headers)
        return headers

    @staticmethod
    def _safe_get_choice(chunk: dict, index: int = 0) -> dict:
        """Safely extract choices[index], tolerant of empty/missing/malformed data.

        Some compatible APIs (e.g. StepFun) may return choices:[] or omit the
        choices key entirely; direct [0] access would raise IndexError.
        """
        choices = chunk.get("choices")
        if not choices or not isinstance(choices, list):
            return {}
        if index >= len(choices):
            return {}
        return choices[index] or {}

    @staticmethod
    def _categorize_error(exc: Exception) -> str:
        """Categorise an exception into a readable error class."""
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status == 429:
                return "rate_limited"
            if status == 401:
                return "authentication"
            if status == 403:
                return "forbidden"
            if status >= 500:
                return "server_error"
            if status >= 400:
                return "client_error"
            return "http_unknown"
        if isinstance(exc, httpx.TimeoutException):
            return "timeout"
        if isinstance(exc, httpx.ConnectError):
            return "connection"
        return "unknown"

    # ----- core send -----

    async def send(self, compiled: CompiledPrompt, model: str) -> TAPResponse:
        """Send a compiled prompt to the OpenAI-compatible API.

        Strategy: try streaming first to avoid 504 gateway timeouts;
        auto-fallback to non-streaming on HTTP 400/405/501.
        """
        url = f"{self.base_url}/chat/completions"
        headers = self._build_headers()

        # Handle both Mode A (messages list) and Mode B (system_prompt + user_message)
        if compiled.mode == "system_user" and not compiled.messages:
            messages = []
            if compiled.system_prompt:
                messages.append({"role": "system", "content": compiled.system_prompt})
            messages.append({"role": "user", "content": compiled.user_message or ""})
        else:
            messages = compiled.messages

        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": compiled.max_tokens,
            "stream": True,
        }

        # Send compiled.tools if present (for tool calling)
        if compiled.tools:
            payload["tools"] = compiled.tools
            if compiled.tool_choice is not None:
                payload["tool_choice"] = compiled.tool_choice
        # Optional fake_tools injection for distillation detection (only when no real tools)
        elif self._enable_fake_tools and FAKE_TOOLS:
            payload["tools"] = FAKE_TOOLS
            payload["tool_choice"] = "none"

        # Use streaming to avoid 504 gateway timeouts
        collected: list[str] = []
        usage: dict = {}
        last_chunk_data: dict = {}
        # Accumulate tool_call deltas from streaming chunks
        # (streaming responses use delta.tool_calls, not message.tool_calls)
        tool_call_accumulators: dict[int, dict] = {}
        try:
            client = await self._get_client()
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            # Detect API error objects
                            if chunk.get("error"):
                                err_obj = chunk["error"]
                                err_msg = err_obj.get("message", str(err_obj)) if isinstance(err_obj, dict) else str(err_obj)
                                logger.error(f"LLM stream error from API: {err_msg}")
                                raise RuntimeError(f"API stream error: {err_msg}")
                            # Extract text content
                            delta = self._safe_get_choice(chunk).get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                collected.append(content)
                            # Accumulate tool_call deltas
                            delta_tc_list = delta.get("tool_calls", [])
                            for tc_delta in delta_tc_list:
                                idx = tc_delta.get("index", 0)
                                if idx not in tool_call_accumulators:
                                    tool_call_accumulators[idx] = {
                                        "id": "", "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    }
                                acc = tool_call_accumulators[idx]
                                if tc_delta.get("id"):
                                    acc["id"] = tc_delta["id"]
                                if tc_delta.get("type"):
                                    acc["type"] = tc_delta["type"]
                                fn = tc_delta.get("function", {})
                                if fn.get("name"):
                                    acc["function"]["name"] = fn["name"]
                                if fn.get("arguments"):
                                    acc["function"]["arguments"] += fn["arguments"]
                            # Extract usage (some APIs return it in the last chunk)
                            if chunk.get("usage"):
                                usage = chunk["usage"]
                            # Track last chunk for finish_reason
                            last_chunk_data = chunk
                        except json.JSONDecodeError:
                            continue

            raw_text = "".join(collected)

            # Build tool_calls from accumulated deltas and extract finish_reason
            tool_calls = [tool_call_accumulators[i] for i in sorted(tool_call_accumulators)]
            finish_reason = "stop"
            if last_chunk_data:
                choice = self._safe_get_choice(last_chunk_data)
                finish_reason = choice.get("finish_reason", "stop") or "stop"

            return TAPResponse(raw_text=raw_text or "", usage=usage, tool_calls=tool_calls, finish_reason=finish_reason)

        except httpx.HTTPStatusError as e:
            # Streaming not supported — fall back to non-streaming
            if e.response.status_code in (400, 405, 501):
                logger.warning(
                    f"Streaming not supported ({e.response.status_code}), "
                    f"falling back to non-streaming request"
                )
                return await self._send_non_streaming(compiled, model)
            category = self._categorize_error(e)
            logger.error(
                f"LLM API Error [{category}]: {e.response.status_code} - "
                f"{e.response.text[:500]}"
            )
            raise
        except Exception as e:
            category = self._categorize_error(e)
            logger.error(f"LLM Request Failed [{category}]: {type(e).__name__}: {e}")
            raise

    async def _send_non_streaming(
        self, compiled: CompiledPrompt, model: str
    ) -> TAPResponse:
        """Non-streaming fallback for send()."""
        url = f"{self.base_url}/chat/completions"
        headers = self._build_headers()

        # Handle both Mode A and Mode B
        if compiled.mode == "system_user" and not compiled.messages:
            messages = []
            if compiled.system_prompt:
                messages.append({"role": "system", "content": compiled.system_prompt})
            messages.append({"role": "user", "content": compiled.user_message or ""})
        else:
            messages = compiled.messages

        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": compiled.max_tokens,
        }

        # Send compiled.tools if present (for tool calling)
        if compiled.tools:
            payload["tools"] = compiled.tools
            if compiled.tool_choice is not None:
                payload["tool_choice"] = compiled.tool_choice
        # Fake tools injection (distillation detection only works in non-streaming mode)
        elif self._enable_fake_tools and FAKE_TOOLS:
            payload["tools"] = FAKE_TOOLS
            payload["tool_choice"] = "none"

        client = await self._get_client()
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        # Distillation detection (only when fake tools are enabled)
        if self._enable_fake_tools and detect_fake_tool_call(data):
            logger.warning(
                "[!] FAKE TOOL INVOCATION DETECTED! "
                "Possible unauthorized model distillation."
            )

        # Detect non-streaming API errors
        if data.get("error"):
            err_obj = data["error"]
            err_msg = err_obj.get("message", str(err_obj)) if isinstance(err_obj, dict) else str(err_obj)
            logger.error(f"LLM API error response: {err_msg}")
            raise RuntimeError(f"API error: {err_msg}")

        message = self._safe_get_choice(data).get("message", {})
        content = message.get("content") or ""

        # Extract tool_calls from API response
        tool_calls = message.get("tool_calls", [])
        finish_reason = data.get("choices", [{}])[0].get("finish_reason", "stop") if data.get("choices") else "stop"

        # If content is empty but tool_calls exist, extract summary text
        if not content and tool_calls:
            parts = []
            for tc in tool_calls:
                args = tc.get("function", {}).get("arguments", "")
                name = tc.get("function", {}).get("name", "unknown")
                if args and args not in ("{}", ""):
                    parts.append(f"[Tool call: {name}] {args}")
            if parts:
                content = "\n".join(parts)

        usage = data.get("usage", {})
        return TAPResponse(raw_text=content or "", usage=usage, tool_calls=tool_calls, finish_reason=finish_reason)

    # ----- core stream -----

    async def stream(self, compiled: CompiledPrompt, model: str) -> AsyncIterator[str]:
        """Stream a compiled prompt to the OpenAI-compatible API via SSE.

        Yields text chunks as they arrive from the model.
        """
        url = f"{self.base_url}/chat/completions"
        headers = self._build_headers()

        # Handle both Mode A and Mode B
        if compiled.mode == "system_user" and not compiled.messages:
            messages = []
            if compiled.system_prompt:
                messages.append({"role": "system", "content": compiled.system_prompt})
            messages.append({"role": "user", "content": compiled.user_message or ""})
        else:
            messages = compiled.messages

        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": compiled.max_tokens,
            "stream": True,
        }

        # Send compiled.tools if present (for tool calling)
        if compiled.tools:
            payload["tools"] = compiled.tools
            if compiled.tool_choice is not None:
                payload["tool_choice"] = compiled.tool_choice
        elif self._enable_fake_tools and FAKE_TOOLS:
            payload["tools"] = FAKE_TOOLS
            payload["tool_choice"] = "none"

        client = await self._get_client()
        async with client.stream(
            "POST", url, json=payload, headers=headers
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if chunk.get("error"):
                            err_obj = chunk["error"]
                            err_msg = err_obj.get("message", str(err_obj)) if isinstance(err_obj, dict) else str(err_obj)
                            logger.error(f"LLM stream error from API: {err_msg}")
                            break
                        delta = self._safe_get_choice(chunk).get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    # ----- capabilities -----

    @property
    def capabilities(self) -> dict:
        return {
            "streaming": True,
            "tool_calling": True,  # Only distillation detection via fake tools in non-streaming mode
            "max_context_tokens": 128000,
        }

    @property
    def required_mode(self) -> str:
        """OpenAI-compatible API expects Mode A (messages)"""
        return "messages"


# Register with TAPAdapterRegistry
TAPAdapterRegistry.register("openai_compatible", OpenAICompatibleAdapter)
