"""teragent.core.adapters.anthropic_native — Anthropic native TAP Adapter

Sends CompiledPrompt to the Anthropic Messages API via HTTP.
Uses Mode B (compiled.system_prompt + compiled.user_message) for the
Anthropic-specific request format where system is a top-level field.

Features:
  - Anthropic-specific headers (x-api-key, anthropic-version)
  - SSE streaming with Anthropic event types
      (content_block_delta, message_start, message_delta, etc.)
  - Auto-fallback to non-streaming on HTTP 400/405/501 (send() only; stream() raises on error)
  - Usage parsing from message_start and message_delta events
  - Distillation detection via fake tools in Anthropic tool format (non-streaming only)
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
# Fake-tools definitions (defined in OpenAI format, converted to Anthropic format at runtime)
# ---------------------------------------------------------------------------

_FAKE_TOOLS_OPENAI: list[dict] = [
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

FAKE_TOOL_NAMES: set[str] = {t["function"]["name"] for t in _FAKE_TOOLS_OPENAI}


def _detect_anthropic_fake_tool_call(data: dict) -> bool:
    """Detect fake tool invocation in an Anthropic response (distillation detection)."""
    content_blocks = data.get("content", [])
    for block in content_blocks:
        if block.get("type") == "tool_use":
            tool_name = block.get("name", "")
            if tool_name in FAKE_TOOL_NAMES:
                logger.warning(
                    f"[!] DISTILLATION DETECTION: Anthropic model attempted to call "
                    f"fake tool '{tool_name}'. This indicates the model may be trying "
                    f"to exfiltrate its internal state."
                )
                return True
    return False


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class AnthropicNativeAdapter(TAPAdapter):
    """Anthropic native API adapter for TAP

    Sends CompiledPrompt via the /messages endpoint.
    Uses Mode B: compiled.system_prompt as payload["system"],
                 compiled.user_message as messages[0]["content"].

    Supports:
      - Anthropic-specific SSE event types
      - Auto-fallback to non-streaming
      - Usage parsing from message_start + message_delta
      - Distillation detection via fake tools
      - Persistent httpx.AsyncClient with connection pooling
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 300.0,
        enable_fake_tools: bool = False,
        ssl_verify: bool | str = True,
        http2_enabled: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._timeout = timeout
        self._enable_fake_tools = enable_fake_tools
        self._ssl_verify = ssl_verify
        self._http2_enabled = http2_enabled

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
            f"AnthropicNativeAdapter: base_url={self.base_url}, "
            f"api_key={'***' if self.api_key else '(empty)'}, "
            f"fake_tools={self._enable_fake_tools}, "
            f"timeout={self._timeout}s"
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
                http2=self._http2_enabled,
                verify=self._ssl_verify,
            )
            logger.debug(
                f"{self.__class__.__name__}: created new httpx connection pool "
                f"(max_connections=10, max_keepalive=5, keepalive_expiry=60s, http2={self._http2_enabled})"
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
        """Build Anthropic-specific request headers."""
        headers = {
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    @staticmethod
    def _convert_tools_to_anthropic(tools: list[dict]) -> list[dict]:
        """Convert OpenAI-format tools to Anthropic format.

        OpenAI format:
            {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

        Anthropic format:
            {"name": "...", "description": "...", "input_schema": {...}}
        """
        anthropic_tools: list[dict] = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                anthropic_tool = {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
                if anthropic_tool["name"]:
                    anthropic_tools.append(anthropic_tool)
        return anthropic_tools

    def _build_fake_tools_payload(self) -> list[dict]:
        """Build fake tools in Anthropic format for distillation detection."""
        return self._convert_tools_to_anthropic(_FAKE_TOOLS_OPENAI)

    @staticmethod
    def _convert_tool_choice_to_anthropic(tool_choice: dict | str) -> dict | str:
        """Convert OpenAI-format tool_choice to Anthropic format.

        OpenAI format:
            "auto" | "none" | {"type": "function", "function": {"name": "..."}}
        Anthropic format:
            {"type": "auto"} | {"type": "any"} | {"type": "tool", "name": "..."}
        """
        if isinstance(tool_choice, str):
            if tool_choice == "auto":
                return {"type": "auto"}
            elif tool_choice == "none":
                # Anthropic API 不支持 {"type": "none"}，不传 tool_choice 即可禁用
                return {"type": "auto"}  # 返回 auto 但调用方应不传 tools 参数
            return {"type": "auto"}
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                name = tool_choice.get("function", {}).get("name", "")
                if name:
                    return {"type": "tool", "name": name}
            elif tool_choice.get("type") == "auto":
                return {"type": "auto"}
            elif tool_choice.get("type") == "required":
                return {"type": "any"}
        return {"type": "auto"}

    # ----- core send -----

    async def send(self, compiled: CompiledPrompt, model: str) -> TAPResponse:
        """Send a compiled prompt to the Anthropic Messages API.

        Uses Mode B: compiled.system_prompt + compiled.user_message.
        Strategy: try streaming first to avoid 504 gateway timeouts;
        auto-fallback to non-streaming on HTTP 400/405/501.
        """
        url = f"{self.base_url}/messages"
        headers = self._build_headers()

        # Build payload based on compiled prompt mode
        payload: dict = {
            "model": model,
            "max_tokens": compiled.max_tokens,
            "stream": True,
        }

        # Mode B: system_prompt + user_message
        if compiled.mode == "system_user":
            if compiled.system_prompt:
                payload["system"] = compiled.system_prompt
            payload["messages"] = [{"role": "user", "content": compiled.user_message}]
        else:
            # Mode A: messages list — extract system message and use rest as messages
            messages = list(compiled.messages)
            system_content = ""
            non_system_messages = []
            for msg in messages:
                if msg.get("role") == "system":
                    system_content += msg.get("content", "") + "\n"
                else:
                    non_system_messages.append(msg)
            if system_content.strip():
                payload["system"] = system_content.strip()
            if non_system_messages:
                payload["messages"] = non_system_messages
            else:
                payload["messages"] = [{"role": "user", "content": compiled.user_message or "Please proceed."}]

        # Send compiled.tools if present (for tool calling)
        if compiled.tools:
            anthropic_tools = self._convert_tools_to_anthropic(compiled.tools)
            if anthropic_tools:
                payload["tools"] = anthropic_tools
                if compiled.tool_choice is not None:
                    payload["tool_choice"] = self._convert_tool_choice_to_anthropic(compiled.tool_choice)
        # Inject fake tools for distillation detection (only when no real tools)
        elif self._enable_fake_tools:
            fake_tools = self._build_fake_tools_payload()
            if fake_tools:
                payload["tools"] = fake_tools

        # Use streaming to avoid 504 gateway timeouts
        collected: list[str] = []
        tool_use_blocks: list[dict] = []  # Collect tool_use blocks for distillation detection
        # Accumulate tool_use input arguments from streaming deltas
        tool_input_accumulators: dict[int, str] = {}  # block_index -> accumulated input_json_delta
        tool_block_map: dict[int, dict] = {}  # block_index -> {id, name}
        usage: dict = {}
        actual_stop_reason: str = ""
        try:
            client = await self._get_client()
            async with client.stream(
                "POST", url, json=payload, headers=headers
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            event = json.loads(data_str)
                            # Detect API error objects
                            if event.get("type") == "error":
                                err_msg = event.get("error", {}).get(
                                    "message", str(event)
                                )
                                logger.error(f"Anthropic stream error: {err_msg}")
                                raise RuntimeError(
                                    f"Anthropic API stream error: {err_msg}"
                                )

                            event_type = event.get("type", "")

                            # Extract text content from content_block_delta
                            if event_type == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text = delta.get("text", "")
                                    if text:
                                        collected.append(text)
                                elif delta.get("type") == "input_json_delta":
                                    # Accumulate tool input arguments
                                    idx = event.get("index", 0)
                                    partial_json = delta.get("partial_json", "")
                                    if idx not in tool_input_accumulators:
                                        tool_input_accumulators[idx] = ""
                                    tool_input_accumulators[idx] += partial_json

                            # Track tool_use blocks for distillation detection + tool_calls
                            if event_type == "content_block_start":
                                block = event.get("content_block", {})
                                idx = event.get("index", 0)
                                if block.get("type") == "tool_use":
                                    tool_use_blocks.append(block)
                                    tool_block_map[idx] = {
                                        "id": block.get("id", ""),
                                        "name": block.get("name", ""),
                                    }

                            # Extract usage from message_start (input tokens)
                            if (
                                event_type == "message_start"
                                and event.get("message", {}).get("usage")
                            ):
                                msg_usage = event["message"]["usage"]
                                usage["prompt_tokens"] = msg_usage.get(
                                    "input_tokens", 0
                                )

                            # Extract usage from message_delta (output tokens)
                            if event_type == "message_delta" and event.get("usage"):
                                usage["completion_tokens"] = event["usage"].get(
                                    "output_tokens", 0
                                )

                            # Extract stop_reason from message_delta
                            if event_type == "message_delta":
                                delta = event.get("delta", {})
                                if delta.get("stop_reason"):
                                    actual_stop_reason = delta["stop_reason"]

                        except json.JSONDecodeError:
                            continue

            # Distillation detection in streaming path
            if self._enable_fake_tools:
                for block in tool_use_blocks:
                    tool_name = block.get("name", "")
                    if tool_name in FAKE_TOOL_NAMES:
                        logger.warning(
                            f"[!] DISTILLATION DETECTION (streaming): Anthropic model "
                            f"attempted to call fake tool '{tool_name}'. This indicates "
                            f"the model may be trying to exfiltrate its internal state."
                        )

            raw_text = "".join(collected)

            # Build tool_calls from accumulated tool_use blocks + input arguments
            tool_calls = []
            for idx in sorted(tool_block_map):
                block_info = tool_block_map[idx]
                input_json_str = tool_input_accumulators.get(idx, "{}")
                try:
                    input_obj = json.loads(input_json_str) if input_json_str else {}
                except json.JSONDecodeError:
                    input_obj = {}
                tool_calls.append({
                    "id": block_info.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block_info.get("name", ""),
                        "arguments": json.dumps(input_obj),
                    },
                })

            # Map Anthropic stop_reason to standard finish_reason
            _FINISH_REASON_MAP = {
                "end_turn": "stop",
                "max_tokens": "length",
                "stop_sequence": "stop",
                "tool_use": "tool_calls",
            }
            finish_reason = _FINISH_REASON_MAP.get(actual_stop_reason, actual_stop_reason or "stop")

            return TAPResponse(raw_text=raw_text or "", usage=usage, tool_calls=tool_calls, finish_reason=finish_reason)

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (400, 405, 501):
                logger.warning(
                    f"Streaming not supported ({e.response.status_code}), "
                    f"falling back to non-streaming"
                )
                return await self._send_non_streaming(compiled, model)
            logger.error(
                f"Anthropic API Error: {e.response.status_code} - {e.response.text}"
            )
            raise
        except Exception as e:
            logger.error(f"Anthropic Request Failed: {e}")
            raise

    async def _send_non_streaming(
        self, compiled: CompiledPrompt, model: str
    ) -> TAPResponse:
        """Non-streaming fallback for send()."""
        url = f"{self.base_url}/messages"
        headers = self._build_headers()

        # Build payload based on compiled prompt mode
        payload: dict = {
            "model": model,
            "max_tokens": compiled.max_tokens,
        }

        # Mode B: system_prompt + user_message
        if compiled.mode == "system_user":
            if compiled.system_prompt:
                payload["system"] = compiled.system_prompt
            payload["messages"] = [{"role": "user", "content": compiled.user_message}]
        else:
            # Mode A: messages list
            messages = list(compiled.messages)
            system_content = ""
            non_system_messages = []
            for msg in messages:
                if msg.get("role") == "system":
                    system_content += msg.get("content", "") + "\n"
                else:
                    non_system_messages.append(msg)
            if system_content.strip():
                payload["system"] = system_content.strip()
            if non_system_messages:
                payload["messages"] = non_system_messages
            else:
                payload["messages"] = [{"role": "user", "content": compiled.user_message or "Please proceed."}]

        # Send compiled.tools if present (for tool calling)
        if compiled.tools:
            anthropic_tools = self._convert_tools_to_anthropic(compiled.tools)
            if anthropic_tools:
                payload["tools"] = anthropic_tools
                if compiled.tool_choice is not None:
                    payload["tool_choice"] = self._convert_tool_choice_to_anthropic(compiled.tool_choice)
        # Inject fake tools for distillation detection (only when no real tools)
        elif self._enable_fake_tools:
            fake_tools = self._build_fake_tools_payload()
            if fake_tools:
                payload["tools"] = fake_tools

        client = await self._get_client()
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        # Distillation detection (only when fake tools are enabled)
        if self._enable_fake_tools and _detect_anthropic_fake_tool_call(data):
            logger.warning(
                "[!] FAKE TOOL INVOCATION DETECTED on Anthropic! "
                "Possible unauthorized model distillation."
            )

        # Extract text content from response
        content = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )

        # Extract tool_use blocks from response
        tool_calls = []
        for block in data.get("content", []):
            if block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })

        usage = {
            "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
            "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
        }

        stop_reason = data.get("stop_reason", "stop")
        _FINISH_REASON_MAP_NS = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }
        finish_reason = _FINISH_REASON_MAP_NS.get(stop_reason, stop_reason or "stop")

        return TAPResponse(raw_text=content, usage=usage, tool_calls=tool_calls, finish_reason=finish_reason)

    # ----- core stream -----

    async def stream(self, compiled: CompiledPrompt, model: str) -> AsyncIterator[str]:
        """Stream a compiled prompt to the Anthropic Messages API via SSE.

        Uses Mode B: compiled.system_prompt + compiled.user_message.
        Yields raw text chunks (str) from content_block_delta events.

        NOTE: This method yields plain str, NOT StreamEvent objects. It is
        therefore NOT compatible with StreamingToolExecutor, which expects
        an AsyncIterator[StreamEvent]. Tool calling in this adapter is
        handled by send() (non-streaming) only.
        """
        url = f"{self.base_url}/messages"
        headers = self._build_headers()

        payload: dict = {
            "model": model,
            "max_tokens": compiled.max_tokens,
            "stream": True,
        }

        # Mode B: system_prompt + user_message
        if compiled.mode == "system_user":
            if compiled.system_prompt:
                payload["system"] = compiled.system_prompt
            payload["messages"] = [{"role": "user", "content": compiled.user_message}]
        else:
            # Mode A: messages list
            messages = list(compiled.messages)
            system_content = ""
            non_system_messages = []
            for msg in messages:
                if msg.get("role") == "system":
                    system_content += msg.get("content", "") + "\n"
                else:
                    non_system_messages.append(msg)
            if system_content.strip():
                payload["system"] = system_content.strip()
            if non_system_messages:
                payload["messages"] = non_system_messages
            else:
                payload["messages"] = [{"role": "user", "content": compiled.user_message or "Please proceed."}]

        client = await self._get_client()
        async with client.stream(
            "POST", url, json=payload, headers=headers
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        event = json.loads(data_str)
                        # Detect API error objects
                        if event.get("type") == "error":
                            err_msg = event.get("error", {}).get(
                                "message", str(event)
                            )
                            logger.error(f"Anthropic stream error: {err_msg}")
                            # 修复 H15: 抛出异常而非静默 break，让调用者知道流被错误终止
                            raise RuntimeError(f"Anthropic stream error: {err_msg}")

                        event_type = event.get("type", "")
                        if event_type == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    yield text
                    except json.JSONDecodeError:
                        continue

    # ----- capabilities -----

    @property
    def capabilities(self) -> dict:
        """Report adapter capabilities.

        Note: ``streaming_tool_calling`` is intentionally absent because
        stream() yields raw str chunks, not StreamEvent objects.  Tool
        calling is only available via the non-streaming send() path.
        StreamingToolExecutor.can_stream_with_tools() will therefore
        return False for this adapter, which is correct.
        """
        return {
            "streaming": True,
            "tool_calling": True,  # Only distillation detection via fake tools in non-streaming mode
            "max_context_tokens": 200000,
        }

    @property
    def required_mode(self) -> str:
        """Anthropic native API expects Mode B (system_user)"""
        return "system_user"


# Register with TAPAdapterRegistry
TAPAdapterRegistry.register("anthropic_native", AnthropicNativeAdapter)
