"""teragent.core.adapters.openai_compatible — OpenAI-compatible TAP Adapter

Sends CompiledPrompt to OpenAI-compatible model APIs via HTTP.
Uses Mode A (compiled.messages) for the chat message array.

Features:
  - SSE streaming with auto-fallback to non-streaming on HTTP 400/405/501
  - Optional fake_tools injection for distillation detection (non-streaming only)
  - Connection pooling via persistent httpx.AsyncClient
"""

from __future__ import annotations

import base64
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
# Multimodal validation helpers
# ---------------------------------------------------------------------------

# URL 协议白名单
_VALID_URL_SCHEMES = {"http", "https", "ftp", "ftps"}

# Data URI MIME 类型白名单（图片相关）
_VALID_IMAGE_MIME_TYPES = {
    "image/png", "image/jpeg", "image/gif", "image/webp",
    "image/bmp", "image/svg+xml", "image/tiff",
}


def _is_valid_url(url: str) -> bool:
    """检查 URL 格式是否有效

    Args:
        url: 待检查的 URL

    Returns:
        URL 格式是否有效
    """
    if not url:
        return False
    # 简单检查：包含 :// 且协议在白名单中
    for scheme in _VALID_URL_SCHEMES:
        if url.startswith(f"{scheme}://"):
            return True
    return False


def _validate_data_uri(data_uri: str) -> bool:
    """验证 base64 data URI 格式是否有效

    格式：data:<mime_type>;base64,<base64_data>

    Args:
        data_uri: data URI 字符串

    Returns:
        data URI 是否有效
    """
    if not data_uri.startswith("data:"):
        return False

    # 解析 data URI
    # 格式：data:[<mediatype>][;base64],<data>
    rest = data_uri[5:]  # 去掉 "data:" 前缀

    if ";base64," not in rest:
        return False

    mime_part, _, base64_part = rest.partition(";base64,")

    # 检查 MIME 类型
    if mime_part and mime_part not in _VALID_IMAGE_MIME_TYPES:
        # 不是已知图片类型，但不一定无效（可能是 image/* 或其他类型）
        # 仅当明确不是 image/ 开头时报错
        if not mime_part.startswith("image/"):
            logger.debug(f"Data URI MIME type not in image whitelist: {mime_part}")

    # 检查 base64 数据是否非空
    if not base64_part:
        return False

    # 尝试解码一小部分 base64 数据来验证格式
    try:
        # 只解码前 100 字符，避免大图片的性能问题
        sample = base64_part[:100]
        base64.b64decode(sample, validate=True)
        return True
    except Exception:
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
        multimodal_timeout: float = 600.0,
        ssl_verify: bool | str = True,
        http2_enabled: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._timeout = timeout
        self._extra_headers: dict[str, str] = extra_headers or {}
        self._enable_fake_tools = enable_fake_tools
        self._multimodal_timeout = multimodal_timeout
        self._ssl_verify = ssl_verify
        self._http2_enabled = http2_enabled

        # httpx timeout: short connect/write/pool, long read (streaming chunks)
        # 视频处理需要更长的超时时间
        self._http_timeout = httpx.Timeout(
            connect=30.0,
            read=self._timeout,
            write=30.0,
            pool=30.0,
        )

        # 多模态请求的专用超时配置（视频处理等耗时操作）
        self._multimodal_http_timeout = httpx.Timeout(
            connect=60.0,
            read=self._multimodal_timeout,
            write=60.0,
            pool=60.0,
        )

        # Lazy-initialised persistent HTTP client
        self._http_client: httpx.AsyncClient | None = None
        self._multimodal_http_client: httpx.AsyncClient | None = None

        logger.info(
            f"OpenAICompatibleAdapter: base_url={self.base_url}, "
            f"api_key={'***' if self.api_key else '(empty)'}, "
            f"extra_headers={bool(self._extra_headers)}, "
            f"fake_tools={self._enable_fake_tools}, "
            f"timeout={self._timeout}s, multimodal_timeout={self._multimodal_timeout}s"
        )

    # ----- connection pool management -----

    async def _get_client(self, multimodal: bool = False) -> httpx.AsyncClient:
        """Get or create a persistent httpx connection pool client.

        Connection pool configuration:
          - max_connections=10
          - max_keepalive_connections=5
          - keepalive_expiry=60s
          - HTTP/2 enabled

        Args:
            multimodal: 是否使用多模态专用超时配置（更长的超时时间用于视频处理）
        """
        if multimodal:
            if self._multimodal_http_client is None or self._multimodal_http_client.is_closed:
                self._multimodal_http_client = httpx.AsyncClient(
                    timeout=self._multimodal_http_timeout,
                    limits=httpx.Limits(
                        max_connections=5,
                        max_keepalive_connections=2,
                        keepalive_expiry=120.0,
                    ),
                    http2=self._http2_enabled,
                    verify=self._ssl_verify,
                )
                logger.debug(
                    f"{self.__class__.__name__}: created new multimodal httpx connection pool "
                    f"(timeout={self._multimodal_timeout}s, max_connections=5, http2={self._http2_enabled})"
                )
            return self._multimodal_http_client

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
        """Close the httpx connection pools."""
        for client_attr in ("_http_client", "_multimodal_http_client"):
            client = getattr(self, client_attr, None)
            if client is not None and not client.is_closed:
                await client.aclose()
                logger.debug(f"{self.__class__.__name__}: {client_attr} connection pool closed")
            setattr(self, client_attr, None)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    def __del__(self) -> None:
        for client_attr in ("_http_client", "_multimodal_http_client"):
            client = getattr(self, client_attr, None)
            if client is not None and not client.is_closed:
                logger.warning(
                    f"{self.__class__.__name__}: {client_attr} not closed. "
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

    # ----- Multimodal content validation -----

    @staticmethod
    def _detect_multimodal_in_messages(messages: list[dict]) -> bool:
        """检测消息列表中是否包含多模态内容

        检查消息的 content 字段是否为 OpenAI 格式的 content 数组，
        且包含 image_url、video_url 等多模态类型。

        Args:
            messages: 消息列表

        Returns:
            是否包含多模态内容
        """
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in (
                        "image_url", "video_url"
                    ):
                        return True
        return False

    @staticmethod
    def _validate_multimodal_content(messages: list[dict]) -> list[str]:
        """验证多模态内容的有效性

        在发送前检查所有多模态内容是否有效：
        - image_url: URL 必须非空且格式合理
        - image_base64: base64 数据必须有效
        - video_url: URL 必须非空且格式合理

        Args:
            messages: 消息列表

        Returns:
            警告消息列表（空列表表示全部有效）
        """
        warnings: list[str] = []

        for msg_idx, msg in enumerate(messages):
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            for part_idx, part in enumerate(content):
                if not isinstance(part, dict):
                    continue

                part_type = part.get("type", "")

                # 验证 image_url 内容
                if part_type == "image_url":
                    image_url_obj = part.get("image_url", {})
                    url = image_url_obj.get("url", "")

                    if not url:
                        warnings.append(
                            f"消息[{msg_idx}] 内容[{part_idx}]: "
                            f"image_url 的 URL 为空"
                        )
                    elif url.startswith("data:"):
                        # 验证 base64 data URI
                        if not _validate_data_uri(url):
                            warnings.append(
                                f"消息[{msg_idx}] 内容[{part_idx}]: "
                                f"image_url 的 base64 data URI 格式无效"
                            )
                    elif not _is_valid_url(url):
                        warnings.append(
                            f"消息[{msg_idx}] 内容[{part_idx}]: "
                            f"image_url 的 URL 格式可能无效: {url[:50]}..."
                        )

                # 验证 video_url 内容
                elif part_type == "video_url":
                    video_url_obj = part.get("video_url", {})
                    url = video_url_obj.get("url", "")

                    if not url:
                        warnings.append(
                            f"消息[{msg_idx}] 内容[{part_idx}]: "
                            f"video_url 的 URL 为空"
                        )
                    elif not _is_valid_url(url):
                        warnings.append(
                            f"消息[{msg_idx}] 内容[{part_idx}]: "
                            f"video_url 的 URL 格式可能无效: {url[:50]}..."
                        )

        return warnings

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

    # ----- Model name mapping -----

    _MODEL_NAME_MAP: dict[str, str] = {
        # DeepSeek V4: old names → new canonical names
        "deepseek-chat": "deepseek-v4-flash",
        "deepseek-reasoner": "deepseek-v4-flash",
        # GLM-5.2 model name mappings
        "glm_52": "glm-5.2",
    }

    def _resolve_model_name(self, model: str) -> str:
        """Resolve model name, applying backward-compatible mappings.

        Maps old DeepSeek model names to V4 equivalents:
          deepseek-chat → deepseek-v4-flash
          deepseek-reasoner → deepseek-v4-flash (with thinking mode)

        Maps GLM-5.2 model name aliases:
          glm_52 → glm-5.2
          glm-5.2 → glm-5.2 (identity mapping, already canonical)
        """
        # GLM-5.2 model name mappings
        if model in ("glm-5.2", "glm_52"):
            return "glm-5.2"
        return self._MODEL_NAME_MAP.get(model, model)

    # ----- Core send -----

    async def send(self, compiled: CompiledPrompt, model: str) -> TAPResponse:
        """Send a compiled prompt to the OpenAI-compatible API.

        Strategy: try streaming first to avoid 504 gateway timeouts;
        auto-fallback to non-streaming on HTTP 400/405/501.

        Extended for DeepSeek V4 / GLM-5 / MiniMax M3:
          - Passes compiled.extra as extra_body for thinking mode, cache settings
          - Tracks cache_hit_tokens from response usage
          - Supports 384K max output tokens
          - Multimodal content validation before sending
          - Longer timeout for multimodal requests (video processing)
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

        # 多模态内容验证：发送前检查所有多模态内容的有效性
        mm_warnings = self._validate_multimodal_content(messages)
        for w in mm_warnings:
            logger.warning(f"OpenAICompatibleAdapter: {w}")

        # 检测是否包含多模态内容，选择对应的超时配置
        has_multimodal = self._detect_multimodal_in_messages(messages)

        # Resolve model name (backward compatibility)
        resolved_model = self._resolve_model_name(model)

        # Cap max_tokens at 384K (DeepSeek V4 / M3 output limit)
        max_tokens = min(compiled.max_tokens, 384_000)

        payload: dict = {
            "model": resolved_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }

        # Pass compiled.extra as extra_body for model-specific parameters
        # e.g., thinking mode: {"thinking": {"type": "enabled"}}
        if compiled.extra:
            payload.update(compiled.extra)

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
            # 多模态请求使用专用超时配置
            client = await self._get_client(multimodal=has_multimodal)
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

            # Extract cache hit tokens from usage (DeepSeek V4 returns prompt_cache_hit_tokens)
            cache_hit_tokens = 0
            if usage:
                cache_hit_tokens = usage.get("prompt_cache_hit_tokens", 0)

            return TAPResponse(
                raw_text=raw_text or "",
                usage=usage,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                cache_hit_tokens=cache_hit_tokens,
            )

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

        resolved_model = self._resolve_model_name(model)
        max_tokens = min(compiled.max_tokens, 384_000)

        payload: dict = {
            "model": resolved_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        # Pass compiled.extra as extra_body for model-specific parameters
        if compiled.extra:
            payload.update(compiled.extra)

        # Send compiled.tools if present (for tool calling)
        if compiled.tools:
            payload["tools"] = compiled.tools
            if compiled.tool_choice is not None:
                payload["tool_choice"] = compiled.tool_choice
        # Fake tools injection (distillation detection only works in non-streaming mode)
        elif self._enable_fake_tools and FAKE_TOOLS:
            payload["tools"] = FAKE_TOOLS
            payload["tool_choice"] = "none"

        # 多模态请求使用专用超时配置
        has_multimodal = self._detect_multimodal_in_messages(messages)
        client = await self._get_client(multimodal=has_multimodal)
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
        cache_hit_tokens = usage.get("prompt_cache_hit_tokens", 0) if usage else 0

        return TAPResponse(
            raw_text=content or "",
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            cache_hit_tokens=cache_hit_tokens,
        )

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

        resolved_model = self._resolve_model_name(model)
        max_tokens = min(compiled.max_tokens, 384_000)

        payload: dict = {
            "model": resolved_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }

        # Pass compiled.extra as extra_body for model-specific parameters
        if compiled.extra:
            payload.update(compiled.extra)

        # Send compiled.tools if present (for tool calling)
        if compiled.tools:
            payload["tools"] = compiled.tools
            if compiled.tool_choice is not None:
                payload["tool_choice"] = compiled.tool_choice
        elif self._enable_fake_tools and FAKE_TOOLS:
            payload["tools"] = FAKE_TOOLS
            payload["tool_choice"] = "none"

        # 多模态请求使用专用超时配置
        has_multimodal = self._detect_multimodal_in_messages(messages)
        client = await self._get_client(multimodal=has_multimodal)
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
            "max_context_tokens": 1_000_000,  # Supports V4/M3 1M and GLM 200K
            "multimodal": True,  # 支持多模态内容发送
        }

    @property
    def required_mode(self) -> str:
        """OpenAI-compatible API expects Mode A (messages)"""
        return "messages"


# Register with TAPAdapterRegistry
TAPAdapterRegistry.register("openai_compatible", OpenAICompatibleAdapter)
