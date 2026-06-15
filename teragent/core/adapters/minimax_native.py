"""teragent.core.adapters.minimax_native — MiniMax M3 Native TAP Adapter

Extends OpenAICompatibleAdapter with MiniMax-specific features while
delegating all standard OpenAI-compatible operations to the parent class.

MiniMax M3 has TWO API interfaces:
  1. OpenAI-compatible:  https://api.minimaxi.com/v1/chat/completions
     — Standard chat completions format for pure-text quick requests
  2. Anthropic-compatible: https://api.minimaxi.com/anthropic/v1/messages
     — Native thinking, video, count_tokens, reasoning_split

This adapter automatically routes requests to the appropriate interface
based on the compiled prompt's features:
  - Multimodal requests (image/video) → Anthropic interface
  - Thinking mode enabled            → Anthropic interface
  - Token estimation needed          → Anthropic interface
  - Pure text quick requests         → OpenAI interface

MiniMax-specific enhancements:
  1. Group ID handling — MiniMax API requires a group_id for some endpoints
  2. Video content encoding — Enhanced video_url processing for M3 native multimodal
  3. Desktop operation API — Dedicated endpoint for M3 desktop automation
  4. Response parsing — MiniMax-specific usage fields and metadata
  5. Rate limit tracking — Parse X-RateLimit-* headers for cost-aware routing
  6. Token billing tracking — MiniMax-specific billing model with cache awareness
  7. Anthropic dual-interface — Automatic routing between OpenAI & Anthropic formats
  8. Thinking mode — Extended thinking via Anthropic interface
  9. reasoning_split — Structured reasoning details via Anthropic interface
  10. count_tokens — Token counting via Anthropic /messages/count_tokens endpoint
"""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator

import httpx

from teragent.core.adapter import TAPAdapterRegistry
from teragent.core.adapters.openai_compatible import OpenAICompatibleAdapter
from teragent.core.tap import CompiledPrompt, TAPResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MiniMax-specific constants
# ---------------------------------------------------------------------------

# MiniMax default base URL (OpenAI-compatible)
MINIMAX_DEFAULT_BASE_URL = "https://api.minimax.chat/v1"

# MiniMax Anthropic-compatible base URL
MINIMAX_ANTHROPIC_DEFAULT_BASE_URL = "https://api.minimaxi.com/anthropic/v1"

# MiniMax desktop operation endpoint (relative to base_url)
_DESKTOP_ENDPOINT = "/desktop/operations"

# MiniMax-specific rate limit response headers
_RATE_LIMIT_HEADERS = {
    "X-RateLimit-Limit",
    "X-RateLimit-Remaining",
    "X-RateLimit-Reset",
}

# MiniMax-specific usage field names (beyond standard OpenAI fields)
_MINIMAX_USAGE_FIELDS = (
    "prompt_cache_hit_tokens",   # Cache hit tokens (same key as DeepSeek V4)
    "prompt_cache_miss_tokens",  # Cache miss tokens (MiniMax-specific)
    "total_tokens",              # Standard field, but MiniMax may calculate differently
)

# Anthropic API version header for MiniMax's Anthropic-compatible endpoint
_ANTHROPIC_VERSION = "2023-06-01"


class MiniMaxRateLimitInfo:
    """Parsed MiniMax rate limit information from response headers.

    Attributes:
        limit: Maximum requests allowed in the current window
        remaining: Remaining requests in the current window
        reset: Timestamp when the rate limit window resets (Unix epoch)
        last_updated: When this info was last updated
    """

    __slots__ = ("limit", "remaining", "reset", "last_updated")

    def __init__(
        self,
        limit: int = 0,
        remaining: int = 0,
        reset: float = 0.0,
    ) -> None:
        self.limit = limit
        self.remaining = remaining
        self.reset = reset
        self.last_updated: float = time.time()

    def update_from_headers(self, headers: httpx.Headers) -> None:
        """Update rate limit info from HTTP response headers.

        Parses X-RateLimit-Limit, X-RateLimit-Remaining, and
        X-RateLimit-Reset headers if present. Gracefully ignores
        missing or malformed headers.

        Args:
            headers: HTTP response headers
        """
        try:
            val = headers.get("X-RateLimit-Limit")
            if val is not None:
                self.limit = int(val)

            val = headers.get("X-RateLimit-Remaining")
            if val is not None:
                self.remaining = int(val)

            val = headers.get("X-RateLimit-Reset")
            if val is not None:
                self.reset = float(val)

            self.last_updated = time.time()
        except (ValueError, TypeError) as e:
            logger.debug(f"MiniMaxRateLimitInfo: failed to parse rate limit headers: {e}")

    @property
    def is_exhausted(self) -> bool:
        """Whether the rate limit is exhausted (no remaining requests)."""
        return self.remaining <= 0 and self.limit > 0

    @property
    def reset_in_seconds(self) -> float:
        """Seconds until the rate limit window resets."""
        if self.reset <= 0:
            return 0.0
        return max(0.0, self.reset - time.time())

    def __repr__(self) -> str:
        return (
            f"MiniMaxRateLimitInfo("
            f"limit={self.limit}, "
            f"remaining={self.remaining}, "
            f"reset={self.reset}, "
            f"exhausted={self.is_exhausted})"
        )


class MiniMaxNativeAdapter(OpenAICompatibleAdapter):
    """MiniMax M3 Native Adapter

    Extends OpenAICompatibleAdapter with MiniMax-specific enhancements,
    including dual-interface support (OpenAI + Anthropic compatible).

    Interface routing logic:
      - Multimodal requests (image/video) → Anthropic interface
      - Thinking mode enabled             → Anthropic interface
      - Token estimation needed           → Anthropic interface
      - Pure text quick requests          → OpenAI interface

    MiniMax-specific features:
      - Group ID handling for MiniMax API endpoints
      - Enhanced video content processing for M3 native multimodal
      - Desktop operation API endpoint
      - MiniMax-specific response field parsing
      - Rate limit header tracking for cost-aware routing
      - Token billing with cache awareness
      - Anthropic dual-interface for thinking, reasoning_split, count_tokens

    The adapter works even without an anthropic_base_url — it gracefully
    degrades to pure OpenAI-compatible mode when the Anthropic endpoint
    is not configured.
    """

    def __init__(
        self,
        base_url: str = MINIMAX_DEFAULT_BASE_URL,
        api_key: str = "",
        group_id: str = "",
        timeout: float = 300.0,
        extra_headers: dict | None = None,
        enable_fake_tools: bool = False,
        multimodal_timeout: float = 600.0,
        anthropic_base_url: str = MINIMAX_ANTHROPIC_DEFAULT_BASE_URL,
    ) -> None:
        """Initialize MiniMaxNativeAdapter.

        Args:
            base_url: MiniMax API base URL (OpenAI-compatible).
                Defaults to MiniMax's official endpoint.
            api_key: MiniMax API key.
            group_id: MiniMax Group ID. Required for some MiniMax-specific endpoints.
                If empty, the adapter operates in pure OpenAI-compatible mode.
            timeout: HTTP request timeout in seconds.
            extra_headers: Additional HTTP headers.
            enable_fake_tools: Whether to inject fake tools for distillation detection.
            multimodal_timeout: Timeout for multimodal requests (video processing).
            anthropic_base_url: MiniMax Anthropic-compatible API base URL.
                If empty string, the Anthropic interface is disabled and all
                requests use the OpenAI-compatible endpoint.
        """
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            extra_headers=extra_headers,
            enable_fake_tools=enable_fake_tools,
            multimodal_timeout=multimodal_timeout,
        )

        # MiniMax-specific: Group ID
        self.group_id = group_id

        # MiniMax-specific: Anthropic-compatible base URL
        self.anthropic_base_url = anthropic_base_url.rstrip("/") if anthropic_base_url else ""

        # MiniMax-specific: Rate limit tracking
        self._rate_limit_info = MiniMaxRateLimitInfo()

        # MiniMax-specific: Cumulative billing tracker
        self._billing_tracker: dict[str, int | float] = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_hit_tokens": 0,
            "total_cache_miss_tokens": 0,
            "total_requests": 0,
        }

        # Anthropic-specific: persistent HTTP client (separate from OpenAI client)
        self._anthropic_http_client: httpx.AsyncClient | None = None

        logger.info(
            f"MiniMaxNativeAdapter: base_url={self.base_url}, "
            f"api_key={'***' if self.api_key else '(empty)'}, "
            f"group_id={'***' if self.group_id else '(empty)'}, "
            f"anthropic_base_url={self.anthropic_base_url or '(disabled)'}, "
            f"extra_headers={bool(self._extra_headers)}, "
            f"fake_tools={self._enable_fake_tools}, "
            f"timeout={self._timeout}s"
        )

    # ===================================================================
    # Override: close — also close the Anthropic client
    # ===================================================================

    async def close(self) -> None:
        """Close all HTTP connection pools including Anthropic client."""
        # Close the Anthropic client first
        if self._anthropic_http_client is not None and not self._anthropic_http_client.is_closed:
            await self._anthropic_http_client.aclose()
            logger.debug("MiniMaxNativeAdapter: anthropic httpx connection pool closed")
        self._anthropic_http_client = None
        # Close parent's clients
        await super().close()

    # ===================================================================
    # Anthropic interface: HTTP client management
    # ===================================================================

    async def _get_anthropic_client(self) -> httpx.AsyncClient:
        """Get or create a persistent httpx client for the Anthropic interface.

        Uses the same timeout configuration as the parent's multimodal client
        since Anthropic-interface requests often involve multimodal content.

        Returns:
            httpx.AsyncClient for Anthropic-compatible requests
        """
        if self._anthropic_http_client is None or self._anthropic_http_client.is_closed:
            self._anthropic_http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=30.0,
                    read=self._timeout,
                    write=30.0,
                    pool=30.0,
                ),
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                    keepalive_expiry=60.0,
                ),
                http2=self._http2_enabled,
                verify=self._ssl_verify,
            )
            logger.debug(
                "MiniMaxNativeAdapter: created new Anthropic httpx connection pool "
                "(max_connections=10, max_keepalive=5, keepalive_expiry=60s, http2=%s)"
                % self._http2_enabled
            )
        return self._anthropic_http_client

    # ===================================================================
    # Interface routing
    # ===================================================================

    def _route_interface(self, compiled: CompiledPrompt) -> str:
        """Determine which API interface to use for a given compiled prompt.

        Routing logic:
          1. If anthropic_base_url is not configured → always "openai"
          2. If compiled.extra has force_interface → use that
          3. Multimodal content (image/video) in messages → "anthropic"
          4. Thinking mode enabled → "anthropic"
          5. reasoning_split requested → "anthropic"
          6. count_tokens needed (marked in extra) → "anthropic"
          7. Default → "openai" (pure text quick requests)

        Args:
            compiled: The compiled prompt to route

        Returns:
            "openai" or "anthropic"
        """
        # If Anthropic interface is not configured, always use OpenAI
        if not self.anthropic_base_url:
            return "openai"

        # Allow explicit interface override
        force_interface = compiled.extra.get("force_interface")
        if force_interface in ("openai", "anthropic"):
            return force_interface

        # Check for multimodal content in messages
        messages = compiled.messages
        has_multimodal = self._detect_multimodal_in_messages(messages)

        # Mode B (system_user): compiled.messages 为空，需要检查 user_message 中的多模态内容
        if not has_multimodal and compiled.mode == "system_user" and compiled.user_message:
            if isinstance(compiled.user_message, list):
                for part in compiled.user_message:
                    if isinstance(part, dict) and part.get("type") in ("image_url", "video_url"):
                        has_multimodal = True
                        break
            elif isinstance(compiled.user_message, str):
                # 简单字符串不可能包含多模态
                pass

        if has_multimodal:
            logger.debug("MiniMaxNativeAdapter: routing to Anthropic (multimodal content)")
            return "anthropic"

        # Check for thinking mode
        thinking = compiled.extra.get("thinking")
        thinking_mode = compiled.extra.get("thinking_mode")
        if thinking or thinking_mode:
            logger.debug("MiniMaxNativeAdapter: routing to Anthropic (thinking mode)")
            return "anthropic"

        # Check for reasoning_split
        if compiled.extra.get("reasoning_split"):
            logger.debug("MiniMaxNativeAdapter: routing to Anthropic (reasoning_split)")
            return "anthropic"

        # Check for count_tokens hint
        if compiled.extra.get("needs_token_estimation"):
            logger.debug("MiniMaxNativeAdapter: routing to Anthropic (token estimation)")
            return "anthropic"

        # Default: pure text quick requests → OpenAI
        return "openai"

    @staticmethod
    def _detect_multimodal_in_messages(messages: list[dict]) -> bool:
        """Detect if messages contain multimodal content (image/video).

        Args:
            messages: Message list to check

        Returns:
            True if multimodal content is found
        """
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in (
                        "image_url", "video_url",
                    ):
                        return True
        return False

    # ===================================================================
    # Override: Headers — Add MiniMax-specific headers
    # ===================================================================

    def _build_headers(self) -> dict[str, str]:
        """Build request headers with MiniMax-specific additions.

        Extends parent to add MiniMax Group ID header if configured.
        """
        headers = super()._build_headers()

        # MiniMax API may require group_id in headers for certain operations
        if self.group_id:
            headers["X-Group-Id"] = self.group_id

        return headers

    def _build_anthropic_headers(self) -> dict[str, str]:
        """Build Anthropic-compatible request headers for MiniMax.

        MiniMax's Anthropic-compatible endpoint uses the same header format
        as the Anthropic Messages API, but with MiniMax's authentication.

        Returns:
            Dict of HTTP headers for the Anthropic interface
        """
        headers: dict[str, str] = {
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        # MiniMax Anthropic endpoint uses x-api-key like Anthropic
        if self.api_key:
            headers["x-api-key"] = self.api_key
        # Also add Authorization header for compatibility
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # Add group_id if configured
        if self.group_id:
            headers["X-Group-Id"] = self.group_id
        # Add any extra headers
        headers.update(self._extra_headers)
        return headers

    # ===================================================================
    # Override: send — Route to appropriate interface
    # ===================================================================

    async def send(self, compiled: CompiledPrompt, model: str) -> TAPResponse:
        """Send a compiled prompt with interface routing.

        Routes to the Anthropic interface when the compiled prompt
        requires features only available there (multimodal, thinking,
        reasoning_split, count_tokens). Otherwise uses the OpenAI
        interface via the parent class.
        """
        interface = self._route_interface(compiled)

        if interface == "anthropic":
            try:
                response = await self._send_via_anthropic(compiled, model)
                self._update_billing_tracker(response)
                return response
            except Exception as e:
                logger.warning(
                    f"MiniMaxNativeAdapter: Anthropic interface failed "
                    f"({type(e).__name__}: {e}), falling back to OpenAI interface"
                )
                # Graceful fallback to OpenAI interface
                # 修复 H16: fallback 路径也需计费（super().send 不会调用 _update_billing_tracker）
                response = await super().send(compiled, model)
                self._update_billing_tracker(response)
                return response
        else:
            # 修复 H16: super().send() 内部可能回退到 _send_non_streaming，
            # 而 _send_non_streaming 已调用 _update_billing_tracker，避免双重计费
            response = await super().send(compiled, model)
            return response

    async def _send_non_streaming(
        self, compiled: CompiledPrompt, model: str
    ) -> TAPResponse:
        """Non-streaming fallback with MiniMax-specific response parsing.

        Delegates to parent, then parses MiniMax-specific response fields
        and rate limit headers from the raw HTTP response.
        """
        response = await super()._send_non_streaming(compiled, model)

        # Post-process: update billing tracker
        self._update_billing_tracker(response)

        return response

    # ===================================================================
    # Override: stream — Route to appropriate interface
    # ===================================================================

    async def stream(self, compiled: CompiledPrompt, model: str) -> AsyncIterator[str]:
        """Stream with interface routing.

        Routes to the Anthropic streaming interface when needed,
        otherwise uses the OpenAI-compatible streaming via parent class.
        """
        interface = self._route_interface(compiled)

        if interface == "anthropic":
            try:
                async for chunk in self._stream_via_anthropic(compiled, model):
                    yield chunk
                return
            except Exception as e:
                logger.warning(
                    f"MiniMaxNativeAdapter: Anthropic streaming failed "
                    f"({type(e).__name__}: {e}), falling back to OpenAI streaming"
                )
                # Graceful fallback to OpenAI streaming
                async for chunk in super().stream(compiled, model):
                    yield chunk
                return
        else:
            async for chunk in super().stream(compiled, model):
                yield chunk

    # ===================================================================
    # Anthropic interface: Message format conversion
    # ===================================================================

    def _convert_messages_to_anthropic(
        self, compiled: CompiledPrompt
    ) -> tuple[str | list, list[dict]]:
        """Convert CompiledPrompt messages to Anthropic Messages API format.

        Extracts system messages into a top-level system parameter and
        converts multimodal content blocks to Anthropic format.

        Args:
            compiled: The compiled prompt to convert

        Returns:
            Tuple of (system_content, anthropic_messages) where:
            - system_content is a string or list of content blocks for the
              top-level "system" parameter (empty string if no system messages)
            - anthropic_messages is a list of message dicts in Anthropic format
        """
        system_parts: list[str] = []
        anthropic_messages: list[dict] = []

        # Mode B: system_prompt + user_message
        if compiled.mode == "system_user":
            if compiled.system_prompt:
                system_parts.append(compiled.system_prompt)
            anthropic_messages.append({
                "role": "user",
                "content": self._convert_content_to_anthropic(compiled.user_message),
            })
        else:
            # Mode A: messages list
            for msg in compiled.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    # Extract system messages to top-level system parameter
                    if isinstance(content, str):
                        system_parts.append(content)
                    elif isinstance(content, list):
                        # System message as content blocks
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                system_parts.append(block.get("text", ""))
                            elif isinstance(block, str):
                                system_parts.append(block)
                else:
                    # Convert non-system messages
                    converted_content = self._convert_content_to_anthropic(content)
                    anthropic_messages.append({
                        "role": role,
                        "content": converted_content,
                    })

            # If no messages after filtering system, add a default user message
            if not anthropic_messages:
                anthropic_messages.append({
                    "role": "user",
                    "content": compiled.user_message or "Please proceed.",
                })

        system_content = "\n".join(system_parts).strip() if system_parts else ""
        return system_content, anthropic_messages

    @staticmethod
    def _convert_content_to_anthropic(content) -> str | list[dict]:
        """Convert message content to Anthropic format.

        Handles:
          - String content → string (no conversion needed)
          - List content blocks → Anthropic content blocks

        For list content, converts:
          - text blocks → Anthropic text blocks (same format)
          - image_url blocks → Anthropic image blocks with source.type "url" or "base64"
          - video_url blocks → Anthropic-compatible video format

        Args:
            content: Message content (string or list of content blocks)

        Returns:
            Content in Anthropic format (string or list of content blocks)
        """
        if isinstance(content, str):
            return content

        if not isinstance(content, list):
            return str(content)

        anthropic_blocks: list[dict] = []
        for part in content:
            if not isinstance(part, dict):
                # Skip non-dict parts
                continue

            part_type = part.get("type", "")

            if part_type == "text":
                anthropic_blocks.append({
                    "type": "text",
                    "text": part.get("text", ""),
                })

            elif part_type == "image_url":
                image_url_obj = part.get("image_url", {})
                url = image_url_obj.get("url", "") if isinstance(image_url_obj, dict) else ""

                if url.startswith("data:"):
                    # Base64 data URI → Anthropic base64 source
                    # Parse: data:<media_type>;base64,<data>
                    try:
                        rest = url[5:]  # Remove "data:"
                        media_part, _, base64_data = rest.partition(";base64,")
                        if not media_part:
                            media_part = "image/png"
                        anthropic_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_part,
                                "data": base64_data,
                            },
                        })
                    except Exception:
                        # Fallback: include as text description
                        anthropic_blocks.append({
                            "type": "text",
                            "text": f"[image: base64 data]",
                        })
                else:
                    # URL → Anthropic url source
                    anthropic_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": url,
                        },
                    })

            elif part_type == "video_url":
                video_url_obj = part.get("video_url", {})
                url = video_url_obj.get("url", "") if isinstance(video_url_obj, dict) else ""
                # MiniMax's Anthropic-compatible endpoint supports video
                # via a custom content block type
                anthropic_blocks.append({
                    "type": "video",
                    "source": {
                        "type": "url",
                        "url": url,
                    },
                })

            else:
                # Unknown type — try to pass through as text
                text = part.get("text", "")
                if text:
                    anthropic_blocks.append({
                        "type": "text",
                        "text": text,
                    })

        return anthropic_blocks if anthropic_blocks else ""

    # ===================================================================
    # Anthropic interface: Build request payload
    # ===================================================================

    def _build_anthropic_payload(
        self, compiled: CompiledPrompt, model: str
    ) -> dict:
        """Build the Anthropic Messages API request payload.

        Args:
            compiled: The compiled prompt
            model: Model name string

        Returns:
            Dict representing the Anthropic API request body
        """
        system_content, messages = self._convert_messages_to_anthropic(compiled)

        payload: dict = {
            "model": model,
            "max_tokens": compiled.max_tokens,
            "messages": messages,
        }

        # Add system parameter if present
        if system_content:
            payload["system"] = system_content

        # --- Thinking mode ---
        thinking = compiled.extra.get("thinking")
        thinking_mode = compiled.extra.get("thinking_mode")
        if thinking:
            # Direct thinking parameter: {"type": "enabled"} or {"type": "adaptive"}
            if isinstance(thinking, dict):
                payload["thinking"] = thinking
            elif isinstance(thinking, bool) and thinking:
                payload["thinking"] = {"type": "enabled"}
            elif isinstance(thinking, str):
                if thinking in ("enabled", "adaptive"):
                    payload["thinking"] = {"type": thinking}
                # "disabled" — Anthropic API 不支持，省略 thinking 字段即可禁用
        elif thinking_mode:
            # Map thinking_mode to Anthropic thinking parameter
            if thinking_mode == "deep":
                payload["thinking"] = {"type": "enabled"}
            # thinking_mode == "quick" 或其他值：不设置 thinking 字段即可禁用
            # Anthropic API 不支持 {"type": "disabled"}

        # --- Budget tokens for thinking ---
        budget_tokens = compiled.extra.get("thinking_budget")
        if budget_tokens and "thinking" in payload:
            payload["thinking"]["budget_tokens"] = int(budget_tokens)

        # --- reasoning_split ---
        if compiled.extra.get("reasoning_split"):
            payload["reasoning_split"] = True

        # --- Tools ---
        if compiled.tools:
            anthropic_tools = self._convert_tools_to_anthropic(compiled.tools)
            if anthropic_tools:
                payload["tools"] = anthropic_tools
                if compiled.tool_choice is not None:
                    payload["tool_choice"] = self._convert_tool_choice_to_anthropic(
                        compiled.tool_choice
                    )

        return payload

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

    @staticmethod
    def _convert_tool_choice_to_anthropic(tool_choice: dict | str) -> dict | str:
        """Convert OpenAI-format tool_choice to Anthropic format."""
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

    # ===================================================================
    # Anthropic interface: Parse response
    # ===================================================================

    def _parse_anthropic_response(
        self, data: dict, model: str
    ) -> TAPResponse:
        """Parse Anthropic Messages API response into TAPResponse.

        Extracts:
          - Text content from content blocks
          - Thinking content from thinking blocks
          - Tool use from tool_use blocks
          - Usage (input_tokens, output_tokens, cache tokens)
          - Reasoning details (if reasoning_split was enabled)
          - Stop reason mapping

        Args:
            data: Anthropic API response dict
            model: Model name for logging

        Returns:
            TAPResponse with parsed data
        """
        content_blocks = data.get("content", [])

        # Extract text content
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict] = []

        for block in content_blocks:
            block_type = block.get("type", "")

            if block_type == "text":
                text_parts.append(block.get("text", ""))

            elif block_type == "thinking":
                # Thinking content block (extended thinking mode)
                thinking_text = block.get("thinking", "")
                if thinking_text:
                    thinking_parts.append(thinking_text)

            elif block_type == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })

        raw_text = "".join(text_parts)
        thinking_content = "\n".join(thinking_parts) if thinking_parts else None

        # Parse usage (Anthropic format)
        raw_usage = data.get("usage", {})
        usage: dict = {
            "prompt_tokens": raw_usage.get("input_tokens", 0),
            "completion_tokens": raw_usage.get("output_tokens", 0),
        }
        # Anthropic cache token fields
        cache_creation = raw_usage.get("cache_creation_input_tokens", 0)
        cache_read = raw_usage.get("cache_read_input_tokens", 0)
        if cache_creation:
            usage["cache_creation_input_tokens"] = cache_creation
        if cache_read:
            usage["cache_read_input_tokens"] = cache_read
            # Map cache_read to prompt_cache_hit_tokens for consistency
            usage["prompt_cache_hit_tokens"] = cache_read

        # Map Anthropic stop_reason to standard finish_reason
        stop_reason = data.get("stop_reason", "stop")
        _FINISH_REASON_MAP = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }
        finish_reason = _FINISH_REASON_MAP.get(stop_reason, stop_reason or "stop")

        # Extract reasoning details (if reasoning_split was enabled)
        reasoning_details = data.get("reasoning_details")
        extra: dict = {}
        if reasoning_details:
            extra["reasoning_details"] = reasoning_details
        if thinking_content:
            extra["thinking_content"] = thinking_content

        return TAPResponse(
            raw_text=raw_text or "",
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            thinking_content=thinking_content,
            cache_hit_tokens=cache_read,
            extra=extra,
        )

    # ===================================================================
    # Anthropic interface: Send (non-streaming)
    # ===================================================================

    async def _send_via_anthropic(
        self, compiled: CompiledPrompt, model: str
    ) -> TAPResponse:
        """Send a compiled prompt via MiniMax's Anthropic-compatible interface.

        Uses the /messages endpoint with Anthropic format.
        Falls back to non-streaming if streaming fails.

        Args:
            compiled: The compiled prompt
            model: Model name string

        Returns:
            TAPResponse with the model's output

        Raises:
            httpx.HTTPStatusError: On API errors
            RuntimeError: On API-level errors in the response
        """
        url = f"{self.anthropic_base_url}/messages"
        headers = self._build_anthropic_headers()
        payload = self._build_anthropic_payload(compiled, model)

        # Use streaming first to avoid 504 gateway timeouts (like Anthropic adapter)
        payload["stream"] = True

        collected: list[str] = []
        thinking_collected: list[str] = []
        tool_use_blocks: list[dict] = []
        tool_input_accumulators: dict[int, str] = {}
        tool_block_map: dict[int, dict] = {}
        usage: dict = {}
        actual_stop_reason: str = ""

        try:
            client = await self._get_anthropic_client()
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
                                logger.error(
                                    f"MiniMax Anthropic stream error: {err_msg}"
                                )
                                raise RuntimeError(
                                    f"MiniMax Anthropic API stream error: {err_msg}"
                                )

                            event_type = event.get("type", "")

                            # Extract text content from content_block_delta
                            if event_type == "content_block_delta":
                                delta = event.get("delta", {})
                                delta_type = delta.get("type", "")

                                if delta_type == "text_delta":
                                    text = delta.get("text", "")
                                    if text:
                                        collected.append(text)

                                elif delta_type == "thinking_delta":
                                    thinking_text = delta.get("thinking", "")
                                    if thinking_text:
                                        thinking_collected.append(thinking_text)

                                elif delta_type == "input_json_delta":
                                    # Accumulate tool input arguments
                                    idx = event.get("index", 0)
                                    partial_json = delta.get("partial_json", "")
                                    if idx not in tool_input_accumulators:
                                        tool_input_accumulators[idx] = ""
                                    tool_input_accumulators[idx] += partial_json

                            # Track tool_use blocks
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
                                cache_creation = msg_usage.get(
                                    "cache_creation_input_tokens", 0
                                )
                                cache_read = msg_usage.get(
                                    "cache_read_input_tokens", 0
                                )
                                if cache_creation:
                                    usage["cache_creation_input_tokens"] = cache_creation
                                if cache_read:
                                    usage["cache_read_input_tokens"] = cache_read
                                    usage["prompt_cache_hit_tokens"] = cache_read

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

            raw_text = "".join(collected)
            thinking_content = "".join(thinking_collected) if thinking_collected else None

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

            # Map stop reason
            _FINISH_REASON_MAP = {
                "end_turn": "stop",
                "max_tokens": "length",
                "stop_sequence": "stop",
                "tool_use": "tool_calls",
            }
            finish_reason = _FINISH_REASON_MAP.get(
                actual_stop_reason, actual_stop_reason or "stop"
            )

            cache_hit = usage.get("prompt_cache_hit_tokens", 0)

            return TAPResponse(
                raw_text=raw_text or "",
                usage=usage,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                thinking_content=thinking_content,
                cache_hit_tokens=cache_hit,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (400, 405, 501):
                logger.warning(
                    f"MiniMax Anthropic streaming not supported "
                    f"({e.response.status_code}), falling back to non-streaming"
                )
                return await self._send_via_anthropic_non_streaming(compiled, model)
            logger.error(
                f"MiniMax Anthropic API Error: "
                f"{e.response.status_code} - {e.response.text}"
            )
            raise
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"MiniMax Anthropic Request Failed: {e}")
            raise

    async def _send_via_anthropic_non_streaming(
        self, compiled: CompiledPrompt, model: str
    ) -> TAPResponse:
        """Non-streaming send via MiniMax's Anthropic-compatible interface.

        Args:
            compiled: The compiled prompt
            model: Model name string

        Returns:
            TAPResponse with the model's output
        """
        url = f"{self.anthropic_base_url}/messages"
        headers = self._build_anthropic_headers()
        payload = self._build_anthropic_payload(compiled, model)
        # Non-streaming: do not include stream parameter

        client = await self._get_anthropic_client()
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        return self._parse_anthropic_response(data, model)

    # ===================================================================
    # Anthropic interface: Stream
    # ===================================================================

    async def _stream_via_anthropic(
        self, compiled: CompiledPrompt, model: str
    ) -> AsyncIterator[str]:
        """Stream via MiniMax's Anthropic-compatible interface.

        Yields text chunks from content_block_delta events.
        Also collects thinking content and stores it for later retrieval.

        Args:
            compiled: The compiled prompt
            model: Model name string

        Yields:
            Text chunks as they arrive from the model
        """
        url = f"{self.anthropic_base_url}/messages"
        headers = self._build_anthropic_headers()
        payload = self._build_anthropic_payload(compiled, model)
        payload["stream"] = True

        client = await self._get_anthropic_client()
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
                            logger.error(
                                f"MiniMax Anthropic stream error: {err_msg}"
                            )
                            # 修复 H15: 抛出异常而非静默 break
                            raise RuntimeError(f"MiniMax Anthropic stream error: {err_msg}")

                        event_type = event.get("type", "")
                        if event_type == "content_block_delta":
                            delta = event.get("delta", {})
                            delta_type = delta.get("type", "")

                            if delta_type == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    yield text
                            # Note: thinking_delta events are not yielded
                            # as text chunks; they should be collected
                            # separately if needed.
                    except json.JSONDecodeError:
                        continue

    # ===================================================================
    # count_tokens method
    # ===================================================================

    async def count_tokens(
        self, compiled: CompiledPrompt, model: str
    ) -> int:
        """Count tokens for a compiled prompt via the Anthropic count_tokens endpoint.

        POSTs to /anthropic/v1/messages/count_tokens with the same
        message format as the Anthropic Messages API.

        Args:
            compiled: The compiled prompt to count tokens for
            model: Model name string

        Returns:
            Estimated token count

        Raises:
            ValueError: If the Anthropic interface is not configured
            httpx.HTTPStatusError: On API errors
        """
        if not self.anthropic_base_url:
            raise ValueError(
                "MiniMaxNativeAdapter: count_tokens requires the Anthropic "
                "interface to be configured. Set anthropic_base_url to enable."
            )

        url = f"{self.anthropic_base_url}/messages/count_tokens"
        headers = self._build_anthropic_headers()

        # Build the same payload as for the messages endpoint
        payload = self._build_anthropic_payload(compiled, model)
        # Remove stream parameter if present (count_tokens doesn't support it)
        payload.pop("stream", None)

        try:
            client = await self._get_anthropic_client()
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            token_count = data.get("input_tokens", 0)
            logger.debug(
                f"MiniMaxNativeAdapter: count_tokens returned "
                f"{token_count} tokens for model {model}"
            )
            return token_count

        except httpx.HTTPStatusError as e:
            logger.error(
                f"MiniMaxNativeAdapter: count_tokens API error: "
                f"{e.response.status_code} - {e.response.text}"
            )
            raise
        except Exception as e:
            logger.error(
                f"MiniMaxNativeAdapter: count_tokens failed: "
                f"{type(e).__name__}: {e}"
            )
            raise

    # ===================================================================
    # MiniMax-specific: Video content enhancement
    # ===================================================================

    @staticmethod
    def _enhance_video_content(messages: list[dict]) -> list[dict]:
        """Enhance video_url content blocks with MiniMax-specific metadata.

        MiniMax M3 supports native video understanding. This method adds
        optional video metadata that the MiniMax API can use for better
        processing:
          - Video duration hint (if available in the content block)
          - Video frame sampling hint

        This is a no-op if no video content is found or if the video
        blocks already have MiniMax-specific fields.

        Args:
            messages: Message list to enhance

        Returns:
            Enhanced message list (may be the same list if no changes needed)
        """
        enhanced = False

        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            for part in content:
                if not isinstance(part, dict):
                    continue

                if part.get("type") != "video_url":
                    continue

                video_obj = part.get("video_url", {})
                if not isinstance(video_obj, dict):
                    continue

                # Add MiniMax-specific video processing hints if not already present
                # These are optional; the MiniMax API ignores unknown fields gracefully
                if "minimax_video_mode" not in video_obj:
                    # Default to "understand" mode — the model should understand the video
                    # Alternative: "summarize" for lightweight processing
                    video_obj["minimax_video_mode"] = "understand"
                    enhanced = True

                if "minimax_frame_sampling" not in video_obj:
                    # Default frame sampling: "auto" lets the API decide
                    # Alternatives: "uniform", "keyframe", "dense"
                    video_obj["minimax_frame_sampling"] = "auto"
                    enhanced = True

        if enhanced:
            logger.debug("MiniMaxNativeAdapter: enhanced video content with processing hints")

        return messages

    # ===================================================================
    # MiniMax-specific: Desktop operation API
    # ===================================================================

    async def send_desktop_command(
        self,
        command: str,
        params: dict | None = None,
        screenshot: str | None = None,
        interactive_elements: list[dict] | None = None,
        active_window: str = "",
        model: str = "minimax-m3",
    ) -> dict:
        """Send a desktop operation command via MiniMax's dedicated endpoint.

        MiniMax M3 supports desktop automation through a specialized API.
        This method sends desktop commands (click, type, scroll, etc.)
        along with screen context (screenshot, interactive elements).

        This is separate from the chat completions endpoint and uses
        MiniMax's native desktop operation API.

        Args:
            command: Desktop command to execute (e.g., "click", "type", "scroll")
            params: Command parameters (e.g., {"x": 100, "y": 200} for click)
            screenshot: Base64-encoded screenshot of the current screen state
            interactive_elements: List of interactive UI elements on screen
            active_window: Name of the currently active window
            model: Model to use for desktop operations (default: "minimax-m3")

        Returns:
            Dict with the API response, containing:
            - "action": The recommended next action
            - "reasoning": The model's reasoning for the action
            - "raw_response": The full API response

        Raises:
            httpx.HTTPStatusError: On API errors
            ValueError: If the adapter is not configured for desktop operations
            RuntimeError: If the desktop endpoint is not available
        """
        if not self.api_key:
            raise ValueError("MiniMaxNativeAdapter: api_key is required for desktop operations")

        # Build the desktop operation request
        url = f"{self.base_url}{_DESKTOP_ENDPOINT}"
        headers = self._build_headers()

        payload: dict = {
            "model": model,
            "command": command,
            "context": {},
        }

        if params:
            payload["params"] = params

        # Add screen context
        context: dict = {}
        if screenshot:
            context["screenshot"] = screenshot
        if interactive_elements:
            context["interactive_elements"] = interactive_elements
        if active_window:
            context["active_window"] = active_window

        payload["context"] = context

        # Add group_id if available (some MiniMax endpoints require it)
        if self.group_id:
            payload["group_id"] = self.group_id

        logger.info(
            f"MiniMaxNativeAdapter: sending desktop command '{command}' "
            f"with {len(interactive_elements or [])} interactive elements"
        )

        try:
            client = await self._get_client(multimodal=True)  # Desktop ops need multimodal timeout
            response = await client.post(url, json=payload, headers=headers)

            # Update rate limit info from response headers
            self._rate_limit_info.update_from_headers(response.headers)

            response.raise_for_status()
            data = response.json()

            # Extract the action recommendation from the response
            result: dict = {
                "action": data.get("action", {}),
                "reasoning": data.get("reasoning", ""),
                "raw_response": data,
            }

            logger.info(
                f"MiniMaxNativeAdapter: desktop command '{command}' completed, "
                f"action={result['action']}"
            )

            return result

        except httpx.HTTPStatusError as e:
            # Desktop endpoint might not be available for all MiniMax deployments
            if e.response.status_code == 404:
                logger.warning(
                    "MiniMaxNativeAdapter: desktop operation endpoint not available. "
                    "Falling back to chat completions for desktop commands."
                )
                # Graceful degradation: use chat completions instead
                return await self._desktop_command_via_chat(
                    command, params, screenshot, interactive_elements,
                    active_window, model,
                )
            raise

        except Exception as e:
            logger.error(
                f"MiniMaxNativeAdapter: desktop command failed: "
                f"{type(e).__name__}: {e}"
            )
            raise

    async def _desktop_command_via_chat(
        self,
        command: str,
        params: dict | None = None,
        screenshot: str | None = None,
        interactive_elements: list[dict] | None = None,
        active_window: str = "",
        model: str = "minimax-m3",
    ) -> dict:
        """Fallback: Send desktop command via chat completions.

        When the dedicated desktop endpoint is not available, we encode
        the desktop context as a multimodal chat message and ask the model
        to recommend the next action.

        This is a graceful degradation strategy — less optimal than the
        native desktop endpoint, but works with standard MiniMax API.

        Args:
            command: Desktop command
            params: Command parameters
            screenshot: Base64-encoded screenshot
            interactive_elements: Interactive UI elements
            active_window: Active window name
            model: Model name

        Returns:
            Dict with "action", "reasoning", and "raw_response" keys
        """
        # Build a prompt that encodes the desktop context
        content_parts: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"You are a desktop automation assistant. The user wants to "
                    f"execute the command: '{command}'.\n\n"
                    f"Current desktop state:\n"
                    f"- Active window: {active_window or 'unknown'}\n"
                    f"- Interactive elements: {json.dumps(interactive_elements or [], ensure_ascii=False)}\n\n"
                    f"Based on the screenshot and desktop state, recommend the next action "
                    f"to accomplish the command. Respond in JSON format with keys: "
                    f'"action_type", "action_params", "reasoning".'
                ),
            }
        ]

        # Add screenshot as image if provided
        if screenshot:
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{screenshot}",
                },
            })

        messages = [{"role": "user", "content": content_parts}]

        # Use the parent's send method to make the API call
        compiled = CompiledPrompt(messages=messages, max_tokens=4096)
        response = await super().send(compiled, model)

        return {
            "action": {"type": "chat_fallback"},
            "reasoning": response.raw_text,
            "raw_response": {"text": response.raw_text, "usage": response.usage},
        }

    # ===================================================================
    # MiniMax-specific: Response parsing enhancements
    # ===================================================================

    def _parse_minimax_usage(self, usage: dict) -> dict:
        """Parse MiniMax-specific usage fields from API response.

        MiniMax may return additional usage fields beyond the standard
        OpenAI format:
          - prompt_cache_hit_tokens: Tokens served from cache
          - prompt_cache_miss_tokens: Cache miss tokens (MiniMax-specific)
          - total_tokens: Total tokens including cache

        This method extracts these fields and returns a normalized dict.

        Args:
            usage: Raw usage dict from API response

        Returns:
            Normalized usage dict with MiniMax-specific fields
        """
        if not usage:
            return {}

        parsed: dict = dict(usage)  # Copy all standard fields

        # MiniMax-specific: cache miss tokens (not present in DeepSeek V4)
        cache_miss_tokens = usage.get("prompt_cache_miss_tokens", 0)
        if cache_miss_tokens:
            parsed["prompt_cache_miss_tokens"] = cache_miss_tokens

        # Ensure cache_hit_tokens is present (same key as DeepSeek V4)
        if "prompt_cache_hit_tokens" not in parsed:
            # Some MiniMax responses may use a different key
            alt_key = usage.get("cached_tokens", 0)
            parsed["prompt_cache_hit_tokens"] = alt_key

        return parsed

    # ===================================================================
    # MiniMax-specific: Billing tracker
    # ===================================================================

    def _update_billing_tracker(self, response: TAPResponse) -> None:
        """Update cumulative billing tracker from a TAPResponse.

        Tracks MiniMax-specific billing fields:
          - Input/output token counts
          - Cache hit/miss token counts
          - Request count

        Args:
            response: TAPResponse with usage data
        """
        usage = response.usage
        if not usage:
            self._billing_tracker["total_requests"] += 1
            return

        self._billing_tracker["total_input_tokens"] += usage.get("prompt_tokens", 0)
        self._billing_tracker["total_output_tokens"] += usage.get("completion_tokens", 0)
        self._billing_tracker["total_cache_hit_tokens"] += usage.get(
            "prompt_cache_hit_tokens", 0
        )
        self._billing_tracker["total_cache_miss_tokens"] += usage.get(
            "prompt_cache_miss_tokens", 0
        )
        self._billing_tracker["total_requests"] += 1

    @property
    def billing_summary(self) -> dict[str, int | float]:
        """Return a copy of the cumulative billing summary.

        Returns:
            Dict with cumulative token counts and request count
        """
        return dict(self._billing_tracker)

    @property
    def rate_limit_info(self) -> MiniMaxRateLimitInfo:
        """Return current rate limit information.

        Updated from HTTP response headers when available.

        Returns:
            MiniMaxRateLimitInfo instance
        """
        return self._rate_limit_info

    # ===================================================================
    # Override: Model name mapping
    # ===================================================================

    _MINIMAX_MODEL_NAME_MAP: dict[str, str] = {
        # Common aliases → canonical MiniMax model names
        "minimax": "minimax-m3",
        "minimax-m3": "minimax-m3",
        "m3": "minimax-m3",
        "MiniMax-M3": "minimax-m3",
    }

    def _resolve_model_name(self, model: str) -> str:
        """Resolve model name with MiniMax-specific mappings.

        First checks MiniMax-specific aliases, then falls back to
        parent's model name resolution.

        Args:
            model: Model name string

        Returns:
            Resolved model name
        """
        # Check MiniMax-specific aliases first
        resolved = self._MINIMAX_MODEL_NAME_MAP.get(model)
        if resolved:
            return resolved

        # Fall back to parent's resolution (handles DeepSeek V4 aliases, etc.)
        return super()._resolve_model_name(model)

    # ===================================================================
    # Override: Capabilities
    # ===================================================================

    @property
    def capabilities(self) -> dict:
        """Return adapter capabilities with MiniMax M3 specifics.

        Extends parent capabilities with:
          - multimodal: True (M3 native multimodal support)
          - desktop: True (M3 desktop operation support)
          - video: True (M3 native video understanding)
          - msa_efficient: True (MiniMax Sparse Attention)
          - thinking: True (extended thinking via Anthropic interface)
          - reasoning_split: True (structured reasoning via Anthropic interface)
          - count_tokens: True (token counting via Anthropic interface)
          - anthropic_interface: True (dual-interface support)
        """
        caps = super().capabilities
        caps.update({
            "multimodal": True,
            "desktop": True,
            "video": True,
            "msa_efficient": True,
            "max_context_tokens": 1_000_000,  # M3 supports 1M context
            "thinking": True,
            "reasoning_split": True,
            "count_tokens": bool(self.anthropic_base_url),
            "anthropic_interface": bool(self.anthropic_base_url),
        })
        return caps


# ---------------------------------------------------------------------------
# Register with TAPAdapterRegistry
# ---------------------------------------------------------------------------

TAPAdapterRegistry.register("minimax_native", MiniMaxNativeAdapter)
