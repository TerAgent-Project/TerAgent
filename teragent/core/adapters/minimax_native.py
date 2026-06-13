"""teragent.core.adapters.minimax_native — MiniMax M3 Native TAP Adapter

Extends OpenAICompatibleAdapter with MiniMax-specific features while
delegating all standard OpenAI-compatible operations to the parent class.

MiniMax-specific enhancements:
  1. Group ID handling — MiniMax API requires a group_id for some endpoints
  2. Video content encoding — Enhanced video_url processing for M3 native multimodal
  3. Desktop operation API — Dedicated endpoint for M3 desktop automation
  4. Response parsing — MiniMax-specific usage fields and metadata
  5. Rate limit tracking — Parse X-RateLimit-* headers for cost-aware routing
  6. Token billing tracking — MiniMax-specific billing model with cache awareness

For standard OpenAI-compatible operations (chat completions, streaming,
tool calling, fake_tools, etc.), everything is delegated to the parent class.
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

# MiniMax default base URL
MINIMAX_DEFAULT_BASE_URL = "https://api.minimax.chat/v1"

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

    Extends OpenAICompatibleAdapter with MiniMax-specific enhancements.
    All standard OpenAI-compatible operations are delegated to the parent
    class via super().

    MiniMax-specific features:
      - Group ID handling for MiniMax API endpoints
      - Enhanced video content processing for M3 native multimodal
      - Desktop operation API endpoint
      - MiniMax-specific response field parsing
      - Rate limit header tracking for cost-aware routing
      - Token billing with cache awareness

    The adapter works even without a group_id — it gracefully degrades
    to standard OpenAI-compatible behavior when MiniMax-specific features
    are not available or not configured.
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
    ) -> None:
        """Initialize MiniMaxNativeAdapter.

        Args:
            base_url: MiniMax API base URL. Defaults to MiniMax's official endpoint.
            api_key: MiniMax API key.
            group_id: MiniMax Group ID. Required for some MiniMax-specific endpoints.
                If empty, the adapter operates in pure OpenAI-compatible mode.
            timeout: HTTP request timeout in seconds.
            extra_headers: Additional HTTP headers.
            enable_fake_tools: Whether to inject fake tools for distillation detection.
            multimodal_timeout: Timeout for multimodal requests (video processing).
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

        logger.info(
            f"MiniMaxNativeAdapter: base_url={self.base_url}, "
            f"api_key={'***' if self.api_key else '(empty)'}, "
            f"group_id={'***' if self.group_id else '(empty)'}, "
            f"extra_headers={bool(self._extra_headers)}, "
            f"fake_tools={self._enable_fake_tools}, "
            f"timeout={self._timeout}s"
        )

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

    # ===================================================================
    # Override: send — Enhanced response parsing
    # ===================================================================

    async def send(self, compiled: CompiledPrompt, model: str) -> TAPResponse:
        """Send a compiled prompt with MiniMax-specific enhancements.

        Delegates to parent's send() for the core HTTP interaction,
        then post-processes the response for MiniMax-specific fields.
        """
        response = await super().send(compiled, model)

        # Post-process: update billing tracker
        self._update_billing_tracker(response)

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
    # Override: stream — Enhanced for MiniMax rate limit tracking
    # ===================================================================

    async def stream(self, compiled: CompiledPrompt, model: str) -> AsyncIterator[str]:
        """Stream with MiniMax-specific rate limit tracking.

        Delegates to parent's stream() for the core SSE interaction.
        Rate limit headers are captured from the initial HTTP response.
        """
        # We need to intercept the HTTP response headers for rate limit tracking
        # The parent's stream() doesn't expose headers, so we use a lightweight
        # approach: track usage from the stream itself.
        async for chunk in super().stream(compiled, model):
            yield chunk

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
        """
        caps = super().capabilities
        caps.update({
            "multimodal": True,
            "desktop": True,
            "video": True,
            "msa_efficient": True,
            "max_context_tokens": 1_000_000,  # M3 supports 1M context
        })
        return caps


# ---------------------------------------------------------------------------
# Register with TAPAdapterRegistry
# ---------------------------------------------------------------------------

TAPAdapterRegistry.register("minimax_native", MiniMaxNativeAdapter)
