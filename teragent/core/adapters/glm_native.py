"""teragent.core.adapters.glm_native — GLM Native TAP Adapter

Extends OpenAICompatibleAdapter with GLM-specific features while
delegating all standard OpenAI-compatible operations to the parent class.

GLM has two interfaces:
  1. OpenAI compatible: https://open.bigmodel.cn/api/paas/v4 — basic, loses native features
  2. Native HTTP API: Same base URL but with extra fields

GLM-specific enhancements over OpenAICompatibleAdapter:
  1. reasoning_content extraction — Thinking mode returns chain-of-thought in
     `reasoning_content` (same level as `content`), NOT standard OpenAI format
  2. enable_thinking parameter — Controls thinking mode (Alibaba Cloud Bailian
     platform uses this, not OpenAI's `thinking` field)
  3. cached_tokens tracking — Parse `prompt_tokens_details.cached_tokens` from
     response usage for cost optimization
  4. Async chat completion — POST /chat/completions/async endpoint for
     long-running tasks with polling and timeout support
  5. content_filter handling — Content safety filtering results in response

Interface routing:
  - Long-horizon tasks → async interface (avoid timeout)
  - Thinking mode → native interface (reasoning_content extraction)
  - Cache tracking → native interface (cached_tokens)
  - Simple requests → OpenAI compatible (lightweight)

For standard OpenAI-compatible operations (chat completions, streaming,
tool calling, fake_tools, etc.), everything is delegated to the parent class.
"""

from __future__ import annotations

import asyncio
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
# GLM-specific constants
# ---------------------------------------------------------------------------

# GLM default base URL (Zhipu AI / BigModel platform)
GLM_DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

# GLM async chat endpoint (relative to base_url)
_ASYNC_CHAT_ENDPOINT = "/chat/completions/async"

# GLM async task status endpoint (relative to base_url)
_ASYNC_TASK_STATUS_ENDPOINT = "/async-task/{task_id}"

# Default polling interval for async chat tasks (seconds)
_ASYNC_POLL_INTERVAL = 2.0

# Default timeout for async chat completion polling (seconds)
_ASYNC_POLL_TIMEOUT = 600.0


class GLMNativeAdapter(OpenAICompatibleAdapter):
    """GLM Native API Adapter

    Extends OpenAICompatibleAdapter with GLM-specific features:

    1. reasoning_content extraction from responses
       GLM returns thinking chain-of-thought in `reasoning_content` at the
       same level as `content` in the message object, unlike OpenAI's format.

    2. enable_thinking parameter control
       GLM uses `enable_thinking=True` in the request body (not OpenAI's
       `thinking` object). This adapter auto-injects it based on compiler
       hints and default configuration.

    3. cached_tokens tracking for cost optimization
       GLM returns `prompt_tokens_details.cached_tokens` in usage, which
       this adapter extracts into TAPResponse.cache_hit_tokens and
       TAPResponse.extra["cached_tokens"].

    4. Async chat completion for long-horizon tasks
       GLM provides a /chat/completions/async endpoint for long-running
       requests that might exceed normal HTTP timeouts. This adapter
       supports submitting, polling, and retrieving results.

    5. content_filter handling for safety results
       GLM may return a `content_filter` field in the response indicating
       content safety filtering results. This adapter preserves it in
       TAPResponse.extra["content_filter"].

    Interface routing:
      - Long-horizon tasks → async interface (avoid timeout)
      - Thinking mode → native interface (reasoning_content extraction)
      - Cache tracking → native interface (cached_tokens)
      - Simple requests → OpenAI compatible (lightweight)
    """

    def __init__(
        self,
        base_url: str = GLM_DEFAULT_BASE_URL,
        api_key: str = "",
        timeout: float = 300.0,
        extra_headers: dict | None = None,
        enable_fake_tools: bool = False,
        multimodal_timeout: float = 600.0,
        enable_thinking_default: bool = False,
        async_enabled: bool = True,
        cache_tracking: bool = True,
    ) -> None:
        """Initialize GLMNativeAdapter.

        Args:
            base_url: GLM API base URL. Defaults to Zhipu AI's official endpoint.
            api_key: GLM API key.
            timeout: HTTP request timeout in seconds.
            extra_headers: Additional HTTP headers.
            enable_fake_tools: Whether to inject fake tools for distillation detection.
            multimodal_timeout: Timeout for multimodal requests.
            enable_thinking_default: Whether to enable thinking mode by default.
                When True, `enable_thinking=True` is added to the request body
                unless the compiled prompt explicitly disables it.
            async_enabled: Whether the async chat completion endpoint is available.
                When True, long-horizon tasks can use the async interface.
            cache_tracking: Whether to track cached_tokens from prompt_tokens_details.
                When True, cached_tokens are extracted and stored in the response.
        """
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            extra_headers=extra_headers,
            enable_fake_tools=enable_fake_tools,
            multimodal_timeout=multimodal_timeout,
        )

        # GLM-specific configuration
        self.enable_thinking_default = enable_thinking_default
        self.async_enabled = async_enabled
        self.cache_tracking = cache_tracking

        # GLM-specific: Cumulative cache tracker
        self._cache_tracker: dict[str, int] = {
            "total_cached_tokens": 0,
            "total_prompt_tokens": 0,
            "total_requests": 0,
        }

        logger.info(
            f"GLMNativeAdapter: base_url={self.base_url}, "
            f"api_key={'***' if self.api_key else '(empty)'}, "
            f"extra_headers={bool(self._extra_headers)}, "
            f"fake_tools={self._enable_fake_tools}, "
            f"timeout={self._timeout}s, "
            f"enable_thinking_default={self.enable_thinking_default}, "
            f"async_enabled={self.async_enabled}, "
            f"cache_tracking={self.cache_tracking}"
        )

    # ===================================================================
    # Override: _inject_glm_params — Add GLM-specific parameters to compiled
    # ===================================================================

    def _inject_glm_params(self, compiled: CompiledPrompt) -> CompiledPrompt:
        """Inject GLM-specific parameters into compiled.extra before sending.

        Adds `enable_thinking` to compiled.extra when:
          - compiled.extra["thinking"] is set (OpenAI-style thinking hint)
          - compiled.extra["thinking_mode"] is set (GLM-specific hint)
          - compiled.extra["preserve_thinking"] is True
          - self.enable_thinking_default is True

        This ensures the GLM API receives `enable_thinking=True` in the
        request body instead of the OpenAI-style `thinking` object.

        Args:
            compiled: The compiled prompt to inject parameters into.

        Returns:
            The same CompiledPrompt with modified extra dict.
        """
        # Determine if thinking should be enabled
        should_enable_thinking = False

        # Check for OpenAI-style thinking hint
        thinking = compiled.extra.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") == "enabled":
            should_enable_thinking = True

        # Check for GLM-specific thinking_mode hint
        thinking_mode = compiled.extra.get("thinking_mode")
        if thinking_mode in ("deep", "high", "max", True):
            should_enable_thinking = True

        # Check for preserve_thinking hint (used in Coding Plan scenarios)
        if compiled.extra.get("preserve_thinking"):
            should_enable_thinking = True

        # Check default configuration
        if self.enable_thinking_default:
            should_enable_thinking = True

        # Explicit disable takes precedence
        if isinstance(thinking, dict) and thinking.get("type") == "disabled":
            should_enable_thinking = False
        if thinking_mode == "quick" or thinking_mode is False:
            should_enable_thinking = False

        if should_enable_thinking:
            compiled.extra["enable_thinking"] = True

        return compiled

    # ===================================================================
    # Override: send — Enhanced with GLM-specific response parsing
    # ===================================================================

    async def send(self, compiled: CompiledPrompt, model: str) -> TAPResponse:
        """Send a compiled prompt with GLM-specific enhancements.

        Before sending, injects GLM-specific parameters (enable_thinking).
        After receiving, post-processes the response for GLM-specific fields:
          - reasoning_content → TAPResponse.thinking_content + extra
          - cached_tokens → TAPResponse.cache_hit_tokens + extra
          - content_filter → TAPResponse.extra
        """
        # Inject GLM-specific parameters
        compiled = self._inject_glm_params(compiled)

        # Use async interface for long-horizon tasks
        if self.async_enabled and compiled.extra.get("use_async"):
            return await self.async_chat_completion(compiled, model)

        # 使用自定义的流式累积，同时捕获 reasoning_content
        # （父类 send() 的流式累积只捕获 content delta，丢失 reasoning_content）
        response = await self._send_with_reasoning(compiled, model)

        # Post-process GLM-specific response fields
        response = self._post_process_response(response)

        return response

    async def _send_non_streaming(
        self, compiled: CompiledPrompt, model: str
    ) -> TAPResponse:
        """Non-streaming fallback with GLM-specific response parsing."""
        # Inject GLM-specific parameters
        compiled = self._inject_glm_params(compiled)

        response = await super()._send_non_streaming(compiled, model)

        # Post-process GLM-specific response fields
        response = self._post_process_response(response)

        return response

    # ===================================================================
    # GLM-specific: Send with reasoning_content capture
    # ===================================================================

    async def _send_with_reasoning(self, compiled: CompiledPrompt, model: str) -> TAPResponse:
        """Send request with reasoning_content capture from streaming.

        父类 send() 的内部流式累积只捕获 delta.content，丢失 GLM 特有的
        delta.reasoning_content。此方法覆盖流式累积逻辑，同时捕获两者。

        Args:
            compiled: 已编译的提示
            model: 模型名称

        Returns:
            包含完整 reasoning_content 的 TAPResponse
        """
        import json

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

        if compiled.extra:
            payload.update(compiled.extra)

        if compiled.tools:
            payload["tools"] = compiled.tools
            if compiled.tool_choice is not None:
                payload["tool_choice"] = compiled.tool_choice

        # 流式累积
        collected: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict = {}
        last_chunk_data: dict = {}
        tool_call_accumulators: dict[int, dict] = {}

        has_multimodal = self._detect_multimodal_in_messages(messages)
        client = await self._get_client(multimodal=has_multimodal)

        try:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            if chunk.get("error"):
                                err_obj = chunk["error"]
                                err_msg = (
                                    err_obj.get("message", str(err_obj))
                                    if isinstance(err_obj, dict)
                                    else str(err_obj)
                                )
                                logger.error(f"GLMNativeAdapter: API error in send: {err_msg}")
                                break

                            delta = self._safe_get_choice(chunk).get("delta", {})

                            # 标准内容
                            content = delta.get("content", "")
                            if content:
                                collected.append(content)

                            # GLM 特有：reasoning_content（思考过程）
                            reasoning_delta = delta.get("reasoning_content", "")
                            if reasoning_delta:
                                reasoning_parts.append(reasoning_delta)

                            # 工具调用累积
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

                            if chunk.get("usage"):
                                usage = chunk["usage"]
                            last_chunk_data = chunk

                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (400, 405, 501):
                logger.warning(f"GLMNativeAdapter: streaming not supported, falling back")
                return await self._send_non_streaming(compiled, model)
            raise

        # 构建 TAPResponse
        raw_text = "".join(collected)
        tool_calls = [tool_call_accumulators[i] for i in sorted(tool_call_accumulators)]
        finish_reason = "stop"
        if last_chunk_data:
            choice = self._safe_get_choice(last_chunk_data)
            finish_reason = choice.get("finish_reason", "stop") or "stop"

        # 提取缓存命中 token
        cache_hit_tokens = 0
        if usage:
            cache_hit_tokens = usage.get("prompt_cache_hit_tokens", 0)

        # 提取 reasoning_content
        reasoning_content = "".join(reasoning_parts) if reasoning_parts else None

        extra: dict = {}
        if reasoning_content:
            extra["reasoning_content"] = reasoning_content
        if cache_hit_tokens:
            extra["cached_tokens"] = cache_hit_tokens

        return TAPResponse(
            raw_text=raw_text or "",
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            cache_hit_tokens=cache_hit_tokens,
            thinking_content=reasoning_content,
            extra=extra,
        )

    # ===================================================================
    # GLM-specific: Response post-processing
    # ===================================================================

    def _post_process_response(self, response: TAPResponse) -> TAPResponse:
        """Post-process TAPResponse for GLM-specific fields.

        Extracts:
          - reasoning_content from usage or response metadata
          - cached_tokens from prompt_tokens_details
          - content_filter from response metadata

        Also updates the internal cache tracker.

        Args:
            response: TAPResponse from parent's send().

        Returns:
            The same TAPResponse with GLM-specific fields populated.
        """
        # Extract reasoning_content from usage metadata
        # GLM returns reasoning_content at the same level as content in the
        # message object. The parent class extracts `content` into `raw_text`,
        # but `reasoning_content` is preserved in usage metadata by our
        # streaming handler, or it may be available in the raw response.
        usage = response.usage
        if usage:
            # Extract cached_tokens from prompt_tokens_details
            if self.cache_tracking:
                prompt_tokens_details = usage.get("prompt_tokens_details", {})
                if isinstance(prompt_tokens_details, dict):
                    cached_tokens = prompt_tokens_details.get("cached_tokens", 0)
                    if cached_tokens:
                        response.extra["cached_tokens"] = cached_tokens
                        # Also update the primary cache_hit_tokens field
                        if response.cache_hit_tokens == 0:
                            response.cache_hit_tokens = cached_tokens
                        # Update internal tracker
                        self._cache_tracker["total_cached_tokens"] += cached_tokens

                # Update cache tracker totals
                self._cache_tracker["total_prompt_tokens"] += usage.get(
                    "prompt_tokens", 0
                )
                self._cache_tracker["total_requests"] += 1

        # Extract reasoning_content if present in extra (set during streaming)
        reasoning = response.extra.get("reasoning_content")
        if reasoning and not response.thinking_content:
            response.thinking_content = reasoning

        return response

    # ===================================================================
    # Override: stream — Handle GLM-specific streaming events
    # ===================================================================

    async def stream(self, compiled: CompiledPrompt, model: str) -> AsyncIterator[str]:
        """Stream with GLM-specific event handling.

        Handles GLM-specific streaming events including:
          - reasoning_content delta (thinking mode chain-of-thought)

        The reasoning_content is accumulated and can be retrieved from
        the adapter after streaming completes via `last_reasoning_content`.
        """
        # Inject GLM-specific parameters
        compiled = self._inject_glm_params(compiled)

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

        # Send compiled.tools if present
        if compiled.tools:
            payload["tools"] = compiled.tools
            if compiled.tool_choice is not None:
                payload["tool_choice"] = compiled.tool_choice

        # Track reasoning content from streaming
        reasoning_parts: list[str] = []

        # Detect multimodal content for timeout selection
        has_multimodal = self._detect_multimodal_in_messages(messages)
        client = await self._get_client(multimodal=has_multimodal)

        try:
            async with client.stream(
                "POST", url, json=payload, headers=headers
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            if chunk.get("error"):
                                err_obj = chunk["error"]
                                err_msg = (
                                    err_obj.get("message", str(err_obj))
                                    if isinstance(err_obj, dict)
                                    else str(err_obj)
                                )
                                logger.error(
                                    f"GLMNativeAdapter: stream error from API: {err_msg}"
                                )
                                break

                            delta = self._safe_get_choice(chunk).get("delta", {})

                            # Handle standard content delta
                            content = delta.get("content", "")
                            if content:
                                yield content

                            # Handle GLM-specific reasoning_content delta
                            reasoning_delta = delta.get("reasoning_content", "")
                            if reasoning_delta:
                                reasoning_parts.append(reasoning_delta)

                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (400, 405, 501):
                logger.warning(
                    f"GLMNativeAdapter: streaming not supported "
                    f"({e.response.status_code}), falling back"
                )
                # Fall back to non-streaming send
                response = await self._send_non_streaming(compiled, model)
                if response.raw_text:
                    yield response.raw_text
                return
            raise

        # Store accumulated reasoning content for later retrieval
        if reasoning_parts:
            self._last_reasoning_content = "".join(reasoning_parts)
        else:
            self._last_reasoning_content = ""

    # ===================================================================
    # GLM-specific: Last reasoning content accessor
    # ===================================================================

    @property
    def last_reasoning_content(self) -> str:
        """Return the reasoning_content from the most recent stream.

        After calling stream(), this property contains the accumulated
        reasoning_content from the GLM response. Empty string if no
        reasoning_content was received.
        """
        return getattr(self, "_last_reasoning_content", "")

    # ===================================================================
    # GLM-specific: Async chat completion
    # ===================================================================

    async def async_chat_completion(
        self,
        compiled: CompiledPrompt,
        model: str,
        poll_interval: float = _ASYNC_POLL_INTERVAL,
        poll_timeout: float = _ASYNC_POLL_TIMEOUT,
    ) -> TAPResponse:
        """Submit an async chat completion request and poll for results.

        GLM provides a /chat/completions/async endpoint for long-running
        tasks that might exceed normal HTTP timeouts. This method:
          1. POSTs the request to the async endpoint
          2. Receives a task_id
          3. Polls the task status endpoint until completion
          4. Returns the completed TAPResponse

        Args:
            compiled: The compiled prompt to send.
            model: Model identifier string.
            poll_interval: Seconds between status polls (default 2.0).
            poll_timeout: Maximum seconds to wait for completion (default 600.0).

        Returns:
            TAPResponse with the completed result.

        Raises:
            RuntimeError: If the async task fails or times out.
            httpx.HTTPStatusError: On API errors.
        """
        if not self.api_key:
            raise ValueError(
                "GLMNativeAdapter: api_key is required for async chat completion"
            )

        # Inject GLM-specific parameters
        compiled = self._inject_glm_params(compiled)

        # Build the async request
        url = f"{self.base_url}{_ASYNC_CHAT_ENDPOINT}"
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

        # Pass compiled.extra as extra_body
        if compiled.extra:
            payload.update(compiled.extra)

        # Send compiled.tools if present
        if compiled.tools:
            payload["tools"] = compiled.tools
            if compiled.tool_choice is not None:
                payload["tool_choice"] = compiled.tool_choice

        # Step 1: Submit async request
        client = await self._get_client()
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        submit_data = response.json()
        task_id = submit_data.get("id") or submit_data.get("task_id", "")

        if not task_id:
            raise RuntimeError(
                f"GLMNativeAdapter: async chat submission did not return a task_id. "
                f"Response: {submit_data}"
            )

        logger.info(
            f"GLMNativeAdapter: submitted async task {task_id} for model {resolved_model}"
        )

        # Step 2: Poll for completion
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > poll_timeout:
                raise RuntimeError(
                    f"GLMNativeAdapter: async task {task_id} timed out after "
                    f"{poll_timeout}s"
                )

            # Query task status
            status_url = f"{self.base_url}{_ASYNC_TASK_STATUS_ENDPOINT}".format(
                task_id=task_id
            )
            status_response = await client.get(status_url, headers=headers)
            status_response.raise_for_status()
            status_data = status_response.json()

            task_status = status_data.get("task_status", "")

            if task_status == "SUCCESS" or task_status == "COMPLETED":
                # Task completed — extract result
                logger.info(
                    f"GLMNativeAdapter: async task {task_id} completed successfully"
                )
                return self._parse_async_response(status_data, compiled)

            elif task_status in ("FAILED", "ERROR"):
                error_msg = status_data.get("error", "Unknown error")
                raise RuntimeError(
                    f"GLMNativeAdapter: async task {task_id} failed: {error_msg}"
                )

            elif task_status in ("PENDING", "PROCESSING", "RUNNING"):
                # Still in progress — wait and retry
                await asyncio.sleep(poll_interval)
                continue

            else:
                # Unknown status — log and continue polling
                logger.warning(
                    f"GLMNativeAdapter: unknown async task status '{task_status}' "
                    f"for task {task_id}, continuing to poll"
                )
                await asyncio.sleep(poll_interval)
                continue

    def _parse_async_response(
        self, status_data: dict, compiled: CompiledPrompt
    ) -> TAPResponse:
        """Parse the completed async task response into TAPResponse.

        Args:
            status_data: The status endpoint response data.
            compiled: The original compiled prompt (for context).

        Returns:
            TAPResponse with the completed result.
        """
        # The async response may nest the actual chat completion under
        # different keys depending on the GLM API version
        result = status_data.get("result") or status_data.get("choices", [])
        content = ""
        reasoning_content = ""
        usage = status_data.get("usage", {})
        content_filter = status_data.get("content_filter")
        tool_calls = []
        finish_reason = "stop"

        if isinstance(result, list) and result:
            # Standard OpenAI-like choices format
            choice = result[0] if isinstance(result[0], dict) else {}
            message = choice.get("message", {})
            content = message.get("content") or ""
            reasoning_content = message.get("reasoning_content", "")
            tool_calls = message.get("tool_calls", [])
            finish_reason = choice.get("finish_reason", "stop") or "stop"
        elif isinstance(result, dict):
            # Direct result object
            content = result.get("content") or ""
            reasoning_content = result.get("reasoning_content", "")
            tool_calls = result.get("tool_calls", [])
        elif isinstance(result, str):
            content = result

        # Build the response
        response = TAPResponse(
            raw_text=content or "",
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            cache_hit_tokens=0,
        )

        # Set reasoning content
        if reasoning_content:
            response.thinking_content = reasoning_content
            response.extra["reasoning_content"] = reasoning_content

        # Set content_filter
        if content_filter:
            response.extra["content_filter"] = content_filter

        # Extract cached_tokens from prompt_tokens_details
        if self.cache_tracking and usage:
            prompt_tokens_details = usage.get("prompt_tokens_details", {})
            if isinstance(prompt_tokens_details, dict):
                cached_tokens = prompt_tokens_details.get("cached_tokens", 0)
                if cached_tokens:
                    response.extra["cached_tokens"] = cached_tokens
                    response.cache_hit_tokens = cached_tokens

        return response

    # ===================================================================
    # Override: Model name mapping
    # ===================================================================

    _GLM_MODEL_NAME_MAP: dict[str, str] = {
        # GLM-5 family aliases → canonical GLM model names
        "glm5": "glm-5",
        "glm-5": "glm-5",
        "glm-5.1": "glm-5.1",
        "glm51": "glm-5.1",
        "glm-5.2": "glm-5.2",
        "glm52": "glm-5.2",
        "glm_52": "glm-5.2",
        "glm_51": "glm-5.1",
        "glm_5": "glm-5",
    }

    def _resolve_model_name(self, model: str) -> str:
        """Resolve model name with GLM-specific mappings.

        First checks GLM-specific aliases, then falls back to
        parent's model name resolution (DeepSeek V4, etc.).

        Args:
            model: Model name string.

        Returns:
            Resolved model name.
        """
        # Check GLM-specific aliases first
        resolved = self._GLM_MODEL_NAME_MAP.get(model)
        if resolved:
            return resolved

        # Fall back to parent's resolution
        return super()._resolve_model_name(model)

    # ===================================================================
    # Override: Capabilities
    # ===================================================================

    @property
    def capabilities(self) -> dict:
        """Return adapter capabilities with GLM-specific extensions.

        Extends parent capabilities with:
          - thinking: True (GLM enable_thinking support)
          - cache_tracking: True (cached_tokens from prompt_tokens_details)
          - async_chat: True (async completion endpoint)
          - content_filter: True (content safety filtering)
        """
        caps = super().capabilities
        caps.update({
            "thinking": True,
            "cache_tracking": self.cache_tracking,
            "async_chat": self.async_enabled,
            "content_filter": True,
            "max_context_tokens": 1_000_000,  # GLM-5.2 supports 1M context
        })
        return caps

    # ===================================================================
    # GLM-specific: Cache tracker summary
    # ===================================================================

    @property
    def cache_summary(self) -> dict[str, int]:
        """Return a copy of the cumulative cache tracking summary.

        Returns:
            Dict with cumulative cached_tokens, prompt_tokens, and request count.
        """
        return dict(self._cache_tracker)


# ---------------------------------------------------------------------------
# Register with TAPAdapterRegistry
# ---------------------------------------------------------------------------

TAPAdapterRegistry.register("glm_native", GLMNativeAdapter)
