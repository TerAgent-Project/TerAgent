"""tests.test_glm_native — Tests for GLMNativeAdapter

Tests cover:
  - Adapter initialization with defaults
  - enable_thinking parameter injection
  - reasoning_content extraction from response
  - cached_tokens parsing
  - content_filter handling
  - Async chat completion flow (mock HTTP)
  - Model name resolution
  - Streaming with reasoning_content
  - Registry registration
  - Cache tracker
  - Capabilities
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from teragent.core.adapter import TAPAdapterRegistry
from teragent.core.adapters.glm_native import (
    GLM_DEFAULT_BASE_URL,
    GLMNativeAdapter,
    _ASYNC_CHAT_ENDPOINT,
    _ASYNC_POLL_INTERVAL,
    _ASYNC_POLL_TIMEOUT,
    _ASYNC_TASK_STATUS_ENDPOINT,
)
from teragent.core.tap import CompiledPrompt, TAPResponse


# ============================================================================
# Adapter Initialization
# ============================================================================


class TestGLMNativeAdapterInit:
    """Tests for GLMNativeAdapter initialization."""

    def test_default_init(self):
        adapter = GLMNativeAdapter()
        assert adapter.base_url == GLM_DEFAULT_BASE_URL
        assert adapter.api_key == ""
        assert adapter.enable_thinking_default is False
        assert adapter.async_enabled is True
        assert adapter.cache_tracking is True
        assert adapter._cache_tracker["total_requests"] == 0

    def test_default_base_url(self):
        """Default base URL should be GLM's official endpoint."""
        assert GLM_DEFAULT_BASE_URL == "https://open.bigmodel.cn/api/paas/v4"

    def test_init_with_custom_params(self):
        adapter = GLMNativeAdapter(
            base_url="https://custom.glm.api.com/v4",
            api_key="test-key",
            timeout=120.0,
            enable_thinking_default=True,
            async_enabled=False,
            cache_tracking=False,
        )
        assert adapter.base_url == "https://custom.glm.api.com/v4"
        assert adapter.api_key == "test-key"
        assert adapter.enable_thinking_default is True
        assert adapter.async_enabled is False
        assert adapter.cache_tracking is False

    def test_init_strips_trailing_slash(self):
        adapter = GLMNativeAdapter(base_url="https://open.bigmodel.cn/api/paas/v4/")
        assert adapter.base_url == "https://open.bigmodel.cn/api/paas/v4"

    def test_registry_registered(self):
        """Verify GLMNativeAdapter is registered in the adapter registry."""
        adapter = TAPAdapterRegistry.create("glm_native")
        assert isinstance(adapter, GLMNativeAdapter)

    def test_registry_with_kwargs(self):
        """Verify adapter can be created with keyword arguments via registry."""
        adapter = TAPAdapterRegistry.create(
            "glm_native",
            api_key="test-key",
            enable_thinking_default=True,
        )
        assert isinstance(adapter, GLMNativeAdapter)
        assert adapter.api_key == "test-key"
        assert adapter.enable_thinking_default is True


# ============================================================================
# enable_thinking Parameter Injection
# ============================================================================


class TestGLMEnableThinking:
    """Tests for GLM enable_thinking parameter injection."""

    def test_inject_thinking_from_openai_style(self):
        """OpenAI-style thinking hint should add enable_thinking=True."""
        adapter = GLMNativeAdapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "test"}],
            extra={"thinking": {"type": "enabled"}},
        )
        result = adapter._inject_glm_params(compiled)
        assert result.extra.get("enable_thinking") is True

    def test_inject_thinking_from_thinking_mode_deep(self):
        """thinking_mode='deep' should add enable_thinking=True."""
        adapter = GLMNativeAdapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "test"}],
            extra={"thinking_mode": "deep"},
        )
        result = adapter._inject_glm_params(compiled)
        assert result.extra.get("enable_thinking") is True

    def test_inject_thinking_from_thinking_mode_high(self):
        """thinking_mode='high' should add enable_thinking=True."""
        adapter = GLMNativeAdapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "test"}],
            extra={"thinking_mode": "high"},
        )
        result = adapter._inject_glm_params(compiled)
        assert result.extra.get("enable_thinking") is True

    def test_inject_thinking_from_thinking_mode_max(self):
        """thinking_mode='max' should add enable_thinking=True."""
        adapter = GLMNativeAdapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "test"}],
            extra={"thinking_mode": "max"},
        )
        result = adapter._inject_glm_params(compiled)
        assert result.extra.get("enable_thinking") is True

    def test_inject_thinking_from_preserve_thinking(self):
        """preserve_thinking=True should add enable_thinking=True."""
        adapter = GLMNativeAdapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "test"}],
            extra={"preserve_thinking": True},
        )
        result = adapter._inject_glm_params(compiled)
        assert result.extra.get("enable_thinking") is True

    def test_inject_thinking_from_default_config(self):
        """enable_thinking_default=True should add enable_thinking=True."""
        adapter = GLMNativeAdapter(enable_thinking_default=True)
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "test"}],
            extra={},
        )
        result = adapter._inject_glm_params(compiled)
        assert result.extra.get("enable_thinking") is True

    def test_no_inject_thinking_when_disabled(self):
        """OpenAI-style thinking=disabled should NOT add enable_thinking."""
        adapter = GLMNativeAdapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "test"}],
            extra={"thinking": {"type": "disabled"}},
        )
        result = adapter._inject_glm_params(compiled)
        assert result.extra.get("enable_thinking") is None

    def test_no_inject_thinking_when_quick(self):
        """thinking_mode='quick' should NOT add enable_thinking."""
        adapter = GLMNativeAdapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "test"}],
            extra={"thinking_mode": "quick"},
        )
        result = adapter._inject_glm_params(compiled)
        assert result.extra.get("enable_thinking") is None

    def test_no_inject_thinking_when_no_hints(self):
        """No thinking hints and default=False should NOT add enable_thinking."""
        adapter = GLMNativeAdapter(enable_thinking_default=False)
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "test"}],
            extra={},
        )
        result = adapter._inject_glm_params(compiled)
        assert result.extra.get("enable_thinking") is None

    def test_disable_overrides_default(self):
        """Explicit disable should override enable_thinking_default=True."""
        adapter = GLMNativeAdapter(enable_thinking_default=True)
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "test"}],
            extra={"thinking": {"type": "disabled"}},
        )
        result = adapter._inject_glm_params(compiled)
        assert result.extra.get("enable_thinking") is None

    def test_thinking_mode_true(self):
        """thinking_mode=True should add enable_thinking=True."""
        adapter = GLMNativeAdapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "test"}],
            extra={"thinking_mode": True},
        )
        result = adapter._inject_glm_params(compiled)
        assert result.extra.get("enable_thinking") is True

    def test_thinking_mode_false(self):
        """thinking_mode=False should NOT add enable_thinking."""
        adapter = GLMNativeAdapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "test"}],
            extra={"thinking_mode": False},
        )
        result = adapter._inject_glm_params(compiled)
        assert result.extra.get("enable_thinking") is None


# ============================================================================
# reasoning_content Extraction
# ============================================================================


class TestGLMReasoningContent:
    """Tests for GLM reasoning_content extraction from responses."""

    def test_post_process_sets_reasoning_from_extra(self):
        """reasoning_content in extra should populate thinking_content."""
        adapter = GLMNativeAdapter()
        response = TAPResponse(
            raw_text="Final answer",
            usage={},
            extra={"reasoning_content": "Let me think about this..."},
        )
        result = adapter._post_process_response(response)
        assert result.thinking_content == "Let me think about this..."
        assert result.extra["reasoning_content"] == "Let me think about this..."

    def test_post_process_does_not_overwrite_thinking_content(self):
        """If thinking_content already set, extra shouldn't overwrite."""
        adapter = GLMNativeAdapter()
        response = TAPResponse(
            raw_text="Final answer",
            usage={},
            thinking_content="Original thinking",
            extra={"reasoning_content": "New thinking"},
        )
        result = adapter._post_process_response(response)
        # thinking_content should remain as-is (already set)
        assert result.thinking_content == "Original thinking"

    def test_post_process_no_reasoning(self):
        """No reasoning_content in extra should leave thinking_content as None."""
        adapter = GLMNativeAdapter()
        response = TAPResponse(
            raw_text="Simple answer",
            usage={},
        )
        result = adapter._post_process_response(response)
        assert result.thinking_content is None


# ============================================================================
# cached_tokens Parsing
# ============================================================================


class TestGLMCachedTokens:
    """Tests for GLM cached_tokens parsing from prompt_tokens_details."""

    def test_extract_cached_tokens(self):
        """cached_tokens from prompt_tokens_details should populate response."""
        adapter = GLMNativeAdapter(cache_tracking=True)
        response = TAPResponse(
            raw_text="Answer",
            usage={
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "prompt_tokens_details": {
                    "cached_tokens": 800,
                },
            },
        )
        result = adapter._post_process_response(response)
        assert result.extra["cached_tokens"] == 800
        assert result.cache_hit_tokens == 800

    def test_cached_tokens_updates_tracker(self):
        """cached_tokens should update the internal cache tracker."""
        adapter = GLMNativeAdapter(cache_tracking=True)
        response = TAPResponse(
            raw_text="Answer",
            usage={
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "prompt_tokens_details": {
                    "cached_tokens": 800,
                },
            },
        )
        adapter._post_process_response(response)
        assert adapter._cache_tracker["total_cached_tokens"] == 800
        assert adapter._cache_tracker["total_prompt_tokens"] == 1000
        assert adapter._cache_tracker["total_requests"] == 1

    def test_cached_tokens_accumulates(self):
        """Cache tracker should accumulate across multiple responses."""
        adapter = GLMNativeAdapter(cache_tracking=True)
        r1 = TAPResponse(
            raw_text="1",
            usage={
                "prompt_tokens": 1000,
                "prompt_tokens_details": {"cached_tokens": 500},
            },
        )
        r2 = TAPResponse(
            raw_text="2",
            usage={
                "prompt_tokens": 2000,
                "prompt_tokens_details": {"cached_tokens": 1000},
            },
        )
        adapter._post_process_response(r1)
        adapter._post_process_response(r2)
        assert adapter._cache_tracker["total_cached_tokens"] == 1500
        assert adapter._cache_tracker["total_prompt_tokens"] == 3000
        assert adapter._cache_tracker["total_requests"] == 2

    def test_no_cached_tokens_when_tracking_disabled(self):
        """cache_tracking=False should not extract cached_tokens."""
        adapter = GLMNativeAdapter(cache_tracking=False)
        response = TAPResponse(
            raw_text="Answer",
            usage={
                "prompt_tokens": 1000,
                "prompt_tokens_details": {
                    "cached_tokens": 800,
                },
            },
        )
        result = adapter._post_process_response(response)
        assert "cached_tokens" not in result.extra
        assert result.cache_hit_tokens == 0

    def test_missing_prompt_tokens_details(self):
        """Missing prompt_tokens_details should not crash."""
        adapter = GLMNativeAdapter(cache_tracking=True)
        response = TAPResponse(
            raw_text="Answer",
            usage={
                "prompt_tokens": 1000,
                "completion_tokens": 100,
            },
        )
        result = adapter._post_process_response(response)
        assert "cached_tokens" not in result.extra

    def test_zero_cached_tokens(self):
        """cached_tokens=0 should not set extra."""
        adapter = GLMNativeAdapter(cache_tracking=True)
        response = TAPResponse(
            raw_text="Answer",
            usage={
                "prompt_tokens": 1000,
                "prompt_tokens_details": {
                    "cached_tokens": 0,
                },
            },
        )
        result = adapter._post_process_response(response)
        assert "cached_tokens" not in result.extra

    def test_cache_summary_property(self):
        """cache_summary should return a copy of the tracker."""
        adapter = GLMNativeAdapter(cache_tracking=True)
        response = TAPResponse(
            raw_text="Answer",
            usage={
                "prompt_tokens": 1000,
                "prompt_tokens_details": {"cached_tokens": 500},
            },
        )
        adapter._post_process_response(response)
        summary = adapter.cache_summary
        assert summary["total_cached_tokens"] == 500
        assert summary["total_prompt_tokens"] == 1000
        assert summary["total_requests"] == 1
        # Verify it's a copy
        summary["total_cached_tokens"] = 0
        assert adapter._cache_tracker["total_cached_tokens"] == 500


# ============================================================================
# content_filter Handling
# ============================================================================


class TestGLMContentFilter:
    """Tests for GLM content_filter handling."""

    def test_content_filter_in_async_response(self):
        """content_filter should be preserved in TAPResponse.extra."""
        adapter = GLMNativeAdapter()
        status_data = {
            "task_status": "SUCCESS",
            "choices": [
                {
                    "message": {"content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
            "content_filter": {
                "level": "safe",
                "triggered": False,
            },
        }
        compiled = CompiledPrompt(messages=[{"role": "user", "content": "test"}])
        result = adapter._parse_async_response(status_data, compiled)
        assert result.extra["content_filter"]["level"] == "safe"
        assert result.extra["content_filter"]["triggered"] is False

    def test_no_content_filter(self):
        """Missing content_filter should not add extra key."""
        adapter = GLMNativeAdapter()
        status_data = {
            "task_status": "SUCCESS",
            "choices": [
                {
                    "message": {"content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }
        compiled = CompiledPrompt(messages=[{"role": "user", "content": "test"}])
        result = adapter._parse_async_response(status_data, compiled)
        assert "content_filter" not in result.extra


# ============================================================================
# Async Chat Completion
# ============================================================================


class TestGLMAsyncChat:
    """Tests for GLM async chat completion flow."""

    @pytest.mark.asyncio
    async def test_async_requires_api_key(self):
        """async_chat_completion should raise ValueError without api_key."""
        adapter = GLMNativeAdapter(api_key="")
        compiled = CompiledPrompt(messages=[{"role": "user", "content": "test"}])
        with pytest.raises(ValueError, match="api_key is required"):
            await adapter.async_chat_completion(compiled, "glm-5")

    @pytest.mark.asyncio
    async def test_async_submit_and_complete(self):
        """Full async flow: submit → poll → complete."""
        adapter = GLMNativeAdapter(api_key="test-key")

        # Mock HTTP client
        mock_client = AsyncMock()

        # Mock the submit response
        submit_response = MagicMock()
        submit_response.json.return_value = {
            "id": "task-123",
            "task_status": "PENDING",
        }
        submit_response.raise_for_status = MagicMock()

        # Mock the poll response (first poll: still processing, second: success)
        poll_response_processing = MagicMock()
        poll_response_processing.json.return_value = {
            "task_status": "PROCESSING",
        }
        poll_response_processing.raise_for_status = MagicMock()

        poll_response_done = MagicMock()
        poll_response_done.json.return_value = {
            "task_status": "SUCCESS",
            "choices": [
                {
                    "message": {"content": "Async result"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        poll_response_done.raise_for_status = MagicMock()

        # Set up the mock client calls
        mock_client.post.return_value = submit_response
        mock_client.get.side_effect = [
            poll_response_processing,
            poll_response_done,
        ]

        # Patch _get_client to return our mock
        with patch.object(adapter, "_get_client", return_value=mock_client):
            compiled = CompiledPrompt(
                messages=[{"role": "user", "content": "Long task"}],
            )
            result = await adapter.async_chat_completion(
                compiled, "glm-5", poll_interval=0.01, poll_timeout=10.0
            )

        assert result.raw_text == "Async result"
        assert result.usage["prompt_tokens"] == 100
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_async_timeout(self):
        """Async task should raise RuntimeError on timeout."""
        adapter = GLMNativeAdapter(api_key="test-key")

        mock_client = AsyncMock()

        # Submit succeeds
        submit_response = MagicMock()
        submit_response.json.return_value = {"id": "task-456"}
        submit_response.raise_for_status = MagicMock()
        mock_client.post.return_value = submit_response

        # Poll always returns PROCESSING
        poll_response = MagicMock()
        poll_response.json.return_value = {"task_status": "PROCESSING"}
        poll_response.raise_for_status = MagicMock()
        mock_client.get.return_value = poll_response

        with patch.object(adapter, "_get_client", return_value=mock_client):
            compiled = CompiledPrompt(
                messages=[{"role": "user", "content": "Long task"}],
            )
            with pytest.raises(RuntimeError, match="timed out"):
                await adapter.async_chat_completion(
                    compiled, "glm-5", poll_interval=0.01, poll_timeout=0.05
                )

    @pytest.mark.asyncio
    async def test_async_task_failed(self):
        """Async task failure should raise RuntimeError."""
        adapter = GLMNativeAdapter(api_key="test-key")

        mock_client = AsyncMock()

        submit_response = MagicMock()
        submit_response.json.return_value = {"id": "task-789"}
        submit_response.raise_for_status = MagicMock()
        mock_client.post.return_value = submit_response

        poll_response = MagicMock()
        poll_response.json.return_value = {
            "task_status": "FAILED",
            "error": "Content policy violation",
        }
        poll_response.raise_for_status = MagicMock()
        mock_client.get.return_value = poll_response

        with patch.object(adapter, "_get_client", return_value=mock_client):
            compiled = CompiledPrompt(
                messages=[{"role": "user", "content": "test"}],
            )
            with pytest.raises(RuntimeError, match="failed"):
                await adapter.async_chat_completion(
                    compiled, "glm-5", poll_interval=0.01, poll_timeout=10.0
                )

    @pytest.mark.asyncio
    async def test_async_no_task_id(self):
        """Submit without task_id should raise RuntimeError."""
        adapter = GLMNativeAdapter(api_key="test-key")

        mock_client = AsyncMock()

        submit_response = MagicMock()
        submit_response.json.return_value = {"status": "unknown"}
        submit_response.raise_for_status = MagicMock()
        mock_client.post.return_value = submit_response

        with patch.object(adapter, "_get_client", return_value=mock_client):
            compiled = CompiledPrompt(
                messages=[{"role": "user", "content": "test"}],
            )
            with pytest.raises(RuntimeError, match="task_id"):
                await adapter.async_chat_completion(
                    compiled, "glm-5", poll_interval=0.01, poll_timeout=10.0
                )

    @pytest.mark.asyncio
    async def test_async_with_reasoning_content(self):
        """Async response with reasoning_content should be extracted."""
        adapter = GLMNativeAdapter(api_key="test-key")

        mock_client = AsyncMock()

        submit_response = MagicMock()
        submit_response.json.return_value = {"id": "task-thinking"}
        submit_response.raise_for_status = MagicMock()
        mock_client.post.return_value = submit_response

        poll_response = MagicMock()
        poll_response.json.return_value = {
            "task_status": "SUCCESS",
            "choices": [
                {
                    "message": {
                        "content": "Final answer",
                        "reasoning_content": "Step 1: Think... Step 2: Reason...",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 500},
        }
        poll_response.raise_for_status = MagicMock()
        mock_client.get.return_value = poll_response

        with patch.object(adapter, "_get_client", return_value=mock_client):
            compiled = CompiledPrompt(
                messages=[{"role": "user", "content": "Think hard"}],
            )
            result = await adapter.async_chat_completion(
                compiled, "glm-5", poll_interval=0.01, poll_timeout=10.0
            )

        assert result.raw_text == "Final answer"
        assert result.thinking_content == "Step 1: Think... Step 2: Reason..."
        assert result.extra["reasoning_content"] == "Step 1: Think... Step 2: Reason..."

    @pytest.mark.asyncio
    async def test_async_with_cached_tokens(self):
        """Async response with cached_tokens should be extracted."""
        adapter = GLMNativeAdapter(api_key="test-key", cache_tracking=True)

        mock_client = AsyncMock()

        submit_response = MagicMock()
        submit_response.json.return_value = {"id": "task-cache"}
        submit_response.raise_for_status = MagicMock()
        mock_client.post.return_value = submit_response

        poll_response = MagicMock()
        poll_response.json.return_value = {
            "task_status": "SUCCESS",
            "choices": [
                {
                    "message": {"content": "Cached answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 2000,
                "prompt_tokens_details": {"cached_tokens": 1500},
            },
        }
        poll_response.raise_for_status = MagicMock()
        mock_client.get.return_value = poll_response

        with patch.object(adapter, "_get_client", return_value=mock_client):
            compiled = CompiledPrompt(
                messages=[{"role": "user", "content": "test"}],
            )
            result = await adapter.async_chat_completion(
                compiled, "glm-5", poll_interval=0.01, poll_timeout=10.0
            )

        assert result.extra["cached_tokens"] == 1500
        assert result.cache_hit_tokens == 1500


# ============================================================================
# Model Name Resolution
# ============================================================================


class TestGLMModelNameResolution:
    """Tests for GLM model name mapping."""

    def test_glm5_alias(self):
        adapter = GLMNativeAdapter()
        assert adapter._resolve_model_name("glm5") == "glm-5"

    def test_glm5_canonical(self):
        adapter = GLMNativeAdapter()
        assert adapter._resolve_model_name("glm-5") == "glm-5"

    def test_glm51_alias(self):
        adapter = GLMNativeAdapter()
        assert adapter._resolve_model_name("glm51") == "glm-5.1"

    def test_glm51_canonical(self):
        adapter = GLMNativeAdapter()
        assert adapter._resolve_model_name("glm-5.1") == "glm-5.1"

    def test_glm52_alias(self):
        adapter = GLMNativeAdapter()
        assert adapter._resolve_model_name("glm52") == "glm-5.2"

    def test_glm52_canonical(self):
        adapter = GLMNativeAdapter()
        assert adapter._resolve_model_name("glm-5.2") == "glm-5.2"

    def test_glm_52_underscore(self):
        adapter = GLMNativeAdapter()
        assert adapter._resolve_model_name("glm_52") == "glm-5.2"

    def test_glm_51_underscore(self):
        adapter = GLMNativeAdapter()
        assert adapter._resolve_model_name("glm_51") == "glm-5.1"

    def test_glm_5_underscore(self):
        adapter = GLMNativeAdapter()
        assert adapter._resolve_model_name("glm_5") == "glm-5"

    def test_unknown_model_passes_through(self):
        """Unknown model names should pass through unchanged."""
        adapter = GLMNativeAdapter()
        assert adapter._resolve_model_name("some-other-model") == "some-other-model"

    def test_parent_mappings_still_work(self):
        """Parent class model mappings should still work."""
        adapter = GLMNativeAdapter()
        assert adapter._resolve_model_name("deepseek-chat") == "deepseek-v4-flash"


# ============================================================================
# Streaming with reasoning_content
# ============================================================================


class TestGLMStreaming:
    """Tests for GLM streaming with reasoning_content delta."""

    @pytest.mark.asyncio
    async def test_stream_extracts_reasoning_content(self):
        """Stream should accumulate reasoning_content deltas."""
        adapter = GLMNativeAdapter(api_key="test-key")

        # Build mock SSE lines
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"reasoning_content":"Let me think..."}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: {"choices":[{"delta":{"reasoning_content":" Step 2..."}}]}',
            'data: [DONE]',
        ]

        # Build mock streaming context manager
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        # aiter_lines() must return an async iterator directly (not a coroutine)
        mock_response.aiter_lines = MagicMock(return_value=AsyncIterator(sse_lines))

        # client.stream() must return an async context manager
        class MockStreamCM:
            async def __aenter__(self_inner):
                return mock_response
            async def __aexit__(self_inner, *args):
                pass

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=MockStreamCM())

        # _get_client is async, so patch it with an async function
        async def mock_get_client(*args, **kwargs):
            return mock_client

        with patch.object(adapter, "_get_client", side_effect=mock_get_client):
            compiled = CompiledPrompt(
                messages=[{"role": "user", "content": "test"}],
            )
            chunks = []
            async for chunk in adapter.stream(compiled, "glm-5"):
                chunks.append(chunk)

        assert "Hello" in chunks
        assert " world" in chunks
        # reasoning_content should not be yielded as content
        assert "Let me think..." not in chunks

        # But should be accumulated and accessible
        assert adapter.last_reasoning_content == "Let me think... Step 2..."

    @pytest.mark.asyncio
    async def test_stream_no_reasoning(self):
        """Stream without reasoning_content should work normally."""
        adapter = GLMNativeAdapter(api_key="test-key")

        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Just text"}}]}',
            'data: [DONE]',
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = MagicMock(return_value=AsyncIterator(sse_lines))

        class MockStreamCM:
            async def __aenter__(self_inner):
                return mock_response
            async def __aexit__(self_inner, *args):
                pass

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=MockStreamCM())

        async def mock_get_client(*args, **kwargs):
            return mock_client

        with patch.object(adapter, "_get_client", side_effect=mock_get_client):
            compiled = CompiledPrompt(
                messages=[{"role": "user", "content": "test"}],
            )
            chunks = []
            async for chunk in adapter.stream(compiled, "glm-5"):
                chunks.append(chunk)

        assert chunks == ["Just text"]
        assert adapter.last_reasoning_content == ""


# ============================================================================
# Capabilities
# ============================================================================


class TestGLMCapabilities:
    """Tests for GLMNativeAdapter capabilities."""

    def test_capabilities_includes_thinking(self):
        adapter = GLMNativeAdapter()
        assert adapter.capabilities["thinking"] is True

    def test_capabilities_includes_cache_tracking(self):
        adapter = GLMNativeAdapter(cache_tracking=True)
        assert adapter.capabilities["cache_tracking"] is True

    def test_capabilities_cache_tracking_disabled(self):
        adapter = GLMNativeAdapter(cache_tracking=False)
        assert adapter.capabilities["cache_tracking"] is False

    def test_capabilities_includes_async_chat(self):
        adapter = GLMNativeAdapter(async_enabled=True)
        assert adapter.capabilities["async_chat"] is True

    def test_capabilities_async_chat_disabled(self):
        adapter = GLMNativeAdapter(async_enabled=False)
        assert adapter.capabilities["async_chat"] is False

    def test_capabilities_includes_content_filter(self):
        adapter = GLMNativeAdapter()
        assert adapter.capabilities["content_filter"] is True

    def test_capabilities_inherits_parent(self):
        """Should inherit parent capabilities (streaming, tool_calling, etc.)."""
        adapter = GLMNativeAdapter()
        caps = adapter.capabilities
        assert caps["streaming"] is True
        assert caps["tool_calling"] is True
        assert caps["max_context_tokens"] == 1_000_000

    def test_required_mode_messages(self):
        """GLM uses OpenAI-compatible Mode A."""
        adapter = GLMNativeAdapter()
        assert adapter.required_mode == "messages"


# ============================================================================
# TAPResponse Extra Field
# ============================================================================


class TestTAPResponseExtra:
    """Tests for TAPResponse.extra field (added for GLM adapter)."""

    def test_extra_default_empty_dict(self):
        response = TAPResponse(raw_text="test")
        assert response.extra == {}

    def test_extra_preserves_data(self):
        response = TAPResponse(
            raw_text="test",
            extra={"reasoning_content": "thinking...", "cached_tokens": 500},
        )
        assert response.extra["reasoning_content"] == "thinking..."
        assert response.extra["cached_tokens"] == 500

    def test_extra_independent_instances(self):
        """Each TAPResponse should have its own extra dict."""
        r1 = TAPResponse(raw_text="1")
        r2 = TAPResponse(raw_text="2")
        r1.extra["key"] = "value1"
        assert "key" not in r2.extra


# ============================================================================
# Async Iterator Helper
# ============================================================================


class AsyncIterator:
    """Helper async iterator for mock SSE lines."""

    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration
