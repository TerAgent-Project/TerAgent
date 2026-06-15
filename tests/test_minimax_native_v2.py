"""tests.test_minimax_native_v2 — Tests for MiniMaxNativeAdapter v2 (Anthropic dual-interface)

Tests cover:
  1. Interface routing logic (openai vs anthropic selection)
  2. Anthropic-style send (message format conversion)
  3. thinking parameter passing
  4. reasoning_split response parsing
  5. count_tokens method
  6. Fallback to OpenAI when Anthropic unavailable
  7. Streaming via Anthropic interface
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from teragent.core.adapter import TAPAdapterRegistry
from teragent.core.adapters.minimax_native import (
    MINIMAX_ANTHROPIC_DEFAULT_BASE_URL,
    MINIMAX_DEFAULT_BASE_URL,
    MiniMaxNativeAdapter,
    MiniMaxRateLimitInfo,
)
from teragent.core.tap import CompiledPrompt, TAPResponse


# ============================================================================
# Helper: Create adapter with Anthropic interface enabled
# ============================================================================


def _make_adapter(**kwargs) -> MiniMaxNativeAdapter:
    """Create a MiniMaxNativeAdapter with sensible defaults for testing."""
    defaults = {
        "base_url": MINIMAX_DEFAULT_BASE_URL,
        "api_key": "test-key",
        "group_id": "test-group",
        "anthropic_base_url": MINIMAX_ANTHROPIC_DEFAULT_BASE_URL,
    }
    defaults.update(kwargs)
    return MiniMaxNativeAdapter(**defaults)


# ============================================================================
# 1. Interface routing logic
# ============================================================================


class TestInterfaceRouting:
    """Tests for _route_interface method."""

    def test_pure_text_routes_to_openai(self):
        """Pure text messages should route to OpenAI interface."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            max_tokens=1024,
        )
        assert adapter._route_interface(compiled) == "openai"

    def test_image_url_routes_to_anthropic(self):
        """Messages with image_url should route to Anthropic interface."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/img.png"},
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )
        assert adapter._route_interface(compiled) == "anthropic"

    def test_video_url_routes_to_anthropic(self):
        """Messages with video_url should route to Anthropic interface."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this video"},
                        {
                            "type": "video_url",
                            "video_url": {"url": "https://example.com/vid.mp4"},
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )
        assert adapter._route_interface(compiled) == "anthropic"

    def test_thinking_mode_routes_to_anthropic(self):
        """Thinking mode should route to Anthropic interface."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Think about this"}],
            max_tokens=1024,
            extra={"thinking": {"type": "enabled"}},
        )
        assert adapter._route_interface(compiled) == "anthropic"

    def test_thinking_mode_string_routes_to_anthropic(self):
        """thinking_mode string should route to Anthropic interface."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Think deeply"}],
            max_tokens=1024,
            extra={"thinking_mode": "deep"},
        )
        assert adapter._route_interface(compiled) == "anthropic"

    def test_reasoning_split_routes_to_anthropic(self):
        """reasoning_split should route to Anthropic interface."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Solve this problem"}],
            max_tokens=1024,
            extra={"reasoning_split": True},
        )
        assert adapter._route_interface(compiled) == "anthropic"

    def test_needs_token_estimation_routes_to_anthropic(self):
        """needs_token_estimation should route to Anthropic interface."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Count my tokens"}],
            max_tokens=1024,
            extra={"needs_token_estimation": True},
        )
        assert adapter._route_interface(compiled) == "anthropic"

    def test_no_anthropic_url_always_routes_to_openai(self):
        """Without anthropic_base_url, always route to OpenAI."""
        adapter = _make_adapter(anthropic_base_url="")
        # Even with thinking enabled
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Think about this"}],
            max_tokens=1024,
            extra={"thinking": {"type": "enabled"}},
        )
        assert adapter._route_interface(compiled) == "openai"

    def test_force_interface_openai(self):
        """force_interface='openai' overrides routing logic."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                    ],
                }
            ],
            max_tokens=1024,
            extra={"force_interface": "openai"},
        )
        assert adapter._route_interface(compiled) == "openai"

    def test_force_interface_anthropic(self):
        """force_interface='anthropic' overrides routing logic."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Just text"}],
            max_tokens=1024,
            extra={"force_interface": "anthropic"},
        )
        assert adapter._route_interface(compiled) == "anthropic"


# ============================================================================
# 2. Anthropic-style send (message format conversion)
# ============================================================================


class TestAnthropicMessageConversion:
    """Tests for _convert_messages_to_anthropic and _convert_content_to_anthropic."""

    def test_mode_a_system_extraction(self):
        """System messages should be extracted to top-level system parameter."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            max_tokens=1024,
        )
        system, messages = adapter._convert_messages_to_anthropic(compiled)
        assert system == "You are helpful."
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_mode_b_system_user(self):
        """Mode B (system_prompt + user_message) should be converted correctly."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            system_prompt="You are a coding assistant.",
            user_message="Write a function",
            max_tokens=1024,
        )
        system, messages = adapter._convert_messages_to_anthropic(compiled)
        assert system == "You are a coding assistant."
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Write a function"

    def test_image_url_conversion(self):
        """image_url blocks should be converted to Anthropic image format."""
        adapter = _make_adapter()
        content = [
            {"type": "text", "text": "What's this?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/img.png"},
            },
        ]
        result = adapter._convert_content_to_anthropic(content)
        assert isinstance(result, list)
        assert result[0] == {"type": "text", "text": "What's this?"}
        assert result[1]["type"] == "image"
        assert result[1]["source"]["type"] == "url"
        assert result[1]["source"]["url"] == "https://example.com/img.png"

    def test_image_base64_conversion(self):
        """Base64 data URI should be converted to Anthropic base64 source."""
        adapter = _make_adapter()
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
                },
            },
        ]
        result = adapter._convert_content_to_anthropic(content)
        assert isinstance(result, list)
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["media_type"] == "image/png"
        assert result[0]["source"]["data"] == "iVBORw0KGgoAAAANSUhEUg=="

    def test_video_url_conversion(self):
        """video_url blocks should be converted to Anthropic video format."""
        adapter = _make_adapter()
        content = [
            {
                "type": "video_url",
                "video_url": {"url": "https://example.com/vid.mp4"},
            },
        ]
        result = adapter._convert_content_to_anthropic(content)
        assert isinstance(result, list)
        assert result[0]["type"] == "video"
        assert result[0]["source"]["type"] == "url"
        assert result[0]["source"]["url"] == "https://example.com/vid.mp4"

    def test_string_content_passthrough(self):
        """String content should pass through unchanged."""
        adapter = _make_adapter()
        result = adapter._convert_content_to_anthropic("Hello, world!")
        assert result == "Hello, world!"

    def test_empty_messages_gets_default_user(self):
        """If all messages are system, a default user message is added."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[
                {"role": "system", "content": "Only system message"},
            ],
            max_tokens=1024,
        )
        system, messages = adapter._convert_messages_to_anthropic(compiled)
        assert system == "Only system message"
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


class TestBuildAnthropicPayload:
    """Tests for _build_anthropic_payload method."""

    def test_basic_payload_structure(self):
        """Basic payload should have model, max_tokens, and messages."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello"},
            ],
            max_tokens=2048,
        )
        payload = adapter._build_anthropic_payload(compiled, "minimax-m3")
        assert payload["model"] == "minimax-m3"
        assert payload["max_tokens"] == 2048
        assert "messages" in payload
        assert payload["system"] == "Be helpful"

    def test_thinking_parameter_enabled(self):
        """thinking parameter should be included when enabled."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Think"}],
            max_tokens=1024,
            extra={"thinking": {"type": "enabled"}},
        )
        payload = adapter._build_anthropic_payload(compiled, "minimax-m3")
        assert payload["thinking"] == {"type": "enabled"}

    def test_thinking_parameter_adaptive(self):
        """thinking parameter should support 'adaptive' type."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Think"}],
            max_tokens=1024,
            extra={"thinking": {"type": "adaptive"}},
        )
        payload = adapter._build_anthropic_payload(compiled, "minimax-m3")
        assert payload["thinking"] == {"type": "adaptive"}

    def test_thinking_mode_deep_mapping(self):
        """thinking_mode='deep' should map to thinking.type='enabled'."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Think"}],
            max_tokens=1024,
            extra={"thinking_mode": "deep"},
        )
        payload = adapter._build_anthropic_payload(compiled, "minimax-m3")
        assert payload["thinking"] == {"type": "enabled"}

    def test_thinking_budget_tokens(self):
        """thinking_budget should add budget_tokens to thinking parameter."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Think"}],
            max_tokens=1024,
            extra={
                "thinking": {"type": "enabled"},
                "thinking_budget": 10000,
            },
        )
        payload = adapter._build_anthropic_payload(compiled, "minimax-m3")
        assert payload["thinking"]["budget_tokens"] == 10000

    def test_reasoning_split_parameter(self):
        """reasoning_split should be included when True."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Solve"}],
            max_tokens=1024,
            extra={"reasoning_split": True},
        )
        payload = adapter._build_anthropic_payload(compiled, "minimax-m3")
        assert payload["reasoning_split"] is True

    def test_reasoning_split_not_included_by_default(self):
        """reasoning_split should not be in payload when not set."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1024,
        )
        payload = adapter._build_anthropic_payload(compiled, "minimax-m3")
        assert "reasoning_split" not in payload

    def test_tools_converted_to_anthropic_format(self):
        """Tools should be converted from OpenAI to Anthropic format."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Use a tool"}],
            max_tokens=1024,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather info",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        )
        payload = adapter._build_anthropic_payload(compiled, "minimax-m3")
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["name"] == "get_weather"
        assert "input_schema" in payload["tools"][0]
        assert "parameters" not in payload["tools"][0]


# ============================================================================
# 3. thinking parameter passing
# ============================================================================


class TestThinkingParameter:
    """Tests for thinking parameter support."""

    def test_parse_anthropic_response_with_thinking(self):
        """Anthropic response with thinking blocks should extract thinking_content."""
        adapter = _make_adapter()
        data = {
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Let me reason about this step by step...",
                },
                {
                    "type": "text",
                    "text": "The answer is 42.",
                },
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
            },
            "stop_reason": "end_turn",
        }
        response = adapter._parse_anthropic_response(data, "minimax-m3")
        assert response.raw_text == "The answer is 42."
        assert response.thinking_content == "Let me reason about this step by step..."

    def test_parse_anthropic_response_multiple_thinking_blocks(self):
        """Multiple thinking blocks should be joined with newlines."""
        adapter = _make_adapter()
        data = {
            "content": [
                {"type": "thinking", "thinking": "First thought"},
                {"type": "thinking", "thinking": "Second thought"},
                {"type": "text", "text": "Final answer"},
            ],
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "stop_reason": "end_turn",
        }
        response = adapter._parse_anthropic_response(data, "minimax-m3")
        assert response.thinking_content == "First thought\nSecond thought"

    def test_parse_anthropic_response_no_thinking(self):
        """Response without thinking blocks should have thinking_content=None."""
        adapter = _make_adapter()
        data = {
            "content": [{"type": "text", "text": "Simple response"}],
            "usage": {"input_tokens": 50, "output_tokens": 20},
            "stop_reason": "end_turn",
        }
        response = adapter._parse_anthropic_response(data, "minimax-m3")
        assert response.thinking_content is None


# ============================================================================
# 4. reasoning_split response parsing
# ============================================================================


class TestReasoningSplitParsing:
    """Tests for reasoning_split response parsing."""

    def test_parse_anthropic_response_with_reasoning_details(self):
        """reasoning_details from response should be in extra."""
        adapter = _make_adapter()
        data = {
            "content": [{"type": "text", "text": "The result is 42."}],
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "stop_reason": "end_turn",
            "reasoning_details": {
                "steps": [
                    {"step": 1, "description": "Parse the problem"},
                    {"step": 2, "description": "Apply formula"},
                ],
            },
        }
        response = adapter._parse_anthropic_response(data, "minimax-m3")
        assert "reasoning_details" in response.extra
        assert response.extra["reasoning_details"]["steps"][0]["step"] == 1

    def test_parse_anthropic_response_without_reasoning_details(self):
        """Response without reasoning_details should not have it in extra."""
        adapter = _make_adapter()
        data = {
            "content": [{"type": "text", "text": "Simple response"}],
            "usage": {"input_tokens": 50, "output_tokens": 20},
            "stop_reason": "end_turn",
        }
        response = adapter._parse_anthropic_response(data, "minimax-m3")
        assert "reasoning_details" not in response.extra


# ============================================================================
# 5. count_tokens method
# ============================================================================


class TestCountTokens:
    """Tests for count_tokens method."""

    @pytest.mark.asyncio
    async def test_count_tokens_requires_anthropic_url(self):
        """count_tokens should raise ValueError without Anthropic URL."""
        adapter = _make_adapter(anthropic_base_url="")
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1024,
        )
        with pytest.raises(ValueError, match="count_tokens requires the Anthropic"):
            await adapter.count_tokens(compiled, "minimax-m3")

    @pytest.mark.asyncio
    async def test_count_tokens_successful(self):
        """count_tokens should return the token count from API response."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Hello world"}],
            max_tokens=1024,
        )

        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"input_tokens": 42}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(adapter, "_get_anthropic_client", return_value=mock_client):
            result = await adapter.count_tokens(compiled, "minimax-m3")

        assert result == 42

    @pytest.mark.asyncio
    async def test_count_tokens_request_format(self):
        """count_tokens should POST to the correct endpoint."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Test message"}],
            max_tokens=1024,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"input_tokens": 10}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(adapter, "_get_anthropic_client", return_value=mock_client):
            await adapter.count_tokens(compiled, "minimax-m3")

        # Verify the URL
        call_args = mock_client.post.call_args
        assert call_args[0][0] == f"{MINIMAX_ANTHROPIC_DEFAULT_BASE_URL}/messages/count_tokens"

    @pytest.mark.asyncio
    async def test_count_tokens_no_stream_in_payload(self):
        """count_tokens payload should not include 'stream' parameter."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=1024,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"input_tokens": 5}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(adapter, "_get_anthropic_client", return_value=mock_client):
            await adapter.count_tokens(compiled, "minimax-m3")

        # Verify stream is not in payload
        call_args = mock_client.post.call_args
        payload = call_args[1].get("json", {})
        assert "stream" not in payload


# ============================================================================
# 6. Fallback to OpenAI when Anthropic unavailable
# ============================================================================


class TestFallbackToOpenAI:
    """Tests for graceful fallback to OpenAI interface."""

    @pytest.mark.asyncio
    async def test_send_falls_back_on_anthropic_error(self):
        """send() should fall back to OpenAI when Anthropic interface fails."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1024,
            extra={"force_interface": "anthropic"},
        )

        # Mock _send_via_anthropic to raise an error
        with patch.object(
            adapter, "_send_via_anthropic",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            # Mock parent's send to return a valid response
            with patch.object(
                type(adapter).__bases__[0], "send",
                new_callable=AsyncMock,
                return_value=TAPResponse(
                    raw_text="Fallback response",
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                ),
            ):
                response = await adapter.send(compiled, "minimax-m3")
                assert response.raw_text == "Fallback response"

    @pytest.mark.asyncio
    async def test_stream_falls_back_on_anthropic_error(self):
        """stream() should fall back to OpenAI when Anthropic streaming fails."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1024,
            extra={"force_interface": "anthropic"},
        )

        # Create an async generator for the fallback
        async def _openai_stream_fallback(self, compiled, model):
            yield "Fallback"
            yield " stream"

        # Mock _stream_via_anthropic to raise
        with patch.object(
            adapter, "_stream_via_anthropic",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            # Mock parent's stream
            with patch.object(
                type(adapter).__bases__[0], "stream",
                _openai_stream_fallback,
            ):
                chunks = []
                async for chunk in adapter.stream(compiled, "minimax-m3"):
                    chunks.append(chunk)
                assert chunks == ["Fallback", " stream"]

    def test_no_anthropic_url_routes_to_openai(self):
        """Without anthropic_base_url, routing always selects openai."""
        adapter = _make_adapter(anthropic_base_url="")
        # Even with multimodal content
        compiled = CompiledPrompt(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                    ],
                }
            ],
            max_tokens=1024,
            extra={"thinking": {"type": "enabled"}},
        )
        assert adapter._route_interface(compiled) == "openai"


# ============================================================================
# 7. Streaming via Anthropic interface
# ============================================================================


class TestAnthropicStreaming:
    """Tests for _stream_via_anthropic method."""

    @pytest.mark.asyncio
    async def test_stream_yields_text_deltas(self):
        """_stream_via_anthropic should yield text from content_block_delta events."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1024,
        )

        # Create mock SSE lines
        sse_lines = [
            'event: message_start',
            'data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}',
            '',
            'event: content_block_start',
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            '',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
            '',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}',
            '',
            'event: message_delta',
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}',
            '',
        ]

        # Create a mock response with aiter_lines as an async generator
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = MagicMock(return_value=AsyncIteratorFromList(sse_lines))

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        with patch.object(adapter, "_get_anthropic_client", return_value=mock_client):
            chunks = []
            async for chunk in adapter._stream_via_anthropic(compiled, "minimax-m3"):
                chunks.append(chunk)

        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_skips_thinking_deltas(self):
        """_stream_via_anthropic should not yield thinking_delta content."""
        adapter = _make_adapter()
        compiled = CompiledPrompt(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1024,
        )

        sse_lines = [
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"I should think..."}}',
            '',
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}',
            '',
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = MagicMock(return_value=AsyncIteratorFromList(sse_lines))

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        with patch.object(adapter, "_get_anthropic_client", return_value=mock_client):
            chunks = []
            async for chunk in adapter._stream_via_anthropic(compiled, "minimax-m3"):
                chunks.append(chunk)

        # thinking_delta should not be yielded, only text_delta
        assert chunks == ["Answer"]


# ============================================================================
# Anthropic response parsing (non-streaming)
# ============================================================================


class TestParseAnthropicResponse:
    """Tests for _parse_anthropic_response method."""

    def test_basic_text_response(self):
        """Basic text response should be parsed correctly."""
        adapter = _make_adapter()
        data = {
            "content": [
                {"type": "text", "text": "Hello, how can I help?"},
            ],
            "usage": {
                "input_tokens": 50,
                "output_tokens": 20,
            },
            "stop_reason": "end_turn",
        }
        response = adapter._parse_anthropic_response(data, "minimax-m3")
        assert response.raw_text == "Hello, how can I help?"
        assert response.usage["prompt_tokens"] == 50
        assert response.usage["completion_tokens"] == 20
        assert response.finish_reason == "stop"

    def test_cache_tokens_parsed(self):
        """Cache token fields should be parsed from Anthropic usage."""
        adapter = _make_adapter()
        data = {
            "content": [{"type": "text", "text": "Response"}],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 30,
                "cache_creation_input_tokens": 50,
                "cache_read_input_tokens": 80,
            },
            "stop_reason": "end_turn",
        }
        response = adapter._parse_anthropic_response(data, "minimax-m3")
        assert response.usage["cache_creation_input_tokens"] == 50
        assert response.usage["cache_read_input_tokens"] == 80
        assert response.usage["prompt_cache_hit_tokens"] == 80
        assert response.cache_hit_tokens == 80

    def test_tool_use_blocks_parsed(self):
        """tool_use blocks should be converted to OpenAI-format tool_calls."""
        adapter = _make_adapter()
        data = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "get_weather",
                    "input": {"city": "Beijing"},
                },
            ],
            "usage": {"input_tokens": 30, "output_tokens": 10},
            "stop_reason": "tool_use",
        }
        response = adapter._parse_anthropic_response(data, "minimax-m3")
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "get_weather"
        assert response.finish_reason == "tool_calls"

    def test_stop_reason_mapping(self):
        """Anthropic stop_reasons should be mapped to standard finish_reasons."""
        adapter = _make_adapter()
        for anthropic_reason, expected in [
            ("end_turn", "stop"),
            ("max_tokens", "length"),
            ("stop_sequence", "stop"),
            ("tool_use", "tool_calls"),
        ]:
            data = {
                "content": [{"type": "text", "text": "x"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "stop_reason": anthropic_reason,
            }
            response = adapter._parse_anthropic_response(data, "minimax-m3")
            assert response.finish_reason == expected

    def test_thinking_and_text_combined(self):
        """Response with both thinking and text should parse both."""
        adapter = _make_adapter()
        data = {
            "content": [
                {"type": "thinking", "thinking": "Step 1: Analyze..."},
                {"type": "text", "text": "The answer is 42."},
            ],
            "usage": {"input_tokens": 100, "output_tokens": 80},
            "stop_reason": "end_turn",
        }
        response = adapter._parse_anthropic_response(data, "minimax-m3")
        assert response.raw_text == "The answer is 42."
        assert response.thinking_content == "Step 1: Analyze..."


# ============================================================================
# Anthropic headers
# ============================================================================


class TestAnthropicHeaders:
    """Tests for _build_anthropic_headers method."""

    def test_anthropic_headers_include_version(self):
        """Anthropic headers should include anthropic-version."""
        adapter = _make_adapter()
        headers = adapter._build_anthropic_headers()
        assert headers["anthropic-version"] == "2023-06-01"

    def test_anthropic_headers_include_api_key(self):
        """Anthropic headers should include x-api-key."""
        adapter = _make_adapter()
        headers = adapter._build_anthropic_headers()
        assert headers["x-api-key"] == "test-key"

    def test_anthropic_headers_include_authorization(self):
        """Anthropic headers should also include Authorization for compatibility."""
        adapter = _make_adapter()
        headers = adapter._build_anthropic_headers()
        assert headers["Authorization"] == "Bearer test-key"

    def test_anthropic_headers_include_group_id(self):
        """Anthropic headers should include X-Group-Id when configured."""
        adapter = _make_adapter()
        headers = adapter._build_anthropic_headers()
        assert headers["X-Group-Id"] == "test-group"

    def test_anthropic_headers_no_api_key(self):
        """Anthropic headers without API key should not include x-api-key."""
        adapter = _make_adapter(api_key="")
        headers = adapter._build_anthropic_headers()
        assert "x-api-key" not in headers
        assert "Authorization" not in headers


# ============================================================================
# Capabilities
# ============================================================================


class TestCapabilities:
    """Tests for capabilities property."""

    def test_capabilities_with_anthropic(self):
        """Capabilities should reflect Anthropic interface features."""
        adapter = _make_adapter()
        caps = adapter.capabilities
        assert caps["thinking"] is True
        assert caps["reasoning_split"] is True
        assert caps["count_tokens"] is True
        assert caps["anthropic_interface"] is True

    def test_capabilities_without_anthropic(self):
        """Capabilities should reflect disabled Anthropic features."""
        adapter = _make_adapter(anthropic_base_url="")
        caps = adapter.capabilities
        assert caps["thinking"] is True  # Feature exists, just not accessible
        assert caps["count_tokens"] is False
        assert caps["anthropic_interface"] is False


# ============================================================================
# Backward compatibility
# ============================================================================


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with existing MiniMaxNativeAdapter."""

    def test_default_init_without_anthropic(self):
        """Adapter should work without anthropic_base_url parameter."""
        adapter = MiniMaxNativeAdapter()
        assert adapter.base_url == MINIMAX_DEFAULT_BASE_URL
        assert adapter.anthropic_base_url == MINIMAX_ANTHROPIC_DEFAULT_BASE_URL

    def test_default_init_with_empty_anthropic(self):
        """Adapter should work with empty anthropic_base_url."""
        adapter = MiniMaxNativeAdapter(anthropic_base_url="")
        assert adapter.anthropic_base_url == ""

    def test_registry_still_works(self):
        """Adapter should still be registered as 'minimax_native'."""
        adapter = TAPAdapterRegistry.create("minimax_native")
        assert isinstance(adapter, MiniMaxNativeAdapter)

    def test_billing_tracker_still_works(self):
        """Billing tracker should still work with new code."""
        adapter = _make_adapter()
        response = TAPResponse(
            raw_text="test",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        adapter._update_billing_tracker(response)
        assert adapter._billing_tracker["total_input_tokens"] == 100
        assert adapter._billing_tracker["total_output_tokens"] == 50

    def test_rate_limit_info_still_works(self):
        """Rate limit info should still work with new code."""
        adapter = _make_adapter()
        info = adapter.rate_limit_info
        assert isinstance(info, MiniMaxRateLimitInfo)

    def test_model_name_mapping_still_works(self):
        """Model name mapping should still work."""
        adapter = _make_adapter()
        assert adapter._resolve_model_name("minimax") == "minimax-m3"
        assert adapter._resolve_model_name("m3") == "minimax-m3"


# ============================================================================
# Helper class for async iteration in tests
# ============================================================================


class AsyncIteratorFromList:
    """Helper to create an async iterator from a list for mocking."""

    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration
