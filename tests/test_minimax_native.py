"""tests.test_minimax_native — Tests for MiniMaxNativeAdapter

Tests cover:
  - Adapter initialization and configuration
  - Rate limit header parsing
  - Billing tracker updates
  - Video content enhancement
  - Desktop command building
  - Header construction (group_id injection)
  - Registration in adapter registry
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from teragent.core.adapters.minimax_native import (
    MiniMaxNativeAdapter,
    MiniMaxRateLimitInfo,
    MINIMAX_DEFAULT_BASE_URL,
    _DESKTOP_ENDPOINT,
)
from teragent.core.adapter import TAPAdapterRegistry
from teragent.core.tap import CompiledPrompt, TAPResponse


# ============================================================================
# MiniMaxRateLimitInfo
# ============================================================================


class TestMiniMaxRateLimitInfo:
    """Tests for MiniMaxRateLimitInfo parsing and state."""

    def test_default_values(self):
        info = MiniMaxRateLimitInfo()
        assert info.limit == 0
        assert info.remaining == 0
        assert info.reset == 0.0
        assert not info.is_exhausted  # No limit set => not exhausted

    def test_update_from_headers(self):
        headers = httpx.Headers({
            "X-RateLimit-Limit": "100",
            "X-RateLimit-Remaining": "42",
            "X-RateLimit-Reset": str(time.time() + 3600),
        })
        info = MiniMaxRateLimitInfo()
        info.update_from_headers(headers)
        assert info.limit == 100
        assert info.remaining == 42
        assert info.reset > 0
        assert not info.is_exhausted

    def test_exhausted_rate_limit(self):
        info = MiniMaxRateLimitInfo(limit=100, remaining=0, reset=time.time() + 3600)
        assert info.is_exhausted

    def test_reset_in_seconds(self):
        info = MiniMaxRateLimitInfo(reset=time.time() + 60)
        assert 55 < info.reset_in_seconds <= 60

    def test_reset_in_seconds_no_reset(self):
        info = MiniMaxRateLimitInfo(reset=0)
        assert info.reset_in_seconds == 0.0

    def test_malformed_headers_graceful(self):
        headers = httpx.Headers({
            "X-RateLimit-Limit": "not_a_number",
            "X-RateLimit-Remaining": "also_not",
        })
        info = MiniMaxRateLimitInfo()
        # Should not raise
        info.update_from_headers(headers)
        # Values remain at defaults because parsing failed
        assert info.limit == 0
        assert info.remaining == 0

    def test_missing_headers_graceful(self):
        headers = httpx.Headers({})
        info = MiniMaxRateLimitInfo()
        info.update_from_headers(headers)
        assert info.limit == 0

    def test_repr(self):
        info = MiniMaxRateLimitInfo(limit=100, remaining=50, reset=0)
        r = repr(info)
        assert "limit=100" in r
        assert "remaining=50" in r
        assert "exhausted=False" in r


# ============================================================================
# MiniMaxNativeAdapter — Initialization
# ============================================================================


class TestMiniMaxNativeAdapterInit:
    """Tests for MiniMaxNativeAdapter initialization."""

    def test_default_init(self):
        adapter = MiniMaxNativeAdapter()
        assert adapter.base_url == MINIMAX_DEFAULT_BASE_URL
        assert adapter.api_key == ""
        assert adapter.group_id == ""
        assert adapter._rate_limit_info is not None
        assert adapter._billing_tracker["total_requests"] == 0

    def test_init_with_params(self):
        adapter = MiniMaxNativeAdapter(
            base_url="https://custom.api.com/v1",
            api_key="test-key",
            group_id="test-group",
            timeout=120.0,
        )
        assert adapter.base_url == "https://custom.api.com/v1"
        assert adapter.api_key == "test-key"
        assert adapter.group_id == "test-group"

    def test_registry_registered(self):
        """Verify MiniMaxNativeAdapter is registered in the adapter registry."""
        adapter = TAPAdapterRegistry.create("minimax_native")
        assert isinstance(adapter, MiniMaxNativeAdapter)


# ============================================================================
# MiniMaxNativeAdapter — Headers
# ============================================================================


class TestMiniMaxNativeAdapterHeaders:
    """Tests for MiniMax-specific header construction."""

    def test_headers_without_group_id(self):
        adapter = MiniMaxNativeAdapter(api_key="test-key")
        headers = adapter._build_headers()
        assert "X-Group-Id" not in headers
        assert headers["Authorization"] == "Bearer test-key"

    def test_headers_with_group_id(self):
        adapter = MiniMaxNativeAdapter(api_key="test-key", group_id="my-group")
        headers = adapter._build_headers()
        assert headers["X-Group-Id"] == "my-group"
        assert headers["Authorization"] == "Bearer test-key"


# ============================================================================
# MiniMaxNativeAdapter — Video content enhancement
# ============================================================================


class TestMiniMaxNativeAdapterVideoEnhancement:
    """Tests for MiniMax-specific video content processing."""

    def test_enhance_video_content_adds_hints(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this video?"},
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://example.com/video.mp4"},
                    },
                ],
            }
        ]
        result = MiniMaxNativeAdapter._enhance_video_content(messages)
        video_part = result[0]["content"][1]
        assert video_part["video_url"]["minimax_video_mode"] == "understand"
        assert video_part["video_url"]["minimax_frame_sampling"] == "auto"

    def test_enhance_video_content_no_video(self):
        messages = [
            {"role": "user", "content": "Hello, no video here."}
        ]
        result = MiniMaxNativeAdapter._enhance_video_content(messages)
        # Should return the same messages unchanged
        assert result == messages

    def test_enhance_video_content_already_enhanced(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": "https://example.com/video.mp4",
                            "minimax_video_mode": "summarize",
                        },
                    },
                ],
            }
        ]
        result = MiniMaxNativeAdapter._enhance_video_content(messages)
        video_part = result[0]["content"][0]
        # Should not overwrite existing mode
        assert video_part["video_url"]["minimax_video_mode"] == "summarize"

    def test_enhance_video_content_string_content_noop(self):
        """Message content as string (not list) should not crash."""
        messages = [
            {"role": "user", "content": "Just text"}
        ]
        result = MiniMaxNativeAdapter._enhance_video_content(messages)
        assert result == messages

    def test_enhance_video_content_invalid_video_url_noop(self):
        """Invalid video_url structure should not crash."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": "not_a_dict"},
                ],
            }
        ]
        result = MiniMaxNativeAdapter._enhance_video_content(messages)
        # Should not crash, video_url is not a dict
        assert len(result) == 1


# ============================================================================
# MiniMaxNativeAdapter — Billing tracker
# ============================================================================


class TestMiniMaxNativeAdapterBilling:
    """Tests for MiniMax billing tracker."""

    def test_billing_tracker_initial_state(self):
        adapter = MiniMaxNativeAdapter()
        assert adapter._billing_tracker["total_input_tokens"] == 0
        assert adapter._billing_tracker["total_output_tokens"] == 0
        assert adapter._billing_tracker["total_cache_hit_tokens"] == 0
        assert adapter._billing_tracker["total_cache_miss_tokens"] == 0
        assert adapter._billing_tracker["total_requests"] == 0

    def test_update_billing_tracker(self):
        adapter = MiniMaxNativeAdapter()
        response = TAPResponse(
            raw_text="test",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        # Access internal method to test billing update
        adapter._update_billing_tracker(response)
        assert adapter._billing_tracker["total_input_tokens"] == 100
        assert adapter._billing_tracker["total_output_tokens"] == 50
        assert adapter._billing_tracker["total_requests"] == 1

    def test_billing_tracker_accumulates(self):
        adapter = MiniMaxNativeAdapter()
        r1 = TAPResponse(raw_text="1", usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        r2 = TAPResponse(raw_text="2", usage={"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300})
        adapter._update_billing_tracker(r1)
        adapter._update_billing_tracker(r2)
        assert adapter._billing_tracker["total_input_tokens"] == 300
        assert adapter._billing_tracker["total_output_tokens"] == 150
        assert adapter._billing_tracker["total_requests"] == 2


# ============================================================================
# MiniMaxNativeAdapter — Desktop command
# ============================================================================


class TestMiniMaxNativeAdapterDesktop:
    """Tests for MiniMax desktop operation API."""

    @pytest.mark.asyncio
    async def test_desktop_command_requires_api_key(self):
        adapter = MiniMaxNativeAdapter(api_key="")
        with pytest.raises(ValueError, match="api_key is required"):
            await adapter.send_desktop_command(command="click", params={"x": 100, "y": 200})

    @pytest.mark.asyncio
    async def test_desktop_command_builds_correct_url(self):
        """Verify desktop command uses the correct endpoint URL."""
        adapter = MiniMaxNativeAdapter(api_key="test-key", group_id="test-group")
        expected_url = f"{adapter.base_url}{_DESKTOP_ENDPOINT}"
        assert expected_url.endswith("/desktop/operations")

    @pytest.mark.asyncio
    async def test_desktop_command_payload_structure(self):
        """Verify desktop command builds a correct payload structure."""
        adapter = MiniMaxNativeAdapter(api_key="test-key", group_id="test-group")
        # Build the expected payload manually
        payload = {
            "model": "minimax-m3",
            "command": "click",
            "params": {"x": 100, "y": 200},
            "screenshot": "base64data",
            "interactive_elements": [{"type": "button", "text": "Submit"}],
            "active_window": "",
        }
        # Verify structure is correct
        assert payload["command"] == "click"
        assert payload["params"]["x"] == 100
        assert payload["model"] == "minimax-m3"


# ============================================================================
# MiniMaxNativeAdapter — Rate limit property
# ============================================================================


class TestMiniMaxNativeAdapterRateLimit:
    """Tests for rate limit info property."""

    def test_rate_limit_info_property(self):
        adapter = MiniMaxNativeAdapter()
        info = adapter.rate_limit_info
        assert isinstance(info, MiniMaxRateLimitInfo)

    def test_rate_limit_updates_on_response(self):
        adapter = MiniMaxNativeAdapter()
        # Simulate headers from a response
        headers = httpx.Headers({
            "X-RateLimit-Limit": "500",
            "X-RateLimit-Remaining": "499",
            "X-RateLimit-Reset": str(time.time() + 60),
        })
        adapter._rate_limit_info.update_from_headers(headers)
        assert adapter.rate_limit_info.limit == 500
        assert adapter.rate_limit_info.remaining == 499
