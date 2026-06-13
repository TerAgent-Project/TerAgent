"""tests.test_live_api — End-to-end tests with real API endpoints

These tests call real LLM APIs and are skipped by default.
Run with: pytest tests/test_live_api.py -m live

Requirements:
    - Set environment variables: DEEPSEEK_API_KEY, GLM_API_KEY, MINIMAX_API_KEY
    - Network access to API endpoints

Markers:
    @pytest.mark.live — Tests that call real API endpoints (skipped by default)
"""

from __future__ import annotations

import os

import pytest

import teragent
from teragent.config import DriverConfig

# Skip entire module if not running live tests
pytestmark = pytest.mark.live

# ============================================================================
# Helpers
# ============================================================================


def _get_api_key(env_var: str) -> str:
    """Get API key from environment, skip test if not available."""
    key = os.environ.get(env_var, "")
    if not key:
        pytest.skip(f"Environment variable {env_var} not set — skipping live test")
    return key


def _create_deepseek_v4_provider():
    """Create a DeepSeek V4 Flash provider for live testing."""
    api_key = _get_api_key("DEEPSEEK_API_KEY")
    return teragent.create_provider(
        compiler="deepseek_v4",
        adapter="openai_compatible",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key=api_key,
    )


def _create_glm5_provider():
    """Create a GLM-5 provider for live testing."""
    api_key = _get_api_key("GLM_API_KEY")
    return teragent.create_provider(
        compiler="glm_5",
        adapter="openai_compatible",
        model="glm-5",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key=api_key,
    )


def _create_minimax_m3_provider():
    """Create a MiniMax M3 provider for live testing."""
    api_key = _get_api_key("MINIMAX_API_KEY")
    return teragent.create_provider(
        compiler="minimax_m3",
        adapter="openai_compatible",
        model="minimax-m3",
        base_url="https://api.minimaxi.com/v1",
        api_key=api_key,
    )


# ============================================================================
# DeepSeek V4 — Live Tests
# ============================================================================


class TestDeepSeekV4Live:
    """Live end-to-end tests for DeepSeek V4 Flash."""

    @pytest.mark.asyncio
    async def test_simple_chat(self):
        """Test basic chat completion with DeepSeek V4."""
        provider = _create_deepseek_v4_provider()
        result = await provider.chat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2? Reply with just the number."},
            ]
        )
        assert result["content"]
        assert "4" in result["content"]

    @pytest.mark.asyncio
    async def test_tap_execute(self):
        """Test TAP execution pipeline with DeepSeek V4."""
        provider = _create_deepseek_v4_provider()
        response = await provider.execute_tap(
            teragent.TAPRequest(
                meta={"task_id": "live-1", "intent": "execute"},
                instruction="Write a Python function that adds two numbers.",
                constraints=["Must have type hints", "Must include a docstring"],
                output_format_hint="<file path='add.py'>complete code</file>",
            )
        )
        assert response.raw_text
        assert "def " in response.raw_text
        assert response.cost_tokens > 0


# ============================================================================
# GLM-5 — Live Tests
# ============================================================================


class TestGLM5Live:
    """Live end-to-end tests for GLM-5."""

    @pytest.mark.asyncio
    async def test_simple_chat(self):
        """Test basic chat completion with GLM-5."""
        provider = _create_glm5_provider()
        result = await provider.chat(
            messages=[
                {"role": "system", "content": "你是一个有用的助手。"},
                {"role": "user", "content": "2+2等于几？只回答数字。"},
            ]
        )
        assert result["content"]
        assert "4" in result["content"]

    @pytest.mark.asyncio
    async def test_tap_design(self):
        """Test TAP design pipeline with GLM-5."""
        provider = _create_glm5_provider()
        response = await provider.execute_tap(
            teragent.TAPRequest(
                meta={"task_id": "live-2", "intent": "design"},
                instruction="设计一个用户登录模块的技术方案。",
                constraints=["Python 3.10+", "使用 JWT 认证"],
                output_format_hint="中文撰写，技术术语保留英文",
            )
        )
        assert response.raw_text
        assert len(response.raw_text) > 100  # Should produce a substantial design

    @pytest.mark.asyncio
    async def test_recency_effect_compilation(self):
        """Test that GLM-5 compiler applies recency effect (instruction last)."""
        provider = _create_glm5_provider()
        request = teragent.TAPRequest(
            meta={"task_id": "live-3", "intent": "execute"},
            instruction="实现冒泡排序",
            constraints=["Python 3.10+"],
            context={"design": "经典冒泡排序算法"},
        )
        compiled = provider.compiler.compile(request)
        # Recency effect: last user message should contain the instruction
        last_user_msg = [m for m in compiled.messages if m["role"] == "user"][-1]
        assert "冒泡排序" in last_user_msg["content"]


# ============================================================================
# MiniMax M3 — Live Tests
# ============================================================================


class TestMiniMaxM3Live:
    """Live end-to-end tests for MiniMax M3."""

    @pytest.mark.asyncio
    async def test_simple_chat(self):
        """Test basic chat completion with MiniMax M3."""
        provider = _create_minimax_m3_provider()
        result = await provider.chat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2? Reply with just the number."},
            ]
        )
        assert result["content"]
        assert "4" in result["content"]

    @pytest.mark.asyncio
    async def test_tap_execute(self):
        """Test TAP execution pipeline with MiniMax M3."""
        provider = _create_minimax_m3_provider()
        response = await provider.execute_tap(
            teragent.TAPRequest(
                meta={"task_id": "live-4", "intent": "execute"},
                instruction="Write a Python function that checks if a number is prime.",
                constraints=["Must have type hints", "Must include error handling"],
                output_format_hint="<file path='prime.py'>complete code</file>",
            )
        )
        assert response.raw_text
        assert "def " in response.raw_text


# ============================================================================
# Multi-Model Integration — Live Tests
# ============================================================================


class TestMultiModelLive:
    """Live integration tests for multi-model workflows."""

    @pytest.mark.asyncio
    async def test_compiler_adapter_orthogonality(self):
        """Verify that different compilers produce different prompts for the same request."""
        request = teragent.TAPRequest(
            meta={"task_id": "live-5", "intent": "design"},
            instruction="Design a REST API",
            constraints=["OpenAPI 3.0"],
        )

        # Compile with different compilers
        from teragent.core.compilers import GLM5Compiler, DeepSeekV4Compiler, MiniMaxM3Compiler

        glm5 = GLM5Compiler().compile(request)
        v4 = DeepSeekV4Compiler().compile(request)
        m3 = MiniMaxM3Compiler().compile(request)

        # All should produce messages
        assert len(glm5.messages) > 0
        assert len(v4.messages) > 0
        assert len(m3.messages) > 0

        # But the prompts should differ (model-specific optimization)
        glm5_system = glm5.messages[0]["content"]
        v4_system = v4.messages[0]["content"]
        m3_system = m3.messages[0]["content"]

        # At least one should differ from the others
        assert not (glm5_system == v4_system == m3_system), \
            "All compilers produced identical prompts — orthogonality broken"

    @pytest.mark.asyncio
    async def test_cost_tracking_across_models(self):
        """Verify cost tracking works when using multiple providers."""
        # We test the tracking mechanism without requiring all API keys
        tracker = teragent.CostBudgetTracker(teragent.CostBudgetConfig(
            max_session_tokens=1_000_000,
        ))

        # Record some usage
        from teragent.core.tap import TAPCostRecord

        tracker.record(TAPCostRecord(
            model="deepseek-v4-flash",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ))
        tracker.record(TAPCostRecord(
            model="glm-5",
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
        ))

        summary = tracker.get_summary()
        assert summary["total_tokens"] == 450
        assert "deepseek-v4-flash" in summary["by_model"]
        assert "glm-5" in summary["by_model"]
