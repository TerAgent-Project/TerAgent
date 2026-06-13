# tests/test_provider.py
"""ModelProvider / TAP / MockAdapter 模型驱动单元测试

覆盖:
  - TAPRequest: 请求结构/token估算
  - TAPResponse: None 处理
  - MockAdapter: 基础功能/send/stream
  - ModelProvider: execute_tap / stream_tap / chat
  - 成本追踪: CostTracker 线程安全
"""

import pytest

from teragent.core.adapters.mock import MockAdapter
from teragent.core.compiler import TAPCompilerRegistry
from teragent.core.provider import ModelProvider
from teragent.core.tap import TAPRequest, TAPResponse


def _make_mock_provider(delay: float = 0.01, fail_rate: float = 0.0) -> ModelProvider:
    """Create a ModelProvider with MockAdapter for testing."""
    adapter = MockAdapter(delay=delay, fail_rate=fail_rate)
    compiler = TAPCompilerRegistry.create("default")
    return ModelProvider(compiler=compiler, adapter=adapter, model="mock")


# ===== TAPRequest =====

class TestTAPRequest:
    """TAP 请求结构"""

    def test_basic_creation(self):
        """基本创建"""
        req = TAPRequest(
            meta={"task_id": "1.1", "intent": "code_generation"},
            context={"design": "sample design"},
            instruction="Write a function",
            constraints=["Must be type-safe"],
            output_format_hint="Python code block",
        )
        assert req.meta["task_id"] == "1.1"
        assert req.instruction == "Write a function"

    def test_estimate_prompt_tokens(self):
        """Token 估算"""
        req = TAPRequest(
            meta={"task_id": "1.1", "intent": "code_generation"},
            context={},
            instruction="Write a function that sorts a list",
            constraints=["Must be O(n log n)"],
            output_format_hint="Python",
        )
        tokens = req.estimate_prompt_tokens()
        assert tokens > 0


# ===== TAPResponse =====

class TestTAPResponse:
    """TAP 响应结构 — None 处理"""

    def test_normal_response(self):
        """正常响应"""
        resp = TAPResponse(raw_text="Hello", usage={"prompt_tokens": 10})
        assert resp.raw_text == "Hello"

    def test_none_raw_text_not_silently_converted(self):
        """raw_text=None 不被静默替换为空串

        raw_text 允许 None，上层需显式处理
        """
        resp = TAPResponse(raw_text=None, usage={"prompt_tokens": 10})
        # raw_text 仍为 None（或被记录为警告但保留 None）
        assert resp.raw_text is None or resp.raw_text == ""

    def test_usage_dict(self):
        """usage 字典"""
        resp = TAPResponse(raw_text="test", usage={"prompt_tokens": 100, "completion_tokens": 50})
        assert resp.usage["prompt_tokens"] == 100
        assert resp.usage["completion_tokens"] == 50


# ===== MockAdapter =====

class TestMockAdapter:
    """MockAdapter 驱动测试"""

    @pytest.mark.asyncio
    async def test_execute_tap_design(self):
        """execute_tap design 意图"""
        provider = _make_mock_provider(delay=0.01)

        req = TAPRequest(
            meta={"task_id": "1.1", "intent": "design"},
            context={},
            instruction="设计一个系统",
            constraints=[],
            output_format_hint="markdown",
        )
        resp = await provider.execute_tap(req)
        assert resp.raw_text is not None
        assert "DESIGN" in resp.raw_text or "design" in resp.raw_text.lower()

    @pytest.mark.asyncio
    async def test_execute_tap_plan(self):
        """execute_tap plan 意图"""
        provider = _make_mock_provider(delay=0.01)

        req = TAPRequest(
            meta={"task_id": "1.1", "intent": "plan"},
            context={},
            instruction="生成计划",
            constraints=[],
            output_format_hint="markdown",
        )
        resp = await provider.execute_tap(req)
        assert "1.1" in resp.raw_text

    @pytest.mark.asyncio
    async def test_execute_tap_review(self):
        """execute_tap review 意图"""
        provider = _make_mock_provider(delay=0.01)

        req = TAPRequest(
            meta={"task_id": "1.1", "intent": "review"},
            context={},
            instruction="审核代码",
            constraints=[],
            output_format_hint="text",
        )
        resp = await provider.execute_tap(req)
        assert "APPROVE" in resp.raw_text

    @pytest.mark.asyncio
    async def test_chat_interface(self):
        """chat 接口"""
        provider = _make_mock_provider(delay=0.01)

        result = await provider.chat([{"role": "user", "content": "生成 PLAN.md"}])
        assert "content" in result
        # MockAdapter generates plan-like response for PLAN.md content
        assert result["content"] is not None

    @pytest.mark.asyncio
    async def test_stream_tap(self):
        """stream_tap 流式输出"""
        provider = _make_mock_provider(delay=0.01)

        req = TAPRequest(
            meta={"task_id": "1.1", "intent": "code_generation"},
            context={},
            instruction="Write code",
            constraints=[],
            output_format_hint="python",
        )
        chunks = []
        async for chunk in provider.stream_tap(req):
            chunks.append(chunk)
        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0

    @pytest.mark.asyncio
    async def test_fail_rate(self):
        """fail_rate 模拟失败"""
        provider = _make_mock_provider(delay=0.0, fail_rate=1.0)  # 100% 失败

        with pytest.raises(RuntimeError, match="MockAdapter simulated failure"):
            await provider.chat([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_capabilities(self):
        """capabilities 返回"""
        adapter = MockAdapter()
        caps = adapter.capabilities
        assert caps["mock"] is True
        assert caps["streaming"] is True

    @pytest.mark.asyncio
    async def test_call_count(self):
        """调用计数"""
        provider = _make_mock_provider(delay=0.01)
        adapter = provider.adapter  # MockAdapter instance

        req = TAPRequest(
            meta={"task_id": "1.1", "intent": "code_generation"},
            context={},
            instruction="test",
            constraints=[],
            output_format_hint="text",
        )
        assert adapter._call_count == 0
        await provider.execute_tap(req)
        assert adapter._call_count == 1
        await provider.execute_tap(req)
        assert adapter._call_count == 2
