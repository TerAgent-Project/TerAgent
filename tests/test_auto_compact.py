# tests/test_auto_compact.py
"""AutoCompactor 自动上下文压缩器单元测试

覆盖:
  - 4 个守卫条件（不应压缩的场景）
  - 压缩结果（消息数减少，系统提示保留）
  - 连续失败熔断
  - 重置行为
  - _format_for_summary / _fallback_summary
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from teragent.context.auto_compact import AutoCompactor
from teragent.context.context_window import ContextWindow
from teragent.core.types import Message, MessageRole, MessageType

# ===== 辅助 =====

def _make_messages(count: int, start: int = 0) -> list[Message]:
    """生成 count 条 user/assistant 交替消息"""
    msgs = []
    for i in range(start, start + count):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        msgs.append(Message(role=role, content=f"消息 {i}", message_type=MessageType.USER_INPUT))
    return msgs


def _make_compactor(
    retain_count: int = 8,
    max_compacts: int = 5,
    compact_threshold: float = 0.85,
) -> tuple[AutoCompactor, ContextWindow, MagicMock]:
    """创建 AutoCompactor 及其依赖，返回 (compactor, cw, model_mock)"""
    cw = ContextWindow(
        model_token_limit=128_000,
        reserved_for_output=4_096,
        reserved_for_system=2_048,
        compact_threshold=compact_threshold,
    )
    model = AsyncMock()
    compactor = AutoCompactor(
        context_window=cw,
        model=model,
        retain_count=retain_count,
        max_compacts=max_compacts,
    )
    return compactor, cw, model


# ===== 守卫条件 =====

class TestGuardConditions:
    """4 个守卫条件 — 不应压缩时返回原消息列表"""

    @pytest.mark.asyncio
    async def test_no_compact_when_below_threshold(self):
        """上下文未达到压缩阈值时不压缩"""
        compactor, cw, _ = _make_compactor(compact_threshold=0.85)
        # 4 条短消息远低于阈值
        msgs = _make_messages(4)
        result = await compactor.maybe_compact(msgs)
        assert result is msgs  # 原样返回

    @pytest.mark.asyncio
    async def test_no_compact_when_max_compacts_reached(self):
        """达到最大压缩次数后不再压缩"""
        compactor, cw, model = _make_compactor(max_compacts=1)
        # 模拟已经压缩过 1 次
        compactor._compact_count = 1

        # 构造足够多的消息让 should_compact 返回 True
        msgs = _make_messages(20)
        # 手动让 should_compact 返回 True
        with patch.object(cw, "should_compact", return_value=True):
            result = await compactor.maybe_compact(msgs)
        assert result is msgs

    @pytest.mark.asyncio
    async def test_no_compact_when_circuit_breaker_tripped(self):
        """连续失败达到熔断阈值后停止压缩"""
        compactor, cw, model = _make_compactor()
        # 模拟连续失败次数达到上限
        compactor._consecutive_failures = AutoCompactor.MAX_CONSECUTIVE_FAILURES

        msgs = _make_messages(20)
        with patch.object(cw, "should_compact", return_value=True):
            result = await compactor.maybe_compact(msgs)
        assert result is msgs

    @pytest.mark.asyncio
    async def test_no_compact_when_too_few_messages(self):
        """消息数太少（≤ retain_count + 2）时不压缩"""
        compactor, cw, model = _make_compactor(retain_count=8)
        msgs = _make_messages(10)  # 10 <= 8+2

        with patch.object(cw, "should_compact", return_value=True):
            result = await compactor.maybe_compact(msgs)
        assert result is msgs


# ===== 压缩结果 =====

class TestCompactResult:
    """压缩结果 — 消息数减少，摘要消息结构正确"""

    @pytest.mark.asyncio
    async def test_messages_reduced_after_compact(self):
        """压缩后消息数减少"""
        compactor, cw, model = _make_compactor(retain_count=4)
        msgs = _make_messages(20)

        # 模拟 LLM 返回摘要
        model.chat = AsyncMock(return_value={"content": "这是一段摘要"})

        with patch.object(cw, "should_compact", return_value=True):
            result = await compactor.maybe_compact(msgs)

        # 压缩后 = 1 条摘要 + retain_count 条近期 = 5
        assert len(result) == 1 + 4
        assert len(result) < len(msgs)

    @pytest.mark.asyncio
    async def test_summary_message_is_context_summary(self):
        """压缩后第一条消息为上下文摘要类型"""
        compactor, cw, model = _make_compactor(retain_count=4)
        msgs = _make_messages(20)
        model.chat = AsyncMock(return_value={"content": "摘要内容"})

        with patch.object(cw, "should_compact", return_value=True):
            result = await compactor.maybe_compact(msgs)

        assert result[0].message_type == MessageType.CONTEXT_SUMMARY
        assert result[0].role == MessageRole.USER

    @pytest.mark.asyncio
    async def test_recent_messages_preserved(self):
        """近期消息原样保留"""
        compactor, cw, model = _make_compactor(retain_count=4)
        msgs = _make_messages(20)
        model.chat = AsyncMock(return_value={"content": "摘要"})

        with patch.object(cw, "should_compact", return_value=True):
            result = await compactor.maybe_compact(msgs)

        # 后 4 条应与原消息后 4 条相同
        for i in range(4):
            assert result[1 + i] is msgs[-4 + i]

    @pytest.mark.asyncio
    async def test_compact_count_increments_on_success(self):
        """成功压缩后计数递增，连续失败重置"""
        compactor, cw, model = _make_compactor(retain_count=4)
        compactor._consecutive_failures = 1
        msgs = _make_messages(20)
        model.chat = AsyncMock(return_value={"content": "摘要"})

        with patch.object(cw, "should_compact", return_value=True):
            await compactor.maybe_compact(msgs)

        assert compactor.compact_count == 1
        assert compactor._consecutive_failures == 0


# ===== 连续失败熔断 =====

class TestConsecutiveFailureCircuitBreaking:
    """连续失败熔断 — LLM 调用异常后递增失败计数"""

    @pytest.mark.asyncio
    async def test_failure_increments_consecutive_count(self):
        """压缩失败递增连续失败计数"""
        compactor, cw, model = _make_compactor(retain_count=4)
        msgs = _make_messages(20)
        model.chat = AsyncMock(side_effect=RuntimeError("LLM 调用失败"))

        with patch.object(cw, "should_compact", return_value=True):
            result = await compactor.maybe_compact(msgs)

        assert compactor._consecutive_failures == 1
        assert result is msgs  # 失败时返回原消息

    @pytest.mark.asyncio
    async def test_circuit_breaker_after_max_failures(self):
        """连续失败达到上限后熔断器生效"""
        compactor, cw, model = _make_compactor(retain_count=4, max_compacts=10)
        compactor._consecutive_failures = AutoCompactor.MAX_CONSECUTIVE_FAILURES - 1

        msgs = _make_messages(20)
        model.chat = AsyncMock(side_effect=RuntimeError("LLM 失败"))

        with patch.object(cw, "should_compact", return_value=True):
            # 第一次：失败计数增加到 MAX，但原消息返回
            await compactor.maybe_compact(msgs)
            assert compactor._consecutive_failures == AutoCompactor.MAX_CONSECUTIVE_FAILURES

            # 第二次：熔断器生效，不再尝试压缩
            result2 = await compactor.maybe_compact(msgs)
            assert result2 is msgs


# ===== 重置行为 =====

class TestResetBehavior:
    """重置行为 — reset() 清空所有内部状态"""

    def test_reset_clears_state(self):
        """reset 清空计数和历史"""
        compactor, cw, model = _make_compactor()
        compactor._compact_count = 3
        compactor._consecutive_failures = 2
        compactor._compact_history.append({"test": True})

        compactor.reset()

        assert compactor.compact_count == 0
        assert compactor._consecutive_failures == 0
        assert compactor._compact_history == []

    def test_get_stats_reflects_state(self):
        """get_stats 返回正确状态"""
        compactor, cw, model = _make_compactor(max_compacts=3)
        compactor._compact_count = 2
        compactor._consecutive_failures = 1

        stats = compactor.get_stats()
        assert stats["compact_count"] == 2
        assert stats["max_compacts"] == 3
        assert stats["consecutive_failures"] == 1
        assert stats["last_compact"] is None

    def test_last_compact_info(self):
        """last_compact_info 返回最近一次压缩信息"""
        compactor, cw, model = _make_compactor()
        assert compactor.last_compact_info is None

        compactor._compact_history.append({"count": 1, "before_messages": 20})
        assert compactor.last_compact_info == {"count": 1, "before_messages": 20}


# ===== _format_for_summary / _fallback_summary =====

class TestSummaryHelpers:
    """辅助方法 — 格式化和备用摘要"""

    def test_format_skips_system_messages(self):
        """格式化时跳过系统消息"""
        compactor, _, _ = _make_compactor()
        msgs = [
            Message.system_prompt("系统提示"),
            Message.user_input("用户消息"),
        ]
        text = compactor._format_for_summary(msgs)
        assert "系统提示" not in text
        assert "用户消息" in text

    def test_format_truncates_long_message(self):
        """格式化时截断超长消息"""
        compactor, _, _ = _make_compactor()
        long_content = "A" * 500
        msgs = [Message.user_input(long_content)]
        text = compactor._format_for_summary(msgs)
        assert len(text) < 500 + 50  # 截断后 + 前缀

    def test_fallback_summary_extracts_key_info(self):
        """备用摘要提取关键信息"""
        compactor, _, _ = _make_compactor()
        msgs = [
            Message.user_input("帮我写代码"),
            Message.assistant_tool_call("", tool_calls=[
                {"function": {"name": "write_file"}}
            ]),
            Message.assistant_text("已完成"),
        ]
        summary = compactor._fallback_summary(msgs)
        assert "帮我写代码" in summary
        assert "write_file" in summary
