# tests/test_context_window.py
"""ContextWindow Token 预算感知上下文窗口管理器单元测试

覆盖:
  - Token 估算逻辑（中英混合启发式）
  - 阈值判定（should_compact / should_warn）
  - 可用预算计算（available_budget）
  - last_estimated_tokens 追踪
  - usage_ratio / summary
"""

from teragent.context.context_window import ContextWindow
from teragent.core.types import Message

# ===== 辅助 =====

def _make_messages(contents: list[str]) -> list[dict]:
    """生成消息列表（dict 格式）"""
    return [{"role": "user", "content": c} for c in contents]


# ===== Token 估算逻辑 =====

class TestTokenEstimation:
    """Token 估算逻辑 — 中英混合启发式"""

    def test_pure_english_estimation(self):
        """纯英文文本估算"""
        cw = ContextWindow(model_token_limit=128_000)
        # 纯英文：4 字符/Token，再加保守系数 1.3
        msgs = [{"role": "user", "content": "A" * 400}]
        tokens = cw.estimate_tokens(msgs)
        # 每条消息额外加 40 字符格式开销
        # total_chars = 400 + 40 = 440, chinese=0, other=440
        # estimated = 440/4 = 110 * 1.3 = 143
        expected = int(((400 + 40) / 4.0) * 1.3)
        assert tokens == expected

    def test_pure_chinese_estimation(self):
        """纯中文文本估算"""
        cw = ContextWindow(model_token_limit=128_000)
        # 中文字符：1.5 字符/Token
        chinese = "中" * 300
        msgs = [{"role": "user", "content": chinese}]
        tokens = cw.estimate_tokens(msgs)
        # total_chars = 300 + 40 = 340, chinese=300, other=40
        # estimated = (300/1.5) + (40/4) = 200 + 10 = 210 * 1.3 = 273
        expected = int(((300 / 1.5) + (40 / 4.0)) * 1.3)
        assert tokens == expected

    def test_mixed_language_estimation(self):
        """中英混合文本估算"""
        cw = ContextWindow(model_token_limit=128_000)
        content = "Hello你好" * 50  # 50 * 7 = 350 字符
        msgs = [{"role": "user", "content": content}]
        tokens = cw.estimate_tokens(msgs)
        # 总字符 350 + 40 格式 = 390, 中文 100, 其他 290
        chinese_chars = sum(1 for c in content if "\u4e00" <= c <= "\u9fff")
        other_chars = len(content) + 40 - chinese_chars
        expected = int(((chinese_chars / 1.5) + (other_chars / 4.0)) * 1.3)
        assert tokens == expected

    def test_message_object_estimation(self):
        """使用 Message 对象进行估算"""
        cw = ContextWindow(model_token_limit=128_000)
        msgs = [Message.user_input("Hello world")]
        tokens = cw.estimate_tokens(msgs)
        assert tokens > 0

    def test_tool_calls_estimated(self):
        """tool_calls 的 arguments 也被估算"""
        cw = ContextWindow(model_token_limit=128_000)
        msgs = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {"name": "write_file", "arguments": "A" * 200}
            }]
        }]
        tokens = cw.estimate_tokens(msgs)
        assert tokens > 0  # 应包含 arguments 的估算


# ===== 阈值判定 =====

class TestThresholdDetermination:
    """阈值判定 — should_compact / should_warn"""

    def test_should_compact_below_threshold(self):
        """低于 compact_threshold 不需要压缩"""
        cw = ContextWindow(
            model_token_limit=1000,
            reserved_for_output=0,
            reserved_for_system=0,
            compact_threshold=0.85,
        )
        # 400/4*1.3 = 130 tokens, budget = 1000, ratio = 0.13 < 0.85
        msgs = [{"role": "user", "content": "A" * 400}]
        assert cw.should_compact(msgs) is False

    def test_should_compact_above_threshold(self):
        """超过 compact_threshold 需要压缩"""
        cw = ContextWindow(
            model_token_limit=200,
            reserved_for_output=0,
            reserved_for_system=0,
            compact_threshold=0.85,
        )
        # 构造大量消息使估算超过 200*0.85=170 tokens
        msgs = [{"role": "user", "content": "A" * 2000}]
        assert cw.should_compact(msgs) is True

    def test_should_warn_below_threshold(self):
        """低于 warn_threshold 不需要警告"""
        cw = ContextWindow(
            model_token_limit=10000,
            reserved_for_output=0,
            reserved_for_system=0,
            warn_threshold=0.75,
        )
        msgs = [{"role": "user", "content": "short"}]
        assert cw.should_warn(msgs) is False

    def test_should_warn_above_threshold(self):
        """超过 warn_threshold 需要警告"""
        cw = ContextWindow(
            model_token_limit=200,
            reserved_for_output=0,
            reserved_for_system=0,
            warn_threshold=0.75,
        )
        msgs = [{"role": "user", "content": "A" * 2000}]
        assert cw.should_warn(msgs) is True


# ===== 可用预算计算 =====

class TestAvailableBudget:
    """可用预算计算"""

    def test_available_budget_calculation(self):
        """available_budget = model_token_limit - reserved_output - reserved_system"""
        cw = ContextWindow(
            model_token_limit=128_000,
            reserved_for_output=4_096,
            reserved_for_system=2_048,
        )
        assert cw.available_budget == 128_000 - 4_096 - 2_048

    def test_custom_budget_params(self):
        """自定义预算参数"""
        cw = ContextWindow(
            model_token_limit=50_000,
            reserved_for_output=5_000,
            reserved_for_system=1_000,
        )
        assert cw.available_budget == 50_000 - 5_000 - 1_000


# ===== last_estimated_tokens 追踪 =====

class TestLastEstimatedTokens:
    """last_estimated_tokens 追踪"""

    def test_initial_value_zero(self):
        """初始值为 0"""
        cw = ContextWindow()
        assert cw.last_estimated_tokens == 0

    def test_updated_after_estimate(self):
        """estimate_tokens 后更新缓存"""
        cw = ContextWindow()
        msgs = [{"role": "user", "content": "Hello"}]
        tokens = cw.estimate_tokens(msgs)
        assert cw.last_estimated_tokens == tokens

    def test_updated_after_should_compact(self):
        """should_compact 后也更新缓存"""
        cw = ContextWindow()
        msgs = [{"role": "user", "content": "Hello world test"}]
        cw.should_compact(msgs)
        assert cw.last_estimated_tokens > 0

    def test_summary_dict(self):
        """summary 返回正确字典"""
        cw = ContextWindow(
            model_token_limit=1000,
            reserved_for_output=100,
            reserved_for_system=100,
        )
        msgs = [{"role": "user", "content": "test"}]
        s = cw.summary(msgs)
        assert s["model_token_limit"] == 1000
        assert s["available_budget"] == 800
        assert "usage_ratio" in s
        assert "should_warn" in s
        assert "should_compact" in s
        assert s["message_count"] == 1

    def test_usage_ratio_without_messages(self):
        """不传 messages 时返回缓存的使用率"""
        cw = ContextWindow()
        # 初始 ratio = 0.0
        assert cw.usage_ratio() == 0.0
        # 估算后缓存
        cw.estimate_tokens([{"role": "user", "content": "A" * 1000}])
        ratio = cw.usage_ratio()
        assert ratio > 0.0
