# tests/test_recovery.py
"""RecoveryManager 错误恢复策略单元测试

覆盖:
  - 5 种恢复决策类型 (RecoveryType)
  - 错误分类（上下文溢出 vs 可重试）
  - should_continue_after_truncation 逻辑
  - should_retry_streaming 逻辑
  - 恢复统计追踪
  - RecoveryManagerConfig
"""
from unittest.mock import MagicMock

from teragent.reliability.recovery import (
    RecoveryManager,
    RecoveryManagerConfig,
    RecoveryStats,
    RecoveryType,
    is_context_overflow_error,
    is_retryable_error,
)

# ===== RecoveryType 枚举 =====

class TestRecoveryType:
    """5 种恢复决策类型"""

    def test_all_five_types_exist(self):
        """5 种类型都存在"""
        assert RecoveryType.LENGTH.value == "length"
        assert RecoveryType.CONTEXT_OVERFLOW.value == "context_overflow"
        assert RecoveryType.FALLBACK.value == "fallback"
        assert RecoveryType.STREAMING_RETRY.value == "streaming_retry"
        assert RecoveryType.TOOL_REPAIR.value == "tool_repair"

    def test_recovery_type_is_string_enum(self):
        """RecoveryType 是字符串枚举"""
        assert isinstance(RecoveryType.LENGTH, str)
        assert RecoveryType.LENGTH == "length"


# ===== 错误分类 =====

class TestErrorClassification:
    """错误分类 — 上下文溢出 vs 可重试"""

    def test_context_overflow_string(self):
        """字符串匹配上下文溢出模式"""
        assert is_context_overflow_error("context_length_exceeded") is True
        assert is_context_overflow_error("prompt is too long") is True
        assert is_context_overflow_error("too many tokens") is True
        assert is_context_overflow_error("上下文长度超出") is True

    def test_context_overflow_exception(self):
        """Exception 对象匹配上下文溢出模式"""
        assert is_context_overflow_error(ValueError("request too large")) is True
        assert is_context_overflow_error(RuntimeError("413 payload too large")) is True

    def test_non_overflow_error(self):
        """非溢出错误返回 False"""
        assert is_context_overflow_error("connection timeout") is False
        assert is_context_overflow_error("invalid api key") is False

    def test_retryable_error_string(self):
        """字符串匹配可重试模式"""
        assert is_retryable_error("rate_limit exceeded") is True
        assert is_retryable_error("429 too many requests") is True
        assert is_retryable_error("503 service unavailable") is True

    def test_retryable_error_exception(self):
        """Exception 对象匹配可重试模式"""
        assert is_retryable_error(ConnectionError("server error")) is True
        assert is_retryable_error(RuntimeError("temporarily unavailable")) is True

    def test_non_retryable_error(self):
        """不可重试错误返回 False"""
        assert is_retryable_error("invalid api key") is False
        assert is_retryable_error("context_length_exceeded") is False

    def test_context_overflow_and_retryable_are_independent(self):
        """上下文溢出和可重试是独立的分类"""
        # 429 是可重试但不是溢出
        assert is_context_overflow_error("429") is False
        assert is_retryable_error("429") is True
        # context_length 是溢出但不是可重试
        assert is_context_overflow_error("context_length_exceeded") is True
        assert is_retryable_error("context_length_exceeded") is False


# ===== should_continue_after_truncation =====

class TestShouldContinueAfterTruncation:
    """should_continue_after_truncation 逻辑"""

    def test_continue_when_length_and_below_max(self):
        """finish_reason='length' 且未达上限时继续"""
        mgr = RecoveryManager()
        assert mgr.should_continue_after_truncation("length", 0) is True
        assert mgr.should_continue_after_truncation("length", 2) is True

    def test_stop_when_not_length(self):
        """finish_reason 不是 'length' 时不继续"""
        mgr = RecoveryManager()
        assert mgr.should_continue_after_truncation("stop", 0) is False
        assert mgr.should_continue_after_truncation(None, 0) is False

    def test_stop_when_attempt_exceeds_max(self):
        """尝试次数超过上限时停止"""
        mgr = RecoveryManager()
        # 默认 max_output_tokens_recovery=3
        assert mgr.should_continue_after_truncation("length", 3) is False
        assert mgr.should_continue_after_truncation("length", 4) is False


# ===== should_retry_streaming =====

class TestShouldRetryStreaming:
    """should_retry_streaming 逻辑"""

    def test_retry_below_max(self):
        """低于最大重试次数时重试"""
        mgr = RecoveryManager()
        assert mgr.should_retry_streaming(0) is True
        assert mgr.should_retry_streaming(1) is True

    def test_no_retry_at_max(self):
        """达到最大重试次数时不重试"""
        mgr = RecoveryManager()
        # 默认 max_streaming_retries=2
        assert mgr.should_retry_streaming(2) is False

    def test_custom_config(self):
        """自定义配置生效"""
        config = RecoveryManagerConfig(max_streaming_retries=5)
        mgr = RecoveryManager(config=config)
        assert mgr.should_retry_streaming(4) is True
        assert mgr.should_retry_streaming(5) is False


# ===== 恢复统计追踪 =====

class TestRecoveryStatsTracking:
    """恢复统计追踪"""

    def test_record_length_recovery(self):
        """记录 LENGTH 恢复"""
        mgr = RecoveryManager()
        mgr.record_recovery(RecoveryType.LENGTH)
        stats = mgr.get_stats()
        assert stats["length_recoveries"] == 1
        assert stats["last_recovery_type"] == "length"

    def test_record_multiple_types(self):
        """记录多种恢复类型"""
        mgr = RecoveryManager()
        mgr.record_recovery(RecoveryType.LENGTH)
        mgr.record_recovery(RecoveryType.CONTEXT_OVERFLOW)
        mgr.record_recovery(RecoveryType.STREAMING_RETRY)
        stats = mgr.get_stats()
        assert stats["length_recoveries"] == 1
        assert stats["overflow_recoveries"] == 1
        assert stats["streaming_retries"] == 1

    def test_fallback_provider_in_stats(self):
        """回退提供商出现在统计中"""
        mock_provider = MagicMock()
        mock_provider.model = "gpt-3.5-turbo"
        mgr = RecoveryManager(fallback_provider=mock_provider)
        mgr.record_recovery(RecoveryType.FALLBACK)
        stats = mgr.get_stats()
        assert stats["fallback_model"] == "gpt-3.5-turbo"
        assert stats["has_fallback"] is True

    def test_reset_stats(self):
        """重置统计"""
        mgr = RecoveryManager()
        mgr.record_recovery(RecoveryType.LENGTH)
        mgr.record_recovery(RecoveryType.CONTEXT_OVERFLOW)
        mgr.reset_stats()
        stats = mgr.get_stats()
        assert stats["length_recoveries"] == 0
        assert stats["overflow_recoveries"] == 0
        assert stats["last_recovery_type"] is None

    def test_recovery_stats_to_dict(self):
        """RecoveryStats.to_dict 导出"""
        rs = RecoveryStats()
        rs.record(RecoveryType.TOOL_REPAIR)
        d = rs.to_dict()
        assert d["tool_repairs"] == 1
        assert d["last_recovery_type"] == "tool_repair"
        assert d["last_recovery_time"] is not None

    def test_config_from_dict(self):
        """RecoveryManagerConfig.from_dict"""
        data = {
            "max_output_tokens_recovery": 5,
            "max_context_overflow_recovery": 3,
            "max_streaming_retries": 4,
            "enable_fallback": False,
        }
        config = RecoveryManagerConfig.from_dict(data)
        assert config.max_output_tokens_recovery == 5
        assert config.max_context_overflow_recovery == 3
        assert config.max_streaming_retries == 4
        assert config.enable_fallback is False

    def test_config_from_dict_defaults(self):
        """RecoveryManagerConfig.from_dict 缺失字段使用默认值"""
        config = RecoveryManagerConfig.from_dict({})
        assert config.max_output_tokens_recovery == 3
        assert config.max_context_overflow_recovery == 2
        assert config.max_streaming_retries == 2
        assert config.enable_fallback is True

    def test_manager_convenience_methods(self):
        """RecoveryManager 的 is_context_overflow / is_retryable 便捷方法"""
        mgr = RecoveryManager()
        assert mgr.is_context_overflow("context_length_exceeded") is True
        assert mgr.is_context_overflow("timeout") is False
        assert mgr.is_retryable("429 rate limit") is True
        assert mgr.is_retryable("invalid key") is False

    def test_has_fallback_property(self):
        """has_fallback 属性"""
        mgr1 = RecoveryManager()
        assert mgr1.has_fallback is False

        mock_provider = MagicMock()
        mgr2 = RecoveryManager(fallback_provider=mock_provider)
        assert mgr2.has_fallback is True

    def test_fallback_provider_setter(self):
        """fallback_provider 可设置"""
        mgr = RecoveryManager()
        assert mgr.has_fallback is False
        mgr.fallback_provider = MagicMock()
        assert mgr.has_fallback is True
