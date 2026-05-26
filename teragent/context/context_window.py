# teragent/context/context_window.py
"""ContextWindow -- Token 预算感知的上下文窗口管理器

参考 Claude-Code 的上下文管理策略：
  - 维护 Token 预算，在接近上限时触发警告或压缩
  - 提供粗略但快速的 Token 估算（中英混合文本）
  - 不依赖外部 Tokenizer，保持零依赖

设计原则：
  - 估算保守（偏高），宁可早压缩也不溢出
  - warn_threshold < compact_threshold，留出缓冲区
  - usage_ratio() 供 TUI 实时展示
"""

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from teragent.core.types import Message

logger = logging.getLogger(__name__)


class ContextWindow:
    """Token 预算感知的上下文窗口管理器

    使用场景：
      1. AgentLoop 每步调用 should_compact() 判断是否需要压缩
      2. TUI 通过 usage_ratio() 展示当前上下文使用率
      3. AutoCompactor 依赖此对象做压缩决策

    Token 估算采用中英混合启发式：
      - 中文字符：约 1.5 字符 / Token
      - 英文/其他：约 4 字符 / Token
      - 保守系数 x1.3（避免低估导致 API 溢出）
    """

    # 保守系数：估算值乘以此系数，避免低估
    CONSERVATIVE_FACTOR = 1.3

    def __init__(
        self,
        model_token_limit: int = 128_000,
        reserved_for_output: int = 4_096,
        reserved_for_system: int = 2_048,
        warn_threshold: float = 0.75,
        compact_threshold: float = 0.85,
    ) -> None:
        self.model_token_limit = model_token_limit
        self.reserved_for_output = reserved_for_output
        self.reserved_for_system = reserved_for_system
        self.warn_threshold = warn_threshold
        self.compact_threshold = compact_threshold

        # 可用 Token 预算 = 模型上限 - 输出预留 - 系统预留
        self.available_budget = (
            model_token_limit - reserved_for_output - reserved_for_system
        )
        if self.available_budget <= 0:
            logger.warning(
                f"Available budget is {self.available_budget} "
                f"(limit={model_token_limit}, output={reserved_for_output}, "
                f"system={reserved_for_system}), clamping to 1"
            )
            self.available_budget = max(self.available_budget, 1)

        # 最近一次估算结果缓存
        self._last_estimated_tokens: int = 0
        self._last_usage_ratio: float = 0.0

    def estimate_tokens(self, messages: list) -> int:
        """估算消息列表的 Token 数量

        使用中英混合启发式估算，并施加保守系数。

        Args:
            messages: 消息列表（list[Message] 或 list[dict]）

        Returns:
            估算的 Token 数量（已含保守系数）
        """
        total_chars = 0
        chinese_chars = 0

        for msg in messages:
            # Phase 7.3: 支持 Message 对象和 dict
            content = self._get_msg_content(msg)
            total_chars += len(content)
            chinese_chars += sum(
                1 for c in content if "\u4e00" <= c <= "\u9fff"
            )

            # 估算 tool_calls 的 arguments
            for tc in self._get_msg_tool_calls(msg):
                func = tc.get("function", {}) if isinstance(tc, dict) else {}
                args = str(func.get("arguments", ""))
                total_chars += len(args)
                chinese_chars += sum(
                    1 for c in args if "\u4e00" <= c <= "\u9fff"
                )

            # role + tool_call_id 等元数据开销（每条约 40 字符 ≈ 10 Token）
            total_chars += 40  # 粗略：role + 格式开销 ≈ 40 字符

        # 混合估算：中文 1.5 字符/Token，其他 4 字符/Token
        other_chars = total_chars - chinese_chars
        estimated = (chinese_chars / 1.5) + (other_chars / 4.0)

        # 施加保守系数
        estimated = int(estimated * self.CONSERVATIVE_FACTOR)

        # 缓存
        self._last_estimated_tokens = estimated
        self._last_usage_ratio = estimated / self.available_budget if self.available_budget > 0 else 1.0

        return estimated

    def should_compact(self, messages: list) -> bool:
        """是否需要压缩上下文

        当估算 Token 数超过可用预算的 compact_threshold 时返回 True。
        """
        estimated = self.estimate_tokens(messages)
        ratio = estimated / self.available_budget if self.available_budget > 0 else 1.0
        return ratio >= self.compact_threshold

    def should_warn(self, messages: Optional[list] = None) -> bool:
        """是否需要警告上下文即将满

        优先使用缓存的估算值（避免重复估算），仅在需要时重新估算。

        Args:
            messages: 如果提供则重新估算，否则使用缓存值
        """
        if messages is not None:
            self.estimate_tokens(messages)
        ratio = self._last_usage_ratio
        return ratio >= self.warn_threshold

    def usage_ratio(self, messages: Optional[list] = None) -> float:
        """获取当前上下文使用率

        Args:
            messages: 如果提供则重新估算，否则返回缓存值

        Returns:
            0.0 ~ 1.0+ 的使用率
        """
        if messages is not None:
            self.estimate_tokens(messages)
        return self._last_usage_ratio

    @property
    def last_estimated_tokens(self) -> int:
        """最近一次估算的 Token 数"""
        return self._last_estimated_tokens

    def summary(self, messages: list) -> dict:
        """返回上下文使用情况摘要（供 TUI /status 使用）"""
        estimated = self.estimate_tokens(messages)
        ratio = self._last_usage_ratio
        return {
            "estimated_tokens": estimated,
            "available_budget": self.available_budget,
            "model_token_limit": self.model_token_limit,
            "usage_ratio": ratio,
            "usage_percent": f"{ratio * 100:.1f}%",
            "should_warn": ratio >= self.warn_threshold,
            "should_compact": ratio >= self.compact_threshold,
            "message_count": len(messages),
        }

    # ===== Phase 7.3: Message 兼容辅助方法 =====

    @staticmethod
    def _get_msg_content(msg) -> str:
        """从 Message 对象或 dict 中获取 content 字段"""
        if hasattr(msg, "content"):
            return str(msg.content) if msg.content else ""
        return str(msg.get("content", ""))

    @staticmethod
    def _get_msg_tool_calls(msg) -> list:
        """从 Message 对象或 dict 中获取 tool_calls 字段"""
        if hasattr(msg, "tool_calls"):
            return msg.tool_calls or []
        return msg.get("tool_calls", []) or []
