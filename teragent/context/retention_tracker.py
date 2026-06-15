"""teragent/context/retention_tracker.py — 1M 上下文信息保留率追踪器

GLM-5.2 1M 上下文智能分区的关键组件：
  - 追踪每个分区（zone）中实际被模型引用的信息比例
  - 根据引用率动态调整分区大小
  - 当 1M 上下文整体保留率过低时，自动降级到 200K 极限压缩模式

设计参考：design.md §6 GLM-5.2 深度适配 — 动态分区 + 保留率追踪
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

__all__ = [
    "LongContextRetentionTracker",
    "PartitionUsage",
    "RetentionRecord",
]

logger = logging.getLogger(__name__)


# ===== 数据结构 =====

@dataclass
class PartitionUsage:
    """单个分区的使用统计

    Attributes:
        zone: 分区名称（system / plan / execution_high / execution_mid / recent / tail）
        tokens_provided: 提供给模型的 token 数
        tokens_referenced: 模型响应中引用的 token 数（估算）
        sample_count: 采样次数
        last_updated: 最后更新时间戳
    """

    zone: str
    tokens_provided: int = 0
    tokens_referenced: int = 0
    sample_count: int = 0
    last_updated: float = 0.0

    @property
    def retention_rate(self) -> float:
        """保留率 = 引用 token 数 / 提供 token 数"""
        if self.tokens_provided == 0:
            return 1.0
        return self.tokens_referenced / self.tokens_provided

    def record(self, tokens_provided: int, tokens_referenced: int) -> None:
        """记录一次采样

        使用增量平均更新统计值，避免需要存储完整历史。

        Args:
            tokens_provided: 本次提供的 token 数
            tokens_referenced: 本次引用的 token 数
        """
        self.sample_count += 1
        # 增量平均：new_avg = old_avg + (new_val - old_avg) / n
        # 修复：使用 round() 代替 int()，避免截断导致系统性向下漂移
        alpha = 1.0 / self.sample_count
        self.tokens_provided = round(
            self.tokens_provided * (1 - alpha) + tokens_provided * alpha
        )
        self.tokens_referenced = round(
            self.tokens_referenced * (1 - alpha) + tokens_referenced * alpha
        )
        self.last_updated = time.time()


@dataclass
class RetentionRecord:
    """单次保留率记录

    用于历史追踪和趋势分析。

    Attributes:
        timestamp: 记录时间戳
        overall_retention: 整体保留率
        zone_retentions: 各分区保留率字典
        should_downgrade: 是否建议降级
    """

    timestamp: float
    overall_retention: float
    zone_retentions: dict[str, float]
    should_downgrade: bool = False


# ===== 追踪器主类 =====

class LongContextRetentionTracker:
    """1M 上下文信息保留率追踪器

    监控 GLM-5.2 1M 上下文窗口中各分区的信息利用效率：
    - 追踪每个分区实际被模型响应引用的 token 比例
    - 根据引用模式优化分区分配
    - 在整体效率不足时建议降级到 200K 模式

    使用方式：
        tracker = LongContextRetentionTracker(max_context_tokens=1_000_000)

        # 编译后记录分区使用情况
        tracker.record_partition("system", 50_000, 45_000)
        tracker.record_partition("execution_high", 400_000, 280_000)

        # 查询保留率
        rate = tracker.get_retention_rate("system")

        # 检查是否需要降级
        if tracker.should_downgrade():
            # 切换到 GLM-5 200K 极限压缩模式
            ...

        # 获取优化建议
        suggestions = tracker.get_optimization_suggestions()
    """

    # 降级阈值
    OVERALL_RETENTION_THRESHOLD = 0.70   # 整体保留率低于 70% 触发降级评估
    TAIL_ZONE_RETENTION_THRESHOLD = 0.50 # 尾部分区保留率低于 50% 触发降级评估
    CONSECUTIVE_LOW_RETENTION_LIMIT = 3  # 连续 N 次低保留率触发降级

    # 分区名称常量
    ZONES = ("system", "plan", "execution_high", "execution_mid", "recent", "tail")

    def __init__(self, max_context_tokens: int = 1_000_000) -> None:
        self.max_context_tokens = max_context_tokens

        # 各分区使用统计
        self._partition_usage: dict[str, PartitionUsage] = {
            zone: PartitionUsage(zone=zone) for zone in self.ZONES
        }

        # 保留率历史记录（最多保留 100 条）
        self._retention_history: list[RetentionRecord] = []
        self._max_history = 100

        # 连续低保留率计数器
        self._consecutive_low_retention: int = 0

        # 是否已降级
        self._downgraded: bool = False
        self._downgrade_reason: str = ""

    # ===== 核心方法 =====

    def record_partition(
        self, zone: str, tokens_provided: int, tokens_referenced: int
    ) -> None:
        """记录分区使用情况

        在每次编译后调用，更新分区的使用统计。

        Args:
            zone: 分区名称
            tokens_provided: 提供给模型的 token 数
            tokens_referenced: 模型响应中引用的 token 数（估算）

        Raises:
            ValueError: 不支持的分区名称
        """
        if zone not in self._partition_usage:
            logger.warning(f"Unknown zone: {zone!r}, ignoring record")
            return

        self._partition_usage[zone].record(tokens_provided, tokens_referenced)

        # 更新整体保留率并检查降级条件
        overall = self.get_overall_retention_rate()
        zone_retentions = {
            z: usage.retention_rate
            for z, usage in self._partition_usage.items()
            if usage.sample_count > 0
        }

        should_down = self._check_downgrade(overall, zone_retentions)

        record = RetentionRecord(
            timestamp=time.time(),
            overall_retention=overall,
            zone_retentions=dict(zone_retentions),
            should_downgrade=should_down,
        )
        self._retention_history.append(record)

        # 限制历史长度
        if len(self._retention_history) > self._max_history:
            self._retention_history = self._retention_history[-self._max_history:]

    def get_retention_rate(self, zone: str) -> float:
        """获取分区的平均保留率

        Args:
            zone: 分区名称

        Returns:
            保留率（0.0 ~ 1.0），如果分区无数据返回 1.0
        """
        if zone not in self._partition_usage:
            return 1.0
        return self._partition_usage[zone].retention_rate

    def get_overall_retention_rate(self) -> float:
        """获取整体加权保留率

        按各分区 token 预算加权计算整体保留率。

        Returns:
            加权保留率（0.0 ~ 1.0）
        """
        total_provided = 0
        total_referenced = 0

        for usage in self._partition_usage.values():
            if usage.sample_count > 0:
                total_provided += usage.tokens_provided
                total_referenced += usage.tokens_referenced

        if total_provided == 0:
            return 1.0

        return total_referenced / total_provided

    def should_downgrade(self) -> bool:
        """检查是否应从 1M 降级到 200K 模式

        降级条件（满足任一即触发）：
        1. 整体保留率 < 70% 连续 3+ 次请求
        2. 尾部分区（tail / recent）保留率 < 50%

        Returns:
            是否应降级
        """
        return self._downgraded

    def get_downgrade_reason(self) -> str:
        """获取降级原因

        Returns:
            降级原因字符串，如果未降级返回空字符串
        """
        return self._downgrade_reason

    def reset_downgrade(self) -> None:
        """重置降级状态

        当模型切换回 1M 模式时调用。
        """
        self._downgraded = False
        self._downgrade_reason = ""
        self._consecutive_low_retention = 0

    def get_optimization_suggestions(self) -> list[str]:
        """根据使用模式生成分区优化建议

        分析各分区的保留率，提出重平衡建议。

        Returns:
            优化建议列表
        """
        suggestions: list[str] = []

        # 收集有数据的分区
        active_zones = {
            z: usage for z, usage in self._partition_usage.items()
            if usage.sample_count > 0
        }

        if not active_zones:
            return ["数据不足，需要更多请求后才能提供优化建议"]

        overall = self.get_overall_retention_rate()

        # 1. 整体保留率低
        if overall < self.OVERALL_RETENTION_THRESHOLD:
            suggestions.append(
                f"整体保留率 {overall:.1%} 低于阈值 {self.OVERALL_RETENTION_THRESHOLD:.0%}，"
                f"建议考虑降级到 200K 极限压缩模式"
            )

        # 2. 高保留率分区可以缩减
        for zone, usage in active_zones.items():
            if usage.retention_rate > 0.95 and usage.tokens_provided > 10_000:
                suggestions.append(
                    f"分区 '{zone}' 保留率 {usage.retention_rate:.1%} 极高，"
                    f"可考虑缩减预算（当前约 {usage.tokens_provided:,} tokens）"
                )

        # 3. 低保留率分区需要增加
        for zone, usage in active_zones.items():
            if usage.retention_rate < 0.50:
                suggestions.append(
                    f"分区 '{zone}' 保留率 {usage.retention_rate:.1%} 过低，"
                    f"建议检查该分区内容质量或增加预算"
                )

        # 4. 尾部分区特殊检查
        tail_usage = self._partition_usage.get("tail")
        if tail_usage and tail_usage.sample_count > 0:
            if tail_usage.retention_rate < self.TAIL_ZONE_RETENTION_THRESHOLD:
                suggestions.append(
                    f"尾部分区保留率 {tail_usage.retention_rate:.1%} < "
                    f"{self.TAIL_ZONE_RETENTION_THRESHOLD:.0%}，"
                    f"尾强化内容可能冗余，建议精简"
                )

        # 5. 如果没有问题，给出正面反馈
        if not suggestions:
            suggestions.append(
                f"各分区保留率正常（整体 {overall:.1%}），当前分区策略有效"
            )

        return suggestions

    def get_stats(self) -> dict:
        """返回追踪器统计信息

        Returns:
            统计信息字典
        """
        return {
            "max_context_tokens": self.max_context_tokens,
            "overall_retention": self.get_overall_retention_rate(),
            "downgraded": self._downgraded,
            "downgrade_reason": self._downgrade_reason,
            "consecutive_low_retention": self._consecutive_low_retention,
            "partitions": {
                zone: {
                    "retention_rate": usage.retention_rate,
                    "tokens_provided": usage.tokens_provided,
                    "tokens_referenced": usage.tokens_referenced,
                    "sample_count": usage.sample_count,
                }
                for zone, usage in self._partition_usage.items()
            },
            "history_size": len(self._retention_history),
        }

    def reset(self) -> None:
        """重置追踪器状态"""
        self._partition_usage = {
            zone: PartitionUsage(zone=zone) for zone in self.ZONES
        }
        self._retention_history.clear()
        self._consecutive_low_retention = 0
        self._downgraded = False
        self._downgrade_reason = ""

    # ===== 内部方法 =====

    def _check_downgrade(
        self,
        overall_retention: float,
        zone_retentions: dict[str, float],
    ) -> bool:
        """检查降级条件并更新降级状态

        Args:
            overall_retention: 当前整体保留率
            zone_retentions: 各分区保留率

        Returns:
            本次检查是否触发了降级
        """
        # 条件 1：整体保留率 < 70%
        if overall_retention < self.OVERALL_RETENTION_THRESHOLD:
            self._consecutive_low_retention += 1
            if self._consecutive_low_retention >= self.CONSECUTIVE_LOW_RETENTION_LIMIT:
                if not self._downgraded:
                    self._downgraded = True
                    self._downgrade_reason = (
                        f"整体保留率 {overall_retention:.1%} < "
                        f"{self.OVERALL_RETENTION_THRESHOLD:.0%} "
                        f"连续 {self._consecutive_low_retention} 次"
                    )
                    logger.warning(
                        f"GLM-5.2 降级触发: {self._downgrade_reason}"
                    )
                return True
        else:
            # 保留率恢复正常，重置连续计数（但不重置降级状态，需手动 reset_downgrade）
            self._consecutive_low_retention = 0

        # 条件 2：尾部分区保留率 < 50%
        tail_retention = zone_retentions.get("tail", 1.0)
        recent_retention = zone_retentions.get("recent", 1.0)

        if tail_retention < self.TAIL_ZONE_RETENTION_THRESHOLD:
            if not self._downgraded:
                self._downgraded = True
                self._downgrade_reason = (
                    f"尾部分区保留率 {tail_retention:.1%} < "
                    f"{self.TAIL_ZONE_RETENTION_THRESHOLD:.0%}"
                )
                logger.warning(
                    f"GLM-5.2 降级触发: {self._downgrade_reason}"
                )
            return True

        if recent_retention < self.TAIL_ZONE_RETENTION_THRESHOLD:
            if not self._downgraded:
                self._downgraded = True
                self._downgrade_reason = (
                    f"近区分区保留率 {recent_retention:.1%} < "
                    f"{self.TAIL_ZONE_RETENTION_THRESHOLD:.0%}"
                )
                logger.warning(
                    f"GLM-5.2 降级触发: {self._downgrade_reason}"
                )
            return True

        return False

    def estimate_referenced_tokens(
        self, provided_tokens: int, response_text: str, zone_content: str
    ) -> int:
        """估算模型响应中引用了某分区多少 token

        基于规则的方法：检查分区内容中的关键术语是否出现在响应中。

        Args:
            provided_tokens: 该分区提供的 token 数
            response_text: 模型响应文本
            zone_content: 分区内容文本

        Returns:
            估算的引用 token 数
        """
        if not zone_content or not response_text:
            return 0

        # 提取分区内容中的关键短语（按行分割，每行作为一个可引用单元）
        zone_lines = [line.strip() for line in zone_content.split("\n") if line.strip()]
        if not zone_lines:
            return 0

        # 检查每行是否在响应中出现
        referenced_lines = 0
        for line in zone_lines:
            # 对长行只检查前 50 个字符作为匹配键
            key = line[:50].lower()
            if key and key in response_text.lower():
                referenced_lines += 1

        # 按行数比例估算引用的 token 数
        reference_ratio = referenced_lines / len(zone_lines) if zone_lines else 0
        return int(provided_tokens * reference_ratio)
