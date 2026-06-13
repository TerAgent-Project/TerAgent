"""teragent.context.profiles — 模型专属上下文分区配置

不同模型有不同的上下文窗口大小和注意力特性，需要不同的上下文分区策略。
本模块定义了模型专属的 ContextProfile，用于：
  1. 将上下文窗口按比例划分为不同区域（系统提示、对话历史、大文件检索、尾部强化）
  2. 为 ContextWindow 提供分区预算计算
  3. 为 Compiler 提供上下文布局指导

设计参考：design.md §3 V4 1M 上下文管理 / §4 GLM-5 200K 极限压缩 / §5 MiniMax M3 1M MSA 全文注入
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContextProfile:
    """Base context profile for context window management

    上下文分区策略基类，定义了上下文窗口的四个核心区域：
    - system_ratio: 系统提示区域占比（固定前缀，最大化缓存命中）
    - history_ratio: 对话历史区域占比（包含历史对话和参考上下文）
    - large_file_ratio: 大文件检索注入区域占比（检索到的代码片段/文档段落）
    - tail_reinforcement_ratio: 尾部强化区域占比（关键约束和输出格式重申）

    所有 ratio 之和应 <= 1.0。如果之和 < 1.0，剩余部分为弹性缓冲区。

    Attributes:
        max_tokens: 模型上下文窗口上限（tokens）
        system_ratio: 系统提示区域占比
        history_ratio: 对话历史区域占比
        large_file_ratio: 大文件检索注入区域占比
        tail_reinforcement_ratio: 尾部强化区域占比
        description: 配置描述
    """

    max_tokens: int = 128_000
    system_ratio: float = 0.05
    history_ratio: float = 0.55
    large_file_ratio: float = 0.30
    tail_reinforcement_ratio: float = 0.10
    description: str = ""

    @property
    def system_budget(self) -> int:
        """系统提示区域 token 预算"""
        return int(self.max_tokens * self.system_ratio)

    @property
    def history_budget(self) -> int:
        """对话历史区域 token 预算"""
        return int(self.max_tokens * self.history_ratio)

    @property
    def large_file_budget(self) -> int:
        """大文件检索注入区域 token 预算"""
        return int(self.max_tokens * self.large_file_ratio)

    @property
    def tail_reinforcement_budget(self) -> int:
        """尾部强化区域 token 预算"""
        return int(self.max_tokens * self.tail_reinforcement_ratio)

    @property
    def total_allocated_ratio(self) -> float:
        """已分配比例总和（应 <= 1.0）"""
        return self.system_ratio + self.history_ratio + self.large_file_ratio + self.tail_reinforcement_ratio

    @property
    def buffer_ratio(self) -> float:
        """弹性缓冲区比例"""
        return max(0.0, 1.0 - self.total_allocated_ratio)

    def section_budget(self, section: str) -> int:
        """返回指定区域的 token 预算

        Args:
            section: 区域名称，支持 "system", "history", "large_file", "tail_reinforcement"

        Returns:
            该区域的 token 预算

        Raises:
            ValueError: 不支持的区域名称
        """
        mapping = {
            "system": self.system_budget,
            "history": self.history_budget,
            "large_file": self.large_file_budget,
            "tail_reinforcement": self.tail_reinforcement_budget,
        }
        if section not in mapping:
            raise ValueError(
                f"Unknown section: {section!r}. "
                f"Available: {list(mapping.keys())}"
            )
        return mapping[section]


@dataclass
class DeepSeekV4ContextProfile(ContextProfile):
    """DeepSeek V4 1M context profile

    分区策略：
    - [0-50K] 系统提示（冻结前缀，最大化缓存命中）
    - [50K-500K] 对话历史（包含设计文档、执行计划等参考上下文）
    - [500K-900K] 大文件检索注入（CodeIndexer 检索结果）
    - [900K-1M] 尾部强化（关键约束重复 + 输出格式提醒）

    利用 V4 CSA (Causal Sparse Attention) 注意力特性：
    - 首部：系统提示冻结，跨请求缓存命中
    - 中部：对话历史 + 大文件，注意力自然衰减但仍在有效范围
    - 尾部：关键信息强化，利用 Recency Effect 提升遵从率
    """

    max_tokens: int = 1_000_000
    system_ratio: float = 0.05  # 50K
    history_ratio: float = 0.45  # 450K
    large_file_ratio: float = 0.40  # 400K
    tail_reinforcement_ratio: float = 0.10  # 100K
    description: str = "DeepSeek V4 1M context partitioning"


@dataclass
class GLM5ContextProfile(ContextProfile):
    """GLM-5 200K context profile

    分区策略（极限压缩模式）：
    - [0-20K] 系统提示
    - [20K-60K] 压缩设计文档（ADR）
    - [60K-150K] 激进压缩历史
    - [150K-180K] 最近完整消息
    - [180K-200K] 尾部强化

    GLM-5 上下文窗口仅 200K，需要极限压缩策略：
    - 设计文档压缩为 ADR（架构决策记录）
    - 历史对话激进压缩为摘要
    - 保留最近几轮完整消息确保连续性
    - 尾部强化补偿压缩带来的信息损失
    """

    max_tokens: int = 200_000
    system_ratio: float = 0.10  # 20K
    history_ratio: float = 0.45  # 90K (压缩设计 + 历史)
    large_file_ratio: float = 0.15  # 30K (最近完整消息)
    tail_reinforcement_ratio: float = 0.10  # 20K
    description: str = "GLM-5 200K extreme compression"

    design_ratio: float = 0.20  # 40K for compressed design doc
    recent_complete_ratio: float = 0.15  # 30K for recent complete messages

    @property
    def design_budget(self) -> int:
        """压缩设计文档区域 token 预算"""
        return int(self.max_tokens * self.design_ratio)

    @property
    def recent_complete_budget(self) -> int:
        """最近完整消息区域 token 预算"""
        return int(self.max_tokens * self.recent_complete_ratio)

    def section_budget(self, section: str) -> int:
        """返回指定区域的 token 预算

        扩展基类，增加 GLM-5 专属区域：
        - "design": 压缩设计文档
        - "recent_complete": 最近完整消息

        Args:
            section: 区域名称

        Returns:
            该区域的 token 预算
        """
        if section == "design":
            return self.design_budget
        if section == "recent_complete":
            return self.recent_complete_budget
        return super().section_budget(section)


@dataclass
class GLM5CompactionStrategy:
    """GLM-5 200K 极限压缩策略

    分区：
    - [0-20K] 系统提示（不变部分）
    - [20K-60K] 压缩设计文档（ADR格式）
    - [60K-150K] 激进压缩历史（关键决策点 + 结果 + 错误 + 策略切换）
    - [150K-180K] 近期完整消息（最近10条完整保留）
    - [180K-200K] 尾部强化（当前指令 + 约束 + 自评估提示）

    Tokens:
    - 20K = 20,480 tokens
    - 60K = 61,440 tokens
    - 150K = 153,600 tokens
    - 180K = 184,320 tokens
    - 200K = 204,800 tokens
    """

    system_budget: int = 20_480
    design_budget: int = 40_960  # 60K - 20K
    history_budget: int = 92_160  # 150K - 60K
    recent_budget: int = 30_720  # 180K - 150K
    tail_budget: int = 20_480  # 200K - 180K

    # 压缩参数
    keep_recent_messages: int = 10  # 保留最近10条完整消息
    design_compression_target: float = 0.3  # 设计文档压缩到30%原始长度
    history_compression_target: float = 0.15  # 历史压缩到15%原始长度
    min_information_retention: float = 0.8  # 压缩后信息保留率最低80%

    @property
    def total_budget(self) -> int:
        """总 token 预算（应等于 204,800）"""
        return (
            self.system_budget
            + self.design_budget
            + self.history_budget
            + self.recent_budget
            + self.tail_budget
        )


@dataclass
class MiniMaxM3ContextProfile(ContextProfile):
    """MiniMax M3 1M context profile (MSA full-text)

    分区策略：
    - [0-30K] 系统提示
    - [30K-500K] 对话历史
    - [500K-900K] 大文件检索注入（MSA 可以处理全文，直接注入）
    - [900K-1M] 尾部强化

    MSA (Multi-Source Attention) 特性：
    - 支持全文注入，无需像 V4 那样依赖检索
    - 大文件区域直接放入完整源码，而不是检索片段
    - 适合需要完整代码库上下文的任务
    """

    max_tokens: int = 1_000_000
    system_ratio: float = 0.03  # 30K
    history_ratio: float = 0.47  # 470K
    large_file_ratio: float = 0.40  # 400K (MSA can handle full text)
    tail_reinforcement_ratio: float = 0.10  # 100K
    description: str = "MiniMax M3 1M MSA full-text injection"
