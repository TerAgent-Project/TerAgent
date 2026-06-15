# teragent/context/auto_compact.py
"""AutoCompactor — 自动上下文压缩器

参考 Claude-Code 的 Autocompact + Reactive Compact 策略：
  - 当对话接近 Token 上限时，自动压缩早期对话为摘要
  - 保留最近 N 轮对话，将更早的对话压缩为结构化摘要
  - 熔断器保护：连续压缩失败后停止尝试
  - 单次会话压缩次数上限，防止无限循环

设计原则：
  - 保留近期对话（最近 4 轮 = 8 条消息）
  - 摘要包含 5 个必填维度（需求、完成、决策、问题、文件）
  - 压缩后对话连贯性由摘要保证
  - 系统消息不参与压缩（每次重新注入）
"""

import logging
import time
from typing import TYPE_CHECKING, Optional

__all__ = [
    "AutoCompactor",
]

from teragent.context.context_window import ContextWindow
from teragent.context.microcompactor import Microcompactor
from teragent.context.profiles import GLM5CompactionStrategy
from teragent.core.types import Message, MessageRole, MessageType

if TYPE_CHECKING:
    from teragent.core.compilers.glm_52 import GLM52CompactionProfile
    from teragent.core.provider import ModelProvider

logger = logging.getLogger(__name__)


class AutoCompactor:
    """自动上下文压缩器 — 当接近 Token 限制时压缩历史对话

    使用场景：
      1. AgentLoop._tool_loop 每步开头调用 maybe_compact()
      2. 如果 should_compact() 返回 True，执行压缩
      3. 压缩后消息列表缩短，Token 使用率下降

    压缩流程：
      1. 将消息列表分为「早期」（需压缩）和「近期」（保留原样）
      2. 将早期消息格式化为文本
      3. 调用 LLM 生成结构化摘要
      4. 用摘要消息替换早期消息
      5. 合并为新的消息列表

    安全机制：
      - 熔断器：连续 2 次压缩失败后停止
      - 次数上限：单次会话最多 5 次压缩
      - 消息数下限：消息太少时不压缩
    """

    # 保留的最近消息数量（8 条 = 4 轮 user+assistant）
    RETAIN_COUNT: int = 8

    # 单次会话最大压缩次数
    MAX_COMPACTS: int = 5

    # 连续压缩失败后停止的阈值
    MAX_CONSECUTIVE_FAILURES: int = 2

    # 摘要中每条消息保留的最大字符数
    SUMMARY_MSG_MAX_CHARS: int = 300

    # 摘要输入的最大总字符数
    SUMMARY_INPUT_MAX_CHARS: int = 12_000

    def __init__(
        self,
        context_window: ContextWindow,
        model: "ModelProvider",
        retain_count: int = 8,
        max_compacts: int = 5,
    ) -> None:
        self.context_window = context_window
        self.model = model
        self.retain_count = retain_count
        self.max_compacts = max_compacts

        self._compact_count: int = 0
        self._consecutive_failures: int = 0

        # 历史摘要列表（用于 /status 展示）
        self._compact_history: list[dict] = []

    @property
    def compact_count(self) -> int:
        """当前会话已执行的压缩次数"""
        return self._compact_count

    @property
    def last_compact_info(self) -> Optional[dict]:
        """最近一次压缩的信息"""
        return self._compact_history[-1] if self._compact_history else None

    async def maybe_compact(
        self,
        messages: list[Message],
        system_prompt: str = "",
        cache_hit_rate: float | None = None,
    ) -> list[Message]:
        """检查并执行上下文压缩

        Args:
            messages: 当前对话消息列表（不含系统消息）
            system_prompt: 当前系统提示（用于传递给摘要模型）
            cache_hit_rate: 缓存命中率 (0.0-1.0)，来自适配器响应。
                None = 无缓存信息（默认行为）
                >0.7 = 低压缩（保留缓存友好的内容）
                <0.3 = 激进压缩（释放缓存空间）

        Returns:
            压缩后的消息列表（可能不变）
        """
        # 不需要压缩
        if not self.context_window.should_compact(messages):
            return messages

        # 次数上限
        if self._compact_count >= self.max_compacts:
            logger.warning(
                f"Autocompact limit reached ({self._compact_count}/{self.max_compacts})"
            )
            return messages

        # 熔断器：连续失败后停止
        if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                "Autocompact circuit breaker: too many consecutive failures"
            )
            return messages

        # 消息数太少，压缩无意义
        if len(messages) <= self.retain_count + 2:
            return messages

        # 执行压缩
        try:
            compacted = await self._do_compact(messages, system_prompt, cache_hit_rate=cache_hit_rate)
            self._compact_count += 1
            self._consecutive_failures = 0

            compact_info = {
                "count": self._compact_count,
                "before_messages": len(messages),
                "after_messages": len(compacted),
                "before_tokens": self.context_window.last_estimated_tokens,
                "cache_hit_rate": cache_hit_rate,
                "timestamp": time.time(),
            }
            self._compact_history.append(compact_info)

            logger.info(
                f"Autocompact #{self._compact_count}: "
                f"{len(messages)} messages → {len(compacted)} messages, "
                f"~{self.context_window.last_estimated_tokens} tokens before"
            )
            return compacted

        except Exception as e:
            self._consecutive_failures += 1
            logger.error(f"Autocompact failed (failure #{self._consecutive_failures}): {e}")
            return messages

    async def _do_compact(
        self,
        messages: list[Message],
        system_prompt: str = "",
        cache_hit_rate: float | None = None,
    ) -> list[Message]:
        """执行上下文压缩

        策略：
          1. 保留最近 RETAIN_COUNT 条消息
          2. 将更早的消息压缩为结构化摘要
          3. 摘要消息 + 近期消息 = 新消息列表

        缓存感知调整（cache_hit_rate）：
          - >0.7：低压缩（保留更多上下文，避免破坏缓存前缀）
            retain_count × 1.5, SUMMARY_INPUT_MAX_CHARS × 1.5
          - <0.3：激进压缩（释放缓存空间）
            retain_count × 0.75, SUMMARY_INPUT_MAX_CHARS × 0.75
          - None 或 0.3-0.7：默认行为
        """
        # 根据缓存命中率调整压缩参数
        effective_retain = self.retain_count
        effective_summary_max_chars = self.SUMMARY_INPUT_MAX_CHARS

        if cache_hit_rate is not None:
            if cache_hit_rate > 0.7:
                # 高缓存命中：低压缩，保留更多内容
                effective_retain = int(self.retain_count * 1.5)
                effective_summary_max_chars = int(self.SUMMARY_INPUT_MAX_CHARS * 1.5)
            elif cache_hit_rate < 0.3:
                # 低保存命中：激进压缩，释放空间
                effective_retain = max(2, int(self.retain_count * 0.75))
                effective_summary_max_chars = int(self.SUMMARY_INPUT_MAX_CHARS * 0.75)

        # 修复 H4: 确保 effective_retain 不超过 messages 长度，避免空 to_compact
        effective_retain = min(effective_retain, max(1, len(messages) - 1))
        if effective_retain <= 0 or len(messages) <= 1:
            return None

        to_compact = messages[: -effective_retain]
        to_retain = messages[-effective_retain :]

        # 生成摘要（使用调整后的参数）
        summary = await self._generate_summary(
            to_compact, max_input_chars=effective_summary_max_chars
        )

        # Phase 7.3: 构建摘要消息 — 使用 Message.context_summary
        summary_content = (
            f"[上下文摘要 — 自动压缩 #{self._compact_count + 1}]\n\n"
            f"以下是之前对话的关键信息摘要：\n\n{summary}\n\n"
            f"--- 以上为摘要，以下是最新对话 ---"
        )
        summary_message = Message.context_summary(summary_content)

        return [summary_message, *to_retain]

    async def _generate_summary(
        self,
        messages: list[Message],
        max_input_chars: int | None = None,
    ) -> str:
        """使用 LLM 生成对话摘要

        摘要包含 5 个必填维度：
          1. 用户的需求和目标
          2. 已完成的工作和结果
          3. 重要的决策和原因
          4. 当前的问题和待解决事项
          5. 关键文件和代码变更

        Args:
            messages: 需要压缩的消息列表
            max_input_chars: 摘要输入的最大字符数限制。
                None 时使用默认 SUMMARY_INPUT_MAX_CHARS。
        """
        effective_max = max_input_chars or self.SUMMARY_INPUT_MAX_CHARS
        conversation_text = self._format_for_summary(messages, max_total_chars=effective_max)

        prompt = (
            "请用中文总结以下对话中的关键信息。"
            "必须包含以下 5 个维度：\n\n"
            "1. **用户需求**：用户想要什么？目标是什么？\n"
            "2. **已完成工作**：已经做了什么？结果如何？\n"
            "3. **重要决策**：做出了哪些关键选择？为什么？\n"
            "4. **当前问题**：还有什么未解决？遇到了什么障碍？\n"
            "5. **关键文件**：涉及哪些文件？有哪些代码变更？\n\n"
            f"对话内容：\n{conversation_text}"
        )

        response = await self.model.chat(
            messages=[{"role": "user", "content": prompt}]
        )
        summary = response.get("content", "").strip()

        if not summary:
            # LLM 返回空，使用简单拼接作为备用
            summary = self._fallback_summary(messages)

        return summary

    def _format_for_summary(
        self,
        messages: list[Message],
        max_total_chars: int | None = None,
    ) -> str:
        """将消息列表格式化为摘要输入文本

        每条消息截断到 SUMMARY_MSG_MAX_CHARS，总长度限制在
        max_total_chars 以内。

        Phase 7.3: 使用 Message 属性直接访问，不再通过 dict .get()

        Args:
            messages: 消息列表
            max_total_chars: 总长度限制。None 时使用 SUMMARY_INPUT_MAX_CHARS。
        """
        effective_max = max_total_chars or self.SUMMARY_INPUT_MAX_CHARS
        parts = []
        total_chars = 0

        for msg in messages:
            # 跳过系统消息（每次都会重新注入）
            if msg.role == MessageRole.SYSTEM:
                continue

            # Phase 7.3: 直接访问 Message 属性
            role = msg.role.value
            content = str(msg.content) if msg.content else ""

            # 截断单条消息
            if len(content) > self.SUMMARY_MSG_MAX_CHARS:
                content = content[: self.SUMMARY_MSG_MAX_CHARS] + "..."

            # 处理 tool_calls（简要描述）
            if msg.tool_calls:
                tc_names = [
                    tc.get("function", {}).get("name", "?") for tc in msg.tool_calls
                ]
                content = f"[调用工具: {', '.join(tc_names)}] {content}"

            part = f"[{role}]: {content}"
            parts.append(part)
            total_chars += len(part)

            # 超过总长度限制则停止
            if total_chars >= effective_max:
                parts.append("... (更多对话已省略)")
                break

        return "\n".join(parts)

    def _fallback_summary(self, messages: list[Message]) -> str:
        """LLM 摘要失败时的备用摘要 — 提取关键信息

        Phase 7.3: 使用 Message 属性直接访问
        """
        user_msgs = []
        tool_calls = []
        assistant_msgs = []

        for msg in messages:
            content = str(msg.content)[:200] if msg.content else ""

            if msg.role == MessageRole.USER:
                user_msgs.append(content)
            elif msg.role == MessageRole.ASSISTANT:
                assistant_msgs.append(content)
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append(
                            tc.get("function", {}).get("name", "?")
                        )

        summary_parts = []
        if user_msgs:
            summary_parts.append(
                f"用户需求: {user_msgs[0]}"
            )
        if tool_calls:
            unique_tools = list(dict.fromkeys(tool_calls))  # 去重保序
            summary_parts.append(
                f"使用的工具: {', '.join(unique_tools)}"
            )
        if assistant_msgs:
            summary_parts.append(
                f"助手最近回复: {assistant_msgs[-1]}"
            )

        return "\n".join(summary_parts) if summary_parts else "（摘要生成失败，对话信息已丢失）"

    def get_stats(self) -> dict:
        """返回压缩器统计信息（供 /status 使用）"""
        return {
            "compact_count": self._compact_count,
            "max_compacts": self.max_compacts,
            "consecutive_failures": self._consecutive_failures,
            "last_compact": self.last_compact_info,
        }

    def reset(self) -> None:
        """重置压缩器状态（新会话时调用）"""
        self._compact_count = 0
        self._consecutive_failures = 0
        self._compact_history.clear()

    # ===== GLM-5 专用压缩 =====

    async def _compact_for_glm5(
        self,
        messages: list[Message],
        strategy: GLM5CompactionStrategy,
    ) -> list[Message]:
        """GLM-5 专用压缩流程

        流程：
        1. 分区：系统消息 / 设计文档 / 历史消息 / 近期消息
        2. 对设计文档应用 ADR 压缩
        3. 对历史消息应用激进压缩
        4. 保留近期完整消息
        5. 添加尾部强化
        6. 验证压缩质量（信息保留率 > 80%）

        Args:
            messages: 当前对话消息列表
            strategy: GLM-5 压缩策略配置

        Returns:
            压缩后的消息列表
        """
        microcompactor = Microcompactor()

        # 1. 分区：系统消息 / 设计文档 / 历史消息 / 近期消息
        system_messages: list[Message] = []
        design_messages: list[Message] = []
        history_messages: list[Message] = []
        recent_messages: list[Message] = []

        # 分类消息
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_messages.append(msg)
            elif self._is_design_message(msg):
                design_messages.append(msg)
            else:
                # 非系统、非设计文档的消息
                history_messages.append(msg)

        # 从历史消息中分离近期消息
        keep_count = strategy.keep_recent_messages
        if len(history_messages) > keep_count:
            recent_messages = history_messages[-keep_count:]
            history_messages = history_messages[:-keep_count]
        else:
            recent_messages = history_messages
            history_messages = []

        # 2. 对设计文档应用 ADR 压缩
        compressed_design: list[Message] = []
        if design_messages:
            design_text = "\n\n".join(
                str(msg.content) for msg in design_messages if msg.content
            )
            if design_text:
                compressed_text = microcompactor._compact_design_to_adr(
                    design_text, max_tokens=strategy.design_budget
                )
                # 验证压缩质量
                quality = microcompactor.assess_compression_quality(
                    design_text, compressed_text
                )
                if quality["information_retention"] < strategy.min_information_retention:
                    logger.warning(
                        f"GLM-5 ADR 压缩信息保留率过低: "
                        f"{quality['information_retention']:.2%} < "
                        f"{strategy.min_information_retention:.2%}，"
                        f"退化为原始设计文档"
                    )
                    # 信息保留率不达标，保留原始但截断
                    max_chars = int(strategy.design_budget * 2.5)
                    if len(design_text) > max_chars:
                        compressed_text = design_text[:max_chars] + "\n... [设计文档截断]"
                    else:
                        compressed_text = design_text

                compressed_design.append(
                    Message.context_summary(
                        f"[ADR 压缩设计文档]\n{compressed_text}"
                    )
                )

        # 3. 对历史消息应用激进压缩
        compressed_history: list[Message] = []
        if history_messages:
            history_text = self._format_for_glm5_history(history_messages)
            if history_text:
                compressed_text = microcompactor._compact_history_aggressive(
                    history_text, max_tokens=strategy.history_budget
                )
                # 验证压缩质量
                quality = microcompactor.assess_compression_quality(
                    history_text, compressed_text
                )
                if quality["information_retention"] < strategy.min_information_retention:
                    logger.warning(
                        f"GLM-5 历史压缩信息保留率过低: "
                        f"{quality['information_retention']:.2%}，"
                        f"使用 LLM 摘要替代"
                    )
                    # 退化为 LLM 摘要
                    try:
                        llm_summary = await self._generate_summary(history_messages)
                        compressed_text = f"[激进压缩历史 — LLM 摘要]\n{llm_summary}"
                    except Exception as e:
                        logger.error(f"LLM 摘要生成失败: {e}")
                        # 最终退化为格式化文本
                        compressed_text = f"[历史摘要 — 降级]\n{history_text[:int(strategy.history_budget * 2.5)]}"

                compressed_history.append(
                    Message.context_summary(
                        f"[上下文摘要 — GLM-5 极限压缩]\n{compressed_text}"
                    )
                )

        # 4. 保留近期完整消息
        # recent_messages 已原样保留

        # 5. 添加尾部强化（由 GLM5Compiler 在编译时处理，此处添加标记消息）
        tail_message = Message.system_reminder(
            "【尾部强化占位】GLM5Compiler 将在编译时注入尾部强化内容"
        )

        # 6. 合并消息列表
        result = (
            system_messages
            + compressed_design
            + compressed_history
            + recent_messages
            + [tail_message]
        )

        # 记录压缩信息
        compact_info = {
            "count": self._compact_count + 1,
            "type": "glm5_extreme",
            "before_messages": len(messages),
            "after_messages": len(result),
            "system_count": len(system_messages),
            "design_count": len(compressed_design),
            "history_count": len(compressed_history),
            "recent_count": len(recent_messages),
            "strategy": {
                "keep_recent": strategy.keep_recent_messages,
                "design_target": strategy.design_compression_target,
                "history_target": strategy.history_compression_target,
                "min_retention": strategy.min_information_retention,
            },
            "timestamp": time.time(),
        }
        # 修复 H3/H17: 递增 _compact_count，确保断路器保护生效
        self._compact_count += 1
        self._compact_history.append(compact_info)

        logger.info(
            f"GLM-5 极限压缩 #{self._compact_count}: {len(messages)} messages → {len(result)} messages "
            f"(system={len(system_messages)}, design={len(compressed_design)}, "
            f"history={len(compressed_history)}, recent={len(recent_messages)})"
        )

        return result

    @staticmethod
    def _is_design_message(msg: Message) -> bool:
        """判断消息是否为设计文档内容

        设计文档消息特征：
        - 内容以 <design> 标签开头
        - 内容包含 ADR 格式标记
        - message_type 为 CONTEXT_SUMMARY 且包含设计相关关键词
        """
        content = str(msg.content) if msg.content else ""

        # 包含 <design> 标签
        if content.strip().startswith("<design>"):
            return True

        # 包含 ADR 压缩标记
        if "[ADR" in content and "设计文档" in content:
            return True

        return False

    @staticmethod
    def _format_for_glm5_history(messages: list[Message]) -> str:
        """将历史消息格式化为适合激进压缩的文本

        与 _format_for_summary 不同，此方法保留更多结构信息
        （如决策标记、错误标记），以便激进压缩提取关键行。

        Args:
            messages: 历史消息列表

        Returns:
            格式化后的文本
        """
        parts: list[str] = []

        for msg in messages:
            content = str(msg.content) if msg.content else ""

            # 跳过系统消息
            if msg.role == MessageRole.SYSTEM:
                continue

            # 截断单条消息（比 _format_for_summary 更宽松的截断限制）
            if len(content) > 500:
                content = content[:500] + "..."

            # 标注角色和关键信息
            role = msg.role.value

            # 标注工具调用
            if msg.tool_calls:
                tc_names = [
                    tc.get("function", {}).get("name", "?") for tc in msg.tool_calls
                ]
                content = f"[调用工具: {', '.join(tc_names)}] {content}"

            parts.append(f"[{role}]: {content}")

        return "\n".join(parts)

    # ===== GLM-5.2 时间衰减压缩 =====

    def _apply_time_decay_compression(
        self,
        messages: list[Message],
        profile: "GLM52CompactionProfile",
    ) -> list[Message]:
        """GLM-5.2 时间衰减压缩策略

        利用 1M 上下文窗口的巨大容量，按时间远近分层保留：
        - 最近 200K tokens: 完整保留（所有细节）
        - 200K-600K: 关键步骤 + 结果保留 80%
        - 600K-900K: 中度压缩保留 40%
        - 900K+: 近期结果 + 尾部强化

        与 GLM-5 极限压缩的区别：
        - 不需要 ADR 压缩（设计文档完整保留）
        - 历史按时间衰减而非激进裁剪
        - 保留率远高于 GLM-5 的 15%

        Args:
            messages: 当前对话消息列表
            profile: GLM-5.2 1M 上下文分区配置

        Returns:
            时间衰减压缩后的消息列表
        """
        microcompactor = Microcompactor()

        # 1. 分类消息
        system_messages: list[Message] = []
        user_messages: list[Message] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_messages.append(msg)
            else:
                user_messages.append(msg)

        if not user_messages:
            return messages

        # 2. 估算消息 token 数（粗略：1 token ≈ 2.5 字符）
        def _estimate_msg_tokens(msg: Message) -> int:
            content = str(msg.content) if msg.content else ""
            return max(1, len(content) // 2)

        # 3. 按时间衰减分层
        # 从最新消息向前累加，确定各层边界
        total_tokens = sum(_estimate_msg_tokens(m) for m in user_messages)

        # 如果总量 < 200K，无需压缩
        recent_budget = 200_000  # 完整保留区
        if total_tokens <= recent_budget:
            return messages

        # 从后向前分配层级
        msg_tokens = [_estimate_msg_tokens(m) for m in user_messages]
        cumulative_from_end: list[int] = [0] * len(user_messages)
        cumsum = 0
        for i in range(len(user_messages) - 1, -1, -1):
            cumsum += msg_tokens[i]
            cumulative_from_end[i] = cumsum

        # 确定层级边界
        # Zone 1: 最近 200K — 完整保留
        # Zone 2: 200K-600K — 80% 保留
        # Zone 3: 600K-900K — 40% 保留
        # Zone 4: 900K+ — 仅保留关键结果

        zone1_end = 200_000      # 最近 200K
        zone2_end = 600_000      # 200K-600K
        zone3_end = 900_000      # 600K-900K

        result_messages: list[Message] = []

        for i, msg in enumerate(user_messages):
            dist_from_end = cumulative_from_end[i] - msg_tokens[i]
            msg_tok = msg_tokens[i]

            if dist_from_end + msg_tok <= zone1_end:
                # Zone 1: 完整保留
                result_messages.append(msg)

            elif dist_from_end < zone1_end:
                # 跨越 Zone 1 和 Zone 2 的消息，完整保留
                result_messages.append(msg)

            elif dist_from_end + msg_tok <= zone2_end:
                # Zone 2: 关键步骤 + 结果保留 80%
                if self._is_key_message_for_retention(msg):
                    result_messages.append(msg)
                else:
                    # 轻度压缩：截断长内容
                    compressed = self._light_compress_message(msg, target_ratio=0.8)
                    result_messages.append(compressed)

            elif dist_from_end < zone2_end:
                # 跨越 Zone 2 和 Zone 3
                if self._is_key_message_for_retention(msg):
                    result_messages.append(msg)
                else:
                    compressed = self._light_compress_message(msg, target_ratio=0.6)
                    result_messages.append(compressed)

            elif dist_from_end + msg_tok <= zone3_end:
                # Zone 3: 中度压缩 40%
                if self._is_key_message_for_retention(msg):
                    # 关键消息轻度压缩
                    compressed = self._light_compress_message(msg, target_ratio=0.7)
                    result_messages.append(compressed)
                else:
                    compressed = self._light_compress_message(msg, target_ratio=0.4)
                    result_messages.append(compressed)

            else:
                # Zone 4: 900K+ — 仅保留关键结果
                if self._is_key_message_for_retention(msg):
                    compressed = self._light_compress_message(msg, target_ratio=0.5)
                    result_messages.append(compressed)
                # 非关键消息直接丢弃

        # 4. 合并系统消息 + 压缩后的用户消息
        final = system_messages + result_messages

        # 记录压缩信息
        compact_info = {
            "count": self._compact_count + 1,
            "type": "glm52_time_decay",
            "before_messages": len(messages),
            "after_messages": len(final),
            "total_tokens_estimated": total_tokens,
            "strategy": {
                "zone1_budget": zone1_end,
                "zone2_budget": zone2_end,
                "zone3_budget": zone3_end,
            },
            "timestamp": time.time(),
        }
        # 修复 H3/H17: 递增 _compact_count，确保断路器保护生效
        self._compact_count += 1
        self._compact_history.append(compact_info)

        logger.info(
            f"GLM-5.2 时间衰减压缩 #{self._compact_count}: {len(messages)} messages → {len(final)} messages "
            f"(estimated {total_tokens:,} tokens)"
        )

        return final

    @staticmethod
    def _is_key_message_for_retention(msg: Message) -> bool:
        """判断消息是否为关键消息，在时间衰减压缩中应优先保留

        关键消息特征：
        - 包含决策/错误/结果关键词
        - 上下文摘要消息（之前压缩的产出）
        - 用户消息（包含需求信息）
        - 工具调用消息

        Args:
            msg: 消息对象

        Returns:
            是否为关键消息
        """
        # 上下文摘要始终保留
        if msg.message_type == MessageType.CONTEXT_SUMMARY:
            return True

        # 用户消息始终保留
        if msg.role == MessageRole.USER:
            return True

        # 工具调用消息
        if msg.tool_calls:
            return True

        content = str(msg.content) if msg.content else ""
        if not content:
            return False

        # 检查关键词
        key_keywords = (
            "决定", "决策", "选择", "采用", "切换到", "改为",
            "成功", "完成", "失败", "通过", "未通过",
            "Error", "error", "错误", "异常", "Exception",
            "切换策略", "换方法", "更换方案", "改用",
        )
        if any(kw in content for kw in key_keywords):
            return True

        return False

    @staticmethod
    def _light_compress_message(msg: Message, target_ratio: float = 0.8) -> Message:
        """轻度压缩单条消息

        策略：
        - 保留消息前 80% 和尾部 20% 内容
        - 中间用省略标记替代
        - 保留消息元数据（role, message_type, tool_calls 等）

        Args:
            msg: 原始消息
            target_ratio: 目标保留比例（0.0 ~ 1.0）

        Returns:
            压缩后的新消息
        """
        content = str(msg.content) if msg.content else ""
        if not content:
            return msg

        target_len = max(50, int(len(content) * target_ratio))
        if len(content) <= target_len:
            return msg

        # 保留前 70% 和后 30% 的目标长度
        head_len = int(target_len * 0.7)
        tail_len = target_len - head_len
        omitted = len(content) - head_len - tail_len

        compressed_content = (
            content[:head_len]
            + f"\n... [省略 {omitted} 字符] ...\n"
            + content[-tail_len:]
        )

        # 创建新消息，保留原始元数据
        return Message(
            role=msg.role,
            content=compressed_content,
            message_type=msg.message_type,
            tool_calls=msg.tool_calls,
            tool_call_id=msg.tool_call_id,
            tool_name=msg.tool_name,
            metadata={**msg.metadata, "time_decay_compressed": True},
            timestamp=msg.timestamp,
        )
