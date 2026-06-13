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
from typing import Optional, TYPE_CHECKING

from teragent.context.context_window import ContextWindow
from teragent.context.profiles import GLM5CompactionStrategy
from teragent.context.microcompactor import Microcompactor
from teragent.core.types import Message, MessageRole, MessageType

if TYPE_CHECKING:
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
    ) -> list[Message]:
        """检查并执行上下文压缩

        Args:
            messages: 当前对话消息列表（不含系统消息）
            system_prompt: 当前系统提示（用于传递给摘要模型）

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
            compacted = await self._do_compact(messages, system_prompt)
            self._compact_count += 1
            self._consecutive_failures = 0

            compact_info = {
                "count": self._compact_count,
                "before_messages": len(messages),
                "after_messages": len(compacted),
                "before_tokens": self.context_window.last_estimated_tokens,
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
    ) -> list[Message]:
        """执行上下文压缩

        策略：
          1. 保留最近 RETAIN_COUNT 条消息
          2. 将更早的消息压缩为结构化摘要
          3. 摘要消息 + 近期消息 = 新消息列表
        """
        to_compact = messages[: -self.retain_count]
        to_retain = messages[-self.retain_count :]

        # 生成摘要
        summary = await self._generate_summary(to_compact)

        # Phase 7.3: 构建摘要消息 — 使用 Message.context_summary
        summary_content = (
            f"[上下文摘要 — 自动压缩 #{self._compact_count + 1}]\n\n"
            f"以下是之前对话的关键信息摘要：\n\n{summary}\n\n"
            f"--- 以上为摘要，以下是最新对话 ---"
        )
        summary_message = Message.context_summary(summary_content)

        return [summary_message, *to_retain]

    async def _generate_summary(self, messages: list[Message]) -> str:
        """使用 LLM 生成对话摘要

        摘要包含 5 个必填维度：
          1. 用户的需求和目标
          2. 已完成的工作和结果
          3. 重要的决策和原因
          4. 当前的问题和待解决事项
          5. 关键文件和代码变更
        """
        conversation_text = self._format_for_summary(messages)

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

    def _format_for_summary(self, messages: list[Message]) -> str:
        """将消息列表格式化为摘要输入文本

        每条消息截断到 SUMMARY_MSG_MAX_CHARS，总长度限制在
        SUMMARY_INPUT_MAX_CHARS 以内。

        Phase 7.3: 使用 Message 属性直接访问，不再通过 dict .get()
        """
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
            if total_chars >= self.SUMMARY_INPUT_MAX_CHARS:
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
        self._compact_history.append(compact_info)

        logger.info(
            f"GLM-5 极限压缩: {len(messages)} messages → {len(result)} messages "
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
