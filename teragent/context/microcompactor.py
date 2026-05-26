# teragent/context/microcompactor.py
"""Microcompactor — 工具结果微压缩器

参考 Claude-Code 的 Microcompact 策略：
  - 超过阈值的工具输出自动压缩
  - 按工具类型采用不同压缩策略（文件内容 / 搜索结果 / 通用）
  - 压缩结果保留关键信息，丢弃冗余细节
  - LLM 摘要作为最后手段（有额外 Token 成本）

设计原则：
  - 不丢失定位信息（文件名、行号、函数名）
  - 保留头尾（文件内容头尾各 N 行）
  - 压缩后标注"[已压缩]"让模型知晓
"""

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from teragent.core.provider import ModelProvider

logger = logging.getLogger(__name__)


class Microcompactor:
    """工具结果微压缩器 — 缩减大型工具输出

    使用场景：
      1. AgentLoop 执行工具后，将原始结果经过 microcompactor 压缩
      2. 压缩后的内容替换原始结果加入对话历史
      3. 大幅减少 Token 消耗，避免单次工具输出占满上下文

    压缩策略（按优先级）：
      1. ≤ MAX_INLINE_LENGTH → 原样保留
      2. read_file 结果 → 保留头尾行
      3. explore_codebase / list_directory → 保留结构信息
      4. get_pipeline_status → 保留状态摘要
      5. 其他 → LLM 摘要（可选）或截断
    """

    # 超过此长度的工具结果触发压缩
    MAX_INLINE_LENGTH: int = 2000

    # 文件内容：保留的头/尾行数
    FILE_HEAD_LINES: int = 20
    FILE_TAIL_LINES: int = 10

    # 搜索结果：保留的最大行数
    SEARCH_MAX_LINES: int = 25

    # 通用截断：保留的最大字符数
    TRUNCATE_MAX_CHARS: int = 1500

    # LLM 摘要的最大输入字符数
    LLM_SUMMARY_MAX_INPUT: int = 8000

    # LLM 摘要的最大输出字符数
    LLM_SUMMARY_MAX_OUTPUT: int = 500

    def __init__(
        self,
        max_inline_length: int = 2000,
        file_head_lines: int = 20,
        file_tail_lines: int = 10,
        search_max_lines: int = 25,
    ) -> None:
        self.max_inline_length = max_inline_length
        self.file_head_lines = file_head_lines
        self.file_tail_lines = file_tail_lines
        self.search_max_lines = search_max_lines

    async def compact_tool_result(
        self,
        tool_name: str,
        result: str,
        model: Optional["ModelProvider"] = None,
    ) -> str:
        """压缩工具结果

        Args:
            tool_name: 工具名称（决定压缩策略）
            result: 原始工具输出
            model: 可选的 LLM（用于高级摘要）

        Returns:
            压缩后的内容（≤ MAX_INLINE_LENGTH 或略大）
        """
        # 短内容直接返回
        if len(result) <= self.max_inline_length:
            return result

        # 按工具类型选择压缩策略
        if tool_name == "read_file":
            return self._compact_file_content(result)
        elif tool_name in ("explore_codebase", "list_directory"):
            return self._compact_search_results(result)
        elif tool_name == "get_pipeline_status":
            return self._compact_status(result)
        elif tool_name in ("generate_design", "generate_plan"):
            return self._compact_design_doc(result)
        else:
            return await self._compact_generic(result, model)

    def _compact_file_content(self, content: str) -> str:
        """压缩文件内容 — 保留头尾行"""
        lines = content.split("\n")
        total = len(lines)

        if total <= self.file_head_lines + self.file_tail_lines + 5:
            # 行数不多，直接截断字符
            return self._truncate(content)

        head = "\n".join(lines[: self.file_head_lines])
        tail = "\n".join(lines[-self.file_tail_lines :])
        omitted = total - self.file_head_lines - self.file_tail_lines

        result = (
            f"[已压缩 — 原始 {total} 行，省略中间 {omitted} 行]\n"
            f"{head}\n"
            f"... (省略 {omitted} 行) ...\n"
            f"{tail}"
        )
        if len(result) > self.max_inline_length * 5:
            # Further truncate if still too long — preserve tail lines for context
            tail_content = "\n".join(lines[-self.file_tail_lines:])
            available = max(self.max_inline_length * 3 - len(tail_content) - 100, 200)
            result = result[:available] + f"\n... [进一步截断: 原始 {total} 行] ...\n{tail_content}"
        return result

    def _compact_search_results(self, content: str) -> str:
        """压缩搜索/目录结果 — 保留结构信息"""
        lines = content.split("\n")
        total = len(lines)

        if total <= self.search_max_lines:
            return self._truncate(content)

        # 保留前 N 行（通常包含文件路径和关键匹配）
        head_count = max(self.search_max_lines - 5, 0)
        tail_count = min(5, self.search_max_lines - head_count)
        kept_head = lines[:head_count]
        kept_tail = lines[-tail_count:] if tail_count > 0 else []
        omitted = max(total - len(kept_head) - len(kept_tail), 0)

        return (
            f"[已压缩 — 原始 {total} 行匹配，省略 {omitted} 行]\n"
            + "\n".join(kept_head)
            + f"\n... (省略 {omitted} 行) ...\n"
            + "\n".join(kept_tail)
        )

    def _compact_status(self, content: str) -> str:
        """压缩流水线状态 — 只保留关键状态行"""
        lines = content.split("\n")
        # 状态信息通常较短，如果超长则截断
        if len(lines) <= 15:
            return self._truncate(content)

        # 保留前 10 行（状态摘要）+ 最后 5 行
        head = "\n".join(lines[:10])
        tail = "\n".join(lines[-5:])
        omitted = len(lines) - 15
        return (
            f"[已压缩 — 原始 {len(lines)} 行状态，省略 {omitted} 行]\n"
            f"{head}\n... (省略 {omitted} 行) ...\n{tail}"
        )

    def _compact_design_doc(self, content: str) -> str:
        """压缩设计文档 — 保留章节标题和关键段落

        策略：
          1. 提取所有标题行（# 开头）
          2. 每个标题下最多保留 1 行内容
          3. 总输出限制在 max_inline_length 以内
          4. 如果标题提取导致输出膨胀，退化为头尾保留
        """
        lines = content.split("\n")
        total = len(lines)

        if total <= 40:
            return self._truncate(content)

        # 提取所有标题行（以 # 开头的行）
        heading_lines = []
        for i, line in enumerate(lines):
            if line.strip().startswith("#"):
                heading_lines.append((i, line.strip()))

        # 如果标题行足够描述结构，只保留标题 + 每个标题后的前 1 行
        if len(heading_lines) >= 3:
            result_lines = []
            total_chars = 0
            max_output_chars = self.max_inline_length

            for idx, (line_no, heading) in enumerate(heading_lines):
                # 预算检查：如果已接近限制，只添加标题
                if total_chars + len(heading) > max_output_chars * 0.9:
                    result_lines.append(heading)
                    result_lines.append("... (更多章节省略)")
                    break

                result_lines.append(heading)
                total_chars += len(heading)

                # 每个标题下只保留 1 行内容
                start = line_no + 1
                end = min(start + 1, len(lines))
                if idx < len(heading_lines) - 1:
                    end = min(end, heading_lines[idx + 1][0])
                for j in range(start, end):
                    result_lines.append(lines[j])
                    total_chars += len(lines[j])

            result_text = "\n".join(result_lines)

            # 安全检查：如果提取后反而更长，退化为头尾保留
            if len(result_text) >= len(content):
                return self._compact_file_content(content)

            return (
                f"[已压缩 — 原始 {total} 行设计文档，"
                f"仅保留章节结构和关键内容]\n"
                + result_text
            )

        # 退化为头尾保留
        return self._compact_file_content(content)

    async def _compact_generic(
        self,
        content: str,
        model: Optional["ModelProvider"] = None,
    ) -> str:
        """通用压缩策略 — 先尝试 LLM 摘要，失败则截断"""
        if model is not None:
            try:
                summary = await self._llm_summarize(content, model)
                if summary:
                    return summary
            except Exception as e:
                logger.warning(f"LLM 摘要生成失败，退化为截断: {e}")

        return self._truncate(content)

    async def _llm_summarize(
        self, text: str, model: "ModelProvider"
    ) -> Optional[str]:
        """使用 LLM 生成工具输出摘要

        注意：此方法会消耗额外 Token，仅作为最后手段。
        """
        truncated_input = text[: self.LLM_SUMMARY_MAX_INPUT]
        prompt = (
            "请用中文简洁总结以下工具输出的关键信息"
            f"（不超过 {self.LLM_SUMMARY_MAX_OUTPUT} 字）：\n\n"
            f"{truncated_input}"
        )
        try:
            response = await model.chat(
                messages=[{"role": "user", "content": prompt}]
            )
            summary = response.get("content", "").strip()
            if summary and len(summary) < len(text) * 0.5:
                return f"[LLM 摘要] {summary}"
            return None  # 摘要太长，不如直接截断
        except Exception as e:
            logger.debug(f"LLM summary failed: {e}")
            return None

    def _truncate(self, content: str) -> str:
        """简单截断 — 保留前 TRUNCATE_MAX_CHARS 个字符"""
        if len(content) <= self.TRUNCATE_MAX_CHARS:
            return content
        return (
            f"[已截断 — 原始 {len(content)} 字符，"
            f"仅保留前 {self.TRUNCATE_MAX_CHARS} 字符]\n"
            + content[: self.TRUNCATE_MAX_CHARS]
            + "\n..."
        )
