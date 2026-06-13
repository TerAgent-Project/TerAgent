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
import re
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

    # ===== GLM-5 200K 极限压缩方法 =====

    def _compact_design_to_adr(self, content: str, max_tokens: int = 40960) -> str:
        """将设计文档压缩为 ADR（Architecture Decision Record）格式

        ADR 只保留 What 和 Why，去掉 How：
        - What: 做了什么决策（标题 + 状态）
        - Why: 为什么这样决策（上下文 + 理由）
        - 不保留：具体实现细节、代码示例、详细配置

        压缩流程：
        1. 提取所有标题和子标题
        2. 对每个章节，只保留决策描述和理由
        3. 去掉代码块、配置示例、详细步骤
        4. 生成 ADR 格式摘要

        Args:
            content: 原始设计文档内容
            max_tokens: 最大 token 预算（默认 40960）

        Returns:
            ADR 格式的压缩结果
        """
        # 粗略估算：1 token ≈ 1.5 中文字符 或 4 英文字符
        # 为安全起见，用字符数预算 ≈ max_tokens * 2.5
        max_chars = int(max_tokens * 2.5)

        lines = content.split("\n")

        # 1. 去掉代码块（```...```）
        cleaned_lines = self._strip_code_blocks(lines)

        # 2. 提取标题结构和标题下首段内容
        sections = self._extract_adr_sections(cleaned_lines)

        if not sections:
            # 无标题结构，退化为关键句提取
            return self._extract_key_sentences(content, max_chars)

        # 3. 构建 ADR 格式
        adr_parts = ["[ADR 压缩设计文档]\n"]
        total_chars = len(adr_parts[0])

        for section in sections:
            adr_entry = self._format_adr_entry(section)
            if total_chars + len(adr_entry) > max_chars:
                adr_parts.append("\n... (更多决策记录已省略)")
                break
            adr_parts.append(adr_entry)
            total_chars += len(adr_entry)

        result = "\n".join(adr_parts)
        original_len = len(content)
        if len(result) >= original_len:
            # 压缩后反而更长，退化为关键句提取
            return self._extract_key_sentences(content, max_chars)
        return result

    def _compact_history_aggressive(self, content: str, max_tokens: int = 92160) -> str:
        """激进压缩执行历史

        只保留：
        1. 关键决策点（用户明确指示的转折点）
        2. 成功/失败结果（每个阶段的最终结果）
        3. 关键错误信息（导致策略切换的错误）
        4. 策略切换节点（何时切换策略，原因是什么）

        丢弃：
        - 中间步骤的详细输出
        - 重复的尝试过程
        - 成功的工具调用细节
        - 调试过程的中间状态

        Args:
            content: 原始历史内容
            max_tokens: 最大 token 预算（默认 92160）

        Returns:
            激进压缩后的历史摘要
        """
        max_chars = int(max_tokens * 2.5)

        lines = content.split("\n")

        # 提取关键行
        key_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 保留：关键决策点
            if self._is_decision_point(stripped):
                key_lines.append(f"◆ 决策: {stripped}")
                continue

            # 保留：成功/失败结果
            if self._is_result_line(stripped):
                key_lines.append(f"● 结果: {stripped}")
                continue

            # 保留：关键错误
            if self._is_key_error(stripped):
                key_lines.append(f"✖ 错误: {stripped}")
                continue

            # 保留：策略切换
            if self._is_strategy_switch(stripped):
                key_lines.append(f"↻ 切换: {stripped}")
                continue

            # 保留：标题/结构行
            if stripped.startswith("#") or stripped.startswith("##"):
                key_lines.append(stripped)
                continue

        if not key_lines:
            # 没有提取到关键行，退化为头尾保留
            return self._compact_file_content(content)

        result = "[激进压缩历史]\n" + "\n".join(key_lines)

        # 如果仍然超长，截断
        if len(result) > max_chars:
            result = result[:max_chars] + "\n... [进一步截断]"

        original_len = len(content)
        if len(result) >= original_len:
            return self._compact_file_content(content)

        return result

    def assess_compression_quality(self, original: str, compressed: str) -> dict:
        """评估压缩质量

        指标：
        - compression_ratio: 压缩比 (compressed_length / original_length)
        - information_retention: 信息保留率（关键信息覆盖率）
        - key_terms_preserved: 关键术语保留率
        - structure_preserved: 结构保留（标题层级是否完整）

        信息保留率评估方法：
        1. 从原始文本中提取关键术语（标题、类名、函数名等）
        2. 检查压缩后的文本中是否保留了这些关键术语
        3. 计算保留率 = 保留的关键术语数 / 总关键术语数

        Args:
            original: 原始文本
            compressed: 压缩后的文本

        Returns:
            压缩质量评估字典
        """
        if not original:
            return {
                "compression_ratio": 0.0,
                "information_retention": 1.0,
                "key_terms_preserved": 1.0,
                "structure_preserved": 1.0,
            }

        # 1. 压缩比
        compression_ratio = len(compressed) / len(original) if len(original) > 0 else 0.0

        # 2. 关键术语提取与保留率
        key_terms = self._extract_key_terms(original)
        if key_terms:
            preserved = sum(
                1 for term in key_terms if term.lower() in compressed.lower()
            )
            key_terms_preserved = preserved / len(key_terms)
        else:
            key_terms_preserved = 1.0

        # 3. 信息保留率 = 关键术语保留率（基于规则，无 LLM 调用）
        information_retention = key_terms_preserved

        # 4. 结构保留 — 检查标题层级
        original_headings = self._extract_headings(original)
        if original_headings:
            compressed_headings = self._extract_headings(compressed)
            # 检查原始标题中有多少出现在压缩后
            preserved_headings = sum(
                1 for h in original_headings
                if h.lower() in compressed.lower()
            )
            structure_preserved = preserved_headings / len(original_headings)
        else:
            structure_preserved = 1.0

        return {
            "compression_ratio": round(compression_ratio, 4),
            "information_retention": round(information_retention, 4),
            "key_terms_preserved": round(key_terms_preserved, 4),
            "structure_preserved": round(structure_preserved, 4),
        }

    # ===== GLM-5 压缩辅助方法 =====

    @staticmethod
    def _strip_code_blocks(lines: list[str]) -> list[str]:
        """去掉代码块（```...```）"""
        result: list[str] = []
        in_code_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if not in_code_block:
                result.append(line)
        return result

    @staticmethod
    def _extract_adr_sections(lines: list[str]) -> list[dict]:
        """提取标题结构，构建 ADR 章节列表

        返回格式: [{"heading": str, "content": str, "level": int}, ...]
        """
        sections: list[dict] = []
        current_heading = ""
        current_content: list[str] = []
        current_level = 0

        for line in lines:
            # 检测 Markdown 标题
            match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if match:
                # 保存前一个章节
                if current_heading:
                    sections.append({
                        "heading": current_heading,
                        "content": "\n".join(current_content).strip(),
                        "level": current_level,
                    })
                current_level = len(match.group(1))
                current_heading = match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)

        # 最后一个章节
        if current_heading:
            sections.append({
                "heading": current_heading,
                "content": "\n".join(current_content).strip(),
                "level": current_level,
            })

        return sections

    @staticmethod
    def _format_adr_entry(section: dict) -> str:
        """将章节格式化为 ADR 条目

        ADR 格式：
        ## [决策标题]
        - 状态: [已采纳/候选/已弃用]
        - 上下文: [为什么需要做这个决策]
        - 决策: [做了什么选择]
        - 理由: [为什么这样选择]
        """
        heading = section.get("heading", "未命名决策")
        content = section.get("content", "")
        level = section.get("level", 2)

        # 从内容中提取关键信息
        # 尝试识别"状态"、"上下文"、"决策"、"理由"等关键词
        lines = content.split("\n") if content else []
        context_lines: list[str] = []
        reason_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # 去掉列表标记前缀
            cleaned = re.sub(r'^[-*]\s+', '', stripped)
            cleaned = re.sub(r'^\d+\.\s+', '', cleaned)

            # 分类到上下文或理由
            lower = cleaned.lower()
            if any(kw in lower for kw in ("因为", "由于", "原因是", "因为", "理由", "为了", "旨在", "目的是")):
                reason_lines.append(cleaned)
            elif any(kw in lower for kw in ("基于", "考虑到", "根据", "依赖", "背景")):
                context_lines.append(cleaned)
            else:
                # 默认归入上下文
                context_lines.append(cleaned)

        prefix = "#" * min(level, 4)
        parts = [f"{prefix} {heading}"]

        if context_lines:
            # 只保留前 3 行上下文
            parts.append(f"上下文: {'; '.join(context_lines[:3])}")
        if reason_lines:
            # 只保留前 3 行理由
            parts.append(f"理由: {'; '.join(reason_lines[:3])}")

        return "\n".join(parts)

    @staticmethod
    def _extract_key_sentences(content: str, max_chars: int) -> str:
        """提取关键句（ADR 压缩退化策略）

        保留以关键词开头的句子和标题行。
        """
        lines = content.split("\n")
        key_lines: list[str] = []
        total_chars = 0

        # 关键句指示词
        key_indicators = (
            "决策", "选择", "采用", "架构", "设计", "目标", "原则",
            "原因", "理由", "约束", "风险", "权衡", "方案", "策略",
        )

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 保留标题
            if stripped.startswith("#"):
                key_lines.append(stripped)
                total_chars += len(stripped)
            # 保留包含关键指示词的行
            elif any(kw in stripped for kw in key_indicators):
                # 截断过长行
                if len(stripped) > 200:
                    stripped = stripped[:200] + "..."
                key_lines.append(stripped)
                total_chars += len(stripped)

            if total_chars >= max_chars:
                key_lines.append("... (更多内容已省略)")
                break

        if not key_lines:
            # 没有提取到关键句，退化为简单截断
            return content[:max_chars]

        return "[ADR 压缩设计文档 — 关键句提取]\n" + "\n".join(key_lines)

    @staticmethod
    def _is_decision_point(line: str) -> bool:
        """判断是否为关键决策点"""
        decision_keywords = (
            "决定", "决策", "选择", "采用", "切换到", "改为", "调整方案",
            "用户要求", "用户指示", "明确要求", "转折点",
        )
        return any(kw in line for kw in decision_keywords)

    @staticmethod
    def _is_result_line(line: str) -> bool:
        """判断是否为成功/失败结果行"""
        result_keywords = (
            "成功", "完成", "失败", "通过", "未通过", "已实现", "已修复",
            "结果", "最终", "结论", "达成", "未达成",
        )
        return any(kw in line for kw in result_keywords)

    @staticmethod
    def _is_key_error(line: str) -> bool:
        """判断是否为关键错误信息"""
        error_keywords = (
            "Error", "error", "错误", "异常", "Exception", "崩溃",
            "Traceback", "FAILED", "失败原因",
        )
        return any(kw in line for kw in error_keywords)

    @staticmethod
    def _is_strategy_switch(line: str) -> bool:
        """判断是否为策略切换节点"""
        switch_keywords = (
            "切换策略", "换方法", "更换方案", "改用", "改回",
            "回退到", "策略调整", "方法调整", "换方向",
        )
        return any(kw in line for kw in switch_keywords)

    @staticmethod
    def _extract_key_terms(text: str) -> list[str]:
        """从文本中提取关键术语

        提取规则：
        1. Markdown 标题内容
        2. 类名风格标识符（大驼峰，至少 3 字符）
        3. 函数名风格标识符（snake_case，至少 3 字符）
        4. 中文专有名词（2-6 字的中文词组，出现在标题或列表项中）

        Returns:
            去重后的关键术语列表
        """
        terms: list[str] = []

        # 1. Markdown 标题
        for match in re.finditer(r'^#{1,6}\s+(.+)$', text, re.MULTILINE):
            heading = match.group(1).strip()
            # 去掉标题中的格式标记
            heading = re.sub(r'[`*]', '', heading)
            if heading:
                terms.append(heading)

        # 2. 大驼峰标识符（类名）
        for match in re.finditer(r'\b([A-Z][a-zA-Z0-9]{2,})\b', text):
            terms.append(match.group(1))

        # 3. snake_case 标识符（函数名）
        for match in re.finditer(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+){1,})\b', text):
            terms.append(match.group(1))

        # 4. 加粗/代码标记中的内容
        for match in re.finditer(r'[`*]{1,2}([^`*]+)[`*]{1,2}', text):
            content = match.group(1).strip()
            if content and len(content) >= 2:
                terms.append(content)

        # 去重（保持顺序）
        seen: set[str] = set()
        unique_terms: list[str] = []
        for t in terms:
            lower = t.lower()
            if lower not in seen:
                seen.add(lower)
                unique_terms.append(t)

        return unique_terms

    @staticmethod
    def _extract_headings(text: str) -> list[str]:
        """提取所有 Markdown 标题文本"""
        headings: list[str] = []
        for match in re.finditer(r'^#{1,6}\s+(.+)$', text, re.MULTILINE):
            heading = match.group(1).strip()
            heading = re.sub(r'[`*]', '', heading)
            if heading:
                headings.append(heading)
        return headings
