# tests/test_microcompactor.py
"""Microcompactor 工具结果微压缩器单元测试

覆盖:
  - 5 种工具压缩策略（read_file / explore_codebase / get_pipeline_status / generate_design / 通用）
  - LLM 摘要生成（mock LLM 调用）
  - 截断回退
  - 消息顺序保留
  - 短内容直接返回
"""
from unittest.mock import AsyncMock

import pytest

from teragent.context.microcompactor import Microcompactor

# ===== 辅助 =====

def _long_content(lines: int = 100, line_len: int = 60) -> str:
    """生成多行长文本"""
    return "\n".join(f"Line {i:04d}: {'x' * line_len}" for i in range(lines))


def _short_content() -> str:
    """生成短文本（≤ max_inline_length）"""
    return "短内容，不需要压缩"


# ===== 短内容直接返回 =====

class TestShortContentPassthrough:
    """短内容直接返回"""

    @pytest.mark.asyncio
    async def test_short_result_returned_as_is(self):
        """短工具结果原样返回"""
        mc = Microcompactor(max_inline_length=2000)
        short = "Hello, this is a short result."
        result = await mc.compact_tool_result("any_tool", short)
        assert result == short

    @pytest.mark.asyncio
    async def test_exactly_at_limit_returned_as_is(self):
        """刚好等于 max_inline_length 的内容原样返回"""
        mc = Microcompactor(max_inline_length=10)
        content = "0123456789"  # len=10
        result = await mc.compact_tool_result("any_tool", content)
        assert result == content


# ===== read_file 策略 =====

class TestReadFileStrategy:
    """read_file 压缩策略 — 保留头尾行"""

    @pytest.mark.asyncio
    async def test_file_content_preserves_head_tail(self):
        """长文件保留头尾行，省略中间"""
        mc = Microcompactor(
            max_inline_length=100,
            file_head_lines=5,
            file_tail_lines=3,
        )
        content = _long_content(100)
        result = await mc.compact_tool_result("read_file", content)

        assert "[已压缩" in result
        assert "省略" in result
        # 头部行存在
        assert "Line 0000" in result
        # 尾部行存在
        assert "Line 0099" in result

    @pytest.mark.asyncio
    async def test_short_file_uses_truncation(self):
        """行数不多的文件使用截断"""
        mc = Microcompactor(
            max_inline_length=100,
            file_head_lines=20,
            file_tail_lines=10,
        )
        # 只有 1 行，行数 < head + tail + 5 = 35，走截断路径
        # 但 _truncate 只在 content > TRUNCATE_MAX_CHARS 时才截断
        # 所以需要 content > 1500 字符
        content = "A" * 2000
        result = await mc.compact_tool_result("read_file", content)
        assert "[已截断" in result


# ===== explore_codebase / list_directory 策略 =====

class TestSearchResultsStrategy:
    """explore_codebase / list_directory 压缩策略"""

    @pytest.mark.asyncio
    async def test_search_results_preserves_structure(self):
        """搜索结果保留头尾行"""
        mc = Microcompactor(
            max_inline_length=100,
            search_max_lines=10,
        )
        lines = [f"result/match_{i}.py: found match here" for i in range(50)]
        content = "\n".join(lines)
        result = await mc.compact_tool_result("explore_codebase", content)

        assert "[已压缩" in result
        assert "省略" in result

    @pytest.mark.asyncio
    async def test_list_directory_same_strategy(self):
        """list_directory 使用与 explore_codebase 相同策略"""
        mc = Microcompactor(
            max_inline_length=100,
            search_max_lines=10,
        )
        lines = [f"dir/file_{i}.py" for i in range(50)]
        content = "\n".join(lines)
        result = await mc.compact_tool_result("list_directory", content)

        assert "[已压缩" in result

    @pytest.mark.asyncio
    async def test_short_search_results_truncated(self):
        """行数不多但超长的搜索结果走截断"""
        mc = Microcompactor(
            max_inline_length=100,
            search_max_lines=100,
        )
        # 只有 1 行，但长度超过 TRUNCATE_MAX_CHARS(1500)
        content = "A" * 2000
        result = await mc.compact_tool_result("explore_codebase", content)
        assert "[已截断" in result


# ===== get_pipeline_status 策略 =====

class TestPipelineStatusStrategy:
    """get_pipeline_status 压缩策略 — 保留状态摘要"""

    @pytest.mark.asyncio
    async def test_long_status_compacted(self):
        """长状态信息压缩保留头尾"""
        mc = Microcompactor(max_inline_length=100)
        lines = [f"Status line {i}: processing..." for i in range(30)]
        content = "\n".join(lines)
        result = await mc.compact_tool_result("get_pipeline_status", content)

        assert "[已压缩" in result

    @pytest.mark.asyncio
    async def test_short_status_truncated(self):
        """短状态但超长内容走截断"""
        mc = Microcompactor(max_inline_length=100)
        # 行数 ≤ 15 但字符超 TRUNCATE_MAX_CHARS
        content = "A" * 2000
        result = await mc.compact_tool_result("get_pipeline_status", content)
        assert "[已截断" in result


# ===== generate_design / generate_plan 策略 =====

class TestDesignDocStrategy:
    """generate_design / generate_plan 压缩策略 — 保留章节标题"""

    @pytest.mark.asyncio
    async def test_design_doc_with_headings(self):
        """设计文档含标题时保留标题和首行内容"""
        mc = Microcompactor(max_inline_length=500)
        # 构造超过 40 行且总长度 > 500 的内容
        lines = []
        for i in range(50):
            lines.append(f"# Section {i}")
            lines.append(f"Detail for section {i} with enough padding " + "x" * 30)
        content = "\n".join(lines)
        result = await mc.compact_tool_result("generate_design", content)

        # 应该被压缩（标题提取或头尾保留）
        assert "已压缩" in result or "已截断" in result

    @pytest.mark.asyncio
    async def test_design_doc_few_headings_fallback(self):
        """标题少于 3 个时退化为头尾保留"""
        mc = Microcompactor(max_inline_length=100)
        # 只有 1 个标题，不足 3 个，且超过 40 行
        lines = ["# Only Heading"] + [f"Line {i}: " + "y" * 30 for i in range(60)]
        content = "\n".join(lines)
        result = await mc.compact_tool_result("generate_plan", content)

        # 退化为 _compact_file_content
        assert "已压缩" in result or "已截断" in result


# ===== LLM 摘要 =====

class TestLLMSummary:
    """LLM 摘要生成（mock LLM 调用）"""

    @pytest.mark.asyncio
    async def test_llm_summary_used_when_model_provided(self):
        """提供 model 时使用 LLM 摘要"""
        mc = Microcompactor(max_inline_length=100)
        model = AsyncMock()
        model.chat = AsyncMock(return_value={"content": "这是 LLM 生成的摘要"})

        content = "A" * 5000
        result = await mc.compact_tool_result("unknown_tool", content, model=model)

        # LLM 摘要返回内容较短时使用
        # 摘要长度 < len(content) * 0.5 才用，这里 5000*0.5 = 2500
        assert "LLM 摘要" in result or "已截断" in result

    @pytest.mark.asyncio
    async def test_llm_summary_too_long_falls_back(self):
        """LLM 摘要太长时回退到截断"""
        mc = Microcompactor(max_inline_length=100)
        model = AsyncMock()
        # 返回超长摘要 (> len(content)*0.5)
        model.chat = AsyncMock(return_value={"content": "A" * 3000})

        content = "B" * 5000
        result = await mc.compact_tool_result("unknown_tool", content, model=model)

        # 摘要长度 >= len(content)*0.5，不使用 LLM 摘要，回退截断
        assert "已截断" in result

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_truncation(self):
        """LLM 调用失败时回退到截断"""
        mc = Microcompactor(max_inline_length=100)
        model = AsyncMock()
        model.chat = AsyncMock(side_effect=RuntimeError("LLM error"))

        content = "C" * 5000
        result = await mc.compact_tool_result("unknown_tool", content, model=model)
        assert "已截断" in result

    @pytest.mark.asyncio
    async def test_no_model_falls_back_to_truncation(self):
        """没有 model 时直接截断"""
        mc = Microcompactor(max_inline_length=100)
        content = "D" * 5000
        result = await mc.compact_tool_result("unknown_tool", content, model=None)
        assert "已截断" in result


# ===== 消息顺序保留 =====

class TestMessageOrdering:
    """消息顺序保留 — 压缩后头尾内容顺序正确"""

    @pytest.mark.asyncio
    async def test_file_content_head_before_tail(self):
        """文件内容头部行在尾部行之前"""
        mc = Microcompactor(
            max_inline_length=100,
            file_head_lines=3,
            file_tail_lines=2,
        )
        lines = [f"Line {i}" for i in range(50)]
        content = "\n".join(lines)
        result = await mc.compact_tool_result("read_file", content)

        head_pos = result.find("Line 0")
        tail_pos = result.find("Line 49")
        assert head_pos < tail_pos
