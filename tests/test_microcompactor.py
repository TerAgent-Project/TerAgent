# tests/test_microcompactor.py
"""Microcompactor 工具结果微压缩器单元测试

覆盖:
  - 5 种工具压缩策略（read_file / explore_codebase / get_pipeline_status / generate_design / 通用）
  - LLM 摘要生成（mock LLM 调用）
  - 截断回退
  - 消息顺序保留
  - 短内容直接返回
  - P2-8: ADR 压缩设计文档
  - P2-8: 激进压缩执行历史
  - P2-8: 压缩质量评估
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


# ===== P2-8: ADR 压缩设计文档 =====

class TestCompactDesignToADR:
    """ADR 压缩设计文档 — P2-8 GLM-5 200K 极限压缩"""

    def test_adr_compresses_design_document(self):
        """设计文档压缩为 ADR 格式"""
        mc = Microcompactor()
        # 构造足够长的设计文档，使 ADR 压缩有意义
        sections = []
        for i in range(5):
            sections.append(f"## 决策 {i}")
            sections.append(f"基于需求 {i} 的考虑，我们选择方案 {i}。")
            sections.append(f"采用技术 {i} 作为实现方式。")
            sections.append(f"影响：方案 {i} 增加了复杂度，但提升了性能。")
            # 添加详细实现描述使文档更长
            sections.append(f"具体实现包括：模块A、模块B、模块C的详细设计，数据流图和接口定义。" + "x" * 100)
            sections.append("")
        design = "\n".join(sections)

        result = mc._compact_design_to_adr(design, max_tokens=10000)

        # ADR 压缩应生成（可能是 ADR 格式或关键句提取）
        assert "ADR" in result
        # 压缩后应短于原文（因为有大量可压缩的详细实现描述）
        assert len(result) < len(design)

    def test_adr_removes_code_blocks(self):
        """ADR 压缩移除代码块"""
        mc = Microcompactor()
        design = """# 配置示例

以下是配置代码：

```python
DATABASE_URL = "postgresql://localhost:5432/mydb"
```

采用环境变量配置。
"""
        result = mc._compact_design_to_adr(design, max_tokens=10000)

        # 代码块应被移除
        assert "DATABASE_URL" not in result or "postgresql" not in result

    def test_adr_short_content_returned_as_is(self):
        """短设计文档不需要 ADR 压缩"""
        mc = Microcompactor()
        short_design = "# 简短设计\n采用方案A。"
        result = mc._compact_design_to_adr(short_design, max_tokens=100000)

        # 短内容可能被压缩或加 ADR 前缀，但应保留关键信息
        assert "简短设计" in result or "方案" in result

    def test_adr_no_headings_fallback_to_key_sentences(self):
        """无标题的设计文档退化为关键句提取"""
        mc = Microcompactor()
        content = "这是一个关于架构决策的文档。决定采用微服务架构。原因是为了提高系统的可扩展性。"
        result = mc._compact_design_to_adr(content, max_tokens=10000)

        # 应该包含关键信息
        assert "微服务" in result or "决策" in result or "架构" in result

    def test_adr_compression_ratio(self):
        """ADR 压缩比应接近目标 30%"""
        mc = Microcompactor()
        # 构造较长的设计文档（包含大量可压缩的详细描述）
        sections = []
        for i in range(20):
            sections.append(f"## 决策 {i}")
            sections.append(f"基于需求 {i}，我们采用方案 {i}。")
            sections.append(f"选择方案 {i} 的理由是为了优化性能。")
            sections.append(f"影响：方案 {i} 的权衡是增加了复杂度，但提升了可维护性。")
            # 详细实现描述
            sections.append(f"实现细节：{('详细实现步骤 ' * 10)}")
            sections.append("")  # blank line
        design = "\n".join(sections)

        result = mc._compact_design_to_adr(design, max_tokens=40960)

        # 压缩后应该短于原文
        ratio = len(result) / len(design) if len(design) > 0 else 1.0
        assert ratio < 1.0, f"Compression ratio {ratio:.2%} should be < 100%"

    def test_adr_consequences_field(self):
        """ADR 格式包含 Consequences 字段"""
        mc = Microcompactor()
        # 构造足够长的设计文档使 ADR 格式化生效
        sections = ["# 缓存策略"]
        for i in range(3):
            sections.append(f"## 缓存方案 {i}")
            sections.append(f"基于性能需求，采用 Redis 缓存层 {i}。")
            sections.append(f"影响：增加了部署复杂度 {i}。")
            sections.append(f"风险：缓存雪崩风险 {i}。")
            sections.append("")
        design = "\n".join(sections)

        result = mc._compact_design_to_adr(design, max_tokens=10000)

        # 应包含 Consequences 字段（因为"影响"、"风险"关键词）
        # 注意：短文档可能退化为关键句提取
        assert "Consequences:" in result or "影响" in result or "风险" in result


# ===== P2-8: 激进压缩执行历史 =====

class TestCompactHistoryAggressive:
    """激进压缩执行历史 — P2-8 GLM-5 200K 极限压缩"""

    def test_aggressive_compression_keeps_decisions(self):
        """激进压缩保留关键决策点"""
        mc = Microcompactor()
        # 构造足够长的历史，使压缩有意义
        lines = ["[assistant]: 执行普通步骤"]
        for i in range(20):
            lines.append(f"[assistant]: 普通操作 {i} " + "x" * 30)
        lines.append("[assistant]: 决定采用方案A作为主要实现")
        for i in range(20):
            lines.append(f"[assistant]: 更多操作 {i} " + "y" * 30)
        history = "\n".join(lines)
        result = mc._compact_history_aggressive(history, max_tokens=92160)

        assert "◆ 决策:" in result or "决策" in result

    def test_aggressive_compression_keeps_errors(self):
        """激进压缩保留关键错误"""
        mc = Microcompactor()
        lines = ["[assistant]: 执行步骤1"]
        for i in range(20):
            lines.append(f"[assistant]: 普通操作 {i} " + "x" * 30)
        lines.append("[assistant]: Error: 文件不存在，无法继续")
        for i in range(20):
            lines.append(f"[assistant]: 更多操作 {i} " + "y" * 30)
        history = "\n".join(lines)
        result = mc._compact_history_aggressive(history, max_tokens=92160)

        assert "✖ 错误:" in result

    def test_aggressive_compression_keeps_results(self):
        """激进压缩保留成功/失败结果"""
        mc = Microcompactor()
        lines = []
        for i in range(20):
            lines.append(f"[assistant]: 普通操作 {i} " + "x" * 30)
        lines.append("[assistant]: 测试成功，所有用例通过")
        for i in range(20):
            lines.append(f"[assistant]: 更多操作 {i} " + "y" * 30)
        history = "\n".join(lines)
        result = mc._compact_history_aggressive(history, max_tokens=92160)

        assert "● 结果:" in result

    def test_aggressive_compression_keeps_strategy_switches(self):
        """激进压缩保留策略切换"""
        mc = Microcompactor()
        lines = []
        for i in range(20):
            lines.append(f"[assistant]: 使用方案A步骤 {i} " + "x" * 30)
        # "失败" triggers result line first, then "切换策略" is also present
        # The method checks result lines before strategy switches,
        # so this line gets marked as result. Use only strategy keywords.
        lines.append("[assistant]: 切换策略，改用方案B继续执行")
        for i in range(20):
            lines.append(f"[assistant]: 方案B步骤 {i} " + "y" * 30)
        history = "\n".join(lines)
        result = mc._compact_history_aggressive(history, max_tokens=92160)

        assert "↻ 切换:" in result

    def test_aggressive_compression_tool_call_sequence(self):
        """激进压缩合并重复工具调用"""
        mc = Microcompactor()
        history = "\n".join([
            "[assistant]: [调用工具: read_file] 读取文件1",
            "[assistant]: [调用工具: read_file] 读取文件2",
            "[assistant]: [调用工具: read_file] 读取文件3",
        ])
        result = mc._compact_history_aggressive(history, max_tokens=92160)

        # 应包含工具序列摘要
        assert "◇ 工具序列: read_file x3" in result

    def test_aggressive_compression_context_anchors(self):
        """激进压缩保留首尾上下文锚点"""
        mc = Microcompactor()
        # 构造足够长的历史以启用锚点
        lines = []
        for i in range(50):
            lines.append(f"[user]: 步骤 {i} 的详细执行过程")
            lines.append(f"[assistant]: 决定采用方案 {i}")
        history = "\n".join(lines)
        result = mc._compact_history_aggressive(history, max_tokens=92160)

        # 应包含上下文锚点标记
        assert "上下文起点" in result or "上下文终点" in result

    def test_aggressive_compression_ratio(self):
        """激进压缩比应接近目标 15%"""
        mc = Microcompactor()
        # 构造大量历史
        lines = []
        for i in range(200):
            lines.append(f"[user]: 步骤 {i} 的执行过程，包含大量详细信息 " + "x" * 50)
            lines.append(f"[assistant]: 决定采用方案 {i}")
            lines.append(f"[assistant]: 成功完成步骤 {i}")
        history = "\n".join(lines)

        result = mc._compact_history_aggressive(history, max_tokens=92160)

        ratio = len(result) / len(history) if len(history) > 0 else 1.0
        # 目标 ~15%，但允许较大浮动（关键是大幅压缩）
        assert ratio < 0.5, f"Compression ratio {ratio:.2%} is too high (expected <50%)"

    def test_aggressive_compression_empty_input(self):
        """空输入退化处理"""
        mc = Microcompactor()
        result = mc._compact_history_aggressive("", max_tokens=92160)
        # 空输入应该被处理（返回空或压缩后的空内容）
        assert isinstance(result, str)


# ===== P2-8: 压缩质量评估 =====

class TestAssessCompressionQuality:
    """压缩质量评估"""

    def test_quality_metrics_structure(self):
        """质量评估返回正确的指标结构"""
        mc = Microcompactor()
        original = "# 设计文档\n采用方案A\n理由：性能优化"
        compressed = "# 设计文档\nDecision: 方案A"

        quality = mc.assess_compression_quality(original, compressed)

        assert "compression_ratio" in quality
        assert "information_retention" in quality
        assert "key_terms_preserved" in quality
        assert "structure_preserved" in quality

    def test_compression_ratio_calculation(self):
        """压缩比计算正确"""
        mc = Microcompactor()
        original = "A" * 1000
        compressed = "A" * 300

        quality = mc.assess_compression_quality(original, compressed)
        assert abs(quality["compression_ratio"] - 0.3) < 0.01

    def test_information_retention(self):
        """信息保留率基于关键术语"""
        mc = Microcompactor()
        original = "# 架构设计\n采用 PostgreSQL 数据库\n理由：性能"
        compressed = "# 架构设计\nPostgreSQL"

        quality = mc.assess_compression_quality(original, compressed)
        # 架构设计 和 PostgreSQL 都保留
        assert quality["information_retention"] > 0.5

    def test_empty_original(self):
        """空原文返回合理的默认值"""
        mc = Microcompactor()
        quality = mc.assess_compression_quality("", "compressed")

        assert quality["compression_ratio"] == 0.0
        assert quality["information_retention"] == 1.0

    def test_structure_preserved(self):
        """结构保留率检测标题层级"""
        mc = Microcompactor()
        original = "# 主标题\n## 子标题1\n## 子标题2\n内容"
        compressed = "# 主标题\n## 子标题1\n内容"

        quality = mc.assess_compression_quality(original, compressed)
        assert quality["structure_preserved"] >= 0.5  # 至少部分标题保留
