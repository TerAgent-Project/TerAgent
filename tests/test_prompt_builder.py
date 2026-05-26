# tests/test_prompt_builder.py
"""Prompt 构建器单元测试

测试 teragent.pipeline.prompt_builder 模块:
  - 模板填充
  - Token 预算验证
  - build_subagent_prompt 向后兼容
"""
import pytest

from teragent.pipeline.prompt_builder import (
    CHARS_PER_TOKEN,
    DEFAULT_SYSTEM_TEMPLATE,
    DEFAULT_TOKEN_BUDGET,
    build_prompt,
    build_subagent_prompt,
    validate_prompt_tokens,
)


class TestBuildPrompt:
    """build_prompt 模板填充测试"""

    def test_basic_template_filling(self):
        """基本模板填充"""
        template = "你好 {name}，你的任务是 {task}"
        context = {"name": "Agent", "task": "写代码"}
        messages = build_prompt(template, context)
        assert len(messages) >= 1
        assert messages[0]["role"] == "system"
        assert "你好 Agent" in messages[0]["content"]
        assert "写代码" in messages[0]["content"]

    def test_empty_context_becomes_na(self):
        """空值上下文替换为 N/A"""
        template = "设计: {design_md}\n计划: {plan_md}"
        context = {"design_md": "", "plan_md": "有计划"}
        messages = build_prompt(template, context)
        assert "N/A" in messages[0]["content"]
        assert "有计划" in messages[0]["content"]

    def test_task_desc_adds_user_message(self):
        """task_desc 生成用户消息"""
        template = "系统提示"
        context = {"task_desc": "实现功能A"}
        messages = build_prompt(template, context)
        assert len(messages) == 2
        assert messages[1]["role"] == "user"
        assert "实现功能A" in messages[1]["content"]

    def test_no_task_desc_no_user_message(self):
        """无 task_desc 时不生成用户消息"""
        template = "系统提示"
        context = {}
        messages = build_prompt(template, context)
        assert len(messages) == 1
        assert messages[0]["role"] == "system"


class TestBuildSubagentPrompt:
    """build_subagent_prompt 向后兼容测试"""

    def test_default_template(self):
        """默认模板正确填充"""
        messages = build_subagent_prompt(
            design_md="设计文档",
            plan_md="计划文档",
            task_desc="实现功能",
            code_summary="代码摘要",
        )
        assert len(messages) >= 1
        content = messages[0]["content"]
        assert "设计文档" in content
        assert "计划文档" in content
        assert "实现功能" in content

    def test_custom_template(self):
        """自定义模板"""
        custom = "自定义: {design_md} | {task_desc}"
        messages = build_subagent_prompt(
            design_md="D",
            plan_md="P",
            task_desc="T",
            code_summary="C",
            system_template=custom,
        )
        assert "自定义: D" in messages[0]["content"]

    def test_agent_md_included(self):
        """agent_md 记忆内容包含在模板中"""
        messages = build_subagent_prompt(
            design_md="D",
            plan_md="P",
            task_desc="T",
            code_summary="C",
            agent_md="记忆内容",
        )
        assert "记忆内容" in messages[0]["content"]


class TestValidatePromptTokens:
    """validate_prompt_tokens 测试"""

    def test_within_budget_no_error(self):
        """预算内不抛异常"""
        # 小字符数，远在预算内
        validate_prompt_tokens(100, token_budget=120000)

    def test_exceeds_budget_logs_warning(self, caplog):
        """超出预算记录警告"""
        import logging
        # 需要 estimated_tokens > token_budget
        # estimated_tokens = char_count // CHARS_PER_TOKEN
        # 所以 char_count 需要足够大，使得 char_count // 3 > 120000
        # 即 char_count > 360000
        with caplog.at_level(logging.WARNING):
            validate_prompt_tokens(360004, token_budget=120000)
        assert any("EXCEEDS" in r.message for r in caplog.records)

    def test_above_80_percent_logs_warning(self, caplog):
        """超过 80% 预算记录警告"""
        import logging
        # 80% * 120000 = 96000 tokens ≈ 288000 chars
        chars = int(120000 * 0.85) * CHARS_PER_TOKEN
        with caplog.at_level(logging.WARNING):
            validate_prompt_tokens(chars, token_budget=120000)
        assert any("80%" in r.message or "above" in r.message.lower() for r in caplog.records)
