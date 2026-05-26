# tests/test_intent_classifier.py
"""意图分类器单元测试

测试 teragent.intent.classifier 模块:
  - IntentType 枚举
  - 三层分类（确定性规则 → 启发式规则 → LLM 回退）
  - 默认回退到 CREATE_PROJECT
  - 分类统计
"""
import pytest

from teragent.intent.classifier import IntentClassifier, IntentType


class TestIntentType:
    """IntentType 枚举测试"""

    def test_enum_values(self):
        """IntentType 有 3 种意图"""
        assert IntentType.CHAT.value == "chat"
        assert IntentType.DEBUG.value == "debug"
        assert IntentType.CREATE_PROJECT.value == "create_project"
        assert len(IntentType) == 3

    def test_enum_comparison(self):
        """IntentType 可以进行比较"""
        assert IntentType.CHAT == IntentType.CHAT
        assert IntentType.CHAT != IntentType.DEBUG


class TestLayer1Deterministic:
    """第一层：确定性规则测试"""

    @pytest.mark.asyncio
    async def test_chat_greeting_hello(self):
        """简单问候 → CHAT"""
        classifier = IntentClassifier()
        result = await classifier.classify("hello")
        assert result == IntentType.CHAT

    @pytest.mark.asyncio
    async def test_chat_greeting_chinese(self):
        """中文问候 → CHAT"""
        classifier = IntentClassifier()
        result = await classifier.classify("你好")
        assert result == IntentType.CHAT

    @pytest.mark.asyncio
    async def test_chat_thanks(self):
        """感谢 → CHAT"""
        classifier = IntentClassifier()
        result = await classifier.classify("thanks")
        assert result == IntentType.CHAT

    @pytest.mark.asyncio
    async def test_create_project_keyword(self):
        """创建项目关键词 → CREATE_PROJECT"""
        classifier = IntentClassifier()
        result = await classifier.classify("创建一个 Python 爬虫")
        assert result == IntentType.CREATE_PROJECT

    @pytest.mark.asyncio
    async def test_create_project_english(self):
        """英文创建关键词 → CREATE_PROJECT"""
        classifier = IntentClassifier()
        result = await classifier.classify("build a web scraper")
        assert result == IntentType.CREATE_PROJECT

    @pytest.mark.asyncio
    async def test_debug_keyword(self):
        """调试关键词 → DEBUG"""
        classifier = IntentClassifier()
        result = await classifier.classify("修复这段代码的 bug")
        assert result == IntentType.DEBUG

    @pytest.mark.asyncio
    async def test_debug_english(self):
        """英文调试关键词 → DEBUG"""
        classifier = IntentClassifier()
        result = await classifier.classify("fix the error in my code")
        assert result == IntentType.DEBUG

    @pytest.mark.asyncio
    async def test_both_keywords_prefers_debug(self):
        """同时有创建和调试关键词 → 优先 DEBUG（安全优先）"""
        classifier = IntentClassifier()
        result = await classifier.classify("创建并修复这个功能")
        assert result == IntentType.DEBUG


class TestLayer2Heuristic:
    """第二层：启发式规则测试"""

    @pytest.mark.asyncio
    async def test_short_non_code_is_chat(self):
        """短文本无代码标记 → CHAT"""
        classifier = IntentClassifier()
        result = await classifier.classify("天气真好啊")
        assert result == IntentType.CHAT

    @pytest.mark.asyncio
    async def test_problem_pattern_is_debug(self):
        """问题描述模式 → DEBUG（输入需 >= 15 字符以避免短文本 CHAT 规则）"""
        classifier = IntentClassifier()
        result = await classifier.classify("为什么这个程序突然不工作了，请帮我看看")
        assert result == IntentType.DEBUG

    @pytest.mark.asyncio
    async def test_error_traceback_is_debug(self):
        """包含 Traceback/Error → DEBUG"""
        classifier = IntentClassifier()
        result = await classifier.classify("运行时出现 Traceback Error: invalid syntax")
        assert result == IntentType.DEBUG

    @pytest.mark.asyncio
    async def test_long_text_is_create(self):
        """长文本描述 → CREATE_PROJECT"""
        classifier = IntentClassifier()
        long_text = "我需要一个完整的电商系统" + "，包含用户注册登录商品展示购物车订单管理" * 5
        result = await classifier.classify(long_text)
        assert result == IntentType.CREATE_PROJECT


class TestLayer3LLMFallback:
    """第三层：LLM 回退测试"""

    @pytest.mark.asyncio
    async def test_no_model_defaults_create(self):
        """无模型时默认回退到 CREATE_PROJECT"""
        classifier = IntentClassifier(model=None)
        # 用一个既不匹配 L1 也不匹配 L2 的输入：15-80字符，无关键词/代码标记/问题模式
        result = await classifier.classify("请分析一下这段文字的深层含义和作者想要表达的观点")
        assert result == IntentType.CREATE_PROJECT

    @pytest.mark.asyncio
    async def test_llm_classify_chat(self):
        """LLM 返回 chat → CHAT"""
        from unittest.mock import AsyncMock

        mock_model = AsyncMock()
        mock_model.chat = AsyncMock(return_value={"content": "chat"})

        classifier = IntentClassifier(model=mock_model)
        # 输入需 >= 15 字符且无关键词/代码标记/问题模式，才能落入 L3
        result = await classifier.classify("请分析一下这段文字的深层含义和作者想要表达的观点")
        assert result == IntentType.CHAT
        assert mock_model.chat.called

    @pytest.mark.asyncio
    async def test_llm_classify_debug(self):
        """LLM 返回 debug → DEBUG"""
        from unittest.mock import AsyncMock

        mock_model = AsyncMock()
        mock_model.chat = AsyncMock(return_value={"content": "debug"})

        classifier = IntentClassifier(model=mock_model)
        result = await classifier.classify("请分析一下这段文字的深层含义和作者想要表达的观点")
        assert result == IntentType.DEBUG

    @pytest.mark.asyncio
    async def test_llm_classify_create(self):
        """LLM 返回 create_project → CREATE_PROJECT"""
        from unittest.mock import AsyncMock

        mock_model = AsyncMock()
        mock_model.chat = AsyncMock(return_value={"content": "create_project"})

        classifier = IntentClassifier(model=mock_model)
        result = await classifier.classify("做一个全新的东西")
        assert result == IntentType.CREATE_PROJECT

    @pytest.mark.asyncio
    async def test_llm_failure_defaults_create(self):
        """LLM 调用失败时回退到 CREATE_PROJECT"""
        from unittest.mock import AsyncMock

        mock_model = AsyncMock()
        mock_model.chat = AsyncMock(side_effect=RuntimeError("API down"))

        classifier = IntentClassifier(model=mock_model)
        result = await classifier.classify("请分析一下这段文字的深层含义和作者想要表达的观点")
        assert result == IntentType.CREATE_PROJECT


class TestClassificationStats:
    """分类统计测试"""

    @pytest.mark.asyncio
    async def test_stats_total_count(self):
        """统计记录总分类次数"""
        classifier = IntentClassifier()
        await classifier.classify("hello")
        await classifier.classify("fix bug")
        stats = classifier.get_stats()
        assert stats["total_classifications"] == 2

    @pytest.mark.asyncio
    async def test_stats_llm_count(self):
        """统计记录 LLM 分类次数"""
        from unittest.mock import AsyncMock

        mock_model = AsyncMock()
        mock_model.chat = AsyncMock(return_value={"content": "chat"})

        classifier = IntentClassifier(model=mock_model)
        # L1 会命中，不走 LLM
        await classifier.classify("hello")
        # 模糊输入走 LLM（>= 15 字符，无关键词/代码标记/问题模式）
        await classifier.classify("请分析一下这段文字的深层含义和作者想要表达的观点")
        stats = classifier.get_stats()
        assert stats["llm_classifications"] == 1
        assert stats["llm_ratio"] > 0

    @pytest.mark.asyncio
    async def test_stats_zero_ratio_when_no_classifications(self):
        """无分类时 llm_ratio 为 0"""
        classifier = IntentClassifier()
        stats = classifier.get_stats()
        assert stats["total_classifications"] == 0
        assert stats["llm_ratio"] == 0.0
