# teragent/intent/classifier.py
"""多层漏斗意图分类器

三层分类策略:
  1. 确定性规则（零成本）：正则匹配问候语、强工程动词
  2. 启发式规则（零成本）：输入长度 + 标点模式 + 语言特征
  3. LLM 分类（低成本的轻量模型调用）：仅模糊输入触发

分类结果:
  - CHAT: 简单对话 / 知识问答 / 问候
  - DEBUG: 调试 / 修复已有项目 / 分析错误
  - CREATE_PROJECT: 创建新项目 / 从零实现功能
"""
import logging
import re
from enum import Enum
from typing import TYPE_CHECKING

__all__ = [
    "IntentClassifier",
    "IntentType",
]

if TYPE_CHECKING:
    from teragent.core.provider import ModelProvider

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """用户意图类型"""

    CHAT = "chat"                       # 简单对话，不触发任何工具
    DEBUG = "debug"                     # 调试/修改已有项目
    CREATE_PROJECT = "create_project"   # 创建新项目（走完整流水线）


# ===== 第一层：确定性规则（零成本）=====

# 简单问候 / 非编程意图 — 整行匹配
_CHAT_PATTERN = re.compile(
    r"^[\s]*("
    r"你好|hi|hello|hey|嗨|哈喽|谢谢|拜拜|再见|ok|好的|嗯|yes|no|"
    r"help|帮助|version|版本|什么是|解释|explain|who|what is|"
    r"测试|test|天呐|哇|哈哈|"
    r"cool|nice|great|thanks|thx|bye"
    r")[\s!！.。?？~～]*$",
    re.IGNORECASE,
)

# 强工程动词 — 创建项目
_CREATE_KEYWORDS = [
    # 中文
    "创建", "新建", "从零", "搭建", "重构", "实现一个", "写一个", "做一个",
    "开发", "构建", "写个", "做个", "实现一个", "帮我写", "帮我做", "帮我创建",
    "我要一个", "我要一个", "生成一个", "设计一个", "编一个",
    # 英文
    "create", "build", "implement", "develop", "write a", "make a",
    "scaffold", "generate a", "set up a",
]

# 调试 / 修复动词
_DEBUG_KEYWORDS = [
    # 中文
    "修复", "调试", "debug", "fix", "改bug", "报错", "报异常", "异常",
    "为什么不工作", "运行失败", "报错信息", "出错", "崩溃", "crash",
    "修改", "补丁", "patch", "排查", "定位问题", "修一下", "帮我改",
    "修正", "调试一下", "看看问题",
    # 英文
    "fix", "debug", "troubleshoot", "patch", "repair",
    "broken", "crashing", "error in", "bug in", "issue with",
    "not working", "doesn't work", "fails to",
]

# ===== 第二层：启发式规则（零成本）=====

# 长输入 (>50 chars) 且包含代码标记 → 多半是 DEBUG 或 CREATE
_CODE_INDICATORS = re.compile(
    r"(```|`[a-z]+|def |class |import |from \w+|function |const |var |Traceback|Error:|Exception)",
    re.IGNORECASE,
)

# 问题描述模式 → DEBUG
_PROBLEM_PATTERNS = re.compile(
    r"(为什么|why does|why is|how to fix|怎么解决|如何修复|怎么办|出了什么|什么原因|"
    r"报了|报了.*错|运行.*报|启动.*失败|无法.*启动|cannot|doesn't|won't|shouldn't)",
    re.IGNORECASE,
)


class IntentClassifier:
    """多层漏斗意图分类器

    使用方式::

        classifier = IntentClassifier(model=provider)  # 可选传入 model 用于 LLM 分类
        intent = await classifier.classify("创建一个 Python 爬虫")
        # → IntentType.CREATE_PROJECT
    """

    def __init__(self, model: "ModelProvider | None" = None) -> None:
        self.model = model
        self._classification_count: int = 0
        self._llm_classification_count: int = 0

    async def classify(self, user_input: str) -> IntentType:
        """分类用户输入的意图

        三层漏斗:
          1. 确定性规则（零成本）
          2. 启发式规则（零成本）
          3. LLM 分类（仅模糊输入触发，低成本）

        Args:
            user_input: 用户的原始输入文本

        Returns:
            IntentType 枚举值
        """
        text = user_input.strip()
        self._classification_count += 1

        # -- 第一层：确定性规则 --
        intent = self._classify_deterministic(text)
        if intent is not None:
            logger.debug(f"Intent (L1 deterministic): {intent.value} for: {text[:50]}")
            return intent

        # -- 第二层：启发式规则 --
        intent = self._classify_heuristic(text)
        if intent is not None:
            logger.debug(f"Intent (L2 heuristic): {intent.value} for: {text[:50]}")
            return intent

        # -- 第三层：LLM 分类（仅模糊输入）--
        if self.model:
            intent = await self._llm_classify(text)
            logger.debug(f"Intent (L3 LLM): {intent.value} for: {text[:50]}")
            return intent

        # 无模型时默认 CREATE_PROJECT（向后兼容：走原有流水线）
        logger.debug(f"Intent (default): CREATE_PROJECT for: {text[:50]}")
        return IntentType.CREATE_PROJECT

    def _classify_deterministic(self, text: str) -> "IntentType | None":
        """第一层：确定性规则，零成本"""

        # 简单问候 / 非编程意图 → CHAT
        if _CHAT_PATTERN.match(text):
            return IntentType.CHAT

        # 统计关键词命中
        has_create_kw = any(kw in text for kw in _CREATE_KEYWORDS)
        has_debug_kw = any(kw in text for kw in _DEBUG_KEYWORDS)

        if has_create_kw and not has_debug_kw:
            return IntentType.CREATE_PROJECT

        if has_debug_kw and not has_create_kw:
            return IntentType.DEBUG

        # 两者都有 → 优先 DEBUG（安全优先：先修后建）
        if has_create_kw and has_debug_kw:
            return IntentType.DEBUG

        return None

    def _classify_heuristic(self, text: str) -> "IntentType | None":
        """第二层：启发式规则，零成本

        根据输入长度、代码标记、问题模式等判断意图。
        """

        # 很短的输入 (<15 chars) 且不含代码标记 → 多半是 CHAT
        if len(text) < 15 and not _CODE_INDICATORS.search(text):
            return IntentType.CHAT

        # 包含问题描述模式 → DEBUG
        if _PROBLEM_PATTERNS.search(text):
            return IntentType.DEBUG

        # 包含代码标记 → 视上下文判断
        if _CODE_INDICATORS.search(text):
            # 如果包含 Traceback/Error → DEBUG
            if re.search(r"(Traceback|Error:|Exception|异常|报错)", text, re.IGNORECASE):
                return IntentType.DEBUG
            # 如果有 "创建/写" 动词 + 代码标记 → CREATE_PROJECT
            if any(kw in text for kw in _CREATE_KEYWORDS):
                return IntentType.CREATE_PROJECT
            # 否则，代码 + 描述多半是 DEBUG
            return IntentType.DEBUG

        # 较长的纯文本描述 → 倾向 CREATE_PROJECT
        if len(text) > 80:
            return IntentType.CREATE_PROJECT

        return None

    async def _llm_classify(self, text: str) -> IntentType:
        """第三层：LLM 分类，仅模糊输入触发

        使用轻量模型（当前配置的 execute_model）进行意图判断。
        Prompt 设计为极简，最小化 Token 消耗。
        """
        self._llm_classification_count += 1
        prompt = (
            "判断以下用户输入的意图，只回复一个词：chat、debug 或 create_project\n\n"
            "- chat：简单对话、问候、知识问答、闲聊\n"
            "- debug：修复bug、调试已有项目、分析错误、修改代码\n"
            "- create_project：创建新项目、从零实现功能、搭建系统\n\n"
            f"用户输入：{text}\n\n意图："
        )
        try:
            response = await self.model.chat(
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.get("content", "").strip().lower()

            if "debug" in result:
                return IntentType.DEBUG
            if "chat" in result:
                return IntentType.CHAT
            if "create" in result:
                return IntentType.CREATE_PROJECT

            # LLM 返回了无法解析的结果 → 默认 CREATE_PROJECT
            return IntentType.CREATE_PROJECT
        except Exception as e:
            logger.warning(
                f"LLM intent classification failed: {e}, "
                f"defaulting to CREATE_PROJECT"
            )
            return IntentType.CREATE_PROJECT

    def get_stats(self) -> dict:
        """返回分类器统计信息"""
        return {
            "total_classifications": self._classification_count,
            "llm_classifications": self._llm_classification_count,
            "llm_ratio": (
                self._llm_classification_count / self._classification_count
                if self._classification_count > 0
                else 0.0
            ),
        }
