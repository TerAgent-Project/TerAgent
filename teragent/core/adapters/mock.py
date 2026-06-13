"""teragent.core.adapters.mock — Mock TAP Adapter

Local testing adapter that simulates model responses without network I/O.
Used for CI environments, unit tests, and development.

Features:
  - Configurable delay and failure rate
  - Intent-aware mock responses based on CompiledPrompt content
  - Streaming simulation (yields chunks with small delays)
  - Multimodal content detection and metadata reporting
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import AsyncIterator

from teragent.core.adapter import TAPAdapter, TAPAdapterRegistry
from teragent.core.tap import CompiledPrompt, TAPResponse

logger = logging.getLogger(__name__)


class MockAdapter(TAPAdapter):
    """Mock adapter for TAP — simulates model responses for local testing

    No network I/O is performed. Responses are generated based on the
    intent detected in the CompiledPrompt content.

    Enhanced for multimodal support:
      - Detects multimodal content in message content arrays
      - Reports multimodal_detected flag in response metadata
      - Includes cache_hit_tokens in mock responses when relevant
      - Intent detection includes multimodal-related keywords

    Args:
        delay: Simulated response latency in seconds (default 0.1)
        fail_rate: Probability of simulated failure, 0.0–1.0 (default 0.0)
    """

    def __init__(
        self,
        delay: float = 0.1,
        fail_rate: float = 0.0,
    ) -> None:
        self.delay = delay
        self.fail_rate = fail_rate
        self._call_count = 0

        logger.info(
            f"MockAdapter: delay={self.delay}s, fail_rate={self.fail_rate}"
        )

    # ----- intent detection -----

    @staticmethod
    def _extract_text_from_content(content) -> str:
        """Extract text from message content, handling both string and list formats.

        OpenAI API messages can have content as:
        - str: "Hello"
        - list[dict]: [{"type": "text", "text": "Hello"}, {"type": "image_url", ...}]

        Args:
            content: Message content (str or list)

        Returns:
            Extracted text content
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            return " ".join(text_parts)
        return str(content) if content else ""

    @staticmethod
    def _detect_multimodal_in_content(content) -> bool:
        """检测消息内容中是否包含多模态类型

        检查 content 数组中是否包含 image_url、video_url 等多模态类型。

        Args:
            content: 消息内容（str 或 list）

        Returns:
            是否包含多模态内容
        """
        if not isinstance(content, list):
            return False
        for part in content:
            if isinstance(part, dict) and part.get("type") in (
                "image_url", "video_url"
            ):
                return True
        return False

    @staticmethod
    def _count_multimodal_content(content) -> dict:
        """统计消息内容中的多模态类型数量

        Args:
            content: 消息内容（str 或 list）

        Returns:
            各类型数量统计字典，如 {"image_url": 2, "video_url": 1}
        """
        counts: dict[str, int] = {}
        if not isinstance(content, list):
            return counts
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type", "")
                if part_type in ("image_url", "video_url", "image_base64"):
                    counts[part_type] = counts.get(part_type, 0) + 1
        return counts

    @staticmethod
    def _detect_intent(compiled: CompiledPrompt) -> str:
        """Detect the intent from CompiledPrompt content for mock response selection.

        Inspects Mode A messages or Mode B system_prompt/user_message for
        intent keywords. Falls back to "code_generation".

        Handles both string content and OpenAI content array format
        (used by multimodal compilers like MiniMaxM3Compiler).

        Also checks for multimodal-related keywords in content arrays.
        """
        # Collect all text content from the compiled prompt
        all_text_parts: list[str] = []
        has_multimodal = False

        # Mode A: messages
        if compiled.messages:
            for msg in compiled.messages:
                content = msg.get("content", "")
                all_text_parts.append(MockAdapter._extract_text_from_content(content))
                # 检查多模态内容
                if MockAdapter._detect_multimodal_in_content(content):
                    has_multimodal = True

        # Mode B: system_prompt + user_message
        if compiled.system_prompt:
            all_text_parts.append(compiled.system_prompt)
        if compiled.user_message:
            all_text_parts.append(compiled.user_message)

        all_content = " ".join(all_text_parts)

        # 多模态相关关键词检测
        multimodal_keywords = ["截图", "UI布局", "设计稿", "视觉风格", "桌面操作",
                              "多模态", "图片分析", "视频分析"]
        if has_multimodal or any(kw in all_content for kw in multimodal_keywords):
            # 检查多模态相关意图
            if "设计稿" in all_content or "对比" in all_content:
                return "multimodal_review"
            if "设计" in all_content or "视觉" in all_content:
                return "multimodal_design"

        # Match intents — check more specific patterns first
        if (
            "生成 PLAN" in all_content
            or "拆解" in all_content
            or "PLAN.md" in all_content
        ):
            return "plan"
        if "APPROVE" in all_content or "审核" in all_content or "审查" in all_content:
            return "review"
        if "核对" in all_content or "checklist" in all_content.lower():
            return "checklist"
        if (
            "需求" in all_content
            or "DESIGN" in all_content
            or "设计" in all_content
        ):
            return "design"
        return "code_generation"

    @staticmethod
    def _get_mock_response(intent: str, task_id: str = "unknown") -> str:
        """Return a mock response based on the detected intent."""
        mock_responses: dict[str, str] = {
            "multimodal_design": (
                "# 多模态设计方案\n"
                "## 1. 视觉风格分析\n"
                "基于截图分析，识别到以下设计特征：\n"
                "- 色彩方案: 主色调 + 辅助色\n"
                "- 排版风格: 居中对齐，卡片式布局\n"
                "## 2. 新方案设计\n"
                "```html\n"
                '<div class="card">Mock Design</div>\n'
                "```\n"
            ),
            "multimodal_review": (
                "# 多模态审查报告\n"
                "## 一致性检查结果\n"
                "| 项目 | 状态 | 说明 |\n"
                "|------|------|------|\n"
                "| 布局 | ✅ | 一致 |\n"
                "| 色彩 | ✅ | 一致 |\n"
                "| 文字 | ⚠️ | 字号偏差 |\n"
                "## 建议修改\n"
                "1. 调整标题字号为 16px\n"
            ),
            "design": (
                "# DESIGN\n"
                "## 1. Background & Motivation\n"
                "Mock design document.\n"
                "## 2. Design Goals\n"
                "| Goal | Description |\n"
                "|------|-------------|\n"
                "| Usability | System works correctly |\n"
                "## 3. Tech Stack\n"
                "Python + asyncio\n"
                "## 4. Core Interface Contract\n"
                "```python\n"
                "class MockSystem:\n"
                "    async def run(self) -> None: ...\n"
                "```"
            ),
            "plan": (
                "### 1.1 Implement core module\n"
                "- **Prerequisites**: None\n"
                "- **Output files**: mock_module.py\n"
                "- **Priority**: Required\n\n"
                "### 1.2 Implement tests\n"
                "- **Prerequisites**: 1.1\n"
                "- **Output files**: test_mock.py\n"
                "- **Priority**: Optional"
            ),
            "review": "APPROVE",
            "checklist": (
                "- [x] 1.1 Implement core module\n"
                "- [ ] 1.2 Implement tests"
            ),
            "code_generation": (
                f'<file path="mock_output.py">\n'
                f"# Auto-generated by MockAdapter\n"
                f"def main() -> None:\n"
                f'    """Mock implementation for task: {task_id}"""\n'
                f"    pass\n"
                f"</file>"
            ),
        }
        return mock_responses.get(intent, mock_responses["code_generation"])

    # ----- core send -----

    async def send(self, compiled: CompiledPrompt, model: str) -> TAPResponse:
        """Send a compiled prompt to the mock adapter.

        Simulates delay and possible failure. Returns a mock TAPResponse
        based on the detected intent in the prompt content.

        Enhanced for multimodal support:
          - Detects multimodal content and reports in metadata
          - Includes cache_hit_tokens in mock responses when relevant
          - Multimodal intent detection
        """
        self._call_count += 1
        await asyncio.sleep(self.delay)

        if random.random() < self.fail_rate:
            raise RuntimeError("MockAdapter simulated failure")

        intent = self._detect_intent(compiled)

        # 检测多模态内容
        has_multimodal = False
        multimodal_counts: dict[str, int] = {}
        if compiled.messages:
            for msg in compiled.messages:
                content = msg.get("content", "")
                if self._detect_multimodal_in_content(content):
                    has_multimodal = True
                # 统计多模态内容类型
                msg_counts = self._count_multimodal_content(content)
                for k, v in msg_counts.items():
                    multimodal_counts[k] = multimodal_counts.get(k, 0) + v

        # Check messages for task_id keyword (Mode A); use a placeholder if found
        task_id = "unknown"
        if compiled.messages:
            for msg in compiled.messages:
                content_text = self._extract_text_from_content(msg.get("content", ""))
                # Placeholder: real extraction would parse the actual task_id value
                if "task_id" in content_text:
                    task_id = "mock_task"

        raw_text = self._get_mock_response(intent, task_id)

        # Simulate token usage estimation
        prompt_text = ""
        if compiled.messages:
            prompt_text = " ".join(
                self._extract_text_from_content(msg.get("content", ""))
                for msg in compiled.messages
            )
        else:
            prompt_text = (
                f"{compiled.system_prompt} {compiled.user_message}"
            )

        prompt_tokens = max(1, len(prompt_text) // 4)
        # 多模态内容增加额外的 token 估算
        if has_multimodal:
            for mm_type, count in multimodal_counts.items():
                if mm_type in ("image_url", "image_base64"):
                    prompt_tokens += count * 1000
                elif mm_type == "video_url":
                    prompt_tokens += count * 3000
        completion_tokens = max(1, len(raw_text) // 4)

        # 构建模拟 usage 字段
        usage: dict = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

        # 对于 V4/GLM/M3 模型，包含 cache_hit_tokens 字段
        # 模拟 60-80% 的缓存命中率（长上下文场景下常见）
        is_cache_model = any(
            name in model.lower()
            for name in ("deepseek", "glm", "minimax", "v4", "m3")
        )
        cache_hit_tokens = 0
        if is_cache_model and prompt_tokens > 100:
            # 模拟缓存命中（系统提示和上下文部分通常被缓存）
            cache_rate = random.uniform(0.6, 0.8)
            cache_hit_tokens = int(prompt_tokens * cache_rate)
            usage["prompt_cache_hit_tokens"] = cache_hit_tokens

        logger.debug(
            f"MockAdapter call #{self._call_count}: intent={intent} "
            f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} "
            f"multimodal={has_multimodal} mm_counts={multimodal_counts} "
            f"cache_hit_tokens={cache_hit_tokens}"
        )

        return TAPResponse(
            raw_text=raw_text,
            usage=usage,
            finish_reason="stop",
            cache_hit_tokens=cache_hit_tokens,
        )

    # ----- core stream -----

    async def stream(self, compiled: CompiledPrompt, model: str) -> AsyncIterator[str]:
        """Stream a compiled prompt from the mock adapter.

        Yields chunks of the mock response with small delays to simulate
        streaming behaviour.
        """
        response = await self.send(compiled, model)
        chunk_size = 10
        text = response.raw_text or ""
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]
            await asyncio.sleep(0.02)

    # ----- capabilities -----

    @property
    def capabilities(self) -> dict:
        return {
            "streaming": True,
            "tool_calling": False,
            "max_context_tokens": 100000,
            "mock": True,
            "multimodal": True,  # 支持多模态内容检测
        }


# Register with TAPAdapterRegistry
TAPAdapterRegistry.register("mock", MockAdapter)
