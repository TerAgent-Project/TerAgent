"""teragent.core.compilers.glm_5v_turbo — GLM5VTurboCompiler

GLM-5V-Turbo 专属编译器，用于视觉理解任务。

核心定位：
  - 视觉模型编译器，专注于图像/视频/设计稿理解
  - 与 GLM52Compiler 配合实现"视觉→编码"协同工作流
  - 原生多模态支持（supports_multimodal = True）
  - 不需要复杂的上下文管理（视觉理解通常是单轮任务）

与 GLM5Compiler 的区别：
  - supports_multimodal: True（GLM5 = False）
  - 无长程任务支持
  - 无思考模式路由（5V-Turbo 是视觉模型，不需要）
  - 上下文窗口: 128K（不需要 1M）
  - 编译策略: 保留多模态内容 + 视觉分析引导 prompt

设计参考：design.md §6.2.5 与 GLM-5V-Turbo 多模态协同
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from teragent.core.compiler import TAPCompiler, TAPCompilerRegistry
from teragent.core.tap import CompiledPrompt, MultimodalContent, TAPRequest

logger = logging.getLogger(__name__)


# ===== 视觉分析 Prompt 模板 =====

VISION_ANALYSIS_SYSTEM_PROMPT = """\
你是一个专业的视觉分析 AI 助手，基于 GLM-5V-Turbo 视觉模型。
你的核心能力是理解和分析图像、设计稿、截图等视觉内容。

分析时请遵循以下结构化输出格式：

## 视觉语义分析

### 1. 整体描述
[对图像/设计稿的整体描述]

### 2. 布局结构
[页面/界面布局结构，包括区域划分、层级关系]
- 使用结构化的层级描述（如：顶部导航栏、左侧边栏、主内容区等）

### 3. UI 元素清单
[逐一列出可识别的 UI 元素]
| # | 类型 | 标签/文本 | 位置描述 | 样式特征 |
|---|------|----------|----------|----------|
| 1 | 按钮 | "提交" | 表单底部 | 蓝色圆角按钮 |

### 4. 颜色方案
[主要颜色和配色方案]
- 主色:
- 辅助色:
- 背景色:
- 文字色:

### 5. 交互逻辑
[从视觉线索推断的交互流程和逻辑]

### 6. 响应式线索
[从设计稿中推断的响应式设计线索，如有]

### 7. 代码实现建议
[基于视觉分析的技术实现建议]
- 推荐框架/库:
- 关键组件:
- 布局策略:
"""

VISION_VERIFICATION_SYSTEM_PROMPT = """\
你是一个专业的视觉验证 AI 助手，基于 GLM-5V-Turbo 视觉模型。
你的任务是对比原始设计稿和生成的代码实现，评估一致性。

请按以下维度评估：

## 视觉一致性验证

### 1. 布局一致性 (0-10)
[布局结构是否与设计稿匹配]

### 2. 颜色一致性 (0-10)
[配色方案是否与设计稿匹配]

### 3. 元素完整性 (0-10)
[所有设计稿中的元素是否都在代码中实现]

### 4. 交互逻辑一致性 (0-10)
[交互行为是否与设计稿暗示的逻辑一致]

### 5. 总体评分 (0-10)
[综合评估]

### 6. 差异清单
[列出所有不一致之处]
- 差异1: ...
- 差异2: ...

### 7. 修复建议
[针对差异的修复建议]
"""

VISION_CODE_GENERATION_HINT = """\
基于上述视觉分析结果，请生成实现该设计的前端代码。
要求：
1. 严格按照视觉分析中的布局结构实现
2. 使用分析中识别的颜色方案
3. 实现所有列出的 UI 元素
4. 遵循交互逻辑描述
5. 考虑响应式设计线索
"""


@dataclass
class VisionAnalysisResult:
    """GLM-5V-Turbo 视觉分析结果

    封装视觉模型的输出，为后续 GLM-5.2 编码提供结构化上下文。

    Attributes:
        raw_analysis: 视觉模型的原始分析文本
        structured_context: 提取的结构化上下文（用于注入 GLM-5.2 prompt）
        has_layout_info: 是否包含布局结构信息
        has_color_scheme: 是否包含颜色方案信息
        has_interaction_logic: 是否包含交互逻辑信息
        element_count: 识别到的 UI 元素数量
        confidence: 分析置信度 (0.0-1.0)
        source_image_type: 源图像类型描述
    """

    raw_analysis: str = ""
    structured_context: str = ""
    has_layout_info: bool = False
    has_color_scheme: bool = False
    has_interaction_logic: bool = False
    element_count: int = 0
    confidence: float = 0.0
    source_image_type: str = "unknown"

    def to_context_string(self) -> str:
        """转换为可注入 GLM-5.2 prompt 的上下文字符串"""
        if self.structured_context:
            return self.structured_context
        if self.raw_analysis:
            return f"<visual_analysis>\n{self.raw_analysis}\n</visual_analysis>"
        return ""

    @property
    def quality_score(self) -> float:
        """分析质量评分 (0.0-1.0)

        基于结构化信息的丰富度评估。
        """
        score = 0.0
        if self.has_layout_info:
            score += 0.25
        if self.has_color_scheme:
            score += 0.25
        if self.has_interaction_logic:
            score += 0.25
        if self.element_count > 0:
            score += min(0.25, self.element_count * 0.025)
        return min(1.0, score)


class GLM5VTurboCompiler(TAPCompiler):
    """GLM-5V-Turbo 视觉模型编译器

    专为视觉理解任务设计的编译器，核心功能：
    1. 原生多模态支持：保留 image/video 内容块，不做降级
    2. 视觉分析引导：注入结构化分析 prompt，引导模型输出结构化结果
    3. 验证模式：支持视觉验证 prompt，用于协同工作流的验证环节
    4. 简洁编译：视觉理解通常是单轮任务，不需要复杂的上下文管理

    使用场景：
    - 设计稿分析：提取布局、颜色、UI 元素信息
    - 截图理解：理解桌面/应用截图内容
    - 视觉验证：对比设计稿与实现的一致性
    - 与 GLM52Compiler 协同：视觉→编码工作流

    设计参考：design.md §6.2.5
    """

    # 编译模式
    CompileMode = Literal["analysis", "verification", "default"]

    def __init__(
        self,
        mode: CompileMode = "default",
        max_context_tokens: int = 128_000,
    ) -> None:
        """初始化 GLM-5V-Turbo 编译器

        Args:
            mode: 编译模式
                - "analysis": 视觉分析模式，注入结构化分析 prompt
                - "verification": 视觉验证模式，注入一致性验证 prompt
                - "default": 默认模式，使用通用视觉 prompt
            max_context_tokens: 最大上下文 token 数
        """
        self._mode = mode
        self._max_context_tokens = max_context_tokens

    # ----- Capability properties -----

    @property
    def supports_multimodal(self) -> bool:
        """GLM-5V-Turbo 原生支持多模态"""
        return True

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    @property
    def mode(self) -> str:
        """当前编译模式"""
        return self._mode

    # ----- Compiler type -----

    def _get_compiler_type(self) -> str:
        return "glm_5v_turbo"

    # ----- Core compile -----

    def compile(self, request: TAPRequest) -> CompiledPrompt:
        """编译 TAP 请求为 GLM-5V-Turbo 特定的 prompt

        编译策略：
        1. 根据编译模式选择系统 prompt
        2. 保留多模态内容块（不做降级）
        3. 构建用户消息：指令 + 多模态内容 + 约束
        4. 使用 Mode A（messages 格式），与 OpenAI 兼容接口一致

        Args:
            request: TAP 请求

        Returns:
            编译后的 CompiledPrompt
        """
        # 1. 选择系统 prompt
        system_prompt = self._select_system_prompt(request)

        # 2. 构建消息列表
        messages: list[dict] = []

        # 系统消息
        messages.append({"role": "system", "content": system_prompt})

        # 3. 注入上下文（设计文档、计划等）
        messages = self._inject_context(messages, request)

        # 4. 构建用户消息（包含多模态内容）
        user_content = self._build_user_content(request)
        messages.append({"role": "user", "content": user_content})

        # 5. 构建 CompiledPrompt
        compiled = CompiledPrompt(
            messages=messages,
            max_tokens=4096,  # 视觉分析输出通常不需要太长
            extra={
                "compiler": "glm_5v_turbo",
                "compile_mode": self._mode,
                "has_multimodal": request.has_multimodal,
            },
        )

        return compiled

    # ----- System prompt selection -----

    def _select_system_prompt(self, request: TAPRequest) -> str:
        """根据编译模式和请求意图选择系统 prompt

        Args:
            request: TAP 请求

        Returns:
            系统 prompt 字符串
        """
        # 优先使用 prompt 注册表中的 prompt
        intent = request.meta.get("intent", "execute")
        registered_prompt = self.get_system_prompt(intent)
        if registered_prompt:
            return registered_prompt

        # 回退到内置 prompt
        if self._mode == "analysis":
            return VISION_ANALYSIS_SYSTEM_PROMPT
        elif self._mode == "verification":
            return VISION_VERIFICATION_SYSTEM_PROMPT
        else:
            # 默认模式：基于意图选择
            if intent in ("design", "review"):
                return VISION_ANALYSIS_SYSTEM_PROMPT
            return VISION_ANALYSIS_SYSTEM_PROMPT

    # ----- User content building -----

    def _build_user_content(self, request: TAPRequest) -> list[dict] | str:
        """构建用户消息内容（支持多模态混合内容）

        GLM-5V-Turbo 支持 OpenAI 格式的多模态内容块：
        [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {...}}]

        Args:
            request: TAP 请求

        Returns:
            混合内容列表（含多模态）或纯文本字符串
        """
        content_parts: list[dict] = []

        # 添加指令文本
        if request.instruction:
            content_parts.append({"type": "text", "text": request.instruction})

        # 添加多模态内容
        if request.has_multimodal:
            for mc in request.multimodal_context:
                content_parts.append(mc.to_openai_format())

        # 添加约束
        if request.constraints:
            constraints_text = "\n".join(
                f"- {c}" for c in request.constraints
            )
            content_parts.append({
                "type": "text",
                "text": f"\n约束条件:\n{constraints_text}",
            })

        # 添加输出格式提示
        if request.output_format_hint:
            content_parts.append({
                "type": "text",
                "text": f"\n输出格式: {request.output_format_hint}",
            })

        # 如果只有文本内容，返回字符串（兼容性更好）
        if len(content_parts) == 1 and content_parts[0].get("type") == "text":
            return content_parts[0]["text"]

        # 如果没有任何内容，返回空字符串
        if not content_parts:
            return request.instruction or "请分析提供的图像。"

        return content_parts

    # ----- Analysis result parsing -----

    @staticmethod
    def parse_analysis_result(response_text: str) -> VisionAnalysisResult:
        """解析 GLM-5V-Turbo 的视觉分析响应

        从模型输出中提取结构化信息，生成 VisionAnalysisResult。

        Args:
            response_text: GLM-5V-Turbo 的响应文本

        Returns:
            VisionAnalysisResult 包含结构化分析结果
        """
        text = response_text or ""
        text_lower = text.lower()

        # 检测结构化信息
        has_layout = any(
            kw in text_lower
            for kw in ["布局", "layout", "结构", "structure", "区域", "层级", "导航", "侧边栏"]
        )
        has_color = any(
            kw in text_lower
            for kw in ["颜色", "color", "配色", "色彩", "主题色", "主色", "背景色"]
        )
        has_interaction = any(
            kw in text_lower
            for kw in ["交互", "interaction", "点击", "click", "导航", "跳转", "表单", "提交"]
        )

        # 估算 UI 元素数量（通过表格行或列表项计数）
        element_count = 0
        # 统计表格行（| 开头的行）
        table_rows = [line for line in text.split("\n") if line.strip().startswith("|")]
        if len(table_rows) > 2:  # 至少有表头和分隔行
            element_count = len(table_rows) - 2  # 减去表头和分隔行
        # 统计列表项（- 开头的行或数字. 开头的行）
        if element_count == 0:
            list_items = [
                line for line in text.split("\n")
                if line.strip().startswith("- ") or line.strip().startswith("  - ")
            ]
            element_count = len(list_items)

        # 构建结构化上下文
        structured = f"<visual_analysis>\n{text}\n</visual_analysis>"

        # 置信度估算
        confidence = 0.5
        if has_layout:
            confidence += 0.15
        if has_color:
            confidence += 0.1
        if has_interaction:
            confidence += 0.1
        if element_count > 3:
            confidence += 0.1
        confidence = min(1.0, confidence)

        return VisionAnalysisResult(
            raw_analysis=text,
            structured_context=structured,
            has_layout_info=has_layout,
            has_color_scheme=has_color,
            has_interaction_logic=has_interaction,
            element_count=element_count,
            confidence=confidence,
            source_image_type="detected_from_content",
        )


# Register compiler
TAPCompilerRegistry.register("glm_5v_turbo", GLM5VTurboCompiler)
