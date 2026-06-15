"""teragent.core.compilers.minimax_m3 — MiniMaxM3Compiler

MiniMax M3 专属编译器，支持：
  1. 原生多模态编译（图像/视频/文本 → OpenAI 格式 content 数组）
  2. MSA 全文注入策略（大文档直接全文灌入，不做检索裁剪）
  3. Agent 编程增强（SWE-Bench Pro 59.0% 验证能力）
  4. 浏览增强（BrowseComp 83.5 分领先）
  5. 桌面操作上下文格式化
  6. 1M 上下文优化
  7. 多模态意图感知 prompt 模板
  8. 混合内容编译（多图/图文交织/视频+图片）
  9. 多模态 token 预算估算

设计参考：design.md §4 MiniMax M3 深度适配方案
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from teragent.core.compiler import TAPCompiler, TAPCompilerRegistry
from teragent.core.tap import CompiledPrompt, MultimodalContent, TAPRequest

logger = logging.getLogger(__name__)

# MiniMax M3 支持的视频 URL 格式
_SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v",
}

# 视频元数据默认值
_DEFAULT_VIDEO_DURATION_SECONDS = 60
_DEFAULT_VIDEO_FRAME_RATE = 24


class MiniMaxM3Compiler(TAPCompiler):
    """MiniMax M3 专属 TAP 编译器

    策略核心：
    1. 多模态感知：将 multimodal_context 编码为 OpenAI 格式的 content 数组
    2. 1M 上下文优化：利用 MSA 架构特性，大文档全文注入（无需检索裁剪）
    3. Agent 编程增强：M3 在 SWE-Bench Pro 上领先，编程 prompt 深度优化
    4. 桌面操作格式化：将 desktop_context 转换为 M3 的桌面操作指令格式
    5. 浏览器/信息检索增强：M3 BrowseComp 领先，优化 browse 相关意图 prompt
    6. 多模态意图感知：根据意图和内容类型注入针对性 prompt 模板
    7. 混合内容编译：支持多图、图文交织、图片+视频等混合场景
    8. Token 预算估算：多模态内容的 token 消耗估算，确保不超过 1M 限制

    Returns CompiledPrompt in Mode A (messages list).
    """

    # ----- Capability overrides -----

    @property
    def supports_multimodal(self) -> bool:
        """M3 原生支持多模态（图像+视频+文本）"""
        return True

    @property
    def max_context_tokens(self) -> int:
        """M3 支持 1M tokens 上下文"""
        return 1_000_000

    def _get_compiler_type(self) -> str:
        """Compiler type for prompt registry lookup"""
        return "minimax_m3"

    # ----- Main compile -----

    def compile(self, request: TAPRequest) -> CompiledPrompt:
        """编译 TAP 请求为 MiniMax M3 专属 prompt

        根据 multimodal_context 是否存在选择编译路径：
        - 有多模态内容 → _compile_multimodal()
        - 纯文本 → _compile_text()
        """
        if request.has_multimodal or request.has_desktop_context:
            return self._compile_multimodal(request)
        else:
            return self._compile_text(request)

    # ----- Text-only compilation -----

    def _compile_text(self, request: TAPRequest) -> CompiledPrompt:
        """纯文本编译模式

        MSA 全文注入策略：大文档直接全文灌入，不做检索裁剪。
        M3 的 MSA 架构在 1M 上下文下效率极高（1/20 计算量），
        因此不需要像 V4 那样做检索式注入。
        """
        messages: list[dict] = []
        intent = request.meta.get("intent", "execute")
        system_prompt = self.get_system_prompt(intent)

        # 1. System message
        # M3 适合较完整的系统提示
        system_parts = [system_prompt]

        # 编程增强：根据意图附加 SWE-Bench Pro 验证的编程引导
        programming_guidance = self._build_programming_guidance(intent, request)
        if programming_guidance:
            system_parts.append(programming_guidance)

        # 浏览增强
        browse_guidance = self._build_browse_guidance(intent, request)
        if browse_guidance:
            system_parts.append(browse_guidance)

        if request.constraints:
            constraint_text = "约束：\n" + "\n".join(
                f"  {i+1}. {c}" for i, c in enumerate(request.constraints)
            )
            system_parts.append(constraint_text)

        if request.output_format_hint:
            system_parts.append(f"输出格式：{request.output_format_hint}")

        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        # 2. Context injection — MSA 全文注入策略
        # M3 的 MSA 在长上下文下效率极高，可以全文灌入
        self._inject_context_fulltext(messages, request)

        # 3. Core instruction
        messages.append({"role": "user", "content": request.instruction})

        return CompiledPrompt(messages=messages, max_tokens=16384)

    # ----- Multimodal compilation -----

    def _compile_multimodal(self, request: TAPRequest) -> CompiledPrompt:
        """多模态编译模式

        将 multimodal_context 编码为 OpenAI 格式的 content 数组。
        图像 URL → image_url 类型
        图像 Base64 → data URI 格式的 image_url
        视频 URL → video_url 类型（通过 _process_video_input 增强）
        文本 → text 类型

        增强特性：
        - 多模态意图感知 prompt 模板
        - 混合内容编译（多图/图文交织/图片+视频）
        - 正确排序：文本上下文 → 图片 → 约束 → 指令
        """
        messages: list[dict] = []
        intent = request.meta.get("intent", "execute")
        system_prompt = self.get_system_prompt(intent)

        # 1. System message (纯文本)
        system_parts = [system_prompt]

        # 编程增强
        programming_guidance = self._build_programming_guidance(intent, request)
        if programming_guidance:
            system_parts.append(programming_guidance)

        # 多模态意图感知 prompt 模板
        multimodal_guidance = self._build_multimodal_system_addition(intent, request)
        if multimodal_guidance:
            system_parts.append(multimodal_guidance)

        if request.constraints:
            constraint_text = "约束：\n" + "\n".join(
                f"  {i+1}. {c}" for i, c in enumerate(request.constraints)
            )
            system_parts.append(constraint_text)

        if request.output_format_hint:
            system_parts.append(f"输出格式：{request.output_format_hint}")

        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        # 2. Context injection (text parts)
        self._inject_context_fulltext(messages, request)

        # 3. Desktop context (if present)
        if request.has_desktop_context:
            desktop_msg = self._build_desktop_context_message(request)
            if desktop_msg:
                messages.append(desktop_msg)

        # 4. Multimodal user message — 增强的混合内容编译
        if request.has_multimodal:
            content_parts = self._build_multimodal_content_parts(request)
            messages.append({"role": "user", "content": content_parts})
        else:
            # Desktop context only, no multimodal
            messages.append({"role": "user", "content": request.instruction})

        # 5. 估算多模态 token 预算，确保不超过 1M 限制
        estimated_tokens = self.estimate_multimodal_tokens(request)
        if estimated_tokens > self.max_context_tokens:
            logger.warning(
                f"MiniMaxM3Compiler: 估算 token 数 {estimated_tokens} 超过 "
                f"M3 上限 {self.max_context_tokens}，可能导致截断"
            )

        return CompiledPrompt(messages=messages, max_tokens=16384)

    # ----- Multimodal content part building -----

    def _build_multimodal_content_parts(self, request: TAPRequest) -> list[dict]:
        """构建多模态 content 数组，支持混合内容编译

        排序策略：文本上下文 → 图片 → 约束 → 指令
        支持：多图、图文交织、图片+视频混合

        Args:
            request: TAP 请求

        Returns:
            OpenAI 格式的 content 数组
        """
        content_parts: list[dict] = []

        # 分类收集多模态内容
        text_parts: list[MultimodalContent] = []
        image_parts: list[MultimodalContent] = []
        video_parts: list[MultimodalContent] = []

        if request.multimodal_context:
            for mc in request.multimodal_context:
                if mc.type == "text":
                    text_parts.append(mc)
                elif mc.type in ("image_url", "image_base64"):
                    image_parts.append(mc)
                elif mc.type == "video_url":
                    video_parts.append(mc)

        # 1. 文本指令在前（提供上下文）
        if request.instruction:
            content_parts.append({"type": "text", "text": request.instruction})

        # 2. 多模态文本内容（如图片说明等）
        for mc in text_parts:
            content_parts.append(mc.to_openai_format())

        # 3. 图片内容（支持多图）
        for i, mc in enumerate(image_parts):
            if len(image_parts) > 1:
                # 多图场景：为每张图片添加序号标注
                label = f"\n[图片 {i+1}/{len(image_parts)}]"
                content_parts.append({"type": "text", "text": label})
            content_parts.append(mc.to_openai_format())

        # 4. 视频内容（通过 _process_video_input 增强）
        for mc in video_parts:
            video_part = self._process_video_input(mc)
            content_parts.append(video_part)

        # 5. 约束条件（放在图片后，增强注意力）
        if request.constraints:
            constraint_text = "约束：\n" + "\n".join(
                f"  {i+1}. {c}" for i, c in enumerate(request.constraints)
            )
            content_parts.append({"type": "text", "text": constraint_text})

        # 6. 输出格式提示
        if request.output_format_hint:
            content_parts.append({
                "type": "text",
                "text": f"输出格式：{request.output_format_hint}"
            })

        return content_parts

    # ----- Video input processing -----

    def _process_video_input(self, mc: MultimodalContent) -> dict:
        """处理视频输入，生成 MiniMax API 格式的 content 块

        对视频 URL 进行格式检查和元数据处理：
        - 检查视频 URL 是否为支持的格式
        - 编码为 MiniMax API 要求的格式
        - 支持可选的视频元数据（duration, frame_rate）
        - 不支持的格式优雅降级

        Args:
            mc: 多模态内容块（type="video_url"）

        Returns:
            MiniMax API 格式的视频 content 块
        """
        url = mc.url or ""

        # 检查视频 URL 是否为支持的格式
        is_supported = self._is_supported_video_url(url)

        if not is_supported and url:
            logger.warning(
                f"MiniMaxM3Compiler: 视频 URL 格式可能不受支持: {url[-30:] if len(url) > 30 else url}。"
                f"支持的格式: {', '.join(sorted(_SUPPORTED_VIDEO_EXTENSIONS))}"
            )

        # 构建基础视频 content 块
        video_content: dict = {
            "type": "video_url",
            "video_url": {"url": url},
        }

        # 处理可选的视频元数据
        # media_type 字段可存放 JSON 字符串格式的元数据
        metadata = self._parse_video_metadata(mc.media_type)
        if metadata:
            video_content["video_url"].update(metadata)
            logger.debug(
                f"MiniMaxM3Compiler: 视频元数据: "
                f"duration={metadata.get('duration', 'N/A')}s, "
                f"frame_rate={metadata.get('frame_rate', 'N/A')}fps"
            )

        return video_content

    @staticmethod
    def _is_supported_video_url(url: str) -> bool:
        """检查视频 URL 是否为支持的格式

        支持的格式通过文件扩展名判断。
        对于流媒体 URL（如 HLS/DASH），默认视为支持。

        Args:
            url: 视频 URL

        Returns:
            是否为支持的视频格式
        """
        if not url:
            return False

        # 流媒体协议默认支持
        if url.startswith(("rtmp://", "rtsp://", "hls://")):
            return True

        # 检查文件扩展名
        url_lower = url.lower().split("?")[0]  # 去掉查询参数
        for ext in _SUPPORTED_VIDEO_EXTENSIONS:
            if url_lower.endswith(ext):
                return True

        # 某些 CDN URL 可能不含扩展名，但仍是有效视频
        # 例如: https://api.example.com/videos/12345
        # 这种情况下不判定为不支持，仅当明确是其他文件类型时才报错
        known_non_video = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg",
                          ".pdf", ".txt", ".html", ".json", ".xml")
        for ext in known_non_video:
            if url_lower.endswith(ext):
                return False

        # 无法确定时默认支持（避免误报）
        return True

    @staticmethod
    def _parse_video_metadata(media_type: Optional[str]) -> dict:
        """解析视频元数据

        media_type 字段可存放 JSON 格式的元数据：
        {"duration": 120, "frame_rate": 30}

        Args:
            media_type: 可能是 MIME 类型或 JSON 元数据字符串

        Returns:
            解析后的元数据字典，可能为空
        """
        if not media_type:
            return {}

        # 如果是标准 MIME 类型（如 "video/mp4"），不是元数据
        if media_type.startswith("video/"):
            return {}

        try:
            metadata = json.loads(media_type)
            if isinstance(metadata, dict):
                result = {}
                # 提取已知的元数据字段
                if "duration" in metadata:
                    result["duration"] = float(metadata["duration"])
                if "frame_rate" in metadata:
                    result["frame_rate"] = float(metadata["frame_rate"])
                return result
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        return {}

    # ----- Multimodal system prompt templates -----

    def _build_multimodal_system_addition(self, intent: str, request: TAPRequest) -> str:
        """构建多模态意图感知的系统 prompt 补充

        根据意图和内容类型，注入针对性的多模态引导 prompt：
        - execute/code_generation + 图片: 分析截图 UI 布局，生成代码
        - review + 图片: 对比截图与设计稿，逐项检查一致性
        - design + 图片: 参考截图视觉风格，设计新方案
        - desktop context: 根据桌面截图和交互元素，执行桌面操作

        Args:
            intent: 请求意图
            request: TAP 请求

        Returns:
            多模态引导 prompt 字符串，无多模态内容时返回空字符串
        """
        has_images = False
        has_videos = False
        has_desktop = request.has_desktop_context

        if request.multimodal_context:
            for mc in request.multimodal_context:
                if mc.type in ("image_url", "image_base64"):
                    has_images = True
                elif mc.type == "video_url":
                    has_videos = True

        # 无多模态内容时不添加
        if not has_images and not has_videos and not has_desktop:
            return ""

        # 桌面操作上下文优先
        if has_desktop:
            return (
                "【多模态桌面操作模式】\n"
                "根据桌面截图和可交互元素列表，执行桌面操作。注意：\n"
                "1. 仔细观察截图中各元素的位置和状态\n"
                "2. 结合可交互元素列表定位目标元素\n"
                "3. 生成精确的坐标和操作指令\n"
                "4. 操作前确认目标元素的可操作性"
            )

        # 视频 + 图片混合场景（优先于单一类型，因为混合场景更特殊）
        if has_videos and has_images:
            return (
                "【多模态混合分析模式】\n"
                "同时处理图片和视频内容。注意：\n"
                "1. 先分析静态图片获取关键信息\n"
                "2. 再结合视频内容理解动态变化\n"
                "3. 综合图片和视频信息给出完整分析\n"
                "4. 注意视频中的时序关系和变化趋势"
            )

        # 根据意图和内容类型选择 prompt 模板
        if has_images and intent in ("execute", "code_generation"):
            return (
                "【多模态代码生成模式】\n"
                "分析截图中的UI布局，然后生成对应代码。注意：\n"
                "1. 仔细识别截图中的 UI 组件和布局结构\n"
                "2. 将视觉元素映射为代码组件\n"
                "3. 保持视觉一致性和功能完整性\n"
                "4. 使用合理的样式和布局代码还原截图效果"
            )

        if has_images and intent == "review":
            return (
                "【多模态审查模式】\n"
                "对比截图与设计稿，逐项检查一致性。注意：\n"
                "1. 逐个对比视觉元素的位置、大小、颜色\n"
                "2. 检查文字内容和字体是否一致\n"
                "3. 验证交互状态和响应式布局\n"
                "4. 列出所有不一致之处并给出修改建议"
            )

        if has_images and intent == "design":
            return (
                "【多模态设计模式】\n"
                "参考截图中的视觉风格，设计新方案。注意：\n"
                "1. 分析截图的视觉语言和设计风格\n"
                "2. 提取色彩、排版、间距等设计特征\n"
                "3. 在保持风格一致性的基础上创新\n"
                "4. 输出详细的设计规范和组件说明"
            )

        # 纯视频场景
        if has_videos:
            return (
                "【多模态视频分析模式】\n"
                "分析视频内容。注意：\n"
                "1. 关注视频中的关键帧和重要场景\n"
                "2. 理解视频的时序逻辑和变化趋势\n"
                "3. 提取视频中的文字、UI元素等关键信息\n"
                "4. 如有元数据，结合时长和帧率进行分析"
            )

        # 纯图片场景（其他意图）
        if has_images:
            return (
                "【多模态图像分析模式】\n"
                "分析截图中的内容。注意：\n"
                "1. 详细描述截图中的可视内容\n"
                "2. 识别关键 UI 组件和交互元素\n"
                "3. 分析布局结构和层级关系\n"
                "4. 根据分析结果执行后续操作"
            )

        return ""

    # ----- Context injection (MSA full-text strategy) -----

    def _inject_context_fulltext(self, messages: list[dict], request: TAPRequest) -> None:
        """MSA 全文注入策略

        与基类 _inject_context 不同，M3 使用 MSA 架构，
        长上下文效率极高，不需要对大文档做裁剪或检索。
        直接全文注入即可。
        """
        if request.context.get("memory") and request.context["memory"] != "N/A":
            messages.append(
                {"role": "user", "content": f"<memory>\n{request.context['memory']}\n</memory>"}
            )
            messages.append({"role": "assistant", "content": "收到项目记忆。"})

        if request.context.get("design") and request.context["design"] != "N/A":
            messages.append(
                {"role": "user", "content": f"<design>\n{request.context['design']}\n</design>"}
            )
            messages.append({"role": "assistant", "content": "收到设计文档。"})

        if request.context.get("plan") and request.context["plan"] != "N/A":
            messages.append(
                {"role": "user", "content": f"<plan>\n{request.context['plan']}\n</plan>"}
            )
            messages.append({"role": "assistant", "content": "收到执行计划。"})

        if (
            request.context.get("dependency_report")
            and request.context["dependency_report"] not in ["N/A", ""]
        ):
            messages.append(
                {
                    "role": "user",
                    "content": f"<dependency_report>\n{request.context['dependency_report']}\n</dependency_report>",
                }
            )
            messages.append({"role": "assistant", "content": "收到依赖报告。"})

    # ----- Desktop context -----

    def _build_desktop_context_message(self, request: TAPRequest) -> dict | None:
        """构建桌面操作上下文消息

        将 desktop_context 转换为包含截图和交互元素的消息。
        截图作为 image_url 编码到 content 数组中。
        """
        if not request.has_desktop_context:
            return None

        dc = request.desktop_context
        content_parts: list[dict] = []

        # Desktop context header
        content_parts.append({"type": "text", "text": "当前桌面状态："})

        # Screenshot
        if dc.screenshot:
            content_parts.append(dc.screenshot.to_openai_format())

        # Text description of interactive elements
        desktop_text = dc.format_for_prompt()
        if desktop_text and dc.interactive_elements:
            content_parts.append({
                "type": "text",
                "text": desktop_text
            })

        if len(content_parts) <= 1:
            return None

        return {"role": "user", "content": content_parts}

    # ----- Programming enhancement -----

    def _build_programming_guidance(self, intent: str, request: TAPRequest) -> str:
        """构建编程增强 prompt

        M3 在 SWE-Bench Pro 上达到 59.0%，
        编程场景下需要特定的 prompt 引导来充分发挥能力。
        """
        if intent not in ("execute", "code_generation"):
            return ""

        return (
            "【编程增强模式】\n"
            "输出可直接运行的完整项目代码，包含：\n"
            "1. 完整的错误处理和边界检查\n"
            "2. 合理的代码结构和模块划分\n"
            "3. 必要的测试用例\n"
            "4. 清晰的代码注释"
        )

    # ----- Browse enhancement -----

    def _build_browse_guidance(self, intent: str, request: TAPRequest) -> str:
        """构建浏览增强 prompt

        M3 在 BrowseComp 上达到 83.5 分，
        信息检索和浏览场景需要优化 prompt。
        """
        # 检查 meta 中是否有 browse 相关标记
        meta_intent = request.meta.get("browse_intent", "")
        if intent != "chat" and not meta_intent:
            return ""

        return (
            "【信息检索增强】\n"
            "当需要查找信息时：\n"
            "1. 先明确需要查找的具体信息\n"
            "2. 使用精确的搜索关键词\n"
            "3. 对搜索结果进行交叉验证\n"
            "4. 综合多个来源给出准确答案"
        )

    # ----- Multimodal token budget estimation -----

    def estimate_multimodal_tokens(self, request: TAPRequest) -> int:
        """估算多模态请求的 token 总消耗

        基于 TAPRequest 的内容估算总 token 数，包括：
        - 文本内容（指令、上下文、约束等）
        - 图片内容（基于分辨率或固定估算）
        - 视频内容（基于时长估算）
        - 系统提示和其他固定开销

        用于确保请求不超过 M3 的 1M token 限制。

        Args:
            request: TAP 请求

        Returns:
            估算的 token 总数
        """
        from teragent.utils.token_counter import estimate_tokens

        total = 0

        # 1. 文本部分 token 估算
        total += estimate_tokens(request.instruction)
        total += estimate_tokens(str(request.constraints))
        total += estimate_tokens(request.output_format_hint)

        for v in request.context.values():
            if isinstance(v, str):
                total += estimate_tokens(v)

        # 2. 多模态内容 token 估算
        if request.multimodal_context:
            for mc in request.multimodal_context:
                total += self._estimate_single_content_tokens(mc)

        # 3. 桌面上下文 token 估算
        if request.has_desktop_context:
            dc = request.desktop_context
            # 截图按图片估算
            if dc.screenshot:
                total += self._estimate_single_content_tokens(dc.screenshot)
            # 交互元素列表按文本估算
            if dc.interactive_elements:
                total += estimate_tokens(str(dc.interactive_elements))
            if dc.active_window:
                total += estimate_tokens(dc.active_window)

        # 4. 系统提示和固定开销（约 500-1000 tokens）
        total += 800

        return total

    @staticmethod
    def _estimate_single_content_tokens(mc: MultimodalContent) -> int:
        """估算单个多模态内容块的 token 消耗

        估算策略：
        - 文本: 使用 token_counter 精确估算
        - 图片 URL: 固定 1000 tokens（标准分辨率）
        - 图片 Base64: 根据数据长度估算，最低 1000 tokens
        - 视频 URL: 根据时长估算，默认 3000 tokens（60秒视频）

        Args:
            mc: 多模态内容块

        Returns:
            估算的 token 数
        """
        from teragent.utils.token_counter import estimate_tokens

        if mc.type == "text":
            return estimate_tokens(mc.text or "")

        elif mc.type in ("image_url", "image_base64"):
            # 图片 token 估算
            # 标准 1024x1024 图片约 1000 tokens
            # 高分辨率可能更多，但我们使用保守固定估算
            base_tokens = 1000

            # 对于 base64 图片，可以根据数据大小做更精确的估算
            if mc.type == "image_base64" and mc.base64_data:
                # base64 数据长度 → 原始字节数 → 粗略分辨率估算
                # base64 编码后大小约为原始数据的 4/3
                raw_size = len(mc.base64_data) * 3 // 4
                # 假设 PNG 压缩比约 2:1，像素大小约 3 bytes (RGB)
                estimated_pixels = raw_size * 2 // 3
                # 假设正方形图片
                import math
                estimated_side = int(math.sqrt(max(1, estimated_pixels)))
                # OpenAI 风格：高分辨率图片按 tile 计算
                # 每个 tile 512x512 约 170 tokens
                tiles = max(1, (estimated_side // 512) ** 2)
                return max(base_tokens, tiles * 170)

            return base_tokens

        elif mc.type == "video_url":
            # 视频 token 估算基于时长
            metadata = MiniMaxM3Compiler._parse_video_metadata(mc.media_type)
            duration = metadata.get("duration", _DEFAULT_VIDEO_DURATION_SECONDS)

            # 每秒视频约 50 tokens（基于关键帧采样）
            # 最低 1000 tokens
            video_tokens = max(1000, int(duration * 50))
            return video_tokens

        return 0


# Register compiler
TAPCompilerRegistry.register("minimax_m3", MiniMaxM3Compiler)
