"""teragent.core.compilers.deepseek_v4 — DeepSeekV4Compiler

DeepSeek V4 专属编译器，支持：
  1. Flash/Pro 双模式感知（Flash 极简 prompt，Pro 完整 prompt）
  2. 思考模式控制（deep/quick/auto）
  3. 1M 上下文分区优化（系统提示固定 + 大文件检索 + 尾部强化）
  4. 多模态降级处理（V4 不支持多模态，降级为文本描述）
  5. 缓存感知布局（不变内容前置，最大化缓存命中）
  6. 缓存前缀冻结（系统提示 + 工具定义固定在前，最大化跨请求缓存命中）
  7. 缓存预热机制（build_warmup_request 生成预热请求）
  8. 缓存感知压缩策略（根据缓存命中率决定是否激进压缩）
  9. 尾部强化增强（_build_tail_reinforcement 利用 V4 CSA 注意力特性）
 10. 大文件检索注入（_inject_large_file_context 超阈值文件用检索代替全文）
 11. 三级缓存友好布局（冻结前缀 → 半静态 → 动态消息）

设计参考：design.md §3 DeepSeek V4 深度适配方案
"""

from __future__ import annotations

import logging
from typing import Literal

from teragent.context.profiles import ContextProfile, DeepSeekV4ContextProfile
from teragent.core.compiler import TAPCompiler, TAPCompilerRegistry
from teragent.core.tap import CompiledPrompt, TAPRequest

logger = logging.getLogger(__name__)


class DeepSeekV4Compiler(TAPCompiler):
    """DeepSeek V4 专属 TAP 编译器

    策略核心：
    1. 极简系统提示 + 约束内联到用户消息（延续 V3 策略）
    2. 新增：思考模式控制（thinking_mode 字段映射到 API 参数）
    3. 新增：Flash/Pro 模式感知（不同模式使用不同的约束强度）
    4. 新增：1M 上下文优化（关键信息前置 + 尾部强化）
    5. 优化：数学/代码推理的专用 prompt 增强
    6. 新增：缓存前缀冻结（系统提示 + 工具定义前置，最大化缓存命中）
    7. 新增：缓存预热机制（首次请求前可发送预热请求初始化缓存）
    8. 新增：缓存感知压缩（低命中率时激进压缩，高命中率时保持缓存稳定）

    Returns CompiledPrompt in Mode A (messages list).

    Args:
        variant: "flash" or "pro" — 决定编译策略的精细度
            flash: 极简 prompt，更短的约束描述，适合快速响应
            pro: 完整 prompt，详细约束 + 推理引导，适合复杂任务
        tools: 可选的工具定义列表，用于 function calling
            工具定义会被前置到消息列表中以利用缓存
    """

    # 大文件检索注入的 token 阈值：超过此值的文件用检索代替全文
    # 默认值 50K，当提供 profile 时由 profile.large_file_budget 决定
    LARGE_FILE_TOKEN_THRESHOLD: int = 50_000

    def __init__(
        self,
        variant: str = "pro",
        tools: list[dict] | None = None,
        profile: ContextProfile | None = None,
    ) -> None:
        if variant not in ("flash", "pro"):
            raise ValueError(f"Invalid variant: {variant!r}. Must be 'flash' or 'pro'.")
        self.variant: Literal["flash", "pro"] = variant
        # 工具定义：缓存感知布局时前置到消息列表
        self.tools: list[dict] = tools or []
        # 跟踪是否为会话中的首次编译（用于缓存预热推荐）
        self._session_compile_count: int = 0
        # 上下文分区配置：默认使用 DeepSeek V4 1M 分区
        self._context_profile: DeepSeekV4ContextProfile = (
            profile if isinstance(profile, DeepSeekV4ContextProfile)
            else profile or DeepSeekV4ContextProfile()
        )

    @property
    def profile(self) -> ContextProfile:
        """当前上下文分区配置（向后兼容属性）"""
        return self._context_profile

    @profile.setter
    def profile(self, value: ContextProfile) -> None:
        """设置上下文分区配置"""
        self._context_profile = value

    def get_context_profile(self) -> DeepSeekV4ContextProfile:
        """返回当前的上下文分区配置

        Returns:
            当前使用的 DeepSeekV4ContextProfile 实例
        """
        return self._context_profile

    # ----- Capability overrides -----

    @property
    def supports_thinking_mode(self) -> bool:
        """DeepSeek V4 支持 thinking mode 控制"""
        return True

    @property
    def max_context_tokens(self) -> int:
        """DeepSeek V4 支持 1M tokens 上下文（从 profile 获取）"""
        return self._context_profile.max_tokens

    def _get_compiler_type(self) -> str:
        """Compiler type for prompt registry lookup"""
        return "deepseek_v4"

    # ----- Main compile -----

    def compile(self, request: TAPRequest) -> CompiledPrompt:
        """编译 TAP 请求为 DeepSeek V4 专属 prompt

        根据 variant 选择编译路径：
        - flash: _compile_flash() — 极简编译
        - pro: _compile_pro() — 深度编译

        当 cache_preference 为 "aggressive" 或 "auto" 时，
        额外执行缓存感知布局：冻结前缀 + 工具定义前置 + 缓存元数据注入。
        """
        # Handle multimodal degradation
        multimodal_text = ""
        if request.has_multimodal and not self.supports_multimodal:
            multimodal_text = self._handle_multimodal_degradation(request)

        if self.variant == "flash":
            compiled = self._compile_flash(request, multimodal_text)
        else:
            compiled = self._compile_pro(request, multimodal_text)

        # Apply thinking mode parameters to compiled.extra
        self._apply_thinking_mode(compiled, request)

        # 注入大文件检索上下文（在缓存布局之前，确保文件内容在正确位置）
        self._inject_large_file_context(compiled, request)

        # Apply cache-aware layout — 缓存感知布局增强（三级分组）
        cache_aware = request.cache_preference and request.cache_preference != "none"
        if cache_aware:
            self._apply_cache_aware_layout(compiled, request)

        # 构建尾部强化（利用 V4 CSA 注意力的 Recency Effect）
        self._build_tail_reinforcement(compiled, request)

        # 更新会话编译计数
        self._session_compile_count += 1

        return compiled

    # ----- Cache-aware layout -----

    def _apply_cache_aware_layout(self, compiled: CompiledPrompt, request: TAPRequest) -> None:
        """应用缓存感知布局策略（三级分组）

        P2-2 增强：将消息按可变性分为三级，最大化缓存命中：
        1. 冻结前缀（frozen）：系统提示 + 工具定义 — 跨请求不变
        2. 半静态区（semi-static）：大文件注入内容、设计文档 — 会话内偶尔变化
        3. 动态区（dynamic）：最近对话、用户指令 — 每次请求都变

        布局顺序：frozen → semi-static → dynamic
        这样 frozen 和 semi-static 的前缀缓存可以在多次请求中命中。

        Args:
            compiled: 已编译的 CompiledPrompt，会被就地修改
            request: 原始 TAP 请求
        """
        # 标记缓存感知已启用，并记录 profile 派生的分区预算
        compiled.extra["cache_aware"] = True
        compiled.extra["profile_budgets"] = {
            "system": self._context_profile.system_budget,
            "history": self._context_profile.history_budget,
            "large_file": self._context_profile.large_file_budget,
            "tail_reinforcement": self._context_profile.tail_reinforcement_budget,
        }

        # ---- 第一步：分类现有消息 ----
        _frozen_messages: list[dict] = []
        semi_static_messages: list[dict] = []
        dynamic_messages: list[dict] = []

        for msg in compiled.messages:
            # 大文件注入内容标记为 semi-static
            if msg.get("_section") == "large_file":
                semi_static_messages.append(msg)
            # 系统提示属于冻结前缀（收集内容，由 _build_cache_prefix 统一重建）
            elif msg.get("role") == "system":
                # 保留系统消息内容，避免丢失约束和格式提示
                _frozen_messages.append(msg)
            else:
                dynamic_messages.append(msg)

        # ---- 第二步：构建冻结前缀消息（系统提示 + 工具定义） ----
        # 传入已有的系统消息内容，确保约束和格式提示不被丢失
        cache_prefix_messages = self._build_cache_prefix(request, _frozen_messages)

        if cache_prefix_messages:
            # 标记缓存前缀已冻结
            compiled.extra["cache_prefix_frozen"] = True
            compiled.extra["cache_prefix_message_count"] = len(cache_prefix_messages)

            # 在 compiled.extra 中记录冻结前缀的消息引用
            compiled.extra["cache_prefix"] = cache_prefix_messages
        else:
            # 无冻结前缀（无系统提示且无工具定义）
            compiled.extra["cache_prefix_frozen"] = False
            compiled.extra["cache_prefix_message_count"] = 0

        # ---- 第三步：组装三级布局 frozen → semi-static → dynamic ----
        final_messages: list[dict] = []

        # Level 1: 冻结前缀（系统提示 + 工具定义）
        final_messages.extend(cache_prefix_messages)

        # Level 2: 半静态区（大文件注入内容）
        # 清除内部标记 _section（仅用于分类，不应出现在最终 API 消息中）
        for msg in semi_static_messages:
            clean_msg = {k: v for k, v in msg.items() if k != "_section"}
            final_messages.append(clean_msg)

        # Level 3: 动态区（对话历史 + 用户指令）
        final_messages.extend(dynamic_messages)

        # 记录三级布局元数据
        compiled.extra["layout_sections"] = {
            "frozen": len(cache_prefix_messages),
            "semi_static": len(semi_static_messages),
            "dynamic": len(dynamic_messages),
        }

        compiled.messages = final_messages

        # 首次编译推荐缓存预热
        compiled.extra["cache_warmup_recommended"] = (self._session_compile_count == 0)

        # aggressive 模式的额外处理
        if request.cache_preference == "aggressive":
            compiled.extra["cache_prefix_frozen"] = True

    def _build_cache_prefix(self, request: TAPRequest, existing_system_messages: list[dict] | None = None) -> list[dict]:
        """构建缓存冻结前缀消息列表

        冻结前缀包含：
        1. 系统提示（如果存在）— 始终放在第一位
        2. 工具定义（如果存在）— 紧随系统提示之后

        这些消息在多次请求中保持不变，因此可以被 DeepSeek V4
        的前缀缓存机制命中，从而减少 token 计费和延迟。

        注意：此方法只构建前缀部分，不包含对话历史和用户指令。
        调用方需要将前缀消息插入到消息列表的头部。

        Args:
            request: TAP 请求（用于提取系统提示）
            existing_system_messages: 已有的系统消息列表（含约束和格式提示），
                如果提供，将合并到缓存前缀中，避免丢失编译器添加的约束。

        Returns:
            构成缓存冻结前缀的消息列表，可能为空
        """
        prefix_messages: list[dict] = []

        # 1. 系统提示 — 始终在缓存前缀的第一位
        intent = request.meta.get("intent", "execute")
        system_prompt = self.get_system_prompt(intent)
        if system_prompt:
            prefix_messages.append({"role": "system", "content": system_prompt})

        # 合并已有的系统消息内容（约束、格式提示、推理引导等）
        # 这些是由 _compile_pro/_compile_flash 添加的，不能丢失
        if existing_system_messages:
            for msg in existing_system_messages:
                content = msg.get("content", "")
                if not content:
                    continue
                # 避免重复：如果新系统提示已包含此内容，跳过
                if system_prompt and content == system_prompt:
                    continue
                # 追加为额外的系统消息
                prefix_messages.append({"role": "system", "content": content})

        # 2. 工具定义 — 紧随系统提示之后，确保跨请求缓存命中
        # 将工具定义序列化为一条系统级消息，放在前缀中
        if self.tools:
            tools_content = self._format_tools_as_message()
            if tools_content:
                prefix_messages.append({
                    "role": "system",
                    "content": tools_content,
                })

        return prefix_messages

    def _format_tools_as_message(self) -> str:
        """将工具定义格式化为消息内容字符串

        将 self.tools（OpenAI function calling 格式的工具定义列表）
        序列化为可嵌入到消息中的文本表示，以便利用缓存前缀。

        Returns:
            工具定义的文本表示，如果无工具则返回空字符串
        """
        if not self.tools:
            return ""

        import json

        parts: list[str] = ["<tools>"]
        for tool in self.tools:
            # 提取工具的核心信息
            func = tool.get("function", tool)
            name = func.get("name", "unknown")
            description = func.get("description", "")
            parameters = func.get("parameters", {})

            parts.append(f"<tool name=\"{name}\">")
            if description:
                parts.append(f"  <description>{description}</description>")
            if parameters:
                parts.append(f"  <parameters>{json.dumps(parameters, ensure_ascii=False)}</parameters>")
            parts.append("</tool>")

        parts.append("</tools>")
        return "\n".join(parts)

    # ----- Tail reinforcement (P2-2) -----

    def _build_tail_reinforcement(self, compiled: CompiledPrompt, request: TAPRequest) -> None:
        """构建尾部强化区域（利用 V4 CSA 注意力的 Recency Effect）

        P2-2 新增：将关键信息在消息末尾重复，利用 DeepSeek V4 的
        Causal Sparse Attention (CSA) 特性——靠近末尾的 token 获得更高的
        注意力权重，从而提升模型对关键约束的遵从率。

        尾部强化内容：
        1. 关键约束重复（原文重申，不省略）
        2. 关键指令摘要
        3. 输出格式提醒
        4. Pro 模式额外：推理要求和边界检查提醒
        5. Flash 模式：紧凑格式，仅保留最关键的约束重申

        使用 profile.tail_reinforcement_budget 控制尾部强化的 token 预算。

        Args:
            compiled: 已编译的 CompiledPrompt，会被就地修改
            request: 原始 TAP 请求
        """
        tail_budget = self._context_profile.tail_reinforcement_budget
        # 将 token 预算转换为粗略字符预算（4 字符/token）
        tail_char_budget = int(tail_budget * 4)

        tail_parts: list[str] = []
        intent = request.meta.get("intent", "execute")

        if self.variant == "pro":
            # Pro 模式：详细尾部强化
            if request.constraints:
                constraint_list = "\n".join(
                    f"  - {c}" for c in request.constraints
                )
                tail_parts.append(
                    f"【关键约束重申】请严格遵守以下约束，不可省略：\n{constraint_list}"
                )

            # 输出格式提醒
            if request.output_format_hint:
                tail_parts.append(
                    f"【输出格式】务必使用 {request.output_format_hint} 格式输出。"
                )

            # 意图特定的尾部强化
            if intent in ("design", "plan"):
                tail_parts.append(
                    "【推理要求】请深入分析后再给出方案，确保方案的完整性和可行性，不要遗漏关键细节。"
                )
            elif intent in ("execute", "code_generation"):
                tail_parts.append(
                    "【代码要求】输出完整可运行代码，包含错误处理和边界检查，不省略任何实现细节。"
                )
            elif intent == "review":
                tail_parts.append(
                    "【审查要求】逐项检查，每条问题附具体修改建议和示例代码。"
                )

        else:
            # Flash 模式：紧凑尾部强化
            if request.constraints and len(request.constraints) <= 3:
                tail_parts.append("重要：严格遵守上述约束。")
            elif request.constraints:
                # 约束超过 3 条时，只重申前 3 条
                top3 = "、".join(request.constraints[:3])
                tail_parts.append(f"重要：{top3}，以及其余约束。")

            if request.output_format_hint:
                tail_parts.append(f"务必使用 {request.output_format_hint} 格式输出。")

        if not tail_parts:
            return

        tail_content = "\n".join(tail_parts)

        # 截断到尾部强化预算内
        if len(tail_content) > tail_char_budget:
            tail_content = tail_content[:tail_char_budget] + "\n... [尾部强化截断]"

        # 修复 H22: 使用 system 消息代替伪造的 assistant 消息
        # 伪造 assistant 消息（"理解任务要求"）违反 API 协议，部分提供商会拒绝
        compiled.messages.append({"role": "system", "content": f"[尾部强化 — 务必遵守]\n{tail_content}"})

        # 在 extra 中记录尾部强化信息（含 profile 预算）
        compiled.extra["tail_reinforcement"] = True
        compiled.extra["tail_reinforcement_variant"] = self.variant
        compiled.extra["tail_reinforcement_budget"] = tail_budget

    # ----- Large file retrieval injection (P2-2) -----

    def _inject_large_file_context(self, compiled: CompiledPrompt, request: TAPRequest) -> None:
        """注入大文件检索上下文

        P2-2 新增：处理 request.context 中的大文件内容。
        对于超过 token 阈值的文件，使用 CodeIndexer 风格的检索策略
        （仅注入与当前任务相关的代码片段），而非全文注入。
        对于较小的文件，直接包含在历史区域中。

        文件上下文来源：request.context["large_files"]
        格式：{"large_files": [{"path": "...", "content": "...", "tokens": 50000}]}

        注入策略：
        - 小文件（tokens < LARGE_FILE_TOKEN_THRESHOLD）：注入到历史区域
        - 大文件（tokens >= LARGE_FILE_TOKEN_THRESHOLD）：注入到大文件区域（semi-static），
          使用检索策略只注入相关片段
        - 所有注入的大文件消息标记 _section="large_file"，供缓存布局分类使用

        Args:
            compiled: 已编译的 CompiledPrompt，会被就地修改
            request: 原始 TAP 请求
        """
        large_files = request.context.get("large_files")
        if not large_files or not isinstance(large_files, list):
            return

        large_file_budget = self._context_profile.large_file_budget
        remaining_budget = large_file_budget
        injected_files: list[str] = []

        for file_info in large_files:
            if not isinstance(file_info, dict):
                continue

            file_path = file_info.get("path", "unknown")
            file_content = file_info.get("content", "")
            estimated_tokens = file_info.get("tokens", 0)

            # 如果没有提供 token 估算，粗略估算
            if estimated_tokens <= 0 and file_content:
                # 粗略估算：4 字符/token + 1.3 保守系数
                estimated_tokens = int(len(file_content) / 4.0 * 1.3)

            if not file_content:
                continue

            # 使用 profile 派生的阈值：large_file_budget 的 10% 作为单个大文件阈值
            large_file_threshold = max(
                self.LARGE_FILE_TOKEN_THRESHOLD,
                int(self._context_profile.large_file_budget * 0.1),
            )

            if estimated_tokens >= large_file_threshold:
                # 大文件：注入检索摘要而非全文
                # 使用 _retrieve_relevant_snippets 按相关性检索
                snippet = self._retrieve_relevant_snippets(
                    file_content, file_path, request.instruction,
                    budget=self._context_profile.large_file_budget,
                )
                snippet_tokens = int(len(snippet) / 4.0 * 1.3)

                if snippet_tokens > remaining_budget:
                    # 预算不足，跳过此文件
                    logger.debug(
                        f"大文件 {file_path} 检索片段超出剩余预算 "
                        f"({snippet_tokens} > {remaining_budget})，跳过"
                    )
                    continue

                # 注入到大文件区域，标记为 semi-static
                compiled.messages.append({
                    "role": "user",
                    "content": f"<large_file path=\"{file_path}\" mode=\"retrieval\">\n{snippet}\n</large_file>",
                    "_section": "large_file",
                })
                compiled.messages.append({
                    "role": "assistant",
                    "content": f"收到大文件检索结果：{file_path}",
                    "_section": "large_file",
                })

                remaining_budget -= snippet_tokens
                injected_files.append(f"{file_path}(retrieval)")

            else:
                # 小文件：直接注入到历史区域（不标记 _section）
                # 这些消息将被归入 dynamic 区域
                if estimated_tokens > remaining_budget:
                    logger.debug(
                        f"小文件 {file_path} 超出剩余预算 ({estimated_tokens} > {remaining_budget})，跳过"
                    )
                    continue

                compiled.messages.append({
                    "role": "user",
                    "content": f"<file path=\"{file_path}\">\n{file_content}\n</file>",
                })
                compiled.messages.append({
                    "role": "assistant",
                    "content": f"收到文件：{file_path}",
                })

                remaining_budget -= estimated_tokens
                injected_files.append(f"{file_path}(full)")

        # 记录注入元数据
        if injected_files:
            compiled.extra["large_file_injection"] = {
                "files": injected_files,
                "budget_used": large_file_budget - remaining_budget,
                "budget_total": large_file_budget,
            }

    def _extract_retrieval_snippet(self, content: str, file_path: str) -> str:
        """从大文件中提取检索摘要片段

        CodeIndexer 风格的检索策略：
        1. 提取文件头部的类/函数定义（签名 + docstring）
        2. 提取文件尾部（最新修改通常在末尾）
        3. 限制总长度，确保不超出大文件区域预算

        这是一种简化的检索策略，无需 tree-sitter 依赖。
        如果 CodeIndexer 可用（teragent[ast]），应优先使用其检索结果。

        Args:
            content: 文件完整内容
            file_path: 文件路径（用于错误提示）

        Returns:
            检索摘要片段字符串
        """
        lines = content.split("\n")
        max_lines = 200  # 检索摘要最多 200 行

        if len(lines) <= max_lines:
            # 文件行数不多，直接返回全文
            return content

        # 策略：头部 100 行 + 尾部 100 行
        head_lines = lines[:100]
        tail_lines = lines[-100:]

        snippet_parts: list[str] = []
        snippet_parts.append(f"# 文件: {file_path} (检索摘要，共 {len(lines)} 行，仅展示首尾)")
        snippet_parts.append("")
        snippet_parts.append("## 文件头部（类/函数定义）")
        snippet_parts.extend(head_lines)
        snippet_parts.append("")
        snippet_parts.append(f"## 文件尾部（最新修改区域，省略中间 {len(lines) - 200} 行）")
        snippet_parts.extend(tail_lines)

        return "\n".join(snippet_parts)

    # ----- Relevant snippet retrieval (P2-2 enhanced) -----

    def _retrieve_relevant_snippets(
        self,
        content: str,
        file_path: str,
        instruction: str,
        budget: int,
    ) -> str:
        """从大文件中按相关性检索代码片段

        CodeIndexer 增强版检索策略：
        1. 将文件解析为逻辑段（函数、类、导入、全局代码）
        2. 按与当前指令的相关性评分（关键词匹配 + 位置加权）
        3. 返回 top-K 最相关的段落，总长度不超过 budget

        Args:
            content: 文件完整内容
            file_path: 文件路径
            instruction: 当前用户指令（用于相关性评分）
            budget: token 预算限制

        Returns:
            检索到的代码片段字符串
        """
        lines = content.split("\n")

        # Step 1: 解析文件为逻辑段
        sections = self._parse_file_sections(lines, file_path)

        # Step 2: 按相关性评分
        scored = self._score_sections(sections, instruction)

        # Step 3: 贪心选择，直到用完预算
        # 将 token 预算转换为字符预算
        char_budget = int(budget * 4)
        selected: list[tuple[float, str]] = []
        used_chars = 0

        # 始终包含文件头（imports + module docstring）
        header_section = next(
            (s for s in scored if s[0].startswith("__header__")), None
        )
        if header_section:
            _, header_text, header_score = header_section
            header_chars = len(header_text)
            if header_chars <= char_budget:
                selected.append((header_score, header_text))
                used_chars += header_chars

        # 按 score 降序排列，贪心选择
        remaining = sorted(
            [s for s in scored if not s[0].startswith("__header__")],
            key=lambda x: x[2],
            reverse=True,
        )

        for section_name, section_text, section_score in remaining:
            section_chars = len(section_text)
            if used_chars + section_chars <= char_budget:
                selected.append((section_score, section_text))
                used_chars += section_chars

        if not selected:
            # 回退到简单的头部+尾部策略
            return self._extract_retrieval_snippet(content, file_path)

        # 组装输出
        parts = [f"# 文件: {file_path} (相关性检索，共 {len(lines)} 行，检索到 {len(selected)} 段)"]
        parts.append("")
        for score, text in sorted(selected, key=lambda x: x[0], reverse=True):
            parts.append(text)
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _parse_file_sections(lines: list[str], file_path: str) -> list[tuple[str, str]]:
        """将文件解析为逻辑段

        段类型：
        - __header__: 文件头部（imports, module docstring）
        - class_XXX: 类定义
        - func_XXX: 函数/方法定义
        - global_XXX: 全局代码块

        Args:
            lines: 文件行列表
            file_path: 文件路径

        Returns:
            (section_name, section_text) 列表
        """
        sections: list[tuple[str, str]] = []
        current_name = "__header__"
        current_lines: list[str] = []
        global_counter = 0

        def _flush() -> None:
            nonlocal current_lines
            if current_lines:
                text = "\n".join(current_lines)
                sections.append((current_name, text))
                current_lines = []

        for line in lines:
            stripped = line.strip()

            # 检测类定义
            if stripped.startswith("class ") and stripped.endswith(":"):
                _flush()
                class_name = stripped[6:stripped.index("(") if "(" in stripped else stripped.index(":")]
                current_name = f"class_{class_name.strip()}"
                current_lines.append(line)

            # 检测函数/方法定义
            elif stripped.startswith("def ") and stripped.endswith(":"):
                _flush()
                func_name = stripped[4:stripped.index("(") if "(" in stripped else stripped.index(":")]
                current_name = f"func_{func_name.strip()}"
                current_lines.append(line)

            # 检测 async 函数
            elif stripped.startswith("async def ") and stripped.endswith(":"):
                _flush()
                func_name = stripped[10:stripped.index("(") if "(" in stripped else stripped.index(":")]
                current_name = f"func_{func_name.strip()}"
                current_lines.append(line)

            else:
                # 如果当前在 __header__ 且遇到了非 import/docstring 的代码，
                # 转换为 global 段
                if current_name == "__header__" and current_lines:
                    # 检查 header 是否已经结束（空行后非 import/docstring/装饰器）
                    if stripped and not stripped.startswith(("#", "import ", "from ", "\"\"\"", "'''", "@")):
                        _flush()
                        global_counter += 1
                        current_name = f"global_{global_counter}"

                current_lines.append(line)

        _flush()
        return sections

    @staticmethod
    def _score_sections(
        sections: list[tuple[str, str]],
        instruction: str,
    ) -> list[tuple[str, str, float]]:
        """按与指令的相关性为段落评分

        评分策略：
        1. 关键词匹配：指令中的关键词出现在段落中的数量
        2. 位置加权：靠近文件末尾的段落获得额外权重（recency）
        3. 段名匹配：段名包含指令关键词的段落加分

        Args:
            sections: (section_name, section_text) 列表
            instruction: 当前用户指令

        Returns:
            (section_name, section_text, score) 列表
        """
        # 提取指令关键词（简单分词：按空格和标点）
        import re
        instruction_words = set(
            w.lower() for w in re.split(r'[\s\-\.,;:!?(){}\[\]<>]+', instruction)
            if len(w) > 2  # 忽略太短的词
        )

        total_sections = len(sections)
        result: list[tuple[str, str, float]] = []

        for idx, (name, text) in enumerate(sections):
            score = 0.0
            text_lower = text.lower()
            name_lower = name.lower()

            # 关键词匹配
            for word in instruction_words:
                if word in text_lower:
                    score += 1.0
                if word in name_lower:
                    score += 2.0  # 段名匹配加权

            # 位置加权：靠后的段落获得小量加分
            position_bonus = (idx / max(1, total_sections)) * 0.5
            score += position_bonus

            # header 始终有基础分（imports 通常重要）
            if name == "__header__":
                score += 1.0

            result.append((name, text, score))

        return result

    # ----- Cache warmup -----

    def build_warmup_request(self, request: TAPRequest | None = None) -> TAPRequest:
        """构建缓存预热请求

        在长对话开始时，发送一个仅包含系统提示 + 工具定义的请求，
        以触发 DeepSeek V4 的前缀缓存。后续请求将命中缓存，
        减少重复计算和计费。

        用法示例::

            compiler = DeepSeekV4Compiler(variant="pro", tools=[...])
            warmup_req = compiler.build_warmup_request()
            warmup_compiled = compiler.compile(warmup_req)
            # 将 warmup_compiled 发送给 API，预热缓存

        Args:
            request: 可选的 TAP 请求，用于提取系统提示和意图。
                如果为 None，则使用默认意图 "execute"。

        Returns:
            一个最小的 TAPRequest，仅包含系统提示和工具定义，
            编译后可用于预热 DeepSeek V4 的前缀缓存。
        """
        # 从原始请求提取意图，或使用默认意图
        if request is not None:
            intent = request.meta.get("intent", "execute")
            meta = {"task_id": "cache_warmup", "intent": intent}
        else:
            intent = "execute"
            meta = {"task_id": "cache_warmup", "intent": intent}

        # 构建一个最小的请求：仅包含系统提示 + 工具定义
        # 指令为空或最小化，目的是触发缓存而非获取有意义的回复
        warmup_request = TAPRequest(
            meta=meta,
            instruction="[缓存预热] 请确认已准备好。",
            constraints=[],
            output_format_hint="",
            context={},
            # 保持与原始请求相同的缓存偏好
            cache_preference="aggressive",
        )

        return warmup_request

    # ----- Cache-aware compression -----

    def _should_compress_aggressively(self, cache_hit_rate: float) -> bool:
        """根据缓存命中率判断是否应进行激进压缩

        缓存感知压缩策略：
        - 命中率 < 30%：缓存效果差，激进压缩以节省 token 开销
        - 命中率 30%~70%：缓存效果中等，保持当前策略
        - 命中率 > 70%：缓存效果好，不压缩以避免破坏缓存前缀

        此方法由 AutoCompactor 集成调用，用于决定上下文管理策略。

        Args:
            cache_hit_rate: 缓存命中率，范围 [0.0, 1.0]

        Returns:
            True 表示应激进压缩，False 表示保持当前策略或避免压缩
        """
        if cache_hit_rate < 0.3:
            # 缓存命中率低，压缩以节省 token
            logger.debug(
                f"缓存命中率 {cache_hit_rate:.1%} < 30%，建议激进压缩以节省 token"
            )
            return True
        elif cache_hit_rate > 0.7:
            # 缓存命中率高，避免压缩导致缓存失效
            logger.debug(
                f"缓存命中率 {cache_hit_rate:.1%} > 70%，缓存效果好，避免压缩破坏缓存"
            )
            return False
        else:
            # 中间区域，保持当前策略
            logger.debug(
                f"缓存命中率 {cache_hit_rate:.1%} 在 30%-70% 之间，保持当前压缩策略"
            )
            return False

    # ----- Flash mode compilation -----

    def _compile_flash(self, request: TAPRequest, multimodal_text: str = "") -> CompiledPrompt:
        """Flash 模式：极简编译

        - 系统提示压缩到 200 tokens 以内
        - 约束以 JSON 列表内联到用户消息
        - 不附加推理引导
        - 适合 CHAT / CHAT_FRIENDLY / 简单 EXECUTE
        """
        messages: list[dict] = []

        # 1. 极简系统消息（只包含角色身份）
        # 注意：缓存感知模式下，系统提示由 _apply_cache_aware_layout 统一前置
        # 这里仍然生成系统消息，_apply_cache_aware_layout 会去重处理
        intent = request.meta.get("intent", "execute")
        system_prompt = self.get_system_prompt(intent)
        messages.append({"role": "system", "content": system_prompt})

        # 2. Context 作为单一用户消息
        context_parts = self._build_context_string(request)
        if request.context.get("memory") and request.context["memory"] != "N/A":
            context_parts = (
                f"<memory>\n{request.context['memory']}\n</memory>\n\n{context_parts}"
                if context_parts
                else f"<memory>\n{request.context['memory']}\n</memory>"
            )

        if context_parts:
            messages.append({"role": "user", "content": context_parts})
            messages.append({"role": "assistant", "content": "收到。"})

        # 3. Core instruction with inlined constraints (Flash 优化：紧凑格式)
        instruction_parts: list[str] = []

        # Multimodal degradation text
        if multimodal_text:
            instruction_parts.append(f"附加信息：\n{multimodal_text}")

        # Constraints as compact JSON list
        if request.constraints:
            constraints_json = str(request.constraints)
            instruction_parts.append(f"约束：{constraints_json}")

        if request.output_format_hint:
            instruction_parts.append(f"输出格式：{request.output_format_hint}")

        instruction_parts.append(request.instruction)

        messages.append({"role": "user", "content": "\n\n".join(instruction_parts)})

        # 注意：尾部强化已由 _build_tail_reinforcement() 统一处理，此处不再内联

        compiled = CompiledPrompt(messages=messages, max_tokens=8192)

        # 设置工具定义到 CompiledPrompt（供 Adapter 传入 API）
        if self.tools:
            compiled.tools = self.tools

        return compiled

    # ----- Pro mode compilation -----

    def _compile_pro(self, request: TAPRequest, multimodal_text: str = "") -> CompiledPrompt:
        """Pro 模式：深度编译

        - 完整系统提示（角色 + 能力 + 约束 + 推理引导）
        - 约束以自然语言详细描述
        - 附加推理引导（数学/代码场景）
        - 输出格式更精确的描述
        - 适合 DESIGN / PLAN / EXECUTE（复杂任务）/ REVIEW
        """
        messages: list[dict] = []

        # 1. 系统消息：角色 + 推理增强引导
        intent = request.meta.get("intent", "execute")
        system_prompt = self.get_system_prompt(intent)

        # Pro 模式：根据意图附加推理引导
        reasoning_guidance = self._build_reasoning_guidance(intent, request)
        full_system = f"{system_prompt}\n\n{reasoning_guidance}" if reasoning_guidance else system_prompt

        messages.append({"role": "system", "content": full_system})

        # 2. Context injection (multi-turn dialogue for enhanced attention)
        self._inject_context(messages, request)

        # 3. 多模态降级文本
        if multimodal_text:
            messages.append({"role": "user", "content": f"附加视觉信息：\n{multimodal_text}"})
            messages.append({"role": "assistant", "content": "收到视觉信息。"})

        # 4. Core instruction with detailed constraints (Pro: 自然语言描述)
        instruction_parts: list[str] = []

        if request.constraints:
            constraint_text = "约束：\n" + "\n".join(
                f"  {i+1}. {c}" for i, c in enumerate(request.constraints)
            )
            instruction_parts.append(constraint_text)

        if request.output_format_hint:
            instruction_parts.append(f"输出格式：{request.output_format_hint}")

        instruction_parts.append(request.instruction)

        messages.append({"role": "user", "content": "\n\n".join(instruction_parts)})

        # 注意：尾部强化已由 _build_tail_reinforcement() 统一处理，此处不再内联

        compiled = CompiledPrompt(messages=messages, max_tokens=16384)

        # 设置工具定义到 CompiledPrompt（供 Adapter 传入 API）
        if self.tools:
            compiled.tools = self.tools

        return compiled

    # ----- Thinking mode -----

    def _apply_thinking_mode(self, compiled: CompiledPrompt, request: TAPRequest) -> None:
        """将 thinking_mode 转换为 API 参数并注入到 compiled.extra

        DeepSeek V4 API 参数映射：
        - thinking_mode="deep" → extra_body={"thinking": {"type": "enabled"}}
        - thinking_mode="quick" → extra_body={"thinking": {"type": "disabled"}}
        - thinking_mode="auto" → 根据意图自动判断
          DESIGN/PLAN → deep, CHAT/CHAT_FRIENDLY → quick, 其他 → deep
        """
        mode = request.effective_thinking_mode

        if mode == "auto":
            # Auto: 根据意图自动判断
            intent = request.meta.get("intent", "execute")
            if intent in ("chat", "chat_friendly"):
                mode = "quick"
            elif intent in ("design", "plan"):
                mode = "deep"
            else:
                # EXECUTE / CODE_GENERATION / REVIEW → Pro 用 deep, Flash 用 quick
                mode = "deep" if self.variant == "pro" else "quick"

        if mode == "deep":
            compiled.extra["thinking"] = {"type": "enabled"}
        elif mode == "quick":
            compiled.extra["thinking"] = {"type": "disabled"}

    # ----- Reasoning guidance -----

    def _build_reasoning_guidance(self, intent: str, request: TAPRequest) -> str:
        """根据意图构建推理引导语（Pro 模式专用）

        DeepSeek V4 在数学和代码推理方面表现优秀，
        通过 prompt 引导可以进一步提升推理质量。
        """
        guidance_parts: list[str] = []

        if intent in ("execute", "code_generation"):
            guidance_parts.append(
                "请写出完整的推理链条，包括中间步骤验证。"
                "输出可运行的完整代码，包含错误处理和边界检查。"
            )
        elif intent == "design":
            guidance_parts.append(
                "请系统分析需求，考虑多种方案的优劣，选择最优方案并说明理由。"
                "注意 UI 美观性，使用现代设计风格。"
            )
        elif intent == "plan":
            guidance_parts.append(
                "请仔细分析任务依赖关系，确保计划的可执行性和完整性。"
            )
        elif intent == "review":
            guidance_parts.append(
                "请逐项深入检查，不仅要发现表面问题，还要发现潜在的逻辑缺陷。"
            )

        return "\n".join(guidance_parts)


# Register compiler with both variant names
TAPCompilerRegistry.register("deepseek_v4", DeepSeekV4Compiler)
TAPCompilerRegistry.register("deepseek_v4_flash", lambda **kw: DeepSeekV4Compiler(variant="flash", **kw))
TAPCompilerRegistry.register("deepseek_v4_pro", lambda **kw: DeepSeekV4Compiler(variant="pro", **kw))
