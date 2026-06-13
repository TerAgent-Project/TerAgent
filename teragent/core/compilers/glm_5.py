"""teragent.core.compilers.glm_5 — GLM5Compiler

GLM-5 专属编译器，支持：
  1. Recency Effect 强化：关键指令放在消息列表最后
  2. 200K 上下文极限压缩：激进压缩历史，确保当前任务上下文完整
  3. 思考模式控制（deep/quick/auto）
  4. 长程任务引导：形成"实验→分析→优化"闭环的 prompt 工程
  5. 自评估注入：在长程任务中周期性注入自评估检查点
  6. 策略切换引导：当模型陷入局部最优时，通过 prompt 引导切换策略

设计参考：design.md §5 GLM-5 深度适配方案
"""

from __future__ import annotations

import logging

from teragent.context.profiles import GLM5CompactionStrategy
from teragent.core.compiler import TAPCompiler, TAPCompilerRegistry
from teragent.core.tap import CompiledPrompt, TAPRequest

logger = logging.getLogger(__name__)


class GLM5Compiler(TAPCompiler):
    """GLM-5 专属 TAP 编译器

    策略核心：
    1. Recency Effect 强化：关键指令放在消息列表最后（延续 GLM 系列策略）
    2. 200K 上下文极限压缩：激进压缩历史，确保当前任务上下文完整
    3. 长程任务引导：形成"实验→分析→优化"闭环的 prompt 工程
    4. 策略切换提示：当模型陷入局部最优时，通过 prompt 引导切换策略
    5. 自评估触发：在长程任务中周期性注入自评估检查点

    Returns CompiledPrompt in Mode A (messages list).
    """

    # ----- Capability overrides -----

    @property
    def supports_thinking_mode(self) -> bool:
        """GLM-5 支持 thinking mode 控制"""
        return True

    @property
    def max_context_tokens(self) -> int:
        """GLM-5 支持 200K tokens 上下文"""
        return 200_000

    def _get_compiler_type(self) -> str:
        """Compiler type for prompt registry lookup"""
        return "glm_5"

    # ----- Main compile -----

    def compile(self, request: TAPRequest) -> CompiledPrompt:
        """编译 TAP 请求为 GLM-5 专属 prompt

        核心策略：Recency Effect — 关键指令放在消息列表最后
        """
        # Handle multimodal degradation
        multimodal_text = ""
        if request.has_multimodal and not self.supports_multimodal:
            multimodal_text = self._handle_multimodal_degradation(request)

        messages: list[dict] = []
        intent = request.meta.get("intent", "execute")
        system_prompt = self.get_system_prompt(intent)

        # 1. System message: 角色提示 + 约束 + 输出格式
        #    GLM 对系统提示的敏感度较高，可以把约束放在系统消息中
        constraint_list = "\n".join(
            f"  {i+1}. {c}" for i, c in enumerate(request.constraints)
        )
        format_hint = request.output_format_hint or "用 <file path='...'> 输出代码。"

        system_parts = [system_prompt]
        if request.constraints:
            system_parts.append(f"约束：\n{constraint_list}")
        system_parts.append(format_hint)

        # 长程任务模式：在系统提示中注入长程任务引导
        if request.is_long_horizon:
            system_parts.append(self._build_long_horizon_system_addition(request))

        messages.append({
            "role": "system",
            "content": "\n\n".join(system_parts)
        })

        # 2. Context as a single concatenated block in the middle
        #    (保持指令更接近尾部 — Recency Effect)
        context_str = self._build_glm_context(request)
        if multimodal_text:
            context_str = f"{context_str}\n\n{multimodal_text}" if context_str else multimodal_text

        if context_str:
            messages.append({"role": "user", "content": f"参考上下文：\n{context_str}"})
            messages.append({"role": "assistant", "content": "收到上下文。"})

        # 3. Desktop context degradation (GLM doesn't support desktop ops)
        if request.has_desktop_context:
            desktop_text = request.desktop_context.format_for_prompt()
            if desktop_text:
                logger.warning(
                    "GLM5Compiler: Request contains desktop_context but GLM-5 "
                    "does not support desktop operations. Degraded to text."
                )
                messages.append({"role": "user", "content": f"桌面状态：\n{desktop_text}"})
                messages.append({"role": "assistant", "content": "收到桌面状态信息。"})

        # 4. Chinese constraints injection (GLM 系列策略)
        chinese_constraints = self._inject_chinese_constraints(request)

        # 5. Long-horizon self-evaluation checkpoint (if applicable)
        self_evaluation_prompt = ""
        if request.is_long_horizon and request.long_horizon and request.long_horizon.self_evaluation_enabled:
            self_evaluation_prompt = self._build_self_evaluation_prompt(request)

        # 6. Core instruction LAST (Recency Effect — 最关键的优化)
        final_instruction = request.instruction
        prefix_parts: list[str] = []

        if chinese_constraints:
            prefix_parts.append(chinese_constraints)
        if self_evaluation_prompt:
            prefix_parts.append(self_evaluation_prompt)

        if prefix_parts:
            final_instruction = "\n\n".join(prefix_parts) + "\n\n" + final_instruction

        messages.append({"role": "user", "content": final_instruction})

        # 7. Tail reinforcement (Recency Effect — 最后一条消息强化关键信息)
        tail_content = self._build_tail_reinforcement(request)
        if tail_content:
            messages.append({"role": "assistant", "content": "收到指令，正在执行。"})
            messages.append({"role": "user", "content": tail_content})

        compiled = CompiledPrompt(messages=messages, max_tokens=16384)

        # Apply thinking mode parameters
        self._apply_thinking_mode(compiled, request)

        return compiled

    # ----- Context building -----

    def _build_glm_context(self, request: TAPRequest) -> str:
        """构建 GLM 风格的单一上下文块

        与基类 _inject_context 的多轮对话不同，
        GLM 使用单一上下文块来保持指令更接近末尾（Recency Effect）。
        """
        parts: list[str] = []

        design = request.context.get("design", "N/A")
        if design and design != "N/A":
            parts.append(f"<design>\n{design}\n</design>")

        plan = request.context.get("plan", "N/A")
        if plan and plan != "N/A":
            parts.append(f"<plan>\n{plan}\n</plan>")

        dep = request.context.get("dependency_report", "N/A")
        if dep and dep not in ["N/A", ""]:
            parts.append(f"<dependency_report>\n{dep}\n</dependency_report>")

        memory = request.context.get("memory", "N/A")
        if memory and memory != "N/A":
            parts.append(f"<memory>\n{memory}\n</memory>")

        return "\n".join(parts)

    # ----- Chinese constraints -----

    def _inject_chinese_constraints(self, request: TAPRequest) -> str:
        """注入中文输出格式约束（GLM 系列策略）

        GLM 模型在指令附近放置显式格式约束时，
        中文输出质量更高（Recency Effect）。
        """
        intent = request.meta.get("intent", "execute")
        parts: list[str] = []

        if intent in ("execute", "code_generation"):
            parts.append("中文注释，英文标识符。用 <file path='...'> 包裹代码。")
        elif intent == "design":
            parts.append("中文撰写，技术术语保留英文。")
        elif intent == "plan":
            parts.append("中文撰写执行计划。")
        elif intent == "review":
            parts.append("中文给出审查意见。")

        if request.output_format_hint:
            parts.append(f"输出格式：{request.output_format_hint}")

        return "\n".join(parts)

    # ----- Thinking mode -----

    def _apply_thinking_mode(self, compiled: CompiledPrompt, request: TAPRequest) -> None:
        """将 thinking_mode 转换为 API 参数

        GLM-5 的思考模式控制：
        - thinking_mode="deep" → 启用深度推理
        - thinking_mode="quick" → 快速响应模式
        - thinking_mode="auto" → 根据意图自动判断
        """
        mode = request.effective_thinking_mode

        if mode == "auto":
            intent = request.meta.get("intent", "execute")
            if intent in ("chat", "chat_friendly"):
                mode = "quick"
            elif intent in ("design", "plan", "review"):
                mode = "deep"
            else:
                mode = "deep"  # GLM-5 默认深度推理

        if mode == "deep":
            compiled.extra["thinking"] = {"type": "enabled"}
        elif mode == "quick":
            compiled.extra["thinking"] = {"type": "disabled"}

    # ----- Long-horizon task support -----

    def _build_long_horizon_system_addition(self, request: TAPRequest) -> str:
        """构建长程任务的系统提示追加内容

        GLM-5 支持 8 小时持续工作，需要在系统提示中
        预设"先规划阶段→每阶段自验收→发现问题自主修复"的工作模式。
        """
        config = request.long_horizon
        if not config:
            return ""

        parts: list[str] = [
            "【长程任务模式】",
            f"最大持续时间：{config.max_duration_hours}小时",
            f"检查点间隔：{config.checkpoint_interval_minutes}分钟",
            "",
            "工作流程：",
            "1. 先将整体目标分解为阶段性子目标",
            "2. 每个阶段自主执行，遇到问题自主调试和修复",
            "3. 每完成一个阶段，进行自我评估",
            "4. 发现停滞时，主动切换策略（换方法、换工具、简化目标）",
            "5. 定期输出进度报告，包含已完成/进行中/待做的内容",
        ]

        if config.self_evaluation_enabled:
            parts.append("")
            parts.append("自评估规则：每完成一个子目标后，评估：")
            parts.append("  - 当前方法是否有效？")
            parts.append("  - 是否需要调整策略？")
            parts.append("  - 距离最终目标还有多远？")

        return "\n".join(parts)

    def _build_self_evaluation_prompt(self, request: TAPRequest) -> str:
        """构建自评估检查点的 prompt 注入

        在长程任务执行过程中，周期性注入此 prompt
        触发模型进行自我评估。
        """
        config = request.long_horizon
        if not config:
            return ""

        return (
            "【自评估检查点】\n"
            "请暂停执行，对当前进度进行自我评估：\n"
            "1. 已完成的步骤和成果\n"
            "2. 当前遇到的问题\n"
            "3. 下一步计划\n"
            "4. 是否需要切换策略？"
        )

    def build_strategy_switch_prompt(self, reason: str = "") -> str:
        """构建策略切换引导 prompt

        当检测到模型陷入局部最优（连续相同结果）时，
        注入此 prompt 引导切换策略。

        Args:
            reason: 策略切换的原因（如"连续3次相同结果"）

        Returns:
            策略切换引导 prompt
        """
        parts: list[str] = [
            "【策略切换引导】",
            "检测到当前方法可能陷入局部最优。",
        ]
        if reason:
            parts.append(f"原因：{reason}")
        parts.extend([
            "",
            "建议尝试以下策略：",
            "1. 换一种完全不同的方法/算法",
            "2. 简化问题，先解决核心子问题",
            "3. 回退到上一个成功状态，换方向推进",
            "4. 将问题分解为更小的独立子问题",
        ])
        return "\n".join(parts)

    # ----- Tail reinforcement for GLM-5 -----

    def _build_tail_reinforcement(self, request: TAPRequest) -> str:
        """构建尾部强化内容

        利用 GLM-5 的 Recency Effect（对最近消息更关注），
        在消息列表末尾放置关键信息的强化重复。

        尾部强化包含：
        1. 当前指令重申：确保模型明确当前任务
        2. 约束条件重申：关键约束在尾部再次强调
        3. 自评估提示：引导模型自我检查

        Budget: tail_budget from GLM5CompactionStrategy (20,480 tokens)

        Args:
            request: TAP 请求对象

        Returns:
            尾部强化文本
        """
        strategy = GLM5CompactionStrategy()
        # 尾部预算字符数估算 ≈ tail_budget * 2.5
        max_chars = int(strategy.tail_budget * 2.5)

        parts: list[str] = []

        # 1. 当前指令重申
        if request.instruction:
            instruction_preview = request.instruction
            if len(instruction_preview) > 300:
                instruction_preview = instruction_preview[:300] + "..."
            parts.append("【当前指令重申】")
            parts.append(instruction_preview)

        # 2. 约束条件重申
        if request.constraints:
            parts.append("")
            parts.append("【关键约束】")
            for i, constraint in enumerate(request.constraints[:5], 1):
                # 每个约束截断到 200 字符
                c = constraint if len(constraint) <= 200 else constraint[:200] + "..."
                parts.append(f"  {i}. {c}")

        # 3. 自评估提示
        parts.append("")
        parts.append("【自评估检查】")
        parts.append("完成输出前，请确认：")
        parts.append("  - 是否满足所有约束条件？")
        parts.append("  - 输出格式是否符合要求？")
        parts.append("  - 是否遗漏了指令中的关键要求？")

        result = "\n".join(parts)

        # 如果超出预算，截断
        if len(result) > max_chars:
            result = result[:max_chars] + "\n... [尾部强化截断]"

        return result


# Register compiler
TAPCompilerRegistry.register("glm_5", GLM5Compiler)
