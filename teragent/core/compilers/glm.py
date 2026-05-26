"""teragent.core.compilers.glm — GLMCompiler

GLM-specific compilation with recency effect optimization and Chinese constraint injection.
"""

from __future__ import annotations

from teragent.core.tap import TAPRequest, CompiledPrompt
from teragent.core.compiler import TAPCompiler, TAPCompilerRegistry


class GLMCompiler(TAPCompiler):
    """GLM-optimized TAP compiler

    Key optimizations:
        - Recency effect: core instruction placed LAST (closest to model output)
        - Context concatenated as a single block in the middle (not multi-turn)
        - Chinese constraints injection for improved Chinese output quality

    Returns CompiledPrompt in Mode A (messages list).
    """

    def _get_compiler_type(self) -> str:
        """GLM compiler type for prompt registry lookup"""
        return "glm"

    def compile(self, request: TAPRequest) -> CompiledPrompt:
        """Compile TAPRequest with GLM recency effect optimization"""
        intent = request.meta.get("intent", "execute")
        system_prompt = self.get_system_prompt(intent)
        return self._do_compile(system_prompt, request)

    def _do_compile(self, system_prompt: str, request: TAPRequest) -> CompiledPrompt:
        """Build messages with recency effect: system → context → instruction"""
        messages: list[dict] = []

        # 1. System message: role prompt + constraints + format hint (GLM is less sensitive to long system prompts)
        constraint_list = "\n".join(
            f"  {i+1}. {c}" for i, c in enumerate(request.constraints)
        )
        format_hint = request.output_format_hint or "用 <file path='...'> 输出代码。"
        messages.append(
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n"
                    + (f"约束：\n{constraint_list}\n" if request.constraints else "")
                    + f"{format_hint}"
                ),
            }
        )

        # 2. Context as a single concatenated block in the middle
        context_str = self._build_glm_context(request)
        if context_str:
            messages.append({"role": "user", "content": f"参考上下文：\n{context_str}"})
            messages.append({"role": "assistant", "content": "收到上下文。"})

        # 3. Chinese constraints injection
        chinese_constraints = self._inject_chinese_constraints(request)

        # 4. Core instruction LAST (recency effect optimization)
        final_instruction = request.instruction
        if chinese_constraints:
            final_instruction = f"{chinese_constraints}\n\n{final_instruction}"
        messages.append({"role": "user", "content": final_instruction})

        return CompiledPrompt(messages=messages)

    def _build_glm_context(self, request: TAPRequest) -> str:
        """Build a single concatenated context block for GLM

        Unlike the base class _inject_context which creates multi-turn dialogue,
        GLM benefits from a single context block to keep the instruction closer
        to the end (recency effect).
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

    def _inject_chinese_constraints(self, request: TAPRequest) -> str:
        """Inject Chinese-friendly output format constraints for GLM

        GLM models produce better Chinese output when explicit format
        constraints are placed near the instruction (recency effect).
        """
        intent = request.meta.get("intent", "execute")
        parts: list[str] = []

        if intent == "execute" or intent == "code_generation":
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


# Register compiler
TAPCompilerRegistry.register("glm", GLMCompiler)
