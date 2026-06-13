"""teragent.core.compilers.deepseek — DeepSeekCompiler

DeepSeek-specific minimalist compilation strategy.
DeepSeek doesn't benefit from elaborate system prompts; constraints are more
effective when inlined directly into the user message.
"""

from __future__ import annotations

from teragent.core.compiler import TAPCompiler, TAPCompilerRegistry
from teragent.core.tap import CompiledPrompt, TAPRequest


class DeepSeekCompiler(TAPCompiler):
    """DeepSeek-optimized TAP compiler

    Key optimizations:
        - Minimalist system prompt (just role identity, no constraints)
        - Constraints inlined into user message (more effective for DeepSeek)
        - Direct, concise prompts without verbose formatting
        - Context injected as single user message before instruction

    Returns CompiledPrompt in Mode A (messages list).
    """

    def _get_compiler_type(self) -> str:
        """DeepSeek compiler type for prompt registry lookup"""
        return "deepseek"

    def compile(self, request: TAPRequest) -> CompiledPrompt:
        """Compile TAPRequest with DeepSeek minimalist strategy"""
        intent = request.meta.get("intent", "execute")
        system_prompt = self.get_system_prompt(intent)
        return self._do_compile(system_prompt, request)

    def _do_compile(self, system_prompt: str, request: TAPRequest) -> CompiledPrompt:
        """Build messages with minimalist system prompt + inlined constraints"""
        messages: list[dict] = []

        # 1. Minimal system message (just role identity)
        messages.append({"role": "system", "content": system_prompt})

        # 2. Context as single user message (if any)
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

        # 3. Core instruction with inlined constraints (DeepSeek optimization)
        instruction_parts: list[str] = []

        if request.constraints:
            instruction_parts.append(
                ("约束：\n" + "\n".join(f"  {i+1}. {c}" for i, c in enumerate(request.constraints))) if request.constraints else ""
            )

        if request.output_format_hint:
            instruction_parts.append(f"输出格式：{request.output_format_hint}")

        instruction_parts.append(request.instruction)

        messages.append({"role": "user", "content": "\n\n".join(instruction_parts)})

        return CompiledPrompt(messages=messages)


# Register compiler
TAPCompilerRegistry.register("deepseek", DeepSeekCompiler)
