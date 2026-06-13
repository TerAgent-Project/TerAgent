"""teragent.core.compilers.anthropic — AnthropicCompiler

Anthropic-specific compilation with XML tag structured optimization (Claude preference).
"""

from __future__ import annotations

from teragent.core.compiler import TAPCompiler, TAPCompilerRegistry
from teragent.core.tap import CompiledPrompt, TAPRequest


class AnthropicCompiler(TAPCompiler):
    """Anthropic-optimized TAP compiler

    Key optimizations:
        - Mode B: separate system_prompt and user_message (Anthropic native protocol)
        - XML tag structured context (Claude processes XML tags well)
        - System prompt contains: role + constraints + format hint + memory
        - User message contains: design/plan/dependency_report context + instruction

    Returns CompiledPrompt in Mode B (system_prompt + user_message).
    """

    def _get_compiler_type(self) -> str:
        """Anthropic compiler type for prompt registry lookup"""
        return "anthropic"

    def compile(self, request: TAPRequest) -> CompiledPrompt:
        """Compile TAPRequest into Anthropic-style system_prompt + user_message"""
        intent = request.meta.get("intent", "execute")
        system_prompt = self.get_system_prompt(intent)
        return self._do_compile(system_prompt, request)

    def _do_compile(self, system_prompt: str, request: TAPRequest) -> CompiledPrompt:
        """Build system_prompt + user_message for Anthropic native protocol"""

        # 1. System prompt: role + constraints + format hint + memory
        system_parts: list[str] = [
            system_prompt,
            ("约束：\n" + "\n".join(f"  {i+1}. {c}" for i, c in enumerate(request.constraints))) if request.constraints else "",
            request.output_format_hint,
        ]

        if request.context.get("memory") and request.context["memory"] != "N/A":
            system_parts.append(f"<memory>\n{request.context['memory']}\n</memory>")

        system_prompt_final = "\n\n".join(p for p in system_parts if p)

        # 2. User message: context (XML tagged) + instruction
        user_parts: list[str] = []

        if request.context.get("design") and request.context["design"] != "N/A":
            user_parts.append(f"<design>\n{request.context['design']}\n</design>")
        if request.context.get("plan") and request.context["plan"] != "N/A":
            user_parts.append(f"<plan>\n{request.context['plan']}\n</plan>")
        if (
            request.context.get("dependency_report")
            and request.context["dependency_report"] not in ["N/A", ""]
        ):
            user_parts.append(
                f"<dependency_report>\n{request.context['dependency_report']}\n</dependency_report>"
            )

        user_parts.append(request.instruction)

        user_message = "\n\n".join(user_parts)

        return CompiledPrompt(
            system_prompt=system_prompt_final,
            user_message=user_message,
        )


# Register compiler
TAPCompilerRegistry.register("anthropic", AnthropicCompiler)
