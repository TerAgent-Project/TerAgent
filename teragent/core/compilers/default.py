"""teragent.core.compilers.default — DefaultCompiler

Generic OpenAI-compatible compilation strategy using multi-turn context injection.
"""

from __future__ import annotations

from teragent.core.compiler import TAPCompiler, TAPCompilerRegistry
from teragent.core.tap import CompiledPrompt, TAPRequest


class DefaultCompiler(TAPCompiler):
    """Default OpenAI-compatible TAP compiler

    Compilation strategy:
        1. System message: role description + constraints + output_format_hint + memory
        2. Context injection: multi-turn dialogue (design → plan → dependency_report)
        3. Core instruction as final user message

    Returns CompiledPrompt in Mode A (messages list).
    """

    def _get_compiler_type(self) -> str:
        """Default compiler type for prompt registry lookup"""
        return "default"

    def compile(self, request: TAPRequest) -> CompiledPrompt:
        """Compile TAPRequest into OpenAI-compatible messages array"""
        intent = request.meta.get("intent", "execute")
        system_prompt = self.get_system_prompt(intent)
        return self._do_compile(system_prompt, request)

    def _do_compile(self, system_prompt: str, request: TAPRequest) -> CompiledPrompt:
        """Build messages list from system prompt + context + instruction"""
        messages: list[dict] = []

        # 1. System message: role + constraints + format hint (memory injected via _inject_context)
        system_parts = [
            system_prompt,
            ("约束：\n" + "\n".join(f"  {i+1}. {c}" for i, c in enumerate(request.constraints))) if request.constraints else "",
            request.output_format_hint,
        ]
        system_content = "\n\n".join(p for p in system_parts if p)

        messages.append({"role": "system", "content": system_content})

        # 2. Context injection (multi-turn dialogue for enhanced attention)
        self._inject_context(messages, request)

        # 3. Core instruction
        messages.append({"role": "user", "content": request.instruction})

        return CompiledPrompt(messages=messages)


# Register compiler
TAPCompilerRegistry.register("default", DefaultCompiler)
