"""teragent.core.compiler — TAPCompiler ABC + CompiledPrompt + TAPCompilerRegistry

A Compiler transforms TAPRequest (IR) into CompiledPrompt (model-specific prompt).
Different models need different compilation strategies:
  - Default: Generic OpenAI-compatible format
  - GLM: Recency effect optimization (key instruction last)
  - Anthropic: XML tag structured optimization
  - DeepSeek: Minimalist compilation
  - DeepSeekV4: Thinking mode + Flash/Pro dual variants + 1M context (new)
  - GLM5: Recency effect + 200K extreme compression + long-horizon (new)
  - MiniMaxM3: Multi-modal + MSA full-text injection + desktop ops (new)
"""

from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from teragent.core.tap import TAPRequest, CompiledPrompt

logger = logging.getLogger(__name__)


class TAPCompiler(ABC):
    """TAP Compiler abstract base class

    Responsibility: Compile TAPRequest into model-specific prompt format.
    Each Compiler also provides intent-specific system prompts.

    Subclasses conventionally implement:
        _do_compile(system_prompt, request) -> CompiledPrompt
    
    The only strict requirement is overriding the abstract `compile()` method.
    Concrete subclasses typically factor their logic into `_do_compile()`,
    but this is a convention, not enforced by the ABC.

    Optionally override:
        _default_prompts -> dict[str, str]
        supports_multimodal -> bool
        supports_thinking_mode -> bool
        max_context_tokens -> int
    """

    # ----- Capability properties (override in subclasses) -----

    @property
    def supports_multimodal(self) -> bool:
        """Whether this Compiler can handle multimodal input (images, videos).

        Only MiniMaxM3Compiler returns True; other Compilers should degrade
        multimodal content to text descriptions and log a warning.
        """
        return False

    @property
    def supports_thinking_mode(self) -> bool:
        """Whether this Compiler supports thinking/reasoning mode control.

        DeepSeekV4Compiler and GLM5Compiler return True.
        When True, the Compiler reads request.thinking_mode and configures
        the prompt and API parameters accordingly.
        """
        return False

    @property
    def max_context_tokens(self) -> int:
        """Maximum context window size in tokens for this Compiler's model.

        Compilers should override this to reflect their model's actual limit:
        - DeepSeekV4Compiler: 1_000_000
        - MiniMaxM3Compiler: 1_000_000
        - GLM5Compiler: 200_000
        - Default: 128_000
        """
        return 128_000

    # ----- Abstract method -----

    @abstractmethod
    def compile(self, request: TAPRequest) -> CompiledPrompt:
        """Compile a TAP request into a model-specific prompt

        Concrete subclasses typically follow this pattern:
        1. Resolve the intent from request.meta
        2. Get the intent-specific system prompt via get_system_prompt()
        3. Delegate to _do_compile(system_prompt, request)

        Args:
            request: The TAP request to compile

        Returns:
            CompiledPrompt ready for an Adapter to send
        """
        ...

    # ----- Prompt resolution -----

    def get_system_prompt(self, intent: str) -> str:
        """Get intent-specific system prompt

        Args:
            intent: One of: design | plan | replan | execute | code_generation | review | chat | chat_friendly | sub_agent

        Returns:
            System prompt string for the given intent, or empty string if not found
        """
        return self._default_prompts.get(intent, "")

    def _get_compiler_type(self) -> str:
        """Return the compiler type name for prompt registry lookup.

        Subclasses should override to return their specific type:
        'default', 'glm', 'anthropic', 'deepseek',
        'deepseek_v4', 'glm_5', 'minimax_m3'.
        """
        return "default"

    @functools.cached_property
    def _default_prompts(self) -> dict[str, str]:
        """Provide intent-specific system prompts from the Prompt Registry.

        Uses _get_compiler_type() to select the correct compiler variant.
        Subclasses that override this property will bypass the Registry.
        """
        from teragent.core.prompts import get_system_prompt_for_intent, list_intents
        compiler_type = self._get_compiler_type()
        prompts = {}
        for intent in list_intents():
            prompt = get_system_prompt_for_intent(intent, compiler_type)
            if prompt:
                prompts[intent] = prompt
        return prompts

    # ----- Context injection helpers -----

    def _inject_context(self, messages: list[dict], request: TAPRequest) -> list[dict]:
        """Shared context injection logic (all Compilers share this)

        Injects memory/design/plan/dependency_report as multi-turn dialogue
        to enhance attention. Subclasses can override to adjust injection
        order and format.

        Args:
            messages: Existing message list to append context to
            request: TAP request containing context fields

        Returns:
            Updated message list with context injected
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
        return messages

    def _build_context_string(self, request: TAPRequest) -> str:
        """Build a single context string (for Compilers that use system+user mode)

        Unlike _inject_context which creates multi-turn dialogue,
        this method concatenates context into a single string suitable
        for the user_message in Anthropic-style protocols.

        Note: This method does NOT include `memory` — Compilers that use
        this method handle memory separately (e.g., AnthropicCompiler
        places it in the system prompt, DeepSeekCompiler appends it manually).

        Args:
            request: TAP request containing context fields

        Returns:
            Newline-joined context string (design + plan + dependency_report only)
        """
        parts: list[str] = []
        if request.context.get("design") and request.context["design"] != "N/A":
            parts.append(f"<design>\n{request.context['design']}\n</design>")
        if request.context.get("plan") and request.context["plan"] != "N/A":
            parts.append(f"<plan>\n{request.context['plan']}\n</plan>")
        if (
            request.context.get("dependency_report")
            and request.context["dependency_report"] not in ["N/A", ""]
        ):
            parts.append(
                f"<dependency_report>\n{request.context['dependency_report']}\n</dependency_report>"
            )
        return "\n\n".join(parts)

    # ----- Multimodal degradation -----

    def _handle_multimodal_degradation(self, request: TAPRequest) -> str:
        """Handle multimodal content when this Compiler doesn't support it.

        Extracts text descriptions from multimodal content and emits a warning.
        Called by Compilers that have supports_multimodal=False when they
        receive a TAPRequest with multimodal_context.

        Args:
            request: The TAP request with multimodal content

        Returns:
            Concatenated text descriptions of all multimodal content
        """
        if not request.has_multimodal:
            return ""

        descriptions: list[str] = []
        for mc in request.multimodal_context:
            desc = mc.extract_text_description()
            if desc:
                descriptions.append(desc)

        if descriptions:
            logger.warning(
                f"{self.__class__.__name__}: Request contains multimodal content but "
                f"this Compiler does not support it. Degraded to text descriptions. "
                f"Use MiniMaxM3Compiler for native multimodal support."
            )

        return "\n".join(descriptions)


class TAPCompilerRegistry:
    """Compiler registry — maps compiler names to Compiler classes"""

    _compilers: dict[str, type[TAPCompiler]] = {}

    @classmethod
    def _get_registry(cls) -> dict[str, type[TAPCompiler]]:
        """Return the compiler registry for *this* class, creating one if needed.

        Without this, subclasses would share the parent's ``_compilers`` dict
        because mutable class variables are inherited by reference.
        """
        if "_compilers" not in cls.__dict__:
            cls._compilers = {}
        return cls._compilers

    @classmethod
    def register(cls, name: str, compiler_cls: type[TAPCompiler]) -> None:
        """Register a Compiler class under a name"""
        cls._get_registry()[name] = compiler_cls
        logger.debug(f"Registered compiler: {name} -> {compiler_cls.__name__}")

    @classmethod
    def get(cls, name: str) -> type[TAPCompiler] | None:
        """Get a registered Compiler class by name"""
        return cls._get_registry().get(name)

    @classmethod
    def create(cls, name: str, **kwargs) -> TAPCompiler:
        """Create a Compiler instance by name

        Args:
            name: Registered compiler name
            **kwargs: Arguments to pass to the Compiler constructor

        Returns:
            New Compiler instance

        Raises:
            ValueError: If no compiler is registered under the given name
        """
        compiler_cls = cls._get_registry().get(name)
        if compiler_cls is None:
            raise ValueError(
                f"Unknown compiler: {name}. Available: {list(cls._get_registry().keys())}"
            )
        return compiler_cls(**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """List all registered compiler names"""
        return list(cls._get_registry().keys())
