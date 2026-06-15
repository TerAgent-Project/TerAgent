"""teragent.core.prompts — Intent-specific system prompt templates

This package contains the actual prompt content for each intent, organized
separately so that:
  1. Each Compiler can reference model-specific optimized prompts
  2. Prompt content is centralized, not scattered across modules
  3. Different Compilers produce different prompts for the same intent

9 intents:
  - design: Generate DESIGN.md from requirements
  - plan: Generate PLAN.md from design
  - replan: Generate partial fix plan from failure
  - execute: Code generation / tool execution
  - code_generation: Alias for execute
  - review: Code review with APPROVE/REJECT
  - chat: Conversational agent mode (tool-using agent)
  - chat_friendly: Friendly non-programming chat
  - sub_agent: Autonomous sub-agent execution

8 compiler types:
  - default: Generic OpenAI-compatible format
  - glm: GLM recency effect optimization + Chinese constraints
  - anthropic: XML tag structured optimization (Claude preference)
  - deepseek: Minimalist compilation
  - deepseek_v4: DeepSeek V4 thinking mode + Flash/Pro (new)
  - glm_5: GLM-5 recency + long-horizon (new)
  - minimax_m3: MiniMax M3 multi-modal + MSA (new)
  - glm_52: GLM-5.2 1M context + dual thinking (new)
"""

import logging

from teragent.core.prompts.chat import (
    AGENT_PROMPT_ANTHROPIC,
    AGENT_PROMPT_DEEPSEEK,
    AGENT_PROMPT_DEEPSEEK_V4,
    AGENT_PROMPT_DEFAULT,
    AGENT_PROMPT_GLM,
    AGENT_PROMPT_GLM_5,
    AGENT_PROMPT_GLM_52,
    AGENT_PROMPT_MINIMAX_M3,
    CHAT_PROMPT_ANTHROPIC,
    CHAT_PROMPT_DEEPSEEK,
    CHAT_PROMPT_DEEPSEEK_V4,
    CHAT_PROMPT_DEFAULT,
    CHAT_PROMPT_GLM,
    CHAT_PROMPT_GLM_5,
    CHAT_PROMPT_GLM_52,
    CHAT_PROMPT_MINIMAX_M3,
)
from teragent.core.prompts.design import (
    DESIGN_PROMPT_ANTHROPIC,
    DESIGN_PROMPT_DEEPSEEK,
    DESIGN_PROMPT_DEEPSEEK_V4,
    DESIGN_PROMPT_DEFAULT,
    DESIGN_PROMPT_GLM,
    DESIGN_PROMPT_GLM_5,
    DESIGN_PROMPT_GLM_52,
    DESIGN_PROMPT_MINIMAX_M3,
)
from teragent.core.prompts.execute import (
    CUDA_TRITON_PROMPT_GLM_5,
    EXECUTE_PROMPT_ANTHROPIC,
    EXECUTE_PROMPT_DEEPSEEK,
    EXECUTE_PROMPT_DEEPSEEK_V4,
    EXECUTE_PROMPT_DEFAULT,
    EXECUTE_PROMPT_GLM,
    EXECUTE_PROMPT_GLM_5,
    EXECUTE_PROMPT_GLM_52,
    EXECUTE_PROMPT_MINIMAX_M3,
)
from teragent.core.prompts.plan import (
    PLAN_PROMPT_ANTHROPIC,
    PLAN_PROMPT_DEEPSEEK,
    PLAN_PROMPT_DEEPSEEK_V4,
    PLAN_PROMPT_DEFAULT,
    PLAN_PROMPT_GLM,
    PLAN_PROMPT_GLM_5,
    PLAN_PROMPT_GLM_52,
    PLAN_PROMPT_MINIMAX_M3,
    REPLAN_PROMPT_ANTHROPIC,
    REPLAN_PROMPT_DEEPSEEK,
    REPLAN_PROMPT_DEEPSEEK_V4,
    REPLAN_PROMPT_DEFAULT,
    REPLAN_PROMPT_GLM,
    REPLAN_PROMPT_GLM_5,
    REPLAN_PROMPT_GLM_52,
    REPLAN_PROMPT_MINIMAX_M3,
)
from teragent.core.prompts.review import (
    REVIEW_PROMPT_ANTHROPIC,
    REVIEW_PROMPT_DEEPSEEK,
    REVIEW_PROMPT_DEEPSEEK_V4,
    REVIEW_PROMPT_DEFAULT,
    REVIEW_PROMPT_GLM,
    REVIEW_PROMPT_GLM_5,
    REVIEW_PROMPT_GLM_52,
    REVIEW_PROMPT_MINIMAX_M3,
)
from teragent.core.prompts.sub_agent import (
    SUB_AGENT_PROMPT_ANTHROPIC,
    SUB_AGENT_PROMPT_DEEPSEEK,
    SUB_AGENT_PROMPT_DEEPSEEK_V4,
    SUB_AGENT_PROMPT_DEFAULT,
    SUB_AGENT_PROMPT_GLM,
    SUB_AGENT_PROMPT_GLM_5,
    SUB_AGENT_PROMPT_GLM_52,
    SUB_AGENT_PROMPT_MINIMAX_M3,
)

logger = logging.getLogger(__name__)


# ===== Intent → compiler_type → prompt mapping =====

_PROMPT_REGISTRY: dict[str, dict[str, str]] = {
    "design": {
        "default": DESIGN_PROMPT_DEFAULT,
        "glm": DESIGN_PROMPT_GLM,
        "anthropic": DESIGN_PROMPT_ANTHROPIC,
        "deepseek": DESIGN_PROMPT_DEEPSEEK,
        "deepseek_v4": DESIGN_PROMPT_DEEPSEEK_V4,
        "glm_5": DESIGN_PROMPT_GLM_5,
        "glm_52": DESIGN_PROMPT_GLM_52,
        "minimax_m3": DESIGN_PROMPT_MINIMAX_M3,
    },
    "plan": {
        "default": PLAN_PROMPT_DEFAULT,
        "glm": PLAN_PROMPT_GLM,
        "anthropic": PLAN_PROMPT_ANTHROPIC,
        "deepseek": PLAN_PROMPT_DEEPSEEK,
        "deepseek_v4": PLAN_PROMPT_DEEPSEEK_V4,
        "glm_5": PLAN_PROMPT_GLM_5,
        "glm_52": PLAN_PROMPT_GLM_52,
        "minimax_m3": PLAN_PROMPT_MINIMAX_M3,
    },
    "replan": {
        "default": REPLAN_PROMPT_DEFAULT,
        "glm": REPLAN_PROMPT_GLM,
        "anthropic": REPLAN_PROMPT_ANTHROPIC,
        "deepseek": REPLAN_PROMPT_DEEPSEEK,
        "deepseek_v4": REPLAN_PROMPT_DEEPSEEK_V4,
        "glm_5": REPLAN_PROMPT_GLM_5,
        "glm_52": REPLAN_PROMPT_GLM_52,
        "minimax_m3": REPLAN_PROMPT_MINIMAX_M3,
    },
    "review": {
        "default": REVIEW_PROMPT_DEFAULT,
        "glm": REVIEW_PROMPT_GLM,
        "anthropic": REVIEW_PROMPT_ANTHROPIC,
        "deepseek": REVIEW_PROMPT_DEEPSEEK,
        "deepseek_v4": REVIEW_PROMPT_DEEPSEEK_V4,
        "glm_5": REVIEW_PROMPT_GLM_5,
        "glm_52": REVIEW_PROMPT_GLM_52,
        "minimax_m3": REVIEW_PROMPT_MINIMAX_M3,
    },
    "chat": {
        "default": AGENT_PROMPT_DEFAULT,
        "glm": AGENT_PROMPT_GLM,
        "anthropic": AGENT_PROMPT_ANTHROPIC,
        "deepseek": AGENT_PROMPT_DEEPSEEK,
        "deepseek_v4": AGENT_PROMPT_DEEPSEEK_V4,
        "glm_5": AGENT_PROMPT_GLM_5,
        "glm_52": AGENT_PROMPT_GLM_52,
        "minimax_m3": AGENT_PROMPT_MINIMAX_M3,
    },
    "chat_friendly": {
        "default": CHAT_PROMPT_DEFAULT,
        "glm": CHAT_PROMPT_GLM,
        "anthropic": CHAT_PROMPT_ANTHROPIC,
        "deepseek": CHAT_PROMPT_DEEPSEEK,
        "deepseek_v4": CHAT_PROMPT_DEEPSEEK_V4,
        "glm_5": CHAT_PROMPT_GLM_5,
        "glm_52": CHAT_PROMPT_GLM_52,
        "minimax_m3": CHAT_PROMPT_MINIMAX_M3,
    },
    "sub_agent": {
        "default": SUB_AGENT_PROMPT_DEFAULT,
        "glm": SUB_AGENT_PROMPT_GLM,
        "anthropic": SUB_AGENT_PROMPT_ANTHROPIC,
        "deepseek": SUB_AGENT_PROMPT_DEEPSEEK,
        "deepseek_v4": SUB_AGENT_PROMPT_DEEPSEEK_V4,
        "glm_5": SUB_AGENT_PROMPT_GLM_5,
        "glm_52": SUB_AGENT_PROMPT_GLM_52,
        "minimax_m3": SUB_AGENT_PROMPT_MINIMAX_M3,
    },
    "execute": {
        "default": EXECUTE_PROMPT_DEFAULT,
        "glm": EXECUTE_PROMPT_GLM,
        "anthropic": EXECUTE_PROMPT_ANTHROPIC,
        "deepseek": EXECUTE_PROMPT_DEEPSEEK,
        "deepseek_v4": EXECUTE_PROMPT_DEEPSEEK_V4,
        "glm_5": EXECUTE_PROMPT_GLM_5,
        "glm_52": EXECUTE_PROMPT_GLM_52,
        "minimax_m3": EXECUTE_PROMPT_MINIMAX_M3,
    },
    # CUDA/Triton specialized intent for GLM-5
    "cuda_triton": {
        "default": EXECUTE_PROMPT_DEFAULT,
        "glm_5": CUDA_TRITON_PROMPT_GLM_5,
    },
    # Backward compatibility: code_generation is an alias for execute
    "code_generation": {
        "default": EXECUTE_PROMPT_DEFAULT,
        "glm": EXECUTE_PROMPT_GLM,
        "anthropic": EXECUTE_PROMPT_ANTHROPIC,
        "deepseek": EXECUTE_PROMPT_DEEPSEEK,
        "deepseek_v4": EXECUTE_PROMPT_DEEPSEEK_V4,
        "glm_5": EXECUTE_PROMPT_GLM_5,
        "glm_52": EXECUTE_PROMPT_GLM_52,
        "minimax_m3": EXECUTE_PROMPT_MINIMAX_M3,
    },
}


def get_system_prompt_for_intent(
    intent: str,
    compiler_type: str = "default",
) -> str:
    """Get the system prompt for a given intent and compiler type.

    This is the centralized prompt management function for the teragent library.
    All prompts are managed through teragent/core/prompts/ and selected based
    on intent and compiler type.

    Args:
        intent: One of: design | plan | replan | execute | code_generation |
            review | chat | chat_friendly | sub_agent | cuda_triton
        compiler_type: One of: default | glm | anthropic | deepseek |
            deepseek_v4 | glm_5 | minimax_m3

    Returns:
        System prompt string for the given intent and compiler type.
        Falls back to "default" compiler if specific one not found.
        Returns empty string if intent is not recognized.
    """
    intent_prompts = _PROMPT_REGISTRY.get(intent)
    if intent_prompts is None:
        logger.warning(f"Unknown intent: {intent!r}. Available: {list(_PROMPT_REGISTRY.keys())}")
        return ""

    prompt = intent_prompts.get(compiler_type)
    if prompt is None:
        # Fall back strategy: try base compiler type, then default
        # e.g., deepseek_v4 -> deepseek, glm_5 -> glm, minimax_m3 -> default
        fallback_map = {
            "deepseek_v4": "deepseek",
            "glm_5": "glm",
            "glm_52": "glm_5",
            "minimax_m3": "default",
        }
        fallback_type = fallback_map.get(compiler_type)
        if fallback_type:
            prompt = intent_prompts.get(fallback_type)

        if prompt is None:
            prompt = intent_prompts.get("default", "")

        if prompt and compiler_type != "default":
            logger.debug(
                f"No prompt for intent={intent!r} compiler={compiler_type!r}, "
                f"falling back to default"
            )

    return prompt


def list_intents() -> list[str]:
    """List all available intent names."""
    return list(_PROMPT_REGISTRY.keys())


def list_compiler_types() -> list[str]:
    """List all available compiler type names."""
    return ["default", "glm", "anthropic", "deepseek", "deepseek_v4", "glm_5", "glm_52", "minimax_m3"]


__all__ = [
    # Prompt selection function (Phase 4)
    "get_system_prompt_for_intent",
    "list_intents",
    "list_compiler_types",
    # Design intent
    "DESIGN_PROMPT_DEFAULT", "DESIGN_PROMPT_GLM", "DESIGN_PROMPT_ANTHROPIC", "DESIGN_PROMPT_DEEPSEEK",
    "DESIGN_PROMPT_DEEPSEEK_V4", "DESIGN_PROMPT_GLM_5", "DESIGN_PROMPT_GLM_52", "DESIGN_PROMPT_MINIMAX_M3",
    # Plan intent
    "PLAN_PROMPT_DEFAULT", "PLAN_PROMPT_GLM", "PLAN_PROMPT_ANTHROPIC", "PLAN_PROMPT_DEEPSEEK",
    "PLAN_PROMPT_DEEPSEEK_V4", "PLAN_PROMPT_GLM_5", "PLAN_PROMPT_GLM_52", "PLAN_PROMPT_MINIMAX_M3",
    # Replan intent
    "REPLAN_PROMPT_DEFAULT", "REPLAN_PROMPT_GLM", "REPLAN_PROMPT_ANTHROPIC", "REPLAN_PROMPT_DEEPSEEK",
    "REPLAN_PROMPT_DEEPSEEK_V4", "REPLAN_PROMPT_GLM_5", "REPLAN_PROMPT_GLM_52", "REPLAN_PROMPT_MINIMAX_M3",
    # Review intent
    "REVIEW_PROMPT_DEFAULT", "REVIEW_PROMPT_GLM", "REVIEW_PROMPT_ANTHROPIC", "REVIEW_PROMPT_DEEPSEEK",
    "REVIEW_PROMPT_DEEPSEEK_V4", "REVIEW_PROMPT_GLM_5", "REVIEW_PROMPT_GLM_52", "REVIEW_PROMPT_MINIMAX_M3",
    # Chat intent (agent mode)
    "AGENT_PROMPT_DEFAULT", "AGENT_PROMPT_GLM", "AGENT_PROMPT_ANTHROPIC", "AGENT_PROMPT_DEEPSEEK",
    "AGENT_PROMPT_DEEPSEEK_V4", "AGENT_PROMPT_GLM_5", "AGENT_PROMPT_GLM_52", "AGENT_PROMPT_MINIMAX_M3",
    # Chat intent (friendly mode)
    "CHAT_PROMPT_DEFAULT", "CHAT_PROMPT_GLM", "CHAT_PROMPT_ANTHROPIC", "CHAT_PROMPT_DEEPSEEK",
    "CHAT_PROMPT_DEEPSEEK_V4", "CHAT_PROMPT_GLM_5", "CHAT_PROMPT_GLM_52", "CHAT_PROMPT_MINIMAX_M3",
    # Sub-agent intent
    "SUB_AGENT_PROMPT_DEFAULT", "SUB_AGENT_PROMPT_GLM", "SUB_AGENT_PROMPT_ANTHROPIC", "SUB_AGENT_PROMPT_DEEPSEEK",
    "SUB_AGENT_PROMPT_DEEPSEEK_V4", "SUB_AGENT_PROMPT_GLM_5", "SUB_AGENT_PROMPT_GLM_52", "SUB_AGENT_PROMPT_MINIMAX_M3",
    # Execute intent
    "EXECUTE_PROMPT_DEFAULT", "EXECUTE_PROMPT_GLM", "EXECUTE_PROMPT_ANTHROPIC", "EXECUTE_PROMPT_DEEPSEEK",
    "EXECUTE_PROMPT_DEEPSEEK_V4", "EXECUTE_PROMPT_GLM_5", "EXECUTE_PROMPT_GLM_52", "EXECUTE_PROMPT_MINIMAX_M3",
    # CUDA/Triton specialized prompt
    "CUDA_TRITON_PROMPT_GLM_5",
]
