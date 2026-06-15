"""teragent.pipeline.prompt_builder — Prompt construction with template externalization

Key changes:
    - SYSTEM_TEMPLATE is now a parameter, not a hardcoded constant
    - build_prompt() accepts any system_template string + context dict
    - build_subagent_prompt() preserved for backward compatibility
    - validate_prompt_tokens() preserved unchanged

Library design principle: the library provides the mechanism (token validation,
context injection), the caller provides the persona (system template).
"""
import logging

__all__ = [
    "build_prompt",
    "build_subagent_prompt",
    "validate_prompt_tokens",
]

logger = logging.getLogger(__name__)

# Flat average estimate (~3 chars/token); for CJK-aware counting, use utils.token_counter
CHARS_PER_TOKEN = 3

DEFAULT_TOKEN_BUDGET = 120_000

# Default system template — kept for backward compatibility with build_subagent_prompt()
DEFAULT_SYSTEM_TEMPLATE = """你是一位专业软件工程师，根据设计和计划完成当前子任务。

<memory>
{agent_md}
</memory>

<design>
{design_md}
</design>

<plan>
{plan_md}
</plan>

<dependency_report>
{code_summary}
</dependency_report>

当前子任务：{task_desc}

## 输出格式
1. 代码文件：<file path="相对路径">完整代码</file>
2. 终端命令：<command cwd="工作目录">命令</command>
3. 每个文件只输出一次，严禁省略（# ...）或 TODO 占位
4. 不输出废话、解释或总结

## 代码质量
- Python 3.10+（X | None 而非 Optional[X]）
- 公开函数须有类型注解和 docstring
- 用 logging 替代 print()
- except Exception as e，禁止裸 except
- 外部调用（I/O、网络、第三方库）须 try/except
- 需第三方包时用 <command cwd=".">pip install 包名</command>"""


def build_prompt(
    system_template: str,
    context: dict,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> list[dict[str, str]]:
    """Build prompt messages with an externalized system template.

    This is the library-level API: the caller provides the template and context,
    the library handles the assembly and token budget validation.

    Args:
        system_template: System prompt template string with {placeholders}
            Supported placeholders depend on the template, common ones:
            {agent_md}, {design_md}, {plan_md}, {code_summary}, {task_desc}
        context: Dict of values to fill into the template
        token_budget: Maximum token budget for the prompt

    Returns:
        A list of message dicts with "role" and "content" keys
    """
    def safe_fill(text: str) -> str:
        return text.strip() if text and text.strip() else "N/A"

    class _SafeDict(dict):
        def __missing__(self, key):
            return f"{{{key}}}"

    # Fill context values with safe defaults; convert None to "N/A" explicitly
    filled_context = {
        k: safe_fill(v) if isinstance(v, str) else ("N/A" if v is None else str(v))
        for k, v in context.items()
    }

    system_content = system_template.format_map(_SafeDict(filled_context))
    task_desc = context.get("task_desc", "")
    user_content = f"执行子任务：{task_desc}" if task_desc else ""

    # Validate token budget
    total_chars = len(system_content) + len(user_content)
    validate_prompt_tokens(total_chars, token_budget)

    messages = [
        {"role": "system", "content": system_content},
    ]
    if user_content:
        messages.append({"role": "user", "content": user_content})
    return messages


def build_subagent_prompt(
    design_md: str,
    plan_md: str,
    task_desc: str,
    code_summary: str,
    agent_md: str = "",
    system_template: str = DEFAULT_SYSTEM_TEMPLATE,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> list[dict[str, str]]:
    """Build the prompt messages for a sub-agent execution.

    Backward-compatible API. Now supports custom system_template parameter.

    Args:
        design_md: The DESIGN.md content providing architectural context.
        plan_md: The PLAN.md content describing the execution plan.
        task_desc: Description of the specific sub-task to execute.
        code_summary: Dependency report summarizing existing code.
        agent_md: Optional AGENT.md memory content.
        system_template: System prompt template (default: DEFAULT_SYSTEM_TEMPLATE)
        token_budget: Maximum token budget for the prompt.

    Returns:
        A list of message dicts with "role" and "content" keys.
    """
    context = {
        "agent_md": agent_md,
        "design_md": design_md,
        "plan_md": plan_md,
        "code_summary": code_summary,
        "task_desc": task_desc,
    }
    return build_prompt(system_template, context, token_budget)


def validate_prompt_tokens(char_count: int, token_budget: int = DEFAULT_TOKEN_BUDGET) -> None:
    """Check if the estimated token count for a prompt is within budget.

    Uses a rough character-to-token ratio to estimate tokens. Emits a warning
    if the prompt is estimated to exceed 80% of the budget, and a stronger
    warning if it exceeds the full budget.

    Args:
        char_count: Total character count of the prompt.
        token_budget: Maximum allowed tokens for the prompt.
    """
    estimated_tokens = char_count // CHARS_PER_TOKEN
    threshold_80 = int(token_budget * 0.8)

    if estimated_tokens > token_budget:
        logger.warning(
            f"Prompt estimated at ~{estimated_tokens} tokens, which EXCEEDS "
            f"the budget of {token_budget} tokens. The prompt may be truncated "
            f"or cause an API error. Consider reducing context size."
        )
    elif estimated_tokens > threshold_80:
        logger.warning(
            f"Prompt estimated at ~{estimated_tokens} tokens, which is above "
            f"80% of the budget ({token_budget}). Consider reducing context "
            f"size to leave room for the model's response."
        )
    else:
        logger.debug(
            f"Prompt estimated at ~{estimated_tokens} tokens "
            f"(budget: {token_budget}, usage: {estimated_tokens / token_budget:.0%})"
        )
