"""teragent.core.tap — TAP (TerAgent Protocol) IR data structures

TAP is the intermediate representation (IR) between user intent and model API calls.
Like LLVM IR, TAP is a client-side in-memory data structure, not a wire protocol.

Flow:
    TAPRequest → Compiler (compile) → CompiledPrompt → Adapter (send/stream) → TAPResponse
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TAPRequest:
    """TerAgent Protocol unified request structure

    The IR that captures *what* the user wants, not *how* to ask the model.
    The Compiler decides the best prompt format for each model.

    Attributes:
        meta: Task metadata, e.g. {"task_id": "1.1", "intent": "code_generation"}
        context: Reference material, e.g. {"design": "...", "plan": "...", "dependency_report": "...", "memory": "..."}
        instruction: The core instruction / user request
        constraints: Hard constraints the output must satisfy
        output_format_hint: Desired output format description
    """

    meta: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    instruction: str = ""
    constraints: list[str] = field(default_factory=list)
    output_format_hint: str = ""

    def estimate_prompt_tokens(self) -> int:
        """Rough token count estimation for this request"""
        from teragent.utils.token_counter import estimate_tokens

        total = estimate_tokens(self.instruction)
        total += estimate_tokens(str(self.constraints))
        total += estimate_tokens(self.output_format_hint)
        for v in self.context.values():
            if isinstance(v, str):
                total += estimate_tokens(v)
        return total


@dataclass
class TAPResponse:
    """TerAgent Protocol unified response structure

    Attributes:
        raw_text: Model's raw text output (None = API error / abnormal response)
        usage: Token usage dict, e.g. {"prompt_tokens": int, "completion_tokens": int}
        tool_calls: Structured tool calls from the API response
        finish_reason: Why the model stopped generating (e.g. "stop", "length")
    """

    raw_text: str | None = ""
    usage: dict = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = ""

    def __post_init__(self) -> None:
        """Log warning if raw_text is None (possible API error)"""
        if self.raw_text is None:
            logger.warning(
                "TAPResponse.raw_text is None — possible API error or empty response. "
                "Caller should handle this explicitly."
            )

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class TAPCostRecord:
    """Cost record for a single TAP call"""

    task_id: str = ""
    intent: str = ""
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class CompiledPrompt:
    """Compiled prompt — protocol-agnostic

    Two intended-exclusive expression modes (mutual exclusivity not enforced at runtime;
    if both are set, `mode` property returns "messages"):
    - Mode A: messages list (for OpenAI / GLM / DeepSeek — chat message array)
    - Mode B: system_prompt + user_message (for Anthropic native — system field separate)

    The Compiler determines which mode to use by populating the corresponding fields.
    Mode A compilers fill `messages`; Mode B compilers fill `system_prompt` + `user_message`.
    The Adapter reads the mode from `compiled.mode` and formats the request accordingly.

    Attributes:
        messages: Chat message array, e.g. [{"role": "system", "content": "..."}, ...]
        system_prompt: System prompt string (Anthropic native protocol)
        user_message: User message string (Anthropic native protocol)
        max_tokens: Maximum output tokens
        tools: Optional tool definitions for function calling
        tool_choice: Optional tool choice strategy
    """

    # Mode A: message array
    messages: list[dict] = field(default_factory=list)

    # Mode B: system + user separation
    system_prompt: str = ""
    user_message: str = ""

    # Common fields
    max_tokens: int = 8192
    tools: list[dict] = field(default_factory=list)
    tool_choice: str | dict | None = None

    @property
    def mode(self) -> str:
        """Determine which mode this CompiledPrompt uses"""
        if self.messages:
            return "messages"
        if self.system_prompt or self.user_message:
            return "system_user"
        return "empty"


class CostTracker:
    """Thread-safe cost tracker for TAP calls"""

    def __init__(self) -> None:
        self._records: list[TAPCostRecord] = []
        self._lock = threading.Lock()

    def append(self, record: TAPCostRecord) -> None:
        with self._lock:
            self._records.append(record)

    def get_all(self) -> list[TAPCostRecord]:
        with self._lock:
            return list(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
