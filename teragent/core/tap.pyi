"""Type stubs for teragent.core.tap — TAP IR data structures"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

__all__ = [
    "CompiledPrompt",
    "CostTracker",
    "DesktopContext",
    "LongHorizonConfig",
    "LongHorizonStatus",
    "MultimodalContent",
    "TAPCostRecord",
    "TAPRequest",
    "TAPResponse",
]

@dataclass
class MultimodalContent:
    """Multi-modal content block for TAP IR."""

    type: Literal["text", "image_url", "video_url", "image_base64"]
    text: str | None
    url: str | None
    base64_data: str | None
    media_type: str | None

    def to_openai_format(self) -> dict: ...
    def extract_text_description(self) -> str: ...

@dataclass
class DesktopContext:
    """Desktop operation context for MiniMax M3."""

    screenshot: MultimodalContent | None
    interactive_elements: list[dict]
    active_window: str

    def format_for_prompt(self) -> str: ...

@dataclass
class LongHorizonConfig:
    """Configuration for long-horizon autonomous tasks (GLM-5 specific)."""

    max_duration_hours: float
    checkpoint_interval_minutes: float
    self_evaluation_enabled: bool
    stagnation_threshold: int

@dataclass
class LongHorizonStatus:
    """Status of a long-horizon task execution (GLM-5 specific)."""

    phase: str
    steps_completed: int
    elapsed_minutes: float
    last_checkpoint: str
    strategy_switches: int

    @property
    def is_active(self) -> bool: ...
    @property
    def is_stagnant(self) -> bool: ...

@dataclass
class TAPRequest:
    """TerAgent Protocol unified request structure."""

    meta: dict
    context: dict
    instruction: str
    constraints: list[str]
    output_format_hint: str
    thinking_mode: Literal["deep", "quick", "auto"] | None
    multimodal_context: list[MultimodalContent] | None
    desktop_context: DesktopContext | None
    long_horizon: LongHorizonConfig | None
    cache_preference: Literal["auto", "aggressive", "none"] | None

    def estimate_prompt_tokens(self) -> int: ...

    @property
    def has_multimodal(self) -> bool: ...
    @property
    def has_desktop_context(self) -> bool: ...
    @property
    def is_long_horizon(self) -> bool: ...
    @property
    def effective_thinking_mode(self) -> Literal["deep", "quick", "auto"]: ...

@dataclass
class TAPResponse:
    """TerAgent Protocol unified response structure."""

    raw_text: str | None
    usage: dict
    tool_calls: list[dict]
    finish_reason: str
    cache_hit_tokens: int
    thinking_content: str | None
    long_horizon_status: LongHorizonStatus | None
    extra: dict

    def __post_init__(self) -> None: ...

    @property
    def prompt_tokens(self) -> int: ...
    @property
    def completion_tokens(self) -> int: ...
    @property
    def total_tokens(self) -> int: ...
    @property
    def cache_miss_tokens(self) -> int: ...

@dataclass
class TAPCostRecord:
    """Cost record for a single TAP call."""

    task_id: str
    intent: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    success: bool
    error: str
    cache_hit_tokens: int
    cache_miss_tokens: int

@dataclass
class CompiledPrompt:
    """Compiled prompt — protocol-agnostic."""

    messages: list[dict]
    system_prompt: str
    user_message: str
    max_tokens: int
    tools: list[dict]
    tool_choice: str | dict | None
    extra: dict

    @property
    def mode(self) -> str: ...
    @property
    def thinking_enabled(self) -> bool | None: ...

class CostTracker:
    """Thread-safe cost tracker for TAP calls."""

    def __init__(self) -> None: ...
    def append(self, record: TAPCostRecord) -> None: ...
    def get_all(self) -> list[TAPCostRecord]: ...
    def clear(self) -> None: ...
    def __len__(self) -> int: ...
    def total_cache_hit_tokens(self) -> int: ...
    def total_cache_miss_tokens(self) -> int: ...
    def cache_hit_rate(self) -> float: ...
    def total_estimated_cost(self, pricing: dict) -> float: ...
