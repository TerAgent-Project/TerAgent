"""teragent.core.tap — TAP (TerAgent Protocol) IR data structures

TAP is the intermediate representation (IR) between user intent and model API calls.
Like LLVM IR, TAP is a client-side in-memory data structure, not a wire protocol.

Flow:
    TAPRequest → Compiler (compile) → CompiledPrompt → Adapter (send/stream) → TAPResponse

Extended for DeepSeek V4 / MiniMax M3 / GLM-5 deep adaptation:
    - MultimodalContent: Multi-modal input (image/video/text)
    - DesktopContext: Desktop operation context (M3 specific)
    - LongHorizonConfig: Long-horizon task configuration (GLM-5 specific)
    - LongHorizonStatus: Long-horizon task execution status
    - TAPRequest extensions: thinking_mode, multimodal_context, long_horizon, cache_preference
    - TAPResponse extensions: cache_hit_tokens, thinking_content, long_horizon_status
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Literal, Optional

logger = logging.getLogger(__name__)


# ===== Multi-modal Support =====

@dataclass
class MultimodalContent:
    """Multi-modal content block for TAP IR

    Represents a single content unit that can be text, image URL, video URL,
    or base64-encoded image. Used by MiniMax M3's native multimodal support,
    and gracefully degraded by non-multimodal Compilers (V4, GLM-5).

    Attributes:
        type: Content type — "text", "image_url", "video_url", "image_base64"
        text: Text content (when type="text")
        url: URL for image/video (when type="image_url" or "video_url")
        base64_data: Base64-encoded image data (when type="image_base64")
        media_type: MIME type for base64 data (e.g., "image/png", "image/jpeg")
    """

    type: Literal["text", "image_url", "video_url", "image_base64"] = "text"
    text: Optional[str] = None
    url: Optional[str] = None
    base64_data: Optional[str] = None
    media_type: Optional[str] = None

    def to_openai_format(self) -> dict:
        """Convert to OpenAI API content block format

        Returns:
            Dict suitable for use in messages[].content[] arrays.
            For "text": {"type": "text", "text": "..."}
            For "image_url": {"type": "image_url", "image_url": {"url": "..."}}
            For "video_url": {"type": "video_url", "video_url": {"url": "..."}}
            For "image_base64": {"type": "image_url", "image_url": {"url": "data:...;base64,..."}}
        """
        if self.type == "text":
            return {"type": "text", "text": self.text or ""}
        elif self.type == "image_url":
            return {"type": "image_url", "image_url": {"url": self.url or ""}}
        elif self.type == "video_url":
            return {"type": "video_url", "video_url": {"url": self.url or ""}}
        elif self.type == "image_base64":
            media_type = self.media_type or "image/png"
            data_uri = f"data:{media_type};base64,{self.base64_data or ''}"
            return {"type": "image_url", "image_url": {"url": data_uri}}
        else:
            return {"type": "text", "text": str(self.text or self.url or "")}

    def extract_text_description(self) -> str:
        """Extract a text description for degradation by non-multimodal compilers.

        When a non-multimodal Compiler (V4, GLM-5) receives multimodal content,
        it should call this method to get a text approximation and log a warning.

        Returns:
            Text description of the content for fallback use.
        """
        if self.type == "text":
            return self.text or ""
        elif self.type == "image_url":
            return f"[图片: {self.url}]"
        elif self.type == "video_url":
            return f"[视频: {self.url}]"
        elif self.type == "image_base64":
            return f"[Base64图片: {self.media_type or 'image/png'}, {len(self.base64_data or '')} bytes]"
        return ""


@dataclass
class DesktopContext:
    """Desktop operation context for MiniMax M3

    Contains the information needed for M3's desktop operation capability:
    screen state, interactive elements, and active window info.

    This is M3-specific; other Compilers should ignore this field.

    Attributes:
        screenshot: Screen capture as MultimodalContent
        interactive_elements: List of clickable/interactable UI elements
            Each dict has keys: "type", "label", "bbox" (x, y, w, h), "action"
        active_window: Name/title of the currently active window
    """

    screenshot: Optional[MultimodalContent] = None
    interactive_elements: list[dict] = field(default_factory=list)
    active_window: str = ""

    def format_for_prompt(self) -> str:
        """Format desktop context as text for prompt injection

        Returns:
            Formatted string describing the desktop state
        """
        parts: list[str] = []

        if self.screenshot:
            parts.append("[当前屏幕截图已提供]")

        if self.active_window:
            parts.append(f"活动窗口: {self.active_window}")

        if self.interactive_elements:
            parts.append("可交互元素:")
            for i, elem in enumerate(self.interactive_elements, 1):
                label = elem.get("label", f"元素{i}")
                etype = elem.get("type", "unknown")
                bbox = elem.get("bbox", {})
                parts.append(f"  {i}. [{etype}] {label} @ ({bbox.get('x', 0)}, {bbox.get('y', 0)})")

        return "\n".join(parts)


# ===== Long-Horizon Task Support =====

@dataclass
class LongHorizonConfig:
    """Configuration for long-horizon autonomous tasks (GLM-5 specific)

    GLM-5 supports 8-hour continuous work capability. This config
    controls how the Compiler and LongHorizonTaskManager handle such tasks.

    Attributes:
        max_duration_hours: Maximum task duration in hours (default 8)
        checkpoint_interval_minutes: Minutes between automatic checkpoints
        self_evaluation_enabled: Whether to inject self-evaluation prompts at checkpoints
        stagnation_threshold: Number of consecutive identical results before triggering strategy switch
    """

    max_duration_hours: float = 8.0
    checkpoint_interval_minutes: float = 30.0
    self_evaluation_enabled: bool = True
    stagnation_threshold: int = 3


@dataclass
class LongHorizonStatus:
    """Status of a long-horizon task execution (GLM-5 specific)

    Tracks the progress and state of long-running autonomous tasks.

    Attributes:
        phase: Current phase — "planning", "executing", "evaluating", "stagnant", "completed", "failed"
        steps_completed: Number of steps completed so far
        elapsed_minutes: Elapsed time in minutes since task start
        last_checkpoint: ISO timestamp of the last checkpoint
        strategy_switches: Number of times the strategy has been switched
    """

    phase: str = "planning"
    steps_completed: int = 0
    elapsed_minutes: float = 0.0
    last_checkpoint: str = ""
    strategy_switches: int = 0

    @property
    def is_active(self) -> bool:
        """Whether the task is still actively running"""
        return self.phase in ("planning", "executing", "evaluating")

    @property
    def is_stagnant(self) -> bool:
        """Whether the task has stagnated and needs strategy switch"""
        return self.phase == "stagnant"


# ===== Core TAP Data Structures =====

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
        thinking_mode: Reasoning depth control — "deep" (full CoT), "quick" (no CoT), "auto" (Compiler decides)
        multimodal_context: Multi-modal content blocks (images, videos) for M3
        desktop_context: Desktop operation context (M3 specific)
        long_horizon: Long-horizon task configuration (GLM-5 specific)
        cache_preference: Cache behavior hint — "auto", "aggressive" (maximize cache hits), "none"
    """

    meta: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    instruction: str = ""
    constraints: list[str] = field(default_factory=list)
    output_format_hint: str = ""

    # --- DeepSeek V4 / GLM-5: Thinking mode control ---
    thinking_mode: Optional[Literal["deep", "quick", "auto"]] = None

    # --- MiniMax M3: Multi-modal content ---
    multimodal_context: Optional[list[MultimodalContent]] = None

    # --- MiniMax M3: Desktop operation context ---
    desktop_context: Optional[DesktopContext] = None

    # --- GLM-5: Long-horizon autonomous task configuration ---
    long_horizon: Optional[LongHorizonConfig] = None

    # --- DeepSeek V4: Cache preference for cost optimization ---
    cache_preference: Optional[Literal["auto", "aggressive", "none"]] = None

    def estimate_prompt_tokens(self) -> int:
        """Rough token count estimation for this request"""
        from teragent.utils.token_counter import estimate_tokens

        total = estimate_tokens(self.instruction)
        total += estimate_tokens(str(self.constraints))
        total += estimate_tokens(self.output_format_hint)
        for v in self.context.values():
            if isinstance(v, str):
                total += estimate_tokens(v)
        # Estimate multimodal content tokens (images ≈ 1000 tokens each)
        if self.multimodal_context:
            for mc in self.multimodal_context:
                if mc.type == "text" and mc.text:
                    total += estimate_tokens(mc.text)
                elif mc.type in ("image_url", "image_base64"):
                    total += 1000  # Rough estimate for image tokens
                elif mc.type == "video_url":
                    total += 3000  # Rough estimate for video tokens
        return total

    @property
    def has_multimodal(self) -> bool:
        """Whether this request contains multimodal content"""
        return bool(self.multimodal_context)

    @property
    def has_desktop_context(self) -> bool:
        """Whether this request contains desktop operation context"""
        return self.desktop_context is not None

    @property
    def is_long_horizon(self) -> bool:
        """Whether this request is a long-horizon autonomous task"""
        return self.long_horizon is not None

    @property
    def effective_thinking_mode(self) -> Literal["deep", "quick"]:
        """Resolve thinking_mode to a concrete value.

        If thinking_mode is None or "auto", the Compiler should decide.
        This property returns the explicit mode if set, otherwise "auto".

        Returns:
            The thinking mode: "deep", "quick", or "auto" for Compiler to decide.
        """
        return self.thinking_mode or "auto"


@dataclass
class TAPResponse:
    """TerAgent Protocol unified response structure

    Attributes:
        raw_text: Model's raw text output (None = API error / abnormal response)
        usage: Token usage dict, e.g. {"prompt_tokens": int, "completion_tokens": int}
        tool_calls: Structured tool calls from the API response
        finish_reason: Why the model stopped generating (e.g. "stop", "length")
        cache_hit_tokens: Number of prompt tokens served from cache (DeepSeek V4)
        thinking_content: The reasoning/thinking content extracted from the response
            (DeepSeek V4, GLM-5 when thinking_mode="deep")
        long_horizon_status: Status update for long-horizon tasks (GLM-5)
    """

    raw_text: str | None = ""
    usage: dict = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = ""

    # --- DeepSeek V4: Cache hit tracking ---
    cache_hit_tokens: int = 0

    # --- DeepSeek V4 / GLM-5: Thinking content ---
    thinking_content: Optional[str] = None

    # --- GLM-5: Long-horizon status ---
    long_horizon_status: Optional[LongHorizonStatus] = None

    def __post_init__(self) -> None:
        """Log warning if raw_text is None (possible API error)"""
        if self.raw_text is None:
            logger.warning(
                "TAPResponse.raw_text is None — possible API error or empty response. "
                "Caller should handle this explicitly."
            )

        # Extract cache_hit_tokens from usage if present (DeepSeek V4 API returns it there)
        if self.usage and self.cache_hit_tokens == 0:
            self.cache_hit_tokens = self.usage.get("prompt_cache_hit_tokens", 0)

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cache_miss_tokens(self) -> int:
        """Tokens that were NOT served from cache (prompt_tokens - cache_hit_tokens)"""
        return max(0, self.prompt_tokens - self.cache_hit_tokens)


@dataclass
class TAPCostRecord:
    """Cost record for a single TAP call

    Attributes:
        task_id: Task identifier
        intent: Task intent (e.g. 'execute', 'design')
        provider: Provider name
        model: Model name
        prompt_tokens: Total prompt tokens consumed
        completion_tokens: Total completion tokens generated
        latency_ms: Request latency in milliseconds
        success: Whether the call succeeded
        error: Error message if the call failed
        cache_hit_tokens: Prompt tokens served from cache (DeepSeek V4)
        cache_miss_tokens: Prompt tokens NOT served from cache (DeepSeek V4)
    """

    task_id: str = ""
    intent: str = ""
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""
    # --- DeepSeek V4: 缓存命中追踪 ---
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0


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

    # --- Extended fields for model-specific API parameters ---
    # Compilers populate this dict with model-specific parameters that Adapters
    # should pass through to the API (e.g., thinking mode, cache settings).
    # This keeps CompiledPrompt model-agnostic while allowing Compiler → Adapter
    # parameter passing without adding a new field for every model feature.
    extra: dict = field(default_factory=dict)

    @property
    def mode(self) -> str:
        """Determine which mode this CompiledPrompt uses"""
        if self.messages:
            return "messages"
        if self.system_prompt or self.user_message:
            return "system_user"
        return "empty"

    # --- Convenience accessors for common extra fields ---

    @property
    def thinking_enabled(self) -> bool | None:
        """Whether thinking/reasoning mode is enabled.

        Returns True if extra["thinking"]["type"] == "enabled",
        False if "disabled", None if not set.
        """
        thinking = self.extra.get("thinking")
        if isinstance(thinking, dict):
            t = thinking.get("type")
            if t == "enabled":
                return True
            if t == "disabled":
                return False
        return None


class CostTracker:
    """Thread-safe cost tracker for TAP calls

    支持缓存命中追踪和基于定价表的成本估算。
    cache_hit_tokens 通常比 cache_miss_tokens 便宜（DeepSeek V4 定价模型）。
    """

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

    # ----- 缓存命中统计方法 -----

    def total_cache_hit_tokens(self) -> int:
        """所有记录的缓存命中 token 总数

        Returns:
            所有 TAPCostRecord 的 cache_hit_tokens 之和
        """
        with self._lock:
            return sum(r.cache_hit_tokens for r in self._records)

    def total_cache_miss_tokens(self) -> int:
        """所有记录的缓存未命中 token 总数

        Returns:
            所有 TAPCostRecord 的 cache_miss_tokens 之和
        """
        with self._lock:
            return sum(r.cache_miss_tokens for r in self._records)

    def cache_hit_rate(self) -> float:
        """缓存命中率（0.0 ~ 1.0）

        计算方式：cache_hit_tokens / (cache_hit_tokens + cache_miss_tokens)
        如果总 prompt token 数为 0，返回 0.0。

        Returns:
            缓存命中率，范围 [0.0, 1.0]
        """
        with self._lock:
            total_hit = sum(r.cache_hit_tokens for r in self._records)
            total_miss = sum(r.cache_miss_tokens for r in self._records)
            total_prompt = total_hit + total_miss
            if total_prompt == 0:
                return 0.0
            return total_hit / total_prompt

    def total_estimated_cost(self, pricing: dict) -> float:
        """基于定价表估算总成本

        pricing 字典格式示例::
            {
                "prompt_per_million": 1.0,      # 每百万 prompt token 价格
                "completion_per_million": 2.0,   # 每百万 completion token 价格
                "cache_hit_per_million": 0.1,    # 每百万缓存命中 token 价格
                "cache_miss_per_million": 1.0,   # 每百万缓存未命中 token 价格
            }

        如果 pricing 中包含 cache_hit_per_million / cache_miss_per_million，
        则优先使用它们计算 prompt 成本（缓存命中更便宜）。
        否则回退到 prompt_per_million 统一计算。

        Args:
            pricing: 定价表字典，包含每百万 token 的价格

        Returns:
            估算总成本（货币单位与 pricing 一致）
        """
        with self._lock:
            total_completion = sum(r.completion_tokens for r in self._records)
            total_hit = sum(r.cache_hit_tokens for r in self._records)
            total_miss = sum(r.cache_miss_tokens for r in self._records)

        # 计算完成 token 成本
        completion_cost = (
            total_completion * pricing.get("completion_per_million", 0.0) / 1_000_000
        )

        # 计算 prompt token 成本：优先使用缓存感知定价
        if "cache_hit_per_million" in pricing and "cache_miss_per_million" in pricing:
            prompt_cost = (
                total_hit * pricing["cache_hit_per_million"] / 1_000_000
                + total_miss * pricing["cache_miss_per_million"] / 1_000_000
            )
        else:
            # 回退：使用统一的 prompt 价格（hit + miss = 总 prompt token）
            total_prompt = total_hit + total_miss
            prompt_cost = (
                total_prompt * pricing.get("prompt_per_million", 0.0) / 1_000_000
            )

        return prompt_cost + completion_cost
