"""examples.full_agent.design_generator — Design generation using library APIs

Reference implementation that combines:
    - teragent.pipeline.retry.retry_with_backoff (generic retry)
    - teragent.core.provider.ModelProvider.execute_tap (TAP-compiled prompts)
    - teragent.core.tap.TAPRequest (unified request IR)
    - teragent.event_bus (orchestration)

Phase 4 change: Now uses execute_tap(TAPRequest(intent="design")) instead of
chat(messages). The system prompt comes from the Compiler's _default_prompts
via get_system_prompt("design"), ensuring model-specific prompt optimization.
"""
import logging

from teragent.core.provider import ModelProvider
from teragent.core.tap import TAPRequest
from teragent.event_bus import EventBus
from teragent.pipeline.retry import retry_with_backoff
from teragent.utils.text import strip_code_block

logger = logging.getLogger(__name__)

REQUIRED_DESIGN_SECTIONS = ["背景与动机", "设计目标", "技术选型", "目录结构", "核心接口契约"]


class DesignGenerator:
    """Reference implementation: Design generation using TAP-compiled prompts.

    Uses retry_with_backoff from teragent.pipeline.retry for robust retry logic
    and EventBus for event-driven orchestration.

    Phase 4: The system prompt is no longer hardcoded here — it comes from
    the Compiler's _default_prompts["design"], which provides model-specific
    optimizations (e.g., GLM recency effect, Anthropic XML tags).
    """

    def __init__(self, bus: EventBus, model: ModelProvider) -> None:
        self.bus = bus
        self.model = model
        self._generation_count = 0
        self._is_generating: bool = False  # Prevent concurrent generation
        bus.on("request_design", self.on_request)

    async def on_request(self, requirement: str) -> None:
        # Prevent concurrent generation
        if self._is_generating:
            logger.warning(
                f"Design generation already in progress, ignoring: {requirement[:50]}..."
            )
            return

        self._is_generating = True
        logger.info(f"Generating DESIGN for: {requirement[:100]}...")
        self._generation_count += 1

        # Phase 4: Use execute_tap() with TAPRequest instead of chat()
        # The Compiler automatically injects the intent-specific system prompt
        tap_request = TAPRequest(
            meta={"task_id": "design", "intent": "design"},
            context={},
            instruction=f"需求：{requirement}",
            constraints=[
                "严禁代码实现，只输出设计和签名",
                "不要将项目目录名当作包名写入 import",
            ],
            output_format_hint="DESIGN.md 格式，包含背景与动机、设计目标、技术选型、目录结构、核心接口契约、依赖管理、运行方式",
        )

        try:
            async def _generate() -> str:
                response = await self.model.execute_tap(tap_request)
                content = response.raw_text or ""
                return strip_code_block(content)

            def _validate(content: str) -> list[str]:
                missing = _validate_sections(content)
                return [f"Missing sections: {missing}"] if missing else []

            clean_content = await retry_with_backoff(
                fn=_generate,
                max_retries=3,
                backoff_base=10.0,
                validate=_validate,
                on_retry=lambda attempt, err: logger.warning(
                    f"Design retry {attempt + 1}: {err}"
                ),
            )

            logger.info(
                f"Design generated successfully ({len(clean_content)} chars)"
            )
            await self.bus.emit("design_ready", clean_content)

        except Exception as e:
            logger.error(f"Design generation failed after all retries: {e}")
            await self.bus.emit("design_generation_failed", str(e))
        finally:
            self._is_generating = False

    @staticmethod
    def _validate_sections_static(content: str) -> list[str]:
        """Check that all required design sections are present."""
        missing: list[str] = []
        for section in REQUIRED_DESIGN_SECTIONS:
            if section not in content:
                missing.append(section)
        return missing


# Module-level alias for backward compatibility
_validate_sections = DesignGenerator._validate_sections_static
