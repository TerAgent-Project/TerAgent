"""examples.full_agent.plan_generator — Plan generation using library APIs

Reference implementation that combines:
    - teragent.pipeline.retry.retry_with_backoff (generic retry)
    - teragent.core.provider.ModelProvider.execute_tap (TAP-compiled prompts)
    - teragent.core.tap.TAPRequest (unified request IR)
    - teragent.event_bus (orchestration)

Phase 4 change: Now uses execute_tap(TAPRequest(intent="plan"/"replan")) instead
of chat(messages). The system prompt comes from the Compiler's _default_prompts
via get_system_prompt("plan"/"replan"), ensuring model-specific prompt optimization.
"""
import logging

from teragent.core.provider import ModelProvider
from teragent.core.tap import TAPRequest
from teragent.event_bus import EventBus
from teragent.pipeline.retry import retry_with_backoff
from teragent.utils.text import strip_code_block

logger = logging.getLogger(__name__)


class PlanGenerator:
    """Reference implementation: Plan generation using TAP-compiled prompts.

    Uses retry_with_backoff from teragent.pipeline.retry for robust retry logic
    and EventBus for event-driven orchestration. Supports both initial planning and replanning.

    Phase 4: The system prompt is no longer hardcoded here — it comes from
    the Compiler's _default_prompts["plan"] or _default_prompts["replan"],
    which provides model-specific optimizations.
    """

    def __init__(self, bus: EventBus, model: ModelProvider) -> None:
        self.bus = bus
        self.model = model
        self._plan_count = 0
        self._replan_count = 0
        self._is_generating: bool = False  # Prevent concurrent generation
        bus.on("request_plan_generation", self.on_request)
        bus.on("request_replan", self.on_replan_request)

    async def on_request(self, design_md: str) -> None:
        if self._is_generating:
            logger.warning("Plan generation already in progress, ignoring request.")
            return

        self._is_generating = True
        logger.info("Generating PLAN based on DESIGN...")
        self._plan_count += 1

        # Phase 4: Use execute_tap() with TAPRequest(intent="plan")
        tap_request = TAPRequest(
            meta={"task_id": "plan", "intent": "plan"},
            context={"design": design_md},
            instruction="请根据以上 DESIGN.md 生成 PLAN.md",
            constraints=[
                "编号从 1.1 递增，连续不跳过",
                "原子化：每任务 1-3 个文件",
                "严禁废话、总结或代码实现",
            ],
            output_format_hint="### 任务编号 标题 + 输出文件排他声明 + 前置依赖 + 实现要点 + 优先级",
        )

        try:
            clean_content = await self._generate_with_retry(tap_request)
            await self.bus.emit("plan_ready", clean_content)
            logger.info(f"Plan generated successfully ({len(clean_content)} chars)")
        except Exception as e:
            logger.error(f"Plan generation failed after all retries: {e}")
            await self.bus.emit("plan_generation_failed", str(e))
        finally:
            self._is_generating = False

    async def on_replan_request(self, group_id: str, error: str, old_plan_md: str) -> None:
        logger.info(f"Handling replan request for group {group_id}...")
        self._replan_count += 1

        # Phase 4: Use execute_tap() with TAPRequest(intent="replan")
        tap_request = TAPRequest(
            meta={"task_id": f"replan_{group_id}", "intent": "replan"},
            context={"plan": old_plan_md},
            instruction=(
                f"当前计划在 {group_id} 处失败，错误：{error}\n\n"
                "请生成局部修复计划。只输出从失败任务开始的替代任务。"
            ),
            constraints=[
                "只输出从失败任务开始的替代任务",
                "优先解决失败原因",
                "保持与已成功任务的接口一致",
                "严禁输出代码实现",
            ],
            output_format_hint="### 编号 标题 + 输出文件排他声明 + 前置依赖 + 实现要点 + 优先级",
        )

        try:
            clean_content = await self._generate_with_retry(tap_request)
            logger.warning("Replan generated. Replacing entire plan for M2 simplicity.")
            # Emit a distinct "replan_ready" event so downstream consumers can
            # differentiate a replanned plan from an initial plan. Consumers
            # that only care about "any new plan" can subscribe to both
            # "plan_ready" and "replan_ready".
            await self.bus.emit("replan_ready", clean_content)
            logger.info(f"Replan generated successfully ({len(clean_content)} chars)")
        except Exception as e:
            logger.error(f"Replan generation failed after all retries: {e}")
            await self.bus.emit("plan_generation_failed", str(e))

    async def _generate_with_retry(self, tap_request: TAPRequest) -> str:
        """Generate plan content with retry and validation using TAP API."""

        async def _generate() -> str:
            response = await self.model.execute_tap(tap_request)
            content = response.raw_text or ""
            return strip_code_block(content)

        def _validate(content: str) -> list[str]:
            return _validate_plan(content)

        return await retry_with_backoff(
            fn=_generate,
            max_retries=3,
            backoff_base=15.0,
            validate=_validate,
            on_retry=lambda attempt, err: logger.warning(
                f"Plan retry {attempt + 1}: {err}"
            ),
        )


def _validate_plan(plan_content: str) -> list[str]:
    """Validate that the generated plan has proper formatting.

    Expects pre-stripped content (caller already applies strip_code_block).

    Returns a list of validation error descriptions. An empty list means the plan
    passes all validation checks.
    """
    content = plan_content
    errors: list[str] = []

    if "### " not in content:
        errors.append("No task headers (###) found in plan")

    # Accept the correct term "输出文件排他声明" plus the common LLM-output
    # variant "输出文件声明" (without the "排他" qualifier). The earlier
    # "排它" / "排再" entries were typos and have been removed.
    has_output_decl = any(
        kw in content
        for kw in ("输出文件排他声明", "输出文件声明")
    )
    if not has_output_decl:
        errors.append("No output file declarations found in plan")

    if "优先级" not in content:
        errors.append("No priority declarations found in plan")

    if "前置依赖" not in content and "输入" not in content:
        errors.append("No dependency declarations found in plan")

    return errors
