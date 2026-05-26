"""examples.full_agent.reviewer — Review using library APIs

Reference implementation that combines:
    - teragent.core.provider.ModelProvider.execute_tap (TAP-compiled prompts)
    - teragent.core.tap.TAPRequest (unified request IR)
    - teragent.event_bus (orchestration)

Phase 4 change: Now uses execute_tap(TAPRequest(intent="review")) instead of
chat(messages). The system prompt comes from the Compiler's _default_prompts
via get_system_prompt("review"), ensuring model-specific prompt optimization.

Note: The reviewer's core value is its prompt engineering and structured
suggestion parsing, not retry logic (it doesn't use retry_with_backoff
because a failed review is an auto-approve, not a retry scenario).
"""
import logging
import re
from dataclasses import dataclass, field

from teragent.core.provider import ModelProvider
from teragent.core.tap import TAPRequest

from teragent.event_bus import EventBus

logger = logging.getLogger(__name__)

# Follow-up prompt when REJECT lacks suggestions
_SUGGESTION_FOLLOWUP_PROMPT = (
    "你拒绝了该方案但没有给出具体的修改建议。请重新审查，"
    "如果仍然拒绝，必须列出具体的修改建议。格式：\n"
    "1. [问题描述] -> [具体修改建议]\n"
    "2. [问题描述] -> [具体修改建议]\n"
    "每条建议必须包含问题位置和明确修改方向。"
)

# Patterns that match various APPROVE formats from different LLMs
_APPROVE_PATTERNS = [
    re.compile(r"\bAPPROVE\b", re.IGNORECASE),
    re.compile(r"\bAPPROVED\b", re.IGNORECASE),
    re.compile(r"\b通过\b"),
    re.compile(r"\b同意\b"),
    re.compile(r"\bLGTM\b", re.IGNORECASE),
]

# Patterns that match various REJECT formats
_REJECT_PATTERNS = [
    re.compile(r"\bREJECT\b", re.IGNORECASE),
    re.compile(r"\bREJECTED\b", re.IGNORECASE),
    re.compile(r"\b拒绝\b"),
    re.compile(r"\b不通过\b"),
    re.compile(r"\b需要修改\b"),
]

# Patterns for extracting structured suggestions from text
_SUGGESTION_PATTERNS = [
    # Numbered list: "1. [problem] -> [suggestion]"
    re.compile(
        r"(?:^|\n)\s*\d+[\.\)、]\s*(.+?)(?:\s*(?:->|->|-->)\s*|\s*[-:]\s*)(.+?)(?=\n\s*\d+[\.\)、]|\n\n|$)",
        re.MULTILINE,
    ),
    # Bullet list: "- [problem] -> [suggestion]"
    re.compile(
        r"(?:^|\n)\s*[-*]\s*(.+?)(?:\s*(?:->|->|-->)\s*|\s*[:：]\s*)(.+?)(?=\n\s*[-*]|\n\n|$)",
        re.MULTILINE,
    ),
    # Bracket format: "[问题描述] -> [建议修改]"
    re.compile(
        r"\[(.+?)\]\s*(?:->|->|-->)\s*\[(.+?)\]",
    ),
    # Simple "问题: xxx  建议: xxx" format
    re.compile(
        r"问题[：:]\s*(.+?)\s+建议[：:]\s*(.+?)(?=\n|$)",
    ),
]


@dataclass
class ReviewResult:
    """Structured review result with mandatory suggestions on REJECT."""

    approved: bool
    notes: str = ""
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "notes": self.notes,
            "suggestions": self.suggestions,
        }


class Reviewer:
    """Reference implementation: Reviewer using TAP-compiled prompts.

    Phase 4: The system prompt is no longer hardcoded here — it comes from
    the Compiler's _default_prompts["review"], which provides model-specific
    optimizations.

    Configuration:
      - user_review=True: Block for user approval
      - llm_review=True: Use LLM for review (requires model)
      - Both False (default): Auto-approve (M0 mode)
    """

    def __init__(
        self,
        bus: EventBus,
        model: ModelProvider | None = None,
        user_review: bool = False,
        llm_review: bool = False,
    ) -> None:
        self.bus = bus
        self.model = model
        self.user_review = user_review
        self.llm_review = llm_review
        self._design_review_count = 0
        self._plan_review_count = 0
        bus.on("request_design_review", self.on_design_review)
        bus.on("request_plan_review", self.on_plan_review)

    async def on_design_review(self, content: str) -> None:
        self._design_review_count += 1
        await self._handle_review(content, "design")

    async def on_plan_review(self, content: str) -> None:
        self._plan_review_count += 1
        await self._handle_review(content, "plan")

    async def _handle_review(self, content: str, review_type: str) -> None:

        # 1. User review mode
        if self.user_review:
            logger.info(f"User review required for {review_type}. Emitting intervention request.")
            await self.bus.emit(
                "request_user_intervention",
                "review",
                f"Please review the generated {review_type} content",
            )
            return

        # 2. LLM review mode
        if self.llm_review and self.model:
            await self._llm_review(content, review_type)
            return

        # 3. Default: M0 auto-approve
        logger.info(f"Review auto-approved for M0 ({review_type}): {content[:50]}...")
        await self.bus.emit("review_result", True, "Auto-approved M0", [])

    async def _llm_review(self, content: str, review_type: str) -> None:
        """Use LLM for review with structured suggestion parsing.

        Phase 4: Uses execute_tap(TAPRequest(intent="review")) instead of
        chat(messages). The system prompt comes from the Compiler.
        """
        try:
            # Phase 4: Build TAPRequest instead of hardcoded messages
            tap_request = TAPRequest(
                meta={"task_id": f"review_{review_type}", "intent": "review"},
                context={},
                instruction=content,
                constraints=[
                    "审核结果格式：APPROVE 或 REJECT + 修改建议",
                    "每条建议包含：问题位置、修改方向、优先级",
                    "不要只指出问题不给建议",
                ],
            )

            response = await self.model.execute_tap(tap_request)
            result_text = (response.raw_text or "").strip()

            approved, notes, suggestions = self._parse_review_result(result_text)

            # REJECT must have suggestions, otherwise follow up
            if not approved and not suggestions:
                logger.warning(
                    f"Review REJECTED for {review_type} without suggestions. "
                    f"Requesting re-review with specific suggestions."
                )
                suggestions = await self._request_suggestions(
                    content, result_text, tap_request
                )
                if not suggestions and notes:
                    suggestions = self._extract_fallback_suggestions(notes)

            result = ReviewResult(
                approved=approved,
                notes=notes,
                suggestions=suggestions,
            )

            logger.info(
                f"Review result for {review_type}: "
                f"{'APPROVED' if approved else 'REJECTED'}, "
                f"notes={notes[:100]}, "
                f"suggestions={len(suggestions)}"
            )
            await self.bus.emit(
                "review_result",
                result.approved,
                result.notes,
                result.suggestions,
            )
        except Exception as e:
            logger.error(f"Review failed, auto-approving: {e}")
            await self.bus.emit(
                "review_result",
                True,
                f"Auto-approved: Review failed - {e}",
                [],
            )

    async def _request_suggestions(
        self,
        original_content: str,
        reject_text: str,
        original_request: TAPRequest,
    ) -> list[str]:
        """Follow up with model when REJECT lacks suggestions (max 1 follow-up).

        Phase 4: Uses execute_tap() for the follow-up, maintaining TAP
        compilation for consistency with the original review.
        """
        try:
            # Build follow-up as a chat continuation
            # The follow-up needs the original context plus the reject + followup prompt
            followup_request = TAPRequest(
                meta={"task_id": original_request.meta.get("task_id", "review"), "intent": "review"},
                context=original_request.context,
                instruction=f"之前的审查结果：\n{reject_text}\n\n{_SUGGESTION_FOLLOWUP_PROMPT}",
                constraints=["每条建议必须包含问题位置和明确修改方向"],
            )
            response = await self.model.execute_tap(followup_request)
            result_text = (response.raw_text or "").strip()

            suggestions = self._parse_suggestions(result_text)

            if suggestions:
                logger.info(f"Follow-up review yielded {len(suggestions)} suggestions")
            else:
                logger.warning("Follow-up review still yielded no structured suggestions")

            return suggestions
        except Exception as e:
            logger.error(f"Follow-up suggestion request failed: {e}")
            return []

    def _parse_review_result(self, text: str) -> tuple[bool, str, list[str]]:
        """Parse LLM review output, return (approved, notes, suggestions)."""
        # Check REJECT first (more specific)
        for pattern in _REJECT_PATTERNS:
            if pattern.search(text):
                reason = ""
                for p in [
                    re.compile(r"\bREJECT\b[:：]?\s*(.*)", re.IGNORECASE),
                    re.compile(r"\bREJECTED\b[:：]?\s*(.*)", re.IGNORECASE),
                    re.compile(r"\b拒绝\b[:：]?\s*(.*)"),
                    re.compile(r"\b不通过\b[:：]?\s*(.*)"),
                    re.compile(r"\b需要修改\b[:：]?\s*(.*)"),
                ]:
                    match = p.search(text)
                    if match and match.lastindex:
                        reason = match.group(1).strip()
                        break

                notes = reason or text[:500]
                suggestions = self._parse_suggestions(text)

                return False, notes, suggestions

        # Check APPROVE
        for pattern in _APPROVE_PATTERNS:
            if pattern.search(text):
                return True, "Model approved", []

        # Unclear verdict → default approve (M0 lenient mode)
        logger.warning(
            f"Could not parse review verdict from text: {text[:200]!r}. "
            f"Defaulting to APPROVE (M0 lenient mode)."
        )
        return True, f"Auto-approved (unclear verdict): {text[:200]}", []

    def _parse_suggestions(self, text: str) -> list[str]:
        """Parse structured suggestion list from LLM review output."""
        suggestions: list[str] = []

        for pattern in _SUGGESTION_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        problem = match[0].strip()
                        suggestion = match[1].strip()
                        if problem and suggestion:
                            suggestions.append(f"{problem} -> {suggestion}")
                    elif isinstance(match, str) and match.strip():
                        suggestions.append(match.strip())
                if suggestions:
                    return suggestions

        # Fallback: extract non-empty lines
        skip_prefixes = (
            "REJECT", "REJECTED", "拒绝", "不通过", "需要修改",
            "##", "#", "---", "***", "```",
        )
        for line in text.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if any(line_stripped.upper().startswith(p.upper()) for p in skip_prefixes):
                continue
            if line_stripped.startswith("问题与修改建议"):
                continue
            if len(line_stripped) < 5:
                continue
            suggestions.append(line_stripped)

        return suggestions

    def _extract_fallback_suggestions(self, notes: str) -> list[str]:
        """Extract fallback suggestions from notes text."""
        if not notes:
            return []

        parts = []
        for sep in ["; ", "。", "；", "\n"]:
            parts = [p.strip() for p in notes.split(sep) if p.strip()]
            if len(parts) > 1:
                break

        suggestions = []
        for part in parts[:5]:
            if len(part) < 5:
                continue
            if any(kw in part for kw in ["修改", "修复", "补充", "添加", "删除", "替换", "调整"]):
                suggestions.append(part)
            else:
                suggestions.append(f"请检查并修改: {part}")

        return suggestions
