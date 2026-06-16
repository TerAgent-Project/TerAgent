"""examples.full_agent.checklist_generator — Checklist generation using library APIs

Reference implementation that combines:
    - teragent.pipeline.checklist.run_deterministic_checks (library primitive)
    - teragent.pipeline.checklist.TaskInfo (decoupled data structure)
    - teragent.event_bus (orchestration)

This demonstrates how the library-level checklist primitive can be used
with any task representation — just convert your tasks to TaskInfo objects.

Note: The plan parameter is typed as Any; callers should convert their task
representation to list[TaskInfo] using plan_to_task_info_list().
"""
import logging
import os
from typing import Any

from teragent.pipeline.checklist import run_deterministic_checks, TaskInfo

from teragent.event_bus import EventBus

logger = logging.getLogger(__name__)


def plan_to_task_info_list(plan: Any) -> list[TaskInfo]:
    """Convert a plan-like object to list[TaskInfo] for library-level checklist.

    This is the bridge function between an external task representation
    and the library-level TaskInfo dataclass. Any caller with their own
    task representation can write a similar conversion function.

    The plan object is expected to have a ``tasks`` attribute where each
    task has ``id``, ``title``, ``status`` (with ``.value`` returning a
    string like "completed"/"pending"/"blocked"/"skipped"), and
    ``output_files`` (an iterable of file paths).

    Args:
        plan: A plan-like object with a .tasks attribute

    Returns:
        List of TaskInfo objects suitable for run_deterministic_checks()
    """
    task_info_list: list[TaskInfo] = []
    for t in plan.tasks:
        # Support both enum .value and plain string status
        status_val = t.status.value if hasattr(t.status, 'value') else str(t.status)
        task_info_list.append(TaskInfo(
            id=t.id,
            title=t.title,
            status=status_val,
            output_files=list(t.output_files),
        ))
    return task_info_list


class ChecklistGenerator:
    """Reference implementation: Checklist generation using library APIs.

    Uses run_deterministic_checks() from teragent.pipeline.checklist
    (library primitive) with Plan → TaskInfo conversion.
    """

    def __init__(
        self,
        bus: EventBus,
        workspace_root: str | None = None,
    ) -> None:
        self.bus = bus
        self._workspace_root = workspace_root
        self._generation_count = 0
        bus.on("request_checklist", self.on_request)

    async def on_request(self, plan: Any) -> None:
        logger.info("Running deterministic code checks...")
        self._generation_count += 1

        workspace_root = self._get_workspace_root()

        # Use library-level deterministic checks with TaskInfo conversion
        if workspace_root and os.path.isdir(workspace_root):
            task_info_list = plan_to_task_info_list(plan)
            checklist, structured_data = run_deterministic_checks(workspace_root, task_info_list)
            await self.bus.emit("checklist_ready", checklist)
            logger.info(f"Deterministic checklist generated ({len(checklist)} chars)")

            # When issues detected, emit checklist_issues_detected event
            if structured_data.get("needs_repair", False):
                logger.warning(
                    f"Checklist detected issues: {structured_data['fail_count']} FAIL, "
                    f"{structured_data['warn_count']} WARN, "
                    f"has_critical_warn={structured_data['has_critical_warn']}. "
                    f"Emitting checklist_issues_detected for auto-repair."
                )
                await self.bus.emit(
                    "checklist_issues_detected",
                    fail_count=structured_data["fail_count"],
                    warn_count=structured_data["warn_count"],
                    ok_count=structured_data["ok_count"],
                    has_critical_warn=structured_data["has_critical_warn"],
                    issues=structured_data["issues"],
                    workspace_root=workspace_root,
                    plan=plan,
                )
            return

        # Fallback: no workspace, generate simple report
        fallback = self._generate_fallback_checklist(plan)
        await self.bus.emit("checklist_ready", fallback)

    def _get_workspace_root(self) -> str | None:
        """Get workspace root from constructor or environment."""
        if self._workspace_root:
            return self._workspace_root
        # NOTE: EventBus._shared was removed; use constructor injection instead.
        return os.environ.get("TERAGENT_WORKSPACE")

    def _generate_fallback_checklist(self, plan: Any) -> str:
        """Generate a simple deterministic checklist from plan data."""
        lines: list[str] = []
        completed_count = 0
        blocked_count = 0
        for t in plan.tasks:
            # Support both enum .value and plain string status
            status_val = t.status.value if hasattr(t.status, 'value') else str(t.status)
            is_completed = status_val in ("completed", "skipped")
            is_blocked = status_val == "blocked"
            if is_completed:
                marker = "[x]"
            elif is_blocked:
                marker = "[!]"
            else:
                marker = "[ ]"
            lines.append(f"- {marker} {t.id} {t.title} - {status_val}")
            if is_completed:
                completed_count += 1
            if is_blocked:
                blocked_count += 1

        total = len(plan.tasks)
        lines.append(f"\n完成 {completed_count}/{total} 项任务")
        if blocked_count > 0:
            lines.append(f"\n**警告**: {blocked_count} 个任务被权限拦截 (BLOCKED)，未执行。")

        if completed_count < total:
            lines.append("\n**需修复**: 有任务未完成")

        return "\n".join(lines)
