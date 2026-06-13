"""teragent.long_horizon.progress — 进度追踪器

记录长程任务的执行步数、耗时、成果产出、策略切换次数。
为 LongHorizonTaskManager 提供进度报告，支持停滞检测和自评估。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ProgressReport:
    """进度报告数据结构

    长程任务在任意时刻的执行进度快照。

    Attributes:
        task_id: 任务唯一标识
        goal: 原始大目标描述
        total_sub_goals: 子目标总数
        completed_sub_goals: 已完成的子目标数
        current_phase: 当前阶段 — "planning" | "executing" | "evaluating" | "stagnant"
        steps_completed: 已完成的步骤数
        elapsed_minutes: 已耗时间（分钟）
        strategy_switches: 策略切换次数
        sub_goal_statuses: 各子目标的状态列表
        estimated_remaining_minutes: 预估剩余时间（分钟）
        last_checkpoint: 最后一个检查点的ID
    """

    task_id: str
    goal: str
    total_sub_goals: int
    completed_sub_goals: int
    current_phase: str
    steps_completed: int
    elapsed_minutes: float
    strategy_switches: int
    sub_goal_statuses: list[dict]  # [{id, description, status, result_summary}]
    estimated_remaining_minutes: float
    last_checkpoint: str


class ProgressTracker:
    """进度追踪器

    记录执行步数、耗时、成果产出、策略切换次数。
    为 LongHorizonTaskManager 提供进度报告，支持停滞检测。

    使用方式::

        tracker = ProgressTracker(task_id="task_1", goal="实现用户系统")
        tracker.start_sub_goal("sg_1", "设计数据库")
        tracker.record_step("创建 User 表")
        tracker.complete_sub_goal("sg_1", "User 表创建完成")
        report = tracker.get_report()

    Attributes:
        task_id: 任务唯一标识
        goal: 原始大目标描述
    """

    def __init__(self, task_id: str, goal: str) -> None:
        """初始化进度追踪器

        Args:
            task_id: 任务唯一标识
            goal: 原始大目标描述
        """
        self.task_id = task_id
        self.goal = goal

        # 时间追踪
        self._start_time: float = time.monotonic()
        self._current_sub_goal_start: float | None = None

        # 步数追踪
        self._steps_completed: int = 0
        self._step_descriptions: list[str] = []

        # 子目标追踪
        # {sub_goal_id: {"description": str, "status": str, "result_summary": str}}
        self._sub_goals: dict[str, dict] = {}

        # 策略切换追踪
        self._strategy_switches: int = 0
        self._strategy_switch_reasons: list[str] = []

        # 当前阶段
        self._current_phase: str = "planning"

        # 最后检查点
        self._last_checkpoint: str = ""

    def start_sub_goal(self, sub_goal_id: str, description: str) -> None:
        """开始一个子目标

        将子目标状态设为 in_progress，并记录开始时间。

        Args:
            sub_goal_id: 子目标ID
            description: 子目标描述
        """
        self._sub_goals[sub_goal_id] = {
            "id": sub_goal_id,
            "description": description,
            "status": "in_progress",
            "result_summary": "",
        }
        self._current_sub_goal_start = time.monotonic()
        self._current_phase = "executing"

    def complete_sub_goal(self, sub_goal_id: str, result_summary: str) -> None:
        """完成一个子目标

        将子目标状态设为 completed，并记录结果摘要。

        Args:
            sub_goal_id: 子目标ID
            result_summary: 结果摘要
        """
        if sub_goal_id in self._sub_goals:
            self._sub_goals[sub_goal_id]["status"] = "completed"
            self._sub_goals[sub_goal_id]["result_summary"] = result_summary
        self._current_sub_goal_start = None

        # 检查是否所有子目标都已完成
        all_completed = all(
            sg["status"] == "completed" for sg in self._sub_goals.values()
        )
        if all_completed and self._sub_goals:
            self._current_phase = "completed"
        else:
            self._current_phase = "evaluating"

    def fail_sub_goal(self, sub_goal_id: str, error: str) -> None:
        """标记子目标为失败

        Args:
            sub_goal_id: 子目标ID
            error: 错误描述
        """
        if sub_goal_id in self._sub_goals:
            self._sub_goals[sub_goal_id]["status"] = "failed"
            self._sub_goals[sub_goal_id]["result_summary"] = f"失败: {error}"

    def record_step(self, description: str = "") -> None:
        """记录一个执行步骤

        Args:
            description: 步骤描述（可选）
        """
        self._steps_completed += 1
        if description:
            self._step_descriptions.append(description)

    def record_strategy_switch(self, reason: str) -> None:
        """记录一次策略切换

        当检测到停滞并触发策略切换时调用此方法。

        Args:
            reason: 策略切换原因
        """
        self._strategy_switches += 1
        self._strategy_switch_reasons.append(reason)
        self._current_phase = "stagnant"

    def set_phase(self, phase: str) -> None:
        """设置当前阶段

        Args:
            phase: 阶段名称 — "planning" | "executing" | "evaluating" | "stagnant"
        """
        self._current_phase = phase

    def set_last_checkpoint(self, checkpoint_id: str) -> None:
        """设置最后一个检查点的ID

        Args:
            checkpoint_id: 检查点ID
        """
        self._last_checkpoint = checkpoint_id

    def register_sub_goal(self, sub_goal_id: str, description: str) -> None:
        """预注册一个子目标（在开始执行前）

        用于在目标分解后注册所有子目标，使进度报告能显示完整的目标列表。

        Args:
            sub_goal_id: 子目标ID
            description: 子目标描述
        """
        if sub_goal_id not in self._sub_goals:
            self._sub_goals[sub_goal_id] = {
                "id": sub_goal_id,
                "description": description,
                "status": "pending",
                "result_summary": "",
            }

    def get_report(self) -> ProgressReport:
        """获取当前进度报告

        计算预估剩余时间：基于已完成子目标的平均耗时外推。

        Returns:
            ProgressReport 进度报告
        """
        elapsed = self.get_elapsed_minutes()

        # 计算已完成子目标数
        completed = sum(
            1 for sg in self._sub_goals.values() if sg["status"] == "completed"
        )
        total = len(self._sub_goals)

        # 预估剩余时间
        estimated_remaining = 0.0
        if completed > 0 and total > completed:
            avg_per_sub_goal = elapsed / completed
            estimated_remaining = avg_per_sub_goal * (total - completed)

        # 构建子目标状态列表
        sub_goal_statuses = list(self._sub_goals.values())

        return ProgressReport(
            task_id=self.task_id,
            goal=self.goal,
            total_sub_goals=total,
            completed_sub_goals=completed,
            current_phase=self._current_phase,
            steps_completed=self._steps_completed,
            elapsed_minutes=elapsed,
            strategy_switches=self._strategy_switches,
            sub_goal_statuses=sub_goal_statuses,
            estimated_remaining_minutes=estimated_remaining,
            last_checkpoint=self._last_checkpoint,
        )

    def get_elapsed_minutes(self) -> float:
        """获取从任务开始到现在的耗时（分钟）

        Returns:
            耗时（分钟），保留两位小数
        """
        elapsed_seconds = time.monotonic() - self._start_time
        return round(elapsed_seconds / 60.0, 2)

    @property
    def steps_completed(self) -> int:
        """已完成的步骤数"""
        return self._steps_completed

    @property
    def strategy_switches(self) -> int:
        """策略切换次数"""
        return self._strategy_switches

    @property
    def current_phase(self) -> str:
        """当前阶段"""
        return self._current_phase
