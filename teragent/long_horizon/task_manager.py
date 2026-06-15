"""teragent.long_horizon.task_manager — 长程任务管理器

编排 GLM-5 的8小时持续工作能力，核心功能：
  1. 将大目标分解为阶段性子目标
  2. 管理检查点的保存和恢复
  3. 追踪执行进度
  4. 集成自评估和策略切换
  5. 支持断点续执行

与 AgentLoop 的集成：
  - AgentLoop 在长程任务模式下创建 LongHorizonTaskManager
  - 每个阶段由 AgentLoop 的正常 tool loop 执行
  - 阶段结束后由 TaskManager 进行评估和决策
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import deque
from typing import TYPE_CHECKING

__all__ = [
    "LongHorizonTaskManager",
]

from teragent.core.tap import LongHorizonConfig
from teragent.long_horizon.checkpoint import Checkpoint, CheckpointStore
from teragent.long_horizon.progress import ProgressReport, ProgressTracker
from teragent.long_horizon.self_evaluation import SelfEvaluator
from teragent.long_horizon.strategy_switch import StrategySwitcher
from teragent.long_horizon.types import LongHorizonResult, PhaseResult, SubGoal

if TYPE_CHECKING:
    from teragent.core.provider import ModelProvider

logger = logging.getLogger(__name__)


class LongHorizonTaskManager:
    """长程任务管理器 — 编排 GLM-5 的8小时持续工作能力

    核心流程：
      1. decompose_goal() — 分解大目标为子目标
      2. 循环执行每个子目标：
         a. execute_phase() — 执行一个阶段
         b. save_checkpoint() — 保存检查点
         c. evaluate_progress() — 评估进度
         d. 检测停滞 → 触发策略切换
      3. 生成最终结果

    使用方式::

        manager = LongHorizonTaskManager(
            goal="实现一个完整的用户管理系统",
            model_provider=provider,
            config=LongHorizonConfig(max_duration_hours=4),
        )
        result = await manager.execute_long_task()

    Attributes:
        goal: 原始大目标描述
        model_provider: GLM-5 模型提供者
        config: 长程任务配置
        checkpoint_store: 检查点存储
        progress_tracker: 进度追踪器
    """

    def __init__(
        self,
        goal: str,
        model_provider: ModelProvider,
        config: LongHorizonConfig | None = None,
        checkpoint_store: CheckpointStore | None = None,
        progress_tracker: ProgressTracker | None = None,
        self_evaluator: SelfEvaluator | None = None,
        strategy_switcher: StrategySwitcher | None = None,
    ) -> None:
        """初始化长程任务管理器

        Args:
            goal: 原始大目标描述
            model_provider: GLM-5 的 ModelProvider 实例
            config: 长程任务配置，默认使用 LongHorizonConfig()
            checkpoint_store: 检查点存储，默认创建 CheckpointStore()
            progress_tracker: 进度追踪器，默认根据 goal 创建
            self_evaluator: 自评估执行器，默认根据 model_provider 创建
            strategy_switcher: 策略切换管理器，默认根据 model_provider 创建
        """
        self.goal = goal
        self.model_provider = model_provider
        self.config = config or LongHorizonConfig()

        # 生成唯一的任务ID
        self.task_id: str = str(uuid.uuid4())

        # 检查点存储
        self.checkpoint_store = checkpoint_store or CheckpointStore()

        # 进度追踪器
        self.progress_tracker = progress_tracker or ProgressTracker(
            task_id=self.task_id, goal=goal
        )

        # 自评估执行器
        self.self_evaluator = self_evaluator or SelfEvaluator(
            model_provider=model_provider,
            evaluation_interval_steps=10,
            evaluation_interval_minutes=self.config.checkpoint_interval_minutes,
        )

        # 策略切换管理器
        self.strategy_switcher = strategy_switcher or StrategySwitcher(
            model_provider=model_provider,
            stagnation_threshold=self.config.stagnation_threshold,
        )

        # 子目标列表
        self._sub_goals: list[SubGoal] = []

        # 阶段结果列表
        self._phase_results: list[PhaseResult] = []

        # 检查点计数
        self._checkpoints_saved: int = 0

        # 停滞检测：记录最近几次执行结果的摘要，用于检测重复
        self._recent_result_summaries: list[str] = []

        # 自评估追踪：距上次评估的步数和时间
        self._steps_since_last_eval: int = 0
        self._last_eval_time: float = __import__("time").monotonic()

        # 最近N步的描述（用于策略切换器的停滞检测）
        self._recent_step_descriptions: list[str] = []

        # 当前策略描述
        self._current_strategy: str = "初始策略"

    # ==================================================================
    # 主入口
    # ==================================================================

    async def execute_long_task(self) -> LongHorizonResult:
        """执行长程任务的主入口

        流程：
          1. decompose_goal() — 分解大目标为子目标
          2. 循环执行每个子目标：
             a. execute_phase() — 执行一个阶段
             b. save_checkpoint() — 保存检查点
             c. evaluate_progress() — 评估进度
             d. 检测停滞 → 触发策略切换
          3. 生成最终结果

        Returns:
            LongHorizonResult 长程任务最终结果
        """
        logger.info(
            f"Starting long-horizon task: task_id={self.task_id} "
            f"goal={self.goal[:100]}..."
        )

        _start_time = __import__("time").monotonic()

        try:
            # 1. 分解目标
            self.progress_tracker.set_phase("planning")
            self._sub_goals = await self.decompose_goal(self.goal)

            if not self._sub_goals:
                logger.warning("Goal decomposition returned no sub-goals")
                return self._build_result(success=False, final_summary="目标分解失败：无法生成子目标")

            # 注册所有子目标到进度追踪器
            for sg in self._sub_goals:
                self.progress_tracker.register_sub_goal(sg.id, sg.description)

            # 保存初始检查点
            await self._save_checkpoint()

            # 2. 按拓扑顺序执行子目标
            execution_order = self._topological_sort()
            completed_ids: set[str] = set()

            for sub_goal in execution_order:
                # 检查时间预算
                elapsed = self.progress_tracker.get_elapsed_minutes()
                max_minutes = self.config.max_duration_hours * 60
                if elapsed >= max_minutes:
                    logger.warning(
                        f"Time budget exhausted: {elapsed:.1f}min >= {max_minutes:.1f}min"
                    )
                    break

                # 跳过已完成的子目标（断点续执行场景）
                if sub_goal.status == "completed":
                    completed_ids.add(sub_goal.id)
                    continue

                # 检查依赖是否满足
                unmet = [dep_id for dep_id in sub_goal.dependencies if dep_id not in completed_ids]
                if unmet:
                    logger.warning(
                        f"Skipping sub-goal {sub_goal.id}: unmet dependencies {unmet}"
                    )
                    continue

                # 执行阶段
                sub_goal.status = "in_progress"
                self.progress_tracker.start_sub_goal(sub_goal.id, sub_goal.description)

                phase_result = await self.execute_phase(sub_goal)
                self._phase_results.append(phase_result)

                if phase_result.success:
                    sub_goal.status = "completed"
                    completed_ids.add(sub_goal.id)
                    self.progress_tracker.complete_sub_goal(
                        sub_goal.id, phase_result.result_text[:200]
                    )
                else:
                    sub_goal.status = "failed"
                    self.progress_tracker.fail_sub_goal(
                        sub_goal.id, "; ".join(phase_result.errors)
                    )

                # 保存检查点
                await self._save_checkpoint()

                # 更新自评估追踪
                self._steps_since_last_eval += phase_result.steps_taken
                if phase_result.result_text:
                    self._recent_step_descriptions.append(
                        phase_result.result_text[:100]
                    )
                    # 只保留最近 20 条描述
                    if len(self._recent_step_descriptions) > 20:
                        self._recent_step_descriptions = (
                            self._recent_step_descriptions[-20:]
                        )

                # 评估进度
                self.progress_tracker.set_phase("evaluating")
                _evaluation = await self.evaluate_progress()

                # 自评估检查（如果启用了自评估）
                should_switch = False
                switch_reason = ""

                if self.config.self_evaluation_enabled:
                    import time as _time
                    minutes_since_eval = (
                        _time.monotonic() - self._last_eval_time
                    ) / 60.0

                    if self.self_evaluator.should_evaluate(
                        steps_since_last=self._steps_since_last_eval,
                        minutes_since_last=minutes_since_eval,
                    ):
                        # 执行自评估
                        progress_report = self.progress_tracker.get_report()
                        eval_result = await self.self_evaluator.evaluate(
                            goal=self.goal,
                            progress_report=progress_report,
                            recent_results=self._phase_results[-5:],
                        )

                        # 重置评估计时器
                        self._steps_since_last_eval = 0
                        self._last_eval_time = _time.monotonic()

                        # 根据自评估结果判断是否需要策略切换
                        if eval_result.should_switch_strategy:
                            should_switch = True
                            switch_reason = (
                                f"自评估建议切换策略: "
                                f"目标对齐度={eval_result.goal_alignment}, "
                                f"产出质量={eval_result.output_quality}, "
                                f"瓶颈={eval_result.bottleneck_identified}"
                            )
                            logger.info(
                                f"Self-evaluation suggests strategy switch: "
                                f"overall={eval_result.overall_score:.1f}"
                            )

                # 停滞检测（使用 StrategySwitcher 的检测方法）
                is_stagnant, stagnation_reason = self.strategy_switcher.detect_stagnation(
                    recent_results=self._phase_results[-10:],
                    recent_steps=self._recent_step_descriptions,
                )

                if is_stagnant:
                    should_switch = True
                    switch_reason = switch_reason or stagnation_reason

                # 也保留原有的简单停滞检测作为补充
                if self._detect_stagnation(phase_result):
                    should_switch = True
                    switch_reason = switch_reason or (
                        f"连续{self.config.stagnation_threshold}次相似结果"
                    )

                # 执行策略切换
                if should_switch:
                    logger.warning(
                        f"Strategy switch triggered: {switch_reason[:100]}"
                    )
                    try:
                        progress_report = self.progress_tracker.get_report()
                        new_strategy, switch_record = (
                            await self.strategy_switcher.switch_strategy(
                                current_strategy=self._current_strategy,
                                reason=switch_reason,
                                goal=self.goal,
                                progress_report=progress_report,
                            )
                        )
                        self._current_strategy = new_strategy

                        # 记录策略切换到进度追踪器
                        self.progress_tracker.record_strategy_switch(switch_reason)

                        logger.info(
                            f"Strategy switched to: {new_strategy[:100]}"
                        )
                    except Exception as e:
                        logger.error(f"Strategy switch failed: {e}")
                        # 即使切换失败也记录事件
                        self.progress_tracker.record_strategy_switch(
                            f"策略切换失败: {e}"
                        )

            # 3. 生成最终结果
            success = all(sg.status == "completed" for sg in self._sub_goals)
            final_summary = await self._generate_final_summary()
            result = self._build_result(success=success, final_summary=final_summary)

            logger.info(
                f"Long-horizon task completed: task_id={self.task_id} "
                f"success={result.success} "
                f"sub_goals={result.completed_sub_goals}/{result.total_sub_goals} "
                f"steps={result.total_steps} "
                f"elapsed={result.total_elapsed_minutes:.1f}min"
            )

            return result

        except Exception as e:
            logger.error(f"Long-horizon task failed: {e}", exc_info=True)
            # 尝试保存检查点
            try:
                await self._save_checkpoint()
            except Exception:
                pass

            return self._build_result(success=False, final_summary=f"任务异常终止: {e}")

    # ==================================================================
    # 目标分解
    # ==================================================================

    async def decompose_goal(self, goal: str) -> list[SubGoal]:
        """将大目标分解为子目标

        使用 GLM-5 进行目标分解。通过 ModelProvider.chat() 发送分解请求，
        要求模型以 JSON 格式返回子目标列表。

        每个子目标包含：
          - description: 子目标描述
          - completion_criteria: 完成标准
          - estimated_steps: 预估步骤数
          - dependencies: 依赖的子目标ID列表

        Args:
            goal: 大目标描述

        Returns:
            子目标列表
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个任务规划专家。请将用户给出的大目标分解为具体的阶段性子目标。\n"
                    "要求：\n"
                    "1. 每个子目标应该是一个可独立验证的阶段性成果\n"
                    "2. 子目标之间可以有依赖关系（通过 dependencies 字段指定）\n"
                    "3. 每个子目标需要有明确的完成标准\n"
                    "4. 请以 JSON 格式返回，格式如下：\n"
                    '[\n'
                    '  {\n'
                    '    "id": "sg_1",\n'
                    '    "description": "子目标描述",\n'
                    '    "completion_criteria": "完成标准",\n'
                    '    "estimated_steps": 5,\n'
                    '    "dependencies": []\n'
                    '  },\n'
                    '  ...\n'
                    ']\n'
                    "5. ID 格式为 sg_1, sg_2, sg_3, ...\n"
                    "6. dependencies 为所依赖的子目标ID列表\n"
                    "7. 只返回 JSON 数组，不要包含其他文字\n"
                ),
            },
            {
                "role": "user",
                "content": f"请将以下大目标分解为子目标：\n\n{goal}",
            },
        ]

        try:
            response = await self.model_provider.chat(messages)

            content = response.get("content", "")
            sub_goals = self._parse_sub_goals_json(content)

            if not sub_goals:
                logger.warning("Failed to parse sub-goals from model response, using fallback")
                # 回退方案：创建一个默认的子目标
                sub_goals = [
                    SubGoal(
                        id="sg_1",
                        description=goal,
                        completion_criteria="完成大目标的所有要求",
                        estimated_steps=10,
                        dependencies=[],
                    )
                ]

            return sub_goals

        except Exception as e:
            logger.error(f"Goal decomposition failed: {e}")
            # 回退方案
            return [
                SubGoal(
                    id="sg_1",
                    description=goal,
                    completion_criteria="完成大目标的所有要求",
                    estimated_steps=10,
                    dependencies=[],
                )
            ]

    # ==================================================================
    # 阶段执行
    # ==================================================================

    async def execute_phase(self, sub_goal: SubGoal) -> PhaseResult:
        """执行一个阶段的子目标

        通过 ModelProvider.chat() 调用 GLM-5，
        使用 GLM5Compiler 编译的长程任务 prompt。

        执行步骤：
          1. 构建阶段执行 prompt（包含子目标描述、完成标准、进度上下文）
          2. 调用模型获取执行结果
          3. 解析结果，记录步骤数和文件变更
          4. 返回 PhaseResult

        Args:
            sub_goal: 要执行的子目标

        Returns:
            PhaseResult 阶段执行结果
        """
        # 获取当前进度上下文
        progress_report = self.progress_tracker.get_report()
        progress_context = self._format_progress_context(progress_report)

        # 构建阶段执行消息
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个高效的执行者。请按照子目标的要求完成任务。\n"
                    "要求：\n"
                    "1. 严格遵循子目标的完成标准\n"
                    "2. 记录你的执行步骤\n"
                    "3. 如果遇到问题，说明原因和尝试的解决方案\n"
                    "4. 输出你的执行结果和总结\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"【当前进度】\n{progress_context}\n\n"
                    f"【子目标】\n"
                    f"描述：{sub_goal.description}\n"
                    f"完成标准：{sub_goal.completion_criteria}\n"
                    f"预估步骤数：{sub_goal.estimated_steps}\n\n"
                    f"请执行此子目标，并在完成后输出结果总结。"
                ),
            },
        ]

        try:
            response = await self.model_provider.chat(messages)

            content = response.get("content", "")
            steps_taken = self._estimate_steps_from_response(content)
            files = self._extract_files_from_response(content)

            # 记录步骤到进度追踪器
            for _ in range(steps_taken):
                self.progress_tracker.record_step()

            # 记录结果摘要用于停滞检测
            summary = content[:200] if content else ""
            self._recent_result_summaries.append(summary)
            # 只保留最近 N 个结果摘要（N = stagnation_threshold + 1）
            max_history = self.config.stagnation_threshold + 2
            if len(self._recent_result_summaries) > max_history:
                self._recent_result_summaries = self._recent_result_summaries[-max_history:]

            success = bool(content and not self._is_failure_response(content))
            errors = []
            if not success:
                errors.append("执行结果可能不理想")

            return PhaseResult(
                sub_goal_id=sub_goal.id,
                success=success,
                result_text=content,
                steps_taken=steps_taken,
                files_created=files.get("created", []),
                files_modified=files.get("modified", []),
                errors=errors,
            )

        except Exception as e:
            logger.error(f"Phase execution failed for sub-goal {sub_goal.id}: {e}")
            return PhaseResult(
                sub_goal_id=sub_goal.id,
                success=False,
                result_text="",
                steps_taken=0,
                errors=[str(e)],
            )

    # ==================================================================
    # 进度评估
    # ==================================================================

    async def evaluate_progress(self) -> dict:
        """评估当前进度

        使用模型对当前执行进度进行自评估，返回评估结果。
        如果 self_evaluation_enabled 为 True，会通过模型进行深度评估；
        否则返回基于进度报告的简单评估。

        Returns:
            评估结果字典，包含：
              - assessment: 总体评估（"on_track" | "behind" | "stagnant"）
              - summary: 评估摘要
              - recommendation: 建议
        """
        if not self.config.self_evaluation_enabled:
            return self._simple_evaluation()

        progress_report = self.progress_tracker.get_report()

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个项目管理评估专家。请根据当前进度评估任务执行状况。\n"
                    "请以 JSON 格式返回评估结果：\n"
                    '{\n'
                    '  "assessment": "on_track" | "behind" | "stagnant",\n'
                    '  "summary": "评估摘要",\n'
                    '  "recommendation": "建议"\n'
                    '}\n'
                    "只返回 JSON，不要包含其他文字。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"任务目标：{self.goal}\n\n"
                    f"当前进度：\n"
                    f"- 已完成子目标：{progress_report.completed_sub_goals}/{progress_report.total_sub_goals}\n"
                    f"- 已执行步骤：{progress_report.steps_completed}\n"
                    f"- 已耗时间：{progress_report.elapsed_minutes:.1f}分钟\n"
                    f"- 策略切换次数：{progress_report.strategy_switches}\n"
                    f"- 预估剩余时间：{progress_report.estimated_remaining_minutes:.1f}分钟\n\n"
                    f"请评估当前进度。"
                ),
            },
        ]

        try:
            response = await self.model_provider.chat(messages)
            content = response.get("content", "")
            return self._parse_evaluation_json(content)
        except Exception as e:
            logger.warning(f"Progress evaluation failed: {e}")
            return self._simple_evaluation()

    # ==================================================================
    # 检查点管理
    # ==================================================================

    async def _save_checkpoint(self) -> str:
        """保存当前状态为检查点

        Returns:
            检查点ID
        """
        progress_report = self.progress_tracker.get_report()

        completed_ids = [
            sg["id"]
            for sg in progress_report.sub_goal_statuses
            if sg["status"] == "completed"
        ]

        current_sg = ""
        for sg_status in progress_report.sub_goal_statuses:
            if sg_status["status"] == "in_progress":
                current_sg = sg_status["id"]
                break

        # 构建子目标状态数据（用于恢复）
        state_data = {
            "sub_goals": [
                {
                    "id": sg.id,
                    "description": sg.description,
                    "completion_criteria": sg.completion_criteria,
                    "estimated_steps": sg.estimated_steps,
                    "dependencies": sg.dependencies,
                    "status": sg.status,
                }
                for sg in self._sub_goals
            ],
            "recent_result_summaries": self._recent_result_summaries[-5:],
        }

        checkpoint = Checkpoint(
            checkpoint_id=CheckpointStore.generate_checkpoint_id(),
            task_id=self.task_id,
            timestamp=CheckpointStore.now_iso(),
            phase=progress_report.current_phase,
            completed_sub_goals=completed_ids,
            current_sub_goal=current_sg,
            steps_completed=progress_report.steps_completed,
            elapsed_minutes=progress_report.elapsed_minutes,
            strategy_switches=progress_report.strategy_switches,
            state_data=state_data,
        )

        checkpoint_id = await self.checkpoint_store.save(checkpoint)
        self._checkpoints_saved += 1
        self.progress_tracker.set_last_checkpoint(checkpoint_id)

        # 自动清理旧检查点
        if self._checkpoints_saved % 10 == 0:
            await self.checkpoint_store.cleanup(self.task_id, keep_last=5)

        return checkpoint_id

    async def resume_from_checkpoint(self, checkpoint_id: str) -> LongHorizonResult:
        """从检查点恢复执行

        从 CheckpointStore 加载指定检查点，
        跳过已完成的步骤，从断点继续。

        Args:
            checkpoint_id: 检查点ID

        Returns:
            LongHorizonResult 长程任务最终结果
        """
        checkpoint = await self.checkpoint_store.load(checkpoint_id)
        if checkpoint is None:
            logger.error(f"Checkpoint not found: {checkpoint_id}")
            return self._build_result(
                success=False, final_summary=f"检查点未找到: {checkpoint_id}"
            )

        logger.info(
            f"Resuming from checkpoint: {checkpoint_id} "
            f"phase={checkpoint.phase} "
            f"completed={len(checkpoint.completed_sub_goals)}"
        )

        # 恢复子目标状态
        sub_goals_data = checkpoint.state_data.get("sub_goals", [])
        if sub_goals_data:
            self._sub_goals = [
                SubGoal(
                    id=sg["id"],
                    description=sg["description"],
                    completion_criteria=sg["completion_criteria"],
                    estimated_steps=sg["estimated_steps"],
                    dependencies=sg.get("dependencies", []),
                    status=sg.get("status", "pending"),
                )
                for sg in sub_goals_data
            ]
        else:
            # 如果检查点没有子目标数据，重新分解
            self._sub_goals = await self.decompose_goal(self.goal)

        # 恢复进度追踪器状态
        for sg in self._sub_goals:
            self.progress_tracker.register_sub_goal(sg.id, sg.description)
            if sg.id in checkpoint.completed_sub_goals:
                self.progress_tracker.complete_sub_goal(sg.id, "从检查点恢复")

        self.progress_tracker.set_phase(checkpoint.phase)
        self.progress_tracker.set_last_checkpoint(checkpoint_id)

        # 恢复停滞检测历史
        self._recent_result_summaries = checkpoint.state_data.get(
            "recent_result_summaries", []
        )

        # 继续执行
        return await self.execute_long_task()

    # ==================================================================
    # 进度报告
    # ==================================================================

    def get_progress_report(self) -> ProgressReport:
        """获取当前进度报告

        Returns:
            ProgressReport 进度报告
        """
        return self.progress_tracker.get_report()

    # ==================================================================
    # 内部辅助方法
    # ==================================================================

    def _topological_sort(self) -> list[SubGoal]:
        """对子目标进行拓扑排序

        基于依赖关系进行拓扑排序，确保依赖在前、被依赖在后。

        Returns:
            排序后的子目标列表
        """
        if not self._sub_goals:
            return []

        # 构建邻接表和入度表
        id_to_sg = {sg.id: sg for sg in self._sub_goals}
        in_degree: dict[str, int] = {sg.id: 0 for sg in self._sub_goals}
        dependents: dict[str, list[str]] = {sg.id: [] for sg in self._sub_goals}

        for sg in self._sub_goals:
            for dep_id in sg.dependencies:
                if dep_id in dependents:
                    dependents[dep_id].append(sg.id)
                    in_degree[sg.id] += 1

        # Kahn 算法 — use deque for O(1) popleft (vs O(n) list.pop(0))
        queue: deque[str] = deque(sg_id for sg_id, deg in in_degree.items() if deg == 0)
        result: list[SubGoal] = []

        while queue:
            sg_id = queue.popleft()
            if sg_id in id_to_sg:
                result.append(id_to_sg[sg_id])
            for dependent_id in dependents.get(sg_id, []):
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)

        # 如果有环或孤立节点，将未排序的节点追加到末尾
        sorted_ids = {sg.id for sg in result}
        for sg in self._sub_goals:
            if sg.id not in sorted_ids:
                result.append(sg)

        return result

    def _detect_stagnation(self, phase_result: PhaseResult) -> bool:
        """检测执行停滞

        当连续 N 次执行结果的摘要高度相似时，判定为停滞。
        N 由 config.stagnation_threshold 决定。

        Args:
            phase_result: 最新的阶段执行结果

        Returns:
            是否检测到停滞
        """
        threshold = self.config.stagnation_threshold
        if len(self._recent_result_summaries) < threshold:
            return False

        # 取最近 threshold 个结果摘要
        recent = self._recent_result_summaries[-threshold:]

        # 简单的相似度检测：如果所有摘要都相同或非常短，判定为停滞
        if all(s == recent[0] for s in recent):
            return True

        # 如果最近的结果都是空字符串，判定为停滞
        if all(len(s.strip()) < 10 for s in recent):
            return True

        return False

    def _simple_evaluation(self) -> dict:
        """基于进度报告的简单评估（不调用模型）

        Returns:
            评估结果字典
        """
        report = self.progress_tracker.get_report()
        if report.total_sub_goals == 0:
            return {
                "assessment": "on_track",
                "summary": "任务尚未开始",
                "recommendation": "开始执行",
            }

        completion_rate = report.completed_sub_goals / report.total_sub_goals

        if report.strategy_switches >= 3:
            assessment = "stagnant"
        elif completion_rate < 0.3 and report.elapsed_minutes > report.estimated_remaining_minutes:
            assessment = "behind"
        else:
            assessment = "on_track"

        return {
            "assessment": assessment,
            "summary": f"已完成 {report.completed_sub_goals}/{report.total_sub_goals} 个子目标",
            "recommendation": "继续执行" if assessment == "on_track" else "考虑调整策略",
        }

    async def _generate_final_summary(self) -> str:
        """生成最终摘要

        使用模型生成任务的最终总结，如果模型调用失败则使用简单摘要。

        Returns:
            最终摘要文本
        """
        report = self.progress_tracker.get_report()

        messages = [
            {
                "role": "system",
                "content": "你是一个项目总结专家。请根据任务执行情况生成简洁的最终总结。",
            },
            {
                "role": "user",
                "content": (
                    f"任务目标：{self.goal}\n\n"
                    f"执行情况：\n"
                    f"- 子目标完成：{report.completed_sub_goals}/{report.total_sub_goals}\n"
                    f"- 总步骤数：{report.steps_completed}\n"
                    f"- 总耗时：{report.elapsed_minutes:.1f}分钟\n"
                    f"- 策略切换：{report.strategy_switches}次\n\n"
                    f"请生成简洁的最终总结（100字以内）。"
                ),
            },
        ]

        try:
            response = await self.model_provider.chat(messages)
            return response.get("content", self._fallback_summary())
        except Exception:
            return self._fallback_summary()

    def _fallback_summary(self) -> str:
        """生成简单回退摘要（不调用模型）

        Returns:
            简单摘要文本
        """
        report = self.progress_tracker.get_report()
        status = "成功" if report.completed_sub_goals == report.total_sub_goals else "部分完成"
        return (
            f"任务{status}：完成 {report.completed_sub_goals}/{report.total_sub_goals} "
            f"个子目标，耗时 {report.elapsed_minutes:.1f} 分钟"
        )

    def _build_result(self, success: bool, final_summary: str) -> LongHorizonResult:
        """构建 LongHorizonResult

        Args:
            success: 整体是否成功
            final_summary: 最终摘要

        Returns:
            LongHorizonResult 实例
        """
        report = self.progress_tracker.get_report()

        return LongHorizonResult(
            task_id=self.task_id,
            goal=self.goal,
            success=success,
            total_steps=report.steps_completed,
            total_elapsed_minutes=report.elapsed_minutes,
            completed_sub_goals=report.completed_sub_goals,
            total_sub_goals=report.total_sub_goals,
            strategy_switches=report.strategy_switches,
            phase_results=list(self._phase_results),
            final_summary=final_summary,
            checkpoints_saved=self._checkpoints_saved,
        )

    # ==================================================================
    # 解析辅助方法
    # ==================================================================

    @staticmethod
    def _parse_sub_goals_json(content: str) -> list[SubGoal]:
        """从模型响应中解析子目标 JSON

        尝试从文本中提取 JSON 数组，并转换为 SubGoal 列表。
        如果解析失败，返回空列表。

        Args:
            content: 模型返回的文本

        Returns:
            SubGoal 列表
        """
        # 尝试提取 JSON 数组
        json_str = content.strip()

        # 如果文本包含 markdown 代码块，提取其中的内容
        if "```" in json_str:
            import re
            match = re.search(r"```(?:json)?\s*\n(.*?)\n```", json_str, re.DOTALL)
            if match:
                json_str = match.group(1).strip()

        # 找到第一个 [ 和最后一个 ]
        start = json_str.find("[")
        end = json_str.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []

        json_str = json_str[start:end + 1]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return []

        if not isinstance(data, list):
            return []

        sub_goals: list[SubGoal] = []
        for item in data:
            if not isinstance(item, dict):
                continue

            sub_goals.append(
                SubGoal(
                    id=item.get("id", f"sg_{len(sub_goals) + 1}"),
                    description=item.get("description", ""),
                    completion_criteria=item.get("completion_criteria", ""),
                    estimated_steps=item.get("estimated_steps", 5),
                    dependencies=item.get("dependencies", []),
                )
            )

        return sub_goals

    @staticmethod
    def _is_failure_response(content: str) -> bool:
        """判断模型响应是否表示失败

        使用结构化的失败标记而非简单的关键词匹配，避免误判包含
        "失败"二字但实际成功的正常回复（如"部分测试失败是因为环境问题"）。

        Args:
            content: 模型返回的文本

        Returns:
            True 如果响应明确表示失败
        """
        # 结构化失败标记（高置信度）
        high_confidence_markers = [
            "【失败】", "STATUS: FAILED", "STATUS:ERROR",
            "执行失败：", "任务失败：", "ERROR:", "FATAL:",
        ]
        for marker in high_confidence_markers:
            if marker in content:
                return True

        # 开头就是明确的失败声明（前20字符）
        content_start = content[:20]
        failure_prefixes = ["失败", "Failed", "failed", "Error", "error", "无法完成"]
        for prefix in failure_prefixes:
            if content_start.startswith(prefix):
                return True

        return False

    @staticmethod
    def _estimate_steps_from_response(content: str) -> int:
        """从模型响应中估算步骤数

        基于响应内容估算执行步骤数（简单的启发式方法）。

        Args:
            content: 模型返回的文本

        Returns:
            估算的步骤数
        """
        # 每出现一个步骤标记算一步
        import re
        step_markers = len(re.findall(r"(?:步骤|Step|step)\s*\d+", content))
        # 每个代码文件算一步
        file_markers = len(re.findall(r"<file\s+path=", content))
        # 最少1步
        return max(1, step_markers + file_markers)

    @staticmethod
    def _extract_files_from_response(content: str) -> dict:
        """从模型响应中提取文件信息

        Args:
            content: 模型返回的文本

        Returns:
            {"created": [...], "modified": [...]}
        """
        import re

        # 提取 <file path="..."> 中的文件路径
        files = re.findall(r'<file\s+path=[\'"](.*?)[\'"]', content)

        return {
            "created": files,  # 简单处理：所有提到的文件都视为创建
            "modified": [],
        }

    @staticmethod
    def _parse_evaluation_json(content: str) -> dict:
        """解析评估结果的 JSON

        Args:
            content: 模型返回的文本

        Returns:
            评估结果字典
        """
        json_str = content.strip()

        # 提取 markdown 代码块中的内容
        if "```" in json_str:
            import re
            match = re.search(r"```(?:json)?\s*\n(.*?)\n```", json_str, re.DOTALL)
            if match:
                json_str = match.group(1).strip()

        # 找到 JSON 对象
        start = json_str.find("{")
        end = json_str.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = json_str[start:end + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # 解析失败，返回默认评估
        return {
            "assessment": "on_track",
            "summary": content[:100] if content else "无法解析评估结果",
            "recommendation": "继续执行",
        }

    @staticmethod
    def _format_progress_context(report: ProgressReport) -> str:
        """格式化进度上下文文本

        Args:
            report: 进度报告

        Returns:
            格式化的进度上下文文本
        """
        parts = [
            f"已完成子目标：{report.completed_sub_goals}/{report.total_sub_goals}",
            f"已执行步骤：{report.steps_completed}",
            f"已耗时间：{report.elapsed_minutes:.1f}分钟",
            f"策略切换次数：{report.strategy_switches}",
        ]

        # 添加各子目标状态
        if report.sub_goal_statuses:
            parts.append("\n子目标详情：")
            for sg in report.sub_goal_statuses:
                status_icon = {
                    "completed": "✓",
                    "in_progress": "→",
                    "pending": "○",
                    "failed": "✗",
                }.get(sg.get("status", ""), "?")
                parts.append(
                    f"  {status_icon} {sg.get('id', '?')}: {sg.get('description', '')} "
                    f"[{sg.get('status', '?')}]"
                )

        return "\n".join(parts)
