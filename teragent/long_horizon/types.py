"""teragent.long_horizon.types — 长程任务数据类型定义

定义长程任务管理器的核心数据结构：
  - SubGoal: 子目标
  - PhaseResult: 阶段执行结果
  - LongHorizonResult: 长程任务最终结果

与 teragent.core.tap 中的 LongHorizonConfig / LongHorizonStatus 配合使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SubGoal:
    """子目标

    大目标分解后的阶段性目标，每个子目标包含：
      - description: 子目标描述
      - completion_criteria: 完成标准（可量化的判断条件）
      - estimated_steps: 预估步骤数
      - dependencies: 依赖的子目标ID列表（DAG 拓扑）

    Attributes:
        id: 子目标唯一标识
        description: 子目标描述
        completion_criteria: 完成标准
        estimated_steps: 预估步骤数
        dependencies: 依赖的子目标ID列表
        status: 当前状态 — pending | in_progress | completed | failed
    """

    id: str
    description: str
    completion_criteria: str
    estimated_steps: int
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | in_progress | completed | failed


@dataclass
class PhaseResult:
    """阶段执行结果

    执行一个子目标后产出的结果记录，包含：
      - 执行是否成功
      - 模型返回的文本
      - 步骤计数
      - 创建/修改的文件列表
      - 错误信息

    Attributes:
        sub_goal_id: 对应的子目标ID
        success: 是否成功完成
        result_text: 模型返回的结果文本
        steps_taken: 本阶段消耗的步骤数
        files_created: 本阶段创建的文件列表
        files_modified: 本阶段修改的文件列表
        errors: 错误信息列表
    """

    sub_goal_id: str
    success: bool
    result_text: str
    steps_taken: int
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class LongHorizonResult:
    """长程任务最终结果

    整个长程任务执行完毕后的汇总结果，包含：
      - 整体是否成功
      - 总步骤数、总耗时
      - 子目标完成情况
      - 策略切换次数
      - 各阶段结果的详细列表
      - 最终摘要
      - 保存的检查点数量

    Attributes:
        task_id: 任务唯一标识
        goal: 原始大目标描述
        success: 整体是否成功
        total_steps: 总步骤数
        total_elapsed_minutes: 总耗时（分钟）
        completed_sub_goals: 已完成的子目标数
        total_sub_goals: 子目标总数
        strategy_switches: 策略切换次数
        phase_results: 各阶段执行结果列表
        final_summary: 最终摘要文本
        checkpoints_saved: 保存的检查点数量
    """

    task_id: str
    goal: str
    success: bool
    total_steps: int
    total_elapsed_minutes: float
    completed_sub_goals: int
    total_sub_goals: int
    strategy_switches: int
    phase_results: list[PhaseResult]
    final_summary: str
    checkpoints_saved: int
