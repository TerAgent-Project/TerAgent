"""teragent.long_horizon — GLM-5 长程任务管理模块

编排 GLM-5 的8小时持续工作能力，提供：
  - LongHorizonTaskManager: 长程任务管理器（目标分解、阶段执行、断点续执行）
  - CheckpointStore: 检查点持久化存储（JSON 文件存储）
  - ProgressTracker: 进度追踪器（步数、耗时、策略切换）
  - SelfEvaluator: 自评估执行器（周期性评估目标对齐度和产出质量）
  - StrategySwitcher: 策略切换管理器（检测停滞并引导策略切换）
  - SubGoal / PhaseResult / LongHorizonResult: 数据类型
  - Checkpoint / ProgressReport: 检查点和进度报告数据类型
  - SelfEvaluationResult / StrategySwitchRecord: 自评估和策略切换数据类型

与 AgentLoop 的集成方式::

    loop = AgentLoop(model=provider, tool_registry=registry)
    result = await loop.run_long_task(
        goal="实现一个完整的用户管理系统",
        config=LongHorizonConfig(max_duration_hours=4),
    )
"""

from teragent.long_horizon.checkpoint import Checkpoint, CheckpointStore
from teragent.long_horizon.progress import ProgressReport, ProgressTracker
from teragent.long_horizon.self_evaluation import SelfEvaluationResult, SelfEvaluator
from teragent.long_horizon.strategy_switch import StrategySwitcher, StrategySwitchRecord
from teragent.long_horizon.task_manager import LongHorizonTaskManager
from teragent.long_horizon.types import LongHorizonResult, PhaseResult, SubGoal

__all__ = [
    "LongHorizonTaskManager",
    "CheckpointStore",
    "Checkpoint",
    "ProgressTracker",
    "ProgressReport",
    "SubGoal",
    "PhaseResult",
    "LongHorizonResult",
    "SelfEvaluator",
    "SelfEvaluationResult",
    "StrategySwitcher",
    "StrategySwitchRecord",
]
