"""teragent.long_horizon.checkpoint — 检查点持久化存储

使用 JSON 文件存储检查点（简单可靠，不依赖 aiosqlite）。
每个任务一个目录，每次保存写入新的 JSON 文件。

设计原则：
  1. 简单可靠 — JSON 文件，无数据库依赖
  2. 支持断点续执行 — 可从任意检查点恢复
  3. 自动清理 — 保留最近 N 个检查点，防止磁盘占用过多
  4. 同步 I/O — 文件写入在 async 方法中直接执行（检查点数据量小，
     JSON 序列化速度快，不阻塞事件循环）
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """检查点数据结构

    记录长程任务在某一时刻的完整状态，用于断点续执行。

    Attributes:
        checkpoint_id: 检查点唯一标识（UUID）
        task_id: 所属任务的ID
        timestamp: ISO 8601 格式的时间戳
        phase: 当前阶段 — "planning" | "executing" | "evaluating" | "stagnant"
        completed_sub_goals: 已完成的子目标ID列表
        current_sub_goal: 当前正在执行的子目标ID
        steps_completed: 已完成的步骤数
        elapsed_minutes: 已耗时间（分钟）
        strategy_switches: 策略切换次数
        state_data: 任意状态数据（用于恢复执行上下文）
    """

    checkpoint_id: str
    task_id: str
    timestamp: str  # ISO format
    phase: str  # "planning" | "executing" | "evaluating" | "stagnant"
    completed_sub_goals: list[str]  # IDs of completed sub-goals
    current_sub_goal: str  # ID of current sub-goal
    steps_completed: int
    elapsed_minutes: float
    strategy_switches: int
    state_data: dict  # Arbitrary state for resumption

    def to_dict(self) -> dict:
        """序列化为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Checkpoint:
        """从字典反序列化

        Args:
            data: 包含检查点数据的字典

        Returns:
            Checkpoint 实例
        """
        # 兼容性处理：缺少字段时使用默认值
        return cls(
            checkpoint_id=data.get("checkpoint_id", ""),
            task_id=data.get("task_id", ""),
            timestamp=data.get("timestamp", ""),
            phase=data.get("phase", "planning"),
            completed_sub_goals=data.get("completed_sub_goals", []),
            current_sub_goal=data.get("current_sub_goal", ""),
            steps_completed=data.get("steps_completed", 0),
            elapsed_minutes=data.get("elapsed_minutes", 0.0),
            strategy_switches=data.get("strategy_switches", 0),
            state_data=data.get("state_data", {}),
        )


class CheckpointStore:
    """检查点持久化存储

    使用 JSON 文件存储检查点，每个任务一个目录。

    目录结构::

        .teragent/checkpoints/
        ├── {task_id_1}/
        │   ├── {checkpoint_id_1}.json
        │   ├── {checkpoint_id_2}.json
        │   └── ...
        └── {task_id_2}/
            ├── {checkpoint_id_3}.json
            └── ...

    使用方式::

        store = CheckpointStore(base_dir=".teragent/checkpoints")
        cp = Checkpoint(...)
        checkpoint_id = await store.save(cp)
        latest = await store.load_latest(task_id)

    Attributes:
        base_dir: 检查点文件的根目录
    """

    def __init__(self, base_dir: str = ".teragent/checkpoints") -> None:
        """初始化检查点存储

        Args:
            base_dir: 检查点文件的根目录，默认为 .teragent/checkpoints
        """
        self.base_dir = Path(base_dir)

    def _task_dir(self, task_id: str) -> Path:
        """获取任务对应的目录路径

        Args:
            task_id: 任务ID

        Returns:
            任务目录的 Path 对象
        """
        return self.base_dir / task_id

    def _checkpoint_path(self, task_id: str, checkpoint_id: str) -> Path:
        """获取检查点文件的完整路径

        Args:
            task_id: 任务ID
            checkpoint_id: 检查点ID

        Returns:
            检查点 JSON 文件的 Path 对象
        """
        return self._task_dir(task_id) / f"{checkpoint_id}.json"

    async def save(self, checkpoint: Checkpoint) -> str:
        """保存检查点

        创建任务目录（如果不存在），将检查点序列化为 JSON 写入文件。

        Args:
            checkpoint: 要保存的检查点对象

        Returns:
            检查点ID（与 checkpoint.checkpoint_id 相同）
        """
        task_dir = self._task_dir(checkpoint.task_id)
        task_dir.mkdir(parents=True, exist_ok=True)

        file_path = self._checkpoint_path(checkpoint.task_id, checkpoint.checkpoint_id)
        data = checkpoint.to_dict()

        # 同步写入（数据量小，不阻塞事件循环）
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Checkpoint saved: task={checkpoint.task_id} "
            f"checkpoint={checkpoint.checkpoint_id} phase={checkpoint.phase}"
        )
        return checkpoint.checkpoint_id

    async def load_latest(self, task_id: str) -> Optional[Checkpoint]:
        """加载指定任务的最新检查点

        按时间戳排序，返回最近的检查点。

        Args:
            task_id: 任务ID

        Returns:
            最新的 Checkpoint 对象，如果没有检查点则返回 None
        """
        checkpoints = await self.list_checkpoints(task_id)
        if not checkpoints:
            return None

        # 按时间戳降序排序，返回最新的
        checkpoints.sort(key=lambda cp: cp.timestamp, reverse=True)
        return checkpoints[0]

    async def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """根据检查点ID加载检查点

        在所有任务目录中搜索指定的检查点ID。
        由于检查点ID是 UUID，冲突概率极低。

        Args:
            checkpoint_id: 检查点ID

        Returns:
            Checkpoint 对象，如果未找到则返回 None
        """
        if not self.base_dir.exists():
            return None

        # 遍历所有任务目录查找
        for task_dir in self.base_dir.iterdir():
            if not task_dir.is_dir():
                continue
            file_path = task_dir / f"{checkpoint_id}.json"
            if file_path.exists():
                return self._read_checkpoint_file(file_path)

        return None

    async def list_checkpoints(self, task_id: str) -> list[Checkpoint]:
        """列出指定任务的所有检查点

        Args:
            task_id: 任务ID

        Returns:
            检查点列表，按时间戳升序排列
        """
        task_dir = self._task_dir(task_id)
        if not task_dir.exists():
            return []

        checkpoints: list[Checkpoint] = []
        for file_path in task_dir.glob("*.json"):
            cp = self._read_checkpoint_file(file_path)
            if cp is not None:
                checkpoints.append(cp)

        # 按时间戳升序排序
        checkpoints.sort(key=lambda cp: cp.timestamp)
        return checkpoints

    async def cleanup(self, task_id: str, keep_last: int = 5) -> int:
        """清理旧检查点，仅保留最近的 N 个

        按时间戳排序，删除最旧的检查点文件。

        Args:
            task_id: 任务ID
            keep_last: 保留最近的检查点数量，默认5

        Returns:
            删除的检查点数量
        """
        checkpoints = await self.list_checkpoints(task_id)
        if len(checkpoints) <= keep_last:
            return 0

        # 保留最新的 keep_last 个
        to_delete = checkpoints[:-keep_last]
        deleted = 0

        for cp in to_delete:
            file_path = self._checkpoint_path(task_id, cp.checkpoint_id)
            try:
                file_path.unlink()
                deleted += 1
            except OSError as e:
                logger.warning(
                    f"Failed to delete checkpoint {cp.checkpoint_id}: {e}"
                )

        logger.info(
            f"Cleaned up {deleted} old checkpoints for task={task_id}, "
            f"kept {keep_last} latest"
        )
        return deleted

    def _read_checkpoint_file(self, file_path: Path) -> Optional[Checkpoint]:
        """从 JSON 文件读取检查点

        Args:
            file_path: JSON 文件路径

        Returns:
            Checkpoint 对象，如果读取失败则返回 None
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Checkpoint.from_dict(data)
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.warning(f"Failed to read checkpoint file {file_path}: {e}")
            return None

    @staticmethod
    def generate_checkpoint_id() -> str:
        """生成唯一的检查点ID

        Returns:
            UUID 格式的检查点ID
        """
        return str(uuid.uuid4())

    @staticmethod
    def now_iso() -> str:
        """获取当前时间的 ISO 8601 格式字符串

        Returns:
            UTC 时间的 ISO 格式字符串
        """
        return datetime.now(timezone.utc).isoformat()
