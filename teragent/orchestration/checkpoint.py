"""teragent.orchestration.checkpoint — 编排检查点

保存编排的完整状态，支持恢复。
复用现有 teragent/long_horizon/checkpoint.py 的存储机制。

设计原则:
  1. 复用 CheckpointStore 的目录结构和文件管理能力
  2. 原子写入（先写临时文件再 rename）确保数据完整性
  3. 自动生成检查点 ID (UUID)
  4. 支持列出和删除指定运行的所有检查点
  5. 将编排状态映射到 Checkpoint.state_data，兼顾两种数据模型
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from teragent.long_horizon.checkpoint import Checkpoint, CheckpointStore

if TYPE_CHECKING:
    from teragent.orchestration.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

__all__ = [
    "OrchestrationCheckpoint",
]


class OrchestrationCheckpoint:
    """编排检查点

    保存编排的完整状态，支持恢复。
    复用现有 teragent/long_horizon/checkpoint.py 的存储机制。

    编排状态保存为 Checkpoint.state_data，包含:
      - shared_state: SharedState 快照 (data, scopes, write_log)
      - current_agent: 当前活跃的 Agent 名称
      - turn: 当前执行轮次
      - messages: 消息历史
      - agent_metadata: Agent 元数据列表 (name, description, output_key)
      - mode: 编排模式 (sequential / swarm / parallel / conditional / loop)

    使用方式::

        cp = OrchestrationCheckpoint()

        # 保存检查点
        cp_id = await cp.save(
            orchestrator,
            run_id="run_001",
            current_agent="researcher",
            turn=3,
            messages=[{"role": "user", "content": "..."}],
        )

        # 恢复最新检查点
        state = await cp.restore("run_001")

        # 恢复指定检查点
        state = await cp.restore("run_001", checkpoint_id=cp_id)

        # 列出检查点
        summaries = await cp.list_checkpoints("run_001")

        # 删除检查点
        await cp.delete_checkpoint("run_001", cp_id)

        # 清理旧检查点，仅保留最近 5 个
        deleted = await cp.cleanup("run_001", keep_last=5)

    Attributes:
        store: 底层 CheckpointStore 实例
    """

    def __init__(self, base_dir: str = ".teragent/orchestration_checkpoints") -> None:
        """初始化编排检查点管理器

        Args:
            base_dir: 检查点文件的根目录，
                默认为 .teragent/orchestration_checkpoints
        """
        self._store = CheckpointStore(base_dir)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def store(self) -> CheckpointStore:
        """底层 CheckpointStore 实例"""
        return self._store

    # ------------------------------------------------------------------
    # Core: save / restore
    # ------------------------------------------------------------------

    async def save(
        self,
        orchestrator: Orchestrator,
        run_id: str,
        *,
        current_agent: str = "",
        turn: int = 0,
        messages: list[dict[str, Any]] | None = None,
    ) -> str:
        """保存检查点

        将编排器的完整状态保存为检查点文件。
        自动生成唯一的检查点 ID (UUID)，使用原子写入确保数据完整性。

        Args:
            orchestrator: 编排器实例
            run_id: 编排运行 ID，用于分组检查点。
                对应 CheckpointStore 中的 task_id。
            current_agent: 当前活跃的 Agent 名称。
                为空时默认使用编排器第一个 Agent 的名称。
            turn: 当前执行轮次
            messages: 消息历史列表，每个元素为一个消息字典

        Returns:
            生成的检查点 ID (UUID 字符串)

        Raises:
            OSError: 文件写入失败
        """
        checkpoint_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # 确定当前 Agent
        effective_current_agent = current_agent
        if not effective_current_agent and orchestrator.agents:
            effective_current_agent = orchestrator.agents[0].name

        # 构建完整的编排状态
        state: dict[str, Any] = {
            "shared_state": orchestrator._shared_state.snapshot(),
            "current_agent": effective_current_agent,
            "turn": turn,
            "messages": messages or [],
            "agent_metadata": [
                {
                    "name": agent.name,
                    "description": agent.description,
                    "output_key": agent.output_key,
                }
                for agent in orchestrator.agents
            ],
            "mode": orchestrator.mode.value,
        }

        # 映射到 Checkpoint 数据结构，编排状态存入 state_data
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            task_id=run_id,
            timestamp=timestamp,
            phase="orchestrating",
            completed_sub_goals=[],
            current_sub_goal=effective_current_agent,
            steps_completed=turn,
            elapsed_minutes=0.0,
            strategy_switches=0,
            state_data=state,
        )

        # 原子写入 JSON
        data = checkpoint.to_dict()
        await self._atomic_write_json(run_id, checkpoint_id, data)

        logger.info(
            "Orchestration checkpoint saved: run_id=%s "
            "checkpoint_id=%s current_agent=%s turn=%d",
            run_id,
            checkpoint_id,
            effective_current_agent,
            turn,
        )

        return checkpoint_id

    async def restore(
        self,
        run_id: str,
        checkpoint_id: str | None = None,
    ) -> dict[str, Any]:
        """恢复检查点

        从检查点文件恢复编排状态。
        如果指定 checkpoint_id，恢复该特定检查点；
        否则恢复指定 run_id 下最新的检查点。

        返回的状态字典可直接用于重建编排器::

            state = await cp.restore("run_001")

            # 重建 SharedState
            shared_state = SharedState()
            shared_state.restore(state["shared_state"])

            # 获取其他字段
            current_agent = state["current_agent"]
            turn = state["turn"]
            messages = state["messages"]
            mode = state["mode"]

        Args:
            run_id: 编排运行 ID
            checkpoint_id: 检查点 ID，None 则恢复最新的

        Returns:
            包含完整编排状态的字典，结构如下:
              - shared_state: SharedState 快照字典
              - current_agent: 当前 Agent 名称
              - turn: 当前轮次
              - messages: 消息历史
              - agent_metadata: Agent 元数据列表
              - mode: 编排模式
              - checkpoint_id: 检查点 ID
              - timestamp: 检查点时间戳

        Raises:
            FileNotFoundError: 指定的检查点不存在
        """
        if checkpoint_id is not None:
            checkpoint = await self._store.load(checkpoint_id)
        else:
            checkpoint = await self._store.load_latest(run_id)

        if checkpoint is None:
            raise FileNotFoundError(
                f"No checkpoint found for run_id={run_id}, "
                f"checkpoint_id={checkpoint_id}"
            )

        # 从 Checkpoint.state_data 中提取编排状态
        state = dict(checkpoint.state_data)
        # 补充 Checkpoint 级别的元信息
        state["checkpoint_id"] = checkpoint.checkpoint_id
        state["timestamp"] = checkpoint.timestamp

        logger.info(
            "Orchestration checkpoint restored: run_id=%s "
            "checkpoint_id=%s current_agent=%s turn=%d",
            run_id,
            checkpoint.checkpoint_id,
            state.get("current_agent", ""),
            state.get("turn", 0),
        )

        return state

    # ------------------------------------------------------------------
    # Management: list / delete / cleanup
    # ------------------------------------------------------------------

    async def list_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        """列出指定运行的所有检查点

        委托给 CheckpointStore.list_checkpoints() 并提取编排相关的摘要信息。

        Args:
            run_id: 编排运行 ID

        Returns:
            检查点摘要列表，按时间戳升序排列。
            每个元素包含:
              - checkpoint_id: 检查点 ID
              - timestamp: 时间戳
              - current_agent: 当前 Agent
              - turn: 轮次
              - mode: 编排模式
        """
        checkpoints = await self._store.list_checkpoints(run_id)
        return [
            {
                "checkpoint_id": cp.checkpoint_id,
                "timestamp": cp.timestamp,
                "current_agent": cp.current_sub_goal,
                "turn": cp.steps_completed,
                "mode": cp.state_data.get("mode", ""),
            }
            for cp in checkpoints
        ]

    async def delete_checkpoint(self, run_id: str, checkpoint_id: str) -> bool:
        """删除指定检查点

        Args:
            run_id: 编排运行 ID
            checkpoint_id: 检查点 ID

        Returns:
            True 表示成功删除，False 表示检查点不存在或删除失败
        """
        file_path = self._store._checkpoint_path(run_id, checkpoint_id)
        loop = asyncio.get_running_loop()

        def _delete() -> bool:
            try:
                file_path.unlink()
                return True
            except FileNotFoundError:
                logger.warning(
                    "Checkpoint not found for deletion: %s", checkpoint_id
                )
                return False
            except OSError as e:
                logger.error(
                    "Failed to delete checkpoint %s: %s", checkpoint_id, e
                )
                return False

        result = await loop.run_in_executor(None, _delete)
        if result:
            logger.info(
                "Checkpoint deleted: run_id=%s checkpoint_id=%s",
                run_id,
                checkpoint_id,
            )
        return result

    async def cleanup(self, run_id: str, keep_last: int = 5) -> int:
        """清理旧检查点，仅保留最近的 N 个

        委托给 CheckpointStore.cleanup() 执行。
        按时间戳排序，删除最旧的检查点文件。

        Args:
            run_id: 编排运行 ID
            keep_last: 保留最近的检查点数量，默认 5

        Returns:
            删除的检查点数量
        """
        return await self._store.cleanup(run_id, keep_last)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _atomic_write_json(
        self,
        run_id: str,
        checkpoint_id: str,
        data: dict[str, Any],
    ) -> None:
        """原子写入 JSON 文件

        先写入临时文件，然后原子性地重命名为目标文件。
        确保在写入过程中崩溃不会损坏已有数据。

        使用 os.replace() 实现原子重命名，在 POSIX 和 Windows
        上均能保证目标路径的原子替换。

        Args:
            run_id: 编排运行 ID，用于确定目录
            checkpoint_id: 检查点 ID，用于确定文件名
            data: 要序列化为 JSON 的数据字典

        Raises:
            OSError: 文件写入或重命名失败
        """
        task_dir = self._store._task_dir(run_id)
        target_path = self._store._checkpoint_path(run_id, checkpoint_id)

        loop = asyncio.get_running_loop()

        def _write() -> None:
            # 确保目录存在
            task_dir.mkdir(parents=True, exist_ok=True)

            # 在同一目录下创建临时文件，保证同一文件系统
            fd, tmp_path_str = tempfile.mkstemp(
                suffix=".json.tmp",
                dir=str(task_dir),
            )
            tmp_path = Path(tmp_path_str)

            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                # 原子性重命名（os.replace 在同一文件系统上是原子的）
                os.replace(str(tmp_path), str(target_path))
            except BaseException:
                # 写入或重命名失败时清理临时文件
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise

        await loop.run_in_executor(None, _write)
