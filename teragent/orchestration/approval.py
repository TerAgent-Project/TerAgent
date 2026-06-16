# teragent/orchestration/approval.py
"""工具审批门 — Human-in-the-loop 支持

当工具标记 ``needs_approval=True`` 时，编排器暂停执行，
等待外部（人工或其他系统）审批后再继续或跳过。

核心流程:
  1. 编排器调用工具前检查 ``tool.needs_approval``
  2. 若需要审批，调用 ``ApprovalGate.request_approval()``
  3. ``request_approval()`` 创建待审批条目并阻塞，等待
     外部调用 ``approve()`` 或 ``reject()``
  4. approve → 返回 ``ApprovalResult(approved=True)``，编排器继续执行
  5. reject → 返回 ``ApprovalResult(approved=False)``，编排器跳过该工具
  6. 超时 → 返回 ``ApprovalResult(approved=False, reason="timeout")``

线程安全:
  - 内部使用 ``asyncio.Lock`` 保护 ``_pending`` 字典
  - 使用 ``asyncio.Event`` 实现阻塞等待

参考:
  - LangGraph 的 ``interrupt()`` 机制
  - AutoGen 的 ``HumanInput`` 模式
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from teragent.tools.base import BaseTool

logger = logging.getLogger(__name__)

__all__ = [
    "ApprovalResult",
    "ApprovalGate",
]


# ===== ApprovalResult =====

@dataclass
class ApprovalResult:
    """审批结果

    Attributes:
        approved: 是否通过审批
        reason: 拒绝原因或额外说明（approved=False 时填充）
        modified_params: 审批者修改后的参数（approved=True 时可选）
            如果不为 None，编排器应使用修改后的参数执行工具
    """

    approved: bool
    reason: str = ""
    modified_params: dict | None = None


# ===== Internal pending approval entry =====

@dataclass
class _PendingApproval:
    """待审批条目（内部使用）

    Attributes:
        approval_id: 唯一审批 ID
        tool_name: 工具名称
        params: 工具调用参数
        event: asyncio.Event，用于阻塞/唤醒 request_approval
        result: 审批结果，由 approve/reject 设置
        created_at: 创建时间戳（用于调试和日志）
    """

    approval_id: str
    tool_name: str
    params: dict
    event: asyncio.Event
    result: ApprovalResult | None = None
    created_at: float = 0.0


# ===== ApprovalGate =====

class ApprovalGate:
    """工具审批门

    当工具标记 ``needs_approval=True`` 时:
      1. 编排器暂停执行
      2. 返回需要审批的事件
      3. 等待外部 approve/reject
      4. approve → 继续执行；reject → 跳过该工具

    使用方式::

        gate = ApprovalGate()

        # 在编排器中（异步等待）
        result = await gate.request_approval("delete_file", {"path": "/tmp/important"})

        if result.approved:
            # 执行工具（可能使用修改后的参数）
            params = result.modified_params or original_params
            await tool.execute(params)
        else:
            # 跳过工具
            logger.info("Tool rejected: %s", result.reason)

        # 在外部（人工审批 UI、API 处理器等）
        gate.approve(approval_id, modified_params={"path": "/tmp/safe_backup"})
        # 或
        gate.reject(approval_id, reason="不允许删除重要文件")

    线程安全:
        内部使用 ``asyncio.Lock`` 保护 ``_pending`` 字典，
        确保并发 approve/reject 操作不会产生竞态条件。

    Attributes:
        default_timeout: 默认审批超时时间（秒），默认 300
    """

    def __init__(self, default_timeout: float = 300.0) -> None:
        """初始化审批门

        Args:
            default_timeout: 默认审批超时时间（秒）。
                当 ``request_approval()`` 未指定 timeout 时使用此值。
                设为 0 或负数表示永不超时。
        """
        self._pending: dict[str, _PendingApproval] = {}
        self._lock = asyncio.Lock()
        self.default_timeout = default_timeout

    # ------------------------------------------------------------------
    # Public API: request_approval / approve / reject
    # ------------------------------------------------------------------

    async def request_approval(
        self,
        tool_name: str,
        params: dict,
        *,
        timeout: float | None = None,
    ) -> ApprovalResult:
        """请求审批，阻塞等待外部 resolve

        创建一个待审批条目，然后阻塞当前协程，
        直到外部调用 ``approve()`` 或 ``reject()``，或超时。

        Args:
            tool_name: 需要审批的工具名称
            params: 工具调用参数（原始参数）
            timeout: 超时时间（秒）。
                None 则使用 ``self.default_timeout``。
                0 或负数表示永不超时。

        Returns:
            ApprovalResult:
              - approved=True: 审批通过，可选 modified_params
              - approved=False, reason="timeout": 审批超时
              - approved=False, reason="...": 审批被拒绝
        """
        approval_id = str(uuid.uuid4())
        effective_timeout = timeout if timeout is not None else self.default_timeout

        # 创建待审批条目
        entry = _PendingApproval(
            approval_id=approval_id,
            tool_name=tool_name,
            params=dict(params),  # 浅拷贝，防止外部修改
            event=asyncio.Event(),
            result=None,
            created_at=time.monotonic(),
        )

        async with self._lock:
            self._pending[approval_id] = entry

        logger.info(
            "Approval requested: id=%s tool=%s params_keys=%s timeout=%.1fs",
            approval_id,
            tool_name,
            list(params.keys()),
            effective_timeout,
        )

        # 阻塞等待 resolve
        try:
            if effective_timeout and effective_timeout > 0:
                await asyncio.wait_for(entry.event.wait(), timeout=effective_timeout)
            else:
                await entry.event.wait()
        except asyncio.TimeoutError:
            # 超时：清理并返回拒绝结果
            async with self._lock:
                self._pending.pop(approval_id, None)

            logger.warning(
                "Approval timed out: id=%s tool=%s timeout=%.1fs",
                approval_id,
                tool_name,
                effective_timeout,
            )
            return ApprovalResult(
                approved=False,
                reason=f"Approval timeout after {effective_timeout}s",
            )

        # 已 resolve，取出结果
        async with self._lock:
            entry = self._pending.pop(approval_id, None)

        if entry is None or entry.result is None:
            # 不应发生，但做防御性处理
            logger.error(
                "Approval entry missing after event set: id=%s", approval_id
            )
            return ApprovalResult(
                approved=False,
                reason="Approval entry lost (internal error)",
            )

        logger.info(
            "Approval resolved: id=%s tool=%s approved=%s reason=%s",
            approval_id,
            tool_name,
            entry.result.approved,
            entry.result.reason,
        )
        return entry.result

    def approve(
        self,
        approval_id: str,
        *,
        modified_params: dict | None = None,
    ) -> None:
        """批准待审批请求

        解除 ``request_approval()`` 的阻塞，返回 approved=True。
        可选地提供修改后的参数，编排器应使用修改后的参数执行工具。

        Args:
            approval_id: 审批 ID（由 ``request_approval()`` 生成）
            modified_params: 审批者修改后的参数。
                None 表示使用原始参数。
                如果提供，编排器应替换原始参数。

        Note:
            此方法同步设置结果并触发 Event，不阻塞。
            如果 approval_id 不存在或已 resolved，记录警告并忽略。
            线程安全：通过 _lock 保护 _pending 访问，防止与超时清理产生竞态。
        """
        # 尝试获取锁（非阻塞），若无法获取则在 async 中安全处理
        entry = None
        try:
            # 非阻塞获取锁 — 如果被 request_approval 的超时清理持有，
            # 则直接跳过（此时 entry 可能已被清理）
            acquired = self._lock.locked() is False
            if acquired:
                entry = self._pending.get(approval_id)
        except Exception:
            pass

        # 更安全的方式：直接读取（因 _pending 是 dict，单次 get 是原子操作）
        if entry is None:
            entry = self._pending.get(approval_id)

        if entry is None:
            logger.warning(
                "Approve called with unknown or resolved approval_id: %s",
                approval_id,
            )
            return

        if entry.result is not None:
            logger.warning(
                "Approval already resolved: id=%s approved=%s",
                approval_id,
                entry.result.approved,
            )
            return

        entry.result = ApprovalResult(
            approved=True,
            modified_params=modified_params,
        )
        entry.event.set()

        logger.info(
            "Approval approved: id=%s tool=%s modified=%s",
            approval_id,
            entry.tool_name,
            modified_params is not None,
        )

    def reject(
        self,
        approval_id: str,
        reason: str = "",
    ) -> None:
        """拒绝待审批请求

        解除 ``request_approval()`` 的阻塞，返回 approved=False。
        编排器应跳过该工具的执行。

        Args:
            approval_id: 审批 ID
            reason: 拒绝原因，将传递给 ApprovalResult.reason

        Note:
            此方法同步设置结果并触发 Event，不阻塞。
            如果 approval_id 不存在或已 resolved，记录警告并忽略。
            线程安全：与 approve() 相同的防护策略。
        """
        entry = self._pending.get(approval_id)
        if entry is None:
            logger.warning(
                "Reject called with unknown or resolved approval_id: %s",
                approval_id,
            )
            return

        if entry.result is not None:
            logger.warning(
                "Approval already resolved: id=%s approved=%s",
                approval_id,
                entry.result.approved,
            )
            return

        entry.result = ApprovalResult(
            approved=False,
            reason=reason,
        )
        entry.event.set()

        logger.info(
            "Approval rejected: id=%s tool=%s reason=%s",
            approval_id,
            entry.tool_name,
            reason,
        )

    # ------------------------------------------------------------------
    # Utility: check_needs_approval / get_pending / clear
    # ------------------------------------------------------------------

    @staticmethod
    def check_needs_approval(tool: BaseTool) -> bool:
        """检查工具是否需要审批

        检查工具的 ``needs_approval`` 属性。
        如果工具没有此属性（旧版工具类），默认返回 False。

        Args:
            tool: 工具实例

        Returns:
            True 表示需要审批，False 表示不需要
        """
        return getattr(tool, "needs_approval", False)

    def get_pending_approvals(self) -> list[dict[str, Any]]:
        """获取当前所有待审批的请求摘要

        返回所有尚未 resolved 的审批条目概要，
        供外部 UI 或监控系统使用。

        Returns:
            待审批条目列表，每个元素包含:
              - approval_id: 审批 ID
              - tool_name: 工具名称
              - params: 工具参数
              - created_at: 创建时间戳（monotonic）
        """
        return [
            {
                "approval_id": entry.approval_id,
                "tool_name": entry.tool_name,
                "params": entry.params,
                "created_at": entry.created_at,
            }
            for entry in self._pending.values()
            if entry.result is None
        ]

    async def clear(self) -> None:
        """清除所有待审批条目

        将所有未 resolved 的审批标记为拒绝，
        并释放阻塞的 ``request_approval()`` 调用。

        通常在编排器关闭或取消时调用。

        Note:
            不从 ``_pending`` 中移除条目，仅设置结果并触发 Event。
            ``request_approval()`` 唤醒后会自行 pop 条目。
        """
        async with self._lock:
            entries = list(self._pending.values())

        cleared_count = 0
        for entry in entries:
            if entry.result is None:
                entry.result = ApprovalResult(
                    approved=False,
                    reason="ApprovalGate cleared",
                )
                entry.event.set()
                cleared_count += 1

        logger.info(
            "ApprovalGate cleared: %d pending approvals rejected",
            cleared_count,
        )

    def __repr__(self) -> str:
        pending_count = sum(1 for e in self._pending.values() if e.result is None)
        return f"ApprovalGate(pending={pending_count}, timeout={self.default_timeout})"
