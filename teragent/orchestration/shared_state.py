# teragent/orchestration/shared_state.py
"""跨 Agent 共享状态

参考 Google ADK 的 session.state, OpenAI Agents SDK 的 RunContext。

功能:
  - set/get/delete 操作
  - 作用域支持（session, agent, global）
  - snapshot/restore 用于检查点
  - 写入日志记录
  - 冲突检测
  - 异步读写锁（可选，用于并行编排场景）

并发保护（Phase 3 / W12 新增）:
  - SharedState 可选启用 AsyncRWLock 保护并发读写
  - 启用后: get() 获取读锁, set()/delete() 获取写锁
  - 未启用时: 行为与之前完全一致（零开销）
  - 也可通过 read_lock()/write_lock() 上下文管理器手动控制
  - Lazy 初始化: lock 仅在首次需要时创建

用法::

    # 不使用锁（向后兼容，原有代码无需修改）
    state = SharedState()
    state.set("key", "value")
    value = state.get("key")

    # 启用锁（并行编排场景）
    state = SharedState(enable_lock=True)

    # 自动加锁的 get/set/delete
    await state.async_get("key")     # 自动获取读锁
    await state.async_set("key", v)  # 自动获取写锁
    await state.async_delete("key")  # 自动获取写锁

    # 手动控制锁的范围（批量操作）
    async with state.read_lock():
        v1 = state.get("k1")
        v2 = state.get("k2")

    async with state.write_lock():
        state.set("k1", v1 + v2)
        state.delete("k2")
"""

from __future__ import annotations

import copy
import time
import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from teragent.orchestration.rwlock import AsyncRWLock, ReadLockContext, WriteLockContext

logger = logging.getLogger(__name__)

__all__ = [
    "SharedState",
    "ScopedState",
    "StateWrite",
]


@dataclass
class StateWrite:
    """状态写入记录

    记录每次状态写入的元数据，用于审计和冲突检测。

    Attributes:
        agent_name: 执行写入的 Agent 名称
        key: 状态键
        value: 写入的值
        timestamp: 写入时间戳
        scope: 作用域
    """

    agent_name: str
    key: str
    value: Any
    timestamp: float
    scope: str = "session"


class SharedState:
    """跨 Agent 共享状态

    参考 Google ADK 的 session.state, OpenAI Agents SDK 的 RunContext。

    功能:
    - set/get/delete 操作
    - 作用域支持（session, agent, global）
    - snapshot/restore 用于检查点
    - 写入日志记录
    - 冲突检测
    - 异步读写锁（可选，用于并行编排场景）

    作用域约定:
    - session (默认): 会话级状态，key 不带前缀
    - agent: Agent 级状态，key 格式为 "agent:key"
    - global: 全局状态，key 格式为 "global:key"

    并发保护:
    - enable_lock=True 时启用 AsyncRWLock
    - get() / async_get() 获取读锁
    - set() / async_set() 获取写锁
    - delete() / async_delete() 获取写锁
    - 未启用锁时，所有操作无锁开销

    Args:
        enable_lock: 是否启用异步读写锁（默认 False，保持向后兼容）
    """

    def __init__(self, enable_lock: bool = False) -> None:
        self._data: dict[str, Any] = {}
        self._scopes: dict[str, str] = {}  # full_key → scope
        self._write_log: list[StateWrite] = []
        self._enable_lock = enable_lock
        self._rwlock: AsyncRWLock | None = None

        if enable_lock:
            self._init_lock()

    def _init_lock(self) -> None:
        """延迟初始化读写锁

        仅在需要时创建 AsyncRWLock 实例，
        避免在不需要锁的场景中引入额外开销。
        """
        if self._rwlock is None:
            from teragent.orchestration.rwlock import AsyncRWLock
            self._rwlock = AsyncRWLock()
            self._enable_lock = True

    @property
    def lock_enabled(self) -> bool:
        """是否已启用读写锁"""
        return self._enable_lock and self._rwlock is not None

    # ===== 同步方法（向后兼容，不加锁） =====

    def set(self, key: str, value: Any, scope: str = "session", agent_name: str = "") -> None:
        """设置状态值（同步，不加锁）

        在并行编排场景中，建议使用 async_set() 替代此方法。

        Args:
            key: 状态键
            value: 状态值
            scope: 作用域（session, agent, global）
            agent_name: 写入的 Agent 名称
        """
        full_key = f"{scope}:{key}" if scope != "session" else key

        # 冲突检测（使用 try/except 防止不可比较类型的 != 抛异常）
        if full_key in self._data:
            old_value = self._data[full_key]
            try:
                is_different = old_value != value
            except Exception:
                is_different = True
            if is_different:
                logger.debug(f"SharedState key '{full_key}' overwritten by {agent_name}")

        self._data[full_key] = value
        self._scopes[full_key] = scope
        self._write_log.append(StateWrite(
            agent_name=agent_name,
            key=full_key,
            value=value,
            timestamp=time.time(),
            scope=scope,
        ))

    def get(self, key: str, default: Any = None, scope: str = "session") -> Any:
        """获取状态值（同步，不加锁）

        在并行编排场景中，建议使用 async_get() 替代此方法。

        Args:
            key: 状态键
            default: 默认值
            scope: 作用域

        Returns:
            状态值，或 default
        """
        full_key = f"{scope}:{key}" if scope != "session" else key
        return self._data.get(full_key, default)

    def delete(self, key: str, scope: str = "session") -> bool:
        """删除状态值（同步，不加锁）

        在并行编排场景中，建议使用 async_delete() 替代此方法。

        Args:
            key: 状态键
            scope: 作用域

        Returns:
            True 表示成功删除，False 表示键不存在
        """
        full_key = f"{scope}:{key}" if scope != "session" else key
        if full_key in self._data:
            del self._data[full_key]
            self._scopes.pop(full_key, None)
            return True
        return False

    # ===== 异步方法（带锁保护） =====

    async def async_set(self, key: str, value: Any, scope: str = "session", agent_name: str = "") -> None:
        """设置状态值（异步，自动获取写锁）

        如果启用了读写锁，此方法会自动获取写锁后执行写操作。
        如果未启用锁，行为与 set() 完全一致。

        Args:
            key: 状态键
            value: 状态值
            scope: 作用域（session, agent, global）
            agent_name: 写入的 Agent 名称
        """
        if self._rwlock is not None:
            async with self._rwlock.write_lock():
                self.set(key, value, scope=scope, agent_name=agent_name)
        else:
            self.set(key, value, scope=scope, agent_name=agent_name)

    async def async_get(self, key: str, default: Any = None, scope: str = "session") -> Any:
        """获取状态值（异步，自动获取读锁）

        如果启用了读写锁，此方法会自动获取读锁后执行读操作。
        如果未启用锁，行为与 get() 完全一致。

        Args:
            key: 状态键
            default: 默认值
            scope: 作用域

        Returns:
            状态值，或 default
        """
        if self._rwlock is not None:
            async with self._rwlock.read_lock():
                return self.get(key, default=default, scope=scope)
        return self.get(key, default=default, scope=scope)

    async def async_delete(self, key: str, scope: str = "session") -> bool:
        """删除状态值（异步，自动获取写锁）

        如果启用了读写锁，此方法会自动获取写锁后执行删除操作。
        如果未启用锁，行为与 delete() 完全一致。

        Args:
            key: 状态键
            scope: 作用域

        Returns:
            True 表示成功删除，False 表示键不存在
        """
        if self._rwlock is not None:
            async with self._rwlock.write_lock():
                return self.delete(key, scope=scope)
        return self.delete(key, scope=scope)

    # ===== 锁上下文管理器 =====

    def read_lock(self) -> ReadLockContext:
        """返回读锁的异步上下文管理器

        用于批量读操作，手动控制锁的范围。
        如果未启用锁，会自动初始化锁。

        Usage::

            async with state.read_lock():
                v1 = state.get("k1")
                v2 = state.get("k2")

        Returns:
            ReadLockContext 实例
        """
        self._init_lock()
        return self._rwlock.read_lock()  # type: ignore[union-attr]

    def write_lock(self) -> WriteLockContext:
        """返回写锁的异步上下文管理器

        用于批量写操作，手动控制锁的范围。
        如果未启用锁，会自动初始化锁。

        Usage::

            async with state.write_lock():
                state.set("k1", "v1")
                state.delete("k2")

        Returns:
            WriteLockContext 实例
        """
        self._init_lock()
        return self._rwlock.write_lock()  # type: ignore[union-attr]

    # ===== 原有方法（保持不变） =====

    def has(self, key: str, scope: str = "session") -> bool:
        """检查键是否存在

        Args:
            key: 状态键
            scope: 作用域

        Returns:
            键是否存在
        """
        full_key = f"{scope}:{key}" if scope != "session" else key
        return full_key in self._data

    def keys(self, scope: str | None = None) -> list[str]:
        """获取所有键

        Args:
            scope: 作用域过滤，None 表示全部

        Returns:
            键列表
        """
        if scope is None:
            return list(self._data.keys())
        if scope == "session":
            return [k for k, s in self._scopes.items() if s == "session"]
        return [k for k, s in self._scopes.items() if s == scope]

    def to_dict(self) -> dict[str, Any]:
        """返回状态字典的浅拷贝"""
        return dict(self._data)

    async def async_snapshot(self) -> dict:
        """创建快照（深拷贝），用于检查点（异步，带锁保护）

        如果启用了读写锁，获取读锁后创建快照，确保快照一致性。
        如果未启用锁，行为与 snapshot() 完全一致。

        Returns:
            包含 data、scopes 和 write_log 的深拷贝字典
        """
        if self._rwlock is not None:
            async with self._rwlock.read_lock():
                return {
                    "data": copy.deepcopy(self._data),
                    "scopes": dict(self._scopes),
                    "write_log": list(self._write_log),
                }
        return self.snapshot()

    def snapshot(self) -> dict:
        """创建快照（深拷贝），用于检查点（同步，不加锁）

        注意：在并行编排场景中，建议使用 async_snapshot() 替代此方法，
        以确保快照时数据的一致性。

        Returns:
            包含 data、scopes 和 write_log 的深拷贝字典
        """
        return {
            "data": copy.deepcopy(self._data),
            "scopes": dict(self._scopes),
            "write_log": list(self._write_log),
        }

    def restore(self, snapshot: dict) -> None:
        """从快照恢复状态

        Args:
            snapshot: 之前 snapshot() 返回的字典
        """
        self._data = copy.deepcopy(snapshot.get("data", {}))
        self._scopes = dict(snapshot.get("scopes", {}))
        self._write_log = list(snapshot.get("write_log", []))

    def scope(self, prefix: str) -> ScopedState:
        """获取子作用域视图

        返回一个 ScopedState 视图，所有操作自动添加作用域前缀。

        Args:
            prefix: 作用域前缀

        Returns:
            ScopedState 实例
        """
        return ScopedState(self, prefix)

    def get_write_log(self, agent_name: str | None = None) -> list[StateWrite]:
        """获取写入日志

        Args:
            agent_name: 过滤指定 Agent 的写入记录，None 表示全部

        Returns:
            写入记录列表
        """
        if agent_name:
            return [w for w in self._write_log if w.agent_name == agent_name]
        return list(self._write_log)

    def clear(self) -> None:
        """清空状态"""
        self._data.clear()
        self._scopes.clear()
        self._write_log.clear()

    def __repr__(self) -> str:
        lock_status = f", locked={self._enable_lock}" if self._enable_lock else ""
        return f"SharedState(keys={len(self._data)}, writes={len(self._write_log)}{lock_status})"


class ScopedState:
    """作用域状态视图

    提供对 SharedState 的作用域访问，
    所有操作自动添加作用域前缀。

    使用方式:
        state = SharedState()
        agent_state = state.scope("agent:researcher")
        agent_state.set("findings", "...")
        agent_state.get("findings")  # 自动添加 "agent:researcher:" 前缀
    """

    def __init__(self, state: SharedState, scope: str) -> None:
        self._state = state
        self._scope = scope

    def set(self, key: str, value: Any, agent_name: str = "") -> None:
        """设置作用域内的状态值"""
        self._state.set(key, value, scope=self._scope, agent_name=agent_name)

    def get(self, key: str, default: Any = None) -> Any:
        """获取作用域内的状态值"""
        return self._state.get(key, default=default, scope=self._scope)

    def delete(self, key: str) -> bool:
        """删除作用域内的状态值"""
        return self._state.delete(key, scope=self._scope)

    def has(self, key: str) -> bool:
        """检查作用域内的键是否存在"""
        return self._state.has(key, scope=self._scope)

    async def async_set(self, key: str, value: Any, agent_name: str = "") -> None:
        """设置作用域内的状态值（异步，带锁保护）"""
        await self._state.async_set(key, value, scope=self._scope, agent_name=agent_name)

    async def async_get(self, key: str, default: Any = None) -> Any:
        """获取作用域内的状态值（异步，带锁保护）"""
        return await self._state.async_get(key, default=default, scope=self._scope)

    async def async_delete(self, key: str) -> bool:
        """删除作用域内的状态值（异步，带锁保护）"""
        return await self._state.async_delete(key, scope=self._scope)

    def __repr__(self) -> str:
        return f"ScopedState(scope={self._scope!r})"
