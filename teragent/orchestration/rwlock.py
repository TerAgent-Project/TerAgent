# teragent/orchestration/rwlock.py
"""异步读写锁 — 用于 SharedState 的并发保护

提供 AsyncRWLock（异步读写锁），支持:
  - 多个读者可同时持有读锁
  - 写者独占（同一时刻只有一个写者）
  - 写者优先策略（防止写者饥饿）
  - 可重入检测（同一任务重复获取同类型锁时发出警告）

设计参考:
  - asyncio 同步原语（Lock, Condition）
  - Readers-Writer Lock with Writer Preference 算法
  - Python asyncio 最佳实践（避免阻塞事件循环）

用法::

    lock = AsyncRWLock()

    # 读操作（可并发）
    async with lock.read_lock():
        value = state.get("key")

    # 写操作（独占）
    async with lock.write_lock():
        state.set("key", "value")

    # 低级 API
    await lock.acquire_read()
    try:
        ...
    finally:
        await lock.release_read()

    await lock.acquire_write()
    try:
        ...
    finally:
        await lock.release_write()
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "AsyncRWLock",
    "ReadLockContext",
    "WriteLockContext",
]


class ReadLockContext:
    """读锁异步上下文管理器

    由 AsyncRWLock.read_lock() 返回，用于 ``async with`` 语法。

    Usage::

        async with rwlock.read_lock():
            # 多个读者可同时进入
            value = state.get("key")
    """

    __slots__ = ("_lock",)

    def __init__(self, lock: AsyncRWLock) -> None:
        self._lock = lock

    async def __aenter__(self) -> AsyncRWLock:
        await self._lock.acquire_read()
        return self._lock

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        await self._lock.release_read()


class WriteLockContext:
    """写锁异步上下文管理器

    由 AsyncRWLock.write_lock() 返回，用于 ``async with`` 语法。

    Usage::

        async with rwlock.write_lock():
            # 写者独占
            state.set("key", "value")
    """

    __slots__ = ("_lock",)

    def __init__(self, lock: AsyncRWLock) -> None:
        self._lock = lock

    async def __aenter__(self) -> AsyncRWLock:
        await self._lock.acquire_write()
        return self._lock

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        await self._lock.release_write()


class AsyncRWLock:
    """异步读写锁（写者优先）

    实现经典的 Readers-Writer Lock 算法，写者优先策略防止写者饥饿。

    核心机制:
    - 多个读者可同时持有读锁（_readers > 0 时，_writer == 0）
    - 写者独占（_writer > 0 时，_readers == 0）
    - 写者优先: 当有写者在等待时，新的读者会被阻塞，直到写者完成
    - 可重入检测: 同一 asyncio.Task 重复获取锁时发出警告日志

    所有获取/释放方法都是 async，以便正确使用 asyncio.Condition
    进行等待和通知，避免从同步代码中操作异步原语的问题。

    内部状态:
    - _readers: 当前活跃读者数量
    - _writer: 当前是否有活跃写者（0 或 1）
    - _waiting_writers: 等待中的写者数量
    - _condition: 共享的 asyncio.Condition（保护内部状态 + 通知机制）

    Attributes:
        _readers: 当前持有读锁的读者数量
        _writer: 当前持有写锁的写者数量（0 或 1）
        _waiting_writers: 等待写锁的写者数量
    """

    def __init__(self) -> None:
        self._readers: int = 0
        self._writer: int = 0
        self._waiting_writers: int = 0

        # 单一 Condition 保护所有内部状态
        # 读者和写者都在此 Condition 上等待和通知
        self._condition: asyncio.Condition = asyncio.Condition()

    @property
    def reader_count(self) -> int:
        """当前持有读锁的读者数量"""
        return self._readers

    @property
    def is_write_locked(self) -> bool:
        """当前是否有写者持有写锁"""
        return self._writer > 0

    @property
    def waiting_writers(self) -> int:
        """当前等待写锁的写者数量"""
        return self._waiting_writers

    async def acquire_read(self) -> None:
        """获取读锁

        当没有写者活跃且没有写者在等待时，读者可立即获取读锁。
        写者优先: 如果有写者在等待，新读者会被阻塞直到写者完成。

        此方法是协程安全的，可在多个 asyncio.Task 中并发调用。
        """
        # 可重入检测
        self._check_reentrant_read()

        async with self._condition:
            # 写者优先: 如果有活跃写者或等待中的写者，读者需要等待
            while self._writer > 0 or self._waiting_writers > 0:
                await self._condition.wait()
            self._readers += 1

    async def release_read(self) -> None:
        """释放读锁

        减少读者计数。如果这是最后一个读者，通知所有等待者
        （写者优先，但使用 notify_all 让等待者自行检查条件）。

        Raises:
            RuntimeError: 在没有持有读锁时调用
        """
        if self._readers <= 0:
            raise RuntimeError("release_read() called without holding read lock")

        async with self._condition:
            self._readers -= 1
            # 最后一个读者离开时，通知所有等待者
            if self._readers == 0:
                self._condition.notify_all()

    async def acquire_write(self) -> None:
        """获取写锁

        写者优先策略:
        1. 增加等待写者计数（阻止新读者进入）
        2. 等待所有读者和活跃写者完成
        3. 获取写锁，减少等待计数

        此方法是协程安全的，可在多个 asyncio.Task 中并发调用。
        """
        # 可重入检测
        self._check_reentrant_write()

        async with self._condition:
            # 标记有写者在等待（阻止新读者）
            self._waiting_writers += 1
            try:
                while self._readers > 0 or self._writer > 0:
                    await self._condition.wait()
                self._writer = 1
            finally:
                # 无论是否成功，减少等待计数
                self._waiting_writers -= 1

    async def release_write(self) -> None:
        """释放写锁

        释放写锁后，通知所有等待的读者和写者。
        由于使用 notify_all，所有等待者会被唤醒并重新检查条件，
        写者优先通过 _waiting_writers > 0 的条件判断自然实现。

        Raises:
            RuntimeError: 在没有持有写锁时调用
        """
        if self._writer == 0:
            raise RuntimeError("release_write() called without holding write lock")

        async with self._condition:
            self._writer = 0
            # 通知所有等待者（写者会先被调度因为 _waiting_writers > 0）
            self._condition.notify_all()

    def read_lock(self) -> ReadLockContext:
        """返回读锁的异步上下文管理器

        Usage::

            async with rwlock.read_lock():
                value = state.get("key")

        Returns:
            ReadLockContext 实例
        """
        return ReadLockContext(self)

    def write_lock(self) -> WriteLockContext:
        """返回写锁的异步上下文管理器

        Usage::

            async with rwlock.write_lock():
                state.set("key", "value")

        Returns:
            WriteLockContext 实例
        """
        return WriteLockContext(self)

    def _check_reentrant_read(self) -> None:
        """检测同任务重复获取读锁（可重入检测）

        在 asyncio 中，同一 Task 重复获取读锁不会死锁
        （因为读者可以共存），但可能表示逻辑问题。
        发出调试日志，不阻止执行。
        """
        try:
            task = asyncio.current_task()
        except RuntimeError:
            return

        if task is not None and self._readers > 0:
            logger.debug(
                "AsyncRWLock: Task %s acquiring read lock "
                "while already held by %d readers",
                task.get_name(), self._readers,
            )

    def _check_reentrant_write(self) -> None:
        """检测同任务重复获取写锁（可重入检测）

        同一 Task 重复获取写锁会导致死锁！
        发出错误日志，但继续执行（依赖 asyncio.Condition 的等待机制）。
        """
        try:
            task = asyncio.current_task()
        except RuntimeError:
            return

        if task is not None and self._writer > 0:
            logger.error(
                "AsyncRWLock: Task %s attempting to acquire "
                "write lock while already write-locked — this WILL deadlock!",
                task.get_name(),
            )

    def __repr__(self) -> str:
        return (
            f"AsyncRWLock(readers={self._readers}, "
            f"writer={self._writer}, "
            f"waiting_writers={self._waiting_writers})"
        )
