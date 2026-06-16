"""teragent.orchestration.cancellation — 取消令牌

CancellationToken 支持编排执行的取消操作：
- 同步检查: is_cancelled 属性
- 同步抛出: throw_if_cancelled() 方法
- 异步等待: wait() 方法

参考:
- AutoGen 的 CancellationToken + ExternalTermination
- .NET 的 CancellationToken 模式
"""
from __future__ import annotations

import asyncio
import threading


class CancellationToken:
    """取消令牌

    参考 AutoGen 的 CancellationToken + ExternalTermination。
    支持同步检查和异步等待。

    用法:
        # 创建令牌
        token = CancellationToken()

        # 在编排器中检查
        token.throw_if_cancelled()

        # 从外部触发取消
        token.cancel()

        # 异步等待取消信号
        await token.wait(timeout=30.0)

    线程安全说明:
        CancellationToken 主要用于异步环境（asyncio）。
        cancel() 方法可从任意线程调用。
        wait() 方法必须在 asyncio 事件循环中调用。
    """

    def __init__(self) -> None:
        self._cancelled: bool = False
        self._waiters: list[asyncio.Future] = []
        self._lock = threading.Lock()

    def cancel(self) -> None:
        """触发取消

        通知所有等待者取消信号已发出。
        可以安全地多次调用（幂等）。
        线程安全：可从任意线程调用。
        """
        with self._lock:
            if self._cancelled:
                return

            self._cancelled = True
            waiters = list(self._waiters)
            self._waiters.clear()

        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(None)

    @property
    def is_cancelled(self) -> bool:
        """是否已取消

        线程安全：通过 lock 保护读取，避免与 cancel() 的写操作产生数据竞争。
        """
        with self._lock:
            return self._cancelled

    def throw_if_cancelled(self) -> None:
        """如果已取消，抛出 CancelledError

        用于在长时间运行的操作中插入取消检查点。
        不会阻塞，适合在同步代码中使用。
        线程安全：通过 lock 保护读取。

        Raises:
            asyncio.CancelledError: 如果已取消
        """
        with self._lock:
            cancelled = self._cancelled
        if cancelled:
            raise asyncio.CancelledError(
                "Operation was cancelled via CancellationToken"
            )

    async def wait(self, timeout: float | None = None) -> bool:
        """等待取消信号

        阻塞当前协程，直到取消信号发出或超时。
        必须在 asyncio 事件循环中调用。

        Args:
            timeout: 超时秒数，None 表示无限等待

        Returns:
            True 如果被取消，False 如果超时

        Note:
            如果已取消，立即返回 True。
        """
        with self._lock:
            if self._cancelled:
                return True

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        with self._lock:
            if self._cancelled:
                return True
            self._waiters.append(future)

        try:
            if timeout is not None:
                await asyncio.wait_for(future, timeout=timeout)
            else:
                await future
        except asyncio.TimeoutError:
            return False
        finally:
            with self._lock:
                if future in self._waiters:
                    self._waiters.remove(future)

        return True

    def __repr__(self) -> str:
        return f"CancellationToken(cancelled={self._cancelled})"
