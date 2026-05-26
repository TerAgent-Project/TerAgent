# teragent/event_bus.py
"""EventBus — 信号驱动的事件总线

Part of the teragent library.

Phase 7.3 增强：支持结构化 Message 事件
  - emit_message(): 发射结构化消息事件
  - get_message_history(): 按消息类型查询事件历史
  - 事件历史存储增强：记录事件数据（不仅是名称和时间戳）

设计原则：
  - 即发即忘、永不阻塞主循环
  - 异步 handler 通过 create_task 调度
  - 同步 handler 通过 run_in_executor 调度
  - 错误隔离：单个 handler 失败不影响其他 handler
"""

import asyncio
import time
import logging
from collections import defaultdict
from typing import Callable, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from teragent.core.types import Message, MessageType

logger = logging.getLogger(__name__)


class EventBus:
    """事件总线 — 组件间异步通信的核心

    Phase 7.3 增强：
      - emit_message(): 发射结构化 Message 事件，handler 接收 Message 对象
      - get_message_history(): 按 MessageType 查询事件历史
      - _event_data_history: 存储最近事件的完整数据（用于调试和状态查询）
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._event_history: list[tuple[str, float]] = []

        # _shared 标记为 deprecated
        # 新代码应通过构造器传递依赖，不再通过 bus._shared 中转
        # 迁移指南: 搜索 `bus._shared[` 找到所有使用点，改为构造器注入
        self._shared: dict[str, Any] = {}  # DEPRECATED: 组件间共享状态

        # Phase 7.3: 增强事件历史 — 存储事件数据
        self._event_data_history: list[dict] = []
        self._max_event_data_history: int = 200

    def on(self, event_name: str, handler: Callable) -> None:
        """注册事件处理器"""
        self._subscribers[event_name].append(handler)

    async def emit(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        """发射事件（即发即忘）

        通知所有订阅了 event_name 的 handler。
        异步 handler 通过 create_task 调度，同步 handler 通过
        run_in_executor 调度。单个 handler 失败不影响其他 handler。

        注意: 异步 handler 是 fire-and-forget，不保证执行顺序或完成时机。
        如需等待所有 handler 完成，请使用 emit_and_wait()。
        """
        handlers = self._subscribers.get(event_name, [])
        timestamp = time.time()
        self._event_history.append((event_name, timestamp))
        if len(self._event_history) > 100:
            self._event_history = self._event_history[-100:]

        # Phase 7.3: 存储事件数据
        self._store_event_data(event_name, timestamp, args, kwargs)

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(self._safe_invoke_async(handler, *args, **kwargs))
                else:
                    loop = asyncio.get_running_loop()
                    # run_in_executor only accepts *args, not **kwargs;
                    # wrap the call in a lambda to pass keyword arguments.
                    # Capture handler by value via default arg to avoid late-binding closure.
                    future = loop.run_in_executor(
                        None, lambda h=handler: h(*args, **kwargs)
                    )
                    future.add_done_callback(self._on_sync_done)
            except Exception as e:
                logger.error(f"Error dispatching '{event_name}': {e}")

    async def emit_and_wait(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        """2.4: 发射事件并等待所有 handler 完成

        与 emit() 的区别:
          - emit() 是 fire-and-forget（即发即忘），异步 handler 通过 create_task 调度
          - emit_and_wait() 会等待所有异步 handler 执行完毕后才返回

        适用场景:
          - 状态更新关键路径（如 subtask_completed、subtask_failed）
          - 需要确保 handler 完成后才继续的场景
          - 避免异步 handler 并发执行导致状态更新竞态

        向后兼容: 原有 emit() 方法不变，仅关键路径升级为 emit_and_wait()。
        """
        handlers = self._subscribers.get(event_name, [])
        timestamp = time.time()
        self._event_history.append((event_name, timestamp))
        if len(self._event_history) > 100:
            self._event_history = self._event_history[-100:]

        # 存储事件数据
        self._store_event_data(event_name, timestamp, args, kwargs)

        if not handlers:
            return

        # 收集所有需要等待的任务
        async_tasks: list = []
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    async_tasks.append(
                        self._safe_invoke_async(handler, *args, **kwargs)
                    )
                else:
                    # 同步 handler 直接执行（在当前协程中）
                    try:
                        handler(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Sync handler {handler.__name__} error for event '{event_name}': {e}")
            except Exception as e:
                logger.error(f"Error preparing '{event_name}' handler: {e}")

        # 等待所有异步 handler 完成
        if async_tasks:
            await asyncio.gather(*async_tasks, return_exceptions=True)

    async def emit_message(self, event_name: str, message: "Message") -> None:
        """Phase 7.3: 发射结构化消息事件

        与 emit() 不同，emit_message() 专门用于消息相关事件，
        自动记录消息的元数据（role, type, content 预览）。

        Handler 接收 Message 对象作为第一个位置参数。

        Args:
            event_name: 事件名称（如 "agent_loop_message_added"）
            message: Message 对象
        """
        timestamp = time.time()
        self._event_history.append((event_name, timestamp))
        if len(self._event_history) > 100:
            self._event_history = self._event_history[-100:]

        # 存储消息事件的增强数据
        self._store_event_data(
            event_name,
            timestamp,
            args=(),
            kwargs={"message": message},
            message_meta={
                "role": message.role.value,
                "message_type": message.message_type.value,
                "content_preview": message.content[:100] if message.content else "",
                "tool_name": message.tool_name,
                "has_tool_calls": bool(message.tool_calls),
            },
        )

        # 通知所有 handler
        handlers = self._subscribers.get(event_name, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(
                        self._safe_invoke_async(handler, message)
                    )
                else:
                    loop = asyncio.get_running_loop()
                    future = loop.run_in_executor(None, handler, message)
                    future.add_done_callback(self._on_sync_done)
            except Exception as e:
                logger.error(f"Error dispatching message event '{event_name}': {e}")

    def _store_event_data(
        self,
        event_name: str,
        timestamp: float,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        message_meta: Optional[dict] = None,
    ) -> None:
        """Phase 7.3: 存储事件数据到历史记录

        Args:
            event_name: 事件名称
            timestamp: 事件时间戳
            args: 位置参数（摘要存储，避免内存膨胀）
            kwargs: 关键字参数（摘要存储）
            message_meta: 消息元数据（仅 emit_message 时提供）
        """
        event_data = {
            "event_name": event_name,
            "timestamp": timestamp,
            "args_count": len(args),
            "kwargs_keys": list(kwargs.keys()) if kwargs else [],
        }

        # 存储关键字参数的摘要（截断长值）
        if kwargs:
            event_data["kwargs_summary"] = {
                k: (str(v)[:100] if isinstance(v, (str, int, float, bool)) else type(v).__name__)
                for k, v in kwargs.items()
            }

        # 存储消息元数据
        if message_meta:
            event_data["message_meta"] = message_meta

        self._event_data_history.append(event_data)
        if len(self._event_data_history) > self._max_event_data_history:
            self._event_data_history = self._event_data_history[-self._max_event_data_history:]

    async def _safe_invoke_async(self, handler: Callable, *args: Any, **kwargs: Any) -> None:
        """安全调用异步 handler"""
        try:
            await handler(*args, **kwargs)
        except Exception as e:
            logger.error(f"Async handler {handler.__name__} error: {e}", exc_info=True)

    def _on_sync_done(self, future: asyncio.Future) -> None:
        """同步 handler 完成回调"""
        if future.cancelled():
            return
        exc = future.exception()
        if exc:
            logger.error(f"Synchronous handler failed: {exc}", exc_info=True)

    def remove(self, event_name: str, handler: Callable) -> None:
        """移除事件处理器"""
        if event_name in self._subscribers:
            self._subscribers[event_name] = [h for h in self._subscribers[event_name] if h != handler]

    def clear(self) -> None:
        """清除所有订阅和事件数据历史"""
        self._subscribers.clear()
        self._event_history.clear()
        self._event_data_history.clear()

    def once(self, event_name: str, handler: Callable) -> None:
        """注册一次性事件处理器，触发一次后自动移除"""
        _fired = False
        async def _async_wrapper(*args: Any, **kwargs: Any) -> None:
            nonlocal _fired
            if _fired:
                return
            _fired = True
            self.remove(event_name, _async_wrapper)
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(*args, **kwargs)
                else:
                    handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"once handler error: {e}", exc_info=True)

        self.on(event_name, _async_wrapper)

    async def wait_for(self, event_name: str, timeout: float = 30.0) -> tuple:
        """等待某个事件触发，返回事件参数。超时抛出 asyncio.TimeoutError"""
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        wrapper_ref: list[Callable | None] = [None]

        def _resolver(*args: Any, **kwargs: Any) -> None:
            if not future.done():
                future.set_result((args, kwargs))

        # once() wraps _resolver in an async _async_wrapper; we need to
        # track the wrapper so we can remove it on timeout.
        original_subscribers = list(self._subscribers.get(event_name, []))
        self.once(event_name, _resolver)
        # Find the newly added wrapper (last subscriber for this event)
        current_subscribers = self._subscribers.get(event_name, [])
        for sub in reversed(current_subscribers):
            if sub not in original_subscribers:
                wrapper_ref[0] = sub
                break

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            if wrapper_ref[0] is not None:
                self.remove(event_name, wrapper_ref[0])
            raise

    # ===== 查询方法 =====

    def get_event_names(self) -> list[str]:
        """返回所有已注册的事件名称"""
        return list(self._subscribers.keys())

    def handler_count(self, event_name: str) -> int:
        """返回指定事件的处理器数量"""
        return len(self._subscribers.get(event_name, []))

    def get_event_history(self, limit: int = 50) -> list[tuple[str, float]]:
        """返回最近的事件历史记录（事件名, 时间戳）"""
        return self._event_history[-limit:]

    def get_event_data_history(
        self,
        event_name: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Phase 7.3: 返回增强的事件数据历史

        Args:
            event_name: 如果指定，只返回该名称的事件
            limit: 最多返回的记录数

        Returns:
            事件数据列表，每条记录包含 event_name, timestamp, args_count, kwargs_keys 等
        """
        history = self._event_data_history
        if event_name:
            history = [e for e in history if e["event_name"] == event_name]
        return history[-limit:]

    def get_message_event_history(
        self,
        message_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Phase 7.3: 返回消息事件历史

        Args:
            message_type: 如果指定，只返回该 MessageType 的消息事件
            limit: 最多返回的记录数

        Returns:
            包含 message_meta 的事件数据列表
        """
        history = [
            e for e in self._event_data_history
            if "message_meta" in e
        ]
        if message_type:
            history = [
                e for e in history
                if e["message_meta"].get("message_type") == message_type
            ]
        return history[-limit:]
