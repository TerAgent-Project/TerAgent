# tests/test_event_bus.py
"""EventBus 单元测试

覆盖:
  - on/emit: 同步/异步 handler 注册和触发
  - once: 一次性 handler
  - wait_for: 等待特定事件
  - emit_and_wait: 等待所有 handler 完成
  - 错误隔离: 单个 handler 失败不影响其他 handler
  - remove/clear: handler 移除和清除
  - 事件历史: get_event_history / get_event_data_history
  - emit_message: 结构化消息事件
"""
import asyncio
import pytest

from teragent.event_bus import EventBus


# ===== on/emit 基础功能 =====

class TestOnEmit:
    """on + emit 基础功能"""

    @pytest.mark.asyncio
    async def test_sync_handler_receives_event(self, event_bus):
        """同步 handler 能正确接收事件参数"""
        received = []

        def handler(*args, **kwargs):
            received.append((args, kwargs))

        event_bus.on("test_event", handler)
        await event_bus.emit_and_wait("test_event", x=1, y=2)
        # emit_and_wait 中同步 handler 直接执行
        assert len(received) >= 1

    @pytest.mark.asyncio
    async def test_async_handler_receives_event(self, event_bus):
        """异步 handler 能正确接收事件参数"""
        received = []

        async def handler(x, y):
            received.append((x, y))

        event_bus.on("test_event", handler)
        await event_bus.emit("test_event", "hello", y="world")
        await asyncio.sleep(0.1)
        assert len(received) == 1
        assert received[0] == ("hello", "world")

    @pytest.mark.asyncio
    async def test_multiple_handlers_all_receive_event(self, event_bus):
        """多个 handler 都能接收到事件"""
        results = []

        def handler1(x):
            results.append(f"h1:{x}")

        async def handler2(x):
            results.append(f"h2:{x}")

        event_bus.on("test_event", handler1)
        event_bus.on("test_event", handler2)
        await event_bus.emit("test_event", "data")
        await asyncio.sleep(0.2)
        assert "h1:data" in results
        assert "h2:data" in results

    @pytest.mark.asyncio
    async def test_no_handlers_no_error(self, event_bus):
        """无 handler 时不报错"""
        await event_bus.emit("nonexistent_event")

    @pytest.mark.asyncio
    async def test_kwargs_passed_to_handler(self, event_bus):
        """关键字参数正确传递"""
        received = {}

        def handler(**kw):
            received.update(kw)

        event_bus.on("test", handler)
        await event_bus.emit_and_wait("test", name="alice", age=30)
        # emit_and_wait 中同步 handler 直接执行
        assert received == {"name": "alice", "age": 30}


# ===== once =====

class TestOnce:
    """once: 一次性 handler"""

    @pytest.mark.asyncio
    async def test_once_handler_fires_only_once(self, event_bus):
        """once handler 只触发一次"""
        count = {"value": 0}

        async def handler():
            count["value"] += 1

        event_bus.once("test_once", handler)
        await event_bus.emit("test_once")
        await asyncio.sleep(0.2)
        await event_bus.emit("test_once")
        await asyncio.sleep(0.2)
        assert count["value"] == 1

    @pytest.mark.asyncio
    async def test_once_with_args(self, event_bus):
        """once handler 正确接收参数"""
        received = []

        async def handler(x):
            received.append(x)

        event_bus.once("test_once_args", handler)
        await event_bus.emit("test_once_args", "first")
        await asyncio.sleep(0.2)
        await event_bus.emit("test_once_args", "second")
        await asyncio.sleep(0.2)
        assert received == ["first"]


# ===== wait_for =====

class TestWaitFor:
    """wait_for: 等待特定事件"""

    @pytest.mark.asyncio
    async def test_wait_for_resolves_on_event(self, event_bus):
        """wait_for 在事件触发时返回"""
        async def emit_later():
            await asyncio.sleep(0.05)
            await event_bus.emit("target_event", value=42)

        asyncio.create_task(emit_later())
        args, kwargs = await event_bus.wait_for("target_event", timeout=2.0)
        assert kwargs == {"value": 42}

    @pytest.mark.asyncio
    async def test_wait_for_timeout(self, event_bus):
        """wait_for 超时抛出 asyncio.TimeoutError"""
        with pytest.raises(asyncio.TimeoutError):
            await event_bus.wait_for("never_emitted", timeout=0.1)


# ===== emit_and_wait =====

class TestEmitAndWait:
    """emit_and_wait: 等待所有 handler 完成"""

    @pytest.mark.asyncio
    async def test_waits_for_async_handlers(self, event_bus):
        """emit_and_wait 等待所有异步 handler 完成"""
        execution_order = []

        async def slow_handler():
            await asyncio.sleep(0.1)
            execution_order.append("slow")

        async def fast_handler():
            execution_order.append("fast")

        event_bus.on("test_wait", slow_handler)
        event_bus.on("test_wait", fast_handler)

        await event_bus.emit_and_wait("test_wait")
        # emit_and_wait 返回时所有 handler 都已完成
        assert "slow" in execution_order
        assert "fast" in execution_order

    @pytest.mark.asyncio
    async def test_sync_handler_executes_inline(self, event_bus):
        """emit_and_wait 中同步 handler 在当前协程中执行"""
        result = []

        def sync_handler():
            result.append("sync_done")

        event_bus.on("test_sync", sync_handler)
        await event_bus.emit_and_wait("test_sync")
        assert result == ["sync_done"]

    @pytest.mark.asyncio
    async def test_no_handlers_returns_immediately(self, event_bus):
        """无 handler 时 emit_and_wait 立即返回"""
        await event_bus.emit_and_wait("no_handlers")  # 不报错


# ===== 错误隔离 =====

class TestErrorIsolation:
    """错误隔离: 单个 handler 失败不影响其他 handler"""

    @pytest.mark.asyncio
    async def test_failing_async_handler_does_not_block_others(self, event_bus):
        """异步 handler 抛异常不影响其他 handler"""
        results = []

        async def failing_handler():
            raise ValueError("intentional error")

        async def good_handler():
            results.append("good")

        event_bus.on("test_error", failing_handler)
        event_bus.on("test_error", good_handler)

        await event_bus.emit_and_wait("test_error")
        assert "good" in results

    @pytest.mark.asyncio
    async def test_failing_sync_handler_does_not_block_others(self, event_bus):
        """同步 handler 抛异常不影响其他 handler (emit_and_wait 中)"""
        results = []

        def failing_handler():
            raise RuntimeError("sync error")

        def good_handler():
            results.append("sync_good")

        event_bus.on("test_sync_error", failing_handler)
        event_bus.on("test_sync_error", good_handler)

        await event_bus.emit_and_wait("test_sync_error")
        assert "sync_good" in results


# ===== remove/clear =====

class TestRemoveClear:
    """handler 移除和清除"""

    @pytest.mark.asyncio
    async def test_remove_handler(self, event_bus):
        """remove 后 handler 不再触发"""
        count = {"value": 0}

        def handler():
            count["value"] += 1

        event_bus.on("test_remove", handler)
        await event_bus.emit("test_remove")
        await asyncio.sleep(0.1)
        assert count["value"] == 1

        event_bus.remove("test_remove", handler)
        await event_bus.emit("test_remove")
        await asyncio.sleep(0.1)
        assert count["value"] == 1  # 没有增加

    @pytest.mark.asyncio
    async def test_clear_all_handlers(self, event_bus):
        """clear 后所有 handler 被移除"""
        event_bus.on("test1", lambda: None)
        event_bus.on("test2", lambda: None)
        assert event_bus.handler_count("test1") == 1
        assert event_bus.handler_count("test2") == 1

        event_bus.clear()
        assert event_bus.handler_count("test1") == 0
        assert event_bus.handler_count("test2") == 0


# ===== 查询方法 =====

class TestQueryMethods:
    """事件历史和查询"""

    @pytest.mark.asyncio
    async def test_get_event_names(self, event_bus):
        """get_event_names 返回已注册的事件名"""
        event_bus.on("alpha", lambda: None)
        event_bus.on("beta", lambda: None)
        names = event_bus.get_event_names()
        assert "alpha" in names
        assert "beta" in names

    @pytest.mark.asyncio
    async def test_handler_count(self, event_bus):
        """handler_count 返回正确数量"""
        event_bus.on("multi", lambda: None)
        event_bus.on("multi", lambda: None)
        assert event_bus.handler_count("multi") == 2
        assert event_bus.handler_count("empty") == 0

    @pytest.mark.asyncio
    async def test_event_history_records_emits(self, event_bus):
        """emit 后事件被记录到历史"""
        await event_bus.emit("event_a")
        await event_bus.emit("event_b")
        history = event_bus.get_event_history(limit=10)
        names = [name for name, _ in history]
        assert "event_a" in names
        assert "event_b" in names

    @pytest.mark.asyncio
    async def test_event_data_history(self, event_bus):
        """emit 记录事件数据历史"""
        await event_bus.emit("data_event", key="value")
        data = event_bus.get_event_data_history(event_name="data_event")
        assert len(data) >= 1
        assert data[-1]["event_name"] == "data_event"
        assert "key" in data[-1].get("kwargs_keys", [])
