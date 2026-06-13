# tests/test_confirmation.py
"""用户确认门控单元测试

测试 teragent.intent.confirmation 模块:
  - M0 模式自动确认
  - M1 模式交互确认
  - 超时自动确认
  - 技术约束检测
  - cancel_all 取消
"""
import asyncio

import pytest

from teragent.event_bus import EventBus
from teragent.intent.confirmation import ConfirmationGate


class TestM0AutoConfirm:
    """M0 自动确认模式测试"""

    @pytest.mark.asyncio
    async def test_auto_confirm_returns_true(self):
        """M0 模式自动确认返回 True"""
        bus = EventBus()
        gate = ConfirmationGate(bus=bus, auto_confirm=True)
        result = await gate.confirm_create_project("创建一个 Python 爬虫")
        assert result is True

    @pytest.mark.asyncio
    async def test_auto_confirm_emits_event(self):
        """M0 自动确认发射 confirmation_request 事件"""
        bus = EventBus()
        gate = ConfirmationGate(bus=bus, auto_confirm=True)

        received = {}

        async def handler(**kwargs):
            received.update(kwargs)

        bus.on("confirmation_request", handler)
        await gate.confirm_create_project("创建一个 Flask 应用")
        # 等待异步 handler 完成
        await asyncio.sleep(0.05)
        assert received.get("auto_confirmed") is True

    @pytest.mark.asyncio
    async def test_auto_confirm_timeout_is_zero(self):
        """M0 模式超时为 0"""
        bus = EventBus()
        gate = ConfirmationGate(bus=bus, auto_confirm=True, timeout=0.0)
        assert gate.timeout == 0.0


class TestM1InteractiveConfirm:
    """M1 交互确认模式测试"""

    @pytest.mark.asyncio
    async def test_m1_timeout_default(self):
        """M1 默认超时 30 秒"""
        bus = EventBus()
        gate = ConfirmationGate(bus=bus, auto_confirm=False)
        assert gate.timeout == 30.0

    @pytest.mark.asyncio
    async def test_m1_custom_timeout(self):
        """M1 自定义超时"""
        bus = EventBus()
        gate = ConfirmationGate(bus=bus, auto_confirm=False, timeout=5.0)
        assert gate.timeout == 5.0


class TestTimeoutAutoConfirm:
    """超时自动确认测试"""

    @pytest.mark.asyncio
    async def test_timeout_auto_confirms(self):
        """超时后自动确认返回 True"""
        bus = EventBus()
        gate = ConfirmationGate(bus=bus, auto_confirm=False, timeout=0.01)

        # 不发送确认响应 → 超时
        result = await gate.confirm_create_project("测试超时确认")
        assert result is True  # 超时自动确认

    @pytest.mark.asyncio
    async def test_user_confirms_before_timeout(self):
        """用户在超时前确认返回 True"""
        bus = EventBus()
        gate = ConfirmationGate(bus=bus, auto_confirm=False, timeout=5.0)

        # 后台发送确认响应
        async def respond_later():
            await asyncio.sleep(0.05)
            await gate._on_confirmation_response(
                request_id=list(gate._pending_confirmations.keys())[0] if gate._pending_confirmations else "",
                confirmed=True,
            )

        asyncio.create_task(respond_later())
        result = await gate.confirm_create_project("测试用户确认")
        assert result is True


class TestSummaryGeneration:
    """摘要生成测试"""

    def test_basic_summary(self):
        """基本摘要包含需求描述"""
        bus = EventBus()
        gate = ConfirmationGate(bus=bus)
        summary = gate._generate_summary("创建一个 Python 爬虫")
        assert "创建一个 Python 爬虫" in summary

    def test_tech_constraint_detection(self):
        """检测技术约束"""
        bus = EventBus()
        gate = ConfirmationGate(bus=bus)
        constraint = gate._detect_tech_constraints("创建一个纯python的爬虫")
        assert "纯Python" in constraint

    def test_no_constraint(self):
        """无技术约束返回空字符串"""
        bus = EventBus()
        gate = ConfirmationGate(bus=bus)
        constraint = gate._detect_tech_constraints("创建一个爬虫")
        assert constraint == ""


class TestCancelAll:
    """取消所有确认请求测试"""

    @pytest.mark.asyncio
    async def test_cancel_all_sets_false(self):
        """cancel_all 将所有等待中的确认设为 False"""
        bus = EventBus()
        gate = ConfirmationGate(bus=bus, auto_confirm=False, timeout=10.0)

        # 创建一个等待中的确认
        future = asyncio.get_running_loop().create_future()
        gate._pending_confirmations["test_req"] = future

        gate.cancel_all()
        assert future.done()
        assert future.result() is False

    @pytest.mark.asyncio
    async def test_cancel_all_clears_pending(self):
        """cancel_all 清空等待队列"""
        bus = EventBus()
        gate = ConfirmationGate(bus=bus, auto_confirm=False, timeout=10.0)

        future = asyncio.get_running_loop().create_future()
        gate._pending_confirmations["test_req"] = future

        gate.cancel_all()
        assert len(gate._pending_confirmations) == 0
