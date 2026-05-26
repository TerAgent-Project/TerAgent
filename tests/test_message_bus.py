# tests/test_message_bus.py
"""Agent 消息总线单元测试

测试 teragent.coordination.message_bus 模块:
  - 点对点消息发送
  - 广播消息发送
  - 队列满行为
  - Agent 注册/注销
  - 非阻塞接收
"""
import asyncio
import pytest

from teragent.coordination.message_bus import (
    AgentMessage,
    AgentMessageBus,
    BROADCAST,
    MAILBOX_MAX_SIZE,
)
from teragent.event_bus import EventBus


class TestAgentMessage:
    """AgentMessage 数据类测试"""

    def test_auto_fill_timestamp(self):
        """自动填充时间戳"""
        msg = AgentMessage(
            from_agent="a", to_agent="b",
            message_type="request", content="hello",
        )
        assert msg.timestamp > 0

    def test_auto_fill_message_id(self):
        """自动填充消息 ID"""
        msg = AgentMessage(
            from_agent="a", to_agent="b",
            message_type="request", content="hello",
        )
        assert msg.message_id.startswith("msg_")

    def test_to_dict(self):
        """to_dict 包含所有字段"""
        msg = AgentMessage(
            from_agent="a", to_agent="b",
            message_type="request", content="hello",
            metadata={"key": "val"},
        )
        d = msg.to_dict()
        assert d["from_agent"] == "a"
        assert d["to_agent"] == "b"
        assert d["message_type"] == "request"
        assert d["content"] == "hello"
        assert d["metadata"]["key"] == "val"

    def test_unique_message_ids(self):
        """每条消息 ID 唯一"""
        msg1 = AgentMessage(from_agent="a", to_agent="b", message_type="req", content="1")
        msg2 = AgentMessage(from_agent="a", to_agent="b", message_type="req", content="2")
        assert msg1.message_id != msg2.message_id


class TestAgentRegistration:
    """Agent 注册/注销测试"""

    def test_register_agent(self):
        """注册 Agent 创建邮箱"""
        bus = EventBus()
        mbus = AgentMessageBus(bus)
        mbus.register_agent("agent_1")
        assert mbus.is_registered("agent_1")

    def test_unregister_agent(self):
        """注销 Agent 移除邮箱"""
        bus = EventBus()
        mbus = AgentMessageBus(bus)
        mbus.register_agent("agent_1")
        mbus.unregister_agent("agent_1")
        assert not mbus.is_registered("agent_1")

    def test_get_registered_agents(self):
        """返回已注册 Agent 列表"""
        bus = EventBus()
        mbus = AgentMessageBus(bus)
        mbus.register_agent("a")
        mbus.register_agent("b")
        agents = mbus.get_registered_agents()
        assert "a" in agents
        assert "b" in agents

    def test_agent_metadata(self):
        """Agent 元数据存储"""
        bus = EventBus()
        mbus = AgentMessageBus(bus)
        mbus.register_agent("agent_1", metadata={"role": "worker"})
        assert mbus.get_agent_metadata("agent_1")["role"] == "worker"


class TestPointToPointMessaging:
    """点对点消息测试"""

    @pytest.mark.asyncio
    async def test_send_and_receive(self):
        """点对点发送和接收消息"""
        bus = EventBus()
        mbus = AgentMessageBus(bus)
        mbus.register_agent("sender")
        mbus.register_agent("receiver")

        msg = AgentMessage(
            from_agent="sender", to_agent="receiver",
            message_type="request", content="分析这段代码",
        )
        await mbus.send(msg)

        received = await mbus.receive("receiver", timeout=1.0)
        assert received is not None
        assert received.content == "分析这段代码"
        assert received.from_agent == "sender"

    @pytest.mark.asyncio
    async def test_send_to_unregistered_agent(self):
        """发送到未注册 Agent 不抛异常"""
        bus = EventBus()
        mbus = AgentMessageBus(bus)
        mbus.register_agent("sender")

        msg = AgentMessage(
            from_agent="sender", to_agent="ghost",
            message_type="request", content="hello",
        )
        # 不应抛异常
        await mbus.send(msg)


class TestBroadcastMessaging:
    """广播消息测试"""

    @pytest.mark.asyncio
    async def test_broadcast_excludes_sender(self):
        """广播消息不发送给发送者自身"""
        bus = EventBus()
        mbus = AgentMessageBus(bus)
        mbus.register_agent("main")
        mbus.register_agent("sub_1")
        mbus.register_agent("sub_2")

        msg = AgentMessage(
            from_agent="main", to_agent=BROADCAST,
            message_type="notification", content="广播消息",
        )
        await mbus.send(msg)

        # 发送者邮箱为空
        assert mbus.get_mailbox_size("main") == 0
        # 其他 Agent 各收到 1 条
        assert mbus.get_mailbox_size("sub_1") == 1
        assert mbus.get_mailbox_size("sub_2") == 1

    @pytest.mark.asyncio
    async def test_broadcast_content_same(self):
        """广播消息内容一致"""
        bus = EventBus()
        mbus = AgentMessageBus(bus)
        mbus.register_agent("main")
        mbus.register_agent("sub_1")

        msg = AgentMessage(
            from_agent="main", to_agent=BROADCAST,
            message_type="notification", content="重要通知",
        )
        await mbus.send(msg)

        received = await mbus.receive("sub_1", timeout=1.0)
        assert received.content == "重要通知"


class TestQueueFullBehavior:
    """队列满行为测试"""

    @pytest.mark.asyncio
    async def test_queue_full_drops_message(self):
        """邮箱满时丢弃消息"""
        bus = EventBus()
        mbus = AgentMessageBus(bus)
        mbus.register_agent("receiver")

        # 填满邮箱
        for i in range(MAILBOX_MAX_SIZE):
            msg = AgentMessage(
                from_agent="sender", to_agent="receiver",
                message_type="request", content=f"msg_{i}",
            )
            await mbus.send(msg)

        # 邮箱已满，下一条应被丢弃
        overflow_msg = AgentMessage(
            from_agent="sender", to_agent="receiver",
            message_type="request", content="overflow",
        )
        await mbus.send(overflow_msg)

        # 邮箱大小不超过最大容量
        assert mbus.get_mailbox_size("receiver") <= MAILBOX_MAX_SIZE


class TestNonBlockingReceive:
    """非阻塞接收测试"""

    def test_try_receive_empty(self):
        """空邮箱返回 None"""
        bus = EventBus()
        mbus = AgentMessageBus(bus)
        mbus.register_agent("agent_1")
        result = mbus.try_receive("agent_1")
        assert result is None

    def test_try_receive_unregistered(self):
        """未注册 Agent 返回 None"""
        bus = EventBus()
        mbus = AgentMessageBus(bus)
        result = mbus.try_receive("ghost")
        assert result is None


class TestStatusReport:
    """状态报告测试"""

    def test_status_report_structure(self):
        """状态报告结构正确"""
        bus = EventBus()
        mbus = AgentMessageBus(bus)
        mbus.register_agent("a")
        mbus.register_agent("b")

        report = mbus.get_status_report()
        assert report["registered_agents"] == 2
        assert len(report["agents"]) == 2
