# teragent/coordination/message_bus.py
"""AgentMessageBus -- Agent 间消息传递总线 (Phase 9.2.1)

核心职责:
  - 为每个注册 Agent 创建独立邮箱 (asyncio.Queue)
  - 支持点对点消息发送和广播消息发送
  - 支持阻塞接收 (带超时) 和非阻塞接收
  - 通过 EventBus 发射 agent_message_sent 事件

设计原则:
  - 使用 asyncio.Queue 实现邮箱, 天然适合异步上下文
  - 所有状态修改在事件循环内完成, 无需额外锁
  - 消息设计为不可变（约定，非强制; AgentMessage 为 mutable dataclass）
  - 广播消息不发送给发送者自身
"""

import asyncio
import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "AgentMessage",
    "AgentMessageBus",
    "BROADCAST",
    "MAILBOX_MAX_SIZE",
]

from teragent.event_bus import EventBus

logger = logging.getLogger(__name__)

# 邮箱最大容量
MAILBOX_MAX_SIZE = 100

# 广播目标标识
BROADCAST = "broadcast"


@dataclass
class AgentMessage:
    """Agent 间消息数据类

    属性:
        from_agent: 发送者 Agent ID
        to_agent: 接收者 Agent ID ("broadcast" 表示广播)
        message_type: 消息类型 (request/result/notification)
        content: 消息内容
        metadata: 扩展元数据
        timestamp: 消息创建时间戳
        message_id: 自动生成的唯一消息 ID (格式: "msg_{counter}")
    """

    from_agent: str
    to_agent: str
    message_type: str
    content: str
    metadata: dict = field(default_factory=dict)
    timestamp: float = 0.0
    message_id: str = ""

    def __post_init__(self) -> None:
        """自动填充时间戳和消息 ID"""
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if not self.message_id:
            self.message_id = AgentMessage._next_id()

    _counter = itertools.count(1)

    @classmethod
    def _next_id(cls) -> str:
        """生成下一个消息 ID"""
        return f"msg_{next(cls._counter)}"

    def to_dict(self) -> dict:
        """转换为可序列化字典"""
        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "message_type": self.message_type,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
        }


class AgentMessageBus:
    """Agent 间消息传递总线

    为每个注册的 Agent 创建独立的 asyncio.Queue 邮箱,
    支持点对点和广播消息发送, 以及阻塞/非阻塞接收。

    使用方式::

        bus = AgentMessageBus(event_bus)
        bus.register_agent("main")
        bus.register_agent("sub_1")

        msg = AgentMessage(from_agent="main", to_agent="sub_1",
                          message_type="request", content="分析这段代码")
        await bus.send(msg)

        received = await bus.receive("sub_1", timeout=30.0)
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._mailboxes: dict[str, asyncio.Queue[AgentMessage]] = {}
        self._agent_metadata: dict[str, dict] = {}

    def register_agent(self, agent_id: str, metadata: dict | None = None) -> None:
        """注册 Agent, 创建邮箱

        如果 agent_id 已注册, 发出警告并覆盖。
        旧邮箱中的未读消息将被丢弃。

        Args:
            agent_id: Agent 唯一标识
            metadata: Agent 元数据 (可选)
        """
        if agent_id in self._mailboxes:
            old_queue = self._mailboxes[agent_id]
            discarded = old_queue.qsize()
            if discarded > 0:
                logger.warning(
                    f"Agent '{agent_id}' 已注册, 覆盖旧邮箱, "
                    f"丢弃 {discarded} 条未读消息"
                )
            else:
                logger.warning(f"Agent '{agent_id}' 已注册, 覆盖旧邮箱")
        self._mailboxes[agent_id] = asyncio.Queue(maxsize=MAILBOX_MAX_SIZE)
        self._agent_metadata[agent_id] = metadata or {}
        logger.info(f"Agent '{agent_id}' 已注册到消息总线")

    def unregister_agent(self, agent_id: str) -> None:
        """注销 Agent, 移除邮箱

        Args:
            agent_id: Agent 唯一标识
        """
        self._mailboxes.pop(agent_id, None)
        self._agent_metadata.pop(agent_id, None)
        logger.info(f"Agent '{agent_id}' 已从消息总线注销")

    def is_registered(self, agent_id: str) -> bool:
        """检查 Agent 是否已注册

        Args:
            agent_id: Agent 唯一标识

        Returns:
            True 表示已注册
        """
        return agent_id in self._mailboxes

    async def send(self, message: AgentMessage) -> None:
        """发送消息到目标邮箱

        点对点: 将消息放入 to_agent 的邮箱
        广播 (to_agent == "broadcast"): 将消息放入所有已注册 Agent 的邮箱 (排除发送者)

        如果目标 Agent 未注册, 记录警告但不抛出异常。

        Args:
            message: 要发送的消息
        """
        if message.to_agent == BROADCAST:
            # 广播: 发送给所有 Agent (排除发送者)
            sent_count = 0
            for agent_id, mailbox in self._mailboxes.items():
                if agent_id == message.from_agent:
                    continue
                try:
                    mailbox.put_nowait(message)
                    sent_count += 1
                except asyncio.QueueFull:
                    logger.warning(
                        f"Agent '{agent_id}' 邮箱已满, 广播消息丢弃: "
                        f"msg_id={message.message_id}"
                    )
            logger.info(
                f"广播消息已发送: from={message.from_agent}, "
                f"msg_id={message.message_id}, recipients={sent_count}"
            )
        else:
            # 点对点
            mailbox = self._mailboxes.get(message.to_agent)
            if mailbox is None:
                logger.warning(
                    f"目标 Agent '{message.to_agent}' 未注册, 消息丢弃: "
                    f"msg_id={message.message_id}"
                )
            else:
                try:
                    mailbox.put_nowait(message)
                except asyncio.QueueFull:
                    logger.warning(
                        f"Agent '{message.to_agent}' 邮箱已满, 消息丢弃: "
                        f"msg_id={message.message_id}"
                    )

        # 通过 EventBus 发射消息事件
        await self._event_bus.emit(
            "agent_message_sent",
            message.to_dict(),
        )

    async def receive(
        self, agent_id: str, timeout: float = 30.0
    ) -> Optional[AgentMessage]:
        """阻塞接收消息 (带超时)

        等待指定 Agent 的邮箱中有消息到达, 超时返回 None。

        Args:
            agent_id: 接收消息的 Agent ID
            timeout: 等待超时 (秒)

        Returns:
            接收到的消息, 或 None (超时 / Agent 未注册)
        """
        mailbox = self._mailboxes.get(agent_id)
        if mailbox is None:
            logger.warning(f"Agent '{agent_id}' 未注册, 无法接收消息")
            return None
        try:
            message = await asyncio.wait_for(mailbox.get(), timeout=timeout)
            return message
        except asyncio.TimeoutError:
            return None

    def try_receive(self, agent_id: str) -> Optional[AgentMessage]:
        """非阻塞接收消息

        尝试从指定 Agent 的邮箱中立即获取一条消息,
        如果邮箱为空则返回 None。

        Args:
            agent_id: 接收消息的 Agent ID

        Returns:
            接收到的消息, 或 None (邮箱为空 / Agent 未注册)
        """
        mailbox = self._mailboxes.get(agent_id)
        if mailbox is None:
            return None
        try:
            return mailbox.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def get_registered_agents(self) -> list[str]:
        """返回所有已注册的 Agent ID 列表

        Returns:
            Agent ID 列表
        """
        return list(self._mailboxes.keys())

    def get_agent_metadata(self, agent_id: str) -> dict | None:
        """获取指定 Agent 的元数据

        Args:
            agent_id: Agent 唯一标识

        Returns:
            元数据字典, 或 None (Agent 未注册)
        """
        return self._agent_metadata.get(agent_id)

    def get_mailbox_size(self, agent_id: str) -> int:
        """获取指定 Agent 邮箱中的待处理消息数

        Args:
            agent_id: Agent 唯一标识

        Returns:
            待处理消息数 (Agent 未注册时返回 0)
        """
        mailbox = self._mailboxes.get(agent_id)
        if mailbox is None:
            return 0
        return mailbox.qsize()

    def get_status_report(self) -> dict:
        """返回消息总线状态摘要 (供调试和 TUI 使用)

        Returns:
            {
                "registered_agents": int,
                "agents": [{"id": str, "mailbox_size": int, "metadata": dict}],
            }
        """
        agents = []
        for agent_id in self._mailboxes:
            agents.append({
                "id": agent_id,
                "mailbox_size": self.get_mailbox_size(agent_id),
                "metadata": self._agent_metadata.get(agent_id, {}),
            })
        return {
            "registered_agents": len(self._mailboxes),
            "agents": agents,
        }
