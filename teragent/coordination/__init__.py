"""teragent.coordination — 多 Agent 协作包

提供 Agent 间消息传递和子 Agent 管理能力。
"""

from teragent.coordination.message_bus import (
    BROADCAST,
    MAILBOX_MAX_SIZE,
    AgentMessage,
    AgentMessageBus,
)
from teragent.coordination.sub_agent_manager import (
    SUB_AGENT_SYSTEM_PROMPT_PREFIX,
    AgentMode,
    SubAgentInfo,
    SubAgentManager,
    SubAgentStatus,
)

__all__ = [
    # Message bus
    "AgentMessageBus",
    "AgentMessage",
    "MAILBOX_MAX_SIZE",
    "BROADCAST",
    # Sub-agent management
    "SubAgentManager",
    "AgentMode",
    "SubAgentStatus",
    "SubAgentInfo",
    "SUB_AGENT_SYSTEM_PROMPT_PREFIX",
]
