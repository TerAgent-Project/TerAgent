# teragent/core/types.py
"""公共数据类型 — 工具安全级别等跨模块共享类型定义

安全类型统一在 teragent/core/types.py 或 teragent/security/ 下，
tools 模块只负责工具实现，不再承载安全类型定义。

Message, MessageRole, MessageType 统一在 teragent/core/types.py 定义，
context 模块和其他模块从此处导入。

所有需要 ToolSafety 或 Message 的模块应从此处导入:
    from teragent.core.types import ToolSafety
    from teragent.core.types import Message, MessageRole, MessageType
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "Message",
    "MessageRole",
    "MessageType",
    "ToolSafety",
    "messages_from_dicts",
    "messages_to_api_format",
]


class ToolSafety(Enum):
    """工具安全级别

    参考 Claude-Code 的 isReadOnly / isDestructive / isConcurrencySafe 设计。

    层级说明:
      - READ_ONLY: 只读，无副作用，可安全并行（如 read_file）
      - SAFE_WRITE: 安全写入，可回退，不应并行（如 generate_design）
      - DESTRUCTIVE: 破坏性操作，不可逆，需串行+谨慎（如 execute_subtask）
      - HIGH_RISK: 高风险，需用户确认（如 create_project 创建整个项目）
    """
    READ_ONLY = "read_only"
    SAFE_WRITE = "safe_write"
    DESTRUCTIVE = "destructive"
    HIGH_RISK = "high_risk"


# ======================================================================
# Message 类型系统
# ======================================================================

class MessageRole(Enum):
    """消息角色 -- 对应 OpenAI Chat API 的 role 字段"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MessageType(Enum):
    """消息类型 -- 区分消息的具体语义

    与 MessageRole 不同，MessageType 描述消息的业务语义：
      - USER_INPUT vs CONTEXT_SUMMARY 都是 user 角色，但语义不同
      - SYSTEM_REMINDER vs SYSTEM_WARNING 都是 system 角色，但严重程度不同
      - ASSISTANT_TEXT vs ASSISTANT_TOOL_CALL 都是 assistant 角色，但用途不同
    """

    # === 基础对话 ===
    USER_INPUT = "user_input"                     # 用户输入
    ASSISTANT_TEXT = "assistant_text"             # 模型文本回复
    ASSISTANT_TOOL_CALL = "assistant_tool_call"   # 模型工具调用请求
    TOOL_RESULT = "tool_result"                   # 工具执行结果

    # === 系统消息 ===
    SYSTEM_PROMPT = "system_prompt"               # 系统提示（AGENT_SYSTEM_PROMPT 等）
    SYSTEM_REMINDER = "system_reminder"           # 系统提醒（周期对齐、状态提醒）
    SYSTEM_WARNING = "system_warning"             # 系统警告（轮询检测、重复熔断）
    SYSTEM_ERROR = "system_error"                 # 系统错误（不可恢复的错误）

    # === 进度与状态 ===
    TOOL_PROGRESS = "tool_progress"               # 工具执行进度
    TOOL_SUMMARY = "tool_summary"                 # 工具使用摘要

    # === 审查与修复 ===
    REVIEW_FEEDBACK = "review_feedback"           # 审查反馈（Reviewer 结果）
    REPAIR_INSTRUCTION = "repair_instruction"     # 修复指令（自动修复引导）

    # === 编排 ===
    HANDOFF = "handoff"                           # Agent 转交信号（Swarm 编排中 Agent 切换控制权）

    # === 元消息 ===
    CONTEXT_SUMMARY = "context_summary"           # 上下文压缩摘要（AutoCompactor 产出）
    TOMBSTONE = "tombstone"                       # 已删除消息的占位符


@dataclass
class Message:
    """统一消息类型

    替代原始 dict 消息格式，提供类型安全和语义标注。

    属性：
        role: 消息角色（对应 OpenAI API 的 role）
        content: 消息内容文本
        message_type: 消息语义类型（区分同一 role 下不同语义）
        tool_calls: 工具调用列表（仅 ASSISTANT_TOOL_CALL 类型使用）
        tool_call_id: 工具调用 ID（仅 TOOL_RESULT 类型使用）
        tool_name: 工具名称（仅 TOOL_RESULT 类型使用，方便调试）
        metadata: 扩展元数据（如压缩前长度、持久化 ID 等）
        timestamp: 消息创建时间戳

    使用方式：
        # 工厂方法创建（推荐）
        msg = Message.user_input("帮我写一个贪吃蛇游戏")
        msg = Message.assistant_tool_call("", tool_calls=[...])
        msg = Message.tool_result("call_abc", "read_file", "文件内容...")

        # 转换为 API 格式
        api_msg = msg.to_api_format()  # -> {"role": "user", "content": "..."}

        # 从旧格式转换
        msg = Message.from_dict({"role": "user", "content": "..."})
    """

    role: MessageRole
    content: str
    message_type: MessageType = MessageType.USER_INPUT

    # 工具调用相关
    tool_calls: list[dict] = field(default_factory=list)
    tool_call_id: str = ""
    tool_name: str = ""

    # 元数据
    metadata: dict = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        """自动填充时间戳"""
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    # ===== 兼容方法 =====

    def to_api_format(self) -> dict:
        """转换为 OpenAI Chat API 格式（dict）

        保证与现有 model.chat(messages=[...]) 调用完全兼容。
        仅包含 API 需要的字段，忽略 metadata、message_type 等内部字段。
        """
        msg: dict[str, Any] = {"role": self.role.value, "content": self.content}

        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id

        return msg

    def get(self, key: str, default: Any = None) -> Any:
        """dict 兼容访问接口

        支持旧代码中 msg.get("role") 风格的访问，
        迁移期间使用，新代码应直接访问属性。
        """
        _DICT_COMPAT = {
            "role": self.role.value,
            "content": self.content,
            "message_type": self.message_type.value if self.message_type else None,
            "tool_calls": self.tool_calls if self.tool_calls else None,
            "tool_call_id": self.tool_call_id if self.tool_call_id else None,
            "tool_name": self.tool_name if self.tool_name else None,
            "metadata": self.metadata if self.metadata else None,
            "timestamp": self.timestamp if self.timestamp else None,
        }
        return _DICT_COMPAT.get(key, default)

    # ===== 工厂方法 =====

    @classmethod
    def user_input(cls, content: str) -> Message:
        """创建用户输入消息"""
        return cls(
            role=MessageRole.USER,
            content=content,
            message_type=MessageType.USER_INPUT,
        )

    @classmethod
    def assistant_text(cls, content: str) -> Message:
        """创建模型文本回复消息"""
        return cls(
            role=MessageRole.ASSISTANT,
            content=content,
            message_type=MessageType.ASSISTANT_TEXT,
        )

    @classmethod
    def assistant_tool_call(cls, content: str, tool_calls: list[dict]) -> Message:
        """创建模型工具调用请求消息

        Args:
            content: 模型的文本输出（通常为空或简短描述）
            tool_calls: OpenAI 格式的工具调用列表
        """
        return cls(
            role=MessageRole.ASSISTANT,
            content=content,
            message_type=MessageType.ASSISTANT_TOOL_CALL,
            tool_calls=tool_calls,
        )

    @classmethod
    def tool_result(cls, tool_call_id: str, tool_name: str, content: str) -> Message:
        """创建工具执行结果消息

        Args:
            tool_call_id: 对应的 assistant tool_call ID
            tool_name: 工具名称（方便调试和追踪）
            content: 工具执行结果文本
        """
        return cls(
            role=MessageRole.TOOL,
            content=content,
            message_type=MessageType.TOOL_RESULT,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )

    @classmethod
    def system_prompt(cls, content: str) -> Message:
        """创建系统提示消息（AGENT_SYSTEM_PROMPT 等）"""
        return cls(
            role=MessageRole.SYSTEM,
            content=content,
            message_type=MessageType.SYSTEM_PROMPT,
        )

    @classmethod
    def system_reminder(cls, content: str) -> Message:
        """创建系统提醒消息（周期对齐、状态提醒等）"""
        return cls(
            role=MessageRole.SYSTEM,
            content=content,
            message_type=MessageType.SYSTEM_REMINDER,
        )

    @classmethod
    def system_warning(cls, content: str) -> Message:
        """创建系统警告消息（轮询检测、重复熔断等）"""
        return cls(
            role=MessageRole.SYSTEM,
            content=content,
            message_type=MessageType.SYSTEM_WARNING,
        )

    @classmethod
    def system_error(cls, content: str) -> Message:
        """创建系统错误消息（不可恢复的错误）"""
        return cls(
            role=MessageRole.SYSTEM,
            content=content,
            message_type=MessageType.SYSTEM_ERROR,
        )

    @classmethod
    def context_summary(cls, content: str) -> Message:
        """创建上下文压缩摘要消息（AutoCompactor 产出）

        注意：role 为 USER 而非 SYSTEM，因为上下文摘要需要模型能够"看到"并理解，
        放在 user 角色下确保模型将其视为对话的一部分。
        """
        return cls(
            role=MessageRole.USER,
            content=content,
            message_type=MessageType.CONTEXT_SUMMARY,
        )

    @classmethod
    def repair_instruction(cls, content: str) -> Message:
        """创建修复指令消息（自动修复引导）"""
        return cls(
            role=MessageRole.SYSTEM,
            content=content,
            message_type=MessageType.REPAIR_INSTRUCTION,
        )

    @classmethod
    def review_feedback(cls, content: str) -> Message:
        """创建审查反馈消息（Reviewer 结果）"""
        return cls(
            role=MessageRole.SYSTEM,
            content=content,
            message_type=MessageType.REVIEW_FEEDBACK,
        )

    @classmethod
    def handoff(cls, content: str, target_agent: str = "") -> Message:
        """创建 Agent 转交消息（Swarm 编排中使用）

        当 Agent 通过 Handoff 机制转交控制权时产生此消息，
        记录转交目标和转交上下文。

        Args:
            content: 转交说明文本
            target_agent: 目标 Agent 名称
        """
        return cls(
            role=MessageRole.SYSTEM,
            content=content,
            message_type=MessageType.HANDOFF,
            metadata={"target_agent": target_agent} if target_agent else {},
        )

    @classmethod
    def tombstone(cls, original_role: MessageRole = MessageRole.SYSTEM) -> Message:
        """创建墓碑消息（标记已删除的消息位置）

        墓碑消息用于在消息序列中标记被删除消息的位置，
        防止索引偏移导致对话不连贯。
        """
        return cls(
            role=original_role,
            content="[TOMBSTONE] This message has been removed.",
            message_type=MessageType.TOMBSTONE,
        )

    # ===== 从旧格式转换 =====

    @classmethod
    def from_dict(cls, data: dict) -> Message:
        """从 dict（OpenAI 格式）创建 Message

        用于将现有的 dict 消息转换为 Message 对象。
        自动推断 MessageType 以保留语义信息。

        Args:
            data: OpenAI 格式的消息字典

        Returns:
            对应的 Message 对象
        """
        role_str = data.get("role", "user")
        try:
            role = MessageRole(role_str)
        except ValueError:
            role = MessageRole.USER

        content = data.get("content") or ""

        # 推断 MessageType
        message_type = cls._infer_message_type(role, data)

        return cls(
            role=role,
            content=content,
            message_type=message_type,
            tool_calls=data.get("tool_calls", []),
            tool_call_id=data.get("tool_call_id", ""),
            tool_name=data.get("tool_name", ""),
        )

    @classmethod
    def _infer_message_type(cls, role: MessageRole, data: dict) -> MessageType:
        """根据 role 和消息内容推断 MessageType

        推断规则：
          - assistant + tool_calls -> ASSISTANT_TOOL_CALL
          - assistant 无 tool_calls -> ASSISTANT_TEXT
          - user + [上下文摘要] -> CONTEXT_SUMMARY
          - user 其他 -> USER_INPUT
          - tool -> TOOL_RESULT
          - system + [AUTO-REPAIR] / [LOCK] -> REPAIR_INSTRUCTION
          - system + [WARN] / [WARNING]（大小写不敏感） -> SYSTEM_WARNING
          - system + [RECOVERY] -> SYSTEM_WARNING
          - system + [CHART] -> SYSTEM_REMINDER
          - system 其他 -> SYSTEM_PROMPT
        """
        content = data.get("content") or ""

        if role == MessageRole.USER:
            if "[上下文摘要" in content or "[CONTEXT SUMMARY" in content.upper():
                return MessageType.CONTEXT_SUMMARY
            return MessageType.USER_INPUT

        elif role == MessageRole.ASSISTANT:
            if data.get("tool_calls"):
                return MessageType.ASSISTANT_TOOL_CALL
            return MessageType.ASSISTANT_TEXT

        elif role == MessageRole.TOOL:
            return MessageType.TOOL_RESULT

        elif role == MessageRole.SYSTEM:
            # 系统消息的细分类（统一大小写不敏感匹配）
            content_upper = content.upper()
            if "[AUTO-REPAIR]" in content_upper:
                return MessageType.REPAIR_INSTRUCTION
            if "[LOCK]" in content_upper:
                return MessageType.REPAIR_INSTRUCTION
            if "[WARN]" in content_upper or "[WARNING]" in content_upper:
                return MessageType.SYSTEM_WARNING
            if "[RECOVERY]" in content_upper:
                return MessageType.SYSTEM_WARNING
            if "[CHART]" in content_upper:
                return MessageType.SYSTEM_REMINDER
            # 默认系统消息归为 SYSTEM_PROMPT
            return MessageType.SYSTEM_PROMPT

        return MessageType.USER_INPUT

    # ===== 便捷属性 =====

    @property
    def is_system(self) -> bool:
        """是否为系统消息"""
        return self.role == MessageRole.SYSTEM

    @property
    def is_user(self) -> bool:
        """是否为用户消息"""
        return self.role == MessageRole.USER

    @property
    def is_assistant(self) -> bool:
        """是否为助手消息"""
        return self.role == MessageRole.ASSISTANT

    @property
    def is_tool(self) -> bool:
        """是否为工具结果消息"""
        return self.role == MessageRole.TOOL

    @property
    def has_tool_calls(self) -> bool:
        """是否包含工具调用"""
        return bool(self.tool_calls)

    @property
    def is_warning(self) -> bool:
        """是否为警告消息"""
        return self.message_type in (
            MessageType.SYSTEM_WARNING,
            MessageType.SYSTEM_ERROR,
        )

    @property
    def is_context_summary(self) -> bool:
        """是否为上下文压缩摘要"""
        return self.message_type == MessageType.CONTEXT_SUMMARY

    @property
    def is_repair(self) -> bool:
        """是否为修复相关消息"""
        return self.message_type in (
            MessageType.REPAIR_INSTRUCTION,
            MessageType.REVIEW_FEEDBACK,
        )

    @property
    def is_handoff(self) -> bool:
        """是否为 Agent 转交消息"""
        return self.message_type == MessageType.HANDOFF

    def __repr__(self) -> str:
        """简洁的表示形式，方便调试"""
        content_preview = self.content[:60].replace("\n", "\\n")
        if len(self.content) > 60:
            content_preview += "..."
        return (
            f"Message(role={self.role.value}, "
            f"type={self.message_type.value}, "
            f"content={content_preview!r})"
        )


def messages_to_api_format(messages: list[Message]) -> list[dict]:
    """将 Message 列表批量转换为 OpenAI API 格式

    Args:
        messages: Message 对象列表

    Returns:
        dict 列表，可直接传递给 model.chat(messages=...)
    """
    return [msg.to_api_format() for msg in messages]


def messages_from_dicts(dicts: list[dict]) -> list[Message]:
    """将 dict 列表批量转换为 Message 对象

    Args:
        dicts: OpenAI 格式的消息字典列表

    Returns:
        Message 对象列表
    """
    return [Message.from_dict(d) for d in dicts]
