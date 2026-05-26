# teragent/session/__init__.py
"""Session 模块 -- 会话持久化管理

支持中断后恢复对话，保存完整对话历史和会话状态。

功能:
  - SessionPersistence: 会话持久化管理器
  - SessionData: 完整会话数据（含对话历史）
  - SessionInfo: 会话摘要信息
  - create: 创建新会话
  - save / save_message: 保存会话或单条消息
  - load / restore: 加载/恢复会话
  - list_sessions: 列出所有会话
  - cleanup: 清理过期会话
"""

from teragent.session.persistence import (
    SessionPersistence,
    SessionData,
    SessionInfo,
    _message_to_dict,
    _message_from_dict,
)

__all__ = [
    "SessionPersistence",
    "SessionData",
    "SessionInfo",
    "_message_to_dict",
    "_message_from_dict",
]
