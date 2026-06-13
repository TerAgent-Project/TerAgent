# teragent/session/persistence.py
"""SessionPersistence -- 会话持久化管理器

Phase 9.3 核心组件：支持中断后恢复对话。

设计原则:
  - 每个会话一个 JSON 文件，保存在 .teragent/sessions/ 目录
  - 文件名格式: session_<id>.json
  - 内容包含会话元数据 + 完整对话历史（Message 序列化）
  - 自动保存：每次对话后自动追加新消息（可配置）
  - 原子写入：使用 tempfile + os.replace 保证写入安全
  - 过期清理：自动清理超过 max_age_days 天的旧会话

存储格式 (session_<id>.json)::

    {
        "id": "sess_1709856000_a1b2c3",
        "created_at": 1709856000.0,
        "updated_at": 1709856060.0,
        "title": "创建贪吃蛇游戏",
        "intent": "CREATE_PROJECT",
        "message_count": 12,
        "step_count": 5,
        "metadata": {},
        "messages": [
            {"role": "user", "content": "...", "message_type": "user_input", ...},
            {"role": "assistant", "content": "...", "message_type": "assistant_text", ...},
            ...
        ]
    }

使用示例::

    from teragent.session import SessionPersistence

    persister = SessionPersistence(workspace_root=".")
    session_id = persister.create("创建贪吃蛇游戏", intent="CREATE_PROJECT")
    persister.save_message(session_id, Message.user_input("帮我写一个贪吃蛇"))
    persister.save_message(session_id, Message.assistant_text("好的，我来帮你..."))

    # 恢复会话
    data = persister.load(session_id)
    messages = persister.restore(session_id)

    # 列出所有会话
    sessions = persister.list_sessions()

    # /resume <id> 命令后恢复到 AgentLoop
    agent_loop._conversation = persister.restore(session_id)
"""
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass

from teragent.core.types import Message, MessageRole, MessageType

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """会话摘要信息 -- 用于 /sessions 命令展示

    不包含完整消息列表，仅用于列表展示。
    """

    id: str
    title: str
    intent: str
    created_at: float
    updated_at: float
    message_count: int
    step_count: int

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "id": self.id,
            "title": self.title,
            "intent": self.intent,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
            "step_count": self.step_count,
        }


@dataclass
class SessionData:
    """完整会话数据 -- 包含对话历史

    用于 /resume <id> 恢复会话。
    """

    id: str
    title: str
    intent: str
    created_at: float
    updated_at: float
    message_count: int
    step_count: int
    metadata: dict
    messages: list[Message]

    def to_dict(self) -> dict:
        """转换为可序列化字典"""
        return {
            "id": self.id,
            "title": self.title,
            "intent": self.intent,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
            "step_count": self.step_count,
            "metadata": self.metadata,
            "messages": [_message_to_dict(m) for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionData":
        """从字典反序列化"""
        # [M1 修复] 逐条处理消息，跳过无效条目
        messages_data = data.get("messages", [])
        messages: list[Message] = []
        for i, m in enumerate(messages_data):
            try:
                if isinstance(m, dict):
                    messages.append(_message_from_dict(m))
                else:
                    logger.warning(f"Skipping invalid message at index {i}: not a dict")
            except Exception as e:
                logger.warning(f"Skipping invalid message at index {i}: {e}")

        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            intent=data.get("intent", ""),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
            message_count=data.get("message_count", 0),
            step_count=data.get("step_count", 0),
            metadata=data.get("metadata", {}),
            messages=messages,
        )

    @property
    def info(self) -> SessionInfo:
        """提取摘要信息"""
        return SessionInfo(
            id=self.id,
            title=self.title,
            intent=self.intent,
            created_at=self.created_at,
            updated_at=self.updated_at,
            message_count=self.message_count,
            step_count=self.step_count,
        )


def _message_to_dict(msg: Message) -> dict:
    """将 Message 对象序列化为可 JSON 化的字典

    保留所有字段，包括 message_type 和 metadata。
    """
    return {
        "role": msg.role.value,
        "content": msg.content,
        "message_type": msg.message_type.value,
        "tool_calls": msg.tool_calls,
        "tool_call_id": msg.tool_call_id,
        "tool_name": msg.tool_name,
        "metadata": msg.metadata,
        "timestamp": msg.timestamp,
    }


def _message_from_dict(data: dict) -> Message:
    """从字典反序列化为 Message 对象"""
    try:
        role = MessageRole(data.get("role", "user"))
    except ValueError:
        role = MessageRole.USER

    try:
        message_type = MessageType(data.get("message_type", "user_input"))
    except ValueError:
        message_type = MessageType.USER_INPUT

    return Message(
        role=role,
        content=data.get("content", ""),
        message_type=message_type,
        tool_calls=data.get("tool_calls", []),
        tool_call_id=data.get("tool_call_id", ""),
        tool_name=data.get("tool_name", ""),
        metadata=data.get("metadata", {}),
        timestamp=data.get("timestamp", 0.0),
    )


class SessionPersistence:
    """会话持久化管理器

    支持:
      - 创建新会话 (create)
      - 保存单条消息 (save_message)
      - 保存完整会话 (save_session)
      - 加载会话 (load)
      - 恢复对话历史 (restore)
      - 列出所有会话 (list_sessions)
      - 删除会话 (delete)
      - 清理过期会话 (cleanup)

    线程安全：使用 threading.Lock 保护文件操作。
    原子写入：使用 tempfile + os.replace 保证写入安全。
    """

    # 会话文件前缀
    FILE_PREFIX = "session_"
    FILE_SUFFIX = ".json"

    def __init__(
        self,
        workspace_root: str = ".",
        session_dir: str = ".teragent/sessions",
        max_sessions: int = 50,
        auto_save: bool = True,
        max_age_days: int = 30,
    ) -> None:
        """初始化会话持久化管理器

        Args:
            workspace_root: 工作区根目录
            session_dir: 会话存储目录（相对于 workspace_root）
            max_sessions: 最大保存会话数（超出时清理最旧的）
            auto_save: 是否在每次 save_message 时自动写入磁盘
            max_age_days: 会话最大保留天数（超过则被 cleanup 清理）
        """
        self.workspace_root = workspace_root
        self.session_dir = os.path.join(workspace_root, session_dir)
        self.max_sessions = max_sessions
        self.auto_save = auto_save
        self.max_age_days = max_age_days

        # 线程安全锁
        self._lock = threading.Lock()

        # 当前活跃会话 ID（最近创建或恢复的会话）
        self._current_session_id: str = ""

        # 内存缓存: session_id -> SessionData
        self._cache: dict[str, SessionData] = {}

        # 确保目录存在
        os.makedirs(self.session_dir, exist_ok=True)

        logger.info(
            f"SessionPersistence initialized: "
            f"dir={self.session_dir}, "
            f"max_sessions={max_sessions}, "
            f"auto_save={auto_save}, "
            f"max_age_days={max_age_days}"
        )

    # ===== 核心操作 =====

    def create(
        self,
        title: str = "",
        intent: str = "",
        metadata: dict | None = None,
    ) -> str:
        """创建新会话

        [H2 修复] 创建时检查 max_sessions 限制，超出时自动清理最旧会话。

        Args:
            title: 会话标题（通常为用户首次输入的摘要）
            intent: 意图类型（如 "CREATE_PROJECT", "DEBUG", "CHAT"）
            metadata: 扩展元数据

        Returns:
            新会话的 ID
        """
        now = time.time()
        session_id = f"sess_{int(now)}_{os.urandom(4).hex()}"

        session = SessionData(
            id=session_id,
            title=title[:100] if title else "Untitled",
            intent=intent,
            created_at=now,
            updated_at=now,
            message_count=0,
            step_count=0,
            metadata=metadata or {},
            messages=[],
        )

        with self._lock:
            # [H2] 强制执行 max_sessions 限制
            self._enforce_max_sessions()

            self._cache[session_id] = session
            self._current_session_id = session_id
            if self.auto_save:
                self._write_to_disk(session_id)

        logger.info(f"Session created: {session_id} (title={title[:50]})")
        return session_id

    def save_message(self, session_id: str, message: Message) -> None:
        """保存单条消息到指定会话

        Args:
            session_id: 目标会话 ID
            message: 要保存的 Message 对象
        """
        with self._lock:
            session = self._cache.get(session_id)
            if session is None:
                # 尝试从磁盘加载
                session = self._read_from_disk(session_id)
                if session is None:
                    logger.warning(f"Session not found: {session_id}")
                    return
                self._cache[session_id] = session

            session.messages.append(message)
            session.message_count = len(session.messages)
            session.updated_at = time.time()

            # 更新标题（如果第一条用户消息且标题为空/Untitled）
            if (
                message.role == MessageRole.USER
                and session.title in ("", "Untitled")
                and message.content.strip()
            ):
                session.title = message.content[:100]

            if self.auto_save:
                self._write_to_disk(session_id)

    def save_session(self, session_id: str) -> None:
        """保存当前会话完整状态到磁盘

        即使 auto_save=False，也可以手动调用此方法保存。

        Args:
            session_id: 要保存的会话 ID
        """
        with self._lock:
            if session_id in self._cache:
                self._write_to_disk(session_id)
            else:
                logger.warning(f"Session not in cache: {session_id}")

    def load(self, session_id: str) -> SessionData | None:
        """加载指定会话的完整数据

        [H5 修复] 返回缓存引用（内部使用）。调用者不应直接修改返回值。
        如需独立副本，请使用 load_copy()。

        Args:
            session_id: 会话 ID

        Returns:
            SessionData 或 None（会话不存在时）
        """
        with self._lock:
            # 优先从缓存获取
            if session_id in self._cache:
                return self._cache[session_id]

            # 从磁盘加载
            session = self._read_from_disk(session_id)
            if session is not None:
                self._cache[session_id] = session
            return session

    def load_copy(self, session_id: str) -> SessionData | None:
        """加载指定会话的深拷贝（用于只读访问）

        返回的 SessionData 与内部缓存完全独立，修改不会影响持久化。

        Args:
            session_id: 会话 ID

        Returns:
            SessionData 深拷贝或 None
        """
        import copy
        with self._lock:
            if session_id in self._cache:
                return copy.deepcopy(self._cache[session_id])
            session = self._read_from_disk(session_id)
            if session is not None:
                self._cache[session_id] = session
                return copy.deepcopy(session)
            return None

    def restore(self, session_id: str) -> list[Message] | None:
        """恢复会话对话历史

        用于 /resume <id> 命令，将保存的对话历史恢复到 AgentLoop。

        Args:
            session_id: 会话 ID

        Returns:
            Message 列表或 None（会话不存在时）
        """
        session = self.load(session_id)
        if session is None:
            return None

        # [C3 修复] 在锁内复制消息列表，避免并发 save_message 导致竞态
        import copy
        with self._lock:
            self._current_session_id = session_id
            messages_copy = copy.deepcopy(session.messages)
            message_count = session.message_count

        logger.info(
            f"Session restored: {session_id} "
            f"({message_count} messages, "
            f"title={session.title[:50]})"
        )
        return messages_copy

    def list_sessions(self, limit: int = 20) -> list[SessionInfo]:
        """列出所有会话（按最近更新时间排序）

        Args:
            limit: 最多返回的会话数

        Returns:
            SessionInfo 列表（不含完整对话历史）
        """
        sessions: list[SessionInfo] = []

        with self._lock:
            # 扫描磁盘上的所有会话文件
            for filename in os.listdir(self.session_dir):
                if not filename.startswith(self.FILE_PREFIX):
                    continue
                if not filename.endswith(self.FILE_SUFFIX):
                    continue

                filepath = os.path.join(self.session_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    sessions.append(SessionInfo(
                        id=data.get("id", ""),
                        title=data.get("title", ""),
                        intent=data.get("intent", ""),
                        created_at=data.get("created_at", 0.0),
                        updated_at=data.get("updated_at", 0.0),
                        message_count=data.get("message_count", 0),
                        step_count=data.get("step_count", 0),
                    ))
                except (json.JSONDecodeError, OSError, KeyError) as e:
                    logger.warning(f"Failed to read session file {filename}: {e}")
                    continue

        # 按更新时间降序排列
        sessions.sort(key=lambda s: s.updated_at, reverse=True)

        return sessions[:limit]

    def delete(self, session_id: str) -> bool:
        """删除指定会话

        Args:
            session_id: 要删除的会话 ID

        Returns:
            True 表示删除成功
        """
        with self._lock:
            # 从缓存移除
            self._cache.pop(session_id, None)

            # 从磁盘删除
            filepath = self._session_filepath(session_id)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logger.info(f"Session deleted: {session_id}")
                    return True
                except OSError as e:
                    logger.warning(f"Failed to delete session {session_id}: {e}")
                    return False
            else:
                logger.warning(f"Session file not found: {session_id}")
                return False

    def cleanup(self, max_age_days: int | None = None) -> int:
        """清理过期会话

        [H1 修复] 使用 JSON 内的 updated_at 判断过期，而非文件 mtime。
        文件 mtime 可能被系统操作（如复制、git checkout）修改，不可靠。

        Args:
            max_age_days: 最大保留天数（None 则使用实例默认值）

        Returns:
            清理的会话数
        """
        if max_age_days is None:
            max_age_days = self.max_age_days

        now = time.time()
        max_age_seconds = max_age_days * 86400
        cleaned = 0

        with self._lock:
            for filename in os.listdir(self.session_dir):
                if not filename.startswith(self.FILE_PREFIX):
                    continue
                if not filename.endswith(self.FILE_SUFFIX):
                    continue

                filepath = os.path.join(self.session_dir, filename)
                try:
                    # [H1 修复] 读取 JSON 中的 updated_at 判断过期
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    updated_at = data.get("updated_at", 0.0)
                    if now - updated_at > max_age_seconds:
                        session_id = filename[len(self.FILE_PREFIX):-len(self.FILE_SUFFIX)]
                        os.remove(filepath)
                        self._cache.pop(session_id, None)
                        cleaned += 1
                except (json.JSONDecodeError, OSError, KeyError) as e:
                    logger.warning(f"Failed to cleanup session {filename}: {e}")

        if cleaned > 0:
            logger.info(f"Session cleanup: removed {cleaned} sessions older than {max_age_days} days")
        return cleaned

    def get_current_session_id(self) -> str:
        """获取当前活跃会话 ID

        [M3 修复] 使用锁保证线程安全。
        """
        with self._lock:
            return self._current_session_id

    def set_current_session_id(self, session_id: str) -> None:
        """设置当前活跃会话 ID"""
        with self._lock:
            self._current_session_id = session_id

    def update_step_count(self, session_id: str, step_count: int) -> None:
        """更新会话的步数计数

        Args:
            session_id: 会话 ID
            step_count: 当前步数
        """
        with self._lock:
            session = self._cache.get(session_id)
            if session is None:
                session = self._read_from_disk(session_id)
                if session is None:
                    return
                self._cache[session_id] = session

            session.step_count = step_count
            session.updated_at = time.time()

            if self.auto_save:
                self._write_to_disk(session_id)

    def get_status_report(self) -> dict:
        """获取状态报告（供 /status 和调试使用）"""
        sessions = self.list_sessions(limit=100)
        return {
            "session_dir": self.session_dir,
            "total_sessions": len(sessions),
            "current_session_id": self.get_current_session_id(),
            "max_sessions": self.max_sessions,
            "auto_save": self.auto_save,
            "max_age_days": self.max_age_days,
            "cached_sessions": len(self._cache),
        }

    # ===== 内部方法 =====

    def _enforce_max_sessions(self) -> None:
        """[H2] 强制执行 max_sessions 限制

        当会话数量超过 max_sessions 时，删除最旧的会话。
        使用 JSON 内的 updated_at 判断最旧会话，与 cleanup() 保持一致。
        注意：调用方必须持有 self._lock。
        """
        # Scan session files and use JSON updated_at for age determination
        session_data: list[tuple[str, float]] = []
        for filename in os.listdir(self.session_dir):
            if not filename.startswith(self.FILE_PREFIX):
                continue
            if not filename.endswith(self.FILE_SUFFIX):
                continue
            filepath = os.path.join(self.session_dir, filename)
            session_id = filename[len(self.FILE_PREFIX):-len(self.FILE_SUFFIX)]
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                updated_at = data.get("updated_at", 0.0)
                session_data.append((session_id, updated_at))
            except (json.JSONDecodeError, OSError, KeyError):
                # Fallback to file mtime for corrupted/invalid JSON files
                try:
                    mtime = os.path.getmtime(filepath)
                    session_data.append((session_id, mtime))
                except OSError:
                    continue

        # Also count sessions that are only in cache (not yet on disk)
        disk_ids = {sid for sid, _ in session_data}
        for sid, session in self._cache.items():
            if sid not in disk_ids:
                session_data.append((sid, session.updated_at))

        # 如果超过限制，删除最旧的
        if len(session_data) >= self.max_sessions:
            # 按 updated_at 排序，最旧的在前
            session_data.sort(key=lambda x: x[1])
            to_remove = len(session_data) - self.max_sessions + 1
            for session_id, _ in session_data[:to_remove]:
                filepath = self._session_filepath(session_id)
                try:
                    os.remove(filepath)
                    self._cache.pop(session_id, None)
                    logger.info(f"Enforce max_sessions: removed oldest session {session_id}")
                except OSError as e:
                    logger.warning(f"Failed to remove session {session_id}: {e}")

    def _session_filepath(self, session_id: str) -> str:
        """获取会话文件路径"""
        return os.path.join(
            self.session_dir,
            f"{self.FILE_PREFIX}{session_id}{self.FILE_SUFFIX}",
        )

    def _write_to_disk(self, session_id: str) -> None:
        """将会话数据原子写入磁盘

        使用 tempfile + os.replace 保证写入安全。
        注意：调用方必须持有 self._lock。
        """
        session = self._cache.get(session_id)
        if session is None:
            return

        filepath = self._session_filepath(session_id)

        try:
            # 序列化为 JSON
            data = session.to_dict()
            json_str = json.dumps(data, ensure_ascii=False, indent=2)

            # 原子写入：先写临时文件，再 replace
            dir_path = os.path.dirname(filepath)
            os.makedirs(dir_path, exist_ok=True)

            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp",
                prefix="session_",
                dir=dir_path,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(json_str)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, filepath)
            except Exception:
                # 清理临时文件
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        except Exception as e:
            logger.warning(f"Failed to write session {session_id}: {e}")

    def _read_from_disk(self, session_id: str) -> SessionData | None:
        """从磁盘读取会话数据

        [M1 增强] 捕获所有异常类型，防止损坏的会话文件导致崩溃。
        注意：调用方必须持有 self._lock。
        """
        filepath = self._session_filepath(session_id)

        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return SessionData.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to read session {session_id}: {e}")
            return None
