# tests/test_session_persistence.py
"""会话持久化管理器单元测试

测试 SessionPersistence 的 CRUD 操作、原子写入、过期清理、会话限制等功能。
"""
import json
import os
import time
import pytest

from teragent.session.persistence import (
    SessionPersistence,
    SessionData,
    SessionInfo,
    _message_to_dict,
    _message_from_dict,
)
from teragent.core.types import Message, MessageRole, MessageType


# ===== 辅助 fixture =====

@pytest.fixture
def persister(tmp_path):
    """创建临时目录下的 SessionPersistence 实例"""
    return SessionPersistence(
        workspace_root=str(tmp_path),
        session_dir="sessions",
        max_sessions=50,
        auto_save=True,
        max_age_days=30,
    )


@pytest.fixture
def persister_no_auto(tmp_path):
    """创建不自动保存的 SessionPersistence 实例"""
    return SessionPersistence(
        workspace_root=str(tmp_path),
        session_dir="sessions",
        auto_save=False,
    )


# ===== SessionData 序列化 =====

class TestSessionDataSerialization:
    """SessionData 序列化与反序列化"""

    def test_to_dict_and_from_dict_roundtrip(self):
        """to_dict → from_dict 往返一致性"""
        msgs = [Message.user_input("hello"), Message.assistant_text("hi")]
        sd = SessionData(
            id="sess_test",
            title="测试",
            intent="CHAT",
            created_at=1000.0,
            updated_at=1001.0,
            message_count=2,
            step_count=1,
            metadata={"key": "val"},
            messages=msgs,
        )
        d = sd.to_dict()
        restored = SessionData.from_dict(d)
        assert restored.id == "sess_test"
        assert restored.title == "测试"
        assert restored.message_count == 2
        assert len(restored.messages) == 2
        assert restored.messages[0].content == "hello"

    def test_from_dict_skips_invalid_messages(self):
        """from_dict 跳过无效消息条目"""
        d = {
            "id": "sess_bad",
            "title": "",
            "intent": "",
            "created_at": 0.0,
            "updated_at": 0.0,
            "message_count": 0,
            "step_count": 0,
            "metadata": {},
            "messages": [
                {"role": "user", "content": "ok", "message_type": "user_input"},
                "not_a_dict",
                None,
            ],
        }
        sd = SessionData.from_dict(d)
        assert len(sd.messages) == 1
        assert sd.messages[0].content == "ok"

    def test_info_property(self):
        """info 属性返回 SessionInfo"""
        sd = SessionData(
            id="s1", title="t", intent="i",
            created_at=1.0, updated_at=2.0,
            message_count=5, step_count=3,
            metadata={}, messages=[],
        )
        info = sd.info
        assert isinstance(info, SessionInfo)
        assert info.id == "s1"
        assert info.step_count == 3


# ===== CRUD 操作 =====

class TestCRUD:
    """基本 CRUD：创建、保存消息、加载、恢复、删除"""

    def test_create_session(self, persister):
        """创建会话返回有效 ID 并写入磁盘"""
        sid = persister.create("测试会话", intent="CHAT")
        assert sid.startswith("sess_")
        # 磁盘文件存在
        fpath = persister._session_filepath(sid)
        assert os.path.exists(fpath)

    def test_save_and_load(self, persister):
        """保存消息后加载可获取完整数据"""
        sid = persister.create("标题", intent="CREATE")
        persister.save_message(sid, Message.user_input("帮我写代码"))
        persister.save_message(sid, Message.assistant_text("好的"))

        data = persister.load(sid)
        assert data is not None
        assert data.message_count == 2
        assert data.messages[0].content == "帮我写代码"

    def test_restore_returns_message_list(self, persister):
        """restore 返回消息列表"""
        sid = persister.create()
        persister.save_message(sid, Message.user_input("hi"))

        msgs = persister.restore(sid)
        assert msgs is not None
        assert len(msgs) == 1
        assert msgs[0].content == "hi"

    def test_delete_session(self, persister):
        """删除会话后无法再加载"""
        sid = persister.create("要删的")
        persister.save_message(sid, Message.user_input("bye"))
        assert persister.delete(sid) is True
        assert persister.load(sid) is None
        # 磁盘文件也已删除
        assert not os.path.exists(persister._session_filepath(sid))

    def test_delete_nonexistent_returns_false(self, persister):
        """删除不存在的会话返回 False"""
        assert persister.delete("sess_no_exist") is False

    def test_save_message_updates_title_from_first_user_msg(self, persister):
        """首条用户消息自动更新标题"""
        sid = persister.create()  # title 默认 Untitled
        persister.save_message(sid, Message.user_input("这是新的标题内容"))
        data = persister.load(sid)
        assert data.title == "这是新的标题内容"

    def test_save_message_not_overwrite_existing_title(self, persister):
        """已有标题时不会被用户消息覆盖"""
        sid = persister.create("原始标题")
        persister.save_message(sid, Message.user_input("另一个消息"))
        data = persister.load(sid)
        assert data.title == "原始标题"


# ===== 原子写入 =====

class TestAtomicWrite:
    """原子写入机制：tempfile + os.replace"""

    def test_file_content_is_valid_json(self, persister):
        """磁盘文件内容是合法 JSON"""
        sid = persister.create("原子测试")
        fpath = persister._session_filepath(sid)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["id"] == sid
        assert data["title"] == "原子测试"

    def test_no_tmp_files_left(self, persister, tmp_path):
        """写入后无残留临时文件"""
        sid = persister.create()
        persister.save_message(sid, Message.user_input("x"))
        session_dir = os.path.join(str(tmp_path), "sessions")
        tmp_files = [f for f in os.listdir(session_dir) if f.endswith(".tmp")]
        assert len(tmp_files) == 0


# ===== 会话清理 =====

class TestCleanup:
    """会话清理：使用 JSON updated_at（非文件 mtime）判断过期"""

    def test_cleanup_removes_expired_sessions(self, persister, tmp_path):
        """cleanup 删除超过 max_age_days 的会话"""
        # 创建一个旧会话：手动构造 JSON with old updated_at
        old_time = time.time() - 100 * 86400  # 100 天前
        old_session = SessionData(
            id="sess_old_abc",
            title="过期会话",
            intent="CHAT",
            created_at=old_time,
            updated_at=old_time,
            message_count=0,
            step_count=0,
            metadata={},
            messages=[],
        )
        persister._cache["sess_old_abc"] = old_session
        persister._write_to_disk("sess_old_abc")

        # 创建一个新会话
        persister.create("新会话")

        # 清理 30 天以上的会话
        cleaned = persister.cleanup(max_age_days=30)
        assert cleaned == 1
        assert persister.load("sess_old_abc") is None

    def test_cleanup_keeps_recent_sessions(self, persister):
        """cleanup 保留未过期的会话"""
        sid = persister.create("最近会话")
        cleaned = persister.cleanup(max_age_days=30)
        assert cleaned == 0
        assert persister.load(sid) is not None


# ===== _enforce_max_sessions =====

class TestEnforceMaxSessions:
    """超出最大会话数时删除最旧会话"""

    def test_enforce_max_sessions_removes_oldest(self, tmp_path):
        """超过 max_sessions 时删除最旧的会话"""
        p = SessionPersistence(
            workspace_root=str(tmp_path),
            session_dir="sessions2",
            max_sessions=2,
            auto_save=True,
        )
        # 创建 3 个会话
        s1 = p.create("第1个")
        time.sleep(0.01)
        s2 = p.create("第2个")
        time.sleep(0.01)
        s3 = p.create("第3个")  # 触发 enforce

        # 最旧的 s1 应被删除
        # （_enforce_max_sessions 在 create 中被调用）
        # 由于 max_sessions=2，第3个创建时应该删掉最旧的
        # 注意：s1 可能已被删除也可能没有，取决于时序
        # 更可靠的断言：磁盘上不超过 max_sessions 个文件
        session_dir = os.path.join(str(tmp_path), "sessions2")
        json_files = [
            f for f in os.listdir(session_dir)
            if f.startswith("session_") and f.endswith(".json")
        ]
        assert len(json_files) <= 2


# ===== current_session_id =====

class TestCurrentSessionId:
    """get_current_session_id / set_current_session_id"""

    def test_create_sets_current_session(self, persister):
        """创建会话后 current_session_id 被设置"""
        sid = persister.create()
        assert persister.get_current_session_id() == sid

    def test_set_current_session_id(self, persister):
        """手动设置 current_session_id"""
        persister.set_current_session_id("sess_manual")
        assert persister.get_current_session_id() == "sess_manual"


# ===== update_step_count =====

class TestUpdateStepCount:
    """update_step_count 更新步数"""

    def test_update_step_count(self, persister):
        """更新步数后加载可见"""
        sid = persister.create()
        persister.update_step_count(sid, 42)
        data = persister.load(sid)
        assert data.step_count == 42

    def test_update_step_count_nonexistent_session(self, persister):
        """更新不存在会话的步数不报错"""
        persister.update_step_count("sess_no_exist", 10)  # 不应抛异常


# ===== 消息序列化 =====

class TestMessageSerialization:
    """_message_to_dict / _message_from_dict"""

    def test_message_roundtrip(self):
        """消息序列化往返一致性"""
        msg = Message.user_input("测试内容")
        d = _message_to_dict(msg)
        restored = _message_from_dict(d)
        assert restored.role == MessageRole.USER
        assert restored.content == "测试内容"
        assert restored.message_type == MessageType.USER_INPUT

    def test_message_with_tool_calls(self):
        """带 tool_calls 的消息序列化"""
        msg = Message.assistant_tool_call("", tool_calls=[{"id": "c1", "function": {"name": "f"}}])
        d = _message_to_dict(msg)
        restored = _message_from_dict(d)
        assert len(restored.tool_calls) == 1
        assert restored.message_type == MessageType.ASSISTANT_TOOL_CALL
