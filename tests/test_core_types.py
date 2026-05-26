# tests/test_core_types.py
"""core/types.py 单元测试

覆盖:
  - MessageRole 枚举值
  - MessageType 枚举值与分类
  - Message 工厂方法: user_input / assistant_text / system_prompt / tool_result / assistant_tool_call
  - Message.to_api_format() 字典结构
  - Message.from_dict() 与 _infer_message_type() 推断逻辑
  - TOMBSTONE 墓碑消息行为
  - 便捷属性: is_system / is_user / is_assistant / is_tool / has_tool_calls
  - messages_to_api_format / messages_from_dicts 批量转换
"""
import pytest
import time

from teragent.core.types import (
    Message,
    MessageRole,
    MessageType,
    ToolSafety,
    messages_to_api_format,
    messages_from_dicts,
)


# ===== MessageRole 枚举 =====

class TestMessageRole:
    """MessageRole 枚举测试"""

    def test_role_values(self):
        """各角色对应正确的字符串值"""
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.TOOL.value == "tool"

    def test_role_from_string(self):
        """通过字符串构造 MessageRole"""
        assert MessageRole("user") == MessageRole.USER
        assert MessageRole("assistant") == MessageRole.ASSISTANT
        assert MessageRole("system") == MessageRole.SYSTEM
        assert MessageRole("tool") == MessageRole.TOOL


# ===== MessageType 枚举 =====

class TestMessageType:
    """MessageType 枚举测试"""

    def test_basic_message_types(self):
        """基础对话类型值正确"""
        assert MessageType.USER_INPUT.value == "user_input"
        assert MessageType.ASSISTANT_TEXT.value == "assistant_text"
        assert MessageType.ASSISTANT_TOOL_CALL.value == "assistant_tool_call"
        assert MessageType.TOOL_RESULT.value == "tool_result"

    def test_system_message_types(self):
        """系统消息类型值正确"""
        assert MessageType.SYSTEM_PROMPT.value == "system_prompt"
        assert MessageType.SYSTEM_REMINDER.value == "system_reminder"
        assert MessageType.SYSTEM_WARNING.value == "system_warning"
        assert MessageType.SYSTEM_ERROR.value == "system_error"

    def test_special_message_types(self):
        """特殊消息类型值正确"""
        assert MessageType.TOMBSTONE.value == "tombstone"
        assert MessageType.CONTEXT_SUMMARY.value == "context_summary"
        assert MessageType.REPAIR_INSTRUCTION.value == "repair_instruction"
        assert MessageType.REVIEW_FEEDBACK.value == "review_feedback"


# ===== Message 工厂方法 =====

class TestMessageFactory:
    """Message 工厂方法测试"""

    def test_user_input(self):
        """user_input 创建正确的用户输入消息"""
        msg = Message.user_input("帮我写一个贪吃蛇游戏")
        assert msg.role == MessageRole.USER
        assert msg.content == "帮我写一个贪吃蛇游戏"
        assert msg.message_type == MessageType.USER_INPUT
        assert msg.tool_calls == []
        assert msg.tool_call_id == ""

    def test_assistant_text(self):
        """assistant_text 创建正确的模型文本回复消息"""
        msg = Message.assistant_text("好的，我来帮你写")
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "好的，我来帮你写"
        assert msg.message_type == MessageType.ASSISTANT_TEXT
        assert msg.tool_calls == []

    def test_system_prompt(self):
        """system_prompt 创建正确的系统提示消息"""
        msg = Message.system_prompt("你是一个代码助手")
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "你是一个代码助手"
        assert msg.message_type == MessageType.SYSTEM_PROMPT

    def test_tool_result(self):
        """tool_result 创建正确的工具执行结果消息"""
        msg = Message.tool_result("call_abc", "read_file", "文件内容...")
        assert msg.role == MessageRole.TOOL
        assert msg.content == "文件内容..."
        assert msg.message_type == MessageType.TOOL_RESULT
        assert msg.tool_call_id == "call_abc"
        assert msg.tool_name == "read_file"

    def test_assistant_tool_call(self):
        """assistant_tool_call 创建正确的模型工具调用请求消息"""
        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]
        msg = Message.assistant_tool_call("", tool_calls=tool_calls)
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == ""
        assert msg.message_type == MessageType.ASSISTANT_TOOL_CALL
        assert msg.tool_calls == tool_calls


# ===== Message.to_api_format() =====

class TestMessageToApiFormat:
    """Message.to_api_format() 测试"""

    def test_user_input_api_format(self):
        """用户消息的 API 格式只有 role 和 content"""
        msg = Message.user_input("你好")
        api = msg.to_api_format()
        assert api == {"role": "user", "content": "你好"}
        # 不应包含 tool_calls / tool_call_id 等字段
        assert "tool_calls" not in api
        assert "tool_call_id" not in api

    def test_assistant_tool_call_api_format(self):
        """工具调用消息的 API 格式包含 tool_calls"""
        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]
        msg = Message.assistant_tool_call("思考中...", tool_calls=tool_calls)
        api = msg.to_api_format()
        assert api["role"] == "assistant"
        assert api["content"] == "思考中..."
        assert "tool_calls" in api
        assert api["tool_calls"] == tool_calls

    def test_tool_result_api_format(self):
        """工具结果消息的 API 格式包含 tool_call_id"""
        msg = Message.tool_result("call_abc", "read_file", "文件内容")
        api = msg.to_api_format()
        assert api["role"] == "tool"
        assert api["content"] == "文件内容"
        assert api["tool_call_id"] == "call_abc"


# ===== Message._infer_message_type() =====

class TestInferMessageType:
    """Message._infer_message_type() 推断逻辑测试"""

    def test_infer_assistant_with_tool_calls(self):
        """assistant + tool_calls → ASSISTANT_TOOL_CALL"""
        data = {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]}
        msg_type = Message._infer_message_type(MessageRole.ASSISTANT, data)
        assert msg_type == MessageType.ASSISTANT_TOOL_CALL

    def test_infer_assistant_without_tool_calls(self):
        """assistant 无 tool_calls → ASSISTANT_TEXT"""
        data = {"role": "assistant", "content": "文本回复"}
        msg_type = Message._infer_message_type(MessageRole.ASSISTANT, data)
        assert msg_type == MessageType.ASSISTANT_TEXT

    def test_infer_user_context_summary(self):
        """user + [上下文摘要 → CONTEXT_SUMMARY"""
        data = {"role": "user", "content": "[上下文摘要] 之前讨论了..."}
        msg_type = Message._infer_message_type(MessageRole.USER, data)
        assert msg_type == MessageType.CONTEXT_SUMMARY

    def test_infer_system_warning(self):
        """system + [WARN] → SYSTEM_WARNING"""
        data = {"role": "system", "content": "[WARN] 轮询检测"}
        msg_type = Message._infer_message_type(MessageRole.SYSTEM, data)
        assert msg_type == MessageType.SYSTEM_WARNING

    def test_infer_system_repair(self):
        """system + [AUTO-REPAIR] → REPAIR_INSTRUCTION"""
        data = {"role": "system", "content": "[AUTO-REPAIR] 请修复代码"}
        msg_type = Message._infer_message_type(MessageRole.SYSTEM, data)
        assert msg_type == MessageType.REPAIR_INSTRUCTION

    def test_infer_system_default_reminder(self):
        """system 无特殊标记 → SYSTEM_PROMPT（通用系统消息归为提示词）"""
        data = {"role": "system", "content": "普通系统消息"}
        msg_type = Message._infer_message_type(MessageRole.SYSTEM, data)
        assert msg_type == MessageType.SYSTEM_PROMPT


# ===== TOMBSTONE 墓碑消息 =====

class TestTombstone:
    """TOMBSTONE 墓碑消息测试"""

    def test_tombstone_default_role(self):
        """默认墓碑消息角色为 SYSTEM"""
        msg = Message.tombstone()
        assert msg.message_type == MessageType.TOMBSTONE
        assert msg.role == MessageRole.SYSTEM
        assert "[TOMBSTONE]" in msg.content

    def test_tombstone_preserves_original_role(self):
        """墓碑消息保留原始消息角色"""
        msg = Message.tombstone(original_role=MessageRole.USER)
        assert msg.role == MessageRole.USER
        assert msg.message_type == MessageType.TOMBSTONE

    def test_tombstone_api_format(self):
        """墓碑消息的 API 格式仍然可用"""
        msg = Message.tombstone(original_role=MessageRole.ASSISTANT)
        api = msg.to_api_format()
        assert api["role"] == "assistant"
        assert "[TOMBSTONE]" in api["content"]


# ===== Message from_dict 与便捷属性 =====

class TestMessageFromDictAndProperties:
    """Message.from_dict() 及便捷属性测试"""

    def test_from_dict_basic(self):
        """从基本 dict 创建 Message"""
        data = {"role": "user", "content": "你好"}
        msg = Message.from_dict(data)
        assert msg.role == MessageRole.USER
        assert msg.content == "你好"
        assert msg.message_type == MessageType.USER_INPUT

    def test_from_dict_with_tool_calls(self):
        """从带 tool_calls 的 dict 创建 Message"""
        tool_calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        data = {"role": "assistant", "content": "", "tool_calls": tool_calls}
        msg = Message.from_dict(data)
        assert msg.message_type == MessageType.ASSISTANT_TOOL_CALL
        assert msg.tool_calls == tool_calls

    def test_convenience_properties(self):
        """便捷属性 is_system / is_user / is_assistant / is_tool"""
        assert Message.system_prompt("x").is_system is True
        assert Message.user_input("x").is_user is True
        assert Message.assistant_text("x").is_assistant is True
        assert Message.tool_result("id", "name", "x").is_tool is True

    def test_has_tool_calls_property(self):
        """has_tool_calls 属性正确反映工具调用状态"""
        msg_with = Message.assistant_tool_call("", tool_calls=[{"id": "1"}])
        msg_without = Message.assistant_text("纯文本")
        assert msg_with.has_tool_calls is True
        assert msg_without.has_tool_calls is False


# ===== 批量转换函数 =====

class TestBatchConversion:
    """messages_to_api_format / messages_from_dicts 批量转换测试"""

    def test_messages_to_api_format(self):
        """批量转换为 API 格式"""
        msgs = [
            Message.user_input("你好"),
            Message.assistant_text("你好！"),
        ]
        result = messages_to_api_format(msgs)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "你好"}
        assert result[1] == {"role": "assistant", "content": "你好！"}

    def test_messages_from_dicts(self):
        """从 dict 列表批量创建 Message"""
        dicts = [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "回答"},
        ]
        msgs = messages_from_dicts(dicts)
        assert len(msgs) == 2
        assert msgs[0].role == MessageRole.USER
        assert msgs[1].role == MessageRole.ASSISTANT
