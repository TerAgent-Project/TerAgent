# tests/test_builtin_hooks.py
"""内置 Hook 单元测试

测试 teragent.hooks.builtin 模块:
  - audit_hook: 审计日志记录 Hook
  - dangerous_command_hook: 危险命令拦截 Hook
"""
import pytest

from teragent.hooks.manager import (
    HookContext,
    HookDecision,
    HookEvent,
    HookResult,
)
from teragent.hooks.builtin.dangerous_command_hook import (
    DangerousCommandHook,
    create_dangerous_command_hook,
    DEFAULT_DANGEROUS_PATTERNS,
    WARNING_PATTERNS,
)
from teragent.hooks.builtin.audit_hook import (
    AuditHook,
    create_audit_hook,
)


# ===== DangerousCommandHook 测试 =====


class TestDangerousCommandHookInit:
    """DangerousCommandHook 初始化测试"""

    def test_default_patterns(self):
        """Default patterns are delegated to sandbox.classify_command_risk(), not stored locally"""
        hook = DangerousCommandHook()
        # No local _dangerous_patterns; risk classification is delegated to sandbox
        assert not hasattr(hook, '_dangerous_patterns') or not hook._extra_dangerous_patterns
        # But the hook should still detect dangerous commands via sandbox delegation
        assert hook.name == "dangerous_command"

    def test_extra_patterns(self):
        """自定义额外危险模式 stored in _extra_dangerous_patterns"""
        hook = DangerousCommandHook(extra_patterns=["my_dangerous_cmd"])
        assert "my_dangerous_cmd" in hook._extra_dangerous_patterns
        # Default patterns are NOT stored locally — they're delegated to sandbox

    def test_extra_warning_patterns(self):
        """自定义额外警告模式 stored in _extra_warning_patterns"""
        hook = DangerousCommandHook(extra_warning_patterns=["cargo install"])
        assert "cargo install" in hook._extra_warning_patterns

    def test_hook_properties(self):
        """Hook 名称和事件类型正确"""
        hook = DangerousCommandHook()
        assert hook.name == "dangerous_command"
        assert hook.event == HookEvent.PRE_TOOL_USE

    def test_factory_function(self):
        """工厂函数创建实例"""
        hook = create_dangerous_command_hook()
        assert isinstance(hook, DangerousCommandHook)


class TestDangerousCommandHookDeny:
    """DangerousCommandHook DENY 拦截测试"""

    @pytest.mark.asyncio
    async def test_deny_rm_rf(self):
        """rm -rf 命令被 DENY"""
        hook = DangerousCommandHook()
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="execute_subtask",
            params={"command": "rm -rf /"},
        )
        result = await hook.execute(ctx)
        assert result.decision == HookDecision.DENY
        assert "CRITICAL" in result.reason

    @pytest.mark.asyncio
    async def test_deny_sudo(self):
        """sudo 命令被 DENY"""
        hook = DangerousCommandHook()
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="execute_subtask",
            params={"command": "sudo apt-get install something"},
        )
        result = await hook.execute(ctx)
        assert result.decision == HookDecision.DENY

    @pytest.mark.asyncio
    async def test_passthrough_non_execute_subtask(self):
        """非 execute_subtask 工具直接 PASSTHROUGH"""
        hook = DangerousCommandHook()
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="read_file",
            params={"command": "rm -rf /"},
        )
        result = await hook.execute(ctx)
        assert result.decision == HookDecision.PASSTHROUGH


class TestDangerousCommandHookWarning:
    """DangerousCommandHook 警告模式测试"""

    @pytest.mark.asyncio
    async def test_warning_pip_install(self):
        """pip install 触发 WARNING 但 ALLOW"""
        hook = DangerousCommandHook()
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="execute_subtask",
            params={"command": "pip install requests"},
        )
        result = await hook.execute(ctx)
        assert result.decision == HookDecision.ALLOW
        assert "WARN" in result.reason

    @pytest.mark.asyncio
    async def test_safe_command_allowed(self):
        """安全命令直接 ALLOW（无 reason）"""
        hook = DangerousCommandHook()
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="execute_subtask",
            params={"command": "ls -la /home/user"},
        )
        result = await hook.execute(ctx)
        assert result.decision == HookDecision.ALLOW
        assert result.reason == ""

    @pytest.mark.asyncio
    async def test_empty_command_allowed(self):
        """空命令直接 ALLOW"""
        hook = DangerousCommandHook()
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="execute_subtask",
            params={"command": ""},
        )
        result = await hook.execute(ctx)
        assert result.decision == HookDecision.ALLOW


class TestDangerousCommandHookTruncate:
    """DangerousCommandHook 截断方法测试"""

    def test_truncate_short_text(self):
        """短文本不截断"""
        assert DangerousCommandHook._truncate("hello", 100) == "hello"

    def test_truncate_long_text(self):
        """长文本截断并添加省略号"""
        long_text = "a" * 200
        result = DangerousCommandHook._truncate(long_text, 100)
        assert len(result) == 103  # 100 + "..."
        assert result.endswith("...")


# ===== AuditHook 测试 =====


class TestAuditHookInit:
    """AuditHook 初始化测试"""

    def test_hook_properties(self):
        """Hook 名称和事件类型正确"""
        hook = AuditHook()
        assert hook.name == "audit"
        assert hook.event == HookEvent.PRE_TOOL_USE

    def test_post_hook_attached(self):
        """POST_TOOL_USE Hook 作为附加属性"""
        hook = AuditHook()
        assert hook.post_hook is not None
        assert hook.post_hook.name == "audit_post"
        assert hook.post_hook.event == HookEvent.POST_TOOL_USE

    def test_factory_function(self):
        """工厂函数创建实例"""
        hook = create_audit_hook()
        assert isinstance(hook, AuditHook)


class TestAuditHookExecution:
    """AuditHook 执行测试"""

    @pytest.mark.asyncio
    async def test_pre_tool_use_returns_allow(self):
        """PRE_TOOL_USE 始终返回 ALLOW"""
        hook = AuditHook()
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="read_file",
            params={"path": "/tmp/test"},
            metadata={"call_id": "test_123"},
        )
        result = await hook.execute(ctx)
        assert result.decision == HookDecision.ALLOW

    @pytest.mark.asyncio
    async def test_post_tool_use_returns_allow(self):
        """POST_TOOL_USE 始终返回 ALLOW"""
        hook = AuditHook()
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="read_file",
            result={"success": True},
            metadata={"call_id": "test_123"},
        )
        result = await hook.post_hook.execute(ctx)
        assert result.decision == HookDecision.ALLOW

    @pytest.mark.asyncio
    async def test_pre_records_call_time(self):
        """PRE_TOOL_USE 记录调用开始时间"""
        hook = AuditHook()
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="test_tool",
            params={},
            metadata={"call_id": "call_1"},
        )
        await hook.execute(ctx)
        assert "call_1" in hook._call_times


class TestAuditHookTruncateParams:
    """AuditHook 参数截断测试"""

    def test_short_params(self):
        """短参数不截断"""
        result = AuditHook._truncate_params({"key": "value"})
        assert "key" in result
        assert "..." not in result

    def test_long_params_truncated(self):
        """长参数被截断"""
        long_params = {"data": "x" * 300}
        result = AuditHook._truncate_params(long_params, max_len=100)
        assert result.endswith("...")
