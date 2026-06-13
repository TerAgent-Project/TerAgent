# tests/test_hooks_manager.py
"""Hook 管理器单元测试

测试 teragent.hooks.manager 模块的核心功能:
  - 链式执行（多个 Hook 按顺序运行）
  - DENY 短路终止
  - MODIFY 修改参数后继续
  - 错误隔离（单个 Hook 异常不中断链）
  - Hook 注册/注销/排序
  - 统计和状态报告
"""
import pytest

from teragent.hooks.manager import (
    HookContext,
    HookDecision,
    HookEvent,
    HookManager,
    HookResult,
    PythonHook,
    ShellHook,
)

# ===== 辅助：构建 Python Hook =====

def _make_hook(name: str, event: HookEvent, decision: HookDecision, reason: str = "",
               modified_params: dict | None = None):
    """快速构建一个返回指定决策的 PythonHook"""
    async def handler(context: HookContext) -> HookResult:
        return HookResult(
            decision=decision,
            reason=reason,
            modified_params=modified_params,
        )
    return PythonHook(name=name, event=event, handler=handler)


def _make_error_hook(name: str, event: HookEvent):
    """构建一个会抛异常的 PythonHook"""
    async def handler(context: HookContext) -> HookResult:
        raise RuntimeError(f"Hook {name} failed!")
    return PythonHook(name=name, event=event, handler=handler)


# ===== 测试类 =====


class TestHookEventAndDecision:
    """HookEvent 和 HookDecision 枚举测试"""

    def test_hook_event_values(self):
        """HookEvent 有 4 种事件类型"""
        assert HookEvent.PRE_TOOL_USE.value == "pre_tool_use"
        assert HookEvent.POST_TOOL_USE.value == "post_tool_use"
        assert HookEvent.PRE_MODEL_CALL.value == "pre_model_call"
        assert HookEvent.POST_MODEL_CALL.value == "post_model_call"
        assert len(HookEvent) == 4

    def test_hook_decision_values(self):
        """HookDecision 有 4 种决策类型"""
        assert HookDecision.ALLOW.value == "allow"
        assert HookDecision.DENY.value == "deny"
        assert HookDecision.MODIFY.value == "modify"
        assert HookDecision.PASSTHROUGH.value == "passthrough"
        assert len(HookDecision) == 4


class TestHookRegistration:
    """Hook 注册/注销/查询测试"""

    def test_register_and_get_hooks(self):
        """注册 Hook 后可以通过 get_hooks 查询"""
        manager = HookManager()
        hook = _make_hook("test_hook", HookEvent.PRE_TOOL_USE, HookDecision.ALLOW)
        manager.register(hook)
        hooks = manager.get_hooks(HookEvent.PRE_TOOL_USE)
        assert len(hooks) == 1
        assert hooks[0].name == "test_hook"

    def test_register_multiple_events(self):
        """不同事件的 Hook 相互隔离"""
        manager = HookManager()
        h1 = _make_hook("pre", HookEvent.PRE_TOOL_USE, HookDecision.ALLOW)
        h2 = _make_hook("post", HookEvent.POST_TOOL_USE, HookDecision.ALLOW)
        manager.register(h1)
        manager.register(h2)
        assert len(manager.get_hooks(HookEvent.PRE_TOOL_USE)) == 1
        assert len(manager.get_hooks(HookEvent.POST_TOOL_USE)) == 1

    def test_unregister_hook(self):
        """按名称注销 Hook"""
        manager = HookManager()
        hook = _make_hook("to_remove", HookEvent.PRE_TOOL_USE, HookDecision.ALLOW)
        manager.register(hook)
        result = manager.unregister("to_remove")
        assert result is True
        assert len(manager.get_hooks(HookEvent.PRE_TOOL_USE)) == 0

    def test_unregister_nonexistent(self):
        """注销不存在的 Hook 返回 False"""
        manager = HookManager()
        result = manager.unregister("ghost")
        assert result is False

    def test_get_hooks_all(self):
        """get_hooks(None) 返回所有 Hook"""
        manager = HookManager()
        h1 = _make_hook("a", HookEvent.PRE_TOOL_USE, HookDecision.ALLOW)
        h2 = _make_hook("b", HookEvent.POST_TOOL_USE, HookDecision.ALLOW)
        manager.register(h1)
        manager.register(h2)
        all_hooks = manager.get_hooks()
        assert len(all_hooks) == 2

    def test_clear_all_hooks(self):
        """clear() 清除所有已注册 Hook"""
        manager = HookManager()
        manager.register(_make_hook("a", HookEvent.PRE_TOOL_USE, HookDecision.ALLOW))
        manager.register(_make_hook("b", HookEvent.POST_TOOL_USE, HookDecision.ALLOW))
        manager.clear()
        assert len(manager.get_hooks()) == 0


class TestChainExecution:
    """链式执行测试"""

    @pytest.mark.asyncio
    async def test_chain_order(self):
        """多个 Hook 按注册顺序执行"""
        execution_order = []

        async def handler_a(context: HookContext) -> HookResult:
            execution_order.append("a")
            return HookResult(decision=HookDecision.PASSTHROUGH)

        async def handler_b(context: HookContext) -> HookResult:
            execution_order.append("b")
            return HookResult(decision=HookDecision.PASSTHROUGH)

        manager = HookManager()
        manager.register(PythonHook("a", HookEvent.PRE_TOOL_USE, handler_a))
        manager.register(PythonHook("b", HookEvent.PRE_TOOL_USE, handler_b))

        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="test")
        await manager.run_hooks(HookEvent.PRE_TOOL_USE, ctx)
        assert execution_order == ["a", "b"]

    @pytest.mark.asyncio
    async def test_no_hooks_returns_allow(self):
        """没有注册 Hook 时返回 ALLOW"""
        manager = HookManager()
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="test")
        result = await manager.run_hooks(HookEvent.PRE_TOOL_USE, ctx)
        assert result.decision == HookDecision.ALLOW


class TestDenyShortCircuit:
    """DENY 短路终止测试"""

    @pytest.mark.asyncio
    async def test_deny_stops_chain(self):
        """DENY 决策立即终止后续 Hook 执行"""
        execution_order = []

        async def handler_a(context: HookContext) -> HookResult:
            execution_order.append("a")
            return HookResult(decision=HookDecision.DENY, reason="blocked")

        async def handler_b(context: HookContext) -> HookResult:
            execution_order.append("b")
            return HookResult(decision=HookDecision.ALLOW)

        manager = HookManager()
        manager.register(PythonHook("a", HookEvent.PRE_TOOL_USE, handler_a))
        manager.register(PythonHook("b", HookEvent.PRE_TOOL_USE, handler_b))

        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="test")
        result = await manager.run_hooks(HookEvent.PRE_TOOL_USE, ctx)
        assert result.decision == HookDecision.DENY
        assert result.reason == "blocked"
        assert execution_order == ["a"]  # b 未执行


class TestModifyHook:
    """MODIFY Hook 修改参数测试"""

    @pytest.mark.asyncio
    async def test_modify_updates_context_params(self):
        """MODIFY 修改上下文参数后继续执行"""
        async def handler_modify(context: HookContext) -> HookResult:
            return HookResult(
                decision=HookDecision.MODIFY,
                reason="sanitized",
                modified_params={"command": "ls -la"},
            )

        manager = HookManager()
        manager.register(PythonHook("mod", HookEvent.PRE_TOOL_USE, handler_modify))

        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="execute_subtask",
            params={"command": "rm -rf /"},
        )
        result = await manager.run_hooks(HookEvent.PRE_TOOL_USE, ctx)
        assert result.decision == HookDecision.MODIFY
        assert ctx.params["command"] == "ls -la"
        assert result.modified_params["command"] == "ls -la"

    @pytest.mark.asyncio
    async def test_modify_then_allow(self):
        """MODIFY 后继续执行，后续 Hook 能看到修改后的参数"""
        async def handler_modify(context: HookContext) -> HookResult:
            return HookResult(
                decision=HookDecision.MODIFY,
                reason="rewrite",
                modified_params={"command": "safe_cmd"},
            )

        seen_params = {}

        async def handler_observe(context: HookContext) -> HookResult:
            seen_params.update(context.params)
            return HookResult(decision=HookDecision.ALLOW)

        manager = HookManager()
        manager.register(PythonHook("mod", HookEvent.PRE_TOOL_USE, handler_modify))
        manager.register(PythonHook("obs", HookEvent.PRE_TOOL_USE, handler_observe))

        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="execute_subtask",
            params={"command": "original"},
        )
        result = await manager.run_hooks(HookEvent.PRE_TOOL_USE, ctx)
        assert result.decision == HookDecision.MODIFY
        assert seen_params["command"] == "safe_cmd"


class TestErrorIsolation:
    """错误隔离测试"""

    @pytest.mark.asyncio
    async def test_hook_exception_doesnt_break_chain(self):
        """单个 Hook 异常不中断后续 Hook 执行"""
        execution_order = []

        async def handler_error(context: HookContext) -> HookResult:
            execution_order.append("error")
            raise ValueError("boom")

        async def handler_ok(context: HookContext) -> HookResult:
            execution_order.append("ok")
            return HookResult(decision=HookDecision.ALLOW)

        manager = HookManager()
        manager.register(PythonHook("err", HookEvent.PRE_TOOL_USE, handler_error))
        manager.register(PythonHook("ok", HookEvent.PRE_TOOL_USE, handler_ok))

        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="test")
        result = await manager.run_hooks(HookEvent.PRE_TOOL_USE, ctx)
        assert result.decision == HookDecision.ALLOW
        assert execution_order == ["error", "ok"]

    @pytest.mark.asyncio
    async def test_error_count_in_stats(self):
        """Hook 异常增加 error_count（使用自定义 Hook 让异常从 execute 传播）"""
        from teragent.hooks.manager import Hook

        class BrokenHook(Hook):
            """execute() 直接抛异常的 Hook（模拟 ShellHook 级别的错误）"""
            name = "broken"
            event = HookEvent.PRE_TOOL_USE

            async def execute(self, context: HookContext) -> HookResult:
                raise RuntimeError("broken!")

        manager = HookManager()
        manager.register(BrokenHook())

        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="test")
        result = await manager.run_hooks(HookEvent.PRE_TOOL_USE, ctx)
        # 异常被 HookManager 捕获，返回 ALLOW（因为无其他 Hook）
        assert result.decision == HookDecision.ALLOW
        report = manager.get_status_report()
        assert report["stats"]["error_count"] == 1


class TestHookManagerStats:
    """统计和状态报告测试"""

    @pytest.mark.asyncio
    async def test_stats_deny_count(self):
        """DENY 决策增加 deny_count"""
        manager = HookManager()
        manager.register(_make_hook("deny", HookEvent.PRE_TOOL_USE, HookDecision.DENY, "no"))
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="test")
        await manager.run_hooks(HookEvent.PRE_TOOL_USE, ctx)
        report = manager.get_status_report()
        assert report["stats"]["deny_count"] == 1

    @pytest.mark.asyncio
    async def test_stats_modify_count(self):
        """MODIFY 决策增加 modify_count"""
        manager = HookManager()
        manager.register(_make_hook(
            "mod", HookEvent.PRE_TOOL_USE, HookDecision.MODIFY,
            modified_params={"k": "v"},
        ))
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="test")
        await manager.run_hooks(HookEvent.PRE_TOOL_USE, ctx)
        report = manager.get_status_report()
        assert report["stats"]["modify_count"] == 1

    @pytest.mark.asyncio
    async def test_status_report_hooks_info(self):
        """状态报告包含已注册 Hook 信息"""
        manager = HookManager()
        manager.register(_make_hook("my_hook", HookEvent.PRE_TOOL_USE, HookDecision.ALLOW))
        report = manager.get_status_report()
        pre_hooks = report["hooks"]["pre_tool_use"]
        assert len(pre_hooks) == 1
        assert pre_hooks[0]["name"] == "my_hook"
        assert pre_hooks[0]["type"] == "PythonHook"


class TestLoadFromConfig:
    """从配置加载 Hook 测试"""

    def test_load_shell_hook_from_config(self):
        """从配置加载 ShellHook"""
        manager = HookManager()
        config = {
            "pre_tool_use": [
                {"type": "shell", "name": "my_shell", "command": "echo hello", "timeout": 5.0},
            ],
        }
        manager.load_from_config(config)
        hooks = manager.get_hooks(HookEvent.PRE_TOOL_USE)
        assert len(hooks) == 1
        assert isinstance(hooks[0], ShellHook)
        assert hooks[0].name == "my_shell"

    def test_load_invalid_type_ignored(self):
        """未知类型的 Hook 配置被忽略"""
        manager = HookManager()
        config = {
            "pre_tool_use": [
                {"type": "unknown_type", "name": "bad"},
            ],
        }
        manager.load_from_config(config)
        assert len(manager.get_hooks(HookEvent.PRE_TOOL_USE)) == 0

    def test_load_missing_command_ignored(self):
        """缺少 command 的 shell Hook 被忽略"""
        manager = HookManager()
        config = {
            "pre_tool_use": [
                {"type": "shell", "name": "no_cmd"},
            ],
        }
        manager.load_from_config(config)
        assert len(manager.get_hooks(HookEvent.PRE_TOOL_USE)) == 0
