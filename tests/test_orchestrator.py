# tests/test_orchestrator.py
"""工具编排器单元测试

测试 ToolOrchestrator 的分区策略、并行/串行执行、Hook 集成、权限检查等。
"""
import asyncio
import pytest

from teragent.tools.orchestrator import ToolOrchestrator
from teragent.tools.base import BaseTool, ToolResult
from teragent.tools.registry import ToolRegistry
from teragent.core.types import ToolSafety
from teragent.hooks.manager import (
    HookManager, HookEvent, HookDecision, HookContext, HookResult, PythonHook,
)


# ===== 测试用工具 =====

class ReadOnlyConcurrentTool(BaseTool):
    """只读 + 并发安全工具"""
    name = "read_file"
    description = "读取文件"
    parameters_schema = {"type": "object", "properties": {}}
    _safety = ToolSafety.READ_ONLY
    _concurrency_safe = True

    async def execute(self, params, progress_callback=None):
        await asyncio.sleep(0.01)  # 模拟IO
        return ToolResult(success=True, data={"content": "data"})


class ReadOnlyNonConcurrentTool(BaseTool):
    """只读但不可并发的工具"""
    name = "search_code"
    description = "搜索代码"
    parameters_schema = {"type": "object", "properties": {}}
    _safety = ToolSafety.READ_ONLY
    _concurrency_safe = False

    async def execute(self, params, progress_callback=None):
        return ToolResult(success=True, data={"matches": []})


class WriteTool(BaseTool):
    """写入工具"""
    name = "write_file"
    description = "写入文件"
    parameters_schema = {"type": "object", "properties": {}}
    _safety = ToolSafety.SAFE_WRITE
    _concurrency_safe = False

    async def execute(self, params, progress_callback=None):
        return ToolResult(success=True, data={"written": True})


class DestructiveTool(BaseTool):
    """破坏性工具"""
    name = "delete_file"
    description = "删除文件"
    parameters_schema = {"type": "object", "properties": {}}
    _safety = ToolSafety.DESTRUCTIVE
    _concurrency_safe = False

    async def execute(self, params, progress_callback=None):
        return ToolResult(success=True, data={"deleted": True})


class FailingTool(BaseTool):
    """执行失败的工具"""
    name = "fail_tool"
    description = "总是失败"
    parameters_schema = {}
    _safety = ToolSafety.SAFE_WRITE

    async def execute(self, params, progress_callback=None):
        raise RuntimeError("执行出错")


# ===== 辅助 fixture =====

@pytest.fixture
def registry():
    """注册所有测试工具"""
    reg = ToolRegistry()
    reg.register(ReadOnlyConcurrentTool())
    reg.register(ReadOnlyNonConcurrentTool())
    reg.register(WriteTool())
    reg.register(DestructiveTool())
    reg.register(FailingTool())
    return reg


@pytest.fixture
def orchestrator(registry):
    """创建默认编排器"""
    return ToolOrchestrator(tool_registry=registry, permission_level=2)


# ===== 分区策略 =====

class TestPartitionStrategy:
    """分区策略：只读并行 vs 写入串行"""

    def test_read_only_tools_grouped_parallel(self, orchestrator):
        """连续的只读+并发安全工具分入并行区"""
        calls = [
            {"name": "read_file", "arguments": {}, "id": "1"},
            {"name": "read_file", "arguments": {}, "id": "2"},
        ]
        partitions = orchestrator._partition(calls)
        assert len(partitions) == 1
        assert partitions[0]["parallel"] is True
        assert len(partitions[0]["calls"]) == 2

    def test_write_tools_are_serial(self, orchestrator):
        """写入工具分入串行区"""
        calls = [
            {"name": "write_file", "arguments": {}, "id": "1"},
            {"name": "write_file", "arguments": {}, "id": "2"},
        ]
        partitions = orchestrator._partition(calls)
        # 写入工具不可并发，每个单独串行
        for p in partitions:
            assert p["parallel"] is False

    def test_mixed_read_write_partitioned(self, orchestrator):
        """混合读写：读并行，写串行，正确分区"""
        calls = [
            {"name": "read_file", "arguments": {}, "id": "1"},
            {"name": "read_file", "arguments": {}, "id": "2"},
            {"name": "write_file", "arguments": {}, "id": "3"},
            {"name": "read_file", "arguments": {}, "id": "4"},
        ]
        partitions = orchestrator._partition(calls)
        # 期望：[parallel(read, read), serial(write), parallel(read)]
        assert len(partitions) == 3
        assert partitions[0]["parallel"] is True
        assert partitions[1]["parallel"] is False
        assert partitions[2]["parallel"] is True

    def test_unknown_tool_is_serial(self, orchestrator):
        """未知工具走串行"""
        calls = [
            {"name": "unknown_tool", "arguments": {}, "id": "1"},
        ]
        partitions = orchestrator._partition(calls)
        assert len(partitions) == 1
        assert partitions[0]["parallel"] is False

    def test_destructive_tool_is_serial(self, orchestrator):
        """破坏性工具走串行"""
        calls = [
            {"name": "delete_file", "arguments": {}, "id": "1"},
        ]
        partitions = orchestrator._partition(calls)
        assert partitions[0]["parallel"] is False

    def test_get_execution_plan(self, orchestrator):
        """get_execution_plan 返回调试信息"""
        calls = [
            {"name": "read_file", "arguments": {}, "id": "1"},
            {"name": "write_file", "arguments": {}, "id": "2"},
        ]
        plan = orchestrator.get_execution_plan(calls)
        assert len(plan) == 2
        assert plan[0]["parallel"] is True
        assert plan[0]["tool_names"] == ["read_file"]
        assert plan[1]["parallel"] is False
        assert plan[1]["tool_names"] == ["write_file"]


# ===== 执行 =====

class TestExecution:
    """并行/串行执行"""

    @pytest.mark.asyncio
    async def test_execute_batch_empty(self, orchestrator):
        """空调用列表返回空结果"""
        results = await orchestrator.execute_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_execute_batch_single(self, orchestrator):
        """单个工具直接执行"""
        calls = [{"name": "read_file", "arguments": {}, "id": "1"}]
        results = await orchestrator.execute_batch(calls)
        assert len(results) == 1
        _, result = results[0]
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_batch_parallel_read_only(self, orchestrator):
        """并行执行多个只读工具"""
        calls = [
            {"name": "read_file", "arguments": {}, "id": "1"},
            {"name": "read_file", "arguments": {}, "id": "2"},
            {"name": "read_file", "arguments": {}, "id": "3"},
        ]
        results = await orchestrator.execute_batch(calls)
        assert len(results) == 3
        for _, result in results:
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_batch_preserves_order(self, orchestrator):
        """结果保持原始调用顺序"""
        calls = [
            {"name": "read_file", "arguments": {}, "id": "1"},
            {"name": "write_file", "arguments": {}, "id": "2"},
        ]
        results = await orchestrator.execute_batch(calls)
        assert results[0][0]["id"] == "1"
        assert results[1][0]["id"] == "2"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, orchestrator):
        """未知工具返回失败结果"""
        calls = [{"name": "nonexistent", "arguments": {}, "id": "1"}]
        results = await orchestrator.execute_batch(calls)
        _, result = results[0]
        assert result.success is False
        assert "未知工具" in result.error

    @pytest.mark.asyncio
    async def test_execute_failing_tool_returns_error(self, orchestrator):
        """工具执行异常返回失败结果"""
        calls = [{"name": "fail_tool", "arguments": {}, "id": "1"}]
        results = await orchestrator.execute_batch(calls)
        _, result = results[0]
        assert result.success is False
        assert "执行出错" in result.error


# ===== 权限检查 =====

class TestPermissionCheck:
    """权限检查集成"""

    @pytest.mark.asyncio
    async def test_destructive_tool_denied_at_level_0(self, registry):
        """level=0 时破坏性工具被拒绝"""
        orch = ToolOrchestrator(tool_registry=registry, permission_level=0)
        calls = [{"name": "delete_file", "arguments": {}, "id": "1"}]
        results = await orch.execute_batch(calls)
        _, result = results[0]
        assert result.success is False
        assert "权限不足" in result.error

    @pytest.mark.asyncio
    async def test_destructive_tool_allowed_at_level_1(self, registry):
        """level=1 时破坏性工具允许"""
        orch = ToolOrchestrator(tool_registry=registry, permission_level=1)
        calls = [{"name": "delete_file", "arguments": {}, "id": "1"}]
        results = await orch.execute_batch(calls)
        _, result = results[0]
        assert result.success is True

    @pytest.mark.asyncio
    async def test_set_permission_level(self, registry):
        """动态设置权限级别"""
        orch = ToolOrchestrator(tool_registry=registry, permission_level=0)
        orch.set_permission_level(2)
        calls = [{"name": "delete_file", "arguments": {}, "id": "1"}]
        results = await orch.execute_batch(calls)
        _, result = results[0]
        assert result.success is True


# ===== Hook 集成 =====

class TestHookIntegration:
    """Hook 集成"""

    @pytest.mark.asyncio
    async def test_pre_hook_deny(self, registry):
        """PRE_TOOL_USE Hook 拒绝执行"""
        hook_mgr = HookManager()

        async def deny_all(context: HookContext) -> HookResult:
            return HookResult(decision=HookDecision.DENY, reason="全局禁止")

        hook_mgr.register(PythonHook("deny_all", HookEvent.PRE_TOOL_USE, deny_all))

        orch = ToolOrchestrator(
            tool_registry=registry, permission_level=2, hook_manager=hook_mgr
        )
        calls = [{"name": "read_file", "arguments": {}, "id": "1"}]
        results = await orch.execute_batch(calls)
        _, result = results[0]
        assert result.success is False
        assert "Hook denied" in result.error

    @pytest.mark.asyncio
    async def test_pre_hook_modify_params(self, registry):
        """PRE_TOOL_USE Hook 修改参数"""
        hook_mgr = HookManager()

        async def modify_params(context: HookContext) -> HookResult:
            context.params["injected"] = True
            return HookResult(
                decision=HookDecision.MODIFY,
                reason="注入参数",
                modified_params=context.params,
            )

        hook_mgr.register(PythonHook("modifier", HookEvent.PRE_TOOL_USE, modify_params))

        orch = ToolOrchestrator(
            tool_registry=registry, permission_level=2, hook_manager=hook_mgr
        )
        calls = [{"name": "read_file", "arguments": {}, "id": "1"}]
        results = await orch.execute_batch(calls)
        _, result = results[0]
        assert result.success is True  # Hook 只是修改，不阻止

    @pytest.mark.asyncio
    async def test_post_hook_runs_after_execution(self, registry):
        """POST_TOOL_USE Hook 在执行后运行"""
        hook_mgr = HookManager()
        post_called = []

        async def post_handler(context: HookContext) -> HookResult:
            post_called.append(context.tool_name)
            return HookResult(decision=HookDecision.PASSTHROUGH)

        hook_mgr.register(PythonHook("post_log", HookEvent.POST_TOOL_USE, post_handler))

        orch = ToolOrchestrator(
            tool_registry=registry, permission_level=2, hook_manager=hook_mgr
        )
        calls = [{"name": "read_file", "arguments": {}, "id": "1"}]
        await orch.execute_batch(calls)
        assert "read_file" in post_called

    @pytest.mark.asyncio
    async def test_no_hook_manager_skips_hooks(self, registry):
        """无 HookManager 时跳过 Hook"""
        orch = ToolOrchestrator(
            tool_registry=registry, permission_level=2, hook_manager=None
        )
        calls = [{"name": "read_file", "arguments": {}, "id": "1"}]
        results = await orch.execute_batch(calls)
        _, result = results[0]
        assert result.success is True


# ===== 增强权限管理器集成 =====

class TestEnhancedPermissionIntegration:
    """增强权限管理器集成"""

    @pytest.mark.asyncio
    async def test_enhanced_perm_manager_deny(self, registry):
        """增强权限管理器拒绝时返回失败"""
        from unittest.mock import MagicMock, AsyncMock

        mock_perm = MagicMock()
        # 删除 acheck_tool_params 使编排器回退到同步方法
        del mock_perm.acheck_tool_params
        mock_perm.check_tool_params.return_value = (False, "规则拒绝")

        orch = ToolOrchestrator(
            tool_registry=registry, permission_level=2,
            enhanced_perm_manager=mock_perm
        )
        calls = [{"name": "read_file", "arguments": {}, "id": "1"}]
        results = await orch.execute_batch(calls)
        _, result = results[0]
        assert result.success is False
        assert "规则拒绝" in result.error

    @pytest.mark.asyncio
    async def test_enhanced_perm_manager_allow(self, registry):
        """增强权限管理器允许时继续执行"""
        from unittest.mock import MagicMock

        mock_perm = MagicMock()
        del mock_perm.acheck_tool_params
        mock_perm.check_tool_params.return_value = (True, "")

        orch = ToolOrchestrator(
            tool_registry=registry, permission_level=2,
            enhanced_perm_manager=mock_perm
        )
        calls = [{"name": "read_file", "arguments": {}, "id": "1"}]
        results = await orch.execute_batch(calls)
        _, result = results[0]
        assert result.success is True

    @pytest.mark.asyncio
    async def test_enhanced_perm_manager_async_check(self, registry):
        """增强权限管理器异步检查方法"""
        from unittest.mock import MagicMock, AsyncMock

        mock_perm = MagicMock()
        mock_perm.acheck_tool_params = AsyncMock(return_value=(False, "异步拒绝"))

        orch = ToolOrchestrator(
            tool_registry=registry, permission_level=2,
            enhanced_perm_manager=mock_perm
        )
        calls = [{"name": "read_file", "arguments": {}, "id": "1"}]
        results = await orch.execute_batch(calls)
        _, result = results[0]
        assert result.success is False
        assert "异步拒绝" in result.error
