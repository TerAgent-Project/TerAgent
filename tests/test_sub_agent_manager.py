# tests/test_sub_agent_manager.py
"""子 Agent 管理器单元测试

测试 teragent.coordination.sub_agent_manager 模块:
  - 三种执行模式 (SYNC / ASYNC / FORK)
  - 工具白名单
  - 步数预算控制
  - 并发限制
  - spawn/cancel 生命周期
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from teragent.coordination.message_bus import AgentMessageBus
from teragent.coordination.sub_agent_manager import (
    AgentMode,
    SubAgentInfo,
    SubAgentManager,
    SubAgentStatus,
    _detect_compiler_type,
)
from teragent.event_bus import EventBus

# ===== 辅助 fixture =====

def _make_mock_tool_registry(tool_names=None):
    """创建 mock ToolRegistry"""
    registry = MagicMock()
    if tool_names is None:
        tool_names = ["read_file", "write_file"]
    registry.list_tool_names.return_value = tool_names
    registry.get.return_value = None
    return registry


def _make_mock_model_provider(responses=None):
    """创建 mock ModelProvider"""
    provider = AsyncMock()
    if responses is None:
        # 默认：无工具调用，直接返回文本
        responses = [{"content": "任务完成", "tool_calls": []}]
    provider.chat = AsyncMock(side_effect=responses)
    provider._tracer = None
    provider.compiler = None
    provider.model_name = "test-model"
    return provider


def _make_manager(model=None, tool_registry=None):
    """创建 SubAgentManager 实例"""
    bus = EventBus()
    mbus = AgentMessageBus(bus)
    if model is None:
        model = _make_mock_model_provider()
    if tool_registry is None:
        tool_registry = _make_mock_tool_registry()
    return SubAgentManager(bus, model, tool_registry, mbus)


# ===== 测试类 =====


class TestAgentModeAndStatus:
    """AgentMode 和 SubAgentStatus 枚举测试"""

    def test_agent_mode_values(self):
        """AgentMode 有 3 种模式"""
        assert AgentMode.SYNC.value == "sync"
        assert AgentMode.ASYNC.value == "async"
        assert AgentMode.FORK.value == "fork"
        assert len(AgentMode) == 3

    def test_sub_agent_status_values(self):
        """SubAgentStatus 有 6 种状态"""
        assert SubAgentStatus.PENDING.value == "pending"
        assert SubAgentStatus.RUNNING.value == "running"
        assert SubAgentStatus.COMPLETED.value == "completed"
        assert SubAgentStatus.FAILED.value == "failed"
        assert SubAgentStatus.STOPPED.value == "stopped"
        assert SubAgentStatus.BUDGET_EXHAUSTED.value == "budget_exhausted"


class TestSubAgentInfo:
    """SubAgentInfo 数据类测试"""

    def test_auto_fill_created_at(self):
        """自动填充创建时间"""
        info = SubAgentInfo(
            agent_id="sub_1", parent_id="main",
            mode=AgentMode.SYNC, status=SubAgentStatus.PENDING,
            task="test",
        )
        assert info.created_at > 0

    def test_to_dict(self):
        """to_dict 包含所有字段"""
        info = SubAgentInfo(
            agent_id="sub_1", parent_id="main",
            mode=AgentMode.SYNC, status=SubAgentStatus.COMPLETED,
            task="test", result="done",
        )
        d = info.to_dict()
        assert d["agent_id"] == "sub_1"
        assert d["mode"] == "sync"
        assert d["status"] == "completed"
        assert d["result"] == "done"


class TestDetectCompilerType:
    """_detect_compiler_type 测试"""

    def test_glm_compiler(self):
        """GLM 编译器类型"""
        from teragent.core.compilers.glm import GLMCompiler
        provider = MagicMock()
        provider.compiler = GLMCompiler()
        assert _detect_compiler_type(provider) == "glm"

    def test_default_by_model_name(self):
        """通过模型名称推断默认类型"""
        provider = MagicMock()
        provider.compiler = None
        provider.model_name = "some-random-model"
        assert _detect_compiler_type(provider) == "default"


class TestSyncMode:
    """SYNC 模式测试"""

    @pytest.mark.asyncio
    async def test_sync_returns_result(self):
        """SYNC 模式返回子 Agent 执行结果"""
        manager = _make_manager()
        result = await manager.spawn("简单任务", mode=AgentMode.SYNC)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_sync_agent_completed(self):
        """SYNC 模式完成后状态为 COMPLETED"""
        manager = _make_manager()
        await manager.spawn("简单任务", mode=AgentMode.SYNC)
        # 查找最近完成的子 Agent
        agents = manager.list_active_agents()
        assert len(agents) >= 1
        # 最后一个 agent 应该是刚创建的
        latest = agents[-1]
        assert latest["status"] == "completed"


class TestAsyncMode:
    """ASYNC 模式测试"""

    @pytest.mark.asyncio
    async def test_async_returns_agent_id(self):
        """ASYNC 模式立即返回 agent_id"""
        manager = _make_manager()
        result = await manager.spawn("后台任务", mode=AgentMode.ASYNC)
        assert result.startswith("sub_agent_")

    @pytest.mark.asyncio
    async def test_async_task_completes(self):
        """ASYNC 模式后台任务最终完成"""
        manager = _make_manager()
        agent_id = await manager.spawn("后台任务", mode=AgentMode.ASYNC)
        # 等待异步任务完成
        task = manager._async_tasks.get(agent_id)
        if task:
            await asyncio.wait_for(task, timeout=2.0)
        status = manager.get_status(agent_id)
        assert status is not None


class TestForkMode:
    """FORK 模式测试"""

    @pytest.mark.asyncio
    async def test_fork_returns_result(self):
        """FORK 模式返回执行结果"""
        manager = _make_manager()
        result = await manager.spawn("FORK 任务", mode=AgentMode.FORK)
        assert isinstance(result, str)


class TestToolWhitelist:
    """工具白名单测试"""

    @pytest.mark.asyncio
    async def test_custom_tool_whitelist(self):
        """自定义工具白名单"""
        manager = _make_manager()
        _result = await manager.spawn(
            "受限任务",
            mode=AgentMode.SYNC,
            allowed_tools=["read_file"],
        )
        agent_id = f"sub_agent_{manager._agent_counter}"
        status = manager.get_status(agent_id)
        assert "read_file" in status["allowed_tools"]

    @pytest.mark.asyncio
    async def test_default_tool_whitelist(self):
        """默认使用注册表中所有工具"""
        registry = _make_mock_tool_registry(["read_file", "write_file", "execute_subtask"])
        manager = _make_manager(tool_registry=registry)
        _result = await manager.spawn("任务", mode=AgentMode.SYNC)
        agent_id = f"sub_agent_{manager._agent_counter}"
        status = manager.get_status(agent_id)
        assert len(status["allowed_tools"]) == 3


class TestConcurrencyLimit:
    """并发限制测试"""

    @pytest.mark.asyncio
    async def test_exceed_concurrent_limit(self):
        """超过最大并发数抛出 RuntimeError"""
        manager = _make_manager()
        # 设置一个很小的并发限制
        manager.MAX_CONCURRENT_SUB_AGENTS = 1

        # 启动一个长时间运行的子 Agent（ASYNC 模式）
        # 需要让模型不返回，保持 running 状态
        slow_provider = _make_mock_model_provider()
        # 模拟长时间运行
        async def slow_chat(**kwargs):
            await asyncio.sleep(5.0)
            return {"content": "done", "tool_calls": []}
        slow_provider.chat = slow_chat

        manager._model_provider = slow_provider
        await manager.spawn("长时间任务", mode=AgentMode.ASYNC)

        # 尝试启动第二个
        with pytest.raises(RuntimeError, match="并发子 Agent 数已达上限"):
            await manager.spawn("第二个任务", mode=AgentMode.SYNC)


class TestStepBudget:
    """步数预算测试"""

    @pytest.mark.asyncio
    async def test_budget_exhausted(self):
        """步数预算耗尽时返回相应提示"""
        manager = _make_manager()
        manager.MAX_SUB_AGENT_STEPS = 2

        # 模拟连续返回工具调用
        tool_call = {
            "id": "tc_1",
            "function": {"name": "read_file", "arguments": {"path": "/tmp/test"}},
        }
        responses = [
            {"content": "执行工具", "tool_calls": [tool_call]},
            {"content": "再执行", "tool_calls": [tool_call]},
            # 第三次模型返回无工具调用（但步数已耗尽，不会再调用模型）
        ]

        # 让工具注册表返回一个 mock 工具
        mock_tool = AsyncMock()
        mock_tool_result = MagicMock()
        mock_tool_result.success = True
        mock_tool_result.data = "file content"
        mock_tool.execute = AsyncMock(return_value=mock_tool_result)

        registry = _make_mock_tool_registry(["read_file"])
        registry.get.return_value = mock_tool

        model = _make_mock_model_provider(responses=responses)
        manager._model_provider = model
        manager._tool_registry = registry

        result = await manager.spawn("循环任务", mode=AgentMode.SYNC)
        # 步数预算耗尽
        assert "预算耗尽" in result or "完成" in result


class TestSpawnCancelLifecycle:
    """spawn/cancel 生命周期测试"""

    @pytest.mark.asyncio
    async def test_stop_agent(self):
        """停止子 Agent"""
        manager = _make_manager()
        slow_provider = _make_mock_model_provider()
        async def slow_chat(**kwargs):
            await asyncio.sleep(10.0)
            return {"content": "done", "tool_calls": []}
        slow_provider.chat = slow_chat
        manager._model_provider = slow_provider

        agent_id = await manager.spawn("长时间任务", mode=AgentMode.ASYNC)
        await manager.stop(agent_id)

        status = manager.get_status(agent_id)
        assert status["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_stop_all_agents(self):
        """停止所有子 Agent"""
        manager = _make_manager()
        slow_provider = _make_mock_model_provider()
        async def slow_chat(**kwargs):
            await asyncio.sleep(10.0)
            return {"content": "done", "tool_calls": []}
        slow_provider.chat = slow_chat
        manager._model_provider = slow_provider

        a1 = await manager.spawn("任务1", mode=AgentMode.ASYNC)
        a2 = await manager.spawn("任务2", mode=AgentMode.ASYNC)
        await manager.stop_all()

        for aid in [a1, a2]:
            status = manager.get_status(aid)
            assert status["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_list_active_agents(self):
        """列出活跃子 Agent"""
        manager = _make_manager()
        await manager.spawn("任务1", mode=AgentMode.SYNC)
        agents = manager.list_active_agents()
        assert len(agents) >= 1

    @pytest.mark.asyncio
    async def test_status_report(self):
        """状态报告结构正确"""
        manager = _make_manager()
        report = manager.get_status_report()
        assert "total_agents" in report
        assert "by_status" in report
        assert "agents" in report
