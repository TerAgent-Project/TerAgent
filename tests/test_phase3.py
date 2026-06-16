"""Phase 3 集成测试 — OrchestrationCheckpoint、ApprovalGate、OrchestratorTool、
ToolHubClient、AuthScheme/AuthCredential/AuthManager、AsyncRWLock、ResultCache

测试覆盖:
1. OrchestrationCheckpoint: 保存/恢复/列表/删除/清理/不存在检查点
2. ApprovalGate: 审批/拒绝/修改参数/超时/检查/清理/待审批列表
3. OrchestratorTool / Nested Orchestration: 创建/执行/错误处理/属性/as_tool
4. ToolHubClient: 搜索/安装/发布/已安装列表/网络错误/上下文管理器
5. AuthScheme / AuthCredential / AuthManager: 注册/查询/应用/环境变量/遮蔽
6. AsyncRWLock: 多读者/写者独占/写者优先/SharedState 集成
7. ResultCache: set/get/TTL/LRU/统计/失效
8. Boundary: 0 agents/1 agent/空 SharedState/无待审批
9. Error Recovery: AgentTool 失败/OrchestratorTool 内部失败/无效 AuthScheme
10. Concurrent Safety: SharedState 并发写入/并发审批请求
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from teragent.orchestration import (
    Agent,
    AgentHooks,
    CancellationToken,
    Handoff,
    Orchestrator,
    OrchestrationMode,
    SharedState,
    RunContext,
    UsageTracker,
    OrchestrationResult,
    AsyncRWLock,
    ApprovalGate,
    ApprovalResult,
    OrchestrationCheckpoint,
    OrchestratorTool,
)
from teragent.orchestration.checkpoint import OrchestrationCheckpoint as CheckpointDirect
from teragent.orchestration.approval import ApprovalResult as ApprovalResultDirect
from teragent.orchestration.rwlock import ReadLockContext, WriteLockContext
from teragent.orchestration.shared_state import ScopedState, StateWrite
from teragent.tools.base import BaseTool, ToolResult
from teragent.tools.auth import AuthScheme, AuthCredential, AuthManager
from teragent.tools.result_cache import ResultCache, CacheEntry, CacheStats
from teragent.tools.orchestrator_tool import OrchestratorTool as OrchestratorToolDirect
from teragent.tools.agent_tool import AgentTool
from teragent.tools.hub.client import ToolHubClient, ToolHubEntry, ToolHubError, HubTool
from teragent.core.types import ToolSafety
from teragent.core.tap import TAPRequest, TAPResponse, CompiledPrompt
from teragent.core.adapter import TAPAdapter
from teragent.core.compiler import TAPCompiler
from teragent.core.provider import ModelProvider


# ===== Mock Provider (same as test_phase2) =====

class MockCompiler(TAPCompiler):
    def compile(self, request: TAPRequest) -> CompiledPrompt:
        return CompiledPrompt(
            messages=[{"role": "user", "content": request.instruction}],
            max_tokens=1024,
        )


class MockAdapter(TAPAdapter):
    def __init__(self, response_text: str = "Mock response"):
        self._response_text = response_text
        self._call_count = 0

    @property
    def capabilities(self) -> dict:
        return {"streaming": False, "tool_calling": True}

    @property
    def required_mode(self) -> str:
        return "any"

    async def send(self, compiled: CompiledPrompt, model: str) -> TAPResponse:
        self._call_count += 1
        return TAPResponse(
            raw_text=self._response_text,
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            finish_reason="stop",
        )

    async def stream(self, compiled: CompiledPrompt, model: str):
        yield self._response_text


def create_mock_provider(response_text: str = "Mock response") -> ModelProvider:
    compiler = MockCompiler()
    adapter = MockAdapter(response_text)
    return ModelProvider(compiler=compiler, adapter=adapter, model="mock-model")


# ===== Test tool with needs_approval =====

class ApprovalRequiredTool(BaseTool):
    """A tool that requires approval for testing"""
    name = "delete_file"
    description = "Delete a file"
    parameters_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    needs_approval = True
    _safety = ToolSafety.DESTRUCTIVE
    _concurrency_safe = False

    async def execute(self, params, progress_callback=None):
        return ToolResult(success=True, data={"deleted": params.get("path", "")})


class NormalTool(BaseTool):
    """A tool that does not require approval"""
    name = "read_file"
    description = "Read a file"
    parameters_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    needs_approval = False
    _safety = ToolSafety.READ_ONLY
    _concurrency_safe = True

    async def execute(self, params, progress_callback=None):
        return ToolResult(success=True, data={"content": "file data"})


# ======================================================================
# 1. OrchestrationCheckpoint Tests
# ======================================================================

class TestOrchestrationCheckpoint(unittest.TestCase):
    """OrchestrationCheckpoint — 编排检查点保存/恢复/管理"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.checkpoint = OrchestrationCheckpoint(base_dir=self.tmpdir)

    def test_save_and_restore(self):
        """保存编排状态到文件，然后恢复它"""
        async def _test():
            provider = create_mock_provider("checkpoint test")
            agent_a = Agent(name="agent_a", provider=provider, output_key="a_output")
            orchestrator = Orchestrator(
                agents=[agent_a],
                mode=OrchestrationMode.SEQUENTIAL,
            )
            # Set some shared state
            orchestrator._shared_state.set("key1", "value1")
            orchestrator._shared_state.set("key2", 42)

            cp_id = await self.checkpoint.save(
                orchestrator,
                run_id="run_001",
                current_agent="agent_a",
                turn=3,
                messages=[{"role": "user", "content": "hello"}],
            )
            self.assertIsInstance(cp_id, str)
            self.assertTrue(len(cp_id) > 0)

            # Restore
            state = await self.checkpoint.restore("run_001", checkpoint_id=cp_id)
            self.assertIsInstance(state, dict)
            self.assertEqual(state["current_agent"], "agent_a")
            self.assertEqual(state["turn"], 3)
            self.assertEqual(len(state["messages"]), 1)
            self.assertEqual(state["messages"][0]["content"], "hello")
            self.assertEqual(state["mode"], "sequential")
            self.assertIn("shared_state", state)
            self.assertIn("agent_metadata", state)
            self.assertEqual(state["agent_metadata"][0]["name"], "agent_a")
            self.assertEqual(state["checkpoint_id"], cp_id)
            self.assertIn("timestamp", state)

        asyncio.run(_test())

    def test_restore_latest_without_id(self):
        """不指定 checkpoint_id 时恢复最新的检查点"""
        async def _test():
            provider = create_mock_provider("test")
            agent_a = Agent(name="agent_a", provider=provider)
            orchestrator = Orchestrator(agents=[agent_a], mode=OrchestrationMode.SEQUENTIAL)

            # Save two checkpoints
            await self.checkpoint.save(orchestrator, run_id="run_002", turn=1)
            import asyncio as _asyncio
            await _asyncio.sleep(0.01)  # ensure different timestamps
            cp2 = await self.checkpoint.save(orchestrator, run_id="run_002", turn=5)

            # Restore latest (no checkpoint_id specified)
            state = await self.checkpoint.restore("run_002")
            self.assertEqual(state["turn"], 5)
            self.assertEqual(state["checkpoint_id"], cp2)

        asyncio.run(_test())

    def test_list_checkpoints(self):
        """列出指定运行的所有检查点"""
        async def _test():
            provider = create_mock_provider("test")
            agent_a = Agent(name="agent_a", provider=provider)
            orchestrator = Orchestrator(agents=[agent_a], mode=OrchestrationMode.SEQUENTIAL)

            await self.checkpoint.save(orchestrator, run_id="run_003", turn=1)
            await self.checkpoint.save(orchestrator, run_id="run_003", turn=2)

            summaries = await self.checkpoint.list_checkpoints("run_003")
            self.assertEqual(len(summaries), 2)
            for s in summaries:
                self.assertIn("checkpoint_id", s)
                self.assertIn("timestamp", s)
                self.assertIn("turn", s)
                self.assertIn("mode", s)

        asyncio.run(_test())

    def test_delete_checkpoint(self):
        """删除指定检查点"""
        async def _test():
            provider = create_mock_provider("test")
            agent_a = Agent(name="agent_a", provider=provider)
            orchestrator = Orchestrator(agents=[agent_a], mode=OrchestrationMode.SEQUENTIAL)

            cp_id = await self.checkpoint.save(orchestrator, run_id="run_004", turn=1)
            result = await self.checkpoint.delete_checkpoint("run_004", cp_id)
            self.assertTrue(result)

            # Verify it's gone
            summaries = await self.checkpoint.list_checkpoints("run_004")
            self.assertEqual(len(summaries), 0)

        asyncio.run(_test())

    def test_delete_nonexistent_checkpoint(self):
        """删除不存在的检查点返回 False"""
        async def _test():
            result = await self.checkpoint.delete_checkpoint("run_999", "nonexistent_id")
            self.assertFalse(result)

        asyncio.run(_test())

    def test_cleanup_old_checkpoints(self):
        """清理旧检查点，仅保留 keep_last 个"""
        async def _test():
            provider = create_mock_provider("test")
            agent_a = Agent(name="agent_a", provider=provider)
            orchestrator = Orchestrator(agents=[agent_a], mode=OrchestrationMode.SEQUENTIAL)

            # Save 5 checkpoints
            for i in range(5):
                await self.checkpoint.save(orchestrator, run_id="run_005", turn=i)
                await asyncio.sleep(0.01)

            # Cleanup, keep only last 2
            deleted = await self.checkpoint.cleanup("run_005", keep_last=2)
            self.assertEqual(deleted, 3)

            summaries = await self.checkpoint.list_checkpoints("run_005")
            self.assertEqual(len(summaries), 2)

        asyncio.run(_test())

    def test_restore_nonexistent_checkpoint(self):
        """从不存在的检查点恢复抛出 FileNotFoundError"""
        async def _test():
            with self.assertRaises(FileNotFoundError):
                await self.checkpoint.restore("nonexistent_run")

        asyncio.run(_test())

    def test_shared_state_round_trip_fidelity(self):
        """SharedState 快照-恢复的完整保真度"""
        async def _test():
            provider = create_mock_provider("test")
            agent_a = Agent(name="agent_a", provider=provider)
            orchestrator = Orchestrator(agents=[agent_a], mode=OrchestrationMode.SEQUENTIAL)

            # Populate shared state
            orchestrator._shared_state.set("string_key", "hello")
            orchestrator._shared_state.set("int_key", 123)
            orchestrator._shared_state.set("list_key", [1, 2, 3])
            orchestrator._shared_state.set("dict_key", {"nested": True})
            orchestrator._shared_state.set("agent_key", "scoped_val", scope="agent", agent_name="agent_a")

            cp_id = await self.checkpoint.save(orchestrator, run_id="run_fidelity", turn=0)

            state = await self.checkpoint.restore("run_fidelity", checkpoint_id=cp_id)

            # Restore into a new SharedState
            new_ss = SharedState()
            new_ss.restore(state["shared_state"])

            self.assertEqual(new_ss.get("string_key"), "hello")
            self.assertEqual(new_ss.get("int_key"), 123)
            self.assertEqual(new_ss.get("list_key"), [1, 2, 3])
            self.assertEqual(new_ss.get("dict_key"), {"nested": True})
            self.assertEqual(new_ss.get("agent_key", scope="agent"), "scoped_val")

        asyncio.run(_test())


# ======================================================================
# 2. ApprovalGate Tests
# ======================================================================

class TestApprovalGate(unittest.TestCase):
    """ApprovalGate — 工具审批门测试"""

    def test_approve_request(self):
        """请求审批，批准后返回 ApprovalResult(approved=True)"""
        async def _test():
            gate = ApprovalGate(default_timeout=5.0)

            async def _request():
                return await gate.request_approval("delete_file", {"path": "/tmp/test"})

            async def _approve():
                await asyncio.sleep(0.05)
                pending = gate.get_pending_approvals()
                self.assertEqual(len(pending), 1)
                self.assertEqual(pending[0]["tool_name"], "delete_file")
                gate.approve(pending[0]["approval_id"])

            result, _ = await asyncio.gather(_request(), _approve())
            self.assertTrue(result.approved)
            self.assertEqual(result.reason, "")

        asyncio.run(_test())

    def test_reject_request(self):
        """请求审批，拒绝后返回 ApprovalResult(approved=False)"""
        async def _test():
            gate = ApprovalGate(default_timeout=5.0)

            async def _request():
                return await gate.request_approval("delete_file", {"path": "/important"})

            async def _reject():
                await asyncio.sleep(0.05)
                pending = gate.get_pending_approvals()
                gate.reject(pending[0]["approval_id"], reason="Not allowed")

            result, _ = await asyncio.gather(_request(), _reject())
            self.assertFalse(result.approved)
            self.assertEqual(result.reason, "Not allowed")

        asyncio.run(_test())

    def test_approve_with_modified_params(self):
        """审批通过并修改参数"""
        async def _test():
            gate = ApprovalGate(default_timeout=5.0)

            async def _request():
                return await gate.request_approval("delete_file", {"path": "/tmp/important"})

            async def _approve():
                await asyncio.sleep(0.05)
                pending = gate.get_pending_approvals()
                gate.approve(
                    pending[0]["approval_id"],
                    modified_params={"path": "/tmp/safe_backup"},
                )

            result, _ = await asyncio.gather(_request(), _approve())
            self.assertTrue(result.approved)
            self.assertIsNotNone(result.modified_params)
            self.assertEqual(result.modified_params["path"], "/tmp/safe_backup")

        asyncio.run(_test())

    def test_timeout_on_approval(self):
        """审批请求超时"""
        async def _test():
            gate = ApprovalGate(default_timeout=0.1)
            result = await gate.request_approval("slow_tool", {"x": 1}, timeout=0.1)
            self.assertFalse(result.approved)
            self.assertIn("timeout", result.reason.lower())

        asyncio.run(_test())

    def test_check_needs_approval(self):
        """检查工具是否需要审批"""
        tool_needs = ApprovalRequiredTool()
        tool_normal = NormalTool()
        self.assertTrue(ApprovalGate.check_needs_approval(tool_needs))
        self.assertFalse(ApprovalGate.check_needs_approval(tool_normal))

    def test_check_needs_approval_missing_attr(self):
        """没有 needs_approval 属性的工具默认不需要审批"""
        mock_tool = MagicMock(spec=BaseTool)
        # MagicMock by default returns a MagicMock for any attribute access,
        # but getattr with default should return False
        del mock_tool.needs_approval
        self.assertFalse(ApprovalGate.check_needs_approval(mock_tool))

    def test_clear_pending_approvals(self):
        """清除所有待审批条目"""
        async def _test():
            gate = ApprovalGate(default_timeout=0.5)

            # Create a pending approval that will be blocked
            request_task = asyncio.create_task(
                gate.request_approval("tool_a", {"x": 1}, timeout=0.5)
            )
            await asyncio.sleep(0.05)

            self.assertEqual(len(gate.get_pending_approvals()), 1)

            await gate.clear()

            # After clear, the request should resolve as rejected
            result = await request_task
            self.assertFalse(result.approved)
            self.assertEqual(result.reason, "ApprovalGate cleared")

        asyncio.run(_test())

    def test_get_pending_approvals_empty(self):
        """没有待审批时返回空列表"""
        gate = ApprovalGate()
        self.assertEqual(gate.get_pending_approvals(), [])

    def test_approval_result_dataclass(self):
        """ApprovalResult 数据类的基本属性"""
        r1 = ApprovalResult(approved=True)
        self.assertTrue(r1.approved)
        self.assertEqual(r1.reason, "")
        self.assertIsNone(r1.modified_params)

        r2 = ApprovalResult(approved=False, reason="denied", modified_params=None)
        self.assertFalse(r2.approved)
        self.assertEqual(r2.reason, "denied")

    def test_approve_unknown_id(self):
        """批准不存在的审批 ID 不抛异常"""
        gate = ApprovalGate()
        # Should not raise
        gate.approve("nonexistent_id")

    def test_reject_unknown_id(self):
        """拒绝不存在的审批 ID 不抛异常"""
        gate = ApprovalGate()
        gate.reject("nonexistent_id", reason="test")

    def test_repr(self):
        """ApprovalGate __repr__"""
        gate = ApprovalGate(default_timeout=60.0)
        r = repr(gate)
        self.assertIn("ApprovalGate", r)
        self.assertIn("60", r)


# ======================================================================
# 3. OrchestratorTool / Nested Orchestration Tests
# ======================================================================

class TestOrchestratorTool(unittest.TestCase):
    """OrchestratorTool — 嵌套编排工具"""

    def test_create_orchestrator_tool(self):
        """从 Orchestrator 创建 OrchestratorTool"""
        provider = create_mock_provider()
        agent_a = Agent(name="researcher", provider=provider)
        agent_b = Agent(name="writer", provider=provider)
        inner = Orchestrator(
            agents=[agent_a, agent_b],
            mode=OrchestrationMode.SEQUENTIAL,
        )
        tool = OrchestratorTool(orchestrator=inner)

        self.assertIn("sequential", tool.name)
        self.assertIsInstance(tool.description, str)
        self.assertIsInstance(tool.parameters_schema, dict)

    def test_tool_name_and_description_override(self):
        """自定义工具名和描述"""
        provider = create_mock_provider()
        agent_a = Agent(name="worker", provider=provider)
        inner = Orchestrator(agents=[agent_a], mode=OrchestrationMode.SEQUENTIAL)
        tool = OrchestratorTool(
            orchestrator=inner,
            tool_name="my_custom_tool",
            tool_description="Custom description",
        )
        self.assertEqual(tool.name, "my_custom_tool")
        self.assertEqual(tool.description, "Custom description")

    def test_tool_schema(self):
        """验证工具参数 Schema"""
        provider = create_mock_provider()
        agent_a = Agent(name="worker", provider=provider)
        inner = Orchestrator(agents=[agent_a], mode=OrchestrationMode.SEQUENTIAL)
        tool = OrchestratorTool(orchestrator=inner)

        schema = tool.parameters_schema
        self.assertIn("task", schema.get("required", []))
        self.assertIn("properties", schema)
        self.assertIn("task", schema["properties"])

    def test_execute_nested_orchestration(self):
        """执行嵌套编排（mock inner orchestrator.run）"""
        async def _test():
            provider = create_mock_provider("nested result")
            agent_a = Agent(name="inner_agent", provider=provider)
            inner = Orchestrator(agents=[agent_a], mode=OrchestrationMode.SEQUENTIAL)
            tool = OrchestratorTool(orchestrator=inner)

            # Mock inner.run to avoid actual orchestration
            mock_result = OrchestrationResult(
                final_output="nested output",
                agent_outputs={"inner_agent": "done"},
                total_turns=1,
                last_agent="inner_agent",
                total_prompt_tokens=10,
                total_completion_tokens=20,
                metadata={},
            )
            inner.run = AsyncMock(return_value=mock_result)

            result = await tool.execute({"task": "Do something"})
            self.assertTrue(result.success)
            self.assertIn("output", result.data)
            self.assertEqual(result.data["output"], "nested output")
            self.assertIn("orchestrator_mode", result.metadata)

        asyncio.run(_test())

    def test_handle_inner_error(self):
        """内部编排器失败时优雅处理"""
        async def _test():
            provider = create_mock_provider()
            agent_a = Agent(name="failing_agent", provider=provider)
            inner = Orchestrator(agents=[agent_a], mode=OrchestrationMode.SEQUENTIAL)
            tool = OrchestratorTool(orchestrator=inner)

            # Mock inner.run to raise exception
            inner.run = AsyncMock(side_effect=RuntimeError("Inner orchestration crashed"))

            result = await tool.execute({"task": "Fail this"})
            self.assertFalse(result.success)
            self.assertIn("failed", result.error.lower())

        asyncio.run(_test())

    def test_handle_inner_cancelled(self):
        """内部编排器被取消时处理"""
        async def _test():
            provider = create_mock_provider()
            agent_a = Agent(name="agent", provider=provider)
            inner = Orchestrator(agents=[agent_a], mode=OrchestrationMode.SEQUENTIAL)
            tool = OrchestratorTool(orchestrator=inner)

            mock_result = OrchestrationResult(
                final_output="",
                agent_outputs={},
                total_turns=0,
                last_agent="",
                total_prompt_tokens=0,
                total_completion_tokens=0,
                metadata={"cancelled": True},
            )
            inner.run = AsyncMock(return_value=mock_result)

            result = await tool.execute({"task": "cancelled task"})
            self.assertFalse(result.success)
            self.assertIn("cancelled", result.error.lower())

        asyncio.run(_test())

    def test_handle_inner_timeout(self):
        """内部编排器超时时处理"""
        async def _test():
            provider = create_mock_provider()
            agent_a = Agent(name="agent", provider=provider)
            inner = Orchestrator(agents=[agent_a], mode=OrchestrationMode.SEQUENTIAL)
            tool = OrchestratorTool(orchestrator=inner)

            mock_result = OrchestrationResult(
                final_output="",
                agent_outputs={},
                total_turns=0,
                last_agent="",
                total_prompt_tokens=0,
                total_completion_tokens=0,
                metadata={"timeout": True, "timeout_seconds": 30},
            )
            inner.run = AsyncMock(return_value=mock_result)

            result = await tool.execute({"task": "timed out task"})
            self.assertFalse(result.success)
            self.assertIn("timed out", result.error.lower())

        asyncio.run(_test())

    def test_handle_inner_error_metadata(self):
        """内部编排器返回错误元数据时处理"""
        async def _test():
            provider = create_mock_provider()
            agent_a = Agent(name="agent", provider=provider)
            inner = Orchestrator(agents=[agent_a], mode=OrchestrationMode.SEQUENTIAL)
            tool = OrchestratorTool(orchestrator=inner)

            mock_result = OrchestrationResult(
                final_output="",
                agent_outputs={},
                total_turns=0,
                last_agent="",
                total_prompt_tokens=0,
                total_completion_tokens=0,
                metadata={"error": "Something went wrong", "error_type": "RuntimeError"},
            )
            inner.run = AsyncMock(return_value=mock_result)

            result = await tool.execute({"task": "error task"})
            self.assertFalse(result.success)
            self.assertIn("failed", result.error.lower())

        asyncio.run(_test())

    def test_orchestrator_as_tool_convenience(self):
        """Orchestrator.as_tool() 便捷方法"""
        provider = create_mock_provider()
        agent_a = Agent(name="researcher", provider=provider)
        inner = Orchestrator(agents=[agent_a], mode=OrchestrationMode.SEQUENTIAL)

        tool = inner.as_tool()
        self.assertIsInstance(tool, OrchestratorTool)

        # With overrides
        tool2 = inner.as_tool(tool_name="custom", tool_description="Custom desc")
        self.assertEqual(tool2.name, "custom")
        self.assertEqual(tool2.description, "Custom desc")

    def test_validate_input(self):
        """OrchestratorTool 输入验证"""
        provider = create_mock_provider()
        agent_a = Agent(name="worker", provider=provider)
        inner = Orchestrator(agents=[agent_a], mode=OrchestrationMode.SEQUENTIAL)
        tool = OrchestratorTool(orchestrator=inner)

        # Empty task should fail
        errors = tool.validate_input({"task": ""})
        self.assertTrue(len(errors) > 0)

        # Valid task should pass
        errors = tool.validate_input({"task": "do something"})
        self.assertEqual(len(errors), 0)

    def test_describe_usage(self):
        """OrchestratorTool 描述用法"""
        provider = create_mock_provider()
        agent_a = Agent(name="worker", provider=provider)
        inner = Orchestrator(agents=[agent_a], mode=OrchestrationMode.SEQUENTIAL)
        tool = OrchestratorTool(orchestrator=inner)

        desc = tool.describe_usage({"task": "research AI safety"})
        self.assertIn("sequential", desc)
        self.assertIn("research AI safety", desc)

    def test_repr(self):
        """OrchestratorTool __repr__"""
        provider = create_mock_provider()
        agent_a = Agent(name="worker", provider=provider)
        inner = Orchestrator(agents=[agent_a], mode=OrchestrationMode.SEQUENTIAL)
        tool = OrchestratorTool(orchestrator=inner)

        r = repr(tool)
        self.assertIn("OrchestratorTool", r)
        self.assertIn("worker", r)

    def test_execute_with_context_param(self):
        """执行嵌套编排时传递 context 参数"""
        async def _test():
            provider = create_mock_provider("nested with context")
            agent_a = Agent(name="agent", provider=provider)
            inner = Orchestrator(agents=[agent_a], mode=OrchestrationMode.SEQUENTIAL)
            tool = OrchestratorTool(orchestrator=inner)

            mock_result = OrchestrationResult(
                final_output="result with context",
                agent_outputs={"agent": "done"},
                total_turns=1,
                last_agent="agent",
                total_prompt_tokens=10,
                total_completion_tokens=20,
                metadata={},
            )
            inner.run = AsyncMock(return_value=mock_result)

            result = await tool.execute({"task": "Do something", "context": "Extra info"})
            self.assertTrue(result.success)
            # Verify the task instruction included context
            call_args = inner.run.call_args
            task_arg = call_args.kwargs.get("task") or call_args[0][0]
            self.assertIn("Extra info", task_arg)

        asyncio.run(_test())


# ======================================================================
# 4. ToolHubClient Tests
# ======================================================================

class TestToolHubClient(unittest.TestCase):
    """ToolHubClient — 工具市场客户端"""

    def test_search_tools(self):
        """搜索工具（mock httpx）"""
        async def _test():
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "tools": [
                    {
                        "name": "postgres_query",
                        "version": "1.2.0",
                        "author": "teragent",
                        "description": "Query PostgreSQL",
                        "category": "database",
                        "downloads": 100,
                        "rating": 4.5,
                    }
                ]
            }

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.is_closed = False
            mock_client.aclose = AsyncMock()

            with patch("httpx.AsyncClient", return_value=mock_client):
                async with ToolHubClient() as client:
                    entries = await client.search("database")

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].name, "postgres_query")
            self.assertEqual(entries[0].version, "1.2.0")

        asyncio.run(_test())

    def test_install_tool(self):
        """安装工具（mock httpx）"""
        async def _test():
            install_response = MagicMock()
            install_response.status_code = 200
            install_response.json.return_value = {
                "name": "redis_get",
                "description": "Get from Redis",
                "parameters_schema": {"type": "object", "properties": {}},
                "safety": "read_only",
                "version": "2.0.0",
                "category": "cache",
            }

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=install_response)
            mock_client.is_closed = False
            mock_client.aclose = AsyncMock()

            with patch("httpx.AsyncClient", return_value=mock_client):
                async with ToolHubClient() as client:
                    tool = await client.install("redis_get", version="2.0.0")

                    self.assertIsInstance(tool, HubTool)
                    self.assertEqual(tool.name, "redis_get")

                    # Check the installed list
                    installed = await client.list_installed()
                    self.assertIn("redis_get", installed)

        asyncio.run(_test())

    def test_publish_tool(self):
        """发布工具（mock httpx）"""
        async def _test():
            publish_response = MagicMock()
            publish_response.status_code = 201

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=publish_response)
            mock_client.is_closed = False
            mock_client.aclose = AsyncMock()

            test_tool = NormalTool()

            with patch("httpx.AsyncClient", return_value=mock_client):
                async with ToolHubClient(auth_token="sk-test") as client:
                    await client.publish(test_tool, {"category": "filesystem"})

            # Should not raise - publish succeeded

        asyncio.run(_test())

    def test_list_installed_empty(self):
        """已安装列表初始为空"""
        async def _test():
            client = ToolHubClient()
            installed = await client.list_installed()
            self.assertEqual(installed, [])

        asyncio.run(_test())

    def test_network_error_search(self):
        """搜索时网络错误 → ToolHubError"""
        async def _test():
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=__import__("httpx").RequestError("Network error"))
            mock_client.is_closed = False
            mock_client.aclose = AsyncMock()

            with patch("httpx.AsyncClient", return_value=mock_client):
                async with ToolHubClient() as client:
                    with self.assertRaises(ToolHubError):
                        await client.search("test")

        asyncio.run(_test())

    def test_network_error_install(self):
        """安装时网络错误 → ToolHubError"""
        async def _test():
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=__import__("httpx").RequestError("Network error"))
            mock_client.is_closed = False
            mock_client.aclose = AsyncMock()

            with patch("httpx.AsyncClient", return_value=mock_client):
                async with ToolHubClient() as client:
                    with self.assertRaises(ToolHubError):
                        await client.install("nonexistent_tool")

        asyncio.run(_test())

    def test_timeout_error(self):
        """超时错误 → ToolHubError"""
        async def _test():
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=__import__("httpx").TimeoutException("Timeout"))
            mock_client.is_closed = False
            mock_client.aclose = AsyncMock()

            with patch("httpx.AsyncClient", return_value=mock_client):
                async with ToolHubClient() as client:
                    with self.assertRaises(ToolHubError):
                        await client.search("slow_query")

        asyncio.run(_test())

    def test_async_context_manager(self):
        """异步上下文管理器自动管理连接"""
        async def _test():
            mock_client = AsyncMock()
            mock_client.is_closed = False
            mock_client.aclose = AsyncMock()

            with patch("httpx.AsyncClient", return_value=mock_client):
                client = ToolHubClient()
                self.assertIsNone(client._client)

                async with client as c:
                    self.assertIs(c, client)
                    self.assertIsNotNone(client._client)

                # After exiting, client should be closed
                # (aclose was called)

        asyncio.run(_test())

    def test_toolhub_entry_from_dict(self):
        """ToolHubEntry.from_dict 创建"""
        data = {
            "name": "test_tool",
            "version": "1.0.0",
            "author": "tester",
            "description": "A test tool",
            "category": "test",
            "downloads": 50,
            "rating": 4.2,
        }
        entry = ToolHubEntry.from_dict(data)
        self.assertEqual(entry.name, "test_tool")
        self.assertEqual(entry.version, "1.0.0")
        self.assertEqual(entry.rating, 4.2)

    def test_toolhub_entry_to_dict(self):
        """ToolHubEntry.to_dict 序列化"""
        entry = ToolHubEntry(
            name="t", version="1.0", author="a",
            description="d", category="c",
            downloads=10, rating=3.0,
        )
        d = entry.to_dict()
        self.assertEqual(d["name"], "t")
        self.assertEqual(d["downloads"], 10)

    def test_toolhub_error_attributes(self):
        """ToolHubError 异常属性"""
        err = ToolHubError("test error", status_code=500, detail="Internal error")
        self.assertEqual(err.status_code, 500)
        self.assertIn("test error", str(err))
        self.assertEqual(err.detail, "Internal error")


# ======================================================================
# 5. AuthScheme / AuthCredential / AuthManager Tests
# ======================================================================

class TestAuthScheme(unittest.TestCase):
    """AuthScheme — 认证方案定义"""

    def test_bearer_scheme(self):
        scheme = AuthScheme(type="bearer")
        self.assertEqual(scheme.type, "bearer")
        self.assertEqual(scheme.header_name, "Authorization")

    def test_api_key_scheme(self):
        scheme = AuthScheme(type="api_key", header_name="X-API-Key")
        self.assertEqual(scheme.type, "api_key")
        self.assertEqual(scheme.header_name, "X-API-Key")

    def test_oauth2_scheme(self):
        scheme = AuthScheme(type="oauth2", token_url="https://auth.example.com/token")
        self.assertEqual(scheme.type, "oauth2")
        self.assertEqual(scheme.token_url, "https://auth.example.com/token")

    def test_basic_scheme(self):
        scheme = AuthScheme(type="basic")
        self.assertEqual(scheme.type, "basic")

    def test_invalid_scheme_type(self):
        with self.assertRaises(ValueError):
            AuthScheme(type="invalid_type")

    def test_repr(self):
        scheme = AuthScheme(type="bearer")
        r = repr(scheme)
        self.assertIn("bearer", r)


class TestAuthCredential(unittest.TestCase):
    """AuthCredential — 认证凭据存储"""

    def test_mask_empty(self):
        self.assertEqual(AuthCredential._mask(""), "<empty>")

    def test_mask_short(self):
        result = AuthCredential._mask("ab")
        self.assertIn("****", result)

    def test_mask_long(self):
        result = AuthCredential._mask("abcdefghij")
        self.assertIn("ab", result)
        self.assertIn("ij", result)
        self.assertIn("***", result)

    def test_repr_masks_sensitive_values(self):
        """__repr__ 遮蔽敏感字段"""
        cred = AuthCredential(
            api_key="sk-1234567890abcdef",
            access_token="token-abcdef123456",
            client_secret="secret-xyz",
        )
        r = repr(cred)
        # Should NOT contain the raw values
        self.assertNotIn("sk-1234567890abcdef", r)
        self.assertNotIn("token-abcdef123456", r)
        self.assertNotIn("secret-xyz", r)
        # Should contain masked versions
        self.assertIn("***", r)

    def test_is_empty(self):
        empty = AuthCredential()
        self.assertTrue(empty.is_empty())

        non_empty = AuthCredential(api_key="test")
        self.assertFalse(non_empty.is_empty())

    def test_is_empty_with_env_only(self):
        cred = AuthCredential(api_key_env="MY_API_KEY")
        self.assertFalse(cred.is_empty())


class TestAuthManager(unittest.TestCase):
    """AuthManager — 认证方案与凭据的中心化管理"""

    def test_register_and_get_scheme(self):
        """注册和查询认证方案"""
        manager = AuthManager()
        scheme = AuthScheme(type="bearer")
        manager.register_scheme("github", scheme)

        retrieved = manager.get_scheme("github")
        self.assertIs(retrieved, scheme)

    def test_get_nonexistent_scheme(self):
        """查询不存在的方案返回 None"""
        manager = AuthManager()
        self.assertIsNone(manager.get_scheme("nonexistent"))

    def test_set_and_get_credential(self):
        """存储和获取凭据"""
        manager = AuthManager()
        scheme = AuthScheme(type="bearer")
        manager.register_scheme("github", scheme)

        cred = AuthCredential(access_token="ghp_test123")
        manager.set_credential("github", cred)

        retrieved = manager.get_credential("github")
        self.assertIs(retrieved, cred)

    def test_get_nonexistent_credential(self):
        """查询不存在的凭据返回 None"""
        manager = AuthManager()
        self.assertIsNone(manager.get_credential("nonexistent"))

    def test_apply_bearer_auth(self):
        """应用 Bearer Token 认证到 HTTP 请求"""
        manager = AuthManager()
        manager.register_scheme("github", AuthScheme(type="bearer"))
        manager.set_credential("github", AuthCredential(access_token="ghp_test123"))

        headers = {}
        params = {}
        result = manager.apply_auth("github", headers, params)

        self.assertTrue(result)
        self.assertEqual(headers["Authorization"], "Bearer ghp_test123")

    def test_apply_api_key_header(self):
        """应用 API Key 认证（Header 模式）"""
        manager = AuthManager()
        manager.register_scheme("weather", AuthScheme(
            type="api_key", header_name="X-API-Key"
        ))
        manager.set_credential("weather", AuthCredential(api_key="key_12345"))

        headers = {}
        params = {}
        result = manager.apply_auth("weather", headers, params)

        self.assertTrue(result)
        self.assertEqual(headers["X-API-Key"], "key_12345")

    def test_apply_api_key_query_param(self):
        """应用 API Key 认证（查询参数模式）"""
        manager = AuthManager()
        manager.register_scheme("maps", AuthScheme(
            type="api_key", query_param="api_key"
        ))
        manager.set_credential("maps", AuthCredential(api_key="map_key_123"))

        headers = {}
        params = {}
        result = manager.apply_auth("maps", headers, params)

        self.assertTrue(result)
        self.assertEqual(params["api_key"], "map_key_123")

    def test_apply_oauth2_auth(self):
        """应用 OAuth2 认证"""
        manager = AuthManager()
        manager.register_scheme("google", AuthScheme(
            type="oauth2", token_url="https://oauth2.googleapis.com/token"
        ))
        manager.set_credential("google", AuthCredential(access_token="ya29.test_token"))

        headers = {}
        params = {}
        result = manager.apply_auth("google", headers, params)

        self.assertTrue(result)
        self.assertEqual(headers["Authorization"], "Bearer ya29.test_token")

    def test_apply_basic_auth(self):
        """应用 Basic 认证"""
        manager = AuthManager()
        manager.register_scheme("legacy", AuthScheme(type="basic"))
        manager.set_credential("legacy", AuthCredential(
            client_id="user", client_secret="pass"
        ))

        headers = {}
        params = {}
        result = manager.apply_auth("legacy", headers, params)

        self.assertTrue(result)
        self.assertIn("Basic", headers["Authorization"])
        # Decode and verify
        encoded = headers["Authorization"].replace("Basic ", "")
        decoded = base64.b64decode(encoded).decode("utf-8")
        self.assertEqual(decoded, "user:pass")

    def test_resolve_credential_from_env(self):
        """从环境变量解析凭据"""
        manager = AuthManager()
        manager.register_scheme("envtest", AuthScheme(type="api_key"))

        with patch.dict(os.environ, {"MY_SECRET_KEY": "env_value_123"}):
            cred = AuthCredential(api_key_env="MY_SECRET_KEY")
            resolved = manager.resolve_credential(cred)
            self.assertEqual(resolved.api_key, "env_value_123")

    def test_resolve_credential_env_missing(self):
        """环境变量不存在时保持原凭据不变"""
        manager = AuthManager()
        cred = AuthCredential(api_key_env="NONEXISTENT_VAR_XYZ")
        resolved = manager.resolve_credential(cred)
        self.assertEqual(resolved.api_key, "")

    def test_missing_scheme_apply_auth(self):
        """缺少认证方案时 apply_auth 返回 False"""
        manager = AuthManager()
        headers = {}
        params = {}
        result = manager.apply_auth("nonexistent", headers, params)
        self.assertFalse(result)

    def test_missing_credential_apply_auth(self):
        """缺少凭据时 apply_auth 返回 False"""
        manager = AuthManager()
        manager.register_scheme("github", AuthScheme(type="bearer"))
        # No credential set
        headers = {}
        params = {}
        result = manager.apply_auth("github", headers, params)
        self.assertFalse(result)

    def test_bearer_missing_token(self):
        """Bearer 认证缺少 token 时 apply_auth 返回 False"""
        manager = AuthManager()
        manager.register_scheme("test", AuthScheme(type="bearer"))
        manager.set_credential("test", AuthCredential())  # Empty credential
        headers = {}
        params = {}
        result = manager.apply_auth("test", headers, params)
        self.assertFalse(result)

    def test_list_schemes(self):
        """列出所有已注册方案"""
        manager = AuthManager()
        manager.register_scheme("a", AuthScheme(type="bearer"))
        manager.register_scheme("b", AuthScheme(type="api_key"))

        schemes = manager.list_schemes()
        self.assertEqual(len(schemes), 2)
        self.assertIn("a", schemes)
        self.assertIn("b", schemes)

    def test_remove_scheme(self):
        """移除方案及其关联凭据"""
        manager = AuthManager()
        manager.register_scheme("test", AuthScheme(type="bearer"))
        manager.set_credential("test", AuthCredential(access_token="tok"))

        result = manager.remove_scheme("test")
        self.assertTrue(result)
        self.assertIsNone(manager.get_scheme("test"))
        self.assertIsNone(manager.get_credential("test"))

    def test_remove_nonexistent_scheme(self):
        """移除不存在的方案返回 False"""
        manager = AuthManager()
        self.assertFalse(manager.remove_scheme("nonexistent"))

    def test_register_invalid_scheme_type(self):
        """注册无效类型的方案对象"""
        manager = AuthManager()
        with self.assertRaises(TypeError):
            manager.register_scheme("test", "not_an_auth_scheme")  # type: ignore

    def test_register_empty_name(self):
        """注册空名称的方案"""
        manager = AuthManager()
        with self.assertRaises(ValueError):
            manager.register_scheme("", AuthScheme(type="bearer"))

    def test_set_credential_invalid_type(self):
        """设置无效类型的凭据"""
        manager = AuthManager()
        with self.assertRaises(TypeError):
            manager.set_credential("test", "not_a_credential")  # type: ignore

    def test_repr(self):
        """AuthManager __repr__"""
        manager = AuthManager()
        manager.register_scheme("github", AuthScheme(type="bearer"))
        r = repr(manager)
        self.assertIn("AuthManager", r)
        self.assertIn("github", r)


# ======================================================================
# 6. AsyncRWLock Tests
# ======================================================================

class TestAsyncRWLock(unittest.TestCase):
    """AsyncRWLock — 异步读写锁"""

    def test_multiple_concurrent_readers(self):
        """多个读者可以同时持有读锁"""
        async def _test():
            lock = AsyncRWLock()
            read_count = 0

            async def reader():
                nonlocal read_count
                async with lock.read_lock():
                    read_count += 1
                    await asyncio.sleep(0.05)
                    # While holding the lock, there should be multiple readers
                    self.assertTrue(lock.reader_count >= 1)

            await asyncio.gather(reader(), reader(), reader())
            self.assertEqual(read_count, 3)
            self.assertEqual(lock.reader_count, 0)

        asyncio.run(_test())

    def test_writer_exclusivity(self):
        """写者独占，同一时刻只有一个写者"""
        async def _test():
            lock = AsyncRWLock()
            write_count = 0

            async def writer(n):
                nonlocal write_count
                async with lock.write_lock():
                    # While holding write lock, no other writers
                    self.assertTrue(lock.is_write_locked)
                    write_count += 1
                    await asyncio.sleep(0.05)

            await asyncio.gather(writer(1), writer(2))
            self.assertEqual(write_count, 2)
            self.assertFalse(lock.is_write_locked)

        asyncio.run(_test())

    def test_writer_priority(self):
        """写者优先：写者等待时新读者被阻塞"""
        async def _test():
            lock = AsyncRWLock()
            events = []

            async def reader1():
                async with lock.read_lock():
                    events.append("r1_start")
                    await asyncio.sleep(0.1)
                    events.append("r1_end")

            async def writer1():
                await asyncio.sleep(0.02)  # Wait for reader1 to start
                async with lock.write_lock():
                    events.append("w1_start")
                    await asyncio.sleep(0.05)
                    events.append("w1_end")

            async def reader2():
                await asyncio.sleep(0.04)  # Start after writer1 is waiting
                async with lock.read_lock():
                    events.append("r2_start")
                    events.append("r2_end")

            await asyncio.gather(reader1(), writer1(), reader2())

            # reader2 should start after writer1 completes (writer priority)
            # r1_start -> w1 waiting -> r2 blocked -> r1_end -> w1_start -> w1_end -> r2_start
            w1_idx = events.index("w1_start")
            r2_idx = events.index("r2_start")
            self.assertGreater(r2_idx, w1_idx, "reader2 should start after writer1")

        asyncio.run(_test())

    def test_read_lock_context_manager(self):
        """读锁上下文管理器"""
        async def _test():
            lock = AsyncRWLock()
            ctx = lock.read_lock()
            self.assertIsInstance(ctx, ReadLockContext)

            async with lock.read_lock():
                self.assertEqual(lock.reader_count, 1)

            self.assertEqual(lock.reader_count, 0)

        asyncio.run(_test())

    def test_write_lock_context_manager(self):
        """写锁上下文管理器"""
        async def _test():
            lock = AsyncRWLock()
            ctx = lock.write_lock()
            self.assertIsInstance(ctx, WriteLockContext)

            async with lock.write_lock():
                self.assertTrue(lock.is_write_locked)

            self.assertFalse(lock.is_write_locked)

        asyncio.run(_test())

    def test_shared_state_async_methods_with_lock(self):
        """SharedState 异步方法与 AsyncRWLock 集成"""
        async def _test():
            state = SharedState(enable_lock=True)
            self.assertTrue(state.lock_enabled)

            await state.async_set("key1", "value1")
            result = await state.async_get("key1")
            self.assertEqual(result, "value1")

            deleted = await state.async_delete("key1")
            self.assertTrue(deleted)
            self.assertIsNone(await state.async_get("key1"))

        asyncio.run(_test())

    def test_shared_state_lock_context_managers(self):
        """SharedState 锁上下文管理器"""
        async def _test():
            state = SharedState()  # enable_lock=False initially
            # read_lock / write_lock auto-init the lock
            async with state.read_lock():
                state.set("k", "v")
                self.assertEqual(state.get("k"), "v")

            async with state.write_lock():
                state.set("k2", "v2")
                state.delete("k")

            self.assertTrue(state.lock_enabled)

        asyncio.run(_test())

    def test_release_read_without_acquire(self):
        """没有持有读锁时 release_read 抛异常"""
        async def _test():
            lock = AsyncRWLock()
            with self.assertRaises(RuntimeError):
                await lock.release_read()

        asyncio.run(_test())

    def test_release_write_without_acquire(self):
        """没有持有写锁时 release_write 抛异常"""
        async def _test():
            lock = AsyncRWLock()
            with self.assertRaises(RuntimeError):
                await lock.release_write()

        asyncio.run(_test())

    def test_repr(self):
        """AsyncRWLock __repr__"""
        lock = AsyncRWLock()
        r = repr(lock)
        self.assertIn("AsyncRWLock", r)
        self.assertIn("readers=0", r)


# ======================================================================
# 7. ResultCache Tests
# ======================================================================

class TestResultCache(unittest.TestCase):
    """ResultCache — 工具结果缓存"""

    def test_set_and_get(self):
        """设置和获取缓存结果"""
        async def _test():
            cache = ResultCache(max_size=128, default_ttl=60.0)
            await cache.set("key1", "value1")
            result = await cache.get("key1")
            self.assertEqual(result, "value1")

        asyncio.run(_test())

    def test_get_nonexistent(self):
        """获取不存在的缓存键返回 None"""
        async def _test():
            cache = ResultCache()
            result = await cache.get("nonexistent")
            self.assertIsNone(result)

        asyncio.run(_test())

    def test_ttl_expiration(self):
        """TTL 过期后获取返回 None"""
        async def _test():
            cache = ResultCache(max_size=128, default_ttl=0.05)
            await cache.set("key1", "value1")

            # Should be available immediately
            result = await cache.get("key1")
            self.assertEqual(result, "value1")

            # Wait for expiration
            await asyncio.sleep(0.1)
            result = await cache.get("key1")
            self.assertIsNone(result)

        asyncio.run(_test())

    def test_lru_eviction(self):
        """LRU 淘汰：max_size 满时淘汰最旧条目"""
        async def _test():
            cache = ResultCache(max_size=3, default_ttl=0)

            await cache.set("key1", "val1")
            await cache.set("key2", "val2")
            await cache.set("key3", "val3")

            # Cache is full; adding key4 should evict key1 (LRU)
            await cache.set("key4", "val4")

            self.assertIsNone(await cache.get("key1"))
            self.assertEqual(await cache.get("key4"), "val4")
            self.assertEqual(len(cache), 3)

        asyncio.run(_test())

    def test_cache_stats(self):
        """缓存统计信息"""
        async def _test():
            cache = ResultCache(max_size=128, default_ttl=60.0)

            await cache.set("key1", "val1")
            await cache.get("key1")  # hit
            await cache.get("key1")  # hit
            await cache.get("nonexistent")  # miss

            stats = cache.stats()
            self.assertIsInstance(stats, CacheStats)
            self.assertEqual(stats.hits, 2)
            self.assertEqual(stats.misses, 1)
            self.assertEqual(stats.size, 1)
            self.assertEqual(stats.max_size, 128)

        asyncio.run(_test())

    def test_invalidation(self):
        """缓存失效"""
        async def _test():
            cache = ResultCache(default_ttl=0)
            await cache.set("key1", "val1")

            result = await cache.invalidate("key1")
            self.assertTrue(result)

            self.assertIsNone(await cache.get("key1"))

            # Invalidating non-existent key
            result = await cache.invalidate("key1")
            self.assertFalse(result)

        asyncio.run(_test())

    def test_has(self):
        """检查缓存键是否存在"""
        async def _test():
            cache = ResultCache(default_ttl=60.0)
            await cache.set("key1", "val1")

            self.assertTrue(await cache.has("key1"))
            self.assertFalse(await cache.has("nonexistent"))

        asyncio.run(_test())

    def test_clear(self):
        """清空缓存"""
        async def _test():
            cache = ResultCache(default_ttl=0)
            await cache.set("key1", "val1")
            await cache.set("key2", "val2")

            count = await cache.clear()
            self.assertEqual(count, 2)
            self.assertEqual(len(cache), 0)

        asyncio.run(_test())

    def test_cleanup_expired(self):
        """清理过期条目"""
        async def _test():
            cache = ResultCache(max_size=10, default_ttl=0.05)
            await cache.set("key1", "val1")
            await cache.set("key2", "val2")

            # Wait for expiration
            await asyncio.sleep(0.1)

            cleaned = await cache.cleanup_expired()
            self.assertEqual(cleaned, 2)
            self.assertEqual(len(cache), 0)

        asyncio.run(_test())

    def test_make_key(self):
        """缓存键生成"""
        key1 = ResultCache.make_key("tool_a", {"task": "hello"})
        key2 = ResultCache.make_key("tool_a", {"task": "hello"})
        key3 = ResultCache.make_key("tool_a", {"task": "world"})
        key4 = ResultCache.make_key("tool_b", {"task": "hello"})

        self.assertEqual(key1, key2)  # Same tool + params → same key
        self.assertNotEqual(key1, key3)  # Different params → different key
        self.assertNotEqual(key1, key4)  # Different tool → different key

    def test_make_key_deterministic(self):
        """缓存键生成确定性（参数顺序无关）"""
        key1 = ResultCache.make_key("tool", {"a": 1, "b": 2})
        key2 = ResultCache.make_key("tool", {"b": 2, "a": 1})
        self.assertEqual(key1, key2)

    def test_hit_rate(self):
        """缓存命中率"""
        cache = ResultCache()
        self.assertEqual(cache.hit_rate, 0.0)

    def test_invalid_max_size(self):
        """负数 max_size 抛异常"""
        with self.assertRaises(ValueError):
            ResultCache(max_size=-1)

    def test_invalid_default_ttl(self):
        """负数 default_ttl 抛异常"""
        with self.assertRaises(ValueError):
            ResultCache(default_ttl=-1)

    def test_repr(self):
        """ResultCache __repr__"""
        cache = ResultCache(max_size=128)
        r = repr(cache)
        self.assertIn("ResultCache", r)
        self.assertIn("128", r)

    def test_overwrite_existing_key(self):
        """覆盖已存在的缓存键"""
        async def _test():
            cache = ResultCache(default_ttl=0)
            await cache.set("key1", "old_value")
            await cache.set("key1", "new_value")

            result = await cache.get("key1")
            self.assertEqual(result, "new_value")

        asyncio.run(_test())

    def test_lru_access_updates_position(self):
        """访问缓存键更新 LRU 位置"""
        async def _test():
            cache = ResultCache(max_size=3, default_ttl=0)

            await cache.set("key1", "val1")
            await cache.set("key2", "val2")
            await cache.set("key3", "val3")

            # Access key1 → moves it to most-recently-used
            await cache.get("key1")

            # Adding key4 should evict key2 (now LRU)
            await cache.set("key4", "val4")

            self.assertEqual(await cache.get("key1"), "val1")  # Still present
            self.assertIsNone(await cache.get("key2"))  # Evicted

        asyncio.run(_test())


# ======================================================================
# 8. Boundary Tests
# ======================================================================

class TestBoundaryConditions(unittest.TestCase):
    """边界条件测试"""

    def test_zero_agents_orchestrator(self):
        """0 agents 编排器"""
        orchestrator = Orchestrator(agents=[], mode=OrchestrationMode.SEQUENTIAL)
        self.assertEqual(len(orchestrator.agents), 0)
        self.assertEqual(orchestrator.mode, OrchestrationMode.SEQUENTIAL)

    def test_one_agent_orchestrator(self):
        """1 agent 编排器"""
        async def _test():
            provider = create_mock_provider("single agent result")
            agent = Agent(name="solo", provider=provider)
            orchestrator = Orchestrator(agents=[agent], mode=OrchestrationMode.SEQUENTIAL)
            self.assertEqual(len(orchestrator.agents), 1)

            result = await orchestrator.run("Do something")
            self.assertIsInstance(result, OrchestrationResult)

        asyncio.run(_test())

    def test_empty_shared_state_snapshot_restore(self):
        """空 SharedState 快照/恢复"""
        state = SharedState()
        snap = state.snapshot()

        self.assertEqual(snap["data"], {})
        self.assertEqual(snap["scopes"], {})
        self.assertEqual(snap["write_log"], [])

        new_state = SharedState()
        new_state.restore(snap)
        self.assertEqual(len(new_state.keys()), 0)

    def test_approval_gate_no_pending(self):
        """ApprovalGate 无待审批"""
        gate = ApprovalGate()
        self.assertEqual(gate.get_pending_approvals(), [])

    def test_orchestration_checkpoint_empty_orchestrator(self):
        """空编排器的检查点"""
        async def _test():
            tmpdir = tempfile.mkdtemp()
            cp = OrchestrationCheckpoint(base_dir=tmpdir)
            orchestrator = Orchestrator(agents=[], mode=OrchestrationMode.SEQUENTIAL)

            cp_id = await cp.save(orchestrator, run_id="empty_run", turn=0)
            state = await cp.restore("empty_run", checkpoint_id=cp_id)
            self.assertEqual(state["agent_metadata"], [])
            self.assertEqual(state["current_agent"], "")

        asyncio.run(_test())

    def test_result_cache_zero_max_size(self):
        """max_size=0 的 ResultCache（无限制）"""
        async def _test():
            cache = ResultCache(max_size=0, default_ttl=0)
            for i in range(200):
                await cache.set(f"key_{i}", f"val_{i}")

            # No eviction should occur with max_size=0
            self.assertEqual(len(cache), 200)

        asyncio.run(_test())


# ======================================================================
# 9. Error Recovery Tests
# ======================================================================

class TestErrorRecovery(unittest.TestCase):
    """错误恢复测试"""

    def test_agent_tool_execution_failure(self):
        """AgentTool 执行失败时返回错误 ToolResult"""
        async def _test():
            provider = create_mock_provider()
            # Make the provider raise an error
            provider.adapter.send = AsyncMock(side_effect=RuntimeError("Provider crashed"))

            agent = Agent(name="failing_agent", provider=provider)
            tool = AgentTool(agent=agent)

            result = await tool.execute({"task": "Do something that fails"})
            self.assertFalse(result.success)
            self.assertIn("failed", result.error.lower())

        asyncio.run(_test())

    def test_agent_tool_no_provider(self):
        """AgentTool 没有可用 provider"""
        async def _test():
            agent = Agent(name="no_provider_agent")
            tool = AgentTool(agent=agent)

            result = await tool.execute({"task": "Do something"})
            self.assertFalse(result.success)

        asyncio.run(_test())

    def test_orchestrator_tool_inner_failure(self):
        """OrchestratorTool 内部编排失败"""
        async def _test():
            provider = create_mock_provider()
            agent = Agent(name="inner_agent", provider=provider)
            inner = Orchestrator(agents=[agent], mode=OrchestrationMode.SEQUENTIAL)
            tool = OrchestratorTool(orchestrator=inner)

            # Mock inner.run to raise
            inner.run = AsyncMock(side_effect=ConnectionError("Network down"))

            result = await tool.execute({"task": "Do something"})
            self.assertFalse(result.success)
            self.assertIn("failed", result.error.lower())

        asyncio.run(_test())

    def test_auth_manager_invalid_scheme_type_apply(self):
        """AuthManager 使用空凭据应用认证"""
        manager = AuthManager()
        manager.register_scheme("empty_bearer", AuthScheme(type="bearer"))
        manager.set_credential("empty_bearer", AuthCredential())

        headers = {}
        params = {}
        result = manager.apply_auth("empty_bearer", headers, params)
        self.assertFalse(result)

    def test_auth_manager_api_key_empty(self):
        """API Key 认证但 api_key 为空"""
        manager = AuthManager()
        manager.register_scheme("no_key", AuthScheme(type="api_key"))
        manager.set_credential("no_key", AuthCredential())

        headers = {}
        params = {}
        result = manager.apply_auth("no_key", headers, params)
        self.assertFalse(result)

    def test_auth_manager_basic_missing_username(self):
        """Basic 认证缺少用户名"""
        manager = AuthManager()
        manager.register_scheme("no_user", AuthScheme(type="basic"))
        manager.set_credential("no_user", AuthCredential())

        headers = {}
        params = {}
        result = manager.apply_auth("no_user", headers, params)
        self.assertFalse(result)

    def test_orchestrator_tool_validate_empty_task(self):
        """OrchestratorTool 空任务验证"""
        provider = create_mock_provider()
        agent = Agent(name="agent", provider=provider)
        inner = Orchestrator(agents=[agent], mode=OrchestrationMode.SEQUENTIAL)
        tool = OrchestratorTool(orchestrator=inner)

        errors = tool.validate_input({"task": "  "})
        self.assertTrue(len(errors) > 0)


# ======================================================================
# 10. Concurrent Safety Tests
# ======================================================================

class TestConcurrentSafety(unittest.TestCase):
    """并发安全测试"""

    def test_concurrent_shared_state_writes_with_lock(self):
        """多个并发 SharedState 写入（带锁）"""
        async def _test():
            state = SharedState(enable_lock=True)
            num_writers = 10

            async def writer(i):
                await state.async_set(f"key_{i}", f"value_{i}", agent_name=f"writer_{i}")

            await asyncio.gather(*[writer(i) for i in range(num_writers)])

            # All keys should be present
            for i in range(num_writers):
                val = await state.async_get(f"key_{i}")
                self.assertEqual(val, f"value_{i}")

            self.assertEqual(len(state.keys()), num_writers)

        asyncio.run(_test())

    def test_concurrent_shared_state_writes_without_lock(self):
        """多个并发 SharedState 写入（无锁，同步方法）"""
        async def _test():
            state = SharedState()
            num_writers = 10

            def writer(i):
                state.set(f"key_{i}", f"value_{i}", agent_name=f"writer_{i}")

            # Simulate concurrent access via tasks (though sync methods)
            await asyncio.gather(*[
                asyncio.get_running_loop().run_in_executor(None, writer, i)
                for i in range(num_writers)
            ])

            # All keys should be present (best effort; no lock means potential
            # race conditions, but for simple set operations it should work)
            for i in range(num_writers):
                val = state.get(f"key_{i}")
                self.assertEqual(val, f"value_{i}")

        asyncio.run(_test())

    def test_concurrent_approval_requests(self):
        """多个并发审批请求"""
        async def _test():
            gate = ApprovalGate(default_timeout=5.0)
            num_requests = 5
            results = {}

            async def make_request(i):
                result = await gate.request_approval(f"tool_{i}", {"idx": i})
                results[i] = result

            async def approve_all():
                await asyncio.sleep(0.1)
                pending = gate.get_pending_approvals()
                for entry in pending:
                    gate.approve(entry["approval_id"])

            await asyncio.gather(
                *[make_request(i) for i in range(num_requests)],
                approve_all(),
            )

            for i in range(num_requests):
                self.assertTrue(results[i].approved)

        asyncio.run(_test())

    def test_rwlock_read_write_interleaving(self):
        """读写交替场景"""
        async def _test():
            lock = AsyncRWLock()
            counter = {"value": 0, "reads": 0, "writes": 0}

            async def reader():
                for _ in range(5):
                    async with lock.read_lock():
                        counter["reads"] += 1
                        _ = counter["value"]
                    await asyncio.sleep(0.001)

            async def writer():
                for _ in range(5):
                    async with lock.write_lock():
                        counter["writes"] += 1
                        counter["value"] += 1
                    await asyncio.sleep(0.002)

            await asyncio.gather(reader(), reader(), writer())

            self.assertEqual(counter["value"], 5)
            self.assertEqual(counter["writes"], 5)
            self.assertTrue(counter["reads"] > 0)

        asyncio.run(_test())


if __name__ == "__main__":
    unittest.main()
