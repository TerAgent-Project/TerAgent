"""Phase 2 集成测试 — 并行编排、条件路由、循环迭代、Guardrail、ToolPack、OpenAPI

测试覆盖:
1. ParallelPattern: 扇出/扇入并行编排
2. ConditionalPattern: 条件路由（受限 Swarm）
3. LoopPattern: 循环迭代（Generator-Critic）
4. Guardrail: 输入/输出守卫
5. ToolPack: 工具包生命周期
6. OpenAPIToolset: OpenAPI 规范解析与工具生成
7. MCPToolset: MCP 工具集基础验证
8. ToolRegistry 增强: 分类注册、意图推荐
9. Config: AgentConfig + MCPServerConfig
"""
from __future__ import annotations

import asyncio
import json
import sys
import os
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
)
from teragent.orchestration.patterns import (
    OrchestrationResult,
    SequentialPattern,
    SwarmPattern,
)
from teragent.orchestration.patterns.parallel import ParallelPattern
from teragent.orchestration.patterns.conditional import ConditionalPattern
from teragent.orchestration.patterns.loop import LoopPattern
from teragent.orchestration.guardrail import (
    Guardrail,
    GuardrailResult,
    GuardrailTripwireTriggered,
    run_input_guardrails,
    run_output_guardrails,
)
from teragent.tools.base import BaseTool, ToolResult
from teragent.tools.registry import ToolRegistry, ToolInfo
from teragent.tools.toolpack import ToolPack
from teragent.tools.decorator import tool
from teragent.core.types import ToolSafety
from teragent.core.tap import TAPRequest, TAPResponse, CompiledPrompt
from teragent.core.adapter import TAPAdapter
from teragent.core.compiler import TAPCompiler
from teragent.core.provider import ModelProvider
from teragent.config.mcp_config import MCPServerConfig
from teragent.config.agent_config import AgentConfig


# ===== Mock Provider =====

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


# ===== 测试工具 =====

@tool(safety=ToolSafety.READ_ONLY, concurrency_safe=True)
def search_info(query: str) -> str:
    """搜索信息"""
    return f"搜索结果: {query}"


@tool(safety=ToolSafety.SAFE_WRITE)
def write_data(key: str, value: str) -> str:
    """写入数据"""
    return f"已写入: {key}={value}"


# ===== 1. ParallelPattern 测试 =====

class TestParallelPattern(unittest.TestCase):
    """并行编排模式测试"""

    def test_parallel_fan_out_only(self):
        """仅扇出（无扇入）: 多 Agent 并行执行"""
        async def _test():
            provider = create_mock_provider("并行结果")
            agent_a = Agent(name="agent_a", provider=provider, output_key="a_output")
            agent_b = Agent(name="agent_b", provider=provider, output_key="b_output")
            agent_c = Agent(name="agent_c", provider=provider, output_key="c_output")

            pattern = ParallelPattern()
            shared_state = SharedState()
            context = RunContext(
                shared_state=shared_state,
                usage=UsageTracker(),
                current_agent="agent_a",
                turn=0,
                max_turns=10,
                cancellation_token=CancellationToken(),
            )

            result = await pattern.run(
                task="并行处理任务",
                agents=[agent_a, agent_b, agent_c],
                shared_state=shared_state,
                context=context,
            )

            self.assertIsInstance(result, OrchestrationResult)
            # 至少扇出的 agents 有输出
            self.assertIn("a_output", result.agent_outputs)
            self.assertIn("b_output", result.agent_outputs)
            self.assertIn("c_output", result.agent_outputs)

        asyncio.run(_test())

    def test_parallel_with_fan_in(self):
        """扇出 + 扇入: 并行执行后汇聚"""
        async def _test():
            provider = create_mock_provider("汇聚结果")
            agent_a = Agent(name="agent_a", provider=provider, output_key="a_result")
            agent_b = Agent(name="agent_b", provider=provider, output_key="b_result")
            fan_in_agent = Agent(name="aggregator", provider=provider, output_key="aggregated")

            pattern = ParallelPattern()
            shared_state = SharedState()
            context = RunContext(
                shared_state=shared_state,
                usage=UsageTracker(),
                current_agent="agent_a",
                turn=0,
                max_turns=10,
                cancellation_token=CancellationToken(),
            )

            # 创建明确的 config，不使用 MagicMock 以避免自动属性问题
            from teragent.orchestration.orchestrator import RuntimeOrchestratorConfig
            config = RuntimeOrchestratorConfig(mode=OrchestrationMode.PARALLEL)
            # 手动设置 fan_in_agent
            config.fan_in_agent = fan_in_agent  # type: ignore

            result = await pattern.run(
                task="并行处理后汇聚",
                agents=[agent_a, agent_b, fan_in_agent],
                shared_state=shared_state,
                context=context,
                config=config,
            )

            self.assertIsInstance(result, OrchestrationResult)

        asyncio.run(_test())

    def test_parallel_empty_agents(self):
        """空 Agent 列表"""
        async def _test():
            pattern = ParallelPattern()
            shared_state = SharedState()
            context = RunContext(
                shared_state=shared_state,
                usage=UsageTracker(),
                current_agent="",
                turn=0,
                max_turns=10,
                cancellation_token=CancellationToken(),
            )

            result = await pattern.run(
                task="空任务",
                agents=[],
                shared_state=shared_state,
                context=context,
            )

            self.assertIsInstance(result, OrchestrationResult)
            self.assertEqual(result.total_turns, 0)

        asyncio.run(_test())

    def test_parallel_execution_plan(self):
        """获取执行计划"""
        pattern = ParallelPattern()
        plan = pattern.get_execution_plan()
        self.assertIsInstance(plan, list)

    def test_parallel_via_orchestrator(self):
        """通过 Orchestrator 使用并行模式"""
        async def _test():
            provider = create_mock_provider("并行结果")
            agent_a = Agent(name="agent_a", provider=provider, output_key="a_out")
            agent_b = Agent(name="agent_b", provider=provider, output_key="b_out")

            orchestrator = Orchestrator(
                agents=[agent_a, agent_b],
                mode=OrchestrationMode.PARALLEL,
            )

            result = await orchestrator.run("并行任务")
            self.assertIsInstance(result, OrchestrationResult)

        asyncio.run(_test())


# ===== 2. ConditionalPattern 测试 =====

class TestConditionalPattern(unittest.TestCase):
    """条件路由编排模式测试"""

    def test_conditional_basic(self):
        """基本条件路由"""
        async def _test():
            provider = create_mock_provider("路由结果")
            router = Agent(
                name="router",
                provider=provider,
                handoffs=[Handoff(target_agent=Agent(name="specialist", provider=provider, output_key="spec_out"))],
            )
            specialist = Agent(name="specialist", provider=provider, output_key="spec_out")

            pattern = ConditionalPattern()
            shared_state = SharedState()
            context = RunContext(
                shared_state=shared_state,
                usage=UsageTracker(),
                current_agent="router",
                turn=0,
                max_turns=10,
                cancellation_token=CancellationToken(),
            )

            result = await pattern.run(
                task="需要路由的任务",
                agents=[router, specialist],
                shared_state=shared_state,
                context=context,
            )

            self.assertIsInstance(result, OrchestrationResult)

        asyncio.run(_test())

    def test_conditional_execution_plan(self):
        """获取执行计划"""
        pattern = ConditionalPattern()
        plan = pattern.get_execution_plan()
        self.assertIsInstance(plan, list)

    def test_conditional_via_orchestrator(self):
        """通过 Orchestrator 使用条件路由模式"""
        async def _test():
            provider = create_mock_provider()
            agent_a = Agent(name="router", provider=provider)
            agent_b = Agent(name="handler", provider=provider, output_key="result")

            orchestrator = Orchestrator(
                agents=[agent_a, agent_b],
                mode=OrchestrationMode.CONDITIONAL,
            )

            result = await orchestrator.run("条件路由任务")
            self.assertIsInstance(result, OrchestrationResult)

        asyncio.run(_test())


# ===== 3. LoopPattern 测试 =====

class TestLoopPattern(unittest.TestCase):
    """循环迭代编排模式测试"""

    def test_loop_max_iterations(self):
        """循环达到最大迭代次数后停止"""
        async def _test():
            provider = create_mock_provider("迭代结果")
            generator = Agent(name="generator", provider=provider, output_key="gen_out")
            critic = Agent(name="critic", provider=provider, output_key="critic_out")

            pattern = LoopPattern()
            shared_state = SharedState()
            context = RunContext(
                shared_state=shared_state,
                usage=UsageTracker(),
                current_agent="generator",
                turn=0,
                max_turns=20,
                cancellation_token=CancellationToken(),
            )

            result = await pattern.run(
                task="循环生成和审查",
                agents=[generator, critic],
                shared_state=shared_state,
                context=context,
                max_iterations=2,
            )

            self.assertIsInstance(result, OrchestrationResult)
            # 应该完成 2 次迭代
            self.assertGreaterEqual(result.total_turns, 2)

        asyncio.run(_test())

    def test_loop_with_exit_condition(self):
        """满足退出条件时提前退出"""
        async def _test():
            provider = create_mock_provider("PASS: 质量达标")
            generator = Agent(name="generator", provider=provider, output_key="gen_out")
            critic = Agent(name="critic", provider=provider, output_key="critic_out")

            pattern = LoopPattern()
            shared_state = SharedState()
            context = RunContext(
                shared_state=shared_state,
                usage=UsageTracker(),
                current_agent="generator",
                turn=0,
                max_turns=20,
                cancellation_token=CancellationToken(),
            )

            result = await pattern.run(
                task="生成和审查直到通过",
                agents=[generator, critic],
                shared_state=shared_state,
                context=context,
                max_iterations=5,
                exit_condition="PASS",
            )

            self.assertIsInstance(result, OrchestrationResult)

        asyncio.run(_test())

    def test_loop_with_callable_exit_condition(self):
        """可调用退出条件"""
        async def _test():
            call_count = 0

            def exit_check(shared_state, iteration):
                nonlocal call_count
                call_count += 1
                return iteration >= 1  # 第 2 轮退出

            provider = create_mock_provider("迭代结果")
            agent = Agent(name="worker", provider=provider, output_key="work_out")

            pattern = LoopPattern()
            shared_state = SharedState()
            context = RunContext(
                shared_state=shared_state,
                usage=UsageTracker(),
                current_agent="worker",
                turn=0,
                max_turns=10,
                cancellation_token=CancellationToken(),
            )

            result = await pattern.run(
                task="迭代任务",
                agents=[agent],
                shared_state=shared_state,
                context=context,
                max_iterations=5,
                exit_condition=exit_check,
            )

            self.assertIsInstance(result, OrchestrationResult)
            self.assertGreaterEqual(call_count, 1)

        asyncio.run(_test())

    def test_loop_empty_agents(self):
        """空 Agent 列表"""
        async def _test():
            pattern = LoopPattern()
            shared_state = SharedState()
            context = RunContext(
                shared_state=shared_state,
                usage=UsageTracker(),
                current_agent="",
                turn=0,
                max_turns=10,
                cancellation_token=CancellationToken(),
            )

            result = await pattern.run(
                task="空任务",
                agents=[],
                shared_state=shared_state,
                context=context,
                max_iterations=3,
            )

            self.assertIsInstance(result, OrchestrationResult)
            self.assertEqual(result.total_turns, 0)

        asyncio.run(_test())

    def test_loop_execution_plan(self):
        """获取执行计划"""
        pattern = LoopPattern()
        plan = pattern.get_execution_plan()
        self.assertIsInstance(plan, list)

    def test_loop_via_orchestrator(self):
        """通过 Orchestrator 使用循环模式"""
        async def _test():
            provider = create_mock_provider("循环结果")
            agent = Agent(name="worker", provider=provider, output_key="work")

            config = MagicMock()
            config.max_turns = 20
            config.max_iterations = 2
            config.exit_condition = ""
            config.timeout = 60

            orchestrator = Orchestrator(
                agents=[agent],
                mode=OrchestrationMode.LOOP,
                config=config,
            )

            result = await orchestrator.run("循环任务")
            self.assertIsInstance(result, OrchestrationResult)

        asyncio.run(_test())


# ===== 4. Guardrail 测试 =====

class TestGuardrail(unittest.TestCase):
    """守卫机制测试"""

    def test_guardrail_result_passed(self):
        """GuardrailResult 通过"""
        result = GuardrailResult(passed=True)
        self.assertTrue(result.passed)
        self.assertEqual(result.output_info, "")
        self.assertIsNone(result.modified_data)

    def test_guardrail_result_failed(self):
        """GuardrailResult 失败"""
        result = GuardrailResult(passed=False, output_info="包含敏感信息")
        self.assertFalse(result.passed)
        self.assertEqual(result.output_info, "包含敏感信息")

    def test_guardrail_tripwire_exception(self):
        """GuardrailTripwireTriggered 异常"""
        exc = GuardrailTripwireTriggered("test_guard", "触发跳闸")
        self.assertEqual(exc.guardrail_name, "test_guard")
        self.assertEqual(exc.output_info, "触发跳闸")
        self.assertIn("test_guard", str(exc))

    def test_guardrail_creation(self):
        """Guardrail 创建"""
        async def my_check(agent, data, context):
            return GuardrailResult(passed=True)

        guard = Guardrail(name="test_guard", check=my_check, mode="input")
        self.assertEqual(guard.name, "test_guard")
        self.assertEqual(guard.mode, "input")
        self.assertTrue(guard.run_in_parallel)

    def test_input_guardrails_all_pass(self):
        """输入守卫全部通过"""
        async def _test():
            async def check_a(agent, data, context):
                return GuardrailResult(passed=True)

            async def check_b(agent, data, context):
                return GuardrailResult(passed=True)

            guardrails = [
                Guardrail(name="a", check=check_a, mode="input"),
                Guardrail(name="b", check=check_b, mode="input"),
            ]
            agent = Agent(name="test", provider=create_mock_provider())
            context = RunContext(
                shared_state=SharedState(),
                usage=UsageTracker(),
                current_agent="test",
                turn=0,
                max_turns=10,
                cancellation_token=CancellationToken(),
            )

            results = await run_input_guardrails(guardrails, agent, "test input", context)
            self.assertEqual(len(results), 2)
            self.assertTrue(all(r.passed for r in results))

        asyncio.run(_test())

    def test_input_guardrails_tripwire_triggered(self):
        """输入守卫触发跳闸"""
        async def _test():
            async def check_pass(agent, data, context):
                return GuardrailResult(passed=True)

            async def check_fail(agent, data, context):
                return GuardrailResult(passed=False, output_info="检测到有害内容")

            guardrails = [
                Guardrail(name="pass_guard", check=check_pass, mode="input"),
                Guardrail(name="fail_guard", check=check_fail, mode="input"),
            ]
            agent = Agent(name="test", provider=create_mock_provider())
            context = RunContext(
                shared_state=SharedState(),
                usage=UsageTracker(),
                current_agent="test",
                turn=0,
                max_turns=10,
                cancellation_token=CancellationToken(),
            )

            with self.assertRaises(GuardrailTripwireTriggered) as cm:
                await run_input_guardrails(guardrails, agent, "harmful input", context)
            self.assertEqual(cm.exception.guardrail_name, "fail_guard")

        asyncio.run(_test())

    def test_output_guardrails(self):
        """输出守卫测试"""
        async def _test():
            async def check_output(agent, output, context):
                return GuardrailResult(passed=True, output_info="输出安全")

            guardrails = [
                Guardrail(name="output_check", check=check_output, mode="output"),
            ]
            agent = Agent(name="test", provider=create_mock_provider())
            context = RunContext(
                shared_state=SharedState(),
                usage=UsageTracker(),
                current_agent="test",
                turn=0,
                max_turns=10,
                cancellation_token=CancellationToken(),
            )

            results = await run_output_guardrails(guardrails, agent, "safe output", context)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].passed)

        asyncio.run(_test())

    def test_agent_with_guardrails(self):
        """Agent 携带 Guardrail 字段"""
        async def my_input_check(agent, data, context):
            return GuardrailResult(passed=True)

        agent = Agent(
            name="guarded_agent",
            input_guardrails=[Guardrail(name="input_guard", check=my_input_check)],
            output_guardrails=[],
        )
        self.assertEqual(len(agent.input_guardrails), 1)
        self.assertEqual(len(agent.output_guardrails), 0)


# ===== 5. ToolPack 测试 =====

class TestToolPack(unittest.TestCase):
    """工具包测试"""

    def test_toolpack_creation(self):
        """ToolPack 创建"""
        @tool(safety=ToolSafety.READ_ONLY)
        def tool_a(query: str) -> str:
            """查询工具"""
            return f"查询: {query}"

        @tool(safety=ToolSafety.SAFE_WRITE)
        def tool_b(key: str, value: str) -> str:
            """写入工具"""
            return f"写入: {key}={value}"

        pack = ToolPack(tools=[tool_a, tool_b], name="io_pack")
        self.assertEqual(pack.name, "io_pack")
        self.assertEqual(len(pack.list_tools()), 2)

    def test_toolpack_register_to(self):
        """ToolPack 注册到 ToolRegistry"""
        @tool(safety=ToolSafety.READ_ONLY)
        def tool_x(data: str) -> str:
            """测试工具"""
            return data

        pack = ToolPack(tools=[tool_x], name="test_pack")
        registry = ToolRegistry()
        count = pack.register_to(registry)
        self.assertEqual(count, 1)
        self.assertTrue(registry.has_tool("tool_x"))

    def test_toolpack_context_manager(self):
        """ToolPack 异步上下文管理器"""
        async def _test():
            @tool(safety=ToolSafety.READ_ONLY)
            def tool_y(data: str) -> str:
                """测试工具"""
                return data

            start_called = False
            stop_called = False

            async def on_start(shared_state):
                nonlocal start_called
                start_called = True

            async def on_stop(shared_state):
                nonlocal stop_called
                stop_called = True

            pack = ToolPack(
                tools=[tool_y],
                name="ctx_pack",
                on_start=on_start,
                on_stop=on_stop,
            )

            async with pack:
                self.assertTrue(start_called)
                self.assertFalse(stop_called)

            self.assertTrue(stop_called)

        asyncio.run(_test())

    def test_toolpack_get_tool(self):
        """ToolPack 获取工具"""
        @tool(safety=ToolSafety.READ_ONLY)
        def tool_z(data: str) -> str:
            """测试工具"""
            return data

        pack = ToolPack(tools=[tool_z], name="get_pack")
        found = pack.get_tool("tool_z")
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "tool_z")

        not_found = pack.get_tool("nonexistent")
        self.assertIsNone(not_found)

    def test_toolpack_shared_state(self):
        """ToolPack 共享状态"""
        pack = ToolPack(
            tools=[],
            name="state_pack",
            shared_state={"connection": "active"},
        )
        self.assertEqual(pack.shared_state["connection"], "active")


# ===== 6. OpenAPIToolset 测试 =====

class TestOpenAPIToolset(unittest.TestCase):
    """OpenAPI 工具集测试"""

    def _get_petstore_spec(self) -> dict:
        """创建简化的 Petstore OpenAPI 规范"""
        return {
            "openapi": "3.0.0",
            "info": {"title": "Petstore", "version": "1.0.0"},
            "servers": [{"url": "https://petstore.example.com/v1"}],
            "paths": {
                "/pets": {
                    "get": {
                        "operationId": "listPets",
                        "summary": "List all pets",
                        "parameters": [
                            {
                                "name": "limit",
                                "in": "query",
                                "required": False,
                                "schema": {"type": "integer"},
                            }
                        ],
                        "responses": {"200": {"description": "A list of pets"}},
                    },
                    "post": {
                        "operationId": "createPet",
                        "summary": "Create a pet",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "tag": {"type": "string"},
                                        },
                                        "required": ["name"],
                                    }
                                }
                            }
                        },
                        "responses": {"201": {"description": "Pet created"}},
                    },
                },
                "/pets/{petId}": {
                    "get": {
                        "operationId": "showPetById",
                        "summary": "Info for a specific pet",
                        "parameters": [
                            {
                                "name": "petId",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "string"},
                            }
                        ],
                        "responses": {"200": {"description": "Pet details"}},
                    },
                },
            },
        }

    def test_openapi_parse(self):
        """解析 OpenAPI 规范生成工具"""
        async def _test():
            from teragent.tools.openapi_toolset import OpenAPIToolset

            spec = self._get_petstore_spec()
            toolset = OpenAPIToolset(spec=spec, base_url="https://petstore.example.com/v1")
            tools = await toolset.parse()

            self.assertGreaterEqual(len(tools), 2)
            tool_names = [t.name for t in tools]
            self.assertIn("listPets", tool_names)
            self.assertIn("createPet", tool_names)

        asyncio.run(_test())

    def test_openapi_tool_filter(self):
        """OpenAPI 工具过滤"""
        async def _test():
            from teragent.tools.openapi_toolset import OpenAPIToolset

            spec = self._get_petstore_spec()
            toolset = OpenAPIToolset(
                spec=spec,
                base_url="https://petstore.example.com/v1",
                tool_filter=["listPets"],
            )
            tools = await toolset.parse()

            self.assertEqual(len(tools), 1)
            self.assertEqual(tools[0].name, "listPets")

        asyncio.run(_test())

    def test_openapi_register_to(self):
        """OpenAPI 工具注册到 ToolRegistry"""
        async def _test():
            from teragent.tools.openapi_toolset import OpenAPIToolset

            spec = self._get_petstore_spec()
            toolset = OpenAPIToolset(spec=spec, base_url="https://petstore.example.com/v1")
            await toolset.parse()

            registry = ToolRegistry()
            toolset.register_to(registry)
            self.assertGreaterEqual(len(registry), 2)

        asyncio.run(_test())

    def test_openapi_safety_inference(self):
        """OpenAPI 安全级别推断"""
        async def _test():
            from teragent.tools.openapi_toolset import OpenAPIToolset

            spec = self._get_petstore_spec()
            toolset = OpenAPIToolset(spec=spec, base_url="https://petstore.example.com/v1")
            tools = await toolset.parse()

            tool_map = {t.name: t for t in tools}
            self.assertEqual(tool_map["listPets"].safety_level, ToolSafety.READ_ONLY)
            self.assertEqual(tool_map["createPet"].safety_level, ToolSafety.SAFE_WRITE)

        asyncio.run(_test())

    def test_openapi_operation_tool_url_building(self):
        """OpenAPIOperationTool URL 构建"""
        from teragent.tools.openapi_toolset import OpenAPIOperationTool

        op_tool = OpenAPIOperationTool(
            operation_id="showPetById",
            method="get",
            path="/pets/{petId}",
            base_url="https://petstore.example.com/v1",
            parameters_schema={
                "type": "object",
                "properties": {
                    "petId": {"type": "string"},
                },
                "required": ["petId"],
            },
            path_params=["petId"],
            safety=ToolSafety.READ_ONLY,
        )
        self.assertEqual(op_tool.name, "showPetById")
        self.assertEqual(op_tool.safety_level, ToolSafety.READ_ONLY)


# ===== 7. MCPToolset 基础验证 =====

class TestMCPToolset(unittest.TestCase):
    """MCP 工具集基础验证（不连接真实 MCP 服务器）"""

    def test_mcp_config_creation(self):
        """MCPServerConfig 创建"""
        config = MCPServerConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            name="filesystem",
        )
        self.assertEqual(config.command, "npx")
        self.assertEqual(config.transport, "stdio")
        self.assertEqual(config.name, "filesystem")

    def test_mcp_config_from_dict(self):
        """MCPServerConfig 从字典创建"""
        config = MCPServerConfig.from_dict("fs", {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem"],
            "transport": "stdio",
        })
        self.assertEqual(config.name, "fs")
        self.assertEqual(config.command, "npx")

    def test_mcp_config_http(self):
        """MCPServerConfig HTTP 模式"""
        config = MCPServerConfig(
            url="https://mcp.example.com/sse",
            transport="sse",
            name="remote_mcp",
        )
        self.assertEqual(config.url, "https://mcp.example.com/sse")
        self.assertEqual(config.transport, "sse")

    def test_mcp_toolset_creation(self):
        """MCPToolset 创建（不连接）"""
        from teragent.tools.mcp_toolset import MCPToolset

        config = MCPServerConfig(command="npx", args=["-y", "test"], name="test")
        toolset = MCPToolset(server_params=config, name="test_toolset")
        self.assertEqual(toolset.name, "test_toolset")

    def test_mcp_tool_creation(self):
        """MCPTool 创建"""
        from teragent.tools.mcp_toolset import MCPTool, MCPToolset

        config = MCPServerConfig(command="npx", args=["-y", "test"])
        toolset = MCPToolset(server_params=config)
        mcp_tool = MCPTool(
            toolset=toolset,
            name="test_mcp_tool",
            description="Test MCP tool",
            input_schema={"type": "object", "properties": {}},
        )
        self.assertEqual(mcp_tool.name, "test_mcp_tool")
        self.assertEqual(mcp_tool.safety_level, ToolSafety.SAFE_WRITE)

    def test_mcp_config_to_dict(self):
        """MCPServerConfig 序列化"""
        config = MCPServerConfig(command="npx", name="fs")
        d = config.to_dict()
        self.assertEqual(d["command"], "npx")
        self.assertEqual(d["name"], "fs")


# ===== 8. ToolRegistry 增强 =====

class TestToolRegistryEnhanced(unittest.TestCase):
    """增强工具注册表测试"""

    def test_tool_info(self):
        """ToolInfo 创建"""
        info = ToolInfo(name="test", category="io", source="builtin")
        self.assertEqual(info.name, "test")
        self.assertEqual(info.category, "io")
        self.assertEqual(info.source, "builtin")

    def test_register_category(self):
        """按分类注册工具"""
        @tool(safety=ToolSafety.READ_ONLY)
        def cat_tool_a(query: str) -> str:
            """分类工具A"""
            return query

        @tool(safety=ToolSafety.READ_ONLY)
        def cat_tool_b(query: str) -> str:
            """分类工具B"""
            return query

        registry = ToolRegistry()
        registry.register_category("search", [cat_tool_a, cat_tool_b])

        tools = registry.get_tools_by_category("search")
        self.assertEqual(len(tools), 2)

    def test_get_tools_for_intent(self):
        """按意图推荐工具"""
        @tool(safety=ToolSafety.READ_ONLY)
        def search_tool(query: str) -> str:
            """搜索信息"""
            return query

        registry = ToolRegistry()
        registry.register_category("search", [search_tool])

        tools = registry.get_tools_for_intent("search")
        self.assertGreaterEqual(len(tools), 1)

    def test_register_toolpack(self):
        """注册 ToolPack"""
        @tool(safety=ToolSafety.READ_ONLY)
        def pack_tool(data: str) -> str:
            """包内工具"""
            return data

        pack = ToolPack(tools=[pack_tool], name="my_pack")
        registry = ToolRegistry()
        count = registry.register_toolpack(pack)
        self.assertEqual(count, 1)
        self.assertTrue(registry.has_tool("pack_tool"))

    def test_backward_compatibility(self):
        """向后兼容 — 原有方法仍正常工作"""
        registry = ToolRegistry()

        @tool(safety=ToolSafety.READ_ONLY)
        def compat_tool(data: str) -> str:
            """兼容工具"""
            return data

        registry.register(compat_tool)
        self.assertTrue(registry.has_tool("compat_tool"))
        self.assertIsNotNone(registry.get("compat_tool"))
        self.assertIn("compat_tool", registry)


# ===== 9. Config 测试 =====

class TestConfig(unittest.TestCase):
    """配置类测试"""

    def test_agent_config(self):
        """AgentConfig 创建"""
        config = AgentConfig(
            name="test_agent",
            driver="openai_compatible.glm_5",
            description="Test agent",
            tools=["read_file", "write_file"],
            max_steps=10,
            mcp_servers=["filesystem"],
            input_guardrails=["content_check"],
        )
        self.assertEqual(config.name, "test_agent")
        self.assertEqual(config.max_steps, 10)
        self.assertIn("filesystem", config.mcp_servers)

    def test_agent_config_from_dict(self):
        """AgentConfig 从字典创建"""
        config = AgentConfig.from_dict("worker", {
            "driver": "openai_compatible.deepseek_v4",
            "description": "Worker agent",
            "tools": ["search", "write"],
            "max_steps": 20,
        })
        self.assertEqual(config.name, "worker")
        self.assertEqual(config.max_steps, 20)
        self.assertEqual(len(config.tools), 2)


# ===== 10. 编排模式 OrchestrationMode 枚举完整性 =====

class TestOrchestrationModeComplete(unittest.TestCase):
    """验证所有编排模式枚举值"""

    def test_all_modes_exist(self):
        """所有 Phase 2 编排模式已注册"""
        self.assertEqual(OrchestrationMode.SEQUENTIAL.value, "sequential")
        self.assertEqual(OrchestrationMode.SWARM.value, "swarm")
        self.assertEqual(OrchestrationMode.PARALLEL.value, "parallel")
        self.assertEqual(OrchestrationMode.CONDITIONAL.value, "conditional")
        self.assertEqual(OrchestrationMode.LOOP.value, "loop")

    def test_orchestrator_creates_all_patterns(self):
        """Orchestrator 可创建所有模式的 Pattern"""
        for mode in OrchestrationMode:
            orchestrator = Orchestrator(agents=[], mode=mode)
            self.assertIsNotNone(orchestrator._pattern)


if __name__ == "__main__":
    unittest.main()
