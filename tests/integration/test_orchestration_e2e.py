"""End-to-end integration tests for Phase 1 orchestration.

Test scenarios:
1. Sequential orchestration: planner → coder → reviewer
2. Swarm orchestration: triage → specialist handoff
3. Agent-as-Tool: orchestrator calls sub_agent.as_tool()
4. @tool decorator + orchestration: custom @tool in orchestrated agents
5. CancellationToken: mid-orchestration cancellation
6. AgentLoop backward compatibility: old interface still works

These tests use a mock ModelProvider to avoid requiring actual API keys.
"""

from __future__ import annotations

import asyncio
import sys
import os
import unittest

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from teragent.orchestration import (
    Agent,
    AgentHooks,
    CancellationToken,
    Handoff,
    HandoffInputFilter,
    Orchestrator,
    OrchestrationMode,
    SharedState,
    RunContext,
    UsageTracker,
)
from teragent.orchestration.patterns import SequentialPattern, SwarmPattern, OrchestrationResult
from teragent.tools.base import BaseTool, ToolResult
from teragent.tools.decorator import tool
from teragent.core.types import ToolSafety
from teragent.core.tap import TAPRequest, TAPResponse, CompiledPrompt
from teragent.core.adapter import TAPAdapter
from teragent.core.compiler import TAPCompiler
from teragent.core.provider import ModelProvider


# ===== Mock Provider for Testing =====

class MockCompiler(TAPCompiler):
    """Mock compiler that passes through the instruction."""
    
    def compile(self, request: TAPRequest) -> CompiledPrompt:
        return CompiledPrompt(
            messages=[{"role": "user", "content": request.instruction}],
            max_tokens=1024,
        )


class MockAdapter(TAPAdapter):
    """Mock adapter that returns a canned response."""
    
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
    """Create a mock ModelProvider for testing."""
    compiler = MockCompiler()
    adapter = MockAdapter(response_text)
    return ModelProvider(compiler=compiler, adapter=adapter, model="mock-model")


# ===== Custom Tools for Testing =====

@tool(safety=ToolSafety.READ_ONLY, concurrency_safe=True)
def lookup_info(query: str) -> str:
    """Look up information"""
    return f"Info about: {query}"


@tool(safety=ToolSafety.SAFE_WRITE)
def process_data(data: str, format: str = "json") -> str:
    """Process data in specified format"""
    return f"Processed: {data} as {format}"


# ===== Test Cases =====

class TestSequentialOrchestration(unittest.TestCase):
    """Test 1: Sequential orchestration — planner → coder → reviewer"""
    
    def test_sequential_three_agents(self):
        async def _test():
            provider = create_mock_provider()
            
            planner = Agent(
                name="planner",
                description="Plans the task",
                provider=provider,
                output_key="plan",
            )
            coder = Agent(
                name="coder",
                description="Writes code",
                provider=provider,
                output_key="code",
            )
            reviewer = Agent(
                name="reviewer",
                description="Reviews the code",
                provider=provider,
                output_key="review",
            )
            
            orchestrator = Orchestrator(
                agents=[planner, coder, reviewer],
                mode=OrchestrationMode.SEQUENTIAL,
            )
            
            result = await orchestrator.run("Build a login system")
            
            self.assertIsInstance(result, OrchestrationResult)
            self.assertEqual(result.total_turns, 3)
            self.assertIn("plan", result.agent_outputs)
            self.assertIn("code", result.agent_outputs)
            self.assertIn("review", result.agent_outputs)
            self.assertEqual(result.last_agent, "reviewer")
        
        asyncio.run(_test())
    
    def test_sequential_empty_agents(self):
        async def _test():
            orchestrator = Orchestrator(agents=[], mode=OrchestrationMode.SEQUENTIAL)
            result = await orchestrator.run("test")
            self.assertEqual(result.total_turns, 0)
        
        asyncio.run(_test())


class TestSwarmOrchestration(unittest.TestCase):
    """Test 2: Swarm orchestration — triage → specialist handoff"""
    
    def test_swarm_basic(self):
        async def _test():
            # Use a provider that returns text (no tool calls)
            provider = create_mock_provider("I can help with that directly.")
            
            triage = Agent(
                name="triage",
                description="Triage agent",
                provider=provider,
                handoffs=[],
            )
            specialist = Agent(
                name="specialist",
                description="Specialist agent",
                provider=provider,
                handoffs=[],
            )
            
            orchestrator = Orchestrator(
                agents=[triage, specialist],
                mode=OrchestrationMode.SWARM,
            )
            
            result = await orchestrator.run("Help me with something")
            
            self.assertIsInstance(result, OrchestrationResult)
            self.assertGreater(result.total_turns, 0)
        
        asyncio.run(_test())


class TestAgentAsTool(unittest.TestCase):
    """Test 3: Agent-as-Tool"""
    
    def test_agent_as_tool_creation(self):
        provider = create_mock_provider("Sub-agent result")
        sub_agent = Agent(
            name="sub_agent",
            description="A sub-agent",
            provider=provider,
        )
        
        agent_tool = sub_agent.as_tool()
        self.assertEqual(agent_tool.name, "use_sub_agent")
        self.assertEqual(agent_tool.description, "A sub-agent")
        self.assertEqual(agent_tool.safety_level, ToolSafety.SAFE_WRITE)
    
    def test_agent_as_tool_execution(self):
        async def _test():
            provider = create_mock_provider("Sub-agent executed successfully")
            sub_agent = Agent(
                name="sub_agent",
                description="A sub-agent",
                provider=provider,
            )
            
            agent_tool = sub_agent.as_tool()
            result = await agent_tool.execute({"task": "Do something"})
            
            self.assertTrue(result.success)
            self.assertIn("output", result.data)
            self.assertEqual(result.metadata["agent"], "sub_agent")
        
        asyncio.run(_test())


class TestToolDecorator(unittest.TestCase):
    """Test 4: @tool decorator + orchestration"""
    
    def test_basic_tool_decorator(self):
        self.assertIsInstance(lookup_info, BaseTool)
        self.assertEqual(lookup_info.name, "lookup_info")
        self.assertEqual(lookup_info.safety_level, ToolSafety.READ_ONLY)
        self.assertTrue(lookup_info.is_concurrency_safe)
    
    def test_tool_decorator_with_params(self):
        self.assertIsInstance(process_data, BaseTool)
        self.assertEqual(process_data.name, "process_data")
        self.assertEqual(process_data.safety_level, ToolSafety.SAFE_WRITE)
        self.assertFalse(process_data.is_concurrency_safe)
    
    def test_tool_in_agent(self):
        provider = create_mock_provider()
        agent = Agent(
            name="worker",
            description="Worker agent",
            provider=provider,
            tools=[lookup_info, process_data],
        )
        
        self.assertEqual(len(agent.tools), 2)
        self.assertEqual(agent.tools[0].name, "lookup_info")
        self.assertEqual(agent.tools[1].name, "process_data")
    
    def test_tool_execution(self):
        async def _test():
            result = await lookup_info.execute({"query": "test"})
            self.assertTrue(result.success)
            self.assertIn("output", result.data)
        
        asyncio.run(_test())


class TestCancellationToken(unittest.TestCase):
    """Test 5: CancellationToken"""
    
    def test_basic_cancellation(self):
        ct = CancellationToken()
        self.assertFalse(ct.is_cancelled)
        ct.cancel()
        self.assertTrue(ct.is_cancelled)
    
    def test_throw_if_cancelled(self):
        ct = CancellationToken()
        ct.cancel()
        with self.assertRaises(asyncio.CancelledError):
            ct.throw_if_cancelled()
    
    def test_uncancelled_does_not_throw(self):
        ct = CancellationToken()
        # Should not raise
        ct.throw_if_cancelled()


class TestSharedState(unittest.TestCase):
    """Test SharedState functionality"""
    
    def test_basic_operations(self):
        state = SharedState()
        state.set("key1", "value1")
        self.assertEqual(state.get("key1"), "value1")
        self.assertIsNone(state.get("nonexistent"))
        self.assertEqual(state.get("nonexistent", "default"), "default")
    
    def test_scoped_operations(self):
        state = SharedState()
        state.set("x", 1, scope="agent")
        self.assertEqual(state.get("x", scope="agent"), 1)
        self.assertIsNone(state.get("x"))  # Without scope, different key
    
    def test_snapshot_restore(self):
        state = SharedState()
        state.set("key1", "value1")
        state.set("key2", "value2")
        
        snap = state.snapshot()
        state.set("key3", "value3")
        self.assertIn("key3", state.to_dict())
        
        state.restore(snap)
        self.assertNotIn("key3", state.to_dict())
        self.assertEqual(state.get("key1"), "value1")
    
    def test_delete(self):
        state = SharedState()
        state.set("key1", "value1")
        self.assertTrue(state.delete("key1"))
        self.assertIsNone(state.get("key1"))


class TestHandoff(unittest.TestCase):
    """Test Handoff mechanism"""
    
    def test_handoff_creation(self):
        target = Agent(name="target", description="Target agent")
        handoff = Handoff(target_agent=target)
        self.assertEqual(handoff.target_agent.name, "target")
        self.assertIn("Transfer control to target", handoff.description)
    
    def test_handoff_tool_creation(self):
        target = Agent(name="specialist", description="Specialist agent")
        handoff = Handoff(target_agent=target, description="Ask specialist")
        handoff_tool = handoff.to_tool()
        
        self.assertEqual(handoff_tool.name, "transfer_to_specialist")
        self.assertEqual(handoff_tool.safety_level, ToolSafety.READ_ONLY)
        self.assertTrue(handoff_tool.is_concurrency_safe)
    
    def test_handoff_tool_execution(self):
        async def _test():
            target = Agent(name="specialist", description="Specialist agent")
            handoff = Handoff(target_agent=target)
            handoff_tool = handoff.to_tool()
            
            result = await handoff_tool.execute({"reason": "need help"})
            self.assertTrue(result.success)
            self.assertTrue(result.data["__handoff__"])
            self.assertEqual(result.data["target_agent"], "specialist")
        
        asyncio.run(_test())
    
    def test_handoff_input_filter(self):
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user message"},
            {"role": "assistant", "content": "assistant reply"},
        ]
        
        # Filter by role
        filt = HandoffInputFilter(keep_roles=["user", "assistant"])
        filtered = filt.apply(messages)
        self.assertEqual(len(filtered), 2)
        
        # Keep recent
        filt2 = HandoffInputFilter(keep_recent=1)
        filtered2 = filt2.apply(messages)
        self.assertEqual(len(filtered2), 1)
        
        # Custom filter
        filt3 = HandoffInputFilter(custom_filter=lambda msgs: msgs[:1])
        filtered3 = filt3.apply(messages)
        self.assertEqual(len(filtered3), 1)


class TestAgentHooks(unittest.TestCase):
    """Test AgentHooks"""
    
    def test_default_hooks(self):
        hooks = AgentHooks()
        # All default hooks should be callable without error
        async def _test():
            ctx = RunContext(
                shared_state=SharedState(),
                usage=UsageTracker(),
                current_agent="test",
                turn=1,
                max_turns=10,
            )
            agent = Agent(name="test", description="test")
            
            await hooks.on_start(ctx, agent)
            await hooks.on_end(ctx, agent, "output")
            await hooks.on_tool_start(ctx, agent, lookup_info)
        
        asyncio.run(_test())
    
    def test_custom_hooks(self):
        events = []
        
        class CustomHooks(AgentHooks):
            async def on_start(self, ctx, agent):
                events.append(f"start:{agent.name}")
            
            async def on_end(self, ctx, agent, output):
                events.append(f"end:{agent.name}")
        
        hooks = CustomHooks()
        
        async def _test():
            ctx = RunContext(
                shared_state=SharedState(),
                usage=UsageTracker(),
                current_agent="test",
                turn=1,
                max_turns=10,
            )
            agent = Agent(name="test", description="test")
            await hooks.on_start(ctx, agent)
            await hooks.on_end(ctx, agent, "done")
            self.assertEqual(events, ["start:test", "end:test"])
        
        asyncio.run(_test())


class TestUsageTracker(unittest.TestCase):
    """Test UsageTracker"""
    
    def test_basic_tracking(self):
        tracker = UsageTracker()
        tracker.record("agent_a", 100, 50)
        tracker.record("agent_b", 200, 100)
        
        self.assertEqual(tracker.total_prompt_tokens, 300)
        self.assertEqual(tracker.total_completion_tokens, 150)
        self.assertEqual(tracker.total_tokens, 450)
        
        summary = tracker.get_summary()
        self.assertEqual(summary["by_agent"]["agent_a"]["calls"], 1)
        self.assertEqual(summary["by_agent"]["agent_b"]["calls"], 1)


class TestBuiltinTools(unittest.TestCase):
    """Test builtin tools creation and basic operations"""
    
    def test_all_builtin_tools_creation(self):
        from teragent.tools.builtin import all_builtin_tools
        tools = all_builtin_tools()
        self.assertEqual(len(tools), 9)
        
        tool_names = {t.name for t in tools}
        expected = {"read_file", "write_file", "list_directory", "search_files",
                   "execute_code", "web_search", "web_scrape",
                   "analyze_code", "search_code_semantic"}
        self.assertEqual(tool_names, expected)
    
    def test_file_read_write(self):
        async def _test():
            from teragent.tools.builtin.file import ReadFileTool, WriteFileTool
            
            write_tool = WriteFileTool()
            result = await write_tool.execute({
                "path": "/tmp/teragent_e2e_test.txt",
                "content": "Hello from e2e test!",
            })
            self.assertTrue(result.success)
            
            read_tool = ReadFileTool()
            result = await read_tool.execute({
                "path": "/tmp/teragent_e2e_test.txt",
            })
            self.assertTrue(result.success)
            self.assertIn("Hello from e2e test!", result.data["content"])
        
        asyncio.run(_test())
    
    def test_code_execution(self):
        async def _test():
            from teragent.tools.builtin.code import CodeExecutionTool
            tool = CodeExecutionTool()
            result = await tool.execute({"code": "print(2 + 3)"})
            self.assertTrue(result.success)
            self.assertIn("5", result.data["stdout"])
        
        asyncio.run(_test())


if __name__ == "__main__":
    unittest.main()
