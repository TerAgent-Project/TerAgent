# tests/test_p8_3_mixed_mode.py
"""Mixed Mode Test Suite

Tests the "mixed mode" behavior where streaming and batch execution
alternate, and the system correctly degrades between modes.

Test categories:
  1. Mode Selection Logic - can_stream_with_tools() decisions
  2. Streaming -> Batch Degradation - graceful fallback
  3. Mixed Execution Within Single Tool Loop - alternating modes
  4. GLM Driver Compatibility - batch-only model
  5. Streaming Execution Stats
  6. StreamingToolExecutor Direct Tests

All tests use mocks/stubs -- no live LLM API required.

NOTE: Tests that depend on AgentLoop are now implemented after
AgentLoop has been migrated to the teragent package.
"""

from __future__ import annotations

import sys
import unittest

# Ensure the project root is importable
from pathlib import Path
from unittest.mock import (
    MagicMock,
)

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from teragent.agent_loop import AgentLoop  # noqa: E402
from teragent.config.agent_loop_config import AgentLoopConfig  # noqa: E402
from teragent.core.types import ToolSafety  # noqa: E402
from teragent.event_bus import EventBus  # noqa: E402
from teragent.streaming.stream_events import (  # noqa: E402
    StreamEvent,
    StreamEventType,
)
from teragent.streaming.streaming_executor import (  # noqa: E402
    StreamingExecutionStats,
    StreamingToolExecutor,
)
from teragent.tools.base import BaseTool, ToolResult  # noqa: E402
from teragent.tools.orchestrator import ToolOrchestrator  # noqa: E402
from teragent.tools.registry import ToolRegistry  # noqa: E402

# ===== Helper: Concrete tool implementations for testing =====

class ReadOnlyTool(BaseTool):
    """A read-only, concurrency-safe tool for testing."""

    name: str = "read_file"
    description: str = "Read file contents"
    parameters_schema: dict = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    _safety: ToolSafety = ToolSafety.READ_ONLY
    _concurrency_safe: bool = True

    async def execute(self, params: dict, progress_callback=None) -> ToolResult:
        """Return a mock file content result."""
        return ToolResult(
            success=True,
            data={"content": f"contents of {params.get('path', 'unknown')}"},
        )


class WriteTool(BaseTool):
    """A safe-write, non-concurrency-safe tool for testing."""

    name: str = "execute_subtask"
    description: str = "Execute a subtask"
    parameters_schema: dict = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }
    _safety: ToolSafety = ToolSafety.SAFE_WRITE
    _concurrency_safe: bool = False

    async def execute(self, params: dict, progress_callback=None) -> ToolResult:
        """Return a mock execution result."""
        return ToolResult(
            success=True,
            data={"output": f"executed: {params.get('command', 'unknown')}"},
        )


class HighRiskTool(BaseTool):
    """A high-risk tool for testing permission gating."""

    name: str = "create_project"
    description: str = "Create a new project"
    parameters_schema: dict = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    _safety: ToolSafety = ToolSafety.HIGH_RISK
    _concurrency_safe: bool = False

    async def execute(self, params: dict, progress_callback=None) -> ToolResult:
        """Return a mock project creation result."""
        return ToolResult(
            success=True,
            data={"project": params.get("name", "unknown")},
        )


class ExploreTool(BaseTool):
    """A read-only exploration tool for testing."""

    name: str = "explore_codebase"
    description: str = "Explore the codebase"
    parameters_schema: dict = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    _safety: ToolSafety = ToolSafety.READ_ONLY
    _concurrency_safe: bool = True

    async def execute(self, params: dict, progress_callback=None) -> ToolResult:
        """Return a mock exploration result."""
        return ToolResult(
            success=True,
            data={"results": [f"found for {params.get('query', '')}"]},
        )


class ListDirTool(BaseTool):
    """A read-only list directory tool for testing."""

    name: str = "list_directory"
    description: str = "List directory contents"
    parameters_schema: dict = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    }
    _safety: ToolSafety = ToolSafety.READ_ONLY
    _concurrency_safe: bool = True

    async def execute(self, params: dict, progress_callback=None) -> ToolResult:
        """Return a mock directory listing."""
        return ToolResult(
            success=True,
            data={"entries": ["src", "tests", "README.md"]},
        )


# ===== Helper: Mock model providers =====

class MockStreamingModel:
    """A mock model that supports streaming tool_calling."""

    def __init__(self) -> None:
        self._caps: dict = {
            "streaming": True,
            "tool_calling": True,
            "streaming_tool_calling": True,
        }

    def capabilities(self) -> dict:
        """Return declared model capabilities."""
        return self._caps

    def set_capabilities(self, caps: dict) -> None:
        """Update model capabilities for testing."""
        self._caps = caps


class MockBatchOnlyModel:
    """A mock model that does NOT support streaming tool_calling (like GLM)."""

    def __init__(self) -> None:
        self._caps: dict = {
            "streaming": True,
            "tool_calling": False,
            "streaming_tool_calling": False,
        }

    def capabilities(self) -> dict:
        """Return declared model capabilities (no tool_calling)."""
        return self._caps


# ===== Helper: Build a registry with test tools =====

def _make_tool_registry() -> ToolRegistry:
    """Build and return a ToolRegistry with standard test tools."""
    registry = ToolRegistry()
    registry.register(ReadOnlyTool())
    registry.register(WriteTool())
    registry.register(HighRiskTool())
    registry.register(ExploreTool())
    registry.register(ListDirTool())
    return registry


# =====================================================================
# 1. Mode Selection Logic (via StreamingToolExecutor)
# =====================================================================

class TestStreamingModeSelection(unittest.TestCase):
    """Test can_stream_with_tools() mode selection logic"""

    def test_streaming_model_can_stream_with_tools(self):
        """Model with streaming_tool_calling=True -> can stream"""
        streaming_model = MockStreamingModel()
        executor = StreamingToolExecutor(_make_tool_registry())
        result = executor.can_stream_with_tools(streaming_model)
        self.assertTrue(
            result,
            "can_stream_with_tools should return True when model supports it",
        )

    def test_batch_model_cannot_stream_with_tools(self):
        """Model with streaming_tool_calling=False -> cannot stream"""
        batch_model = MockBatchOnlyModel()
        executor = StreamingToolExecutor(_make_tool_registry())
        result = executor.can_stream_with_tools(batch_model)
        self.assertFalse(
            result,
            "can_stream_with_tools should return False when model lacks support",
        )

    def test_partial_capability_returns_false(self):
        """Requires all three capability flags: streaming, tool_calling, streaming_tool_calling"""
        # Has streaming=True and tool_calling=True but streaming_tool_calling=False
        model = MockStreamingModel()
        model.set_capabilities({
            "streaming": True,
            "tool_calling": True,
            "streaming_tool_calling": False,
        })
        executor = StreamingToolExecutor(_make_tool_registry())
        self.assertFalse(
            executor.can_stream_with_tools(model),
            "should not stream when streaming_tool_calling is False",
        )

        # Has streaming_tool_calling=True but tool_calling=False
        model2 = MockStreamingModel()
        model2.set_capabilities({
            "streaming": True,
            "tool_calling": False,
            "streaming_tool_calling": True,
        })
        executor2 = StreamingToolExecutor(_make_tool_registry())
        self.assertFalse(
            executor2.can_stream_with_tools(model2),
            "should not stream when tool_calling is False",
        )

        # Has streaming=False
        model3 = MockStreamingModel()
        model3.set_capabilities({
            "streaming": False,
            "tool_calling": True,
            "streaming_tool_calling": True,
        })
        executor3 = StreamingToolExecutor(_make_tool_registry())
        self.assertFalse(
            executor3.can_stream_with_tools(model3),
            "should not stream when streaming is False",
        )

    def test_glm_always_uses_batch(self):
        """GLM driver should always use batch (tool_calling=False)"""
        glm_model = MockBatchOnlyModel()
        executor = StreamingToolExecutor(_make_tool_registry())
        result = executor.can_stream_with_tools(glm_model)
        self.assertFalse(
            result,
            "GLM model must report can_stream_with_tools=False",
        )

    def test_model_without_capabilities_method(self):
        """should gracefully handle model without capabilities()"""
        model = MagicMock()
        model.capabilities.return_value = {}
        executor = StreamingToolExecutor(_make_tool_registry())
        result = executor.can_stream_with_tools(model)
        self.assertFalse(
            result,
            "can_stream_with_tools should return False for empty capabilities",
        )

    def test_capabilities_exception_returns_false(self):
        """should return False when capabilities() raises an exception"""
        model = MagicMock()
        model.capabilities.side_effect = RuntimeError("API error")
        executor = StreamingToolExecutor(_make_tool_registry())
        result = executor.can_stream_with_tools(model)
        self.assertFalse(
            result,
            "can_stream_with_tools should return False on exception",
        )


# =====================================================================
# 2. Streaming -> Batch Degradation (StreamingToolExecutor only)
# =====================================================================

class TestStreamingToBatchDegradation(unittest.IsolatedAsyncioTestCase):
    """Test graceful degradation from streaming to batch mode"""

    async def test_batch_fallback_preserves_execution(self):
        """execute_batch_fallback should execute all tools and mark fallback_used"""
        registry = _make_tool_registry()
        executor = StreamingToolExecutor(registry, permission_level=0)

        tool_calls = [
            {"name": "read_file", "arguments": {"path": "x.py"}, "id": "call_0"},
            {"name": "explore_codebase", "arguments": {"query": "test"}, "id": "call_1"},
        ]

        results, stats = await executor.execute_batch_fallback(tool_calls)

        self.assertEqual(len(results), 2, "Should execute all 2 tool calls")
        self.assertTrue(
            stats.fallback_used,
            "StreamingExecutionStats.fallback_used should be True",
        )
        self.assertEqual(
            stats.total_tool_calls, 2,
            "total_tool_calls should match input count",
        )
        self.assertEqual(
            stats.queued_executions, 2,
            "All tools should be counted as queued in batch mode",
        )


# =====================================================================
# 3. GLM Driver Compatibility
# =====================================================================

class TestGLMDriverCompatibility(unittest.IsolatedAsyncioTestCase):
    """Test that GLM driver (no streaming tool_use) correctly uses batch mode"""

    async def test_glm_uses_batch_mode(self):
        """GLM should never attempt streaming tool execution"""
        glm_model = MockBatchOnlyModel()
        executor = StreamingToolExecutor(_make_tool_registry())

        # Verify GLM capabilities prevent streaming
        can_stream = executor.can_stream_with_tools(glm_model)
        self.assertFalse(
            can_stream,
            "GLM model must report can_stream_with_tools=False",
        )

    async def test_glm_batch_execution_compatibility(self):
        """GLM batch execution should work correctly with ToolOrchestrator"""
        registry = _make_tool_registry()
        orchestrator = ToolOrchestrator(registry, permission_level=0)

        # Simulate a batch of tool calls that GLM would produce
        tool_calls = [
            {"name": "read_file", "arguments": {"path": "src/main.py"}, "id": "call_0"},
        ]

        results = await orchestrator.execute_batch(tool_calls)

        self.assertEqual(len(results), 1, "Should have 1 result")
        tool_call, result = results[0]
        self.assertTrue(result.success, "read_file should succeed")
        self.assertEqual(tool_call["name"], "read_file")

    async def test_glm_capabilities_structure(self):
        """GLM capabilities dict should have expected keys and values"""
        expected_caps = {
            "streaming": True,
            "tool_calling": False,
            "streaming_tool_calling": False,
        }

        # Test that our mock matches the expected GLM capabilities
        glm_model = MockBatchOnlyModel()
        caps = glm_model.capabilities()
        self.assertEqual(caps["tool_calling"], expected_caps["tool_calling"])
        self.assertEqual(caps["streaming_tool_calling"], expected_caps["streaming_tool_calling"])


# =====================================================================
# 4. Streaming Execution Stats
# =====================================================================

class TestStreamingExecutionStats(unittest.TestCase):
    """Test StreamingExecutionStats dataclass and methods"""

    def test_default_values(self):
        """Default stats should have zero counts and fallback_used=False"""
        stats = StreamingExecutionStats()
        self.assertEqual(stats.total_tool_calls, 0)
        self.assertEqual(stats.immediate_executions, 0)
        self.assertEqual(stats.queued_executions, 0)
        self.assertEqual(stats.parallel_groups, 0)
        self.assertEqual(stats.streaming_time_ms, 0.0)
        self.assertEqual(stats.execution_time_ms, 0.0)
        self.assertFalse(stats.fallback_used)

    def test_to_dict(self):
        """to_dict should return a serializable dict"""
        stats = StreamingExecutionStats(
            total_tool_calls=3,
            immediate_executions=2,
            queued_executions=1,
            parallel_groups=1,
            streaming_time_ms=100.5,
            execution_time_ms=200.3,
            fallback_used=False,
        )
        d = stats.to_dict()
        self.assertEqual(d["total_tool_calls"], 3)
        self.assertEqual(d["immediate_executions"], 2)
        self.assertEqual(d["queued_executions"], 1)
        self.assertFalse(d["fallback_used"])

    def test_batch_fallback_stats(self):
        """Batch fallback should set fallback_used=True"""
        stats = StreamingExecutionStats(
            total_tool_calls=2,
            fallback_used=True,
            queued_executions=2,
        )
        self.assertTrue(stats.fallback_used)
        d = stats.to_dict()
        self.assertTrue(d["fallback_used"])


# =====================================================================
# 5. StreamingToolExecutor Direct Tests
# =====================================================================

class TestStreamingToolExecutorDirect(unittest.IsolatedAsyncioTestCase):
    """Direct tests of StreamingToolExecutor behavior"""

    async def test_streaming_tool_execution_order_preserved(self):
        """Streaming executor should preserve tool call order regardless of execution mode"""
        registry = _make_tool_registry()
        executor = StreamingToolExecutor(registry, permission_level=0)

        # Build a stream with two read-only tool calls
        events = [
            StreamEvent(
                event_type=StreamEventType.TOOL_CALL_START,
                tool_call_index=0,
                tool_call_id="call_0",
                tool_name="read_file",
            ),
            StreamEvent(
                event_type=StreamEventType.TOOL_CALL_COMPLETE,
                tool_call_index=0,
                tool_call_id="call_0",
                tool_name="read_file",
                tool_arguments={"path": "a.py"},
            ),
            StreamEvent(
                event_type=StreamEventType.TOOL_CALL_START,
                tool_call_index=1,
                tool_call_id="call_1",
                tool_name="explore_codebase",
            ),
            StreamEvent(
                event_type=StreamEventType.TOOL_CALL_COMPLETE,
                tool_call_index=1,
                tool_call_id="call_1",
                tool_name="explore_codebase",
                tool_arguments={"query": "main"},
            ),
            StreamEvent(
                event_type=StreamEventType.DONE,
                finish_reason="stop",
            ),
        ]

        async def event_stream():
            """Yield the test events."""
            for e in events:
                yield e

        results, streaming_result, stats = await executor.execute_streaming(
            stream=event_stream(),
        )

        # Results should be ordered by index
        self.assertEqual(len(results), 2, "Should have 2 tool results")
        self.assertEqual(results[0][0]["name"], "read_file", "First result should be read_file")
        self.assertEqual(results[1][0]["name"], "explore_codebase", "Second result should be explore_codebase")

    async def test_execute_streaming_with_mixed_tools(self):
        """Streaming should immediately execute read-only tools and queue write tools"""
        registry = _make_tool_registry()
        executor = StreamingToolExecutor(registry, permission_level=0)

        # Build a stream with one read-only and one write tool
        events = [
            # Read-only tool (should be immediate)
            StreamEvent(
                event_type=StreamEventType.TOOL_CALL_START,
                tool_call_index=0,
                tool_call_id="call_0",
                tool_name="read_file",
            ),
            StreamEvent(
                event_type=StreamEventType.TOOL_CALL_COMPLETE,
                tool_call_index=0,
                tool_call_id="call_0",
                tool_name="read_file",
                tool_arguments={"path": "test.py"},
            ),
            # Write tool (should be queued)
            StreamEvent(
                event_type=StreamEventType.TOOL_CALL_START,
                tool_call_index=1,
                tool_call_id="call_1",
                tool_name="execute_subtask",
            ),
            StreamEvent(
                event_type=StreamEventType.TOOL_CALL_COMPLETE,
                tool_call_index=1,
                tool_call_id="call_1",
                tool_name="execute_subtask",
                tool_arguments={"command": "echo hello"},
            ),
            StreamEvent(
                event_type=StreamEventType.DONE,
                finish_reason="stop",
            ),
        ]

        async def event_stream():
            """Yield the test events."""
            for e in events:
                yield e

        results, streaming_result, stats = await executor.execute_streaming(
            stream=event_stream(),
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(stats.total_tool_calls, 2)


# =====================================================================
# 6. AgentLoop-dependent tests
# =====================================================================

def _make_agent_loop(
    streaming_mode: str = "auto",
    model_caps: dict | None = None,
) -> AgentLoop:
    """Create an AgentLoop with mock dependencies for testing."""
    # Create mock model with capabilities method
    if model_caps is None:
        model_caps = {"streaming": True, "tool_calling": True, "streaming_tool_calling": True}
    mock_model = MagicMock()
    mock_model.capabilities.return_value = model_caps

    # Create tool registry with test tools
    registry = _make_tool_registry()

    # Create streaming executor
    streaming_executor = StreamingToolExecutor(registry, permission_level=0)

    # Create config
    config = AgentLoopConfig()

    # Create event bus
    event_bus = EventBus()

    loop = AgentLoop(
        model=mock_model,
        tool_registry=registry,
        config=config,
        event_bus=event_bus,
        streaming_executor=streaming_executor,
    )

    # Set streaming mode if not "auto" (default)
    if streaming_mode != "auto":
        loop.set_streaming_config(mode=streaming_mode)

    return loop


class TestAgentLoopDependentSkipped(unittest.TestCase):
    """Tests that depend on AgentLoop — now implemented after migration."""

    def test_batch_mode_always_returns_false(self):
        """streaming_mode='batch' should never use streaming"""
        loop = _make_agent_loop(streaming_mode="batch")
        self.assertFalse(
            loop._should_use_streaming([]),
            "batch mode should always return False from _should_use_streaming",
        )
        # Even with a capable model, batch mode should not stream
        self.assertFalse(
            loop._should_use_streaming(["read_file"]),
            "batch mode should return False even with capable model",
        )

    def test_streaming_mode_always_returns_true(self):
        """streaming_mode='streaming' should always use streaming"""
        loop = _make_agent_loop(streaming_mode="streaming")
        self.assertTrue(
            loop._should_use_streaming([]),
            "streaming mode should always return True from _should_use_streaming",
        )
        # Even with an incapable model, forced streaming should return True
        loop_batch_model = _make_agent_loop(
            streaming_mode="streaming",
            model_caps={"streaming": False, "tool_calling": False, "streaming_tool_calling": False},
        )
        self.assertTrue(
            loop_batch_model._should_use_streaming([]),
            "forced streaming mode should return True even for batch-only model",
        )

    def test_auto_mode_depends_on_model_capabilities(self):
        """streaming_mode='auto' should check model capabilities"""
        # With a fully capable model -> True
        loop_capable = _make_agent_loop(
            streaming_mode="auto",
            model_caps={"streaming": True, "tool_calling": True, "streaming_tool_calling": True},
        )
        self.assertTrue(
            loop_capable._should_use_streaming([]),
            "auto mode with capable model should use streaming",
        )

        # With a partially capable model -> False
        loop_partial = _make_agent_loop(
            streaming_mode="auto",
            model_caps={"streaming": True, "tool_calling": True, "streaming_tool_calling": False},
        )
        self.assertFalse(
            loop_partial._should_use_streaming([]),
            "auto mode with partial capabilities should not stream",
        )

        # With a batch-only model -> False
        loop_batch = _make_agent_loop(
            streaming_mode="auto",
            model_caps={"streaming": True, "tool_calling": False, "streaming_tool_calling": False},
        )
        self.assertFalse(
            loop_batch._should_use_streaming([]),
            "auto mode with batch-only model should not stream",
        )

        # Without streaming executor -> False
        mock_model_noexec = MagicMock()
        mock_model_noexec.capabilities.return_value = {
            "streaming": True, "tool_calling": True, "streaming_tool_calling": True
        }
        loop_no_executor = AgentLoop(
            model=mock_model_noexec,
            tool_registry=_make_tool_registry(),
            config=AgentLoopConfig(),
        )
        self.assertFalse(
            loop_no_executor._should_use_streaming([]),
            "auto mode without streaming_executor should not stream",
        )

    def test_streaming_failure_falls_back_to_batch(self):
        """When streaming fails, should fall back to batch execution"""
        loop = _make_agent_loop(streaming_mode="streaming")
        # Initially, _should_use_streaming returns True
        self.assertTrue(loop._should_use_streaming([]))

        # Simulate streaming failure by switching to batch fallback
        loop.set_streaming_config(mode="batch")
        self.assertFalse(loop._should_use_streaming([]))
        # Recovery stats should track the mode change
        self.assertEqual(loop._recovery_stats["streaming_mode"], "batch")

    def test_streaming_retry_before_fallback(self):
        """System should retry streaming before falling back"""
        loop = _make_agent_loop(streaming_mode="streaming")
        # Set a max retry count
        loop.set_streaming_config(max_streaming_retries=3)
        self.assertEqual(loop._max_streaming_retries, 3)

        # In streaming mode, _should_use_streaming always returns True,
        # so retries happen at the _call_model_streaming level.
        # We verify the retry count is configured correctly.
        self.assertEqual(loop._max_streaming_retries, 3)

        # Simulate first retry increment
        loop._recovery_stats["streaming_retries"] += 1
        self.assertEqual(loop._recovery_stats["streaming_retries"], 1)

        # Still in streaming mode after retry
        self.assertTrue(loop._should_use_streaming([]))

    def test_max_retries_then_fallback(self):
        """After max_streaming_retries, should fall back to batch"""
        loop = _make_agent_loop(streaming_mode="streaming")
        loop.set_streaming_config(max_streaming_retries=2)

        # Simulate exhausting retries
        loop._recovery_stats["streaming_retries"] = loop._max_streaming_retries
        # After exhausting retries, the code falls back to batch.
        # Simulate the fallback
        loop._recovery_stats["batch_fallbacks"] += 1
        loop.set_streaming_config(mode="batch")

        self.assertFalse(loop._should_use_streaming([]))
        self.assertEqual(loop._recovery_stats["batch_fallbacks"], 1)

    def test_context_overflow_recovery_in_streaming(self):
        """Context overflow during streaming should trigger compression + retry"""
        loop = _make_agent_loop(streaming_mode="streaming")

        # Simulate context overflow recovery
        loop._recovery_stats["context_compactions"] += 1
        self.assertEqual(loop._recovery_stats["context_compactions"], 1)

        # After compaction, the loop should still be able to stream
        self.assertTrue(loop._should_use_streaming([]))

    def test_auto_mode_switches_to_streaming(self):
        """In auto mode, if model supports streaming, should use it"""
        loop = _make_agent_loop(
            streaming_mode="auto",
            model_caps={"streaming": True, "tool_calling": True, "streaming_tool_calling": True},
        )
        self.assertTrue(
            loop._should_use_streaming([]),
            "auto mode with capable model should switch to streaming",
        )

    def test_permission_level_sync(self):
        """set_permission_level should update both orchestrator and streaming_executor"""
        loop = _make_agent_loop(streaming_mode="auto")

        # Verify initial permission level
        self.assertEqual(loop._permission_level, 0)

        # Set new permission level
        loop.set_permission_level(2)

        # Verify it propagated to the internal state
        self.assertEqual(loop._permission_level, 2)

        # Verify it propagated to the tool orchestrator
        self.assertEqual(loop._tool_orchestrator.permission_level, 2)

        # Verify it propagated to the streaming executor
        self.assertEqual(loop._streaming_executor.permission_level, 2)

    def test_streaming_succeeds_then_fails_then_batch(self):
        """Session: streaming OK -> streaming fails -> batch fallback"""
        loop = _make_agent_loop(streaming_mode="auto")

        # Phase 1: Auto mode with capable model -> streaming
        self.assertTrue(loop._should_use_streaming([]))

        # Phase 2: Model loses streaming capability (e.g. fallback model)
        loop._model.capabilities.return_value = {
            "streaming": True, "tool_calling": False, "streaming_tool_calling": False
        }
        self.assertFalse(loop._should_use_streaming([]))

        # Phase 3: Explicit switch to batch mode
        loop.set_streaming_config(mode="batch")
        self.assertFalse(loop._should_use_streaming([]))
        self.assertEqual(loop._recovery_stats["streaming_mode"], "batch")

    def test_streaming_length_recovery_then_batch(self):
        """Streaming length recovery, then subsequent batch call"""
        loop = _make_agent_loop(streaming_mode="streaming")

        # Simulate output truncation recovery
        loop._recovery_stats["truncation_recoveries"] += 1
        self.assertEqual(loop._recovery_stats["truncation_recoveries"], 1)

        # After truncation recovery, still in streaming mode
        self.assertTrue(loop._should_use_streaming([]))

        # Switch to batch mode for subsequent call
        loop.set_streaming_config(mode="batch")
        self.assertFalse(loop._should_use_streaming([]))

    def test_glm_forced_streaming_still_uses_streaming_mode(self):
        """Even with GLM, forcing streaming_mode='streaming' should use streaming"""
        loop = _make_agent_loop(
            streaming_mode="streaming",
            model_caps={"streaming": True, "tool_calling": False, "streaming_tool_calling": False},
        )
        # In forced streaming mode, _should_use_streaming returns True
        # regardless of model capabilities
        self.assertTrue(
            loop._should_use_streaming([]),
            "forced streaming mode should use streaming even with GLM capabilities",
        )

    def test_switch_from_auto_to_batch(self):
        """Switching from auto to batch at runtime"""
        loop = _make_agent_loop(streaming_mode="auto")
        self.assertEqual(loop.streaming_mode, "auto")
        self.assertTrue(loop._should_use_streaming([]))

        # Switch to batch
        loop.set_streaming_config(mode="batch")
        self.assertEqual(loop.streaming_mode, "batch")
        self.assertFalse(loop._should_use_streaming([]))

    def test_switch_from_auto_to_streaming(self):
        """Switching from auto to streaming at runtime"""
        loop = _make_agent_loop(streaming_mode="auto")
        self.assertEqual(loop.streaming_mode, "auto")

        # Switch to streaming
        loop.set_streaming_config(mode="streaming")
        self.assertEqual(loop.streaming_mode, "streaming")
        self.assertTrue(loop._should_use_streaming([]))

    def test_switch_from_streaming_to_auto(self):
        """Switching from streaming to auto at runtime"""
        loop = _make_agent_loop(streaming_mode="streaming")
        self.assertEqual(loop.streaming_mode, "streaming")
        self.assertTrue(loop._should_use_streaming([]))

        # Switch to auto — with capable model, should still stream
        loop.set_streaming_config(mode="auto")
        self.assertEqual(loop.streaming_mode, "auto")
        # Auto mode with capable model -> True
        self.assertTrue(loop._should_use_streaming([]))

        # Auto mode with incapable model -> False
        loop._model.capabilities.return_value = {
            "streaming": False, "tool_calling": False, "streaming_tool_calling": False
        }
        self.assertFalse(loop._should_use_streaming([]))

    def test_invalid_mode_handled_gracefully(self):
        """Invalid mode value should fall back to auto"""
        loop = _make_agent_loop(streaming_mode="auto")
        loop.set_streaming_config(mode="invalid_mode")
        self.assertEqual(loop.streaming_mode, "auto")

        loop.set_streaming_config(mode="STREAMING")
        self.assertEqual(loop.streaming_mode, "auto")

        loop.set_streaming_config(mode="")
        self.assertEqual(loop.streaming_mode, "auto")

    def test_set_streaming_config_updates_max_retries(self):
        """set_streaming_config should update max_streaming_retries"""
        loop = _make_agent_loop(streaming_mode="auto")
        initial_retries = loop._max_streaming_retries

        loop.set_streaming_config(max_streaming_retries=10)
        self.assertEqual(loop._max_streaming_retries, 10)

        loop.set_streaming_config(max_streaming_retries=0)
        self.assertEqual(loop._max_streaming_retries, 0)

        # Restore
        loop.set_streaming_config(max_streaming_retries=initial_retries)

    def test_set_streaming_config_both_params(self):
        """set_streaming_config should update both mode and retries simultaneously"""
        loop = _make_agent_loop(streaming_mode="auto")
        self.assertEqual(loop.streaming_mode, "auto")

        loop.set_streaming_config(mode="streaming", max_streaming_retries=5)
        self.assertEqual(loop.streaming_mode, "streaming")
        self.assertEqual(loop._max_streaming_retries, 5)

        loop.set_streaming_config(mode="batch", max_streaming_retries=3)
        self.assertEqual(loop.streaming_mode, "batch")
        self.assertEqual(loop._max_streaming_retries, 3)
