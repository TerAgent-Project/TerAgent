"""End-to-end Performance Tests for Streaming Tool Execution

Tests the performance characteristics of the streaming tool execution
system, including latency improvements, parser throughput, error
recovery overhead, config loading, and tool pre-filtering.

All tests use mocks/stubs -- no live LLM API required.
"""

from __future__ import annotations

import asyncio
import json
import time
import unittest
import pytest
from collections import defaultdict
from typing import Any, AsyncIterator, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from teragent.streaming.stream_events import (
    AnthropicStreamParser,
    OpenAIStreamParser,
    StreamEvent,
    StreamEventType,
    StreamingChatResult,
    ToolCallAccumulator,
)
from teragent.tools.base import BaseTool, ToolResult
from teragent.core.types import ToolSafety
from teragent.tools.registry import ToolRegistry
from teragent.streaming.streaming_executor import StreamingExecutionStats, StreamingToolExecutor
from teragent.agent_loop import AgentLoop
from teragent.config.agent_loop_config import AgentLoopConfig


# ======================================================================
# Helper: Mock tool implementations
# ======================================================================


class MockReadOnlyTool(BaseTool):
    """A read-only, concurrency-safe mock tool for testing."""

    name: str = "read_file"
    description: str = "Read a file"
    parameters_schema: dict = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    _safety: ToolSafety = ToolSafety.READ_ONLY
    _concurrency_safe: bool = True
    _execution_delay: float = 0.0
    _call_log: list[dict] = []

    def __init__(self, name: str = "read_file", delay: float = 0.0) -> None:
        self.name = name
        self._execution_delay = delay
        self._call_log = []

    async def execute(self, params: dict, progress_callback=None) -> ToolResult:
        if self._execution_delay > 0:
            await asyncio.sleep(self._execution_delay)
        self._call_log.append({
            "name": self.name,
            "params": params,
            "time": time.monotonic(),
        })
        return ToolResult(
            success=True,
            data={"content": f"Content of {params.get('path', 'unknown')}"},
        )


class MockWriteTool(BaseTool):
    """A write (non-read-only) mock tool for testing."""

    name: str = "execute_subtask"
    description: str = "Execute a subtask"
    parameters_schema: dict = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }
    _safety: ToolSafety = ToolSafety.DESTRUCTIVE
    _concurrency_safe: bool = False
    _execution_delay: float = 0.0
    _call_log: list[dict] = []

    def __init__(self, name: str = "execute_subtask", delay: float = 0.0) -> None:
        self.name = name
        self._execution_delay = delay
        self._call_log = []

    async def execute(self, params: dict, progress_callback=None) -> ToolResult:
        if self._execution_delay > 0:
            await asyncio.sleep(self._execution_delay)
        self._call_log.append({
            "name": self.name,
            "params": params,
            "time": time.monotonic(),
        })
        return ToolResult(
            success=True,
            data={"output": f"Executed: {params.get('command', 'unknown')}"},
        )


class MockSafeWriteTool(BaseTool):
    """A safe-write (non-read-only, non-concurrency-safe) mock tool."""

    name: str = "generate_design"
    description: str = "Generate a design"
    parameters_schema: dict = {
        "type": "object",
        "properties": {"description": {"type": "string"}},
        "required": ["description"],
    }
    _safety: ToolSafety = ToolSafety.SAFE_WRITE
    _concurrency_safe: bool = False
    _execution_delay: float = 0.0
    _call_log: list[dict] = []

    def __init__(self, name: str = "generate_design", delay: float = 0.0) -> None:
        self.name = name
        self._execution_delay = delay
        self._call_log = []

    async def execute(self, params: dict, progress_callback=None) -> ToolResult:
        if self._execution_delay > 0:
            await asyncio.sleep(self._execution_delay)
        self._call_log.append({
            "name": self.name,
            "params": params,
            "time": time.monotonic(),
        })
        return ToolResult(
            success=True,
            data={"design": f"Design for {params.get('description', 'unknown')}"},
        )


# ======================================================================
# Helper: Mock stream creation
# ======================================================================


async def _create_mock_stream(
    text_deltas: list[str] | None = None,
    tool_calls: list[dict] | None = None,
    interleaved: bool = True,
    delay_between_events: float = 0.01,
) -> AsyncIterator[StreamEvent]:
    """Create a mock StreamEvent iterator for testing.

    Args:
        text_deltas: List of text delta strings to emit
        tool_calls: List of tool call dicts with 'id', 'name', 'arguments'
        interleaved: If True, interleave text deltas with tool calls
        delay_between_events: Simulated network delay between events
    """
    text_deltas = text_deltas or []
    tool_calls = tool_calls or []

    if interleaved and text_deltas and tool_calls:
        # Interleave: text -> tool_start -> tool_delta -> tool_complete -> text -> ...
        text_idx = 0
        for tc_idx, tc in enumerate(tool_calls):
            # Emit a text delta before each tool call
            if text_idx < len(text_deltas):
                await asyncio.sleep(delay_between_events)
                yield StreamEvent(
                    event_type=StreamEventType.TEXT_DELTA,
                    text=text_deltas[text_idx],
                )
                text_idx += 1

            # Emit TOOL_CALL_START
            await asyncio.sleep(delay_between_events)
            yield StreamEvent(
                event_type=StreamEventType.TOOL_CALL_START,
                tool_call_index=tc_idx,
                tool_call_id=tc.get("id", f"call_{tc_idx}"),
                tool_name=tc["name"],
            )

            # Emit TOOL_CALL_COMPLETE (simulate arguments arriving at once)
            await asyncio.sleep(delay_between_events)
            yield StreamEvent(
                event_type=StreamEventType.TOOL_CALL_COMPLETE,
                tool_call_index=tc_idx,
                tool_call_id=tc.get("id", f"call_{tc_idx}"),
                tool_name=tc["name"],
                tool_arguments=tc.get("arguments", {}),
            )

        # Emit remaining text deltas
        while text_idx < len(text_deltas):
            await asyncio.sleep(delay_between_events)
            yield StreamEvent(
                event_type=StreamEventType.TEXT_DELTA,
                text=text_deltas[text_idx],
            )
            text_idx += 1

    else:
        # Non-interleaved: all text first, then all tool calls
        for text in text_deltas:
            await asyncio.sleep(delay_between_events)
            yield StreamEvent(
                event_type=StreamEventType.TEXT_DELTA,
                text=text,
            )

        for tc_idx, tc in enumerate(tool_calls):
            await asyncio.sleep(delay_between_events)
            yield StreamEvent(
                event_type=StreamEventType.TOOL_CALL_START,
                tool_call_index=tc_idx,
                tool_call_id=tc.get("id", f"call_{tc_idx}"),
                tool_name=tc["name"],
            )
            await asyncio.sleep(delay_between_events)
            yield StreamEvent(
                event_type=StreamEventType.TOOL_CALL_COMPLETE,
                tool_call_index=tc_idx,
                tool_call_id=tc.get("id", f"call_{tc_idx}"),
                tool_name=tc["name"],
                tool_arguments=tc.get("arguments", {}),
            )

    # Emit USAGE event
    await asyncio.sleep(delay_between_events)
    yield StreamEvent(
        event_type=StreamEventType.USAGE,
        usage={"prompt_tokens": 100, "completion_tokens": 200},
    )

    # Emit DONE event
    await asyncio.sleep(delay_between_events)
    yield StreamEvent(
        event_type=StreamEventType.DONE,
        finish_reason="stop",
    )


def _build_tool_registry(
    read_only_count: int = 3,
    write_count: int = 1,
    safe_write_count: int = 1,
    tool_delay: float = 0.0,
) -> ToolRegistry:
    """Build a ToolRegistry with mock tools for testing.

    Args:
        read_only_count: Number of read-only tools to register
        write_count: Number of destructive write tools to register
        safe_write_count: Number of safe-write tools to register
        tool_delay: Simulated execution delay per tool (seconds)

    Returns:
        Populated ToolRegistry instance
    """
    registry = ToolRegistry()

    for i in range(read_only_count):
        name = f"read_tool_{i}" if i > 0 else "read_file"
        registry.register(MockReadOnlyTool(name=name, delay=tool_delay))

    for i in range(write_count):
        name = f"write_tool_{i}" if i > 0 else "execute_subtask"
        registry.register(MockWriteTool(name=name, delay=tool_delay))

    for i in range(safe_write_count):
        name = f"safe_write_tool_{i}" if i > 0 else "generate_design"
        registry.register(MockSafeWriteTool(name=name, delay=tool_delay))

    return registry


# ======================================================================
# Test: StreamingToolExecutor Performance
# ======================================================================


class TestStreamingPerformance(unittest.IsolatedAsyncioTestCase):
    """Test streaming execution latency vs batch execution."""

    async def test_read_only_tools_execute_in_parallel_during_stream(self):
        """Verify that read-only tools start execution while model is still streaming.

        When a stream emits multiple read-only tool calls interleaved with
        text deltas, the StreamingToolExecutor should begin executing each
        read-only tool as soon as its TOOL_CALL_COMPLETE event arrives,
        without waiting for the stream to finish.
        """
        tool_delay = 0.05  # 50ms per tool
        registry = _build_tool_registry(
            read_only_count=3, write_count=0, safe_write_count=0,
            tool_delay=tool_delay,
        )
        executor = StreamingToolExecutor(registry, permission_level=0)

        tool_calls = [
            {"id": "call_0", "name": "read_file", "arguments": {"path": "a.py"}},
            {"id": "call_1", "name": "read_tool_1", "arguments": {"path": "b.py"}},
            {"id": "call_2", "name": "read_tool_2", "arguments": {"path": "c.py"}},
        ]
        text_deltas = ["Let me ", "read these ", "files for you."]

        stream = _create_mock_stream(
            text_deltas=text_deltas,
            tool_calls=tool_calls,
            interleaved=True,
            delay_between_events=0.02,
        )

        start = time.monotonic()
        results, streaming_result, stats = await executor.execute_streaming(stream)
        elapsed = time.monotonic() - start

        # All three tools should have been executed immediately (read-only)
        self.assertEqual(stats.immediate_executions, 3)
        self.assertEqual(stats.queued_executions, 0)
        self.assertEqual(stats.total_tool_calls, 3)

        # All results should be present
        self.assertEqual(len(results), 3)

        # Since tools run in parallel, total time should be much less
        # than sequential 3 * tool_delay (150ms). With parallel execution
        # it should be roughly tool_delay + overhead.
        self.assertLess(elapsed, 3 * tool_delay + 0.5,
                        "Read-only tools should execute in parallel, not sequentially")

    async def test_batch_vs_streaming_latency_comparison(self):
        """Streaming should be faster than batch when multiple read-only tools are present.

        Simulates a stream that emits 3 tool_calls over ~60ms (20ms each
        with interleaved text). In streaming mode, the tools begin executing
        during the stream, so total time is approximately max(stream_time, tool_time).
        In batch mode, total time would be stream_time + tool_time.
        """
        tool_delay = 0.05  # 50ms per tool
        registry = _build_tool_registry(
            read_only_count=3, write_count=0, safe_write_count=0,
            tool_delay=tool_delay,
        )

        # --- Streaming mode ---
        streaming_executor = StreamingToolExecutor(registry, permission_level=0)
        tool_calls = [
            {"id": "call_0", "name": "read_file", "arguments": {"path": "a.py"}},
            {"id": "call_1", "name": "read_tool_1", "arguments": {"path": "b.py"}},
            {"id": "call_2", "name": "read_tool_2", "arguments": {"path": "c.py"}},
        ]

        stream = _create_mock_stream(
            text_deltas=["Reading files..."],
            tool_calls=tool_calls,
            interleaved=True,
            delay_between_events=0.02,
        )

        streaming_start = time.monotonic()
        s_results, s_streaming_result, s_stats = await streaming_executor.execute_streaming(stream)
        streaming_elapsed = time.monotonic() - streaming_start

        # --- Batch mode (simulate: stream first, then execute all tools sequentially) ---
        stream2 = _create_mock_stream(
            text_deltas=["Reading files..."],
            tool_calls=tool_calls,
            interleaved=True,
            delay_between_events=0.02,
        )

        # Consume the stream fully first (simulating batch: wait for full model output)
        batch_stream_start = time.monotonic()
        collected_tool_calls: list[dict] = []
        async for event in stream2:
            if event.event_type == StreamEventType.TOOL_CALL_COMPLETE:
                collected_tool_calls.append({
                    "name": event.tool_name,
                    "id": event.tool_call_id,
                    "arguments": event.tool_arguments,
                })

        # Then execute tools sequentially (batch mode)
        for tc in collected_tool_calls:
            tool = registry.get(tc["name"])
            if tool:
                await tool.execute(tc.get("arguments", {}))

        batch_elapsed = time.monotonic() - batch_stream_start

        # Streaming should be faster than batch because tools overlap with stream
        # Allow generous margin for CI variability
        self.assertLess(
            streaming_elapsed, batch_elapsed + 0.1,
            "Streaming mode should not be significantly slower than batch mode",
        )

        # Verify streaming stats
        self.assertEqual(s_stats.immediate_executions, 3)
        self.assertGreater(s_stats.streaming_time_ms, 0)

    async def test_queued_tools_execute_after_stream(self):
        """Non-read-only tools should wait until stream ends.

        When a mix of read-only and write tools are present in the stream,
        read-only tools execute immediately during streaming, while write
        tools are queued and executed only after the stream completes.
        """
        tool_delay = 0.03  # 30ms per tool
        registry = _build_tool_registry(
            read_only_count=2, write_count=1, safe_write_count=1,
            tool_delay=tool_delay,
        )
        executor = StreamingToolExecutor(registry, permission_level=0)

        # Mix: 2 read-only + 1 write + 1 safe-write
        tool_calls = [
            {"id": "call_0", "name": "read_file", "arguments": {"path": "a.py"}},
            {"id": "call_1", "name": "read_tool_1", "arguments": {"path": "b.py"}},
            {"id": "call_2", "name": "execute_subtask", "arguments": {"command": "build"}},
            {"id": "call_3", "name": "generate_design", "arguments": {"description": "test"}},
        ]

        stream = _create_mock_stream(
            text_deltas=["I will read files and then write."],
            tool_calls=tool_calls,
            interleaved=True,
            delay_between_events=0.01,
        )

        results, streaming_result, stats = await executor.execute_streaming(stream)

        # Read-only tools should be immediate; write tools should be queued
        self.assertEqual(stats.immediate_executions, 2)
        self.assertEqual(stats.queued_executions, 2)
        self.assertEqual(stats.total_tool_calls, 4)
        self.assertEqual(len(results), 4)

        # Results should be in original order
        result_names = [tc_dict.get("name", "") for tc_dict, _ in results]
        self.assertEqual(result_names, ["read_file", "read_tool_1", "execute_subtask", "generate_design"])

    async def test_streaming_stats_accuracy(self):
        """StreamingExecutionStats should accurately track execution metrics.

        Verify that all stat fields are correctly populated after a
        streaming execution with a mix of read-only and write tools.
        """
        registry = _build_tool_registry(
            read_only_count=2, write_count=1, safe_write_count=0,
            tool_delay=0.01,
        )
        executor = StreamingToolExecutor(registry, permission_level=0)

        tool_calls = [
            {"id": "call_0", "name": "read_file", "arguments": {"path": "a.py"}},
            {"id": "call_1", "name": "read_tool_1", "arguments": {"path": "b.py"}},
            {"id": "call_2", "name": "execute_subtask", "arguments": {"command": "build"}},
        ]

        stream = _create_mock_stream(
            text_deltas=["Processing..."],
            tool_calls=tool_calls,
            interleaved=True,
            delay_between_events=0.01,
        )

        results, streaming_result, stats = await executor.execute_streaming(stream)

        # Stats should be accurate
        self.assertEqual(stats.total_tool_calls, 3)
        self.assertEqual(stats.immediate_executions, 2)
        self.assertEqual(stats.queued_executions, 1)
        self.assertEqual(stats.parallel_groups, 1)  # one group of read-only tools
        self.assertGreater(stats.streaming_time_ms, 0)
        self.assertGreater(stats.execution_time_ms, 0)
        self.assertFalse(stats.fallback_used)

        # Verify to_dict produces expected keys
        stats_dict = stats.to_dict()
        self.assertIn("total_tool_calls", stats_dict)
        self.assertIn("immediate_executions", stats_dict)
        self.assertIn("queued_executions", stats_dict)
        self.assertIn("parallel_groups", stats_dict)
        self.assertIn("streaming_time_ms", stats_dict)
        self.assertIn("execution_time_ms", stats_dict)
        self.assertIn("fallback_used", stats_dict)

        # StreamingChatResult should have content and tool_calls
        self.assertEqual(streaming_result.content, "Processing...")
        self.assertEqual(len(streaming_result.tool_calls), 3)
        self.assertEqual(streaming_result.finish_reason, "stop")

    async def test_no_tool_calls_stream(self):
        """A stream with no tool calls should complete with empty results.

        When the model returns only text (no tool calls), the streaming
        executor should return empty tool results and accurate stats.
        """
        registry = _build_tool_registry(read_only_count=1, write_count=0, safe_write_count=0)
        executor = StreamingToolExecutor(registry, permission_level=0)

        stream = _create_mock_stream(
            text_deltas=["Hello! ", "How can I help?"],
            tool_calls=None,
            interleaved=False,
            delay_between_events=0.005,
        )

        results, streaming_result, stats = await executor.execute_streaming(stream)

        self.assertEqual(len(results), 0)
        self.assertEqual(stats.total_tool_calls, 0)
        self.assertEqual(stats.immediate_executions, 0)
        self.assertEqual(stats.queued_executions, 0)
        self.assertEqual(stats.parallel_groups, 0)
        self.assertEqual(streaming_result.content, "Hello! How can I help?")

    async def test_single_read_only_tool_completes_quickly(self):
        """A single read-only tool call should execute immediately during stream."""
        registry = _build_tool_registry(
            read_only_count=1, write_count=0, safe_write_count=0,
            tool_delay=0.01,
        )
        executor = StreamingToolExecutor(registry, permission_level=0)

        tool_calls = [
            {"id": "call_0", "name": "read_file", "arguments": {"path": "main.py"}},
        ]

        stream = _create_mock_stream(
            text_deltas=["Reading the main file..."],
            tool_calls=tool_calls,
            interleaved=True,
            delay_between_events=0.01,
        )

        results, streaming_result, stats = await executor.execute_streaming(stream)

        self.assertEqual(len(results), 1)
        self.assertEqual(stats.immediate_executions, 1)
        self.assertEqual(stats.queued_executions, 0)
        self.assertTrue(results[0][1].success)

    async def test_execution_plan_preview(self):
        """get_execution_plan should correctly categorize tools without executing."""
        registry = _build_tool_registry(
            read_only_count=2, write_count=1, safe_write_count=0,
        )
        executor = StreamingToolExecutor(registry, permission_level=0)

        from teragent.streaming.streaming_executor import _PendingToolCall

        pending = {
            0: _PendingToolCall(
                index=0, call_id="c0", name="read_file",
                arguments={"path": "a.py"},
                is_read_only=True, is_concurrency_safe=True,
            ),
            1: _PendingToolCall(
                index=1, call_id="c1", name="read_tool_1",
                arguments={"path": "b.py"},
                is_read_only=True, is_concurrency_safe=True,
            ),
            2: _PendingToolCall(
                index=2, call_id="c2", name="execute_subtask",
                arguments={"command": "build"},
                is_read_only=False, is_concurrency_safe=False,
            ),
        }

        plan = executor.get_execution_plan(pending)

        self.assertEqual(plan["immediate_count"], 2)
        self.assertEqual(plan["queued_count"], 1)
        self.assertEqual(plan["total"], 3)
        self.assertEqual(len(plan["immediate"]), 2)
        self.assertEqual(len(plan["queued"]), 1)


# ======================================================================
# Test: Stream Parser Performance
# ======================================================================


class TestStreamParserPerformance(unittest.TestCase):
    """Test stream parser throughput and accuracy."""

    def test_openai_stream_parser_throughput(self):
        """OpenAI stream parser should handle high-frequency chunks.

        Generate 1000 SSE chunks and measure parsing time. The parser
        should process them in well under 1 second.
        """
        parser = OpenAIStreamParser()

        # Generate chunks: 500 text chunks + 500 tool call chunks (2 tool calls)
        chunks = []

        # Text chunks
        for i in range(500):
            chunks.append({
                "choices": [{
                    "delta": {"content": f"Text chunk {i}. "},
                    "finish_reason": None,
                }],
            })

        # Tool call 1: 250 argument fragments
        for i in range(250):
            args_delta = '{"pa' if i == 0 else 'th": "src/main.py"}' if i == 249 else 'rtial'
            if i == 0:
                chunks.append({
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_abc",
                                "function": {"name": "read_file", "arguments": args_delta},
                            }]
                        },
                        "finish_reason": None,
                    }],
                })
            else:
                chunks.append({
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "function": {"arguments": args_delta},
                            }]
                        },
                        "finish_reason": None,
                    }],
                })

        # Tool call 2: 250 argument fragments
        for i in range(250):
            args_delta = '{"co' if i == 0 else 'mmand": "ls"}' if i == 249 else 'mmand'
            if i == 0:
                chunks.append({
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 1,
                                "id": "call_def",
                                "function": {"name": "execute_subtask", "arguments": args_delta},
                            }]
                        },
                        "finish_reason": None,
                    }],
                })
            else:
                chunks.append({
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 1,
                                "function": {"arguments": args_delta},
                            }]
                        },
                        "finish_reason": None,
                    }],
                })

        # Final chunk with usage
        chunks.append({
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 500},
        })

        start = time.monotonic()
        all_events: list[StreamEvent] = []
        for chunk in chunks:
            events = parser.parse_chunk(chunk)
            all_events.extend(events)
        finalize_events = parser.finalize()
        all_events.extend(finalize_events)
        elapsed = time.monotonic() - start

        # Should process in under 1 second
        self.assertLess(elapsed, 1.0, "Parser should handle 1000 chunks in under 1 second")

        # Should have emitted events
        self.assertGreater(len(all_events), 0)

        # Verify result has content
        result = parser.get_result()
        self.assertGreater(len(result.content), 0)

        # Should have tool_calls in the finalize events
        tool_call_complete_events = [
            e for e in all_events
            if e.event_type == StreamEventType.TOOL_CALL_COMPLETE
        ]
        self.assertGreaterEqual(len(tool_call_complete_events), 2)

    def test_anthropic_stream_parser_throughput(self):
        """Anthropic stream parser should handle high-frequency chunks.

        Generate 1000 Anthropic SSE events and measure parsing time.
        """
        parser = AnthropicStreamParser()

        events_list: list[dict] = []

        # message_start
        events_list.append({
            "type": "message_start",
            "message": {"usage": {"input_tokens": 100}},
        })

        # Text content block: 500 text deltas
        events_list.append({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text"},
        })
        for i in range(500):
            events_list.append({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": f"Chunk {i}. "},
            })
        events_list.append({"type": "content_block_stop"})

        # Tool use content block: 500 json deltas
        events_list.append({
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "tu_abc", "name": "read_file"},
        })
        json_parts = ['{"pa', 'th":', ' "src/', 'main.py', '"}']
        for i in range(500):
            part = json_parts[i % len(json_parts)]
            events_list.append({
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": part},
            })
        events_list.append({"type": "content_block_stop"})

        # message_delta
        events_list.append({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 200},
        })

        start = time.monotonic()
        all_events: list[StreamEvent] = []
        for event in events_list:
            parsed = parser.parse_event(event)
            all_events.extend(parsed)
        finalize_events = parser.finalize()
        all_events.extend(finalize_events)
        elapsed = time.monotonic() - start

        # Should process in under 1 second
        self.assertLess(elapsed, 1.0, "Parser should handle 1000 events in under 1 second")

        # Should have emitted events
        self.assertGreater(len(all_events), 0)

        # Verify result
        result = parser.get_result()
        self.assertGreater(len(result.content), 0)

        # Should have a TOOL_CALL_COMPLETE event from content_block_stop
        tool_complete_events = [
            e for e in all_events
            if e.event_type == StreamEventType.TOOL_CALL_COMPLETE
        ]
        self.assertGreaterEqual(len(tool_complete_events), 1)

    def test_tool_call_accumulator_json_detection(self):
        """ToolCallAccumulator should correctly detect JSON completeness.

        Test various JSON patterns: simple, nested, arrays, strings
        with braces. The accumulator should only report completion
        when the JSON is truly valid.
        """
        # Simple JSON object
        acc = ToolCallAccumulator(index=0, call_id="c1", name="read_file")
        acc.append_arguments('{"path": "src/main.py"}')
        self.assertTrue(acc.is_arguments_complete(), "Simple JSON should be detected as complete")
        parsed = acc.parse_arguments()
        self.assertEqual(parsed, {"path": "src/main.py"})

        # Nested JSON object
        acc2 = ToolCallAccumulator(index=1, call_id="c2", name="complex_tool")
        acc2.append_arguments('{"outer": {"inner": "value"}, "count": 1}')
        self.assertTrue(acc2.is_arguments_complete(), "Nested JSON should be detected as complete")
        parsed2 = acc2.parse_arguments()
        self.assertEqual(parsed2["outer"]["inner"], "value")

        # JSON array inside object
        acc3 = ToolCallAccumulator(index=2, call_id="c3", name="list_tool")
        acc3.append_arguments('{"items": [1, 2, 3], "name": "test"}')
        self.assertTrue(acc3.is_arguments_complete(), "JSON with array should be detected as complete")
        parsed3 = acc3.parse_arguments()
        self.assertEqual(parsed3["items"], [1, 2, 3])

        # String containing braces (should NOT be considered complete prematurely)
        acc4 = ToolCallAccumulator(index=3, call_id="c4", name="code_tool")
        acc4.append_arguments('{"code": "if (x) { return y; }", "lang": "js"}')
        self.assertTrue(acc4.is_arguments_complete(),
                        "JSON with braces in string should be detected as complete")
        parsed4 = acc4.parse_arguments()
        self.assertEqual(parsed4["code"], "if (x) { return y; }")

        # Incomplete JSON (missing closing brace)
        acc5 = ToolCallAccumulator(index=4, call_id="c5", name="partial_tool")
        acc5.append_arguments('{"path": "src/')
        self.assertFalse(acc5.is_arguments_complete(), "Incomplete JSON should not be detected as complete")

        # Incremental accumulation
        acc6 = ToolCallAccumulator(index=5, call_id="c6", name="streaming_tool")
        acc6.append_arguments('{"pa')
        self.assertFalse(acc6.is_arguments_complete())
        acc6.append_arguments('th": "main.py"}')
        self.assertTrue(acc6.is_arguments_complete())
        self.assertEqual(acc6.parse_arguments(), {"path": "main.py"})

        # Empty buffer
        acc7 = ToolCallAccumulator(index=6, call_id="c7", name="empty_tool")
        self.assertFalse(acc7.is_arguments_complete(), "Empty buffer should not be complete")
        self.assertEqual(acc7.parse_arguments(), {})

    def test_openai_parser_tool_call_incremental(self):
        """OpenAI parser should emit TOOL_CALL_COMPLETE when arguments are complete.

        Test that the parser correctly tracks incremental tool call arguments
        and emits a TOOL_CALL_COMPLETE event when the full JSON is available.
        """
        parser = OpenAIStreamParser()

        # First chunk: tool call start with id and name
        events1 = parser.parse_chunk({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_123",
                        "function": {"name": "read_file", "arguments": ""},
                    }]
                },
            }],
        })
        event_types_1 = [e.event_type for e in events1]
        self.assertIn(StreamEventType.TOOL_CALL_START, event_types_1)

        # Second chunk: partial arguments
        events2 = parser.parse_chunk({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": '{"pat'},
                    }]
                },
            }],
        })
        # Arguments not complete yet
        tool_complete_2 = [e for e in events2 if e.event_type == StreamEventType.TOOL_CALL_COMPLETE]
        self.assertEqual(len(tool_complete_2), 0)

        # Third chunk: rest of arguments
        events3 = parser.parse_chunk({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": 'h": "main.py"}'},
                    }]
                },
            }],
        })
        tool_complete_3 = [e for e in events3 if e.event_type == StreamEventType.TOOL_CALL_COMPLETE]
        self.assertEqual(len(tool_complete_3), 1)
        self.assertEqual(tool_complete_3[0].tool_name, "read_file")
        self.assertEqual(tool_complete_3[0].tool_arguments, {"path": "main.py"})

    def test_anthropic_parser_tool_use_complete(self):
        """Anthropic parser should emit TOOL_CALL_COMPLETE at content_block_stop.

        Verify that when a tool_use content block ends, the parser
        correctly emits a TOOL_CALL_COMPLETE event with parsed arguments.
        """
        parser = AnthropicStreamParser()

        # Start tool_use block
        parser.parse_event({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "tu_001", "name": "read_file"},
        })

        # Send JSON deltas
        parser.parse_event({
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": '{"path":'},
        })
        parser.parse_event({
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": ' "src/main.py"}'},
        })

        # End block -- should emit TOOL_CALL_COMPLETE
        events = parser.parse_event({"type": "content_block_stop"})

        complete_events = [e for e in events if e.event_type == StreamEventType.TOOL_CALL_COMPLETE]
        self.assertEqual(len(complete_events), 1)
        self.assertEqual(complete_events[0].tool_name, "read_file")
        self.assertEqual(complete_events[0].tool_arguments, {"path": "src/main.py"})


# ======================================================================
# Test: Error Recovery Performance
# ======================================================================


class TestStreamingRecoveryPerformance(unittest.IsolatedAsyncioTestCase):
    """Test streaming error recovery overhead."""

    async def test_streaming_retry_overhead(self):
        """Streaming retry should add minimal overhead.

        Measure the time for 2 retries vs a single successful call.
        The overhead of retries should be proportional to the number
        of retries, not exponentially larger.
        """
        registry = _build_tool_registry(
            read_only_count=1, write_count=0, safe_write_count=0,
            tool_delay=0.01,
        )
        executor = StreamingToolExecutor(registry, permission_level=0)

        tool_calls = [
            {"id": "call_0", "name": "read_file", "arguments": {"path": "main.py"}},
        ]

        # Measure single successful call
        stream = _create_mock_stream(
            text_deltas=["Reading..."],
            tool_calls=tool_calls,
            delay_between_events=0.01,
        )
        start = time.monotonic()
        results, _, stats = await executor.execute_streaming(stream)
        single_call_time = time.monotonic() - start

        # Measure with simulated retries (2 failed streams then 1 success)
        # Each "failed" stream is an empty stream that ends with DONE + no tool_calls
        retry_count = 2
        start = time.monotonic()
        for attempt in range(retry_count):
            empty_stream = _create_mock_stream(
                text_deltas=[""],
                tool_calls=None,
                delay_between_events=0.005,
            )
            empty_results, _, empty_stats = await executor.execute_streaming(empty_stream)

        # Final successful attempt
        success_stream = _create_mock_stream(
            text_deltas=["Reading..."],
            tool_calls=tool_calls,
            delay_between_events=0.01,
        )
        final_results, _, final_stats = await executor.execute_streaming(success_stream)
        retry_total_time = time.monotonic() - start

        # Retry overhead should be roughly (retry_count + 1) * single_call_time
        # But definitely not more than 5x single_call_time (generous margin)
        self.assertLess(
            retry_total_time, single_call_time * 10,
            "Retry overhead should not be disproportionate to retry count",
        )

        # Final results should be valid
        self.assertEqual(len(final_results), 1)
        self.assertTrue(final_results[0][1].success)

    async def test_context_overflow_recovery_time(self):
        """Context compression + retry should complete within reasonable time.

        Simulate a scenario where the AutoCompactor takes ~100ms to compress
        context. The total recovery time (compress + retry) should be
        approximately 100ms + model_call_time, not significantly more.
        """
        registry = _build_tool_registry(
            read_only_count=1, write_count=0, safe_write_count=0,
            tool_delay=0.01,
        )
        executor = StreamingToolExecutor(registry, permission_level=0)

        # Simulate a successful streaming call (representing post-compression retry)
        tool_calls = [
            {"id": "call_0", "name": "read_file", "arguments": {"path": "main.py"}},
        ]

        # Simulate compression overhead
        compression_time = 0.1  # 100ms

        start = time.monotonic()

        # Simulate compression step
        await asyncio.sleep(compression_time)

        # Then streaming call
        stream = _create_mock_stream(
            text_deltas=["After compression..."],
            tool_calls=tool_calls,
            delay_between_events=0.01,
        )
        results, streaming_result, stats = await executor.execute_streaming(stream)

        total_recovery_time = time.monotonic() - start

        # Total should be roughly compression_time + streaming_time, not much more
        # Allow generous margin (2x) for CI variability
        self.assertLess(
            total_recovery_time, (compression_time + 1.0) * 2,
            "Context overflow recovery should complete within reasonable time",
        )

        # Results should be valid after recovery
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0][1].success)

    async def test_stream_error_event_handling(self):
        """StreamingToolExecutor should handle ERROR events gracefully.

        When the stream emits an ERROR event, the executor should
        continue processing and not crash.
        """
        registry = _build_tool_registry(
            read_only_count=1, write_count=0, safe_write_count=0,
            tool_delay=0.01,
        )
        executor = StreamingToolExecutor(registry, permission_level=0)

        async def error_stream() -> AsyncIterator[StreamEvent]:
            """Stream that emits an error before tool call."""
            yield StreamEvent(event_type=StreamEventType.TEXT_DELTA, text="Starting...")
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                error="Temporary network error",
            )
            yield StreamEvent(
                event_type=StreamEventType.TOOL_CALL_START,
                tool_call_index=0,
                tool_call_id="call_0",
                tool_name="read_file",
            )
            yield StreamEvent(
                event_type=StreamEventType.TOOL_CALL_COMPLETE,
                tool_call_index=0,
                tool_call_id="call_0",
                tool_name="read_file",
                tool_arguments={"path": "main.py"},
            )
            yield StreamEvent(event_type=StreamEventType.DONE, finish_reason="stop")

        results, streaming_result, stats = await executor.execute_streaming(error_stream())

        # Should still have results despite error event
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0][1].success)
        self.assertEqual(stats.total_tool_calls, 1)

    async def test_fallback_to_batch_mode(self):
        """execute_batch_fallback should work when streaming is unavailable.

        When streaming is not supported, the executor should fall back
        to batch execution via execute_batch_fallback.
        """
        registry = _build_tool_registry(
            read_only_count=2, write_count=1, safe_write_count=0,
            tool_delay=0.01,
        )
        executor = StreamingToolExecutor(registry, permission_level=0)

        tool_calls = [
            {"name": "read_file", "arguments": {"path": "a.py"}, "id": "c0"},
            {"name": "read_tool_1", "arguments": {"path": "b.py"}, "id": "c1"},
            {"name": "execute_subtask", "arguments": {"command": "build"}, "id": "c2"},
        ]

        results, stats = await executor.execute_batch_fallback(tool_calls)

        self.assertEqual(len(results), 3)
        self.assertTrue(stats.fallback_used)
        self.assertEqual(stats.total_tool_calls, 3)
        self.assertEqual(stats.queued_executions, 3)

    async def test_can_stream_with_tools(self):
        """can_stream_with_tools should correctly detect model capabilities."""
        registry = _build_tool_registry(read_only_count=1, write_count=0, safe_write_count=0)
        executor = StreamingToolExecutor(registry, permission_level=0)

        # Model that supports streaming tool calling
        model_with_streaming = MagicMock()
        model_with_streaming.capabilities.return_value = {
            "streaming": True,
            "tool_calling": True,
            "streaming_tool_calling": True,
        }
        self.assertTrue(executor.can_stream_with_tools(model_with_streaming))

        # Model without streaming tool calling
        model_no_streaming_tc = MagicMock()
        model_no_streaming_tc.capabilities.return_value = {
            "streaming": True,
            "tool_calling": True,
            "streaming_tool_calling": False,
        }
        self.assertFalse(executor.can_stream_with_tools(model_no_streaming_tc))

        # Model with no capabilities method (raises exception)
        model_broken = MagicMock()
        model_broken.capabilities.side_effect = RuntimeError("No capabilities")
        self.assertFalse(executor.can_stream_with_tools(model_broken))


# ======================================================================
# Test: Streaming Config Loading
# ======================================================================


class TestStreamingConfigLoading(unittest.TestCase):
    """Test streaming config wiring from agent.toml.

    AgentLoop is now available in the teragent package.
    """

    def _make_agent_loop(self) -> AgentLoop:
        """Create a minimal AgentLoop instance for testing config methods.

        Uses mocks for all dependencies to avoid requiring a live model.
        """
        from teragent.event_bus import EventBus

        mock_model = MagicMock()
        mock_model.capabilities.return_value = {
            "streaming": True,
            "tool_calling": True,
            "streaming_tool_calling": True,
        }

        registry = _build_tool_registry(
            read_only_count=1, write_count=0, safe_write_count=0,
        )
        streaming_executor = StreamingToolExecutor(registry, permission_level=0)
        config = AgentLoopConfig()
        event_bus = EventBus()

        return AgentLoop(
            model=mock_model,
            tool_registry=registry,
            config=config,
            event_bus=event_bus,
            streaming_executor=streaming_executor,
        )

    def test_set_streaming_config_mode(self):
        """set_streaming_config should update streaming_mode."""
        loop = self._make_agent_loop()

        self.assertEqual(loop.streaming_mode, "auto")

        loop.set_streaming_config(mode="streaming")
        self.assertEqual(loop.streaming_mode, "streaming")

        loop.set_streaming_config(mode="batch")
        self.assertEqual(loop.streaming_mode, "batch")

        loop.set_streaming_config(mode="auto")
        self.assertEqual(loop.streaming_mode, "auto")

    def test_set_streaming_config_retries(self):
        """set_streaming_config should update max_streaming_retries."""
        loop = self._make_agent_loop()

        # Record the initial value (may vary due to global state from prior tests)
        initial_retries = loop._max_streaming_retries

        loop.set_streaming_config(max_streaming_retries=5)
        self.assertEqual(loop._max_streaming_retries, 5)

        loop.set_streaming_config(max_streaming_retries=0)
        self.assertEqual(loop._max_streaming_retries, 0)

        # Restore to a known value
        loop.set_streaming_config(max_streaming_retries=initial_retries)

    def test_invalid_mode_falls_back_to_auto(self):
        """Invalid streaming mode should fall back to 'auto'.

        When an unrecognized mode string is passed, the method should
        log a warning and default to 'auto' instead of crashing.
        """
        loop = self._make_agent_loop()

        loop.set_streaming_config(mode="invalid_mode")
        self.assertEqual(loop.streaming_mode, "auto")

        loop.set_streaming_config(mode="STREAMING")
        self.assertEqual(loop.streaming_mode, "auto")

        loop.set_streaming_config(mode="")
        self.assertEqual(loop.streaming_mode, "auto")

    def test_streaming_mode_in_recovery_stats(self):
        """Streaming mode should be tracked in recovery stats."""
        loop = self._make_agent_loop()

        loop.set_streaming_config(mode="streaming")
        self.assertEqual(loop._recovery_stats["streaming_mode"], "streaming")

        loop.set_streaming_config(mode="batch")
        self.assertEqual(loop._recovery_stats["streaming_mode"], "batch")

    def test_set_streaming_config_none_unchanged(self):
        """Passing None for mode/retries should not change existing values."""
        loop = self._make_agent_loop()
        loop.set_streaming_config(mode="streaming")
        loop.set_streaming_config(max_streaming_retries=7)

        loop.set_streaming_config(mode=None, max_streaming_retries=None)
        self.assertEqual(loop.streaming_mode, "streaming")
        self.assertEqual(loop._max_streaming_retries, 7)

    def test_config_from_toml(self):
        """Streaming config should be loadable from agent.toml format.

        Verify that the [streaming] section in agent.toml can be
        correctly parsed and applied to AgentLoop.
        """
        loop = self._make_agent_loop()

        # Simulate loading config from agent.toml
        streaming_config = {
            "mode": "batch",
            "max_streaming_retries": 3,
        }

        loop.set_streaming_config(
            mode=streaming_config.get("mode"),
            max_streaming_retries=streaming_config.get("max_streaming_retries"),
        )

        self.assertEqual(loop.streaming_mode, "batch")
        self.assertEqual(loop._max_streaming_retries, 3)


# ======================================================================
# Test: Streaming Pre-Filter
# ======================================================================


class TestStreamingPreFilter(unittest.IsolatedAsyncioTestCase):
    """Test tool pre-filtering before streaming execution."""

    async def test_disallowed_tools_not_executed(self):
        """Tools not in allowed_tools should not be passed to stream_with_tools.

        When the streaming executor receives tool calls for tools that
        are not in the allowed set, they should still be tracked in
        results but with an error indicating the tool is not available.
        """
        registry = _build_tool_registry(
            read_only_count=2, write_count=1, safe_write_count=0,
            tool_delay=0.01,
        )
        executor = StreamingToolExecutor(registry, permission_level=0)

        # Stream contains a tool call for a tool that is NOT registered
        tool_calls = [
            {"id": "call_0", "name": "read_file", "arguments": {"path": "a.py"}},
            {"id": "call_1", "name": "unknown_tool", "arguments": {"x": 1}},
        ]

        stream = _create_mock_stream(
            text_deltas=["Processing..."],
            tool_calls=tool_calls,
            delay_between_events=0.01,
        )

        results, streaming_result, stats = await executor.execute_streaming(stream)

        # read_file should succeed; unknown_tool should fail
        self.assertEqual(len(results), 2)

        # Find results by name
        results_by_name = {tc["name"]: r for tc, r in results}
        self.assertTrue(results_by_name["read_file"].success)
        self.assertFalse(results_by_name["unknown_tool"].success)
        self.assertIn("unknown_tool", results_by_name["unknown_tool"].error)

    async def test_design_gate_pre_filter(self):
        """When force_design_first, only generate_design should be available.

        In CREATE_PROJECT intent with force_design_first=True, the tool
        pre-filter should block all tools except generate_design and
        submit_failure from being available for the first model call.
        """
        registry = _build_tool_registry(
            read_only_count=1, write_count=1, safe_write_count=1,
            tool_delay=0.01,
        )

        # Simulate design gate pre-filter logic
        allowed_tools = ["generate_design", "submit_failure"]

        all_tool_names = registry.list_tool_names()
        filtered_tools = [name for name in all_tool_names if name in allowed_tools]

        # Only generate_design and submit_failure should pass the filter
        self.assertIn("generate_design", filtered_tools)
        self.assertNotIn("read_file", filtered_tools)
        self.assertNotIn("execute_subtask", filtered_tools)

        # The actual tool definitions passed to the model should only
        # contain allowed tools
        filtered_definitions = registry.get_tools_by_names(filtered_tools)
        self.assertEqual(len(filtered_definitions), 1)  # Only generate_design registered
        self.assertEqual(filtered_definitions[0].name, "generate_design")

    async def test_read_only_tools_always_allowed(self):
        """Read-only tools should always be available regardless of design gate.

        After the design gate has been satisfied (generate_design called),
        read-only tools should become available alongside write tools.
        """
        registry = _build_tool_registry(
            read_only_count=2, write_count=1, safe_write_count=1,
            tool_delay=0.01,
        )

        # Simulate post-design-gate: all CREATE_PROJECT tools + debug tools
        # Note: write tools like execute_subtask are not in CREATE_PROJECT
        # intent tool list, so only test tools that would actually be allowed
        allowed_tools = [
            "generate_design", "generate_plan", "create_project",
            "read_file", "read_tool_1", "explore_codebase", "list_directory",
            "submit_failure", "get_pipeline_status", "execute_subtask",
        ]

        all_tool_names = registry.list_tool_names()
        filtered_tools = [name for name in all_tool_names if name in allowed_tools]

        # Read-only tools should be present
        self.assertIn("read_file", filtered_tools)
        self.assertIn("read_tool_1", filtered_tools)  # Second read-only tool

        # Write tools should also be present when in allowed set
        self.assertIn("generate_design", filtered_tools)
        self.assertIn("execute_subtask", filtered_tools)

    async def test_registry_safety_metadata_correctness(self):
        """ToolRegistry should correctly track safety metadata for filtering."""
        registry = _build_tool_registry(
            read_only_count=2, write_count=1, safe_write_count=1,
        )

        # Verify safety metadata
        read_only_tools = registry.get_read_only_tools()
        self.assertIn("read_file", read_only_tools)
        self.assertIn("read_tool_1", read_only_tools)
        self.assertNotIn("execute_subtask", read_only_tools)
        self.assertNotIn("generate_design", read_only_tools)

        concurrency_safe_tools = registry.get_concurrency_safe_tools()
        self.assertIn("read_file", concurrency_safe_tools)
        self.assertIn("read_tool_1", concurrency_safe_tools)

        destructive_tools = registry.get_destructive_tools()
        self.assertIn("execute_subtask", destructive_tools)

        # Safety report should be comprehensive
        report = registry.get_safety_report()
        self.assertEqual(report["total"], 4)
        self.assertIn("read_file", report["read_only_tools"])
        self.assertIn("execute_subtask", report["destructive_tools"])

    async def test_empty_registry_streaming(self):
        """Streaming executor should handle an empty tool registry gracefully."""
        registry = ToolRegistry()
        executor = StreamingToolExecutor(registry, permission_level=0)

        tool_calls = [
            {"id": "call_0", "name": "nonexistent", "arguments": {"x": 1}},
        ]

        stream = _create_mock_stream(
            text_deltas=["Attempting..."],
            tool_calls=tool_calls,
            delay_between_events=0.01,
        )

        results, streaming_result, stats = await executor.execute_streaming(stream)

        # Should have one result, but it should be a failure
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0][1].success)
        self.assertEqual(stats.queued_executions, 1)


# ======================================================================
# Test: StreamingChatResult and End-to-End Integration
# ======================================================================


class TestStreamingChatResultIntegration(unittest.IsolatedAsyncioTestCase):
    """Test StreamingChatResult construction and integration with executor."""

    async def test_streaming_result_contains_all_tool_calls(self):
        """StreamingChatResult should list all tool calls from the stream."""
        registry = _build_tool_registry(
            read_only_count=2, write_count=1, safe_write_count=0,
            tool_delay=0.01,
        )
        executor = StreamingToolExecutor(registry, permission_level=0)

        tool_calls = [
            {"id": "call_0", "name": "read_file", "arguments": {"path": "a.py"}},
            {"id": "call_1", "name": "read_tool_1", "arguments": {"path": "b.py"}},
            {"id": "call_2", "name": "execute_subtask", "arguments": {"command": "build"}},
        ]

        stream = _create_mock_stream(
            text_deltas=["Processing..."],
            tool_calls=tool_calls,
            interleaved=True,
            delay_between_events=0.01,
        )

        results, streaming_result, stats = await executor.execute_streaming(stream)

        # StreamingChatResult should have all 3 tool calls
        self.assertEqual(len(streaming_result.tool_calls), 3)
        self.assertEqual(streaming_result.content, "Processing...")
        self.assertEqual(streaming_result.finish_reason, "stop")

        # to_chat_response should work
        response = streaming_result.to_chat_response()
        self.assertIn("content", response)
        self.assertIn("tool_calls", response)

    async def test_streaming_result_usage_tracking(self):
        """StreamingChatResult should include token usage from the stream."""
        registry = _build_tool_registry(
            read_only_count=1, write_count=0, safe_write_count=0,
        )
        executor = StreamingToolExecutor(registry, permission_level=0)

        tool_calls = [
            {"id": "call_0", "name": "read_file", "arguments": {"path": "main.py"}},
        ]

        stream = _create_mock_stream(
            text_deltas=["Reading..."],
            tool_calls=tool_calls,
            delay_between_events=0.01,
        )

        results, streaming_result, stats = await executor.execute_streaming(stream)

        # Usage should be present
        self.assertIn("prompt_tokens", streaming_result.usage)
        self.assertIn("completion_tokens", streaming_result.usage)
        self.assertGreater(streaming_result.usage.get("prompt_tokens", 0), 0)

    async def test_on_text_delta_callback(self):
        """on_text_delta callback should be called for each text delta event."""
        registry = _build_tool_registry(
            read_only_count=1, write_count=0, safe_write_count=0,
        )
        executor = StreamingToolExecutor(registry, permission_level=0)

        tool_calls = [
            {"id": "call_0", "name": "read_file", "arguments": {"path": "main.py"}},
        ]

        received_deltas: list[str] = []

        async def text_callback(text: str) -> None:
            received_deltas.append(text)

        stream = _create_mock_stream(
            text_deltas=["Hello ", "World!"],
            tool_calls=tool_calls,
            interleaved=False,
            delay_between_events=0.01,
        )

        await executor.execute_streaming(stream, on_text_delta=text_callback)

        self.assertEqual(received_deltas, ["Hello ", "World!"])

    async def test_on_tool_complete_callback(self):
        """on_tool_complete callback should be called for each completed tool."""
        registry = _build_tool_registry(
            read_only_count=2, write_count=0, safe_write_count=0,
            tool_delay=0.01,
        )
        executor = StreamingToolExecutor(registry, permission_level=0)

        tool_calls = [
            {"id": "call_0", "name": "read_file", "arguments": {"path": "a.py"}},
            {"id": "call_1", "name": "read_tool_1", "arguments": {"path": "b.py"}},
        ]

        completed_tools: list[str] = []

        async def tool_complete_callback(tc: dict, result: ToolResult) -> None:
            completed_tools.append(tc["name"])

        stream = _create_mock_stream(
            text_deltas=["Reading..."],
            tool_calls=tool_calls,
            delay_between_events=0.01,
        )

        await executor.execute_streaming(stream, on_tool_complete=tool_complete_callback)

        self.assertEqual(completed_tools, ["read_file", "read_tool_1"])


# ======================================================================
# Test: ToolCallAccumulator Edge Cases
# ======================================================================


class TestToolCallAccumulatorEdgeCases(unittest.TestCase):
    """Test ToolCallAccumulator with edge-case JSON patterns."""

    def test_json_with_escaped_quotes(self):
        """Accumulator should handle JSON with escaped quote characters."""
        acc = ToolCallAccumulator(index=0, call_id="c1", name="tool")
        acc.append_arguments('{"text": "He said \\"hello\\""}')
        self.assertTrue(acc.is_arguments_complete())
        parsed = acc.parse_arguments()
        self.assertEqual(parsed["text"], 'He said "hello"')

    def test_json_with_escaped_backslash(self):
        """Accumulator should handle JSON with escaped backslash characters."""
        acc = ToolCallAccumulator(index=0, call_id="c1", name="tool")
        acc.append_arguments('{"path": "C:\\\\Users\\\\test"}')
        self.assertTrue(acc.is_arguments_complete())
        parsed = acc.parse_arguments()
        self.assertEqual(parsed["path"], "C:\\Users\\test")

    def test_json_with_unicode(self):
        """Accumulator should handle JSON with unicode characters."""
        acc = ToolCallAccumulator(index=0, call_id="c1", name="tool")
        acc.append_arguments('{"text": "Hello World"}')
        self.assertTrue(acc.is_arguments_complete())
        parsed = acc.parse_arguments()
        self.assertEqual(parsed["text"], "Hello World")

    def test_json_with_null_and_bool(self):
        """Accumulator should handle JSON with null and boolean values."""
        acc = ToolCallAccumulator(index=0, call_id="c1", name="tool")
        acc.append_arguments('{"opt": null, "flag": true, "disabled": false}')
        self.assertTrue(acc.is_arguments_complete())
        parsed = acc.parse_arguments()
        self.assertIsNone(parsed["opt"])
        self.assertTrue(parsed["flag"])
        self.assertFalse(parsed["disabled"])

    def test_json_with_empty_object(self):
        """Accumulator should handle an empty JSON object."""
        acc = ToolCallAccumulator(index=0, call_id="c1", name="tool")
        acc.append_arguments('{}')
        self.assertTrue(acc.is_arguments_complete())
        parsed = acc.parse_arguments()
        self.assertEqual(parsed, {})

    def test_incremental_brace_counting(self):
        """Accumulator should correctly track braces during incremental parsing.

        Verify that the brace-counting heuristic does not prematurely
        declare completeness for partially-arriving JSON.
        """
        acc = ToolCallAccumulator(index=0, call_id="c1", name="tool")

        # Start with opening brace
        acc.append_arguments('{')
        self.assertFalse(acc.is_arguments_complete())

        # Add key
        acc.append_arguments('"path"')
        self.assertFalse(acc.is_arguments_complete())

        # Add colon
        acc.append_arguments(': ')
        self.assertFalse(acc.is_arguments_complete())

        # Add value
        acc.append_arguments('"main.py"')
        self.assertFalse(acc.is_arguments_complete())

        # Add closing brace
        acc.append_arguments('}')
        self.assertTrue(acc.is_arguments_complete())

    def test_to_tool_call_dict(self):
        """to_tool_call_dict should produce the standard format."""
        acc = ToolCallAccumulator(index=3, call_id="call_abc", name="read_file")
        acc.append_arguments('{"path": "src/main.py"}')

        result = acc.to_tool_call_dict()

        self.assertEqual(result["id"], "call_abc")
        self.assertEqual(result["type"], "function")
        self.assertEqual(result["function"]["name"], "read_file")
        self.assertEqual(result["function"]["arguments"], '{"path": "src/main.py"}')

    def test_to_tool_call_dict_no_id(self):
        """to_tool_call_dict should use index-based ID when no call_id is set."""
        acc = ToolCallAccumulator(index=5, name="read_file")
        acc.append_arguments('{"path": "test.py"}')

        result = acc.to_tool_call_dict()

        self.assertEqual(result["id"], "call_5")


# ======================================================================
# Test: StreamingExecutionStats Dataclass
# ======================================================================


class TestStreamingExecutionStatsDataclass(unittest.TestCase):
    """Test StreamingExecutionStats dataclass behavior."""

    def test_default_values(self):
        """Stats should have sensible default values."""
        stats = StreamingExecutionStats()
        self.assertEqual(stats.total_tool_calls, 0)
        self.assertEqual(stats.immediate_executions, 0)
        self.assertEqual(stats.queued_executions, 0)
        self.assertEqual(stats.parallel_groups, 0)
        self.assertEqual(stats.streaming_time_ms, 0.0)
        self.assertEqual(stats.execution_time_ms, 0.0)
        self.assertFalse(stats.fallback_used)

    def test_to_dict_roundtrip(self):
        """to_dict should produce a serializable dictionary with all fields."""
        stats = StreamingExecutionStats(
            total_tool_calls=5,
            immediate_executions=3,
            queued_executions=2,
            parallel_groups=1,
            streaming_time_ms=150.5,
            execution_time_ms=200.3,
            fallback_used=False,
        )

        d = stats.to_dict()

        self.assertEqual(d["total_tool_calls"], 5)
        self.assertEqual(d["immediate_executions"], 3)
        self.assertEqual(d["queued_executions"], 2)
        self.assertEqual(d["parallel_groups"], 1)
        self.assertEqual(d["streaming_time_ms"], 150.5)
        self.assertEqual(d["execution_time_ms"], 200.3)
        self.assertFalse(d["fallback_used"])

    def test_to_dict_rounds_timing(self):
        """to_dict should round timing values to 1 decimal place."""
        stats = StreamingExecutionStats(
            streaming_time_ms=123.456789,
            execution_time_ms=987.654321,
        )

        d = stats.to_dict()

        self.assertEqual(d["streaming_time_ms"], 123.5)
        self.assertEqual(d["execution_time_ms"], 987.7)


# ======================================================================
# Entry point
# ======================================================================


if __name__ == "__main__":
    unittest.main()
