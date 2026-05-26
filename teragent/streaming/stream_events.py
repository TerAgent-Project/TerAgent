# teragent/streaming/stream_events.py
"""Phase 8.1: Streaming Chat Event Types

Defines structured streaming events for model chat with tool_use support.
These events are yielded by stream_chat() and stream_with_tools() methods,
enabling real-time tool execution while the model is still streaming.

Design principle:
  - Text deltas are yielded immediately for TUI rendering
  - Tool calls are tracked incrementally; a tool_call_complete event fires
    as soon as the full JSON arguments are available
  - This allows StreamingToolExecutor (Phase 8.2) to start executing
    read-only tools before the model finishes generating all tool_calls
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class StreamEventType(str, Enum):
    """Streaming event types

    Order of typical occurrence in an OpenAI-compatible stream:
      1. text_delta        -- model outputs text content (zero or more)
      2. tool_call_start   -- a new tool_call begins (id + name known)
      3. tool_call_delta   -- arguments fragment arrives (zero or more)
      4. tool_call_complete -- full arguments JSON is valid and complete
      5. ... (more text_delta or tool_call events)
      6. usage             -- token usage info (optional, at end)
      7. done              -- stream finished
    """

    TEXT_DELTA = "text_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_COMPLETE = "tool_call_complete"
    USAGE = "usage"
    DONE = "done"
    ERROR = "error"


@dataclass
class StreamEvent:
    """A single streaming event from the model

    Attributes:
        event_type: Type of this event
        text: Text content (for TEXT_DELTA events)
        tool_call_index: Index of the tool_call in the model response (for tool_call events)
        tool_call_id: Unique ID of the tool_call (for TOOL_CALL_START / TOOL_CALL_COMPLETE)
        tool_name: Name of the tool being called (for TOOL_CALL_START)
        tool_arguments_delta: Partial arguments JSON fragment (for TOOL_CALL_DELTA)
        tool_arguments: Complete parsed arguments dict (for TOOL_CALL_COMPLETE)
        usage: Token usage dict with prompt_tokens / completion_tokens (for USAGE events)
        error: Error message (for ERROR events)
        finish_reason: Why the model stopped generating (for DONE events)
    """

    event_type: StreamEventType
    text: str = ""
    tool_call_index: int = -1
    tool_call_id: str = ""
    tool_name: str = ""
    tool_arguments_delta: str = ""
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    error: str = ""
    finish_reason: str = ""


@dataclass
class ToolCallAccumulator:
    """Accumulates incremental tool_call deltas into a complete tool_call

    OpenAI streaming sends tool_calls in fragments:
      chunk 1: {"index":0, "id":"call_abc", "function":{"name":"read_file", "arguments":""}}
      chunk 2: {"index":0, "function":{"arguments":"{\"pa"}}
      chunk 3: {"index":0, "function":{"arguments":"th\": \"src/"}}
      chunk 4: {"index":0, "function":{"arguments":"main.py\"}"}}

    This accumulator tracks each tool_call by index, buffering the
    arguments fragments until they form a complete JSON string.

    Anthropic streaming sends tool_calls differently:
      event 1: content_block_start with type=tool_use, id, name
      event 2+: content_block_delta with input_json_delta partial JSON
      final: content_block_stop
    """

    index: int
    call_id: str = ""
    name: str = ""
    arguments_buffer: str = ""

    def append_arguments(self, delta: str) -> None:
        """Append a arguments fragment to the buffer"""
        self.arguments_buffer += delta

    def is_arguments_complete(self) -> bool:
        """Check if the buffered arguments form a valid JSON object

        Uses a brace-counting heuristic first (fast), then validates
        with json.loads() (accurate). This is necessary because:
        - JSON may contain nested objects/arrays
        - Simple brace counting can be fooled by strings containing braces
        - But full json.loads() on every delta is expensive

        Strategy: only attempt json.loads() when brace count suggests
        the JSON might be complete.
        """
        stripped = self.arguments_buffer.strip()
        if not stripped:
            return False

        # Quick check: does it look like a complete JSON object?
        if stripped.startswith("{"):
            # Count unescaped braces (rough heuristic)
            depth = 0
            in_string = False
            escape_next = False
            for ch in stripped:
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\":
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1

            # If braces are balanced, try parsing
            if depth == 0:
                try:
                    json.loads(stripped)
                    return True
                except json.JSONDecodeError:
                    return False
            return False

        # Non-object JSON (shouldn't happen for tool args, but handle gracefully)
        try:
            json.loads(stripped)
            return True
        except json.JSONDecodeError:
            return False

    def parse_arguments(self) -> dict[str, Any]:
        """Parse the accumulated arguments buffer into a dict

        Returns:
            Parsed arguments dict, or empty dict on parse failure
        """
        stripped = self.arguments_buffer.strip()
        if not stripped:
            return {}
        try:
            result = json.loads(stripped)
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            logger.warning(
                f"Failed to parse tool_call arguments for "
                f"'{self.name}' (id={self.call_id}): "
                f"{stripped[:200]}"
            )
            return {}

    def to_tool_call_dict(self) -> dict[str, Any]:
        """Convert to the standard tool_call dict format used by AgentLoop

        Returns:
            dict with id, type, function.name, function.arguments
            Note: function.arguments is a JSON string (OpenAI API format),
            not a dict.
        """
        parsed = self.parse_arguments()
        try:
            args_str = json.dumps(parsed, ensure_ascii=False)
        except (TypeError, ValueError):
            args_str = "{}"
        return {
            "id": self.call_id or f"call_{self.index}",
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": args_str,
            },
        }


@dataclass
class StreamingChatResult:
    """Final result of a streaming chat call

    After the stream completes, this object contains the full response
    in the same format as chat(), enabling seamless fallback.

    Attributes:
        content: Full text content
        tool_calls: List of parsed tool_calls (same format as chat() response)
        usage: Token usage dict
        finish_reason: Why the model stopped generating
    """

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""

    def to_chat_response(self) -> dict[str, Any]:
        """Convert to the standard chat() response format

        Returns:
            dict compatible with the return type of chat(), including
            content, tool_calls (if any), usage, and finish_reason.
        """
        result: dict[str, Any] = {"content": self.content}
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.usage:
            result["usage"] = self.usage
        if self.finish_reason:
            result["finish_reason"] = self.finish_reason
        return result


class OpenAIStreamParser:
    """Parse OpenAI-compatible SSE stream into structured StreamEvents

    Handles the OpenAI Chat Completions streaming format with tool_calls.
    Tracks multiple tool_calls by index, emitting events as they arrive.

    Usage:
        parser = OpenAIStreamParser()
        async for event in parser.parse(sse_line_iterable):
            # handle event
        result = parser.get_result()
    """

    def __init__(self) -> None:
        self._accumulators: dict[int, ToolCallAccumulator] = {}
        self._content_parts: list[str] = []
        self._usage: dict[str, int] = {}
        self._finish_reason: str = ""
        self._started_tool_indices: set[int] = set()
        self._completed_tool_indices: set[int] = set()

    def _get_or_create_accumulator(self, index: int) -> ToolCallAccumulator:
        """Get or create a ToolCallAccumulator for the given index"""
        if index not in self._accumulators:
            self._accumulators[index] = ToolCallAccumulator(index=index)
        return self._accumulators[index]

    def parse_chunk(self, chunk: dict) -> list[StreamEvent]:
        """Parse a single SSE chunk into StreamEvents

        Args:
            chunk: Parsed JSON object from an SSE "data: " line

        Returns:
            List of StreamEvents emitted by this chunk (usually 0-2)
        """
        events: list[StreamEvent] = []
        choice = chunk.get("choices", [{}])[0] if chunk.get("choices") else {}
        if not choice:
            return events

        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Handle finish_reason
        if finish_reason:
            self._finish_reason = finish_reason

        # 1. Text content delta
        text = delta.get("content", "")
        if text:
            self._content_parts.append(text)
            events.append(StreamEvent(
                event_type=StreamEventType.TEXT_DELTA,
                text=text,
            ))

        # 2. Tool calls delta
        tool_calls_delta = delta.get("tool_calls")
        if tool_calls_delta and isinstance(tool_calls_delta, list):
            for tc_delta in tool_calls_delta:
                tc_index = tc_delta.get("index", 0)
                acc = self._get_or_create_accumulator(tc_index)

                # Tool call start: first delta for this index has id + name
                if tc_index not in self._started_tool_indices:
                    tc_id = tc_delta.get("id", "")
                    func = tc_delta.get("function", {})
                    tc_name = func.get("name", "")

                    if tc_id:
                        acc.call_id = tc_id
                    if tc_name:
                        acc.name = tc_name

                    self._started_tool_indices.add(tc_index)
                    events.append(StreamEvent(
                        event_type=StreamEventType.TOOL_CALL_START,
                        tool_call_index=tc_index,
                        tool_call_id=acc.call_id,
                        tool_name=acc.name,
                    ))

                # Arguments delta
                func = tc_delta.get("function", {})
                args_delta = func.get("arguments", "")
                if args_delta:
                    acc.append_arguments(args_delta)
                    events.append(StreamEvent(
                        event_type=StreamEventType.TOOL_CALL_DELTA,
                        tool_call_index=tc_index,
                        tool_call_id=acc.call_id,
                        tool_name=acc.name,
                        tool_arguments_delta=args_delta,
                    ))

                    # Check if arguments are now complete
                    if acc.is_arguments_complete() and tc_index not in self._completed_tool_indices:
                        self._completed_tool_indices.add(tc_index)
                        events.append(StreamEvent(
                            event_type=StreamEventType.TOOL_CALL_COMPLETE,
                            tool_call_index=tc_index,
                            tool_call_id=acc.call_id,
                            tool_name=acc.name,
                            tool_arguments=acc.parse_arguments(),
                        ))

        # 3. Usage (some APIs include usage in the final chunk)
        if chunk.get("usage"):
            self._usage = chunk["usage"]
            events.append(StreamEvent(
                event_type=StreamEventType.USAGE,
                usage=chunk["usage"],
            ))

        return events

    def finalize(self) -> list[StreamEvent]:
        """Finalize the stream, emitting any remaining tool_call_complete events

        Called after the stream ends ([DONE] marker or connection close).
        Some APIs don't send complete arguments in delta form - they only
        become parseable after all fragments are received.

        Returns:
            List of remaining StreamEvents (tool_call_complete for any
            accumulators that weren't already marked complete)
        """
        events: list[StreamEvent] = []

        # Finalize any incomplete tool calls (only those not already completed)
        for index in sorted(self._accumulators.keys()):
            acc = self._accumulators[index]
            # Only emit TOOL_CALL_COMPLETE for indices not already completed during streaming
            if acc.arguments_buffer.strip() and acc.name and index not in self._completed_tool_indices:
                self._completed_tool_indices.add(index)
                parsed = acc.parse_arguments()
                events.append(StreamEvent(
                    event_type=StreamEventType.TOOL_CALL_COMPLETE,
                    tool_call_index=acc.index,
                    tool_call_id=acc.call_id,
                    tool_name=acc.name,
                    tool_arguments=parsed,
                ))

        # Done event
        events.append(StreamEvent(
            event_type=StreamEventType.DONE,
            finish_reason=self._finish_reason,
        ))

        return events

    def get_result(self) -> StreamingChatResult:
        """Build the final StreamingChatResult from accumulated data

        Returns:
            StreamingChatResult with full content, tool_calls, usage
        """
        tool_calls: list[dict[str, Any]] = []
        for index in sorted(self._accumulators.keys()):
            acc = self._accumulators[index]
            if acc.name:
                tool_calls.append(acc.to_tool_call_dict())

        return StreamingChatResult(
            content="".join(self._content_parts),
            tool_calls=tool_calls,
            usage=self._usage,
            finish_reason=self._finish_reason,
        )


class AnthropicStreamParser:
    """Parse Anthropic SSE stream into structured StreamEvents

    Handles the Anthropic Messages API streaming format with tool_use.
    Anthropic uses a different event structure than OpenAI:
      - content_block_start: begins a text or tool_use block
      - content_block_delta: text_delta or input_json_delta
      - content_block_stop: ends a block
      - message_start / message_delta: usage info

    Usage:
        parser = AnthropicStreamParser()
        async for event in parser.parse(sse_line_iterable):
            # handle event
        result = parser.get_result()
    """

    def __init__(self) -> None:
        self._accumulators: dict[int, ToolCallAccumulator] = {}
        self._content_parts: list[str] = []
        self._usage: dict[str, int] = {}
        self._finish_reason: str = ""
        self._current_block_index: int = -1
        self._completed_tool_indices: set[int] = set()

    def _get_or_create_accumulator(self, index: int) -> ToolCallAccumulator:
        """Get or create a ToolCallAccumulator for the given block index"""
        if index not in self._accumulators:
            self._accumulators[index] = ToolCallAccumulator(index=index)
        return self._accumulators[index]

    def parse_event(self, event: dict) -> list[StreamEvent]:
        """Parse a single Anthropic SSE event into StreamEvents

        Args:
            event: Parsed JSON object from an Anthropic SSE "data: " line

        Returns:
            List of StreamEvents emitted by this event
        """
        events: list[StreamEvent] = []
        event_type = event.get("type", "")

        # 1. message_start: contains initial usage
        if event_type == "message_start":
            msg = event.get("message", {})
            if msg.get("usage"):
                self._usage["prompt_tokens"] = msg["usage"].get("input_tokens", 0)
                events.append(StreamEvent(
                    event_type=StreamEventType.USAGE,
                    usage=self._usage.copy(),
                ))
            return events

        # 2. content_block_start: begins a text or tool_use block
        if event_type == "content_block_start":
            block = event.get("content_block", {})
            block_index = event.get("index", 0)
            self._current_block_index = block_index
            block_type = block.get("type", "")

            if block_type == "tool_use":
                acc = self._get_or_create_accumulator(block_index)
                acc.call_id = block.get("id", "")
                acc.name = block.get("name", "")
                events.append(StreamEvent(
                    event_type=StreamEventType.TOOL_CALL_START,
                    tool_call_index=block_index,
                    tool_call_id=acc.call_id,
                    tool_name=acc.name,
                ))
            # text blocks don't need a start event
            return events

        # 3. content_block_delta: text or tool arguments
        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            delta_type = delta.get("type", "")

            if delta_type == "text_delta":
                text = delta.get("text", "")
                if text:
                    self._content_parts.append(text)
                    events.append(StreamEvent(
                        event_type=StreamEventType.TEXT_DELTA,
                        text=text,
                    ))

            elif delta_type == "input_json_delta":
                block_index = self._current_block_index
                acc = self._get_or_create_accumulator(block_index)
                partial_json = delta.get("partial_json", "")
                if partial_json:
                    acc.append_arguments(partial_json)
                    events.append(StreamEvent(
                        event_type=StreamEventType.TOOL_CALL_DELTA,
                        tool_call_index=block_index,
                        tool_call_id=acc.call_id,
                        tool_name=acc.name,
                        tool_arguments_delta=partial_json,
                    ))
            return events

        # 4. content_block_stop: a block is complete
        if event_type == "content_block_stop":
            block_index = self._current_block_index
            if block_index in self._accumulators:
                acc = self._accumulators[block_index]
                if acc.name and block_index not in self._completed_tool_indices:
                    self._completed_tool_indices.add(block_index)
                    events.append(StreamEvent(
                        event_type=StreamEventType.TOOL_CALL_COMPLETE,
                        tool_call_index=block_index,
                        tool_call_id=acc.call_id,
                        tool_name=acc.name,
                        tool_arguments=acc.parse_arguments(),
                    ))
            return events

        # 5. message_delta: final usage
        if event_type == "message_delta":
            delta = event.get("delta", {})
            if delta.get("stop_reason"):
                self._finish_reason = delta["stop_reason"]
            if event.get("usage"):
                self._usage["completion_tokens"] = event["usage"].get("output_tokens", 0)
                events.append(StreamEvent(
                    event_type=StreamEventType.USAGE,
                    usage=self._usage.copy(),
                ))
            return events

        return events

    def finalize(self) -> list[StreamEvent]:
        """Finalize the stream, emitting any remaining events

        Returns:
            List of remaining StreamEvents
        """
        events: list[StreamEvent] = []

        # Finalize any incomplete tool calls
        for index in sorted(self._accumulators.keys()):
            if index not in self._completed_tool_indices:
                acc = self._accumulators[index]
                if acc.name:
                    self._completed_tool_indices.add(index)
                    events.append(StreamEvent(
                        event_type=StreamEventType.TOOL_CALL_COMPLETE,
                        tool_call_index=acc.index,
                        tool_call_id=acc.call_id,
                        tool_name=acc.name,
                        tool_arguments=acc.parse_arguments(),
                    ))

        # Done event
        events.append(StreamEvent(
            event_type=StreamEventType.DONE,
            finish_reason=self._finish_reason,
        ))

        return events

    def get_result(self) -> StreamingChatResult:
        """Build the final StreamingChatResult from accumulated data

        Returns:
            StreamingChatResult with full content, tool_calls, usage
        """
        tool_calls: list[dict[str, Any]] = []
        for index in sorted(self._accumulators.keys()):
            acc = self._accumulators[index]
            if acc.name:
                tool_calls.append(acc.to_tool_call_dict())

        return StreamingChatResult(
            content="".join(self._content_parts),
            tool_calls=tool_calls,
            usage=self._usage,
            finish_reason=self._finish_reason,
        )
