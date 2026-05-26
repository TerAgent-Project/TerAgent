# Streaming Execution

TerAgent supports streaming tool execution that significantly reduces latency between model output and tool execution.

## Overview

When the model streams `tool_use` blocks, read-only tools can be executed **immediately** during the stream, rather than waiting for the full stream to complete. This is particularly valuable when the model outputs multiple tool calls — early tools can start executing while later ones are still being streamed.

## Dispatch Strategy

The `StreamingToolExecutor` dispatches tool calls based on their safety attributes:

| Tool Safety Attributes | Execution Strategy |
|------------------------|--------------------|
| `read_only` + `concurrency_safe` | **Immediate** async execution during stream (no waiting) |
| Non-read-only or non-concurrency-safe | **Queued** for serial execution after stream ends |
| Unknown tool | **Queued** (conservative default) |

### Example

```
Model streams:
  Tool call 1: read_file(src/main.py)     ← read_only + concurrency_safe → IMMEDIATE
  Tool call 2: read_file(src/utils.py)     ← read_only + concurrency_safe → IMMEDIATE
  Tool call 3: write_file(src/output.py)   ← SAFE_WRITE → QUEUED
  Tool call 4: read_file(src/config.yaml)  ← read_only + concurrency_safe → IMMEDIATE
  Stream ends
  Tool call 3: write_file(src/output.py)   ← Execute serially (write operation)
```

## Degradation Path

```
Streaming + tool_use
         ↓ failed
     Streaming retry (up to N times)
         ↓ failed
     Batch fallback (ToolOrchestrator.execute_batch())
```

1. **First attempt**: Stream the model output and dispatch tools
2. **Streaming retry**: If streaming fails, retry up to `max_streaming_retries` times
3. **Batch fallback**: If all streaming retries fail, fall back to non-streaming batch mode

## Configuration

### AgentLoop Streaming Mode

```python
from teragent import AgentLoop

loop = AgentLoop(
    model=provider,
    tool_registry=registry,
    streaming_executor=StreamingToolExecutor(registry),
)

# Set streaming mode
loop.set_streaming_config(
    mode="auto",  # "auto" | "streaming" | "batch"
    max_streaming_retries=2,
)
```

**Mode options:**
- `"auto"` — Check model capabilities; use streaming if supported
- `"streaming"` — Always use streaming (fallback to batch if no executor)
- `"batch"` — Always use batch mode (no streaming)

### StreamingToolExecutor

```python
from teragent.streaming import StreamingToolExecutor

executor = StreamingToolExecutor(
    tool_registry=registry,
    permission_level=0,
    max_concurrent=10,  # Max concurrent read-only tool executions
)
```

## Usage

### Via AgentLoop (Recommended)

```python
loop = AgentLoop(
    model=provider,
    tool_registry=registry,
    streaming_executor=executor,
)

# The loop automatically determines streaming vs batch
messages = await loop.run("Read all Python files and fix the bugs")
```

### Direct Usage

```python
from teragent.core.tap import TAPRequest, CompiledPrompt

# Build a compiled prompt
compiled = CompiledPrompt(messages=messages, tools=tools)

# Create the stream
stream = provider.adapter.stream(compiled, provider.model)

# Execute streaming
results, streaming_result, stats = await executor.execute_streaming(
    stream,
    on_text_delta=lambda text: print(text, end=""),  # Real-time text
    on_tool_complete=lambda tc, r: print(f"Tool done: {tc['name']}"),  # Tool completion
    on_progress=lambda msg, frac: print(f"Progress: {msg} ({frac:.0%})"),  # Progress
)

# results: [(tool_call_dict, ToolResult), ...] in original order
# streaming_result: StreamingChatResult with content, tool_calls, usage
# stats: StreamingExecutionStats
```

### Batch Fallback

```python
results, stats = await executor.execute_batch_fallback(
    tool_calls=[...],
    on_progress=lambda msg, frac: ...,
)
# stats.fallback_used = True
```

## Stream Events

The streaming system processes these event types:

| Event | Description |
|-------|-------------|
| `TEXT_DELTA` | Incremental text from the model |
| `TOOL_CALL_START` | Beginning of a tool call block |
| `TOOL_CALL_COMPLETE` | Tool call arguments fully received |
| `USAGE` | Token usage information |
| `ERROR` | Stream error |
| `DONE` | Stream complete |

### Stream Parsers

| Parser | Protocol |
|--------|----------|
| `OpenAIStreamParser` | OpenAI `/chat/completions` SSE format |
| `AnthropicStreamParser` | Anthropic `/messages` SSE format |

## Execution Statistics

`StreamingExecutionStats` provides detailed metrics:

```python
stats.to_dict()
# → {
#     "total_tool_calls": 4,
#     "immediate_executions": 3,  # Read-only tools executed during stream
#     "queued_executions": 1,     # Write tools executed after stream
#     "parallel_groups": 1,       # Parallel execution groups
#     "streaming_time_ms": 1200,  # Time to receive stream
#     "execution_time_ms": 800,   # Time to execute tools
#     "fallback_used": False,     # Whether batch fallback was used
# }
```

## Tool Safety Declaration

Tools declare their safety attributes, which the streaming executor uses for dispatch:

```python
from teragent.tools import BaseTool, ToolResult
from teragent.core.types import ToolSafety

class ReadFileTool(BaseTool):
    name = "read_file"
    _safety = ToolSafety.READ_ONLY
    _concurrency_safe = True  # Can be executed in parallel
    ...

class WriteFileTool(BaseTool):
    name = "write_file"
    _safety = ToolSafety.SAFE_WRITE
    _concurrency_safe = False  # Must be executed serially
    ...
```

## Checking Streaming Capability

```python
# Check if the model supports streaming tool calls
can_stream = executor.can_stream_with_tools(provider)
# → Checks: streaming AND tool_calling AND streaming_tool_calling capabilities
```

## Execution Plan Preview

For debugging and TUI purposes, you can preview the execution plan without actually executing:

```python
plan = executor.get_execution_plan(pending_calls)
# → {
#     "immediate": [{"index": 0, "name": "read_file", ...}],
#     "queued": [{"index": 2, "name": "write_file", ...}],
#     "total": 4,
#     "immediate_count": 3,
#     "queued_count": 1,
# }
```
