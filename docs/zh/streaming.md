# 流式执行

TerAgent 支持流式工具执行，可显著降低模型输出与工具执行之间的延迟。

## 概述

当模型流式输出 `tool_use` 块时，只读工具可以在流式传输过程中**立即**执行，而无需等待整个流完成。这在模型输出多个工具调用时尤其有价值——较早的工具可以在后续工具仍在流式传输时开始执行。

## 调度策略

`StreamingToolExecutor` 根据工具的安全属性来调度工具调用：

| 工具安全属性 | 执行策略 |
|------------------------|--------------------|
| `read_only` + `concurrency_safe` | 流式传输期间**立即**异步执行（无需等待） |
| 非只读或非并发安全 | 流式传输结束后**排队**串行执行 |
| 未知工具 | **排队**（保守默认策略） |

### 示例

```
Model streams:
  Tool call 1: read_file(src/main.py)     ← read_only + concurrency_safe → IMMEDIATE
  Tool call 2: read_file(src/utils.py)     ← read_only + concurrency_safe → IMMEDIATE
  Tool call 3: write_file(src/output.py)   ← SAFE_WRITE → QUEUED
  Tool call 4: read_file(src/config.yaml)  ← read_only + concurrency_safe → IMMEDIATE
  Stream ends
  Tool call 3: write_file(src/output.py)   ← Execute serially (write operation)
```

## 降级路径

```
Streaming + tool_use
         ↓ failed
     Streaming retry (up to N times)
         ↓ failed
     Batch fallback (ToolOrchestrator.execute_batch())
```

1. **首次尝试**：流式传输模型输出并调度工具
2. **流式重试**：如果流式传输失败，最多重试 `max_streaming_retries` 次
3. **批量降级**：如果所有流式重试均失败，回退到非流式批量模式

## 配置

### AgentLoop 流式模式

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

**模式选项：**
- `"auto"` — 检查模型能力；如果支持则使用流式
- `"streaming"` — 始终使用流式（若无执行器则降级为批量）
- `"batch"` — 始终使用批量模式（不使用流式）

### StreamingToolExecutor

```python
from teragent.streaming import StreamingToolExecutor

executor = StreamingToolExecutor(
    tool_registry=registry,
    permission_level=0,
    max_concurrent=10,  # Max concurrent read-only tool executions
)
```

## 使用方式

### 通过 AgentLoop（推荐）

```python
loop = AgentLoop(
    model=provider,
    tool_registry=registry,
    streaming_executor=executor,
)

# The loop automatically determines streaming vs batch
messages = await loop.run("Read all Python files and fix the bugs")
```

### 直接使用

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

### 批量降级

```python
results, stats = await executor.execute_batch_fallback(
    tool_calls=[...],
    on_progress=lambda msg, frac: ...,
)
# stats.fallback_used = True
```

## 流事件

流式系统处理以下事件类型：

| 事件 | 描述 |
|-------|-------------|
| `TEXT_DELTA` | 模型的增量文本输出 |
| `TOOL_CALL_START` | 工具调用块开始 |
| `TOOL_CALL_COMPLETE` | 工具调用参数接收完毕 |
| `USAGE` | Token 使用信息 |
| `ERROR` | 流错误 |
| `DONE` | 流完成 |

### 流解析器

| 解析器 | 协议 |
|--------|----------|
| `OpenAIStreamParser` | OpenAI `/chat/completions` SSE 格式 |
| `AnthropicStreamParser` | Anthropic `/messages` SSE 格式 |

## 执行统计

`StreamingExecutionStats` 提供详细指标：

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

## 工具安全声明

工具声明其安全属性，流式执行器根据这些属性进行调度：

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

## 检查流式能力

```python
# Check if the model supports streaming tool calls
can_stream = executor.can_stream_with_tools(provider)
# → Checks: streaming AND tool_calling AND streaming_tool_calling capabilities
```

## 执行计划预览

用于调试和 TUI 目的，你可以在不实际执行的情况下预览执行计划：

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
