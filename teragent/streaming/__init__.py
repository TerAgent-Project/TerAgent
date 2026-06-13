# teragent/streaming/__init__.py
"""Streaming 模块 -- 流式聊天事件与工具执行

核心组件:
  - StreamEventType: 流式事件类型枚举
  - StreamEvent: 单个流式事件
  - ToolCallAccumulator: 工具调用增量累积器
  - StreamingChatResult: 流式聊天最终结果
  - OpenAIStreamParser: OpenAI SSE 流解析器
  - AnthropicStreamParser: Anthropic SSE 流解析器
  - StreamingToolExecutor: 流式工具执行器
  - StreamingExecutionStats: 流式执行统计
"""

from teragent.streaming.stream_events import (
    AnthropicStreamParser,
    OpenAIStreamParser,
    StreamEvent,
    StreamEventType,
    StreamingChatResult,
    ToolCallAccumulator,
)
from teragent.streaming.streaming_executor import (
    StreamingExecutionStats,
    StreamingToolExecutor,
)

__all__ = [
    "StreamEventType",
    "StreamEvent",
    "ToolCallAccumulator",
    "StreamingChatResult",
    "OpenAIStreamParser",
    "AnthropicStreamParser",
    "StreamingToolExecutor",
    "StreamingExecutionStats",
]
