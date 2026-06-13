# teragent/streaming/streaming_executor.py
"""StreamingToolExecutor -- streaming tool executor

When the model streams tool_use blocks, read-only tools can be executed
immediately, significantly reducing the latency between model output
and tool execution.

Core design:
  1. Consume the StreamEvent stream produced by stream_with_tools()
  2. On TOOL_CALL_COMPLETE: dispatch based on tool safety attributes
     (see class docstring for dispatch rules)
  3. After the stream ends, execute queued non-read-only tools serially
  4. Return all results preserving the original tool_call order

Degradation:
  When the model does not support streaming tool calls (determined by
  can_stream_with_tools(), which checks streaming AND tool_calling AND
  streaming_tool_calling capabilities), automatically falls back to
  ToolOrchestrator.execute_batch().

Relationship with ToolOrchestrator:
  StreamingToolExecutor delegates individual tool executions to
  ToolOrchestrator (reusing the full validate -> permission -> execute
  lifecycle) but manages its own dispatch strategy (streaming-triggered
  vs batch).

Usage::

    executor = StreamingToolExecutor(tool_registry, permission_level=0)
    results = await executor.execute_streaming(
        stream=model.stream_with_tools(messages, tools),
        on_progress=progress_callback,
        on_text_delta=text_callback,
    )
    for tool_call, result in results:
        # Process results (order matches model output)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from teragent.core.types import ToolSafety
from teragent.streaming.stream_events import (
    StreamEventType,
    StreamingChatResult,
)
from teragent.tools.base import ToolResult
from teragent.tools.orchestrator import ToolOrchestrator
from teragent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class _PendingToolCall:
    """待执行的工具调用（从流式事件中提取）"""

    index: int
    call_id: str
    name: str
    arguments: dict[str, Any]
    is_read_only: bool = False
    is_concurrency_safe: bool = False


@dataclass
class StreamingExecutionStats:
    """流式执行统计信息"""

    total_tool_calls: int = 0
    immediate_executions: int = 0  # 只读工具立即执行数
    queued_executions: int = 0  # 非只读工具排队执行数
    parallel_groups: int = 0  # 并行执行组数
    streaming_time_ms: float = 0.0  # 流式接收时间
    execution_time_ms: float = 0.0  # 工具执行时间
    fallback_used: bool = False  # 是否退化到批量模式

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tool_calls": self.total_tool_calls,
            "immediate_executions": self.immediate_executions,
            "queued_executions": self.queued_executions,
            "parallel_groups": self.parallel_groups,
            "streaming_time_ms": round(self.streaming_time_ms, 1),
            "execution_time_ms": round(self.execution_time_ms, 1),
            "fallback_used": self.fallback_used,
        }


class StreamingToolExecutor:
    """Streaming tool executor -- core component.

    When the model streams tool_use blocks, read-only tools can be
    executed immediately rather than waiting for the full stream to
    complete.  This significantly reduces latency, especially when
    the model outputs multiple tool calls.

    Dispatch strategy (authoritative):
      On TOOL_CALL_COMPLETE, the tool's safety attributes are checked:
        - read_only + concurrency_safe -> immediate async execution
        - non-read-only -> queued for serial execution after the stream
        - unknown tool -> queued (conservative default)

    Concurrency safety:
      - asyncio primitives protect shared state internally
      - concurrent immediate executions are capped by a semaphore

    Degradation:
      - Model does not support streaming tool_use -> falls back to
        ToolOrchestrator.execute_batch()
      - Error during streaming -> collect completed results + batch the rest
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        permission_level: int = 0,
        max_concurrent: int = 10,
        enhanced_perm_manager=None,
    ) -> None:
        """初始化流式工具执行器

        Args:
            tool_registry: 工具注册表
            permission_level: 当前权限级别
            max_concurrent: 最大并发只读工具数
            enhanced_perm_manager: 增强权限管理器
        """
        self.tool_registry = tool_registry
        self.permission_level = permission_level
        self.max_concurrent = max_concurrent

        # 内部使用 ToolOrchestrator 的 _execute_single 逻辑
        # 但自己管理调度策略（流式触发 vs 批量）
        self._orchestrator = ToolOrchestrator(
            tool_registry, permission_level, max_concurrent,
            enhanced_perm_manager=enhanced_perm_manager,
        )

    async def execute_streaming(
        self,
        stream: Any,
        on_tool_complete: Optional[Callable[[dict, ToolResult], Awaitable[None]]] = None,
        on_text_delta: Optional[Callable[[str], Awaitable[None]]] = None,
        on_progress: Optional[Callable[[str, float], Awaitable[None]]] = None,
    ) -> tuple[list[tuple[dict, ToolResult]], StreamingChatResult, StreamingExecutionStats]:
        """Consume stream events and execute tool calls in real time.

        Iterates over the StreamEvent stream. Text deltas are forwarded
        via on_text_delta; tool calls are dispatched per the class-level
        dispatch strategy (immediate for read-only+concurrency_safe,
        queued otherwise). After the stream ends, queued tools are
        executed serially. Results preserve the original tool_call order.

        Args:
            stream: AsyncIterator[StreamEvent] from model.stream_with_tools()
            on_tool_complete: Callback per completed tool (tool_call_dict, result)
            on_text_delta: Callback for text chunks (for TUI real-time rendering)
            on_progress: Progress callback (message, fraction)

        Returns:
            (results, streaming_result, stats) tuple:
              - results: [(tool_call_dict, ToolResult), ...] in original order
              - streaming_result: StreamingChatResult with full stream output
              - stats: StreamingExecutionStats
        """
        stats = StreamingExecutionStats()
        stream_start = time.time()

        # 收集流式过程中的状态
        pending_calls: dict[int, _PendingToolCall] = {}  # index -> pending
        immediate_tasks: dict[int, asyncio.Task[ToolResult]] = {}  # index -> task
        content_parts: list[str] = []
        usage: dict[str, int] = {}
        finish_reason = ""
        error_events: list[str] = []

        # 用于收集流式结果
        streaming_result = StreamingChatResult()

        # 并发信号量
        semaphore = asyncio.Semaphore(self.max_concurrent)

        try:
            async for event in stream:
                if event.event_type == StreamEventType.TEXT_DELTA:
                    content_parts.append(event.text)
                    if on_text_delta:
                        try:
                            await on_text_delta(event.text)
                        except Exception as e:
                            logger.debug(f"on_text_delta callback error: {e}")

                elif event.event_type == StreamEventType.TOOL_CALL_START:
                    # 工具调用开始 -- 仅记录日志
                    logger.debug(
                        f"StreamingToolExecutor: tool_call_start "
                        f"index={event.tool_call_index} "
                        f"name={event.tool_name} id={event.tool_call_id}"
                    )

                elif event.event_type == StreamEventType.TOOL_CALL_COMPLETE:
                    # 工具调用参数完整 -- 判断是否立即执行
                    tc_index = event.tool_call_index
                    tc_name = event.tool_name
                    tc_id = event.tool_call_id
                    tc_args = event.tool_arguments

                    # 查找工具安全属性
                    tool = self.tool_registry.get(tc_name)
                    is_read_only = tool.is_read_only if tool else False
                    is_concurrency_safe = tool.is_concurrency_safe if tool else False

                    pending = _PendingToolCall(
                        index=tc_index,
                        call_id=tc_id,
                        name=tc_name,
                        arguments=tc_args,
                        is_read_only=is_read_only,
                        is_concurrency_safe=is_concurrency_safe,
                    )
                    pending_calls[tc_index] = pending

                    # 判断是否立即执行
                    if is_read_only and is_concurrency_safe:
                        # 只读 + 并发安全 -> 立即异步执行
                        stats.immediate_executions += 1
                        logger.info(
                            f"StreamingToolExecutor: immediate execution "
                            f"of {tc_name} (index={tc_index})"
                        )

                        # 构建 tool_call dict 供执行
                        tool_call_dict = self._build_tool_call_dict(pending)

                        # NOTE: _execute_immediate takes the tool_call_dict,
                        # semaphore, and progress callback as explicit parameters
                        # (not via closure over loop variables) to avoid the classic
                        # Python closure-over-mutable-loop-variable pitfall.
                        async def _execute_immediate(
                            tc: dict, sem: asyncio.Semaphore,
                            prog_cb: Optional[Callable],
                        ) -> ToolResult:
                            async with sem:
                                return await self._orchestrator._execute_single(
                                    tc, prog_cb
                                )

                        task = asyncio.create_task(
                            _execute_immediate(tool_call_dict, semaphore, on_progress)
                        )
                        immediate_tasks[tc_index] = task
                    else:
                        # 非只读 -> 排队等待
                        stats.queued_executions += 1
                        logger.info(
                            f"StreamingToolExecutor: queued execution "
                            f"of {tc_name} (index={tc_index}, "
                            f"read_only={is_read_only}, "
                            f"concurrent={is_concurrency_safe})"
                        )

                elif event.event_type == StreamEventType.USAGE:
                    incoming_usage = event.usage
                    if incoming_usage:
                        # 增量合并: 保留已有字段，用新数据覆盖/补充
                        usage.update(incoming_usage)

                elif event.event_type == StreamEventType.ERROR:
                    error_events.append(event.error)
                    logger.warning(
                        f"StreamingToolExecutor: stream error: {event.error[:200]}"
                    )

                elif event.event_type == StreamEventType.DONE:
                    finish_reason = event.finish_reason
                    break

        except Exception as e:
            logger.error(
                f"StreamingToolExecutor: stream iteration failed: {e}",
                exc_info=True,
            )
            error_events.append(str(e))

        # 流结束，记录流式接收时间
        stats.streaming_time_ms = (time.time() - stream_start) * 1000
        execution_start = time.time()

        # 构建 StreamingChatResult
        streaming_result.content = "".join(content_parts)
        streaming_result.usage = usage
        streaming_result.finish_reason = finish_reason

        # 收集流式结果中的 tool_calls
        # NOTE: OpenAI API requires function.arguments to be a JSON string,
        # not a dict. Convert pc.arguments (dict) to JSON string.
        for idx in sorted(pending_calls.keys()):
            pc = pending_calls[idx]
            try:
                args_str = json.dumps(pc.arguments, ensure_ascii=False)
            except (TypeError, ValueError):
                args_str = "{}"
            streaming_result.tool_calls.append({
                "id": pc.call_id or f"call_{idx}",
                "type": "function",
                "function": {
                    "name": pc.name,
                    "arguments": args_str,
                },
            })

        stats.total_tool_calls = len(pending_calls)

        # 等待所有立即执行的任务完成
        if immediate_tasks:
            stats.parallel_groups = 1  # 所有只读工具形成一个并行组
            results_map: dict[int, tuple[dict, ToolResult]] = {}

            # 等待所有立即执行的任务
            for idx, task in immediate_tasks.items():
                try:
                    result = await task
                    pc = pending_calls[idx]
                    tool_call_dict = self._build_tool_call_dict(pc)
                    results_map[idx] = (tool_call_dict, result)

                    # 回调通知
                    if on_tool_complete:
                        try:
                            await on_tool_complete(tool_call_dict, result)
                        except Exception as e:
                            logger.debug(f"on_tool_complete callback error: {e}")
                except Exception as e:
                    logger.error(
                        f"StreamingToolExecutor: immediate task {idx} failed: {e}"
                    )
                    pc = pending_calls[idx]
                    tool_call_dict = self._build_tool_call_dict(pc)
                    # Immediate tasks are always read_only+concurrency_safe,
                    # so safety=READ_ONLY is correct here
                    results_map[idx] = (
                        tool_call_dict,
                        ToolResult(
                            success=False,
                            data={},
                            error=f"流式执行失败: {e}",
                            safety=ToolSafety.READ_ONLY,
                        ),
                    )

        else:
            results_map = {}

        # 执行排队的非只读工具（串行执行）
        queued_indices = [
            idx for idx in sorted(pending_calls.keys())
            if idx not in immediate_tasks
        ]

        if queued_indices:
            for idx in queued_indices:
                pc = pending_calls[idx]
                tool_call_dict = self._build_tool_call_dict(pc)

                logger.info(
                    f"StreamingToolExecutor: executing queued tool "
                    f"{pc.name} (index={idx})"
                )

                try:
                    result = await self._orchestrator._execute_single(
                        tool_call_dict, on_progress
                    )
                except Exception as e:
                    # Look up the tool's actual safety level for accurate metadata
                    tool = self.tool_registry.get(pc.name)
                    result = ToolResult(
                        success=False,
                        data={},
                        error=f"工具执行失败: {e}",
                        safety=tool.safety_level if tool else ToolSafety.SAFE_WRITE,
                    )

                results_map[idx] = (tool_call_dict, result)

                # 回调通知
                if on_tool_complete:
                    try:
                        await on_tool_complete(tool_call_dict, result)
                    except Exception as e:
                        logger.debug(f"on_tool_complete callback error: {e}")

        # 记录执行时间
        stats.execution_time_ms = (time.time() - execution_start) * 1000

        # 按原始顺序排列结果
        ordered_results: list[tuple[dict, ToolResult]] = []
        for idx in sorted(results_map.keys()):
            ordered_results.append(results_map[idx])

        # 如果流中有错误且没有工具调用，记录日志
        if error_events and not pending_calls:
            logger.warning(
                f"StreamingToolExecutor: stream had errors but no tool_calls: "
                f"{'; '.join(e[:100] for e in error_events)}"
            )

        return ordered_results, streaming_result, stats

    async def execute_batch_fallback(
        self,
        tool_calls: list[dict],
        on_progress: Optional[Callable[[str, float], Awaitable[None]]] = None,
    ) -> tuple[list[tuple[dict, ToolResult]], StreamingExecutionStats]:
        """退化到批量执行模式

        当模型不支持流式 tool_use 时使用此方法。
        内部委托给 ToolOrchestrator.execute_batch()。

        Args:
            tool_calls: [{"name": ..., "arguments": ..., "id": ...}]
            on_progress: 进度回调

        Returns:
            (results, stats) 二元组:
              - results: [(tool_call, result), ...]
              - stats: StreamingExecutionStats（fallback_used=True）
        """
        stats = StreamingExecutionStats(
            total_tool_calls=len(tool_calls),
            fallback_used=True,
        )
        execution_start = time.time()

        results = await self._orchestrator.execute_batch(tool_calls, on_progress)

        stats.execution_time_ms = (time.time() - execution_start) * 1000
        stats.queued_executions = len(tool_calls)

        return results, stats

    def can_stream_with_tools(self, model: Any) -> bool:
        """检查模型是否支持流式 tool_use

        Args:
            model: ModelProvider 实例

        Returns:
            True 表示模型支持流式 tool_use，应使用流式执行
        """
        try:
            caps = model.capabilities()
            return bool(
                caps.get("streaming", False)
                and caps.get("tool_calling", False)
                and caps.get("streaming_tool_calling", False)
            )
        except Exception:
            return False

    def _build_tool_call_dict(self, pending: _PendingToolCall) -> dict[str, Any]:
        """将 _PendingToolCall 转换为编排器格式的 tool_call dict

        Args:
            pending: 待执行工具调用信息

        Returns:
            {"name": ..., "arguments": ..., "id": ...}
        """
        return {
            "name": pending.name,
            "arguments": pending.arguments,
            "id": pending.call_id or f"call_{pending.index}",
        }

    def get_execution_plan(
        self, pending_calls: dict[int, _PendingToolCall]
    ) -> dict[str, Any]:
        """预览执行计划（调试/TUI 用，不执行）

        Args:
            pending_calls: 从流式事件收集的待执行工具调用

        Returns:
            执行计划 dict，包含立即执行和排队执行的工具列表
        """
        immediate = []
        queued = []

        for idx in sorted(pending_calls.keys()):
            pc = pending_calls[idx]
            entry = {
                "index": idx,
                "name": pc.name,
                "call_id": pc.call_id,
                "read_only": pc.is_read_only,
                "concurrency_safe": pc.is_concurrency_safe,
            }
            if pc.is_read_only and pc.is_concurrency_safe:
                immediate.append(entry)
            else:
                queued.append(entry)

        return {
            "immediate": immediate,
            "queued": queued,
            "total": len(pending_calls),
            "immediate_count": len(immediate),
            "queued_count": len(queued),
        }

    def set_permission_level(self, level: int) -> None:
        """更新权限级别（同步更新内部编排器）"""
        self.permission_level = level
        self._orchestrator.set_permission_level(level)

    def set_enhanced_perm_manager(self, enhanced_perm_manager) -> None:
        """设置增强权限管理器（透传到内部编排器）

        Args:
            enhanced_perm_manager: EnhancedPermissionManager 实例
        """
        self._orchestrator.set_enhanced_perm_manager(enhanced_perm_manager)

    def set_hook_manager(self, hook_manager) -> None:
        """设置 Hook 管理器（透传到内部编排器）

        Args:
            hook_manager: HookManager 实例
        """
        self._orchestrator.set_hook_manager(hook_manager)
