"""teragent.tools.agent_tool — Agent-as-Tool implementation

Wraps an Agent as a BaseTool, enabling hierarchical agent delegation.
When a parent agent calls an AgentTool, the sub-agent's TAP compilation
chain is executed, and control returns to the parent agent.

This is the key difference from Handoff: AgentTool returns control to
the parent, while Handoff transfers control permanently.

Phase 3 (W12) additions:
    - Optional result caching via ResultCache
    - Controlled by cache parameter in __init__ and cache_ttl

Design reference:
    - OpenAI Agents SDK: Agent-as-Tool pattern
    - Google ADK: Agent.as_tool()
    - Claude-Code: Tool wrapping for sub-agent invocation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from teragent.core.tap import TAPRequest
from teragent.core.types import ToolSafety
from teragent.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from teragent.orchestration.agent import Agent
    from teragent.tools.result_cache import ResultCache

logger = logging.getLogger(__name__)

__all__ = [
    "AgentTool",
]


class AgentTool(BaseTool):
    """将Agent封装为工具

    父Agent调用时，内部执行子Agent的TAP编译链路。
    控制权返回父Agent（与Handoff的区别）。

    参考 OpenAI Agents SDK 的 Agent-as-Tool 模式。

    Phase 3 (W12) 新增:
    - 可选的结果缓存: 重复调用相同参数时直接返回缓存结果
    - 通过 cache 参数启用，cache_ttl 控制缓存时间

    Usage::

        from teragent.orchestration import Agent
        from teragent.tools.agent_tool import AgentTool

        # Create a sub-agent
        coder = Agent(
            name="coder",
            description="Writes code based on specifications",
            provider=some_provider,
            tools=[read_file, write_file],
        )

        # Wrap as a tool for the parent agent (no caching)
        coder_tool = AgentTool(agent=coder)

        # Wrap with result caching (TTL 60s)
        from teragent.tools.result_cache import ResultCache
        cache = ResultCache(max_size=128, default_ttl=60.0)
        coder_tool = AgentTool(agent=coder, cache=cache, cache_ttl=60.0)

        # Or use the convenience method
        coder_tool = coder.as_tool()

        # Register with parent's tool registry
        registry.register(coder_tool)
    """

    _safety = ToolSafety.SAFE_WRITE
    _concurrency_safe = False

    def __init__(
        self,
        agent: Agent,
        tool_name: str | None = None,
        tool_description: str | None = None,
        output_extractor: Callable[[str], str] | None = None,
        cache: ResultCache | None = None,
        cache_ttl: float | None = None,
    ) -> None:
        """Initialize AgentTool with an Agent instance.

        Args:
            agent: The Agent to wrap as a tool
            tool_name: Override tool name (default: "use_{agent.name}")
            tool_description: Override tool description (default: agent.description)
            output_extractor: Optional callable to extract/transform the agent's output
            cache: Optional ResultCache instance for caching execution results.
                When provided, execute() will check the cache before running
                the sub-agent, and store results in the cache after execution.
            cache_ttl: TTL (seconds) for cached results. None uses the cache's
                default_ttl. 0 means never cache (but cache is still used for
                stats tracking). Only used when cache is not None.
        """
        self._agent = agent
        self._output_extractor = output_extractor
        self._cache = cache
        self._cache_ttl = cache_ttl

        self.name = tool_name or f"use_{agent.name}"
        self.description = tool_description or agent.description
        self.parameters_schema = self._default_schema()

    def _default_schema(self) -> dict:
        """默认参数 schema

        Generates a simple schema with 'task' (required) and 'context' (optional)
        parameters, suitable for most agent delegation use cases.
        """
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": f"Task to delegate to {self._agent.name}",
                },
                "context": {
                    "type": "string",
                    "description": "Additional context for the agent",
                },
            },
            "required": ["task"],
        }

    async def execute(
        self,
        params: dict,
        progress_callback: Any | None = None,
    ) -> ToolResult:
        """执行子Agent

        通过 TAP 请求调用子Agent的 provider，
        返回结果作为 ToolResult 给父Agent。

        如果设置了缓存，会先检查缓存，命中则直接返回缓存结果。
        缓存仅对成功的执行结果生效，失败结果不会被缓存。

        Args:
            params: Tool parameters, must include 'task'
            progress_callback: Optional progress callback (forwarded to sub-agent)

        Returns:
            ToolResult with the sub-agent's output
        """
        # 缓存查找
        if self._cache is not None:
            cache_key = self._cache.make_key(self.name, params)
            cached = await self._cache.get(cache_key)
            if cached is not None:
                logger.debug(
                    "AgentTool '%s' cache hit for key '%s'",
                    self.name, cache_key,
                )
                # 缓存命中：返回副本，避免修改缓存对象
                if isinstance(cached, ToolResult):
                    # 创建新的 ToolResult 副本，添加缓存命中标记
                    result_copy = ToolResult(
                        success=cached.success,
                        data=cached.data.copy() if isinstance(cached.data, dict) else cached.data,
                        error=cached.error,
                        metadata={
                            **(cached.metadata or {}),
                            "cache_hit": True,
                            "cache_key": cache_key,
                        },
                        safety=cached.safety,
                    )
                    if cached.extra_messages:
                        result_copy.extra_messages = list(cached.extra_messages)
                    return result_copy
                # 非 ToolResult 类型的缓存值，包装为 ToolResult
                return ToolResult(
                    success=True,
                    data={"output": cached},
                    metadata={"cache_hit": True, "cache_key": cache_key},
                    safety=self._safety,
                )

        task = params.get("task", params.get("input", str(params)))
        context = params.get("context", "")

        # 构建指令
        instruction = task
        if context:
            instruction = f"{task}\n\nContext: {context}"

        tap_request = TAPRequest(
            instruction=instruction,
            meta={"intent": "sub_agent", "parent_call": True},
        )

        # 解析 provider
        try:
            provider = self._agent.resolve_provider()
        except ValueError as e:
            return ToolResult(
                success=False,
                error=str(e),
                safety=self._safety,
            )

        # 执行
        try:
            response = await provider.execute_tap(tap_request)

            output = response.raw_text or ""
            if self._output_extractor:
                output = self._output_extractor(output)

            result = ToolResult(
                success=True,
                data={"output": output},
                metadata={
                    "agent": self._agent.name,
                    "usage": response.usage,
                },
                safety=self._safety,
            )

            # 缓存成功结果
            if self._cache is not None and result.success:
                cache_key = self._cache.make_key(self.name, params)
                await self._cache.set(cache_key, result, ttl=self._cache_ttl)
                logger.debug(
                    "AgentTool '%s' cached result for key '%s'",
                    self.name, cache_key,
                )

            return result
        except Exception as e:
            logger.error(f"AgentTool '{self.name}' execution failed: {e}")
            # 失败结果不缓存
            return ToolResult(
                success=False,
                error=f"Agent '{self._agent.name}' execution failed: {e}",
                safety=self._safety,
            )

    def validate_input(self, params: dict) -> list[str]:
        """Validate that required 'task' parameter is present.

        Args:
            params: Tool parameters to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = super().validate_input(params)
        # Additional validation: task must be a non-empty string
        task = params.get("task", "")
        if isinstance(task, str) and not task.strip():
            errors.append("task must be a non-empty string")
        return errors

    def describe_usage(self, params: dict) -> str:
        """动态描述当前工具调用（供 TUI 展示）

        Args:
            params: Tool parameters

        Returns:
            Human-readable description of this agent invocation
        """
        task = params.get("task", "")
        task_preview = task[:50] + "..." if len(task) > 50 else task
        return f"Delegate to {self._agent.name}: {task_preview}"

    def invalidate_cache(self, params: dict | None = None) -> bool:
        """使缓存失效

        如果指定 params，仅使该参数组合的缓存失效。
        如果不指定 params，使该工具的所有缓存失效。

        Args:
            params: 工具参数字典，None 表示清除该工具的所有缓存

        Returns:
            True 如果有缓存被清除，False 如果没有
        """
        if self._cache is None:
            return False

        if params is not None:
            cache_key = self._cache.make_key(self.name, params)
            # invalidate 是 async 方法，但这里提供同步接口
            # 使用 try/except 处理无事件循环的情况
            try:
                loop = __import__("asyncio").get_running_loop()
                loop.create_task(self._cache.invalidate(cache_key))
                return True
            except RuntimeError:
                logger.warning(
                    "AgentTool '%s': invalidate_cache called outside async context, "
                    "use async invalidate_cache_async instead",
                    self.name,
                )
                return False
        else:
            # 清除该工具的所有缓存键
            try:
                loop = __import__("asyncio").get_running_loop()
                loop.create_task(self._invalidate_all_tool_cache())
                return True
            except RuntimeError:
                logger.warning(
                    "AgentTool '%s': invalidate_cache called outside async context, "
                    "use async invalidate_cache_async instead",
                    self.name,
                )
                return False

    async def invalidate_cache_async(self, params: dict | None = None) -> bool:
        """异步使缓存失效

        如果指定 params，仅使该参数组合的缓存失效。
        如果不指定 params，使该工具的所有缓存失效。

        Args:
            params: 工具参数字典，None 表示清除该工具的所有缓存

        Returns:
            True 如果有缓存被清除，False 如果没有
        """
        if self._cache is None:
            return False

        if params is not None:
            cache_key = self._cache.make_key(self.name, params)
            return await self._cache.invalidate(cache_key)
        else:
            return await self._invalidate_all_tool_cache()

    async def _invalidate_all_tool_cache(self) -> bool:
        """使该工具的所有缓存失效

        扫描缓存，删除以工具名开头的所有键。

        Returns:
            True 如果有缓存被清除
        """
        if self._cache is None:
            return False

        # 前缀匹配删除
        prefix = f"{self.name}:"
        keys_to_remove = [
            key for key in self._cache._cache  # noqa: SLF001
            if key.startswith(prefix)
        ]

        for key in keys_to_remove:
            await self._cache.invalidate(key)

        if keys_to_remove:
            logger.debug(
                "AgentTool '%s': invalidated %d cache entries",
                self.name, len(keys_to_remove),
            )
        return len(keys_to_remove) > 0

    def __repr__(self) -> str:
        cache_info = f", cached=True" if self._cache is not None else ""
        return f"AgentTool(name={self.name!r}, agent={self._agent.name!r}{cache_info})"
