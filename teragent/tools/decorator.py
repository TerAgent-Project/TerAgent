"""teragent.tools.decorator — @tool 装饰器

将普通 Python 函数封装为 BaseTool 实例（DecoratorTool），支持：
- 自动从函数签名生成 JSON Schema
- 自动从 docstring 提取工具描述
- 安全级别标注
- 并发安全标记
- RunContext 注入
- 异步/同步函数支持

用法:
    @tool
    def read_config(path: str) -> str:
        '''Read config file'''
        ...

    @tool(safety=ToolSafety.READ_ONLY, concurrency_safe=True)
    def search(query: str, limit: int = 10) -> list:
        '''Search for items'''
        ...

    @tool(requires_context=True)
    def process_with_context(ctx: RunContext, data: str) -> str:
        '''Process with run context'''
        ...
"""
from __future__ import annotations

import inspect
from functools import wraps
from typing import Callable, Any, TYPE_CHECKING

from teragent.tools.base import BaseTool, ToolResult
from teragent.core.types import ToolSafety
from teragent.tools.schema_gen import generate_schema_from_hints

if TYPE_CHECKING:
    from teragent.orchestration.run_context import RunContext


class DecoratorTool(BaseTool):
    """从 @tool 装饰器创建的工具

    将普通 Python 函数封装为 BaseTool 实例。
    自动从函数签名生成 JSON Schema，从 docstring 提取描述。

    支持的函数返回类型:
    - ToolResult: 直接返回
    - dict: 封装为 ToolResult(success=True, data=result)
    - str: 封装为 ToolResult(success=True, data={"output": result})
    - bool: 封装为 ToolResult(success=result, data={"result": result})
    - 其他: 封装为 ToolResult(success=True, data={"output": str(result)})

    Attributes:
        _func: 被装饰的原始函数
        _requires_context: 是否需要注入 RunContext
        _cache_enabled: 是否启用缓存
        _needs_approval: 是否需要审批
    """

    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        safety: ToolSafety = ToolSafety.SAFE_WRITE,
        concurrency_safe: bool = False,
        requires_context: bool = False,
        cache_enabled: bool = False,
        needs_approval: bool = False,
    ):
        """初始化 DecoratorTool

        Args:
            func: 被装饰的 Python 函数
            name: 自定义工具名称，默认为函数名
            description: 自定义工具描述，默认为函数 docstring
            safety: 工具安全级别
            concurrency_safe: 是否可并发执行
            requires_context: 是否需要 RunContext 注入
            cache_enabled: 是否启用结果缓存
            needs_approval: 是否需要用户审批
        """
        self._func = func
        self._requires_context = requires_context
        self._cache_enabled = cache_enabled
        self._needs_approval = needs_approval

        self.name = name or func.__name__
        self.description = description or inspect.getdoc(func) or f"Tool: {self.name}"
        self._safety = safety
        self._concurrency_safe = concurrency_safe
        self.parameters_schema = generate_schema_from_hints(func, requires_context)

    async def execute(
        self,
        params: dict,
        progress_callback=None,
    ) -> ToolResult:
        """执行被装饰的函数

        自动处理:
        - RunContext 注入（如果 requires_context=True）
        - 同步/异步函数适配
        - 返回值类型转换

        Args:
            params: 工具参数字典
            progress_callback: 进度回调（未使用）

        Returns:
            ToolResult 实例
        """
        try:
            # 构建调用参数
            kwargs = dict(params)

            # 如果函数需要 RunContext，从 params 中提取
            ctx = kwargs.pop("ctx", None) or kwargs.pop("context", None)
            if self._requires_context and ctx is not None:
                kwargs["ctx"] = ctx

            # 调用函数（适配同步/异步）
            if inspect.iscoroutinefunction(self._func):
                result = await self._func(**kwargs)
            else:
                result = self._func(**kwargs)

            # 格式化结果
            if isinstance(result, ToolResult):
                return result
            if isinstance(result, dict):
                return ToolResult(success=True, data=result)
            if isinstance(result, str):
                return ToolResult(success=True, data={"output": result})
            if isinstance(result, bool):
                return ToolResult(success=result, data={"result": result})

            return ToolResult(success=True, data={"output": str(result)})

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                data={"exception": type(e).__name__},
            )

    def __repr__(self) -> str:
        return (
            f"DecoratorTool(name={self.name!r}, "
            f"func={self._func.__name__}, "
            f"safety={self._safety.value})"
        )


def tool(
    func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    safety: ToolSafety = ToolSafety.SAFE_WRITE,
    concurrency_safe: bool = False,
    requires_context: bool = False,
    cache_enabled: bool = False,
    needs_approval: bool = False,
) -> Any:
    """将 Python 函数转换为 BaseTool 的装饰器

    支持两种用法:

    1. 无参数装饰器（@tool）:
        @tool
        def read_config(path: str) -> str:
            '''Read config file'''
            ...

    2. 带参数装饰器（@tool(...)）:
        @tool(safety=ToolSafety.READ_ONLY, concurrency_safe=True)
        def search(query: str, limit: int = 10) -> list:
            '''Search for items'''
            ...

    Args:
        func: 被装饰的函数（无括号调用时自动传入）
        name: 自定义工具名称，默认为函数名
        description: 自定义工具描述，默认为函数 docstring
        safety: 工具安全级别，默认 SAFE_WRITE
        concurrency_safe: 是否可并发执行，默认 False
        requires_context: 是否需要 RunContext 注入，默认 False
        cache_enabled: 是否启用结果缓存，默认 False
        needs_approval: 是否需要用户审批，默认 False

    Returns:
        DecoratorTool 实例（当用作装饰器时）
        或装饰器函数（当带参数调用时）
    """

    def decorator(fn: Callable) -> DecoratorTool:
        return DecoratorTool(
            func=fn,
            name=name,
            description=description,
            safety=safety,
            concurrency_safe=concurrency_safe,
            requires_context=requires_context,
            cache_enabled=cache_enabled,
            needs_approval=needs_approval,
        )

    if func is not None:
        # @tool without parentheses — 直接装饰
        return decorator(func)

    # @tool(...) with parentheses — 返回装饰器
    return decorator
