# teragent/tools/mcp_toolset.py
"""MCP 工具集 — 连接 MCP 服务器，发现并代理工具调用

核心组件:
  - MCPToolset: MCP 工具集，管理与单个 MCP 服务器的连接、工具发现和代理调用
  - MCPTool: MCP 远程工具的本地代理（BaseTool 子类）

支持的传输模式:
  - stdio: 启动本地 MCP 服务器进程（通过 stdin/stdout 通信）
  - sse: 连接远程 MCP 服务器（通过 SSE + HTTP POST 通信）
  - streamable_http: 连接远程 MCP 服务器（通过 Streamable HTTP 通信）

设计参考:
  - OpenAI Agents SDK: MCPServer, MCPServerStdio, MCPServerSse
  - MCP Python SDK: ClientSession, stdio_client, sse_client, streamablehttp_client

用法:
    from teragent.tools.mcp_toolset import MCPToolset
    from teragent.config.mcp_config import MCPServerConfig

    # 创建配置
    config = MCPServerConfig(
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        name="filesystem",
    )

    # 创建工具集
    toolset = MCPToolset(server_params=config)

    # 连接并发现工具
    await toolset.connect()
    tools = toolset.to_base_tools()

    # 使用工具
    result = await toolset.call_tool("read_file", {"path": "/tmp/test.txt"})

    # 断开连接
    await toolset.disconnect()
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Callable, Awaitable, Optional

from teragent.core.types import ToolSafety
from teragent.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from mcp.client.session import ClientSession
    from teragent.config.mcp_config import MCPServerConfig
    from teragent.tools.registry import ToolRegistry

__all__ = [
    "MCPToolset",
    "MCPTool",
    "MCPConnectionPool",
]

logger = logging.getLogger(__name__)


# ===== MCP SDK 延迟导入 =====

_MCP_AVAILABLE: bool | None = None


def _check_mcp_available() -> None:
    """检查 mcp SDK 是否可用，不可用时抛出 ImportError

    Raises:
        ImportError: mcp 包未安装时抛出
    """
    global _MCP_AVAILABLE
    if _MCP_AVAILABLE is None:
        try:
            import mcp  # noqa: F401
            _MCP_AVAILABLE = True
        except ImportError:
            _MCP_AVAILABLE = False

    if not _MCP_AVAILABLE:
        raise ImportError(
            "mcp 包未安装。MCPToolset 需要 mcp SDK 支持远程工具调用。"
            "请运行: pip install mcp"
        )


def _extract_text_content(result_content: list[Any]) -> str:
    """从 MCP CallToolResult 的 content 列表中提取文本内容

    MCP 工具返回的 content 可能包含多种类型:
    - TextContent: 文本内容
    - ImageContent: 图片内容（跳过）
    - EmbeddedResource: 嵌入资源（跳过）

    Args:
        result_content: MCP CallToolResult.content 列表

    Returns:
        拼接后的文本内容
    """
    text_parts: list[str] = []
    for item in result_content:
        # TextContent 具有 text 属性
        item_type = getattr(item, "type", None)
        if item_type == "text":
            text = getattr(item, "text", None)
            if text is not None:
                text_parts.append(text)
        elif hasattr(item, "text"):
            # 兼容：某些实现可能不设置 type 但有 text
            text_parts.append(item.text)
    return "\n".join(text_parts) if text_parts else str(result_content)


# ===== MCPTool =====

class MCPTool(BaseTool):
    """MCP 远程工具的本地代理

    每个 MCP 服务器提供的工具被封装为 MCPTool，
    调用时通过 MCPToolset 代理到远程服务器执行。

    设计参考:
        - OpenAI Agents SDK: FunctionTool 代理 MCP 工具调用
        - Google ADK: MCP 工具代理模式

    Attributes:
        _toolset: 所属的 MCPToolset 实例
        _input_schema: MCP 服务器提供的原始 JSON Schema
    """

    _safety = ToolSafety.SAFE_WRITE
    _concurrency_safe = False

    def __init__(
        self,
        toolset: MCPToolset,
        name: str,
        description: str,
        input_schema: dict,
        safety: ToolSafety = ToolSafety.SAFE_WRITE,
    ) -> None:
        """初始化 MCPTool

        Args:
            toolset: 所属的 MCPToolset 实例（用于代理调用）
            name: 工具名称（来自 MCP 服务器的 tools/list）
            description: 工具描述（来自 MCP 服务器的 tools/list）
            input_schema: 工具参数 JSON Schema（来自 MCP 服务器的 tools/list）
            safety: 安全级别，默认 SAFE_WRITE（MCP 工具属于远程调用，保守标记）
        """
        self._toolset = toolset
        self._input_schema = input_schema

        self.name = name
        self.description = description
        self.parameters_schema = input_schema
        self._safety = safety

    async def execute(
        self,
        params: dict,
        progress_callback: Any | None = None,
    ) -> ToolResult:
        """执行 MCP 远程工具调用

        通过 MCPToolset 代理到远程 MCP 服务器执行工具调用。

        Args:
            params: 工具参数字典
            progress_callback: 进度回调（暂不使用，保留接口兼容）

        Returns:
            ToolResult: 统一返回格式
        """
        return await self._toolset.call_tool(self.name, params)

    def describe_usage(self, params: dict) -> str:
        """动态描述当前工具调用（供 TUI 展示）

        Args:
            params: 工具参数

        Returns:
            可读的调用描述
        """
        server_name = self._toolset._name or "mcp"
        return f"MCP[{server_name}].{self.name}({', '.join(f'{k}={v!r}' for k, v in list(params.items())[:3])})"

    def get_tool_prompt(self) -> str:
        """工具专属提示

        标注此工具来自 MCP 远程服务器，提醒 LLM 注意远程调用延迟。

        Returns:
            提示文本
        """
        server_name = self._toolset._name or "mcp"
        return (
            f"此工具由 MCP 服务器 '{server_name}' 提供，"
            f"通过远程调用执行，可能存在网络延迟。"
        )

    def __repr__(self) -> str:
        server_name = self._toolset._name or "mcp"
        return f"MCPTool(name={self.name!r}, server={server_name!r}, safety={self._safety.value})"


# ===== MCPToolset =====

class MCPToolset:
    """MCP 工具集 — 连接 MCP 服务器，发现并代理工具调用

    支持:
    - stdio 传输（启动本地 MCP 服务器进程）
    - HTTP/SSE 传输（连接远程 MCP 服务器）
    - 工具发现（tools/list）
    - 工具代理调用（tools/call）
    - 工具过滤（允许/阻止列表）

    设计参考:
        - OpenAI Agents SDK: MCPServer 抽象 + MCPServerStdio/MCPServerSse 实现
        - MCP Python SDK: ClientSession + transport 层

    用法:
        config = MCPServerConfig(transport="stdio", command="npx", args=[...])
        toolset = MCPToolset(server_params=config)
        await toolset.connect()
        tools = toolset.to_base_tools()
        # ... 使用工具 ...
        await toolset.disconnect()

    生命周期:
        1. 创建 MCPToolset（未连接）
        2. connect() → 建立传输通道 → 创建 ClientSession → initialize → list_tools → 缓存
        3. call_tool() / to_base_tools() → 使用缓存的工具信息
        4. disconnect() → 关闭 ClientSession → 关闭传输通道

    Attributes:
        _server_params: MCP 服务器配置
        _name: 工具集名称
        _tool_filter: 工具过滤列表
        _cache_tools: 是否缓存工具列表
        _connect_timeout: 连接超时（秒）
        _session: MCP ClientSession 实例
        _exit_stack: 用于管理异步上下文的生命周期
        _cached_tools: 缓存的 MCPTool 实例列表
        _connected: 是否已连接
    """

    def __init__(
        self,
        server_params: MCPServerConfig,
        name: str = "",
        tool_filter: list[str] | None = None,
        cache_tools: bool = True,
        connect_timeout: float = 30.0,
    ) -> None:
        """初始化 MCPToolset

        Args:
            server_params: MCP 服务器配置（MCPServerConfig 实例）
            name: 工具集名称（默认使用 server_params.name）
            tool_filter: 工具过滤列表 — 仅暴露指定名称的工具
                None 表示不过滤（暴露全部工具）
            cache_tools: 是否缓存工具列表（避免重复 list_tools 调用）
            connect_timeout: 连接超时（秒）
        """
        from teragent.config.mcp_config import MCPServerConfig

        if not isinstance(server_params, MCPServerConfig):
            raise TypeError(
                f"server_params 必须是 MCPServerConfig 类型，"
                f"实际: {type(server_params).__name__}"
            )

        self._server_params = server_params
        self._name = name or server_params.name
        self._tool_filter = tool_filter or list(server_params.tool_filter)
        self._cache_tools = cache_tools
        self._connect_timeout = connect_timeout

        # 连接状态
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._cached_tools: list[MCPTool] = []
        self._connected: bool = False

    @property
    def name(self) -> str:
        """工具集名称"""
        return self._name

    @property
    def is_connected(self) -> bool:
        """是否已连接到 MCP 服务器"""
        return self._connected and self._session is not None

    async def connect(self) -> None:
        """连接到 MCP 服务器

        执行流程:
        1. 检查 mcp SDK 可用性
        2. 根据传输模式创建传输通道
        3. 创建 ClientSession 并初始化
        4. 调用 initialize() 握手
        5. 发现工具（list_tools）
        6. 标记为已连接

        Raises:
            ImportError: mcp SDK 未安装
            ConnectionError: 连接失败
            TimeoutError: 连接超时
        """
        if self._connected:
            logger.debug(f"MCPToolset[{self._name}] 已连接，跳过重复连接")
            return

        _check_mcp_available()

        try:
            self._exit_stack = AsyncExitStack()

            if self._server_params.transport == "stdio":
                await self._connect_stdio()
            elif self._server_params.transport == "sse":
                await self._connect_sse()
            elif self._server_params.transport == "streamable_http":
                await self._connect_streamable_http()
            else:
                raise ValueError(
                    f"不支持的传输模式: {self._server_params.transport}，"
                    f"支持: stdio, sse, streamable_http"
                )

            # 初始化会话
            assert self._session is not None
            await self._session.initialize()
            logger.info(
                f"MCPToolset[{self._name}] 会话初始化成功 "
                f"(transport={self._server_params.transport})"
            )

            # 发现工具
            await self._discover_tools()

            self._connected = True
            logger.info(
                f"MCPToolset[{self._name}] 连接成功，"
                f"发现 {len(self._cached_tools)} 个工具"
            )

        except Exception as e:
            # 连接失败时清理资源
            logger.error(f"MCPToolset[{self._name}] 连接失败: {e}")
            await self._cleanup()
            raise

    async def _connect_stdio(self) -> None:
        """通过 stdio 传输连接 MCP 服务器

        使用 mcp.client.stdio.stdio_client 启动本地进程，
        并创建 ClientSession 管理通信。
        """
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.client.session import ClientSession

        assert self._exit_stack is not None

        server_params = StdioServerParameters(
            command=self._server_params.command,
            args=self._server_params.args,
            env=self._server_params.env or None,
            cwd=self._server_params.cwd or None,
        )

        # stdio_client 是异步上下文管理器，返回 (read_stream, write_stream)
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

    async def _connect_sse(self) -> None:
        """通过 SSE 传输连接 MCP 服务器

        使用 mcp.client.sse.sse_client 连接远程 SSE 端点，
        并创建 ClientSession 管理通信。
        """
        from mcp.client.sse import sse_client
        from mcp.client.session import ClientSession

        assert self._exit_stack is not None

        read_stream, write_stream = await self._exit_stack.enter_async_context(
            sse_client(
                url=self._server_params.url,
                headers=self._server_params.headers or None,
                timeout=self._server_params.timeout,
            )
        )

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

    async def _connect_streamable_http(self) -> None:
        """通过 Streamable HTTP 传输连接 MCP 服务器

        使用 mcp.client.streamable_http.streamablehttp_client 连接远程端点，
        并创建 ClientSession 管理通信。
        """
        from mcp.client.streamable_http import streamable_http_client
        from mcp.client.session import ClientSession

        assert self._exit_stack is not None

        read_stream, write_stream = await self._exit_stack.enter_async_context(
            streamable_http_client(
                url=self._server_params.url,
            )
        )

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

    async def _discover_tools(self) -> None:
        """从 MCP 服务器发现工具并创建本地代理

        调用 MCP ClientSession.list_tools() 获取服务器提供的工具列表，
        根据 tool_filter 过滤后创建 MCPTool 代理。
        """
        assert self._session is not None

        result = await self._session.list_tools()
        tools: list[MCPTool] = []

        for mcp_tool in result.tools:
            # 工具过滤
            if self._tool_filter and mcp_tool.name not in self._tool_filter:
                logger.debug(
                    f"MCPToolset[{self._name}] 工具 '{mcp_tool.name}' "
                    f"不在过滤列表中，跳过"
                )
                continue

            # 构建 input_schema
            input_schema: dict = {}
            if mcp_tool.inputSchema:
                # inputSchema 可能是 dict 或 None
                if isinstance(mcp_tool.inputSchema, dict):
                    input_schema = mcp_tool.inputSchema
                else:
                    # Pydantic BaseModel 转换
                    input_schema = mcp_tool.inputSchema

            # 创建 MCPTool 代理
            proxy = MCPTool(
                toolset=self,
                name=mcp_tool.name,
                description=mcp_tool.description or "",
                input_schema=input_schema,
            )
            tools.append(proxy)

        if self._cache_tools:
            self._cached_tools = tools

        logger.debug(
            f"MCPToolset[{self._name}] 发现 {len(result.tools)} 个工具，"
            f"过滤后保留 {len(tools)} 个"
        )

    async def disconnect(self) -> None:
        """断开与 MCP 服务器的连接

        关闭 ClientSession 和传输通道，清理所有资源。
        断开后可再次调用 connect() 重新连接。
        """
        if not self._connected:
            logger.debug(f"MCPToolset[{self._name}] 未连接，跳过断开操作")
            return

        await self._cleanup()
        self._connected = False
        logger.info(f"MCPToolset[{self._name}] 已断开连接")

    async def _cleanup(self) -> None:
        """清理连接资源

        关闭 AsyncExitStack 管理的所有异步上下文。
        """
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.warning(f"MCPToolset[{self._name}] 清理资源时出错: {e}")
            finally:
                self._exit_stack = None
                self._session = None

    async def list_tools(self) -> list[BaseTool]:
        """获取 MCP 服务器提供的工具列表

        如果启用了缓存且已发现工具，直接返回缓存。
        否则重新调用 list_tools 发现。

        Returns:
            BaseTool 实例列表（实际为 MCPTool 代理实例）
        """
        if self._cache_tools and self._cached_tools:
            return list(self._cached_tools)

        # 未连接时抛出错误
        if not self._connected or self._session is None:
            raise RuntimeError(
                f"MCPToolset[{self._name}] 未连接，"
                f"请先调用 connect()"
            )

        await self._discover_tools()
        return list(self._cached_tools)

    async def call_tool(self, name: str, arguments: dict) -> ToolResult:
        """代理调用 MCP 服务器上的工具

        通过 ClientSession.call_tool() 调用远程工具，
        并将 MCP 的 CallToolResult 转换为 TerAgent 的 ToolResult。

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            ToolResult: 统一返回格式

        Raises:
            RuntimeError: 未连接时调用
        """
        if not self._connected or self._session is None:
            return ToolResult(
                success=False,
                error=f"MCPToolset[{self._name}] 未连接，无法调用工具 '{name}'",
                safety=ToolSafety.SAFE_WRITE,
            )

        try:
            logger.debug(
                f"MCPToolset[{self._name}] 调用工具 '{name}' "
                f"参数: {list(arguments.keys())}"
            )

            result = await self._session.call_tool(name, arguments)

            # 检查 MCP 层面的错误
            if result.isError:
                error_text = _extract_text_content(result.content) if result.content else "MCP 工具调用失败"
                logger.warning(
                    f"MCPToolset[{self._name}] 工具 '{name}' 返回错误: {error_text}"
                )
                return ToolResult(
                    success=False,
                    error=error_text,
                    data={"mcp_error": True, "tool_name": name},
                    safety=ToolSafety.SAFE_WRITE,
                )

            # 提取成功结果
            text_content = _extract_text_content(result.content) if result.content else ""

            return ToolResult(
                success=True,
                data={"output": text_content},
                metadata={
                    "mcp_server": self._name,
                    "tool_name": name,
                    "content_types": [
                        getattr(item, "type", "unknown") for item in (result.content or [])
                    ],
                },
                safety=ToolSafety.SAFE_WRITE,
            )

        except Exception as e:
            logger.error(
                f"MCPToolset[{self._name}] 调用工具 '{name}' 异常: {e}"
            )
            return ToolResult(
                success=False,
                error=f"MCP 工具 '{name}' 调用异常: {e}",
                data={"exception": type(e).__name__, "tool_name": name},
                safety=ToolSafety.SAFE_WRITE,
            )

    def to_base_tools(self) -> list[BaseTool]:
        """将缓存的 MCP 工具导出为 BaseTool 列表

        返回缓存的 MCPTool 代理实例列表，可直接注册到 ToolRegistry
        或添加到 Agent 的 tools 列表。

        如果尚未连接或没有缓存工具，返回空列表。

        Returns:
            BaseTool 实例列表
        """
        return list(self._cached_tools)

    def register_to(self, registry: ToolRegistry) -> None:
        """将所有 MCP 工具注册到指定的 ToolRegistry

        Args:
            registry: ToolRegistry 实例
        """
        for tool in self._cached_tools:
            registry.register(tool)
        logger.info(
            f"MCPToolset[{self._name}] 注册了 {len(self._cached_tools)} 个工具到注册表"
        )

    def get_tool_names(self) -> list[str]:
        """获取所有已发现工具的名称列表

        Returns:
            工具名称列表
        """
        return [t.name for t in self._cached_tools]

    def get_summary(self) -> dict:
        """返回工具集摘要信息

        Returns:
            包含名称、传输模式、连接状态和工具列表的字典
        """
        return {
            "name": self._name,
            "transport": self._server_params.transport,
            "connected": self._connected,
            "tool_count": len(self._cached_tools),
            "tool_names": self.get_tool_names(),
            "tool_filter": self._tool_filter,
        }

    async def __aenter__(self) -> MCPToolset:
        """异步上下文管理器入口 — 自动连接"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器出口 — 自动断开"""
        await self.disconnect()

    def __repr__(self) -> str:
        return (
            f"MCPToolset(name={self._name!r}, "
            f"transport={self._server_params.transport!r}, "
            f"connected={self._connected}, "
            f"tools={len(self._cached_tools)})"
        )


# ===== MCPConnectionPool =====

class MCPConnectionPool:
    """MCP 连接池 — 管理多个 MCPToolset 实例的连接复用

    当多个 Agent 或组件需要连接同一个 MCP 服务器时，
    MCPConnectionPool 确保复用已有的 MCPToolset 连接，
    避免重复创建连接和进程。

    特性:
    - 按服务器名称复用 MCPToolset 连接
    - LRU 淘汰不活跃的连接
    - 连接健康检查和自动重连
    - 最大连接数限制
    - 异步安全（asyncio.Lock）

    设计参考:
        - 数据库连接池模式（HikariCP, SQLAlchemy Pool）
        - OpenAI Agents SDK: MCP 服务器连接管理

    用法::

        from teragent.tools.mcp_toolset import MCPConnectionPool
        from teragent.config.mcp_config import MCPServerConfig

        pool = MCPConnectionPool(max_connections=10)

        # 获取连接（自动创建或复用）
        config = MCPServerConfig(transport="stdio", command="npx", ...)
        toolset = await pool.get(config)

        # 使用工具
        result = await toolset.call_tool("read_file", {"path": "/tmp/test.txt"})

        # 释放连接（归还到池中）
        await pool.release(toolset)

        # 清理
        await pool.close_all()

    Args:
        max_connections: 最大连接数（默认 16）
        idle_timeout: 空闲连接超时（秒，默认 300.0）
        health_check_interval: 健康检查间隔（秒，默认 60.0，0 禁用）
    """

    def __init__(
        self,
        max_connections: int = 16,
        idle_timeout: float = 300.0,
        health_check_interval: float = 60.0,
    ) -> None:
        if max_connections <= 0:
            raise ValueError(f"max_connections must be > 0, got {max_connections}")
        if idle_timeout < 0:
            raise ValueError(f"idle_timeout must be >= 0, got {idle_timeout}")
        if health_check_interval < 0:
            raise ValueError(f"health_check_interval must be >= 0, got {health_check_interval}")

        self._max_connections = max_connections
        self._idle_timeout = idle_timeout
        self._health_check_interval = health_check_interval

        # 连接池: server_key → MCPToolset
        self._pool: OrderedDict[str, MCPToolset] = OrderedDict()
        # 正在使用的连接
        self._in_use: set[str] = set()
        # 最后活跃时间
        self._last_used: dict[str, float] = {}

        self._lock = asyncio.Lock()
        self._closed = False

        # 健康检查任务
        self._health_task: asyncio.Task | None = None

    @property
    def max_connections(self) -> int:
        """最大连接数"""
        return self._max_connections

    @property
    def idle_timeout(self) -> float:
        """空闲连接超时（秒）"""
        return self._idle_timeout

    @property
    def active_count(self) -> int:
        """当前活跃连接数（包括使用中和空闲的）"""
        return len(self._pool)

    @property
    def in_use_count(self) -> int:
        """当前正在使用的连接数"""
        return len(self._in_use)

    @staticmethod
    def _make_key(config: MCPServerConfig) -> str:
        """生成连接池键

        基于服务器配置的确定性键，确保相同配置复用同一连接。

        Args:
            config: MCP 服务器配置

        Returns:
            连接池键字符串
        """
        # 基于 transport + 唯一标识生成键
        if config.transport == "stdio":
            cmd = config.command or ""
            args_str = ",".join(config.args or [])
            return f"stdio:{cmd}:{args_str}"
        else:
            # sse / streamable_http
            url = config.url or ""
            return f"{config.transport}:{url}"

    async def get(
        self,
        config: MCPServerConfig,
        name: str = "",
        tool_filter: list[str] | None = None,
    ) -> MCPToolset:
        """获取 MCPToolset 连接

        如果池中已有相同服务器配置的空闲连接，复用该连接。
        否则创建新的 MCPToolset 并连接。

        Args:
            config: MCP 服务器配置
            name: 工具集名称（默认使用 config.name）
            tool_filter: 工具过滤列表

        Returns:
            已连接的 MCPToolset 实例

        Raises:
            RuntimeError: 连接池已关闭或超过最大连接数
            ConnectionError: MCP 连接失败
        """
        if self._closed:
            raise RuntimeError("MCPConnectionPool is closed")

        key = self._make_key(config)

        async with self._lock:
            # 检查是否有空闲的复用连接
            if key in self._pool and key not in self._in_use:
                toolset = self._pool[key]
                # 验证连接是否仍然活跃
                if toolset.is_connected:
                    self._in_use.add(key)
                    self._last_used[key] = time.monotonic()
                    # 移动到 LRU 尾部
                    self._pool.move_to_end(key)
                    logger.debug(
                        "MCPConnectionPool: reusing connection '%s'", key
                    )
                    return toolset
                else:
                    # 连接已断开，移除并重建
                    logger.debug(
                        "MCPConnectionPool: connection '%s' is stale, reconnecting",
                        key,
                    )
                    try:
                        await toolset.disconnect()
                    except Exception:
                        pass
                    del self._pool[key]
                    self._last_used.pop(key, None)

            # 检查最大连接数
            if len(self._pool) >= self._max_connections:
                # 尝试淘汰空闲连接
                evicted = await self._evict_idle()
                if not evicted:
                    raise RuntimeError(
                        f"MCPConnectionPool: max connections ({self._max_connections}) "
                        f"reached and no idle connections to evict"
                    )

            # 创建新连接
            toolset = MCPToolset(
                server_params=config,
                name=name,
                tool_filter=tool_filter,
            )

        # 在锁外连接（避免长时间持锁）
        await toolset.connect()

        async with self._lock:
            self._pool[key] = toolset
            self._in_use.add(key)
            self._last_used[key] = time.monotonic()

        logger.debug(
            "MCPConnectionPool: created new connection '%s'", key
        )

        # 启动健康检查（如果尚未启动）
        self._start_health_check()

        return toolset

    async def release(self, toolset: MCPToolset) -> None:
        """释放 MCPToolset 连接回连接池

        不断开连接，而是将其标记为空闲，可供后续复用。

        Args:
            toolset: 要释放的 MCPToolset 实例
        """
        key = self._make_key(toolset._server_params)  # noqa: SLF001

        async with self._lock:
            self._in_use.discard(key)
            self._last_used[key] = time.monotonic()

        logger.debug(
            "MCPConnectionPool: released connection '%s'", key
        )

    async def remove(self, toolset: MCPToolset) -> None:
        """移除并断开 MCPToolset 连接

        从连接池中移除连接并断开。

        Args:
            toolset: 要移除的 MCPToolset 实例
        """
        key = self._make_key(toolset._server_params)  # noqa: SLF001

        async with self._lock:
            self._pool.pop(key, None)
            self._in_use.discard(key)
            self._last_used.pop(key, None)

        try:
            await toolset.disconnect()
        except Exception as e:
            logger.warning(
                "MCPConnectionPool: error disconnecting '%s': %s", key, e
            )

        logger.debug(
            "MCPConnectionPool: removed connection '%s'", key
        )

    async def close_all(self) -> None:
        """关闭所有连接并停止连接池"""
        self._closed = True

        # 停止健康检查
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

        async with self._lock:
            keys = list(self._pool.keys())
            for key in keys:
                toolset = self._pool.pop(key, None)
                if toolset is not None:
                    try:
                        await toolset.disconnect()
                    except Exception as e:
                        logger.warning(
                            "MCPConnectionPool: error disconnecting '%s': %s",
                            key, e,
                        )
            self._in_use.clear()
            self._last_used.clear()

        logger.info(
            "MCPConnectionPool: closed %d connections", len(keys)
        )

    async def _evict_idle(self) -> bool:
        """淘汰空闲连接

        从 LRU 头部开始淘汰不在使用中的连接。

        Returns:
            True 如果成功淘汰了至少一个连接
        """
        evicted = False
        # 从 LRU 头部（最旧）开始检查
        for key in list(self._pool.keys()):
            if key in self._in_use:
                continue
            toolset = self._pool.pop(key, None)
            self._last_used.pop(key, None)
            if toolset is not None:
                try:
                    await toolset.disconnect()
                except Exception:
                    pass
            evicted = True
            logger.debug(
                "MCPConnectionPool: evicted idle connection '%s'", key
            )
            break  # 只淘汰一个，给新连接腾位

        return evicted

    async def _cleanup_expired(self) -> int:
        """清理过期的空闲连接

        断开超过 idle_timeout 未使用的空闲连接。

        Returns:
            清理的连接数
        """
        now = time.monotonic()
        expired_keys: list[str] = []

        async with self._lock:
            for key, last_used in list(self._last_used.items()):
                if key in self._in_use:
                    continue
                if self._idle_timeout > 0 and (now - last_used) > self._idle_timeout:
                    expired_keys.append(key)

            for key in expired_keys:
                toolset = self._pool.pop(key, None)
                self._last_used.pop(key, None)
                if toolset is not None:
                    try:
                        await toolset.disconnect()
                    except Exception:
                        pass

        if expired_keys:
            logger.debug(
                "MCPConnectionPool: cleaned up %d expired connections",
                len(expired_keys),
            )

        return len(expired_keys)

    async def _health_check_loop(self) -> None:
        """健康检查循环

        定期检查连接池中的连接是否仍然活跃，
        清理不活跃的连接和过期的空闲连接。
        """
        try:
            while not self._closed:
                await asyncio.sleep(self._health_check_interval)
                if self._closed:
                    break

                # 清理过期连接
                await self._cleanup_expired()

                # 检查活跃连接
                stale_keys: list[str] = []
                async with self._lock:
                    for key, toolset in list(self._pool.items()):
                        if not toolset.is_connected:
                            stale_keys.append(key)

                for key in stale_keys:
                    async with self._lock:
                        toolset = self._pool.pop(key, None)
                        self._in_use.discard(key)
                        self._last_used.pop(key, None)
                    if toolset is not None:
                        try:
                            await toolset.disconnect()
                        except Exception:
                            pass

                if stale_keys:
                    logger.debug(
                        "MCPConnectionPool: removed %d stale connections",
                        len(stale_keys),
                    )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("MCPConnectionPool: health check loop error: %s", e)

    def _start_health_check(self) -> None:
        """启动健康检查任务（如果未启动且间隔 > 0）"""
        if (
            self._health_check_interval > 0
            and self._health_task is None
            and not self._closed
        ):
            try:
                self._health_task = asyncio.get_running_loop().create_task(
                    self._health_check_loop()
                )
            except RuntimeError:
                pass

    def get_stats(self) -> dict[str, Any]:
        """获取连接池统计信息

        Returns:
            包含连接池状态和统计的字典
        """
        return {
            "max_connections": self._max_connections,
            "active_count": len(self._pool),
            "in_use_count": len(self._in_use),
            "idle_count": len(self._pool) - len(self._in_use),
            "idle_timeout": self._idle_timeout,
            "health_check_interval": self._health_check_interval,
            "closed": self._closed,
        }

    async def __aenter__(self) -> MCPConnectionPool:
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器出口 — 关闭所有连接"""
        await self.close_all()

    def __repr__(self) -> str:
        return (
            f"MCPConnectionPool("
            f"active={len(self._pool)}, "
            f"in_use={len(self._in_use)}, "
            f"max={self._max_connections})"
        )
