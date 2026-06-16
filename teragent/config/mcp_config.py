# teragent/config/mcp_config.py
"""MCP 服务器配置 — 定义 MCP 连接参数

支持三种传输模式:
  - stdio: 启动本地 MCP 服务器进程（通过 stdin/stdout 通信）
  - sse: 连接远程 MCP 服务器（通过 SSE + HTTP POST 通信）
  - streamable_http: 连接远程 MCP 服务器（通过 Streamable HTTP 通信）

用法:
    # stdio 模式
    cfg = MCPServerConfig(
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        name="filesystem",
    )

    # SSE 模式
    cfg = MCPServerConfig(
        transport="sse",
        url="http://localhost:8080/sse",
        headers={"Authorization": "Bearer xxx"},
        name="remote-search",
    )

    # 从配置字典创建
    cfg = MCPServerConfig.from_dict("filesystem", {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    })
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

__all__ = [
    "MCPServerConfig",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPServerConfig:
    """MCP 服务器配置

    描述如何连接到 MCP 服务器，包括传输模式、连接参数和工具过滤选项。

    Attributes:
        command: stdio 模式下启动服务器的可执行命令
        args: stdio 模式下的命令行参数
        env: stdio 模式下额外的环境变量（合并到默认环境变量上）
        cwd: stdio 模式下的工作目录

        url: HTTP/SSE 模式下的服务器端点 URL
        headers: HTTP/SSE 模式下的请求头
        timeout: HTTP/SSE 模式下的连接超时（秒）

        transport: 传输模式 — "stdio" | "sse" | "streamable_http"
        name: 服务器名称（用于日志和标识）
        tool_filter: 工具过滤列表 — 仅暴露指定名称的工具（空列表 = 全部暴露）
    """

    # stdio 模式
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""

    # HTTP/SSE 模式
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0

    # 公共
    transport: str = "stdio"  # "stdio" | "sse" | "streamable_http"
    name: str = ""
    tool_filter: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """校验配置一致性"""
        if self.transport == "stdio":
            if not self.command:
                logger.warning(
                    f"MCP 服务器配置 '{self.name or '<unnamed>'}': "
                    f"stdio 模式需要指定 command"
                )
        elif self.transport in ("sse", "streamable_http"):
            if not self.url:
                logger.warning(
                    f"MCP 服务器配置 '{self.name or '<unnamed>'}': "
                    f"{self.transport} 模式需要指定 url"
                )
        else:
            logger.warning(
                f"MCP 服务器配置 '{self.name or '<unnamed>'}': "
                f"未知的传输模式 '{self.transport}'，支持: stdio, sse, streamable_http"
            )

    @classmethod
    def from_dict(cls, name: str, data: dict) -> MCPServerConfig:
        """从字典创建配置

        Args:
            name: 服务器名称
            data: 配置字典，支持以下键:
                - transport: 传输模式
                - command: stdio 命令
                - args: 命令行参数
                - env: 环境变量
                - cwd: 工作目录
                - url: 服务器 URL
                - headers: 请求头
                - timeout: 连接超时
                - tool_filter: 工具过滤列表

        Returns:
            MCPServerConfig 实例
        """
        return cls(
            name=name,
            transport=data.get("transport", "stdio"),
            command=data.get("command", ""),
            args=data.get("args", []),
            env=data.get("env", {}),
            cwd=data.get("cwd", ""),
            url=data.get("url", ""),
            headers=data.get("headers", {}),
            timeout=data.get("timeout", 30.0),
            tool_filter=data.get("tool_filter", []),
        )

    def to_dict(self) -> dict:
        """转换为可序列化字典

        Returns:
            包含所有配置项的字典
        """
        result: dict = {
            "transport": self.transport,
            "name": self.name,
            "timeout": self.timeout,
            "tool_filter": self.tool_filter,
        }
        if self.transport == "stdio":
            result["command"] = self.command
            result["args"] = self.args
            result["env"] = self.env
            result["cwd"] = self.cwd
        elif self.transport in ("sse", "streamable_http"):
            result["url"] = self.url
            result["headers"] = self.headers
        return result

    def __repr__(self) -> str:
        if self.transport == "stdio":
            cmd_preview = f"{self.command} {' '.join(self.args)}".strip()
            return (
                f"MCPServerConfig(name={self.name!r}, "
                f"transport={self.transport!r}, "
                f"command={cmd_preview!r})"
            )
        return (
            f"MCPServerConfig(name={self.name!r}, "
            f"transport={self.transport!r}, "
            f"url={self.url!r})"
        )
