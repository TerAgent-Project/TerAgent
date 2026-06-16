# teragent/tools/__init__.py
"""工具注册子系统 — 统一工具接口 + 注册表

核心组件:
  - BaseTool: 工具基类（安全属性 + 生命周期 + 进度回调 + 注册元数据）
  - ToolResult: 工具执行结果统一返回格式
  - ToolSafety: 工具安全级别枚举（来自 teragent.core.types）
  - ToolRegistry: 工具注册表
  - ToolOrchestrator: 工具并行编排器
  - DecoratorTool / @tool: 装饰器方式创建工具
  - AgentTool: Agent-as-Tool 封装
  - ToolPack: 工具包（共享状态和资源的工具集合）
  - ToolInfo: 工具扩展信息（分类、来源、标签）
  - MCPToolset: MCP 工具集（连接 MCP 服务器，发现并代理工具调用）
  - MCPTool: MCP 远程工具的本地代理
  - OpenAPIToolset / OpenAPIOperationTool: OpenAPI 规范自动生成工具
  - builtin: 内置工具集（文件、代码、Web、分析）
"""

from teragent.core.types import ToolSafety
from teragent.tools.base import BaseTool, ToolResult
from teragent.tools.desktop import DesktopSafetyConfig, DesktopTool
from teragent.tools.orchestrator import MAX_CONCURRENT_TOOLS, ToolOrchestrator
from teragent.tools.registry import ToolInfo, ToolRegistry

# Phase 1 W1: @tool decorator and Agent-as-Tool
from teragent.tools.decorator import DecoratorTool, tool
from teragent.tools.agent_tool import AgentTool

# Phase 1 W4: Built-in tools
from teragent.tools.builtin import all_builtin_tools

# Phase 2 W8: ToolPack
from teragent.tools.toolpack import ToolPack

# Phase 2 W5: MCP Toolset
from teragent.tools.mcp_toolset import MCPToolset, MCPTool, MCPConnectionPool

# Phase 2 W7: OpenAPI Toolset
from teragent.tools.openapi_toolset import OpenAPIOperationTool, OpenAPIToolset

# Phase 3 W11: Orchestrator-as-Tool (nested orchestration)
from teragent.tools.orchestrator_tool import OrchestratorTool

# Phase 3 W11: Auth
from teragent.tools.auth import AuthScheme, AuthCredential, AuthManager

# Phase 3 W12: Result Cache
from teragent.tools.result_cache import ResultCache, CacheEntry, CacheStats

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolSafety",
    "ToolRegistry",
    "ToolOrchestrator",
    "MAX_CONCURRENT_TOOLS",
    "DesktopTool",
    "DesktopSafetyConfig",
    # Phase 1 W1
    "DecoratorTool",
    "tool",
    "AgentTool",
    # Phase 1 W4: Built-in tools
    "all_builtin_tools",
    # Phase 2 W8: ToolPack + ToolInfo
    "ToolPack",
    "ToolInfo",
    # Phase 2 W5: MCP Toolset
    "MCPToolset",
    "MCPTool",
    "MCPConnectionPool",
    # Phase 2 W7: OpenAPI Toolset
    "OpenAPIOperationTool",
    "OpenAPIToolset",
    # Phase 3 W11: Nested orchestration
    "OrchestratorTool",
    # Phase 3 W11: Auth
    "AuthScheme",
    "AuthCredential",
    "AuthManager",
    # Phase 3 W12: Result Cache
    "ResultCache",
    "CacheEntry",
    "CacheStats",
]
