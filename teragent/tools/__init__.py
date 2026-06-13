# teragent/tools/__init__.py
"""工具注册子系统 — 统一工具接口 + 注册表

核心组件:
  - BaseTool: 工具基类（安全属性 + 生命周期 + 进度回调 + 注册元数据）
  - ToolResult: 工具执行结果统一返回格式
  - ToolSafety: 工具安全级别枚举（来自 teragent.core.types）
  - ToolRegistry: 工具注册表
  - ToolOrchestrator: 工具并行编排器
"""

from teragent.core.types import ToolSafety
from teragent.tools.base import BaseTool, ToolResult
from teragent.tools.desktop import DesktopSafetyConfig, DesktopTool
from teragent.tools.orchestrator import MAX_CONCURRENT_TOOLS, ToolOrchestrator
from teragent.tools.registry import ToolRegistry

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolSafety",
    "ToolRegistry",
    "ToolOrchestrator",
    "MAX_CONCURRENT_TOOLS",
    "DesktopTool",
    "DesktopSafetyConfig",
]
