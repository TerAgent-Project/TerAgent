# teragent/tools/registry.py
"""工具注册表 — 管理所有可用工具的生命周期

核心职责:
  - 注册 / 注销工具
  - 按 name 查找工具
  - 列举所有工具（function calling 格式）
  - 批量注册（按意图分类）
  - 工具存在性检查

Phase 2 增强:
  - ToolInfo 扩展元数据（category / source / tags）
  - 按分类注册和查询工具
  - 按意图推荐工具（基于分类元数据匹配）
  - ToolPack / MCP 工具集注册

设计原则:
  - 注册表是"平面"的，不关心工具的执行权限
  - 权限控制由 AgentLoop / Pipeline 负责（按意图过滤可用工具）
  - 注册表不关心工具的内部状态，只管理"注册"这个维度
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

__all__ = [
    "ToolInfo",
    "ToolRegistry",
]

from teragent.core.types import ToolSafety
from teragent.tools.base import BaseTool

if TYPE_CHECKING:
    from teragent.tools.toolpack import ToolPack

logger = logging.getLogger(__name__)


# ===== ToolInfo 数据类 =====

@dataclass
class ToolInfo:
    """工具扩展信息

    存储工具的分类、来源、标签等扩展元数据，
    供 ToolRegistry 按分类/意图查询使用。

    Attributes:
        name: 工具名称
        category: 工具分类（如 "filesystem", "database", "web"）
        description: 工具描述
        safety: 工具安全级别
        source: 工具来源（"builtin" | "mcp:{server_name}" | "openapi:{spec}" |
                "toolpack:{name}" | "custom"）
        tags: 工具标签列表（用于意图匹配和过滤）
    """

    name: str
    category: str = ""
    description: str = ""
    safety: ToolSafety = ToolSafety.SAFE_WRITE
    source: str = ""  # "builtin" | "mcp:{server_name}" | "openapi:{spec}" | "toolpack:{name}" | "custom"
    tags: list[str] = field(default_factory=list)


# ===== ToolRegistry =====

class ToolRegistry:
    """工具注册表 — 管理所有可用工具的生命周期和安全元数据"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        # 安全元数据缓存
        self._safety_metadata: dict[str, dict] = {}
        # Phase 2: 扩展元数据
        self._tool_info: dict[str, ToolInfo] = {}
        # Phase 2: 分类索引 — category → tool names
        self._categories: dict[str, list[str]] = {}

    def register(self, tool: BaseTool) -> None:
        """注册一个工具实例

        如果同名工具已存在，发出警告并覆盖。
        注册时自动提取安全元数据和扩展信息。

        Args:
            tool: BaseTool 子类实例
        """
        if not tool.name:
            logger.error(f"Cannot register tool with empty name: {tool.__class__.__name__}")
            return

        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting.")
        self._tools[tool.name] = tool
        # 提取并缓存安全元数据
        self._safety_metadata[tool.name] = tool.to_registry_metadata()
        # 初始化扩展信息（如果尚未存在）
        if tool.name not in self._tool_info:
            self._tool_info[tool.name] = ToolInfo(
                name=tool.name,
                description=tool.description,
                safety=tool.safety_level,
                source="custom",
            )
        else:
            # 已有 ToolInfo（可能由 register_category 等方法预设），
            # 仅更新 description 和 safety
            existing = self._tool_info[tool.name]
            if not existing.description:
                existing.description = tool.description
            existing.safety = tool.safety_level
        logger.debug(f"Tool registered: {tool.name} (safety={tool.safety_level.value})")

    def unregister(self, name: str) -> bool:
        """注销一个工具

        Args:
            name: 工具名称

        Returns:
            True 表示成功注销，False 表示工具不存在
        """
        if name in self._tools:
            del self._tools[name]
            self._safety_metadata.pop(name, None)
            # 清理分类索引
            tool_info = self._tool_info.pop(name, None)
            if tool_info and tool_info.category:
                cat = tool_info.category
                if cat in self._categories:
                    self._categories[cat] = [
                        n for n in self._categories[cat] if n != name
                    ]
                    if not self._categories[cat]:
                        del self._categories[cat]
            logger.debug(f"Tool unregistered: {name}")
            return True
        return False

    def get(self, name: str) -> BaseTool | None:
        """按名称查找工具

        Args:
            name: 工具名称

        Returns:
            BaseTool 实例或 None
        """
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """检查工具是否已注册"""
        return name in self._tools

    def list_tools(self) -> list[dict]:
        """返回所有工具的 function calling 定义

        Returns:
            符合 OpenAI tools 格式的字典列表
        """
        return [tool.to_function_definition() for tool in self._tools.values()]

    def list_tool_names(self) -> list[str]:
        """返回所有已注册工具的名称列表"""
        return list(self._tools.keys())

    def get_tools_by_names(self, names: list[str]) -> list[BaseTool]:
        """按名称列表批量获取工具

        Args:
            names: 工具名称列表

        Returns:
            找到的 BaseTool 实例列表（跳过不存在的）
        """
        result: list[BaseTool] = []
        for name in names:
            tool = self._tools.get(name)
            if tool:
                result.append(tool)
            else:
                logger.warning(f"Tool '{name}' not found in registry, skipping.")
        return result

    def batch_register(self, tools: list[BaseTool]) -> int:
        """批量注册工具

        Args:
            tools: BaseTool 实例列表

        Returns:
            成功注册的工具数量
        """
        count = 0
        for tool in tools:
            if tool.name:
                self.register(tool)
                count += 1
        return count

    # ===== 安全元数据查询 =====

    def get_safety_metadata(self, name: str) -> dict | None:
        """获取工具的安全元数据

        Args:
            name: 工具名称

        Returns:
            安全元数据字典，包含 safety / concurrency_safe / read_only / destructive，
            或 None 表示工具不存在
        """
        return self._safety_metadata.get(name)

    def get_read_only_tools(self) -> list[str]:
        """返回所有只读工具的名称列表

        只读工具可安全并行执行，供 ToolOrchestrator 使用。
        """
        return [
            name for name, meta in self._safety_metadata.items()
            if meta.get("read_only", False)
        ]

    def get_concurrency_safe_tools(self) -> list[str]:
        """返回所有并发安全工具的名称列表"""
        return [
            name for name, meta in self._safety_metadata.items()
            if meta.get("concurrency_safe", False)
        ]

    def get_tools_by_safety(self, safety: ToolSafety) -> list[str]:
        """按安全级别查询工具

        Args:
            safety: ToolSafety 枚举值

        Returns:
            匹配的工具名称列表
        """
        safety_value = safety.value
        return [
            name for name, meta in self._safety_metadata.items()
            if meta.get("safety") == safety_value
        ]

    def get_destructive_tools(self) -> list[str]:
        """返回所有破坏性工具的名称列表"""
        return [
            name for name, meta in self._safety_metadata.items()
            if meta.get("destructive", False)
        ]

    def invalidate_metadata(self, tool_name: str) -> None:
        """Force-refresh the safety metadata cache for a specific tool.

        Call this when a tool's safety attributes may have changed
        after registration (e.g., dynamic permission updates).

        Args:
            tool_name: Name of the tool to refresh
        """
        tool = self._tools.get(tool_name)
        if tool:
            self._safety_metadata[tool_name] = tool.to_registry_metadata()
        else:
            logger.warning(f"Cannot invalidate metadata: tool '{tool_name}' not found")

    def get_safety_report(self) -> dict:
        """返回工具安全报告（供 TUI /status 和调试使用）

        Returns:
            {
                "total": 10,
                "by_safety": {"read_only": 5, "safe_write": 2, ...},
                "read_only_tools": [...],
                "destructive_tools": [...],
                "concurrency_safe_tools": [...],
            }
        """
        by_safety: dict[str, int] = {}
        for meta in self._safety_metadata.values():
            safety = meta.get("safety", "safe_write")
            by_safety[safety] = by_safety.get(safety, 0) + 1

        return {
            "total": len(self._tools),
            "by_safety": by_safety,
            "read_only_tools": self.get_read_only_tools(),
            "destructive_tools": self.get_destructive_tools(),
            "concurrency_safe_tools": self.get_concurrency_safe_tools(),
        }

    # ===== Phase 2: 分类与意图查询 =====

    def register_category(self, category: str, tools: list[BaseTool]) -> None:
        """按分类注册工具

        将工具注册到注册表，并标记它们的分类信息。
        如果工具已注册，则更新其分类；如果未注册，则先注册再标记分类。

        Args:
            category: 分类名称（如 "filesystem", "database", "web"）
            tools: 该分类下的工具列表
        """
        if category not in self._categories:
            self._categories[category] = []

        for tool in tools:
            if not tool.name:
                logger.warning(
                    f"Skipping tool with empty name in category '{category}': "
                    f"{tool.__class__.__name__}"
                )
                continue

            # 注册工具（如果尚未注册）
            if tool.name not in self._tools:
                self.register(tool)

            # 更新分类索引
            if tool.name not in self._categories[category]:
                self._categories[category].append(tool.name)

            # 更新扩展信息
            if tool.name in self._tool_info:
                self._tool_info[tool.name].category = category
            else:
                self._tool_info[tool.name] = ToolInfo(
                    name=tool.name,
                    category=category,
                    description=tool.description,
                    safety=tool.safety_level,
                    source="custom",
                )

        logger.debug(
            f"Category '{category}' registered with {len(tools)} tools."
        )

    def get_tools_by_category(self, category: str) -> list[BaseTool]:
        """按分类查询工具

        Args:
            category: 分类名称

        Returns:
            该分类下的 BaseTool 实例列表
        """
        tool_names = self._categories.get(category, [])
        result: list[BaseTool] = []
        for name in tool_names:
            tool = self._tools.get(name)
            if tool:
                result.append(tool)
            else:
                logger.warning(
                    f"Tool '{name}' in category '{category}' not found in registry."
                )
        return result

    def get_tools_for_intent(self, intent: str) -> list[BaseTool]:
        """按意图推荐工具（基于分类元数据匹配）

        将意图字符串与工具的分类、标签、描述进行模糊匹配，
        返回最相关的工具列表。

        匹配策略:
          1. 精确匹配分类名
          2. 分类名包含意图关键词
          3. 标签包含意图关键词
          4. 描述包含意图关键词

        Args:
            intent: 意图字符串（如 "file", "database", "web_search"）

        Returns:
            匹配的 BaseTool 实例列表
        """
        intent_lower = intent.lower()
        matched_names: list[str] = []

        # 1. 精确匹配分类名
        if intent_lower in self._categories:
            matched_names.extend(self._categories[intent_lower])

        # 2. 分类名包含意图关键词
        for cat, names in self._categories.items():
            if cat not in [intent_lower] and intent_lower in cat.lower():
                for name in names:
                    if name not in matched_names:
                        matched_names.append(name)

        # 3 & 4. 标签和描述匹配
        for name, info in self._tool_info.items():
            if name in matched_names:
                continue
            # 标签匹配
            if any(intent_lower in tag.lower() for tag in info.tags):
                matched_names.append(name)
                continue
            # 描述匹配
            if intent_lower in info.description.lower():
                matched_names.append(name)

        # 转换为 BaseTool 实例
        result: list[BaseTool] = []
        for name in matched_names:
            tool = self._tools.get(name)
            if tool:
                result.append(tool)

        return result

    # ===== Phase 2: ToolPack / MCP 注册 =====

    def register_toolpack(self, toolpack: ToolPack) -> int:
        """注册 ToolPack 中的所有工具

        将 ToolPack 中的工具批量注册到注册表，
        并为每个工具设置 source 为 "toolpack:{name}"。

        Args:
            toolpack: ToolPack 实例

        Returns:
            成功注册的工具数量
        """
        source = f"toolpack:{toolpack.name}"
        count = 0
        for tool in toolpack.tools:
            if tool.name:
                self.register(tool)
                # 更新扩展信息中的 source
                if tool.name in self._tool_info:
                    self._tool_info[tool.name].source = source
                else:
                    self._tool_info[tool.name] = ToolInfo(
                        name=tool.name,
                        description=tool.description,
                        safety=tool.safety_level,
                        source=source,
                    )
                count += 1
            else:
                logger.warning(
                    f"ToolPack '{toolpack.name}': skipping tool with empty name "
                    f"({tool.__class__.__name__})."
                )

        logger.debug(
            f"ToolPack '{toolpack.name}' registered {count} tools to registry."
        )
        return count

    def register_mcp_toolset(self, toolset) -> int:
        """注册 MCP 工具集中的所有工具

        使用延迟导入避免对 mcp 包的硬依赖。
        MCP 工具集中的工具 source 标记为 "mcp:{server_name}"。

        Args:
            toolset: MCPToolset 实例（需提供 name 属性和 tools 方法/属性）

        Returns:
            成功注册的工具数量
        """
        # 延迟导入 — 避免硬依赖 mcp 包
        try:
            server_name = getattr(toolset, "name", "unknown")
            source = f"mcp:{server_name}"

            # MCPToolset 可能有 tools 属性或 get_tools() 方法
            if hasattr(toolset, "tools"):
                mcp_tools = toolset.tools
                if callable(mcp_tools):
                    mcp_tools = mcp_tools()
            elif hasattr(toolset, "get_tools"):
                mcp_tools = toolset.get_tools()
            else:
                logger.warning(
                    f"MCPToolset '{server_name}' has no tools attribute or get_tools method."
                )
                return 0

            if not isinstance(mcp_tools, list):
                logger.warning(
                    f"MCPToolset '{server_name}' tools is not a list, skipping."
                )
                return 0

            count = 0
            for tool in mcp_tools:
                if isinstance(tool, BaseTool) and tool.name:
                    self.register(tool)
                    # 更新扩展信息
                    if tool.name in self._tool_info:
                        self._tool_info[tool.name].source = source
                    else:
                        self._tool_info[tool.name] = ToolInfo(
                            name=tool.name,
                            description=tool.description,
                            safety=tool.safety_level,
                            source=source,
                        )
                    count += 1

            logger.debug(
                f"MCPToolset '{server_name}' registered {count} tools to registry."
            )
            return count

        except Exception:
            logger.error(
                f"Failed to register MCP toolset: {getattr(toolset, 'name', 'unknown')}",
                exc_info=True,
            )
            return 0

    # ===== Phase 2: 扩展元数据查询 =====

    def get_tool_info(self, name: str) -> ToolInfo | None:
        """获取工具的扩展信息

        Args:
            name: 工具名称

        Returns:
            ToolInfo 实例或 None
        """
        return self._tool_info.get(name)

    def list_categories(self) -> list[str]:
        """返回所有已注册的分类名称列表"""
        return list(self._categories.keys())

    # ===== 通用方法 =====

    def get_summary(self) -> dict:
        """返回注册表摘要信息"""
        return {
            "total_tools": len(self._tools),
            "tool_names": list(self._tools.keys()),
            "safety_report": self.get_safety_report(),
            "categories": list(self._categories.keys()),
        }

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={list(self._tools.keys())}>"
