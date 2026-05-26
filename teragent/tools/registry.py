# teragent/tools/registry.py
"""工具注册表 — 管理所有可用工具的生命周期

核心职责:
  - 注册 / 注销工具
  - 按 name 查找工具
  - 列举所有工具（function calling 格式）
  - 批量注册（按意图分类）
  - 工具存在性检查

设计原则:
  - 注册表是"平面"的，不关心工具的执行权限
  - 权限控制由 AgentLoop / Pipeline 负责（按意图过滤可用工具）
  - 注册表不关心工具的内部状态，只管理"注册"这个维度
"""
import logging
from teragent.tools.base import BaseTool
from teragent.core.types import ToolSafety

logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册表 — 管理所有可用工具的生命周期和安全元数据"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        # 安全元数据缓存
        self._safety_metadata: dict[str, dict] = {}

    def register(self, tool: BaseTool) -> None:
        """注册一个工具实例

        如果同名工具已存在，发出警告并覆盖。
        注册时自动提取安全元数据。

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

    # ===== 通用方法 =====

    def get_summary(self) -> dict:
        """返回注册表摘要信息"""
        return {
            "total_tools": len(self._tools),
            "tool_names": list(self._tools.keys()),
            "safety_report": self.get_safety_report(),
        }

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={list(self._tools.keys())}>"
