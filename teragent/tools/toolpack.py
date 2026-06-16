# teragent/tools/toolpack.py
"""工具包 — 共享状态和资源的工具集合

参考 OpenAI Agents SDK 的 ShellToolSet/FilesystemToolSet 模式。
将一组相关工具打包，共享生命周期和资源。

使用方式:
    db_pack = ToolPack(
        tools=[db_query_tool, db_insert_tool, db_update_tool],
        name="database",
        shared_state={"connection_pool": None},
    )
    await db_pack.start()  # 初始化共享资源
    db_pack.register_to(tool_registry)  # 注册到工具注册表
    await db_pack.stop()  # 清理共享资源

    # 或使用 async context manager
    async with ToolPack(tools=[...], name="fs", on_start=_init, on_stop=_cleanup) as pack:
        pack.register_to(registry)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Awaitable, Callable

__all__ = [
    "ToolPack",
]

from teragent.tools.base import BaseTool

if TYPE_CHECKING:
    from teragent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolPack:
    """工具包 — 共享状态和资源的工具集合

    参考 OpenAI Agents SDK 的 ShellToolSet/FilesystemToolSet 模式。
    将一组相关工具打包，共享生命周期和资源。

    ToolPack 支持:
      - 共享状态（shared_state）: 工具间共享的运行时数据
      - 生命周期钩子（on_start / on_stop）: 初始化和清理共享资源
      - 批量注册: 一键注册到 ToolRegistry
      - Async context manager: 优雅的生命周期管理

    Attributes:
        tools: 工具实例列表
        name: 工具包名称（用于日志和 ToolInfo.source）
        shared_state: 工具间共享的运行时状态字典
        on_start: 启动时调用的异步回调（用于初始化共享资源）
        on_stop: 停止时调用的异步回调（用于清理共享资源）
    """

    def __init__(
        self,
        tools: list[BaseTool],
        name: str = "",
        shared_state: dict | None = None,
        on_start: Callable[..., Awaitable[None]] | None = None,
        on_stop: Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        self.tools = tools
        self.name = name or self._infer_name()
        self.shared_state: dict = shared_state if shared_state is not None else {}
        self._on_start = on_start
        self._on_stop = on_stop
        self._started = False

    def _infer_name(self) -> str:
        """从工具列表推断工具包名称

        如果未指定 name，则使用第一个工具的类名作为包名。

        Returns:
            推断的名称字符串
        """
        if self.tools:
            return self.tools[0].__class__.__module__.split(".")[-1]
        return "unnamed_pack"

    async def start(self) -> None:
        """初始化工具包的共享资源

        调用 on_start 回调（如果提供）。
        仅在未启动时执行，重复调用是安全的（幂等）。
        """
        if self._started:
            logger.debug(f"ToolPack '{self.name}' already started, skipping.")
            return

        if self._on_start is not None:
            try:
                await self._on_start(self.shared_state)
                logger.debug(f"ToolPack '{self.name}' started (on_start callback executed).")
            except Exception:
                logger.error(f"ToolPack '{self.name}' on_start callback failed.", exc_info=True)
                raise
        else:
            logger.debug(f"ToolPack '{self.name}' started (no on_start callback).")

        self._started = True

    async def stop(self) -> None:
        """清理工具包的共享资源

        调用 on_stop 回调（如果提供）。
        仅在已启动时执行，重复调用是安全的（幂等）。
        """
        if not self._started:
            logger.debug(f"ToolPack '{self.name}' not started, skipping stop.")
            return

        if self._on_stop is not None:
            try:
                await self._on_stop(self.shared_state)
                logger.debug(f"ToolPack '{self.name}' stopped (on_stop callback executed).")
            except Exception:
                logger.error(f"ToolPack '{self.name}' on_stop callback failed.", exc_info=True)
                raise
        else:
            logger.debug(f"ToolPack '{self.name}' stopped (no on_stop callback).")

        self._started = False

    def list_tools(self) -> list[BaseTool]:
        """返回工具包中的所有工具

        Returns:
            BaseTool 实例列表
        """
        return list(self.tools)

    def register_to(self, registry: ToolRegistry) -> int:
        """将工具包中的所有工具注册到工具注册表

        注册时自动为每个工具设置 ToolInfo 的 source 为 "toolpack:{name}"。

        Args:
            registry: ToolRegistry 实例

        Returns:
            成功注册的工具数量
        """
        count = 0
        source = f"toolpack:{self.name}"

        for tool in self.tools:
            if tool.name:
                registry.register(tool)
                # 注册后更新扩展信息中的 source
                if hasattr(registry, "_tool_info") and tool.name in registry._tool_info:
                    registry._tool_info[tool.name].source = source
                elif hasattr(registry, "_tool_info"):
                    from teragent.tools.registry import ToolInfo
                    registry._tool_info[tool.name] = ToolInfo(
                        name=tool.name,
                        description=tool.description,
                        safety=tool.safety_level,
                        source=source,
                    )
                count += 1
            else:
                logger.warning(
                    f"ToolPack '{self.name}': skipping tool with empty name "
                    f"({tool.__class__.__name__})."
                )

        logger.debug(f"ToolPack '{self.name}' registered {count} tools to registry.")
        return count

    def get_tool(self, name: str) -> BaseTool | None:
        """按名称获取工具包中的工具

        Args:
            name: 工具名称

        Returns:
            BaseTool 实例或 None
        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    # ===== Async context manager support =====

    async def __aenter__(self) -> ToolPack:
        """进入异步上下文: 自动启动工具包"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """退出异步上下文: 自动停止工具包"""
        await self.stop()

    def __repr__(self) -> str:
        tool_names = [t.name for t in self.tools if t.name]
        return (
            f"<ToolPack name={self.name!r} "
            f"tools={tool_names} "
            f"started={self._started}>"
        )
