# teragent/tools/hub/client.py
"""工具市场客户端 — 搜索、安装、发布远程工具

提供与 TerAgent Tool Hub 交互的异步客户端，支持:
  - 搜索远程工具市场中的工具
  - 安装远程工具到本地注册表（自动实例化为 HubTool）
  - 发布本地工具到远程市场
  - 查询已安装的市场工具列表

使用方式:
    # 异步上下文管理器（推荐，自动管理 httpx 连接池）
    async with ToolHubClient() as client:
        entries = await client.search("database")
        tool = await client.install("postgres_query")
        await client.publish(my_tool, {"category": "database"})

    # 手动管理
    client = ToolHubClient(auth_token="sk-xxx")
    try:
        await client.connect()
        entries = await client.search("file")
    finally:
        await client.close()

设计原则:
  - 客户端只负责"市场交互"，不负责"工具执行"（执行由 HubTool 代理到 Hub API）
  - 网络错误统一封装为 ToolHubError，不暴露 httpx 异常细节
  - 已安装工具缓存在本地 dict 中，支持可选持久化
  - 连接池复用，避免频繁建立/断开 HTTP 连接
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

from teragent.core.types import ToolSafety
from teragent.tools.base import BaseTool, ToolResult

__all__ = [
    "ToolHubClient",
    "ToolHubEntry",
    "ToolHubError",
    "HubTool",
]

logger = logging.getLogger(__name__)

# ======================================================================
# Exception
# ======================================================================


class ToolHubError(Exception):
    """工具市场异常

    所有与 Tool Hub 交互过程中产生的错误统一封装为此异常，
    包括网络错误、API 错误响应、工具安装失败等。

    Attributes:
        status_code: HTTP 状态码（如果适用），-1 表示非 HTTP 错误
        detail: 错误详情
    """

    def __init__(
        self,
        message: str,
        status_code: int = -1,
        detail: str = "",
    ) -> None:
        self.status_code = status_code
        self.detail = detail or message
        super().__init__(message)

    def __repr__(self) -> str:
        return (
            f"ToolHubError(message={self.args[0]!r}, "
            f"status_code={self.status_code})"
        )


# ======================================================================
# ToolHubEntry — 工具市场条目
# ======================================================================


@dataclass
class ToolHubEntry:
    """工具市场搜索结果条目

    表示工具市场中的一个工具条目，包含工具的元数据信息。
    搜索接口返回此数据类的列表。

    Attributes:
        name: 工具唯一标识符（如 "postgres_query"）
        version: 工具版本号（如 "1.2.0"）
        author: 工具作者
        description: 工具描述
        category: 工具分类（如 "database", "filesystem", "web"）
        downloads: 累计下载次数
        rating: 用户评分（0.0 ~ 5.0）
    """

    name: str
    version: str
    author: str
    description: str
    category: str
    downloads: int
    rating: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolHubEntry:
        """从 API 响应字典创建 ToolHubEntry

        Args:
            data: API 返回的工具条目字典

        Returns:
            ToolHubEntry 实例
        """
        return cls(
            name=data.get("name", ""),
            version=data.get("version", "0.0.0"),
            author=data.get("author", ""),
            description=data.get("description", ""),
            category=data.get("category", ""),
            downloads=data.get("downloads", 0),
            rating=float(data.get("rating", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化字典

        Returns:
            包含所有字段的字典
        """
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "category": self.category,
            "downloads": self.downloads,
            "rating": self.rating,
        }


# ======================================================================
# HubTool — 从市场安装的工具
# ======================================================================


class HubTool(BaseTool):
    """从工具市场安装的远程工具代理

    HubTool 是 BaseTool 的子类，代表从 Tool Hub 安装的工具。
    它将本地调用代理到 Hub API 执行，实现"本地接口 + 远程执行"模式。

    Attributes:
        hub_url: Hub API 基础 URL
        auth_token: 认证令牌
        installed_version: 安装的版本号
        category: 工具分类
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters_schema: dict,
        hub_url: str,
        auth_token: str | None = None,
        safety: ToolSafety = ToolSafety.SAFE_WRITE,
        installed_version: str = "latest",
        category: str = "",
    ) -> None:
        """初始化 HubTool

        Args:
            name: 工具名称
            description: 工具描述
            parameters_schema: 参数 JSON Schema
            hub_url: Hub API 基础 URL
            auth_token: 认证令牌（用于远程执行鉴权）
            safety: 工具安全级别
            installed_version: 安装的版本号
            category: 工具分类
        """
        self.name = name
        self.description = description
        self.parameters_schema = parameters_schema
        self._safety = safety
        self._concurrency_safe = False
        self.hub_url = hub_url.rstrip("/")
        self.auth_token = auth_token
        self.installed_version = installed_version
        self.category = category
        self._shared_client: httpx.AsyncClient | None = None  # 由 ToolHubClient 设置

    async def execute(
        self,
        params: dict,
        progress_callback=None,
    ) -> ToolResult:
        """通过 Hub API 远程执行工具

        将本地工具调用参数发送到 Hub API 的执行端点，
        并将远程执行结果转换为 ToolResult 返回。

        如果设置了 _shared_client，复用父 ToolHubClient 的连接池；
        否则创建临时 httpx.AsyncClient（不推荐高频调用）。

        Args:
            params: 工具参数，格式由 parameters_schema 定义
            progress_callback: 进度回调（远程执行暂不支持）

        Returns:
            ToolResult: 远程执行结果
        """
        execute_url = f"{self.hub_url}/tools/{self.name}/execute"
        headers = self._build_headers()

        try:
            # 优先复用共享客户端（连接池），否则创建临时客户端
            if self._shared_client is not None and not self._shared_client.is_closed:
                response = await self._shared_client.post(
                    execute_url,
                    json={"params": params, "version": self.installed_version},
                    headers=headers,
                )
            else:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        execute_url,
                        json={"params": params, "version": self.installed_version},
                        headers=headers,
                    )

            if response.status_code == 200:
                result_data = response.json()
                return ToolResult(
                    success=result_data.get("success", True),
                    data=result_data.get("data", {}),
                    error=result_data.get("error", ""),
                    metadata={
                        "source": "hub",
                        "tool_name": self.name,
                        "version": self.installed_version,
                        **result_data.get("metadata", {}),
                    },
                )
            else:
                error_detail = ""
                try:
                    error_body = response.json()
                    error_detail = error_body.get("detail", "")
                except (json.JSONDecodeError, ValueError):
                    error_detail = response.text[:500]

                return ToolResult(
                    success=False,
                    error=f"Hub execution failed (HTTP {response.status_code}): {error_detail}",
                    metadata={"status_code": response.status_code},
                )

        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                error=f"Hub execution timed out for tool '{self.name}'",
            )
        except httpx.RequestError as exc:
            return ToolResult(
                success=False,
                error=f"Hub execution request failed: {exc}",
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Hub execution unexpected error: {exc}",
            )

    def _build_headers(self) -> dict[str, str]:
        """构建 HTTP 请求头

        Returns:
            包含 Content-Type 和可选 Authorization 的字典
        """
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def __repr__(self) -> str:
        return (
            f"<HubTool name={self.name!r} version={self.installed_version!r} "
            f"safety={self._safety.value}>"
        )


# ======================================================================
# ToolHubClient — 工具市场客户端
# ======================================================================


class ToolHubClient:
    """工具市场客户端

    提供与 TerAgent Tool Hub 交互的异步接口，支持工具搜索、安装、发布
    和已安装工具管理。

    使用 httpx.AsyncClient 实现 HTTP 连接池复用，支持异步上下文管理器
    模式自动管理客户端生命周期。

    Examples:
        异步上下文管理器（推荐）::

            async with ToolHubClient() as client:
                entries = await client.search("database")
                tool = await client.install("postgres_query", version="1.2.0")

        手动管理生命周期::

            client = ToolHubClient(auth_token="sk-xxx")
            try:
                await client.connect()
                await client.publish(my_tool, {"category": "web"})
            finally:
                await client.close()
    """

    DEFAULT_HUB_URL = "https://hub.teragent.dev/api/v1"

    def __init__(
        self,
        hub_url: str = DEFAULT_HUB_URL,
        auth_token: str | None = None,
        timeout: float = 30.0,
        cache_dir: str | Path | None = None,
    ) -> None:
        """初始化工具市场客户端

        Args:
            hub_url: Hub API 基础 URL，默认为官方 Hub 地址
            auth_token: 认证令牌（发布和安装私有工具时需要）
            timeout: HTTP 请求超时时间（秒），默认 30
            cache_dir: 本地缓存目录路径，用于持久化已安装工具信息。
                设为 None 则仅使用内存缓存（不持久化）。
        """
        self._hub_url = hub_url.rstrip("/")
        self._auth_token = auth_token
        self._timeout = timeout
        self._cache_dir = Path(cache_dir) if cache_dir else None

        # httpx 客户端（延迟初始化）
        self._client: httpx.AsyncClient | None = None

        # 已安装工具缓存: tool_name → {"entry": ToolHubEntry, "tool": HubTool}
        self._installed: dict[str, dict[str, Any]] = {}

        # 如果有缓存目录，从磁盘加载已安装工具信息
        if self._cache_dir is not None:
            self._load_cache_from_disk()

    # ===== 生命周期管理 =====

    async def connect(self) -> None:
        """显式建立 HTTP 连接

        创建 httpx.AsyncClient 实例，启用连接池复用。
        通常不需要手动调用，建议使用 async with 上下文管理器。
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._hub_url,
                timeout=self._timeout,
                headers=self._build_headers(),
            )
            logger.debug(f"ToolHubClient connected to {self._hub_url}")

    async def close(self) -> None:
        """关闭 HTTP 连接

        释放 httpx.AsyncClient 资源。通常不需要手动调用，
        建议使用 async with 上下文管理器。
        """
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            logger.debug("ToolHubClient connection closed")

    async def __aenter__(self) -> ToolHubClient:
        """异步上下文管理器入口"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """异步上下文管理器出口"""
        await self.close()

    # ===== 核心方法 =====

    async def search(self, query: str) -> list[ToolHubEntry]:
        """搜索工具市场中的工具

        根据关键词搜索远程工具市场，返回匹配的工具条目列表。
        搜索结果按相关度排序。

        Args:
            query: 搜索关键词（如 "database", "file reader"）

        Returns:
            匹配的 ToolHubEntry 列表

        Raises:
            ToolHubError: 网络错误或 API 返回错误响应时抛出
        """
        client = await self._ensure_client()

        try:
            response = await client.get(
                "/tools/search",
                params={"q": query},
            )
        except httpx.TimeoutException:
            raise ToolHubError(
                f"Search request timed out for query: {query!r}",
                detail="Connection to Tool Hub timed out",
            )
        except httpx.RequestError as exc:
            raise ToolHubError(
                f"Search request failed: {exc}",
                detail=f"Network error while searching for {query!r}",
            )

        if response.status_code != 200:
            self._raise_api_error(response, "Search failed")

        try:
            data = response.json()
            entries_raw = data.get("tools", []) if isinstance(data, dict) else data
            entries = [ToolHubEntry.from_dict(item) for item in entries_raw]
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise ToolHubError(
                f"Failed to parse search response: {exc}",
                detail="Invalid JSON in search response",
            )

        logger.info(
            f"Tool Hub search for {query!r} returned {len(entries)} results"
        )
        return entries

    async def install(
        self,
        tool_name: str,
        version: str = "latest",
    ) -> BaseTool:
        """从工具市场安装工具

        从远程 Hub 下载工具配置，实例化为 HubTool 并注册到本地缓存。
        安装后的工具可直接通过 ToolRegistry 使用。

        Args:
            tool_name: 工具名称（如 "postgres_query"）
            version: 安装版本号，默认为 "latest"

        Returns:
            安装成功的 BaseTool（HubTool）实例

        Raises:
            ToolHubError: 网络错误、工具不存在或安装失败时抛出
        """
        client = await self._ensure_client()

        # 如果已安装相同版本，直接返回缓存
        if tool_name in self._installed:
            cached = self._installed[tool_name]
            cached_entry = cached["entry"]
            if version == "latest" or cached_entry.version == version:
                logger.info(
                    f"Tool '{tool_name}' v{cached_entry.version} already installed, "
                    f"using cached version"
                )
                return cached["tool"]

        # 从 Hub API 获取工具版本信息
        try:
            response = await client.get(
                f"/tools/{tool_name}/versions/{version}",
            )
        except httpx.TimeoutException:
            raise ToolHubError(
                f"Install request timed out for tool: {tool_name!r}",
                detail="Connection to Tool Hub timed out during install",
            )
        except httpx.RequestError as exc:
            raise ToolHubError(
                f"Install request failed: {exc}",
                detail=f"Network error while installing {tool_name!r}",
            )

        if response.status_code == 404:
            raise ToolHubError(
                f"Tool '{tool_name}' version '{version}' not found",
                status_code=404,
                detail="The requested tool or version does not exist in the Hub",
            )

        if response.status_code != 200:
            self._raise_api_error(response, f"Install failed for '{tool_name}'")

        try:
            config = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ToolHubError(
                f"Failed to parse install response for '{tool_name}': {exc}",
                detail="Invalid JSON in install response",
            )

        # 从配置创建 HubTool 实例
        tool = self._create_hub_tool_from_config(config, version)

        # 共享客户端连接池，避免 HubTool 每次执行都创建新的 HTTP 连接
        if self._client is not None and not self._client.is_closed:
            tool._shared_client = self._client

        # 创建 ToolHubEntry 用于缓存
        entry = ToolHubEntry(
            name=tool.name,
            version=config.get("version", version),
            author=config.get("author", ""),
            description=tool.description,
            category=config.get("category", ""),
            downloads=config.get("downloads", 0),
            rating=float(config.get("rating", 0.0)),
        )

        # 存入本地缓存
        self._installed[tool_name] = {
            "entry": entry,
            "tool": tool,
        }

        # 可选持久化
        if self._cache_dir is not None:
            self._save_cache_to_disk()

        logger.info(
            f"Tool '{tool_name}' v{entry.version} installed successfully "
            f"(category={entry.category!r})"
        )
        return tool

    async def publish(
        self,
        tool: BaseTool,
        metadata: dict,
    ) -> None:
        """发布工具到远程市场

        将本地工具的元数据和 Schema 上传到远程 Hub 市场。
        发布后，其他用户可以通过搜索和安装使用该工具。

        Args:
            tool: 要发布的 BaseTool 实例
            metadata: 工具发布元数据，包含:
                - category: 工具分类（如 "database"）
                - author: 作者名称
                - tags: 标签列表（可选）
                - visibility: 可见性（"public" | "private"，默认 "public"）

        Raises:
            ToolHubError: 网络错误、认证失败或发布失败时抛出
        """
        client = await self._ensure_client()

        # 构建发布载荷
        payload = {
            "name": tool.name,
            "description": tool.description,
            "parameters_schema": tool.parameters_schema or {},
            "safety": tool.safety_level.value,
            "concurrency_safe": tool.is_concurrency_safe,
            "metadata": metadata,
            "function_definition": tool.to_function_definition(),
            "registry_metadata": tool.to_registry_metadata(),
        }

        try:
            response = await client.post(
                "/tools/publish",
                json=payload,
            )
        except httpx.TimeoutException:
            raise ToolHubError(
                f"Publish request timed out for tool: {tool.name!r}",
                detail="Connection to Tool Hub timed out during publish",
            )
        except httpx.RequestError as exc:
            raise ToolHubError(
                f"Publish request failed: {exc}",
                detail=f"Network error while publishing {tool.name!r}",
            )

        if response.status_code == 401:
            raise ToolHubError(
                "Authentication required for publishing",
                status_code=401,
                detail="Set auth_token when creating ToolHubClient to enable publishing",
            )

        if response.status_code == 409:
            raise ToolHubError(
                f"Tool '{tool.name}' already exists in the Hub",
                status_code=409,
                detail="Use a different tool name or update the existing version",
            )

        if response.status_code not in (200, 201):
            self._raise_api_error(response, f"Publish failed for '{tool.name}'")

        logger.info(f"Tool '{tool.name}' published successfully to Hub")

    async def list_installed(self) -> list[str]:
        """返回已安装的市场工具名称列表

        返回通过 ToolHubClient.install() 安装到本地的工具名称列表。
        该列表独立于 ToolRegistry，仅包含通过 Hub 安装的工具。

        Returns:
            已安装工具的名称列表
        """
        return list(self._installed.keys())

    # ===== 辅助方法 =====

    def get_installed_entry(self, tool_name: str) -> ToolHubEntry | None:
        """获取已安装工具的市场条目信息

        Args:
            tool_name: 工具名称

        Returns:
            ToolHubEntry 实例，如果工具未安装则返回 None
        """
        cached = self._installed.get(tool_name)
        if cached:
            return cached["entry"]
        return None

    def get_installed_tool(self, tool_name: str) -> HubTool | None:
        """获取已安装的工具实例

        Args:
            tool_name: 工具名称

        Returns:
            HubTool 实例，如果工具未安装则返回 None
        """
        cached = self._installed.get(tool_name)
        if cached:
            return cached["tool"]
        return None

    def uninstall(self, tool_name: str) -> bool:
        """卸载已安装的市场工具

        从本地缓存中移除工具。此操作仅移除本地缓存，
        不会影响远程 Hub 上的工具。

        Args:
            tool_name: 要卸载的工具名称

        Returns:
            True 表示成功卸载，False 表示工具未安装
        """
        if tool_name in self._installed:
            del self._installed[tool_name]
            if self._cache_dir is not None:
                self._save_cache_to_disk()
            logger.info(f"Tool '{tool_name}' uninstalled from local cache")
            return True
        logger.warning(f"Tool '{tool_name}' not found in installed cache")
        return False

    # ===== 内部方法 =====

    def _build_headers(self) -> dict[str, str]:
        """构建 HTTP 请求头

        Returns:
            包含 Content-Type 和可选 Authorization 的字典
        """
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        return headers

    async def _ensure_client(self) -> httpx.AsyncClient:
        """确保 HTTP 客户端已初始化

        如果客户端尚未创建或已关闭，则自动创建新客户端。

        Returns:
            可用的 httpx.AsyncClient 实例

        Raises:
            ToolHubError: 客户端初始化失败时抛出
        """
        if self._client is None or self._client.is_closed:
            await self.connect()
        assert self._client is not None
        return self._client

    def _create_hub_tool_from_config(
        self,
        config: dict[str, Any],
        version: str,
    ) -> HubTool:
        """从 Hub API 返回的配置创建 HubTool 实例

        Args:
            config: Hub API 返回的工具配置字典，包含:
                - name: 工具名称
                - description: 工具描述
                - parameters_schema: 参数 JSON Schema
                - safety: 安全级别字符串
                - category: 分类
            version: 安装版本号

        Returns:
            HubTool 实例
        """
        # 解析安全级别
        safety_str = config.get("safety", "safe_write")
        try:
            safety = ToolSafety(safety_str)
        except ValueError:
            logger.warning(
                f"Unknown safety level '{safety_str}', defaulting to SAFE_WRITE"
            )
            safety = ToolSafety.SAFE_WRITE

        return HubTool(
            name=config.get("name", ""),
            description=config.get("description", ""),
            parameters_schema=config.get("parameters_schema", {}),
            hub_url=self._hub_url,
            auth_token=self._auth_token,
            safety=safety,
            installed_version=config.get("version", version),
            category=config.get("category", ""),
        )
        # NOTE: _shared_client 由 _ensure_client 后设置，见 install()

    def _raise_api_error(
        self,
        response: httpx.Response,
        context: str,
    ) -> None:
        """从 HTTP 响应构建并抛出 ToolHubError

        Args:
            response: httpx 响应对象
            context: 错误上下文描述

        Raises:
            ToolHubError: 始终抛出
        """
        error_detail = ""
        try:
            error_body = response.json()
            error_detail = error_body.get("detail", "")
        except (json.JSONDecodeError, ValueError):
            error_detail = response.text[:500]

        raise ToolHubError(
            f"{context} (HTTP {response.status_code}): {error_detail}",
            status_code=response.status_code,
            detail=error_detail,
        )

    # ===== 缓存持久化 =====

    def _load_cache_from_disk(self) -> None:
        """从磁盘加载已安装工具缓存

        从 cache_dir 下的 hub_cache.json 文件读取已安装工具信息，
        重建 HubTool 实例存入内存缓存。
        """
        if self._cache_dir is None:
            return

        cache_file = self._cache_dir / "hub_cache.json"
        if not cache_file.exists():
            return

        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            installed_list = data.get("installed", [])
            for item in installed_list:
                entry = ToolHubEntry.from_dict(item.get("entry", {}))
                tool_config = item.get("tool_config", {})
                version = entry.version

                tool = self._create_hub_tool_from_config(tool_config, version)
                self._installed[entry.name] = {
                    "entry": entry,
                    "tool": tool,
                }

            logger.debug(
                f"Loaded {len(self._installed)} installed tools from cache "
                f"at {cache_file}"
            )
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning(f"Failed to load hub cache from disk: {exc}")

    def _save_cache_to_disk(self) -> None:
        """将已安装工具缓存持久化到磁盘

        将内存中的已安装工具信息写入 cache_dir 下的 hub_cache.json 文件。
        如果 cache_dir 不存在则自动创建。
        """
        if self._cache_dir is None:
            return

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self._cache_dir / "hub_cache.json"

            installed_list = []
            for name, cached in self._installed.items():
                entry = cached["entry"]
                tool = cached["tool"]
                installed_list.append({
                    "entry": entry.to_dict(),
                    "tool_config": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters_schema": tool.parameters_schema,
                        "safety": tool.safety_level.value,
                        "category": tool.category,
                        "version": tool.installed_version,
                    },
                })

            data = {"installed": installed_list}
            cache_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.debug(f"Saved {len(installed_list)} installed tools to {cache_file}")

        except (OSError, TypeError) as exc:
            logger.warning(f"Failed to save hub cache to disk: {exc}")

    # ===== 表示方法 =====

    def __repr__(self) -> str:
        installed_count = len(self._installed)
        client_status = "connected" if (
            self._client is not None and not self._client.is_closed
        ) else "disconnected"
        return (
            f"<ToolHubClient url={self._hub_url!r} "
            f"installed={installed_count} "
            f"status={client_status}>"
        )
