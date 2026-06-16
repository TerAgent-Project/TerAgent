# teragent/tools/openapi_toolset.py
"""OpenAPI 工具集 — 从 OpenAPI 规范自动生成工具

解析 OpenAPI 3.0/Swagger 规范，为每个 operation 生成
OpenAPIOperationTool 实例，支持 HTTP 调用。

核心组件:
  - OpenAPIOperationTool: 单个 OpenAPI 操作的工具封装
  - OpenAPIToolset: 从 OpenAPI 规范批量生成工具集

设计原则:
  - 简单 dict 遍历解析，不依赖 openapi-python_client 等重型库
  - 只处理常见场景（path/query/header 参数 + JSON body）
  - 使用 httpx.AsyncClient 发送 HTTP 请求
  - 安全级别根据 HTTP method 自动推断（GET → READ_ONLY, 其他 → SAFE_WRITE）

参考: OpenAPI 3.0 Specification, openapi-python-client 的 Pydantic 模型
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, TYPE_CHECKING

from teragent.tools.base import BaseTool, ToolResult
from teragent.core.types import ToolSafety

if TYPE_CHECKING:
    from teragent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

__all__ = [
    "OpenAPIOperationTool",
    "OpenAPIToolset",
]


# ===== OpenAPIOperationTool =====

class OpenAPIOperationTool(BaseTool):
    """单个 OpenAPI 操作的工具封装

    从 OpenAPI 规范中提取的单个操作（HTTP method + path），
    封装为 BaseTool 实例，调用时通过 httpx 发送 HTTP 请求。

    Attributes:
        operation_id: 操作唯一标识符（用作工具名称）
        method: HTTP 方法（GET/POST/PUT/PATCH/DELETE）
        path: URL 路径模板（如 /pets/{petId}）
        base_url: API 基础 URL
        parameters_schema: JSON Schema 格式的参数定义
        description: 工具描述
        path_params: 路径参数名列表
        query_params: 查询参数名列表
        header_params: 请求头参数名列表
        body_content_type: 请求体内容类型
        auth: 认证配置
    """

    def __init__(
        self,
        operation_id: str,
        method: str,
        path: str,
        base_url: str,
        parameters_schema: dict,
        description: str = "",
        path_params: list[str] | None = None,
        query_params: list[str] | None = None,
        header_params: list[str] | None = None,
        body_content_type: str = "application/json",
        auth: dict | None = None,
        safety: ToolSafety = ToolSafety.SAFE_WRITE,
    ):
        self.operation_id = operation_id
        self.method = method.upper()
        self.path = path
        self.base_url = base_url.rstrip("/")
        self.parameters_schema = parameters_schema
        self.description = description or f"{self.method} {self.path}"
        self.path_params = path_params or []
        self.query_params = query_params or []
        self.header_params = header_params or []
        self.body_content_type = body_content_type
        self.auth = auth

        # BaseTool 属性
        self.name = operation_id
        self._safety = safety
        self._concurrency_safe = self.method == "GET"

    async def execute(
        self,
        params: dict,
        progress_callback: Callable[[str, float], Awaitable[None]] | None = None,
    ) -> ToolResult:
        """执行 HTTP 请求

        1. 构建 URL（替换路径参数如 {petId}）
        2. 提取查询参数
        3. 构建请求体
        4. 通过 httpx.AsyncClient 发送 HTTP 请求
        5. 返回 ToolResult

        Args:
            params: 工具参数字典
            progress_callback: 进度回调（未使用）

        Returns:
            ToolResult 实例
        """
        try:
            import httpx
        except ImportError:
            return ToolResult(
                success=False,
                error="httpx is required for OpenAPIToolset. Install it with: pip install httpx",
            )

        try:
            # 1. 构建 URL（替换路径参数）
            url_path = self.path
            for param_name in self.path_params:
                placeholder = "{" + param_name + "}"
                value = params.get(param_name, "")
                url_path = url_path.replace(placeholder, str(value))

            url = f"{self.base_url}{url_path}"

            # 2. 提取查询参数
            query: dict[str, Any] = {}
            for param_name in self.query_params:
                if param_name in params:
                    query[param_name] = params[param_name]

            # 3. 提取请求头参数
            headers: dict[str, str] = {"Content-Type": self.body_content_type}
            for param_name in self.header_params:
                if param_name in params:
                    headers[param_name] = str(params[param_name])

            # 4. 应用认证
            if self.auth:
                auth_type = self.auth.get("type", "")
                if auth_type == "bearer":
                    token = self.auth.get("token", "")
                    if token:
                        headers["Authorization"] = f"Bearer {token}"
                elif auth_type == "api_key":
                    key_name = self.auth.get("key_name", "X-API-Key")
                    key_value = self.auth.get("key_value", "")
                    location = self.auth.get("in", "header")
                    if location == "header" and key_value:
                        headers[key_name] = key_value
                    elif location == "query" and key_value:
                        query[key_name] = key_value
                elif auth_type == "basic":
                    # httpx 原生支持 basic auth
                    pass

            # 5. 构建请求体
            body_param_names = set(self.path_params + self.query_params + self.header_params)
            body_params = {
                k: v for k, v in params.items()
                if k not in body_param_names and k != "body"
            }
            # 如果 params 中有 'body' 键，直接使用
            request_body = params.get("body", body_params) if body_params or "body" in params else None

            # 6. 发送 HTTP 请求
            async with httpx.AsyncClient(timeout=30.0) as client:
                request_kwargs: dict[str, Any] = {
                    "method": self.method,
                    "url": url,
                    "params": query or None,
                    "headers": headers,
                }

                # Basic auth
                if self.auth and self.auth.get("type") == "basic":
                    request_kwargs["auth"] = (
                        self.auth.get("username", ""),
                        self.auth.get("password", ""),
                    )

                # 添加请求体（GET/HEAD 通常没有 body）
                if request_body is not None and self.method not in ("GET", "HEAD"):
                    if isinstance(request_body, str):
                        request_kwargs["content"] = request_body
                    else:
                        request_kwargs["json"] = request_body

                if progress_callback:
                    await progress_callback(f"Sending {self.method} {url}", 0.5)

                response = await client.request(**request_kwargs)

            # 7. 解析响应
            status_code = response.status_code
            is_success = 200 <= status_code < 300

            # 尝试解析 JSON 响应
            try:
                response_data = response.json()
            except (json.JSONDecodeError, ValueError):
                response_data = response.text

            metadata = {
                "status_code": status_code,
                "method": self.method,
                "url": str(response.url),
                "headers": dict(response.headers),
            }

            if is_success:
                return ToolResult(
                    success=True,
                    data={
                        "status_code": status_code,
                        "body": response_data,
                    },
                    metadata=metadata,
                    safety=self._safety,
                )
            else:
                return ToolResult(
                    success=False,
                    error=f"HTTP {status_code}: {response.text[:500]}",
                    data={
                        "status_code": status_code,
                        "body": response_data,
                    },
                    metadata=metadata,
                    safety=self._safety,
                )

        except Exception as e:
            logger.error(f"OpenAPIOperationTool '{self.name}' execution failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                data={"exception": type(e).__name__},
            )

    def __repr__(self) -> str:
        return (
            f"OpenAPIOperationTool(name={self.name!r}, "
            f"method={self.method}, "
            f"path={self.path!r}, "
            f"safety={self._safety.value})"
        )


# ===== OpenAPIToolset =====

class OpenAPIToolset:
    """OpenAPI 工具集 — 从 OpenAPI 规范自动生成工具

    解析 OpenAPI 3.0/Swagger 规范，为每个 operation 生成
    OpenAPIOperationTool 实例，支持 HTTP 调用。

    支持:
      - dict 格式的已解析规范
      - JSON/YAML 文件路径
      - URL 拉取规范
      - tool_filter 过滤特定操作
      - 安全级别自动推断

    用法:
        toolset = OpenAPIToolset(spec="openapi.yaml", base_url="http://localhost:8080")
        tools = await toolset.parse()
        toolset.register_to(registry)

    Attributes:
        spec: OpenAPI 规范（dict / URL / 文件路径）
        base_url: API 基础 URL
        auth: 认证配置
        tool_filter: 工具过滤列表（只保留指定 operationId）
        default_safety: 默认安全级别
    """

    def __init__(
        self,
        spec: dict | str,
        base_url: str | None = None,
        auth: dict | None = None,
        tool_filter: list[str] | None = None,
        default_safety: ToolSafety = ToolSafety.SAFE_WRITE,
    ):
        self._spec_input = spec
        self.base_url = base_url or ""
        self.auth = auth
        self.tool_filter = tool_filter
        self.default_safety = default_safety
        self._tools: list[OpenAPIOperationTool] = []
        self._parsed_spec: dict | None = None

    async def parse(self) -> list[OpenAPIOperationTool]:
        """解析 OpenAPI 规范，生成工具列表

        支持三种输入:
        1. dict — 直接使用
        2. 文件路径（.json/.yaml/.yml）— 读取并解析
        3. URL — 通过 httpx 拉取

        Returns:
            OpenAPIOperationTool 实例列表
        """
        spec = await self._load_spec()
        self._parsed_spec = spec

        # 提取 base_url
        base_url = self.base_url
        if not base_url:
            base_url = self._extract_base_url(spec)
            self.base_url = base_url  # 缓存提取结果

        # 解析 paths
        tools: list[OpenAPIOperationTool] = []
        paths = spec.get("paths", {})

        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue

            # 遍历 HTTP methods
            for method in ("get", "post", "put", "patch", "delete", "head", "options"):
                operation = path_item.get(method)
                if not operation or not isinstance(operation, dict):
                    continue

                tool = self._parse_operation(path, method, operation, base_url)
                if tool is None:
                    continue

                # 应用 tool_filter
                if self.tool_filter and tool.name not in self.tool_filter:
                    continue

                tools.append(tool)

        self._tools = tools
        logger.info(f"OpenAPIToolset parsed {len(tools)} operations from spec")
        return tools

    def to_base_tools(self) -> list[BaseTool]:
        """返回 BaseTool 列表

        Returns:
            OpenAPIOperationTool 实例列表（作为 BaseTool 类型）
        """
        return list(self._tools)

    def register_to(self, registry: ToolRegistry) -> None:
        """将所有工具注册到 ToolRegistry

        Args:
            registry: 工具注册表实例
        """
        for tool in self._tools:
            registry.register(tool)
        logger.info(f"OpenAPIToolset registered {len(self._tools)} tools to registry")

    # ===== 内部方法 =====

    async def _load_spec(self) -> dict:
        """加载 OpenAPI 规范

        根据输入类型选择加载方式:
        - dict: 直接返回
        - str: 自动检测是文件路径还是 URL

        Returns:
            解析后的规范字典
        """
        if isinstance(self._spec_input, dict):
            return self._spec_input

        spec_str = self._spec_input
        if not isinstance(spec_str, str):
            raise TypeError(f"spec must be dict or str, got {type(self._spec_input)}")

        # 判断是文件路径还是 URL
        if spec_str.startswith(("http://", "https://")):
            return await self._load_spec_from_url(spec_str)
        else:
            return self._load_spec_from_file(spec_str)

    async def _load_spec_from_url(self, url: str) -> dict:
        """从 URL 拉取 OpenAPI 规范

        Args:
            url: 规范文件的 URL

        Returns:
            解析后的规范字典
        """
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required to load spec from URL. Install it with: pip install httpx")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "yaml" in content_type or url.endswith((".yaml", ".yml")):
                return self._parse_yaml(response.text)
            return response.json()

    def _load_spec_from_file(self, file_path: str) -> dict:
        """从文件加载 OpenAPI 规范

        支持 JSON 和 YAML 格式。

        Args:
            file_path: 规范文件路径

        Returns:
            解析后的规范字典
        """
        import os

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"OpenAPI spec file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if file_path.endswith((".yaml", ".yml")):
            return self._parse_yaml(content)
        else:
            return json.loads(content)

    @staticmethod
    def _parse_yaml(content: str) -> dict:
        """解析 YAML 内容

        Args:
            content: YAML 文本内容

        Returns:
            解析后的字典

        Raises:
            ImportError: 如果未安装 pyyaml
        """
        try:
            import yaml
            return yaml.safe_load(content)
        except ImportError:
            raise ImportError(
                "pyyaml is required to parse YAML specs. "
                "Install it with: pip install pyyaml"
            )

    def _extract_base_url(self, spec: dict) -> str:
        """从规范中提取 base_url

        优先级:
        1. 构造函数传入的 base_url
        2. servers[0].url
        3. host + basePath (Swagger 2.0)
        4. 空字符串

        Args:
            spec: OpenAPI 规范字典

        Returns:
            base URL 字符串
        """
        # OpenAPI 3.0: servers
        servers = spec.get("servers", [])
        if servers and isinstance(servers, list):
            first_server = servers[0]
            if isinstance(first_server, dict):
                url = first_server.get("url", "")
                if url:
                    return url.rstrip("/")

        # Swagger 2.0: host + basePath + schemes
        host = spec.get("host", "")
        base_path = spec.get("basePath", "")
        schemes = spec.get("schemes", ["https"])
        scheme = schemes[0] if schemes else "https"
        if host:
            return f"{scheme}://{host}{base_path}".rstrip("/")

        return ""

    def _parse_operation(
        self,
        path: str,
        method: str,
        operation: dict,
        base_url: str,
    ) -> OpenAPIOperationTool | None:
        """解析单个操作，生成 OpenAPIOperationTool

        从操作定义中提取:
        - operationId（工具名称）
        - summary + description（工具描述）
        - parameters（分类为 path/query/header）
        - requestBody（请求体）
        - 安全级别

        Args:
            path: URL 路径
            method: HTTP 方法
            operation: 操作定义字典
            base_url: API 基础 URL

        Returns:
            OpenAPIOperationTool 实例，或 None 表示跳过
        """
        # 提取 operationId
        operation_id = operation.get("operationId", "")
        if not operation_id:
            # 自动生成 operationId: method + path
            # /pets/{petId} → get_pets_petId
            clean_path = re.sub(r"[{}]", "", path)
            clean_path = re.sub(r"[/.]", "_", clean_path).strip("_")
            operation_id = f"{method}_{clean_path}"

        # 提取描述
        summary = operation.get("summary", "")
        description = operation.get("description", "")
        tool_description = summary
        if description:
            tool_description = f"{summary}: {description}" if summary else description

        # 解析参数
        path_params: list[str] = []
        query_params: list[str] = []
        header_params: list[str] = []
        schema_properties: dict[str, Any] = {}
        required_params: list[str] = []

        # 处理 parameters 列表
        parameters = operation.get("parameters", [])
        for param in parameters:
            if not isinstance(param, dict):
                continue

            param_name = param.get("name", "")
            param_in = param.get("in", "query")
            param_required = param.get("required", False)
            param_schema = param.get("schema", {})
            param_description = param.get("description", "")

            # 构建参数 schema
            prop: dict[str, Any] = {}
            if param_schema:
                prop.update(param_schema)
            else:
                # Swagger 2.0: type 直接在 parameter 上
                if "type" in param:
                    prop["type"] = param["type"]
                if "enum" in param:
                    prop["enum"] = param["enum"]

            if param_description:
                prop["description"] = param_description

            if param_name:
                schema_properties[param_name] = prop
                if param_required:
                    required_params.append(param_name)

                if param_in == "path":
                    path_params.append(param_name)
                elif param_in == "query":
                    query_params.append(param_name)
                elif param_in == "header":
                    header_params.append(param_name)

        # 处理 requestBody
        request_body = operation.get("requestBody", {})
        if request_body and isinstance(request_body, dict):
            content = request_body.get("content", {})
            # 优先使用 application/json
            json_content = content.get("application/json", {})
            body_schema = json_content.get("schema", {})

            if body_schema:
                # 如果 schema 有 $ref，暂时跳过解析（简化处理）
                if "$ref" not in body_schema:
                    body_props = body_schema.get("properties", {})
                    body_required = body_schema.get("required", [])

                    for prop_name, prop_schema in body_props.items():
                        if prop_name not in schema_properties:
                            schema_properties[prop_name] = prop_schema
                            if prop_name in body_required:
                                required_params.append(prop_name)

        # 构建参数 JSON Schema
        parameters_schema: dict[str, Any] = {
            "type": "object",
            "properties": schema_properties,
        }
        if required_params:
            parameters_schema["required"] = required_params

        # 推断安全级别
        safety = self._infer_safety(method.upper())

        return OpenAPIOperationTool(
            operation_id=operation_id,
            method=method,
            path=path,
            base_url=base_url,
            parameters_schema=parameters_schema,
            description=tool_description,
            path_params=path_params,
            query_params=query_params,
            header_params=header_params,
            body_content_type="application/json",
            auth=self.auth,
            safety=safety,
        )

    def _infer_safety(self, method: str) -> ToolSafety:
        """根据 HTTP 方法推断安全级别

        GET/HEAD/OPTIONS → READ_ONLY
        POST/PUT/PATCH/DELETE → default_safety（通常为 SAFE_WRITE）

        Args:
            method: HTTP 方法（大写）

        Returns:
            ToolSafety 枚举值
        """
        if method in ("GET", "HEAD", "OPTIONS"):
            return ToolSafety.READ_ONLY
        return self.default_safety

    def __repr__(self) -> str:
        tool_count = len(self._tools)
        return f"OpenAPIToolset(tools={tool_count}, base_url={self.base_url!r})"
