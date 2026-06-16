# teragent/tools/auth.py
"""认证方案与凭据管理 — AuthScheme / AuthCredential / AuthManager

为工具（如 OpenAPIToolset、MCPToolset）提供统一的认证基础设施:

  - AuthScheme: 声明认证方案类型（bearer / api_key / oauth2 / basic）
  - AuthCredential: 存储认证凭据（API Key、Token、Client Secret 等）
  - AuthManager: 中心化管理认证方案与凭据，支持:
      - 注册 / 查询认证方案
      - 存储 / 获取凭据
      - 将凭据应用到 HTTP 请求（headers / query params）
      - 从环境变量解析凭据
      - OAuth2 Token 刷新（基本结构）

安全设计:
  - 凭据仅存储于内存，不持久化
  - AuthCredential.__repr__ 遮蔽所有敏感字段
  - 日志中不输出任何凭据值
  - 支持 api_key_env 从环境变量读取，避免硬编码

参考: OpenAPI 3.0 Security Scheme, RFC 6750 (Bearer Token), RFC 6749 (OAuth2)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

__all__ = [
    "AuthScheme",
    "AuthCredential",
    "AuthManager",
]

logger = logging.getLogger(__name__)

# Type alias for auth scheme types
AuthType = Literal["bearer", "api_key", "oauth2", "basic"]


# ===== AuthScheme =====

@dataclass
class AuthScheme:
    """认证方案定义

    声明 API 所需的认证方式，对应 OpenAPI 3.0 Security Scheme。

    Attributes:
        type: 认证类型（bearer / api_key / oauth2 / basic）
        header_name: 认证头名称，默认 "Authorization"；
            api_key 模式下可自定义为 "X-API-Key" 等
        query_param: 查询参数名，仅 api_key 模式下使用；
            如 "api_key"，凭据将附加到 URL 查询参数中
        token_url: OAuth2 token 端点 URL，仅 oauth2 模式下使用
    """

    type: AuthType
    header_name: str = "Authorization"
    query_param: str = ""
    token_url: str = ""  # OAuth2

    def __post_init__(self) -> None:
        """验证认证方案参数合法性"""
        valid_types: set[str] = {"bearer", "api_key", "oauth2", "basic"}
        if self.type not in valid_types:
            raise ValueError(
                f"Invalid auth type: {self.type!r}. Must be one of {valid_types}"
            )

        if self.type == "oauth2" and not self.token_url:
            logger.warning(
                "AuthScheme with type='oauth2' should specify token_url"
            )

    def __repr__(self) -> str:
        return (
            f"AuthScheme(type={self.type!r}, "
            f"header_name={self.header_name!r}, "
            f"query_param={self.query_param!r}, "
            f"token_url={self.token_url!r})"
        )


# ===== AuthCredential =====

@dataclass
class AuthCredential:
    """认证凭据存储

    存储各种认证方式所需的凭据。支持环境变量引用（api_key_env），
    避免 API Key 等敏感信息硬编码在代码中。

    安全说明:
      - __repr__ 遮蔽所有敏感字段，仅显示是否已设置
      - 凭据仅存储于内存，绝不持久化或写入日志

    Attributes:
        api_key: API Key 值（直接设置或通过 api_key_env 从环境变量读取）
        api_key_env: API Key 对应的环境变量名；
            resolve_credential() 会优先从此环境变量读取值
        client_id: OAuth2 Client ID
        client_secret: OAuth2 Client Secret
        access_token: Bearer / OAuth2 Access Token
        refresh_token: OAuth2 Refresh Token
    """

    api_key: str = ""
    api_key_env: str = ""  # 环境变量名
    client_id: str = ""
    client_secret: str = ""
    access_token: str = ""
    refresh_token: str = ""

    @staticmethod
    def _mask(value: str) -> str:
        """遮蔽敏感值，仅显示长度和首尾字符

        Args:
            value: 待遮蔽的字符串

        Returns:
            遮蔽后的字符串，如 "ab***yz (8 chars)" 或 "<empty>"
        """
        if not value:
            return "<empty>"
        if len(value) <= 4:
            return f"**** ({len(value)} chars)"
        return f"{value[:2]}***{value[-2:]} ({len(value)} chars)"

    def __repr__(self) -> str:
        """遮蔽敏感字段的 repr，防止凭据泄露到日志或调试输出"""
        return (
            f"AuthCredential("
            f"api_key={self._mask(self.api_key)}, "
            f"api_key_env={self.api_key_env!r}, "
            f"client_id={self._mask(self.client_id)}, "
            f"client_secret={self._mask(self.client_secret)}, "
            f"access_token={self._mask(self.access_token)}, "
            f"refresh_token={self._mask(self.refresh_token)})"
        )

    def is_empty(self) -> bool:
        """检查凭据是否全部为空

        Returns:
            True 如果所有凭据字段都为空字符串
        """
        return not any([
            self.api_key,
            self.api_key_env,
            self.client_id,
            self.client_secret,
            self.access_token,
            self.refresh_token,
        ])


# ===== AuthManager =====

class AuthManager:
    """认证方案与凭据的中心化管理器

    管理认证方案（AuthScheme）和凭据（AuthCredential）的注册、
    查询和应用。工具（如 OpenAPIToolset）可通过 AuthManager
    获取认证信息并应用到 HTTP 请求中。

    用法:
        manager = AuthManager()

        # 注册认证方案
        manager.register_scheme("github", AuthScheme(type="bearer"))
        manager.register_scheme("weather", AuthScheme(
            type="api_key", header_name="X-API-Key"
        ))

        # 设置凭据
        manager.set_credential("github", AuthCredential(access_token="ghp_xxx"))
        manager.set_credential("weather", AuthCredential(
            api_key_env="WEATHER_API_KEY"
        ))

        # 应用认证到请求
        headers = {}
        params = {}
        manager.apply_auth("github", headers, params)
        # headers => {"Authorization": "Bearer ghp_xxx"}

    线程安全说明:
        当前实现为单线程设计。如需在多线程环境中使用，
        调用方应自行加锁保护 _schemes 和 _credentials 字典。
    """

    def __init__(self) -> None:
        self._schemes: Dict[str, AuthScheme] = {}
        self._credentials: Dict[str, AuthCredential] = {}

    # ===== 认证方案管理 =====

    def register_scheme(self, name: str, scheme: AuthScheme) -> None:
        """注册认证方案

        Args:
            name: 方案名称（唯一标识符，如 "github", "stripe"）
            scheme: AuthScheme 实例

        Raises:
            TypeError: 如果 scheme 不是 AuthScheme 实例
            ValueError: 如果 name 为空字符串
        """
        if not name or not isinstance(name, str):
            raise ValueError("Scheme name must be a non-empty string")
        if not isinstance(scheme, AuthScheme):
            raise TypeError(
                f"scheme must be an AuthScheme instance, got {type(scheme).__name__}"
            )

        if name in self._schemes:
            logger.warning(
                "Overwriting existing auth scheme: %s", name
            )

        self._schemes[name] = scheme
        logger.debug("Registered auth scheme: %s (type=%s)", name, scheme.type)

    def get_scheme(self, name: str) -> Optional[AuthScheme]:
        """获取认证方案

        Args:
            name: 方案名称

        Returns:
            AuthScheme 实例，若不存在则返回 None
        """
        return self._schemes.get(name)

    def list_schemes(self) -> Dict[str, AuthScheme]:
        """列出所有已注册的认证方案

        Returns:
            方案名称到 AuthScheme 的映射（浅拷贝）
        """
        return dict(self._schemes)

    def remove_scheme(self, name: str) -> bool:
        """移除认证方案

        同时移除关联的凭据。

        Args:
            name: 方案名称

        Returns:
            True 如果方案存在并被移除，False 如果方案不存在
        """
        if name in self._schemes:
            del self._schemes[name]
            self._credentials.pop(name, None)
            logger.debug("Removed auth scheme and credential: %s", name)
            return True
        return False

    # ===== 凭据管理 =====

    def set_credential(self, scheme_name: str, credential: AuthCredential) -> None:
        """存储凭据

        如果 scheme_name 对应的认证方案尚未注册，将记录警告但仍然存储凭据，
        允许先设置凭据再注册方案的使用模式。

        Args:
            scheme_name: 关联的认证方案名称
            credential: AuthCredential 实例

        Raises:
            TypeError: 如果 credential 不是 AuthCredential 实例
            ValueError: 如果 scheme_name 为空字符串
        """
        if not scheme_name or not isinstance(scheme_name, str):
            raise ValueError("Scheme name must be a non-empty string")
        if not isinstance(credential, AuthCredential):
            raise TypeError(
                f"credential must be an AuthCredential instance, "
                f"got {type(credential).__name__}"
            )

        if scheme_name not in self._schemes:
            logger.warning(
                "Setting credential for unregistered scheme: %s", scheme_name
            )

        self._credentials[scheme_name] = credential
        logger.debug("Set credential for scheme: %s", scheme_name)

    def get_credential(self, scheme_name: str) -> Optional[AuthCredential]:
        """获取凭据

        Args:
            scheme_name: 关联的认证方案名称

        Returns:
            AuthCredential 实例，若不存在则返回 None
        """
        return self._credentials.get(scheme_name)

    # ===== 认证应用 =====

    def resolve_credential(self, credential: AuthCredential) -> AuthCredential:
        """解析凭据中的环境变量引用

        如果 credential.api_key_env 已设置，则从 os.environ 中读取
        对应的值并填充到 api_key 字段。原始 api_key 值仅在环境变量
        不存在时作为回退。

        Args:
            credential: 待解析的 AuthCredential 实例

        Returns:
            新的 AuthCredential 实例，api_key 字段已从环境变量解析
            （如果 api_key_env 已设置且环境变量存在）
        """
        if not credential.api_key_env:
            return credential

        env_value = os.environ.get(credential.api_key_env, "")
        if env_value:
            # 创建新实例，避免修改原始凭据
            return AuthCredential(
                api_key=env_value,
                api_key_env=credential.api_key_env,
                client_id=credential.client_id,
                client_secret=credential.client_secret,
                access_token=credential.access_token,
                refresh_token=credential.refresh_token,
            )
        else:
            logger.warning(
                "Environment variable %r is not set or empty",
                credential.api_key_env,
            )
            return credential

    def apply_auth(
        self,
        scheme_name: str,
        request_headers: Dict[str, str],
        request_params: Dict[str, Any],
    ) -> bool:
        """将认证信息应用到 HTTP 请求

        根据认证方案类型，将凭据注入到请求头或查询参数中:

          - bearer: 添加 "Authorization: Bearer <token>" 头
          - api_key: 根据 header_name 或 query_param 添加到请求头或查询参数
          - oauth2: 添加 "Authorization: Bearer <access_token>" 头
          - basic: 添加 "Authorization: Basic <encoded>" 头

        Args:
            scheme_name: 认证方案名称
            request_headers: 请求头字典（会被就地修改）
            request_params: 查询参数字典（会被就地修改）

        Returns:
            True 如果认证成功应用，False 如果方案或凭据不存在

        Raises:
            ValueError: 如果凭据缺少必需字段
        """
        scheme = self._schemes.get(scheme_name)
        if scheme is None:
            logger.warning("Auth scheme not found: %s", scheme_name)
            return False

        credential = self._credentials.get(scheme_name)
        if credential is None:
            logger.warning("Credential not found for scheme: %s", scheme_name)
            return False

        # 解析环境变量引用
        credential = self.resolve_credential(credential)

        try:
            if scheme.type == "bearer":
                self._apply_bearer(credential, request_headers)
            elif scheme.type == "api_key":
                self._apply_api_key(scheme, credential, request_headers, request_params)
            elif scheme.type == "oauth2":
                self._apply_oauth2(scheme, credential, request_headers)
            elif scheme.type == "basic":
                self._apply_basic(credential, request_headers)
            else:
                logger.error("Unsupported auth type: %s", scheme.type)
                return False
        except ValueError as e:
            logger.error("Failed to apply auth for scheme %s: %s", scheme_name, e)
            return False

        return True

    def _apply_bearer(
        self,
        credential: AuthCredential,
        headers: Dict[str, str],
    ) -> None:
        """应用 Bearer Token 认证

        Args:
            credential: 凭据实例
            headers: 请求头字典（就地修改）

        Raises:
            ValueError: 如果 access_token 为空
        """
        token = credential.access_token
        if not token:
            # 回退：尝试用 api_key 作为 token
            token = credential.api_key
        if not token:
            raise ValueError(
                "Bearer auth requires access_token or api_key, both are empty"
            )
        headers["Authorization"] = f"Bearer {token}"

    def _apply_api_key(
        self,
        scheme: AuthScheme,
        credential: AuthCredential,
        headers: Dict[str, str],
        params: Dict[str, Any],
    ) -> None:
        """应用 API Key 认证

        根据 scheme 配置将 API Key 注入到请求头或查询参数。

        Args:
            scheme: 认证方案
            credential: 凭据实例
            headers: 请求头字典（就地修改）
            params: 查询参数字典（就地修改）

        Raises:
            ValueError: 如果 api_key 为空
        """
        key_value = credential.api_key
        if not key_value:
            raise ValueError("API Key auth requires api_key, but it is empty")

        if scheme.query_param:
            # API Key 通过查询参数传递
            params[scheme.query_param] = key_value
        else:
            # API Key 通过请求头传递
            header_name = scheme.header_name or "X-API-Key"
            headers[header_name] = key_value

    def _apply_oauth2(
        self,
        scheme: AuthScheme,
        credential: AuthCredential,
        headers: Dict[str, str],
    ) -> None:
        """应用 OAuth2 认证

        使用 access_token 作为 Bearer Token。如果 access_token 过期
        且 refresh_token 可用，尝试刷新（基本结构）。

        Args:
            scheme: 认证方案
            credential: 凭据实例
            headers: 请求头字典（就地修改）

        Raises:
            ValueError: 如果 access_token 和 refresh_token 都为空
        """
        token = credential.access_token
        if not token and credential.refresh_token:
            # 尝试刷新 token（基本结构，子类或使用者可覆盖此逻辑）
            token = self._try_refresh_token(scheme, credential)

        if not token:
            raise ValueError(
                "OAuth2 auth requires access_token or refresh_token, both are empty"
            )
        headers["Authorization"] = f"Bearer {token}"

    def _apply_basic(
        self,
        credential: AuthCredential,
        headers: Dict[str, str],
    ) -> None:
        """应用 Basic 认证

        使用 client_id 作为 username，client_secret 作为 password，
        进行 Base64 编码后添加到 Authorization 头。

        也可使用 api_key 作为 username（某些 API 的简化认证方式）。

        Args:
            credential: 凭据实例
            headers: 请求头字典（就地修改）

        Raises:
            ValueError: 如果缺少必需的用户名/密码
        """
        import base64

        username = credential.client_id or credential.api_key
        password = credential.client_secret

        if not username:
            raise ValueError(
                "Basic auth requires client_id (or api_key) as username"
            )

        credentials_str = f"{username}:{password}"
        encoded = base64.b64encode(credentials_str.encode("utf-8")).decode("utf-8")
        headers["Authorization"] = f"Basic {encoded}"

    def _try_refresh_token(
        self,
        scheme: AuthScheme,
        credential: AuthCredential,
    ) -> str:
        """尝试刷新 OAuth2 Access Token

        基本结构：发送 POST 请求到 token_url，使用 refresh_token
        换取新的 access_token。实际项目中应考虑:
          - Token 缓存与过期时间
          - 并发刷新的锁保护
          - 刷新失败的重试策略

        Args:
            scheme: 认证方案（需包含 token_url）
            credential: 凭据实例（需包含 refresh_token 和 client_id）

        Returns:
            新的 access_token，刷新失败时返回空字符串
        """
        if not scheme.token_url:
            logger.warning(
                "OAuth2 token refresh failed: token_url not configured"
            )
            return ""

        if not credential.refresh_token:
            logger.warning(
                "OAuth2 token refresh failed: refresh_token not available"
            )
            return ""

        try:
            import httpx
        except ImportError:
            logger.warning(
                "OAuth2 token refresh requires httpx. "
                "Install it with: pip install httpx"
            )
            return ""

        try:
            # 同步发送 token 刷新请求
            # 注意: 在异步上下文中应使用 httpx.AsyncClient
            # 此处为基本结构，使用者可根据需要改为异步版本
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    scheme.token_url,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": credential.refresh_token,
                        "client_id": credential.client_id,
                        "client_secret": credential.client_secret,
                    },
                )
                response.raise_for_status()
                token_data = response.json()

                new_access_token = token_data.get("access_token", "")
                if new_access_token:
                    # 更新内存中的凭据
                    credential.access_token = new_access_token
                    # 如果响应包含新的 refresh_token，也更新
                    new_refresh_token = token_data.get("refresh_token", "")
                    if new_refresh_token:
                        credential.refresh_token = new_refresh_token
                    logger.info("OAuth2 token refreshed successfully")
                    return new_access_token
                else:
                    logger.warning(
                        "OAuth2 token refresh response missing access_token"
                    )
                    return ""

        except Exception as e:
            # 不记录异常详情以防凭据泄露
            logger.warning("OAuth2 token refresh failed: %s", type(e).__name__)
            return ""

    def __repr__(self) -> str:
        scheme_names = list(self._schemes.keys())
        return (
            f"AuthManager(schemes={scheme_names}, "
            f"credential_count={len(self._credentials)})"
        )
