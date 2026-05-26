"""teragent.config.driver_config — DriverConfig dataclass

Represents the complete configuration for a single model driver.

New config format (agent.toml):
    [drivers.openai_compatible.glm]
    base_url = "https://open.bigmodel.cn/api/paas/v4"
    api_key_env = "GLM_API_KEY"
    model = "glm-5.1"
    compiler = "glm"

    [drivers.anthropic_native.claude]
    base_url = "https://api.anthropic.com/v1"
    api_key_env = "ANTHROPIC_API_KEY"
    model = "claude-sonnet-4-20250514"
    compiler = "anthropic"

Old config format (still supported):
    [drivers.openai_compatible]
    base_url = "https://integrate.api.nvidia.com/v1"
    api_key = "nvapi-..."
    model = "stepfun-ai/step-3.5-flash"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DriverConfig:
    """Complete configuration for a single model driver.

    The key insight: protocol (adapter) and compiler are two independent dimensions.
    The same adapter (e.g., OpenAI-compatible) can serve different compilers,
    and the same compiler (e.g., Anthropic XML) can be used with different adapters.

    Attributes:
        adapter: Protocol name — determines how to send HTTP requests.
            Values: "openai_compatible" | "anthropic_native" | "mock"
        identity: Model identity — determines *what* the model is, regardless of protocol.
            Values: "glm" | "claude" | "deepseek" | "step" | custom
        base_url: API endpoint URL
        api_key: Resolved API key (from env var, .env, or plaintext — never stored in config)
        model: Model version string (e.g., "glm-5.1", "claude-sonnet-4-20250514")
        compiler: Compiler name — determines how to compile TAP prompts.
            Values: "default" | "glm" | "anthropic" | "deepseek"
        timeout: HTTP request timeout in seconds
        extra_headers: Additional HTTP headers (for gateway auth, etc.)
        full_name: Fully qualified driver name (e.g., "openai_compatible.glm")
        api_key_env: Environment variable name for the API key (for reference/debugging)
        enable_fake_tools: Whether to inject fake tools for distillation detection
    """

    adapter: str = ""
    identity: str = ""
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    compiler: str = "default"
    timeout: float = 300.0
    extra_headers: dict[str, str] = field(default_factory=dict)
    full_name: str = ""
    api_key_env: str = ""
    enable_fake_tools: bool = False

    def __post_init__(self) -> None:
        """Derive full_name if not explicitly set"""
        if not self.full_name and self.adapter and self.identity:
            self.full_name = f"{self.adapter}.{self.identity}"

    @property
    def is_configured(self) -> bool:
        """Whether this driver has enough configuration to be usable"""
        return bool(self.adapter and self.model)

    @property
    def has_api_key(self) -> bool:
        """Whether an API key is available"""
        return bool(self.api_key)

    def to_create_provider_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs suitable for create_provider()

        Returns:
            Dict that can be unpacked into create_provider(**kwargs)
        """
        return {
            "compiler": self.compiler,
            "adapter": self.adapter,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "api_key_env": self.api_key_env,
            "timeout": self.timeout,
            "extra_headers": self.extra_headers or None,
            "enable_fake_tools": self.enable_fake_tools,
        }

    def __repr__(self) -> str:
        # Mask API key in repr for security
        masked_key = "***" if self.api_key else "(empty)"
        return (
            f"DriverConfig("
            f"full_name={self.full_name!r}, "
            f"adapter={self.adapter!r}, "
            f"identity={self.identity!r}, "
            f"compiler={self.compiler!r}, "
            f"model={self.model!r}, "
            f"base_url={self.base_url!r}, "
            f"api_key={masked_key}, "
            f"timeout={self.timeout})"
        )
