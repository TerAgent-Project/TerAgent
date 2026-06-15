"""teragent.config.loader — Configuration loading with new and old format support

Supports two config formats in agent.toml:

## New format (protocol.identity two-level structure):

    [drivers.openai_compatible.glm_5]
    base_url = "https://open.bigmodel.cn/api/paas/v4"
    api_key_env = "GLM_API_KEY"
    model = "glm-5"
    compiler = "glm_5"

    [drivers.anthropic_native.claude]
    base_url = "https://api.anthropic.com/v1"
    api_key_env = "ANTHROPIC_API_KEY"
    model = "claude-sonnet-4-20250514"
    compiler = "anthropic"

## Old format (flat structure, still supported with auto-inference):

    [drivers.openai_compatible]
    base_url = "https://integrate.api.nvidia.com/v1"
    api_key = "nvapi-..."
    model = "stepfun-ai/step-3.5-flash"

The loader automatically detects which format is being used and handles both.
For old format, compiler is auto-inferred from the identity/adapter name.

## API Key resolution priority (Phase 9):
1. Environment variable referenced by `api_key_env`
2. .env file (if python-dotenv is available)
3. Explicit `api_key` in config (DEPRECATED — emits warning)
4. Empty string (warning logged)

## Pipeline config:

    [execution.pipeline]
    design_driver = "openai_compatible.glm_5"
    plan_driver = "openai_compatible.glm_5"
    execute_driver = "openai_compatible.glm_5"
    review_driver = "openai_compatible.glm_5"

Old format also supported:

    [execution.pipeline]
    design_model = "openai_compatible"
    plan_model = "openai_compatible"
    execute_model = "openai_compatible"
    review_model = "openai_compatible"
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

# tomllib is stdlib in 3.11+; fall back to tomli (pip) for 3.10
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

from teragent.config.api_key_security import (
    ApiKeyVault,
    ResolvedKey,
    mask_api_key,
)
from teragent.config.driver_config import DriverConfig

logger = logging.getLogger(__name__)


# ===== API Key Resolution =====
# Phase 9: All key resolution delegated to ApiKeyVault

# Module-level vault for backward compat with resolve_api_key()
_loader_vault = ApiKeyVault()


def resolve_api_key(
    settings: dict[str, Any],
    full_name: str = "",
) -> tuple[str, str]:
    """Resolve API key from config settings.

    Priority (Phase 9): env var > .env file > plaintext api_key (DEPRECATED) > empty

    This function delegates to ApiKeyVault.resolve_from_settings() for
    consistent security handling across the library.

    Args:
        settings: Driver settings dict from TOML config
        full_name: Fully qualified driver name (for logging)

    Returns:
        (api_key, api_key_env) tuple
    """
    resolved = _loader_vault.resolve_from_settings(settings, full_name=full_name)
    return resolved.key, resolved.env_var


def resolve_api_key_detailed(
    settings: dict[str, Any],
    full_name: str = "",
) -> ResolvedKey:
    """Resolve API key from config settings with detailed result.

    Same as resolve_api_key() but returns a ResolvedKey object with
    additional metadata (source, provider, is_plaintext, etc.).

    Args:
        settings: Driver settings dict from TOML config
        full_name: Fully qualified driver name (for logging)

    Returns:
        ResolvedKey with detailed resolution information
    """
    return _loader_vault.resolve_from_settings(settings, full_name=full_name)


# ===== Compiler Auto-Inference =====

# Map of identity → inferred compiler
_COMPILER_INFERENCE_MAP: dict[str, str] = {
    "glm": "glm",
    "claude": "anthropic",
    "anthropic": "anthropic",
    "deepseek": "deepseek",
    "deepseek_v4": "deepseek_v4",
    "deepseek_v4_flash": "deepseek_v4",
    "deepseek_v4_pro": "deepseek_v4",
    "minimax": "minimax_m3",
    "minimax_m3": "minimax_m3",
    "glm_5": "glm_5",
    "glm_52": "glm_52",
    "glm-5.2": "glm_52",
    "step": "default",
    "gpt": "default",
    "qwen": "default",
}

# Map of old driver name → (adapter, identity)
_OLD_DRIVER_NAME_MAP: dict[str, tuple[str, str]] = {
    "openai_compatible": ("openai_compatible", "default"),
    "anthropic_compatible": ("anthropic_native", "claude"),
    "minimax_native": ("minimax_native", "minimax"),
    "minimax": ("minimax_native", "minimax"),
    "glm": ("openai_compatible", "glm"),
    "mock": ("mock", "mock"),
}


def infer_compiler(identity: str, adapter: str = "") -> str:
    """Auto-infer compiler from model identity.

    Args:
        identity: Model identity name (e.g., "glm", "claude", "deepseek")
        adapter: Adapter/protocol name (optional, for additional heuristics)

    Returns:
        Compiler name string
    """
    # Direct lookup by identity
    if identity in _COMPILER_INFERENCE_MAP:
        return _COMPILER_INFERENCE_MAP[identity]

    # Heuristic: check if identity contains known model family names
    identity_lower = identity.lower()
    for key, compiler in _COMPILER_INFERENCE_MAP.items():
        if key in identity_lower:
            return compiler

    # Default
    return "default"


def _parse_old_driver_name(driver_name: str) -> tuple[str, str]:
    """Parse old flat driver name into (adapter, identity).

    Args:
        driver_name: Old-style driver name like "openai_compatible", "glm", "anthropic_compatible"

    Returns:
        (adapter, identity) tuple
    """
    if driver_name in _OLD_DRIVER_NAME_MAP:
        return _OLD_DRIVER_NAME_MAP[driver_name]

    # Unknown driver — treat as openai_compatible with the name as identity
    return ("openai_compatible", driver_name)


# ===== Config Loading =====

def _is_new_format(drivers_section: dict[str, Any]) -> bool:
    """Detect whether the drivers section uses new two-level format.

    New format: [drivers.protocol.identity] where values are dicts containing
    model/compiler/etc.

    Old format: [drivers.name] where values are dicts containing
    base_url/api_key/model but NOT nested identity dicts.
    """
    for key, value in drivers_section.items():
        if not isinstance(value, dict):
            continue
        # Check if any value in this dict is itself a dict (nested identity)
        for sub_key, sub_value in value.items():
            if isinstance(sub_value, dict):
                # It's a nested structure → new format
                # But we need to distinguish from old format's extra_headers (which is also a dict)
                # New format identity dicts contain model/compiler keys
                if isinstance(sub_value, dict) and (
                    "model" in sub_value
                    or "compiler" in sub_value
                    or "base_url" in sub_value
                    or "api_key_env" in sub_value
                ):
                    return True
    return False


def load_driver_configs(config: dict[str, Any]) -> dict[str, DriverConfig]:
    """Load all driver configurations from a TOML config dict.

    Automatically detects and supports both old (flat) and new (two-level) formats.

    Args:
        config: The full TOML config dict (from tomllib.load or similar)

    Returns:
        Dict mapping full_name (e.g., "openai_compatible.glm_5") to DriverConfig instances.
        For old format, the full_name may be just the adapter name (e.g., "openai_compatible").
    """
    drivers_section = config.get("drivers", {})
    if not drivers_section:
        return {}

    if _is_new_format(drivers_section):
        return _load_new_format(drivers_section)
    else:
        return _load_old_format(drivers_section)


def _load_new_format(drivers_section: dict[str, Any]) -> dict[str, DriverConfig]:
    """Load drivers using new protocol.identity two-level format.

    Example TOML:
        [drivers.openai_compatible.glm_5]
        base_url = "https://open.bigmodel.cn/api/paas/v4"
        api_key_env = "GLM_API_KEY"
        model = "glm-5"
        compiler = "glm_5"
    """
    drivers: dict[str, DriverConfig] = {}

    for protocol, identities in drivers_section.items():
        if not isinstance(identities, dict):
            continue
        for identity, settings in identities.items():
            if not isinstance(settings, dict):
                continue

            full_name = f"{protocol}.{identity}"
            api_key, api_key_env = resolve_api_key(settings, full_name=full_name)

            # Compiler: explicit > inferred from identity > default
            compiler = settings.get("compiler", "")
            if not compiler:
                compiler = infer_compiler(identity, adapter=protocol)

            drivers[full_name] = DriverConfig(
                adapter=protocol,
                identity=identity,
                base_url=settings.get("base_url", ""),
                api_key=api_key,
                model=settings.get("model", ""),
                compiler=compiler,
                timeout=float(settings.get("timeout", 300.0)),
                extra_headers=settings.get("extra_headers", {}),
                full_name=full_name,
                api_key_env=api_key_env,
                enable_fake_tools=settings.get("enable_fake_tools", False),
                # Extended fields for V4/M3/GLM-5
                compiler_variant=settings.get("compiler_variant", ""),
                max_context_tokens=int(settings.get("max_context_tokens", 0)),
                max_output_tokens=int(settings.get("max_output_tokens", 0)),
                thinking_mode=settings.get("thinking_mode", ""),
                cache_aware=settings.get("cache_aware", False),
                multimodal_enabled=settings.get("multimodal_enabled", False),
                desktop_enabled=settings.get("desktop_enabled", False),
                long_horizon_enabled=settings.get("long_horizon_enabled", False),
                msa_efficient=settings.get("msa_efficient", False),
            )

            logger.info(
                f"Loaded driver [{full_name}]: "
                f"adapter={protocol}, compiler={compiler}, "
                f"model={settings.get('model', '(empty)')}, "
                f"api_key={mask_api_key(api_key)}"
            )

    return drivers


def _load_old_format(drivers_section: dict[str, Any]) -> dict[str, DriverConfig]:
    """Load drivers using old flat format (backward compatibility).

    Example TOML:
        [drivers.openai_compatible]
        base_url = "https://integrate.api.nvidia.com/v1"
        api_key = "nvapi-..."
        model = "stepfun-ai/step-3.5-flash"

    Auto-infers compiler and adapter from the driver name.
    """
    drivers: dict[str, DriverConfig] = {}

    for driver_name, settings in drivers_section.items():
        if not isinstance(settings, dict):
            continue

        # Parse old driver name → (adapter, identity)
        adapter, identity = _parse_old_driver_name(driver_name)

        # Use driver_name as full_name for backward compat
        full_name = driver_name

        api_key, api_key_env = resolve_api_key(settings, full_name=full_name)

        # Auto-infer compiler
        compiler = settings.get("compiler", "")
        if not compiler:
            compiler = infer_compiler(identity, adapter=adapter)

        drivers[full_name] = DriverConfig(
            adapter=adapter,
            identity=identity,
            base_url=settings.get("base_url", ""),
            api_key=api_key,
            model=settings.get("model", ""),
            compiler=compiler,
            timeout=float(settings.get("timeout", 300.0)),
            extra_headers=settings.get("extra_headers", {}),
            full_name=full_name,
            api_key_env=api_key_env,
            enable_fake_tools=settings.get("enable_fake_tools", False),
            # Extended fields for V4/M3/GLM-5
            compiler_variant=settings.get("compiler_variant", ""),
            max_context_tokens=int(settings.get("max_context_tokens", 0)),
            max_output_tokens=int(settings.get("max_output_tokens", 0)),
            thinking_mode=settings.get("thinking_mode", ""),
            cache_aware=settings.get("cache_aware", False),
            multimodal_enabled=settings.get("multimodal_enabled", False),
            desktop_enabled=settings.get("desktop_enabled", False),
            long_horizon_enabled=settings.get("long_horizon_enabled", False),
            msa_efficient=settings.get("msa_efficient", False),
        )

        logger.info(
            f"Loaded driver [{full_name}] (old format, auto-inferred): "
            f"adapter={adapter}, identity={identity}, compiler={compiler}, "
            f"model={settings.get('model', '(empty)')}, "
            f"api_key={mask_api_key(api_key)}"
        )

    return drivers


# ===== Pipeline Config =====

def load_pipeline_config(config: dict[str, Any]) -> dict[str, str]:
    """Load execution pipeline driver assignments.

    Supports both old and new config formats:

    New format:
        [execution.pipeline]
        design_driver = "openai_compatible.glm_5"
        plan_driver = "openai_compatible.glm_5"
        execute_driver = "openai_compatible.glm_5"
        review_driver = "openai_compatible.glm_5"

    Old format:
        [execution.pipeline]
        design_model = "openai_compatible"
        plan_model = "openai_compatible"
        execute_model = "openai_compatible"
        review_model = "openai_compatible"

    Returns:
        Dict with keys: "design", "plan", "execute", "review"
        Values are driver full_names (e.g., "openai_compatible.glm_5")
    """
    # 修复 H12: 添加 isinstance 检查，避免 config["execution"] 非字典时 AttributeError
    execution_config = config.get("execution", {})
    if not isinstance(execution_config, dict):
        execution_config = {}
    pipeline = execution_config.get("pipeline", {})

    result: dict[str, str] = {}

    # New format keys (preferred)
    new_keys = {
        "design": "design_driver",
        "plan": "plan_driver",
        "execute": "execute_driver",
        "review": "review_driver",
    }

    # Old format keys (backward compat)
    old_keys = {
        "design": "design_model",
        "plan": "plan_model",
        "execute": "execute_model",
        "review": "review_model",
    }

    for stage, new_key in new_keys.items():
        old_key = old_keys[stage]
        # Prefer new key, fall back to old key
        driver_name = pipeline.get(new_key, "") or pipeline.get(old_key, "")
        result[stage] = driver_name

    return result


# ===== High-level Provider Creation =====

def create_provider_from_config(
    driver_config: DriverConfig,
    **extra_kwargs: Any,
) -> Any:
    """Create a ModelProvider from a DriverConfig.

    This is a convenience function that bridges the config layer with the
    provider factory.

    Args:
        driver_config: DriverConfig instance with all necessary settings
        **extra_kwargs: Additional kwargs passed to create_provider()
            (e.g., fallback, circuit_breaker)

    Returns:
        ModelProvider instance

    Raises:
        ValueError: If driver_config is not properly configured
    """
    from teragent import create_provider

    if not driver_config.is_configured:
        raise ValueError(
            f"DriverConfig '{driver_config.full_name}' is not fully configured: "
            f"adapter={driver_config.adapter!r}, model={driver_config.model!r}"
        )

    kwargs = driver_config.to_create_provider_kwargs()
    kwargs.update(extra_kwargs)

    return create_provider(**kwargs)


# ===== Full Config Loading =====

def load_full_config(config_path: str | None = None) -> dict[str, Any]:
    """Load the full agent.toml configuration.

    This is the main entry point for loading all configuration.

    Args:
        config_path: Path to agent.toml. If None, searches default locations.

    Returns:
        Dict with keys:
        - "drivers": dict[str, DriverConfig]
        - "pipeline": dict[str, str]
        - "raw": dict (the raw TOML dict)
    """
    if tomllib is None:
        raise ImportError(
            "TOML config loading requires 'tomli' on Python 3.10. "
            "Install it with: pip install tomli"
        )

    if config_path is None:
        # 搜索配置文件: CWD → 平台配置目录 → 项目源码根目录 → CWD默认
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # 构建搜索路径列表（按优先级排序）
        search_paths = [
            os.path.join(os.getcwd(), "agent.toml"),  # 当前工作目录
        ]

        # 平台特定配置目录
        if sys.platform == "win32":
            appdata = os.environ.get("APPDATA", "")
            if appdata:
                search_paths.append(os.path.join(appdata, "teragent", "agent.toml"))
        elif sys.platform == "darwin":
            search_paths.append(os.path.expanduser(
                "~/Library/Application Support/teragent/agent.toml"
            ))
        else:
            # Linux: XDG Base Directory
            xdg_config = os.environ.get(
                "XDG_CONFIG_HOME",
                os.path.expanduser("~/.config"),
            )
            search_paths.append(os.path.join(xdg_config, "teragent", "agent.toml"))

        # 项目源码根目录
        search_paths.append(os.path.join(_project_root, "agent.toml"))

        # 按顺序搜索，找到第一个存在的
        config_path = "agent.toml"  # 最终回退
        for candidate in search_paths:
            if os.path.exists(candidate):
                config_path = candidate
                break

    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found, using defaults.")
        return {
            "drivers": {},
            "pipeline": {
                "design": "",
                "plan": "",
                "execute": "",
                "review": "",
            },
            "raw": {},
        }

    try:
        with open(config_path, "rb") as f:
            raw_config = tomllib.load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {
            "drivers": {},
            "pipeline": {
                "design": "",
                "plan": "",
                "execute": "",
                "review": "",
            },
            "raw": {},
        }

    drivers = load_driver_configs(raw_config)
    pipeline = load_pipeline_config(raw_config)

    return {
        "drivers": drivers,
        "pipeline": pipeline,
        "raw": raw_config,
    }


def get_driver_config(
    drivers: dict[str, DriverConfig],
    driver_name: str,
) -> DriverConfig | None:
    """Look up a DriverConfig by name.

    Supports both full names ("openai_compatible.glm_5") and short names ("glm_5").

    Args:
        drivers: Dict of full_name → DriverConfig
        driver_name: Name to look up (full or short)

    Returns:
        DriverConfig if found, None otherwise
    """
    # Direct lookup by full name
    if driver_name in drivers:
        return drivers[driver_name]

    # Search by identity
    for full_name, cfg in drivers.items():
        if cfg.identity == driver_name:
            return cfg

    # Note: This block is unreachable — the dict key lookup above (line 520)
    # already handles this case. Kept for readability; could be removed.
    for full_name, cfg in drivers.items():
        if full_name == driver_name:
            return cfg

    return None


# ===== Typed Config Loading (Phase 5) =====

def load_typed_config(config_path: str | None = None) -> Any:
    """Load the full agent.toml as a typed TerAgentConfig.

    This is the Phase 5 main entry point for loading all configuration
    with full type safety. Returns a TerAgentConfig dataclass instead
    of a raw dict.

    NOTE: TerAgentConfig is imported inside this function to avoid a circular
    import with teragent.config.teragent_config (which imports load_driver_configs
    from this module at function scope). This is an intentional pattern — both
    sides use function-scoped imports to break the cycle. Do NOT hoist this
    import to module level.

    Args:
        config_path: Path to agent.toml. If None, searches default locations.

    Returns:
        TerAgentConfig instance with all sections typed
    """
    from teragent.config.teragent_config import TerAgentConfig
    return TerAgentConfig.from_toml(config_path)


# ===== Security Audit (Phase 9) =====

def audit_config_security(config_path: str | None = None) -> list[dict[str, str]]:
    """Run a security audit on the configuration file.

    Phase 9 convenience function that loads a config and checks for
    API key security issues.

    Args:
        config_path: Path to agent.toml. If None, searches default locations.

    Returns:
        List of security finding dicts with keys:
        - "severity": "critical" | "warning" | "info"
        - "message": Human-readable description
        - "location": Where the issue was found
        - "recommendation": How to fix the issue
    """
    from teragent.config.api_key_security import audit_config_security as _audit

    # Load raw config
    if tomllib is None:
        raise ImportError(
            "TOML config loading requires 'tomli' on Python 3.10. "
            "Install it with: pip install tomli"
        )

    if config_path is None:
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidate = os.path.join(_project_root, "agent.toml")
        config_path = candidate if os.path.exists(candidate) else "agent.toml"

    if not os.path.exists(config_path):
        return [{
            "severity": "info",
            "message": f"Config file not found at '{config_path}'",
            "location": "",
            "recommendation": "Create an agent.toml configuration file",
        }]

    try:
        with open(config_path, "rb") as f:
            raw_config = tomllib.load(f)
    except Exception as e:
        return [{
            "severity": "warning",
            "message": f"Failed to load config: {e}",
            "location": config_path,
            "recommendation": "Fix the config file syntax",
        }]

    findings = _audit(raw_config)
    return [
        {
            "severity": f.severity.value,
            "message": f.message,
            "location": f.location,
            "recommendation": f.recommendation,
        }
        for f in findings
    ]
