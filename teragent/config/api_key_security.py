"""teragent.config.api_key_security — API Key security management (Phase 9)

Centralized API key management with:
- Secure storage with masking in logs/repr
- Config security auditing (detect plaintext keys)
- Key validation and strength checking
- .env file support with lazy loading
- Key masking and secure storage

Design principles:
1. API keys are NEVER logged or printed in plaintext
2. All key resolution goes through a single code path
3. Plaintext keys in config files are flagged with deprecation warnings
4. Environment variables and .env files are the recommended approach
5. Security audit can scan configs and report issues

Usage::

    from teragent.config.api_key_security import ApiKeyVault, audit_config_security

    # Create a vault and resolve keys
    vault = ApiKeyVault()
    key = vault.resolve("GLM_API_KEY")

    # Audit a config dict for security issues
    issues = audit_config_security(raw_config)
    for issue in issues:
        print(f"[{issue.severity}] {issue.message}")

    # Validate key strength
    vault.validate_key_strength("sk-...", min_length=20)
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ===== Security Issue Types =====

class SecuritySeverity(str, Enum):
    """Severity levels for security audit findings."""
    CRITICAL = "critical"   # Plaintext API key found in config
    WARNING = "warning"     # Deprecated pattern, missing key
    INFO = "info"           # Best practice suggestions


@dataclass
class SecurityFinding:
    """A single security audit finding.

    Attributes:
        severity: Finding severity level
        message: Human-readable description
        location: Where the issue was found (e.g., "drivers.openai_compatible.glm_5.api_key")
        recommendation: How to fix the issue
    """
    severity: SecuritySeverity
    message: str
    location: str = ""
    recommendation: str = ""

    def __str__(self) -> str:
        prefix = f"[{self.severity.value.upper()}]"
        parts = [prefix, self.message]
        if self.location:
            parts.append(f"(at: {self.location})")
        if self.recommendation:
            parts.append(f"→ {self.recommendation}")
        return " ".join(parts)


# ===== API Key Pattern Detection =====

# Patterns that look like API keys (common prefixes from major providers)
_API_KEY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Anthropic", re.compile(r"^sk-ant-[a-zA-Z0-9\-]{20,}$")),
    ("NVIDIA", re.compile(r"^nvapi-[a-zA-Z0-9_\-]{20,}$")),
    ("OpenRouter", re.compile(r"^sk-or-[a-zA-Z0-9\-]{20,}$")),
    ("OpenAI", re.compile(r"^sk-(?:proj-)?[a-zA-Z0-9\-]{20,}$")),
    ("DeepSeek", re.compile(r"^sk-[a-f0-9]{32,}$")),
    ("GLM/BigModel", re.compile(r"^[a-f0-9]{32,}\.[a-zA-Z0-9]{16,}$")),
    ("Generic Bearer", re.compile(r"^[A-Za-z0-9_\-]{32,}$")),
]

# Environment variable names that typically hold API keys
_KNOWN_KEY_ENV_VARS: set[str] = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GLM_API_KEY",
    "DEEPSEEK_API_KEY",
    "NVIDIA_API_KEY",
    "OPENROUTER_API_KEY",
    "HUGGINGFACE_API_KEY",
    "MISTRAL_API_KEY",
    "QWEN_API_KEY",
    "ONEAPI_API_KEY",
}


def detect_api_key_provider(value: str) -> str | None:
    """Detect which provider an API key belongs to based on its format.

    Args:
        value: The API key string to check

    Returns:
        Provider name if detected, None otherwise
    """
    if not value or len(value) < 16:
        return None
    for provider, pattern in _API_KEY_PATTERNS:
        if pattern.match(value):
            return provider
    return None


def mask_api_key(key: str, visible_chars: int = 4) -> str:
    """Mask an API key for safe display in logs and repr.

    Args:
        key: The API key to mask
        visible_chars: Number of characters to show at the end

    Returns:
        Masked key string (e.g., "sk-...abc1")

    Examples:
        >>> mask_api_key("sk-1234567890abcdef")
        'sk-...cdef'
        >>> mask_api_key("")
        '(empty)'
        >>> mask_api_key("short")
        '***'
    """
    if not key:
        return "(empty)"
    if len(key) <= visible_chars + 3:
        return "***"
    prefix = key[:3] if key.startswith("sk-") or key.startswith("nv") else "***"
    suffix = key[-visible_chars:]
    return f"{prefix}...{suffix}"


# ===== ApiKeyVault =====

_dotenv_loaded = False


def _ensure_dotenv_loaded() -> None:
    """Load .env file once (lazy, best-effort).

    Tries to load the .env file from the current working directory.
    If python-dotenv is not installed, silently skips.
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    try:
        from dotenv import load_dotenv
        # Try loading .env from multiple locations (按优先级排序)
        # 1. 当前工作目录 (最高优先级)
        load_dotenv()
        # 2. 用户主目录
        home_env = os.path.join(os.path.expanduser("~"), ".env")
        if os.path.exists(home_env):
            load_dotenv(home_env, override=False)
        # 3. 项目源码根目录 (where agent.toml typically lives)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )))
        env_path = os.path.join(project_root, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=False)
    except ImportError:
        logger.debug("python-dotenv not installed, .env file support disabled")


@dataclass
class ResolvedKey:
    """Result of an API key resolution attempt.

    Attributes:
        key: The resolved API key (empty string if not found)
        source: Where the key came from
        env_var: The environment variable name used (if applicable)
        provider: Detected provider name (if identifiable)
        is_plaintext: Whether the key came from a plaintext config value
    """
    key: str = ""
    source: str = "none"  # "env" | "dotenv" | "plaintext" | "direct" | "none"
    env_var: str = ""
    provider: str | None = None
    is_plaintext: bool = False

    @property
    def found(self) -> bool:
        """Whether a key was successfully resolved."""
        return bool(self.key)

    @property
    def masked(self) -> str:
        """Masked representation for safe display."""
        return mask_api_key(self.key)

    def __repr__(self) -> str:
        return (
            f"ResolvedKey(source={self.source!r}, "
            f"env_var={self.env_var!r}, "
            f"key={self.masked!r}, "
            f"provider={self.provider!r}, "
            f"is_plaintext={self.is_plaintext})"
        )


class ApiKeyVault:
    """Centralized API key storage and resolution.

    The vault provides a single code path for all API key resolution,
    ensuring consistent security practices across the library.

    Resolution priority (resolve_from_settings() only):
        1. Environment variable (os.getenv)
        2. .env file (python-dotenv)
        3. Plaintext in config (DEPRECATED, emits warning)

    Note: resolve() only uses priorities 1 and 2 (env var + .env).
    It does NOT fall back to plaintext config.

    Features:
        - Keys are stored securely and never exposed in repr/str
        - Plaintext keys trigger deprecation warnings
        - Security audit capabilities
        - Key validation

    Usage::

        vault = ApiKeyVault()

        # Resolve a key from environment
        resolved = vault.resolve("GLM_API_KEY")
        if resolved.found:
            print(f"Key found: {resolved.masked}")

        # Resolve from config settings
        resolved = vault.resolve_from_settings(
            settings={"api_key_env": "GLM_API_KEY", "api_key": "..."},
            full_name="openai_compatible.glm_5"
        )

        # Validate key strength
        vault.validate_key_strength(resolved.key, min_length=20)

        # Get all resolved keys (masked)
        for name, resolved in vault.all_resolved():
            print(f"{name}: {resolved.masked}")
    """

    def __init__(self, strict_mode: bool = False) -> None:
        """Initialize the API key vault.

        Args:
            strict_mode: If True, raise errors instead of warnings for
                         plaintext keys. Useful in CI/CD pipelines.
        """
        self._resolved: dict[str, ResolvedKey] = {}
        self._strict_mode = strict_mode

    def resolve(self, env_var: str) -> ResolvedKey:
        """Resolve an API key from an environment variable name.

        Follows the resolution priority:
            1. Direct environment variable
            2. .env file (lazy loaded)

        Args:
            env_var: Environment variable name (e.g., "GLM_API_KEY")

        Returns:
            ResolvedKey with the resolution result
        """
        if not env_var:
            return ResolvedKey(source="none", env_var=env_var)

        # Check cache
        if env_var in self._resolved:
            return self._resolved[env_var]

        # 1. Try direct environment variable
        key = os.getenv(env_var, "")
        if key:
            provider = detect_api_key_provider(key)
            result = ResolvedKey(
                key=key,
                source="env",
                env_var=env_var,
                provider=provider,
            )
            self._resolved[env_var] = result
            return result

        # 2. Try .env file
        _ensure_dotenv_loaded()
        key = os.getenv(env_var, "")
        if key:
            provider = detect_api_key_provider(key)
            result = ResolvedKey(
                key=key,
                source="dotenv",
                env_var=env_var,
                provider=provider,
            )
            self._resolved[env_var] = result
            return result

        # Not found
        result = ResolvedKey(source="none", env_var=env_var)
        self._resolved[env_var] = result
        logger.debug(
            f"API key not found for env var: {env_var}. "
            f"Set the environment variable or create a .env file."
        )
        return result

    def resolve_from_settings(
        self,
        settings: dict[str, Any],
        full_name: str = "",
    ) -> ResolvedKey:
        """Resolve an API key from config settings dict.

        This is the primary method used by the config loader.
        It checks api_key_env first, then falls back to plaintext api_key
        with a deprecation warning.

        Args:
            settings: Driver settings dict from TOML config
            full_name: Fully qualified driver name (for logging)

        Returns:
            ResolvedKey with the resolution result
        """
        api_key_env = settings.get("api_key_env", "")

        # 1. Try environment variable (if api_key_env is specified)
        if api_key_env:
            resolved = self.resolve(api_key_env)
            if resolved.found:
                return resolved

        # 2. Fallback to plaintext api_key (DEPRECATED)
        api_key = settings.get("api_key", "")
        if api_key:
            provider = detect_api_key_provider(api_key)
            result = ResolvedKey(
                key=api_key,
                source="plaintext",
                env_var=api_key_env,
                provider=provider,
                is_plaintext=True,
            )

            # Emit deprecation warning
            msg = (
                f"DEPRECATED: api_key in plain text for driver '{full_name}'. "
                f"Use api_key_env instead. "
                f"Plain text api_key will be removed in v2.0."
            )
            if self._strict_mode:
                raise SecurityError(msg)
            else:
                warnings.warn(msg, DeprecationWarning, stacklevel=3)
                logger.warning(msg)

            self._resolved[full_name or "plaintext"] = result
            return result

        # No key found
        if api_key_env:
            logger.warning(
                f"API key not found for driver '{full_name}' "
                f"(env: {api_key_env}). "
                f"Set the environment variable or create a .env file."
            )
        else:
            logger.debug(
                f"No API key configured for driver '{full_name}'. "
                f"Specify api_key_env or api_key in the config."
            )

        return ResolvedKey(source="none", env_var=api_key_env)

    def store_direct(self, key: str, name: str = "direct") -> ResolvedKey:
        """Store an API key provided directly (e.g., from create_provider()).

        Direct keys are not flagged as plaintext (they come from code, not config).

        Args:
            key: The API key string
            name: Identifier for the key (for logging/debugging)

        Returns:
            ResolvedKey with the stored result
        """
        if not key:
            return ResolvedKey(source="none")

        provider = detect_api_key_provider(key)
        result = ResolvedKey(
            key=key,
            source="direct",
            provider=provider,
        )
        self._resolved[name] = result
        return result

    def validate_key_strength(
        self,
        key: str,
        min_length: int = 20,
    ) -> list[str]:
        """Validate the strength of an API key.

        Args:
            key: The API key to validate
            min_length: Minimum acceptable key length

        Returns:
            List of warning strings (empty = key is acceptable)
        """
        warnings_list: list[str] = []

        if not key:
            warnings_list.append("API key is empty")
            return warnings_list

        if len(key) < min_length:
            warnings_list.append(
                f"API key is too short ({len(key)} chars, minimum {min_length}). "
                f"Short keys may be insecure."
            )

        # Check for common weak patterns
        if key in ("test", "demo", "example", "placeholder", "your_key_here", "xxx"):
            warnings_list.append(
                f"API key appears to be a placeholder: '{mask_api_key(key)}'"
            )

        # Check for all same characters
        if len(set(key)) <= 3:
            warnings_list.append(
                "API key has very low entropy (few unique characters)"
            )

        return warnings_list

    def all_resolved(self) -> list[tuple[str, ResolvedKey]]:
        """Get all resolved keys (for auditing).

        Returns:
            List of (name, ResolvedKey) tuples. Keys are masked in repr.
        """
        return list(self._resolved.items())

    def audit(self) -> list[SecurityFinding]:
        """Audit all resolved keys for security issues.

        Returns:
            List of SecurityFinding objects
        """
        findings: list[SecurityFinding] = []

        for name, resolved in self._resolved.items():
            if resolved.is_plaintext:
                findings.append(SecurityFinding(
                    severity=SecuritySeverity.CRITICAL,
                    message=f"Plaintext API key found for '{name}'",
                    location=name,
                    recommendation="Use api_key_env to reference an environment variable instead",
                ))

            if resolved.found:
                strength_warnings = self.validate_key_strength(resolved.key)
                for w in strength_warnings:
                    findings.append(SecurityFinding(
                        severity=SecuritySeverity.WARNING,
                        message=w,
                        location=name,
                    ))

            if not resolved.found and resolved.env_var:
                findings.append(SecurityFinding(
                    severity=SecuritySeverity.WARNING,
                    message=f"API key not found for env var '{resolved.env_var}'",
                    location=name,
                    recommendation=f"Set {resolved.env_var} in your environment or .env file",
                ))

        return findings

    def clear(self) -> None:
        """Clear all stored keys (for testing or key rotation)."""
        self._resolved.clear()

    def __repr__(self) -> str:
        count = len(self._resolved)
        plaintext_count = sum(1 for r in self._resolved.values() if r.is_plaintext)
        return (
            f"ApiKeyVault(keys={count}, "
            f"plaintext={plaintext_count}, "
            f"strict_mode={self._strict_mode})"
        )


# ===== Config Security Audit =====

def audit_config_security(config: dict[str, Any]) -> list[SecurityFinding]:
    """Audit a TOML config dict for API key security issues.

    Scans the config for:
    - Plaintext API keys (api_key field with non-empty value)
    - Missing api_key_env for drivers
    - Weak or placeholder keys
    - Keys that match known API key formats

    Args:
        config: Raw TOML config dict (as loaded by tomllib)

    Returns:
        List of SecurityFinding objects, sorted by severity

    Examples:
        >>> issues = audit_config_security(raw_config)
        >>> critical = [i for i in issues if i.severity == SecuritySeverity.CRITICAL]
        >>> if critical:
        ...     print("SECURITY ISSUES FOUND:")
        ...     for issue in critical:
        ...         print(f"  {issue}")
    """
    findings: list[SecurityFinding] = []

    drivers_section = config.get("drivers", {})
    if not drivers_section:
        return findings

    # Handle both new format and old format
    for protocol, identities_or_settings in drivers_section.items():
        if not isinstance(identities_or_settings, dict):
            continue

        # Detect if this is new format (nested) or old format (flat)
        is_nested = False
        for _key, value in identities_or_settings.items():
            if isinstance(value, dict) and (
                "model" in value or "base_url" in value or "api_key" in value or "api_key_env" in value
            ):
                is_nested = True
                break

        if is_nested:
            # New format: [drivers.protocol.identity]
            for identity, settings in identities_or_settings.items():
                if not isinstance(settings, dict):
                    continue
                full_name = f"{protocol}.{identity}"
                _audit_driver_settings(settings, full_name, findings)
        else:
            # Old format: [drivers.name]
            _audit_driver_settings(identities_or_settings, protocol, findings)

    # Sort by severity (CRITICAL first)
    severity_order = {
        SecuritySeverity.CRITICAL: 0,
        SecuritySeverity.WARNING: 1,
        SecuritySeverity.INFO: 2,
    }
    findings.sort(key=lambda f: severity_order.get(f.severity, 99))

    return findings


def _audit_driver_settings(
    settings: dict[str, Any],
    full_name: str,
    findings: list[SecurityFinding],
) -> None:
    """Audit a single driver's settings for security issues.

    Args:
        settings: The driver's settings dict
        full_name: Fully qualified driver name
        findings: List to append findings to
    """
    api_key = settings.get("api_key", "")
    api_key_env = settings.get("api_key_env", "")

    # Check for plaintext API key
    if api_key:
        provider = detect_api_key_provider(api_key)
        provider_info = f" (detected: {provider})" if provider else ""
        findings.append(SecurityFinding(
            severity=SecuritySeverity.CRITICAL,
            message=f"Plaintext API key found for driver '{full_name}'{provider_info}",
            location=f"drivers.{full_name}.api_key",
            recommendation=(
                f"Remove api_key from config and use api_key_env instead. "
                f"Add: api_key_env = \"{api_key_env or 'YOUR_KEY_ENV_VAR'}\" "
                f"to [drivers.{full_name}] in agent.toml"
            ),
        ))

        # Check for weak keys
        if len(api_key) < 20:
            findings.append(SecurityFinding(
                severity=SecuritySeverity.WARNING,
                message=f"API key for '{full_name}' appears too short ({len(api_key)} chars)",
                location=f"drivers.{full_name}.api_key",
            ))

        # Check for placeholder keys
        if api_key in ("test", "demo", "example", "placeholder", "your_key_here", "xxx"):
            findings.append(SecurityFinding(
                severity=SecuritySeverity.WARNING,
                message=f"API key for '{full_name}' appears to be a placeholder",
                location=f"drivers.{full_name}.api_key",
            ))

    # Check for missing api_key_env (if no api_key either)
    if not api_key and not api_key_env:
        findings.append(SecurityFinding(
            severity=SecuritySeverity.INFO,
            message=f"Driver '{full_name}' has no API key configured",
            location=f"drivers.{full_name}",
            recommendation=(
                f"Add api_key_env = \"YOUR_KEY_ENV_VAR\" to [drivers.{full_name}] "
                f"in agent.toml"
            ),
        ))

    # If api_key_env is set but api_key is also set, recommend removing api_key
    if api_key and api_key_env:
        findings.append(SecurityFinding(
            severity=SecuritySeverity.WARNING,
            message=(
                f"Driver '{full_name}' has both api_key and api_key_env set. "
                f"The api_key_env takes precedence over plaintext api_key. "
                f"Remove api_key and use api_key_env only."
            ),
            location=f"drivers.{full_name}",
        ))


def audit_env_file(env_path: str = ".env") -> list[SecurityFinding]:
    """Audit a .env file for security best practices.

    Checks for:
    - Empty values
    - Placeholder values
    - Keys that don't follow naming conventions

    Args:
        env_path: Path to the .env file

    Returns:
        List of SecurityFinding objects
    """
    findings: list[SecurityFinding] = []

    if not os.path.exists(env_path):
        findings.append(SecurityFinding(
            severity=SecuritySeverity.INFO,
            message=f".env file not found at '{env_path}'",
            recommendation="Create a .env file with your API keys (see .env.example)",
        ))
        return findings

    try:
        with open(env_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if "=" not in line:
                    findings.append(SecurityFinding(
                        severity=SecuritySeverity.WARNING,
                        message=f"Invalid .env line {line_num}: missing '='",
                        location=f".env:{line_num}",
                    ))
                    continue

                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                if not value or value in ("your_key_here", "xxx", "placeholder", "CHANGEME"):
                    findings.append(SecurityFinding(
                        severity=SecuritySeverity.WARNING,
                        message=f".env key '{key}' has a placeholder or empty value",
                        location=f".env:{line_num}",
                        recommendation=f"Set a real API key for {key}",
                    ))

    except Exception as e:
        findings.append(SecurityFinding(
            severity=SecuritySeverity.WARNING,
            message=f"Failed to read .env file: {e}",
            location=env_path,
        ))

    return findings


# ===== Security Error =====

class SecurityError(Exception):
    """Raised when a security policy is violated in strict mode.

    For example, when a plaintext API key is found in config
    while running in strict mode (CI/CD pipelines).
    """
    pass


# ===== Module-level convenience functions =====

# Global vault instance (lazy-initialized)
_global_vault: ApiKeyVault | None = None


def get_vault() -> ApiKeyVault:
    """Get the global ApiKeyVault instance.

    Returns:
        The singleton ApiKeyVault instance
    """
    global _global_vault
    if _global_vault is None:
        _global_vault = ApiKeyVault()
    return _global_vault


def reset_vault() -> None:
    """Reset the global vault (for testing)."""
    global _global_vault, _dotenv_loaded
    _global_vault = None
    _dotenv_loaded = False
