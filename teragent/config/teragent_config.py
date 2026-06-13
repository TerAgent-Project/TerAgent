"""teragent.config.teragent_config — Top-level typed configuration

Replaces all raw dict.get() access to agent.toml with typed dataclass access.
All configuration sections are accessible as typed attributes.

Usage::

    from teragent.config import TerAgentConfig

    # From agent.toml file
    config = TerAgentConfig.from_toml()

    # From raw TOML dict
    config = TerAgentConfig.from_dict(raw_config)

    # Access typed config
    config.max_parallel
    config.circuit_breaker.budget.max_session_tokens
    config.context_management.model_token_limit
    config.permission.mode
    config.drivers  # dict[str, DriverConfig]
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field

# tomllib is stdlib in 3.11+; fall back to tomli (pip) for 3.10
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

from teragent.config.circuit_breaker_config import CircuitBreakerConfig
from teragent.config.context_management_config import ContextManagementConfig
from teragent.config.coordination_config import CoordinationConfig
from teragent.config.driver_config import DriverConfig
from teragent.config.execution_pipeline_config import ExecutionPipelineConfig
from teragent.config.file_safety_config import FileSafetyConfig
from teragent.config.hooks_config import HooksConfig
from teragent.config.model_fallback_config import ModelFallbackConfig
from teragent.config.permission_config import PermissionConfig
from teragent.config.recovery_config import RecoveryConfig
from teragent.config.session_config import SessionConfig
from teragent.config.streaming_config import StreamingConfig
from teragent.config.tools_config import ToolsConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentdSectionConfig:
    """Agentd-specific configuration.

    Maps to [agentd] section in agent.toml.
    These parameters were previously accessed via config.get("agentd", {})
    but never had a corresponding TOML section — they were "ghost parameters".

    The [agentd] section in agent.toml makes these user-configurable.
    """
    max_tool_steps: int = 30
    repetition_window: int = 3
    repetition_similarity_threshold: float = 0.9
    cross_tool_repetition_limit: int = 3
    polling_detection_limit: int = 3
    periodic_check_interval: int = 5
    tool_execution_timeout: float = 120.0

    @classmethod
    def from_dict(cls, data: dict) -> AgentdSectionConfig:
        """Create AgentdSectionConfig from a raw TOML dict.

        Args:
            data: The [agentd] section dict from agent.toml

        Returns:
            Typed AgentdSectionConfig instance
        """
        return cls(
            max_tool_steps=data.get("max_tool_steps", 30),
            repetition_window=data.get("repetition_window", 3),
            repetition_similarity_threshold=data.get("repetition_similarity_threshold", 0.9),
            cross_tool_repetition_limit=data.get("cross_tool_repetition_limit", 3),
            polling_detection_limit=data.get("polling_detection_limit", 3),
            periodic_check_interval=data.get("periodic_check_interval", 5),
            tool_execution_timeout=data.get("tool_execution_timeout", 120.0),
        )


@dataclass(frozen=True)
class TerAgentConfig:
    """Top-level TerAgent configuration — all typed, no dict.get().

    Maps to the entire agent.toml file.
    Every configuration section is accessible as a typed dataclass attribute.

    Usage::

        config = TerAgentConfig.from_toml()
        print(config.max_parallel)
        print(config.circuit_breaker.budget.max_session_tokens)
        print(config.context_management.model_token_limit)
        print(config.drivers["openai_compatible.glm_5"].model)
    """

    # --- Root-level parameters ---
    max_parallel: int = 4
    max_retries: int = 2
    workspace_root: str = "."
    user_review: bool = False
    llm_review: bool = False
    max_pipeline_steps: int = 300
    max_repair_rounds: int = 3
    max_review_repair_rounds: int = 2
    max_tool_repair_attempts: int = 3
    max_consecutive_tool_failures: int = 5
    max_model_call_retries: int = 2

    # --- Sub-section configs ---
    agentd: AgentdSectionConfig = field(default_factory=AgentdSectionConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    context_management: ContextManagementConfig = field(default_factory=ContextManagementConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    file_safety: FileSafetyConfig = field(default_factory=FileSafetyConfig)
    permission: PermissionConfig = field(default_factory=PermissionConfig)
    hooks: HooksConfig = field(default_factory=HooksConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    coordination: CoordinationConfig = field(default_factory=CoordinationConfig)
    execution_pipeline: ExecutionPipelineConfig = field(default_factory=ExecutionPipelineConfig)
    model_fallback: ModelFallbackConfig = field(default_factory=ModelFallbackConfig)
    session: SessionConfig = field(default_factory=SessionConfig)

    # --- Driver configs ---
    drivers: dict[str, DriverConfig] = field(default_factory=dict)

    # --- Raw TOML dict (for backward compat) ---
    _raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: dict) -> TerAgentConfig:
        """Create TerAgentConfig from a raw TOML dict.

        Args:
            raw: The full TOML dict as loaded from agent.toml

        Returns:
            Typed TerAgentConfig instance
        """
        from teragent.config.loader import (
            load_driver_configs,  # noqa: E402 — delayed import to avoid circular dep with loader.load_typed_config
        )

        # Load driver configs
        drivers = load_driver_configs(raw)

        # Model fallback: [model.fallback]
        _model_val = raw.get("model")
        model_data = _model_val if isinstance(_model_val, dict) else {}
        fallback_data = model_data.get("fallback", {})

        instance = cls(
            # Root-level params
            max_parallel=raw.get("max_parallel", 4),
            max_retries=raw.get("max_retries", 2),
            workspace_root=raw.get("workspace_root", "."),
            user_review=raw.get("user_review", False),
            llm_review=raw.get("llm_review", False),
            max_pipeline_steps=raw.get("max_pipeline_steps", 300),
            max_repair_rounds=raw.get("max_repair_rounds", 3),
            max_review_repair_rounds=raw.get("max_review_repair_rounds", 2),
            max_tool_repair_attempts=raw.get("max_tool_repair_attempts", 3),
            max_consecutive_tool_failures=raw.get("max_consecutive_tool_failures", 5),
            max_model_call_retries=raw.get("max_model_call_retries", 2),

            # Sub-sections
            agentd=AgentdSectionConfig.from_dict(raw.get("agentd", {})),
            circuit_breaker=CircuitBreakerConfig.from_dict(raw.get("circuit_breaker", {})),
            context_management=ContextManagementConfig.from_dict(raw.get("context_management", {})),
            tools=ToolsConfig.from_dict(raw.get("tools", {})),
            file_safety=FileSafetyConfig.from_dict(raw.get("file_safety", {})),
            permission=PermissionConfig.from_dict(raw.get("permission", {})),
            hooks=HooksConfig.from_dict(raw.get("hooks", {})),
            recovery=RecoveryConfig.from_dict(raw.get("recovery", {})),
            streaming=StreamingConfig.from_dict(raw.get("streaming", {})),
            coordination=CoordinationConfig.from_dict(raw.get("coordination", {})),
            execution_pipeline=ExecutionPipelineConfig.from_dict(
                (raw.get("execution") or {}).get("pipeline", {}) if isinstance(raw.get("execution"), dict) else {}
            ),
            model_fallback=ModelFallbackConfig.from_dict(fallback_data),
            session=SessionConfig.from_dict(raw.get("session", {})),

            # Drivers
            drivers=drivers,

            # Keep raw dict for backward compat
            _raw=raw,
        )

        return instance

    @classmethod
    def from_toml(cls, config_path: str | None = None) -> TerAgentConfig:
        """Create TerAgentConfig from an agent.toml file.

        Args:
            config_path: Optional path to agent.toml.
                         If None, searches default locations.

        Returns:
            Typed TerAgentConfig instance
        """

        if tomllib is None:
            raise ImportError(
                "TOML config loading requires 'tomli' on Python 3.10. "
                "Install it with: pip install tomli"
            )

        if config_path is None:
            # Search in project root, then CWD
            _project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            candidate = os.path.join(_project_root, "agent.toml")
            config_path = candidate if os.path.exists(candidate) else "agent.toml"

        if not os.path.exists(config_path):
            logger.warning(f"Config file {config_path} not found, using defaults.")
            return cls()

        try:
            with open(config_path, "rb") as f:
                raw = tomllib.load(f)
            return cls.from_dict(raw)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return cls()

    def validate(self) -> list[str]:
        """Validate configuration values, return warning list.

        Includes both value validation and API Key security audit.

        Returns:
            List of warning strings (empty = all valid)
        """
        warnings: list[str] = []

        if self.max_parallel <= 0:
            warnings.append(f"max_parallel = {self.max_parallel} (should be > 0)")
        if self.max_retries < 0:
            warnings.append(f"max_retries = {self.max_retries} (should be >= 0)")
        if not (0.0 < self.agentd.repetition_similarity_threshold <= 1.0):
            warnings.append(
                f"repetition_similarity_threshold = {self.agentd.repetition_similarity_threshold} "
                f"(should be in (0, 1])"
            )
        if self.agentd.tool_execution_timeout <= 0:
            warnings.append(f"tool_execution_timeout = {self.agentd.tool_execution_timeout} (should be > 0)")

        # API Key security audit
        security_warnings = self._audit_api_key_security()
        warnings.extend(security_warnings)

        if warnings:
            logger.warning(f"TerAgentConfig validation warnings: {warnings}")

        return warnings

    def _audit_api_key_security(self) -> list[str]:
        """Audit API key security across all driver configs.

        Checks for plaintext keys, missing keys, and weak keys.

        Returns:
            List of security warning strings
        """
        from teragent.config.api_key_security import (
            SecuritySeverity,
            audit_config_security,
        )

        security_warnings: list[str] = []

        # Run audit on the raw config if available
        if self._raw:
            findings = audit_config_security(self._raw)
            for finding in findings:
                prefix = f"[{finding.severity.value.upper()}]"
                if finding.severity == SecuritySeverity.CRITICAL:
                    security_warnings.append(
                        f"SECURITY {prefix} {finding.message}"
                        + (f" (at: {finding.location})" if finding.location else "")
                        + (f" → {finding.recommendation}" if finding.recommendation else "")
                    )
                elif finding.severity == SecuritySeverity.WARNING:
                    security_warnings.append(
                        f"SECURITY {prefix} {finding.message}"
                    )

        # Also check DriverConfig objects for plaintext keys
        for full_name, driver_cfg in self.drivers.items():
            if driver_cfg.api_key and not driver_cfg.api_key_env:
                security_warnings.append(
                    f"SECURITY [CRITICAL] Driver '{full_name}' uses plaintext api_key "
                    f"without api_key_env. Add api_key_env for secure key management."
                )

        return security_warnings

    def to_dict(self) -> dict:
        """Convert back to a raw dict format for backward compatibility.

        This is useful for components that still expect the old dict format.

        Always constructs the dict from current field values to ensure
        consistency even after dataclasses.replace() or other mutations.

        Returns:
            Dict representation matching the old load_config() return format
        """
        # Always build from current field values to avoid returning stale _raw data
        # after dataclasses.replace() or other mutations.
        return {
            "max_parallel": self.max_parallel,
            "max_retries": self.max_retries,
            "workspace_root": self.workspace_root,
            "user_review": self.user_review,
            "llm_review": self.llm_review,
            "max_pipeline_steps": self.max_pipeline_steps,
            "max_repair_rounds": self.max_repair_rounds,
            "max_review_repair_rounds": self.max_review_repair_rounds,
            "max_tool_repair_attempts": self.max_tool_repair_attempts,
            "max_consecutive_tool_failures": self.max_consecutive_tool_failures,
            "max_model_call_retries": self.max_model_call_retries,
            "agentd": {
                "max_tool_steps": self.agentd.max_tool_steps,
                "repetition_window": self.agentd.repetition_window,
                "repetition_similarity_threshold": self.agentd.repetition_similarity_threshold,
                "cross_tool_repetition_limit": self.agentd.cross_tool_repetition_limit,
                "polling_detection_limit": self.agentd.polling_detection_limit,
                "periodic_check_interval": self.agentd.periodic_check_interval,
                "tool_execution_timeout": self.agentd.tool_execution_timeout,
            },
            "circuit_breaker": {
                "budget": {
                    "max_session_tokens": self.circuit_breaker.budget.max_session_tokens,
                    "warning_threshold": self.circuit_breaker.budget.warning_threshold,
                    "critical_threshold": self.circuit_breaker.budget.critical_threshold,
                    "cost_per_million_input": self.circuit_breaker.budget.cost_per_million_input,
                    "cost_per_million_output": self.circuit_breaker.budget.cost_per_million_output,
                    "enable_hard_limit": self.circuit_breaker.budget.enable_hard_limit,
                    "auto_downgrade_model": self.circuit_breaker.budget.auto_downgrade_model,
                },
                "failure_breaker": {
                    "max_consecutive": self.circuit_breaker.failure_breaker.max_consecutive,
                    "window_seconds": self.circuit_breaker.failure_breaker.window_seconds,
                },
                "latency_breaker": {
                    "warn_latency_ms": self.circuit_breaker.latency_breaker.warn_latency_ms,
                    "avg_window": self.circuit_breaker.latency_breaker.avg_window,
                },
                "progress_detector": {
                    "stall_threshold": self.circuit_breaker.progress_detector.stall_threshold,
                },
            },
            "context_management": {
                "model_token_limit": self.context_management.model_token_limit,
                "reserved_for_output": self.context_management.reserved_for_output,
                "reserved_for_system": self.context_management.reserved_for_system,
                "warn_threshold": self.context_management.warn_threshold,
                "compact_threshold": self.context_management.compact_threshold,
                "max_inline_tool_result": self.context_management.max_inline_tool_result,
                "max_compacts_per_session": self.context_management.max_compacts_per_session,
                "retain_count": self.context_management.retain_count,
            },
            "tools": {
                "max_concurrent": self.tools.max_concurrent,
                "output_persist_threshold": self.tools.output_persist_threshold,
                "output_dir": self.tools.output_dir,
                "preview_head_lines": self.tools.preview_head_lines,
                "preview_tail_lines": self.tools.preview_tail_lines,
                "max_persisted_outputs": self.tools.max_persisted_outputs,
            },
            "file_safety": {
                "enable_hash_validation": self.file_safety.enable_hash_validation,
                "max_history_per_file": self.file_safety.max_history_per_file,
            },
            "permission": {
                "mode": self.permission.mode,
                "default_effect": self.permission.default_effect,
                "enable_ai_classifier": self.permission.enable_ai_classifier,
                "ai_classifier_confidence": self.permission.ai_classifier_confidence,
                "rules": {
                    "allow": {"rules": self.permission.allow_rules},
                    "deny": {"rules": self.permission.deny_rules},
                },
            },
            "hooks": self.hooks.to_hook_manager_config(),
            "recovery": {
                "max_output_tokens_recovery": self.recovery.max_output_tokens_recovery,
                "max_context_overflow_recovery": self.recovery.max_context_overflow_recovery,
            },
            "streaming": {
                "mode": self.streaming.mode,
                "max_streaming_retries": self.streaming.max_streaming_retries,
            },
            "coordination": {
                "max_concurrent_sub_agents": self.coordination.max_concurrent_sub_agents,
                "max_sub_agent_steps": self.coordination.max_sub_agent_steps,
                "mailbox_max_size": self.coordination.mailbox_max_size,
                "async_notification_timeout": self.coordination.async_notification_timeout,
            },
            "execution": {
                "pipeline": {
                    "design_driver": self.execution_pipeline.design_driver,
                    "plan_driver": self.execution_pipeline.plan_driver,
                    "execute_driver": self.execution_pipeline.execute_driver,
                    "review_driver": self.execution_pipeline.review_driver,
                },
            },
            "model": {
                "fallback": {
                    "driver": self.model_fallback.driver,
                    "model": self.model_fallback.model,
                },
            },
            "session": {
                "session_dir": self.session.session_dir,
                "max_sessions": self.session.max_sessions,
                "auto_save": self.session.auto_save,
                "max_age_days": self.session.max_age_days,
            },
            "drivers": {
                name: {
                    "adapter": dc.adapter,
                    "identity": dc.identity,
                    "base_url": dc.base_url,
                    "api_key_env": dc.api_key_env,
                    "model": dc.model,
                    "compiler": dc.compiler,
                    "timeout": dc.timeout,
                    "extra_headers": dc.extra_headers,
                    "full_name": dc.full_name,
                    "enable_fake_tools": dc.enable_fake_tools,
                }
                for name, dc in self.drivers.items()
            },
        }
