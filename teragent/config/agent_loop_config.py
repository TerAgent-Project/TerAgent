# teragent/config/agent_loop_config.py
"""AgentLoopConfig — typed configuration for AgentLoop

Replaces the global `_config = load_config()` pattern in agent_loop/constants.py.
All configuration values are now dataclass fields with sensible defaults,
loadable from agent.toml via `AgentLoopConfig.from_toml()`.

Design principles:
  1. No global state — each AgentLoop instance gets its own config
  2. Backward compatible — module-level constants still work (deprecated)
  3. Immutable after creation — use `replace()` for modifications
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from teragent.config.teragent_config import TerAgentConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentLoopConfig:
    """AgentLoop configuration — all tunable parameters in one place

    Usage::

        # From defaults
        config = AgentLoopConfig()

        # From agent.toml
        config = AgentLoopConfig.from_toml()

        # Override specific values
        config = AgentLoopConfig(max_tool_steps=50)
        config = replace(config, max_tool_steps=50)

    Validation:
        Call `config.validate()` to check for invalid values.
        Validation warnings are logged at WARNING level.
    """

    # --- 步数与循环控制 ---
    max_tool_steps: int = 30
    repetition_window: int = 3
    repetition_similarity_threshold: float = 0.9
    cross_tool_repetition_limit: int = 3
    polling_detection_limit: int = 3
    periodic_check_interval: int = 5
    tool_execution_timeout: float = 120.0

    # --- 工具失败自动修复 ---
    max_tool_repair_attempts: int = 3
    max_consecutive_tool_failures: int = 5
    max_model_call_retries: int = 2

    # --- 错误恢复 ---
    max_output_tokens_recovery: int = 3
    max_context_overflow_recovery: int = 2

    # --- 流式模式 ---
    max_streaming_retries: int = 2

    # --- 上下文管理 ---
    conversation_window: int = 8

    # --- 各意图模式允许的工具 ---
    # Stored as dict[str, list[str]], keyed by IntentType value name
    intent_tools: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set default intent_tools if not provided"""
        if not self.intent_tools:
            # Cannot use self.x on frozen dataclass; use object.__setattr__
            from teragent.intent.classifier import IntentType
            object.__setattr__(self, "intent_tools", {
                IntentType.CHAT: [],
                IntentType.DEBUG: [
                    "read_file", "explore_codebase", "list_directory",
                    "execute_subtask", "submit_failure", "get_pipeline_status",
                    "send_message",
                ],
                IntentType.CREATE_PROJECT: [
                    "generate_design", "generate_plan", "create_project",
                    "read_file", "explore_codebase", "list_directory",
                    "submit_failure", "get_pipeline_status",
                    "spawn_agent", "send_message",
                ],
            })

    def validate(self) -> list[str]:
        """Validate configuration values, return warning list

        Checks:
          - Positive integer bounds
          - Threshold ranges

        Returns:
            List of warning strings (empty = all valid)
        """
        warnings: list[str] = []

        if self.max_tool_steps <= 0:
            warnings.append(f"max_tool_steps = {self.max_tool_steps} (should be > 0)")
        if self.max_tool_repair_attempts <= 0:
            warnings.append(f"max_tool_repair_attempts = {self.max_tool_repair_attempts} (should be > 0)")
        if self.max_consecutive_tool_failures <= 0:
            warnings.append(f"max_consecutive_tool_failures = {self.max_consecutive_tool_failures} (should be > 0)")
        if not (0.0 < self.repetition_similarity_threshold <= 1.0):
            warnings.append(
                f"repetition_similarity_threshold = {self.repetition_similarity_threshold} "
                f"(should be in (0, 1])"
            )
        if self.tool_execution_timeout <= 0:
            warnings.append(f"tool_execution_timeout = {self.tool_execution_timeout} (should be > 0)")
        if self.conversation_window <= 0:
            warnings.append(f"conversation_window = {self.conversation_window} (should be > 0)")

        if warnings:
            logger.warning(f"AgentLoopConfig validation warnings: {warnings}")

        return warnings

    @classmethod
    def from_toml(cls, config_path: str | None = None) -> AgentLoopConfig:
        """Create AgentLoopConfig from agent.toml

        Args:
            config_path: Optional path to agent.toml.
                         If None, uses default search strategy.

        Returns:
            AgentLoopConfig with values loaded from config file,
            falling back to defaults for missing fields.
        """
        from teragent.config.loader import load_full_config

        full_config = load_full_config(config_path)
        # load_full_config returns {"drivers": ..., "pipeline": ..., "raw": raw_toml_dict}
        # The raw TOML dict contains the original sections (agentd, recovery, etc.)
        raw = full_config.get("raw", {})
        _agentd_config = raw.get("agentd", {})
        _recovery_config = raw.get("recovery", {})
        _streaming_config = raw.get("streaming", {})
        _context_config = raw.get("context_management", {})

        instance = cls(
            max_tool_steps=_agentd_config.get("max_tool_steps", cls.max_tool_steps.default if hasattr(cls.max_tool_steps, 'default') else 30),
            repetition_window=_agentd_config.get("repetition_window", 3),
            repetition_similarity_threshold=_agentd_config.get("repetition_similarity_threshold", 0.9),
            cross_tool_repetition_limit=_agentd_config.get("cross_tool_repetition_limit", 3),
            polling_detection_limit=_agentd_config.get("polling_detection_limit", 3),
            periodic_check_interval=_agentd_config.get("periodic_check_interval", 5),
            tool_execution_timeout=_agentd_config.get("tool_execution_timeout", 120.0),
            max_tool_repair_attempts=_agentd_config.get("max_tool_repair_attempts", 3),
            max_consecutive_tool_failures=_agentd_config.get("max_consecutive_tool_failures", 5),
            max_model_call_retries=_agentd_config.get("max_model_call_retries", 2),
            max_output_tokens_recovery=_recovery_config.get("max_output_tokens_recovery", 3),
            max_context_overflow_recovery=_recovery_config.get("max_context_overflow_recovery", 2),
            max_streaming_retries=_streaming_config.get("max_streaming_retries", 2),
            conversation_window=_context_config.get("retain_count", 8),
        )

        # Validate and log
        validation_warnings = instance.validate()
        if validation_warnings:
            for w in validation_warnings:
                logger.warning(w)
        else:
            # Log non-default values
            defaults = cls()
            non_defaults = []
            for f_name in [
                "max_tool_steps", "max_tool_repair_attempts",
                "max_consecutive_tool_failures", "max_model_call_retries",
                "max_output_tokens_recovery", "max_context_overflow_recovery",
                "max_streaming_retries", "conversation_window",
            ]:
                val = getattr(instance, f_name)
                default_val = getattr(defaults, f_name)
                if val != default_val:
                    non_defaults.append(f"{f_name}={val}")
            if non_defaults:
                logger.info(f"AgentLoopConfig overridden from agent.toml: {', '.join(non_defaults)}")

        return instance

    @classmethod
    def from_typed_config(cls, config: TerAgentConfig) -> AgentLoopConfig:
        """Create AgentLoopConfig from a typed TerAgentConfig.

        This provides a direct conversion from the new typed configuration
        dataclass to AgentLoopConfig, without going through the intermediate
        dict representation.

        Args:
            config: A TerAgentConfig instance (from teragent.config)

        Returns:
            AgentLoopConfig with values populated from the typed config
        """
        return cls(
            max_tool_steps=config.agentd.max_tool_steps,
            repetition_window=config.agentd.repetition_window,
            repetition_similarity_threshold=config.agentd.repetition_similarity_threshold,
            cross_tool_repetition_limit=config.agentd.cross_tool_repetition_limit,
            polling_detection_limit=config.agentd.polling_detection_limit,
            periodic_check_interval=config.agentd.periodic_check_interval,
            tool_execution_timeout=config.agentd.tool_execution_timeout,
            max_tool_repair_attempts=config.max_tool_repair_attempts,
            max_consecutive_tool_failures=config.max_consecutive_tool_failures,
            max_model_call_retries=config.max_model_call_retries,
            max_output_tokens_recovery=config.recovery.max_output_tokens_recovery,
            max_context_overflow_recovery=config.recovery.max_context_overflow_recovery,
            max_streaming_retries=config.streaming.max_streaming_retries,
            conversation_window=config.context_management.retain_count,
        )
