"""teragent.config.context_management_config — Context management typed configuration (Phase 5)

Replaces raw dict.get() access to [context_management] section in agent.toml.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextManagementConfig:
    """Context window and compaction configuration.

    Maps to [context_management] section in agent.toml.
    """
    model_token_limit: int = 128_000
    reserved_for_output: int = 4_096
    reserved_for_system: int = 2_048
    warn_threshold: float = 0.75
    compact_threshold: float = 0.85
    max_inline_tool_result: int = 2000
    max_compacts_per_session: int = 5
    retain_count: int = 8

    @classmethod
    def from_dict(cls, data: dict) -> ContextManagementConfig:
        """Create ContextManagementConfig from a raw TOML dict.

        Args:
            data: The [context_management] section dict from agent.toml

        Returns:
            Typed ContextManagementConfig instance
        """
        return cls(
            model_token_limit=data.get("model_token_limit", 128_000),
            reserved_for_output=data.get("reserved_for_output", 4_096),
            reserved_for_system=data.get("reserved_for_system", 2_048),
            warn_threshold=data.get("warn_threshold", 0.75),
            compact_threshold=data.get("compact_threshold", 0.85),
            max_inline_tool_result=data.get("max_inline_tool_result", 2000),
            max_compacts_per_session=data.get("max_compacts_per_session", 5),
            retain_count=data.get("retain_count", 8),
        )
