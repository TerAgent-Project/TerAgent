"""teragent.config.recovery_config — Recovery typed configuration (Phase 5)

Replaces raw dict.get() access to [recovery] section in agent.toml.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryConfig:
    """Error recovery configuration.

    Maps to [recovery] section in agent.toml.
    """
    max_output_tokens_recovery: int = 3
    max_context_overflow_recovery: int = 2

    @classmethod
    def from_dict(cls, data: dict) -> RecoveryConfig:
        """Create RecoveryConfig from a raw TOML dict.

        Args:
            data: The [recovery] section dict from agent.toml

        Returns:
            Typed RecoveryConfig instance
        """
        return cls(
            max_output_tokens_recovery=data.get("max_output_tokens_recovery", 3),
            max_context_overflow_recovery=data.get("max_context_overflow_recovery", 2),
        )
