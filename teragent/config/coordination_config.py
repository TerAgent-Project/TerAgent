"""teragent.config.coordination_config — Coordination typed configuration (Phase 5)

Replaces raw dict.get() access to [coordination] section in agent.toml.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoordinationConfig:
    """Multi-agent coordination configuration.

    Maps to [coordination] section in agent.toml.
    """
    max_concurrent_sub_agents: int = 5
    max_sub_agent_steps: int = 15
    mailbox_max_size: int = 100
    async_notification_timeout: float = 30.0

    @classmethod
    def from_dict(cls, data: dict) -> CoordinationConfig:
        """Create CoordinationConfig from a raw TOML dict.

        Args:
            data: The [coordination] section dict from agent.toml

        Returns:
            Typed CoordinationConfig instance
        """
        return cls(
            max_concurrent_sub_agents=data.get("max_concurrent_sub_agents", 5),
            max_sub_agent_steps=data.get("max_sub_agent_steps", 15),
            mailbox_max_size=data.get("mailbox_max_size", 100),
            async_notification_timeout=data.get("async_notification_timeout", 30.0),
        )
