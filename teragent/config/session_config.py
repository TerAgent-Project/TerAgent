"""teragent.config.session_config — Session persistence typed configuration (Phase 5)

Replaces raw dict.get() access to [session] section in agent.toml.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SessionConfig:
    """Session persistence configuration.

    Maps to [session] section in agent.toml.
    """
    session_dir: str = ".teragent/sessions"
    max_sessions: int = 50
    auto_save: bool = True
    max_age_days: int = 30

    @classmethod
    def from_dict(cls, data: dict) -> SessionConfig:
        """Create SessionConfig from a raw TOML dict.

        Args:
            data: The [session] section dict from agent.toml

        Returns:
            Typed SessionConfig instance
        """
        return cls(
            session_dir=data.get("session_dir", ".teragent/sessions"),
            max_sessions=data.get("max_sessions", 50),
            auto_save=data.get("auto_save", True),
            max_age_days=data.get("max_age_days", 30),
        )
