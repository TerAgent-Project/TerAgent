"""teragent.config.streaming_config — Streaming typed configuration (Phase 5)

Replaces raw dict.get() access to [streaming] section in agent.toml.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamingConfig:
    """Streaming execution configuration.

    Maps to [streaming] section in agent.toml.
    """
    mode: str = "auto"
    max_streaming_retries: int = 2

    @classmethod
    def from_dict(cls, data: dict) -> StreamingConfig:
        """Create StreamingConfig from a raw TOML dict.

        Args:
            data: The [streaming] section dict from agent.toml

        Returns:
            Typed StreamingConfig instance
        """
        return cls(
            mode=data.get("mode", "auto"),
            max_streaming_retries=data.get("max_streaming_retries", 2),
        )
