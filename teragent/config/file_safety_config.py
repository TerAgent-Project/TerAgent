"""teragent.config.file_safety_config — File safety typed configuration (Phase 5)

Replaces raw dict.get() access to [file_safety] section in agent.toml.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FileSafetyConfig:
    """File safety and state tracking configuration.

    Maps to [file_safety] section in agent.toml.
    """
    enable_hash_validation: bool = True
    max_history_per_file: int = 50

    @classmethod
    def from_dict(cls, data: dict) -> FileSafetyConfig:
        """Create FileSafetyConfig from a raw TOML dict.

        Args:
            data: The [file_safety] section dict from agent.toml

        Returns:
            Typed FileSafetyConfig instance
        """
        return cls(
            enable_hash_validation=data.get("enable_hash_validation", True),
            max_history_per_file=data.get("max_history_per_file", 50),
        )
