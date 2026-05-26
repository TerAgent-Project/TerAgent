"""teragent.config.model_fallback_config — Model fallback typed configuration (Phase 5)

Replaces raw dict.get() access to [model.fallback] section in agent.toml.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFallbackConfig:
    """Model fallback/degradation configuration.

    Maps to [model.fallback] section in agent.toml.
    """
    driver: str = ""
    model: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> ModelFallbackConfig:
        """Create ModelFallbackConfig from a raw TOML dict.

        Args:
            data: The [model.fallback] section dict from agent.toml

        Returns:
            Typed ModelFallbackConfig instance
        """
        return cls(
            driver=data.get("driver", ""),
            model=data.get("model", ""),
        )
