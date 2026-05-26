"""teragent.config.tools_config — Tools typed configuration (Phase 5)

Replaces raw dict.get() access to [tools] section in agent.toml.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolsConfig:
    """Tool execution and output persistence configuration.

    Maps to [tools] section in agent.toml.
    """
    max_concurrent: int = 10
    output_persist_threshold: int = 3000
    output_dir: str = ".teragent/tool_outputs"
    preview_head_lines: int = 20
    preview_tail_lines: int = 5
    max_persisted_outputs: int = 100

    @classmethod
    def from_dict(cls, data: dict) -> ToolsConfig:
        """Create ToolsConfig from a raw TOML dict.

        Args:
            data: The [tools] section dict from agent.toml

        Returns:
            Typed ToolsConfig instance
        """
        return cls(
            max_concurrent=data.get("max_concurrent", 10),
            output_persist_threshold=data.get("output_persist_threshold", 3000),
            output_dir=data.get("output_dir", ".teragent/tool_outputs"),
            preview_head_lines=data.get("preview_head_lines", 20),
            preview_tail_lines=data.get("preview_tail_lines", 5),
            max_persisted_outputs=data.get("max_persisted_outputs", 100),
        )
