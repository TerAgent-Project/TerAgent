"""teragent.config.hooks_config — Hooks typed configuration (Phase 5)

Replaces raw dict.get() access to [hooks] section in agent.toml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HooksConfig:
    """Hook system configuration.

    Maps to [hooks] section in agent.toml.
    """
    enable_audit_hook: bool = True
    enable_dangerous_command_hook: bool = True
    pre_tool_use: list[dict[str, Any]] = field(default_factory=list)
    post_tool_use: list[dict[str, Any]] = field(default_factory=list)
    pre_model_call: list[dict[str, Any]] = field(default_factory=list)
    post_model_call: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> HooksConfig:
        """Create HooksConfig from a raw TOML dict.

        Args:
            data: The [hooks] section dict from agent.toml

        Returns:
            Typed HooksConfig instance
        """
        return cls(
            enable_audit_hook=data.get("enable_audit_hook", True),
            enable_dangerous_command_hook=data.get("enable_dangerous_command_hook", True),
            pre_tool_use=data.get("pre_tool_use", []),
            post_tool_use=data.get("post_tool_use", []),
            pre_model_call=data.get("pre_model_call", []),
            post_model_call=data.get("post_model_call", []),
        )

    def to_hook_manager_config(self) -> dict:
        """Convert to the dict format expected by HookManager.load_from_config().

        This provides backward compatibility with the existing HookManager API.
        """
        return {
            "enable_audit_hook": self.enable_audit_hook,
            "enable_dangerous_command_hook": self.enable_dangerous_command_hook,
            "pre_tool_use": self.pre_tool_use,
            "post_tool_use": self.post_tool_use,
            "pre_model_call": self.pre_model_call,
            "post_model_call": self.post_model_call,
        }
