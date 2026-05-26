"""teragent.config.permission_config — Permission typed configuration (Phase 5)

Replaces raw dict.get() access to [permission] section in agent.toml.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PermissionConfig:
    """Permission system configuration.

    Maps to [permission] section in agent.toml.
    """
    mode: str = "plan"
    default_effect: str = "deny"
    enable_ai_classifier: bool = False
    ai_classifier_confidence: float = 0.8
    allow_rules: list[str] = field(default_factory=list)
    deny_rules: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> PermissionConfig:
        """Create PermissionConfig from a raw TOML dict.

        Args:
            data: The [permission] section dict from agent.toml

        Returns:
            Typed PermissionConfig instance
        """
        rules_config = data.get("rules", {})

        # Support both flat and nested rule formats:
        # Flat: rules = { allow = [...], deny = [...] }
        # Nested: rules = { allow = { rules = [...] }, deny = { rules = [...] } }
        raw_allow = rules_config.get("allow", [])
        raw_deny = rules_config.get("deny", [])

        # Handle nested format: { allow: { rules: [...] } }
        if isinstance(raw_allow, dict):
            raw_allow = raw_allow.get("rules", [])
        if isinstance(raw_deny, dict):
            raw_deny = raw_deny.get("rules", [])

        allow_rules = raw_allow if isinstance(raw_allow, list) else []
        deny_rules = raw_deny if isinstance(raw_deny, list) else []

        return cls(
            mode=data.get("mode", "plan"),
            default_effect=data.get("default_effect", "deny"),
            enable_ai_classifier=data.get("enable_ai_classifier", False),
            ai_classifier_confidence=data.get("ai_classifier_confidence", 0.8),
            allow_rules=allow_rules,
            deny_rules=deny_rules,
        )
