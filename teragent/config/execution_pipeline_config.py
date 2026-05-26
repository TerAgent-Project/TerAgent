"""teragent.config.execution_pipeline_config — Execution pipeline typed configuration (Phase 5)

Replaces raw dict.get() access to [execution.pipeline] section in agent.toml.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionPipelineConfig:
    """Execution pipeline driver assignments.

    Maps to [execution.pipeline] section in agent.toml.
    Supports both new format (*_driver) and old format (*_model).
    """
    design_driver: str = ""
    plan_driver: str = ""
    execute_driver: str = ""
    review_driver: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> ExecutionPipelineConfig:
        """Create ExecutionPipelineConfig from a raw TOML dict.

        Supports both new format (*_driver) and old format (*_model) keys.

        Args:
            data: The [execution.pipeline] section dict from agent.toml

        Returns:
            Typed ExecutionPipelineConfig instance
        """
        # New format keys (preferred)
        design = data.get("design_driver", "") or data.get("design_model", "")
        plan = data.get("plan_driver", "") or data.get("plan_model", "")
        execute = data.get("execute_driver", "") or data.get("execute_model", "")
        review = data.get("review_driver", "") or data.get("review_model", "")

        return cls(
            design_driver=design,
            plan_driver=plan,
            execute_driver=execute,
            review_driver=review,
        )
