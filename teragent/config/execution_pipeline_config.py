"""teragent.config.execution_pipeline_config — Execution pipeline typed configuration (Phase 5)

Replaces raw dict.get() access to [execution.pipeline] section in agent.toml.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionPipelineConfig:
    """Execution pipeline driver assignments.

    Maps to [execution.pipeline] section in agent.toml.
    Uses *_driver keys for driver assignment.
    """
    design_driver: str = ""
    plan_driver: str = ""
    execute_driver: str = ""
    review_driver: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> ExecutionPipelineConfig:
        """Create ExecutionPipelineConfig from a raw TOML dict.

        Args:
            data: The [execution.pipeline] section dict from agent.toml

        Returns:
            Typed ExecutionPipelineConfig instance
        """
        return cls(
            design_driver=data.get("design_driver", ""),
            plan_driver=data.get("plan_driver", ""),
            execute_driver=data.get("execute_driver", ""),
            review_driver=data.get("review_driver", ""),
        )
