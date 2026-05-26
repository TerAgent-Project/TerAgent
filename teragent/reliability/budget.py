"""StepBudget — 会话级步数预算

Part of the teragent library.
"""

import logging
from dataclasses import dataclass

# Module-level default — replaces the former MAX_TOOL_STEPS import.
# The value 30 matches the original AgentLoopConfig.max_tool_steps default.
DEFAULT_MAX_STEPS: int = 30

logger = logging.getLogger(__name__)


@dataclass
class StepBudget:
    """会话级步数预算

    追踪 AgentLoop 的工具调用步数，防止无限循环。
    当步数耗尽时暂停，等待用户确认后可追加步数。
    """

    max_steps: int = DEFAULT_MAX_STEPS
    current_steps: int = 0
    _paused: bool = False
    _initial_max_steps: int = 0

    def __post_init__(self):
        self._initial_max_steps = self.max_steps

    def consume(self) -> bool:
        """消耗一步预算

        Returns:
            True 表示仍有预算，False 表示耗尽
        """
        if self._paused:
            return False
        if self.current_steps >= self.max_steps:
            self._paused = True
            logger.warning(
                f"AgentLoop step budget exhausted: {self.current_steps}/{self.max_steps}"
            )
            return False
        self.current_steps += 1
        return True

    def resume(self, extra_steps: int = 10) -> None:
        """用户确认后，追加额外步数"""
        self.max_steps += extra_steps
        self._paused = False
        logger.info(
            f"AgentLoop step budget resumed: {self.current_steps}/{self.max_steps} "
            f"(+{extra_steps} extra)"
        )

    @property
    def exhausted(self) -> bool:
        return self._paused

    @property
    def remaining(self) -> int:
        return max(0, self.max_steps - self.current_steps)

    def reset(self) -> None:
        """重置步数预算"""
        self.current_steps = 0
        self._paused = False
        self.max_steps = self._initial_max_steps
