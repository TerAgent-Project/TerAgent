# tests/test_budget.py
"""StepBudget 会话级步数预算单元测试

覆盖:
  - 步数消耗 (consume)
  - 暂停/恢复 (pause/resume)
  - 耗尽状态 (exhausted)
  - 重置 (reset)
  - remaining 属性
"""

from teragent.reliability.budget import StepBudget

# ===== 步数消耗 =====

class TestConsume:
    """步数消耗"""

    def test_consume_returns_true_when_budget_available(self):
        """有预算时消耗返回 True"""
        budget = StepBudget(max_steps=10)
        assert budget.consume() is True
        assert budget.current_steps == 1

    def test_consume_increments_step_counter(self):
        """每次消耗递增步数计数器"""
        budget = StepBudget(max_steps=10)
        budget.consume()
        budget.consume()
        assert budget.current_steps == 2

    def test_consume_when_paused_returns_false(self):
        """暂停状态消耗返回 False"""
        budget = StepBudget(max_steps=10)
        budget._paused = True
        assert budget.consume() is False
        assert budget.current_steps == 0  # 不递增


# ===== 暂停/恢复 =====

class TestPauseResume:
    """暂停/恢复"""

    def test_exhausted_when_max_steps_reached(self):
        """达到最大步数时暂停"""
        budget = StepBudget(max_steps=3)
        assert budget.consume() is True   # 1
        assert budget.consume() is True   # 2
        assert budget.consume() is True   # 3 → still allowed (max_steps=3 allows 3 steps)
        result = budget.consume()         # 4 → exhausted
        assert result is False
        assert budget.exhausted is True

    def test_resume_adds_extra_steps(self):
        """恢复时追加额外步数"""
        budget = StepBudget(max_steps=3)
        budget.consume()  # 1
        budget.consume()  # 2
        budget.consume()  # 3
        budget.consume()  # 4 → exhausted
        assert budget.exhausted is True

        budget.resume(extra_steps=5)
        assert budget.exhausted is False
        assert budget.max_steps == 8  # 3 + 5

    def test_resume_clears_paused_state(self):
        """恢复清除暂停状态"""
        budget = StepBudget(max_steps=2)
        budget.consume()  # 1
        budget.consume()  # 2
        budget.consume()  # 3 → exhausted
        assert budget._paused is True
        budget.resume(extra_steps=10)
        assert budget._paused is False


# ===== 耗尽状态 =====

class TestExhaustedState:
    """耗尽状态"""

    def test_not_exhausted_initially(self):
        """初始不耗尽"""
        budget = StepBudget()
        assert budget.exhausted is False

    def test_exhausted_after_max_steps(self):
        """达到最大步数后耗尽"""
        budget = StepBudget(max_steps=1)
        budget.consume()  # 1 → allowed
        budget.consume()  # 2 → exhausted
        assert budget.exhausted is True


# ===== 重置 =====

class TestReset:
    """重置"""

    def test_reset_clears_state(self):
        """重置清空所有状态"""
        budget = StepBudget(max_steps=5)
        budget.consume()
        budget.consume()
        budget.consume()
        budget.consume()
        budget.consume()  # 5 → allowed
        budget.consume()  # 6 → exhausted
        assert budget.exhausted is True

        budget.reset()
        assert budget.current_steps == 0
        assert budget.exhausted is False
        # reset() should restore the initial max_steps, not the module default
        assert budget.max_steps == 5

    def test_remaining_property(self):
        """remaining 属性"""
        budget = StepBudget(max_steps=10)
        assert budget.remaining == 10
        budget.consume()
        assert budget.remaining == 9

    def test_remaining_zero_when_exhausted(self):
        """耗尽后 remaining 为 0"""
        budget = StepBudget(max_steps=2)
        budget.consume()
        budget.consume()
        assert budget.remaining == 0
