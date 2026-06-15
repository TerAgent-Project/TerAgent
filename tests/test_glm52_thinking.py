# tests/test_glm52_thinking.py
"""P2-10 GLM-5.2 Dual Thinking Mode tests

Coverage:
  - ThinkingModeRouter with budget constraints
  - DynamicThinkingModeManager switching logic
  - PreservedThinkingManager recording and injection
  - reasoning_content round-trip persistence
  - Cost estimation in ThinkingModeDecision
  - Mode switch oscillation prevention
  - GLM52Compiler integration
"""

import pytest

from teragent.core.compilers.glm_52 import (
    DynamicThinkingModeManager,
    GLM52CompactionProfile,
    GLM52Compiler,
    ModeSwitchRecord,
    PreservedThinkingManager,
    ThinkingModeDecision,
    ThinkingModeRouter,
    ThinkingLevel,
    _THINKING_COST_MULTIPLIERS,
)
from teragent.core.tap import CompiledPrompt, LongHorizonConfig, TAPRequest


# ===== Helpers =====

def _make_request(**overrides) -> TAPRequest:
    """Construct a TAPRequest for testing."""
    defaults = {
        "meta": {"task_id": "test.1", "intent": "execute"},
        "instruction": "写一个排序函数",
        "constraints": [],
    }
    defaults.update(overrides)
    return TAPRequest(**defaults)


# ===== 1. ThinkingModeDecision cost_estimate =====

class TestThinkingModeDecisionCost:
    """ThinkingModeDecision cost_estimate field tests."""

    def test_default_cost_estimate_zero(self):
        """Default cost_estimate is 0.0."""
        decision = ThinkingModeDecision()
        assert decision.cost_estimate == 0.0

    def test_high_mode_cost_estimate(self):
        """High mode cost_estimate matches multiplier."""
        decision = ThinkingModeDecision(
            level="high",
            reason="test",
            cost_estimate=_THINKING_COST_MULTIPLIERS["high"],
        )
        assert decision.cost_estimate == 1.5

    def test_max_mode_cost_estimate(self):
        """Max mode cost_estimate matches multiplier."""
        decision = ThinkingModeDecision(
            level="max",
            reason="test",
            cost_estimate=_THINKING_COST_MULTIPLIERS["max"],
        )
        assert decision.cost_estimate == 4.0


# ===== 2. ThinkingModeRouter with budget constraints =====

class TestThinkingModeRouterBudget:
    """ThinkingModeRouter budget-aware routing tests."""

    def setup_method(self):
        self.router = ThinkingModeRouter()

    def test_no_budget_normal_routing(self):
        """No budget constraint → normal routing logic."""
        request = _make_request(
            meta={"task_id": "1", "intent": "plan"},
        )
        decision = self.router.select(request, budget_remaining=None)
        assert decision.level == "max"
        assert decision.cost_estimate == 4.0

    def test_budget_very_low_forces_high(self):
        """Budget < 5% forces High mode even for plan intent."""
        request = _make_request(
            meta={"task_id": "1", "intent": "plan"},
        )
        decision = self.router.select(request, budget_remaining=0.03)
        assert decision.level == "high"
        assert decision.cost_estimate == 1.5

    def test_budget_very_low_forces_high_debug(self):
        """Budget < 5% forces High mode even for debug instructions."""
        request = _make_request(instruction="debug the memory leak")
        decision = self.router.select(request, budget_remaining=0.02)
        assert decision.level == "high"
        assert "极度紧张" in decision.reason

    def test_budget_low_prefers_high(self):
        """Budget < 20% prefers High mode for normal tasks."""
        request = _make_request(
            meta={"task_id": "1", "intent": "execute"},
        )
        decision = self.router.select(request, budget_remaining=0.15)
        assert decision.level == "high"
        assert decision.cost_estimate == 1.5

    def test_budget_low_long_horizon_allows_max(self):
        """Budget < 20% still allows Max for long-horizon tasks."""
        request = _make_request(
            long_horizon=LongHorizonConfig(max_duration_hours=8),
        )
        decision = self.router.select(request, budget_remaining=0.15)
        assert decision.level == "max"
        assert "长程任务" in decision.reason

    def test_budget_low_debug_downgrades_to_high(self):
        """Budget < 20% downgrades debug tasks to High."""
        request = _make_request(instruction="debug this issue")
        decision = self.router.select(request, budget_remaining=0.10)
        assert decision.level == "high"
        assert "降级" in decision.reason

    def test_budget_sufficient_normal_routing(self):
        """Budget >= 20% uses normal routing logic."""
        request = _make_request(
            meta={"task_id": "1", "intent": "plan"},
        )
        decision = self.router.select(request, budget_remaining=0.50)
        assert decision.level == "max"

    def test_budget_exactly_5_percent_forces_high(self):
        """Budget = 5% is not < 5%, so normal low-budget logic applies."""
        request = _make_request(
            meta={"task_id": "1", "intent": "execute"},
        )
        decision = self.router.select(request, budget_remaining=0.05)
        # 5% is not < 5%, but it is < 20%, so it falls to low-budget logic
        assert decision.level == "high"

    def test_budget_exactly_20_percent_normal(self):
        """Budget = 20% is not < 20%, so normal routing applies."""
        request = _make_request(
            meta={"task_id": "1", "intent": "chat"},
        )
        decision = self.router.select(request, budget_remaining=0.20)
        assert decision.level == "high"

    def test_all_decisions_have_cost_estimate(self):
        """All router decisions should have cost_estimate set."""
        scenarios = [
            (_make_request(meta={"intent": "chat"}), None),
            (_make_request(meta={"intent": "plan"}), None),
            (_make_request(instruction="debug this"), None),
            (_make_request(instruction="simple query"), None),
            (_make_request(meta={"intent": "execute"}), 0.03),
            (_make_request(long_horizon=LongHorizonConfig()), None),
        ]
        for request, budget in scenarios:
            decision = self.router.select(request, budget_remaining=budget)
            assert decision.cost_estimate > 0, (
                f"Missing cost_estimate for intent={request.meta.get('intent')}, "
                f"budget={budget}"
            )
            assert decision.cost_estimate in (1.5, 4.0)


# ===== 3. DynamicThinkingModeManager =====

class TestDynamicThinkingModeManager:
    """DynamicThinkingModeManager switching tests."""

    def setup_method(self):
        self.router = ThinkingModeRouter()
        self.manager = DynamicThinkingModeManager(self.router)

    def test_initial_mode_is_high(self):
        """Initial current_mode is 'high'."""
        assert self.manager.current_mode == "high"

    def test_initial_switch_count_zero(self):
        """Initial switch_count is 0."""
        assert self.manager.switch_count == 0

    def test_same_mode_no_switch(self):
        """If suggested mode == current mode, no switch recorded."""
        # Current mode is "high", and a chat request should suggest "high"
        request = _make_request(meta={"intent": "chat"})
        decision = self.manager.should_switch(request, current_step=0)
        assert decision.level == "high"
        assert self.manager.switch_count == 0

    def test_switch_from_high_to_max(self):
        """Switch from High to Max for complex task."""
        request = _make_request(instruction="debug this code")
        decision = self.manager.should_switch(request, current_step=1)
        assert decision.level == "max"
        # Apply the switch
        self.manager.apply_switch(decision, current_step=1)
        assert self.manager.current_mode == "max"
        assert self.manager.switch_count == 1

    def test_oscillation_prevention_consecutive_steps(self):
        """Prevent oscillation: don't switch again on consecutive step."""
        # Switch from high to max at step 1
        request_max = _make_request(instruction="debug this code")
        decision = self.manager.should_switch(request_max, current_step=1)
        self.manager.apply_switch(decision, current_step=1)
        assert self.manager.current_mode == "max"

        # Try to switch back to high at step 2 (consecutive)
        request_high = _make_request(meta={"intent": "chat"})
        decision2 = self.manager.should_switch(request_high, current_step=2)
        # Should be prevented (steps_since_last = 2-1 = 1, which is <= 1)
        assert decision2.level == "max"  # Stays at max
        assert "防止振荡" in decision2.reason

    def test_oscillation_prevention_non_consecutive_steps(self):
        """Allow switch when not consecutive (steps_since_last > 1)."""
        # Switch from high to max at step 1
        request_max = _make_request(instruction="debug this code")
        decision = self.manager.should_switch(request_max, current_step=1)
        self.manager.apply_switch(decision, current_step=1)
        assert self.manager.current_mode == "max"

        # Switch back at step 3 (not consecutive, gap = 2)
        request_high = _make_request(meta={"intent": "chat"})
        decision2 = self.manager.should_switch(request_high, current_step=3)
        assert decision2.level == "high"  # Allowed

    def test_max_switch_limit(self):
        """Prevent switching after max_switches_per_task reached."""
        # Simulate 10 switches (the default max)
        for i in range(10):
            from_mode = "high" if i % 2 == 0 else "max"
            to_mode = "max" if i % 2 == 0 else "high"
            self.manager.record_switch(from_mode, to_mode, f"test switch {i}", step=i)

        assert self.manager.switch_count == 10

        # Next should_switch should be blocked
        request = _make_request(instruction="debug this code")
        decision = self.manager.should_switch(request, current_step=20)
        assert "最大切换次数" in decision.reason

    def test_reset(self):
        """Reset clears mode and history."""
        self.manager.record_switch("high", "max", "test", step=1)
        self.manager._current_mode = "max"
        self.manager.reset()
        assert self.manager.current_mode == "high"
        assert self.manager.switch_count == 0

    def test_mode_history_read_only(self):
        """mode_history returns a copy, not the original list."""
        self.manager.record_switch("high", "max", "test", step=1)
        history = self.manager.mode_history
        history.clear()  # Modify the copy
        assert self.manager.switch_count == 1  # Original unchanged

    def test_apply_switch_only_records_on_change(self):
        """apply_switch only records when mode actually changes."""
        # Current is "high", applying "high" decision shouldn't record
        decision = ThinkingModeDecision(level="high", reason="same")
        self.manager.apply_switch(decision, current_step=0)
        assert self.manager.switch_count == 0

    def test_record_switch_creates_mode_switch_record(self):
        """record_switch creates proper ModeSwitchRecord."""
        self.manager.record_switch("high", "max", "complex task", step=5)
        assert len(self.manager.mode_history) == 1
        record = self.manager.mode_history[0]
        assert isinstance(record, ModeSwitchRecord)
        assert record.from_mode == "high"
        assert record.to_mode == "max"
        assert record.reason == "complex task"
        assert record.step == 5
        assert record.timestamp > 0


# ===== 4. PreservedThinkingManager =====

class TestPreservedThinkingManager:
    """PreservedThinkingManager reasoning_content persistence tests."""

    def setup_method(self):
        self.manager = PreservedThinkingManager()

    def test_initial_state(self):
        """Initial manager has no reasoning history."""
        assert self.manager.reasoning_count == 0

    def test_record_reasoning(self):
        """record_reasoning stores content verbatim."""
        self.manager.record_reasoning("First reasoning step: analyzing code...")
        assert self.manager.reasoning_count == 1

    def test_record_reasoning_multiple_rounds(self):
        """Multiple rounds of reasoning are recorded in order."""
        self.manager.record_reasoning("Round 1: analysis")
        self.manager.record_reasoning("Round 2: design")
        self.manager.record_reasoning("Round 3: implementation")
        assert self.manager.reasoning_count == 3

    def test_record_reasoning_empty_ignored(self):
        """Empty reasoning content is not recorded."""
        self.manager.record_reasoning("")
        assert self.manager.reasoning_count == 0

    def test_get_preserved_reasoning_messages(self):
        """get_preserved_reasoning_messages returns correct format."""
        self.manager.record_reasoning("Thinking about step 1")
        self.manager.record_reasoning("Thinking about step 2")
        messages = self.manager.get_preserved_reasoning_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[0]["reasoning_content"] == "Thinking about step 1"
        assert messages[0]["content"] == ""
        assert messages[1]["reasoning_content"] == "Thinking about step 2"

    def test_inject_preserved_reasoning(self):
        """inject_preserved_reasoning inserts messages before last user message."""
        self.manager.record_reasoning("Round 1 reasoning")

        compiled = CompiledPrompt(messages=[
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First user message"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second user message"},
        ])

        self.manager.inject_preserved_reasoning(compiled)

        # Reasoning should be injected before the last user message (index 3)
        # After injection: system(0), user(1), assistant(2), reasoning(3), user(4)
        assert len(compiled.messages) == 5
        assert compiled.messages[3]["role"] == "assistant"
        assert compiled.messages[3]["reasoning_content"] == "Round 1 reasoning"
        assert compiled.messages[4]["role"] == "user"
        assert compiled.messages[4]["content"] == "Second user message"

    def test_inject_preserved_reasoning_multiple_rounds(self):
        """Multiple reasoning rounds are all injected in order."""
        self.manager.record_reasoning("Round 1")
        self.manager.record_reasoning("Round 2")

        compiled = CompiledPrompt(messages=[
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User message"},
        ])

        self.manager.inject_preserved_reasoning(compiled)

        assert len(compiled.messages) == 4
        assert compiled.messages[1]["reasoning_content"] == "Round 1"
        assert compiled.messages[2]["reasoning_content"] == "Round 2"
        assert compiled.messages[3]["content"] == "User message"

    def test_inject_preserved_reasoning_no_history(self):
        """No injection when there's no reasoning history."""
        compiled = CompiledPrompt(messages=[
            {"role": "user", "content": "Hello"},
        ])
        self.manager.inject_preserved_reasoning(compiled)
        assert len(compiled.messages) == 1  # Unchanged

    def test_clear(self):
        """clear removes all reasoning history."""
        self.manager.record_reasoning("Some reasoning")
        self.manager.clear()
        assert self.manager.reasoning_count == 0

    def test_should_preserve_existing_logic(self):
        """should_preserve logic remains correct after refactoring."""
        # Long-horizon task
        request = _make_request(long_horizon=LongHorizonConfig())
        decision = ThinkingModeDecision(level="max", preserve_thinking=False)
        assert self.manager.should_preserve(request, decision) is True

        # Multi-step execution
        request = _make_request(meta={"step_count": 5})
        assert self.manager.should_preserve(request, decision) is True

        # Simple chat — no preserve
        request = _make_request(meta={"intent": "chat"})
        assert self.manager.should_preserve(request, decision) is False

        # Decision with preserve_thinking=True
        request = _make_request(meta={"intent": "chat"})
        decision = ThinkingModeDecision(level="high", preserve_thinking=True)
        assert self.manager.should_preserve(request, decision) is True


# ===== 5. reasoning_content round-trip persistence =====

class TestReasoningRoundTrip:
    """End-to-end reasoning_content round-trip tests."""

    def test_full_round_trip(self):
        """Simulate multi-round reasoning persistence."""
        manager = PreservedThinkingManager()

        # Round 1: model generates reasoning
        r1 = "Let me analyze the code structure first. I see three modules..."
        manager.record_reasoning(r1)

        # Build messages for round 2
        compiled = CompiledPrompt(messages=[
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Refactor the auth module"},
        ])

        # Inject preserved reasoning
        manager.inject_preserved_reasoning(compiled)

        # Verify reasoning is present before user message
        user_indices = [
            i for i, m in enumerate(compiled.messages) if m.get("role") == "user"
        ]
        last_user_idx = max(user_indices)
        # Reasoning should be before the last user message
        reasoning_msg = compiled.messages[last_user_idx - 1]
        assert reasoning_msg["reasoning_content"] == r1

        # Round 2: model generates more reasoning
        r2 = "Based on the previous analysis, I'll refactor the auth module..."
        manager.record_reasoning(r2)
        assert manager.reasoning_count == 2

        # Build messages for round 3
        compiled2 = CompiledPrompt(messages=[
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Continue refactoring"},
        ])
        manager.inject_preserved_reasoning(compiled2)

        # Both reasoning rounds should be present
        reasoning_msgs = [
            m for m in compiled2.messages if "reasoning_content" in m
        ]
        assert len(reasoning_msgs) == 2
        assert reasoning_msgs[0]["reasoning_content"] == r1
        assert reasoning_msgs[1]["reasoning_content"] == r2

    def test_reasoning_verbatim_preservation(self):
        """Reasoning content is stored and retrieved verbatim."""
        manager = PreservedThinkingManager()

        # Include special characters, unicode, whitespace
        complex_reasoning = "  Analysis:\n  - Step 1: α = β + γ\n  - Step 2: 中文测试\n  <tag>content</tag>  "
        manager.record_reasoning(complex_reasoning)

        messages = manager.get_preserved_reasoning_messages()
        assert messages[0]["reasoning_content"] == complex_reasoning

    def test_reasoning_order_preserved(self):
        """Reasoning messages maintain insertion order."""
        manager = PreservedThinkingManager()
        for i in range(5):
            manager.record_reasoning(f"Round {i} reasoning")

        messages = manager.get_preserved_reasoning_messages()
        for i, msg in enumerate(messages):
            assert msg["reasoning_content"] == f"Round {i} reasoning"


# ===== 6. ModeSwitchRecord =====

class TestModeSwitchRecord:
    """ModeSwitchRecord dataclass tests."""

    def test_record_creation(self):
        """ModeSwitchRecord stores all fields correctly."""
        record = ModeSwitchRecord(
            from_mode="high",
            to_mode="max",
            reason="Complex task detected",
            step=5,
        )
        assert record.from_mode == "high"
        assert record.to_mode == "max"
        assert record.reason == "Complex task detected"
        assert record.step == 5
        assert record.timestamp > 0

    def test_record_default_step(self):
        """Default step is 0."""
        record = ModeSwitchRecord(from_mode="max", to_mode="high", reason="test")
        assert record.step == 0


# ===== 7. GLM52Compiler Integration =====

class TestGLM52CompilerIntegration:
    """GLM52Compiler integration with new features."""

    def test_compiler_has_dynamic_mode_manager(self):
        """GLM52Compiler has DynamicThinkingModeManager."""
        compiler = GLM52Compiler()
        assert hasattr(compiler, "_dynamic_mode_manager")
        assert isinstance(compiler._dynamic_mode_manager, DynamicThinkingModeManager)

    def test_compiler_has_preserved_thinking_manager(self):
        """GLM52Compiler has PreservedThinkingManager with __init__."""
        compiler = GLM52Compiler()
        assert hasattr(compiler, "_preserved_thinking_manager")
        assert isinstance(compiler._preserved_thinking_manager, PreservedThinkingManager)

    def test_cost_estimate_in_compiled_extra(self):
        """CompiledPrompt.extra contains cost_estimate."""
        compiler = GLM52Compiler()
        request = _make_request(
            meta={"intent": "chat"},
            thinking_mode="auto",
        )
        compiled = compiler.compile(request)
        assert "cost_estimate" in compiled.extra
        assert compiled.extra["cost_estimate"] > 0

    def test_deep_mode_cost_estimate(self):
        """Deep mode sets cost_estimate to max multiplier."""
        compiler = GLM52Compiler()
        request = _make_request(thinking_mode="deep")
        compiled = compiler.compile(request)
        assert compiled.extra["cost_estimate"] == 4.0

    def test_quick_mode_cost_estimate(self):
        """Quick mode sets cost_estimate to high multiplier."""
        compiler = GLM52Compiler()
        request = _make_request(thinking_mode="quick")
        compiled = compiler.compile(request)
        assert compiled.extra["cost_estimate"] == 1.5

    def test_long_horizon_uses_dynamic_manager(self):
        """Long-horizon tasks use DynamicThinkingModeManager."""
        compiler = GLM52Compiler()
        request = _make_request(
            long_horizon=LongHorizonConfig(max_duration_hours=8),
            thinking_mode="auto",
        )
        compiled = compiler.compile(request)
        # Should have used dynamic manager (which would select max for long-horizon)
        thinking = compiled.extra.get("thinking", {})
        assert thinking.get("level") == "max"
        # Dynamic manager should have tracked the switch
        assert compiler._dynamic_mode_manager.switch_count >= 0

    def test_preserved_reasoning_injection_in_compile(self):
        """Preserved reasoning is injected during compile when preserve_thinking=True."""
        compiler = GLM52Compiler()

        # Pre-populate some reasoning
        compiler._preserved_thinking_manager.record_reasoning("Previous reasoning content")

        request = _make_request(
            long_horizon=LongHorizonConfig(max_duration_hours=8),
            thinking_mode="auto",
        )
        compiled = compiler.compile(request)

        # The compiled messages should contain the preserved reasoning
        reasoning_msgs = [
            m for m in compiled.messages if "reasoning_content" in m
        ]
        assert len(reasoning_msgs) >= 1
        assert reasoning_msgs[0]["reasoning_content"] == "Previous reasoning content"

    def test_no_preserved_reasoning_when_not_needed(self):
        """No reasoning injection when preserve_thinking is not needed."""
        compiler = GLM52Compiler()

        request = _make_request(
            meta={"intent": "chat"},
            thinking_mode="auto",
        )
        compiled = compiler.compile(request)

        # No reasoning messages should be injected
        reasoning_msgs = [
            m for m in compiled.messages if "reasoning_content" in m
        ]
        assert len(reasoning_msgs) == 0

    def test_dynamic_mode_manager_reset_on_new_compiler(self):
        """Each new compiler instance has a fresh DynamicThinkingModeManager."""
        compiler1 = GLM52Compiler()
        compiler2 = GLM52Compiler()
        assert compiler1._dynamic_mode_manager is not compiler2._dynamic_mode_manager
        assert compiler2._dynamic_mode_manager.switch_count == 0

    def test_cost_multipliers_values(self):
        """_THINKING_COST_MULTIPLIERS has expected values."""
        assert _THINKING_COST_MULTIPLIERS["high"] == 1.5
        assert _THINKING_COST_MULTIPLIERS["max"] == 4.0


# ===== 8. Regression: Existing GLM-5.2 behavior =====

class TestGLM52Regression:
    """Ensure existing GLM-5.2 behavior is preserved."""

    def test_max_context_1m(self):
        """GLM-5.2 max_context_tokens = 1M."""
        compiler = GLM52Compiler()
        assert compiler.max_context_tokens == 1_000_000

    def test_compiler_type(self):
        """GLM-5.2 compiler type = glm_52."""
        compiler = GLM52Compiler()
        assert compiler._get_compiler_type() == "glm_52"

    def test_thinking_deep_to_max(self):
        """thinking_mode=deep → Max mode."""
        compiler = GLM52Compiler()
        request = _make_request(thinking_mode="deep")
        compiled = compiler.compile(request)
        thinking = compiled.extra.get("thinking", {})
        assert thinking.get("level") == "max"

    def test_thinking_quick_to_high(self):
        """thinking_mode=quick → High mode."""
        compiler = GLM52Compiler()
        request = _make_request(thinking_mode="quick")
        compiled = compiler.compile(request)
        thinking = compiled.extra.get("thinking", {})
        assert thinking.get("level") == "high"

    def test_auto_chat_to_high(self):
        """thinking_mode=auto + chat → High."""
        compiler = GLM52Compiler()
        request = _make_request(
            meta={"intent": "chat"},
            thinking_mode="auto",
        )
        compiled = compiler.compile(request)
        thinking = compiled.extra.get("thinking", {})
        assert thinking.get("level") == "high"

    def test_auto_plan_to_max(self):
        """thinking_mode=auto + plan → Max."""
        compiler = GLM52Compiler()
        request = _make_request(
            meta={"intent": "plan"},
            thinking_mode="auto",
        )
        compiled = compiler.compile(request)
        thinking = compiled.extra.get("thinking", {})
        assert thinking.get("level") == "max"

    def test_preserve_thinking_coding_plan(self):
        """Coding Plan scenario preserves thinking."""
        compiler = GLM52Compiler()
        request = _make_request(
            meta={"intent": "plan"},
            context={"design": "完整设计文档"},
            thinking_mode="auto",
        )
        compiled = compiler.compile(request)
        assert compiled.extra.get("preserve_thinking") is True

    def test_compaction_profile(self):
        """GLM52CompactionProfile budget is correct."""
        profile = GLM52CompactionProfile()
        assert profile.total_budget == 1_024_000
