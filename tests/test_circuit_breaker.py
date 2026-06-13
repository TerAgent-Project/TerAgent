# tests/test_circuit_breaker.py
"""Circuit Breaker 熔断器单元测试

覆盖:
  - CostBudgetTracker: 预算追踪/警告/严重/耗尽/阶段统计
  - ConsecutiveFailureBreaker: 熔断状态机 closed→open→half_open→closed
  - LatencyBreaker: 延迟追踪/慢检测/reset
  - ProgressDetector: 进度检测/停滞评分/停滞警告
  - CircuitBreakerManager: 统一管理/record_model_call/状态报告
"""
import time

from teragent.reliability.circuit_breaker import (
    CostBudgetConfig,
    CostBudgetTracker,
    LatencyBreaker,
    ProgressDetector,
)

# ===== CostBudgetTracker =====

class TestCostBudgetTracker:
    """预算追踪器"""

    def test_initial_state_ok(self, budget_tracker):
        """初始状态为 ok"""
        result = budget_tracker.check_budget()
        assert result.level == "ok"
        assert result.utilization == 0.0

    def test_record_usage(self, budget_tracker):
        """记录使用量"""
        result = budget_tracker.record_usage(prompt_tokens=1000, completion_tokens=500, stage="plan")
        assert result.total_tokens_used == 1500
        assert result.prompt_tokens_used == 1000
        assert result.completion_tokens_used == 500

    def test_warning_threshold(self, budget_tracker):
        """达到警告阈值"""
        # max=10000, warning=0.7 → 7000 tokens
        budget_tracker.record_usage(prompt_tokens=3000, completion_tokens=1000, stage="plan")
        assert budget_tracker.is_warning is False  # 4000 < 70%

        # 继续使用超过 70%
        budget_tracker.record_usage(prompt_tokens=2000, completion_tokens=2000, stage="execute")
        # 总计 8000, 超过 70%
        result = budget_tracker.check_budget()
        assert result.level == "warning"
        assert budget_tracker.is_warning is True

    def test_critical_threshold(self, budget_tracker):
        """达到严重阈值"""
        # max=10000, critical=0.9 → 9000 tokens
        budget_tracker.record_usage(prompt_tokens=5000, completion_tokens=3000, stage="execute")
        # 总计 8000, 还未到 critical (80%)
        result = budget_tracker.check_budget()
        assert result.level == "warning"

        # 继续使用直到超过 90%
        budget_tracker.record_usage(prompt_tokens=1000, completion_tokens=500, stage="execute")
        result = budget_tracker.check_budget()
        assert result.level == "critical"
        assert result.utilization >= 0.9

    def test_hard_limit_exhausted(self):
        """硬限制耗尽"""
        config = CostBudgetConfig(max_session_tokens=1000, enable_hard_limit=True)
        tracker = CostBudgetTracker(config=config)
        result = tracker.record_usage(prompt_tokens=600, completion_tokens=500, stage="test")
        assert result.level == "exhausted"
        assert tracker.is_exhausted is True

    def test_soft_limit_not_exhausted(self):
        """软限制不阻止调用"""
        config = CostBudgetConfig(max_session_tokens=1000, enable_hard_limit=False)
        tracker = CostBudgetTracker(config=config)
        tracker.record_usage(prompt_tokens=600, completion_tokens=500, stage="test")
        # 超过 100% 但软限制不标记为 exhausted
        result = tracker.check_budget()
        # utilization cap at 1.0
        assert result.utilization == 1.0

    def test_per_stage_tracking(self, budget_tracker):
        """阶段统计"""
        budget_tracker.record_usage(prompt_tokens=100, completion_tokens=50, stage="design")
        budget_tracker.record_usage(prompt_tokens=200, completion_tokens=100, stage="plan")

        breakdown = budget_tracker.get_stage_breakdown()
        assert "design" in breakdown
        assert "plan" in breakdown
        assert breakdown["design"]["prompt_tokens"] == 100
        assert breakdown["plan"]["completion_tokens"] == 100

    def test_session_summary(self, budget_tracker):
        """会话摘要"""
        budget_tracker.record_usage(prompt_tokens=100, completion_tokens=50, stage="test")
        summary = budget_tracker.get_session_summary()
        assert "total_tokens" in summary
        assert "estimated_cost" in summary
        assert "stages" in summary

    def test_cost_estimation(self):
        """成本估算"""
        config = CostBudgetConfig(
            max_session_tokens=1_000_000,
            cost_per_million_input=3.0,
            cost_per_million_output=15.0,
        )
        tracker = CostBudgetTracker(config=config)
        tracker.record_usage(prompt_tokens=1_000_000, completion_tokens=100_000, stage="test")
        result = tracker.check_budget()
        # input: 1M * $3/M = $3.0, output: 100K * $15/M = $1.5 → $4.5
        assert abs(result.estimated_cost - 4.5) < 0.01

    def test_estimate_call_cost(self):
        """预估调用成本"""
        config = CostBudgetConfig(cost_per_million_input=5.0)
        tracker = CostBudgetTracker(config=config)
        cost = tracker.estimate_call_cost(prompt_tokens=1_000_000)
        assert abs(cost - 5.0) < 0.01

    def test_reset(self, budget_tracker):
        """重置追踪状态"""
        budget_tracker.record_usage(prompt_tokens=1000, completion_tokens=500, stage="test")
        budget_tracker.reset()
        result = budget_tracker.check_budget()
        assert result.total_tokens_used == 0

    def test_utilization_property(self, budget_tracker):
        """utilization 属性"""
        assert budget_tracker.utilization == 0.0
        budget_tracker.record_usage(prompt_tokens=5000, completion_tokens=0, stage="test")
        assert 0.4 < budget_tracker.utilization < 0.6


# ===== ConsecutiveFailureBreaker =====

class TestConsecutiveFailureBreaker:
    """熔断状态机"""

    def test_initial_state_closed(self, failure_breaker):
        """初始状态为 closed"""
        state = failure_breaker.get_state()
        assert state.name == "closed"
        assert state.consecutive_failures == 0

    def test_record_failure_increments(self, failure_breaker):
        """失败计数递增"""
        failure_breaker.record_failure("error1")
        state = failure_breaker.get_state()
        assert state.consecutive_failures == 1
        assert state.total_failures == 1

    def test_opens_after_max_consecutive(self, failure_breaker):
        """达到阈值后熔断器打开"""
        failure_breaker.record_failure("err1")
        failure_breaker.record_failure("err2")
        failure_breaker.record_failure("err3")  # max_consecutive=3
        assert failure_breaker.is_open is True

    def test_success_resets_consecutive(self, failure_breaker):
        """成功重置连续失败计数"""
        failure_breaker.record_failure("err1")
        failure_breaker.record_failure("err2")
        failure_breaker.record_success()
        state = failure_breaker.get_state()
        assert state.consecutive_failures == 0
        assert state.name == "closed"

    def test_half_open_after_cooldown(self, failure_breaker):
        """冷却后进入半开状态"""
        failure_breaker.record_failure("err1")
        failure_breaker.record_failure("err2")
        failure_breaker.record_failure("err3")
        assert failure_breaker.is_open is True

        # 等待冷却（window_seconds=1.0）
        time.sleep(1.1)
        # get_state() is query-only; use try_half_open() for transition
        state = failure_breaker.get_state()
        assert state.can_retry is True  # Ready to retry
        assert failure_breaker.try_half_open() is True
        state = failure_breaker.get_state()
        assert state.name == "half_open"
        assert state.can_retry is True

    def test_half_open_failure_reopens(self, failure_breaker):
        """半开状态失败重新打开"""
        failure_breaker.record_failure("err1")
        failure_breaker.record_failure("err2")
        failure_breaker.record_failure("err3")

        time.sleep(1.1)
        failure_breaker.try_half_open()  # Explicitly trigger half_open

        failure_breaker.record_failure("err4")
        assert failure_breaker.is_open is True

    def test_half_open_success_closes(self, failure_breaker):
        """半开状态成功关闭熔断器"""
        failure_breaker.record_failure("err1")
        failure_breaker.record_failure("err2")
        failure_breaker.record_failure("err3")

        time.sleep(1.1)
        failure_breaker.try_half_open()  # Explicitly trigger half_open

        failure_breaker.record_success()
        state = failure_breaker.get_state()
        assert state.name == "closed"

    def test_reset(self, failure_breaker):
        """重置熔断器"""
        failure_breaker.record_failure("err1")
        failure_breaker.record_failure("err2")
        failure_breaker.record_failure("err3")
        failure_breaker.reset()
        state = failure_breaker.get_state()
        assert state.name == "closed"
        assert state.consecutive_failures == 0

    def test_total_counts(self, failure_breaker):
        """总计数统计"""
        failure_breaker.record_failure("err1")
        failure_breaker.record_success()
        failure_breaker.record_failure("err2")
        state = failure_breaker.get_state()
        assert state.total_failures == 2
        assert state.total_successes == 1


# ===== LatencyBreaker =====

class TestLatencyBreaker:
    """延迟追踪器"""

    def test_initial_not_slow(self):
        """初始状态不慢"""
        breaker = LatencyBreaker(warn_latency_ms=1000.0)
        assert breaker.is_slow is False

    def test_below_threshold_not_slow(self):
        """低于阈值不慢"""
        breaker = LatencyBreaker(warn_latency_ms=1000.0, avg_window=5)
        for _ in range(5):
            breaker.record_latency(500.0)
        assert breaker.is_slow is False

    def test_above_threshold_slow(self):
        """超过阈值为慢"""
        breaker = LatencyBreaker(warn_latency_ms=1000.0, avg_window=5)
        for _ in range(5):
            breaker.record_latency(2000.0)
        assert breaker.is_slow is True

    def test_needs_min_3_samples(self):
        """至少需要 3 个样本才判定为慢"""
        breaker = LatencyBreaker(warn_latency_ms=1000.0, avg_window=10)
        breaker.record_latency(5000.0)
        breaker.record_latency(5000.0)
        assert breaker.is_slow is False  # 只有 2 个样本

        breaker.record_latency(5000.0)
        assert breaker.is_slow is True  # 3 个样本

    def test_get_avg_latency(self):
        """平均延迟计算"""
        breaker = LatencyBreaker(warn_latency_ms=10000.0, avg_window=5)
        breaker.record_latency(100.0)
        breaker.record_latency(200.0)
        assert abs(breaker.get_avg_latency() - 150.0) < 0.01

    def test_rolling_window(self):
        """滚动窗口"""
        breaker = LatencyBreaker(warn_latency_ms=100000.0, avg_window=3)
        breaker.record_latency(100.0)
        breaker.record_latency(200.0)
        breaker.record_latency(300.0)
        breaker.record_latency(400.0)  # 100 被挤出
        avg = breaker.get_avg_latency()
        assert abs(avg - 300.0) < 0.01  # (200+300+400)/3

    def test_peak_latency(self):
        """峰值延迟"""
        breaker = LatencyBreaker(warn_latency_ms=100000.0)
        breaker.record_latency(100.0)
        breaker.record_latency(500.0)
        breaker.record_latency(200.0)
        assert breaker._peak_latency_ms == 500.0

    def test_get_state(self):
        """状态报告"""
        breaker = LatencyBreaker(warn_latency_ms=1000.0)
        breaker.record_latency(500.0)
        state = breaker.get_state()
        assert "avg_latency_ms" in state
        assert "is_slow" in state
        assert "peak_latency_ms" in state

    def test_reset(self):
        """3.7: LatencyBreaker.reset()"""
        breaker = LatencyBreaker(warn_latency_ms=1000.0)
        breaker.record_latency(5000.0)
        breaker.record_latency(3000.0)
        assert breaker.get_avg_latency() > 0

        breaker.reset()
        assert breaker.get_avg_latency() == 0.0
        assert breaker._total_calls == 0
        assert breaker._peak_latency_ms == 0.0


# ===== ProgressDetector =====

class TestProgressDetector:
    """进度检测器"""

    def test_initial_not_stalled(self):
        """初始不停滞"""
        detector = ProgressDetector(stall_threshold=5)
        assert detector.is_stalled() is False

    def test_effective_steps_not_stalled(self):
        """有效步骤不停滞"""
        detector = ProgressDetector(stall_threshold=5)
        for _ in range(10):
            detector.record_step("write_file", had_effect=True)
        assert detector.is_stalled() is False

    def test_ineffective_steps_stalled(self):
        """无效步骤停滞"""
        detector = ProgressDetector(stall_threshold=5)
        for _ in range(10):
            detector.record_step("read_file", had_effect=False)
        assert detector.is_stalled() is True
        assert detector.get_stall_score() >= 0.8

    def test_stall_score_calculation(self):
        """停滞评分计算"""
        detector = ProgressDetector(stall_threshold=4)
        detector.record_step("write_file", had_effect=True)
        detector.record_step("read_file", had_effect=False)
        detector.record_step("read_file", had_effect=False)
        detector.record_step("read_file", had_effect=False)
        # 最近 4 步: 1 有效, 3 无效 → score = 0.75
        score = detector.get_stall_score()
        assert abs(score - 0.75) < 0.01

    def test_below_threshold_not_stalled(self):
        """低于阈值步数不判定停滞"""
        detector = ProgressDetector(stall_threshold=10)
        for _ in range(5):
            detector.record_step("read_file", had_effect=False)
        assert detector.is_stalled() is False

    def test_reset(self):
        """重置进度检测器"""
        detector = ProgressDetector(stall_threshold=5)
        for _ in range(10):
            detector.record_step("tool", had_effect=False)
        detector.reset()
        assert detector.is_stalled() is False
        assert detector.get_stall_score() == 0.0

    def test_unique_tools_tracked(self):
        """跟踪唯一工具数"""
        detector = ProgressDetector(stall_threshold=5)
        detector.record_step("read_file", had_effect=True)
        detector.record_step("write_file", had_effect=True)
        detector.record_step("read_file", had_effect=True)
        assert len(detector._unique_tools) == 2


# ===== CircuitBreakerManager =====

class TestCircuitBreakerManager:
    """统一熔断管理器"""

    def test_initial_state(self, cb_manager):
        """初始状态"""
        status = cb_manager.get_status()
        assert "budget" in status
        assert "failure_breaker" in status
        assert "latency" in status
        assert "progress" in status

    def test_record_model_call(self, cb_manager):
        """记录模型调用"""
        result = cb_manager.record_model_call(
            prompt_tokens=1000, completion_tokens=500,
            stage="plan", latency_ms=1500.0,
        )
        assert result.total_tokens_used == 1500

    def test_record_success_and_failure(self, cb_manager):
        """记录成功和失败"""
        cb_manager.record_success()
        cb_manager.record_failure("test error")
        status = cb_manager.get_status()
        assert status["failure_breaker"]["total_failures"] == 1
        assert status["failure_breaker"]["total_successes"] == 1

    def test_record_agent_step(self, cb_manager):
        """记录 Agent 步骤"""
        cb_manager.record_agent_step("read_file", had_effect=True)
        cb_manager.record_agent_step("read_file", had_effect=False)
        status = cb_manager.get_status()
        assert status["progress"]["total_steps"] == 2

    def test_check_before_call(self, cb_manager):
        """调用前检查预算"""
        result = cb_manager.check_before_call(estimated_prompt_tokens=1000)
        assert result.level == "ok"

    def test_get_budget_summary(self, cb_manager):
        """预算摘要"""
        cb_manager.record_model_call(1000, 500, "plan", 1500.0)
        summary = cb_manager.get_budget_summary()
        assert "total_tokens" in summary
        assert "stages" in summary

    def test_reset_all(self, cb_manager):
        """重置所有熔断器"""
        cb_manager.record_model_call(5000, 3000, "test", 5000.0)
        cb_manager.record_failure("err")
        cb_manager.record_agent_step("tool", had_effect=False)

        cb_manager.reset_all()

        status = cb_manager.get_status()
        assert status["budget"]["total_tokens"] == 0
        assert status["failure_breaker"]["consecutive_failures"] == 0

    def test_sub_components_accessible(self, cb_manager):
        """子组件可通过属性访问"""
        assert cb_manager.cost_tracker is not None
        assert cb_manager.failure_breaker is not None
        assert cb_manager.latency_breaker is not None
        assert cb_manager.progress_detector is not None
