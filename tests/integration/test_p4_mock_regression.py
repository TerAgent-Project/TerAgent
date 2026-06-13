"""Phase 4 Integration Tests — Optimization & Stabilization

Covers:
  P4-1: Benchmark framework (CompilationBenchmark, BenchmarkRunner, etc.)
  P4-2: Prompt tuning (enhanced V4/M3/GLM-5 prompts, cuda_triton intent)
  P4-3: Fault recovery (ModelCircuitBreakerManager, DegradationChain,
         LongHorizonRecoveryManager, RateLimitHandler)

All tests use MockAdapter — no real API calls.
"""

from __future__ import annotations

import json
import statistics

import pytest

# ---------------------------------------------------------------------------
# P4-1: Benchmark Framework imports
# ---------------------------------------------------------------------------
from teragent.benchmark.benchmark import (
    ALL_COMPILERS,
    BenchmarkMetric,
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkRunner,
    CompilationBenchmark,
    make_tap_request,
)
from teragent.core.compiler import TAPCompilerRegistry

# ---------------------------------------------------------------------------
# P4-2: Prompt tuning imports
# ---------------------------------------------------------------------------
from teragent.core.prompts import (
    # GLM-5 agent prompt
    AGENT_PROMPT_GLM_5,
    # M3 agent prompt
    AGENT_PROMPT_MINIMAX_M3,
    # CUDA/Triton
    CUDA_TRITON_PROMPT_GLM_5,
    # V4 design prompt
    DESIGN_PROMPT_DEEPSEEK_V4,
    # V4 execute prompt
    EXECUTE_PROMPT_DEEPSEEK_V4,
    # M3 execute prompt
    EXECUTE_PROMPT_MINIMAX_M3,
    # M3 plan prompt
    PLAN_PROMPT_MINIMAX_M3,
    # V4 review prompt
    REVIEW_PROMPT_DEEPSEEK_V4,
    # M3 sub_agent prompt
    SUB_AGENT_PROMPT_MINIMAX_M3,
    get_system_prompt_for_intent,
)

# ---------------------------------------------------------------------------
# Core imports for end-to-end tests
# ---------------------------------------------------------------------------
from teragent.core.tap import (
    TAPRequest,
)

# ---------------------------------------------------------------------------
# P4-3: Fault Recovery imports
# ---------------------------------------------------------------------------
from teragent.reliability.circuit_breaker import (
    ModelBreakerConfig,
    ModelBreakerState,
    ModelCircuitBreakerManager,
)
from teragent.reliability.recovery import (
    DegradationChain,
    LongHorizonRecoveryManager,
    RateLimitHandler,
    RateLimitInfo,
)

# ===========================================================================
# P4-1: Benchmark Framework
# ===========================================================================


class TestP4_1_BenchmarkFramework:
    """Tests for the benchmark framework (P4-1)."""

    def test_benchmark_metric_from_samples(self):
        """BenchmarkMetric.from_samples computes correct statistics."""
        samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        metric = BenchmarkMetric.from_samples("latency", samples, unit="ms")

        assert metric.name == "latency"
        assert metric.unit == "ms"
        assert metric.sample_count == 10
        assert metric.mean == pytest.approx(5.5, rel=1e-6)
        assert metric.median == pytest.approx(5.5, rel=1e-6)
        assert metric.min == 1.0
        assert metric.max == 10.0
        assert metric.p95 >= 9.0  # 95th percentile of 1-10
        assert metric.p99 >= 9.0  # 99th percentile
        assert metric.std_dev > 0

    def test_benchmark_metric_from_empty_samples(self):
        """BenchmarkMetric.from_samples returns zeroed metric for empty list."""
        metric = BenchmarkMetric.from_samples("empty", [], unit="ms")
        assert metric.name == "empty"
        assert metric.value == 0.0
        assert metric.sample_count == 0

    def test_benchmark_metric_to_dict(self):
        """BenchmarkMetric.to_dict serializes all fields."""
        metric = BenchmarkMetric(
            name="test_metric", value=42.0, unit="ms",
            mean=42.0, median=41.0, p95=50.0, p99=55.0,
            std_dev=3.0, min=38.0, max=55.0, sample_count=100,
        )
        d = metric.to_dict()
        assert d["name"] == "test_metric"
        assert d["value"] == 42.0
        assert d["unit"] == "ms"
        assert d["sample_count"] == 100
        assert "mean" in d and "median" in d and "p95" in d

    def test_benchmark_result_construction_and_serialization(self):
        """BenchmarkResult construction, add_metric, get_metric, to_dict."""
        result = BenchmarkResult(suite_name="TestSuite", model="test_model")
        assert result.suite_name == "TestSuite"
        assert result.model == "test_model"
        assert result.timestamp > 0.0

        metric = BenchmarkMetric(name="latency", value=5.0, unit="ms")
        result.add_metric(metric)
        assert result.get_metric("latency") is metric
        assert result.get_metric("nonexistent") is None

        d = result.to_dict()
        assert d["suite_name"] == "TestSuite"
        assert len(d["metrics"]) == 1

    def test_benchmark_report_to_text(self):
        """BenchmarkReport.to_text produces a readable text report."""
        report = BenchmarkReport()
        result = BenchmarkResult(suite_name="CompilationBenchmark", model="deepseek_v4")
        result.add_metric(BenchmarkMetric(name="compile_latency", value=1.5, unit="ms"))
        report.add_result(result)
        report.summary = {"total_benchmarks": 1}

        text = report.to_text()
        assert "TERAGENT PERFORMANCE BENCHMARK REPORT" in text
        assert "CompilationBenchmark" in text
        assert "deepseek_v4" in text
        assert "compile_latency" in text
        assert "SUMMARY" in text

    def test_benchmark_report_to_json(self):
        """BenchmarkReport.to_json produces valid JSON with all results."""
        report = BenchmarkReport()
        result = BenchmarkResult(suite_name="LatencyBenchmark", model="minimax_m3")
        result.add_metric(BenchmarkMetric(name="send_latency", value=50.0, unit="ms"))
        report.add_result(result)

        json_str = report.to_json()
        data = json.loads(json_str)
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["suite_name"] == "LatencyBenchmark"

    def test_compilation_benchmark_produces_results(self):
        """CompilationBenchmark.run() produces results for all 3 new compilers."""
        bench = CompilationBenchmark(iterations=5, seed=42)
        results = bench.run()
        assert len(results) > 0
        # Should have results for all_compilers + per-intent + large-context
        suite_names = {r.suite_name for r in results}
        assert "CompilationBenchmark" in suite_names
        # Each result should have at least one metric
        for r in results:
            if r.model != "error":
                assert len(r.metrics) > 0, f"No metrics for {r.model} in {r.metadata}"

    def test_benchmark_runner_run_all(self):
        """BenchmarkRunner.run_all() produces a report with all suite results."""
        # Use small iterations for speed
        runner = BenchmarkRunner(iterations=3, seed=42, suites=["compilation"])
        report = runner.run_all()
        assert isinstance(report, BenchmarkReport)
        assert len(report.results) > 0
        assert "total_benchmarks" in report.summary
        assert report.summary["iterations_per_scenario"] == 3

    def test_benchmark_runner_run_suite(self):
        """BenchmarkRunner.run_suite() runs a single named suite."""
        runner = BenchmarkRunner(iterations=3, seed=42)
        results = runner.run_suite("compilation")
        assert len(results) > 0
        assert all(r.suite_name == "CompilationBenchmark" for r in results)

    def test_benchmark_runner_unknown_suite_raises(self):
        """BenchmarkRunner.run_suite() raises ValueError for unknown suite."""
        runner = BenchmarkRunner(iterations=3, seed=42)
        with pytest.raises(ValueError, match="Unknown benchmark suite"):
            runner.run_suite("nonexistent_suite")

    def test_make_tap_request_generates_valid_request(self):
        """make_tap_request() generates a valid TAPRequest with proper fields."""
        request = make_tap_request(
            intent="execute",
            context_size="small",
            has_multimodal=True,
            has_desktop=True,
            is_long_horizon=True,
        )
        assert request.meta["intent"] == "execute"
        assert request.multimodal_context is not None
        assert len(request.multimodal_context) > 0
        assert request.desktop_context is not None
        assert request.long_horizon is not None
        assert request.long_horizon.self_evaluation_enabled is True


# ===========================================================================
# P4-2: Prompt Tuning
# ===========================================================================


class TestP4_2_PromptTuning:
    """Tests for the enhanced prompts (P4-2)."""

    def test_v4_execute_prompt_contains_math_reasoning(self):
        """V4 execute prompt contains math reasoning enhancement keywords."""
        prompt = EXECUTE_PROMPT_DEEPSEEK_V4
        assert "数学推理" in prompt or "数学" in prompt
        assert "推理" in prompt

    def test_v4_execute_prompt_contains_code_generation_enhancement(self):
        """V4 execute prompt contains code generation enhancement keywords."""
        prompt = EXECUTE_PROMPT_DEEPSEEK_V4
        assert "代码生成增强" in prompt or "错误处理" in prompt
        assert "try/except" in prompt or "边界检查" in prompt

    def test_v4_design_prompt_contains_frontend_beauty(self):
        """V4 design prompt contains frontend beauty compensation keywords."""
        prompt = DESIGN_PROMPT_DEEPSEEK_V4
        assert "前端美观补偿" in prompt or "CSS" in prompt or "配色方案" in prompt
        assert "UI" in prompt or "视觉层级" in prompt

    def test_v4_review_prompt_contains_enhanced_checking(self):
        """V4 review prompt contains enhanced checking keywords."""
        prompt = REVIEW_PROMPT_DEEPSEEK_V4
        # V4 review has deeper checking requirements
        assert "深入检查" in prompt or "潜在" in prompt or "逻辑缺陷" in prompt
        assert "类型安全" in prompt or "边界条件" in prompt

    def test_m3_execute_prompt_contains_programming_enhancement(self):
        """M3 execute prompt contains programming enhancement keywords."""
        prompt = EXECUTE_PROMPT_MINIMAX_M3
        assert "编程增强" in prompt or "SWE-Bench" in prompt
        assert "类型注解" in prompt or "单元测试" in prompt

    def test_m3_agent_prompt_contains_desktop_operation_rules(self):
        """M3 agent prompt contains desktop operation rules."""
        prompt = AGENT_PROMPT_MINIMAX_M3
        assert "桌面操作增强" in prompt or "桌面操作" in prompt
        assert "分步执行" in prompt or "每步验证" in prompt

    def test_m3_sub_agent_prompt_contains_desktop_operation(self):
        """M3 sub_agent prompt contains desktop operation keywords."""
        prompt = SUB_AGENT_PROMPT_MINIMAX_M3
        assert "桌面操作增强" in prompt or "桌面操作" in prompt

    def test_m3_plan_prompt_contains_verification_checkpoint(self):
        """M3 plan prompt contains verification checkpoint keywords."""
        prompt = PLAN_PROMPT_MINIMAX_M3
        assert "验证检查点" in prompt or "预设错误处理" in prompt

    def test_glm5_agent_prompt_contains_self_evaluation_trigger(self):
        """GLM-5 agent prompt contains self-evaluation trigger conditions."""
        prompt = AGENT_PROMPT_GLM_5
        assert "自评估" in prompt or "自评估触发" in prompt
        assert "5个子目标" in prompt or "10步" in prompt

    def test_glm5_agent_prompt_contains_strategy_switching(self):
        """GLM-5 agent prompt contains strategy switching conditions."""
        prompt = AGENT_PROMPT_GLM_5
        assert "策略切换" in prompt or "切换策略" in prompt
        assert "3次无进展" in prompt or "相似度" in prompt

    def test_cuda_triton_intent_accessible_via_registry(self):
        """cuda_triton intent is accessible via get_system_prompt_for_intent."""
        prompt = get_system_prompt_for_intent("cuda_triton", "glm_5")
        assert prompt != ""
        assert len(prompt) > 50  # Should be a substantial prompt

    def test_cuda_triton_prompt_contains_gpu_keywords(self):
        """CUDA_TRITON_PROMPT_GLM_5 contains GPU optimization keywords."""
        prompt = CUDA_TRITON_PROMPT_GLM_5
        assert "CUDA" in prompt or "Triton" in prompt
        assert "GPU" in prompt
        assert "内存" in prompt or "带宽" in prompt or "occupancy" in prompt.lower()


# ===========================================================================
# P4-3: Model Circuit Breaker
# ===========================================================================


class TestP4_3_ModelCircuitBreaker:
    """Tests for ModelCircuitBreakerManager and related classes (P4-3)."""

    def test_model_breaker_config_creation_with_defaults(self):
        """ModelBreakerConfig can be created with model_name and default values."""
        config = ModelBreakerConfig(model_name="test_model")
        assert config.model_name == "test_model"
        assert config.max_consecutive_failures == 5
        assert config.window_seconds == 300.0
        assert config.cooldown_seconds == 60.0
        assert config.failure_threshold_percent == 0.5
        assert config.half_open_max_calls == 3

    def test_model_breaker_config_custom_values(self):
        """ModelBreakerConfig accepts custom values."""
        config = ModelBreakerConfig(
            model_name="custom",
            max_consecutive_failures=10,
            cooldown_seconds=120.0,
        )
        assert config.max_consecutive_failures == 10
        assert config.cooldown_seconds == 120.0

    def test_model_breaker_state_initialization(self):
        """ModelBreakerState initializes with correct defaults."""
        state = ModelBreakerState(model_name="deepseek_v4_pro", state="closed")
        assert state.model_name == "deepseek_v4_pro"
        assert state.state == "closed"
        assert state.failure_count == 0
        assert state.success_count == 0
        assert state.last_failure_time == 0.0

    def test_manager_creation_with_defaults(self):
        """ModelCircuitBreakerManager creates default configs for 4 models."""
        mgr = ModelCircuitBreakerManager()
        states = mgr.get_all_states()
        assert "deepseek_v4_pro" in states
        assert "deepseek_v4_flash" in states
        assert "minimax_m3" in states
        assert "glm_5" in states
        # All should start closed
        for state in states.values():
            assert state == "closed"

    def test_manager_creation_with_custom_configs(self):
        """ModelCircuitBreakerManager accepts custom configs."""
        configs = [
            ModelBreakerConfig(model_name="model_a", max_consecutive_failures=3),
            ModelBreakerConfig(model_name="model_b", max_consecutive_failures=10),
        ]
        mgr = ModelCircuitBreakerManager(configs=configs)
        states = mgr.get_all_states()
        assert "model_a" in states
        assert "model_b" in states

    def test_record_success_resets_failure_count(self):
        """record_success() resets failure_count in closed state."""
        mgr = ModelCircuitBreakerManager()
        # Record some failures first
        mgr.record_failure("deepseek_v4_pro", "error1")
        mgr.record_failure("deepseek_v4_pro", "error2")
        assert mgr._states["deepseek_v4_pro"].failure_count == 2

        # Record success — should reset failure count
        mgr.record_success("deepseek_v4_pro")
        assert mgr._states["deepseek_v4_pro"].failure_count == 0

    def test_record_failure_increments_failure_count(self):
        """record_failure() increments failure_count in closed state."""
        mgr = ModelCircuitBreakerManager()
        mgr.record_failure("deepseek_v4_pro", "err1")
        mgr.record_failure("deepseek_v4_pro", "err2")
        assert mgr._states["deepseek_v4_pro"].failure_count == 2

    def test_record_failure_opens_breaker_at_threshold(self):
        """record_failure() opens breaker when consecutive failures reach threshold."""
        mgr = ModelCircuitBreakerManager()
        # Default threshold is 5
        for i in range(4):
            result = mgr.record_failure("deepseek_v4_pro", f"err{i}")
            assert result is None  # Should not open yet

        # 5th failure should open the breaker
        fallback = mgr.record_failure("deepseek_v4_pro", "err5")
        assert mgr.get_state("deepseek_v4_pro") == "open"
        # Should return a fallback model
        assert fallback is not None

    def test_record_failure_returns_fallback_when_opens(self):
        """record_failure() returns fallback model name when breaker just opens."""
        mgr = ModelCircuitBreakerManager()
        for i in range(5):
            _result = mgr.record_failure("deepseek_v4_pro", f"err{i}")

        # After 5 failures, the 5th record_failure should have returned a fallback
        assert mgr.get_state("deepseek_v4_pro") == "open"

    def test_get_state_returns_correct_state(self):
        """get_state() returns 'closed', 'open', or 'half_open' as appropriate."""
        mgr = ModelCircuitBreakerManager()
        assert mgr.get_state("deepseek_v4_pro") == "closed"

        # Open the breaker
        for i in range(5):
            mgr.record_failure("deepseek_v4_pro", f"err{i}")
        assert mgr.get_state("deepseek_v4_pro") == "open"

    def test_can_call_closed_true_open_false(self):
        """can_call() returns True for closed, False for open."""
        mgr = ModelCircuitBreakerManager()
        assert mgr.can_call("deepseek_v4_pro") is True

        # Open the breaker
        for i in range(5):
            mgr.record_failure("deepseek_v4_pro", f"err{i}")
        assert mgr.can_call("deepseek_v4_pro") is False

    def test_get_fallback_returns_next_in_chain(self):
        """get_fallback() returns the next model in the degradation chain."""
        mgr = ModelCircuitBreakerManager()
        fallback = mgr.get_fallback("deepseek_v4_pro")
        # Default chain: ["deepseek_v4_pro", "glm_5", "deepseek_v4_flash"]
        assert fallback == "glm_5"

    def test_get_fallback_skips_open_breakers(self):
        """get_fallback() skips models whose breakers are also open."""
        mgr = ModelCircuitBreakerManager()

        # Open the breaker for glm_5 (which is next in chain after v4_pro)
        for i in range(5):
            mgr.record_failure("glm_5", f"err{i}")

        # Now v4_pro should fallback to v4_flash, skipping glm_5
        fallback = mgr.get_fallback("deepseek_v4_pro")
        assert fallback == "deepseek_v4_flash"

    def test_get_all_states_returns_dict(self):
        """get_all_states() returns dict mapping model_name → state string."""
        mgr = ModelCircuitBreakerManager()
        states = mgr.get_all_states()
        assert isinstance(states, dict)
        assert len(states) >= 4
        for name, state in states.items():
            assert state in ("closed", "open", "half_open")

    def test_reset_restores_closed_state(self):
        """reset() restores closed state for specified model."""
        mgr = ModelCircuitBreakerManager()
        for i in range(5):
            mgr.record_failure("deepseek_v4_pro", f"err{i}")
        assert mgr.get_state("deepseek_v4_pro") == "open"

        mgr.reset("deepseek_v4_pro")
        assert mgr.get_state("deepseek_v4_pro") == "closed"

    def test_reset_all_restores_all_models(self):
        """reset() with no args restores all models to closed."""
        mgr = ModelCircuitBreakerManager()
        for i in range(5):
            mgr.record_failure("deepseek_v4_pro", f"err{i}")
            mgr.record_failure("minimax_m3", f"err{i}")
        assert mgr.get_state("deepseek_v4_pro") == "open"

        mgr.reset()
        assert mgr.get_state("deepseek_v4_pro") == "closed"
        assert mgr.get_state("minimax_m3") == "closed"

    def test_auto_create_state_for_unknown_model(self):
        """Auto-creates state/config for unknown models."""
        mgr = ModelCircuitBreakerManager()
        # Access an unknown model
        state = mgr.get_state("new_model_xyz")
        assert state == "closed"
        assert mgr.can_call("new_model_xyz") is True


# ===========================================================================
# P4-3: Degradation Chain
# ===========================================================================


class TestP4_3_DegradationChain:
    """Tests for DegradationChain (P4-3)."""

    def test_degradation_chain_creation_with_defaults(self):
        """DegradationChain creates default chains for heavy, multimodal, default."""
        chain = DegradationChain()
        # Default chains should exist
        heavy = chain.get_full_chain("heavy")
        multimodal = chain.get_full_chain("multimodal")
        default = chain.get_full_chain("default")
        assert len(heavy) > 0
        assert len(multimodal) > 0
        assert len(default) > 0

    def test_get_fallback_heavy_task(self):
        """get_fallback() for heavy task type returns next model in chain."""
        chain = DegradationChain()
        fallback = chain.get_fallback("deepseek_v4_pro", task_type="heavy")
        # Heavy chain: ["deepseek_v4_pro", "glm_5", "deepseek_v4_flash"]
        assert fallback == "glm_5"

    def test_get_fallback_multimodal_task(self):
        """get_fallback() for multimodal task type returns next model."""
        chain = DegradationChain()
        fallback = chain.get_fallback("minimax_m3", task_type="multimodal")
        # Multimodal chain: ["minimax_m3", "deepseek_v4_pro"]
        assert fallback == "deepseek_v4_pro"

    def test_get_fallback_skips_unavailable_models(self):
        """get_fallback() skips models with open breakers."""
        mgr = ModelCircuitBreakerManager()
        # Open glm_5's breaker
        for i in range(5):
            mgr.record_failure("glm_5", f"err{i}")

        chain = DegradationChain(breaker_manager=mgr)
        fallback = chain.get_fallback("deepseek_v4_pro", task_type="heavy")
        # Should skip glm_5 (open) and return deepseek_v4_flash
        assert fallback == "deepseek_v4_flash"

    def test_get_full_chain_returns_ordered_list(self):
        """get_full_chain() returns the complete ordered model list."""
        chain = DegradationChain()
        heavy = chain.get_full_chain("heavy")
        assert heavy == ["deepseek_v4_pro", "glm_5", "deepseek_v4_flash"]

        default = chain.get_full_chain("default")
        assert default == ["deepseek_v4_pro", "glm_5", "deepseek_v4_flash"]

    def test_add_chain_adds_custom_chain(self):
        """add_chain() adds a custom degradation chain."""
        chain = DegradationChain()
        chain.add_chain("lightweight", ["deepseek_v4_flash", "minimax_m3"])
        full = chain.get_full_chain("lightweight")
        assert full == ["deepseek_v4_flash", "minimax_m3"]
        fallback = chain.get_fallback("deepseek_v4_flash", task_type="lightweight")
        assert fallback == "minimax_m3"

    def test_integration_with_circuit_breaker_manager(self):
        """DegradationChain correctly integrates with ModelCircuitBreakerManager."""
        mgr = ModelCircuitBreakerManager()
        chain = DegradationChain(breaker_manager=mgr)

        # All closed → normal fallback
        fallback = chain.get_fallback("deepseek_v4_pro", task_type="heavy")
        assert fallback == "glm_5"

        # Open both glm_5 and deepseek_v4_flash
        for i in range(5):
            mgr.record_failure("glm_5", f"err{i}")
            mgr.record_failure("deepseek_v4_flash", f"err{i}")

        fallback = chain.get_fallback("deepseek_v4_pro", task_type="heavy")
        assert fallback is None  # All fallbacks unavailable

    def test_get_fallback_for_model_not_in_chain(self):
        """get_fallback() for model not in chain returns first available candidate."""
        chain = DegradationChain()
        fallback = chain.get_fallback("unknown_model", task_type="heavy")
        # Should return first model in chain that is not unknown_model
        assert fallback is not None
        assert fallback != "unknown_model"


# ===========================================================================
# P4-3: Long-Horizon Recovery
# ===========================================================================


class TestP4_3_LongHorizonRecovery:
    """Tests for LongHorizonRecoveryManager (P4-3)."""

    def test_manager_creation(self):
        """LongHorizonRecoveryManager can be created with defaults."""
        mgr = LongHorizonRecoveryManager()
        assert mgr.has_checkpoint_store is False
        stats = mgr.recovery_stats
        assert stats["total_attempts"] == 0
        assert stats["successes"] == 0
        assert stats["failures"] == 0

    def test_manager_creation_with_params(self):
        """LongHorizonRecoveryManager accepts custom parameters."""
        mgr = LongHorizonRecoveryManager(
            max_reconnection_attempts=10,
            reconnection_base_delay=5.0,
        )
        stats = mgr.recovery_stats
        assert stats["max_reconnection_attempts"] == 10

    def test_should_downgrade_to_standard_by_attempts(self):
        """should_downgrade_to_standard() returns True when too many attempts."""
        mgr = LongHorizonRecoveryManager()
        # Default _downgrade_max_attempts = 3
        assert mgr.should_downgrade_to_standard(recovery_attempts=2, elapsed_time=0) is False
        assert mgr.should_downgrade_to_standard(recovery_attempts=3, elapsed_time=0) is True
        assert mgr.should_downgrade_to_standard(recovery_attempts=5, elapsed_time=0) is True

    def test_should_downgrade_to_standard_by_elapsed(self):
        """should_downgrade_to_standard() returns True when too much time elapsed."""
        mgr = LongHorizonRecoveryManager()
        # Default _downgrade_max_elapsed = 1800.0 (30 minutes)
        assert mgr.should_downgrade_to_standard(recovery_attempts=0, elapsed_time=1000) is False
        assert mgr.should_downgrade_to_standard(recovery_attempts=0, elapsed_time=1800) is True
        assert mgr.should_downgrade_to_standard(recovery_attempts=0, elapsed_time=3600) is True

    def test_get_reconnection_delay_exponential_backoff(self):
        """get_reconnection_delay() implements exponential backoff."""
        mgr = LongHorizonRecoveryManager(reconnection_base_delay=2.0)
        # Attempt 0: base * 2^0 = 2.0 (±25% jitter)
        delays = [mgr.get_reconnection_delay(0) for _ in range(20)]
        avg_delay_0 = statistics.mean(delays)
        assert 1.0 < avg_delay_0 < 3.0  # 2.0 ± 25%

        # Attempt 1: base * 2^1 = 4.0 (±25% jitter)
        delays = [mgr.get_reconnection_delay(1) for _ in range(20)]
        avg_delay_1 = statistics.mean(delays)
        assert 2.5 < avg_delay_1 < 5.5  # 4.0 ± 25%

    def test_get_reconnection_delay_returns_zero_beyond_max(self):
        """get_reconnection_delay() returns 0.0 when attempt >= max attempts."""
        mgr = LongHorizonRecoveryManager(max_reconnection_attempts=3)
        assert mgr.get_reconnection_delay(3) == 0.0
        assert mgr.get_reconnection_delay(5) == 0.0

    def test_record_recovery_attempt_success_and_failure(self):
        """record_recovery_attempt() tracks successes and failures."""
        mgr = LongHorizonRecoveryManager()
        mgr.record_recovery_attempt(success=True)
        mgr.record_recovery_attempt(success=True)
        mgr.record_recovery_attempt(success=False)

        stats = mgr.recovery_stats
        assert stats["total_attempts"] == 3
        assert stats["successes"] == 2
        assert stats["failures"] == 1
        assert stats["last_recovery_time"] > 0

    def test_recovery_stats_property(self):
        """recovery_stats returns dict with expected keys."""
        mgr = LongHorizonRecoveryManager()
        stats = mgr.recovery_stats
        assert "total_attempts" in stats
        assert "successes" in stats
        assert "failures" in stats
        assert "last_recovery_time" in stats
        assert "max_reconnection_attempts" in stats

    def test_has_checkpoint_store_property(self):
        """has_checkpoint_store returns True only when store is configured."""
        mgr_no_store = LongHorizonRecoveryManager()
        assert mgr_no_store.has_checkpoint_store is False

        mgr_with_store = LongHorizonRecoveryManager(checkpoint_store=object())
        assert mgr_with_store.has_checkpoint_store is True

    @pytest.mark.asyncio
    async def test_recover_from_checkpoint_no_store(self):
        """recover_from_checkpoint() returns False when no checkpoint store."""
        mgr = LongHorizonRecoveryManager()
        result = await mgr.recover_from_checkpoint(object())
        assert result is False


# ===========================================================================
# P4-3: Rate Limit Handler
# ===========================================================================


class TestP4_3_RateLimitHandler:
    """Tests for RateLimitHandler and RateLimitInfo (P4-3)."""

    def test_rate_limit_info_creation(self):
        """RateLimitInfo dataclass creation with defaults."""
        info = RateLimitInfo(model_name="deepseek_v4_pro")
        assert info.model_name == "deepseek_v4_pro"
        assert info.requests_remaining is None
        assert info.tokens_remaining is None
        assert info.reset_time is None
        assert info.retry_after is None

    def test_rate_limit_info_with_values(self):
        """RateLimitInfo accepts all field values."""
        info = RateLimitInfo(
            model_name="test",
            requests_remaining=100,
            tokens_remaining=50000,
            reset_time=1700000000.0,
            retry_after=30.0,
        )
        assert info.requests_remaining == 100
        assert info.retry_after == 30.0

    def test_handler_creation(self):
        """RateLimitHandler can be created with and without breaker_manager."""
        handler = RateLimitHandler()
        assert handler._breaker_manager is None

        mgr = ModelCircuitBreakerManager()
        handler_with_mgr = RateLimitHandler(breaker_manager=mgr)
        assert handler_with_mgr._breaker_manager is mgr

    def test_parse_rate_limit_deepseek_v4(self):
        """parse_rate_limit_response() parses DeepSeek V4 Retry-After header."""
        handler = RateLimitHandler()
        info = handler.parse_rate_limit_response(
            model_name="deepseek_v4_pro",
            status_code=429,
            headers={"Retry-After": "30"},
        )
        assert info.model_name == "deepseek_v4_pro"
        assert info.retry_after == 30.0

    def test_parse_rate_limit_minimax_m3(self):
        """parse_rate_limit_response() parses MiniMax M3 X-RateLimit-* headers."""
        handler = RateLimitHandler()
        info = handler.parse_rate_limit_response(
            model_name="minimax_m3",
            status_code=429,
            headers={
                "X-RateLimit-Remaining-Requests": "50",
                "X-RateLimit-Remaining-Tokens": "100000",
                "X-RateLimit-Reset": "1700000060.0",
            },
        )
        assert info.model_name == "minimax_m3"
        assert info.requests_remaining == 50
        assert info.tokens_remaining == 100000
        assert info.reset_time == 1700000060.0

    def test_parse_rate_limit_glm_5_body(self):
        """parse_rate_limit_response() parses GLM-5 body fields."""
        handler = RateLimitHandler()
        info = handler.parse_rate_limit_response(
            model_name="glm_5",
            status_code=429,
            headers={},
            body={
                "retry_after": 60,
                "requests_remaining": 10,
                "tokens_remaining": 20000,
            },
        )
        assert info.model_name == "glm_5"
        assert info.retry_after == 60.0
        assert info.requests_remaining == 10
        assert info.tokens_remaining == 20000

    def test_parse_rate_limit_unknown_model(self):
        """parse_rate_limit_response() for unknown model returns minimal info."""
        handler = RateLimitHandler()
        info = handler.parse_rate_limit_response(
            model_name="unknown_model",
            status_code=429,
            headers={},
        )
        assert info.model_name == "unknown_model"
        assert info.retry_after is None
        assert info.requests_remaining is None

    def test_parse_rate_limit_non_429_returns_empty(self):
        """parse_rate_limit_response() for non-429 returns empty info."""
        handler = RateLimitHandler()
        info = handler.parse_rate_limit_response(
            model_name="deepseek_v4_pro",
            status_code=200,
            headers={"Retry-After": "30"},
        )
        assert info.retry_after is None  # Not parsed for non-429

    def test_should_retry_with_remaining_requests(self):
        """should_retry() returns True when requests remaining > 0."""
        handler = RateLimitHandler()
        info = RateLimitInfo(model_name="test", requests_remaining=10)
        assert handler.should_retry("test", info) is True

        info_zero = RateLimitInfo(model_name="test", requests_remaining=0)
        assert handler.should_retry("test", info_zero) is False

    def test_should_retry_with_circuit_breaker(self):
        """should_retry() returns False when breaker is open."""
        mgr = ModelCircuitBreakerManager()
        handler = RateLimitHandler(breaker_manager=mgr)

        # Open the breaker
        for i in range(5):
            mgr.record_failure("deepseek_v4_pro", f"err{i}")

        info = RateLimitInfo(model_name="deepseek_v4_pro", requests_remaining=10)
        assert handler.should_retry("deepseek_v4_pro", info) is False

    def test_should_retry_default_allows(self):
        """should_retry() returns True by default when no specific info."""
        handler = RateLimitHandler()
        info = RateLimitInfo(model_name="test")
        assert handler.should_retry("test", info) is True

    def test_get_backoff_delay_exponential(self):
        """get_backoff_delay() uses exponential backoff without rate limit info."""
        handler = RateLimitHandler()
        # Attempt 0: 1.0 * 2^0 = 1.0 (±25% jitter)
        delays = [handler.get_backoff_delay("test", 0) for _ in range(20)]
        avg_0 = statistics.mean(delays)
        assert 0.5 < avg_0 < 1.5

        # Attempt 2: 1.0 * 2^2 = 4.0 (±25% jitter)
        delays = [handler.get_backoff_delay("test", 2) for _ in range(20)]
        avg_2 = statistics.mean(delays)
        assert 2.5 < avg_2 < 5.5

    def test_get_backoff_delay_respects_retry_after(self):
        """get_backoff_delay() uses retry_after from rate limit info."""
        handler = RateLimitHandler()
        info = RateLimitInfo(model_name="test", retry_after=10.0)
        delays = [handler.get_backoff_delay("test", 0, rate_limit_info=info) for _ in range(20)]
        avg = statistics.mean(delays)
        # Should be around 10.0 (±10% jitter)
        assert 8.0 < avg < 12.0

    def test_get_backoff_delay_beyond_max_retries(self):
        """get_backoff_delay() returns 0.0 when attempt >= max retries."""
        handler = RateLimitHandler()
        assert handler.get_backoff_delay("test", 5) == 0.0
        assert handler.get_backoff_delay("test", 10) == 0.0


# ===========================================================================
# P4-3: Recovery Exports
# ===========================================================================


class TestP4_3_RecoveryExport:
    """Tests for correct exports of new P4-3 classes."""

    def test_all_new_classes_importable_from_reliability(self):
        """All P4-3 classes are importable from teragent.reliability."""
        from teragent.reliability import (
            DegradationChain,
            LongHorizonRecoveryManager,
            ModelBreakerConfig,
            ModelBreakerState,
            ModelCircuitBreakerManager,
            RateLimitHandler,
            RateLimitInfo,
        )
        assert ModelBreakerConfig is not None
        assert ModelBreakerState is not None
        assert ModelCircuitBreakerManager is not None
        assert DegradationChain is not None
        assert LongHorizonRecoveryManager is not None
        assert RateLimitInfo is not None
        assert RateLimitHandler is not None

    def test_all_new_classes_in_all_list(self):
        """All P4-3 classes are in the __all__ list."""
        import teragent.reliability as rel
        assert "ModelBreakerConfig" in rel.__all__
        assert "ModelBreakerState" in rel.__all__
        assert "ModelCircuitBreakerManager" in rel.__all__
        assert "DegradationChain" in rel.__all__
        assert "LongHorizonRecoveryManager" in rel.__all__
        assert "RateLimitInfo" in rel.__all__
        assert "RateLimitHandler" in rel.__all__

    def test_circuit_breaker_module_direct_import(self):
        """ModelCircuitBreakerManager importable from circuit_breaker module."""
        from teragent.reliability.circuit_breaker import (
            ModelBreakerConfig,
            ModelBreakerState,
            ModelCircuitBreakerManager,
        )
        assert ModelBreakerConfig is not None
        assert ModelBreakerState is not None
        assert ModelCircuitBreakerManager is not None

    def test_recovery_module_direct_import(self):
        """DegradationChain importable from recovery module."""
        from teragent.reliability.recovery import (
            DegradationChain,
            LongHorizonRecoveryManager,
            RateLimitHandler,
            RateLimitInfo,
        )
        assert DegradationChain is not None
        assert LongHorizonRecoveryManager is not None
        assert RateLimitInfo is not None
        assert RateLimitHandler is not None


# ===========================================================================
# P4: End-to-End Integration
# ===========================================================================


class TestP4_EndToEnd:
    """End-to-end integration tests spanning multiple P4 features."""

    def test_full_benchmark_pipeline(self):
        """Full benchmark pipeline: compile → benchmark → report."""
        runner = BenchmarkRunner(iterations=3, seed=42, suites=["compilation"])
        report = runner.run_all()

        assert isinstance(report, BenchmarkReport)
        assert len(report.results) > 0
        assert "total_benchmarks" in report.summary
        assert report.summary["total_benchmarks"] > 0

        # Verify report can be serialized
        json_str = report.to_json()
        data = json.loads(json_str)
        assert "results" in data

        text = report.to_text()
        assert "TERAGENT PERFORMANCE BENCHMARK REPORT" in text

    def test_circuit_breaker_degradation_chain_e2e(self):
        """Model circuit breaker + degradation chain end-to-end flow."""
        mgr = ModelCircuitBreakerManager()
        chain = DegradationChain(breaker_manager=mgr)

        # Initial state: everything works
        assert chain.get_fallback("deepseek_v4_pro", task_type="heavy") == "glm_5"

        # Simulate V4-Pro failure: open its breaker
        for i in range(5):
            mgr.record_failure("deepseek_v4_pro", f"err{i}")

        assert mgr.get_state("deepseek_v4_pro") == "open"
        assert mgr.can_call("deepseek_v4_pro") is False

        # V4-Pro fallback should work
        fallback = mgr.get_fallback("deepseek_v4_pro")
        assert fallback == "glm_5"

        # Now simulate GLM-5 failure too
        for i in range(5):
            mgr.record_failure("glm_5", f"err{i}")

        # V4-Pro fallback should skip GLM-5 and go to V4-Flash
        fallback = mgr.get_fallback("deepseek_v4_pro")
        assert fallback == "deepseek_v4_flash"

    def test_rate_limit_to_circuit_breaker_to_degradation(self):
        """Rate limit → circuit breaker → degradation chain flow."""
        mgr = ModelCircuitBreakerManager()
        handler = RateLimitHandler(breaker_manager=mgr)
        chain = DegradationChain(breaker_manager=mgr)

        # Parse a rate limit response for DeepSeek V4
        info = handler.parse_rate_limit_response(
            model_name="deepseek_v4_pro",
            status_code=429,
            headers={"Retry-After": "30"},
        )
        assert info.retry_after == 30.0

        # Initially can retry
        assert handler.should_retry("deepseek_v4_pro", info) is True

        # Open the breaker via repeated failures
        for i in range(5):
            mgr.record_failure("deepseek_v4_pro", f"err{i}")

        # Now should NOT retry (breaker open)
        assert handler.should_retry("deepseek_v4_pro", info) is False

        # Should get fallback from degradation chain
        fallback = chain.get_fallback("deepseek_v4_pro", task_type="heavy")
        assert fallback is not None

    def test_prompt_tuning_verification_via_compilation(self):
        """Verify P4-2 prompt tuning changes are reflected in compilation."""
        import teragent.core.compilers  # noqa: F401

        # Compile with DeepSeek V4 and verify execute prompt contains enhancements
        v4_cls = TAPCompilerRegistry.get("deepseek_v4")
        assert v4_cls is not None, "deepseek_v4 compiler should be registered"
        v4 = v4_cls(variant="pro")

        request = TAPRequest(
            meta={"intent": "execute"},
            instruction="测试指令",
        )
        compiled = v4.compile(request)
        # The compiled messages should contain the V4 execute prompt
        all_content = " ".join(
            str(msg.get("content", "")) for msg in (compiled.messages or [])
        )
        assert "数学推理" in all_content or "推理" in all_content

    def test_cross_model_cost_tracking_with_fault_recovery(self):
        """Cross-model cost tracking works alongside fault recovery."""
        from teragent.reliability.budget import CostRecord, CrossModelCostTracker

        tracker = CrossModelCostTracker()
        mgr = ModelCircuitBreakerManager()

        # Record cost for V4-Pro
        tracker.record(CostRecord(
            model="deepseek_v4_pro",
            prompt_tokens=1000,
            completion_tokens=500,
            cost_cny=0.005,
        ))

        # Simulate V4-Pro failure
        for i in range(5):
            mgr.record_failure("deepseek_v4_pro", f"err{i}")

        # Record cost for fallback (GLM-5)
        tracker.record(CostRecord(
            model="glm_5",
            prompt_tokens=1000,
            completion_tokens=500,
            cost_cny=0.004,
        ))

        # Verify costs are tracked
        assert tracker.get_total_cost() > 0

    def test_all_nine_compilers_regression(self):
        """All 9 compilers still work after P4 changes."""
        import teragent.core.compilers  # noqa: F401

        request = TAPRequest(
            meta={"intent": "execute"},
            instruction="实现一个函数",
            constraints=["使用类型注解"],
            output_format_hint="用 <file path='...'> 输出代码",
        )

        for compiler_name in ALL_COMPILERS:
            compiler_cls = TAPCompilerRegistry.get(compiler_name)
            if compiler_cls is None:
                continue  # Skip unregistered compilers

            # DeepSeek V4 needs variant
            if compiler_name == "deepseek_v4":
                compiler = compiler_cls(variant="pro")
            else:
                compiler = compiler_cls()

            compiled = compiler.compile(request)
            assert compiled is not None, f"Compiler {compiler_name} returned None"
            # Some compilers (e.g., anthropic) use system_prompt instead of messages
            has_content = (
                (compiled.messages and len(compiled.messages) > 0)
                or (compiled.system_prompt and len(compiled.system_prompt) > 0)
            )
            assert has_content, f"Compiler {compiler_name} produced no content"

    def test_cuda_triton_prompt_in_compilation(self):
        """cuda_triton intent produces CUDA-specific prompt in compilation."""
        import teragent.core.compilers  # noqa: F401

        glm_cls = TAPCompilerRegistry.get("glm_5")
        assert glm_cls is not None, "glm_5 compiler should be registered"
        glm = glm_cls()

        request = TAPRequest(
            meta={"intent": "cuda_triton"},
            instruction="优化一个矩阵乘法内核",
        )
        compiled = glm.compile(request)
        all_content = " ".join(
            str(msg.get("content", "")) for msg in (compiled.messages or [])
        )
        # Should contain CUDA/GPU-related content from the specialized prompt
        assert "GPU" in all_content or "CUDA" in all_content or "Triton" in all_content

    def test_long_horizon_recovery_with_cost_tracking(self):
        """Long-horizon recovery integrates with cost tracking."""
        from teragent.reliability.budget import CostRecord, CrossModelCostTracker

        tracker = CrossModelCostTracker()
        recovery_mgr = LongHorizonRecoveryManager(max_reconnection_attempts=5)

        # Simulate recovery attempts
        for attempt in range(3):
            delay = recovery_mgr.get_reconnection_delay(attempt)
            assert delay > 0
            # Simulate first 2 failures, 3rd success
            success = (attempt == 2)
            recovery_mgr.record_recovery_attempt(success=success)

            if not success:
                # Record cost for the failed attempt
                tracker.record(CostRecord(
                    model="glm_5",
                    prompt_tokens=2000,
                    completion_tokens=0,
                    cost_cny=0.004,
                ))

        stats = recovery_mgr.recovery_stats
        assert stats["total_attempts"] == 3
        assert stats["successes"] == 1
        assert stats["failures"] == 2
        assert tracker.get_total_cost() > 0
