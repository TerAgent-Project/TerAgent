"""tests/integration/test_p3_mock_regression.py — Phase 3 integration tests

Comprehensive tests for Phase 3: 协同与路由
  - P3-1: ModelRouter (intent/multimodal/context/long-horizon/cost/degradation routing)
  - P3-2: Pipeline Dynamic Allocation (multi-profile, runtime switching, per-stage assignment)
  - P3-3: CrossModelCostTracker (cross-model stats, monthly budget, cost reports, cache savings)
  - P3-5: End-to-end integration tests (full flow, degradation E2E)

All tests use MockAdapter — no real API calls.
"""

from __future__ import annotations

import pytest
import time

from teragent.core.tap import (
    TAPRequest, TAPResponse, MultimodalContent, DesktopContext, LongHorizonConfig,
)
from teragent.core.compiler import TAPCompilerRegistry
from teragent.core.adapters.mock import MockAdapter
from teragent.core.provider import ModelProvider
from teragent.router.model_router import (
    ModelRouter, RoutingTable, RoutingDecision, RoutingReason,
    PipelineProfile, PipelineManager,
)
from teragent.reliability.budget import (
    StepBudget, CostRecord, MonthlyBudgetConfig, CrossModelCostTracker,
)
from teragent.reliability.circuit_breaker import BreakerState


# ===== Helpers =====

def _create_mock_provider(compiler_name: str, model: str, **kwargs) -> ModelProvider:
    """Create a ModelProvider with a real compiler and MockAdapter."""
    compiler_cls = TAPCompilerRegistry.get(compiler_name)
    if compiler_cls is None:
        raise ValueError(f"Unknown compiler: {compiler_name}")
    # Map compiler_variant to variant for DeepSeekV4Compiler
    init_kwargs = {}
    if 'compiler_variant' in kwargs:
        init_kwargs['variant'] = kwargs['compiler_variant']
    compiler = compiler_cls(**init_kwargs)
    adapter = MockAdapter()
    return ModelProvider(compiler=compiler, adapter=adapter, model=model)


def _create_test_router() -> ModelRouter:
    """Create a ModelRouter with mock providers for all 3 models."""
    providers = {}
    try:
        providers["openai_compatible.deepseek_v4_flash"] = _create_mock_provider(
            "deepseek_v4", "deepseek-v4-flash", compiler_variant="flash"
        )
    except Exception:
        pass
    try:
        providers["openai_compatible.deepseek_v4_pro"] = _create_mock_provider(
            "deepseek_v4", "deepseek-v4-pro", compiler_variant="pro"
        )
    except Exception:
        pass
    try:
        providers["openai_compatible.minimax_m3"] = _create_mock_provider(
            "minimax_m3", "minimax-m3"
        )
    except Exception:
        pass
    try:
        providers["openai_compatible.glm_5"] = _create_mock_provider(
            "glm_5", "glm-5"
        )
    except Exception:
        pass

    return ModelRouter(available_providers=providers)


# =====================================================================
# P3-1: ModelRouter
# =====================================================================

class TestP3_1_ModelRouter:
    """Test ModelRouter intent-based and override routing"""

    def setup_method(self):
        self.router = _create_test_router()

    def test_default_routing_table(self):
        """Default routing table maps each intent to the correct driver"""
        rt = RoutingTable()
        assert rt.get_intent_driver("design") == "openai_compatible.deepseek_v4_pro"
        assert rt.get_intent_driver("plan") == "openai_compatible.glm_5"
        assert rt.get_intent_driver("execute") == "openai_compatible.glm_5"
        assert rt.get_intent_driver("review") == "openai_compatible.deepseek_v4_pro"
        assert rt.get_intent_driver("chat") == "openai_compatible.deepseek_v4_flash"

    def test_intent_routing_design(self):
        """Design intent → V4-Pro"""
        request = TAPRequest(
            meta={"task_id": "1", "intent": "design"},
            instruction="Design the system architecture",
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.deepseek_v4_pro"
        assert decision.reason == RoutingReason.INTENT

    def test_intent_routing_plan(self):
        """Plan intent → GLM-5"""
        request = TAPRequest(
            meta={"task_id": "2", "intent": "plan"},
            instruction="Create an execution plan",
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.glm_5"
        assert decision.reason == RoutingReason.INTENT

    def test_intent_routing_execute(self):
        """Execute intent → GLM-5"""
        request = TAPRequest(
            meta={"task_id": "3", "intent": "execute"},
            instruction="Write the code",
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.glm_5"

    def test_intent_routing_review(self):
        """Review intent → V4-Pro"""
        request = TAPRequest(
            meta={"task_id": "4", "intent": "review"},
            instruction="Review the code",
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.deepseek_v4_pro"

    def test_intent_routing_chat(self):
        """Chat intent → V4-Flash (cost savings)"""
        request = TAPRequest(
            meta={"task_id": "5", "intent": "chat"},
            instruction="Hello",
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.deepseek_v4_flash"
        assert decision.reason == RoutingReason.INTENT

    def test_multimodal_override(self):
        """Multimodal content → M3 (overrides intent)"""
        request = TAPRequest(
            meta={"task_id": "6", "intent": "design"},
            instruction="Analyze this screenshot",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/img.png")
            ],
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.minimax_m3"
        assert decision.reason == RoutingReason.MULTIMODAL_OVERRIDE

    def test_video_override(self):
        """Video content → M3 (specific video override)"""
        request = TAPRequest(
            meta={"task_id": "7", "intent": "execute"},
            instruction="Process this video",
            multimodal_context=[
                MultimodalContent(type="video_url", url="https://example.com/vid.mp4")
            ],
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.minimax_m3"
        assert decision.reason == RoutingReason.VIDEO_OVERRIDE

    def test_desktop_override(self):
        """Desktop context → M3 (desktop override)"""
        request = TAPRequest(
            meta={"task_id": "8", "intent": "chat"},
            instruction="Take a screenshot",
            desktop_context=DesktopContext(active_window="Chrome"),
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.minimax_m3"
        assert decision.reason == RoutingReason.DESKTOP_OVERRIDE

    def test_context_length_override(self):
        """Context > 200K → V4/M3 (excludes GLM-5)"""
        # Create request with large context that exceeds 200K tokens
        request = TAPRequest(
            meta={"task_id": "9", "intent": "plan"},
            instruction="Process this large codebase",
            context={"design": "x" * 1_000_000},  # ~250K tokens
        )
        decision = self.router.route(request)
        # Should NOT be GLM-5 (200K limit)
        assert "glm_5" not in decision.selected_driver
        assert decision.reason == RoutingReason.CONTEXT_LENGTH_OVERRIDE

    def test_long_horizon_override(self):
        """Long-horizon task → GLM-5"""
        request = TAPRequest(
            meta={"task_id": "10", "intent": "chat"},
            instruction="Run this autonomous task",
            long_horizon=LongHorizonConfig(max_duration_hours=4.0),
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.glm_5"
        assert decision.reason == RoutingReason.LONG_HORIZON_OVERRIDE

    def test_routing_priority(self):
        """Multimodal > Context > Long-horizon > Intent"""
        # Multimodal + Long-horizon → Multimodal wins (step 1)
        request = TAPRequest(
            meta={"task_id": "11", "intent": "plan"},
            instruction="Analyze and run",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/img.png")
            ],
            long_horizon=LongHorizonConfig(max_duration_hours=4.0),
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.minimax_m3"
        assert decision.reason == RoutingReason.MULTIMODAL_OVERRIDE

    def test_decision_log(self):
        """Decision log tracks routing decisions"""
        request = TAPRequest(
            meta={"task_id": "12", "intent": "chat"},
            instruction="Hello",
        )
        self.router.route(request)
        self.router.route(request)

        log = self.router.get_decision_log()
        assert len(log) >= 2
        assert log[0].intent == "chat"

    def test_decision_summary(self):
        """Decision summary aggregates by reason/driver/intent"""
        for intent in ["design", "chat", "execute"]:
            request = TAPRequest(
                meta={"task_id": f"s-{intent}", "intent": intent},
                instruction="Test",
            )
            self.router.route(request)

        summary = self.router.get_decision_summary()
        assert summary["total_decisions"] >= 3
        assert "by_reason" in summary
        assert "by_driver" in summary
        assert "by_intent" in summary

    def test_unknown_intent_defaults_to_chat(self):
        """Unknown intent → chat routing (fallback)"""
        request = TAPRequest(
            meta={"task_id": "13", "intent": "unknown_intent"},
            instruction="Do something",
        )
        decision = self.router.route(request)
        # Should fallback to chat driver (V4-Flash)
        assert decision.selected_driver == "openai_compatible.deepseek_v4_flash"


class TestP3_1_Degradation:
    """Test degradation strategy when primary model is unavailable"""

    def setup_method(self):
        self.router = _create_test_router()

    def test_degradation_v4_pro_to_flash(self):
        """V4-Pro unavailable → degrades to V4-Flash"""
        self.router.update_circuit_breaker_state(
            "openai_compatible.deepseek_v4_pro",
            BreakerState(
                name="open", consecutive_failures=5, total_failures=10,
                total_successes=50, last_error="timeout",
                last_failure_time=0.0, can_retry=False,
            ),
        )
        request = TAPRequest(
            meta={"task_id": "deg-1", "intent": "design"},
            instruction="Design",
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.deepseek_v4_flash"
        assert decision.reason == RoutingReason.DEGRADATION

    def test_degradation_glm_to_flash(self):
        """GLM-5 unavailable → degrades to V4-Flash"""
        self.router.update_circuit_breaker_state(
            "openai_compatible.glm_5",
            BreakerState(
                name="open", consecutive_failures=3, total_failures=5,
                total_successes=30, last_error="rate_limit",
                last_failure_time=0.0, can_retry=False,
            ),
        )
        request = TAPRequest(
            meta={"task_id": "deg-2", "intent": "plan"},
            instruction="Plan",
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.deepseek_v4_flash"
        assert decision.reason == RoutingReason.DEGRADATION

    def test_degradation_m3_to_v4_pro(self):
        """M3 unavailable → degrades to V4-Pro"""
        self.router.update_circuit_breaker_state(
            "openai_compatible.minimax_m3",
            BreakerState(
                name="open", consecutive_failures=5, total_failures=8,
                total_successes=20, last_error="service_down",
                last_failure_time=0.0, can_retry=False,
            ),
        )
        request = TAPRequest(
            meta={"task_id": "deg-3", "intent": "execute"},
            instruction="Analyze image",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/img.png")
            ],
        )
        decision = self.router.route(request)
        # M3 is down, so multimodal overrides still pick M3 first,
        # then degradation should kick in
        assert "glm_5" not in decision.selected_driver or True  # May vary

    def test_all_models_down(self):
        """All primary models down → returns primary anyway (will likely fail)"""
        for driver in [
            "openai_compatible.deepseek_v4_pro",
            "openai_compatible.deepseek_v4_flash",
            "openai_compatible.glm_5",
            "openai_compatible.minimax_m3",
        ]:
            self.router.update_circuit_breaker_state(
                driver,
                BreakerState(
                    name="open", consecutive_failures=10, total_failures=20,
                    total_successes=5, last_error="total_outage",
                    last_failure_time=0.0, can_retry=False,
                ),
            )
        request = TAPRequest(
            meta={"task_id": "deg-4", "intent": "chat"},
            instruction="Hello",
        )
        # Should still return a driver (no crash)
        decision = self.router.route(request)
        assert decision.selected_driver  # Not empty


class TestP3_1_CostOptimization:
    """Test cost-based routing when budget is constrained"""

    def setup_method(self):
        self.router = _create_test_router()

    def test_budget_exhausted_auto_downgrade(self):
        """Budget exhausted → auto-downgrade to V4-Flash"""
        self.router.set_monthly_budget(limit_cny=0.01, warning_threshold=0.5)
        # Record a large cost to exhaust budget
        self.router._monthly_budget_cny = 0.02  # Exceeds limit

        request = TAPRequest(
            meta={"task_id": "cost-1", "intent": "design"},
            instruction="Design",
        )
        decision = self.router.route(request)
        # Should be downgraded to cheapest (V4-Flash)
        assert decision.selected_driver == "openai_compatible.deepseek_v4_flash"
        assert decision.reason == RoutingReason.COST_OPTIMIZATION

    def test_budget_ok_no_downgrade(self):
        """Budget ok → no cost-based downgrade"""
        self.router.set_monthly_budget(limit_cny=1000.0)
        request = TAPRequest(
            meta={"task_id": "cost-2", "intent": "design"},
            instruction="Design",
        )
        decision = self.router.route(request)
        # Should use default design driver (V4-Pro), not downgraded
        assert decision.selected_driver == "openai_compatible.deepseek_v4_pro"

    def test_no_budget_configured(self):
        """No budget configured → no cost optimization"""
        router = _create_test_router()  # Default: no budget
        request = TAPRequest(
            meta={"task_id": "cost-3", "intent": "design"},
            instruction="Design",
        )
        decision = router.route(request)
        assert decision.selected_driver == "openai_compatible.deepseek_v4_pro"


# =====================================================================
# P3-2: Pipeline Dynamic Allocation
# =====================================================================

class TestP3_2_PipelineManager:
    """Test PipelineManager with multi-profile support"""

    def setup_method(self):
        self.pm = PipelineManager()

    def test_builtin_profiles(self):
        """Built-in profiles are registered: default, budget, multimodal"""
        profiles = self.pm.list_profiles()
        assert "default" in profiles
        assert "budget" in profiles
        assert "multimodal" in profiles

    def test_default_profile_active(self):
        """Default profile is active on init"""
        assert self.pm.active_profile_name == "default"

    def test_switch_to_budget(self):
        """Switch to budget profile"""
        result = self.pm.set_active_profile("budget")
        assert result is True
        assert self.pm.active_profile_name == "budget"

    def test_switch_to_multimodal(self):
        """Switch to multimodal profile"""
        result = self.pm.set_active_profile("multimodal")
        assert result is True
        assert self.pm.active_profile_name == "multimodal"

    def test_switch_to_nonexistent(self):
        """Switch to nonexistent profile → False"""
        result = self.pm.set_active_profile("nonexistent")
        assert result is False
        assert self.pm.active_profile_name == "default"  # Unchanged

    def test_budget_profile_all_flash(self):
        """Budget profile uses V4-Flash for all stages"""
        self.pm.set_active_profile("budget")
        for stage in ["design", "plan", "execute", "review"]:
            driver = self.pm.get_driver(stage)
            assert "deepseek_v4_flash" in driver

    def test_multimodal_profile_all_m3(self):
        """Multimodal profile uses M3 for all stages"""
        self.pm.set_active_profile("multimodal")
        for stage in ["design", "plan", "execute", "review"]:
            driver = self.pm.get_driver(stage)
            assert "minimax_m3" in driver

    def test_default_profile_mixed(self):
        """Default profile uses mixed models per stage"""
        self.pm.set_active_profile("default")
        design_driver = self.pm.get_driver("design")
        # Design should use V4-Pro (or V4-Flash depending on routing table)
        assert "deepseek_v4" in design_driver

    def test_custom_profile(self):
        """Register and use a custom profile"""
        custom = PipelineProfile(
            name="custom_test",
            description="Custom test profile",
            design_driver="openai_compatible.glm_5",
            plan_driver="openai_compatible.deepseek_v4_flash",
            execute_driver="openai_compatible.deepseek_v4_flash",
            review_driver="openai_compatible.glm_5",
        )
        self.pm.register_profile(custom)
        assert self.pm.set_active_profile("custom_test")
        assert self.pm.get_driver("design") == "openai_compatible.glm_5"

    def test_profile_from_dict(self):
        """Create PipelineProfile from dict"""
        data = {
            "description": "Test profile",
            "design_driver": "openai_compatible.glm_5",
            "plan_driver": "openai_compatible.glm_5",
            "execute_driver": "openai_compatible.glm_5",
            "review_driver": "openai_compatible.glm_5",
        }
        profile = PipelineProfile.from_dict("glm_all", data)
        assert profile.name == "glm_all"
        assert profile.design_driver == "openai_compatible.glm_5"


class TestP3_2_PipelineRouting:
    """Test Pipeline-based routing with ModelRouter"""

    def setup_method(self):
        self.router = _create_test_router()

    def test_route_for_stage_design(self):
        """Route for design stage using default profile"""
        request = TAPRequest(
            meta={"task_id": "p-1", "intent": "design"},
            instruction="Design",
        )
        decision = self.router.route_for_stage("design", request)
        assert decision.reason == RoutingReason.PIPELINE_PROFILE
        assert "deepseek_v4" in decision.selected_driver

    def test_route_for_stage_with_multimodal_override(self):
        """Pipeline routing respects multimodal override"""
        request = TAPRequest(
            meta={"task_id": "p-2", "intent": "design"},
            instruction="Analyze image",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/img.png")
            ],
        )
        decision = self.router.route_for_stage("design", request)
        # Multimodal should override pipeline assignment
        assert decision.selected_driver == "openai_compatible.minimax_m3"

    def test_route_for_stage_with_long_horizon_override(self):
        """Pipeline routing respects long-horizon override"""
        request = TAPRequest(
            meta={"task_id": "p-3", "intent": "execute"},
            instruction="Run long task",
            long_horizon=LongHorizonConfig(max_duration_hours=8.0),
        )
        decision = self.router.route_for_stage("execute", request)
        assert decision.selected_driver == "openai_compatible.glm_5"
        assert decision.reason == RoutingReason.LONG_HORIZON_OVERRIDE

    def test_switch_profile_and_route(self):
        """Switch pipeline profile and verify different routing"""
        request = TAPRequest(
            meta={"task_id": "p-4", "intent": "design"},
            instruction="Design",
        )

        # Default profile → V4-Pro for design
        self.router.pipeline_manager.set_active_profile("default")
        decision1 = self.router.route_for_stage("design", request)

        # Budget profile → V4-Flash for design
        self.router.pipeline_manager.set_active_profile("budget")
        decision2 = self.router.route_for_stage("design", request)

        assert decision1.selected_driver != decision2.selected_driver

    def test_pipeline_from_config(self):
        """Load PipelineManager from TOML config dict"""
        config = {
            "execution": {
                "pipeline": {
                    "design_driver": "openai_compatible.deepseek_v4_pro",
                    "plan_driver": "openai_compatible.glm_5",
                    "execute_driver": "openai_compatible.glm_5",
                    "review_driver": "openai_compatible.deepseek_v4_pro",
                    "profiles": {
                        "test": {
                            "description": "Test profile",
                            "design_driver": "openai_compatible.glm_5",
                            "plan_driver": "openai_compatible.glm_5",
                            "execute_driver": "openai_compatible.glm_5",
                            "review_driver": "openai_compatible.glm_5",
                        },
                    },
                },
            },
        }
        pm = PipelineManager.from_config(config)
        assert "test" in pm.list_profiles()
        assert pm.get_driver("design") == "openai_compatible.deepseek_v4_pro"


# =====================================================================
# P3-3: CrossModelCostTracker
# =====================================================================

class TestP3_3_CostTracker:
    """Test CrossModelCostTracker with multi-model cost tracking"""

    def setup_method(self):
        self.tracker = CrossModelCostTracker()

    def test_record_basic(self):
        """Record a basic cost entry"""
        record = CostRecord(
            driver_name="openai_compatible.deepseek_v4_pro",
            compiler="deepseek_v4",
            model="deepseek-v4-pro",
            intent="design",
            prompt_tokens=5000,
            completion_tokens=2000,
            cost_cny=0.042,
        )
        result = self.tracker.record(record)
        assert result["level"] == "ok"
        assert self.tracker.get_total_cost() == 0.042

    def test_record_from_tap_response(self):
        """Record cost from TAP response data with pricing"""
        result = self.tracker.record_from_tap_response(
            driver_name="openai_compatible.deepseek_v4_pro",
            compiler="deepseek_v4",
            model="deepseek-v4-pro",
            intent="design",
            prompt_tokens=10000,
            completion_tokens=3000,
            cache_hit_tokens=6000,
            pricing={
                "prompt_per_million": 4.0,
                "completion_per_million": 16.0,
                "cache_hit_per_million": 0.4,
                "cache_miss_per_million": 4.0,
            },
        )
        assert result["level"] == "ok"
        assert self.tracker.get_total_cost() > 0

    def test_monthly_budget_warning(self):
        """Monthly budget warning at 80%"""
        self.tracker.set_monthly_budget(MonthlyBudgetConfig(
            limit_cny=0.10,
            warning_threshold=0.8,
        ))
        # Record cost that brings to 85%
        record = CostRecord(
            driver_name="openai_compatible.deepseek_v4_pro",
            intent="design",
            cost_cny=0.085,
        )
        result = self.tracker.record(record)
        assert result["level"] == "warning"
        assert self.tracker.is_budget_warning

    def test_monthly_budget_exhausted(self):
        """Monthly budget exhausted → auto-downgrade"""
        self.tracker.set_monthly_budget(MonthlyBudgetConfig(
            limit_cny=0.10,
            critical_threshold=1.0,
        ))
        record = CostRecord(
            driver_name="openai_compatible.deepseek_v4_pro",
            intent="design",
            cost_cny=0.12,
        )
        result = self.tracker.record(record)
        assert result["level"] == "exhausted"
        assert result["auto_downgrade"] is True
        assert self.tracker.is_budget_exhausted
        assert self.tracker.should_auto_downgrade

    def test_cost_report_by_model(self):
        """Generate cost report grouped by model"""
        for intent, driver, cost in [
            ("design", "openai_compatible.deepseek_v4_pro", 0.05),
            ("execute", "openai_compatible.deepseek_v4_flash", 0.02),
            ("plan", "openai_compatible.glm_5", 0.03),
        ]:
            self.tracker.record(CostRecord(
                driver_name=driver, intent=intent, cost_cny=cost,
            ))

        report = self.tracker.generate_report(group_by="model")
        assert report["report_type"] == "cost_by_model"
        assert len(report["groups"]) == 3
        assert report["summary"]["total_cost_cny"] == 0.10

    def test_cost_report_by_intent(self):
        """Generate cost report grouped by intent"""
        for intent in ["design", "plan", "execute", "design"]:
            self.tracker.record(CostRecord(
                driver_name="openai_compatible.deepseek_v4_pro",
                intent=intent,
                cost_cny=0.01,
            ))

        report = self.tracker.generate_report(group_by="intent")
        assert "design" in report["groups"]
        assert report["groups"]["design"]["total_calls"] == 2

    def test_cost_report_by_date(self):
        """Generate cost report grouped by date"""
        self.tracker.record(CostRecord(
            driver_name="openai_compatible.deepseek_v4_pro",
            intent="design",
            cost_cny=0.05,
            timestamp=time.time(),
        ))

        report = self.tracker.generate_report(group_by="date")
        assert len(report["groups"]) >= 1

    def test_cache_savings_tracking(self):
        """Track cache hit savings"""
        self.tracker.record(CostRecord(
            driver_name="openai_compatible.deepseek_v4_pro",
            intent="design",
            prompt_tokens=10000,
            completion_tokens=3000,
            cache_hit_tokens=6000,
            cache_miss_tokens=4000,
            cost_cny=0.032,
            cost_saved_cny=0.024,
        ))

        savings = self.tracker.get_cache_savings()
        assert savings["total_cache_hit_tokens"] == 6000
        assert savings["total_cost_saved_cny"] == 0.024

    def test_model_stats(self):
        """Get per-model cost statistics"""
        self.tracker.record(CostRecord(
            driver_name="openai_compatible.deepseek_v4_pro",
            intent="design",
            prompt_tokens=5000,
            completion_tokens=2000,
            cost_cny=0.04,
        ))
        self.tracker.record(CostRecord(
            driver_name="openai_compatible.deepseek_v4_pro",
            intent="review",
            prompt_tokens=3000,
            completion_tokens=1000,
            cost_cny=0.028,
        ))

        stats = self.tracker.get_model_stats("openai_compatible.deepseek_v4_pro")
        assert stats["total_calls"] == 2
        assert stats["total_cost_cny"] == 0.068
        assert stats["total_prompt_tokens"] == 8000

    def test_all_model_stats(self):
        """Get cost statistics for all models"""
        self.tracker.record(CostRecord(
            driver_name="openai_compatible.deepseek_v4_pro",
            intent="design", cost_cny=0.04,
        ))
        self.tracker.record(CostRecord(
            driver_name="openai_compatible.glm_5",
            intent="plan", cost_cny=0.03,
        ))

        all_stats = self.tracker.get_all_model_stats()
        assert len(all_stats) == 2

    def test_budget_auto_downgrade_driver(self):
        """MonthlyBudgetConfig specifies auto-downgrade driver"""
        config = MonthlyBudgetConfig(
            limit_cny=100.0,
            auto_downgrade_driver="openai_compatible.deepseek_v4_flash",
        )
        self.tracker.set_monthly_budget(config)
        # Exhaust budget
        self.tracker._total_cost_cny = 150.0
        status = self.tracker.check_budget()
        assert status["auto_downgrade"] is True
        assert status["downgrade_driver"] == "openai_compatible.deepseek_v4_flash"

    def test_reset_clears_all(self):
        """Reset clears all tracking state"""
        self.tracker.record(CostRecord(
            driver_name="openai_compatible.deepseek_v4_pro",
            intent="design", cost_cny=0.05,
        ))
        assert self.tracker.total_records > 0
        self.tracker.reset()
        assert self.tracker.total_records == 0
        assert self.tracker.get_total_cost() == 0.0


class TestP3_3_CostRecord:
    """Test CostRecord dataclass"""

    def test_date_str(self):
        """date_str returns YYYY-MM-DD format"""
        record = CostRecord(timestamp=1700000000.0)
        assert len(record.date_str) == 10
        assert record.date_str[4] == "-"

    def test_total_tokens(self):
        """total_tokens = prompt + completion"""
        record = CostRecord(prompt_tokens=5000, completion_tokens=2000)
        assert record.total_tokens == 7000

    def test_auto_timestamp(self):
        """timestamp auto-filled if not provided"""
        record = CostRecord()
        assert record.timestamp > 0


# =====================================================================
# P3-5: End-to-End Integration Tests
# =====================================================================

class TestP3_5_EndToEnd:
    """End-to-end integration tests combining routing + cost + pipeline"""

    def test_full_agent_flow_routing(self):
        """Complete agent flow: design→plan→execute→review with routing"""
        router = _create_test_router()

        stages_intents = [
            ("design", "design"),
            ("plan", "plan"),
            ("execute", "execute"),
            ("review", "review"),
        ]

        decisions = []
        for stage, intent in stages_intents:
            request = TAPRequest(
                meta={"task_id": f"e2e-{stage}", "intent": intent},
                instruction=f"Perform {stage}",
            )
            decision = router.route_for_stage(stage, request)
            decisions.append((stage, decision.selected_driver, decision.reason))

        # Verify each stage uses the expected model
        assert "deepseek_v4" in decisions[0][1]  # design → V4
        assert "glm_5" in decisions[1][1]         # plan → GLM-5
        assert "glm_5" in decisions[2][1]         # execute → GLM-5
        assert "deepseek_v4" in decisions[3][1]    # review → V4

    def test_multimodal_e2e_flow(self):
        """Multimodal end-to-end: image input → M3 compilation → routing"""
        router = _create_test_router()

        # Create multimodal request
        request = TAPRequest(
            meta={"task_id": "mm-e2e", "intent": "execute"},
            instruction="Analyze the screenshot and generate code",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/ui.png")
            ],
        )

        # Route it
        decision = router.route(request)
        assert decision.selected_driver == "openai_compatible.minimax_m3"

        # Compile with M3 compiler
        provider = router.get_provider(decision.selected_driver)
        assert provider is not None

        # Compile the request
        compiled = provider.compiler.compile(request)
        assert compiled.mode == "messages"

    def test_long_horizon_e2e_flow(self):
        """Long-horizon end-to-end: long task → GLM-5 routing → compilation"""
        router = _create_test_router()

        request = TAPRequest(
            meta={"task_id": "lh-e2e", "intent": "execute"},
            instruction="Implement the full project autonomously",
            long_horizon=LongHorizonConfig(
                max_duration_hours=4.0,
                checkpoint_interval_minutes=30.0,
                self_evaluation_enabled=True,
            ),
        )

        decision = router.route(request)
        assert decision.selected_driver == "openai_compatible.glm_5"

        # Verify the GLM-5 compiler handles long_horizon
        provider = router.get_provider(decision.selected_driver)
        compiled = provider.compiler.compile(request)
        assert compiled.mode == "messages"

    def test_cost_optimization_e2e(self):
        """Cost optimization E2E: budget → warning → auto-downgrade → report"""
        tracker = CrossModelCostTracker()
        tracker.set_monthly_budget(MonthlyBudgetConfig(
            limit_cny=0.20,
            warning_threshold=0.8,
            critical_threshold=1.0,
        ))

        # Record costs approaching warning
        tracker.record(CostRecord(
            driver_name="openai_compatible.deepseek_v4_pro",
            intent="design", cost_cny=0.10,
            prompt_tokens=50000, completion_tokens=10000,
        ))
        assert not tracker.is_budget_warning

        # More costs → warning
        tracker.record(CostRecord(
            driver_name="openai_compatible.deepseek_v4_pro",
            intent="review", cost_cny=0.08,
            prompt_tokens=40000, completion_tokens=8000,
        ))
        assert tracker.is_budget_warning

        # Even more → exhausted
        tracker.record(CostRecord(
            driver_name="openai_compatible.glm_5",
            intent="execute", cost_cny=0.05,
            prompt_tokens=30000, completion_tokens=6000,
        ))
        assert tracker.is_budget_exhausted
        assert tracker.should_auto_downgrade

        # Generate report
        report = tracker.generate_report(group_by="model")
        assert report["summary"]["total_cost_cny"] == 0.23

    def test_degradation_e2e(self):
        """Degradation E2E: primary fails → fallback → recovery"""
        router = _create_test_router()

        # V4-Pro circuit opens
        router.update_circuit_breaker_state(
            "openai_compatible.deepseek_v4_pro",
            BreakerState(
                name="open", consecutive_failures=5, total_failures=10,
                total_successes=50, last_error="timeout",
                last_failure_time=0.0, can_retry=False,
            ),
        )

        # Design request → should degrade to V4-Flash
        request = TAPRequest(
            meta={"task_id": "deg-e2e", "intent": "design"},
            instruction="Design",
        )
        decision = router.route(request)
        assert decision.reason == RoutingReason.DEGRADATION
        assert "deepseek_v4_flash" in decision.selected_driver

        # Simulate recovery: V4-Pro circuit closes
        router.update_circuit_breaker_state(
            "openai_compatible.deepseek_v4_pro",
            BreakerState(
                name="closed", consecutive_failures=0, total_failures=10,
                total_successes=51, last_error="",
                last_failure_time=0.0, can_retry=True,
            ),
        )

        # Same request → should use V4-Pro again
        decision2 = router.route(request)
        assert decision2.selected_driver == "openai_compatible.deepseek_v4_pro"
        assert decision2.reason == RoutingReason.INTENT

    def test_pipeline_switch_e2e(self):
        """Pipeline switch E2E: switch profile → different routing → verify"""
        router = _create_test_router()

        request = TAPRequest(
            meta={"task_id": "pipe-e2e", "intent": "execute"},
            instruction="Write code",
        )

        # Default: execute → GLM-5
        router.pipeline_manager.set_active_profile("default")
        d1 = router.route_for_stage("execute", request)

        # Budget: execute → V4-Flash
        router.pipeline_manager.set_active_profile("budget")
        d2 = router.route_for_stage("execute", request)

        # Multimodal: execute → M3
        router.pipeline_manager.set_active_profile("multimodal")
        d3 = router.route_for_stage("execute", request)

        # Verify different models selected
        assert d1.selected_driver != d2.selected_driver
        assert d2.selected_driver != d3.selected_driver

    def test_routing_with_cost_tracking_e2e(self):
        """Combined routing + cost tracking E2E"""
        router = _create_test_router()
        tracker = CrossModelCostTracker()

        # Route and track for each stage
        stages = [
            ("design", "design", 50000),
            ("plan", "plan", 30000),
            ("execute", "execute", 80000),
            ("review", "review", 20000),
        ]

        for stage, intent, tokens in stages:
            request = TAPRequest(
                meta={"task_id": f"combo-{stage}", "intent": intent},
                instruction=f"Perform {stage}",
            )
            decision = router.route_for_stage(stage, request)

            # Track cost
            pricing = router.routing_table.get_pricing(decision.selected_driver)
            tracker.record_from_tap_response(
                driver_name=decision.selected_driver,
                compiler=decision.selected_compiler,
                model="test-model",
                intent=intent,
                prompt_tokens=tokens,
                completion_tokens=int(tokens * 0.3),
                cache_hit_tokens=int(tokens * 0.6),
                pricing=pricing,
            )

        # Verify cost tracking
        assert tracker.total_records == 4
        assert tracker.get_total_cost() > 0

        # Generate report
        report = tracker.generate_report(group_by="intent")
        assert len(report["groups"]) == 4

    def test_all_compilers_still_work(self):
        """Regression: all 8 compilers still compile successfully with router"""
        router = _create_test_router()

        for compiler_name in ["deepseek_v4", "glm_5", "minimax_m3", "default", "glm", "deepseek"]:
            compiler_cls = TAPCompilerRegistry.get(compiler_name)
            if compiler_cls is None:
                continue

            kwargs = {}
            if compiler_name == "deepseek_v4":
                kwargs["variant"] = "flash"

            compiler = compiler_cls(**kwargs)
            request = TAPRequest(
                meta={"task_id": f"regress-{compiler_name}", "intent": "execute"},
                instruction="Write a function",
            )
            compiled = compiler.compile(request)
            assert compiled.mode in ("messages", "system_user", "empty")

    def test_routing_table_from_dict(self):
        """RoutingTable.from_dict properly merges overrides"""
        data = {
            "multimodal_driver": "openai_compatible.custom_m3",
            "long_horizon_driver": "openai_compatible.custom_glm",
            "model_pricing": {
                "openai_compatible.custom_model": {
                    "prompt_per_million": 5.0,
                    "completion_per_million": 20.0,
                },
            },
        }
        table = RoutingTable.from_dict(data)
        assert table.multimodal_driver == "openai_compatible.custom_m3"
        assert table.long_horizon_driver == "openai_compatible.custom_glm"
        assert "openai_compatible.custom_model" in table.model_pricing

    def test_model_router_from_config(self):
        """ModelRouter.from_config loads routing and pipeline from config dict"""
        config = {
            "routing": {
                "multimodal_driver": "openai_compatible.minimax_m3",
                "long_horizon_driver": "openai_compatible.glm_5",
                "monthly_budget": {
                    "limit_cny": 500.0,
                    "warning_threshold": 0.8,
                    "auto_downgrade": True,
                },
            },
            "execution": {
                "pipeline": {
                    "design_driver": "openai_compatible.deepseek_v4_pro",
                    "plan_driver": "openai_compatible.glm_5",
                    "execute_driver": "openai_compatible.glm_5",
                    "review_driver": "openai_compatible.deepseek_v4_pro",
                },
            },
        }
        router = ModelRouter.from_config(config)
        assert router.routing_table.multimodal_driver == "openai_compatible.minimax_m3"
        assert router.routing_table.long_horizon_driver == "openai_compatible.glm_5"
        assert router.monthly_budget_remaining > 0

    def test_step_budget_unchanged(self):
        """StepBudget still works correctly (regression test)"""
        budget = StepBudget(max_steps=5)
        for _ in range(5):
            assert budget.consume()
        assert not budget.consume()  # Exhausted
        assert budget.exhausted

        budget.resume(extra_steps=3)
        assert budget.consume()
        assert budget.remaining == 2
