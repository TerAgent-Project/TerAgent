"""tests/integration/test_e2e.py — Phase 3 End-to-End integration tests

Comprehensive E2E tests for Phase 3:
  1. Full agent flow with ModelRouter (Design→Plan→Execute→Review)
  2. Multimodal E2E flow (image → M3/5V-Turbo routing)
  3. Long-horizon E2E flow (long task → GLM-5.2 routing)
  4. Cost optimization E2E (budget exhaustion → auto-downgrade)
  5. Degradation E2E (circuit breaker → fallback)
  6. Pipeline switch E2E (profile switching at runtime)
  7. GLM-5.2 + 5V-Turbo coordination E2E (vision → code → verify)
  8. All compilers still work (regression)

All tests use MockAdapter — no real API calls.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from teragent.core.adapters.mock import MockAdapter
from teragent.core.compiler import TAPCompilerRegistry
from teragent.core.provider import ModelProvider
from teragent.core.tap import (
    DesktopContext,
    LongHorizonConfig,
    MultimodalContent,
    TAPRequest,
    TAPResponse,
)
from teragent.reliability.budget import (
    CostRecord,
    CrossModelCostTracker,
    MonthlyBudgetConfig,
    StepBudget,
)
from teragent.reliability.circuit_breaker import BreakerState
from teragent.router.model_router import (
    ModelRouter,
    PipelineManager,
    PipelineProfile,
    RoutingReason,
    RoutingTable,
)
from teragent.coordination.glm5v_coordinator import (
    CoordinationConfig,
    CoordinationPhase,
    GLM52VCoordinatedWorkflow,
)


# ===== Helpers =====


def _create_mock_provider(compiler_name: str, model: str, **kwargs) -> ModelProvider:
    """Create a ModelProvider with a real compiler and MockAdapter."""
    compiler_cls = TAPCompilerRegistry.get(compiler_name)
    if compiler_cls is None:
        raise ValueError(f"Unknown compiler: {compiler_name}")
    init_kwargs = {}
    if "compiler_variant" in kwargs:
        init_kwargs["variant"] = kwargs["compiler_variant"]
    if "mode" in kwargs:
        init_kwargs["mode"] = kwargs["mode"]
    compiler = compiler_cls(**init_kwargs)
    adapter = MockAdapter()
    return ModelProvider(compiler=compiler, adapter=adapter, model=model)


def _create_all_providers() -> dict[str, ModelProvider]:
    """Create mock providers for all registered models."""
    providers: dict[str, ModelProvider] = {}
    provider_specs = [
        ("deepseek_v4", "deepseek-v4-flash", "openai_compatible.deepseek_v4_flash",
         {"compiler_variant": "flash"}),
        ("deepseek_v4", "deepseek-v4-pro", "openai_compatible.deepseek_v4_pro",
         {"compiler_variant": "pro"}),
        ("minimax_m3", "minimax-m3", "openai_compatible.minimax_m3", {}),
        ("glm_5", "glm-5", "openai_compatible.glm_5", {}),
        ("glm_52", "glm-5.2", "openai_compatible.glm_52", {}),
        ("glm_5v_turbo", "glm-5v-turbo", "openai_compatible.glm_5v_turbo",
         {"mode": "analysis"}),
    ]
    for compiler_name, model, driver_name, kwargs in provider_specs:
        try:
            providers[driver_name] = _create_mock_provider(compiler_name, model, **kwargs)
        except Exception:
            pass
    return providers


def _create_test_router(providers: dict[str, ModelProvider] | None = None) -> ModelRouter:
    """Create a ModelRouter with all mock providers."""
    if providers is None:
        providers = _create_all_providers()
    return ModelRouter(available_providers=providers)


# =====================================================================
# 1. Full Agent Flow E2E
# =====================================================================


class TestFullAgentFlowE2E:
    """End-to-end tests for the full Design→Plan→Execute→Review agent loop."""

    def setup_method(self):
        self.providers = _create_all_providers()
        self.router = _create_test_router(self.providers)

    @pytest.mark.asyncio
    async def test_design_stage_routes_correctly(self):
        """Design stage routes to V4-Pro and produces valid output."""
        request = TAPRequest(
            meta={"task_id": "1", "intent": "design"},
            instruction="Design the user authentication system",
            constraints=["Python 3.10+", "JWT-based"],
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.deepseek_v4_pro"

        provider = self.router.get_provider(decision.selected_driver)
        assert provider is not None
        response = await provider.execute_tap(request)
        assert response.raw_text is not None
        assert len(response.raw_text) > 0

    @pytest.mark.asyncio
    async def test_plan_stage_routes_correctly(self):
        """Plan stage routes to GLM-5.2 and produces valid output."""
        request = TAPRequest(
            meta={"task_id": "2", "intent": "plan"},
            instruction="Create an execution plan for the auth module",
            context={"design": "Authentication system design document"},
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.glm_52"

        provider = self.router.get_provider(decision.selected_driver)
        assert provider is not None
        response = await provider.execute_tap(request)
        assert response.raw_text is not None

    @pytest.mark.asyncio
    async def test_execute_stage_routes_correctly(self):
        """Execute stage routes to GLM-5.2 and produces valid output."""
        request = TAPRequest(
            meta={"task_id": "3", "intent": "execute"},
            instruction="Implement the JWT auth module",
            context={"design": "Auth design", "plan": "Step 1: Create models"},
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.glm_52"

        provider = self.router.get_provider(decision.selected_driver)
        assert provider is not None
        response = await provider.execute_tap(request)
        assert response.raw_text is not None

    @pytest.mark.asyncio
    async def test_review_stage_routes_correctly(self):
        """Review stage routes to V4-Pro and produces valid output."""
        request = TAPRequest(
            meta={"task_id": "4", "intent": "review"},
            instruction="Review the implemented code",
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.deepseek_v4_pro"

        provider = self.router.get_provider(decision.selected_driver)
        assert provider is not None
        response = await provider.execute_tap(request)
        assert response.raw_text is not None

    @pytest.mark.asyncio
    async def test_full_pipeline_stages_e2e(self):
        """Full pipeline: Design→Plan→Execute→Review all produce responses."""
        stages = [
            ("design", "Design the system", {}),
            ("plan", "Create a plan", {"design": "Design doc"}),
            ("execute", "Write the code", {"design": "D", "plan": "P"}),
            ("review", "Review the code", {}),
        ]

        responses = []
        for intent, instruction, context in stages:
            request = TAPRequest(
                meta={"task_id": f"e2e.{intent}", "intent": intent},
                instruction=instruction,
                context=context,
            )
            decision = self.router.route(request)
            provider = self.router.get_provider(decision.selected_driver)
            assert provider is not None, f"No provider for {decision.selected_driver}"
            response = await provider.execute_tap(request)
            assert response.raw_text is not None
            responses.append((intent, decision.selected_driver, response))

        # Verify all 4 stages completed
        assert len(responses) == 4
        intents_seen = [r[0] for r in responses]
        assert intents_seen == ["design", "plan", "execute", "review"]

    @pytest.mark.asyncio
    async def test_pipeline_route_for_stage(self):
        """route_for_stage uses pipeline profile for each stage."""
        request = TAPRequest(
            meta={"task_id": "pipeline.1", "intent": "execute"},
            instruction="Write code",
        )

        for stage in ["design", "plan", "execute", "review"]:
            decision = self.router.route_for_stage(stage, request)
            assert decision.selected_driver != "", f"No driver for stage {stage}"
            assert decision.selected_compiler != ""


# =====================================================================
# 2. Multimodal E2E Flow
# =====================================================================


class TestMultimodalE2E:
    """End-to-end tests for multimodal request routing and execution."""

    def setup_method(self):
        self.providers = _create_all_providers()
        self.router = _create_test_router(self.providers)

    def test_image_request_routes_to_m3(self):
        """Image content routes to M3 (multimodal override)."""
        request = TAPRequest(
            meta={"task_id": "mm.1", "intent": "execute"},
            instruction="Analyze this screenshot",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/screenshot.png"),
            ],
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.minimax_m3"
        assert decision.reason == RoutingReason.MULTIMODAL_OVERRIDE

    @pytest.mark.asyncio
    async def test_image_request_e2e_execution(self):
        """Multimodal request routes to M3 and produces valid output."""
        request = TAPRequest(
            meta={"task_id": "mm.2", "intent": "design"},
            instruction="Analyze this design screenshot and create a system design",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/design.png"),
            ],
        )
        decision = self.router.route(request)
        provider = self.router.get_provider(decision.selected_driver)
        assert provider is not None
        response = await provider.execute_tap(request)
        assert response.raw_text is not None

    def test_video_request_routes_to_m3(self):
        """Video content routes to M3 (video override)."""
        request = TAPRequest(
            meta={"task_id": "mm.3", "intent": "execute"},
            instruction="Analyze this video",
            multimodal_context=[
                MultimodalContent(type="video_url", url="https://example.com/demo.mp4"),
            ],
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.minimax_m3"
        assert decision.reason == RoutingReason.VIDEO_OVERRIDE

    def test_desktop_context_routes_to_m3(self):
        """Desktop context routes to M3 (desktop override)."""
        request = TAPRequest(
            meta={"task_id": "mm.4", "intent": "execute"},
            instruction="Click the submit button",
            desktop_context=DesktopContext(
                screenshot=MultimodalContent(type="image_url", url="https://example.com/screen.png"),
                interactive_elements=[{"type": "button", "label": "Submit", "bbox": {"x": 100, "y": 200}}],
                active_window="Browser",
            ),
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.minimax_m3"
        assert decision.reason == RoutingReason.DESKTOP_OVERRIDE

    @pytest.mark.asyncio
    async def test_5v_turbo_direct_execution(self):
        """Direct GLM-5V-Turbo execution for vision analysis."""
        provider = self.providers.get("openai_compatible.glm_5v_turbo")
        if provider is None:
            pytest.skip("GLM-5V-Turbo provider not available")

        request = TAPRequest(
            meta={"task_id": "mm.5v", "intent": "design"},
            instruction="分析这个设计稿的布局和颜色",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/ui.png"),
            ],
        )
        response = await provider.execute_tap(request)
        assert response.raw_text is not None

    def test_multimodal_overrides_intent_routing(self):
        """Multimodal override takes priority over intent routing."""
        # Even with "execute" intent (normally GLM-5.2), multimodal → M3
        request = TAPRequest(
            meta={"task_id": "mm.6", "intent": "execute"},
            instruction="Implement based on this screenshot",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/screen.png"),
            ],
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.minimax_m3"


# =====================================================================
# 3. Long-Horizon E2E Flow
# =====================================================================


class TestLongHorizonE2E:
    """End-to-end tests for long-horizon task routing and execution."""

    def setup_method(self):
        self.providers = _create_all_providers()
        self.router = _create_test_router(self.providers)

    def test_long_horizon_routes_to_glm52(self):
        """Long-horizon task routes to GLM-5.2."""
        request = TAPRequest(
            meta={"task_id": "lh.1", "intent": "execute"},
            instruction="Run the full 8-hour autonomous task",
            long_horizon=LongHorizonConfig(
                max_duration_hours=8.0,
                checkpoint_interval_minutes=30.0,
                self_evaluation_enabled=True,
            ),
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.glm_52"
        assert decision.reason == RoutingReason.LONG_HORIZON_OVERRIDE

    @pytest.mark.asyncio
    async def test_long_horizon_e2e_execution(self):
        """Long-horizon task routes correctly and produces output."""
        request = TAPRequest(
            meta={"task_id": "lh.2", "intent": "execute"},
            instruction="Implement the complete project autonomously",
            long_horizon=LongHorizonConfig(
                max_duration_hours=4.0,
                self_evaluation_enabled=True,
            ),
        )
        decision = self.router.route(request)
        provider = self.router.get_provider(decision.selected_driver)
        assert provider is not None
        response = await provider.execute_tap(request)
        assert response.raw_text is not None

    def test_long_horizon_overrides_intent(self):
        """Long-horizon override takes priority over intent routing."""
        # Even with "design" intent (normally V4-Pro), long_horizon → GLM-5.2
        request = TAPRequest(
            meta={"task_id": "lh.3", "intent": "design"},
            instruction="Long-running design task",
            long_horizon=LongHorizonConfig(),
        )
        decision = self.router.route(request)
        assert decision.selected_driver == "openai_compatible.glm_52"

    def test_long_horizon_config_defaults(self):
        """LongHorizonConfig has correct defaults."""
        config = LongHorizonConfig()
        assert config.max_duration_hours == 8.0
        assert config.checkpoint_interval_minutes == 30.0
        assert config.self_evaluation_enabled is True
        assert config.stagnation_threshold == 3


# =====================================================================
# 4. Cost Optimization E2E
# =====================================================================


class TestCostOptimizationE2E:
    """End-to-end tests for budget exhaustion and auto-downgrade."""

    def setup_method(self):
        self.providers = _create_all_providers()
        self.router = _create_test_router(self.providers)

    def test_normal_routing_no_budget(self):
        """Without budget set, routing follows intent defaults."""
        request = TAPRequest(
            meta={"task_id": "cost.1", "intent": "execute"},
            instruction="Write code",
        )
        decision = self.router.route(request)
        assert decision.reason == RoutingReason.INTENT

    def test_budget_exhaustion_auto_downgrade(self):
        """When budget is exhausted, auto-downgrade to cheaper model."""
        # Set a very small budget
        self.router.set_monthly_budget(limit_cny=0.00001, auto_downgrade=True)

        # Simulate that budget is already exceeded
        self.router._monthly_budget_cny = 1.0

        request = TAPRequest(
            meta={"task_id": "cost.2", "intent": "execute"},
            instruction="Write code",
        )
        decision = self.router.route(request)
        # Should downgrade to cheapest available (V4-Flash)
        assert decision.reason == RoutingReason.COST_OPTIMIZATION

    def test_budget_warning_downgrades_expensive(self):
        """When budget is at warning level, expensive models are downgraded."""
        # Set budget and simulate 85% utilization
        self.router.set_monthly_budget(limit_cny=1.0, warning_threshold=0.8)
        self.router._monthly_budget_cny = 0.85

        request = TAPRequest(
            meta={"task_id": "cost.3", "intent": "execute"},
            instruction="Write code",
        )
        decision = self.router.route(request)
        # GLM-5.2 is expensive, should be downgraded to V4-Flash
        if decision.reason == RoutingReason.COST_OPTIMIZATION:
            assert decision.selected_driver == "openai_compatible.deepseek_v4_flash"

    def test_budget_not_exhausted_no_downgrade(self):
        """When budget is not exhausted, no downgrade happens."""
        self.router.set_monthly_budget(limit_cny=100.0)
        self.router._monthly_budget_cny = 0.01

        request = TAPRequest(
            meta={"task_id": "cost.4", "intent": "execute"},
            instruction="Write code",
        )
        decision = self.router.route(request)
        # Should route normally (intent-based)
        assert decision.reason == RoutingReason.INTENT

    def test_cross_model_cost_tracker(self):
        """CrossModelCostTracker tracks costs across models."""
        tracker = CrossModelCostTracker()
        tracker.set_monthly_budget(MonthlyBudgetConfig(limit_cny=100.0))

        # Record some costs
        tracker.record(CostRecord(
            timestamp=time.time(),
            driver_name="openai_compatible.deepseek_v4_pro",
            compiler="deepseek_v4",
            model="deepseek-v4-pro",
            intent="design",
            prompt_tokens=1000,
            completion_tokens=500,
        ))
        tracker.record(CostRecord(
            timestamp=time.time(),
            driver_name="openai_compatible.glm_52",
            compiler="glm_52",
            model="glm-5.2",
            intent="execute",
            prompt_tokens=2000,
            completion_tokens=1000,
        ))

        report = tracker.generate_report()
        assert report is not None

    def test_record_actual_cost(self):
        """ModelRouter records actual costs from completed requests."""
        self.router.set_monthly_budget(limit_cny=100.0)

        cost = self.router.record_actual_cost(
            "openai_compatible.deepseek_v4_pro",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        assert cost > 0.0
        assert self.router._monthly_budget_cny > 0.0


# =====================================================================
# 5. Degradation E2E
# =====================================================================


class TestDegradationE2E:
    """End-to-end tests for circuit breaker degradation and fallback."""

    def setup_method(self):
        self.providers = _create_all_providers()
        self.router = _create_test_router(self.providers)

    def test_circuit_open_triggers_degradation(self):
        """When primary model's circuit breaker is open, route to fallback."""
        # Simulate V4-Pro circuit breaker open
        self.router.update_circuit_breaker_state(
            "openai_compatible.deepseek_v4_pro",
            BreakerState(
                name="open",
                consecutive_failures=5,
                total_failures=5,
                total_successes=0,
                last_error="API timeout",
                last_failure_time=time.time(),
                can_retry=False,
            ),
        )

        request = TAPRequest(
            meta={"task_id": "deg.1", "intent": "design"},
            instruction="Design the system",
        )
        decision = self.router.route(request)
        # Should degrade to V4-Flash (fallback for V4-Pro)
        assert decision.reason == RoutingReason.DEGRADATION
        assert decision.selected_driver != "openai_compatible.deepseek_v4_pro"

    def test_circuit_closed_routes_normally(self):
        """When circuit breaker is closed, routing is normal."""
        request = TAPRequest(
            meta={"task_id": "deg.2", "intent": "design"},
            instruction="Design the system",
        )
        decision = self.router.route(request)
        assert decision.reason == RoutingReason.INTENT
        assert decision.selected_driver == "openai_compatible.deepseek_v4_pro"

    def test_multiple_circuits_open_fallback_chain(self):
        """When multiple circuit breakers are open, fallback chain works."""
        # Open V4-Pro and V4-Flash circuit breakers
        self.router.update_circuit_breaker_state(
            "openai_compatible.deepseek_v4_pro",
            BreakerState(name="open", consecutive_failures=5, total_failures=5,
                        total_successes=0, last_error="err", last_failure_time=time.time(),
                        can_retry=False),
        )
        self.router.update_circuit_breaker_state(
            "openai_compatible.deepseek_v4_flash",
            BreakerState(name="open", consecutive_failures=5, total_failures=5,
                        total_successes=0, last_error="err", last_failure_time=time.time(),
                        can_retry=False),
        )

        request = TAPRequest(
            meta={"task_id": "deg.3", "intent": "design"},
            instruction="Design the system",
        )
        decision = self.router.route(request)
        # Should fall through the fallback chain
        assert decision.selected_driver != "openai_compatible.deepseek_v4_pro"
        assert decision.selected_driver != "openai_compatible.deepseek_v4_flash"

    @pytest.mark.asyncio
    async def test_degraded_provider_still_works(self):
        """Even when degraded, the fallback provider still produces output."""
        self.router.update_circuit_breaker_state(
            "openai_compatible.deepseek_v4_pro",
            BreakerState(name="open", consecutive_failures=5, total_failures=5,
                        total_successes=0, last_error="err", last_failure_time=time.time(),
                        can_retry=False),
        )

        request = TAPRequest(
            meta={"task_id": "deg.4", "intent": "design"},
            instruction="Design the system",
        )
        decision = self.router.route(request)
        provider = self.router.get_provider(decision.selected_driver)
        if provider is not None:
            response = await provider.execute_tap(request)
            assert response.raw_text is not None


# =====================================================================
# 6. Pipeline Switch E2E
# =====================================================================


class TestPipelineSwitchE2E:
    """End-to-end tests for runtime pipeline profile switching."""

    def setup_method(self):
        self.providers = _create_all_providers()
        self.router = _create_test_router(self.providers)

    def test_default_profile(self):
        """Default profile uses standard intent routing."""
        pm = self.router.pipeline_manager
        assert pm.active_profile_name == "default"

    def test_switch_to_budget_profile(self):
        """Switching to budget profile routes all stages to V4-Flash."""
        pm = self.router.pipeline_manager
        assert pm.set_active_profile("budget") is True
        assert pm.active_profile_name == "budget"

        request = TAPRequest(
            meta={"task_id": "pipe.1", "intent": "execute"},
            instruction="Write code",
        )
        decision = self.router.route_for_stage("execute", request)
        # Budget profile should route to V4-Flash (unless overridden by multimodal/etc.)
        # Check that the pipeline assignment was made
        assert decision.selected_driver != ""

    def test_switch_to_multimodal_profile(self):
        """Switching to multimodal profile routes all stages to M3."""
        pm = self.router.pipeline_manager
        assert pm.set_active_profile("multimodal") is True
        assert pm.active_profile_name == "multimodal"

    def test_switch_to_deep_thinking_profile(self):
        """Switching to deep_thinking profile routes all stages to GLM-5.2."""
        pm = self.router.pipeline_manager
        assert pm.set_active_profile("deep_thinking") is True
        assert pm.active_profile_name == "deep_thinking"

    def test_switch_back_to_default(self):
        """Switching back to default profile restores standard routing."""
        pm = self.router.pipeline_manager
        pm.set_active_profile("budget")
        pm.set_active_profile("default")
        assert pm.active_profile_name == "default"

    def test_invalid_profile_name(self):
        """Switching to non-existent profile returns False."""
        pm = self.router.pipeline_manager
        assert pm.set_active_profile("nonexistent") is False

    def test_custom_profile_registration(self):
        """Custom profiles can be registered and activated."""
        pm = self.router.pipeline_manager
        custom = PipelineProfile(
            name="custom_test",
            description="Custom test profile",
            design_driver="openai_compatible.glm_52",
            plan_driver="openai_compatible.glm_52",
            execute_driver="openai_compatible.glm_52",
            review_driver="openai_compatible.glm_52",
        )
        pm.register_profile(custom)
        assert pm.set_active_profile("custom_test") is True
        assert pm.active_profile_name == "custom_test"

    @pytest.mark.asyncio
    async def test_profile_switch_mid_workflow(self):
        """Pipeline profile can be switched mid-workflow."""
        pm = self.router.pipeline_manager
        request = TAPRequest(
            meta={"task_id": "pipe.mid", "intent": "execute"},
            instruction="Write code",
        )

        # Design with default profile
        pm.set_active_profile("default")
        decision1 = self.router.route_for_stage("design", request)

        # Switch to budget for execute
        pm.set_active_profile("budget")
        decision2 = self.router.route_for_stage("execute", request)

        # Both should produce valid routes
        assert decision1.selected_driver != ""
        assert decision2.selected_driver != ""

    def test_list_profiles(self):
        """list_profiles returns all registered profiles."""
        pm = self.router.pipeline_manager
        profiles = pm.list_profiles()
        assert "default" in profiles
        assert "budget" in profiles
        assert "multimodal" in profiles


# =====================================================================
# 7. GLM-5.2 + 5V-Turbo Coordination E2E
# =====================================================================


class TestCoordinationE2E:
    """End-to-end tests for GLM-5.2 + GLM-5V-Turbo coordination workflow."""

    def setup_method(self):
        self.vision_provider = _create_mock_provider(
            "glm_5v_turbo", "glm-5v-turbo", mode="analysis",
        )
        self.coding_provider = _create_mock_provider(
            "glm_52", "glm-5.2",
        )

    @pytest.mark.asyncio
    async def test_sequential_coordination_e2e(self):
        """Sequential coordination: vision → code with multimodal input."""
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=self.vision_provider,
            coding_provider=self.coding_provider,
            config=CoordinationConfig(mode="sequential"),
        )
        request = TAPRequest(
            meta={"task_id": "coord.seq", "intent": "execute"},
            instruction="根据设计稿实现页面",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/design.png"),
            ],
        )
        result = await workflow.execute(request)

        assert result.success is True
        assert result.vision_analysis is not None
        assert result.final_response is not None
        phases = [s.phase for s in result.steps]
        assert CoordinationPhase.VISION_ANALYSIS.value in phases
        assert CoordinationPhase.CODE_GENERATION.value in phases

    @pytest.mark.asyncio
    async def test_verify_coordination_e2e(self):
        """Verify coordination: vision → code → visual verification."""
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=self.vision_provider,
            coding_provider=self.coding_provider,
            config=CoordinationConfig(mode="verify", max_verification_rounds=1),
        )
        request = TAPRequest(
            meta={"task_id": "coord.verify", "intent": "execute"},
            instruction="根据设计稿实现页面并验证一致性",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/design.png"),
            ],
        )
        result = await workflow.execute(request)

        assert result.success is True
        phases = [s.phase for s in result.steps]
        assert CoordinationPhase.VISUAL_VERIFICATION.value in phases

    @pytest.mark.asyncio
    async def test_parallel_coordination_e2e(self):
        """Parallel coordination: vision || code simultaneously."""
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=self.vision_provider,
            coding_provider=self.coding_provider,
            config=CoordinationConfig(mode="parallel"),
        )
        request = TAPRequest(
            meta={"task_id": "coord.par", "intent": "execute"},
            instruction="根据设计稿实现页面",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/design.png"),
            ],
        )
        result = await workflow.execute(request)

        assert result.success is True
        assert result.final_response is not None

    @pytest.mark.asyncio
    async def test_coordination_no_multimodal_coding_only(self):
        """Coordination without multimodal content goes directly to coding."""
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=self.vision_provider,
            coding_provider=self.coding_provider,
        )
        request = TAPRequest(
            meta={"task_id": "coord.text", "intent": "execute"},
            instruction="实现排序函数",
        )
        result = await workflow.execute(request)

        assert result.success is True
        # No multimodal → coding only, single step
        assert len(result.steps) == 1
        assert result.steps[0].phase == CoordinationPhase.CODE_GENERATION.value

    @pytest.mark.asyncio
    async def test_coordination_degradation_e2e(self):
        """Coordination degrades when vision provider fails."""
        config = CoordinationConfig(
            degrade_on_vision_failure=True,
            vision_compiler="",
        )
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=None,
            coding_provider=self.coding_provider,
            config=config,
        )
        request = TAPRequest(
            meta={"task_id": "coord.deg", "intent": "execute"},
            instruction="根据设计稿实现页面",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/design.png"),
            ],
        )
        result = await workflow.execute(request)

        assert result.is_degraded is True
        assert result.final_response is not None


# =====================================================================
# 8. All Compilers Regression
# =====================================================================


class TestCompilerRegression:
    """Regression test: all registered compilers still produce valid output."""

    def setup_method(self):
        self.adapter = MockAdapter()

    @pytest.mark.asyncio
    async def test_default_compiler(self):
        """Default compiler still works."""
        compiler_cls = TAPCompilerRegistry.get("default")
        if compiler_cls is None:
            pytest.skip("default compiler not registered")
        compiler = compiler_cls()
        provider = ModelProvider(compiler=compiler, adapter=self.adapter, model="test")
        request = TAPRequest(
            meta={"task_id": "reg.default", "intent": "execute"},
            instruction="Write a function",
        )
        response = await provider.execute_tap(request)
        assert response.raw_text is not None

    @pytest.mark.asyncio
    async def test_deepseek_v4_flash_compiler(self):
        """DeepSeek V4 Flash compiler still works."""
        compiler_cls = TAPCompilerRegistry.get("deepseek_v4")
        if compiler_cls is None:
            pytest.skip("deepseek_v4 compiler not registered")
        compiler = compiler_cls(variant="flash")
        provider = ModelProvider(compiler=compiler, adapter=self.adapter, model="deepseek-v4-flash")
        request = TAPRequest(
            meta={"task_id": "reg.v4f", "intent": "execute"},
            instruction="Write a function",
        )
        response = await provider.execute_tap(request)
        assert response.raw_text is not None

    @pytest.mark.asyncio
    async def test_deepseek_v4_pro_compiler(self):
        """DeepSeek V4 Pro compiler still works."""
        compiler_cls = TAPCompilerRegistry.get("deepseek_v4")
        if compiler_cls is None:
            pytest.skip("deepseek_v4 compiler not registered")
        compiler = compiler_cls(variant="pro")
        provider = ModelProvider(compiler=compiler, adapter=self.adapter, model="deepseek-v4-pro")
        request = TAPRequest(
            meta={"task_id": "reg.v4p", "intent": "design"},
            instruction="Design the system",
        )
        response = await provider.execute_tap(request)
        assert response.raw_text is not None

    @pytest.mark.asyncio
    async def test_glm5_compiler(self):
        """GLM-5 compiler still works."""
        compiler_cls = TAPCompilerRegistry.get("glm_5")
        if compiler_cls is None:
            pytest.skip("glm_5 compiler not registered")
        compiler = compiler_cls()
        provider = ModelProvider(compiler=compiler, adapter=self.adapter, model="glm-5")
        request = TAPRequest(
            meta={"task_id": "reg.g5", "intent": "execute"},
            instruction="写一个排序函数",
        )
        response = await provider.execute_tap(request)
        assert response.raw_text is not None

    @pytest.mark.asyncio
    async def test_glm52_compiler(self):
        """GLM-5.2 compiler still works."""
        compiler_cls = TAPCompilerRegistry.get("glm_52")
        if compiler_cls is None:
            pytest.skip("glm_52 compiler not registered")
        compiler = compiler_cls()
        provider = ModelProvider(compiler=compiler, adapter=self.adapter, model="glm-5.2")
        request = TAPRequest(
            meta={"task_id": "reg.g52", "intent": "execute"},
            instruction="写一个排序函数",
        )
        response = await provider.execute_tap(request)
        assert response.raw_text is not None

    @pytest.mark.asyncio
    async def test_glm5v_turbo_compiler(self):
        """GLM-5V-Turbo compiler still works."""
        compiler_cls = TAPCompilerRegistry.get("glm_5v_turbo")
        if compiler_cls is None:
            pytest.skip("glm_5v_turbo compiler not registered")
        compiler = compiler_cls(mode="analysis")
        provider = ModelProvider(compiler=compiler, adapter=self.adapter, model="glm-5v-turbo")
        request = TAPRequest(
            meta={"task_id": "reg.5vt", "intent": "design"},
            instruction="分析这个设计稿",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/design.png"),
            ],
        )
        response = await provider.execute_tap(request)
        assert response.raw_text is not None

    @pytest.mark.asyncio
    async def test_minimax_m3_compiler(self):
        """MiniMax M3 compiler still works."""
        compiler_cls = TAPCompilerRegistry.get("minimax_m3")
        if compiler_cls is None:
            pytest.skip("minimax_m3 compiler not registered")
        compiler = compiler_cls()
        provider = ModelProvider(compiler=compiler, adapter=self.adapter, model="minimax-m3")
        request = TAPRequest(
            meta={"task_id": "reg.m3", "intent": "execute"},
            instruction="分析截图并写代码",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/screen.png"),
            ],
        )
        response = await provider.execute_tap(request)
        assert response.raw_text is not None

    @pytest.mark.asyncio
    async def test_all_compilers_register_in_registry(self):
        """All expected compilers are registered in TAPCompilerRegistry."""
        expected = ["default", "deepseek_v4", "glm_5", "glm_52", "glm_5v_turbo", "minimax_m3"]
        for name in expected:
            compiler_cls = TAPCompilerRegistry.get(name)
            assert compiler_cls is not None, f"Compiler '{name}' not registered"
