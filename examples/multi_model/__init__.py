"""examples/multi_model — Multi-model collaboration example using teragent

This example demonstrates teragent's Phase 3 multi-model collaboration:
  - ModelRouter: Intelligent model selection based on intent, multimodal, cost
  - PipelineManager: Dynamic pipeline profile switching
  - CrossModelCostTracker: Multi-model cost tracking with budget control
  - Per-stage model assignment (Design→V4-Pro, Plan→GLM-5, etc.)

Usage:
    export DEEPSEEK_API_KEY=your_key
    export MINIMAX_API_KEY=your_key
    export GLM_API_KEY=your_key
    python -m examples.multi_model
"""

from __future__ import annotations

import asyncio

import teragent
from teragent.core.tap import LongHorizonConfig, MultimodalContent, TAPRequest
from teragent.reliability.budget import CrossModelCostTracker, MonthlyBudgetConfig
from teragent.router.model_router import ModelRouter, PipelineProfile, RoutingTable


async def demo_multi_model_compilation() -> None:
    """Demo 1: Same request, different compilers → different optimized prompts"""
    print("\n" + "=" * 70)
    print("Demo 1: Multi-Model Compilation (V4/M3/GLM-5)")
    print("=" * 70)

    # Create providers for each model
    providers = {}

    try:
        providers["openai_compatible.deepseek_v4_flash"] = teragent.create_provider(
            compiler="deepseek_v4",
            adapter="mock",
            model="deepseek-v4-flash",
            compiler_variant="flash",
        )
    except Exception as e:
        print(f"  (Skipping V4-Flash: {e})")

    try:
        providers["openai_compatible.deepseek_v4_pro"] = teragent.create_provider(
            compiler="deepseek_v4",
            adapter="mock",
            model="deepseek-v4-pro",
            compiler_variant="pro",
        )
    except Exception as e:
        print(f"  (Skipping V4-Pro: {e})")

    try:
        providers["openai_compatible.minimax_m3"] = teragent.create_provider(
            compiler="minimax_m3",
            adapter="mock",
            model="minimax-m3",
        )
    except Exception as e:
        print(f"  (Skipping M3: {e})")

    try:
        providers["openai_compatible.glm_5"] = teragent.create_provider(
            compiler="glm_5",
            adapter="mock",
            model="glm-5",
        )
    except Exception as e:
        print(f"  (Skipping GLM-5: {e})")

    # Same request, different compilers
    tap_request = teragent.TAPRequest(
        meta={"task_id": "multi-1", "intent": "execute"},
        instruction="Write a function to check if a string is a palindrome",
        constraints=["Python 3.10+", "Include type hints"],
        output_format_hint="<file path='palindrome.py'>complete code</file>",
    )

    for name, provider in providers.items():
        try:
            response = await provider.execute_tap(tap_request)
            print(f"\n  Model: {name}")
            print(f"  Response: {response.raw_text[:100]}...")
        except Exception as e:
            print(f"\n  Model: {name} — Error: {e}")


async def demo_intelligent_routing() -> None:
    """Demo 2: ModelRouter intelligent routing based on request characteristics"""
    print("\n" + "=" * 70)
    print("Demo 2: Intelligent Model Routing")
    print("=" * 70)

    # Create router with default routing table
    router = ModelRouter()

    # Test 1: Intent-based routing
    design_request = TAPRequest(
        meta={"task_id": "1", "intent": "design"},
        instruction="Design a REST API architecture",
    )
    decision = router.route(design_request)
    print(f"\n  [Design intent] → {decision.selected_driver} (reason: {decision.reason.value})")

    # Test 2: Chat intent → V4-Flash (cost savings)
    chat_request = TAPRequest(
        meta={"task_id": "2", "intent": "chat"},
        instruction="Hello, how are you?",
    )
    decision = router.route(chat_request)
    print(f"  [Chat intent] → {decision.selected_driver} (reason: {decision.reason.value})")

    # Test 3: Multimodal override → M3
    multimodal_request = TAPRequest(
        meta={"task_id": "3", "intent": "execute"},
        instruction="Analyze this screenshot",
        multimodal_context=[MultimodalContent(type="image_url", url="https://example.com/screenshot.png")],
    )
    decision = router.route(multimodal_request)
    print(f"  [Multimodal] → {decision.selected_driver} (reason: {decision.reason.value})")

    # Test 4: Long-horizon override → GLM-5
    long_task_request = TAPRequest(
        meta={"task_id": "4", "intent": "execute"},
        instruction="Implement the full project autonomously",
        long_horizon=LongHorizonConfig(max_duration_hours=4.0),
    )
    decision = router.route(long_task_request)
    print(f"  [Long-horizon] → {decision.selected_driver} (reason: {decision.reason.value})")

    # Test 5: Context > 200K → V4/M3 (not GLM-5)
    large_context_request = TAPRequest(
        meta={"task_id": "5", "intent": "review"},
        instruction="Review this codebase",
        context={"design": "x" * 800_000, "plan": "y" * 200_000},  # ~300K tokens
    )
    decision = router.route(large_context_request)
    print(f"  [Large context] → {decision.selected_driver} (reason: {decision.reason.value})")

    # Print decision summary
    summary = router.get_decision_summary()
    print(f"\n  Routing summary: {summary['total_decisions']} decisions")
    print(f"  By reason: {summary['by_reason']}")
    print(f"  By driver: {summary['by_driver']}")


async def demo_pipeline_switching() -> None:
    """Demo 3: Pipeline profile switching"""
    print("\n" + "=" * 70)
    print("Demo 3: Pipeline Dynamic Switching")
    print("=" * 70)

    router = ModelRouter()
    pm = router.pipeline_manager

    # Show available profiles
    print(f"\n  Available profiles: {pm.list_profiles()}")

    # Default profile: per-stage model assignment
    request = TAPRequest(
        meta={"task_id": "p1", "intent": "execute"},
        instruction="Write code",
    )

    for stage in ["design", "plan", "execute", "review"]:
        driver = pm.get_driver(stage)
        print(f"  [default] {stage} → {driver}")

    # Switch to budget profile
    pm.set_active_profile("budget")
    print(f"\n  Switched to profile: {pm.active_profile_name}")
    for stage in ["design", "plan", "execute", "review"]:
        driver = pm.get_driver(stage)
        print(f"  [budget] {stage} → {driver}")

    # Switch to multimodal profile
    pm.set_active_profile("multimodal")
    print(f"\n  Switched to profile: {pm.active_profile_name}")
    for stage in ["design", "plan", "execute", "review"]:
        driver = pm.get_driver(stage)
        print(f"  [multimodal] {stage} → {driver}")

    # Route using active profile
    pm.set_active_profile("default")
    decision = router.route_for_stage("design", request)
    print(f"\n  Pipeline routing for 'design': {decision.selected_driver} (reason: {decision.reason.value})")


async def demo_cost_tracking() -> None:
    """Demo 4: Cross-model cost tracking with budget control"""
    print("\n" + "=" * 70)
    print("Demo 4: Cross-Model Cost Tracking & Budget Control")
    print("=" * 70)

    tracker = CrossModelCostTracker()
    tracker.set_monthly_budget(MonthlyBudgetConfig(
        limit_cny=100.0,
        warning_threshold=0.8,
        critical_threshold=0.95,
    ))

    # Record costs for different models
    pricing_v4_flash = {"prompt_per_million": 1.0, "completion_per_million": 2.0, "cache_hit_per_million": 0.1}
    pricing_v4_pro = {"prompt_per_million": 4.0, "completion_per_million": 16.0, "cache_hit_per_million": 0.4}
    pricing_glm = {"prompt_per_million": 2.0, "completion_per_million": 8.0}
    pricing_m3 = {"prompt_per_million": 1.0, "completion_per_million": 2.0}

    # Simulate a day of usage
    tracker.record_from_tap_response(
        driver_name="openai_compatible.deepseek_v4_pro",
        compiler="deepseek_v4", model="deepseek-v4-pro", intent="design",
        prompt_tokens=50_000, completion_tokens=10_000,
        cache_hit_tokens=30_000, pricing=pricing_v4_pro,
    )
    tracker.record_from_tap_response(
        driver_name="openai_compatible.deepseek_v4_flash",
        compiler="deepseek_v4", model="deepseek-v4-flash", intent="execute",
        prompt_tokens=100_000, completion_tokens=30_000,
        cache_hit_tokens=60_000, pricing=pricing_v4_flash,
    )
    tracker.record_from_tap_response(
        driver_name="openai_compatible.glm_5",
        compiler="glm_5", model="glm-5", intent="plan",
        prompt_tokens=30_000, completion_tokens=8_000, pricing=pricing_glm,
    )
    tracker.record_from_tap_response(
        driver_name="openai_compatible.minimax_m3",
        compiler="minimax_m3", model="minimax-m3", intent="execute",
        prompt_tokens=20_000, completion_tokens=5_000, pricing=pricing_m3,
    )

    # Generate reports
    report_by_model = tracker.generate_report(group_by="model")
    print(f"\n  Cost by model:")
    for model, stats in report_by_model["groups"].items():
        print(f"    {model}: ¥{stats['total_cost_cny']:.4f} ({stats['total_calls']} calls)")

    report_by_intent = tracker.generate_report(group_by="intent")
    print(f"\n  Cost by intent:")
    for intent, stats in report_by_intent["groups"].items():
        print(f"    {intent}: ¥{stats['total_cost_cny']:.4f}")

    # Cache savings
    cache_stats = tracker.get_cache_savings()
    print(f"\n  Cache savings:")
    print(f"    Total saved: ¥{cache_stats['total_cost_saved_cny']:.4f}")
    print(f"    Cache hit rate: {cache_stats['overall_cache_hit_rate']:.1%}")

    # Budget status
    budget = report_by_model["budget"]
    print(f"\n  Budget status: {budget['level']} (¥{tracker.get_total_cost():.2f} / ¥100.00)")


async def demo_degradation() -> None:
    """Demo 5: Model degradation when primary is unavailable"""
    print("\n" + "=" * 70)
    print("Demo 5: Degradation Strategy")
    print("=" * 70)

    from teragent.reliability.circuit_breaker import BreakerState

    router = ModelRouter()

    # Simulate V4-Pro circuit breaker being open
    router.update_circuit_breaker_state(
        "openai_compatible.deepseek_v4_pro",
        BreakerState(
            name="open",
            consecutive_failures=5,
            total_failures=10,
            total_successes=50,
            last_error="API timeout",
            last_failure_time=0.0,
            can_retry=False,
        ),
    )

    # Route a design request (normally → V4-Pro, but V4-Pro is down)
    request = TAPRequest(
        meta={"task_id": "deg-1", "intent": "design"},
        instruction="Design the API architecture",
    )
    decision = router.route(request)
    print(f"\n  Design with V4-Pro down → {decision.selected_driver} (reason: {decision.reason.value})")

    # Also simulate V4-Flash being down
    router.update_circuit_breaker_state(
        "openai_compatible.deepseek_v4_flash",
        BreakerState(
            name="open",
            consecutive_failures=3,
            total_failures=5,
            total_successes=30,
            last_error="Rate limit exceeded",
            last_failure_time=0.0,
            can_retry=False,
        ),
    )

    # Now GLM-5 should be the fallback
    request2 = TAPRequest(
        meta={"task_id": "deg-2", "intent": "chat"},
        instruction="Hello",
    )
    decision2 = router.route(request2)
    print(f"  Chat with V4-Pro+Flash down → {decision2.selected_driver} (reason: {decision2.reason.value})")


async def demo_multimodal_agent_flow() -> None:
    """Demo 6: Complete multimodal Agent flow using ModelRouter"""
    print("\n" + "=" * 70)
    print("Demo 6: Multimodal Agent Flow (Screenshot→Analyze→Code→Review)")
    print("=" * 70)

    router = ModelRouter()
    router.pipeline_manager.set_active_profile("default")

    # Step 1: Take screenshot with M3 (multimodal)
    screenshot_request = TAPRequest(
        meta={"task_id": "mm-1", "intent": "execute"},
        instruction="Take a screenshot of the current desktop",
        multimodal_context=[MultimodalContent(type="image_url", url="https://example.com/desktop.png")],
        desktop_context=teragent.DesktopContext(active_window="VS Code"),
    )
    decision1 = router.route(screenshot_request)
    print(f"\n  Step 1 (Screenshot): {decision1.selected_driver} (reason: {decision1.reason.value})")

    # Step 2: Analyze screenshot with M3
    analyze_request = TAPRequest(
        meta={"task_id": "mm-2", "intent": "review"},
        instruction="Analyze the screenshot and identify UI elements",
        multimodal_context=[MultimodalContent(type="image_url", url="https://example.com/desktop.png")],
    )
    decision2 = router.route(analyze_request)
    print(f"  Step 2 (Analyze):   {decision2.selected_driver} (reason: {decision2.reason.value})")

    # Step 3: Generate code with GLM-5 (via pipeline)
    code_request = TAPRequest(
        meta={"task_id": "mm-3", "intent": "execute"},
        instruction="Generate code to automate the identified actions",
    )
    decision3 = router.route_for_stage("execute", code_request)
    print(f"  Step 3 (Code Gen):  {decision3.selected_driver} (reason: {decision3.reason.value})")

    # Step 4: Review with V4-Pro (via pipeline)
    review_request = TAPRequest(
        meta={"task_id": "mm-4", "intent": "review"},
        instruction="Review the generated code for correctness",
    )
    decision4 = router.route_for_stage("review", review_request)
    print(f"  Step 4 (Review):    {decision4.selected_driver} (reason: {decision4.reason.value})")


async def demo_5v_turbo_coordination() -> None:
    """Demo 7: GLM-5V-Turbo + GLM-5.2 coordinated workflow"""
    print("\n" + "=" * 70)
    print("Demo 7: GLM-5V-Turbo + GLM-5.2 Vision→Code Coordination")
    print("=" * 70)

    from teragent.coordination.glm5v_coordinator import (
        GLM52VCoordinatedWorkflow,
        CoordinationConfig,
        CoordinationMode,
    )
    from teragent.core.adapters.mock import MockAdapter
    from teragent.core.compiler import TAPCompilerRegistry
    from teragent.core.provider import ModelProvider

    # Create vision and coding providers
    vision_compiler_cls = TAPCompilerRegistry.get("glm_5v_turbo")
    coding_compiler_cls = TAPCompilerRegistry.get("glm_52")

    if vision_compiler_cls and coding_compiler_cls:
        vision_provider = ModelProvider(
            compiler=vision_compiler_cls(mode="analysis"),
            adapter=MockAdapter(),
            model="glm-5v-turbo",
        )
        coding_provider = ModelProvider(
            compiler=coding_compiler_cls(),
            adapter=MockAdapter(),
            model="glm-5.2",
        )

        # Sequential mode: Vision → Code
        config_seq = CoordinationConfig(
            mode="sequential",
            context_sharing=True,
            inject_code_generation_hint=True,
        )
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=vision_provider,
            coding_provider=coding_provider,
            config=config_seq,
        )

        request = TAPRequest(
            meta={"task_id": "coord-1", "intent": "execute"},
            instruction="Implement this design as a React component",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/design-mock.png")
            ],
            constraints=["React 18+", "TypeScript"],
        )

        result = await workflow.execute(request)
        print(f"\n  [Sequential] Phase: {result.phase}")
        print(f"  [Sequential] Success: {result.success}")
        print(f"  [Sequential] Steps: {len(result.steps)}")
        if result.vision_analysis:
            print(f"  [Sequential] Vision analysis confidence: {result.vision_analysis.confidence:.2f}")
        if result.final_response:
            print(f"  [Sequential] Code output preview: {result.final_response.raw_text[:80]}...")

        # Verify mode: Vision → Code → Verify
        config_verify = CoordinationConfig(
            mode="verify",
            max_verification_rounds=1,
            verification_score_threshold=7.0,
        )
        workflow_verify = GLM52VCoordinatedWorkflow(
            vision_provider=vision_provider,
            coding_provider=coding_provider,
            config=config_verify,
        )

        result_verify = await workflow_verify.execute(request)
        print(f"\n  [Verify] Phase: {result_verify.phase}")
        print(f"  [Verify] Steps: {len(result_verify.steps)}")
        if result_verify.verification_result:
            print(f"  [Verify] Verification done: True")
    else:
        print("  (Skipping: GLM-5V-Turbo or GLM-5.2 compiler not available)")


async def main() -> None:
    """Run all Phase 3 multi-model collaboration demos"""
    print("=" * 70)
    print("TerAgent Phase 3: Multi-Model Collaboration Demo")
    print("  - ModelRouter: Intelligent model selection (now with GLM-5.2)")
    print("  - PipelineManager: Dynamic pipeline switching")
    print("  - CrossModelCostTracker: Cost tracking & budget control")
    print("  - Degradation: Automatic fallback on failure")
    print("  - GLM-5V-Turbo + GLM-5.2: Vision→Code coordination")
    print("=" * 70)

    await demo_multi_model_compilation()
    await demo_intelligent_routing()
    await demo_pipeline_switching()
    await demo_cost_tracking()
    await demo_degradation()
    await demo_multimodal_agent_flow()
    await demo_5v_turbo_coordination()

    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
