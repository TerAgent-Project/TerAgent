"""teragent.router.model_router — Intelligent model routing + Pipeline dynamic allocation

The ModelRouter selects the optimal model provider for each TAP request based on
six routing dimensions:
    1. Intent type (design/plan/execute/review/chat)
    2. Multimodal requirements (images, video, desktop)
    3. Context length (>200K excludes GLM-5)
    4. Task duration (long-horizon → GLM-5)
    5. Cost budget (Flash < Pro < M3/GLM-5)
    6. Load / circuit breaker state (degradation to fallback)

The PipelineManager extends this with named Pipeline profiles and runtime switching:
    - "default": Design→V4-Pro, Plan→GLM-5, Execute→GLM-5, Review→V4-Pro
    - "budget": All stages use V4-Flash (extreme cost savings)
    - "multimodal": All stages use M3 (vision/video/desktop)
    - Custom profiles via agent.toml

Design reference: design.md §6 "跨模型协同与智能路由"
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "ModelRouter",
    "PipelineManager",
    "PipelineProfile",
    "RoutingDecision",
    "RoutingReason",
    "RoutingTable",
]

from teragent.core.tap import TAPRequest

logger = logging.getLogger(__name__)


# ===== Enums & Data Classes =====


class RoutingReason(Enum):
    """Why the router selected a particular model"""

    INTENT = "intent"                      # Default intent-based routing
    MULTIMODAL_OVERRIDE = "multimodal"      # Has multimodal content → M3
    DESKTOP_OVERRIDE = "desktop"            # Has desktop context → M3
    VIDEO_OVERRIDE = "video"                # Has video content → M3
    CONTEXT_LENGTH_OVERRIDE = "context_gt_200k"  # Context >200K → V4/M3
    LONG_HORIZON_OVERRIDE = "long_horizon"  # Long-horizon task → GLM-5
    COST_OPTIMIZATION = "cost"              # Budget constraint → cheaper model
    DEGRADATION = "degradation"             # Primary unavailable → fallback
    PIPELINE_PROFILE = "pipeline"           # Explicit pipeline profile assignment
    EXPLICIT = "explicit"                   # User explicitly specified model


@dataclass
class RoutingDecision:
    """Record of a routing decision

    Captures the selected model, the primary reason, and a trace of
    all routing steps evaluated (for debugging and logging).

    Attributes:
        selected_driver: Full driver name (e.g., "openai_compatible.deepseek_v4_pro")
        selected_compiler: Compiler name (e.g., "deepseek_v4")
        reason: Primary routing reason
        intent: The request's intent type
        trace: Ordered list of (reason, candidate, accepted/rejected) tuples
        timestamp: Decision timestamp (epoch seconds)
        estimated_cost: Estimated cost for this request with selected model (0.0 if unknown)
        context_tokens: Estimated context token count
    """

    selected_driver: str = ""
    selected_compiler: str = ""
    reason: RoutingReason = RoutingReason.INTENT
    intent: str = ""
    trace: list[tuple[str, str, str]] = field(default_factory=list)
    timestamp: float = 0.0
    estimated_cost: float = 0.0
    context_tokens: int = 0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def add_trace(self, reason: str, candidate: str, result: str) -> None:
        """Append a trace entry: (reason, candidate_model, "accepted"/"rejected")"""
        self.trace.append((reason, candidate, result))


@dataclass
class RoutingTable:
    """Configurable routing table with intent defaults and override rules

    Maps intent types to default driver names, with override rules
    for special scenarios (multimodal, long context, long-horizon).

    All fields have sensible defaults matching design.md §6.2.
    """

    # Intent → default driver name mapping
    # NOTE: These are CODE defaults used when no config file is loaded.
    # When agent.toml is present, PipelineManager.from_config() overrides these
    # with the [execution.pipeline] section values. See agent.toml for the
    # config-driven pipeline defaults (design→V4-Pro, plan→V4-Pro, execute→V4-Flash).
    intent_routing: dict[str, str] = field(default_factory=lambda: {
        "design": "openai_compatible.deepseek_v4_pro",
        "plan": "openai_compatible.glm_52",       # GLM-5.2 1M context + dual thinking
        "execute": "openai_compatible.glm_52",     # GLM-5.2 for complex coding
        "review": "openai_compatible.deepseek_v4_pro",
        "chat": "openai_compatible.deepseek_v4_flash",
        # Aliases for common intent names
        "code_generation": "openai_compatible.deepseek_v4_flash",
        "debug": "openai_compatible.deepseek_v4_flash",
        "replan": "openai_compatible.glm_52",      # GLM-5.2 for replan
        "sub_agent": "openai_compatible.deepseek_v4_flash",
        "chat_friendly": "openai_compatible.deepseek_v4_flash",
    })

    # Multimodal override: when request has multimodal content → M3
    multimodal_driver: str = "openai_compatible.minimax_m3"

    # Desktop override: when request has desktop context → M3
    desktop_driver: str = "openai_compatible.minimax_m3"

    # Context >200K override candidates (GLM-5.2 preferred for 1M usable context)
    long_context_candidates: list[str] = field(default_factory=lambda: [
        "openai_compatible.glm_52",              # GLM-5.2: 1M truly usable
        "openai_compatible.deepseek_v4_pro",     # V4-Pro: 1M context
        "openai_compatible.minimax_m3",           # M3: 1M context
    ])

    # Long-horizon override: when request has long_horizon config → GLM-5.2
    # (GLM-5.2 inherits 8h+ capability from GLM-5 + 1M context + dual thinking)
    long_horizon_driver: str = "openai_compatible.glm_52"

    # Cost-based fallback order (cheapest first)
    cost_fallback_order: list[str] = field(default_factory=lambda: [
        "openai_compatible.deepseek_v4_flash",
        "openai_compatible.glm_5",
        "openai_compatible.glm_52",
        "openai_compatible.deepseek_v4_pro",
        "openai_compatible.minimax_m3",
    ])

    # Degradation fallback map: primary → fallback
    degradation_map: dict[str, str] = field(default_factory=lambda: {
        "openai_compatible.deepseek_v4_pro": "openai_compatible.deepseek_v4_flash",
        "openai_compatible.deepseek_v4_flash": "openai_compatible.glm_5",
        "openai_compatible.glm_5": "openai_compatible.deepseek_v4_flash",
        "openai_compatible.glm_52": "openai_compatible.glm_5",         # GLM-5.2 → GLM-5 (200K fallback)
        "openai_compatible.glm_5v_turbo": "openai_compatible.minimax_m3", # 5V-Turbo → M3 (both multimodal)
        "openai_compatible.minimax_m3": "openai_compatible.deepseek_v4_pro",
    })

    # Per-model pricing (CNY per million tokens) for cost estimation
    # Format: {driver_name: {"prompt_per_million": float, "completion_per_million": float, "cache_hit_per_million": float}}
    model_pricing: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "openai_compatible.deepseek_v4_flash": {
            "prompt_per_million": 1.0,
            "completion_per_million": 2.0,
            "cache_hit_per_million": 0.1,
        },
        "openai_compatible.deepseek_v4_pro": {
            "prompt_per_million": 4.0,
            "completion_per_million": 16.0,
            "cache_hit_per_million": 0.4,
        },
        "openai_compatible.minimax_m3": {
            "prompt_per_million": 1.0,
            "completion_per_million": 2.0,
        },
        "openai_compatible.glm_5": {
            "prompt_per_million": 2.0,
            "completion_per_million": 8.0,
        },
        "openai_compatible.glm_52": {
            "prompt_per_million": 3.0,
            "completion_per_million": 10.0,
            "cache_hit_per_million": 0.3,
        },
        "openai_compatible.glm_5v_turbo": {
            "prompt_per_million": 1.5,
            "completion_per_million": 5.0,
        },
    })

    # Maximum context tokens per model (for context length override)
    max_context_per_model: dict[str, int] = field(default_factory=lambda: {
        "openai_compatible.deepseek_v4_flash": 1_000_000,
        "openai_compatible.deepseek_v4_pro": 1_000_000,
        "openai_compatible.minimax_m3": 1_000_000,
        "openai_compatible.glm_5": 200_000,
        "openai_compatible.glm_52": 1_000_000,      # GLM-5.2: 1M truly usable
        "openai_compatible.glm_5v_turbo": 128_000,  # 5V-Turbo: 128K
    })

    # Compiler inference from driver name
    compiler_map: dict[str, str] = field(default_factory=lambda: {
        "openai_compatible.deepseek_v4_flash": "deepseek_v4",
        "openai_compatible.deepseek_v4_pro": "deepseek_v4",
        "openai_compatible.minimax_m3": "minimax_m3",
        "openai_compatible.glm_5": "glm_5",
        "openai_compatible.glm_52": "glm_52",
        "openai_compatible.glm_5v_turbo": "glm_5v_turbo",
    })

    def resolve_compiler(self, driver_name: str) -> str:
        """Resolve compiler name from driver name

        Args:
            driver_name: Full driver name (e.g., "openai_compatible.deepseek_v4_pro")

        Returns:
            Compiler name string, or "default" if not found
        """
        return self.compiler_map.get(driver_name, "default")

    def get_intent_driver(self, intent: str) -> str:
        """Get default driver for an intent type

        Args:
            intent: Intent string (e.g., "design", "plan", "execute")

        Returns:
            Driver name string, or intent_routing["chat"] if intent not found
        """
        return self.intent_routing.get(intent, self.intent_routing.get("chat", ""))

    def get_pricing(self, driver_name: str) -> dict[str, float]:
        """Get pricing dict for a model

        Args:
            driver_name: Full driver name

        Returns:
            Pricing dict with prompt_per_million, completion_per_million, etc.
        """
        return self.model_pricing.get(driver_name, {})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoutingTable:
        """Create RoutingTable from a config dict (e.g., from agent.toml)

        Args:
            data: Dict with optional routing configuration keys

        Returns:
            RoutingTable instance with overrides applied
        """
        table = cls()

        # Override intent routing if provided
        if "intent_routing" in data:
            table.intent_routing.update(data["intent_routing"])

        # Override special driver assignments
        if "multimodal_driver" in data:
            table.multimodal_driver = data["multimodal_driver"]
        if "desktop_driver" in data:
            table.desktop_driver = data["desktop_driver"]
        if "long_horizon_driver" in data:
            table.long_horizon_driver = data["long_horizon_driver"]

        # Override long context candidates
        if "long_context_candidates" in data:
            table.long_context_candidates = data["long_context_candidates"]

        # Override cost fallback order
        if "cost_fallback_order" in data:
            table.cost_fallback_order = data["cost_fallback_order"]

        # Override degradation map
        if "degradation_map" in data:
            table.degradation_map.update(data["degradation_map"])

        # Override pricing
        if "model_pricing" in data:
            table.model_pricing.update(data["model_pricing"])

        # Override max context
        if "max_context_per_model" in data:
            table.max_context_per_model.update(data["max_context_per_model"])

        return table


@dataclass
class PipelineProfile:
    """Named pipeline configuration mapping stages to driver names

    A Pipeline profile defines which model driver to use at each
    stage of the agent loop (design, plan, execute, review).

    Attributes:
        name: Profile name (e.g., "default", "budget", "multimodal")
        description: Human-readable description
        design_driver: Driver for design stage
        plan_driver: Driver for plan stage
        execute_driver: Driver for execute stage
        review_driver: Driver for review stage
    """

    name: str = "default"
    description: str = ""
    design_driver: str = ""
    plan_driver: str = ""
    execute_driver: str = ""
    review_driver: str = ""

    def get_driver_for_stage(self, stage: str) -> str:
        """Get the driver name for a pipeline stage

        Args:
            stage: One of "design", "plan", "execute", "review"

        Returns:
            Driver name string, or empty string if stage not found
        """
        return {
            "design": self.design_driver,
            "plan": self.plan_driver,
            "execute": self.execute_driver,
            "review": self.review_driver,
        }.get(stage, "")

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> PipelineProfile:
        """Create PipelineProfile from a config dict

        Args:
            name: Profile name
            data: Dict with stage → driver mappings

        Returns:
            PipelineProfile instance
        """
        return cls(
            name=name,
            description=data.get("description", ""),
            design_driver=data.get("design_driver", ""),
            plan_driver=data.get("plan_driver", ""),
            execute_driver=data.get("execute_driver", ""),
            review_driver=data.get("review_driver", ""),
        )


class PipelineManager:
    """Runtime pipeline switching and per-stage model assignment

    Manages multiple Pipeline profiles and supports runtime switching
    between them. The ModelRouter delegates to PipelineManager when
    a request comes from a known pipeline stage.

    Usage::

        pm = PipelineManager()
        pm.register_profile(PipelineProfile(
            name="default",
            design_driver="openai_compatible.deepseek_v4_pro",
            plan_driver="openai_compatible.glm_5",
            execute_driver="openai_compatible.glm_5",
            review_driver="openai_compatible.deepseek_v4_pro",
        ))
        pm.register_profile(PipelineProfile(name="budget", ...))
        pm.set_active_profile("budget")
        driver = pm.get_driver("execute")  # Returns budget's execute_driver
    """

    def __init__(self, routing_table: RoutingTable | None = None) -> None:
        self._profiles: dict[str, PipelineProfile] = {}
        self._active_profile: str = "default"
        self._routing_table = routing_table or RoutingTable()

        # Register built-in profiles
        self._register_builtin_profiles()

    def _register_builtin_profiles(self) -> None:
        """Register the three built-in pipeline profiles from design.md §6.5"""
        rt = self._routing_table

        # Default profile: optimal quality per stage
        self.register_profile(PipelineProfile(
            name="default",
            description="默认配置：设计/审查→V4-Pro, 规划/执行→GLM-5.2",
            design_driver=rt.get_intent_driver("design"),
            plan_driver=rt.get_intent_driver("plan"),
            execute_driver=rt.get_intent_driver("execute"),
            review_driver=rt.get_intent_driver("review"),
        ))

        # Budget profile: all V4-Flash
        self.register_profile(PipelineProfile(
            name="budget",
            description="极致性价比：所有阶段使用 V4-Flash",
            design_driver="openai_compatible.deepseek_v4_flash",
            plan_driver="openai_compatible.deepseek_v4_flash",
            execute_driver="openai_compatible.deepseek_v4_flash",
            review_driver="openai_compatible.deepseek_v4_flash",
        ))

        # Multimodal profile: all M3
        self.register_profile(PipelineProfile(
            name="multimodal",
            description="多模态模式：所有阶段使用 M3",
            design_driver="openai_compatible.minimax_m3",
            plan_driver="openai_compatible.minimax_m3",
            execute_driver="openai_compatible.minimax_m3",
            review_driver="openai_compatible.minimax_m3",
        ))

        # Deep thinking profile: GLM-5.2 Max mode for all stages (highest quality)
        self.register_profile(PipelineProfile(
            name="deep_thinking",
            description="深度思考模式：所有阶段使用 GLM-5.2（Max 思考模式）",
            design_driver="openai_compatible.glm_52",
            plan_driver="openai_compatible.glm_52",
            execute_driver="openai_compatible.glm_52",
            review_driver="openai_compatible.glm_52",
        ))

    def register_profile(self, profile: PipelineProfile) -> None:
        """Register a pipeline profile

        Args:
            profile: PipelineProfile instance with name and stage mappings
        """
        self._profiles[profile.name] = profile
        logger.debug(f"Registered pipeline profile: {profile.name}")

    def set_active_profile(self, name: str) -> bool:
        """Switch to a named pipeline profile at runtime

        Args:
            name: Profile name to activate

        Returns:
            True if the profile was found and activated, False otherwise
        """
        if name in self._profiles:
            old = self._active_profile
            self._active_profile = name
            logger.info(f"Pipeline profile switched: {old} → {name}")
            return True
        logger.warning(f"Pipeline profile '{name}' not found, staying on '{self._active_profile}'")
        return False

    @property
    def active_profile_name(self) -> str:
        """Name of the currently active pipeline profile"""
        return self._active_profile

    @property
    def active_profile(self) -> PipelineProfile:
        """The currently active PipelineProfile"""
        return self._profiles.get(self._active_profile, PipelineProfile(name="default"))

    def get_driver(self, stage: str) -> str:
        """Get the driver name for a stage from the active profile

        Args:
            stage: Pipeline stage ("design", "plan", "execute", "review")

        Returns:
            Driver name string from active profile
        """
        return self.active_profile.get_driver_for_stage(stage)

    def list_profiles(self) -> list[str]:
        """List all registered profile names"""
        return list(self._profiles.keys())

    def get_profile(self, name: str) -> PipelineProfile | None:
        """Get a profile by name

        Args:
            name: Profile name

        Returns:
            PipelineProfile if found, None otherwise
        """
        return self._profiles.get(name)

    @classmethod
    def from_config(cls, config: dict[str, Any], routing_table: RoutingTable | None = None) -> PipelineManager:
        """Create PipelineManager from a TOML config dict

        Reads [execution.pipeline] and [execution.pipeline.profiles.*] sections.

        Args:
            config: Full TOML config dict
            routing_table: Optional RoutingTable instance

        Returns:
            PipelineManager with profiles loaded from config
        """
        rt = routing_table or RoutingTable()
        pm = cls(routing_table=rt)

        # Load the base pipeline as "default" profile (overrides built-in)
        pipeline_section = config.get("execution", {}).get("pipeline", {})
        if pipeline_section:
            # Extract base stage → driver mappings
            design = pipeline_section.get("design_driver", "")
            plan = pipeline_section.get("plan_driver", "")
            execute = pipeline_section.get("execute_driver", "")
            review = pipeline_section.get("review_driver", "")

            if design or plan or execute or review:
                pm.register_profile(PipelineProfile(
                    name="default",
                    description="From agent.toml [execution.pipeline]",
                    design_driver=design,
                    plan_driver=plan,
                    execute_driver=execute,
                    review_driver=review,
                ))

        # Load named profiles from [execution.pipeline.profiles.*]
        profiles_section = pipeline_section.get("profiles", {})
        if isinstance(profiles_section, dict):
            for profile_name, profile_data in profiles_section.items():
                if isinstance(profile_data, dict):
                    pm.register_profile(PipelineProfile.from_dict(profile_name, profile_data))

        return pm


class ModelRouter:
    """Intelligent model router for multi-model collaboration

    Routes TAP requests to the optimal model provider based on:
      1. Multimodal check → M3 if has multimodal/desktop/video content
      2. Context length check → V4/M3 if context > 200K
      3. Long-horizon check → GLM-5 if long_horizon config present
      4. Intent matching → default routing table
      5. Cost evaluation → downgrade if budget constrained
      6. Load / degradation → fallback if primary unavailable

    All routing decisions are logged with full trace information.

    Usage::

        router = ModelRouter(
            available_providers={"openai_compatible.glm_5": glm_provider, ...},
            routing_table=RoutingTable(),
        )
        decision = router.route(tap_request)
        provider = router.get_provider(decision.selected_driver)
        response = await provider.execute_tap(tap_request)
    """

    def __init__(
        self,
        available_providers: dict[str, Any] | None = None,
        available_driver_configs: dict[str, Any] | None = None,
        routing_table: RoutingTable | None = None,
        pipeline_manager: PipelineManager | None = None,
        circuit_breaker_states: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ModelRouter

        Args:
            available_providers: Map of driver_name → ModelProvider instances
            available_driver_configs: Map of driver_name → DriverConfig instances
            routing_table: RoutingTable with routing rules (default if None)
            pipeline_manager: PipelineManager for pipeline-based routing
            circuit_breaker_states: Map of driver_name → BreakerState for degradation checks
        """
        self._providers = available_providers or {}
        self._driver_configs = available_driver_configs or {}
        self._routing_table = routing_table or RoutingTable()
        self._pipeline_manager = pipeline_manager or PipelineManager(self._routing_table)
        self._circuit_breaker_states = circuit_breaker_states or {}

        # Decision log (thread-safe)
        self._decision_log: list[RoutingDecision] = []
        self._lock = threading.Lock()

        # Monthly budget tracker
        self._monthly_budget_cny: float = 0.0
        self._monthly_budget_limit_cny: float = 0.0  # 0 = no limit
        self._budget_warning_threshold: float = 0.8   # Warn at 80%
        self._budget_auto_downgrade: bool = True       # Auto-downgrade at 100%

    # ===== Core routing =====

    def route(self, request: TAPRequest) -> RoutingDecision:
        """Route a TAP request to the optimal model

        Follows the 6-step decision flow from design.md §6.3:
          1. Multimodal check → M3
          2. Context length check → V4/M3
          3. Long-horizon check → GLM-5
          4. Intent matching → default routing table
          5. Cost evaluation → cheaper model if budget constrained
          6. Load/degradation check → fallback if primary unavailable

        Args:
            request: The TAP request to route

        Returns:
            RoutingDecision with selected driver and trace
        """
        decision = RoutingDecision()
        intent = request.meta.get("intent", "chat")
        decision.intent = intent
        context_tokens = request.estimate_prompt_tokens()
        decision.context_tokens = context_tokens

        # Step 1: Multimodal override
        selected = self._check_multimodal_override(request, decision)
        if selected:
            decision.selected_driver = selected
            decision.selected_compiler = self._routing_table.resolve_compiler(selected)
            decision.estimated_cost = self._estimate_cost(selected, context_tokens)
            self._log_decision(decision)
            return decision

        # Step 2: Context length override
        selected = self._check_context_length_override(request, decision)
        if selected:
            decision.selected_driver = selected
            decision.selected_compiler = self._routing_table.resolve_compiler(selected)
            decision.estimated_cost = self._estimate_cost(selected, context_tokens)
            self._log_decision(decision)
            return decision

        # Step 3: Long-horizon override
        selected = self._check_long_horizon_override(request, decision)
        if selected:
            decision.selected_driver = selected
            decision.selected_compiler = self._routing_table.resolve_compiler(selected)
            decision.estimated_cost = self._estimate_cost(selected, context_tokens)
            self._log_decision(decision)
            return decision

        # Step 4: Intent matching (from routing table or pipeline)
        selected = self._check_intent_routing(request, decision)
        decision.selected_driver = selected
        decision.selected_compiler = self._routing_table.resolve_compiler(selected)

        # Step 5: Cost evaluation
        selected = self._check_cost_optimization(request, decision, selected)
        decision.selected_driver = selected
        decision.selected_compiler = self._routing_table.resolve_compiler(selected)

        # Step 6: Degradation check
        selected = self._check_degradation(request, decision, selected)
        decision.selected_driver = selected
        decision.selected_compiler = self._routing_table.resolve_compiler(selected)

        decision.estimated_cost = self._estimate_cost(selected, context_tokens)
        self._log_decision(decision)
        return decision

    def route_for_stage(self, stage: str, request: TAPRequest) -> RoutingDecision:
        """Route a TAP request using the active pipeline profile for a specific stage

        This method uses the PipelineManager to determine the driver for a
        given stage, then falls through the normal override checks (multimodal,
        context, etc.).

        Args:
            stage: Pipeline stage ("design", "plan", "execute", "review")
            request: The TAP request to route

        Returns:
            RoutingDecision with pipeline-based routing
        """
        # Start with pipeline profile assignment
        profile_driver = self._pipeline_manager.get_driver(stage)
        decision = RoutingDecision(
            selected_driver=profile_driver,
            selected_compiler=self._routing_table.resolve_compiler(profile_driver),
            reason=RoutingReason.PIPELINE_PROFILE,
            intent=request.meta.get("intent", stage),
        )
        decision.add_trace("pipeline_profile", profile_driver, "accepted")
        context_tokens = request.estimate_prompt_tokens()
        decision.context_tokens = context_tokens

        # Still apply multimodal/context/long-horizon overrides
        override = self._check_multimodal_override(request, RoutingDecision())
        if override:
            decision.selected_driver = override
            decision.selected_compiler = self._routing_table.resolve_compiler(override)
            decision.reason = RoutingReason.MULTIMODAL_OVERRIDE
            decision.add_trace("multimodal_override", override, "accepted (overrides pipeline)")

        override = self._check_context_length_override(request, RoutingDecision())
        if override:
            decision.selected_driver = override
            decision.selected_compiler = self._routing_table.resolve_compiler(override)
            decision.reason = RoutingReason.CONTEXT_LENGTH_OVERRIDE
            decision.add_trace("context_length_override", override, "accepted (overrides pipeline)")

        override = self._check_long_horizon_override(request, RoutingDecision())
        if override:
            decision.selected_driver = override
            decision.selected_compiler = self._routing_table.resolve_compiler(override)
            decision.reason = RoutingReason.LONG_HORIZON_OVERRIDE
            decision.add_trace("long_horizon_override", override, "accepted (overrides pipeline)")

        # Degradation check
        final = self._check_degradation(request, decision, decision.selected_driver)
        decision.selected_driver = final
        decision.selected_compiler = self._routing_table.resolve_compiler(final)

        decision.estimated_cost = self._estimate_cost(decision.selected_driver, context_tokens)
        self._log_decision(decision)
        return decision

    # ===== Override checks =====

    def _check_multimodal_override(self, request: TAPRequest, decision: RoutingDecision) -> str:
        """Step 1: Check if request has multimodal content → route to M3

        Returns:
            Driver name if override applies, empty string otherwise
        """
        # Check for video content
        if request.multimodal_context:
            for mc in request.multimodal_context:
                if mc.type == "video_url":
                    driver = self._routing_table.multimodal_driver
                    decision.add_trace("video_content", driver, "accepted")
                    decision.reason = RoutingReason.VIDEO_OVERRIDE
                    return driver

        # Check for desktop context
        if request.has_desktop_context:
            driver = self._routing_table.desktop_driver
            decision.add_trace("desktop_context", driver, "accepted")
            decision.reason = RoutingReason.DESKTOP_OVERRIDE
            return driver

        # Check for any multimodal content (image/video)
        if request.has_multimodal:
            driver = self._routing_table.multimodal_driver
            decision.add_trace("multimodal_content", driver, "accepted")
            decision.reason = RoutingReason.MULTIMODAL_OVERRIDE
            return driver

        return ""

    def _check_context_length_override(self, request: TAPRequest, decision: RoutingDecision) -> str:
        """Step 2: Check if context > 200K → exclude GLM-5

        When estimated context tokens exceed 200K, GLM-5 (200K window)
        cannot handle it. Route to V4-Pro or M3 instead.

        Returns:
            Driver name if override applies, empty string otherwise
        """
        context_tokens = request.estimate_prompt_tokens()
        if context_tokens > 200_000:
            candidates = self._routing_table.long_context_candidates
            # Pick the first available candidate
            for candidate in candidates:
                if self._is_driver_available(candidate):
                    decision.add_trace("context_gt_200k", candidate, "accepted")
                    decision.reason = RoutingReason.CONTEXT_LENGTH_OVERRIDE
                    return candidate
            # Fallback: pick first candidate even if not verified available
            if candidates:
                decision.add_trace("context_gt_200k", candidates[0], "accepted (unverified)")
                decision.reason = RoutingReason.CONTEXT_LENGTH_OVERRIDE
                return candidates[0]

        return ""

    def _check_long_horizon_override(self, request: TAPRequest, decision: RoutingDecision) -> str:
        """Step 3: Check if request has long_horizon config → route to GLM-5

        Returns:
            Driver name if override applies, empty string otherwise
        """
        if request.is_long_horizon:
            driver = self._routing_table.long_horizon_driver
            decision.add_trace("long_horizon", driver, "accepted")
            decision.reason = RoutingReason.LONG_HORIZON_OVERRIDE
            return driver

        return ""

    def _check_intent_routing(self, request: TAPRequest, decision: RoutingDecision) -> str:
        """Step 4: Match intent to default routing table

        Returns:
            Driver name from intent routing table
        """
        intent = request.meta.get("intent", "chat")
        driver = self._routing_table.get_intent_driver(intent)
        if driver:
            decision.add_trace("intent_routing", driver, "accepted")
            decision.reason = RoutingReason.INTENT
        else:
            # Fallback to chat driver
            driver = self._routing_table.get_intent_driver("chat")
            decision.add_trace("intent_routing", driver, "accepted (fallback to chat)")
            decision.reason = RoutingReason.INTENT

        return driver

    def _check_cost_optimization(self, request: TAPRequest, decision: RoutingDecision, current_driver: str) -> str:
        """Step 5: Evaluate cost and potentially downgrade to cheaper model

        If monthly budget is configured and utilization is high, downgrade
        to a cheaper model (typically V4-Flash).

        Returns:
            Driver name (potentially downgraded for cost savings)
        """
        if self._monthly_budget_limit_cny <= 0:
            return current_driver

        utilization = self._monthly_budget_cny / self._monthly_budget_limit_cny

        if utilization >= 1.0 and self._budget_auto_downgrade:
            # Budget exhausted → downgrade to cheapest available
            for candidate in self._routing_table.cost_fallback_order:
                if self._is_driver_available(candidate):
                    decision.add_trace("cost_exhausted", candidate, "accepted (budget exhausted)")
                    decision.reason = RoutingReason.COST_OPTIMIZATION
                    return candidate

        elif utilization >= self._budget_warning_threshold:
            # Budget warning → suggest cheaper but don't force
            # Only downgrade if current driver is expensive
            current_pricing = self._routing_table.get_pricing(current_driver)
            current_cost = current_pricing.get("prompt_per_million", 0.0)
            # If current is more expensive than V4-Flash, suggest downgrade
            flash_pricing = self._routing_table.get_pricing("openai_compatible.deepseek_v4_flash")
            flash_cost = flash_pricing.get("prompt_per_million", 0.0)
            if current_cost > flash_cost and self._is_driver_available("openai_compatible.deepseek_v4_flash"):
                decision.add_trace(
                    "cost_warning",
                    "openai_compatible.deepseek_v4_flash",
                    "accepted (budget warning, downgraded from expensive model)"
                )
                decision.reason = RoutingReason.COST_OPTIMIZATION
                return "openai_compatible.deepseek_v4_flash"

        return current_driver

    def _check_degradation(self, request: TAPRequest, decision: RoutingDecision, current_driver: str) -> str:
        """Step 6: Check circuit breaker state and degrade to fallback if needed

        If the primary model's circuit breaker is open (too many failures),
        automatically fall back to the degradation map's fallback.

        Returns:
            Driver name (potentially degraded to fallback)
        """
        if not self._is_circuit_open(current_driver):
            return current_driver

        # Primary is unavailable → find fallback
        fallback = self._routing_table.degradation_map.get(current_driver, "")
        if fallback and self._is_driver_available(fallback) and not self._is_circuit_open(fallback):
            decision.add_trace("degradation", fallback, f"accepted (primary {current_driver} unavailable)")
            decision.reason = RoutingReason.DEGRADATION
            logger.warning(
                f"Model degradation: {current_driver} circuit open, "
                f"falling back to {fallback}"
            )
            return fallback

        # Fallback also unavailable → try cost_fallback_order
        for candidate in self._routing_table.cost_fallback_order:
            if candidate != current_driver and self._is_driver_available(candidate) and not self._is_circuit_open(candidate):
                decision.add_trace("degradation", candidate, "accepted (both primary and fallback unavailable)")
                decision.reason = RoutingReason.DEGRADATION
                logger.warning(
                    f"Model degradation: primary {current_driver} and fallback unavailable, "
                    f"using {candidate}"
                )
                return candidate

        # All models unavailable → return current_driver (will likely fail, but at least we tried)
        decision.add_trace("degradation", current_driver, "no_fallback_available")
        logger.error(
            f"All model fallbacks exhausted for {current_driver}. "
            f"Request will likely fail."
        )
        return current_driver

    # ===== Helper methods =====

    def _is_driver_available(self, driver_name: str) -> bool:
        """Check if a driver is registered and has a provider"""
        return driver_name in self._providers or driver_name in self._driver_configs

    def _is_circuit_open(self, driver_name: str) -> bool:
        """Check if a driver's circuit breaker is open"""
        state = self._circuit_breaker_states.get(driver_name)
        if state is None:
            return False
        # BreakerState has a 'name' field: "closed" | "open" | "half_open"
        if hasattr(state, "name"):
            return state.name == "open"
        # Dict-style access
        if isinstance(state, dict):
            return state.get("name", "closed") == "open"
        return False

    def _estimate_cost(self, driver_name: str, context_tokens: int) -> float:
        """Estimate cost for a request with the given driver

        Uses the routing table's pricing data. Returns 0.0 if pricing
        not configured for the driver.

        Args:
            driver_name: Full driver name
            context_tokens: Estimated prompt token count

        Returns:
            Estimated cost in CNY
        """
        pricing = self._routing_table.get_pricing(driver_name)
        if not pricing:
            return 0.0
        # Use cache_hit pricing if available (optimistic estimate)
        prompt_rate = pricing.get("cache_hit_per_million", pricing.get("prompt_per_million", 0.0))
        completion_rate = pricing.get("completion_per_million", 0.0)
        # Assume ~30% of context tokens as completion tokens
        est_completion = int(context_tokens * 0.3)
        return (context_tokens * prompt_rate + est_completion * completion_rate) / 1_000_000

    def _log_decision(self, decision: RoutingDecision) -> None:
        """Log a routing decision to the internal decision log"""
        with self._lock:
            self._decision_log.append(decision)

        # Also log to Python logger
        logger.info(
            f"Routing decision: intent={decision.intent} → "
            f"driver={decision.selected_driver} "
            f"compiler={decision.selected_compiler} "
            f"reason={decision.reason.value} "
            f"est_cost=¥{decision.estimated_cost:.4f} "
            f"ctx_tokens={decision.context_tokens}"
        )

        # Update monthly budget tracking
        # NOTE: 仅在此处累加估算成本。record_actual_cost() 会替换为实际成本。
        # 避免双重计数：如果后续会调用 record_actual_cost()，此处不累加。
        # 当前设计：_log_decision 仅用于日志记录，预算追踪统一由 record_actual_cost 管理。
        # self._monthly_budget_cny += decision.estimated_cost  # 已移除，避免双重计数

    # ===== Provider access =====

    def get_provider(self, driver_name: str) -> Any:
        """Get a ModelProvider by driver name

        Args:
            driver_name: Full driver name

        Returns:
            ModelProvider instance, or None if not found
        """
        return self._providers.get(driver_name)

    def register_provider(self, driver_name: str, provider: Any) -> None:
        """Register a ModelProvider for a driver

        Args:
            driver_name: Full driver name
            provider: ModelProvider instance
        """
        self._providers[driver_name] = provider
        logger.debug(f"Registered provider for driver: {driver_name}")

    def update_circuit_breaker_state(self, driver_name: str, state: Any) -> None:
        """Update circuit breaker state for a driver

        Args:
            driver_name: Full driver name
            state: BreakerState instance or dict with "name" key
        """
        self._circuit_breaker_states[driver_name] = state

    # ===== Decision log =====

    def get_decision_log(self, limit: int = 100) -> list[RoutingDecision]:
        """Get recent routing decisions

        Args:
            limit: Maximum number of decisions to return

        Returns:
            List of RoutingDecision entries, most recent first
        """
        with self._lock:
            return list(reversed(self._decision_log[-limit:]))

    def get_decision_summary(self) -> dict[str, Any]:
        """Get aggregated routing decision statistics

        Returns:
            Dict with decision counts by reason, driver, and intent
        """
        with self._lock:
            decisions = list(self._decision_log)

        by_reason: dict[str, int] = {}
        by_driver: dict[str, int] = {}
        by_intent: dict[str, int] = {}
        total_cost = 0.0

        for d in decisions:
            by_reason[d.reason.value] = by_reason.get(d.reason.value, 0) + 1
            by_driver[d.selected_driver] = by_driver.get(d.selected_driver, 0) + 1
            by_intent[d.intent] = by_intent.get(d.intent, 0) + 1
            total_cost += d.estimated_cost

        return {
            "total_decisions": len(decisions),
            "by_reason": by_reason,
            "by_driver": by_driver,
            "by_intent": by_intent,
            "total_estimated_cost_cny": round(total_cost, 4),
            "monthly_budget_cny": round(self._monthly_budget_cny, 4),
            "monthly_budget_limit_cny": self._monthly_budget_limit_cny,
            "budget_utilization": (
                round(self._monthly_budget_cny / self._monthly_budget_limit_cny, 4)
                if self._monthly_budget_limit_cny > 0
                else 0.0
            ),
        }

    # ===== Budget management =====

    def set_monthly_budget(self, limit_cny: float, warning_threshold: float = 0.8, auto_downgrade: bool = True) -> None:
        """Configure monthly budget limits

        Args:
            limit_cny: Monthly budget limit in CNY (0 = no limit)
            warning_threshold: Fraction at which to warn (0.8 = 80%)
            auto_downgrade: If True, auto-downgrade to V4-Flash when budget exhausted
        """
        self._monthly_budget_limit_cny = limit_cny
        self._budget_warning_threshold = warning_threshold
        self._budget_auto_downgrade = auto_downgrade
        logger.info(
            f"Monthly budget set: ¥{limit_cny:.2f} "
            f"(warning at {warning_threshold:.0%}, auto_downgrade={auto_downgrade})"
        )

    def record_actual_cost(self, driver_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Record actual cost from a completed request (not just estimated)

        This provides more accurate budget tracking than estimation alone.

        Args:
            driver_name: Full driver name
            prompt_tokens: Actual prompt tokens consumed
            completion_tokens: Actual completion tokens generated

        Returns:
            Calculated cost in CNY
        """
        pricing = self._routing_table.get_pricing(driver_name)
        if not pricing:
            return 0.0

        prompt_rate = pricing.get("prompt_per_million", 0.0)
        completion_rate = pricing.get("completion_per_million", 0.0)
        cost = (prompt_tokens * prompt_rate + completion_tokens * completion_rate) / 1_000_000

        self._monthly_budget_cny += cost
        return cost

    @property
    def monthly_budget_utilization(self) -> float:
        """Current monthly budget utilization (0.0 - 1.0+)"""
        if self._monthly_budget_limit_cny <= 0:
            return 0.0
        return self._monthly_budget_cny / self._monthly_budget_limit_cny

    @property
    def monthly_budget_remaining(self) -> float:
        """Remaining monthly budget in CNY"""
        if self._monthly_budget_limit_cny <= 0:
            return float("inf")
        return max(0.0, self._monthly_budget_limit_cny - self._monthly_budget_cny)

    @property
    def is_budget_exhausted(self) -> bool:
        """Whether the monthly budget is exhausted"""
        if self._monthly_budget_limit_cny <= 0:
            return False
        return self._monthly_budget_cny >= self._monthly_budget_limit_cny

    @property
    def is_budget_warning(self) -> bool:
        """Whether the monthly budget is in warning state"""
        if self._monthly_budget_limit_cny <= 0:
            return False
        return self._monthly_budget_utilization >= self._budget_warning_threshold

    # ===== Pipeline delegation =====

    @property
    def pipeline_manager(self) -> PipelineManager:
        """Access the PipelineManager for profile switching"""
        return self._pipeline_manager

    # ===== Configuration =====

    @property
    def routing_table(self) -> RoutingTable:
        """Access the current RoutingTable"""
        return self._routing_table

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        available_providers: dict[str, Any] | None = None,
        available_driver_configs: dict[str, Any] | None = None,
    ) -> ModelRouter:
        """Create ModelRouter from a TOML config dict

        Reads [routing] and [execution.pipeline] sections.

        Args:
            config: Full TOML config dict
            available_providers: Map of driver_name → ModelProvider
            available_driver_configs: Map of driver_name → DriverConfig

        Returns:
            Configured ModelRouter instance
        """
        # Load routing table from config
        routing_config = config.get("routing", {})
        routing_table = RoutingTable.from_dict(routing_config)

        # Load pipeline manager from config
        pipeline_manager = PipelineManager.from_config(config, routing_table)

        # Load budget config
        budget_config = routing_config.get("monthly_budget", {})
        budget_limit = float(budget_config.get("limit_cny", 0))
        budget_warning = float(budget_config.get("warning_threshold", 0.8))
        budget_auto_downgrade = budget_config.get("auto_downgrade", True)

        router = cls(
            available_providers=available_providers,
            available_driver_configs=available_driver_configs,
            routing_table=routing_table,
            pipeline_manager=pipeline_manager,
        )

        if budget_limit > 0:
            router.set_monthly_budget(budget_limit, budget_warning, budget_auto_downgrade)

        return router

    def __repr__(self) -> str:
        return (
            f"ModelRouter("
            f"providers={len(self._providers)}, "
            f"decisions={len(self._decision_log)}, "
            f"pipeline={self._pipeline_manager.active_profile_name}, "
            f"budget=¥{self._monthly_budget_cny:.2f}/{self._monthly_budget_limit_cny:.2f})"
        )
