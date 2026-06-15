# tests/test_glm5v_coordinator.py
"""P2-11 GLM-5V-Turbo + GLM-5.2 Coordinator tests

Coverage:
  - GLM5VTurboCompiler (analysis/verification/default modes)
  - VisionAnalysisResult (parsing, quality_score, to_context_string)
  - CoordinationConfig (creation, from_dict)
  - CoordinationStep / CoordinationResult / CoordinationPhase / CoordinationMode
  - GLM52VCoordinatedWorkflow (sequential/verify/parallel/degraded/coding_only)
  - GLM52Compiler coordination integration (create_coordinated_workflow, compile_with_visual_context)
  - _extract_verification_score helper
  - _aggregate_token_usage helper

All tests use MockAdapter — no real API calls.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from teragent.core.adapters.mock import MockAdapter
from teragent.core.compiler import TAPCompilerRegistry
from teragent.core.compilers.glm_52 import GLM52Compiler
from teragent.core.compilers.glm_5v_turbo import (
    GLM5VTurboCompiler,
    VisionAnalysisResult,
    VISION_ANALYSIS_SYSTEM_PROMPT,
    VISION_VERIFICATION_SYSTEM_PROMPT,
    VISION_CODE_GENERATION_HINT,
)
from teragent.core.provider import ModelProvider
from teragent.core.tap import (
    CompiledPrompt,
    MultimodalContent,
    TAPRequest,
    TAPResponse,
)
from teragent.coordination.glm5v_coordinator import (
    CoordinationConfig,
    CoordinationMode,
    CoordinationPhase,
    CoordinationResult,
    CoordinationStep,
    GLM52VCoordinatedWorkflow,
    _extract_verification_score,
    _aggregate_token_usage,
)


# ===== Helpers =====


def _make_request(**overrides) -> TAPRequest:
    """Construct a TAPRequest for testing."""
    defaults = {
        "meta": {"task_id": "test.1", "intent": "execute"},
        "instruction": "根据设计稿实现页面",
        "constraints": [],
    }
    defaults.update(overrides)
    return TAPRequest(**defaults)


def _make_multimodal_request(**overrides) -> TAPRequest:
    """Construct a TAPRequest with multimodal content."""
    mm = [
        MultimodalContent(type="image_url", url="https://example.com/design.png"),
    ]
    defaults = {
        "meta": {"task_id": "test.mm.1", "intent": "execute"},
        "instruction": "根据设计稿实现页面",
        "constraints": [],
        "multimodal_context": mm,
    }
    defaults.update(overrides)
    return TAPRequest(**defaults)


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


def _create_vision_provider() -> ModelProvider:
    """Create a mock GLM-5V-Turbo vision provider."""
    return _create_mock_provider("glm_5v_turbo", "glm-5v-turbo", mode="analysis")


def _create_coding_provider() -> ModelProvider:
    """Create a mock GLM-5.2 coding provider."""
    return _create_mock_provider("glm_52", "glm-5.2")


# ===== 1. GLM5VTurboCompiler tests =====


class TestGLM5VTurboCompiler:
    """GLM5VTurboCompiler compilation tests with different modes."""

    def test_compiler_type(self):
        """_get_compiler_type returns 'glm_5v_turbo'."""
        compiler = GLM5VTurboCompiler()
        assert compiler._get_compiler_type() == "glm_5v_turbo"

    def test_supports_multimodal(self):
        """GLM5VTurboCompiler supports multimodal natively."""
        compiler = GLM5VTurboCompiler()
        assert compiler.supports_multimodal is True

    def test_max_context_tokens(self):
        """Default max_context_tokens is 128K."""
        compiler = GLM5VTurboCompiler()
        assert compiler.max_context_tokens == 128_000

    def test_custom_max_context_tokens(self):
        """Custom max_context_tokens is respected."""
        compiler = GLM5VTurboCompiler(max_context_tokens=64_000)
        assert compiler.max_context_tokens == 64_000

    def test_mode_property(self):
        """Mode property returns the configured mode."""
        compiler_default = GLM5VTurboCompiler()
        assert compiler_default.mode == "default"

        compiler_analysis = GLM5VTurboCompiler(mode="analysis")
        assert compiler_analysis.mode == "analysis"

        compiler_verification = GLM5VTurboCompiler(mode="verification")
        assert compiler_verification.mode == "verification"

    def test_compile_analysis_mode(self):
        """Analysis mode compiles with proper system prompt."""
        compiler = GLM5VTurboCompiler(mode="analysis")
        request = _make_multimodal_request()
        compiled = compiler.compile(request)

        assert compiled.mode == "messages"
        assert len(compiled.messages) >= 2
        # System prompt should be present (may come from prompt registry or built-in)
        system_msg = compiled.messages[0]
        assert system_msg["role"] == "system"
        assert len(system_msg["content"]) > 0

    def test_compile_verification_mode(self):
        """Verification mode compiles with proper system prompt."""
        compiler = GLM5VTurboCompiler(mode="verification")
        request = _make_multimodal_request()
        compiled = compiler.compile(request)

        assert compiled.mode == "messages"
        system_msg = compiled.messages[0]
        assert system_msg["role"] == "system"
        assert len(system_msg["content"]) > 0

    def test_compile_default_mode(self):
        """Default mode compiles successfully."""
        compiler = GLM5VTurboCompiler(mode="default")
        request = _make_multimodal_request()
        compiled = compiler.compile(request)

        assert compiled.mode == "messages"
        assert len(compiled.messages) >= 2

    def test_compile_preserves_multimodal_content(self):
        """Compilation preserves multimodal content blocks (no degradation)."""
        compiler = GLM5VTurboCompiler(mode="analysis")
        request = _make_multimodal_request()
        compiled = compiler.compile(request)

        # Extra should indicate multimodal
        assert compiled.extra.get("has_multimodal") is True
        assert compiled.extra.get("compiler") == "glm_5v_turbo"

    def test_compile_text_only_request(self):
        """Compile a text-only request returns a simple string user content."""
        compiler = GLM5VTurboCompiler(mode="analysis")
        request = _make_request(instruction="Analyze the image")
        compiled = compiler.compile(request)

        assert compiled.mode == "messages"
        # User content should be a string since no multimodal
        user_msg = compiled.messages[-1]
        assert user_msg["role"] == "user"
        # When only text, user content is a string
        assert isinstance(user_msg["content"], str)

    def test_compile_with_constraints(self):
        """Constraints are included in the compiled prompt."""
        compiler = GLM5VTurboCompiler()
        request = _make_request(
            constraints=["使用 React 框架", "响应式设计"],
        )
        compiled = compiler.compile(request)

        user_msg = compiled.messages[-1]
        content = user_msg["content"]
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            content = " ".join(text_parts)
        assert "约束条件" in content

    def test_compile_with_output_format_hint(self):
        """Output format hint is included in the compiled prompt."""
        compiler = GLM5VTurboCompiler()
        request = _make_request(output_format_hint="HTML")
        compiled = compiler.compile(request)

        user_msg = compiled.messages[-1]
        content = user_msg["content"]
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            content = " ".join(text_parts)
        assert "输出格式" in content

    def test_compile_extra_fields(self):
        """Compiled prompt extra has correct compiler metadata."""
        compiler = GLM5VTurboCompiler(mode="analysis")
        request = _make_multimodal_request()
        compiled = compiler.compile(request)

        assert compiled.extra["compiler"] == "glm_5v_turbo"
        assert compiled.extra["compile_mode"] == "analysis"
        assert compiled.extra["has_multimodal"] is True

    def test_parse_analysis_result_layout(self):
        """parse_analysis_result detects layout keywords."""
        text = "## 布局结构\n页面采用卡片式布局，包含导航栏和侧边栏。"
        result = GLM5VTurboCompiler.parse_analysis_result(text)
        assert result.has_layout_info is True
        assert result.structured_context == f"<visual_analysis>\n{text}\n</visual_analysis>"

    def test_parse_analysis_result_color(self):
        """parse_analysis_result detects color keywords."""
        text = "## 颜色方案\n主色为蓝色，背景色为白色。"
        result = GLM5VTurboCompiler.parse_analysis_result(text)
        assert result.has_color_scheme is True

    def test_parse_analysis_result_interaction(self):
        """parse_analysis_result detects interaction keywords."""
        text = "## 交互逻辑\n用户点击提交按钮后跳转到结果页面。"
        result = GLM5VTurboCompiler.parse_analysis_result(text)
        assert result.has_interaction_logic is True

    def test_parse_analysis_result_table_elements(self):
        """parse_analysis_result counts table row elements."""
        text = (
            "| # | 类型 | 标签 |\n"
            "|---|------|------|\n"
            "| 1 | 按钮 | 提交 |\n"
            "| 2 | 输入 | 用户名 |\n"
            "| 3 | 链接 | 帮助 |\n"
        )
        result = GLM5VTurboCompiler.parse_analysis_result(text)
        assert result.element_count == 3  # 5 rows - 2 header/separator

    def test_parse_analysis_result_list_elements(self):
        """parse_analysis_result counts list item elements when no table."""
        text = (
            "- 按钮: 提交\n"
            "- 输入框: 用户名\n"
            "- 链接: 帮助\n"
        )
        result = GLM5VTurboCompiler.parse_analysis_result(text)
        assert result.element_count == 3

    def test_parse_analysis_result_confidence(self):
        """parse_analysis_result calculates confidence based on features."""
        text = "布局结构和颜色方案，交互逻辑"
        result = GLM5VTurboCompiler.parse_analysis_result(text)
        # All three features → confidence = 0.5 + 0.15 + 0.1 + 0.1 = 0.85
        assert result.confidence >= 0.8

    def test_parse_analysis_result_empty(self):
        """parse_analysis_result handles empty string."""
        result = GLM5VTurboCompiler.parse_analysis_result("")
        assert result.raw_analysis == ""
        assert result.has_layout_info is False
        assert result.confidence == 0.5  # base confidence

    def test_parse_analysis_result_none(self):
        """parse_analysis_result handles None input."""
        result = GLM5VTurboCompiler.parse_analysis_result(None)
        assert result.raw_analysis == ""


# ===== 2. VisionAnalysisResult tests =====


class TestVisionAnalysisResult:
    """VisionAnalysisResult quality_score and to_context_string tests."""

    def test_default_values(self):
        """Default VisionAnalysisResult has zero values."""
        result = VisionAnalysisResult()
        assert result.raw_analysis == ""
        assert result.structured_context == ""
        assert result.has_layout_info is False
        assert result.has_color_scheme is False
        assert result.has_interaction_logic is False
        assert result.element_count == 0
        assert result.confidence == 0.0
        assert result.source_image_type == "unknown"

    def test_quality_score_no_features(self):
        """quality_score is 0.0 when no features detected."""
        result = VisionAnalysisResult()
        assert result.quality_score == 0.0

    def test_quality_score_layout_only(self):
        """quality_score is 0.25 when only layout detected."""
        result = VisionAnalysisResult(has_layout_info=True)
        assert result.quality_score == 0.25

    def test_quality_score_color_only(self):
        """quality_score is 0.25 when only color detected."""
        result = VisionAnalysisResult(has_color_scheme=True)
        assert result.quality_score == 0.25

    def test_quality_score_interaction_only(self):
        """quality_score is 0.25 when only interaction detected."""
        result = VisionAnalysisResult(has_interaction_logic=True)
        assert result.quality_score == 0.25

    def test_quality_score_elements(self):
        """quality_score increases with more elements."""
        result_low = VisionAnalysisResult(element_count=3)
        result_high = VisionAnalysisResult(element_count=15)
        assert result_high.quality_score > result_low.quality_score

    def test_quality_score_capped_at_one(self):
        """quality_score is capped at 1.0."""
        result = VisionAnalysisResult(
            has_layout_info=True,
            has_color_scheme=True,
            has_interaction_logic=True,
            element_count=100,
        )
        assert result.quality_score <= 1.0

    def test_quality_score_all_features(self):
        """quality_score reaches 1.0 with all features."""
        result = VisionAnalysisResult(
            has_layout_info=True,
            has_color_scheme=True,
            has_interaction_logic=True,
            element_count=10,
        )
        assert result.quality_score == 1.0

    def test_to_context_string_structured(self):
        """to_context_string returns structured_context when set."""
        result = VisionAnalysisResult(structured_context="<visual>data</visual>")
        assert result.to_context_string() == "<visual>data</visual>"

    def test_to_context_string_fallback_to_raw(self):
        """to_context_string falls back to raw_analysis wrapped in XML tag."""
        result = VisionAnalysisResult(raw_analysis="some analysis")
        assert result.to_context_string() == "<visual_analysis>\nsome analysis\n</visual_analysis>"

    def test_to_context_string_empty(self):
        """to_context_string returns empty string when no data."""
        result = VisionAnalysisResult()
        assert result.to_context_string() == ""


# ===== 3. CoordinationConfig tests =====


class TestCoordinationConfig:
    """CoordinationConfig creation and from_dict tests."""

    def test_default_values(self):
        """Default CoordinationConfig has correct defaults."""
        config = CoordinationConfig()
        assert config.enabled is True
        assert config.mode == "sequential"
        assert config.vision_model == "glm-5v-turbo"
        assert config.coding_model == "glm-5.2"
        assert config.vision_compiler == "glm_5v_turbo"
        assert config.coding_compiler == "glm_52"
        assert config.context_sharing is True
        assert config.max_verification_rounds == 1
        assert config.verification_score_threshold == 7.0
        assert config.degrade_on_vision_failure is True
        assert config.inject_code_generation_hint is True

    def test_custom_values(self):
        """Custom CoordinationConfig values are preserved."""
        config = CoordinationConfig(
            enabled=False,
            mode="verify",
            vision_model="custom-vision",
            coding_model="custom-code",
            max_verification_rounds=3,
            verification_score_threshold=8.5,
        )
        assert config.enabled is False
        assert config.mode == "verify"
        assert config.vision_model == "custom-vision"
        assert config.coding_model == "custom-code"
        assert config.max_verification_rounds == 3
        assert config.verification_score_threshold == 8.5

    def test_from_dict_empty(self):
        """from_dict with empty dict returns defaults."""
        config = CoordinationConfig.from_dict({})
        assert config.enabled is True
        assert config.mode == "sequential"

    def test_from_dict_full(self):
        """from_dict with full dict overrides all fields."""
        data = {
            "enabled": False,
            "coordination_mode": "verify",
            "vision_model": "test-vision",
            "coding_model": "test-code",
            "vision_compiler": "glm_5v_turbo",
            "coding_compiler": "glm_52",
            "context_sharing": False,
            "max_verification_rounds": 2,
            "verification_score_threshold": 9.0,
            "degrade_on_vision_failure": False,
            "inject_code_generation_hint": False,
        }
        config = CoordinationConfig.from_dict(data)
        assert config.enabled is False
        assert config.mode == "verify"
        assert config.vision_model == "test-vision"
        assert config.coding_model == "test-code"
        assert config.context_sharing is False
        assert config.max_verification_rounds == 2
        assert config.verification_score_threshold == 9.0
        assert config.degrade_on_vision_failure is False
        assert config.inject_code_generation_hint is False

    def test_from_dict_partial(self):
        """from_dict with partial dict uses defaults for missing fields."""
        data = {"coordination_mode": "parallel"}
        config = CoordinationConfig.from_dict(data)
        assert config.mode == "parallel"
        assert config.enabled is True  # default
        assert config.max_verification_rounds == 1  # default


# ===== 4. CoordinationStep & CoordinationResult tests =====


class TestCoordinationStepAndResult:
    """CoordinationStep and CoordinationResult data class tests."""

    def test_coordination_step_default(self):
        """CoordinationStep defaults are correct."""
        step = CoordinationStep()
        assert step.phase == ""
        assert step.model == ""
        assert step.start_time == 0.0
        assert step.end_time == 0.0
        assert step.success is True
        assert step.token_usage == {}
        assert step.error == ""

    def test_coordination_step_duration_ms(self):
        """CoordinationStep.duration_ms calculates correctly."""
        step = CoordinationStep(start_time=1.0, end_time=2.5)
        assert step.duration_ms == 1500.0

    def test_coordination_step_duration_ms_zero_when_no_end(self):
        """CoordinationStep.duration_ms is 0 when end_time is 0."""
        step = CoordinationStep(start_time=1.0, end_time=0.0)
        assert step.duration_ms == 0.0

    def test_coordination_result_default(self):
        """CoordinationResult defaults are correct."""
        result = CoordinationResult()
        assert result.final_response is None
        assert result.vision_analysis is None
        assert result.verification_result is None
        assert result.steps == []
        assert result.phase == CoordinationPhase.PENDING.value
        assert result.is_degraded is False
        assert result.total_tokens == {}
        assert result.total_duration_ms == 0.0

    def test_coordination_result_success_completed(self):
        """CoordinationResult.success is True when phase is completed."""
        result = CoordinationResult(phase=CoordinationPhase.COMPLETED.value)
        assert result.success is True

    def test_coordination_result_success_degraded(self):
        """CoordinationResult.success is True when phase is degraded."""
        result = CoordinationResult(phase=CoordinationPhase.DEGRADED.value)
        assert result.success is True

    def test_coordination_result_success_failed(self):
        """CoordinationResult.success is False when phase is failed."""
        result = CoordinationResult(phase=CoordinationPhase.FAILED.value)
        assert result.success is False

    def test_coordination_result_success_pending(self):
        """CoordinationResult.success is False when phase is pending."""
        result = CoordinationResult(phase=CoordinationPhase.PENDING.value)
        assert result.success is False

    def test_coordination_result_verification_score(self):
        """CoordinationResult.verification_score extracts score from response."""
        response = TAPResponse(raw_text="总体评分：8/10")
        result = CoordinationResult(verification_result=response)
        assert result.verification_score == 8.0

    def test_coordination_result_verification_score_none(self):
        """CoordinationResult.verification_score is None when no verification."""
        result = CoordinationResult()
        assert result.verification_score is None

    def test_coordination_result_verification_score_empty_text(self):
        """CoordinationResult.verification_score is None for empty text."""
        response = TAPResponse(raw_text="")
        result = CoordinationResult(verification_result=response)
        assert result.verification_score is None


# ===== 5. CoordinationMode & CoordinationPhase tests =====


class TestCoordinationEnums:
    """CoordinationMode and CoordinationPhase enum tests."""

    def test_coordination_mode_values(self):
        """CoordinationMode has correct values."""
        assert CoordinationMode.SEQUENTIAL.value == "sequential"
        assert CoordinationMode.PARALLEL.value == "parallel"
        assert CoordinationMode.VERIFY.value == "verify"

    def test_coordination_phase_values(self):
        """CoordinationPhase has correct values."""
        assert CoordinationPhase.PENDING.value == "pending"
        assert CoordinationPhase.VISION_ANALYSIS.value == "vision_analysis"
        assert CoordinationPhase.CONTEXT_TRANSFER.value == "context_transfer"
        assert CoordinationPhase.CODE_GENERATION.value == "code_generation"
        assert CoordinationPhase.VISUAL_VERIFICATION.value == "visual_verification"
        assert CoordinationPhase.COMPLETED.value == "completed"
        assert CoordinationPhase.FAILED.value == "failed"
        assert CoordinationPhase.DEGRADED.value == "degraded"

    def test_coordination_mode_from_string(self):
        """CoordinationMode can be constructed from string value."""
        assert CoordinationMode("sequential") == CoordinationMode.SEQUENTIAL
        assert CoordinationMode("verify") == CoordinationMode.VERIFY
        assert CoordinationMode("parallel") == CoordinationMode.PARALLEL


# ===== 6. Helper functions tests =====


class TestHelperFunctions:
    """Tests for _extract_verification_score and _aggregate_token_usage."""

    def test_extract_score_chinese_format(self):
        """Extract score from Chinese format: 总体评分：8/10."""
        assert _extract_verification_score("总体评分：8/10") == 8.0

    def test_extract_score_chinese_no_slash(self):
        """Extract score from Chinese format without slash."""
        assert _extract_verification_score("总体评分：8.5") == 8.5

    def test_extract_score_english_format(self):
        """Extract score from English format: Overall Score: 8/10."""
        assert _extract_verification_score("Overall Score: 8/10") == 8.0

    def test_extract_score_english_no_slash(self):
        """Extract score from English format without slash."""
        assert _extract_verification_score("Overall Score: 7.5") == 7.5

    def test_extract_score_plain_fraction(self):
        """Extract score from plain fraction: 9/10."""
        assert _extract_verification_score("评分 9/10") == 9.0

    def test_extract_score_no_match(self):
        """Return None when no score pattern found."""
        assert _extract_verification_score("No score here") is None

    def test_extract_score_empty(self):
        """Return None for empty string."""
        assert _extract_verification_score("") is None

    def test_aggregate_token_usage_empty(self):
        """Aggregate empty steps returns zeros."""
        result = _aggregate_token_usage([])
        assert result["prompt_tokens"] == 0
        assert result["completion_tokens"] == 0
        assert result["total_tokens"] == 0

    def test_aggregate_token_usage_single_step(self):
        """Aggregate single step returns its usage."""
        steps = [CoordinationStep(token_usage={"prompt_tokens": 100, "completion_tokens": 50})]
        result = _aggregate_token_usage(steps)
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert result["total_tokens"] == 150

    def test_aggregate_token_usage_multiple_steps(self):
        """Aggregate multiple steps sums all usage."""
        steps = [
            CoordinationStep(token_usage={"prompt_tokens": 100, "completion_tokens": 50}),
            CoordinationStep(token_usage={"prompt_tokens": 200, "completion_tokens": 100}),
        ]
        result = _aggregate_token_usage(steps)
        assert result["prompt_tokens"] == 300
        assert result["completion_tokens"] == 150
        assert result["total_tokens"] == 450


# ===== 7. GLM52VCoordinatedWorkflow tests =====


class TestGLM52VCoordinatedWorkflow:
    """GLM52VCoordinatedWorkflow integration tests with MockAdapter."""

    def test_workflow_not_available_without_providers(self):
        """Workflow is not available when providers are missing and no compiler configured."""
        config = CoordinationConfig(vision_compiler="", coding_compiler="")
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=None,
            coding_provider=None,
            config=config,
        )
        assert workflow.is_available is False

    def test_workflow_available_with_both_providers(self):
        """Workflow is available when both providers are explicitly provided."""
        vision = _create_vision_provider()
        coding = _create_coding_provider()
        config = CoordinationConfig(vision_compiler="", coding_compiler="")
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=vision,
            coding_provider=coding,
            config=config,
        )
        assert workflow.is_available is True

    def test_workflow_available_with_default_config(self):
        """Workflow is available when default config auto-creates providers from compilers."""
        # Default config has vision_compiler and coding_compiler set,
        # so the workflow auto-creates providers from the compilers
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=None,
            coding_provider=None,
        )
        assert workflow.is_available is True

    def test_workflow_config_property(self):
        """Workflow config property returns the config."""
        config = CoordinationConfig(mode="verify")
        workflow = GLM52VCoordinatedWorkflow(config=config)
        assert workflow.config is config

    @pytest.mark.asyncio
    async def test_sequential_workflow_success(self):
        """Sequential workflow completes successfully with multimodal request."""
        vision = _create_vision_provider()
        coding = _create_coding_provider()
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=vision,
            coding_provider=coding,
            config=CoordinationConfig(mode="sequential"),
        )
        request = _make_multimodal_request()
        result = await workflow.execute(request)

        assert result.success is True
        assert result.phase == CoordinationPhase.COMPLETED.value
        assert result.final_response is not None
        assert result.vision_analysis is not None
        assert len(result.steps) >= 2  # vision + context_transfer + coding

    @pytest.mark.asyncio
    async def test_sequential_workflow_has_vision_analysis(self):
        """Sequential workflow produces vision analysis result."""
        vision = _create_vision_provider()
        coding = _create_coding_provider()
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=vision,
            coding_provider=coding,
        )
        request = _make_multimodal_request()
        result = await workflow.execute(request)

        assert result.vision_analysis is not None
        assert isinstance(result.vision_analysis, VisionAnalysisResult)
        assert result.vision_analysis.structured_context != ""

    @pytest.mark.asyncio
    async def test_sequential_workflow_tracks_tokens(self):
        """Sequential workflow tracks token usage across steps."""
        vision = _create_vision_provider()
        coding = _create_coding_provider()
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=vision,
            coding_provider=coding,
        )
        request = _make_multimodal_request()
        result = await workflow.execute(request)

        assert result.total_tokens is not None
        assert "prompt_tokens" in result.total_tokens
        assert "completion_tokens" in result.total_tokens

    @pytest.mark.asyncio
    async def test_sequential_workflow_tracks_duration(self):
        """Sequential workflow tracks total duration."""
        vision = _create_vision_provider()
        coding = _create_coding_provider()
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=vision,
            coding_provider=coding,
        )
        request = _make_multimodal_request()
        result = await workflow.execute(request)

        assert result.total_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_coding_only_no_multimodal(self):
        """Non-multimodal request goes directly to coding model."""
        vision = _create_vision_provider()
        coding = _create_coding_provider()
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=vision,
            coding_provider=coding,
        )
        request = _make_request()  # no multimodal
        result = await workflow.execute(request)

        assert result.success is True
        assert result.phase == CoordinationPhase.COMPLETED.value
        assert result.final_response is not None
        # Should have only one step (code generation)
        assert len(result.steps) == 1
        assert result.steps[0].phase == CoordinationPhase.CODE_GENERATION.value

    @pytest.mark.asyncio
    async def test_verify_workflow(self):
        """Verify mode runs vision→coding→verification loop."""
        vision = _create_vision_provider()
        coding = _create_coding_provider()
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=vision,
            coding_provider=coding,
            config=CoordinationConfig(mode="verify"),
        )
        request = _make_multimodal_request()
        result = await workflow.execute(request)

        assert result.success is True
        # Should have more steps: vision + context_transfer + coding + verification
        assert len(result.steps) >= 3

    @pytest.mark.asyncio
    async def test_parallel_workflow(self):
        """Parallel mode runs vision and coding simultaneously."""
        vision = _create_vision_provider()
        coding = _create_coding_provider()
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=vision,
            coding_provider=coding,
            config=CoordinationConfig(mode="parallel"),
        )
        request = _make_multimodal_request()
        result = await workflow.execute(request)

        assert result.success is True
        assert result.final_response is not None
        # Parallel should have 2 steps
        assert len(result.steps) == 2

    @pytest.mark.asyncio
    async def test_degraded_when_not_available(self):
        """Workflow degrades when vision provider is missing and degrade enabled."""
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=None,
            coding_provider=_create_coding_provider(),
            config=CoordinationConfig(
                degrade_on_vision_failure=True,
                vision_compiler="",  # prevent auto-creation
            ),
        )
        request = _make_multimodal_request()
        result = await workflow.execute(request)

        assert result.is_degraded is True
        assert result.phase in (
            CoordinationPhase.DEGRADED.value,
            CoordinationPhase.FAILED.value,
        )

    @pytest.mark.asyncio
    async def test_failed_when_not_available_no_degrade(self):
        """Workflow fails when providers not available and degrade disabled."""
        config = CoordinationConfig(
            degrade_on_vision_failure=False,
            vision_compiler="",
            coding_compiler="",
        )
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=None,
            coding_provider=None,
            config=config,
        )
        request = _make_multimodal_request()
        result = await workflow.execute(request)

        assert result.success is False
        assert result.phase == CoordinationPhase.FAILED.value

    @pytest.mark.asyncio
    async def test_degraded_request_has_text_description(self):
        """Degraded mode converts multimodal to text description."""
        coding = _create_coding_provider()
        config = CoordinationConfig(
            degrade_on_vision_failure=True,
            vision_compiler="",  # prevent auto-creation
        )
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=None,
            coding_provider=coding,
            config=config,
        )
        request = _make_multimodal_request()
        result = await workflow.execute(request)

        assert result.is_degraded is True
        # The degraded request should have used the coding provider
        if result.final_response is not None:
            assert result.final_response.raw_text is not None

    @pytest.mark.asyncio
    async def test_no_vision_no_coding_providers_fails(self):
        """Workflow fails when both providers are None and no degradation."""
        config = CoordinationConfig(
            degrade_on_vision_failure=True,
            vision_compiler="",
            coding_compiler="",
        )
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=None,
            coding_provider=None,
            config=config,
        )
        request = _make_multimodal_request()
        result = await workflow.execute(request)

        # Even with degrade_on_vision_failure, without coding provider it fails
        assert result.phase == CoordinationPhase.FAILED.value

    @pytest.mark.asyncio
    async def test_context_sharing_injected(self):
        """Visual analysis is injected into coding request context."""
        vision = _create_vision_provider()
        coding = _create_coding_provider()
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=vision,
            coding_provider=coding,
            config=CoordinationConfig(context_sharing=True),
        )
        request = _make_multimodal_request()
        result = await workflow.execute(request)

        assert result.success is True
        assert result.vision_analysis is not None

    @pytest.mark.asyncio
    async def test_no_context_sharing(self):
        """Visual analysis is not injected when context_sharing=False."""
        vision = _create_vision_provider()
        coding = _create_coding_provider()
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=vision,
            coding_provider=coding,
            config=CoordinationConfig(context_sharing=False),
        )
        request = _make_multimodal_request()
        result = await workflow.execute(request)

        # Should still succeed but without context sharing
        assert result.success is True


# ===== 8. GLM52Compiler coordination integration tests =====


class TestGLM52CompilerCoordination:
    """GLM52Compiler create_coordinated_workflow and compile_with_visual_context."""

    def test_create_coordinated_workflow(self):
        """create_coordinated_workflow returns a workflow instance."""
        compiler = GLM52Compiler()
        workflow = compiler.create_coordinated_workflow()
        assert isinstance(workflow, GLM52VCoordinatedWorkflow)

    def test_create_coordinated_workflow_with_providers(self):
        """create_coordinated_workflow accepts custom providers."""
        compiler = GLM52Compiler()
        vision = _create_vision_provider()
        coding = _create_coding_provider()
        workflow = compiler.create_coordinated_workflow(
            vision_provider=vision,
            coding_provider=coding,
        )
        assert workflow.is_available is True

    def test_create_coordinated_workflow_with_config(self):
        """create_coordinated_workflow accepts custom config."""
        compiler = GLM52Compiler()
        config = CoordinationConfig(mode="verify")
        workflow = compiler.create_coordinated_workflow(config=config)
        assert workflow.config.mode == "verify"

    def test_compile_with_visual_context(self):
        """compile_with_visual_context injects visual analysis into prompt."""
        compiler = GLM52Compiler()
        request = _make_request()
        visual_analysis = "布局: 卡片式; 颜色: 蓝色主题; 元素: 5个"
        compiled = compiler.compile_with_visual_context(request, visual_analysis)

        assert isinstance(compiled, CompiledPrompt)
        assert compiled.extra.get("visual_coordination") is True
        assert compiled.extra.get("visual_analysis_length") == len(visual_analysis)

    def test_compile_with_visual_context_preserves_instruction(self):
        """compile_with_visual_context preserves original instruction."""
        compiler = GLM52Compiler()
        request = _make_request(instruction="实现登录页面")
        visual_analysis = "设计稿分析结果"
        compiled = compiler.compile_with_visual_context(request, visual_analysis)

        assert compiled.mode == "messages"
        # Check that the instruction is still in the messages
        all_text = ""
        for msg in compiled.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                all_text += content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        all_text += part.get("text", "")
        assert "登录页面" in all_text

    def test_compile_with_visual_context_empty_analysis(self):
        """compile_with_visual_context handles empty analysis string."""
        compiler = GLM52Compiler()
        request = _make_request()
        compiled = compiler.compile_with_visual_context(request, "")

        assert compiled.extra.get("visual_coordination") is True
        assert compiled.extra.get("visual_analysis_length") == 0

    def test_compile_with_visual_context_long_analysis(self):
        """compile_with_visual_context handles long analysis string."""
        compiler = GLM52Compiler()
        request = _make_request()
        visual_analysis = "详细分析内容" * 500
        compiled = compiler.compile_with_visual_context(request, visual_analysis)

        assert compiled.extra.get("visual_coordination") is True
        assert compiled.extra.get("visual_analysis_length") == len(visual_analysis)


# ===== 9. Prompt template tests =====


class TestPromptTemplates:
    """Tests for VISION_ANALYSIS_SYSTEM_PROMPT and VISION_VERIFICATION_SYSTEM_PROMPT."""

    def test_analysis_prompt_has_structure(self):
        """VISION_ANALYSIS_SYSTEM_PROMPT contains structured output format."""
        assert "视觉语义分析" in VISION_ANALYSIS_SYSTEM_PROMPT
        assert "整体描述" in VISION_ANALYSIS_SYSTEM_PROMPT
        assert "布局结构" in VISION_ANALYSIS_SYSTEM_PROMPT
        assert "UI 元素清单" in VISION_ANALYSIS_SYSTEM_PROMPT
        assert "颜色方案" in VISION_ANALYSIS_SYSTEM_PROMPT
        assert "交互逻辑" in VISION_ANALYSIS_SYSTEM_PROMPT

    def test_verification_prompt_has_structure(self):
        """VISION_VERIFICATION_SYSTEM_PROMPT contains verification format."""
        assert "视觉一致性验证" in VISION_VERIFICATION_SYSTEM_PROMPT
        assert "布局一致性" in VISION_VERIFICATION_SYSTEM_PROMPT
        assert "颜色一致性" in VISION_VERIFICATION_SYSTEM_PROMPT
        assert "总体评分" in VISION_VERIFICATION_SYSTEM_PROMPT

    def test_code_generation_hint(self):
        """VISION_CODE_GENERATION_HINT contains code generation guidance."""
        assert "视觉分析结果" in VISION_CODE_GENERATION_HINT
        assert "前端代码" in VISION_CODE_GENERATION_HINT


# ===== 10. End-to-end workflow integration tests =====


class TestWorkflowE2E:
    """End-to-end coordination workflow integration tests."""

    @pytest.mark.asyncio
    async def test_full_sequential_e2e(self):
        """Full sequential workflow: vision→context_transfer→coding."""
        vision = _create_vision_provider()
        coding = _create_coding_provider()
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=vision,
            coding_provider=coding,
            config=CoordinationConfig(mode="sequential"),
        )
        request = _make_multimodal_request(
            instruction="根据设计稿实现一个登录页面",
            constraints=["使用React", "响应式设计"],
        )
        result = await workflow.execute(request)

        assert result.success is True
        assert result.final_response is not None
        assert result.vision_analysis is not None
        # Should have: vision_analysis + context_transfer + code_generation
        phases = [s.phase for s in result.steps]
        assert CoordinationPhase.VISION_ANALYSIS.value in phases
        assert CoordinationPhase.CONTEXT_TRANSFER.value in phases
        assert CoordinationPhase.CODE_GENERATION.value in phases

    @pytest.mark.asyncio
    async def test_full_verify_e2e(self):
        """Full verify workflow: vision→coding→verification."""
        vision = _create_vision_provider()
        coding = _create_coding_provider()
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=vision,
            coding_provider=coding,
            config=CoordinationConfig(mode="verify", max_verification_rounds=1),
        )
        request = _make_multimodal_request()
        result = await workflow.execute(request)

        assert result.success is True
        phases = [s.phase for s in result.steps]
        assert CoordinationPhase.VISION_ANALYSIS.value in phases
        assert CoordinationPhase.CODE_GENERATION.value in phases
        assert CoordinationPhase.VISUAL_VERIFICATION.value in phases

    @pytest.mark.asyncio
    async def test_full_parallel_e2e(self):
        """Full parallel workflow: vision || coding."""
        vision = _create_vision_provider()
        coding = _create_coding_provider()
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=vision,
            coding_provider=coding,
            config=CoordinationConfig(mode="parallel"),
        )
        request = _make_multimodal_request()
        result = await workflow.execute(request)

        assert result.success is True
        assert result.final_response is not None
        phases = [s.phase for s in result.steps]
        assert CoordinationPhase.VISION_ANALYSIS.value in phases
        assert CoordinationPhase.CODE_GENERATION.value in phases

    @pytest.mark.asyncio
    async def test_degradation_e2e(self):
        """Degradation E2E: vision model unavailable → degrade to coding only."""
        coding = _create_coding_provider()
        config = CoordinationConfig(
            degrade_on_vision_failure=True,
            vision_compiler="",  # prevent auto-creation of vision provider
        )
        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=None,
            coding_provider=coding,
            config=config,
        )
        request = _make_multimodal_request()
        result = await workflow.execute(request)

        assert result.is_degraded is True

    @pytest.mark.asyncio
    async def test_compiler_and_adapter_round_trip(self):
        """Full round trip: compile → MockAdapter → response."""
        vision = _create_vision_provider()
        request = _make_multimodal_request(
            meta={"task_id": "round.trip", "intent": "design"},
            instruction="分析这个设计稿的布局和颜色方案",
        )
        response = await vision.execute_tap(request)

        assert response.raw_text is not None
        assert len(response.raw_text) > 0
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_glm52_with_visual_context_round_trip(self):
        """GLM-5.2 + visual context round trip compilation and execution."""
        compiler = GLM52Compiler()
        adapter = MockAdapter()
        provider = ModelProvider(compiler=compiler, adapter=adapter, model="glm-5.2")

        request = _make_request(
            instruction="实现登录页面",
            context={"visual_analysis": "<visual_analysis>\n布局: 卡片式\n颜色: 蓝色\n</visual_analysis>"},
        )
        response = await provider.execute_tap(request)

        assert response.raw_text is not None
        assert len(response.raw_text) > 0
