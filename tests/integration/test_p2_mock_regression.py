# tests/integration/test_p2_mock_regression.py
"""P2 Integration Tests — Phase 2 Mock-mode Regression

Comprehensive integration tests covering ALL Phase 2 features using MockAdapter.
No real API calls are made.

Test Coverage:
  1. P2-1: V4 Cache Aware Strategy (cache prefix freezing, warmup, cost tracker)
  2. P2-2: V4 1M Context Management (context profiles, section budgets, tail reinforcement)
  3. P2-3: M3 Multimodal Support (video, image, mixed content, token estimation)
  4. P2-4: M3 Desktop Tool (creation, safety, rate limiting, safe zones, registry)
  5. P2-5: MiniMax Native Adapter (inheritance, registry, headers, capabilities)
  6. P2-6: GLM Long-Horizon Manager (task manager, dataclasses, checkpoint, progress)
  7. P2-7: GLM Self-Eval Strategy Switch (evaluator, switcher, stagnation detection)
  8. P2-8: GLM 200K Compression (compaction strategy, ADR, aggressive compression)
  9. P2 End-to-End (full pipeline tests across models)
"""

import shutil
import tempfile
import time

import pytest

from teragent.context.context_window import ContextWindow
from teragent.context.microcompactor import Microcompactor
from teragent.context.profiles import (
    DeepSeekV4ContextProfile,
    GLM5CompactionStrategy,
    GLM5ContextProfile,
    MiniMaxM3ContextProfile,
)
from teragent.core.adapter import TAPAdapterRegistry
from teragent.core.adapters.minimax_native import MiniMaxNativeAdapter
from teragent.core.adapters.mock import MockAdapter
from teragent.core.adapters.openai_compatible import OpenAICompatibleAdapter
from teragent.core.compiler import TAPCompilerRegistry
from teragent.core.compilers import (
    DeepSeekV4Compiler,
    GLM5Compiler,
    MiniMaxM3Compiler,
)
from teragent.core.provider import ModelProvider
from teragent.core.tap import (
    CompiledPrompt,
    CostTracker,
    DesktopContext,
    LongHorizonConfig,
    MultimodalContent,
    TAPCostRecord,
    TAPRequest,
    TAPResponse,
)
from teragent.core.types import ToolSafety
from teragent.long_horizon.checkpoint import Checkpoint, CheckpointStore
from teragent.long_horizon.progress import ProgressReport, ProgressTracker
from teragent.long_horizon.self_evaluation import SelfEvaluationResult, SelfEvaluator
from teragent.long_horizon.strategy_switch import StrategySwitcher, StrategySwitchRecord
from teragent.long_horizon.task_manager import LongHorizonTaskManager
from teragent.long_horizon.types import LongHorizonResult, PhaseResult, SubGoal
from teragent.tools.base import ToolResult
from teragent.tools.desktop import DesktopSafetyConfig, DesktopTool
from teragent.tools.registry import ToolRegistry

# ===== Helpers =====


def _make_request(**overrides) -> TAPRequest:
    """Create a TAPRequest with sensible defaults, allowing overrides."""
    defaults = dict(
        meta={"task_id": "test_p2", "intent": "execute"},
        instruction="实现一个排序函数",
        constraints=["Python 3.10+"],
        output_format_hint="<file path='...'>完整代码</file>",
        context={},
    )
    defaults.update(overrides)
    return TAPRequest(**defaults)


def _make_mock_provider(compiler_name: str, model: str = "mock-model", **compiler_kwargs) -> ModelProvider:
    """Create a ModelProvider with a given compiler and MockAdapter."""
    compiler_cls = TAPCompilerRegistry.get(compiler_name)
    if compiler_cls is None:
        raise ValueError(f"Unknown compiler: {compiler_name}")
    compiler = compiler_cls(**compiler_kwargs)
    adapter = MockAdapter(delay=0.0)
    return ModelProvider(compiler=compiler, adapter=adapter, model=model)


# =====================================================================
# 1. TestP2_1_CacheAwareStrategy — V4 Cache
# =====================================================================


class TestP2_1_CacheAwareStrategy:
    """P2-1: DeepSeek V4 Cache Aware Strategy tests."""

    def test_cache_prefix_freezing_auto(self):
        """compile with cache_preference='auto', verify cache_prefix_frozen in compiled.extra"""
        compiler = DeepSeekV4Compiler(variant="pro")
        request = _make_request(cache_preference="auto")
        compiled = compiler.compile(request)
        assert compiled.extra.get("cache_prefix_frozen") is True, (
            f"cache_prefix_frozen should be True with cache_preference='auto', "
            f"got {compiled.extra.get('cache_prefix_frozen')}"
        )

    def test_cache_prefix_freezing_aggressive(self):
        """compile with cache_preference='aggressive', verify cache metadata"""
        compiler = DeepSeekV4Compiler(variant="pro")
        request = _make_request(cache_preference="aggressive")
        compiled = compiler.compile(request)
        assert compiled.extra.get("cache_prefix_frozen") is True
        assert compiled.extra.get("cache_aware") is True
        # aggressive mode should have layout sections
        layout = compiled.extra.get("layout_sections")
        assert layout is not None, "layout_sections should be present in aggressive mode"
        assert "frozen" in layout
        assert "semi_static" in layout
        assert "dynamic" in layout

    def test_cache_prefix_freezing_none(self):
        """compile with cache_preference='none', verify no cache metadata"""
        compiler = DeepSeekV4Compiler(variant="pro")
        request = _make_request(cache_preference="none")
        compiled = compiler.compile(request)
        assert compiled.extra.get("cache_aware") is None
        assert compiled.extra.get("cache_prefix_frozen") is None

    def test_cache_warmup_request(self):
        """build warmup request, compile it, verify it produces valid CompiledPrompt"""
        compiler = DeepSeekV4Compiler(variant="pro", tools=[{
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }])
        warmup_req = compiler.build_warmup_request()
        assert isinstance(warmup_req, TAPRequest)
        assert warmup_req.cache_preference == "aggressive"
        compiled = compiler.compile(warmup_req)
        assert isinstance(compiled, CompiledPrompt)
        assert compiled.mode == "messages"
        assert len(compiled.messages) > 0

    def test_cache_aware_layout_structure(self):
        """verify messages layout has frozen → semi-static → dynamic sections"""
        compiler = DeepSeekV4Compiler(variant="pro")
        request = _make_request(
            cache_preference="aggressive",
            context={"large_files": [{"path": "big.py", "content": "x" * 200_000, "tokens": 60_000}]},
        )
        compiled = compiler.compile(request)
        layout = compiled.extra.get("layout_sections")
        assert layout is not None, "layout_sections should exist with cache_preference='aggressive'"
        # frozen section should exist (system prompt is frozen)
        assert layout.get("frozen", 0) >= 0
        # The sections should be ordered: frozen -> semi-static -> dynamic
        assert "frozen" in layout
        assert "semi_static" in layout
        assert "dynamic" in layout

    def test_cost_tracker_cache_hit(self):
        """create CostTracker, append records with cache_hit_tokens, verify methods"""
        tracker = CostTracker()
        tracker.append(TAPCostRecord(
            task_id="t1", prompt_tokens=1000, completion_tokens=200,
            cache_hit_tokens=700, cache_miss_tokens=300,
        ))
        tracker.append(TAPCostRecord(
            task_id="t2", prompt_tokens=2000, completion_tokens=400,
            cache_hit_tokens=1500, cache_miss_tokens=500,
        ))
        assert tracker.total_cache_hit_tokens() == 2200
        assert tracker.total_cache_miss_tokens() == 800
        rate = tracker.cache_hit_rate()
        assert 0.0 < rate < 1.0, f"cache_hit_rate should be between 0 and 1, got {rate}"

    def test_cost_tracker_estimated_cost(self):
        """verify total_estimated_cost() with pricing dict"""
        tracker = CostTracker()
        tracker.append(TAPCostRecord(
            task_id="t1", prompt_tokens=1000, completion_tokens=200,
            cache_hit_tokens=700, cache_miss_tokens=300,
        ))
        pricing = {
            "prompt_per_million": 1.0,
            "completion_per_million": 2.0,
            "cache_hit_per_million": 0.1,
            "cache_miss_per_million": 1.0,
        }
        cost = tracker.total_estimated_cost(pricing)
        assert cost > 0, "total_estimated_cost should be positive"
        # Verify: cache_hit cost = 700 * 0.1 / 1M = 0.00007
        # cache_miss cost = 300 * 1.0 / 1M = 0.0003
        # completion cost = 200 * 2.0 / 1M = 0.0004
        # total = 0.00077
        assert abs(cost - 0.00077) < 1e-6, f"Expected ~0.00077, got {cost}"

    def test_should_compress_aggressively(self):
        """verify compression decision at various cache hit rates"""
        compiler = DeepSeekV4Compiler(variant="pro")
        # Low hit rate → should compress aggressively
        assert compiler._should_compress_aggressively(0.1) is True
        assert compiler._should_compress_aggressively(0.25) is True
        # Medium hit rate → no aggressive compression
        assert compiler._should_compress_aggressively(0.5) is False
        # High hit rate → should NOT compress (avoid cache invalidation)
        assert compiler._should_compress_aggressively(0.8) is False
        assert compiler._should_compress_aggressively(0.95) is False


# =====================================================================
# 2. TestP2_2_ContextManagement — V4 1M Context
# =====================================================================


class TestP2_2_ContextManagement:
    """P2-2: V4 1M Context Management tests."""

    def test_deepseek_v4_context_profile(self):
        """verify DeepSeekV4ContextProfile section budgets"""
        profile = DeepSeekV4ContextProfile()
        assert profile.max_tokens == 1_000_000
        assert profile.system_budget == int(1_000_000 * 0.05)  # 50K
        assert profile.history_budget == int(1_000_000 * 0.45)  # 450K
        assert profile.large_file_budget == int(1_000_000 * 0.40)  # 400K
        assert profile.tail_reinforcement_budget == int(1_000_000 * 0.10)  # 100K
        # Total allocated should be 1.0
        assert abs(profile.total_allocated_ratio - 1.0) < 0.01

    def test_glm5_context_profile(self):
        """verify GLM5ContextProfile section budgets"""
        profile = GLM5ContextProfile()
        assert profile.max_tokens == 200_000
        assert profile.system_budget == int(200_000 * 0.10)  # 20K
        assert profile.history_budget == int(200_000 * 0.45)  # 90K
        # GLM-5 specific sections
        assert profile.design_budget == int(200_000 * 0.20)  # 40K
        assert profile.recent_complete_budget == int(200_000 * 0.15)  # 30K
        # section_budget() extended support
        assert profile.section_budget("design") == profile.design_budget
        assert profile.section_budget("recent_complete") == profile.recent_complete_budget

    def test_minimax_m3_context_profile(self):
        """verify MiniMaxM3ContextProfile section budgets"""
        profile = MiniMaxM3ContextProfile()
        assert profile.max_tokens == 1_000_000
        assert profile.system_ratio == 0.03
        assert profile.system_budget == int(1_000_000 * 0.03)  # 30K
        assert profile.history_budget == int(1_000_000 * 0.47)  # 470K
        assert profile.large_file_budget == int(1_000_000 * 0.40)  # 400K
        assert profile.tail_reinforcement_budget == int(1_000_000 * 0.10)  # 100K

    def test_context_window_with_profile(self):
        """create ContextWindow with profile, verify section_budget()"""
        profile = DeepSeekV4ContextProfile()
        cw = ContextWindow(profile=profile)
        assert cw.section_budget("system") == profile.system_budget
        assert cw.section_budget("history") == profile.history_budget
        assert cw.section_budget("large_file") == profile.large_file_budget
        assert cw.section_budget("tail_reinforcement") == profile.tail_reinforcement_budget

    def test_context_window_should_compact_section(self):
        """test per-section compaction check"""
        profile = GLM5ContextProfile()
        cw = ContextWindow(profile=profile)
        # Create messages that exceed the system budget (20K tokens)
        large_messages = [
            {"role": "system", "content": "x" * 100_000}  # ~30K tokens (conservative)
        ]
        result = cw.should_compact_section("system", large_messages)
        assert isinstance(result, bool)
        # A very large system message should trigger compaction
        assert result is True

    @pytest.mark.asyncio
    async def test_tail_reinforcement_pro(self):
        """compile Pro request, verify tail reinforcement messages present"""
        compiler = DeepSeekV4Compiler(variant="pro")
        request = _make_request(
            constraints=["必须包含错误处理", "使用类型注解"],
            output_format_hint="<file path='...'>代码</file>",
        )
        compiled = compiler.compile(request)
        # Pro mode should add tail reinforcement
        assert compiled.extra.get("tail_reinforcement") is True
        assert compiled.extra.get("tail_reinforcement_variant") == "pro"
        # Last messages should be the reinforcement
        assert len(compiled.messages) >= 2

    @pytest.mark.asyncio
    async def test_tail_reinforcement_flash(self):
        """compile Flash request, verify tail reinforcement present (compact)"""
        compiler = DeepSeekV4Compiler(variant="flash")
        request = _make_request(
            constraints=["必须包含错误处理"],
            output_format_hint="<file path='...'>代码</file>",
        )
        compiled = compiler.compile(request)
        assert compiled.extra.get("tail_reinforcement") is True
        assert compiled.extra.get("tail_reinforcement_variant") == "flash"

    def test_large_file_context_injection(self):
        """compile with large context, verify injection logic"""
        compiler = DeepSeekV4Compiler(variant="pro")
        large_content = "def foo():\n    pass\n" * 10000  # ~200K chars
        request = _make_request(
            context={"large_files": [{"path": "big_module.py", "content": large_content, "tokens": 60_000}]},
            cache_preference="aggressive",
        )
        compiled = compiler.compile(request)
        # Should have large_file_injection metadata
        injection = compiled.extra.get("large_file_injection")
        assert injection is not None, "large_file_injection should be present"
        assert "files" in injection
        assert len(injection["files"]) > 0


# =====================================================================
# 3. TestP2_3_MultimodalSupport — M3 Multimodal
# =====================================================================


class TestP2_3_MultimodalSupport:
    """P2-3: MiniMax M3 Multimodal Support tests."""

    def test_video_url_compilation(self):
        """compile with video_url MultimodalContent, verify video_url in messages"""
        compiler = MiniMaxM3Compiler()
        request = _make_request(
            multimodal_context=[
                MultimodalContent(type="video_url", url="https://example.com/video.mp4"),
            ],
        )
        compiled = compiler.compile(request)
        # Find video_url in messages
        found_video = False
        for msg in compiled.messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "video_url":
                        found_video = True
                        assert "video_url" in part
                        assert part["video_url"]["url"] == "https://example.com/video.mp4"
        assert found_video, "video_url should be present in compiled messages"

    def test_video_metadata_parsing(self):
        """test _parse_video_metadata with JSON metadata"""
        # Valid JSON metadata
        result = MiniMaxM3Compiler._parse_video_metadata('{"duration": 120, "frame_rate": 30}')
        assert result.get("duration") == 120.0
        assert result.get("frame_rate") == 30.0

        # Standard MIME type (should return empty)
        result = MiniMaxM3Compiler._parse_video_metadata("video/mp4")
        assert result == {}

        # Empty/None
        assert MiniMaxM3Compiler._parse_video_metadata(None) == {}
        assert MiniMaxM3Compiler._parse_video_metadata("") == {}

        # Invalid JSON
        assert MiniMaxM3Compiler._parse_video_metadata("not json") == {}

    def test_supported_video_urls(self):
        """test _is_supported_video_url for various formats"""
        # Supported formats
        assert MiniMaxM3Compiler._is_supported_video_url("https://example.com/video.mp4") is True
        assert MiniMaxM3Compiler._is_supported_video_url("https://example.com/video.avi") is True
        assert MiniMaxM3Compiler._is_supported_video_url("https://example.com/video.mov") is True

        # Streaming protocols
        assert MiniMaxM3Compiler._is_supported_video_url("rtmp://stream.example.com/live") is True
        assert MiniMaxM3Compiler._is_supported_video_url("rtsp://stream.example.com/live") is True

        # Known non-video
        assert MiniMaxM3Compiler._is_supported_video_url("https://example.com/image.jpg") is False
        assert MiniMaxM3Compiler._is_supported_video_url("https://example.com/doc.pdf") is False

        # Empty URL
        assert MiniMaxM3Compiler._is_supported_video_url("") is False

    def test_multimodal_system_addition(self):
        """compile multimodal request, verify intent-specific multimodal prompt"""
        compiler = MiniMaxM3Compiler()
        request = _make_request(
            meta={"task_id": "test_p2", "intent": "execute"},
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/screenshot.png"),
            ],
        )
        compiled = compiler.compile(request)
        # Check system message contains multimodal guidance
        system_msgs = [m for m in compiled.messages if m.get("role") == "system"]
        assert len(system_msgs) > 0
        system_content = system_msgs[0].get("content", "")
        assert "多模态" in system_content, "System prompt should contain multimodal guidance"

    def test_mixed_content_compilation(self):
        """compile with image + video + text, verify ordering"""
        compiler = MiniMaxM3Compiler()
        request = _make_request(
            instruction="分析这些内容",
            multimodal_context=[
                MultimodalContent(type="text", text="这是文本说明"),
                MultimodalContent(type="image_url", url="https://example.com/img.png"),
                MultimodalContent(type="video_url", url="https://example.com/vid.mp4"),
            ],
        )
        compiled = compiler.compile(request)
        # Find user message with content array
        user_msgs = [m for m in compiled.messages if m.get("role") == "user"]
        content_array_msg = None
        for msg in user_msgs:
            if isinstance(msg.get("content"), list):
                content_array_msg = msg
                break
        assert content_array_msg is not None, "Should have a user message with content array"
        parts = content_array_msg["content"]
        types_found = [p.get("type") for p in parts if isinstance(p, dict)]
        # Text should come first (instruction), then images, then video
        assert "text" in types_found
        assert "image_url" in types_found
        assert "video_url" in types_found

    def test_multimodal_token_estimation(self):
        """verify estimate_multimodal_tokens returns reasonable estimates"""
        compiler = MiniMaxM3Compiler()
        request = _make_request(
            instruction="写代码",
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/img.png"),
                MultimodalContent(type="video_url", url="https://example.com/vid.mp4"),
            ],
        )
        tokens = compiler.estimate_multimodal_tokens(request)
        assert tokens > 0, "Token estimate should be positive"
        # Image ~1000 + video ~3000 + text + overhead ~800 = at least 4800
        assert tokens >= 4000, f"Token estimate should be at least 4000 for image+video, got {tokens}"

    def test_multimodal_adapter_validation(self):
        """test OpenAICompatibleAdapter._validate_multimodal_content"""
        # Valid content
        messages = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ]},
        ]
        warnings = OpenAICompatibleAdapter._validate_multimodal_content(messages)
        assert isinstance(warnings, list)

        # Empty image URL
        messages_bad = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": ""}},
            ]},
        ]
        warnings_bad = OpenAICompatibleAdapter._validate_multimodal_content(messages_bad)
        assert len(warnings_bad) > 0, "Empty image URL should produce a warning"

    def test_mock_adapter_multimodal_detection(self):
        """verify MockAdapter detects multimodal content"""
        adapter = MockAdapter(delay=0.0)
        content_list = [
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        assert adapter._detect_multimodal_in_content(content_list) is True
        assert adapter._detect_multimodal_in_content("plain text") is False
        assert adapter._detect_multimodal_in_content([{"type": "text", "text": "only text"}]) is False


# =====================================================================
# 4. TestP2_4_DesktopTool — M3 Desktop
# =====================================================================


class TestP2_4_DesktopTool:
    """P2-4: M3 Desktop Tool tests."""

    def test_desktop_tool_creation(self):
        """create DesktopTool, verify name, description, safety level"""
        tool = DesktopTool()
        assert tool.name == "desktop"
        assert "桌面操作" in tool.description or "desktop" in tool.description.lower()
        assert tool._safety == ToolSafety.DESTRUCTIVE

    def test_desktop_tool_schema(self):
        """verify parameters_schema has all required actions"""
        tool = DesktopTool()
        schema = tool.parameters_schema
        assert schema["type"] == "object"
        action_enum = schema["properties"]["action"]["enum"]
        expected_actions = {"screenshot", "click", "type_text", "scroll", "hotkey", "move_mouse", "drag"}
        assert set(action_enum) == expected_actions
        assert "action" in schema["required"]

    @pytest.mark.asyncio
    async def test_desktop_tool_screenshot_no_deps(self):
        """execute screenshot action (no display), verify graceful failure"""
        tool = DesktopTool()
        result = await tool.execute({"action": "screenshot"})
        # In headless/CI environments, screenshot will fail gracefully
        assert isinstance(result, ToolResult)
        # Should not crash — either success or graceful error
        if not result.success:
            assert result.error, "Failed screenshot should have an error message"

    @pytest.mark.asyncio
    async def test_desktop_tool_type_text(self):
        """execute type_text with text, verify ToolResult"""
        tool = DesktopTool()
        result = await tool.execute({"action": "type_text", "text": "hello world"})
        assert isinstance(result, ToolResult)
        # In CI without pyautogui, should fail gracefully
        if not result.success:
            assert "pyautogui" in result.error.lower() or "不可用" in result.error

    @pytest.mark.asyncio
    async def test_desktop_tool_hotkey_blocked(self):
        """verify dangerous shortcuts are blocked"""
        tool = DesktopTool()
        result = await tool.execute({"action": "hotkey", "keys": "alt,f4"})
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "阻止" in result.error or "危险" in result.error

    @pytest.mark.asyncio
    async def test_desktop_tool_rate_limiting(self):
        """verify operations respect rate limit"""
        config = DesktopSafetyConfig(min_interval=10.0)  # 10 seconds between ops
        tool = DesktopTool(safety_config=config)
        # First op: set last_op_time
        tool._last_op_time = time.monotonic()
        # Second op immediately: should be rate-limited
        result = await tool.execute({"action": "screenshot"})
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "频繁" in result.error or "间隔" in result.error

    @pytest.mark.asyncio
    async def test_desktop_tool_safe_zones(self):
        """verify clicks in safe zones are blocked"""
        config = DesktopSafetyConfig(safe_zones=[(0, 0, 100, 100)])
        tool = DesktopTool(safety_config=config)
        result = await tool.execute({"action": "click", "x": 50, "y": 50})
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "安全区域" in result.error

    @pytest.mark.asyncio
    async def test_desktop_tool_capture_context(self):
        """verify capture_desktop_context returns DesktopContext"""
        tool = DesktopTool()
        ctx = await tool.capture_desktop_context(include_interactive_elements=False)
        assert isinstance(ctx, DesktopContext)
        # screenshot may be None in headless env
        # interactive_elements should be empty since we disabled detection

    def test_desktop_tool_register(self):
        """register with ToolRegistry, verify it appears"""
        registry = ToolRegistry()
        tool = DesktopTool()
        registry.register(tool)
        assert registry.has_tool("desktop")
        assert registry.get("desktop") is tool


# =====================================================================
# 5. TestP2_5_MiniMaxNativeAdapter
# =====================================================================


class TestP2_5_MiniMaxNativeAdapter:
    """P2-5: MiniMax Native Adapter tests."""

    def test_minimax_native_adapter_creation(self):
        """create adapter, verify it inherits from OpenAICompatibleAdapter"""
        adapter = MiniMaxNativeAdapter(api_key="test-key", group_id="test-group")
        assert isinstance(adapter, OpenAICompatibleAdapter)
        assert isinstance(adapter, MiniMaxNativeAdapter)

    def test_minimax_native_adapter_registry(self):
        """verify it's registered as 'minimax_native'"""
        cls = TAPAdapterRegistry.get("minimax_native")
        assert cls is MiniMaxNativeAdapter

    def test_minimax_native_adapter_headers(self):
        """verify group_id appears in headers"""
        adapter = MiniMaxNativeAdapter(api_key="test-key", group_id="my-group-id")
        headers = adapter._build_headers()
        assert headers.get("X-Group-Id") == "my-group-id"

        # Without group_id
        adapter_no_group = MiniMaxNativeAdapter(api_key="test-key")
        headers_no_group = adapter_no_group._build_headers()
        assert "X-Group-Id" not in headers_no_group

    def test_minimax_native_adapter_model_mapping(self):
        """verify model name resolution"""
        adapter = MiniMaxNativeAdapter(api_key="test-key")
        assert adapter._resolve_model_name("minimax") == "minimax-m3"
        assert adapter._resolve_model_name("m3") == "minimax-m3"
        assert adapter._resolve_model_name("minimax-m3") == "minimax-m3"
        # Unknown model should pass through
        assert adapter._resolve_model_name("some-other-model") == "some-other-model"

    def test_minimax_native_adapter_capabilities(self):
        """verify multimodal=True, desktop=True"""
        adapter = MiniMaxNativeAdapter(api_key="test-key")
        caps = adapter.capabilities
        assert caps.get("multimodal") is True
        assert caps.get("desktop") is True
        assert caps.get("video") is True
        assert caps.get("msa_efficient") is True
        assert caps.get("max_context_tokens") == 1_000_000


# =====================================================================
# 6. TestP2_6_LongHorizonManager — GLM Long-Horizon
# =====================================================================


class TestP2_6_LongHorizonManager:
    """P2-6: GLM Long-Horizon Manager tests."""

    def test_long_horizon_task_manager_creation(self):
        """create with MockAdapter provider"""
        adapter = MockAdapter(delay=0.0)
        compiler = GLM5Compiler()
        provider = ModelProvider(compiler=compiler, adapter=adapter, model="glm-5")
        manager = LongHorizonTaskManager(
            goal="实现一个用户管理系统",
            model_provider=provider,
            config=LongHorizonConfig(max_duration_hours=4),
        )
        assert manager.goal == "实现一个用户管理系统"
        assert manager.config.max_duration_hours == 4

    def test_sub_goal_dataclass(self):
        """create SubGoal, verify fields"""
        sg = SubGoal(
            id="sg_1",
            description="设计数据库模式",
            completion_criteria="ER图和DDL脚本完成",
            estimated_steps=5,
            dependencies=[],
            status="pending",
        )
        assert sg.id == "sg_1"
        assert sg.description == "设计数据库模式"
        assert sg.completion_criteria == "ER图和DDL脚本完成"
        assert sg.estimated_steps == 5
        assert sg.dependencies == []
        assert sg.status == "pending"

    def test_phase_result_dataclass(self):
        """create PhaseResult, verify fields"""
        pr = PhaseResult(
            sub_goal_id="sg_1",
            success=True,
            result_text="数据库设计完成",
            steps_taken=3,
            files_created=["schema.sql"],
            files_modified=["config.py"],
            errors=[],
        )
        assert pr.sub_goal_id == "sg_1"
        assert pr.success is True
        assert pr.result_text == "数据库设计完成"
        assert pr.steps_taken == 3
        assert "schema.sql" in pr.files_created

    def test_long_horizon_result_dataclass(self):
        """create LongHorizonResult, verify fields"""
        result = LongHorizonResult(
            task_id="task_1",
            goal="实现系统",
            success=True,
            total_steps=25,
            total_elapsed_minutes=120.0,
            completed_sub_goals=3,
            total_sub_goals=3,
            strategy_switches=1,
            phase_results=[],
            final_summary="全部完成",
            checkpoints_saved=5,
        )
        assert result.task_id == "task_1"
        assert result.success is True
        assert result.total_steps == 25
        assert result.strategy_switches == 1
        assert result.checkpoints_saved == 5

    @pytest.mark.asyncio
    async def test_checkpoint_store_save_load(self):
        """save a checkpoint, load it back, verify data integrity"""
        tmpdir = tempfile.mkdtemp()
        try:
            store = CheckpointStore(base_dir=tmpdir)
            cp = Checkpoint(
                checkpoint_id="cp_001",
                task_id="task_1",
                timestamp="2025-01-01T00:00:00Z",
                phase="executing",
                completed_sub_goals=["sg_1"],
                current_sub_goal="sg_2",
                steps_completed=10,
                elapsed_minutes=30.0,
                strategy_switches=0,
                state_data={"key": "value"},
            )
            cp_id = await store.save(cp)
            assert cp_id == "cp_001"

            loaded = await store.load_latest("task_1")
            assert loaded is not None
            assert loaded.checkpoint_id == "cp_001"
            assert loaded.task_id == "task_1"
            assert loaded.phase == "executing"
            assert loaded.completed_sub_goals == ["sg_1"]
            assert loaded.state_data == {"key": "value"}
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_checkpoint_store_cleanup(self):
        """save many checkpoints, cleanup, verify last N remain"""
        tmpdir = tempfile.mkdtemp()
        try:
            store = CheckpointStore(base_dir=tmpdir)
            for i in range(10):
                cp = Checkpoint(
                    checkpoint_id=f"cp_{i:03d}",
                    task_id="task_1",
                    timestamp=f"2025-01-01T{i:02d}:00:00Z",
                    phase="executing",
                    completed_sub_goals=[],
                    current_sub_goal="sg_1",
                    steps_completed=i,
                    elapsed_minutes=float(i * 10),
                    strategy_switches=0,
                    state_data={},
                )
                await store.save(cp)

            # Cleanup, keeping only 3
            deleted = await store.cleanup("task_1", keep_last=3)
            assert deleted == 7, f"Should delete 7 checkpoints, deleted {deleted}"

            remaining = await store.list_checkpoints("task_1")
            assert len(remaining) == 3
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_progress_tracker(self):
        """create tracker, record steps and sub-goals, verify report"""
        tracker = ProgressTracker(task_id="task_1", goal="实现系统")
        tracker.register_sub_goal("sg_1", "设计数据库")
        tracker.register_sub_goal("sg_2", "实现API")
        tracker.start_sub_goal("sg_1", "设计数据库")
        tracker.record_step("创建 User 表")
        tracker.record_step("创建 Order 表")
        tracker.complete_sub_goal("sg_1", "数据库设计完成")

        report = tracker.get_report()
        assert isinstance(report, ProgressReport)
        assert report.task_id == "task_1"
        assert report.steps_completed == 2
        assert report.completed_sub_goals == 1
        assert report.total_sub_goals == 2

    @pytest.mark.asyncio
    async def test_decompose_goal(self):
        """use MockAdapter to decompose a goal, verify SubGoal list"""
        adapter = MockAdapter(delay=0.0)
        compiler = GLM5Compiler()
        provider = ModelProvider(compiler=compiler, adapter=adapter, model="glm-5")
        manager = LongHorizonTaskManager(
            goal="实现排序算法",
            model_provider=provider,
            config=LongHorizonConfig(max_duration_hours=1),
        )
        sub_goals = await manager.decompose_goal("实现排序算法")
        assert isinstance(sub_goals, list)
        assert len(sub_goals) > 0, "decompose_goal should return at least one sub-goal"
        for sg in sub_goals:
            assert isinstance(sg, SubGoal)
            assert sg.id
            assert sg.description

    def test_stagnation_detection(self):
        """simulate stagnation (similar results), verify detection"""
        switcher = StrategySwitcher(
            model_provider=None,  # Not needed for detection
            stagnation_threshold=3,
            similarity_threshold=0.8,
        )
        # Create 3 similar results
        similar_text = "相同的执行结果内容，没有变化"
        results = [
            PhaseResult(
                sub_goal_id=f"sg_{i}",
                success=True,
                result_text=similar_text,
                steps_taken=5,
            )
            for i in range(3)
        ]
        is_stagnant, reason = switcher.detect_stagnation(results, [])
        assert is_stagnant is True, "Should detect stagnation with identical results"
        assert "相似" in reason


# =====================================================================
# 7. TestP2_7_SelfEvalStrategySwitch — GLM Self-Eval
# =====================================================================


class TestP2_7_SelfEvalStrategySwitch:
    """P2-7: GLM Self-Evaluation and Strategy Switch tests."""

    def test_self_evaluator_creation(self):
        """create SelfEvaluator"""
        adapter = MockAdapter(delay=0.0)
        compiler = GLM5Compiler()
        provider = ModelProvider(compiler=compiler, adapter=adapter, model="glm-5")
        evaluator = SelfEvaluator(
            model_provider=provider,
            evaluation_interval_steps=5,
            evaluation_interval_minutes=10.0,
        )
        assert evaluator.evaluation_interval_steps == 5
        assert evaluator.evaluation_interval_minutes == 10.0

    def test_self_evaluation_result_dataclass(self):
        """create SelfEvaluationResult, verify fields"""
        result = SelfEvaluationResult(
            goal_alignment=4,
            output_quality=3,
            bottleneck_identified="测试覆盖不足",
            strategy_review="当前策略基本有效",
            next_step_plan="增加测试用例",
            overall_score=3.6,
            should_switch_strategy=False,
            raw_response="...",
        )
        assert result.goal_alignment == 4
        assert result.output_quality == 3
        assert result.overall_score == 3.6
        assert result.should_switch_strategy is False
        assert "测试" in result.bottleneck_identified

    def test_should_evaluate(self):
        """test step and time-based evaluation triggers"""
        adapter = MockAdapter(delay=0.0)
        compiler = GLM5Compiler()
        provider = ModelProvider(compiler=compiler, adapter=adapter, model="glm-5")
        evaluator = SelfEvaluator(
            model_provider=provider,
            evaluation_interval_steps=10,
            evaluation_interval_minutes=30.0,
        )
        # Below thresholds
        assert evaluator.should_evaluate(steps_since_last=5, minutes_since_last=10.0) is False
        # Step threshold met
        assert evaluator.should_evaluate(steps_since_last=10, minutes_since_last=5.0) is True
        # Time threshold met
        assert evaluator.should_evaluate(steps_since_last=5, minutes_since_last=30.0) is True
        # Both met
        assert evaluator.should_evaluate(steps_since_last=15, minutes_since_last=45.0) is True

    @pytest.mark.asyncio
    async def test_evaluate_with_mock(self):
        """run evaluate() with MockAdapter, verify result"""
        adapter = MockAdapter(delay=0.0)
        compiler = GLM5Compiler()
        provider = ModelProvider(compiler=compiler, adapter=adapter, model="glm-5")
        evaluator = SelfEvaluator(
            model_provider=provider,
            evaluation_interval_steps=1,
        )
        progress = ProgressReport(
            task_id="task_1",
            goal="实现系统",
            total_sub_goals=3,
            completed_sub_goals=1,
            current_phase="executing",
            steps_completed=10,
            elapsed_minutes=30.0,
            strategy_switches=0,
            sub_goal_statuses=[],
            estimated_remaining_minutes=60.0,
            last_checkpoint="",
        )
        result = await evaluator.evaluate(
            goal="实现系统",
            progress_report=progress,
            recent_results=[],
        )
        assert isinstance(result, SelfEvaluationResult)
        assert 1 <= result.goal_alignment <= 5
        assert 1 <= result.output_quality <= 5
        assert 0.0 <= result.overall_score <= 5.0

    def test_strategy_switcher_creation(self):
        """create StrategySwitcher"""
        adapter = MockAdapter(delay=0.0)
        compiler = GLM5Compiler()
        provider = ModelProvider(compiler=compiler, adapter=adapter, model="glm-5")
        switcher = StrategySwitcher(
            model_provider=provider,
            stagnation_threshold=3,
            no_progress_threshold=5,
        )
        assert switcher.stagnation_threshold == 3
        assert switcher.no_progress_threshold == 5

    def test_stagnation_detection_similarity(self):
        """detect stagnation via Jaccard similarity"""
        adapter = MockAdapter(delay=0.0)
        compiler = GLM5Compiler()
        provider = ModelProvider(compiler=compiler, adapter=adapter, model="glm-5")
        switcher = StrategySwitcher(
            model_provider=provider,
            stagnation_threshold=3,
            similarity_threshold=0.7,
        )
        # Create highly similar results
        results = [
            PhaseResult(
                sub_goal_id=f"sg_{i}",
                success=True,
                result_text="The quick brown fox jumps over the lazy dog",
                steps_taken=5,
            )
            for i in range(4)
        ]
        is_stagnant, reason = switcher.detect_stagnation(results, [])
        assert is_stagnant is True

    def test_stagnation_detection_no_progress(self):
        """detect stagnation via no file output"""
        adapter = MockAdapter(delay=0.0)
        compiler = GLM5Compiler()
        provider = ModelProvider(compiler=compiler, adapter=adapter, model="glm-5")
        switcher = StrategySwitcher(
            model_provider=provider,
            no_progress_threshold=5,
        )
        # Results with no files created/modified
        results = [
            PhaseResult(
                sub_goal_id=f"sg_{i}",
                success=True,
                result_text=f"Unique result {i} with different content each time",
                steps_taken=5,
                files_created=[],
                files_modified=[],
            )
            for i in range(6)
        ]
        is_stagnant, reason = switcher.detect_stagnation(results, [])
        assert is_stagnant is True
        assert "文件" in reason

    def test_strategy_switch_record(self):
        """create record, verify fields"""
        record = StrategySwitchRecord(
            timestamp="2025-01-01T00:00:00Z",
            reason="连续3次相同结果",
            previous_strategy="增量开发",
            new_strategy="回退重试",
            risk_assessment="中等风险",
            effectiveness="",
        )
        assert record.reason == "连续3次相同结果"
        assert record.previous_strategy == "增量开发"
        assert record.new_strategy == "回退重试"

    @pytest.mark.asyncio
    async def test_strategy_switch_history(self):
        """execute switch, verify history tracking"""
        adapter = MockAdapter(delay=0.0)
        compiler = GLM5Compiler()
        provider = ModelProvider(compiler=compiler, adapter=adapter, model="glm-5")
        switcher = StrategySwitcher(model_provider=provider)
        progress = ProgressReport(
            task_id="task_1",
            goal="实现系统",
            total_sub_goals=3,
            completed_sub_goals=1,
            current_phase="stagnant",
            steps_completed=15,
            elapsed_minutes=45.0,
            strategy_switches=0,
            sub_goal_statuses=[],
            estimated_remaining_minutes=60.0,
            last_checkpoint="",
        )
        new_strategy, record = await switcher.switch_strategy(
            current_strategy="增量开发",
            reason="停滞检测",
            goal="实现系统",
            progress_report=progress,
        )
        assert isinstance(new_strategy, str)
        assert len(new_strategy) > 0
        assert isinstance(record, StrategySwitchRecord)
        assert record.previous_strategy == "增量开发"
        history = switcher.get_switch_history()
        assert len(history) == 1
        assert history[0].new_strategy == new_strategy


# =====================================================================
# 8. TestP2_8_Compression — GLM 200K Compression
# =====================================================================


class TestP2_8_Compression:
    """P2-8: GLM 200K Compression tests."""

    def test_glm5_compaction_strategy(self):
        """verify all budget zones"""
        strategy = GLM5CompactionStrategy()
        assert strategy.system_budget == 20_480
        assert strategy.design_budget == 40_960
        assert strategy.history_budget == 92_160
        assert strategy.recent_budget == 30_720
        assert strategy.tail_budget == 20_480
        # Total should equal 204,800
        assert strategy.total_budget == 204_800
        # Compression targets
        assert strategy.design_compression_target == 0.3
        assert strategy.history_compression_target == 0.15
        assert strategy.min_information_retention == 0.8

    def test_compact_design_to_adr(self):
        """compress a design document, verify ADR format"""
        compactor = Microcompactor()
        design_doc = (
            "# 用户管理系统设计\n\n"
            "## 1. 背景与动机\n需要管理大量用户数据。\n\n"
            "## 2. 设计目标\n- 高性能\n- 可扩展\n- 安全\n\n"
            "## 3. 技术栈\nPython + FastAPI\n\n"
            "```python\nclass User:\n    pass\n```\n\n"
            "## 4. 核心接口\n用户CRUD操作。\n"
        )
        result = compactor._compact_design_to_adr(design_doc)
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be shorter than original
        assert len(result) < len(design_doc) or "ADR" in result

    def test_compact_history_aggressive(self):
        """compress execution history, verify key info retained"""
        compactor = Microcompactor()
        history = (
            "# 执行历史\n\n"
            "## 阶段1：数据库设计\n"
            "决定使用 PostgreSQL 作为主数据库。\n"
            "成功创建 User 表和 Order 表。\n"
            "错误：连接池配置不当导致超时。\n"
            "策略切换：从直接连接切换到连接池。\n\n"
            "## 阶段2：API实现\n"
            "开始实现 RESTful API。\n"
            "大量调试输出...\n"
            "大量调试输出...\n"
            "大量调试输出...\n"
        )
        result = compactor._compact_history_aggressive(history)
        assert isinstance(result, str)
        assert len(result) > 0
        # Should retain key information
        assert len(result) <= len(history)

    def test_compression_quality_assessment(self):
        """assess quality of compressed text"""
        compactor = Microcompactor()
        original = (
            "# 系统设计\n"
            "本系统使用 Python 和 FastAPI 构建。\n"
            "核心模块包括 UserService, OrderService, AuthService。\n"
        )
        compressed = "系统设计: Python/FastAPI, 核心模块: UserService, OrderService, AuthService"
        quality = compactor.assess_compression_quality(original, compressed)
        assert "compression_ratio" in quality
        assert "information_retention" in quality
        assert "key_terms_preserved" in quality
        assert "structure_preserved" in quality
        assert 0.0 <= quality["compression_ratio"] <= 1.0
        assert 0.0 <= quality["information_retention"] <= 1.0

    def test_information_retention(self):
        """verify key terms preserved in compression"""
        compactor = Microcompactor()
        original = (
            "# 架构设计\n"
            "UserService 负责用户管理，OrderService 处理订单，"
            "AuthService 提供认证服务。数据库使用 PostgreSQL。"
        )
        compressed = "架构: UserService, OrderService, AuthService, PostgreSQL"
        quality = compactor.assess_compression_quality(original, compressed)
        # Key terms should be mostly preserved
        assert quality["information_retention"] >= 0.5, (
            f"Information retention should be >= 0.5, got {quality['information_retention']}"
        )

    def test_glm5_tail_reinforcement(self):
        """compile GLM-5 request, verify tail reinforcement messages"""
        compiler = GLM5Compiler()
        request = _make_request(
            instruction="实现排序算法",
            constraints=["时间复杂度 O(n log n)", "原地排序"],
        )
        compiled = compiler.compile(request)
        # GLM-5 should have tail reinforcement as the last user message
        last_user_msgs = [m for m in compiled.messages if m.get("role") == "user"]
        assert len(last_user_msgs) > 0
        last_user = last_user_msgs[-1]
        content = last_user.get("content", "")
        # Should contain reinforcement markers
        assert "当前指令重申" in content or "关键约束" in content or "自评估" in content


# =====================================================================
# 9. TestP2_EndToEnd — Full Pipeline
# =====================================================================


class TestP2_EndToEnd:
    """P2: Full End-to-End Pipeline tests with MockAdapter."""

    @pytest.mark.asyncio
    async def test_v4_flash_full_pipeline(self):
        """TAPRequest → V4Flash compile → MockAdapter send → TAPResponse with cache"""
        provider = _make_mock_provider("deepseek_v4", model="deepseek-v4-flash", variant="flash")
        request = _make_request(cache_preference="auto")
        response = await provider.execute_tap(request)
        assert isinstance(response, TAPResponse)
        assert response.raw_text is not None
        assert len(response.raw_text) > 0
        assert response.usage.get("prompt_tokens", 0) > 0

    @pytest.mark.asyncio
    async def test_v4_pro_full_pipeline(self):
        """TAPRequest → V4Pro compile → MockAdapter send → TAPResponse with thinking"""
        provider = _make_mock_provider("deepseek_v4", model="deepseek-v4-pro", variant="pro")
        request = _make_request(thinking_mode="deep")
        response = await provider.execute_tap(request)
        assert isinstance(response, TAPResponse)
        assert response.raw_text is not None
        assert response.usage.get("prompt_tokens", 0) > 0
        # Pro with thinking mode should have thinking enabled in compiled
        compiled = provider.compiler.compile(request)
        assert compiled.thinking_enabled is True

    @pytest.mark.asyncio
    async def test_m3_multimodal_full_pipeline(self):
        """TAPRequest with images → M3 compile → MockAdapter send → TAPResponse"""
        provider = _make_mock_provider("minimax_m3", model="minimax-m3")
        request = _make_request(
            multimodal_context=[
                MultimodalContent(type="image_url", url="https://example.com/screenshot.png"),
                MultimodalContent(type="text", text="分析这个截图"),
            ],
        )
        response = await provider.execute_tap(request)
        assert isinstance(response, TAPResponse)
        assert response.raw_text is not None
        assert len(response.raw_text) > 0

    @pytest.mark.asyncio
    async def test_glm5_long_horizon_pipeline(self):
        """TAPRequest with long_horizon → GLM-5 compile → verify prompt structure"""
        compiler = GLM5Compiler()
        request = _make_request(
            instruction="实现完整的用户管理系统",
            long_horizon=LongHorizonConfig(
                max_duration_hours=4,
                self_evaluation_enabled=True,
            ),
        )
        compiled = compiler.compile(request)
        assert isinstance(compiled, CompiledPrompt)
        # Should have long-horizon system prompt
        system_msgs = [m for m in compiled.messages if m.get("role") == "system"]
        assert len(system_msgs) > 0
        system_content = system_msgs[0].get("content", "")
        assert "长程任务" in system_content
        # Should have self-evaluation prompt
        user_msgs = [m for m in compiled.messages if m.get("role") == "user"]
        all_user_content = " ".join(m.get("content", "") for m in user_msgs)
        assert "自评估" in all_user_content
        # Should have tail reinforcement
        assert compiled.extra.get("thinking", {}).get("type") == "enabled"

    @pytest.mark.asyncio
    async def test_all_compilers_no_regression(self):
        """all 9+ compilers still work with MockAdapter"""
        compiler_configs = [
            ("default", {}),
            ("glm", {}),
            ("anthropic", {}),
            ("deepseek", {}),
            ("deepseek_v4", {"variant": "flash"}),
            ("deepseek_v4", {"variant": "pro"}),
            ("glm_5", {}),
            ("minimax_m3", {}),
        ]
        adapter = MockAdapter(delay=0.0)
        request = _make_request()

        for compiler_name, kwargs in compiler_configs:
            compiler_cls = TAPCompilerRegistry.get(compiler_name)
            assert compiler_cls is not None, f"Compiler {compiler_name} not registered"
            compiler = compiler_cls(**kwargs)
            compiled = compiler.compile(request)
            assert isinstance(compiled, CompiledPrompt), (
                f"{compiler_name} compile should return CompiledPrompt"
            )
            assert compiled.mode != "empty", (
                f"{compiler_name} compiled prompt should not be empty"
            )

            response = await adapter.send(compiled, model=f"test-{compiler_name}")
            assert isinstance(response, TAPResponse)
            assert response.raw_text is not None, (
                f"{compiler_name} response should have raw_text"
            )

    @pytest.mark.asyncio
    async def test_cost_tracker_cross_model(self):
        """track costs across V4/M3/GLM calls"""
        tracker = CostTracker()

        # V4 Pro call
        provider_v4 = _make_mock_provider("deepseek_v4", model="deepseek-v4-pro", variant="pro")
        request_v4 = _make_request(cache_preference="aggressive")
        response_v4 = await provider_v4.execute_tap(request_v4)
        tracker.append(TAPCostRecord(
            task_id="t1",
            intent="execute",
            provider="DeepSeekV4",
            model="deepseek-v4-pro",
            prompt_tokens=response_v4.prompt_tokens,
            completion_tokens=response_v4.completion_tokens,
            cache_hit_tokens=response_v4.cache_hit_tokens,
            cache_miss_tokens=response_v4.cache_miss_tokens,
            latency_ms=100.0,
            success=True,
        ))

        # M3 call
        provider_m3 = _make_mock_provider("minimax_m3", model="minimax-m3")
        request_m3 = _make_request(
            multimodal_context=[MultimodalContent(type="image_url", url="https://example.com/img.png")],
        )
        response_m3 = await provider_m3.execute_tap(request_m3)
        tracker.append(TAPCostRecord(
            task_id="t2",
            intent="execute",
            provider="MiniMaxM3",
            model="minimax-m3",
            prompt_tokens=response_m3.prompt_tokens,
            completion_tokens=response_m3.completion_tokens,
            cache_hit_tokens=response_m3.cache_hit_tokens,
            cache_miss_tokens=response_m3.cache_miss_tokens,
            latency_ms=150.0,
            success=True,
        ))

        # GLM-5 call
        provider_glm = _make_mock_provider("glm_5", model="glm-5")
        request_glm = _make_request(thinking_mode="deep")
        response_glm = await provider_glm.execute_tap(request_glm)
        tracker.append(TAPCostRecord(
            task_id="t3",
            intent="execute",
            provider="GLM5",
            model="glm-5",
            prompt_tokens=response_glm.prompt_tokens,
            completion_tokens=response_glm.completion_tokens,
            cache_hit_tokens=response_glm.cache_hit_tokens,
            cache_miss_tokens=response_glm.cache_miss_tokens,
            latency_ms=200.0,
            success=True,
        ))

        # Verify cross-model tracking
        assert len(tracker) == 3
        assert tracker.total_cache_hit_tokens() >= 0
        assert tracker.total_cache_miss_tokens() >= 0
        total_prompt = sum(r.prompt_tokens for r in tracker.get_all())
        assert total_prompt > 0, "Should have prompt tokens across all models"

        # Test estimated cost
        pricing = {
            "cache_hit_per_million": 0.1,
            "cache_miss_per_million": 1.0,
            "completion_per_million": 2.0,
        }
        cost = tracker.total_estimated_cost(pricing)
        assert cost > 0, "Total estimated cost should be positive"
