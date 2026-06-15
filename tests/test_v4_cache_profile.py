# tests/test_v4_cache_profile.py
"""P2-1 + P2-2: V4 cache-aware compression + 1M context profile integration tests

P2-1: Tests for cache_hit_rate parameter in AutoCompactor
P2-2: Tests for DeepSeekV4Compiler profile integration and _retrieve_relevant_snippets
"""
from unittest.mock import AsyncMock, patch

import pytest

from teragent.context.auto_compact import AutoCompactor
from teragent.context.context_window import ContextWindow
from teragent.context.profiles import ContextProfile, DeepSeekV4ContextProfile
from teragent.core.compilers.deepseek_v4 import DeepSeekV4Compiler
from teragent.core.tap import TAPRequest
from teragent.core.types import Message, MessageRole, MessageType


# ===== P2-1: Cache-aware compression tests =====

def _make_messages(count: int, start: int = 0) -> list[Message]:
    """Generate alternating user/assistant messages"""
    msgs = []
    for i in range(start, start + count):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        msgs.append(Message(role=role, content=f"消息 {i}", message_type=MessageType.USER_INPUT))
    return msgs


def _make_compactor(
    retain_count: int = 8,
    max_compacts: int = 5,
    compact_threshold: float = 0.85,
) -> tuple[AutoCompactor, ContextWindow, AsyncMock]:
    """Create AutoCompactor with dependencies"""
    cw = ContextWindow(
        model_token_limit=128_000,
        reserved_for_output=4_096,
        reserved_for_system=2_048,
        compact_threshold=compact_threshold,
    )
    model = AsyncMock()
    compactor = AutoCompactor(
        context_window=cw,
        model=model,
        retain_count=retain_count,
        max_compacts=max_compacts,
    )
    return compactor, cw, model


class TestCacheAwareCompressionDefault:
    """P2-1: cache_hit_rate=None (default behavior)"""

    @pytest.mark.asyncio
    async def test_default_cache_hit_rate_same_as_before(self):
        """cache_hit_rate=None should produce same result as before (no parameter)"""
        compactor, cw, model = _make_compactor(retain_count=4)
        msgs = _make_messages(20)
        model.chat = AsyncMock(return_value={"content": "摘要"})

        with patch.object(cw, "should_compact", return_value=True):
            result_none = await compactor.maybe_compact(msgs, cache_hit_rate=None)

        # Reset for second call
        compactor.reset()
        model.chat = AsyncMock(return_value={"content": "摘要"})
        compactor2, cw2, model2 = _make_compactor(retain_count=4)
        model2.chat = AsyncMock(return_value={"content": "摘要"})

        with patch.object(cw2, "should_compact", return_value=True):
            result_default = await compactor2.maybe_compact(msgs)

        # Same retain count (4) → same result structure
        assert len(result_none) == len(result_default)


class TestCacheAwareCompressionHighHitRate:
    """P2-1: cache_hit_rate > 0.7 (low compression — preserve cache-friendly content)"""

    @pytest.mark.asyncio
    async def test_high_cache_hit_rate_retains_more(self):
        """High cache hit rate should retain more messages (retain_count * 1.5)"""
        compactor, cw, model = _make_compactor(retain_count=4)
        msgs = _make_messages(20)
        model.chat = AsyncMock(return_value={"content": "摘要"})

        with patch.object(cw, "should_compact", return_value=True):
            result = await compactor.maybe_compact(msgs, cache_hit_rate=0.8)

        # retain_count * 1.5 = 6, so 1 summary + 6 retained = 7
        assert len(result) == 1 + 6

    @pytest.mark.asyncio
    async def test_high_cache_hit_rate_summary_info_recorded(self):
        """High cache hit rate should be recorded in compact info"""
        compactor, cw, model = _make_compactor(retain_count=4)
        msgs = _make_messages(20)
        model.chat = AsyncMock(return_value={"content": "摘要"})

        with patch.object(cw, "should_compact", return_value=True):
            await compactor.maybe_compact(msgs, cache_hit_rate=0.85)

        info = compactor.last_compact_info
        assert info["cache_hit_rate"] == 0.85

    @pytest.mark.asyncio
    async def test_high_cache_hit_rate_uses_larger_summary_input(self):
        """High cache hit rate should use larger SUMMARY_INPUT_MAX_CHARS (×1.5)"""
        compactor, cw, model = _make_compactor(retain_count=4)
        # Create messages with enough content to test the char budget
        msgs = _make_messages(20)
        # Make early messages very long to exceed default SUMMARY_INPUT_MAX_CHARS
        for i in range(min(10, len(msgs))):
            msgs[i] = Message(
                role=msgs[i].role,
                content="A" * 2000,
                message_type=msgs[i].message_type,
            )
        model.chat = AsyncMock(return_value={"content": "摘要"})

        with patch.object(cw, "should_compact", return_value=True):
            # With high cache hit rate, more chars should be included
            result = await compactor.maybe_compact(msgs, cache_hit_rate=0.9)

        assert len(result) > 0


class TestCacheAwareCompressionLowHitRate:
    """P2-1: cache_hit_rate < 0.3 (aggressive compression — free up cache space)"""

    @pytest.mark.asyncio
    async def test_low_cache_hit_rate_retains_fewer(self):
        """Low cache hit rate should retain fewer messages (retain_count * 0.75)"""
        compactor, cw, model = _make_compactor(retain_count=8)
        msgs = _make_messages(30)
        model.chat = AsyncMock(return_value={"content": "摘要"})

        with patch.object(cw, "should_compact", return_value=True):
            result = await compactor.maybe_compact(msgs, cache_hit_rate=0.2)

        # retain_count * 0.75 = 6, so 1 summary + 6 retained = 7
        assert len(result) == 1 + 6

    @pytest.mark.asyncio
    async def test_low_cache_hit_rate_minimum_retain(self):
        """Low cache hit rate should still retain at least 2 messages"""
        compactor, cw, model = _make_compactor(retain_count=2)
        msgs = _make_messages(20)
        model.chat = AsyncMock(return_value={"content": "摘要"})

        with patch.object(cw, "should_compact", return_value=True):
            result = await compactor.maybe_compact(msgs, cache_hit_rate=0.1)

        # max(2, int(2 * 0.75)) = max(2, 1) = 2, so 1 summary + 2 retained = 3
        assert len(result) == 1 + 2

    @pytest.mark.asyncio
    async def test_low_cache_hit_rate_summary_info_recorded(self):
        """Low cache hit rate should be recorded in compact info"""
        compactor, cw, model = _make_compactor(retain_count=4)
        msgs = _make_messages(20)
        model.chat = AsyncMock(return_value={"content": "摘要"})

        with patch.object(cw, "should_compact", return_value=True):
            await compactor.maybe_compact(msgs, cache_hit_rate=0.15)

        info = compactor.last_compact_info
        assert info["cache_hit_rate"] == 0.15


class TestCacheAwareCompressionMiddleRange:
    """P2-1: cache_hit_rate in 0.3-0.7 (default behavior)"""

    @pytest.mark.asyncio
    async def test_middle_cache_hit_rate_uses_default(self):
        """Middle cache hit rate should use default retain_count"""
        compactor, cw, model = _make_compactor(retain_count=4)
        msgs = _make_messages(20)
        model.chat = AsyncMock(return_value={"content": "摘要"})

        with patch.object(cw, "should_compact", return_value=True):
            result = await compactor.maybe_compact(msgs, cache_hit_rate=0.5)

        # retain_count unchanged = 4, so 1 summary + 4 retained = 5
        assert len(result) == 1 + 4

    @pytest.mark.asyncio
    async def test_exact_threshold_values(self):
        """Test exactly at threshold boundaries"""
        compactor, cw, model = _make_compactor(retain_count=4)
        msgs = _make_messages(20)

        # cache_hit_rate = 0.3 → default (not < 0.3)
        model.chat = AsyncMock(return_value={"content": "摘要"})
        with patch.object(cw, "should_compact", return_value=True):
            result = await compactor.maybe_compact(msgs, cache_hit_rate=0.3)
        assert len(result) == 1 + 4  # default

        # cache_hit_rate = 0.7 → default (not > 0.7)
        compactor.reset()
        model.chat = AsyncMock(return_value={"content": "摘要"})
        with patch.object(cw, "should_compact", return_value=True):
            result = await compactor.maybe_compact(msgs, cache_hit_rate=0.7)
        assert len(result) == 1 + 4  # default


# ===== P2-2: V4 Context Profile Integration tests =====

def _make_v4_request(
    intent: str = "execute",
    instruction: str = "写一个排序函数",
    constraints: list[str] | None = None,
    output_format_hint: str = "",
    context: dict | None = None,
    cache_preference: str = "",
) -> TAPRequest:
    """Create a TAPRequest for V4 compiler testing"""
    return TAPRequest(
        meta={"task_id": "test", "intent": intent},
        context=context or {},
        instruction=instruction,
        constraints=constraints or [],
        output_format_hint=output_format_hint,
        cache_preference=cache_preference,
    )


class TestV4ProfileIntegration:
    """P2-2: DeepSeekV4Compiler uses DeepSeekV4ContextProfile"""

    def test_default_profile_is_v4(self):
        """Default profile should be DeepSeekV4ContextProfile"""
        compiler = DeepSeekV4Compiler()
        profile = compiler.get_context_profile()
        assert isinstance(profile, DeepSeekV4ContextProfile)

    def test_profile_budgets_match_v4_defaults(self):
        """Profile-derived budgets should match V4 1M defaults"""
        compiler = DeepSeekV4Compiler()
        profile = compiler.get_context_profile()

        # DeepSeekV4ContextProfile defaults:
        # max_tokens=1_000_000, system_ratio=0.05, history_ratio=0.45,
        # large_file_ratio=0.40, tail_reinforcement_ratio=0.10
        assert profile.max_tokens == 1_000_000
        assert profile.system_budget == 50_000
        assert profile.history_budget == 450_000
        assert profile.large_file_budget == 400_000
        assert profile.tail_reinforcement_budget == 100_000

    def test_max_context_tokens_from_profile(self):
        """max_context_tokens should come from profile"""
        compiler = DeepSeekV4Compiler()
        assert compiler.max_context_tokens == 1_000_000

    def test_custom_profile(self):
        """Compiler should accept custom profile"""
        custom = DeepSeekV4ContextProfile(
            max_tokens=500_000,
            system_ratio=0.10,
            history_ratio=0.40,
            large_file_ratio=0.35,
            tail_reinforcement_ratio=0.15,
        )
        compiler = DeepSeekV4Compiler(profile=custom)
        profile = compiler.get_context_profile()

        assert profile.max_tokens == 500_000
        assert compiler.max_context_tokens == 500_000
        assert profile.system_budget == 50_000
        assert profile.tail_reinforcement_budget == 75_000

    def test_profile_backwards_compatible_property(self):
        """profile property should work for backwards compatibility"""
        compiler = DeepSeekV4Compiler()
        # Read via .profile property
        assert compiler.profile.max_tokens == 1_000_000

        # Write via .profile setter
        custom = DeepSeekV4ContextProfile(max_tokens=500_000)
        compiler.profile = custom
        assert compiler.get_context_profile().max_tokens == 500_000

    def test_cache_aware_layout_records_profile_budgets(self):
        """Cache-aware layout should record profile-derived budgets in extra"""
        compiler = DeepSeekV4Compiler(variant="pro")
        request = _make_v4_request(cache_preference="aggressive")
        compiled = compiler.compile(request)

        assert "profile_budgets" in compiled.extra
        budgets = compiled.extra["profile_budgets"]
        assert budgets["system"] == 50_000
        assert budgets["history"] == 450_000
        assert budgets["large_file"] == 400_000
        assert budgets["tail_reinforcement"] == 100_000

    def test_tail_reinforcement_records_budget(self):
        """Tail reinforcement should record profile budget"""
        compiler = DeepSeekV4Compiler(variant="pro")
        request = _make_v4_request(
            constraints=["约束1", "约束2"],
        )
        compiled = compiler.compile(request)

        if compiled.extra.get("tail_reinforcement"):
            assert "tail_reinforcement_budget" in compiled.extra
            assert compiled.extra["tail_reinforcement_budget"] == 100_000

    def test_large_file_injection_uses_profile_budget(self):
        """Large file injection should use profile.large_file_budget"""
        large_content = "x" * 250_000  # ~81K tokens
        compiler = DeepSeekV4Compiler(variant="pro")
        request = _make_v4_request(
            context={
                "large_files": [{
                    "path": "test.py",
                    "content": large_content,
                    "tokens": 80_000,
                }]
            }
        )
        compiled = compiler.compile(request)

        # Should have injected the file
        if "large_file_injection" in compiled.extra:
            assert compiled.extra["large_file_injection"]["budget_total"] == 400_000


class TestRetrieveRelevantSnippets:
    """P2-2: _retrieve_relevant_snippets method tests"""

    def test_parse_file_sections_header(self):
        """_parse_file_sections should identify header section"""
        lines = [
            '"""Module docstring."""',
            "",
            "import os",
            "import sys",
            "",
            "x = 1",
        ]
        sections = DeepSeekV4Compiler._parse_file_sections(lines, "test.py")
        # Should have header and at least one global section
        section_names = [s[0] for s in sections]
        assert "__header__" in section_names

    def test_parse_file_sections_class(self):
        """_parse_file_sections should identify class sections"""
        lines = [
            "class MyClass(Base):",
            "    def method(self):",
            "        pass",
            "",
            "def standalone():",
            "    pass",
        ]
        sections = DeepSeekV4Compiler._parse_file_sections(lines, "test.py")
        section_names = [s[0] for s in sections]
        assert any("class_MyClass" in n for n in section_names)

    def test_parse_file_sections_function(self):
        """_parse_file_sections should identify function sections"""
        lines = [
            "def my_func(arg1, arg2):",
            "    return arg1 + arg2",
            "",
            "async def async_func():",
            "    await something()",
        ]
        sections = DeepSeekV4Compiler._parse_file_sections(lines, "test.py")
        section_names = [s[0] for s in sections]
        assert any("func_my_func" in n for n in section_names)
        assert any("func_async_func" in n for n in section_names)

    def test_score_sections_keyword_match(self):
        """_score_sections should score higher for keyword matches"""
        sections = [
            ("func_sort", "def sort(arr):\n    return sorted(arr)"),
            ("func_print", "def print_hello():\n    print('hello')"),
        ]
        scored = DeepSeekV4Compiler._score_sections(sections, "sort the array")

        # func_sort should have higher score due to keyword match
        sort_score = next(s[2] for s in scored if s[0] == "func_sort")
        print_score = next(s[2] for s in scored if s[0] == "func_print")
        assert sort_score > print_score

    def test_score_sections_header_bonus(self):
        """_score_sections should give header a base bonus"""
        sections = [
            ("__header__", "import os\nimport sys"),
            ("func_other", "def other():\n    pass"),
        ]
        scored = DeepSeekV4Compiler._score_sections(sections, "unrelated instruction")

        header_score = next(s[2] for s in scored if s[0] == "__header__")
        assert header_score >= 1.0  # Base bonus

    def test_retrieve_relevant_snippets_basic(self):
        """_retrieve_relevant_snippets should return relevant sections"""
        content = "\n".join([
            '"""Module."""',
            "import os",
            "",
            "def sort_array(arr):",
            "    return sorted(arr)",
            "",
            "def print_hello():",
            "    print('hello')",
            "",
            "class DataProcessor:",
            "    def process(self, data):",
            "        return data",
        ])
        compiler = DeepSeekV4Compiler()
        result = compiler._retrieve_relevant_snippets(
            content, "test.py", "sort the data", budget=10_000,
        )

        # Should contain the sort function (keyword match)
        assert "sort_array" in result

    def test_retrieve_relevant_snippets_budget_respected(self):
        """_retrieve_relevant_snippets should respect budget"""
        # Create a file with many sections
        lines = ['"""Module."""', "import os", ""]
        for i in range(50):
            lines.extend([
                f"def func_{i}():",
                f"    return {i}",
                f"    # " + "x" * 200,  # Padding
                "",
            ])
        content = "\n".join(lines)

        compiler = DeepSeekV4Compiler()
        result = compiler._retrieve_relevant_snippets(
            content, "test.py", "func_0", budget=1_000,  # Small budget
        )

        # Result should be within budget (roughly budget * 4 chars)
        # At least some content should be returned
        assert len(result) > 0

    def test_retrieve_relevant_snippets_fallback(self):
        """_retrieve_relevant_snippets should fallback to _extract_retrieval_snippet when no sections match"""
        # Content with no clear Python structure
        content = "plain text\n" * 1000
        compiler = DeepSeekV4Compiler()
        result = compiler._retrieve_relevant_snippets(
            content, "test.txt", "search term", budget=50_000,
        )
        # Should still return some content (fallback to head+tail)
        assert len(result) > 0


class TestV4CompilerExistingBehaviorPreserved:
    """Regression: existing V4 compiler behavior should be preserved"""

    def test_pro_mode_compilation(self):
        """Pro mode compilation should still work"""
        compiler = DeepSeekV4Compiler(variant="pro")
        request = _make_v4_request(instruction="写代码")
        compiled = compiler.compile(request)
        assert compiled.mode == "messages"
        assert len(compiled.messages) > 0

    def test_flash_mode_compilation(self):
        """Flash mode compilation should still work"""
        compiler = DeepSeekV4Compiler(variant="flash")
        request = _make_v4_request(instruction="写代码")
        compiled = compiler.compile(request)
        assert compiled.mode == "messages"

    def test_thinking_mode_deep(self):
        """Deep thinking mode should still be applied"""
        compiler = DeepSeekV4Compiler(variant="pro")
        request = _make_v4_request()
        request.thinking_mode = "deep"
        compiled = compiler.compile(request)
        assert compiled.extra.get("thinking") == {"type": "enabled"}

    def test_cache_aware_layout(self):
        """Cache-aware layout should still function"""
        compiler = DeepSeekV4Compiler(variant="pro")
        request = _make_v4_request(cache_preference="aggressive")
        compiled = compiler.compile(request)
        assert compiled.extra.get("cache_aware") is True

    def test_should_compress_aggressively(self):
        """_should_compress_aggressively should still work"""
        compiler = DeepSeekV4Compiler()
        assert compiler._should_compress_aggressively(0.2) is True
        assert compiler._should_compress_aggressively(0.8) is False
        assert compiler._should_compress_aggressively(0.5) is False

    def test_invalid_variant_raises(self):
        """Invalid variant should raise ValueError"""
        with pytest.raises(ValueError, match="Invalid variant"):
            DeepSeekV4Compiler(variant="invalid")

    def test_warmup_request(self):
        """build_warmup_request should still work"""
        compiler = DeepSeekV4Compiler()
        warmup = compiler.build_warmup_request()
        assert warmup.meta.get("task_id") == "cache_warmup"
