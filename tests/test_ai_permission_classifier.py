# tests/test_ai_permission_classifier.py
"""AI 权限分类器单元测试

覆盖:
  - _HeuristicClassifier: 启发式分类规则（敏感路径、只读工具、破坏性工具、名称前缀推断、未知工具保守拒绝）
  - _ClassificationCache: LRU 缓存（key 生成、TTL 过期、容量淘汰）
  - _CacheEntry: 过期判断
  - _parse_llm_response: LLM 响应解析
  - _build_classification_messages: 消息构建
  - AIPermissionClassifier: 完整分类流程（缓存→LLM→启发式→置信度阈值→缓存写入）
  - 统计计数器
"""
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from teragent.security.ai_permission_classifier import (
    AIPermissionClassifier,
    _build_classification_messages,
    _CacheEntry,
    _ClassificationCache,
    _HeuristicClassifier,
    _parse_llm_response,
)
from teragent.security.permission import PermissionEffect

# ===== _HeuristicClassifier =====

class TestHeuristicClassifier:
    """启发式分类规则"""

    def test_sensitive_path_denied(self):
        """敏感路径高置信度拒绝"""
        effect, conf, reason = _HeuristicClassifier.classify(
            "write_file", {"file_path": "/etc/passwd"}
        )
        assert effect == PermissionEffect.DENY
        assert conf >= 0.90
        assert "sensitive path" in reason.lower()

    def test_env_file_denied(self):
        """环境变量文件被拒绝"""
        effect, conf, reason = _HeuristicClassifier.classify(
            "write_file", {"file_path": ".env.production"}
        )
        assert effect == PermissionEffect.DENY

    def test_ssh_path_denied(self):
        """SSH 路径被拒绝"""
        effect, conf, reason = _HeuristicClassifier.classify(
            "read_file", {"file_path": ".ssh/id_rsa"}
        )
        assert effect == PermissionEffect.DENY

    def test_git_path_denied(self):
        """Git 内部路径被拒绝"""
        effect, conf, reason = _HeuristicClassifier.classify(
            "write_file", {"file_path": ".git/config"}
        )
        assert effect == PermissionEffect.DENY

    def test_read_only_tool_allowed(self):
        """只读工具高置信度允许"""
        effect, conf, reason = _HeuristicClassifier.classify(
            "read_file", {"file_path": "/src/main.py"}
        )
        assert effect == PermissionEffect.ALLOW
        assert conf >= 0.85
        assert "read-only" in reason.lower()

    def test_list_directory_allowed(self):
        """list_directory 工具允许"""
        effect, conf, reason = _HeuristicClassifier.classify(
            "list_directory", {"path": "/project"}
        )
        assert effect == PermissionEffect.ALLOW

    def test_destructive_tool_denied(self):
        """破坏性工具拒绝"""
        effect, conf, reason = _HeuristicClassifier.classify(
            "execute_subtask", {"cmd": "rm -rf /"}
        )
        assert effect == PermissionEffect.DENY
        assert "destructive" in reason.lower()

    def test_high_risk_tool_denied(self):
        """高风险工具拒绝"""
        effect, conf, reason = _HeuristicClassifier.classify(
            "create_project", {"path": "/tmp/evil"}
        )
        assert effect == PermissionEffect.DENY
        assert "high-risk" in reason.lower()

    def test_read_prefix_tool_allowed(self):
        """read_ 前缀工具名称推断允许"""
        effect, conf, reason = _HeuristicClassifier.classify(
            "read_config", {"path": "/project/.config"}
        )
        assert effect == PermissionEffect.ALLOW
        assert "read-only" in reason.lower()

    def test_delete_prefix_tool_denied(self):
        """delete_ 前缀工具名称推断拒绝"""
        effect, conf, reason = _HeuristicClassifier.classify(
            "delete_cache", {"path": "/tmp/cache"}
        )
        assert effect == PermissionEffect.DENY
        assert "destructive" in reason.lower()

    def test_write_prefix_tool_allowed(self):
        """write_ 前缀工具名称推断允许（中等置信度）"""
        effect, conf, reason = _HeuristicClassifier.classify(
            "write_output", {"path": "/project/result.txt"}
        )
        assert effect == PermissionEffect.ALLOW
        assert 0.5 <= conf <= 0.7  # 中等置信度

    def test_write_to_sensitive_path_denied(self):
        """write_ 前缀工具写入敏感路径被拒绝"""
        effect, conf, reason = _HeuristicClassifier.classify(
            "write_file", {"file_path": "/etc/hosts"}
        )
        assert effect == PermissionEffect.DENY
        assert conf >= 0.90

    def test_unknown_tool_conservative_deny(self):
        """未知工具保守拒绝"""
        effect, conf, reason = _HeuristicClassifier.classify(
            "obscure_tool_xyz", {}
        )
        assert effect == PermissionEffect.DENY
        assert "unknown" in reason.lower()

    def test_extract_path_various_keys(self):
        """从不同参数键提取路径"""
        # file_path
        path = _HeuristicClassifier._extract_path({"file_path": "/a/b"})
        assert path == "/a/b"
        # path
        path = _HeuristicClassifier._extract_path({"path": "/c/d"})
        assert path == "/c/d"
        # 无匹配
        path = _HeuristicClassifier._extract_path({"other": "value"})
        assert path == ""


# ===== _ClassificationCache =====

class TestClassificationCache:
    """LRU 缓存"""

    def test_make_key_same_params_same_key(self):
        """相同参数产生相同 key"""
        key1 = _ClassificationCache.make_key("tool_a", {"x": 1, "y": 2})
        key2 = _ClassificationCache.make_key("tool_a", {"y": 2, "x": 1})
        assert key1 == key2

    def test_make_key_unhashable_value_converted(self):
        """不可哈希值转为 JSON 字符串"""
        key = _ClassificationCache.make_key("tool", {"data": [1, 2, 3]})
        assert key is not None  # 不应抛出异常

    def test_put_and_get(self):
        """存入和读取"""
        cache = _ClassificationCache(max_size=10, ttl=300.0)
        key = _ClassificationCache.make_key("tool", {"x": 1})
        entry = _CacheEntry(
            effect=PermissionEffect.ALLOW,
            confidence=0.9,
            reason="test",
            timestamp=time.monotonic(),
        )
        cache.put(key, entry)
        result = cache.get(key)
        assert result is not None
        assert result.effect == PermissionEffect.ALLOW

    def test_expired_entry_returns_none(self):
        """过期条目返回 None"""
        cache = _ClassificationCache(max_size=10, ttl=0.01)  # 极短 TTL
        key = _ClassificationCache.make_key("tool", {"x": 1})
        entry = _CacheEntry(
            effect=PermissionEffect.ALLOW,
            confidence=0.9,
            reason="test",
            timestamp=time.monotonic() - 1,  # 1 秒前
        )
        cache.put(key, entry)
        result = cache.get(key)
        assert result is None

    def test_lru_eviction_at_capacity(self):
        """容量满时淘汰最旧条目"""
        cache = _ClassificationCache(max_size=3, ttl=300.0)
        for i in range(5):
            key = _ClassificationCache.make_key(f"tool_{i}", {"x": i})
            cache.put(key, _CacheEntry(
                effect=PermissionEffect.ALLOW, confidence=0.9,
                reason="test", timestamp=time.monotonic(),
            ))
        assert cache.size == 3
        # 最早的两个应该已被淘汰
        key_0 = _ClassificationCache.make_key("tool_0", {"x": 0})
        assert cache.get(key_0) is None

    def test_clear_removes_all(self):
        """清空缓存"""
        cache = _ClassificationCache()
        key = _ClassificationCache.make_key("tool", {"x": 1})
        cache.put(key, _CacheEntry(
            effect=PermissionEffect.ALLOW, confidence=0.9,
            reason="test", timestamp=time.monotonic(),
        ))
        cache.clear()
        assert cache.size == 0


# ===== _parse_llm_response =====

class TestParseLLMResponse:
    """LLM 响应解析"""

    def test_valid_allow_response(self):
        """解析有效的允许响应"""
        result = _parse_llm_response('{"decision": "allow", "confidence": 0.92, "reason": "Safe op"}')
        assert result is not None
        effect, conf, reason = result
        assert effect == PermissionEffect.ALLOW
        assert conf == 0.92
        assert reason == "Safe op"

    def test_valid_deny_response(self):
        """解析有效的拒绝响应"""
        result = _parse_llm_response('{"decision": "deny", "confidence": 0.88, "reason": "Risky"}')
        assert result is not None
        effect, conf, reason = result
        assert effect == PermissionEffect.DENY

    def test_confidence_clamped(self):
        """置信度钳位到 [0.0, 1.0]"""
        result = _parse_llm_response('{"decision": "allow", "confidence": 1.5, "reason": "test"}')
        assert result is not None
        _, conf, _ = result
        assert conf == 1.0

    def test_empty_response_returns_none(self):
        """空响应返回 None"""
        assert _parse_llm_response("") is None
        assert _parse_llm_response(None) is None

    def test_malformed_json_returns_none(self):
        """畸形 JSON 返回 None"""
        assert _parse_llm_response("not json") is None
        assert _parse_llm_response('{"decision": invalid}') is None

    def test_unknown_decision_returns_none(self):
        """未知决策值返回 None"""
        assert _parse_llm_response('{"decision": "maybe", "confidence": 0.5, "reason": "test"}') is None

    def test_extra_text_around_json(self):
        """JSON 外有额外文本仍可解析"""
        result = _parse_llm_response(
            'Here is my classification:\n{"decision": "deny", "confidence": 0.75, "reason": "Unsafe"}\nDone.'
        )
        assert result is not None
        assert result[0] == PermissionEffect.DENY


# ===== AIPermissionClassifier =====

class TestAIPermissionClassifier:
    """完整分类流程"""

    @pytest.mark.asyncio
    async def test_no_model_uses_heuristic(self):
        """无模型时使用启发式"""
        clf = AIPermissionClassifier(model=None)
        effect, conf, reason = await clf.classify("read_file", {"file_path": "/src/main.py"})
        assert effect == PermissionEffect.ALLOW
        assert clf._heuristic_fallbacks == 1

    @pytest.mark.asyncio
    async def test_llm_fallback_to_heuristic_on_error(self):
        """LLM 错误时回退到启发式"""
        model = MagicMock()
        model.chat = AsyncMock(side_effect=RuntimeError("API error"))
        clf = AIPermissionClassifier(model=model)
        effect, conf, reason = await clf.classify("read_file", {"file_path": "/src/a.py"})
        # 应回退到启发式
        assert clf._heuristic_fallbacks == 1
        assert clf._llm_errors == 1

    @pytest.mark.asyncio
    async def test_llm_successful_classification(self):
        """LLM 成功分类"""
        model = MagicMock()
        model.chat = AsyncMock(return_value={
            "content": '{"decision": "deny", "confidence": 0.95, "reason": "Sensitive path"}'
        })
        clf = AIPermissionClassifier(model=model, confidence_threshold=0.8)
        effect, conf, reason = await clf.classify("write_file", {"file_path": "/etc/passwd"})
        assert effect == PermissionEffect.DENY
        assert clf._llm_calls == 1

    @pytest.mark.asyncio
    async def test_confidence_threshold_overrides_allow(self):
        """置信度低于阈值时 ALLOW 被覆盖为 DENY"""
        model = MagicMock()
        model.chat = AsyncMock(return_value={
            "content": '{"decision": "allow", "confidence": 0.5, "reason": "Maybe safe"}'
        })
        clf = AIPermissionClassifier(model=model, confidence_threshold=0.8)
        effect, conf, reason = await clf.classify("write_file", {"file_path": "/tmp/x"})
        assert effect == PermissionEffect.DENY
        assert "below threshold" in reason.lower()

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_llm_call(self):
        """缓存命中避免 LLM 调用"""
        model = MagicMock()
        model.chat = AsyncMock(return_value={
            "content": '{"decision": "allow", "confidence": 0.9, "reason": "OK"}'
        })
        clf = AIPermissionClassifier(model=model, confidence_threshold=0.8)
        # 第一次调用
        await clf.classify("read_file", {"file_path": "/src/a.py"})
        # 第二次相同调用应命中缓存
        await clf.classify("read_file", {"file_path": "/src/a.py"})
        assert clf._llm_calls == 1  # 只调用一次 LLM
        assert clf._cache_hits == 1

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """清空缓存"""
        clf = AIPermissionClassifier(model=None)
        key = _ClassificationCache.make_key("tool", {"x": 1})
        clf._cache.put(key, _CacheEntry(
            effect=PermissionEffect.ALLOW, confidence=0.9,
            reason="test", timestamp=time.monotonic(),
        ))
        clf.clear_cache()
        assert clf.cache_size == 0

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """统计信息"""
        clf = AIPermissionClassifier(model=None, confidence_threshold=0.7)
        await clf.classify("read_file", {"file_path": "/a"})
        stats = clf.get_stats()
        assert stats["total_calls"] == 1
        assert stats["heuristic_fallbacks"] == 1
        assert stats["confidence_threshold"] == 0.7

    @pytest.mark.asyncio
    async def test_reset_stats(self):
        """重置统计"""
        clf = AIPermissionClassifier(model=None)
        await clf.classify("read_file", {"file_path": "/a"})
        clf.reset_stats()
        stats = clf.get_stats()
        assert stats["total_calls"] == 0
        assert stats["heuristic_fallbacks"] == 0


# ===== _build_classification_messages =====

class TestBuildClassificationMessages:
    """消息构建"""

    def test_builds_system_and_user_messages(self):
        """生成系统和用户消息"""
        msgs = _build_classification_messages("write_file", {"path": "/a"}, "test context")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "write_file" in msgs[1]["content"]
        assert "test context" in msgs[1]["content"]

    def test_truncates_large_values(self):
        """截断大参数值"""
        large_val = "x" * 1000
        msgs = _build_classification_messages("tool", {"data": large_val}, "")
        # 参数应被截断
        user_msg = msgs[1]["content"]
        assert len(user_msg) < 2000  # 不应膨胀过大

    def test_no_context_omits_context_line(self):
        """无上下文时不添加上下文行"""
        msgs = _build_classification_messages("tool", {"x": 1}, "")
        assert "Context:" not in msgs[1]["content"]
