# tests/test_openai_adapter.py
"""OpenAICompatibleAdapter 单元测试

覆盖:
  - __init__ 参数存储 (base_url, api_key, enable_fake_tools, extra_headers)
  - enable_fake_tools 默认值为 False (向后兼容)
  - _build_headers() 包含 Authorization 头
  - _safe_get_choice() 处理空/畸形数据
  - _categorize_error() 异常分类
  - required_mode 属性返回 "messages"
  - capabilities 属性
  - detect_fake_tool_call 假工具检测 (假工具名 vs 真实工具名)
  - FAKE_TOOLS 和 FAKE_TOOL_NAMES 常量
  - 连接池管理 (_get_client 创建客户端, close 销毁)
  - __del__ 方法行为
  - base_url 尾部斜杠去除
"""
import httpx
import pytest

from teragent.core.adapters.openai_compatible import (
    FAKE_TOOL_NAMES,
    FAKE_TOOLS,
    OpenAICompatibleAdapter,
    detect_fake_tool_call,
)

# ===== __init__ 参数存储 =====

class TestOpenAIInit:
    """初始化参数存储"""

    def test_init_stores_base_url(self):
        """base_url 存储并去除尾部斜杠"""
        adapter = OpenAICompatibleAdapter(base_url="https://api.example.com/v1/", api_key="sk-test")
        assert adapter.base_url == "https://api.example.com/v1"

    def test_init_stores_api_key(self):
        """api_key 存储"""
        adapter = OpenAICompatibleAdapter(base_url="https://api.example.com", api_key="sk-abc123")
        assert adapter.api_key == "sk-abc123"

    def test_init_stores_enable_fake_tools_true(self):
        """enable_fake_tools=True 存储"""
        adapter = OpenAICompatibleAdapter(
            base_url="https://api.example.com", api_key="sk-test", enable_fake_tools=True
        )
        assert adapter._enable_fake_tools is True

    def test_enable_fake_tools_defaults_to_false(self):
        """enable_fake_tools 默认值为 False (向后兼容)"""
        adapter = OpenAICompatibleAdapter(base_url="https://api.example.com", api_key="sk-test")
        assert adapter._enable_fake_tools is False

    def test_init_stores_extra_headers(self):
        """extra_headers 存储"""
        extra = {"X-Custom-Header": "value"}
        adapter = OpenAICompatibleAdapter(
            base_url="https://api.example.com", api_key="sk-test", extra_headers=extra
        )
        assert adapter._extra_headers == {"X-Custom-Header": "value"}

    def test_init_http_client_none(self):
        """初始化时 _http_client 为 None (延迟创建)"""
        adapter = OpenAICompatibleAdapter(base_url="https://api.example.com", api_key="sk-test")
        assert adapter._http_client is None


# ===== _build_headers =====

class TestOpenAIBuildHeaders:
    """请求头构建"""

    def test_build_headers_includes_authorization(self):
        """_build_headers 包含 Authorization 头"""
        adapter = OpenAICompatibleAdapter(base_url="https://api.example.com", api_key="sk-test")
        headers = adapter._build_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer sk-test"

    def test_build_headers_includes_content_type(self):
        """_build_headers 包含 Content-Type"""
        adapter = OpenAICompatibleAdapter(base_url="https://api.example.com", api_key="sk-test")
        headers = adapter._build_headers()
        assert headers["Content-Type"] == "application/json"

    def test_build_headers_no_auth_when_empty_key(self):
        """api_key 为空时不包含 Authorization 头"""
        adapter = OpenAICompatibleAdapter(base_url="https://api.example.com", api_key="")
        headers = adapter._build_headers()
        assert "Authorization" not in headers

    def test_build_headers_merges_extra_headers(self):
        """extra_headers 合并到请求头"""
        adapter = OpenAICompatibleAdapter(
            base_url="https://api.example.com",
            api_key="sk-test",
            extra_headers={"X-Trace-Id": "abc"},
        )
        headers = adapter._build_headers()
        assert headers["X-Trace-Id"] == "abc"
        assert headers["Authorization"] == "Bearer sk-test"


# ===== _safe_get_choice =====

class TestOpenAISafeGetChoice:
    """安全提取 choice"""

    def test_normal_data(self):
        """正常数据提取"""
        chunk = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}]}
        result = OpenAICompatibleAdapter._safe_get_choice(chunk)
        assert result["message"]["content"] == "hi"

    def test_empty_choices(self):
        """choices 为空列表"""
        chunk = {"choices": []}
        result = OpenAICompatibleAdapter._safe_get_choice(chunk)
        assert result == {}

    def test_missing_choices(self):
        """无 choices 键"""
        chunk = {"id": "chatcmpl-123"}
        result = OpenAICompatibleAdapter._safe_get_choice(chunk)
        assert result == {}

    def test_choices_not_list(self):
        """choices 不是列表"""
        chunk = {"choices": "not_a_list"}
        result = OpenAICompatibleAdapter._safe_get_choice(chunk)
        assert result == {}

    def test_index_out_of_range(self):
        """索引超出范围"""
        chunk = {"choices": [{"index": 0}]}
        result = OpenAICompatibleAdapter._safe_get_choice(chunk, index=5)
        assert result == {}

    def test_none_choice(self):
        """choice 为 None"""
        chunk = {"choices": [None]}
        result = OpenAICompatibleAdapter._safe_get_choice(chunk)
        assert result == {}


# ===== _categorize_error =====

class TestOpenAICategorizeError:
    """异常分类"""

    def _make_http_error(self, status_code: int) -> httpx.HTTPStatusError:
        """构造 httpx.HTTPStatusError"""
        request = httpx.Request("POST", "https://api.example.com/chat/completions")
        response = httpx.Response(status_code=status_code, request=request)
        return httpx.HTTPStatusError(message="error", request=request, response=response)

    def test_rate_limited(self):
        """429 → rate_limited"""
        exc = self._make_http_error(429)
        assert OpenAICompatibleAdapter._categorize_error(exc) == "rate_limited"

    def test_authentication(self):
        """401 → authentication"""
        exc = self._make_http_error(401)
        assert OpenAICompatibleAdapter._categorize_error(exc) == "authentication"

    def test_forbidden(self):
        """403 → forbidden"""
        exc = self._make_http_error(403)
        assert OpenAICompatibleAdapter._categorize_error(exc) == "forbidden"

    def test_server_error(self):
        """500 → server_error"""
        exc = self._make_http_error(500)
        assert OpenAICompatibleAdapter._categorize_error(exc) == "server_error"

    def test_client_error(self):
        """400 → client_error (非 401/403/429)"""
        exc = self._make_http_error(400)
        assert OpenAICompatibleAdapter._categorize_error(exc) == "client_error"

    def test_timeout(self):
        """TimeoutException → timeout"""
        exc = httpx.TimeoutException("timeout")
        assert OpenAICompatibleAdapter._categorize_error(exc) == "timeout"

    def test_connection(self):
        """ConnectError → connection"""
        exc = httpx.ConnectError("connection refused")
        assert OpenAICompatibleAdapter._categorize_error(exc) == "connection"

    def test_unknown(self):
        """其他异常 → unknown"""
        exc = ValueError("some error")
        assert OpenAICompatibleAdapter._categorize_error(exc) == "unknown"


# ===== capabilities & required_mode =====

class TestOpenAICapabilities:
    """能力和模式属性"""

    def test_required_mode_returns_messages(self):
        """required_mode 返回 'messages'"""
        adapter = OpenAICompatibleAdapter(base_url="https://api.example.com", api_key="sk-test")
        assert adapter.required_mode == "messages"

    def test_capabilities_contains_streaming(self):
        """capabilities 包含 streaming"""
        adapter = OpenAICompatibleAdapter(base_url="https://api.example.com", api_key="sk-test")
        assert adapter.capabilities["streaming"] is True

    def test_capabilities_contains_tool_calling(self):
        """capabilities 包含 tool_calling"""
        adapter = OpenAICompatibleAdapter(base_url="https://api.example.com", api_key="sk-test")
        assert adapter.capabilities["tool_calling"] is True

    def test_capabilities_contains_max_context_tokens(self):
        """capabilities 包含 max_context_tokens (upgraded to 1M for V4/M3 support)"""
        adapter = OpenAICompatibleAdapter(base_url="https://api.example.com", api_key="sk-test")
        assert adapter.capabilities["max_context_tokens"] == 1_000_000


# ===== FAKE_TOOLS & FAKE_TOOL_NAMES =====

class TestOpenAIFakeToolsConstants:
    """假工具常量"""

    def test_fake_tools_is_list_of_dicts(self):
        """FAKE_TOOLS 是字典列表"""
        assert isinstance(FAKE_TOOLS, list)
        assert len(FAKE_TOOLS) > 0
        for tool in FAKE_TOOLS:
            assert isinstance(tool, dict)
            assert tool.get("type") == "function"
            assert "function" in tool

    def test_fake_tool_names_is_set(self):
        """FAKE_TOOL_NAMES 是集合"""
        assert isinstance(FAKE_TOOL_NAMES, set)
        assert len(FAKE_TOOL_NAMES) == len(FAKE_TOOLS)

    def test_fake_tool_names_match(self):
        """FAKE_TOOL_NAMES 与 FAKE_TOOLS 中的名称一致"""
        expected = {t["function"]["name"] for t in FAKE_TOOLS}
        assert FAKE_TOOL_NAMES == expected

    def test_fake_tools_have_deceptive_names(self):
        """假工具包含具有欺骗性的名称"""
        names = FAKE_TOOL_NAMES
        assert "internal_model_profiling_snapshot" in names
        assert "_system_diagnostic_dump" in names
        assert "export_training_data" in names


# ===== detect_fake_tool_call =====

class TestOpenAIDetectFakeToolCall:
    """假工具调用检测"""

    def test_detects_fake_tool_call(self):
        """检测到假工具调用"""
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "internal_model_profiling_snapshot",
                                    "arguments": "{}",
                                }
                            }
                        ]
                    }
                }
            ]
        }
        assert detect_fake_tool_call(response) is True

    def test_no_fake_tool_call_with_real_tool(self):
        """真实工具名不触发检测"""
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "/tmp/test.py"}',
                                }
                            }
                        ]
                    }
                }
            ]
        }
        assert detect_fake_tool_call(response) is False

    def test_no_tool_calls(self):
        """无 tool_calls 不触发检测"""
        response = {"choices": [{"message": {"content": "Hello"}}]}
        assert detect_fake_tool_call(response) is False

    def test_empty_choices(self):
        """空 choices 不触发检测"""
        assert detect_fake_tool_call({"choices": []}) is False

    def test_missing_choices(self):
        """无 choices 键不触发检测"""
        assert detect_fake_tool_call({}) is False

    def test_detects_second_fake_tool(self):
        """检测到第二个假工具名"""
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "export_training_data", "arguments": "{}"}}
                        ]
                    }
                }
            ]
        }
        assert detect_fake_tool_call(response) is True


# ===== 连接池管理 =====

class TestOpenAIConnectionPool:
    """连接池管理"""

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self):
        """_get_client 创建 httpx.AsyncClient"""
        adapter = OpenAICompatibleAdapter(base_url="https://api.example.com", api_key="sk-test")
        assert adapter._http_client is None
        client = await adapter._get_client()
        assert isinstance(client, httpx.AsyncClient)
        assert adapter._http_client is client
        await adapter.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing(self):
        """_get_client 复用已有客户端"""
        adapter = OpenAICompatibleAdapter(base_url="https://api.example.com", api_key="sk-test")
        client1 = await adapter._get_client()
        client2 = await adapter._get_client()
        assert client1 is client2
        await adapter.close()

    @pytest.mark.asyncio
    async def test_close_destroys_client(self):
        """close 销毁客户端"""
        adapter = OpenAICompatibleAdapter(base_url="https://api.example.com", api_key="sk-test")
        await adapter._get_client()
        assert adapter._http_client is not None
        await adapter.close()
        assert adapter._http_client is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        """close 可安全重复调用"""
        adapter = OpenAICompatibleAdapter(base_url="https://api.example.com", api_key="sk-test")
        await adapter._get_client()
        await adapter.close()
        await adapter.close()  # 不应抛出异常
        assert adapter._http_client is None


# ===== __del__ =====

class TestOpenAIDel:
    """__del__ 方法行为"""

    def test_del_with_no_client(self):
        """无客户端时 __del__ 不抛异常"""
        adapter = OpenAICompatibleAdapter(base_url="https://api.example.com", api_key="sk-test")
        adapter.__del__()  # 不应抛出异常

    def test_del_sets_client_to_none(self):
        """__del__ 后 _http_client 为 None"""
        adapter = OpenAICompatibleAdapter(base_url="https://api.example.com", api_key="sk-test")
        # 手动设置一个已关闭的客户端
        adapter._http_client = None
        adapter.__del__()
        assert adapter._http_client is None
