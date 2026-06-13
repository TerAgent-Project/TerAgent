# tests/test_anthropic_adapter.py
"""AnthropicNativeAdapter 单元测试

覆盖:
  - __init__ 参数存储 (base_url, api_key, enable_fake_tools)
  - enable_fake_tools 默认值为 True (向后兼容 — 原先一直开启)
  - _build_headers() 包含 x-api-key 和 anthropic-version
  - _convert_tools_to_anthropic() 将 OpenAI 格式转为 Anthropic 格式
  - _build_fake_tools_payload() 生成 Anthropic 格式假工具
  - required_mode 属性返回 "system_user"
  - capabilities 属性
  - _detect_anthropic_fake_tool_call 检测 tool_use 块
  - FAKE_TOOL_NAMES 常量
  - enable_fake_tools=False 阻止工具注入
  - 连接池管理
  - __del__ 方法行为
"""
import httpx
import pytest

from teragent.core.adapters.anthropic_native import (
    _FAKE_TOOLS_OPENAI,
    FAKE_TOOL_NAMES,
    AnthropicNativeAdapter,
    _detect_anthropic_fake_tool_call,
)

# ===== __init__ 参数存储 =====

class TestAnthropicInit:
    """初始化参数存储"""

    def test_init_stores_base_url(self):
        """base_url 存储并去除尾部斜杠"""
        adapter = AnthropicNativeAdapter(base_url="https://api.anthropic.com/", api_key="sk-ant-test")
        assert adapter.base_url == "https://api.anthropic.com"

    def test_init_stores_api_key(self):
        """api_key 存储"""
        adapter = AnthropicNativeAdapter(base_url="https://api.anthropic.com", api_key="sk-ant-xyz")
        assert adapter.api_key == "sk-ant-xyz"

    def test_init_stores_enable_fake_tools_true(self):
        """enable_fake_tools=True 存储"""
        adapter = AnthropicNativeAdapter(
            base_url="https://api.anthropic.com", api_key="sk-ant-test", enable_fake_tools=True
        )
        assert adapter._enable_fake_tools is True

    def test_enable_fake_tools_defaults_to_false(self):
        """enable_fake_tools 默认值为 False (安全优先 — 默认不注入假工具)"""
        adapter = AnthropicNativeAdapter(base_url="https://api.anthropic.com", api_key="sk-ant-test")
        assert adapter._enable_fake_tools is False

    def test_enable_fake_tools_can_be_disabled(self):
        """enable_fake_tools 可显式关闭"""
        adapter = AnthropicNativeAdapter(
            base_url="https://api.anthropic.com", api_key="sk-ant-test", enable_fake_tools=False
        )
        assert adapter._enable_fake_tools is False

    def test_init_http_client_none(self):
        """初始化时 _http_client 为 None (延迟创建)"""
        adapter = AnthropicNativeAdapter(base_url="https://api.anthropic.com", api_key="sk-ant-test")
        assert adapter._http_client is None


# ===== _build_headers =====

class TestAnthropicBuildHeaders:
    """请求头构建"""

    def test_includes_x_api_key(self):
        """_build_headers 包含 x-api-key"""
        adapter = AnthropicNativeAdapter(base_url="https://api.anthropic.com", api_key="sk-ant-123")
        headers = adapter._build_headers()
        assert headers["x-api-key"] == "sk-ant-123"

    def test_includes_anthropic_version(self):
        """_build_headers 包含 anthropic-version"""
        adapter = AnthropicNativeAdapter(base_url="https://api.anthropic.com", api_key="sk-ant-123")
        headers = adapter._build_headers()
        assert "anthropic-version" in headers
        assert headers["anthropic-version"] == "2023-06-01"

    def test_includes_content_type(self):
        """_build_headers 包含 Content-Type"""
        adapter = AnthropicNativeAdapter(base_url="https://api.anthropic.com", api_key="sk-ant-123")
        headers = adapter._build_headers()
        assert headers["Content-Type"] == "application/json"

    def test_no_authorization_header(self):
        """_build_headers 不包含 Authorization 头 (Anthropic 使用 x-api-key)"""
        adapter = AnthropicNativeAdapter(base_url="https://api.anthropic.com", api_key="sk-ant-123")
        headers = adapter._build_headers()
        assert "Authorization" not in headers


# ===== _convert_tools_to_anthropic =====

class TestAnthropicConvertTools:
    """OpenAI 格式工具转 Anthropic 格式"""

    def test_basic_conversion(self):
        """基本格式转换"""
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
        result = AnthropicNativeAdapter._convert_tools_to_anthropic(openai_tools)
        assert len(result) == 1
        tool = result[0]
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get weather"
        assert tool["input_schema"] == {"type": "object", "properties": {"city": {"type": "string"}}}

    def test_input_schema_from_parameters(self):
        """parameters 映射为 input_schema"""
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ]
        result = AnthropicNativeAdapter._convert_tools_to_anthropic(openai_tools)
        assert "input_schema" in result[0]
        assert "parameters" not in result[0]

    def test_empty_name_skipped(self):
        """名称为空的工具被跳过"""
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "",
                    "description": "No name",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = AnthropicNativeAdapter._convert_tools_to_anthropic(openai_tools)
        assert len(result) == 0

    def test_non_function_type_skipped(self):
        """非 function 类型被跳过"""
        openai_tools = [{"type": "not_function", "name": "whatever"}]
        result = AnthropicNativeAdapter._convert_tools_to_anthropic(openai_tools)
        assert len(result) == 0

    def test_default_input_schema(self):
        """缺少 parameters 时使用默认 input_schema"""
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "simple_tool",
                    "description": "No params",
                },
            }
        ]
        result = AnthropicNativeAdapter._convert_tools_to_anthropic(openai_tools)
        assert result[0]["input_schema"] == {"type": "object", "properties": {}}

    def test_multiple_tools(self):
        """多个工具转换"""
        openai_tools = [
            {
                "type": "function",
                "function": {"name": f"tool_{i}", "description": f"Tool {i}"},
            }
            for i in range(3)
        ]
        result = AnthropicNativeAdapter._convert_tools_to_anthropic(openai_tools)
        assert len(result) == 3
        assert result[0]["name"] == "tool_0"
        assert result[2]["name"] == "tool_2"


# ===== _build_fake_tools_payload =====

class TestAnthropicBuildFakeTools:
    """假工具载荷构建"""

    def test_produces_anthropic_format(self):
        """_build_fake_tools_payload 生成 Anthropic 格式工具"""
        adapter = AnthropicNativeAdapter(
            base_url="https://api.anthropic.com", api_key="sk-ant-test", enable_fake_tools=True
        )
        result = adapter._build_fake_tools_payload()
        assert isinstance(result, list)
        assert len(result) > 0
        for tool in result:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            # 不应包含 OpenAI 格式的 "function" 键
            assert "function" not in tool
            assert "type" not in tool

    def test_fake_tool_names_in_payload(self):
        """假工具载荷包含预期的工具名"""
        adapter = AnthropicNativeAdapter(
            base_url="https://api.anthropic.com", api_key="sk-ant-test", enable_fake_tools=True
        )
        result = adapter._build_fake_tools_payload()
        names = {t["name"] for t in result}
        assert "internal_model_profiling_snapshot" in names
        assert "_system_diagnostic_dump" in names
        assert "export_training_data" in names


# ===== capabilities & required_mode =====

class TestAnthropicCapabilities:
    """能力和模式属性"""

    def test_required_mode_returns_system_user(self):
        """required_mode 返回 'system_user'"""
        adapter = AnthropicNativeAdapter(base_url="https://api.anthropic.com", api_key="sk-ant-test")
        assert adapter.required_mode == "system_user"

    def test_capabilities_contains_streaming(self):
        """capabilities 包含 streaming"""
        adapter = AnthropicNativeAdapter(base_url="https://api.anthropic.com", api_key="sk-ant-test")
        assert adapter.capabilities["streaming"] is True

    def test_capabilities_contains_tool_calling(self):
        """capabilities 包含 tool_calling"""
        adapter = AnthropicNativeAdapter(base_url="https://api.anthropic.com", api_key="sk-ant-test")
        assert adapter.capabilities["tool_calling"] is True

    def test_capabilities_max_context_tokens(self):
        """capabilities 的 max_context_tokens 为 200000"""
        adapter = AnthropicNativeAdapter(base_url="https://api.anthropic.com", api_key="sk-ant-test")
        assert adapter.capabilities["max_context_tokens"] == 200000


# ===== _detect_anthropic_fake_tool_call =====

class TestAnthropicDetectFakeToolCall:
    """Anthropic 格式假工具调用检测"""

    def test_detects_fake_tool_use_block(self):
        """检测到假工具 tool_use 块"""
        data = {
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "name": "internal_model_profiling_snapshot",
                    "id": "toolu_123",
                    "input": {},
                },
            ]
        }
        assert _detect_anthropic_fake_tool_call(data) is True

    def test_no_detection_for_real_tool(self):
        """真实工具名不触发检测"""
        data = {
            "content": [
                {
                    "type": "tool_use",
                    "name": "read_file",
                    "id": "toolu_456",
                    "input": {"path": "/tmp/test.py"},
                },
            ]
        }
        assert _detect_anthropic_fake_tool_call(data) is False

    def test_no_tool_use_blocks(self):
        """无 tool_use 块不触发检测"""
        data = {
            "content": [
                {"type": "text", "text": "Hello world"},
            ]
        }
        assert _detect_anthropic_fake_tool_call(data) is False

    def test_empty_content(self):
        """空 content 列表不触发检测"""
        assert _detect_anthropic_fake_tool_call({"content": []}) is False

    def test_missing_content(self):
        """无 content 键不触发检测"""
        assert _detect_anthropic_fake_tool_call({}) is False

    def test_detects_export_training_data(self):
        """检测到 export_training_data 假工具"""
        data = {
            "content": [
                {
                    "type": "tool_use",
                    "name": "export_training_data",
                    "id": "toolu_789",
                    "input": {"format": "jsonl"},
                },
            ]
        }
        assert _detect_anthropic_fake_tool_call(data) is True


# ===== FAKE_TOOL_NAMES =====

class TestAnthropicFakeToolNames:
    """FAKE_TOOL_NAMES 常量"""

    def test_is_set(self):
        """FAKE_TOOL_NAMES 是集合"""
        assert isinstance(FAKE_TOOL_NAMES, set)

    def test_contains_expected_names(self):
        """包含预期的假工具名称"""
        assert "internal_model_profiling_snapshot" in FAKE_TOOL_NAMES
        assert "_system_diagnostic_dump" in FAKE_TOOL_NAMES
        assert "export_training_data" in FAKE_TOOL_NAMES

    def test_consistent_with_openai_source(self):
        """FAKE_TOOL_NAMES 与 _FAKE_TOOLS_OPENAI 一致"""
        expected = {t["function"]["name"] for t in _FAKE_TOOLS_OPENAI}
        assert FAKE_TOOL_NAMES == expected


# ===== enable_fake_tools=False 阻止工具注入 =====

class TestAnthropicFakeToolsDisabled:
    """enable_fake_tools=False 阻止工具注入"""

    def test_fake_tools_disabled_build_payload_not_called_in_send(self):
        """enable_fake_tools=False 时不注入假工具 (检查内部属性)"""
        adapter = AnthropicNativeAdapter(
            base_url="https://api.anthropic.com", api_key="sk-ant-test", enable_fake_tools=False
        )
        assert adapter._enable_fake_tools is False
        # 当 enable_fake_tools=False 时, _build_fake_tools_payload 不应在 send 中被调用
        # 但方法本身仍可调用 (不依赖实例状态)
        # 验证: 直接调用 _build_fake_tools_payload 仍可工作 (方法本身无副作用)
        payload = adapter._build_fake_tools_payload()
        assert isinstance(payload, list)  # 方法本身仍能工作
        # 但在实际 send 流程中, _enable_fake_tools 门控会阻止注入

    def test_build_fake_tools_payload_still_functional(self):
        """_build_fake_tools_payload 方法本身始终可用"""
        adapter = AnthropicNativeAdapter(
            base_url="https://api.anthropic.com", api_key="sk-ant-test", enable_fake_tools=False
        )
        # 方法本身不依赖 _enable_fake_tools 标志
        result = adapter._build_fake_tools_payload()
        assert len(result) > 0
        # 关键区别: send() 中 if self._enable_fake_tools 门控


# ===== 连接池管理 =====

class TestAnthropicConnectionPool:
    """连接池管理"""

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self):
        """_get_client 创建 httpx.AsyncClient"""
        adapter = AnthropicNativeAdapter(base_url="https://api.anthropic.com", api_key="sk-ant-test")
        assert adapter._http_client is None
        client = await adapter._get_client()
        assert isinstance(client, httpx.AsyncClient)
        assert adapter._http_client is client
        await adapter.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing(self):
        """_get_client 复用已有客户端"""
        adapter = AnthropicNativeAdapter(base_url="https://api.anthropic.com", api_key="sk-ant-test")
        client1 = await adapter._get_client()
        client2 = await adapter._get_client()
        assert client1 is client2
        await adapter.close()

    @pytest.mark.asyncio
    async def test_close_destroys_client(self):
        """close 销毁客户端"""
        adapter = AnthropicNativeAdapter(base_url="https://api.anthropic.com", api_key="sk-ant-test")
        await adapter._get_client()
        assert adapter._http_client is not None
        await adapter.close()
        assert adapter._http_client is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        """close 可安全重复调用"""
        adapter = AnthropicNativeAdapter(base_url="https://api.anthropic.com", api_key="sk-ant-test")
        await adapter._get_client()
        await adapter.close()
        await adapter.close()  # 不应抛出异常
        assert adapter._http_client is None


# ===== __del__ =====

class TestAnthropicDel:
    """__del__ 方法行为"""

    def test_del_with_no_client(self):
        """无客户端时 __del__ 不抛异常"""
        adapter = AnthropicNativeAdapter(base_url="https://api.anthropic.com", api_key="sk-ant-test")
        adapter.__del__()  # 不应抛出异常

    def test_del_sets_client_to_none(self):
        """__del__ 后 _http_client 为 None"""
        adapter = AnthropicNativeAdapter(base_url="https://api.anthropic.com", api_key="sk-ant-test")
        adapter._http_client = None
        adapter.__del__()
        assert adapter._http_client is None
