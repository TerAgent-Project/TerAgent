# tests/test_tools_base.py
"""工具基类与返回格式单元测试

测试 ToolResult、BaseTool 的安全级别、参数验证、注册格式等。
"""

from teragent.core.types import ToolSafety
from teragent.tools.base import BaseTool, ToolResult

# ===== 测试用具体工具 =====

class ReadOnlyTool(BaseTool):
    """只读测试工具"""
    name = "read_file"
    description = "读取文件内容"
    parameters_schema = {
        "type": "object",
        "required": ["path"],
        "properties": {
            "path": {"type": "string", "description": "文件路径"},
        },
    }
    _safety = ToolSafety.READ_ONLY
    _concurrency_safe = True

    async def execute(self, params, progress_callback=None):
        return ToolResult(success=True, data={"content": "file data"})


class DestructiveTool(BaseTool):
    """破坏性测试工具"""
    name = "execute_command"
    description = "执行命令"
    parameters_schema = {
        "type": "object",
        "required": ["command"],
        "properties": {
            "command": {"type": "string", "description": "命令"},
        },
    }
    _safety = ToolSafety.DESTRUCTIVE
    _concurrency_safe = False

    async def execute(self, params, progress_callback=None):
        return ToolResult(success=True, data={"output": "done"})


class HighRiskTool(BaseTool):
    """高风险测试工具"""
    name = "create_project"
    description = "创建项目"
    parameters_schema = {}
    _safety = ToolSafety.HIGH_RISK

    async def execute(self, params, progress_callback=None):
        return ToolResult(success=True, data={"path": "/tmp/project"})


# ===== ToolResult 测试 =====

class TestToolResult:
    """ToolResult 创建与序列化"""

    def test_success_result(self):
        """成功结果创建"""
        r = ToolResult(success=True, data={"key": "val"})
        assert r.success is True
        assert r.data == {"key": "val"}
        assert r.error == ""
        assert r.safety == ToolSafety.READ_ONLY  # 默认

    def test_failure_result(self):
        """失败结果创建"""
        r = ToolResult(success=False, error="出错了")
        assert r.success is False
        assert r.error == "出错了"
        assert r.data == {}

    def test_to_dict_serializable(self):
        """to_dict 返回可序列化字典"""
        r = ToolResult(
            success=True,
            data={"result": 42},
            safety=ToolSafety.SAFE_WRITE,
            metadata={"time": 0.5},
        )
        d = r.to_dict()
        assert d["success"] is True
        assert d["data"] == {"result": 42}
        assert d["safety"] == "safe_write"
        assert d["metadata"] == {"time": 0.5}
        # context_modifier 不可序列化，不包含
        assert "context_modifier" not in d

    def test_to_dict_with_extra_messages(self):
        """to_dict 包含 extra_messages"""
        r = ToolResult(
            success=True,
            extra_messages=[{"role": "system", "content": "警告"}],
        )
        d = r.to_dict()
        assert "extra_messages" in d
        assert len(d["extra_messages"]) == 1

    def test_to_dict_without_extra_messages(self):
        """to_dict 无 extra_messages 时不包含该键"""
        r = ToolResult(success=True)
        d = r.to_dict()
        assert "extra_messages" not in d


# ===== 安全级别检查 =====

class TestSafetyLevelChecking:
    """安全级别检查"""

    def test_read_only_tool(self):
        """只读工具属性"""
        tool = ReadOnlyTool()
        assert tool.is_read_only is True
        assert tool.is_destructive is False
        assert tool.is_concurrency_safe is True
        assert tool.safety_level == ToolSafety.READ_ONLY

    def test_destructive_tool(self):
        """破坏性工具属性"""
        tool = DestructiveTool()
        assert tool.is_read_only is False
        assert tool.is_destructive is True
        assert tool.is_concurrency_safe is False
        assert tool.safety_level == ToolSafety.DESTRUCTIVE

    def test_high_risk_tool(self):
        """高风险工具属性"""
        tool = HighRiskTool()
        assert tool.is_destructive is True  # HIGH_RISK 也算破坏性
        assert tool.safety_level == ToolSafety.HIGH_RISK

    def test_check_permissions_read_only_always_allowed(self):
        """只读工具始终允许"""
        tool = ReadOnlyTool()
        allowed, reason = tool.check_permissions({}, permission_level=0)
        assert allowed is True

    def test_check_permissions_destructive_needs_plan(self):
        """破坏性工具需要 PLAN 权限 (level >= 1)"""
        tool = DestructiveTool()
        allowed, _ = tool.check_permissions({}, permission_level=0)
        assert allowed is False
        allowed, _ = tool.check_permissions({}, permission_level=1)
        assert allowed is True

    def test_check_permissions_high_risk_needs_bypass(self):
        """高风险工具需要 BYPASS 权限 (level >= 2)"""
        tool = HighRiskTool()
        allowed, _ = tool.check_permissions({}, permission_level=1)
        assert allowed is False
        allowed, _ = tool.check_permissions({}, permission_level=2)
        assert allowed is True


# ===== 参数验证 =====

class TestParameterValidation:
    """参数验证"""

    def test_validate_params_missing_required(self):
        """缺少必填参数"""
        tool = ReadOnlyTool()
        errors = tool.validate_params({})
        assert "path" in errors

    def test_validate_params_empty_string_required(self):
        """必填参数为空字符串"""
        tool = ReadOnlyTool()
        errors = tool.validate_params({"path": "  "})
        assert "path" in errors

    def test_validate_params_all_present(self):
        """所有必填参数存在"""
        tool = ReadOnlyTool()
        errors = tool.validate_params({"path": "/tmp/test.py"})
        assert errors == []

    def test_validate_params_no_schema(self):
        """无 schema 时不报错"""
        tool = HighRiskTool()
        errors = tool.validate_params({"any": "thing"})
        assert errors == []

    def test_validate_input_calls_validate_params(self):
        """validate_input 默认调用 validate_params"""
        tool = ReadOnlyTool()
        errors = tool.validate_input({})
        assert "path" in errors


# ===== 注册格式 =====

class TestRegistrationFormats:
    """to_function_definition / to_registry_metadata 输出格式"""

    def test_to_function_definition(self):
        """to_function_definition 输出符合 OpenAI 格式"""
        tool = ReadOnlyTool()
        fd = tool.to_function_definition()
        assert fd["type"] == "function"
        assert fd["function"]["name"] == "read_file"
        assert fd["function"]["description"] == "读取文件内容"
        assert "path" in fd["function"]["parameters"]["properties"]

    def test_to_registry_metadata(self):
        """to_registry_metadata 包含安全属性"""
        tool = ReadOnlyTool()
        meta = tool.to_registry_metadata()
        assert meta["name"] == "read_file"
        assert meta["safety"] == "read_only"
        assert meta["read_only"] is True
        assert meta["destructive"] is False
        assert meta["concurrency_safe"] is True

    def test_repr_format(self):
        """__repr__ 包含名称和安全级别"""
        tool = DestructiveTool()
        r = repr(tool)
        assert "execute_command" in r
        assert "destructive" in r
