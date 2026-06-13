# tests/test_registry.py
"""工具注册表单元测试

测试 ToolRegistry 的注册/注销、安全元数据查询、invalidate_metadata 等。
"""

from teragent.core.types import ToolSafety
from teragent.tools.base import BaseTool, ToolResult
from teragent.tools.registry import ToolRegistry

# ===== 测试用工具 =====

class ReadOnlyTool(BaseTool):
    name = "read_file"
    description = "读取文件"
    parameters_schema = {"type": "object", "properties": {}}
    _safety = ToolSafety.READ_ONLY
    _concurrency_safe = True

    async def execute(self, params, progress_callback=None):
        return ToolResult(success=True, data={})


class WriteTool(BaseTool):
    name = "write_file"
    description = "写入文件"
    parameters_schema = {"type": "object", "properties": {}}
    _safety = ToolSafety.SAFE_WRITE
    _concurrency_safe = False

    async def execute(self, params, progress_callback=None):
        return ToolResult(success=True, data={})


class DestructiveTool(BaseTool):
    name = "delete_file"
    description = "删除文件"
    parameters_schema = {}
    _safety = ToolSafety.DESTRUCTIVE
    _concurrency_safe = False

    async def execute(self, params, progress_callback=None):
        return ToolResult(success=True, data={})


class NoNameTool(BaseTool):
    """无名称工具"""
    name = ""
    description = "无名称"
    parameters_schema = {}

    async def execute(self, params, progress_callback=None):
        return ToolResult(success=True)


class DynamicSafetyTool(BaseTool):
    """安全级别可动态变更的工具"""
    name = "dynamic_tool"
    description = "动态安全"
    parameters_schema = {}
    _safety = ToolSafety.READ_ONLY
    _concurrency_safe = True

    async def execute(self, params, progress_callback=None):
        return ToolResult(success=True, data={})


# ===== 注册/注销 =====

class TestRegisterUnregister:
    """注册与注销"""

    def test_register_tool(self):
        """注册工具后可查询"""
        reg = ToolRegistry()
        reg.register(ReadOnlyTool())
        assert reg.has_tool("read_file") is True
        assert reg.get("read_file") is not None

    def test_unregister_tool(self):
        """注销工具后不可查询"""
        reg = ToolRegistry()
        reg.register(ReadOnlyTool())
        assert reg.unregister("read_file") is True
        assert reg.has_tool("read_file") is False

    def test_unregister_nonexistent(self):
        """注销不存在的工具返回 False"""
        reg = ToolRegistry()
        assert reg.unregister("no_such_tool") is False

    def test_register_empty_name_tool_ignored(self):
        """空名称工具不注册"""
        reg = ToolRegistry()
        reg.register(NoNameTool())
        assert len(reg) == 0

    def test_batch_register(self):
        """批量注册工具"""
        reg = ToolRegistry()
        count = reg.batch_register([ReadOnlyTool(), WriteTool(), DestructiveTool()])
        assert count == 3
        assert len(reg) == 3

    def test_duplicate_registration_warning(self, caplog):
        """重复注册发出警告"""
        import logging
        reg = ToolRegistry()
        reg.register(ReadOnlyTool())
        with caplog.at_level(logging.WARNING):
            reg.register(ReadOnlyTool())  # 再次注册同名
        assert "already registered" in caplog.text

    def test_contains_check(self):
        """__contains__ 支持 'in' 操作"""
        reg = ToolRegistry()
        reg.register(ReadOnlyTool())
        assert "read_file" in reg
        assert "write_file" not in reg

    def test_len(self):
        """__len__ 返回工具数"""
        reg = ToolRegistry()
        reg.register(ReadOnlyTool())
        reg.register(WriteTool())
        assert len(reg) == 2


# ===== 安全元数据查询 =====

class TestSafetyMetadata:
    """安全元数据查询"""

    def test_get_safety_metadata(self):
        """获取工具安全元数据"""
        reg = ToolRegistry()
        reg.register(ReadOnlyTool())
        meta = reg.get_safety_metadata("read_file")
        assert meta is not None
        assert meta["safety"] == "read_only"
        assert meta["read_only"] is True
        assert meta["concurrency_safe"] is True

    def test_get_safety_metadata_nonexistent(self):
        """不存在工具的安全元数据返回 None"""
        reg = ToolRegistry()
        assert reg.get_safety_metadata("no_tool") is None

    def test_get_read_only_tools(self):
        """查询只读工具列表"""
        reg = ToolRegistry()
        reg.register(ReadOnlyTool())
        reg.register(WriteTool())
        ro = reg.get_read_only_tools()
        assert "read_file" in ro
        assert "write_file" not in ro

    def test_get_destructive_tools(self):
        """查询破坏性工具列表"""
        reg = ToolRegistry()
        reg.register(ReadOnlyTool())
        reg.register(DestructiveTool())
        destructive = reg.get_destructive_tools()
        assert "delete_file" in destructive
        assert "read_file" not in destructive

    def test_get_tools_by_safety(self):
        """按安全级别查询"""
        reg = ToolRegistry()
        reg.register(WriteTool())
        result = reg.get_tools_by_safety(ToolSafety.SAFE_WRITE)
        assert "write_file" in result

    def test_get_safety_report(self):
        """安全报告包含汇总"""
        reg = ToolRegistry()
        reg.register(ReadOnlyTool())
        reg.register(WriteTool())
        reg.register(DestructiveTool())
        report = reg.get_safety_report()
        assert report["total"] == 3
        assert "by_safety" in report
        assert "read_only" in report["by_safety"]


# ===== invalidate_metadata (L4 修复) =====

class TestInvalidateMetadata:
    """invalidate_metadata 强制刷新安全元数据缓存"""

    def test_invalidate_metadata_refreshes_cache(self):
        """工具安全属性变更后，invalidate_metadata 刷新缓存"""
        reg = ToolRegistry()
        tool = DynamicSafetyTool()
        reg.register(tool)

        # 验证初始元数据
        meta = reg.get_safety_metadata("dynamic_tool")
        assert meta["safety"] == "read_only"

        # 动态变更安全属性
        tool._safety = ToolSafety.DESTRUCTIVE
        tool._concurrency_safe = False

        # 缓存未刷新，仍是旧值
        meta_cached = reg.get_safety_metadata("dynamic_tool")
        assert meta_cached["safety"] == "read_only"

        # 调用 invalidate_metadata 刷新
        reg.invalidate_metadata("dynamic_tool")
        meta_refreshed = reg.get_safety_metadata("dynamic_tool")
        assert meta_refreshed["safety"] == "destructive"
        assert meta_refreshed["concurrency_safe"] is False

    def test_invalidate_metadata_nonexistent_tool(self, caplog):
        """刷新不存在工具的元数据发出警告"""
        import logging
        reg = ToolRegistry()
        with caplog.at_level(logging.WARNING):
            reg.invalidate_metadata("no_such_tool")
        assert "not found" in caplog.text


# ===== 列举工具 =====

class TestListTools:
    """列举工具"""

    def test_list_tool_names(self):
        """返回工具名称列表"""
        reg = ToolRegistry()
        reg.register(ReadOnlyTool())
        reg.register(WriteTool())
        names = reg.list_tool_names()
        assert "read_file" in names
        assert "write_file" in names

    def test_list_tools_function_definitions(self):
        """返回 function calling 定义"""
        reg = ToolRegistry()
        reg.register(ReadOnlyTool())
        defs = reg.list_tools()
        assert len(defs) == 1
        assert defs[0]["type"] == "function"
        assert defs[0]["function"]["name"] == "read_file"

    def test_get_tools_by_names(self):
        """按名称列表批量获取"""
        reg = ToolRegistry()
        reg.register(ReadOnlyTool())
        reg.register(WriteTool())
        tools = reg.get_tools_by_names(["read_file", "write_file", "nonexistent"])
        assert len(tools) == 2

    def test_get_summary(self):
        """返回注册表摘要"""
        reg = ToolRegistry()
        reg.register(ReadOnlyTool())
        summary = reg.get_summary()
        assert summary["total_tools"] == 1
        assert "read_file" in summary["tool_names"]

    def test_repr(self):
        """__repr__ 格式"""
        reg = ToolRegistry()
        reg.register(ReadOnlyTool())
        r = repr(reg)
        assert "read_file" in r
