# tests/test_desktop_tool.py
"""桌面操作工具单元测试

覆盖:
  - DesktopTool 基础属性（安全级别、参数 schema、注册格式）
  - 安全机制（安全区域、阻止快捷键、频率限制、连续操作限制、屏幕边界）
  - 各操作方法（screenshot/click/type_text/scroll/hotkey/move_mouse/drag）
  - validate_input 输入验证
  - capture_desktop_context 上下文捕获
  - DesktopSafetyConfig 配置
  - register_desktop_tool 注册便捷函数
"""
import time

import pytest

from teragent.core.tap import DesktopContext, MultimodalContent
from teragent.core.types import ToolSafety
from teragent.tools.desktop import (
    _HAS_MSS,
    _HAS_PIL,
    _HAS_PYAUTOGUI,
    DesktopSafetyConfig,
    DesktopTool,
    register_desktop_tool,
)
from teragent.tools.registry import ToolRegistry

# ===== 基础属性 =====

class TestDesktopToolBasics:
    """DesktopTool 基础属性测试"""

    def test_name_and_description(self):
        """工具名称和描述"""
        tool = DesktopTool()
        assert tool.name == "desktop"
        assert "桌面操作" in tool.description

    def test_safety_level(self):
        """安全级别为 DESTRUCTIVE"""
        tool = DesktopTool()
        assert tool.safety_level == ToolSafety.DESTRUCTIVE
        assert tool.is_destructive is True
        assert tool.is_read_only is False

    def test_concurrency_safe(self):
        """桌面操作不可并行"""
        tool = DesktopTool()
        assert tool.is_concurrency_safe is False

    def test_parameters_schema_actions(self):
        """参数 schema 包含所有操作类型"""
        tool = DesktopTool()
        actions = tool.parameters_schema["properties"]["action"]["enum"]
        expected = ["screenshot", "click", "type_text", "scroll", "hotkey", "move_mouse", "drag"]
        assert set(actions) == set(expected)

    def test_parameters_schema_required(self):
        """action 为必填参数"""
        tool = DesktopTool()
        assert "action" in tool.parameters_schema["required"]

    def test_to_function_definition(self):
        """转换为 OpenAI function calling 格式"""
        tool = DesktopTool()
        fd = tool.to_function_definition()
        assert fd["type"] == "function"
        assert fd["function"]["name"] == "desktop"
        assert "properties" in fd["function"]["parameters"]

    def test_to_registry_metadata(self):
        """注册元数据包含安全属性"""
        tool = DesktopTool()
        meta = tool.to_registry_metadata()
        assert meta["name"] == "desktop"
        assert meta["safety"] == "destructive"
        assert meta["destructive"] is True
        assert meta["concurrency_safe"] is False

    def test_repr(self):
        """__repr__ 包含名称和安全级别"""
        tool = DesktopTool()
        r = repr(tool)
        assert "desktop" in r
        assert "destructive" in r

    def test_get_tool_prompt(self):
        """工具提示不为空"""
        tool = DesktopTool()
        prompt = tool.get_tool_prompt()
        assert len(prompt) > 0
        assert "桌面操作" in prompt

    def test_describe_usage(self):
        """describe_usage 返回有意义的描述"""
        tool = DesktopTool()
        assert "点击" in tool.describe_usage({"action": "click", "x": 100, "y": 200})
        assert "截图" in tool.describe_usage({"action": "screenshot"})
        assert "滚动" in tool.describe_usage({"action": "scroll", "direction": "down"})
        assert "快捷键" in tool.describe_usage({"action": "hotkey", "keys": "ctrl,c"})
        assert "拖拽" in tool.describe_usage({"action": "drag", "x": 0, "y": 0, "end_x": 100, "end_y": 100})
        assert "移动鼠标" in tool.describe_usage({"action": "move_mouse", "x": 100, "y": 200})
        assert "输入文本" in tool.describe_usage({"action": "type_text", "text": "hello"})


# ===== 安全配置 =====

class TestDesktopSafetyConfig:
    """DesktopSafetyConfig 测试"""

    def test_default_config(self):
        """默认安全配置"""
        config = DesktopSafetyConfig()
        assert config.min_interval == 0.5
        assert config.max_consecutive_ops == 50
        assert config.screenshot_quality == 75
        assert config.screenshot_format == "jpeg"
        assert len(config.safe_zones) == 0
        assert len(config.blocked_shortcuts) > 0

    def test_custom_config(self):
        """自定义安全配置"""
        config = DesktopSafetyConfig(
            safe_zones=[(0, 0, 100, 100)],
            min_interval=2.0,
            max_consecutive_ops=10,
            screenshot_quality=90,
            screenshot_format="png",
        )
        assert len(config.safe_zones) == 1
        assert config.min_interval == 2.0
        assert config.max_consecutive_ops == 10
        assert config.screenshot_quality == 90
        assert config.screenshot_format == "png"

    def test_blocked_shortcuts_include_dangerous(self):
        """默认阻止列表包含危险快捷键"""
        config = DesktopSafetyConfig()
        # Alt+F4 应被阻止
        assert frozenset({"alt", "f4"}) in config.blocked_shortcuts
        # Ctrl+Alt+Delete 应被阻止
        assert frozenset({"ctrl", "alt", "delete"}) in config.blocked_shortcuts


# ===== 输入验证 =====

class TestInputValidation:
    """输入验证测试"""

    def test_validate_screenshot_no_extra_params(self):
        """screenshot 不需要额外参数"""
        tool = DesktopTool()
        errors = tool.validate_input({"action": "screenshot"})
        assert errors == []

    def test_validate_click_needs_coordinates(self):
        """click 需要 x 和 y"""
        tool = DesktopTool()
        errors = tool.validate_input({"action": "click"})
        assert any("x" in e for e in errors)

    def test_validate_drag_needs_start_and_end(self):
        """drag 需要起点和终点坐标"""
        tool = DesktopTool()
        errors = tool.validate_input({"action": "drag"})
        assert any("x" in e for e in errors)
        assert any("end_x" in e or "end_y" in e for e in errors)

    def test_validate_type_text_needs_text(self):
        """type_text 需要 text"""
        tool = DesktopTool()
        errors = tool.validate_input({"action": "type_text"})
        assert any("text" in e for e in errors)

    def test_validate_hotkey_needs_keys(self):
        """hotkey 需要 keys"""
        tool = DesktopTool()
        errors = tool.validate_input({"action": "hotkey"})
        assert any("keys" in e for e in errors)

    def test_validate_scroll_needs_direction(self):
        """scroll 需要 direction"""
        tool = DesktopTool()
        errors = tool.validate_input({"action": "scroll"})
        assert any("direction" in e for e in errors)

    def test_validate_move_mouse_needs_coordinates(self):
        """move_mouse 需要 x 和 y"""
        tool = DesktopTool()
        errors = tool.validate_input({"action": "move_mouse"})
        assert any("x" in e for e in errors)

    def test_validate_missing_action(self):
        """缺少 action 参数"""
        tool = DesktopTool()
        errors = tool.validate_input({})
        assert "action" in errors


# ===== 安全检查 =====

class TestSafetyChecks:
    """安全检查测试"""

    def test_safe_zone_blocks_click(self):
        """安全区域阻止点击"""
        config = DesktopSafetyConfig(safe_zones=[(0, 0, 100, 100)])
        tool = DesktopTool(safety_config=config)
        error = tool._pre_safety_check({"action": "click", "x": 50, "y": 50})
        assert "安全区域" in error

    def test_safe_zone_allows_outside(self):
        """安全区域外允许点击"""
        config = DesktopSafetyConfig(safe_zones=[(0, 0, 100, 100)])
        tool = DesktopTool(safety_config=config)
        # 确保不在频率限制内
        tool._last_op_time = 0.0
        tool._consecutive_ops = 0
        error = tool._pre_safety_check({"action": "click", "x": 200, "y": 200})
        assert error == ""

    def test_safe_zone_blocks_drag_endpoint(self):
        """安全区域阻止拖拽终点"""
        config = DesktopSafetyConfig(safe_zones=[(500, 500, 600, 600)])
        tool = DesktopTool(safety_config=config)
        tool._last_op_time = 0.0
        tool._consecutive_ops = 0
        error = tool._pre_safety_check({
            "action": "drag", "x": 100, "y": 100, "end_x": 550, "end_y": 550,
        })
        assert "安全区域" in error

    def test_blocked_hotkey_alt_f4(self):
        """阻止 Alt+F4"""
        tool = DesktopTool()
        error = tool._check_blocked_hotkey("alt,f4")
        assert "被阻止" in error

    def test_blocked_hotkey_ctrl_alt_delete(self):
        """阻止 Ctrl+Alt+Delete"""
        tool = DesktopTool()
        error = tool._check_blocked_hotkey("ctrl,alt,delete")
        assert "被阻止" in error

    def test_allowed_hotkey_ctrl_c(self):
        """允许 Ctrl+C"""
        tool = DesktopTool()
        error = tool._check_blocked_hotkey("ctrl,c")
        assert error == ""

    def test_allowed_hotkey_ctrl_v(self):
        """允许 Ctrl+V"""
        tool = DesktopTool()
        error = tool._check_blocked_hotkey("ctrl,v")
        assert error == ""

    def test_rate_limiting(self):
        """频率限制"""
        config = DesktopSafetyConfig(min_interval=1.0)
        tool = DesktopTool(safety_config=config)
        tool._update_op_state()  # 模拟一次成功操作
        error = tool._pre_safety_check({"action": "screenshot"})
        assert "操作过于频繁" in error

    def test_rate_limiting_allows_after_interval(self):
        """频率限制到期后允许"""
        config = DesktopSafetyConfig(min_interval=0.01)
        tool = DesktopTool(safety_config=config)
        tool._update_op_state()
        time.sleep(0.02)
        tool._consecutive_ops = 0  # 重置计数避免干扰
        error = tool._pre_safety_check({"action": "screenshot"})
        assert error == ""

    def test_max_consecutive_ops(self):
        """连续操作次数限制"""
        config = DesktopSafetyConfig(max_consecutive_ops=3, min_interval=0.0)
        tool = DesktopTool(safety_config=config)
        tool._consecutive_ops = 3
        error = tool._pre_safety_check({"action": "screenshot"})
        assert "最大连续操作次数" in error

    def test_reset_consecutive_ops(self):
        """重置连续操作计数"""
        config = DesktopSafetyConfig(max_consecutive_ops=2, min_interval=0.0)
        tool = DesktopTool(safety_config=config)
        tool._consecutive_ops = 2
        tool.reset_consecutive_ops()
        assert tool._consecutive_ops == 0

    def test_screen_bounds_negative_coords(self):
        """负坐标超出屏幕"""
        tool = DesktopTool()
        error = tool._check_screen_bounds(-1, 0)
        assert "超出屏幕范围" in error

    def test_screen_bounds_very_large_coords(self):
        """超大坐标超出屏幕"""
        tool = DesktopTool()
        error = tool._check_screen_bounds(99999, 99999)
        assert "超出屏幕范围" in error

    def test_screen_bounds_valid_coords(self):
        """有效坐标通过检查"""
        tool = DesktopTool()
        error = tool._check_screen_bounds(100, 100)
        assert error == ""


# ===== 操作执行 =====

class TestActionExecution:
    """操作执行测试（模拟模式下）"""

    @pytest.fixture
    def tool(self):
        """创建 DesktopTool 实例"""
        return DesktopTool()

    @pytest.mark.asyncio
    async def test_unknown_action(self, tool):
        """未知操作类型返回错误"""
        result = await tool.execute({"action": "nonexistent"})
        assert result.success is False
        assert "未知" in result.error

    @pytest.mark.asyncio
    async def test_screenshot_no_deps(self, tool):
        """截图在无依赖时返回失败（模拟模式）"""
        if _HAS_PIL or _HAS_MSS:
            pytest.skip("Pillow/mss 已安装，无法测试无依赖场景")
        result = await tool.execute({"action": "screenshot"})
        assert result.success is False
        assert "Pillow" in result.error or "mss" in result.error

    @pytest.mark.asyncio
    async def test_click_no_pyautogui(self, tool):
        """点击在无 pyautogui 时返回失败"""
        if _HAS_PYAUTOGUI:
            pytest.skip("pyautogui 已安装，无法测试模拟模式")
        result = await tool.execute({"action": "click", "x": 100, "y": 200})
        assert result.success is False
        assert "pyautogui" in result.error

    @pytest.mark.asyncio
    async def test_type_text_no_pyautogui(self, tool):
        """文本输入在无 pyautogui 时返回失败"""
        if _HAS_PYAUTOGUI:
            pytest.skip("pyautogui 已安装")
        result = await tool.execute({"action": "type_text", "text": "hello"})
        assert result.success is False
        assert "pyautogui" in result.error

    @pytest.mark.asyncio
    async def test_scroll_no_pyautogui(self, tool):
        """滚动在无 pyautogui 时返回失败"""
        if _HAS_PYAUTOGUI:
            pytest.skip("pyautogui 已安装")
        result = await tool.execute({"action": "scroll", "direction": "down", "scroll_amount": 3})
        assert result.success is False
        assert "pyautogui" in result.error

    @pytest.mark.asyncio
    async def test_hotkey_blocked(self, tool):
        """被阻止的快捷键返回错误"""
        result = await tool.execute({"action": "hotkey", "keys": "alt,f4"})
        assert result.success is False
        assert "被阻止" in result.error

    @pytest.mark.asyncio
    async def test_hotkey_no_pyautogui(self, tool):
        """快捷键在无 pyautogui 时返回失败（安全快捷键）"""
        if _HAS_PYAUTOGUI:
            pytest.skip("pyautogui 已安装")
        result = await tool.execute({"action": "hotkey", "keys": "ctrl,c"})
        assert result.success is False
        assert "pyautogui" in result.error

    @pytest.mark.asyncio
    async def test_move_mouse_no_pyautogui(self, tool):
        """鼠标移动在无 pyautogui 时返回失败"""
        if _HAS_PYAUTOGUI:
            pytest.skip("pyautogui 已安装")
        result = await tool.execute({"action": "move_mouse", "x": 100, "y": 200})
        assert result.success is False
        assert "pyautogui" in result.error

    @pytest.mark.asyncio
    async def test_drag_no_pyautogui(self, tool):
        """拖拽在无 pyautogui 时返回失败"""
        if _HAS_PYAUTOGUI:
            pytest.skip("pyautogui 已安装")
        result = await tool.execute({"action": "drag", "x": 100, "y": 100, "end_x": 200, "end_y": 200})
        assert result.success is False
        assert "pyautogui" in result.error

    @pytest.mark.asyncio
    async def test_screenshot_returns_data_on_success(self, tool):
        """截图成功时返回包含 MultimodalContent 的数据"""
        # 这个测试在无显示环境下可能跳过
        if not _HAS_PIL and not _HAS_MSS:
            pytest.skip("无截图依赖")
        result = await tool.execute({"action": "screenshot"})
        if result.success:
            assert "multimodal_content" in result.data
            assert isinstance(result.data["multimodal_content"], MultimodalContent)


# ===== DesktopContext 集成 =====

class TestDesktopContextIntegration:
    """DesktopContext 集成测试"""

    @pytest.mark.asyncio
    async def test_capture_desktop_context(self):
        """捕获桌面上下文"""
        tool = DesktopTool()
        ctx = await tool.capture_desktop_context()
        assert isinstance(ctx, DesktopContext)
        # screenshot 可能为 None（无显示环境）
        # interactive_elements 应为空列表（暂未实现平台 API）
        assert isinstance(ctx.interactive_elements, list)

    @pytest.mark.asyncio
    async def test_capture_without_interactive_elements(self):
        """不检测交互元素"""
        tool = DesktopTool()
        ctx = await tool.capture_desktop_context(include_interactive_elements=False)
        assert isinstance(ctx, DesktopContext)
        assert ctx.interactive_elements == []

    @pytest.mark.asyncio
    async def test_screenshot_in_desktop_context_is_multimodal(self):
        """截图成功时 DesktopContext.screenshot 为 MultimodalContent"""
        if not _HAS_PIL and not _HAS_MSS:
            pytest.skip("无截图依赖")
        tool = DesktopTool()
        ctx = await tool.capture_desktop_context()
        # 如果截图成功，验证类型
        if ctx.screenshot is not None:
            assert isinstance(ctx.screenshot, MultimodalContent)
            assert ctx.screenshot.type == "image_base64"


# ===== 注册便捷函数 =====

class TestRegisterDesktopTool:
    """register_desktop_tool 注册便捷函数测试"""

    def test_register(self):
        """注册工具到注册表"""
        registry = ToolRegistry()
        tool = register_desktop_tool(registry)
        assert isinstance(tool, DesktopTool)
        assert registry.has_tool("desktop")

    def test_register_returns_tool(self):
        """返回注册的工具实例"""
        registry = ToolRegistry()
        tool = register_desktop_tool(registry)
        assert tool.name == "desktop"

    def test_register_overwrites_existing(self):
        """重复注册覆盖已有工具"""
        registry = ToolRegistry()
        _tool1 = register_desktop_tool(registry)
        tool2 = register_desktop_tool(registry)
        assert registry.get("desktop") is tool2


# ===== 模拟模式 =====

class TestSimulationMode:
    """模拟模式测试"""

    def test_simulation_mode_without_pyautogui(self):
        """无 pyautogui 时启用模拟模式"""
        tool = DesktopTool()
        if not _HAS_PYAUTOGUI:
            assert tool.simulation_mode is True
        else:
            assert tool.simulation_mode is False

    def test_safety_config_accessible(self):
        """安全配置可访问"""
        config = DesktopSafetyConfig(min_interval=2.0)
        tool = DesktopTool(safety_config=config)
        assert tool.safety_config.min_interval == 2.0

    def test_default_safety_config(self):
        """默认安全配置"""
        tool = DesktopTool()
        assert isinstance(tool.safety_config, DesktopSafetyConfig)
        assert tool.safety_config.min_interval == 0.5
