# teragent/tools/desktop.py
"""桌面操作工具 — MiniMax M3 桌面操作能力

支持截图、点击、输入文本、滚动、快捷键等桌面操作。
M3 可以"看"屏幕截图，识别可交互元素，并通过工具调用执行桌面操作。

安全限制：
  - 所有操作需要用户确认（DESTRUCTIVE 级别）
  - 可配置安全区域（禁止点击的区域）
  - 操作频率限制（防止过快操作）
  - 可配置最大连续操作次数
  - 阻止危险快捷键组合

依赖说明：
  - pyautogui: 可选，用于鼠标/键盘操作
  - Pillow / mss: 可选，用于截图
  - 如未安装，工具仍可导入并在模拟模式下运行
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

__all__ = [
    "DesktopSafetyConfig",
    "DesktopTool",
    "register_desktop_tool",
]

from teragent.core.tap import DesktopContext, MultimodalContent
from teragent.core.types import ToolSafety
from teragent.tools.base import BaseTool, ToolResult
from teragent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# ===== 可选依赖导入 =====

try:
    import pyautogui
    _HAS_PYAUTOGUI = True
    # 禁用 pyautogui 的默认 fail-safe（我们在内部做安全检查）
    # 但保留 fail-safe 作为最终防护
except ImportError:
    pyautogui = None  # type: ignore[assignment]
    _HAS_PYAUTOGUI = False

try:
    from PIL import ImageGrab
    _HAS_PIL = True
except ImportError:
    ImageGrab = None  # type: ignore[assignment]
    _HAS_PIL = False

try:
    import mss
    _HAS_MSS = True
except ImportError:
    mss = None  # type: ignore[assignment]
    _HAS_MSS = False


# ===== 安全配置 =====

# 阻止的快捷键组合（大小写不敏感）
# 这些快捷键可能导致系统级不可逆操作
_BLOCKED_SHORTCUTS: set[frozenset[str]] = {
    frozenset({"alt", "f4"}),          # 关闭窗口
    frozenset({"ctrl", "alt", "delete"}),  # 系统安全界面
    frozenset({"ctrl", "alt", "del"}),     # 同上（缩写）
    frozenset({"alt", "tab"}),          # 切换窗口（可能导致不可预测行为）
    frozenset({"ctrl", "shift", "esc"}),  # 任务管理器
    frozenset({"win", "l"}),           # 锁屏
    frozenset({"super", "l"}),         # 锁屏（Linux）
    frozenset({"cmd", "q"}),           # macOS 退出应用
    frozenset({"cmd", "option", "esc"}),  # macOS 强制退出
}

# 默认最小操作间隔（秒）
_DEFAULT_MIN_INTERVAL = 0.5

# 默认最大连续操作次数
_DEFAULT_MAX_CONSECUTIVE_OPS = 50

# 截图默认 JPEG 质量
_DEFAULT_SCREENSHOT_QUALITY = 75


@dataclass
class DesktopSafetyConfig:
    """桌面操作安全配置

    Attributes:
        safe_zones: 禁止操作的区域列表，每个元素为 (x1, y1, x2, y2) 矩形
        blocked_shortcuts: 阻止的快捷键组合集合
        min_interval: 最小操作间隔（秒）
        max_consecutive_ops: 最大连续操作次数
        screenshot_quality: 截图 JPEG 压缩质量 (1-100)
        screenshot_format: 截图格式 "jpeg" 或 "png"
    """
    safe_zones: list[tuple[int, int, int, int]] = field(default_factory=list)
    blocked_shortcuts: set[frozenset[str]] = field(default_factory=lambda: _BLOCKED_SHORTCUTS.copy())
    min_interval: float = _DEFAULT_MIN_INTERVAL
    max_consecutive_ops: int = _DEFAULT_MAX_CONSECUTIVE_OPS
    screenshot_quality: int = _DEFAULT_SCREENSHOT_QUALITY
    screenshot_format: str = "jpeg"


class DesktopTool(BaseTool):
    """桌面操作工具 — 支持截图、点击、输入、滚动、快捷键等桌面操作

    用于 MiniMax M3 的桌面操作能力。M3 可以"看"屏幕截图，
    识别可交互元素，并通过工具调用执行桌面操作。

    安全限制：
    - 所有操作需要用户确认（DESTRUCTIVE 级别）
    - 可配置安全区域（禁止点击的区域）
    - 操作频率限制（防止过快操作）
    - 可配置最大连续操作次数
    """

    name = "desktop"
    description = "桌面操作工具：截图、点击、输入文本、滚动、快捷键等"

    # Parameters schema for OpenAI function calling
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["screenshot", "click", "type_text", "scroll", "hotkey", "move_mouse", "drag"],
                "description": "桌面操作类型"
            },
            "x": {"type": "integer", "description": "X坐标（点击/移动/拖拽起点）"},
            "y": {"type": "integer", "description": "Y坐标（点击/移动/拖拽起点）"},
            "text": {"type": "string", "description": "输入文本（type_text操作）"},
            "direction": {"type": "string", "enum": ["up", "down", "left", "right"], "description": "滚动方向"},
            "scroll_amount": {"type": "integer", "description": "滚动量（像素或次数）"},
            "keys": {"type": "string", "description": "快捷键组合（逗号分隔，如 'ctrl,c'）"},
            "end_x": {"type": "integer", "description": "拖拽终点X坐标"},
            "end_y": {"type": "integer", "description": "拖拽终点Y坐标"},
            "button": {"type": "string", "enum": ["left", "right", "middle"], "description": "鼠标按钮"},
        },
        "required": ["action"],
    }

    _safety = ToolSafety.DESTRUCTIVE  # 桌面操作默认需要确认
    _concurrency_safe = False  # 桌面操作不能并行

    def __init__(self, safety_config: Optional[DesktopSafetyConfig] = None) -> None:
        """初始化桌面操作工具

        Args:
            safety_config: 安全配置，为 None 时使用默认配置
        """
        self._safety_config = safety_config or DesktopSafetyConfig()
        # 操作频率限制状态
        self._last_op_time: float = 0.0
        # 连续操作计数
        self._consecutive_ops: int = 0
        # 模拟模式标志（依赖不可用时自动启用）
        self._simulation_mode: bool = not _HAS_PYAUTOGUI

    # ===== 属性 =====

    @property
    def simulation_mode(self) -> bool:
        """是否处于模拟模式（pyautogui 不可用）"""
        return self._simulation_mode

    @property
    def safety_config(self) -> DesktopSafetyConfig:
        """当前安全配置"""
        return self._safety_config

    # ===== BaseTool 生命周期方法 =====

    async def execute(
        self,
        params: dict,
        progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None,
    ) -> ToolResult:
        """执行桌面操作

        根据 params["action"] 分发到对应的操作方法。

        Args:
            params: 工具参数，格式由 parameters_schema 定义
            progress_callback: 进度回调

        Returns:
            ToolResult: 操作结果
        """
        action = params.get("action", "")

        if progress_callback:
            await progress_callback(f"Starting desktop action: {action}", 0.0)

        # 安全预检查
        safety_error = self._pre_safety_check(params)
        if safety_error:
            if progress_callback:
                await progress_callback(f"Safety check failed: {action}", 1.0)
            return ToolResult(
                success=False,
                error=safety_error,
                safety=self._safety,
            )

        # 操作分发
        action_map = {
            "screenshot": self._action_screenshot,
            "click": self._action_click,
            "type_text": self._action_type_text,
            "scroll": self._action_scroll,
            "hotkey": self._action_hotkey,
            "move_mouse": self._action_move_mouse,
            "drag": self._action_drag,
        }

        handler = action_map.get(action)
        if not handler:
            if progress_callback:
                await progress_callback(f"Unknown action: {action}", 1.0)
            return ToolResult(
                success=False,
                error=f"未知的桌面操作类型: {action!r}，支持的操作: {list(action_map.keys())}",
                safety=self._safety,
            )

        if progress_callback:
            await progress_callback(f"Executing: {action}", 0.5)

        try:
            result = await handler(params)
            # 操作成功时更新频率限制状态
            if result.success:
                self._update_op_state()
            if progress_callback:
                await progress_callback(f"Completed: {action}", 1.0)
            return result
        except Exception as e:
            logger.error(f"DesktopTool: 操作 {action} 执行失败: {e}", exc_info=True)
            if progress_callback:
                await progress_callback(f"Failed: {action}", 1.0)
            return ToolResult(
                success=False,
                error=f"桌面操作 {action} 执行失败: {e}",
                safety=self._safety,
            )

    def validate_input(self, params: dict) -> list[str]:
        """输入验证

        除了基类的必填项检查，还增加对坐标范围、操作参数的验证。
        """
        errors = self.validate_params(params)
        action = params.get("action", "")

        # 点击操作需要坐标
        if action in ("click", "move_mouse"):
            if "x" not in params or "y" not in params:
                errors.append(f"{action} 操作需要 x 和 y 坐标")

        # 拖拽操作需要起点和终点
        if action == "drag":
            if "x" not in params or "y" not in params:
                errors.append("drag 操作需要 x 和 y 坐标（起点）")
            if "end_x" not in params or "end_y" not in params:
                errors.append("drag 操作需要 end_x 和 end_y 坐标（终点）")

        # 输入文本操作需要 text
        if action == "type_text":
            if "text" not in params or not params["text"]:
                errors.append("type_text 操作需要 text 参数")

        # 快捷键操作需要 keys
        if action == "hotkey":
            if "keys" not in params or not params["keys"]:
                errors.append("hotkey 操作需要 keys 参数")

        # 滚动操作需要 direction
        if action == "scroll":
            if "direction" not in params or not params["direction"]:
                errors.append("scroll 操作需要 direction 参数")

        return errors

    def get_tool_prompt(self) -> str:
        """工具专属提示（注入到系统提示中）

        指导 M3 如何使用桌面操作工具。
        """
        return (
            "【桌面操作工具使用指南】\n"
            "1. 先用 screenshot 截取当前屏幕，分析屏幕内容\n"
            "2. 根据分析结果，使用 click/type_text/scroll/hotkey 执行操作\n"
            "3. 操作后再次截图确认结果\n"
            "4. 坐标基于屏幕左上角 (0,0)，向右为 X 正方向，向下为 Y 正方向\n"
            "5. 快捷键使用逗号分隔，如 'ctrl,c' 表示 Ctrl+C\n"
            "6. 所有操作需要用户确认才能执行\n"
        )

    def describe_usage(self, params: dict) -> str:
        """动态描述当前工具调用（供 TUI 展示）"""
        action = params.get("action", "unknown")
        detail = ""

        if action == "click":
            x, y = params.get("x", "?"), params.get("y", "?")
            button = params.get("button", "left")
            detail = f"点击 ({x}, {y}) [{button}]"
        elif action == "type_text":
            text = params.get("text", "")
            preview = text[:20] + "..." if len(text) > 20 else text
            detail = f"输入文本: {preview!r}"
        elif action == "screenshot":
            detail = "截取屏幕截图"
        elif action == "scroll":
            direction = params.get("direction", "?")
            amount = params.get("scroll_amount", 3)
            detail = f"滚动 {direction} {amount}"
        elif action == "hotkey":
            keys = params.get("keys", "?")
            detail = f"快捷键: {keys}"
        elif action == "move_mouse":
            x, y = params.get("x", "?"), params.get("y", "?")
            detail = f"移动鼠标到 ({x}, {y})"
        elif action == "drag":
            x, y = params.get("x", "?"), params.get("y", "?")
            end_x, end_y = params.get("end_x", "?"), params.get("end_y", "?")
            detail = f"拖拽 ({x}, {y}) → ({end_x}, {end_y})"
        else:
            detail = f"未知操作: {action}"

        return f"桌面操作: {detail}"

    # ===== 安全检查方法 =====

    def _pre_safety_check(self, params: dict) -> str:
        """操作前安全检查

        Returns:
            错误消息字符串，空字符串表示检查通过
        """
        # 1. 频率限制
        now = time.monotonic()
        elapsed = now - self._last_op_time
        if elapsed < self._safety_config.min_interval and self._last_op_time > 0:
            return (
                f"操作过于频繁，距离上次操作仅 {elapsed:.2f}s，"
                f"最小间隔为 {self._safety_config.min_interval}s"
            )

        # 2. 连续操作次数限制
        if self._consecutive_ops >= self._safety_config.max_consecutive_ops:
            return (
                f"已达到最大连续操作次数 {self._safety_config.max_consecutive_ops}，"
                f"请确认是否继续"
            )

        # 3. 安全区域检查
        action = params.get("action", "")
        if action in ("click", "move_mouse", "drag"):
            x = params.get("x", 0)
            y = params.get("y", 0)
            zone_error = self._check_safe_zone(x, y)
            if zone_error:
                return zone_error

        if action == "drag":
            end_x = params.get("end_x", 0)
            end_y = params.get("end_y", 0)
            zone_error = self._check_safe_zone(end_x, end_y)
            if zone_error:
                return zone_error

        # 4. 屏幕边界检查
        if action in ("click", "move_mouse", "drag"):
            x = params.get("x", 0)
            y = params.get("y", 0)
            bounds_error = self._check_screen_bounds(x, y)
            if bounds_error:
                return bounds_error

        if action == "drag":
            end_x = params.get("end_x", 0)
            end_y = params.get("end_y", 0)
            bounds_error = self._check_screen_bounds(end_x, end_y)
            if bounds_error:
                return bounds_error

        # 5. 快捷键安全检查
        if action == "hotkey":
            keys = params.get("keys", "")
            hotkey_error = self._check_blocked_hotkey(keys)
            if hotkey_error:
                return hotkey_error

        return ""

    def _check_safe_zone(self, x: int, y: int) -> str:
        """检查坐标是否在安全区域（禁止操作的区域）

        Args:
            x: X 坐标
            y: Y 坐标

        Returns:
            错误消息，空字符串表示不在安全区域
        """
        for i, (x1, y1, x2, y2) in enumerate(self._safety_config.safe_zones):
            if x1 <= x <= x2 and y1 <= y <= y2:
                return (
                    f"坐标 ({x}, {y}) 在安全区域 #{i+1} 内 "
                    f"({x1},{y1})-({x2},{y2})，禁止操作"
                )
        return ""

    def _check_screen_bounds(self, x: int, y: int) -> str:
        """检查坐标是否在屏幕范围内

        当屏幕尺寸为 (0, 0) 时（无法检测），跳过边界检查。

        Args:
            x: X 坐标
            y: Y 坐标

        Returns:
            错误消息，空字符串表示在范围内
        """
        screen_w, screen_h = self._get_screen_size()
        # 屏幕尺寸未知，跳过边界检查
        if screen_w == 0 and screen_h == 0:
            return ""
        if x < 0 or y < 0 or x >= screen_w or y >= screen_h:
            return (
                f"坐标 ({x}, {y}) 超出屏幕范围 "
                f"(0,0)-({screen_w-1},{screen_h-1})"
            )
        return ""

    def _check_blocked_hotkey(self, keys: str) -> str:
        """检查快捷键是否被阻止

        Args:
            keys: 逗号分隔的快捷键组合，如 "ctrl,c"

        Returns:
            错误消息，空字符串表示未被阻止
        """
        if not keys:
            return ""

        key_set = frozenset(k.strip().lower() for k in keys.split(",") if k.strip())

        for blocked in self._safety_config.blocked_shortcuts:
            if key_set == blocked:
                return f"快捷键 {keys!r} 被阻止（危险操作）"

        return ""

    def _update_op_state(self) -> None:
        """更新操作频率限制状态"""
        self._last_op_time = time.monotonic()
        self._consecutive_ops += 1

    def reset_consecutive_ops(self) -> None:
        """重置连续操作计数（用户确认后调用）"""
        self._consecutive_ops = 0

    def _get_screen_size(self) -> tuple[int, int]:
        """获取屏幕尺寸

        跨平台策略:
          - 优先使用 pyautogui.size()
          - Windows 回退: ctypes.windll.user32.GetSystemMetrics
          - macOS 回退: AppKit.NSScreen
          - 最终回退: (0, 0) 表示未知，禁用坐标检查

        Returns:
            (width, height) 元组，(0, 0) 表示无法检测
        """
        if _HAS_PYAUTOGUI and pyautogui is not None:
            try:
                size = pyautogui.size()
                return size.width, size.height
            except Exception:
                pass

        # 平台特定回退
        try:
            if sys.platform == "win32":
                import ctypes
                user32 = ctypes.windll.user32
                width = user32.GetSystemMetrics(0)
                height = user32.GetSystemMetrics(1)
                if width > 0 and height > 0:
                    return width, height
            elif sys.platform == "darwin":
                from AppKit import NSScreen  # type: ignore[import-untyped]
                screen = NSScreen.mainScreen()
                frame = screen.frame()
                return int(frame.size.width), int(frame.size.height)
        except Exception:
            pass

        # 无法检测 — 返回 (0, 0) 表示未知
        logger.warning("DesktopTool: 无法检测屏幕尺寸，坐标边界检查将跳过")
        return 0, 0

    # ===== 操作实现 =====

    async def _action_screenshot(self, params: dict) -> ToolResult:
        """截取屏幕截图

        优先使用 Pillow ImageGrab，回退到 mss，都没有则返回模拟结果。
        截图压缩为 JPEG/PNG 后以 base64 编码返回。
        """
        # 尝试使用 Pillow 截图
        img_data: bytes | None = None
        img_format = self._safety_config.screenshot_format.upper()
        # Pillow 使用 "JPEG" 而非 "JPG"
        if img_format == "JPG":
            img_format = "JPEG"

        # Linux 优先使用 mss (ImageGrab 需要 Xlib)
        if sys.platform.startswith("linux") and _HAS_MSS and mss is not None:
            try:
                with mss.mss() as sct:
                    monitor = sct.monitors[0] if sct.monitors else None
                    if monitor:
                        screenshot = sct.grab(monitor)
                        from PIL import Image as PILImage
                        img = PILImage.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                        buf = io.BytesIO()
                        save_kwargs = {"format": img_format}
                        if img_format == "JPEG":
                            save_kwargs["quality"] = self._safety_config.screenshot_quality
                        img.save(buf, **save_kwargs)
                        img_data = buf.getvalue()
                        logger.debug(
                            f"DesktopTool: mss 截图成功 (Linux优先)，"
                            f"尺寸={img.size}，格式={img_format}，"
                            f"大小={len(img_data)} bytes"
                        )
            except Exception as e:
                logger.warning(f"DesktopTool: mss 截图失败: {e}，尝试 Pillow")

        # 非 Linux 或 mss 失败，尝试 Pillow ImageGrab
        if img_data is None and _HAS_PIL and ImageGrab is not None:
            try:
                img = ImageGrab.grab()
                buf = io.BytesIO()
                save_kwargs: dict[str, Any] = {"format": img_format}
                if img_format == "JPEG":
                    save_kwargs["quality"] = self._safety_config.screenshot_quality
                img.save(buf, **save_kwargs)
                img_data = buf.getvalue()
                logger.debug(
                    f"DesktopTool: Pillow 截图成功，"
                    f"尺寸={img.size}，格式={img_format}，"
                    f"大小={len(img_data)} bytes"
                )
            except Exception as e:
                logger.warning(f"DesktopTool: Pillow 截图失败: {e}")

        # Pillow 也失败 (非 Linux)，尝试 mss
        if img_data is None and not sys.platform.startswith("linux") and _HAS_MSS and mss is not None:
            try:
                with mss.mss() as sct:
                    monitor = sct.monitors[0] if sct.monitors else None
                    if monitor:
                        screenshot = sct.grab(monitor)
                        from PIL import Image as PILImage
                        img = PILImage.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                        buf = io.BytesIO()
                        save_kwargs = {"format": img_format}
                        if img_format == "JPEG":
                            save_kwargs["quality"] = self._safety_config.screenshot_quality
                        img.save(buf, **save_kwargs)
                        img_data = buf.getvalue()
                        logger.debug(
                            f"DesktopTool: mss 截图成功，"
                            f"尺寸={img.size}，格式={img_format}，"
                            f"大小={len(img_data)} bytes"
                        )
            except Exception as e:
                logger.warning(f"DesktopTool: mss 截图失败: {e}")

        # 都没有可用，返回模拟结果
        if img_data is None:
            return ToolResult(
                success=False,
                error=(
                    "截图失败：Pillow 和 mss 均不可用或截图失败。"
                    "请安装 Pillow (pip install Pillow) 或 mss (pip install mss)。"
                    "当前为模拟模式，无法执行真实截图。"
                ),
                data={"action": "screenshot", "mode": "simulation"},
                safety=self._safety,
            )

        # base64 编码
        b64_data = base64.b64encode(img_data).decode("ascii")
        media_type = "image/jpeg" if img_format == "JPEG" else "image/png"

        # 构建 MultimodalContent
        mc = MultimodalContent(
            type="image_base64",
            base64_data=b64_data,
            media_type=media_type,
        )

        return ToolResult(
            success=True,
            data={
                "action": "screenshot",
                "format": img_format.lower(),
                "size_bytes": len(img_data),
                "media_type": media_type,
                "multimodal_content": mc,
            },
            safety=self._safety,
            metadata={"image_size": len(img_data)},
        )

    async def _action_click(self, params: dict) -> ToolResult:
        """点击指定坐标

        使用 pyautogui.click() 执行点击操作。
        """
        x = params.get("x", 0)
        y = params.get("y", 0)
        button = params.get("button", "left")

        if not _HAS_PYAUTOGUI or pyautogui is None:
            return ToolResult(
                success=False,
                error="pyautogui 不可用，无法执行点击操作。请安装: pip install pyautogui",
                data={"action": "click", "x": x, "y": y, "button": button, "mode": "simulation"},
                safety=self._safety,
            )

        try:
            pyautogui.click(x=x, y=y, button=button)
            logger.info(f"DesktopTool: 点击 ({x}, {y}) [{button}]")
            return ToolResult(
                success=True,
                data={"action": "click", "x": x, "y": y, "button": button},
                safety=self._safety,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"点击操作失败: {e}",
                data={"action": "click", "x": x, "y": y, "button": button},
                safety=self._safety,
            )

    async def _action_type_text(self, params: dict) -> ToolResult:
        """在当前光标位置输入文本

        使用 pyautogui.write() 或 pyautogui.typewrite() 输入文本。
        对中文字符等非 ASCII 字符，使用 pyautogui.typewrite() 的替代方案。
        """
        text = params.get("text", "")

        if not _HAS_PYAUTOGUI or pyautogui is None:
            return ToolResult(
                success=False,
                error="pyautogui 不可用，无法执行文本输入。请安装: pip install pyautogui",
                data={"action": "type_text", "text_length": len(text), "mode": "simulation"},
                safety=self._safety,
            )

        try:
            # 检查是否包含非 ASCII 字符（如中文）
            has_unicode = any(ord(c) > 127 for c in text)

            if has_unicode:
                # pyautogui.write() 不支持 Unicode 字符
                # 使用剪贴板方式输入（跨平台方案）
                self._type_unicode_text(text)
            else:
                # ASCII 文本直接输入
                pyautogui.write(text)

            logger.info(f"DesktopTool: 输入文本 (长度={len(text)}, unicode={has_unicode})")
            return ToolResult(
                success=True,
                data={
                    "action": "type_text",
                    "text_length": len(text),
                    "has_unicode": has_unicode,
                },
                safety=self._safety,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"文本输入失败: {e}",
                data={"action": "type_text", "text_length": len(text)},
                safety=self._safety,
            )

    def _type_unicode_text(self, text: str) -> None:
        """通过剪贴板输入 Unicode 文本（中文等）

        使用系统剪贴板临时存储文本，然后粘贴输入。
        尝试保存并恢复原始剪贴板内容。

        Args:
            text: 要输入的文本
        """
        import subprocess

        # 保存剪贴板内容
        saved_clipboard = self._read_clipboard()

        try:
            # 将文本复制到剪贴板
            if sys.platform == "darwin":
                process = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
                process.communicate(text.encode("utf-8"))
            # Linux (检测 X11/Wayland)
            elif sys.platform.startswith("linux"):
                session_type = os.getenv("XDG_SESSION_TYPE", "")
                wayland_display = os.getenv("WAYLAND_DISPLAY", "")
                if session_type == "wayland" or wayland_display:
                    process = subprocess.Popen(
                        ["wl-copy"],
                        stdin=subprocess.PIPE,
                    )
                    process.communicate(text.encode("utf-8"))
                else:
                    process = subprocess.Popen(
                        ["xclip", "-selection", "clipboard"],
                        stdin=subprocess.PIPE,
                    )
                    process.communicate(text.encode("utf-8"))
            # Windows
            elif sys.platform == "win32":
                process = subprocess.Popen(["clip"], stdin=subprocess.PIPE)
                process.communicate(text.encode("utf-8"))
            else:
                logger.warning(f"DesktopTool: 不支持的平台 {sys.platform}，尝试直接输入")
                pyautogui.write(text)
                return

            # 粘贴
            if sys.platform == "darwin":
                pyautogui.hotkey("command", "v")
            else:
                pyautogui.hotkey("ctrl", "v")
        except Exception as e:
            logger.warning(f"DesktopTool: Unicode 输入失败: {e}，尝试直接输入")
            try:
                pyautogui.write(text)
            except Exception:
                logger.error("DesktopTool: 所有文本输入方式均失败")
        finally:
            # 恢复剪贴板内容
            if saved_clipboard:
                self._write_clipboard(saved_clipboard)

    def _read_clipboard(self) -> str:
        """读取当前剪贴板内容（跨平台）

        Returns:
            剪贴板文本内容，失败返回空字符串
        """
        import subprocess
        try:
            if sys.platform == "darwin":
                result = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=2)
                return result.stdout
            elif sys.platform.startswith("linux"):
                session_type = os.getenv("XDG_SESSION_TYPE", "")
                wayland_display = os.getenv("WAYLAND_DISPLAY", "")
                if session_type == "wayland" or wayland_display:
                    result = subprocess.run(["wl-paste"], capture_output=True, text=True, timeout=2)
                else:
                    result = subprocess.run(
                        ["xclip", "-selection", "clipboard", "-o"],
                        capture_output=True, text=True, timeout=2,
                    )
                return result.stdout
            elif sys.platform == "win32":
                try:
                    import pyperclip  # type: ignore[import-untyped]
                    return pyperclip.paste()
                except ImportError:
                    return ""
        except Exception:
            pass
        return ""

    def _write_clipboard(self, text: str) -> None:
        """写入剪贴板内容（跨平台）

        Args:
            text: 要写入的文本
        """
        import subprocess
        try:
            if sys.platform == "darwin":
                process = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
                process.communicate(text.encode("utf-8"))
            elif sys.platform.startswith("linux"):
                session_type = os.getenv("XDG_SESSION_TYPE", "")
                wayland_display = os.getenv("WAYLAND_DISPLAY", "")
                if session_type == "wayland" or wayland_display:
                    process = subprocess.Popen(["wl-copy"], stdin=subprocess.PIPE)
                    process.communicate(text.encode("utf-8"))
                else:
                    process = subprocess.Popen(
                        ["xclip", "-selection", "clipboard"],
                        stdin=subprocess.PIPE,
                    )
                    process.communicate(text.encode("utf-8"))
            elif sys.platform == "win32":
                process = subprocess.Popen(["clip"], stdin=subprocess.PIPE)
                process.communicate(text.encode("utf-8"))
        except Exception as e:
            logger.debug(f"DesktopTool: 写入剪贴板失败: {e}")

    async def _action_scroll(self, params: dict) -> ToolResult:
        """滚动屏幕

        使用 pyautogui.scroll() 执行滚动操作。
        """
        direction = params.get("direction", "down")
        scroll_amount = params.get("scroll_amount", 3)

        if not _HAS_PYAUTOGUI or pyautogui is None:
            return ToolResult(
                success=False,
                error="pyautogui 不可用，无法执行滚动操作。请安装: pip install pyautogui",
                data={"action": "scroll", "direction": direction, "amount": scroll_amount, "mode": "simulation"},
                safety=self._safety,
            )

        try:
            # 方向映射：up=+amount, down=-amount
            if direction == "up":
                pyautogui.scroll(scroll_amount)
            elif direction == "down":
                pyautogui.scroll(-scroll_amount)
            elif direction == "left":
                try:
                    pyautogui.hscroll(-scroll_amount)
                except (NotImplementedError, AttributeError):
                    # hscroll 不可用，回退到 Shift+垂直滚动
                    pyautogui.keyDown('shift')
                    pyautogui.scroll(-scroll_amount)
                    pyautogui.keyUp('shift')
            elif direction == "right":
                try:
                    pyautogui.hscroll(scroll_amount)
                except (NotImplementedError, AttributeError):
                    pyautogui.keyDown('shift')
                    pyautogui.scroll(scroll_amount)
                    pyautogui.keyUp('shift')
            else:
                return ToolResult(
                    success=False,
                    error=f"未知的滚动方向: {direction!r}",
                    data={"action": "scroll", "direction": direction},
                    safety=self._safety,
                )

            logger.info(f"DesktopTool: 滚动 {direction} {scroll_amount}")
            return ToolResult(
                success=True,
                data={"action": "scroll", "direction": direction, "amount": scroll_amount},
                safety=self._safety,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"滚动操作失败: {e}",
                data={"action": "scroll", "direction": direction, "amount": scroll_amount},
                safety=self._safety,
            )

    async def _action_hotkey(self, params: dict) -> ToolResult:
        """按下快捷键组合

        解析逗号分隔的快捷键组合（如 "ctrl,c" → Ctrl+C），
        使用 pyautogui.hotkey() 执行。
        """
        keys = params.get("keys", "")

        if not _HAS_PYAUTOGUI or pyautogui is None:
            return ToolResult(
                success=False,
                error="pyautogui 不可用，无法执行快捷键操作。请安装: pip install pyautogui",
                data={"action": "hotkey", "keys": keys, "mode": "simulation"},
                safety=self._safety,
            )

        try:
            # 解析快捷键组合
            key_list = [k.strip() for k in keys.split(",") if k.strip()]

            if not key_list:
                return ToolResult(
                    success=False,
                    error="快捷键列表为空",
                    data={"action": "hotkey", "keys": keys},
                    safety=self._safety,
                )

            pyautogui.hotkey(*key_list)
            logger.info(f"DesktopTool: 快捷键 {keys}")
            return ToolResult(
                success=True,
                data={"action": "hotkey", "keys": keys, "key_list": key_list},
                safety=self._safety,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"快捷键操作失败: {e}",
                data={"action": "hotkey", "keys": keys},
                safety=self._safety,
            )

    async def _action_move_mouse(self, params: dict) -> ToolResult:
        """移动鼠标到指定坐标

        使用 pyautogui.moveTo() 执行移动操作。
        """
        x = params.get("x", 0)
        y = params.get("y", 0)

        if not _HAS_PYAUTOGUI or pyautogui is None:
            return ToolResult(
                success=False,
                error="pyautogui 不可用，无法执行鼠标移动操作。请安装: pip install pyautogui",
                data={"action": "move_mouse", "x": x, "y": y, "mode": "simulation"},
                safety=self._safety,
            )

        try:
            pyautogui.moveTo(x=x, y=y)
            logger.info(f"DesktopTool: 移动鼠标到 ({x}, {y})")
            return ToolResult(
                success=True,
                data={"action": "move_mouse", "x": x, "y": y},
                safety=self._safety,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"鼠标移动失败: {e}",
                data={"action": "move_mouse", "x": x, "y": y},
                safety=self._safety,
            )

    async def _action_drag(self, params: dict) -> ToolResult:
        """从 (x,y) 拖拽到 (end_x, end_y)

        使用 pyautogui.moveTo + drag 执行拖拽操作。
        """
        x = params.get("x", 0)
        y = params.get("y", 0)
        end_x = params.get("end_x", 0)
        end_y = params.get("end_y", 0)
        button = params.get("button", "left")

        if not _HAS_PYAUTOGUI or pyautogui is None:
            return ToolResult(
                success=False,
                error="pyautogui 不可用，无法执行拖拽操作。请安装: pip install pyautogui",
                data={
                    "action": "drag", "x": x, "y": y,
                    "end_x": end_x, "end_y": end_y,
                    "button": button, "mode": "simulation",
                },
                safety=self._safety,
            )

        try:
            # 先移动到起点，然后拖拽到终点
            pyautogui.moveTo(x=x, y=y)
            dx = end_x - x
            dy = end_y - y
            pyautogui.drag(dx, dy, button=button)
            logger.info(f"DesktopTool: 拖拽 ({x},{y}) → ({end_x},{end_y}) [{button}]")
            return ToolResult(
                success=True,
                data={
                    "action": "drag",
                    "start": {"x": x, "y": y},
                    "end": {"x": end_x, "y": end_y},
                    "button": button,
                },
                safety=self._safety,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"拖拽操作失败: {e}",
                data={
                    "action": "drag",
                    "start": {"x": x, "y": y},
                    "end": {"x": end_x, "y": end_y},
                },
                safety=self._safety,
            )

    # ===== DesktopContext 集成 =====

    async def capture_desktop_context(
        self,
        include_interactive_elements: bool = True,
    ) -> DesktopContext:
        """捕获桌面上下文

        截取屏幕截图，并尝试识别可交互元素。
        返回的 DesktopContext 可用于 MiniMaxM3Compiler 编译。

        Args:
            include_interactive_elements: 是否尝试识别可交互元素

        Returns:
            DesktopContext: 包含截图和交互元素的桌面上下文
        """
        # 1. 截图
        screenshot_result = await self._action_screenshot({})

        screenshot_mc: Optional[MultimodalContent] = None
        if screenshot_result.success and screenshot_result.data:
            screenshot_mc = screenshot_result.data.get("multimodal_content")

        # 2. 识别可交互元素
        interactive_elements: list[dict] = []
        active_window = ""

        if include_interactive_elements:
            try:
                interactive_elements = self._detect_interactive_elements()
            except Exception as e:
                logger.warning(f"DesktopTool: 交互元素检测失败: {e}")

            try:
                active_window = self._get_active_window()
            except Exception as e:
                logger.warning(f"DesktopTool: 活动窗口获取失败: {e}")

        return DesktopContext(
            screenshot=screenshot_mc,
            interactive_elements=interactive_elements,
            active_window=active_window,
        )

    def _detect_interactive_elements(self) -> list[dict]:
        """检测屏幕上的可交互元素

        优先使用系统无障碍 API，回退到基础检测。
        在无依赖模式下返回空列表。

        Returns:
            可交互元素列表，每个元素包含 type, label, bbox, action 字段
        """
        # 无障碍 API 访问需要平台特定库
        # 目前返回空列表作为基础实现
        # 未来可集成:
        #   - macOS: ApplicationServices.framework / pyobjc
        #   - Windows: uiautomation / comtypes
        #   - Linux: AT-SPI / python-at-spi
        logger.debug("DesktopTool: 交互元素检测暂未实现平台特定 API，返回空列表")
        return []

    def _get_active_window(self) -> str:
        """获取当前活动窗口标题

        Returns:
            活动窗口标题字符串
        """
        # 尝试使用 pyautogui 相关功能
        # pyautogui 本身不提供窗口标题获取
        # 需要平台特定实现
        try:
            import sys
            if sys.platform == "win32":
                # Windows: 使用 win32gui
                try:
                    import win32gui  # type: ignore[import-untyped]
                    hwnd = win32gui.GetForegroundWindow()
                    return win32gui.GetWindowText(hwnd)
                except ImportError:
                    pass
            elif sys.platform == "darwin":
                # macOS: 使用 AppKit
                try:
                    from AppKit import NSWorkspace  # type: ignore[import-untyped]
                    active_app = NSWorkspace.sharedWorkspace().frontmostApplication()
                    return active_app.localizedName() if active_app else ""
                except ImportError:
                    pass
            elif sys.platform.startswith("linux"):
                # Linux: 使用 xdotool 获取活动窗口标题
                try:
                    import subprocess as _sp
                    result = _sp.run(
                        ["xdotool", "getwindowfocus", "getwindowname"],
                        capture_output=True, text=True, timeout=2,
                    )
                    if result.returncode == 0:
                        return result.stdout.strip()
                except (FileNotFoundError, _sp.TimeoutExpired, OSError):
                    pass
        except Exception as e:
            logger.debug(f"DesktopTool: 获取活动窗口失败: {e}")

        return ""


def register_desktop_tool(registry: ToolRegistry) -> DesktopTool:
    """Register the desktop tool with a ToolRegistry

    便捷函数：创建 DesktopTool 实例并注册到工具注册表。

    Args:
        registry: 工具注册表实例

    Returns:
        注册后的 DesktopTool 实例
    """
    tool = DesktopTool()
    registry.register(tool)
    return tool
