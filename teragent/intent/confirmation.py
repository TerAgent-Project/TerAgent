# teragent/intent/confirmation.py
"""高风险操作的用户确认门控

当意图被判定为 create_project 时，系统生成简短摘要，
通过 EventBus 请求 TUI 展示确认对话框，等待用户响应。

设计原则:
  - M0 阶段：自动确认（不打断用户体验）
  - M1 阶段：接入 TUI 交互确认
  - 信号驱动：通过 EventBus 异步请求确认
  - 超时兜底：用户无响应时自动确认（避免流水线卡死）
"""
import asyncio
import logging
import uuid

from teragent.event_bus import EventBus

logger = logging.getLogger(__name__)

# 默认确认超时（秒）— M0 自动确认模式不需要等待
_CONFIRM_TIMEOUT = 0.0

# M1 模式下的确认超时（秒）— 用户 30 秒无响应则自动确认
_M1_CONFIRM_TIMEOUT = 30.0


class ConfirmationGate:
    """高风险操作的用户确认门控

    使用方式::

        gate = ConfirmationGate(bus=event_bus)
        confirmed = await gate.confirm_create_project(requirement)
        if confirmed:
            # 启动流水线
        else:
            # 用户取消
    """

    def __init__(
        self,
        bus: EventBus,
        auto_confirm: bool = True,
        timeout: float = 0.0,
    ) -> None:
        """初始化确认门控

        Args:
            bus: EventBus 实例，用于发送确认请求信号
            auto_confirm: M0 模式下是否自动确认（默认 True）
            timeout: 确认等待超时（秒），0 表示不等待
        """
        self.bus = bus
        self.auto_confirm = auto_confirm
        self.timeout = timeout if timeout is not None and timeout > 0 else (
            _CONFIRM_TIMEOUT if auto_confirm else _M1_CONFIRM_TIMEOUT
        )

        # 等待中的确认请求
        self._pending_confirmations: dict[str, asyncio.Future] = {}

        # 注册确认响应事件
        bus.on("confirmation_response", self._on_confirmation_response)

    async def confirm_create_project(self, requirement: str) -> bool:
        """请求用户确认项目创建

        流程:
          1. 生成简短摘要
          2. 通过 EventBus 发射确认请求信号
          3. M0 模式: 自动确认
          4. M1 模式: 等待用户响应（带超时）

        Args:
            requirement: 用户的需求描述

        Returns:
            True 表示用户确认或自动确认，False 表示用户取消
        """
        summary = self._generate_summary(requirement)

        # M0 模式：自动确认
        if self.auto_confirm:
            logger.info(f"Auto-confirmed project creation: {summary[:80]}")
            await self.bus.emit(
                "confirmation_request",
                summary=summary,
                auto_confirmed=True,
            )
            return True

        # M1 模式：请求用户确认
        request_id = f"confirm_{uuid.uuid4().hex[:8]}"
        future = asyncio.get_running_loop().create_future()
        self._pending_confirmations[request_id] = future

        # 发射确认请求信号
        await self.bus.emit(
            "confirmation_request",
            request_id=request_id,
            summary=summary,
            auto_confirmed=False,
        )

        # 等待用户响应（带超时）
        try:
            confirmed = await asyncio.wait_for(future, timeout=self.timeout)
            return bool(confirmed)
        except asyncio.TimeoutError:
            # 超时自动确认
            logger.warning(f"Confirmation timeout for '{summary[:50]}', auto-confirming.")
            self._pending_confirmations.pop(request_id, None)
            return True

    async def _on_confirmation_response(
        self, request_id: str = "", confirmed: bool = False, **kwargs: object
    ) -> None:
        """处理用户确认响应"""
        future = self._pending_confirmations.pop(request_id, None)
        if future and not future.done():
            future.set_result(confirmed)

    def _generate_summary(self, requirement: str) -> str:
        """从需求描述中提取关键信息，生成简短摘要

        检测用户明确的技术约束并包含在摘要中。
        不做主观推断（如"游戏→Pygame"），只传播用户明确表达的约束。
        """
        lines = requirement.strip().split("\n")
        first_line = lines[0][:100] if lines else requirement[:100]

        # 检测用户明确的技术约束
        constraint = self._detect_tech_constraints(requirement)

        summary = f"即将创建项目：{first_line}"
        if constraint:
            summary += f"\n技术约束: {constraint}"

        return summary

    @staticmethod
    def _detect_tech_constraints(requirement: str) -> str:
        """从用户需求中检测明确的技术约束

        只检测用户明确表达的约束，不做主观推断。
        返回约束描述字符串，空字符串表示未检测到约束。
        """
        import re
        text = requirement.lower()

        # 1. 检测排除性约束
        exclusion_patterns = [
            (r"不用\s*(js|javascript|ts|typescript|web|html|css|node|前端)",
             lambda m: f"不使用 {m.group(1).upper()}"),
            (r"不需要\s*(js|javascript|ts|typescript|web|html|css|node|前端)",
             lambda m: f"不使用 {m.group(1).upper()}"),
            (r"no\s*(js|javascript|ts|typescript|web|html|css|node)",
             lambda m: f"不使用 {m.group(1).upper()}"),
            (r"without\s*(js|javascript|ts|typescript|web|html|css|node)",
             lambda m: f"不使用 {m.group(1).upper()}"),
            (r"不要\s*(用|使用)?\s*(web|网页|浏览器|前端|js|javascript|ts|typescript|html)",
             lambda m: "不使用 Web/前端技术"),
        ]
        for pattern, formatter in exclusion_patterns:
            match = re.search(pattern, text)
            if match:
                return formatter(match)

        # 2. 检测限定性约束
        constraint_patterns = [
            (r"纯\s*python|全部\s*(用|使用)?\s*py|只用\s*python|python\s*only|纯py",
             "纯Python，不使用JS/TS"),
            (r"纯\s*web|全部\s*(用|使用)?\s*web|只用\s*web|web\s*only",
             "纯Web技术"),
            (r"纯\s*js|全部\s*(用|使用)?\s*js|只用\s*javascript",
             "纯JavaScript"),
        ]
        for pattern, constraint_desc in constraint_patterns:
            if re.search(pattern, text):
                return constraint_desc

        # 3. 检测明确的技术栈指定
        explicit_tech = {
            "pygame": "Pygame",
            "tkinter": "Tkinter",
            "pyqt": "PyQt",
            "qt": "Qt",
            "flask": "Flask",
            "django": "Django",
            "fastapi": "FastAPI",
            "react": "React",
            "vue": "Vue",
        }
        for kw, desc in explicit_tech.items():
            if kw in text:
                return f"{desc}"

        return ""

    def cancel_all(self) -> None:
        """取消所有等待中的确认请求"""
        for request_id, future in self._pending_confirmations.items():
            if not future.done():
                future.set_result(False)
        self._pending_confirmations.clear()
