# teragent/tools/base.py
"""工具基类与统一返回格式

所有工具必须继承 BaseTool 并实现 execute() 方法。
工具通过 ToolRegistry 统一管理，支持:
  - 注册 / 查询 / 列举
  - 转换为 OpenAI function calling 格式
  - 安全级别标记（ToolSafety）+ 并发安全标记
  - 运行时权限检查（check_permissions）+ 输入验证（validate_input）
  - 进度回调 + 工具专属提示 + 注册元数据

设计原则:
  - 工具只负责"执行"，不负责"决策"（决策由 AgentLoop / 流水线控制）
  - 工具结果通过 ToolResult 统一返回，永远不抛异常给调用者
  - 工具不知道自己被谁调用（信号驱动，解耦）
  - 工具声明自身安全属性，编排器据此决定并行/串行
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional

from teragent.core.types import ToolSafety

logger = logging.getLogger(__name__)


# ===== ToolResult =====

@dataclass
class ToolResult:
    """工具执行结果的统一返回格式

    Attributes:
        success: 是否执行成功
        data: 结构化数据（成功时填充）
        error: 错误描述（失败时填充）
        metadata: 额外元数据（执行耗时、Token 数等）
        safety: 工具安全级别（用于编排器决策）
        context_modifier: 上下文修改器（工具执行后可修改共享上下文）
        extra_messages: 额外消息（如需要注入到对话中的警告/提示）
    """

    success: bool
    data: dict = field(default_factory=dict)
    error: str = ""
    metadata: dict = field(default_factory=dict)
    safety: ToolSafety = ToolSafety.READ_ONLY
    context_modifier: Optional[Callable[[dict], dict]] = None
    extra_messages: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为可序列化字典

        注意: context_modifier 和 extra_messages 中可能包含不可序列化的对象，
        to_dict 只保留可序列化字段。
        """
        result = {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "safety": self.safety.value,
        }
        if self.extra_messages:
            result["extra_messages"] = self.extra_messages
        # context_modifier 是函数，不可序列化，跳过
        return result


# ===== BaseTool =====

class BaseTool(ABC):
    """所有工具的基类

    参考 Claude-Code 的 Tool 接口（15+ 方法），为 TerAgent 工具增加:
      - 安全属性: _safety / _concurrency_safe
      - 生命周期: validate_input → check_permissions → execute
      - 属性方法: is_read_only / is_destructive / is_concurrency_safe / safety_level
      - 可选覆盖: get_tool_prompt / describe_usage / set_progress_callback
      - 注册格式: to_function_definition / to_registry_metadata

    子类必须定义:
      - name: 工具唯一标识符
      - description: 工具描述（供 LLM 理解用途）
      - parameters_schema: JSON Schema 格式的参数定义

    子类必须实现:
      - execute(params: dict) -> ToolResult: 执行工具逻辑

    子类可选覆盖:
      - _safety: ToolSafety 安全级别（默认 SAFE_WRITE）
      - _concurrency_safe: 是否可并行执行（默认 False）
      - validate_input(params) -> list[str]: 输入验证
      - check_permissions(params, level) -> (bool, str): 权限检查
      - get_tool_prompt() -> str: 工具专属提示
      - describe_usage(params) -> str: 动态描述当前调用
    """

    # === 基础属性 ===
    name: str = ""
    description: str = ""
    parameters_schema: dict | None = None

    # === 安全属性（子类覆盖） ===
    _safety: ToolSafety = ToolSafety.SAFE_WRITE
    _concurrency_safe: bool = False
    _progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None

    # === 生命周期方法 ===

    @abstractmethod
    async def execute(self, params: dict, progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None) -> ToolResult:
        """执行工具，返回结构化结果

        Args:
            params: 工具参数，格式由 parameters_schema 定义
            progress_callback: Optional progress callback for this execution only.
                If provided, temporarily sets self._progress_callback for the
                duration of execution, avoiding race conditions in parallel use.

        Returns:
            ToolResult: 统一返回格式
        """
        ...

    def validate_input(self, params: dict) -> list[str]:
        """输入验证（在权限检查前执行）

        在 ToolOrchestrator 的完整生命周期中，validate_input 在
        check_permissions 之前执行，确保参数合法后再检查权限。

        Returns:
            错误列表，空列表表示验证通过
        """
        # 默认实现：复用原有的 validate_params 做必填项检查
        return self.validate_params(params)

    def check_permissions(self, params: dict, permission_level: int = 0) -> tuple[bool, str]:
        """权限检查

        Args:
            params: 工具参数
            permission_level: 当前权限级别

        Returns:
            (allowed, reason) — 是否允许执行 + 原因
        """
        # 默认实现：根据安全级别检查权限
        if self._safety == ToolSafety.READ_ONLY:
            return True, ""
        if self._safety == ToolSafety.HIGH_RISK:
            # HIGH_RISK 需要 BYPASS 权限（level >= 2）
            if permission_level < 2:
                return False, f"工具 {self.name} 需要更高权限（当前: {permission_level}）"
        if self._safety == ToolSafety.DESTRUCTIVE:
            # DESTRUCTIVE 需要 PLAN 权限（level >= 1）
            if permission_level < 1:
                return False, f"工具 {self.name} 需要更高权限（当前: {permission_level}）"
        return True, ""

    # === 属性方法 ===

    @property
    def is_read_only(self) -> bool:
        """是否为只读工具"""
        return self._safety == ToolSafety.READ_ONLY

    @property
    def is_destructive(self) -> bool:
        """是否为破坏性工具"""
        return self._safety in (ToolSafety.DESTRUCTIVE, ToolSafety.HIGH_RISK)

    @property
    def is_concurrency_safe(self) -> bool:
        """是否可并行执行"""
        return self._concurrency_safe

    @property
    def safety_level(self) -> ToolSafety:
        """安全级别"""
        return self._safety

    # === 可选覆盖方法 ===

    def get_tool_prompt(self) -> str:
        """工具专属提示（注入到系统提示中）

        Claude-Code 中每个工具可以提供自己的 prompt 文本，
        用于指导模型如何使用该工具。

        Returns:
            提示文本，空字符串表示无额外提示
        """
        return ""

    def describe_usage(self, params: dict) -> str:
        """动态描述当前工具调用（供 TUI 展示）

        例: read_file → "读取 src/main.py (2.3KB)"
        """
        return f"执行 {self.name}"

    def set_progress_callback(self, callback: Callable[[str, float], Awaitable[None]]) -> None:
        """设置进度回调

        Args:
            callback: (message, progress_fraction) → None
        """
        self._progress_callback = callback

    # === 注册格式 ===

    def to_function_definition(self) -> dict:
        """转换为 OpenAI function calling 格式

        Returns:
            符合 OpenAI tools 格式的字典
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema or {},
            },
        }

    def to_registry_metadata(self) -> dict:
        """返回工具注册元数据（供 ToolRegistry 和 ToolOrchestrator 使用）

        包含安全属性，用于编排器决定并行/串行执行策略。
        """
        return {
            "name": self.name,
            "safety": self._safety.value,
            "concurrency_safe": self._concurrency_safe,
            "read_only": self.is_read_only,
            "destructive": self.is_destructive,
        }

    # === 兼容方法 ===

    def validate_params(self, params: dict) -> list[str]:
        """验证参数是否符合 schema（基本校验：必填项检查）

        此方法保持向后兼容，validate_input 默认调用此方法。

        Args:
            params: 待验证的参数字典

        Returns:
            缺失的必填参数名列表（空列表表示全部通过）
        """
        if not self.parameters_schema:
            return []

        required = self.parameters_schema.get("required", [])
        properties = self.parameters_schema.get("properties", {})
        missing: list[str] = []

        for req_key in required:
            if req_key not in params:
                missing.append(req_key)
            elif req_key in params and properties.get(req_key, {}).get("type") == "string":
                if not isinstance(params[req_key], str) or not params[req_key].strip():
                    missing.append(req_key)

        return missing

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} safety={self._safety.value}>"
