# teragent/security/permission.py
"""权限管理模块

基础权限系统:
  - PermissionLevel: 五级权限枚举
  - PermissionManager: 基础权限管理器（级别提升/降级/审计）

增强权限系统:
  - PermissionEffect: 规则效果枚举（ALLOW / DENY）
  - PermissionRule: 权限规则（工具名 glob + 路径 glob + 来源优先级）
  - EnhancedPermissionManager: 增强版权限管理器（规则匹配 + 分层解析 + 权限级别回退）

参考 Claude-Code 的七层权限解析:
  Layer 1: 用户级规则（source=user） -- 最高优先级
  Layer 2: 配置级规则（source=config）
  Layer 3: 项目级规则（source=project）
  Layer 4: 系统级规则（source=system）
  Layer 5: 权限级别检查（PermissionLevel）
  Layer 6: AI 分类器（可选，仅 acheck() 异步方法可用）
  Layer 7: 默认策略（DENY）

使用示例::

    from teragent.security.permission import (
        EnhancedPermissionManager, PermissionRule, PermissionEffect
    )

    epm = EnhancedPermissionManager()

    # 添加自定义规则
    epm.add_rule(PermissionRule(
        effect=PermissionEffect.DENY,
        tool_pattern="read_file",
        path_pattern="/etc/*",
        description="禁止读取系统目录",
        source="user",
    ))

    # 检查权限
    allowed, reason = epm.check("read_file", path="/etc/passwd")
    # allowed = False, reason = "Denied by rule: 禁止读取系统目录"

    allowed, reason = epm.check("read_file", path="/src/main.py")
    # allowed = True, reason = "Allowed by rule: 读取文件始终允许"

    # 从配置文件加载规则
    epm.load_from_config({
        "rules": {
            "allow": ["read_file:*", "explore_codebase:*"],
            "deny": ["*:**/.env*", "read_file:/etc/*"],
        }
    })
"""
import fnmatch
import functools
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from enum import IntEnum, Enum

from teragent.utils.exceptions import PermissionDenied

logger = logging.getLogger(__name__)


# ===== Phase 5.3: 基础权限系统 =====

class PermissionLevel(IntEnum):
    """数值越高，权限越大"""
    DEFAULT = 0       # 只读
    PLAN = 1          # 允许写项目目录
    BYPASS = 2        # 允许执行用户确认的高风险操作
    ACCEPT_EDITS = 3  # 自动接受代码修改
    AUTO = 4          # 全自动，无需确认
    CUSTOM = 99       # 用户自定义


class PermissionManager:
    """管理权限级别，支持提升、降级、审计追踪和装饰器检查。"""

    def __init__(self, default_level: PermissionLevel = PermissionLevel.PLAN) -> None:
        self.current_level = default_level
        self._default_level = default_level
        self._history: list[tuple[float, PermissionLevel, PermissionLevel]] = []

    def check_level(self, required: PermissionLevel) -> bool:
        """检查当前权限是否满足要求。

        Args:
            required: The minimum required permission level.

        Returns:
            True if the current level meets or exceeds the required level.
        """
        if self.current_level >= required:
            return True
        logger.warning(
            f"Permission denied: Required {required.name}, Current {self.current_level.name}"
        )
        return False

    def elevate(self, new_level: PermissionLevel) -> None:
        """提升权限到指定级别。

        Only elevates if the new level is strictly higher than the current level.
        Records the elevation in audit log (PLAN 5.4).

        Args:
            new_level: The target permission level to elevate to.
        """
        if new_level > self.current_level:
            self._history.append((time.time(), self.current_level, new_level))
            logger.info(
                f"Permission elevated from {self.current_level.name} to {new_level.name}"
            )
            # 审计日志 (PLAN 5.4) -- try async, fall back to sync log
            try:
                import asyncio
                from teragent.security.audit import log_audit
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(log_audit(
                        "permission_elevate",
                        f"From {self.current_level.name} to {new_level.name}"
                    ))
                except RuntimeError:
                    # No event loop — record synchronously as fallback
                    logger.info(
                        f"Permission elevated: {self.current_level.name} → {new_level.name} "
                        f"(audit deferred — no event loop)"
                    )
            except Exception as e:
                # Non-blocking: don't let audit failure prevent elevation
                logger.info(
                    f"Permission elevated: {self.current_level.name} → {new_level.name} "
                    f"(audit deferred — {e})"
                )
            self.current_level = new_level

    def deactivate(self) -> None:
        """重置到默认权限级别。

        Records the deactivation in the history if the level actually changes.
        """
        if self.current_level != self._default_level:
            self._history.append((time.time(), self.current_level, self._default_level))
            logger.info(
                f"Permission deactivated from {self.current_level.name} "
                f"to {self._default_level.name}"
            )
            self.current_level = self._default_level

    def require_level(self, required: PermissionLevel) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        """装饰器：检查权限级别，不满足则抛出 PermissionDenied。

        Usage::

            pm = PermissionManager()
            @pm.require_level(PermissionLevel.BYPASS)
            async def dangerous_operation(): ...

        Args:
            required: The minimum required permission level.

        Returns:
            A decorator that wraps the target async function.
        """
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.check_level(required):
                    raise PermissionDenied(
                        f"Operation requires {required.name} level, "
                        f"current: {self.current_level.name}"
                    )
                return await func(*args, **kwargs)
            return wrapper
        return decorator

    def get_history(self) -> list[dict]:
        """获取权限变更历史。

        Returns:
            A list of dicts with keys ``timestamp``, ``from``, ``to``.
        """
        return [
            {"timestamp": ts, "from": old.name, "to": new.name}
            for ts, old, new in self._history
        ]

    @property
    def default_level(self) -> PermissionLevel:
        """Return the default (initial) permission level."""
        return self._default_level

    def reset_to_default(self) -> None:
        """Alias for :meth:`deactivate` -- reset permission to default level."""
        self.deactivate()


# ===== Phase 9.4: 增强权限系统 =====

class PermissionEffect(Enum):
    """权限规则效果"""
    ALLOW = "allow"
    DENY = "deny"


# 来源优先级数值（数值越大优先级越高）
_SOURCE_PRIORITY = {
    "user": 100,
    "config": 60,     # config > project, matching documented Layer 2 > Layer 3
    "project": 50,
    "system": 10,
    "default": 0,
}


@dataclass
class PermissionRule:
    """权限规则 -- Phase 9.4

    基于 glob 模式匹配工具名和路径，支持来源优先级。

    匹配规则:
      - tool_pattern: 使用 fnmatch glob 匹配工具名
        - "read_file" 匹配 read_file 工具
        - "read_*" 匹配所有 read_ 开头的工具
        - "*" 匹配所有工具
      - path_pattern: 使用 fnmatch glob 匹配路径参数
        - "/etc/*" 匹配 /etc/ 下所有文件
        - "**/.env*" 匹配任意目录下的 .env 文件
        - ""（空）匹配所有路径（不检查路径）

    优先级:
      - 同一来源的规则按添加顺序匹配（先添加优先）
      - 不同来源按优先级排序：user > config > project > system > default

    使用示例::

        rule = PermissionRule(
            effect=PermissionEffect.DENY,
            tool_pattern="read_file",
            path_pattern="/etc/*",
            description="禁止读取系统目录",
            source="user",
        )
        rule.matches_tool("read_file")  # True
        rule.matches_path("/etc/passwd")  # True
    """

    effect: PermissionEffect
    tool_pattern: str       # 工具名匹配模式（glob）
    path_pattern: str = ""  # 路径匹配模式（glob，空=不检查路径）
    description: str = ""   # 规则描述
    source: str = "system"  # 规则来源（user/project/system/config/default）

    def matches_tool(self, tool_name: str) -> bool:
        """检查工具名是否匹配规则

        Args:
            tool_name: 工具名称

        Returns:
            True 表示匹配
        """
        return fnmatch.fnmatch(tool_name, self.tool_pattern)

    def matches_path(self, path: str) -> bool:
        """检查路径是否匹配规则

        Args:
            path: 文件路径

        Returns:
            True 表示匹配（空 path_pattern 匹配所有路径）
        """
        if not self.path_pattern:
            return True
        if not path:
            return False
        return fnmatch.fnmatch(
            os.path.normcase(path),
            os.path.normcase(self.path_pattern),
        )

    @property
    def priority(self) -> int:
        """获取来源优先级"""
        return _SOURCE_PRIORITY.get(self.source, 0)

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "effect": self.effect.value,
            "tool_pattern": self.tool_pattern,
            "path_pattern": self.path_pattern,
            "description": self.description,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PermissionRule":
        """从字典创建 PermissionRule"""
        effect_str = data.get("effect", "deny")
        try:
            effect = PermissionEffect(effect_str)
        except ValueError:
            effect = PermissionEffect.DENY

        return cls(
            effect=effect,
            tool_pattern=data.get("tool_pattern", "*"),
            path_pattern=data.get("path_pattern", ""),
            description=data.get("description", ""),
            source=data.get("source", "system"),
        )


class EnhancedPermissionManager:
    """增强版权限管理器 -- Phase 9.4

    支持规则匹配 + 分层解析 + 权限级别回退 + AI 分类器。

    权限解析流程（优先级从高到低）:
      Layer 1: 用户级规则（source=user） -- 最高优先级
      Layer 2: 配置级规则（source=config）
      Layer 3: 项目级规则（source=project）
      Layer 4: 系统级规则（source=system）
      Layer 5: 权限级别检查（PermissionLevel）
      Layer 6: AI 分类器（可选，仅 acheck() 异步方法可用）
      Layer 7: 默认策略（DENY）

    与 PermissionManager 的关系:
      - EnhancedPermissionManager 内部包含一个 PermissionManager 实例
      - 规则匹配优先于权限级别检查
      - 无匹配规则时回退到权限级别检查
      - AI 分类器仅在 acheck() 异步方法中调用，check() 同步方法不调用

    使用示例::

        epm = EnhancedPermissionManager()

        # 使用默认规则集
        for rule in EnhancedPermissionManager.default_rules():
            epm.add_rule(rule)

        # 添加自定义规则
        epm.add_rule(PermissionRule(
            effect=PermissionEffect.DENY,
            tool_pattern="read_file",
            path_pattern="/etc/*",
            description="禁止读取系统目录",
            source="user",
        ))

        # 同步检查权限（不含 AI 分类器）
        allowed, reason = epm.check("read_file", path="/etc/passwd")

        # 异步检查权限（含 AI 分类器）
        allowed, reason = await epm.acheck("write_file", path="/src/main.py")
    """

    def __init__(
        self,
        default_level: PermissionLevel = PermissionLevel.PLAN,
        default_effect: PermissionEffect = PermissionEffect.DENY,
        ai_classifier: Any | None = None,  # Phase 0.5: 构造器注入
    ) -> None:
        """初始化增强版权限管理器

        Args:
            default_level: 默认权限级别（用于无规则匹配时的回退检查）
            default_effect: 默认规则效果（DENY = 默认拒绝，ALLOW = 默认允许）
            ai_classifier: AI 权限分类器（可选，仅 acheck() 异步方法使用）
        """
        self._base_manager = PermissionManager(default_level=default_level)
        self._rules: list[PermissionRule] = []
        self._default_effect = default_effect
        self._check_count: int = 0
        self._deny_count: int = 0
        # Phase 0.5: AI 权限分类器通过构造器注入（不再使用 setter）
        self._ai_classifier: Any | None = ai_classifier
        # Phase 4.2: 排序规则缓存
        self._sorted_rules_cache: list | None = None
        self._rules_dirty: bool = True

    # ===== 规则管理 =====

    def add_rule(self, rule: PermissionRule) -> None:
        """添加权限规则

        Args:
            rule: PermissionRule 实例
        """
        self._rules.append(rule)
        self._rules_dirty = True
        logger.info(
            f"Permission rule added: {rule.effect.value} "
            f"{rule.tool_pattern}"
            + (f" path={rule.path_pattern}" if rule.path_pattern else "")
            + f" ({rule.source})"
        )

    def add_rules(self, rules: list[PermissionRule]) -> None:
        """批量添加权限规则

        Args:
            rules: PermissionRule 列表
        """
        for rule in rules:
            self.add_rule(rule)

    def remove_rules_by_source(self, source: str) -> int:
        """移除指定来源的所有规则

        Args:
            source: 规则来源（user/project/system/config/default）

        Returns:
            移除的规则数量
        """
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.source != source]
        removed = before - len(self._rules)
        if removed > 0:
            self._rules_dirty = True
            logger.info(f"Removed {removed} permission rules from source={source}")
        return removed

    def clear_rules(self) -> None:
        """清除所有规则"""
        self._rules.clear()
        self._rules_dirty = True
        logger.info("All permission rules cleared")

    def _get_sorted_rules(self) -> list[PermissionRule]:
        """获取排序后的规则列表（带缓存）

        缓存策略：仅在规则列表被修改后重新排序，避免每次 check/acheck
        都执行 O(n log n) 的排序操作。

        Returns:
            按优先级排序的规则列表
        """
        if self._rules_dirty or self._sorted_rules_cache is None:
            self._sorted_rules_cache = sorted(
                self._rules,
                key=lambda r: (
                    r.priority,
                    1 if r.path_pattern else 0,
                    1 if r.effect == PermissionEffect.DENY else 0,
                ),
                reverse=True,
            )
            self._rules_dirty = False
        return self._sorted_rules_cache

    # ===== AI 分类器管理 =====

    @property
    def ai_classifier(self) -> Any | None:
        """获取 AI 权限分类器"""
        return self._ai_classifier

    @ai_classifier.setter
    def ai_classifier(self, value: Any | None) -> None:
        """设置 AI 权限分类器

        DEPRECATED (Phase 0.5): Use EnhancedPermissionManager(ai_classifier=...) instead.

        Args:
            value: AIPermissionClassifier 实例或 None
        """
        self._ai_classifier = value
        if value is not None:
            logger.info("AI permission classifier enabled")
        else:
            logger.info("AI permission classifier disabled")

    # ===== 权限检查（同步） =====

    def check(self, tool_name: str, path: str = "") -> tuple[bool, str]:
        """同步检查权限（不含 AI 分类器）

        规则按优先级匹配:
        1. 用户级规则（source=user）
        2. 配置级规则（source=config）
        3. 项目级规则（source=project）
        4. 系统级规则（source=system）
        5. 权限级别检查
        6. 默认策略

        注意: 此方法不调用 AI 分类器。如需 AI 分类器支持，
        请使用 acheck() 异步方法。

        Args:
            tool_name: 工具名称
            path: 文件路径（可选，某些规则需要检查路径）

        Returns:
            (allowed, reason) -- 是否允许 + 原因
        """
        self._check_count += 1

        # 排序策略：
        # 1. 来源优先级（高优先级先匹配）
        # 2. 同优先级内，有路径模式（更具体）的规则优先
        # 3. 同优先级内，DENY 规则优先于 ALLOW 规则（保守安全）
        # 这样确保：system DENY /etc/* 优先于 system ALLOW read_file:*
        sorted_rules = self._get_sorted_rules()

        for rule in sorted_rules:
            if rule.matches_tool(tool_name) and rule.matches_path(path):
                if rule.effect == PermissionEffect.ALLOW:
                    return True, (
                        f"Allowed by rule: "
                        f"{rule.description or rule.tool_pattern}"
                    )
                else:
                    self._deny_count += 1
                    return False, (
                        f"Denied by rule: "
                        f"{rule.description or rule.tool_pattern}"
                    )

        # 无匹配规则：回退到权限级别检查
        return self._check_fallback(tool_name, path)

    def _check_fallback(self, tool_name: str, path: str = "") -> tuple[bool, str]:
        """权限级别回退检查（同步和异步共用）

        Args:
            tool_name: 工具名称
            path: 文件路径

        Returns:
            (allowed, reason) -- 是否允许 + 原因
        """
        if self._base_manager.current_level >= PermissionLevel.AUTO:
            return True, "Auto-approved (AUTO level)"

        if self._base_manager.current_level >= PermissionLevel.ACCEPT_EDITS:
            return True, "Approved (ACCEPT_EDITS level)"

        if self._base_manager.current_level >= PermissionLevel.BYPASS:
            # BYPASS: user has explicitly confirmed, allow high-risk operations
            return True, "Approved (BYPASS - user confirmed high-risk)"

        if self._base_manager.current_level >= PermissionLevel.PLAN:
            # PLAN: allow project directory writes
            return True, "Approved (PLAN level - project write allowed)"

        # DEFAULT: fall back to default policy
        if self._default_effect == PermissionEffect.ALLOW:
            return True, "Allowed by default policy"
        else:
            self._deny_count += 1
            return False, "Denied by default policy (no matching rule)"

    def check_tool_params(self, tool_name: str, params: dict) -> tuple[bool, str]:
        """检查工具调用的权限（从参数中提取路径）

        从工具参数中自动提取路径字段（file_path, path, filepath, dir, directory, workspace），
        然后调用 check() 进行权限检查。

        Args:
            tool_name: 工具名称
            params: 工具参数

        Returns:
            (allowed, reason) -- 是否允许 + 原因
        """
        # 从参数中提取路径
        path = ""
        for key in ("file_path", "path", "filepath", "dir", "directory", "workspace"):
            val = params.get(key, "")
            if val and isinstance(val, str):
                path = val
                break

        return self.check(tool_name, path=path)

    # ===== 权限检查（异步，含 AI 分类器） =====

    async def acheck(self, tool_name: str, path: str = "", context: str = "") -> tuple[bool, str]:
        """异步检查权限（含 AI 分类器支持）

        与 check() 的区别:
          - check() 是同步方法，不含 AI 分类器（Layer 6）
          - acheck() 是异步方法，在权限级别检查未通过时，
            会调用 AI 分类器做咨询性判断

        流程:
        1. 先执行同步规则匹配（Layer 1-4）
        2. 如果规则匹配，直接返回结果
        3. 如果无匹配规则，尝试权限级别检查（Layer 5）
        4. 如果权限级别允许，直接返回
        5. 如果权限级别不允许且有 AI 分类器，调用 AI 分类器（Layer 6）
        6. AI 分类器 ALLOW 且置信度足够 -> 允许
        7. 否则 -> 默认策略（Layer 7）

        Args:
            tool_name: 工具名称
            path: 文件路径
            context: 上下文信息（传递给 AI 分类器）

        Returns:
            (allowed, reason) -- 是否允许 + 原因
        """
        self._check_count += 1

        # 1. 先执行同步规则匹配
        sorted_rules = self._get_sorted_rules()

        for rule in sorted_rules:
            if rule.matches_tool(tool_name) and rule.matches_path(path):
                if rule.effect == PermissionEffect.ALLOW:
                    return True, (
                        f"Allowed by rule: "
                        f"{rule.description or rule.tool_pattern}"
                    )
                else:
                    self._deny_count += 1
                    return False, (
                        f"Denied by rule: "
                        f"{rule.description or rule.tool_pattern}"
                    )

        # 2. 权限级别检查（复用 _check_fallback 保持同步/异步一致）
        # Note: _check_fallback may increment _deny_count internally,
        # but we treat its result as advisory — the final decision may differ.
        fallback_allowed, fallback_reason = self._check_fallback(tool_name, path)
        if fallback_allowed:
            return True, fallback_reason

        # 3. AI 分类器咨询（仅在权限级别未通过时）
        if self._ai_classifier is not None:
            try:
                params = {"path": path} if path else {}
                effect, confidence, reason = await self._ai_classifier.classify(
                    tool_name=tool_name,
                    params=params,
                    context=context,
                )
                if effect == PermissionEffect.ALLOW:
                    return True, f"AI classifier allowed (confidence={confidence:.2f}): {reason}"
                else:
                    return False, f"AI classifier denied (confidence={confidence:.2f}): {reason}"
            except Exception as e:
                logger.warning(f"AI classifier error, falling back to default policy: {e}")

        # 4. 默认策略 — return the fallback reason from step 2 instead of
        # re-evaluating the default policy (which _check_fallback already did)
        if self._default_effect == PermissionEffect.ALLOW:
            return True, "Allowed by default policy"
        else:
            return False, fallback_reason or "Denied by default policy (no matching rule)"

    async def acheck_tool_params(
        self, tool_name: str, params: dict, context: str = ""
    ) -> tuple[bool, str]:
        """异步检查工具调用的权限（含 AI 分类器，从参数中提取路径）

        Args:
            tool_name: 工具名称
            params: 工具参数
            context: 上下文信息（传递给 AI 分类器）

        Returns:
            (allowed, reason) -- 是否允许 + 原因
        """
        # 从参数中提取路径
        path = ""
        for key in ("file_path", "path", "filepath", "dir", "directory", "workspace"):
            val = params.get(key, "")
            if val and isinstance(val, str):
                path = val
                break

        return await self.acheck(tool_name, path=path, context=context)

    # ===== 权限级别管理（委托给 PermissionManager）=====

    @property
    def current_level(self) -> PermissionLevel:
        """当前权限级别"""
        return self._base_manager.current_level

    def elevate(self, new_level: PermissionLevel) -> None:
        """提升权限级别"""
        self._base_manager.elevate(new_level)

    def deactivate(self) -> None:
        """重置到默认权限级别"""
        self._base_manager.deactivate()

    def set_level(self, level: int | PermissionLevel) -> None:
        """直接设置权限级别

        Args:
            level: 目标权限级别
        """
        if isinstance(level, int):
            level = PermissionLevel(level)

        current = self._base_manager.current_level
        if level > current:
            self.elevate(level)
        elif level < current:
            # Record downgrade in audit history
            self._base_manager._history.append(
                (time.time(), current, level)
            )
            self._base_manager.current_level = level
            logger.info(f"Permission level set to {level.name}")
            # Fire-and-forget audit logging
            try:
                import asyncio
                from teragent.security.audit import log_audit
                loop = asyncio.get_running_loop()
                loop.create_task(log_audit(
                    "permission_downgrade",
                    f"Level changed: {current.name} → {level.name}"
                ))
            except RuntimeError:
                pass  # No event loop
            except Exception:
                pass  # Non-blocking

    # ===== 配置加载 =====

    def load_from_config(self, config: dict) -> None:
        """从配置字典加载权限规则

        配置格式::

            {
                "mode": "default",  # default / accept_edits / bypass / auto / plan
                "rules": {
                    "allow": ["read_file:*", "explore_codebase:*"],
                    "deny": ["*:**/.env*", "read_file:/etc/*"],
                }
            }

        每条规则格式: "<tool_pattern>:<path_pattern>" 或 "<tool_pattern>"（无路径限制）

        Args:
            config: 配置字典
        """
        # 设置权限模式
        mode = config.get("mode", "default")
        mode_map = {
            "default": PermissionLevel.DEFAULT,
            "plan": PermissionLevel.PLAN,
            "bypass": PermissionLevel.BYPASS,
            "accept_edits": PermissionLevel.ACCEPT_EDITS,
            "auto": PermissionLevel.AUTO,
        }
        if mode in mode_map:
            self.set_level(mode_map[mode])

        # 加载规则
        rules_config = config.get("rules", {})

        # Allow 规则
        allow_rules = rules_config.get("allow", [])
        for rule_str in allow_rules:
            rule = self._parse_rule_string(rule_str, PermissionEffect.ALLOW, source="config")
            if rule:
                self.add_rule(rule)

        # Deny 规则
        deny_rules = rules_config.get("deny", [])
        for rule_str in deny_rules:
            rule = self._parse_rule_string(rule_str, PermissionEffect.DENY, source="config")
            if rule:
                self.add_rule(rule)

        logger.info(
            f"Permission config loaded: mode={mode}, "
            f"allow={len(allow_rules)}, deny={len(deny_rules)}"
        )

    @staticmethod
    def _parse_rule_string(
        rule_str: str,
        effect: PermissionEffect,
        source: str = "config",
    ) -> PermissionRule | None:
        """解析规则字符串

        格式: "<tool_pattern>:<path_pattern>" 或 "<tool_pattern>"（无路径限制）

        Args:
            rule_str: 规则字符串
            effect: 规则效果
            source: 规则来源

        Returns:
            PermissionRule 或 None（解析失败时）
        """
        rule_str = rule_str.strip()
        if not rule_str:
            return None

        parts = rule_str.split(":", 1)
        tool_pattern = parts[0].strip()
        path_pattern = parts[1].strip() if len(parts) > 1 else ""

        return PermissionRule(
            effect=effect,
            tool_pattern=tool_pattern,
            path_pattern=path_pattern,
            description=f"Config rule: {rule_str}",
            source=source,
        )

    # ===== 默认规则集 =====

    @staticmethod
    def default_rules() -> list[PermissionRule]:
        """默认权限规则集

        Returns:
            默认规则列表
        """
        return [
            # 只读工具始终允许
            PermissionRule(
                effect=PermissionEffect.ALLOW,
                tool_pattern="read_file",
                description="读取文件始终允许",
                source="system",
            ),
            PermissionRule(
                effect=PermissionEffect.ALLOW,
                tool_pattern="explore_codebase",
                description="搜索代码库始终允许",
                source="system",
            ),
            PermissionRule(
                effect=PermissionEffect.ALLOW,
                tool_pattern="classify_intent",
                description="意图分类始终允许",
                source="system",
            ),
            PermissionRule(
                effect=PermissionEffect.ALLOW,
                tool_pattern="list_directory",
                description="列出目录始终允许",
                source="system",
            ),
            PermissionRule(
                effect=PermissionEffect.ALLOW,
                tool_pattern="get_pipeline_status",
                description="查询流水线状态始终允许",
                source="system",
            ),
            PermissionRule(
                effect=PermissionEffect.ALLOW,
                tool_pattern="submit_failure",
                description="提交失败始终允许",
                source="system",
            ),
            PermissionRule(
                effect=PermissionEffect.ALLOW,
                tool_pattern="spawn_agent",
                description="创建子 Agent 始终允许",
                source="system",
            ),
            PermissionRule(
                effect=PermissionEffect.ALLOW,
                tool_pattern="send_message",
                description="发送消息始终允许",
                source="system",
            ),
            # 敏感路径拒绝
            PermissionRule(
                effect=PermissionEffect.DENY,
                tool_pattern="read_file",
                path_pattern="/etc/*",
                description="禁止读取系统目录",
                source="system",
            ),
            # Root-level sensitive files (fnmatch **/ does not match root-level paths)
            PermissionRule(
                effect=PermissionEffect.DENY,
                tool_pattern="*",
                path_pattern=".env*",
                description="禁止访问根目录环境变量文件",
                source="system",
            ),
            PermissionRule(
                effect=PermissionEffect.DENY,
                tool_pattern="*",
                path_pattern="**/.env*",
                description="禁止访问环境变量文件",
                source="system",
            ),
            PermissionRule(
                effect=PermissionEffect.DENY,
                tool_pattern="*",
                path_pattern=".ssh/*",
                description="禁止访问根目录 SSH 密钥",
                source="system",
            ),
            PermissionRule(
                effect=PermissionEffect.DENY,
                tool_pattern="*",
                path_pattern="**/.ssh/*",
                description="禁止访问 SSH 密钥",
                source="system",
            ),
            PermissionRule(
                effect=PermissionEffect.DENY,
                tool_pattern="*",
                path_pattern=".git/*",
                description="禁止修改根目录 Git 内部文件",
                source="system",
            ),
            PermissionRule(
                effect=PermissionEffect.DENY,
                tool_pattern="*",
                path_pattern="**/.git/*",
                description="禁止修改 Git 内部文件",
                source="system",
            ),
            # 高风险操作需要更高权限（由权限级别回退检查处理）
        ]

    # ===== 状态报告 =====

    def get_status_report(self) -> dict:
        """获取状态报告（供 /perm 和调试使用）"""
        report = {
            "current_level": self._base_manager.current_level.name,
            "default_effect": self._default_effect.value,
            "total_rules": len(self._rules),
            "rules_by_source": {
                source: len([r for r in self._rules if r.source == source])
                for source in set(r.source for r in self._rules)
            },
            "rules_by_effect": {
                "allow": len([r for r in self._rules if r.effect == PermissionEffect.ALLOW]),
                "deny": len([r for r in self._rules if r.effect == PermissionEffect.DENY]),
            },
            "check_count": self._check_count,
            "deny_count": self._deny_count,
            "ai_classifier_enabled": self._ai_classifier is not None,
        }
        # 附加 AI 分类器统计（如果有）
        if self._ai_classifier is not None and hasattr(self._ai_classifier, 'get_stats'):
            report["ai_classifier_stats"] = self._ai_classifier.get_stats()
        return report

    def get_rules_summary(self) -> list[dict]:
        """获取规则摘要列表"""
        sorted_rules = sorted(
            self._rules,
            key=lambda r: r.priority,
            reverse=True,
        )
        return [r.to_dict() for r in sorted_rules]

    def get_history(self) -> list[dict]:
        """获取权限变更历史"""
        return self._base_manager.get_history()

    def reset(self) -> None:
        """重置所有状态"""
        self._base_manager.reset_to_default()
        self._rules.clear()
        self._rules_dirty = True
        self._check_count = 0
        self._deny_count = 0
        if self._ai_classifier is not None and hasattr(self._ai_classifier, 'clear_cache'):
            self._ai_classifier.clear_cache()
            if hasattr(self._ai_classifier, 'reset_stats'):
                self._ai_classifier.reset_stats()
