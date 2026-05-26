# tests/test_permission.py
"""Permission 权限管理单元测试

覆盖:
  - PermissionLevel: 级别比较/提升/降级
  - PermissionManager: check_level/elevate/deactivate/require_level
  - PermissionEffect: ALLOW/DENY 枚举
  - PermissionRule: matches_tool/matches_path/priority/to_dict/from_dict
  - EnhancedPermissionManager: 规则匹配/优先级/排序缓存/配置加载/AI分类器
  - check_tool_params: 参数路径提取
"""
import pytest
import time

from teragent.security.permission import (
    PermissionLevel,
    PermissionManager,
    PermissionEffect,
    PermissionRule,
    EnhancedPermissionManager,
)
from teragent.utils.exceptions import PermissionDenied


# ===== PermissionLevel =====

class TestPermissionLevel:
    """PermissionLevel 枚举"""

    def test_level_ordering(self):
        """权限级别数值越高权限越大"""
        assert PermissionLevel.DEFAULT < PermissionLevel.PLAN
        assert PermissionLevel.PLAN < PermissionLevel.BYPASS
        assert PermissionLevel.BYPASS < PermissionLevel.ACCEPT_EDITS
        assert PermissionLevel.ACCEPT_EDITS < PermissionLevel.AUTO

    def test_level_values(self):
        """各级别数值"""
        assert PermissionLevel.DEFAULT == 0
        assert PermissionLevel.PLAN == 1
        assert PermissionLevel.AUTO == 4


# ===== PermissionManager =====

class TestPermissionManager:
    """基础权限管理器"""

    def test_default_level_is_plan(self):
        """默认权限级别为 PLAN"""
        pm = PermissionManager()
        assert pm.current_level == PermissionLevel.PLAN

    def test_check_level_satisfied(self):
        """当前级别满足要求"""
        pm = PermissionManager()
        assert pm.check_level(PermissionLevel.DEFAULT) is True
        assert pm.check_level(PermissionLevel.PLAN) is True

    def test_check_level_not_satisfied(self):
        """当前级别不满足要求"""
        pm = PermissionManager()
        assert pm.check_level(PermissionLevel.BYPASS) is False

    def test_elevate(self):
        """提升权限"""
        pm = PermissionManager()
        pm.elevate(PermissionLevel.AUTO)
        assert pm.current_level == PermissionLevel.AUTO

    def test_elevate_only_higher(self):
        """只提升不降级"""
        pm = PermissionManager(default_level=PermissionLevel.AUTO)
        pm.elevate(PermissionLevel.PLAN)  # 试图降级
        assert pm.current_level == PermissionLevel.AUTO  # 不变

    def test_deactivate(self):
        """重置到默认级别"""
        pm = PermissionManager(default_level=PermissionLevel.PLAN)
        pm.elevate(PermissionLevel.AUTO)
        pm.deactivate()
        assert pm.current_level == PermissionLevel.PLAN  # 回到默认级别

    def test_history_recorded(self):
        """权限变更被记录"""
        pm = PermissionManager()
        pm.elevate(PermissionLevel.AUTO)
        history = pm.get_history()
        assert len(history) >= 1
        assert history[-1]["from"] == "PLAN"
        assert history[-1]["to"] == "AUTO"

    @pytest.mark.asyncio
    async def test_require_level_decorator(self):
        """require_level 装饰器"""
        pm = PermissionManager(default_level=PermissionLevel.DEFAULT)

        @pm.require_level(PermissionLevel.BYPASS)
        async def dangerous_op():
            return "success"

        with pytest.raises(PermissionDenied):
            await dangerous_op()

    @pytest.mark.asyncio
    async def test_require_level_passes(self):
        """require_level 权限足够时正常执行"""
        pm = PermissionManager(default_level=PermissionLevel.AUTO)

        @pm.require_level(PermissionLevel.BYPASS)
        async def safe_op():
            return "success"

        result = await safe_op()
        assert result == "success"


# ===== PermissionRule =====

class TestPermissionRule:
    """权限规则"""

    def test_matches_tool_exact(self):
        """精确匹配工具名"""
        rule = PermissionRule(effect=PermissionEffect.DENY, tool_pattern="read_file")
        assert rule.matches_tool("read_file") is True
        assert rule.matches_tool("write_file") is False

    def test_matches_tool_glob(self):
        """Glob 模式匹配工具名"""
        rule = PermissionRule(effect=PermissionEffect.ALLOW, tool_pattern="read_*")
        assert rule.matches_tool("read_file") is True
        assert rule.matches_tool("read_directory") is True
        assert rule.matches_tool("write_file") is False

    def test_matches_tool_wildcard(self):
        """通配符匹配所有工具"""
        rule = PermissionRule(effect=PermissionEffect.DENY, tool_pattern="*")
        assert rule.matches_tool("anything") is True

    def test_matches_path_exact(self):
        """匹配路径"""
        rule = PermissionRule(effect=PermissionEffect.DENY, tool_pattern="*", path_pattern="/etc/*")
        assert rule.matches_path("/etc/passwd") is True
        assert rule.matches_path("/home/user/file.py") is False

    def test_matches_path_empty_pattern(self):
        """空路径模式匹配所有路径"""
        rule = PermissionRule(effect=PermissionEffect.ALLOW, tool_pattern="read_file", path_pattern="")
        # 空路径模式匹配所有路径（包括空路径）
        assert rule.matches_path("/any/path") is True
        assert rule.matches_path("") is True  # 空 path_pattern 匹配一切

    def test_matches_path_double_star(self):
        """** glob 匹配任意目录"""
        rule = PermissionRule(effect=PermissionEffect.DENY, tool_pattern="*", path_pattern="**/.env*")
        assert rule.matches_path("/home/user/.env") is True
        assert rule.matches_path("/project/.env.local") is True

    def test_priority_user_highest(self):
        """用户级规则优先级最高"""
        user_rule = PermissionRule(effect=PermissionEffect.ALLOW, tool_pattern="*", source="user")
        system_rule = PermissionRule(effect=PermissionEffect.DENY, tool_pattern="*", source="system")
        assert user_rule.priority > system_rule.priority

    def test_priority_order(self):
        """优先级: user > config > project > system > default"""
        priorities = {
            "user": 100, "config": 60, "project": 50, "system": 10, "default": 0,
        }
        for source, expected_priority in priorities.items():
            rule = PermissionRule(effect=PermissionEffect.ALLOW, tool_pattern="*", source=source)
            assert rule.priority == expected_priority

    def test_to_dict_roundtrip(self):
        """to_dict / from_dict 往返"""
        original = PermissionRule(
            effect=PermissionEffect.DENY,
            tool_pattern="read_file",
            path_pattern="/etc/*",
            description="Block system files",
            source="user",
        )
        d = original.to_dict()
        restored = PermissionRule.from_dict(d)
        assert restored.effect == original.effect
        assert restored.tool_pattern == original.tool_pattern
        assert restored.path_pattern == original.path_pattern
        assert restored.source == original.source

    def test_from_dict_invalid_effect(self):
        """from_dict 无效效果默认 DENY"""
        d = {"effect": "invalid", "tool_pattern": "*"}
        rule = PermissionRule.from_dict(d)
        assert rule.effect == PermissionEffect.DENY


# ===== EnhancedPermissionManager =====

class TestEnhancedPermissionManager:
    """增强版权限管理器"""

    def test_no_rules_default_deny(self):
        """无规则时默认策略拒绝（DEFAULT级别下）
        
        注意：M2修复后，PLAN级别允许SAFE_WRITE操作。
        此测试使用DEFAULT级别（0）验证无规则时的默认拒绝行为。
        """
        epm = EnhancedPermissionManager()
        # 默认级别为PLAN(1)，M2修复后PLAN允许操作
        # 将级别降为DEFAULT(0)来测试真正的"无规则默认拒绝"
        epm._base_manager.current_level = PermissionLevel.DEFAULT
        allowed, reason = epm.check("any_tool")
        assert allowed is False
        assert "default policy" in reason.lower() or "默认" in reason

    def test_allow_rule(self):
        """ALLOW 规则允许操作"""
        epm = EnhancedPermissionManager()
        epm.add_rule(PermissionRule(
            effect=PermissionEffect.ALLOW,
            tool_pattern="read_file",
            source="system",
        ))
        allowed, reason = epm.check("read_file")
        assert allowed is True

    def test_deny_rule(self):
        """DENY 规则拒绝操作"""
        epm = EnhancedPermissionManager()
        epm.add_rule(PermissionRule(
            effect=PermissionEffect.DENY,
            tool_pattern="execute_subtask",
            source="user",
        ))
        allowed, reason = epm.check("execute_subtask")
        assert allowed is False

    def test_user_rule_overrides_system(self):
        """用户级规则覆盖系统级"""
        epm = EnhancedPermissionManager()
        epm.add_rule(PermissionRule(
            effect=PermissionEffect.DENY,
            tool_pattern="read_file",
            path_pattern="/etc/*",
            source="system",
        ))
        epm.add_rule(PermissionRule(
            effect=PermissionEffect.ALLOW,
            tool_pattern="read_file",
            path_pattern="/etc/hostname",
            source="user",
        ))
        # 用户级优先
        allowed, reason = epm.check("read_file", path="/etc/hostname")
        assert allowed is True

    def test_path_specific_rule_overrides_general(self):
        """路径特定规则优先于一般规则"""
        epm = EnhancedPermissionManager()
        epm.add_rule(PermissionRule(
            effect=PermissionEffect.ALLOW,
            tool_pattern="read_file",
            source="system",
        ))
        epm.add_rule(PermissionRule(
            effect=PermissionEffect.DENY,
            tool_pattern="read_file",
            path_pattern="/etc/*",
            source="system",
        ))
        # 有路径的规则更具体，DENY 优先于同优先级 ALLOW
        allowed, reason = epm.check("read_file", path="/etc/passwd")
        assert allowed is False

    def test_auto_level_allows_all(self):
        """AUTO 级别允许所有操作"""
        epm = EnhancedPermissionManager(default_level=PermissionLevel.AUTO)
        allowed, reason = epm.check("any_tool")
        assert allowed is True

    def test_check_tool_params(self):
        """check_tool_params 从参数中提取路径"""
        epm = EnhancedPermissionManager()
        epm.add_rule(PermissionRule(
            effect=PermissionEffect.DENY,
            tool_pattern="read_file",
            path_pattern="/etc/*",
            source="system",
        ))
        allowed, reason = epm.check_tool_params("read_file", {"file_path": "/etc/passwd"})
        assert allowed is False

    def test_add_rules_batch(self):
        """批量添加规则"""
        epm = EnhancedPermissionManager()
        rules = [
            PermissionRule(effect=PermissionEffect.ALLOW, tool_pattern="read_*", source="system"),
            PermissionRule(effect=PermissionEffect.DENY, tool_pattern="write_*", source="system"),
        ]
        epm.add_rules(rules)
        assert len(epm._rules) == 2

    def test_remove_rules_by_source(self):
        """按来源移除规则"""
        epm = EnhancedPermissionManager()
        epm.add_rule(PermissionRule(effect=PermissionEffect.ALLOW, tool_pattern="a", source="user"))
        epm.add_rule(PermissionRule(effect=PermissionEffect.DENY, tool_pattern="b", source="system"))
        removed = epm.remove_rules_by_source("user")
        assert removed == 1
        assert len(epm._rules) == 1

    def test_clear_rules(self):
        """清除所有规则"""
        epm = EnhancedPermissionManager()
        epm.add_rule(PermissionRule(effect=PermissionEffect.ALLOW, tool_pattern="*", source="system"))
        epm.clear_rules()
        assert len(epm._rules) == 0

    def test_sorted_rules_cache(self):
        """4.2: 排序规则缓存"""
        epm = EnhancedPermissionManager()
        epm.add_rule(PermissionRule(effect=PermissionEffect.ALLOW, tool_pattern="*", source="system"))

        # 第一次调用：排序并缓存
        sorted1 = epm._get_sorted_rules()
        assert epm._rules_dirty is False

        # 第二次调用：使用缓存
        sorted2 = epm._get_sorted_rules()
        assert sorted1 is sorted2  # 同一个对象

        # 添加规则后缓存失效
        epm.add_rule(PermissionRule(effect=PermissionEffect.DENY, tool_pattern="x", source="user"))
        assert epm._rules_dirty is True
        sorted3 = epm._get_sorted_rules()
        assert sorted3 is not sorted2

    def test_load_from_config(self):
        """从配置加载规则"""
        epm = EnhancedPermissionManager()
        config = {
            "mode": "plan",
            "rules": {
                "allow": ["read_file:*", "explore_codebase:*"],
                "deny": ["*:**/.env*"],
            }
        }
        epm.load_from_config(config)
        assert epm.current_level == PermissionLevel.PLAN
        assert len(epm._rules) >= 3

    def test_default_rules(self):
        """默认规则集不为空"""
        rules = EnhancedPermissionManager.default_rules()
        assert len(rules) > 0

    def test_get_status_report(self):
        """状态报告"""
        epm = EnhancedPermissionManager()
        epm.add_rule(PermissionRule(effect=PermissionEffect.ALLOW, tool_pattern="a", source="system"))
        report = epm.get_status_report()
        assert "current_level" in report
        assert "total_rules" in report
        assert report["total_rules"] == 1

    def test_elevate_and_deactivate(self):
        """权限提升和降级"""
        epm = EnhancedPermissionManager()
        epm.elevate(PermissionLevel.AUTO)
        assert epm.current_level == PermissionLevel.AUTO
        epm.deactivate()
        assert epm.current_level == PermissionLevel.PLAN

    def test_reset(self):
        """重置所有状态"""
        epm = EnhancedPermissionManager()
        epm.add_rule(PermissionRule(effect=PermissionEffect.ALLOW, tool_pattern="*", source="user"))
        epm.elevate(PermissionLevel.AUTO)
        epm.reset()
        assert len(epm._rules) == 0
        assert epm.current_level == PermissionLevel.PLAN

    @pytest.mark.asyncio
    async def test_acheck_with_ai_classifier(self):
        """异步权限检查 + AI 分类器"""
        # 使用 DEFAULT 级别，避免权限级别回退直接通过
        epm = EnhancedPermissionManager(default_level=PermissionLevel.DEFAULT)

        # Mock AI 分类器
        class MockClassifier:
            async def classify(self, tool_name, params, context=""):
                return PermissionEffect.ALLOW, 0.9, "Looks safe"

        epm.ai_classifier = MockClassifier()

        # 无匹配规则 + DEFAULT 权限 → AI 分类器决定
        allowed, reason = await epm.acheck("some_tool", path="/src/main.py")
        assert allowed is True
        assert "AI classifier" in reason

    @pytest.mark.asyncio
    async def test_acheck_without_ai_classifier(self):
        """异步权限检查无 AI 分类器时回退默认策略"""
        epm = EnhancedPermissionManager(default_level=PermissionLevel.DEFAULT)
        allowed, reason = await epm.acheck("some_tool")
        assert allowed is False  # 默认 DENY
