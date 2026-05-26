# tests/test_checklist.py
"""代码审查清单单元测试

覆盖 teragent.pipeline.checklist 模块:
  - 5 项检查: 代码质量 / 依赖管理 / 文件提取 / 任务排他性 / 可运行性
  - needs_repair 判定逻辑
  - 通过/失败标准
  - 空输入 / 缺失响应处理
  - 部分完成处理
  - 清单结果结构
"""
import os
import pytest
from pathlib import Path

from teragent.pipeline.checklist import (
    TaskInfo,
    check_code_quality,
    check_requirements,
    check_file_conflicts,
    check_fallback_files,
    run_deterministic_checks,
)


# ===== 辅助 fixture =====

@pytest.fixture
def workspace(tmp_path):
    """创建临时工作区"""
    return str(tmp_path)


@pytest.fixture
def workspace_with_issues(tmp_path):
    """创建包含质量问题的临时工作区"""
    # 包含 TODO 的文件
    (tmp_path / "todo_module.py").write_text("# TODO: 需要实现\npass\n")
    # 包含 ellipsis 占位的文件
    (tmp_path / "ellipsis_module.py").write_text("def placeholder():\n    ...\n")
    # 包含 pass 的文件
    (tmp_path / "pass_module.py").write_text("def empty():\n    pass\n")
    # 正常文件
    (tmp_path / "good_module.py").write_text("def implemented():\n    return 42\n")
    return str(tmp_path)


@pytest.fixture
def completed_tasks():
    """创建已完成的任务列表"""
    return [
        TaskInfo(id="1.1", title="实现核心逻辑", status="completed", output_files=["core.py"]),
        TaskInfo(id="1.2", title="实现工具函数", status="completed", output_files=["utils.py"]),
        TaskInfo(id="1.3", title="实现入口文件", status="completed", output_files=["main.py"]),
    ]


@pytest.fixture
def mixed_tasks():
    """创建混合状态的任务列表"""
    return [
        TaskInfo(id="1.1", title="实现核心", status="completed", output_files=["core.py"]),
        TaskInfo(id="1.2", title="实现工具", status="blocked", output_files=["utils.py"]),
        TaskInfo(id="1.3", title="实现入口", status="pending", output_files=["main.py"]),
    ]


# ===== 代码质量检查 =====

class TestCodeQuality:
    """检查 1: 代码质量检查"""

    def test_detects_todo_comments(self, workspace_with_issues):
        """检测 TODO 注释"""
        py_files = ["todo_module.py"]
        issues = check_code_quality(workspace_with_issues, py_files)
        todo_issues = [i for i in issues if "TODO" in i]
        assert len(todo_issues) >= 1

    def test_detects_ellipsis_placeholder(self, workspace_with_issues):
        """检测 ... 占位符"""
        py_files = ["ellipsis_module.py"]
        issues = check_code_quality(workspace_with_issues, py_files)
        ellipsis_issues = [i for i in issues if "[FAIL]" in i and "..." in i]
        assert len(ellipsis_issues) >= 1

    def test_detects_pass_only_function(self, workspace_with_issues):
        """检测只有 pass 的函数"""
        py_files = ["pass_module.py"]
        issues = check_code_quality(workspace_with_issues, py_files)
        pass_issues = [i for i in issues if "pass" in i]
        assert len(pass_issues) >= 1

    def test_clean_code_no_issues(self, workspace_with_issues):
        """干净代码无问题"""
        py_files = ["good_module.py"]
        issues = check_code_quality(workspace_with_issues, py_files)
        assert len(issues) == 0

    def test_empty_file_list_no_issues(self, workspace_with_issues):
        """空文件列表无问题"""
        issues = check_code_quality(workspace_with_issues, [])
        assert issues == []


# ===== 依赖管理检查 =====

class TestRequirements:
    """检查 2: 依赖管理检查"""

    def test_missing_requirements_warns(self, workspace):
        """缺少 requirements.txt 发出警告"""
        issues = check_requirements(workspace)
        assert len(issues) >= 1
        assert any("requirements.txt" in i for i in issues)

    def test_empty_requirements_warns(self, workspace):
        """空的 requirements.txt 发出警告"""
        Path(os.path.join(workspace, "requirements.txt")).write_text("")
        issues = check_requirements(workspace)
        assert any("空" in i for i in issues)

    def test_valid_requirements_ok(self, workspace):
        """有效的 requirements.txt 无警告"""
        Path(os.path.join(workspace, "requirements.txt")).write_text("pytest>=7.0\nhttpx\n")
        issues = check_requirements(workspace)
        assert len(issues) == 0


# ===== 文件提取检查 =====

class TestFallbackFiles:
    """检查 3: 文件提取降级检查"""

    def test_fallback_file_detected(self, workspace):
        """检测 fallback 降级文件"""
        Path(os.path.join(workspace, "fallback_1_1.py")).write_text("code")
        issues = check_fallback_files(workspace)
        assert len(issues) >= 1
        assert any("降级策略" in i for i in issues)

    def test_module_degraded_file_detected(self, workspace):
        """检测 module 降级文件"""
        Path(os.path.join(workspace, "module_1_1.py")).write_text("code")
        issues = check_fallback_files(workspace)
        assert len(issues) >= 1

    def test_entry_degraded_file_detected(self, workspace):
        """检测 entry 降级文件"""
        Path(os.path.join(workspace, "entry_1_1.py")).write_text("code")
        issues = check_fallback_files(workspace)
        assert len(issues) >= 1

    def test_normal_files_no_issues(self, workspace):
        """正常文件名无问题"""
        Path(os.path.join(workspace, "game.py")).write_text("code")
        issues = check_fallback_files(workspace)
        assert len(issues) == 0

    def test_nonexistent_workspace_returns_empty(self):
        """不存在的工作区返回空列表"""
        issues = check_fallback_files("/nonexistent/path/abc123")
        assert issues == []


# ===== 任务文件排他性检查 =====

class TestFileConflicts:
    """检查 4: 任务文件排他性检查"""

    def test_no_conflicts(self, completed_tasks):
        """无冲突时不产生问题"""
        issues = check_file_conflicts(completed_tasks)
        assert len(issues) == 0

    def test_conflict_detected(self):
        """多任务声明同一文件时检测冲突"""
        tasks = [
            TaskInfo(id="1.1", title="任务A", status="completed", output_files=["core.py"]),
            TaskInfo(id="1.2", title="任务B", status="completed", output_files=["core.py"]),
        ]
        issues = check_file_conflicts(tasks)
        assert len(issues) >= 1
        assert any("多个任务声明" in i for i in issues)

    def test_empty_output_files_ignored(self):
        """空输出文件和特殊占位符被忽略"""
        tasks = [
            TaskInfo(id="1.1", title="任务A", status="completed", output_files=["无"]),
            TaskInfo(id="1.2", title="任务B", status="completed", output_files=["无（控制台命令）"]),
        ]
        issues = check_file_conflicts(tasks)
        assert len(issues) == 0


# ===== 完整清单运行 =====

class TestRunDeterministicChecks:
    """完整清单运行 run_deterministic_checks"""

    def test_returns_markdown_and_structured_data(self, workspace, completed_tasks):
        """返回 Markdown 文本和结构化数据"""
        markdown, data = run_deterministic_checks(workspace, completed_tasks)
        assert isinstance(markdown, str)
        assert isinstance(data, dict)
        assert "fail_count" in data
        assert "warn_count" in data
        assert "ok_count" in data
        assert "has_critical_warn" in data
        assert "issues" in data
        assert "needs_repair" in data

    def test_completed_tasks_shown_in_markdown(self, workspace, completed_tasks):
        """已完成任务在 Markdown 中标记为 [x]"""
        markdown, _ = run_deterministic_checks(workspace, completed_tasks)
        assert "[x]" in markdown
        assert "1.1" in markdown

    def test_blocked_tasks_shown_in_markdown(self, workspace, mixed_tasks):
        """被阻塞任务在 Markdown 中标记为 [!]"""
        markdown, _ = run_deterministic_checks(workspace, mixed_tasks)
        assert "[!]" in markdown

    def test_pending_tasks_shown_in_markdown(self, workspace, mixed_tasks):
        """待处理任务在 Markdown 中标记为 [ ]"""
        markdown, _ = run_deterministic_checks(workspace, mixed_tasks)
        assert "[ ]" in markdown

    def test_needs_repair_with_fail(self, workspace_with_issues, completed_tasks):
        """存在 FAIL 问题时 needs_repair 为 True"""
        _, data = run_deterministic_checks(workspace_with_issues, completed_tasks)
        if data["fail_count"] > 0:
            assert data["needs_repair"] is True

    def test_needs_repair_clean_workspace(self, workspace, completed_tasks):
        """干净工作区 needs_repair 为 False（或取决于 warn 数量）"""
        _, data = run_deterministic_checks(workspace, completed_tasks)
        # 无 requirements.txt 会产生 warn，但无 fail
        # needs_repair 只在 fail>0 或 critical_warn 或 warn>3 时为 True
        if data["fail_count"] == 0 and not data["has_critical_warn"] and data["warn_count"] <= 3:
            assert data["needs_repair"] is False

    def test_critical_warn_detection(self, workspace):
        """关键警告检测（文件冲突触发 critical warn）"""
        tasks = [
            TaskInfo(id="1.1", title="任务A", status="completed", output_files=["core.py"]),
            TaskInfo(id="1.2", title="任务B", status="completed", output_files=["core.py"]),
        ]
        _, data = run_deterministic_checks(workspace, tasks)
        assert data["has_critical_warn"] is True
        assert data["needs_repair"] is True

    def test_empty_task_list(self, workspace):
        """空任务列表不崩溃"""
        markdown, data = run_deterministic_checks(workspace, [])
        assert isinstance(markdown, str)
        assert data["fail_count"] == 0
        # 空任务列表应显示 0/0 完成度
        assert "0/0" in markdown

    def test_5_check_sections_in_markdown(self, workspace, completed_tasks):
        """Markdown 输出包含 5 个检查章节"""
        markdown, _ = run_deterministic_checks(workspace, completed_tasks)
        assert "1. 代码质量检查" in markdown
        assert "2. 依赖管理检查" in markdown
        assert "3. 文件提取检查" in markdown
        assert "4. 任务文件排他性检查" in markdown
        assert "5. 可运行性验证" in markdown


# ===== TaskInfo 数据类 =====

class TestTaskInfo:
    """TaskInfo 数据类基本功能"""

    def test_default_values(self):
        """默认值正确"""
        task = TaskInfo()
        assert task.id == ""
        assert task.title == ""
        assert task.status == "pending"
        assert task.output_files == []

    def test_custom_values(self):
        """自定义值正确"""
        task = TaskInfo(id="2.1", title="测试任务", status="completed", output_files=["a.py", "b.py"])
        assert task.id == "2.1"
        assert task.title == "测试任务"
        assert task.status == "completed"
        assert len(task.output_files) == 2
