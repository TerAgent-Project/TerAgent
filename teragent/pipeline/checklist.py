"""teragent.pipeline.checklist — Deterministic code checks (decoupled from Plan)

Key changes:
    - Plan dependency replaced by TaskInfo dataclass
    - run_deterministic_checks() accepts list[TaskInfo] instead of Plan
    - All check functions are standalone (no EventBus, no ModelProvider)
    - _check_file_conflicts() uses TaskInfo.output_files instead of Plan.tasks

Library design principle: deterministic checks are universal primitives that
any AI code tool needs — they should not depend on a specific Plan structure.
"""
import ast
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """Lightweight task information for checklist — decoupled from Plan

    This replaces the Plan object dependency with a simple data structure.
    Any caller can construct TaskInfo from their own task representation.

    Attributes:
        id: Task identifier (e.g., "1.1", "2.3")
        title: Task title/description
        status: Task status string — "completed" | "blocked" | "pending" | "skipped"
        output_files: List of output file paths declared by this task
    """

    id: str = ""
    title: str = ""
    status: str = "pending"  # "completed" | "blocked" | "pending" | "skipped"
    output_files: list[str] = field(default_factory=list)


def _scan_python_files(workspace_root: str) -> list[str]:
    """Recursively scan workspace for .py files, return relative path list."""
    py_files: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(workspace_root):
        rel_dir = os.path.relpath(dirpath, workspace_root)
        # Skip hidden directories and .agent trace directory
        parts = rel_dir.split(os.sep)
        if any(p.startswith('.') for p in parts if p != '.'):
            continue
        if '.agent' in parts:
            continue
        for fn in filenames:
            if fn.endswith('.py'):
                rel_path = os.path.join(rel_dir, fn).replace('\\', '/')
                if rel_path.startswith('./'):
                    rel_path = rel_path[2:]
                py_files.append(rel_path)
    return sorted(py_files)


def check_code_quality(workspace_root: str, py_files: list[str]) -> list[str]:
    """Check code quality: TODO, ellipsis, empty function bodies, print instead of logging.

    Args:
        workspace_root: Project root directory
        py_files: List of relative .py file paths

    Returns:
        List of issue strings with [WARN]/[FAIL] prefixes
    """
    issues: list[str] = []

    for rel_path in py_files:
        abs_path = os.path.join(workspace_root, rel_path)
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (IOError, UnicodeDecodeError):
            continue

        # Check TODO / FIXME
        for i, line in enumerate(content.split('\n'), 1):
            stripped = line.strip()
            if re.match(r'#\s*(TODO|FIXME|HACK|XXX)', stripped, re.IGNORECASE):
                issues.append(f"[WARN] {rel_path}:{i}: 包含 TODO/FIXME 标记")

        # Check ellipsis placeholder # ... or pass as function/class body
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    body = node.body
                    if len(body) == 1:
                        stmt = body[0]
                        if isinstance(stmt, ast.Pass):
                            issues.append(f"[WARN] {rel_path}:{node.lineno}: 函数 {node.name} 只有 pass")
                        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis:
                            issues.append(f"[FAIL] {rel_path}:{node.lineno}: 函数 {node.name} 只有 ... 占位")
                elif isinstance(node, ast.ClassDef):
                    body = node.body
                    if len(body) == 1:
                        stmt = body[0]
                        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis:
                            issues.append(f"[FAIL] {rel_path}:{node.lineno}: 类 {node.name} 只有 ... 占位")
        except SyntaxError:
            pass  # Syntax errors reported in import check

    return issues


def check_requirements(workspace_root: str) -> list[str]:
    """Check if requirements.txt exists and contains necessary dependencies.

    Args:
        workspace_root: Project root directory

    Returns:
        List of issue strings with [WARN] prefix
    """
    issues: list[str] = []
    req_path = os.path.join(workspace_root, 'requirements.txt')

    if not os.path.isfile(req_path):
        issues.append("[WARN] 未找到 requirements.txt（如无第三方依赖可忽略）")
    else:
        try:
            with open(req_path, 'r', encoding='utf-8') as f:
                reqs = f.read().strip()
            if not reqs:
                issues.append("[WARN] requirements.txt 为空")
        except IOError:
            pass

    return issues


def check_runnable(workspace_root: str) -> list[str]:
    """Actually run python main.py, verify it can start (5-second timeout).

    Args:
        workspace_root: Project root directory

    Returns:
        List of issue strings with [OK]/[WARN]/[FAIL] prefixes
    """
    issues: list[str] = []
    main_path = os.path.join(workspace_root, 'main.py')

    if not os.path.isfile(main_path):
        return issues  # Skip if entry file doesn't exist

    try:
        result = subprocess.run(
            [sys.executable, 'main.py'],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            error_lines = stderr.split('\n')
            key_error = ""
            for line in reversed(error_lines):
                if 'Error' in line or 'error' in line:
                    key_error = line.strip()
                    break
            if not key_error and error_lines:
                key_error = error_lines[-1].strip()[:200]

            issues.append(f"[FAIL] python main.py 退出码 {result.returncode}: {key_error[:200]}")
        else:
            issues.append("[OK] python main.py 启动成功（5秒内无崩溃）")
    except subprocess.TimeoutExpired:
        issues.append("[WARN] python main.py 5秒内未退出（可能是游戏循环，也可能是悬挂进程，请手动验证）")
    except Exception as e:
        issues.append(f"[FAIL] python main.py 无法执行: {e}")

    return issues


def check_file_conflicts(task_list: list[TaskInfo]) -> list[str]:
    """Check if multiple tasks declare the same output file (file exclusivity conflict).

    Args:
        task_list: List of TaskInfo objects

    Returns:
        List of issue strings with [WARN] prefix
    """
    issues: list[str] = []
    file_to_tasks: dict[str, list[str]] = {}

    for task in task_list:
        for f in task.output_files:
            f = f.strip()
            if f and f not in ('无', '无（控制台命令）', '无（控制台/游戏窗口）'):
                file_to_tasks.setdefault(f, []).append(task.id)

    for f, task_ids in file_to_tasks.items():
        if len(task_ids) > 1:
            issues.append(f"[WARN] 文件 {f} 被多个任务声明: {', '.join(task_ids)}")

    return issues


def check_fallback_files(workspace_root: str) -> list[str]:
    """Check for fallback/module/entry degraded files (indicates extractor filename failure).

    Phase 1.4.3+ degraded filename format:
    - fallback_{task_id}_{N}.py  (no filename hints at all)
    - module_{task_id}_{N}.py    (inferred as module file)
    - entry_{task_id}.py         (inferred as entry file)

    Args:
        workspace_root: Project root directory

    Returns:
        List of issue strings with [FAIL] prefix
    """
    issues: list[str] = []
    if not os.path.isdir(workspace_root):
        return issues  # Skip if workspace doesn't exist

    # Use regex instead of hardcoded prefixes to handle any task_id
    # (e.g., module_10_5_1.py, entry_10_5.py would be missed by
    # hardcoded ('module_1_', 'module_2_', 'module_3_') patterns)
    fallback_re = re.compile(r'^fallback_(?:output_)?\w+_\d+$')
    module_re = re.compile(r'^module_\w+_\d+$')
    entry_re = re.compile(r'^entry_\w+$')

    for entry in os.listdir(workspace_root):
        if not entry.endswith('.py'):
            continue
        basename = entry[:-3]  # Remove .py
        is_fallback = bool(fallback_re.match(basename))
        is_generic_module = bool(module_re.match(basename))
        is_generic_entry = bool(entry_re.match(basename))
        if is_fallback or is_generic_module or is_generic_entry:
            issues.append(
                f"[FAIL] 存在 {entry}: LLM 输出未使用 <file path='...'> 格式，"
                f"文件名提取使用了降级策略"
            )
    return issues


# Critical warning patterns for has_critical_warn detection
_CRITICAL_WARN_PATTERNS = [
    "被多个任务声明",       # File exclusivity conflict
    "未指定技术栈",        # Misunderstood requirements tech choice
    "空函数体",            # Actually FAIL but fallback check
    "只有 pass",          # Empty implementation
    "悬挂进程",            # Suspicious run verification
    "降级策略",            # File extraction failure
    "5秒内未退出",         # Timeout always treated as critical warning
]


def run_deterministic_checks(
    workspace_root: str,
    task_list: list[TaskInfo],
) -> tuple[str, dict]:
    """Run all deterministic checks, return (checklist markdown, structured data).

    No LLM dependency — pure code verification.

    Key change: uses list[TaskInfo] instead of Plan object,
    making this function fully decoupled from any specific Plan implementation.

    Args:
        workspace_root: Project root directory
        task_list: List of TaskInfo objects (decoupled from Plan)

    Returns:
        Tuple of (markdown checklist text, structured check data dict)
        Structured data contains: fail_count, warn_count, ok_count,
        has_critical_warn, issues list, needs_repair flag
    """
    py_files = _scan_python_files(workspace_root)

    all_lines: list[str] = []
    all_lines.append(f"# 代码审查报告\n")
    all_lines.append(f"Python 文件: {len(py_files)} 个\n")

    # Task completion statistics
    completed = sum(1 for t in task_list if t.status in ("completed", "skipped"))
    blocked = sum(1 for t in task_list if t.status == "blocked")
    total = len(task_list)
    all_lines.append(f"\n## 任务完成度: {completed}/{total}\n")
    for t in task_list:
        if t.status == "blocked":
            icon = "[!]"
        elif t.status in ("completed", "skipped"):
            icon = "[x]"
        else:
            icon = "[ ]"
        all_lines.append(f"- {icon} {t.id} {t.title}")
    if blocked > 0:
        all_lines.append(f"\n**警告**: {blocked} 个任务被权限拦截 (BLOCKED)，未执行。")

    # Check 1: Code quality
    all_lines.append(f"\n## 1. 代码质量检查\n")
    quality_issues = check_code_quality(workspace_root, py_files)
    if quality_issues:
        for issue in quality_issues:
            all_lines.append(f"- {issue}")
    else:
        all_lines.append("- [OK] 无 TODO/FIXME 标记，无空函数体或省略占位")

    # Check 2: Dependency management
    all_lines.append(f"\n## 2. 依赖管理检查\n")
    req_issues = check_requirements(workspace_root)
    if req_issues:
        for issue in req_issues:
            all_lines.append(f"- {issue}")
    else:
        all_lines.append("- [OK] requirements.txt 存在且非空")

    # Check 3: Fallback files
    all_lines.append(f"\n## 3. 文件提取检查\n")
    fallback_issues = check_fallback_files(workspace_root)
    if fallback_issues:
        for issue in fallback_issues:
            all_lines.append(f"- {issue}")
    else:
        all_lines.append("- [OK] 无 fallback_output.py（所有文件名提取正确）")

    # Check 4: Task file exclusivity
    all_lines.append(f"\n## 4. 任务文件排他性检查\n")
    conflict_issues = check_file_conflicts(task_list)
    if conflict_issues:
        for issue in conflict_issues:
            all_lines.append(f"- {issue}")
    else:
        all_lines.append("- [OK] 无文件排他冲突")

    # Check 5: Runnability (last because it may launch GUI)
    all_lines.append(f"\n## 5. 可运行性验证\n")
    run_issues = check_runnable(workspace_root)
    for issue in run_issues:
        all_lines.append(f"- {issue}")

    # Summary
    fail_count = sum(1 for l in all_lines if '[FAIL]' in l)
    warn_count = sum(1 for l in all_lines if '[WARN]' in l)
    ok_count = sum(1 for l in all_lines if '[OK]' in l)

    has_critical_warn = any(
        any(pat in l for pat in _CRITICAL_WARN_PATTERNS)
        for l in all_lines if '[WARN]' in l
    )

    # Extract structured issues list for auto-repair
    issues_list: list[dict] = []
    for l in all_lines:
        issue_type = ""
        if '[FAIL]' in l:
            issue_type = "FAIL"
        elif '[WARN]' in l and any(pat in l for pat in _CRITICAL_WARN_PATTERNS):
            issue_type = "CRITICAL_WARN"
        else:
            continue

        # Try to extract file path from line
        file_match = re.match(r'-\s*\[(?:FAIL|WARN)\]\s*([^:]+)', l)
        file_path = file_match.group(1).strip() if file_match else ""
        # Remove line number suffix
        file_path = re.sub(r':\d+$', '', file_path).strip()

        # Extract description (remove markers and file path prefix)
        desc = l.lstrip('- ').strip()

        issues_list.append({
            "type": issue_type,
            "file": file_path,
            "description": desc,
        })

    # Take snapshot of FAIL lines BEFORE appending summary (which also contains [FAIL])
    fail_lines_snapshot = [l for l in all_lines if '[FAIL]' in l]

    all_lines.append(f"\n## 总结\n")
    if fail_count == 0 and not has_critical_warn and warn_count <= 3:
        all_lines.append(f"**通过** | [OK] {ok_count}  [WARN] {warn_count}  [FAIL] {fail_count}")
        if warn_count > 0:
            all_lines.append("\n有警告项，建议修复但不阻塞交付。")
    else:
        verdict = "需修复" if (fail_count > 0 or has_critical_warn or warn_count > 3) else "建议修复"
        all_lines.append(f"**{verdict}** | [OK] {ok_count}  [WARN] {warn_count}  [FAIL] {fail_count}")
        if has_critical_warn:
            all_lines.append("\n存在关键警告（文件冲突/空实现/超时/可疑运行结果），已触发自动修复。")
        if fail_count > 0:
            all_lines.append("\n关键问题:")
            # Snapshot already taken before summary
            for l in fail_lines_snapshot:
                all_lines.append(f"  - {l.lstrip('- ')}")
        elif warn_count > 3:
            all_lines.append(f"\n警告项较多（{warn_count} 个），已触发自动修复。")

    markdown = "\n".join(all_lines)
    structured_data = {
        "fail_count": fail_count,
        "warn_count": warn_count,
        "ok_count": ok_count,
        "has_critical_warn": has_critical_warn,
        "issues": issues_list,
        "needs_repair": fail_count > 0 or has_critical_warn or warn_count > 3,
    }
    return markdown, structured_data
