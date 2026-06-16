"""teragent.pipeline.subagent_worker — TAP request assembly → execution → file extraction → safe write

Key changes:
    - Uses teragent.core.tap.TAPRequest
    - Uses teragent.pipeline.extractor.extract_files_from_response
    - Accepts teragent.core.provider.ModelProvider
    - Security modules imported from teragent/security/
    - Optional TAPTracer integration for structured tracing + DPO pair generation

This is a library core primitive: TAP request assembly → execute_tap →
file extraction → safe write → sandbox execution. No EventBus dependency.
"""
import asyncio
import logging
import re
from typing import TYPE_CHECKING

__all__ = [
    "SubAgentWorker",
]

from teragent.core.provider import ModelProvider
from teragent.core.tap import TAPRequest
from teragent.pipeline.extractor import extract_files_from_response

# Security modules — migrated to teragent/security/
from teragent.security.file_writer import write_files_safely
from teragent.security.permission import PermissionLevel, PermissionManager
from teragent.security.sandbox import execute_in_sandbox
from teragent.utils.exceptions import SandboxViolation
from teragent.utils.token_counter import estimate_tokens

if TYPE_CHECKING:
    from teragent.pipeline.tracing import TAPTracer
    from teragent.security.file_state import FileStateTracker

logger = logging.getLogger(__name__)

COMMAND_BLOCK_PATTERN = r'<command\s+cwd=["\'](.*?)["\']\s*>(.*?)</command>'


def extract_commands_from_response(content: str) -> list[dict]:
    """Extract <command> tags from TAP response."""
    matches = re.findall(COMMAND_BLOCK_PATTERN, content, re.DOTALL | re.IGNORECASE)
    return [{"cwd": m[0].strip(), "command": m[1].strip()} for m in matches]


def is_dangerous_command(cmd: str) -> bool:
    """Determine if a command is a dangerous operation.

    Uses the unified classify_command_risk() from sandbox module
    for consistent risk assessment with DangerousCommandHook.

    Returns True for DANGEROUS and CRITICAL risk levels.
    Also returns True for WARNING level (package installs) since
    those require user approval in SubAgentWorker.
    """
    from teragent.security.sandbox import CommandRiskLevel, classify_command_risk
    risk_level, _ = classify_command_risk(cmd)
    return risk_level in (CommandRiskLevel.DANGEROUS, CommandRiskLevel.CRITICAL, CommandRiskLevel.WARNING)


class SubAgentWorker:
    """Library-level sub-agent worker: TAP → execute → extract → write → sandbox.

    This is a core library primitive that performs the complete execution cycle
    for a single sub-task. No EventBus dependency.

    Phase 10 enhancement: Optional TAPTracer integration.
        - If a tracer is provided (via constructor or ModelProvider), all TAP
          requests and responses are recorded through TAPTracer for DPO pair
          generation.
        - If no tracer is provided, tracing is skipped (a debug log is emitted).

    Attributes:
        task_id: Task identifier (e.g., "1.1")
        design_md: Design document content
        plan_md: Plan document content
        task_desc: Task description
        code_summary: Dependency report / code summary
        model: ModelProvider instance for TAP execution
        workspace_root: Project root directory
        agent_md: Optional AGENT.md memory content
        perm_mgr: Optional permission manager for write safety
        file_state_tracker: Optional file state tracker for read-after-write contracts
        tracer: Optional TAPTracer for structured tracing (Phase 10)
    """

    def __init__(
        self,
        task_id: str,
        design_md: str,
        plan_md: str,
        task_desc: str,
        code_summary: str,
        model: ModelProvider,
        workspace_root: str,
        agent_md: str = "",
        perm_mgr: PermissionManager | None = None,
        file_state_tracker: "FileStateTracker | None" = None,
        tracer: "TAPTracer | None" = None,
    ) -> None:
        self.task_id = task_id
        self.design_md = design_md
        self.plan_md = plan_md
        self.task_desc = task_desc
        self.code_summary = code_summary
        self.model = model
        self.workspace_root = workspace_root
        self.agent_md = agent_md
        self.perm_mgr = perm_mgr
        self.file_state_tracker = file_state_tracker
        self.tracer = tracer

    def _get_tracer(self) -> "TAPTracer | None":
        """Get the tracer — either explicitly set or from ModelProvider (Phase 10)."""
        if self.tracer is not None:
            return self.tracer
        # Check if ModelProvider has a tracer attached
        return getattr(self.model, '_tracer', None)

    async def execute(self) -> dict:
        """Execute the sub-task: assemble TAP → execute → extract files → safe write → sandbox.

        Returns:
            Result dict with keys: task_id, files, executed_cmds, error
        """
        # Determine tracing strategy
        tracer = self._get_tracer()

        try:
            # Classify task type: command_only / code_generation / mixed
            COMMAND_KEYWORDS = ["安装", "install", "依赖", "dependency", "初始化", "initialize", "配置环境", "setup"]
            CODE_GEN_KEYWORDS = ["创建", "编写", "实现", "生成", "写入", "开发", "添加", "实现代码",
                                "create", "write", "implement", "generate", "develop", "add", "code"]

            has_command_kw = any(kw in self.task_desc.lower() for kw in COMMAND_KEYWORDS)
            has_codegen_kw = any(kw in self.task_desc.lower() for kw in CODE_GEN_KEYWORDS)

            if has_command_kw and not has_codegen_kw:
                task_type = "command_only"
            elif has_command_kw and has_codegen_kw:
                task_type = "mixed"
            else:
                task_type = "code_generation"

            # 2. Assemble TAP request
            intent = "command_execution" if task_type == "command_only" else "code_generation"
            tap_request = TAPRequest(
                meta={"task_id": self.task_id, "intent": intent},
                context={
                    "design": self.design_md,
                    "plan": self.plan_md,
                    "dependency_report": self.code_summary,
                    "memory": self.agent_md,
                    "project_root": self.workspace_root,
                },
                instruction=self.task_desc,
                constraints=[
                    "输出完整文件内容，严禁省略（# ...）或留 TODO 占位",
                    "Python 3.10+，用 logging 替代 print()，公开函数须有类型注解和 docstring",
                    "外部调用（I/O、网络、第三方库）须 try/except，禁止裸 except",
                    "需第三方包时用 <command> 输出 pip install 命令",
                ],
                output_format_hint=(
                    "<file path='相对路径'>完整代码</file> | "
                    "<command cwd='工作目录'>命令</command> | "
                    "禁止省略和废话"
                )
            )

            # Adjust constraints based on task type
            if task_type == "command_only":
                tap_request.constraints = [
                    "此任务是命令执行任务，不是代码生成任务",
                    "输出 <command> 标签包含需要执行的命令（如 pip install）",
                    "如果只需要执行命令而不需要创建文件，不需要输出 <file> 标签",
                    "如果没有文件需要创建，这是正常的 — 只输出命令即可",
                ]
                tap_request.output_format_hint = (
                    "<command cwd='工作目录'>命令</command> | "
                    "如果需要创建文件: <file path='相对路径'>完整代码</file> | "
                    "禁止省略和废话"
                )
            elif task_type == "mixed":
                tap_request.constraints.append(
                    "此任务同时需要安装依赖和生成代码，请先输出 <command> 标签安装依赖，再输出 <file> 标签生成代码"
                )

            # Skeleton task scope constraint
            SKELETON_KEYWORDS = ["骨架", "skeleton", "结构", "目录", "框架", "搭建", "空文件"]
            is_skeleton_task = any(kw in self.task_desc for kw in SKELETON_KEYWORDS)
            if is_skeleton_task:
                skeleton_constraint = (
                    "此任务仅创建项目骨架/目录结构，不要生成具体实现代码。"
                    "文件内容只需要基本的 import 声明和空函数/class 定义即可。"
                )
                tap_request.constraints.insert(0, skeleton_constraint)

            # Phase 10: Record TAP request via tracer
            trace_id = ""

            if tracer is not None:
                # Phase 10: Structured tracing via TAPTracer
                trace_id = await tracer.record_request(tap_request)
            else:
                logger.debug(f"No tracer configured for task {self.task_id}, skipping request trace")

            # 3. Execute TAP request
            prompt_tokens = estimate_tokens(str(tap_request))
            logger.info(f"Task {self.task_id} prompt tokens estimated: {prompt_tokens}")

            tap_response = await self.model.execute_tap(tap_request)

            # Phase 10: Record TAP response via tracer
            raw_text = tap_response.raw_text or ""  # Guard against None

            if tracer is not None:
                # Phase 10: Structured tracing via TAPTracer
                await tracer.record_response(
                    tap_response,
                    task_id=self.task_id,
                    trace_id=trace_id,
                    intent=intent,
                )
            else:
                logger.debug(f"No tracer configured for task {self.task_id}, skipping response trace")

            response_tokens = estimate_tokens(raw_text)
            logger.info(f"Task {self.task_id} response tokens estimated: {response_tokens}")
            logger.info(f"Task {self.task_id} usage: {tap_response.usage}")

            # If model returned empty content, report error
            if not raw_text.strip():
                return {"task_id": self.task_id, "files": [], "error": "Model returned empty content (possible fake tool call instead of text output)"}

            # 4. Extract files from raw text (pass task_id to avoid fallback filename collisions)
            files_dict = extract_files_from_response(raw_text, task_id=self.task_id)
            if not files_dict and not extract_commands_from_response(raw_text):
                return {"task_id": self.task_id, "files": [], "error": "No file blocks found"}

            # 5. Safe file writing (atomic write + FileStateTracker read-after-write contract)
            written_files: list[str] = []
            failed_files: list[str] = []
            if files_dict:
                written_files, failed_files = await write_files_safely(
                    files_dict,
                    self.workspace_root,
                    self.perm_mgr,
                    file_state_tracker=self.file_state_tracker,
                    writer_id=f"subagent_{self.task_id}",
                )
                if failed_files:
                    logger.warning(f"Partial write failure for task {self.task_id}: {failed_files}")

            # 6. Process command execution
            commands_list = extract_commands_from_response(raw_text)
            executed_cmds: list[dict] = []
            for cmd in commands_list:
                try:
                    # Dangerous command detection
                    if is_dangerous_command(cmd['command']):
                        if self.perm_mgr is None or not self.perm_mgr.check_level(PermissionLevel.BYPASS):
                            is_pip_install = bool(re.search(r'\bpip\b.*\binstall\b', cmd['command'], re.IGNORECASE))
                            if is_pip_install:
                                reason = (
                                    f"依赖安装命令需要用户审批: {cmd['command']}。"
                                    "请使用 /approve 命令批准，或 /approve --auto 允许所有后续安装命令。"
                                )
                            else:
                                reason = f"AI 试图执行高危命令: {cmd['command']}"
                            return {
                                "task_id": self.task_id,
                                "files": written_files,
                                "error": "PERMISSION_DENIED",
                                "pending_command": cmd['command'],
                                "reason": reason
                            }

                    code, output = await execute_in_sandbox(cmd['command'], cmd.get('cwd', self.workspace_root))
                    executed_cmds.append({"cmd": cmd['command'], "exit_code": code, "output": output})
                except SandboxViolation as e:
                    return {"task_id": self.task_id, "files": written_files, "error": str(e)}

            error_msg = None
            if failed_files:
                error_msg = f"Partial write failure: {failed_files}"

            return {
                "task_id": self.task_id,
                "files": written_files,
                "executed_cmds": executed_cmds,
                "error": error_msg
            }

        except Exception as e:
            logger.error(f"Worker execution failed for {self.task_id}: {e}", exc_info=True)
            return {"task_id": self.task_id, "files": [], "error": str(e)}
