# teragent/security/sandbox.py
"""安全沙箱执行 — Phase 1 增强版

增强内容:
  1.1 增强黑名单检查：8 大类危险模式 + 管道链拆分 + 命令规范化
  1.2 优先使用 create_subprocess_exec 避免命令注入
  1.3 Docker 模式 Shell 转义（shlex.quote）
  1.4 进程组杀灭，消除孤儿进程
"""
import asyncio
import logging
import os
import re
import shlex
import shutil
import signal
import sys

__all__ = [
    "BLOCKLIST_PATTERNS",
    "CommandRiskLevel",
    "DEFAULT_TIMEOUT",
    "MAX_OUTPUT_SIZE",
    "check_command_safety",
    "classify_command_risk",
    "execute_in_sandbox",
]

from teragent.utils.exceptions import SandboxViolation

logger = logging.getLogger(__name__)

# resource 模块仅 Unix 可用；Windows 上跳过 rlimit 设置
_HAS_RESOURCE = sys.platform != "win32"
if _HAS_RESOURCE:
    import resource

# ============================================================
# 增强黑名单模式：拒绝已知危险，放行未知（非白名单）
# 每条规则均为独立正则，匹配即拒绝
# ============================================================

# --- 特权提升 ---
_PRIVILEGE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'(^|\s|;|&&|\|\|)\s*(\\?/?usr/?bin/)?sudo\s', re.IGNORECASE),
    re.compile(r'(^|\s|;|&&|\|\|)\s*(\\?/?usr/?bin/)?su\s', re.IGNORECASE),
    re.compile(r'\bdoas\s', re.IGNORECASE),
    re.compile(r'\bpkexec\s', re.IGNORECASE),
    re.compile(r'\b(chmod|chown|chgrp)\s', re.IGNORECASE),
]

# --- 反向 Shell / 网络后门 ---
_REVERSE_SHELL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'\b(nc|ncat|socat|netcat)\s', re.IGNORECASE),
    re.compile(r'\b(/dev/tcp|/dev/udp)/', re.IGNORECASE),
    re.compile(r'\b(ssh)\s+-[DRL]\s', re.IGNORECASE),  # ssh 隧道/端口转发
    re.compile(r'\b(telnet)\s', re.IGNORECASE),
]

# --- 内联脚本执行（-c / -e） ---
_INLINE_EXEC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'\b(python[23]?(?:\.\d+)?)\s+-[ce]\s', re.IGNORECASE),
    re.compile(r'\b(perl)\s+-[ce]\s', re.IGNORECASE),
    re.compile(r'\b(ruby)\s+-[ce]\s', re.IGNORECASE),
    re.compile(r'\b(node)\s+-[ce]\s', re.IGNORECASE),
    re.compile(r'\b(bash|sh|zsh|dash|ksh)\s+-[ce]\s', re.IGNORECASE),
    re.compile(r'\benv\s+\w+\s+-[ce]\s', re.IGNORECASE),  # env python -c
]

# --- 系统破坏 ---
_SYSTEM_DESTROY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+/|.*-rf\s+/)', re.IGNORECASE),  # rm -rf /
    re.compile(r'\bmkfs(\.|\s)', re.IGNORECASE),
    re.compile(r'\bdd\s+(if=|of=)', re.IGNORECASE),
    re.compile(r'\b(shutdown|reboot|init\s+[06])\b', re.IGNORECASE),
    re.compile(r'\b:\(\)\s*\{', re.IGNORECASE),  # fork bomb
    re.compile(r'\b(systemctl|service)\s+(stop|disable|mask)\b', re.IGNORECASE),
]

# --- 定时任务 / 持久化 ---
_PERSISTENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'\bcrontab\b', re.IGNORECASE),
    re.compile(r'\bat\s+(now|\d)', re.IGNORECASE),
    re.compile(r'\blaunchctl\s+(load|install)', re.IGNORECASE),
]

# --- 编码绕过 ---
_ENCODING_BYPASS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'\b(base64|b64)\s+(-d|--decode)', re.IGNORECASE),
    re.compile(r'\b(xxd)\s+(-r|--revert)', re.IGNORECASE),
    re.compile(r'\b(od)\s+(-A\s*n|-t\s*x)', re.IGNORECASE),
    re.compile(r'\bpython[23]?\s+.*\\x[0-9a-f]{2}', re.IGNORECASE),  # \x41 绕过
]

# --- 远程脚本执行 ---
_REMOTE_EXEC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'(curl|wget|fetch)\s+.*\|\s*(ba)?sh', re.IGNORECASE),
    re.compile(r'(curl|wget|fetch)\s+.*>\s*/tmp/', re.IGNORECASE),  # 下载到临时再执行
    re.compile(r'\beval\s', re.IGNORECASE),
    re.compile(r'\bsource\s+(/tmp|/dev/shm|/var/tmp)', re.IGNORECASE),
]

# --- 写入系统关键路径 ---
_DANGER_REDIRECT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'[>]\s*/(etc|dev|sys|proc|boot|root|sbin|usr)/', re.IGNORECASE),
    re.compile(r'\b(tee|dd|cp|mv|install)\s+.*/(etc|dev|sys|proc|boot)/', re.IGNORECASE),
    re.compile(r'\bmount\s', re.IGNORECASE),
    re.compile(r'\bumount\s', re.IGNORECASE),
]

# --- Windows 危险命令（仅 Windows 平台生效） ---
_WINDOWS_DANGEROUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'\bformat\s+[A-Za-z]:', re.IGNORECASE),           # 格式化磁盘
    re.compile(r'\bdel\s+/[sS]', re.IGNORECASE),                   # 递归删除
    re.compile(r'\brd\s+/[sS]', re.IGNORECASE),                    # 递归删除目录
    re.compile(r'\breg\s+(delete|add)\s+', re.IGNORECASE),         # 注册表操作
    re.compile(r'\bnet\s+user\b', re.IGNORECASE),                  # 用户管理
    re.compile(r'\bnet\s+localgroup\b', re.IGNORECASE),            # 用户组管理
    re.compile(r'\bpowershell\s+-(enc|encodedcommand)\s', re.IGNORECASE),  # 编码执行绕过
    re.compile(r'\bpwsh\s+-(enc|encodedcommand)\s', re.IGNORECASE),
    re.compile(r'\bdiskpart\b', re.IGNORECASE),                    # 磁盘分区
    re.compile(r'\bcipher\s+/w:', re.IGNORECASE),                  # 安全擦除
    re.compile(r'\btaskkill\b', re.IGNORECASE),                    # 进程杀灭
    re.compile(r'\bnetsh\b', re.IGNORECASE),                       # 网络配置
    re.compile(r'\btakeown\b', re.IGNORECASE),                     # 取得所有权
    re.compile(r'\bicacls\s+.*\/grant\b', re.IGNORECASE),          # 修改权限
    re.compile(r'\bwmic\b', re.IGNORECASE),                        # WMI 命令
    re.compile(r'\bschtasks\s+/(create|delete)\b', re.IGNORECASE), # 计划任务
]

# --- 包安装（需审批，但不硬拦截，仅警告） ---
_PACKAGE_INSTALL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'\b(pip|pip3|conda)\s+install\b', re.IGNORECASE),
    re.compile(r'\b(npm|yarn|pnpm)\s+(install|add|i)\b', re.IGNORECASE),
    re.compile(r'\b(apt|apt-get|yum|dnf|pacman|brew)\s+install\b', re.IGNORECASE),
]

# --- 跨管道链危险组合（需在完整命令中检测，拆分后各段不可见） ---
_CROSS_CHAIN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r'(curl|wget|fetch)\s+.*\|\s*(ba)?sh', re.IGNORECASE),
        "远程脚本执行: 下载并管道到 shell",
    ),
    (
        re.compile(r'(curl|wget|fetch)\s+.*\|\s*(python|perl|ruby|node)', re.IGNORECASE),
        "远程脚本执行: 下载并管道到解释器",
    ),
    # Generic pipe-to-shell and pipe-to-interpreter patterns
    (
        re.compile(r'\|\s*(ba)?sh\b', re.IGNORECASE),
        "管道到 shell 解释器: 任意命令可通过管道注入",
    ),
    (
        re.compile(r'\|\s*(python[23]?|perl|ruby|node)\b', re.IGNORECASE),
        "管道到解释器: 任意代码可通过管道注入",
    ),
]

# 合并所有硬拦截规则（包安装不在此列，单独处理）
BLOCKLIST_PATTERNS: list[re.Pattern[str]] = (
    _PRIVILEGE_PATTERNS
    + _REVERSE_SHELL_PATTERNS
    + _INLINE_EXEC_PATTERNS
    + _SYSTEM_DESTROY_PATTERNS
    + _PERSISTENCE_PATTERNS
    + _ENCODING_BYPASS_PATTERNS
    + _REMOTE_EXEC_PATTERNS
    + _DANGER_REDIRECT_PATTERNS
    + (_WINDOWS_DANGEROUS_PATTERNS if sys.platform == "win32" else [])
)

MAX_OUTPUT_SIZE = 1_048_576  # 1MB
DEFAULT_TIMEOUT = 60  # seconds

# Resource limits for Level 0 subprocesses (Unix only)
_RLIMIT_NOFILE = 256        # Max open file descriptors
_RLIMIT_FSIZE = 50 * 1024 * 1024  # Max file size: 50MB
_RLIMIT_NPROC = 64          # Max processes


# ===== 命令安全检查函数 =====

def _normalize_command(cmd: str) -> str:
    """命令规范化：消除常见绕过手法

    处理:
      - 去除 ANSI 转义序列
      - 去除 null 字节
      - 压缩连续空白
    """
    # 去除 ANSI 转义
    cmd = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', cmd)
    # 去除 null 字节
    cmd = cmd.replace('\x00', '')
    # 压缩连续空白
    cmd = re.sub(r'\s+', ' ', cmd).strip()
    return cmd


def _split_command_chain(cmd: str) -> list[str]:
    """将管道链拆分为独立子命令

    对 |、&&、||、; 连接的每个子命令独立检查，
    防止通过管道链绕过黑名单。

    注意：不拆分单 &（后台运行），因为 & 不是命令链操作符。
    单 & 的语义是“在后台运行”，拆分会改变命令含义。
    """
    parts = re.split(r'\s*(?:\|\|?|&&|;)\s*', cmd)
    return [p.strip() for p in parts if p.strip()]


class CommandRiskLevel:
    """Unified command risk classification — shared by Hook and SubAgentWorker."""
    SAFE = "safe"               # 安全命令
    WARNING = "warning"         # 需警告但允许（包安装）
    DANGEROUS = "dangerous"     # 阻断
    CRITICAL = "critical"       # 严格阻断


def classify_command_risk(cmd: str) -> tuple[str, str]:
    """Unified command risk classification.

    Shared by DangerousCommandHook and SubAgentWorker to ensure
    consistent risk assessment across the system.

    Args:
        cmd: Shell command to classify

    Returns:
        (risk_level, reason) — risk_level is one of CommandRiskLevel values,
        reason is a human-readable description.
    """
    normalized = _normalize_command(cmd)
    sub_cmds = _split_command_chain(normalized)

    for sub_cmd in sub_cmds:
        # CRITICAL: blocklisted patterns (privilege escalation, reverse shell, etc.)
        for pattern in BLOCKLIST_PATTERNS:
            if pattern.search(sub_cmd):
                return CommandRiskLevel.CRITICAL, f"Blocked by pattern: {pattern.pattern}"

    # Check full command for cross-chain patterns
    for pattern, desc in _CROSS_CHAIN_PATTERNS:
        if pattern.search(normalized):
            return CommandRiskLevel.CRITICAL, desc

    # Check shell metacharacters (command substitution, backticks, etc.)
    _SHELL_META_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r'\$\('), "Command substitution $(...)"),
        (re.compile(r'`[^`]+`'), "Backtick command substitution"),
        (re.compile(r'<<'), "Heredoc injection"),
    ]
    for pattern, desc in _SHELL_META_PATTERNS:
        if pattern.search(normalized):
            return CommandRiskLevel.CRITICAL, desc

    # WARNING: package install patterns (allow but warn)
    for pattern in _PACKAGE_INSTALL_PATTERNS:
        if pattern.search(normalized):
            return CommandRiskLevel.WARNING, f"Package installation detected: {pattern.pattern}"

    return CommandRiskLevel.SAFE, ""


def check_command_safety(cmd: str) -> tuple[bool, str]:
    """检查命令安全性（增强黑名单 + 管道链拆分 + 命令规范化）

    6 层防御:
      Layer 1: 命令规范化 — 消除编码绕过
      Layer 2: 管道链拆分 — 逐段检查，防止通过管道链绕过
      Layer 3: 增强黑名单 — 8 大类危险模式
      Layer 4: 危险重定向检测 — 拒绝写入系统关键路径
      Layer 5: 跨管道链危险组合检测 — 检测 curl|sh 等跨段危险模式
      Layer 6: 包安装警告 — 仅日志，不硬拦截

    Args:
        cmd: 待检查的 shell 命令

    Returns:
        (is_safe, reason) — is_safe=True 表示安全，reason 为空串；
        is_safe=False 表示危险，reason 为拦截原因。
    """
    # Layer 1: 命令规范化
    normalized = _normalize_command(cmd)

    # Block background execution with bare &
    if re.search(r'(?<!&)&(?!&)', normalized):
        # Allow >& and 2>&1 redirect patterns
        cleaned = re.sub(r'\d*>&\d*', '', normalized)
        if re.search(r'(?<!&)&(?!&)', cleaned):
            return False, "Background execution (&) is not allowed"

    # Layer 2: 管道链拆分，逐段检查
    sub_cmds = _split_command_chain(normalized)

    for sub_cmd in sub_cmds:
        # Layer 3: 增强黑名单检查
        for pattern in BLOCKLIST_PATTERNS:
            if pattern.search(sub_cmd):
                return False, f"命令匹配危险模式: {pattern.pattern}"

        # Layer 4: 危险重定向精细化检测
        # Detect > target or >> target for any path (absolute or relative)
        redirect_match = re.search(r'>{1,2}\s*(\S+)', sub_cmd)
        if redirect_match:
            target_path = redirect_match.group(1)
            # Remove surrounding quotes if present
            target_path = target_path.strip('\'"')
            # Resolve relative paths against a hypothetical workspace
            target_path = os.path.normpath(target_path)

            # System prefixes to protect
            system_prefixes = ['/etc', '/dev', '/sys', '/proc', '/boot', '/root', '/sbin']
            if sys.platform == "win32":
                system_root = os.environ.get("SystemRoot", r"C:\Windows").lower()
                system_prefixes.extend([
                    system_root,
                    os.path.join(system_root, "system32"),
                    os.path.join(system_root, "system"),
                ])

            # Check if target is an absolute system path
            target_lower = target_path.lower() if sys.platform == "win32" else target_path
            for prefix in system_prefixes:
                prefix_lower = prefix.lower() if sys.platform == "win32" else prefix
                if target_lower.startswith(prefix_lower):
                    return False, f"重定向到系统关键路径: {target_path}"

            # Check for path traversal in relative redirects
            if '..' in target_path.split(os.sep):
                return False, f"重定向路径包含目录穿越: {target_path}"

    # Layer 5: 跨管道链的危险组合检测
    # 某些危险模式只在完整命令中可见（如 curl | sh），拆分后各段都不触发
    for pattern, desc in _CROSS_CHAIN_PATTERNS:
        if pattern.search(normalized):
            return False, desc

    # Layer 6: 包安装警告（不硬拦截，仅记录日志）
    for pattern in _PACKAGE_INSTALL_PATTERNS:
        if pattern.search(normalized):
            logger.warning(
                f"检测到包安装命令: {normalized[:100]}. "
                f"建议审批后再执行。"
            )
            # 不拦截，仅警告；如需硬拦截，将此段移到 BLOCKLIST_PATTERNS

    return True, ""


async def execute_in_sandbox(
    cmd: str,
    workdir: str,
    level: int = 0,
    timeout: float = DEFAULT_TIMEOUT,
    max_output_size: int = MAX_OUTPUT_SIZE,
) -> tuple[int, str]:
    """在安全沙箱中执行命令。

    Level 0: 增强黑名单拦截 + 子进程限制 + 进程组杀灭
    Level 1: Docker 容器隔离 + shlex.quote 转义
    Level 2: Firecracker microVM 隔离

    Args:
        cmd: The shell command to execute.
        workdir: Working directory for command execution.
        level: Sandbox isolation level (0, 1, or 2).
        timeout: Maximum execution time in seconds.
        max_output_size: Maximum output size in bytes before truncation.

    Returns:
        A tuple of (exit_code, stdout_text).

    Raises:
        SandboxViolation: If the command is blocked or times out.
    """
    # 1. 增强黑名单检查（管道链拆分 + 命令规范化）
    is_safe, reason = check_command_safety(cmd)
    if not is_safe:
        logger.error(f"SandboxViolation: {reason}")
        # 审计日志 (PLAN 5.4)
        try:
            from teragent.security.audit import log_audit
            await log_audit("command_blocked", f"Reason: {reason}, Cmd: {cmd[:100]}")
        except Exception as e:
            logger.debug(f"Audit logging for blocked command failed: {e}")
        raise SandboxViolation(reason)

    # 2. 工作目录验证
    abs_workdir = os.path.abspath(workdir)
    if not os.path.isdir(abs_workdir):
        raise SandboxViolation(f"Working directory does not exist: {workdir}")

    # 3. 根据隔离级别选择执行方式
    if level >= 2:
        try:
            from teragent.security.firecracker_sandbox import FirecrackerSandbox
            sandbox = FirecrackerSandbox(abs_workdir)
            return await sandbox.run(cmd, timeout=timeout)
        except ImportError:
            logger.warning("FirecrackerSandbox not available, falling back to Level 1")
            try:
                from teragent.security.audit import log_audit
                await log_audit(
                    "sandbox_downgrade",
                    f"Sandbox level downgraded from 2 to 1: FirecrackerSandbox not available (ImportError)"
                )
            except Exception:
                pass  # Don't fail the sandbox operation if audit logging fails
            level = 1
        except RuntimeError as e:
            logger.warning(f"Firecracker not available ({e}), falling back to Level 1")
            try:
                from teragent.security.audit import log_audit
                await log_audit(
                    "sandbox_downgrade",
                    f"Sandbox level downgraded from 2 to 1: Firecracker not available ({e})"
                )
            except Exception:
                pass  # Don't fail the sandbox operation if audit logging fails
            level = 1

    if level >= 1:
        if not shutil.which("docker"):
            logger.warning("Docker not found, falling back to Level 0")
            try:
                from teragent.security.audit import log_audit
                await log_audit(
                    "sandbox_downgrade",
                    "Sandbox level downgraded from 1 to 0: Docker not found"
                )
            except Exception:
                pass  # Don't fail the sandbox operation if audit logging fails
            return await _execute_level_0(cmd, abs_workdir, timeout, max_output_size)

        # 1.3: Docker 模式下使用 shlex.quote 转义命令，防止双重注入
        # 修复 H14: 使用唯一标签，避免清理时误杀其他实例的容器
        import uuid
        _docker_run_id = f"teragent_{uuid.uuid4().hex[:12]}"
        quoted_cmd = shlex.quote(cmd) if sys.platform != "win32" else f'"{cmd}"'
        docker_cmd = [
            "docker", "run", "--rm",
            "--network=none",
            f"--label=teragent_run={_docker_run_id}",
        ] + (
            [f"--user={os.getuid()}:{os.getgid()}"] if hasattr(os, "getuid")
            else (["--user", "ContainerUser"] if sys.platform == "win32" else [])
        ) + [
            "-v", f"{abs_workdir}:/workspace",
            "-w", "/workspace",
            "--memory=512m", "--cpus=1",
            "--pids-limit=64",
            f"--ulimit=nofile={_RLIMIT_NOFILE}",
            "python:3.10-slim",
            "bash", "-c", quoted_cmd,
        ]
        logger.info(f"Executing (Level 1 Docker): {cmd[:200]}")
        try:
            _docker_kwargs: dict = {}
            if sys.platform == "win32":
                import subprocess as _sp
                _docker_kwargs["creationflags"] = _sp.CREATE_NEW_PROCESS_GROUP
            else:
                _docker_kwargs["start_new_session"] = True
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **_docker_kwargs,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            output = _truncate_output(stdout.decode(errors='replace'), max_output_size)
            if process.returncode != 0:
                stderr_text = stderr.decode(errors='replace')[:500]
                logger.warning(f"Docker command failed (code={process.returncode}): {stderr_text}")
            return process.returncode or 0, output
        except asyncio.TimeoutError:
            # 1.4: 进程组杀灭
            _kill_process_group(process)
            # Force-remove the container to prevent orphaned containers after timeout.
            # 修复 H14: 使用唯一标签过滤，避免误杀其他实例的容器
            try:
                find_cmd = [
                    "docker", "ps", "-q",
                    f"--filter=label=teragent_run={_docker_run_id}",
                ]
                find_proc = await asyncio.create_subprocess_exec(
                    *find_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                find_stdout, _ = await asyncio.wait_for(find_proc.communicate(), timeout=5)
                container_ids = find_stdout.decode(errors='replace').strip().splitlines()
                for cid in container_ids:
                    rm_proc = await asyncio.create_subprocess_exec(
                        "docker", "rm", "-f", cid,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await asyncio.wait_for(rm_proc.communicate(), timeout=5)
                    logger.warning(f"Force-removed orphaned container {cid} after timeout")
            except Exception as cleanup_err:
                # Best-effort cleanup; do not mask the original timeout error
                logger.warning(
                    f"Failed to clean up Docker container after timeout: {cleanup_err}. "
                    f"Orphaned containers may remain — consider running 'docker container prune'."
                )
            raise SandboxViolation(f"Command timed out after {timeout}s: {cmd[:100]}")
        except Exception as e:
            logger.error(f"Docker execution error: {e}")
            return -1, str(e)
    else:
        return await _execute_level_0(cmd, abs_workdir, timeout, max_output_size)


async def _execute_level_0(
    cmd: str,
    workdir: str,
    timeout: float = DEFAULT_TIMEOUT,
    max_output_size: int = MAX_OUTPUT_SIZE,
) -> tuple[int, str]:
    """Level 0: 子进程执行 — 优先使用 exec 模式避免注入

    1.2 改进:
      - 尝试 shlex.split 将命令拆分为参数列表
      - 拆分成功则使用 create_subprocess_exec（无 shell 注入风险）
      - 拆分失败（含管道/重定向等）回退到 shell 模式，但额外检查危险元字符
      - start_new_session=True 为 1.4 进程组管理做准备
    """
    logger.info(f"Executing (Level 0): {cmd[:200]} in {workdir}")

    # preexec_fn 仅 Unix 可用；Windows 上不设置资源限制
    preexec_fn = _set_resource_limits if _HAS_RESOURCE else None

    # 1.2: 优先使用 exec 模式
    use_exec = False
    args: list[str] = []

    try:
        args = shlex.split(cmd, posix=not sys.platform.startswith("win"))
        # 如果 shlex 成功拆分且结果不含 shell 元字符特征，使用 exec 模式
        _SHELL_EXECUTABLES = {"/bin/bash", "/bin/sh", "/bin/zsh", "bash", "sh", "zsh"}
        if sys.platform == "win32":
            _SHELL_EXECUTABLES.update({
                "cmd.exe", "cmd",
                "powershell.exe", "powershell",
                "pwsh.exe", "pwsh",
            })
        if args and args[0] not in _SHELL_EXECUTABLES:
            use_exec = True
    except ValueError:
        # shlex 解析失败（不匹配的引号等），需要 shell 模式
        logger.warning(f"Command shlex parse failed, falling back to shell: {cmd[:50]}...")

    try:
        if use_exec:
            # 使用 create_subprocess_exec — 无 shell 注入风险
            _subprocess_kwargs: dict = {}
            if sys.platform == "win32":
                import subprocess as _sp
                _subprocess_kwargs["creationflags"] = _sp.CREATE_NEW_PROCESS_GROUP
            else:
                _subprocess_kwargs["start_new_session"] = True
                if preexec_fn is not None:
                    _subprocess_kwargs["preexec_fn"] = preexec_fn
            process = await asyncio.create_subprocess_exec(
                *args, cwd=workdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **_subprocess_kwargs,
            )
        else:
            # 回退到 shell 模式，但额外检查危险元字符
            _check_shell_metacharacters(cmd)
            _subprocess_kwargs: dict = {}
            if sys.platform == "win32":
                import subprocess as _sp
                _subprocess_kwargs["creationflags"] = _sp.CREATE_NEW_PROCESS_GROUP
            else:
                _subprocess_kwargs["start_new_session"] = True
                if preexec_fn is not None:
                    _subprocess_kwargs["preexec_fn"] = preexec_fn
            process = await asyncio.create_subprocess_shell(
                cmd, cwd=workdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **_subprocess_kwargs,
            )

        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)

        if process.returncode != 0:
            stderr_text = stderr.decode(errors='replace')[:500]
            logger.warning(f"Command failed (code={process.returncode}): {stderr_text}")

        output = _truncate_output(stdout.decode(errors='replace'), max_output_size)
        return process.returncode or 0, output
    except asyncio.TimeoutError:
        # 1.4: 进程组杀灭，消除孤儿进程
        _kill_process_group(process)
        raise SandboxViolation(f"Command timed out after {timeout}s: {cmd[:100]}")
    except SandboxViolation:
        raise
    except Exception as e:
        logger.error(f"Execution error: {e}")
        return -1, str(e)


def _check_shell_metacharacters(cmd: str) -> None:
    """检查 shell 回退模式下的危险元字符

    当命令必须通过 shell 执行时，检查是否包含可能导致注入的元字符。
    这是 create_subprocess_exec 的补充防御层。

    Raises:
        SandboxViolation: 如果命令包含危险的 shell 元字符
    """
    # 注意: 这里只检查最危险的元字符组合，不拦截合法的管道/重定向
    # 管道链安全性已由 check_command_safety() 保证
    dangerous_patterns = [
        (r'\$\(', "命令替换 $(...) 可能导致注入"),
        (r'`[^`]+`', "反引号命令替换可能导致注入"),
        (r'[<>]\(', "进程替换 <(...) 或 >(...) 可能导致注入"),
        (r'<<', "Heredoc 可能导致注入"),
        (r"\$'", "ANSI-C 引号 $'...' 可编码绕过黑名单"),
    ]
    for pattern_str, reason in dangerous_patterns:
        if re.search(pattern_str, cmd):
            raise SandboxViolation(f"命令包含危险元字符: {reason}")


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    """杀灭整个进程组，消除孤儿进程（跨平台）

    1.4 改进:
      - Unix: 优先使用 os.killpg 杀灭进程组（包含所有子进程）
      - Windows: 使用 taskkill /F /T 杀灭进程树
      - 进程组不存在或无权限时回退到 process.kill()
    """
    try:
        if sys.platform == "win32":
            # Windows: 使用 taskkill 杀灭整个进程树
            import subprocess as _sp
            try:
                _sp.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    capture_output=True,
                    timeout=5,
                )
                logger.debug(f"Killed process tree for pid {process.pid} via taskkill")
            except (FileNotFoundError, _sp.TimeoutExpired, OSError) as e:
                logger.debug(f"taskkill failed for pid {process.pid}: {e}, falling back to process.kill()")
                try:
                    process.kill()
                except (ProcessLookupError, OSError):
                    pass
        else:
            # Unix: 杀灭进程组
            pgid = os.getpgid(process.pid)
            if pgid != os.getpid():  # 不杀自己的进程组
                os.killpg(pgid, signal.SIGKILL)
                logger.debug(f"Killed process group {pgid} for pid {process.pid}")
            else:
                process.kill()
    except (ProcessLookupError, PermissionError, OSError) as e:
        # 进程可能已退出，或无权限杀进程组
        logger.debug(f"Kill failed for pid {process.pid}: {e}, falling back to process.kill()")
        try:
            process.kill()
        except (ProcessLookupError, OSError):
            pass  # 进程已退出


def _set_resource_limits() -> None:
    """设置子进程的资源限制 (在子进程内执行, preexec_fn) — 仅 Unix"""
    if not _HAS_RESOURCE:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (_RLIMIT_NOFILE, _RLIMIT_NOFILE))
        resource.setrlimit(resource.RLIMIT_FSIZE, (_RLIMIT_FSIZE, _RLIMIT_FSIZE))
        resource.setrlimit(resource.RLIMIT_NPROC, (_RLIMIT_NPROC, _RLIMIT_NPROC))
    except (ValueError, OSError) as e:
        logger.warning(f"Failed to set resource limits: {e}")


def _truncate_output(output: str, max_size: int) -> str:
    """截断超长输出（基于字节数而非字符数）

    截断时确保不破坏 UTF-8 多字节字符边界：如果在截断点
    落在 UTF-8 续字节（0x80-0xBF）上，向前回退到字符起始位置。
    """
    encoded = output.encode('utf-8', errors='replace')
    if len(encoded) > max_size:
        # 截断到有效 UTF-8 字符边界
        truncated_bytes = max_size
        while truncated_bytes > 0 and (encoded[truncated_bytes] & 0xC0) == 0x80:
            truncated_bytes -= 1
        truncated = encoded[:truncated_bytes].decode('utf-8', errors='replace')
        logger.warning(f"Output truncated from {len(encoded)} to {max_size} bytes")
        return truncated + f"\n... [TRUNCATED: output exceeded {max_size} bytes]"
    return output
