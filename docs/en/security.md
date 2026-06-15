# Security Architecture

TerAgent provides defense-in-depth security across multiple layers. This document describes each security subsystem and how they work together.

## Overview

```
┌─────────────────────────────────────────────────┐
│              Security Architecture               │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │        7-Layer Permission Resolution       │ │
│  │  User → Config → Project → System →       │ │
│  │  Level → AI Classifier → Default DENY     │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │        6-Layer Command Defense             │ │
│  │  Normalize → Chain Split → Blacklist →    │ │
│  │  Cross-Chain → Package Warning → Meta      │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │        2-Phase Commit File Writes          │ │
│  │  Validate → Write Temp → Atomic Swap →    │ │
│  │  Rollback on failure                       │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │        3-Level Sandbox Degradation         │ │
│  │  Firecracker → Docker → Subprocess         │ │
│  └────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## 7-Layer Permission Resolution

The `EnhancedPermissionManager` resolves permissions through 7 layers, from highest to lowest priority:

| Layer | Source | Priority | Description |
|-------|--------|----------|-------------|
| 1 | `user` | 100 | User-defined rules (highest priority, always wins) |
| 2 | `config` | 60 | Rules loaded from `agent.toml` |
| 3 | `project` | 50 | Project-level rules |
| 4 | `system` | 10 | System default rules (built-in) |
| 5 | Permission Level | — | `DEFAULT` / `PLAN` / `BYPASS` / `ACCEPT_EDITS` / `AUTO` |
| 6 | AI Classifier | — | Consultative LLM-based judgment (async only) |
| 7 | Default DENY | — | When no rule matches, deny by default |

### Rule Matching

Rules use glob patterns for both tool names and paths:

```python
from teragent.security import EnhancedPermissionManager, PermissionRule, PermissionEffect

epm = EnhancedPermissionManager()

# User-level DENY: never read /etc
epm.add_rule(PermissionRule(
    effect=PermissionEffect.DENY,
    tool_pattern="read_file",
    path_pattern="/etc/*",
    description="Block reading system directories",
    source="user",
))

# System-level ALLOW: read files in project
epm.add_rule(PermissionRule(
    effect=PermissionEffect.ALLOW,
    tool_pattern="read_file",
    description="Reading files is always allowed",
    source="system",
))

# Check permissions
allowed, reason = epm.check("read_file", path="/etc/passwd")
# → (False, "Denied by rule: Block reading system directories")

allowed, reason = epm.check("read_file", path="/src/main.py")
# → (True, "Allowed by rule: Reading files is always allowed")
```

### Sorting Strategy

When multiple rules match, the sorted order determines the outcome:

1. **Source priority** (higher = first match): user > config > project > system
2. **Path specificity** (more specific = first match): rules with `path_pattern` beat rules without
3. **DENY priority** (within same source + specificity): DENY beats ALLOW

This ensures that a user-level DENY always wins over a system-level ALLOW, and that more specific rules (with path patterns) take precedence.

### Sync vs Async Checks

```python
# Sync: Layers 1-5 + Layer 7 (no AI classifier)
allowed, reason = epm.check("write_file", path="/src/main.py")

# Async: Layers 1-7 (includes AI classifier)
allowed, reason = await epm.acheck("write_file", path="/src/main.py", context="...")
```

### Permission Levels

| Level | Value | What It Allows |
|-------|-------|---------------|
| `DEFAULT` | 0 | Read-only operations |
| `PLAN` | 1 | Write to project directory |
| `BYPASS` | 2 | Execute user-confirmed high-risk operations |
| `ACCEPT_EDITS` | 3 | Auto-accept code modifications |
| `AUTO` | 4 | Full auto, no confirmation needed |
| `CUSTOM` | 99 | User-defined |

## 6-Layer Command Defense

The `check_command_safety()` function and `DangerousCommandHook` implement 6 layers of command defense:

### Layer 1: Command Normalization

Strips ANSI escape sequences, null bytes, and compresses whitespace to defeat encoding-based bypass attempts.

### Layer 2: Pipeline Chain Splitting

Splits commands on `|`, `&&`, `||`, `;` and checks each sub-command independently. This prevents attackers from hiding dangerous operations in pipeline chains.

### Layer 3: 8-Category Blacklist

| Category | Examples |
|----------|---------|
| Privilege escalation | `sudo`, `su`, `doas`, `pkexec`, `chmod` |
| Reverse shell / backdoor | `nc`, `ncat`, `socat`, `/dev/tcp` |
| Inline script execution | `python -c`, `bash -c`, `node -e` |
| System destruction | `rm -rf /`, `mkfs`, `dd`, `shutdown` |
| Persistence | `crontab`, `at`, `launchctl` |
| Encoding bypass | `base64 -d`, `xxd -r`, `\x41` patterns |
| Remote execution | `curl \| sh`, `eval`, `source /tmp/...` |
| Fork bomb / disk write | `:(){ :\|:& };:`, `> /dev/sd` |

#### Windows-Specific Dangerous Patterns (16 patterns)

On Windows (`sys.platform == "win32"`), an additional 16 dangerous command patterns are checked:

| Pattern | Example | Category |
|---------|---------|----------|
| `format X:` | `format C:` | Disk destruction |
| `del /s`, `rd /s` | `del /s /q C:\` | Recursive deletion |
| `reg delete/add` | `reg delete HKLM\...` | Registry modification |
| `net user` | `net user hacker P@ss /add` | User management |
| `net localgroup` | `net localgroup admins hacker /add` | Group management |
| `powershell -enc` | `powershell -encodedcommand <base64>` | Encoded execution bypass |
| `pwsh -enc` | `pwsh -encodedcommand <base64>` | Encoded execution bypass |
| `diskpart` | `diskpart` | Disk partition operations |
| `cipher /w:` | `cipher /w:C:\` | Secure data erasure |
| `taskkill` | `taskkill /f /im *` | Process termination |
| `netsh` | `netsh advfirewall` | Network/firewall configuration |
| `takeown` | `takeown /f C:\Windows` | Ownership takeover |
| `icacls /grant` | `icacls C:\ /grant user:F` | Permission modification |
| `wmic` | `wmic process call create` | WMI command execution |
| `schtasks /create` | `schtasks /create /tn ...` | Scheduled task creation |
| `schtasks /delete` | `schtasks /delete /tn ...` | Scheduled task deletion |

These patterns are automatically enabled on Windows platforms and supplement the Unix-focused blacklist.

### Layer 4: Dangerous Redirect Detection

Fine-grained detection of redirects to system-critical paths, checked per sub-command:

- `> /etc/passwd` — redirect to system configuration
- `> /dev/sda` — direct disk write
- Any redirect to `/etc`, `/dev`, `/sys`, `/proc`, `/boot`, `/root`, `/sbin`

On Windows, the following system paths are also protected:

| Path | Description |
|------|-------------|
| `%SystemRoot%\` (e.g., `C:\Windows\`) | Windows system directory |
| `%SystemRoot%\System32\` | System binaries and configs |
| `%SystemRoot%\SysWOW64\` | 64-bit system directory |
| `C:\Program Files\` | Installed applications |
| `C:\Program Files (x86)\` | 32-bit applications |
| `C:\ProgramData\` | Global application data |
| `C:\System Volume Information\` | System restore points |

These paths are resolved from environment variables (`SystemRoot`, `ProgramData`) and are automatically checked on Windows.

### Layer 5: Cross-Chain Detection

Some dangerous patterns are only visible in the full command (not in individual sub-commands after splitting):

- `curl | sh` — remote script execution
- `wget | python` — remote code execution
- Any pipe to shell interpreter

### Layer 6: Package Install Warning

`pip install`, `npm install`, `apt install` etc. are logged as warnings but not blocked. This is informational only.

## 2-Phase Commit (2PC) File Writes

The `write_files_safely()` function implements transactional file writes:

```
Phase 1: Validate
  ├── Check permissions for each file
  ├── Check path traversal (all paths must be within workspace_root)
  └── Check read-before-write contract

Phase 2: Write
  └── Write all files to .tmp suffix

Phase 3: Commit
  └── os.replace() atomic swap (all succeed or all roll back)

Phase 4: Rollback (on any commit failure)
  └── Restore from .bak backups
```

### Key Properties

- **Atomic**: `os.replace()` is atomic on POSIX; on Windows NTFS, uses a 3-step rename-rename-delete pattern with backup/rollback to ensure crash recovery
- **Crash-safe**: Intermediate temp files prevent corruption on crash
- **Consistent**: All files commit or none do (transactional)
- **Concurrent-safe**: Readers never see half-written state
- **Path traversal protection**: All paths must be within `workspace_root`

### Usage

```python
from teragent.security import write_files_safely

# Write multiple files atomically
success, results = write_files_safely(
    files=[
        {"path": "/project/src/main.py", "content": "..."},
        {"path": "/project/src/utils.py", "content": "..."},
    ],
    workspace_root="/project",
)
```

## 3-Level Sandbox Degradation

Commands can be executed in progressively less isolated environments:

| Level | Isolation | Constraints | Fallback |
|-------|-----------|-------------|----------|
| Level 2 | Firecracker microVM | Full hardware isolation | → Level 1 (Docker) |
| Level 1 | Docker container | 512MB RAM, 1 CPU, 64 PIDs, no network | → Level 0 (subprocess) |
| Level 0 | Subprocess with rlimit | `RLIMIT_NOFILE=256`, `RLIMIT_FSIZE=50MB`, `RLIMIT_NPROC=64` | — |

### Level 0 Details

Level 0 (subprocess) has two execution modes:

1. **Exec mode** (preferred): Uses `shlex.split()` to split the command into arguments, then `create_subprocess_exec()` — no shell injection risk
2. **Shell mode** (fallback): When the command contains pipes, redirects, or shell features that can't be parsed by `shlex`, falls back to `create_subprocess_shell()` with additional metacharacter checks

### Process Group Management

All subprocess modes create a new process group for isolation. The implementation is platform-specific:

- **Unix** (`sys.platform != "win32"`): Uses `start_new_session=True` to create a new session/process group. On timeout, `os.killpg()` kills the entire group (including child processes), preventing orphan processes.
- **Windows** (`sys.platform == "win32"`): Uses `CREATE_NEW_PROCESS_GROUP` creation flag. On timeout, `taskkill /F /T /PID <pid>` kills the entire process tree, including all child processes.

The `_kill_process_group()` function provides a unified cross-platform API that handles both platforms. On failure (process already terminated, permission denied), it falls back to `process.kill()`.

### Output Truncation

Sandbox output is truncated to 1MB by default, with UTF-8-safe boundary handling to avoid breaking multi-byte characters.

## API Key Security

### ApiKeyVault

The `ApiKeyVault` resolves API keys from environment variables with `.env` file fallback:

```python
from teragent.config import ApiKeyVault, mask_api_key, audit_config_security

vault = ApiKeyVault()
resolved = vault.resolve("GLM_API_KEY")
# → ResolvedKey(key="sk-xxx...", found=True, source="env")

# Mask keys for logging
masked = mask_api_key("sk-1234567890abcdef")
# → "sk-12...cdef"
```

### Security Auditing

```python
# Audit config file for leaked keys
findings = audit_config_security(config_dict)

# Audit .env file
findings = audit_env_file(".env")
```

### Best Practices

- **Always use `api_key_env`** (environment variable name) in config, not `api_key` (direct value)
- Use `audit_config_security()` to scan for leaked keys
- Use `mask_api_key()` when logging key information
- The library logs an info message when `api_key` is used directly, recommending `api_key_env`

## Cross-Platform Security Notes

### Windows Security Considerations

TerAgent's security sandbox was originally designed for Unix systems. Starting from v0.1.2, comprehensive Windows support has been added:

1. **Command Safety**: 16 Windows-specific dangerous command patterns are now blocked (see Layer 3 above)
2. **System Path Protection**: Windows system directories are protected from write redirects (see Layer 4 above)
3. **Shell Detection**: Windows shell executables (`cmd.exe`, `powershell.exe`, `pwsh.exe`) are correctly identified for exec/shell mode classification
4. **Command Parsing**: `shlex.split()` uses `posix=False` on Windows for correct command-line parsing
5. **Docker Isolation**: Windows Docker Desktop uses `ContainerUser` instead of uid/gid mapping (slightly reduced isolation)

### macOS and Linux Notes

- **macOS**: Full sandbox support at all levels (except Firecracker, which requires KVM)
- **Linux**: Full support including Firecracker. Wayland desktops are auto-detected for clipboard operations (`wl-copy`/`wl-paste`)
- **Path Normalization**: On case-sensitive Unix filesystems, paths are not lowercased during security checks (preserving case sensitivity)
