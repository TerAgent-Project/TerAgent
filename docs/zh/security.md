# 安全体系

TerAgent 提供跨多层的纵深防御安全机制。本文档描述每个安全子系统及其协同工作方式。

## 概述

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

## 7 层权限解析

`EnhancedPermissionManager` 通过 7 层解析权限，从最高优先级到最低优先级：

| 层级 | 来源 | 优先级 | 说明 |
|------|------|--------|------|
| 1 | `user` | 100 | 用户定义的规则（最高优先级，始终优先） |
| 2 | `config` | 60 | 从 `agent.toml` 加载的规则 |
| 3 | `project` | 50 | 项目级别规则 |
| 4 | `system` | 10 | 系统默认规则（内置） |
| 5 | Permission Level | — | `DEFAULT` / `PLAN` / `BYPASS` / `ACCEPT_EDITS` / `AUTO` |
| 6 | AI Classifier | — | 咨询性 LLM 判断（仅异步） |
| 7 | Default DENY | — | 无规则匹配时，默认拒绝 |

### 规则匹配

规则对工具名称和路径使用 glob 模式：

```python
from teragent.security import EnhancedPermissionManager, PermissionRule, PermissionEffect

epm = EnhancedPermissionManager()

# 用户级别 DENY：永远不读取 /etc
epm.add_rule(PermissionRule(
    effect=PermissionEffect.DENY,
    tool_pattern="read_file",
    path_pattern="/etc/*",
    description="禁止读取系统目录",
    source="user",
))

# 系统级别 ALLOW：读取项目中的文件
epm.add_rule(PermissionRule(
    effect=PermissionEffect.ALLOW,
    tool_pattern="read_file",
    description="读取文件始终允许",
    source="system",
))

# 检查权限
allowed, reason = epm.check("read_file", path="/etc/passwd")
# → (False, "Denied by rule: 禁止读取系统目录")

allowed, reason = epm.check("read_file", path="/src/main.py")
# → (True, "Allowed by rule: 读取文件始终允许")
```

### 排序策略

当多条规则匹配时，排序顺序决定结果：

1. **来源优先级**（越高越先匹配）：user > config > project > system
2. **路径特异性**（越具体越先匹配）：带 `path_pattern` 的规则优先于不带路径模式的规则
3. **DENY 优先**（在同一来源 + 特异性内）：DENY 优先于 ALLOW

这确保了用户级别的 DENY 始终优先于系统级别的 ALLOW，更具体的规则（带路径模式）优先于通用规则。

### 同步 vs 异步检查

```python
# 同步：第 1-5 层 + 第 7 层（不含 AI 分类器）
allowed, reason = epm.check("write_file", path="/src/main.py")

# 异步：第 1-7 层（包含 AI 分类器）
allowed, reason = await epm.acheck("write_file", path="/src/main.py", context="...")
```

### Permission Levels

| Level | 值 | 允许的操作 |
|-------|----|-----------|
| `DEFAULT` | 0 | 只读操作 |
| `PLAN` | 1 | 写入项目目录 |
| `BYPASS` | 2 | 执行用户确认的高风险操作 |
| `ACCEPT_EDITS` | 3 | 自动接受代码修改 |
| `AUTO` | 4 | 完全自动，无需确认 |
| `CUSTOM` | 99 | 用户自定义 |

## 6 层命令防御

`check_command_safety()` 函数和 `DangerousCommandHook` 实现了 6 层命令防御：

### 第 1 层：命令规范化

剥离 ANSI 转义序列、空字节，并压缩空白，以对抗基于编码的绕过尝试。

### 第 2 层：管道链拆分

按 `|`、`&&`、`||`、`;` 拆分命令，并独立检查每个子命令。这防止攻击者在管道链中隐藏危险操作。

### 第 3 层：8 类黑名单

| 类别 | 示例 |
|------|------|
| 权限提升 | `sudo`、`su`、`doas`、`pkexec`、`chmod` |
| 反弹 Shell / 后门 | `nc`、`ncat`、`socat`、`/dev/tcp` |
| 内联脚本执行 | `python -c`、`bash -c`、`node -e` |
| 系统破坏 | `rm -rf /`、`mkfs`、`dd`、`shutdown` |
| 持久化 | `crontab`、`at`、`launchctl` |
| 编码绕过 | `base64 -d`、`xxd -r`、`\x41` 模式 |
| 远程执行 | `curl \| sh`、`eval`、`source /tmp/...` |
| Fork 炸弹 / 磁盘写入 | `:(){ :\|:& };:`、`> /dev/sd` |

### 第 4 层：危险重定向检测

针对系统关键路径的重定向进行细粒度检测，按子命令逐一检查：

- `> /etc/passwd` — 重定向到系统配置
- `> /dev/sda` — 直接磁盘写入
- 任何重定向到 `/etc`、`/dev`、`/sys`、`/proc`、`/boot`、`/root`、`/sbin` 的操作

### 第 5 层：跨链检测

某些危险模式仅在完整命令中可见（拆分后的子命令中不可见）：

- `curl | sh` — 远程脚本执行
- `wget | python` — 远程代码执行
- 任何管道到 Shell 解释器的操作

### 第 6 层：包安装警告

`pip install`、`npm install`、`apt install` 等操作会记录为警告但不阻止。这仅作为信息提示。

## 2 阶段提交文件写入

`write_files_safely()` 函数实现了事务性文件写入：

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

### 关键特性

- **原子性**：`os.replace()` 在 POSIX 和 Windows 上都是原子操作
- **崩溃安全**：中间临时文件防止崩溃导致数据损坏
- **一致性**：所有文件要么全部提交，要么全部不提交（事务性）
- **并发安全**：读取者永远不会看到半写入状态
- **路径遍历保护**：所有路径必须在 `workspace_root` 内

### 用法

```python
from teragent.security import write_files_safely

# 原子写入多个文件
success, results = write_files_safely(
    files=[
        {"path": "/project/src/main.py", "content": "..."},
        {"path": "/project/src/utils.py", "content": "..."},
    ],
    workspace_root="/project",
)
```

## 3 级沙箱降级

命令可以在隔离程度递减的环境中执行：

| 级别 | 隔离方式 | 约束 | 降级目标 |
|------|---------|------|---------|
| Level 2 | Firecracker microVM | 完整硬件隔离 | → Level 1 (Docker) |
| Level 1 | Docker 容器 | 512MB RAM, 1 CPU, 64 PIDs, 无网络 | → Level 0 (subprocess) |
| Level 0 | 带 rlimit 的子进程 | `RLIMIT_NOFILE=256`, `RLIMIT_FSIZE=50MB`, `RLIMIT_NPROC=64` | — |

### Level 0 详细说明

Level 0（子进程）有两种执行模式：

1. **Exec 模式**（优先）：使用 `shlex.split()` 将命令拆分为参数，然后调用 `create_subprocess_exec()` — 无 Shell 注入风险
2. **Shell 模式**（降级）：当命令包含管道、重定向或 `shlex` 无法解析的 Shell 特性时，降级为 `create_subprocess_shell()`，并附加元字符检查

### 进程组管理

所有子进程模式使用 `start_new_session=True` 创建新进程组。超时时，`os.killpg()` 终止整个组（包括子进程），防止产生孤儿进程。

### 输出截断

沙箱输出默认截断为 1MB，使用 UTF-8 安全边界处理以避免破坏多字节字符。

## API Key 安全

### ApiKeyVault

`ApiKeyVault` 从环境变量解析 API Key，支持 `.env` 文件降级：

```python
from teragent.config import ApiKeyVault, mask_api_key, audit_config_security

vault = ApiKeyVault()
resolved = vault.resolve("GLM_API_KEY")
# → ResolvedKey(key="sk-xxx...", found=True, source="env")

# 对 Key 进行掩码处理用于日志记录
masked = mask_api_key("sk-1234567890abcdef")
# → "sk-12...cdef"
```

### 安全审计

```python
# 审计配置文件中的泄露 Key
findings = audit_config_security(config_dict)

# 审计 .env 文件
findings = audit_env_file(".env")
```

### 最佳实践

- **始终使用 `api_key_env`**（环境变量名称）而非 `api_key`（直接值）配置
- 使用 `audit_config_security()` 扫描泄露的 Key
- 日志记录 Key 信息时使用 `mask_api_key()`
- 当直接使用 `api_key` 时，库会记录信息日志，建议使用 `api_key_env`
