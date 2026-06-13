# 配置指南

TerAgent 使用由 `agent.toml` 文件支持的类型化配置系统。本文档涵盖所有配置选项。

## 配置文件

在项目根目录创建 `agent.toml`：

```toml
[drivers.openai_compatible.glm_5]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5"
compiler = "glm_5"

[drivers.anthropic_native.claude]
base_url = "https://api.anthropic.com/v1"
api_key_env = "ANTHROPIC_API_KEY"
model = "claude-sonnet-4-20250514"
compiler = "anthropic"

[drivers.openai_compatible.deepseek]
base_url = "https://api.deepseek.com/v1"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-chat"
compiler = "deepseek"

[execution.pipeline]
design_driver = "openai_compatible.glm_5"
plan_driver = "openai_compatible.glm_5"
execute_driver = "openai_compatible.glm_5"
review_driver = "openai_compatible.glm_5"

[permission]
mode = "plan"
rules = { allow = ["read_file:*", "explore_codebase:*"], deny = ["*:**/.env*", "read_file:/etc/*"] }
```

## 加载配置

```python
import teragent

# Load all configuration
full_config = teragent.load_full_config()

# Load specific driver configs
drivers = teragent.load_driver_configs()

# Load pipeline config
pipeline = teragent.load_pipeline_config()

# Create a provider from config
provider = teragent.create_provider_from_config(drivers["openai_compatible.glm_5"])

# Load typed configuration
typed_config = teragent.load_typed_config()
# → TerAgentConfig with all typed sub-configs
```

## 驱动配置

每个驱动定义了一个 compiler + adapter + model + API key 的组合：

| 字段 | 类型 | 必填 | 描述 |
|-------|------|----------|-------------|
| `base_url` | string | 是（mock 除外） | API 基础 URL |
| `api_key_env` | string | 推荐 | API key 的环境变量名 |
| `api_key` | string | 不推荐 | 直接 API key 值（安全风险） |
| `model` | string | 是 | 模型标识字符串 |
| `compiler` | string | 是 | 可选值：`default`、`glm`、`anthropic`、`deepseek` |

### API Key 安全

始终使用 `api_key_env`（环境变量名）而非 `api_key`（直接值）：

```toml
# ✅ Recommended
api_key_env = "GLM_API_KEY"

# ❌ Not recommended (key visible in config file)
api_key = "sk-xxxxxxxxxxxx"
```

## 类型化配置模块

### AgentLoopConfig

控制 Agent 循环行为：

| 设置 | 默认值 | 描述 |
|---------|---------|-------------|
| `max_tool_steps` | 50 | 每次对话最大工具调用步数 |
| `max_streaming_retries` | 2 | 最大流式重试次数 |
| `tool_execution_timeout` | 60 | 工具执行超时秒数 |
| `max_consecutive_tool_failures` | 5 | 连续工具失败 N 次后停止循环 |
| `intent_tools` | {} | intent → 允许的工具名称映射 |

### CircuitBreakerConfig

控制可靠性阈值：

| 设置 | 默认值 | 描述 |
|---------|---------|-------------|
| `budget.max_session_tokens` | 10,000,000 | 每会话 Token 预算 |
| `budget.warning_threshold` | 0.7 | 70% 预算使用率时发出警告 |
| `budget.critical_threshold` | 0.9 | 90% 时发出严重警告 |
| `budget.enable_hard_limit` | false | 100% 预算时阻止调用 |
| `failure_breaker.max_consecutive` | 5 | 连续失败 N 次后断开熔断器 |
| `failure_breaker.window_seconds` | 300 | 失败计数的时间窗口 |
| `latency_breaker.warn_latency_ms` | 30,000 | 平均延迟超过此值时发出警告 |
| `progress_detector.stall_threshold` | 10 | 停滞检测触发的步数 |

### PermissionConfig

| 设置 | 默认值 | 描述 |
|---------|---------|-------------|
| `mode` | "default" | 权限级别：default/plan/bypass/accept_edits/auto |
| `rules` | {} | 权限规则（allow/deny 模式） |

### ContextManagementConfig

| 设置 | 默认值 | 描述 |
|---------|---------|-------------|
| `model_token_limit` | 128,000 | 模型最大上下文 Token 数 |
| `compaction_threshold` | 0.8 | 使用率超过此值时压缩 |
| `retain_count` | 8 | 压缩时保留的消息数 |
| `max_compacts` | 5 | 每会话最大压缩次数 |

### StreamingConfig

| 设置 | 默认值 | 描述 |
|---------|---------|-------------|
| `mode` | "auto" | 流式模式：auto/streaming/batch |
| `max_concurrent` | 10 | 最大并发只读工具执行数 |

### SessionConfig

| 设置 | 默认值 | 描述 |
|---------|---------|-------------|
| `db_path` | ".agent/sessions.db" | SQLite 数据库路径 |
| `enabled` | true | 是否启用会话持久化 |

### FileSafetyConfig

| 设置 | 默认值 | 描述 |
|---------|---------|-------------|
| `workspace_root` | "." | 文件操作的根目录 |
| `enable_2pc` | true | 启用两阶段提交文件写入 |
| `max_file_size` | 10MB | 写入的最大文件大小 |

### HooksConfig

| 设置 | 默认值 | 描述 |
|---------|---------|-------------|
| `hooks` | [] | Hook 配置列表 |

## 编程式配置

你也可以通过编程方式创建配置：

```python
from teragent.config import (
    TerAgentConfig,
    AgentLoopConfig,
    CircuitBreakerConfig,
    ContextManagementConfig,
)

# Create typed config
config = TerAgentConfig(
    agent_loop=AgentLoopConfig(
        max_tool_steps=100,
        max_streaming_retries=3,
    ),
    circuit_breaker=CircuitBreakerConfig(
        budget={"max_session_tokens": 20_000_000},
    ),
    context_management=ContextManagementConfig(
        model_token_limit=200_000,
    ),
)
```

## 环境变量

| 变量 | 描述 |
|----------|-------------|
| `GLM_API_KEY` | 智谱 AI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |

TerAgent 使用 `python-dotenv` 自动加载 `.env` 文件。
