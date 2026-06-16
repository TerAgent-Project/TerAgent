# 配置指南

TerAgent 使用由 `agent.toml` 文件支持的类型化配置系统。本文档涵盖所有配置选项。

## 配置文件

在项目根目录创建 `agent.toml`（完整示例见 `examples/agent.toml`）：

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

## 配置文件搜索路径

TerAgent 按以下优先级搜索 `agent.toml`：

| 优先级 | 位置 | 平台 |
|--------|------|------|
| 1 | `./agent.toml`（当前工作目录） | 全部 |
| 2 | `%APPDATA%\teragent\agent.toml` | Windows |
| 2 | `~/Library/Application Support/teragent/agent.toml` | macOS |
| 2 | `$XDG_CONFIG_HOME/teragent/agent.toml`（默认 `~/.config/teragent/`） | Linux |
| 3 | `<project_root>/agent.toml` | 全部 |
| 4 | `agent.toml`（回退） | 全部 |

找到的第一个存在文件即被使用。这允许你在平台标准配置目录中设置全局配置，适用于所有项目。

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
| `compiler` | string | 是 | 可选值：`default`、`glm`、`glm_5`、`glm_52`、`glm_5v_turbo`、`anthropic`、`deepseek`、`deepseek_v4`、`minimax_m3` |
| `compiler_variant` | string | 否 | 编译器变体（如 `deepseek_v4` 的 `"flash"` 或 `"pro"`） |

### 适配器 HTTP 配置

所有基于 HTTP 的适配器（`openai_compatible`、`anthropic_native`、`minimax_native`）接受以下配置参数：

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `ssl_verify` | `bool \| str` | `True` | SSL 证书验证。`True` = 系统 CA，`False` = 禁用（不安全），`str` = 自定义 CA 证书包路径 |
| `http2_enabled` | `bool` | `False` | 启用 HTTP/2 连接池。需要 `h2` 包。在 HTTP/1.1-only 代理环境中需禁用 |

```python
from teragent.core.adapters import OpenAICompatibleAdapter

# 企业代理 + 自定义 CA
adapter = OpenAICompatibleAdapter(
    base_url="https://api.example.com/v1",
    api_key="...",
    ssl_verify="/path/to/ca-bundle.crt",  # 自定义 CA 证书
    http2_enabled=False,                    # 为 HTTP/1.1 代理禁用
)
```

**注意：** `http2_enabled` 默认为 `False` 以确保最大兼容性。仅在确认端点支持 HTTP/2 时启用。

### API Key 安全

始终使用 `api_key_env`（环境变量名）而非 `api_key`（直接值）：

```toml
# ✅ Recommended
api_key_env = "GLM_API_KEY"

# ❌ Not recommended (key visible in config file)
api_key = "sk-xxxxxxxxxxxx"
```

## 新模型驱动

### DeepSeek V4-Flash 驱动

```toml
[drivers.openai_compatible.deepseek_v4_flash]
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-v4-flash"
compiler = "deepseek_v4"
compiler_variant = "flash"              # Flash 模式：极简提示词，快速响应
max_context_tokens = 1_000_000          # V4 支持 1M 上下文
max_output_tokens = 384_000             # V4 最大 384K 输出
thinking_mode = "auto"                  # auto/deep/quick
cache_aware = true                      # 启用缓存感知（12 倍价格差异）
```

| 字段 | 类型 | 默认值 | 描述 |
|-------|------|---------|-------------|
| `base_url` | string | — | DeepSeek API 基础 URL |
| `api_key_env` | string | — | API key 的环境变量名 |
| `model` | string | — | 必须为 `"deepseek-v4-flash"` |
| `compiler` | string | — | 必须为 `"deepseek_v4"` |
| `compiler_variant` | string | `"pro"` | Flash 模式必须为 `"flash"` |
| `max_context_tokens` | int | `1_000_000` | 最大上下文窗口 |
| `max_output_tokens` | int | `384_000` | 最大输出 Token 数 |
| `thinking_mode` | string | `"auto"` | 思考模式：`auto`/`deep`/`quick` |
| `cache_aware` | bool | `false` | 启用缓存感知提示布局 |

### DeepSeek V4-Pro 驱动

```toml
[drivers.openai_compatible.deepseek_v4_pro]
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-v4-pro"
compiler = "deepseek_v4"
compiler_variant = "pro"                # Pro 模式：完整提示词 + 推理引导
max_context_tokens = 1_000_000
max_output_tokens = 384_000
thinking_mode = "deep"                  # Pro 默认深度推理
cache_aware = true
```

| 字段 | 类型 | 默认值 | 描述 |
|-------|------|---------|-------------|
| `compiler_variant` | string | `"pro"` | Pro 模式必须为 `"pro"` |
| `thinking_mode` | string | `"deep"` | Pro 默认深度推理 |

### MiniMax M3 驱动

```toml
[drivers.openai_compatible.minimax_m3]
base_url = "https://api.minimaxi.com/v1"
api_key_env = "MINIMAX_API_KEY"
model = "minimax-m3"
compiler = "minimax_m3"
max_context_tokens = 1_000_000          # M3 支持 1M 上下文
max_output_tokens = 384_000
multimodal_enabled = true               # 启用多模态（图像+视频）
desktop_enabled = true                  # 启用桌面操作
msa_efficient = true                    # MSA 全文注入模式
```

| 字段 | 类型 | 默认值 | 描述 |
|-------|------|---------|-------------|
| `base_url` | string | — | MiniMax API 基础 URL |
| `api_key_env` | string | — | API key 的环境变量名 |
| `model` | string | — | 必须为 `"minimax-m3"` |
| `compiler` | string | — | 必须为 `"minimax_m3"` |
| `multimodal_enabled` | bool | `false` | 启用原生多模态支持 |
| `desktop_enabled` | bool | `false` | 启用桌面操作支持 |
| `msa_efficient` | bool | `false` | 启用 MSA 全文注入 |

### GLM-5 驱动

```toml
[drivers.openai_compatible.glm_5]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5"
compiler = "glm_5"
max_context_tokens = 200_000            # GLM-5 200K 上下文
max_output_tokens = 128_000
thinking_mode = "deep"                  # GLM-5 默认深度推理
long_horizon_enabled = true             # 启用长时模式（8 小时自主）
```

| 字段 | 类型 | 默认值 | 描述 |
|-------|------|---------|-------------|
| `model` | string | — | 必须为 `"glm-5"` |
| `compiler` | string | — | 必须为 `"glm_5"` |
| `max_context_tokens` | int | `200_000` | GLM-5 上下文窗口（200K） |
| `long_horizon_enabled` | bool | `false` | 启用长时自主模式 |

### GLM-5.2 驱动

```toml
[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
max_context_tokens = 1_000_000          # GLM-5.2 支持 1M 上下文
max_output_tokens = 128_000
thinking_mode = "high"                  # 默认："high"（对比 "max" 深度推理）
multimodal_enabled = true               # 启用多模态（视觉协调）
long_horizon_enabled = true             # GLM-5.2 也支持长时任务
# 注意：dual_thinking、preserved_thinking、vision_coordination、context_degradation
# 是 create_provider() 的编译器级 kwargs，不是 TOML 驱动字段。
# 使用 thinking_mode 和按请求覆盖来控制双思考模式。
```

| 字段 | 类型 | 默认值 | 描述 |
|-------|------|---------|-------------|
| `model` | string | — | 必须为 `"glm-5.2"` |
| `compiler` | string | — | 必须为 `"glm_52"` |
| `max_context_tokens` | int | `1_000_000` | GLM-5.2 上下文窗口（1M） |
| `thinking_mode` | string | `"high"` | 默认思考模式：`high` 或 `max` |
| `multimodal_enabled` | bool | `false` | 启用多模态内容（视觉协调） |
| `long_horizon_enabled` | bool | `false` | 启用长时自主模式 |

> **注意：** 双思考、PreservedThinking、视觉协调和上下文降级等功能通过编译器级 kwargs 传递给 `create_provider()`，而不是通过 TOML 驱动字段。在 TOML 配置中，使用 `thinking_mode` 设置默认思考深度，并通过 `meta={"thinking_mode": "max"}` 按请求覆盖。上下文降级由 AutoCompactor 内部处理。

### GLM-5V-Turbo 驱动（用于视觉协调）

```toml
[drivers.openai_compatible.glm_5v_turbo]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5v-turbo"
compiler = "glm_5v_turbo"
```

### GLMNative 驱动

```toml
[drivers.glm_native.glm_5]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5"
compiler = "glm_5"

[drivers.glm_native.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
max_context_tokens = 1_000_000
thinking_mode = "high"
multimodal_enabled = true
# 注意：preserved_thinking_enabled 和 vision_coordination_enabled
# 是 create_provider() 的 kwargs，不是 TOML 驱动字段。
```

### MiniMaxNative 驱动

```toml
[drivers.minimax_native.minimax_m3]
base_url = "https://api.minimaxi.com/v1"
api_key_env = "MINIMAX_API_KEY"
model = "minimax-m3"
compiler = "minimax_m3"
max_context_tokens = 1_000_000
multimodal_enabled = true
desktop_enabled = true
msa_efficient = true
```

## 智能路由配置

### [routing] 部分

控制基于内容类型和任务特征的自动模型选择：

```toml
[routing]
# 多模态内容（图像/视频） → M3
multimodal_driver = "openai_compatible.minimax_m3"

# 桌面操作 → M3
desktop_driver = "openai_compatible.minimax_m3"

# 长时任务 → GLM-5
long_horizon_driver = "openai_compatible.glm_5"
```

| 字段 | 类型 | 默认值 | 描述 |
|-------|------|---------|-------------|
| `multimodal_driver` | string | `"openai_compatible.minimax_m3"` | 多模态内容驱动 |
| `desktop_driver` | string | `"openai_compatible.minimax_m3"` | 桌面上下文驱动 |
| `long_horizon_driver` | string | `"openai_compatible.glm_5"` | 长时任务驱动 |

### [routing.monthly_budget] 部分

带自动降级的月度成本控制：

```toml
[routing.monthly_budget]
limit_cny = 500.0              # 月度预算上限（人民币）
warning_threshold = 0.8        # 80% 使用率时发出警告
critical_threshold = 0.95      # 95% 时自动降级
auto_downgrade = true          # 启用自动降级
auto_downgrade_driver = "openai_compatible.deepseek_v4_flash"  # 降级驱动
notify_on_warning = true       # 预算警告时发出事件
```

| 字段 | 类型 | 默认值 | 描述 |
|-------|------|---------|-------------|
| `limit_cny` | float | `0.0` | 月度预算上限，人民币（0 = 无限制） |
| `warning_threshold` | float | `0.8` | 发出警告的使用率比例 |
| `critical_threshold` | float | `0.95` | 自动降级的使用率比例 |
| `auto_downgrade` | bool | `true` | 预算耗尽时是否自动降级 |
| `auto_downgrade_driver` | string | `"openai_compatible.deepseek_v4_flash"` | 降级目标驱动 |
| `notify_on_warning` | bool | `true` | 预算警告时是否发出事件 |

## 管道命名配置文件

### [execution.pipeline.profiles.*] 部分

定义可在运行时切换的命名管道配置：

```toml
# 经济配置文件：最大成本节省
[execution.pipeline.profiles.budget]
description = "最大成本节省：所有阶段使用 V4-Flash"
design_driver = "openai_compatible.deepseek_v4_flash"
plan_driver = "openai_compatible.deepseek_v4_flash"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.deepseek_v4_flash"

# 多模态配置文件：所有阶段使用 M3
[execution.pipeline.profiles.multimodal]
description = "多模态模式：所有阶段使用 M3"
design_driver = "openai_compatible.minimax_m3"
plan_driver = "openai_compatible.minimax_m3"
execute_driver = "openai_compatible.minimax_m3"
review_driver = "openai_compatible.minimax_m3"

# 质量配置文件：每个阶段使用最佳模型
[execution.pipeline.profiles.quality]
description = "质量优先：设计/评审使用 V4-Pro，规划/执行使用 GLM-5"
design_driver = "openai_compatible.deepseek_v4_pro"
plan_driver = "openai_compatible.glm_5"
execute_driver = "openai_compatible.glm_5"
review_driver = "openai_compatible.deepseek_v4_pro"
```

每个配置文件部分支持：

| 字段 | 类型 | 必填 | 描述 |
|-------|------|----------|-------------|
| `description` | string | 否 | 人类可读的配置文件描述 |
| `design_driver` | string | 是 | 设计阶段驱动 |
| `plan_driver` | string | 是 | 规划阶段驱动 |
| `execute_driver` | string | 是 | 执行阶段驱动 |
| `review_driver` | string | 是 | 评审阶段驱动 |

**内置配置文件**（自动可用）：
- `default` — 使用基础 `[execution.pipeline]` 设置
- `budget` — 所有阶段使用 V4-Flash
- `multimodal` — 所有阶段使用 M3
- `deep_thinking` — 所有阶段使用 GLM-5.2（Max 思考模式）

## 按模型熔断器配置

```toml
[circuit_breaker.models.deepseek_v4_pro]
max_consecutive_failures = 5        # 连续失败 N 次后断开熔断器
window_seconds = 300.0              # 滑动窗口持续时间（秒）
cooldown_seconds = 60.0             # 半开状态转换前的冷却时间
failure_threshold_percent = 0.5     # 窗口内失败率 >50% 时断开
half_open_max_calls = 3             # 半开状态允许的测试调用数

[circuit_breaker.models.deepseek_v4_flash]
max_consecutive_failures = 8        # 轻量模型更宽容
cooldown_seconds = 30.0             # 更短的冷却时间

[circuit_breaker.models.minimax_m3]
max_consecutive_failures = 5
cooldown_seconds = 60.0

[circuit_breaker.models.glm_5]
max_consecutive_failures = 5
cooldown_seconds = 90.0             # 长时模型更长的冷却时间
```

| 字段 | 类型 | 默认值 | 描述 |
|-------|------|---------|-------------|
| `max_consecutive_failures` | int | `5` | 连续失败次数以断开熔断器 |
| `window_seconds` | float | `300.0` | 失败率的滑动窗口 |
| `cooldown_seconds` | float | `60.0` | 半开状态转换前的冷却时间 |
| `failure_threshold_percent` | float | `0.5` | 窗口内失败率阈值 |
| `half_open_max_calls` | int | `3` | 半开状态中的测试调用数 |

## 降级链配置

控制模型不可用时的降级顺序：

```toml
[degradation]
# 四模型架构的默认降级链
heavy = ["deepseek_v4_pro", "glm_52", "glm_5", "deepseek_v4_flash"]
multimodal = ["minimax_m3", "glm_52", "deepseek_v4_pro"]
ultra_context = ["glm_52", "deepseek_v4_pro", "minimax_m3"]
default = ["deepseek_v4_pro", "glm_52", "glm_5", "deepseek_v4_flash"]
```

| 链 | 描述 |
|-------|-------------|
| `heavy` | 复杂任务：V4-Pro → GLM-5.2 → GLM-5 → V4-Flash |
| `multimodal` | 视觉任务：M3 → GLM-5.2 → V4-Pro（降级为纯文本） |
| `ultra_context` | 大上下文：GLM-5.2 → V4-Pro → M3 |
| `default` | 通用任务：V4-Pro → GLM-5.2 → GLM-5 → V4-Flash |

## 长时任务配置

```toml
[long_horizon]
max_duration_hours = 8.0                   # 最大任务持续时间
checkpoint_interval_minutes = 15.0          # 每 N 分钟保存检查点
evaluation_interval_steps = 10             # 每 N 步自评估
evaluation_interval_minutes = 30.0         # 每 N 分钟自评估
stagnation_threshold = 3                   # 连续相似结果 → 停滞
no_progress_threshold = 5                  # 连续无输出步骤 → 停滞
similarity_threshold = 0.8                 # Jaccard 相似度阈值
max_strategy_switches = 5                  # 每任务最大策略切换次数
checkpoint_base_dir = ".teragent/checkpoints"  # 检查点存储目录
checkpoint_keep_last = 5                   # 每任务保留最近 N 个检查点
```

| 字段 | 类型 | 默认值 | 描述 |
|-------|------|---------|-------------|
| `max_duration_hours` | float | `8.0` | 最大任务持续时间（小时） |
| `checkpoint_interval_minutes` | float | `15.0` | 自动检查点间隔 |
| `evaluation_interval_steps` | int | `10` | 自评估触发（按步骤） |
| `evaluation_interval_minutes` | float | `30.0` | 自评估触发（按时间） |
| `stagnation_threshold` | int | `3` | 连续相似结果判定为停滞 |
| `no_progress_threshold` | int | `5` | 连续无输出步骤判定为停滞 |
| `similarity_threshold` | float | `0.8` | Jaccard 相似度阈值 |
| `max_strategy_switches` | int | `5` | 每任务最大策略切换次数 |
| `checkpoint_base_dir` | string | `".teragent/checkpoints"` | 检查点存储目录 |
| `checkpoint_keep_last` | int | `5` | 每任务保留最近 N 个检查点 |

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

## 完整 agent.toml 参考

以下展示所有可用的配置部分：

```toml
# ===== 模型驱动 =====
[drivers.openai_compatible.deepseek_v4_flash]
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-v4-flash"
compiler = "deepseek_v4"
compiler_variant = "flash"
max_context_tokens = 1_000_000
max_output_tokens = 384_000
thinking_mode = "auto"
cache_aware = true

[drivers.openai_compatible.deepseek_v4_pro]
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-v4-pro"
compiler = "deepseek_v4"
compiler_variant = "pro"
max_context_tokens = 1_000_000
max_output_tokens = 384_000
thinking_mode = "deep"
cache_aware = true

[drivers.openai_compatible.minimax_m3]
base_url = "https://api.minimaxi.com/v1"
api_key_env = "MINIMAX_API_KEY"
model = "minimax-m3"
compiler = "minimax_m3"
max_context_tokens = 1_000_000
max_output_tokens = 384_000
multimodal_enabled = true
desktop_enabled = true
msa_efficient = true

[drivers.openai_compatible.glm_5]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5"
compiler = "glm_5"
max_context_tokens = 200_000
max_output_tokens = 128_000
thinking_mode = "deep"
long_horizon_enabled = true

# ===== 执行管道 =====
[execution.pipeline]
design_driver = "openai_compatible.deepseek_v4_pro"
plan_driver = "openai_compatible.deepseek_v4_pro"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.glm_5"

[execution.pipeline.profiles.budget]
description = "最大成本节省"
design_driver = "openai_compatible.deepseek_v4_flash"
plan_driver = "openai_compatible.deepseek_v4_flash"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.deepseek_v4_flash"

[execution.pipeline.profiles.multimodal]
description = "多模态模式"
design_driver = "openai_compatible.minimax_m3"
plan_driver = "openai_compatible.minimax_m3"
execute_driver = "openai_compatible.minimax_m3"
review_driver = "openai_compatible.minimax_m3"

[execution.pipeline.profiles.quality]
description = "质量优先"
design_driver = "openai_compatible.deepseek_v4_pro"
plan_driver = "openai_compatible.glm_5"
execute_driver = "openai_compatible.glm_5"
review_driver = "openai_compatible.deepseek_v4_pro"

# ===== 智能路由 =====
[routing]
multimodal_driver = "openai_compatible.minimax_m3"
desktop_driver = "openai_compatible.minimax_m3"
long_horizon_driver = "openai_compatible.glm_5"

[routing.monthly_budget]
limit_cny = 500.0
warning_threshold = 0.8
auto_downgrade = true

# ===== 熔断器 =====
[circuit_breaker.models.deepseek_v4_pro]
max_consecutive_failures = 5
cooldown_seconds = 60.0

[circuit_breaker.models.minimax_m3]
max_consecutive_failures = 5
cooldown_seconds = 60.0

[circuit_breaker.models.glm_5]
max_consecutive_failures = 5
cooldown_seconds = 90.0

# ===== 长时任务 =====
[long_horizon]
max_duration_hours = 8.0
checkpoint_interval_minutes = 15.0
evaluation_interval_steps = 10
evaluation_interval_minutes = 30.0

# ===== 降级链 =====
[degradation]
heavy = ["deepseek_v4_pro", "glm_5", "deepseek_v4_flash"]
multimodal = ["minimax_m3", "deepseek_v4_pro"]
default = ["deepseek_v4_pro", "glm_5", "deepseek_v4_flash"]

# ===== 标准配置 =====
[permission]
mode = "plan"
rules = { allow = ["read_file:*", "explore_codebase:*"], deny = ["*:**/.env*"] }

[context_management]
model_token_limit = 1_000_000
compaction_threshold = 0.8
retain_count = 8
```

## 环境变量

| 变量 | 描述 |
|----------|-------------|
| `DEEPSEEK_API_KEY` | DeepSeek API key（V4-Flash 和 V4-Pro） |
| `MINIMAX_API_KEY` | MiniMax API key（M3） |
| `GLM_API_KEY` | 智谱 AI API key（GLM-5 和 GLM-5.2） |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |

TerAgent 使用 `python-dotenv` 自动加载 `.env` 文件。搜索顺序如下：

1. 当前工作目录（`./.env`）— 最高优先级
2. 用户主目录（`~/.env`）
3. 项目源码根目录

找到的第一个文件即被加载；后续文件不会覆盖已有的值。
