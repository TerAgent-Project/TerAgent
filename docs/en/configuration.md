# Configuration Guide

TerAgent uses a typed configuration system backed by `agent.toml` files. This document covers all configuration options.

## Configuration File

Create an `agent.toml` in your project root:

```toml
[drivers.openai_compatible.glm]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.1"
compiler = "glm"

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
design_driver = "openai_compatible.glm"
plan_driver = "openai_compatible.glm"
execute_driver = "openai_compatible.glm"
review_driver = "openai_compatible.glm"

[permission]
mode = "plan"
rules = { allow = ["read_file:*", "explore_codebase:*"], deny = ["*:**/.env*", "read_file:/etc/*"] }
```

## Loading Configuration

```python
import teragent

# Load all configuration
full_config = teragent.load_full_config()

# Load specific driver configs
drivers = teragent.load_driver_configs()

# Load pipeline config
pipeline = teragent.load_pipeline_config()

# Create a provider from config
provider = teragent.create_provider_from_config(drivers["openai_compatible.glm"])

# Load typed configuration
typed_config = teragent.load_typed_config()
# → TerAgentConfig with all typed sub-configs
```

## Driver Configuration

Each driver defines a compiler + adapter + model + API key combination:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `base_url` | string | Yes (except mock) | API base URL |
| `api_key_env` | string | Recommended | Environment variable name for API key |
| `api_key` | string | Not recommended | Direct API key value (security risk) |
| `model` | string | Yes | Model identifier string |
| `compiler` | string | Yes | One of: `default`, `glm`, `anthropic`, `deepseek` |

### API Key Security

Always use `api_key_env` (environment variable name) rather than `api_key` (direct value):

```toml
# ✅ Recommended
api_key_env = "GLM_API_KEY"

# ❌ Not recommended (key visible in config file)
api_key = "sk-xxxxxxxxxxxx"
```

## Typed Configuration Modules

### AgentLoopConfig

Controls the agent loop behavior:

| Setting | Default | Description |
|---------|---------|-------------|
| `max_tool_steps` | 50 | Maximum tool-calling steps per conversation |
| `max_streaming_retries` | 2 | Maximum streaming retry attempts |
| `tool_execution_timeout` | 60 | Seconds before tool execution timeout |
| `max_consecutive_tool_failures` | 5 | Stop loop after N consecutive tool failures |
| `intent_tools` | {} | Mapping of intent → allowed tool names |

### CircuitBreakerConfig

Controls reliability thresholds:

| Setting | Default | Description |
|---------|---------|-------------|
| `budget.max_session_tokens` | 10,000,000 | Token budget per session |
| `budget.warning_threshold` | 0.7 | Warn at 70% budget utilization |
| `budget.critical_threshold` | 0.9 | Critical warning at 90% |
| `budget.enable_hard_limit` | false | Block calls at 100% budget |
| `failure_breaker.max_consecutive` | 5 | Open circuit after N consecutive failures |
| `failure_breaker.window_seconds` | 300 | Time window for failure counting |
| `latency_breaker.warn_latency_ms` | 30,000 | Warn when avg latency exceeds this |
| `progress_detector.stall_threshold` | 10 | Steps before stall detection kicks in |

### PermissionConfig

| Setting | Default | Description |
|---------|---------|-------------|
| `mode` | "default" | Permission level: default/plan/bypass/accept_edits/auto |
| `rules` | {} | Permission rules (allow/deny patterns) |

### ContextManagementConfig

| Setting | Default | Description |
|---------|---------|-------------|
| `model_token_limit` | 128,000 | Model's maximum context tokens |
| `compaction_threshold` | 0.8 | Compact when utilization exceeds this |
| `retain_count` | 8 | Messages to retain during compaction |
| `max_compacts` | 5 | Maximum compaction attempts per session |

### StreamingConfig

| Setting | Default | Description |
|---------|---------|-------------|
| `mode` | "auto" | Streaming mode: auto/streaming/batch |
| `max_concurrent` | 10 | Max concurrent read-only tool executions |

### SessionConfig

| Setting | Default | Description |
|---------|---------|-------------|
| `db_path` | ".agent/sessions.db" | SQLite database path |
| `enabled` | true | Whether session persistence is active |

### FileSafetyConfig

| Setting | Default | Description |
|---------|---------|-------------|
| `workspace_root` | "." | Root directory for file operations |
| `enable_2pc` | true | Enable 2-phase commit for file writes |
| `max_file_size` | 10MB | Maximum file size for writes |

### HooksConfig

| Setting | Default | Description |
|---------|---------|-------------|
| `hooks` | [] | List of hook configurations |

## Programmatic Configuration

You can also create configurations programmatically:

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

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GLM_API_KEY` | Zhipu AI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |

TerAgent uses `python-dotenv` to load `.env` files automatically.
