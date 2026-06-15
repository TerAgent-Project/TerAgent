# Configuration Guide

TerAgent uses a typed configuration system backed by `agent.toml` files. This document covers all configuration options.

## Configuration File

Create an `agent.toml` in your project root (see `examples/agent.toml` for a complete example):

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

## Configuration File Search Paths

TerAgent searches for `agent.toml` in the following locations, in priority order:

| Priority | Location | Platform |
|----------|----------|----------|
| 1 | `./agent.toml` (current working directory) | All |
| 2 | `%APPDATA%\teragent\agent.toml` | Windows |
| 2 | `~/Library/Application Support/teragent/agent.toml` | macOS |
| 2 | `$XDG_CONFIG_HOME/teragent/agent.toml` (default `~/.config/teragent/`) | Linux |
| 3 | `<project_root>/agent.toml` | All |
| 4 | `agent.toml` (fallback) | All |

The first existing file found is used. This allows you to set up a global configuration in your platform's standard config directory that applies to all projects.

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
provider = teragent.create_provider_from_config(drivers["openai_compatible.glm_5"])

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
| `compiler` | string | Yes | One of: `default`, `glm`, `glm_5`, `glm_52`, `glm_5v_turbo`, `anthropic`, `deepseek`, `deepseek_v4`, `minimax_m3` |

### Adapter HTTP Configuration

All HTTP-based adapters (`openai_compatible`, `anthropic_native`, `minimax_native`) accept the following configuration parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ssl_verify` | `bool \| str` | `True` | SSL certificate verification. `True` = system CA, `False` = disable (insecure), `str` = path to custom CA bundle |
| `http2_enabled` | `bool` | `False` | Enable HTTP/2 for the connection pool. Requires `h2` package. Disable for HTTP/1.1-only proxy environments |

```python
from teragent.core.adapters import OpenAICompatibleAdapter

# Enterprise proxy with custom CA
adapter = OpenAICompatibleAdapter(
    base_url="https://api.example.com/v1",
    api_key="...",
    ssl_verify="/path/to/ca-bundle.crt",  # Custom CA certificate
    http2_enabled=False,                    # Disable for HTTP/1.1 proxy
)
```

**Note:** `http2_enabled` defaults to `False` for maximum compatibility. Enable it only when you know the endpoint supports HTTP/2.

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

## New Model Drivers

### DeepSeek V4-Flash Driver

```toml
[drivers.openai_compatible.deepseek_v4_flash]
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-v4-flash"
compiler = "deepseek_v4"
compiler_variant = "flash"              # Flash mode: minimal prompt, fast response
max_context_tokens = 1_000_000          # V4 supports 1M context
max_output_tokens = 384_000             # V4 max 384K output
thinking_mode = "auto"                  # auto/deep/quick
cache_aware = true                      # Enable cache awareness (12x price diff)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_url` | string | — | DeepSeek API base URL |
| `api_key_env` | string | — | Environment variable name for API key |
| `model` | string | — | Must be `"deepseek-v4-flash"` |
| `compiler` | string | — | Must be `"deepseek_v4"` |
| `compiler_variant` | string | `"pro"` | Must be `"flash"` for Flash mode |
| `max_context_tokens` | int | `1_000_000` | Maximum context window |
| `max_output_tokens` | int | `384_000` | Maximum output tokens |
| `thinking_mode` | string | `"auto"` | Thinking mode: `auto`/`deep`/`quick` |
| `cache_aware` | bool | `false` | Enable cache-aware prompt layout |

### DeepSeek V4-Pro Driver

```toml
[drivers.openai_compatible.deepseek_v4_pro]
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-v4-pro"
compiler = "deepseek_v4"
compiler_variant = "pro"                # Pro mode: full prompt + reasoning guidance
max_context_tokens = 1_000_000
max_output_tokens = 384_000
thinking_mode = "deep"                  # Pro defaults to deep reasoning
cache_aware = true
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `compiler_variant` | string | `"pro"` | Must be `"pro"` for Pro mode |
| `thinking_mode` | string | `"deep"` | Pro defaults to deep reasoning |

### MiniMax M3 Driver

```toml
[drivers.openai_compatible.minimax_m3]
base_url = "https://api.minimaxi.com/v1"
api_key_env = "MINIMAX_API_KEY"
model = "minimax-m3"
compiler = "minimax_m3"
max_context_tokens = 1_000_000          # M3 supports 1M context
max_output_tokens = 384_000
multimodal_enabled = true               # Enable multimodal (image+video)
desktop_enabled = true                  # Enable desktop operations
msa_efficient = true                    # MSA full-text injection mode
group_id = ""                           # Optional MiniMax Group ID
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_url` | string | — | MiniMax API base URL |
| `api_key_env` | string | — | Environment variable name for API key |
| `model` | string | — | Must be `"minimax-m3"` |
| `compiler` | string | — | Must be `"minimax_m3"` |
| `multimodal_enabled` | bool | `false` | Enable native multimodal support |
| `desktop_enabled` | bool | `false` | Enable desktop operation support |
| `msa_efficient` | bool | `false` | Enable MSA full-text injection |
| `group_id` | string | `""` | MiniMax Group ID (required for some endpoints) |

### GLM-5 Driver

```toml
[drivers.openai_compatible.glm_5]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5"
compiler = "glm_5"
max_context_tokens = 200_000            # GLM-5 200K context
max_output_tokens = 128_000
thinking_mode = "deep"                  # GLM-5 defaults to deep reasoning
long_horizon_enabled = true             # Enable long-horizon mode (8h autonomy)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | — | Must be `"glm-5"` |
| `compiler` | string | — | Must be `"glm_5"` |
| `max_context_tokens` | int | `200_000` | GLM-5 context window (200K) |
| `long_horizon_enabled` | bool | `false` | Enable long-horizon autonomous mode |

### GLM-5.2 Driver

```toml
[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
max_context_tokens = 1_000_000          # GLM-5.2 supports 1M context
max_output_tokens = 128_000
thinking_mode = "high"                  # Default: "high" (vs "max" for deep reasoning)
dual_thinking_enabled = true            # Enable High/Max dual thinking mode
preserved_thinking_enabled = true       # Enable PreservedThinking for coding plans
vision_coordination_enabled = true      # Enable 5V-Turbo vision coordination
long_horizon_enabled = true             # GLM-5.2 also supports long-horizon
context_degradation_enabled = true      # Auto-downgrade 1M → 200K under pressure
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | — | Must be `"glm-5.2"` |
| `compiler` | string | — | Must be `"glm_52"` |
| `max_context_tokens` | int | `1_000_000` | GLM-5.2 context window (1M) |
| `thinking_mode` | string | `"high"` | Default thinking mode: `high` or `max` |
| `dual_thinking_enabled` | bool | `false` | Enable High/Max dual thinking mode |
| `preserved_thinking_enabled` | bool | `false` | Enable PreservedThinking for coding plans |
| `vision_coordination_enabled` | bool | `false` | Enable 5V-Turbo vision coordination |
| `context_degradation_enabled` | bool | `false` | Auto-downgrade 1M → 200K under memory pressure |

### GLM-5V-Turbo Driver (for Vision Coordination)

```toml
[drivers.openai_compatible.glm_5v_turbo]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5v-turbo"
compiler = "glm_5v_turbo"
```

### GLMNative Driver

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
dual_thinking_enabled = true
preserved_thinking_enabled = true
```

### MiniMaxNative Driver

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
group_id = ""
```

## Smart Routing Configuration

### [routing] Section

Controls automatic model selection based on content type and task characteristics:

```toml
[routing]
# Multimodal content (images/video) → M3
multimodal_driver = "openai_compatible.minimax_m3"

# Desktop operations → M3
desktop_driver = "openai_compatible.minimax_m3"

# Long-horizon tasks → GLM-5
long_horizon_driver = "openai_compatible.glm_5"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `multimodal_driver` | string | `"openai_compatible.minimax_m3"` | Driver for multimodal content |
| `desktop_driver` | string | `"openai_compatible.minimax_m3"` | Driver for desktop context |
| `long_horizon_driver` | string | `"openai_compatible.glm_5"` | Driver for long-horizon tasks |

### [routing.monthly_budget] Section

Monthly cost control with automatic downgrade:

```toml
[routing.monthly_budget]
limit_cny = 500.0              # Monthly budget cap in CNY
warning_threshold = 0.8        # Warn at 80% utilization
critical_threshold = 0.95      # Auto-downgrade at 95%
auto_downgrade = true          # Enable automatic downgrade
auto_downgrade_driver = "openai_compatible.deepseek_v4_flash"  # Fallback driver
notify_on_warning = true       # Emit events on budget warning
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `limit_cny` | float | `0.0` | Monthly budget cap in CNY (0 = no limit) |
| `warning_threshold` | float | `0.8` | Fraction at which to emit warning |
| `critical_threshold` | float | `0.95` | Fraction at which to auto-downgrade |
| `auto_downgrade` | bool | `true` | Whether to auto-downgrade when budget exhausted |
| `auto_downgrade_driver` | string | `"openai_compatible.deepseek_v4_flash"` | Driver to fall back to |
| `notify_on_warning` | bool | `true` | Whether to emit events on budget warning |

## Pipeline Named Profiles

### [execution.pipeline.profiles.*] Sections

Define named pipeline configurations that can be switched at runtime:

```toml
# Budget profile: maximum cost savings
[execution.pipeline.profiles.budget]
description = "Maximum cost savings: all stages use V4-Flash"
design_driver = "openai_compatible.deepseek_v4_flash"
plan_driver = "openai_compatible.deepseek_v4_flash"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.deepseek_v4_flash"

# Multimodal profile: all stages use M3
[execution.pipeline.profiles.multimodal]
description = "Multimodal mode: all stages use M3"
design_driver = "openai_compatible.minimax_m3"
plan_driver = "openai_compatible.minimax_m3"
execute_driver = "openai_compatible.minimax_m3"
review_driver = "openai_compatible.minimax_m3"

# Quality profile: best model for each stage
[execution.pipeline.profiles.quality]
description = "Quality first: design/review with V4-Pro, plan/execute with GLM-5"
design_driver = "openai_compatible.deepseek_v4_pro"
plan_driver = "openai_compatible.glm_5"
execute_driver = "openai_compatible.glm_5"
review_driver = "openai_compatible.deepseek_v4_pro"
```

Each profile section supports:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `description` | string | No | Human-readable profile description |
| `design_driver` | string | Yes | Driver for design stage |
| `plan_driver` | string | Yes | Driver for plan stage |
| `execute_driver` | string | Yes | Driver for execute stage |
| `review_driver` | string | Yes | Driver for review stage |

**Built-in profiles** (automatically available):
- `default` — Uses the base `[execution.pipeline]` settings
- `budget` — All stages use V4-Flash
- `multimodal` — All stages use M3
- `deep_thinking` — All stages use GLM-5.2 (Max thinking mode)

## Per-Model Circuit Breaker Configuration

```toml
[circuit_breaker.models.deepseek_v4_pro]
max_consecutive_failures = 5        # Open breaker after N consecutive failures
window_seconds = 300.0              # Sliding window duration (seconds)
cooldown_seconds = 60.0             # Time before half-open transition
failure_threshold_percent = 0.5     # Open if >50% failures in window
half_open_max_calls = 3             # Test calls allowed in half-open state

[circuit_breaker.models.deepseek_v4_flash]
max_consecutive_failures = 8        # More tolerant for lightweight model
cooldown_seconds = 30.0             # Shorter cooldown

[circuit_breaker.models.minimax_m3]
max_consecutive_failures = 5
cooldown_seconds = 60.0

[circuit_breaker.models.glm_5]
max_consecutive_failures = 5
cooldown_seconds = 90.0             # Longer cooldown for long-horizon model
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_consecutive_failures` | int | `5` | Consecutive failures to open breaker |
| `window_seconds` | float | `300.0` | Sliding window for failure rate |
| `cooldown_seconds` | float | `60.0` | Cooldown before half-open transition |
| `failure_threshold_percent` | float | `0.5` | Failure rate threshold in window |
| `half_open_max_calls` | int | `3` | Test calls in half-open state |

## Degradation Chain Configuration

Controls the fallback order when a model becomes unavailable:

```toml
[degradation]
# Default chains for the four-model architecture
heavy = ["deepseek_v4_pro", "glm_52", "glm_5", "deepseek_v4_flash"]
multimodal = ["minimax_m3", "glm_52", "deepseek_v4_pro"]
ultra_context = ["glm_52", "deepseek_v4_pro", "minimax_m3"]
default = ["deepseek_v4_pro", "glm_52", "glm_5", "deepseek_v4_flash"]
```

| Chain | Description |
|-------|-------------|
| `heavy` | Complex tasks: V4-Pro → GLM-5.2 → GLM-5 → V4-Flash |
| `multimodal` | Visual tasks: M3 → GLM-5.2 → V4-Pro (degrades to text-only) |
| `ultra_context` | Large context: GLM-5.2 → V4-Pro → M3 |
| `default` | General tasks: V4-Pro → GLM-5.2 → GLM-5 → V4-Flash |

## Long-Horizon Task Configuration

```toml
[long_horizon]
max_duration_hours = 8.0                   # Maximum task duration
checkpoint_interval_minutes = 15.0          # Save checkpoint every N minutes
evaluation_interval_steps = 10             # Self-evaluate every N steps
evaluation_interval_minutes = 30.0         # Self-evaluate every N minutes
stagnation_threshold = 3                   # Consecutive similar results → stagnation
no_progress_threshold = 5                  # Consecutive steps without output → stagnation
similarity_threshold = 0.8                 # Jaccard similarity threshold for stagnation
max_strategy_switches = 5                  # Maximum strategy switches per task
checkpoint_base_dir = ".teragent/checkpoints"  # Checkpoint storage directory
checkpoint_keep_last = 5                   # Keep last N checkpoints per task
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_duration_hours` | float | `8.0` | Maximum task duration in hours |
| `checkpoint_interval_minutes` | float | `15.0` | Auto-checkpoint interval |
| `evaluation_interval_steps` | int | `10` | Self-evaluation trigger (by steps) |
| `evaluation_interval_minutes` | float | `30.0` | Self-evaluation trigger (by time) |
| `stagnation_threshold` | int | `3` | Consecutive similar results for stagnation |
| `no_progress_threshold` | int | `5` | Consecutive no-output steps for stagnation |
| `similarity_threshold` | float | `0.8` | Jaccard similarity threshold |
| `max_strategy_switches` | int | `5` | Maximum strategy switches per task |
| `checkpoint_base_dir` | string | `".teragent/checkpoints"` | Checkpoint storage directory |
| `checkpoint_keep_last` | int | `5` | Keep last N checkpoints per task |

## Complete agent.toml Reference

The following shows all available configuration sections:

```toml
# ===== Model Drivers =====
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
group_id = ""

[drivers.openai_compatible.glm_5]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5"
compiler = "glm_5"
max_context_tokens = 200_000
max_output_tokens = 128_000
thinking_mode = "deep"
long_horizon_enabled = true

# ===== Execution Pipeline =====
[execution.pipeline]
design_driver = "openai_compatible.deepseek_v4_pro"
plan_driver = "openai_compatible.deepseek_v4_pro"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.glm_5"

[execution.pipeline.profiles.budget]
description = "Maximum cost savings"
design_driver = "openai_compatible.deepseek_v4_flash"
plan_driver = "openai_compatible.deepseek_v4_flash"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.deepseek_v4_flash"

[execution.pipeline.profiles.multimodal]
description = "Multimodal mode"
design_driver = "openai_compatible.minimax_m3"
plan_driver = "openai_compatible.minimax_m3"
execute_driver = "openai_compatible.minimax_m3"
review_driver = "openai_compatible.minimax_m3"

[execution.pipeline.profiles.quality]
description = "Quality first"
design_driver = "openai_compatible.deepseek_v4_pro"
plan_driver = "openai_compatible.glm_5"
execute_driver = "openai_compatible.glm_5"
review_driver = "openai_compatible.deepseek_v4_pro"

# ===== Smart Routing =====
[routing]
multimodal_driver = "openai_compatible.minimax_m3"
desktop_driver = "openai_compatible.minimax_m3"
long_horizon_driver = "openai_compatible.glm_5"

[routing.monthly_budget]
limit_cny = 500.0
warning_threshold = 0.8
auto_downgrade = true

# ===== Circuit Breakers =====
[circuit_breaker.models.deepseek_v4_pro]
max_consecutive_failures = 5
cooldown_seconds = 60.0

[circuit_breaker.models.minimax_m3]
max_consecutive_failures = 5
cooldown_seconds = 60.0

[circuit_breaker.models.glm_5]
max_consecutive_failures = 5
cooldown_seconds = 90.0

# ===== Long-Horizon Tasks =====
[long_horizon]
max_duration_hours = 8.0
checkpoint_interval_minutes = 15.0
evaluation_interval_steps = 10
evaluation_interval_minutes = 30.0

# ===== Degradation Chains =====
[degradation]
heavy = ["deepseek_v4_pro", "glm_5", "deepseek_v4_flash"]
multimodal = ["minimax_m3", "deepseek_v4_pro"]
default = ["deepseek_v4_pro", "glm_5", "deepseek_v4_flash"]

# ===== Standard Configuration =====
[permission]
mode = "plan"
rules = { allow = ["read_file:*", "explore_codebase:*"], deny = ["*:**/.env*"] }

[context_management]
model_token_limit = 1_000_000
compaction_threshold = 0.8
retain_count = 8
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DEEPSEEK_API_KEY` | DeepSeek API key (V4-Flash and V4-Pro) |
| `MINIMAX_API_KEY` | MiniMax API key (M3) |
| `GLM_API_KEY` | Zhipu AI API key (GLM-5 and GLM-5.2) |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |

TerAgent uses `python-dotenv` to load `.env` files automatically. The following search order is used:

1. Current working directory (`./.env`) — highest priority
2. User home directory (`~/.env`)
3. Project source root directory

The first file found is loaded; subsequent files do not override existing values.
