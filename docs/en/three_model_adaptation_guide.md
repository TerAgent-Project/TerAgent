# Three-Model Adaptation Guide

This guide covers how to configure and use TerAgent's three-model deep adaptation architecture with **DeepSeek V4**, **MiniMax M3**, and **GLM-5**.

---

## Table of Contents

- [Overview](#overview)
- [Model Capabilities](#model-capabilities)
- [Configuring the Three Models](#configuring-the-three-models)
- [Choosing the Right Model](#choosing-the-right-model)
- [Differentiated Capabilities](#differentiated-capabilities)
- [Migration from Single-Model](#migration-from-single-model)
- [Common Patterns](#common-patterns)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

TerAgent's three-model architecture assigns each model to the tasks it does best:

| Model | Role | Strength |
|-------|------|----------|
| **DeepSeek V4-Flash** | Lightweight tasks | Fast response, low cost, 1M context |
| **DeepSeek V4-Pro** | Complex reasoning | Deep thinking mode, 1M context, cache-aware |
| **MiniMax M3** | Multimodal & desktop | Image/video understanding, desktop automation, 1M context |
| **GLM-5** | Long-horizon & review | 8-hour autonomous tasks, deep review, 200K context |

The **ModelRouter** automatically selects the optimal model based on a 6-step decision flow:

1. **Intent** — Match task intent to default model
2. **Multimodal** — Route visual/video content to M3
3. **Context length** — Exclude models with insufficient context window
4. **Long-horizon** — Route extended tasks to GLM-5
5. **Cost** — Downgrade to cheaper model if budget constrained
6. **Degradation** — Fall back if primary model is unavailable

---

## Model Capabilities

### DeepSeek V4-Flash

- **Context**: 1,000,000 tokens
- **Max output**: 384,000 tokens
- **Thinking mode**: `auto` / `quick`
- **Compiler variant**: `flash` — minimalist prompt, fast response
- **Cache-aware**: Yes (12x price difference between cache hit and miss)
- **Best for**: Chat, quick code generation, debugging, simple tasks

### DeepSeek V4-Pro

- **Context**: 1,000,000 tokens
- **Max output**: 384,000 tokens
- **Thinking mode**: `deep` — extended reasoning
- **Compiler variant**: `pro` — full prompt, reasoning guidance
- **Cache-aware**: Yes
- **Best for**: Design, complex planning, architecture decisions, deep code review

### MiniMax M3

- **Context**: 1,000,000 tokens
- **Max output**: 384,000 tokens
- **Multimodal**: Image + video understanding
- **Desktop**: Screenshot, click, type, scroll, hotkey
- **MSA efficient**: Full-text injection (1/20 compute cost at 1M context)
- **Best for**: Visual tasks, desktop automation, video analysis, multimodal Q&A

### GLM-5

- **Context**: 200,000 tokens
- **Max output**: 128,000 tokens
- **Thinking mode**: `deep`
- **Long-horizon**: 8-hour autonomous task execution
- **Best for**: Extended autonomous tasks, thorough code review, multi-step refactoring

---

## Configuring the Three Models

### Complete agent.toml Example

```toml
# =============================================================================
# DeepSeek V4 — Flash (lightweight tasks, default model)
# =============================================================================
[drivers.openai_compatible.deepseek_v4_flash]
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-v4-flash"
compiler = "deepseek_v4"
compiler_variant = "flash"              # Flash: minimal prompt, fast response
max_context_tokens = 1_000_000          # V4 supports 1M context
max_output_tokens = 384_000             # V4 max 384K output
thinking_mode = "auto"                  # auto/deep/quick
cache_aware = true                      # Enable cache awareness (12x price diff)

# =============================================================================
# DeepSeek V4 — Pro (complex tasks)
# =============================================================================
[drivers.openai_compatible.deepseek_v4_pro]
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-v4-pro"
compiler = "deepseek_v4"
compiler_variant = "pro"                # Pro: full prompt + reasoning guidance
max_context_tokens = 1_000_000
max_output_tokens = 384_000
thinking_mode = "deep"                  # Pro defaults to deep reasoning
cache_aware = true

# =============================================================================
# MiniMax M3 — Multimodal model
# =============================================================================
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

# =============================================================================
# GLM-5 — Long-horizon task model
# =============================================================================
[drivers.openai_compatible.glm_5]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5"
compiler = "glm_5"
max_context_tokens = 200_000            # GLM-5 200K context
max_output_tokens = 128_000
thinking_mode = "deep"                  # GLM-5 defaults to deep reasoning
long_horizon_enabled = true             # Enable long-horizon mode (8h autonomy)

# =============================================================================
# Execution Pipeline — Multi-model collaboration
# =============================================================================
[execution.pipeline]
design_driver = "openai_compatible.deepseek_v4_pro"   # Design: V4-Pro (deep reasoning)
plan_driver = "openai_compatible.deepseek_v4_pro"     # Plan: V4-Pro (structured planning)
execute_driver = "openai_compatible.deepseek_v4_flash" # Execute: V4-Flash (fast, low cost)
review_driver = "openai_compatible.glm_5"            # Review: GLM-5 (thorough review)

# =============================================================================
# Smart Routing Configuration
# =============================================================================
[routing]
multimodal_driver = "openai_compatible.minimax_m3"    # Multimodal → M3
desktop_driver = "openai_compatible.minimax_m3"       # Desktop → M3
long_horizon_driver = "openai_compatible.glm_5"      # Long-horizon → GLM-5

[routing.monthly_budget]
limit_cny = 500.0                      # Monthly budget cap ¥500
warning_threshold = 0.8                 # Warn at 80%
auto_downgrade = true                   # Auto-downgrade to V4-Flash when over budget

# =============================================================================
# Pipeline Named Profiles
# =============================================================================
[execution.pipeline.profiles.budget]
description = "Maximum cost savings: all stages use V4-Flash"
design_driver = "openai_compatible.deepseek_v4_flash"
plan_driver = "openai_compatible.deepseek_v4_flash"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.deepseek_v4_flash"

[execution.pipeline.profiles.multimodal]
description = "Multimodal mode: all stages use M3"
design_driver = "openai_compatible.minimax_m3"
plan_driver = "openai_compatible.minimax_m3"
execute_driver = "openai_compatible.minimax_m3"
review_driver = "openai_compatible.minimax_m3"

[execution.pipeline.profiles.quality]
description = "Quality first: design/review with V4-Pro, plan/execute with GLM-5"
design_driver = "openai_compatible.deepseek_v4_pro"
plan_driver = "openai_compatible.glm_5"
execute_driver = "openai_compatible.glm_5"
review_driver = "openai_compatible.deepseek_v4_pro"
```

### Environment Variables

Set these in your `.env` file or environment:

```bash
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxx        # Required for V4-Flash and V4-Pro
MINIMAX_API_KEY=xxxxxxxxxxxx            # Required for M3
GLM_API_KEY=xxxxxxxxxxxx.xxxxxx         # Required for GLM-5
```

---

## Choosing the Right Model

### By Task Type

| Task Type | Recommended Model | Reason |
|-----------|-------------------|--------|
| Chat / Q&A | V4-Flash | Fast, cheap, good quality |
| Code generation | V4-Flash | Quick response, low cost |
| Debugging | V4-Flash | Fast iteration cycles |
| Design / Architecture | V4-Pro | Deep reasoning, full context |
| Planning | V4-Pro | Structured output, reasoning |
| Code review | GLM-5 | Thorough analysis, logic verification |
| Image understanding | M3 | Native multimodal support |
| Video processing | M3 | Native video understanding |
| Desktop automation | M3 | Dedicated desktop API |
| Long-horizon tasks | GLM-5 | 8-hour autonomous execution |
| Large codebase (>200K) | V4-Pro or M3 | 1M context window |

### By Cost Sensitivity

| Scenario | Strategy | Pipeline Profile |
|----------|----------|------------------|
| Development / prototyping | Use V4-Flash everywhere | `budget` |
| Production (balanced) | Mix models by stage | `default` |
| Quality-critical | V4-Pro for design/review | `quality` |
| Visual tasks | M3 for all stages | `multimodal` |

---

## Differentiated Capabilities

### V4 Thinking Mode

DeepSeek V4 supports three thinking modes that control reasoning depth:

```python
from teragent import create_provider

# Quick thinking — fast responses
flash_provider = create_provider(
    compiler="deepseek_v4",
    adapter="openai_compatible",
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com",
    api_key_env="DEEPSEEK_API_KEY",
)

# Deep thinking — extended reasoning
pro_provider = create_provider(
    compiler="deepseek_v4",
    adapter="openai_compatible",
    model="deepseek-v4-pro",
    base_url="https://api.deepseek.com",
    api_key_env="DEEPSEEK_API_KEY",
)
```

The compiler variant (`flash` vs `pro`) automatically selects the appropriate prompt strategy:
- **Flash**: Minimalist system prompt, shorter constraint descriptions, faster token output
- **Pro**: Full system prompt, detailed constraints, reasoning guidance injection

### M3 Multimodal

MiniMax M3 provides native multimodal capabilities:

```python
from teragent import TAPRequest, create_provider

m3_provider = create_provider(
    compiler="minimax_m3",
    adapter="minimax_native",
    model="minimax-m3",
    base_url="https://api.minimaxi.com/v1",
    api_key_env="MINIMAX_API_KEY",
)

# Image understanding
request = TAPRequest(
    instruction="Describe what's in this image",
    multimodal_context=[
        MultimodalContent(type="image_url", image_url={"url": "https://example.com/photo.jpg"}),
    ],
)

# Video processing
request = TAPRequest(
    instruction="Summarize this video",
    multimodal_context=[
        MultimodalContent(type="video_url", video_url={"url": "https://example.com/video.mp4"}),
    ],
)
```

### GLM-5 Long-Horizon

GLM-5 supports 8-hour autonomous task execution:

```python
from teragent import TAPRequest, create_provider
from teragent.core.tap import LongHorizonConfig

glm_provider = create_provider(
    compiler="glm_5",
    adapter="openai_compatible",
    model="glm-5",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
)

# Long-horizon task configuration
request = TAPRequest(
    instruction="Implement a complete user management system with auth, roles, and audit logging",
    long_horizon_config=LongHorizonConfig(
        max_duration_hours=4,
        checkpoint_interval_minutes=15,
        evaluation_interval_steps=10,
    ),
)
```

---

## Migration from Single-Model

### Step 1: Add New Model Drivers

Keep your existing single-model configuration and add the new drivers:

```toml
# Your existing configuration (keep it)
[drivers.openai_compatible.glm_5]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5"
compiler = "glm_5"

# Add V4-Flash
[drivers.openai_compatible.deepseek_v4_flash]
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-v4-flash"
compiler = "deepseek_v4"
compiler_variant = "flash"

# Add V4-Pro
[drivers.openai_compatible.deepseek_v4_pro]
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-v4-pro"
compiler = "deepseek_v4"
compiler_variant = "pro"

# Add M3
[drivers.openai_compatible.minimax_m3]
base_url = "https://api.minimaxi.com/v1"
api_key_env = "MINIMAX_API_KEY"
model = "minimax-m3"
compiler = "minimax_m3"
multimodal_enabled = true
desktop_enabled = true
```

### Step 2: Update Pipeline Configuration

Replace the single-model pipeline with multi-model assignments:

```toml
# Before: single model for everything
[execution.pipeline]
design_driver = "openai_compatible.glm_5"
plan_driver = "openai_compatible.glm_5"
execute_driver = "openai_compatible.glm_5"
review_driver = "openai_compatible.glm_5"

# After: specialized model per stage
[execution.pipeline]
design_driver = "openai_compatible.deepseek_v4_pro"
plan_driver = "openai_compatible.deepseek_v4_pro"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.glm_5"
```

### Step 3: Add Routing Configuration

Add the `[routing]` section for automatic model selection:

```toml
[routing]
multimodal_driver = "openai_compatible.minimax_m3"
desktop_driver = "openai_compatible.minimax_m3"
long_horizon_driver = "openai_compatible.glm_5"

[routing.monthly_budget]
limit_cny = 500.0
warning_threshold = 0.8
auto_downgrade = true
```

### Step 4: Add Pipeline Profiles

Add named profiles for quick switching:

```toml
[execution.pipeline.profiles.budget]
description = "Maximum cost savings"
design_driver = "openai_compatible.deepseek_v4_flash"
plan_driver = "openai_compatible.deepseek_v4_flash"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.deepseek_v4_flash"
```

### Step 5: Test Incrementally

1. Start with the `budget` profile (all V4-Flash) to validate the new model
2. Switch to `default` profile to test per-stage routing
3. Add M3 for multimodal tasks
4. Enable GLM-5 for long-horizon tasks

---

## Common Patterns

### Pattern 1: Development with Budget Control

```toml
[execution.pipeline]
design_driver = "openai_compatible.deepseek_v4_flash"
plan_driver = "openai_compatible.deepseek_v4_flash"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.deepseek_v4_flash"

[routing.monthly_budget]
limit_cny = 100.0        # Tight budget for development
auto_downgrade = true     # Auto-downgrade if over budget
```

### Pattern 2: Production Quality

```toml
[execution.pipeline]
design_driver = "openai_compatible.deepseek_v4_pro"
plan_driver = "openai_compatible.deepseek_v4_pro"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.glm_5"
```

### Pattern 3: Multimodal-First

```python
from teragent.router import ModelRouter, RoutingTable

# Force all multimodal content to M3
routing_table = RoutingTable(
    multimodal_driver="openai_compatible.minimax_m3",
    desktop_driver="openai_compatible.minimax_m3",
)
```

### Pattern 4: Runtime Profile Switching

```python
from teragent.router import PipelineManager, PipelineProfile

# Switch to budget profile at runtime
pipeline_manager.set_active_profile("budget")

# Switch back to default
pipeline_manager.set_active_profile("default")

# Create a custom profile on the fly
pipeline_manager.register_profile(PipelineProfile(
    name="review_only",
    description="GLM-5 for all stages (thorough review)",
    design_driver="openai_compatible.glm_5",
    plan_driver="openai_compatible.glm_5",
    execute_driver="openai_compatible.glm_5",
    review_driver="openai_compatible.glm_5",
))
```

---

## Best Practices

1. **Start with V4-Flash** for new projects, then upgrade specific stages to Pro or GLM-5
2. **Use the `budget` profile** for development and testing to minimize costs
3. **Enable monthly budget** to prevent unexpected charges from long-running sessions
4. **Let the router handle model selection** — only override when you have specific requirements
5. **Use GLM-5 for long-horizon tasks** that need more than a few minutes of autonomous execution
6. **Route multimodal content to M3** — other models will degrade visual content to text descriptions
7. **Leverage V4 cache awareness** — keep system prompts and tool definitions stable across requests for maximum cache hit rates
8. **Monitor circuit breaker states** — check `ModelCircuitBreakerManager.get_all_states()` regularly
9. **Set up fallback chains** — configure degradation chains so your agent never deadlocks when a model is down
10. **Use pipeline profiles** for different environments (dev/staging/prod)

---

## Troubleshooting

### Model Not Found Error

```
KeyError: "openai_compatible.deepseek_v4_flash"
```

**Solution**: Ensure the driver section exists in `agent.toml` and the API key environment variable is set.

### Context Length Exceeded with GLM-5

GLM-5 has a 200K token context window. If your request exceeds this:

**Solution**: The ModelRouter automatically routes requests with >200K context to V4-Pro or M3. Ensure `[routing]` is configured.

### Multimodal Content Not Processed

If M3 is not available, multimodal content is degraded to text descriptions by other compilers.

**Solution**: Ensure the M3 driver is configured and the API key is valid. Check routing configuration.

### High Costs Despite Budget Configuration

**Possible causes**:
- Budget limit is too high
- `auto_downgrade` is set to `false`
- Many requests bypass the router

**Solution**: Set a reasonable `limit_cny`, enable `auto_downgrade`, and ensure all requests go through the ModelRouter.

### Circuit Breaker Opens Too Easily

If a model's circuit breaker opens prematurely:

```python
from teragent.reliability.circuit_breaker import ModelCircuitBreakerManager, ModelBreakerConfig

# Customize breaker thresholds
manager = ModelCircuitBreakerManager(configs=[
    ModelBreakerConfig(
        model_name="deepseek_v4_pro",
        max_consecutive_failures=10,     # More tolerant
        cooldown_seconds=30.0,           # Shorter cooldown
    ),
])
```

### Pipeline Profile Not Taking Effect

**Solution**: Ensure you've called `set_active_profile()` and that the profile name matches exactly. Check `pipeline_manager.list_profiles()` for available profiles.

### Cache Hit Rate Low on V4

**Possible causes**:
- System prompts change between requests
- Tool definitions vary per request
- Context is restructured frequently

**Solution**: Keep system prompts and tool definitions consistent. Use the `cache_aware = true` driver option. The DeepSeek V4 compiler automatically freezes tool definitions at the beginning of the message list for maximum cache hit rates.
