# Four-Model Adaptation Guide

This guide covers how to configure and use TerAgent's four-model deep adaptation architecture with **DeepSeek V4**, **MiniMax M3**, **GLM-5**, and **GLM-5.2**.

---

## Table of Contents

- [Overview](#overview)
- [Model Capabilities at a Glance](#model-capabilities-at-a-glance)
- [Configuring the Four Models](#configuring-the-four-models)
- [Choosing the Right Model for Each Use Case](#choosing-the-right-model-for-each-use-case)
- [Per-Model Feature Guide](#per-model-feature-guide)
  - [DeepSeek V4 — Thinking Modes & Cache Awareness](#deepseek-v4--thinking-modes--cache-awareness)
  - [MiniMax M3 — Multimodal & Desktop Operations](#minimax-m3--multimodal--desktop-operations)
  - [GLM-5 — Long-Horizon Autonomous Tasks](#glm-5--long-horizon-autonomous-tasks)
  - [GLM-5.2 — 1M Context & Dual Thinking](#glm-52--1m-context--dual-thinking)
- [Migration Guide: Single-Model to Multi-Model](#migration-guide-single-model-to-multi-model)
- [Common Patterns and Recipes](#common-patterns-and-recipes)
- [Best Practices](#best-practices)
- [Troubleshooting Common Issues](#troubleshooting-common-issues)
- [Quick Reference Card](#quick-reference-card)

---

## Overview

TerAgent's four-model architecture assigns each model to the tasks it does best, maximizing quality while minimizing cost:

| Model | Role | Key Strength |
|-------|------|--------------|
| **DeepSeek V4-Flash** | Lightweight tasks | Fast response, low cost, 1M context |
| **DeepSeek V4-Pro** | Complex reasoning | Deep thinking mode, 1M context, cache-aware |
| **MiniMax M3** | Multimodal & desktop | Image/video understanding, desktop automation, 1M context |
| **GLM-5** | Long-horizon & review | 8-hour autonomous tasks, self-evaluation, strategy switching, 200K context |
| **GLM-5.2** | Ultra-long context & dual thinking | 1M context, High/Max dual thinking, PreservedThinking, 5V-Turbo vision coordination |

The **ModelRouter** automatically selects the optimal model based on a 6-step decision flow:

1. **Intent** — Match task intent to the default model
2. **Multimodal** — Route visual/video content to M3
3. **Context length** — Exclude models with insufficient context window
4. **Long-horizon** — Route extended tasks to GLM-5 or GLM-5.2
5. **Cost** — Downgrade to a cheaper model if budget constrained
6. **Degradation** — Fall back if the primary model is unavailable

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        TerAgent Four-Model Layer                     │
│                                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────┐  ┌───────────────┐ │
│  │ DeepSeek V4  │  │  MiniMax M3  │  │  GLM-5  │  │   GLM-5.2     │ │
│  │ Flash + Pro  │  │  Multimodal  │  │ 8h Auto │  │ 1M + Dual     │ │
│  │ 1M context   │  │  Desktop     │  │ 200K    │  │ Thinking      │ │
│  └──────┬───────┘  └──────┬───────┘  └────┬────┘  └───────┬───────┘ │
│         │                  │               │                │         │
│  ┌──────▼──────────────────▼───────────────▼────────────────▼───────┐│
│  │                     ModelRouter (6 dimensions)                   ││
│  │  Intent → Multimodal → Context → Long-Horizon → Cost → Fallback ││
│  └──────────────────────────────────────────────────────────────────┘│
│         │                                                            │
│  ┌──────▼───────────────────────────────────────────────────────────┐│
│  │               PipelineManager (Design→Plan→Execute→Review)       ││
│  └──────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────┘
```

---

## Model Capabilities at a Glance

| Feature | V4-Flash | V4-Pro | M3 | GLM-5 | GLM-5.2 |
|---------|:--------:|:------:|:--:|:-----:|:-------:|
| Context window | 1M | 1M | 1M | 200K | **1M** |
| Max output | 384K | 384K | 384K | 128K | 128K |
| Thinking mode | auto/quick | deep | — | deep | **High/Max** |
| Multimodal | ❌ | ❌ | ✅ | ❌ | ❌ (via 5V) |
| Desktop ops | ❌ | ❌ | ✅ | ❌ | ❌ |
| Long-horizon | ❌ | ❌ | ❌ | ✅ (8h) | ✅ (8h+) |
| Cache-aware | ✅ | ✅ | MSA | ❌ | ✅ |
| PreservedThinking | ❌ | ❌ | ❌ | ❌ | ✅ |
| Vision coordination | ❌ | ❌ | ❌ | ❌ | ✅ (5V-Turbo) |
| Cost (relative) | ★★★★★ | ★★ | ★★★★ | ★★★ | ★★★ |

---

## Configuring the Four Models

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
# GLM-5.2 — 1M context + dual thinking model
# =============================================================================
[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
max_context_tokens = 1_000_000          # GLM-5.2 supports 1M context
max_output_tokens = 128_000
thinking_mode = "high"                  # Default: "high" (vs "max" for deep)
multimodal_enabled = true               # Enable multimodal (vision coordination)
long_horizon_enabled = true             # GLM-5.2 also supports long-horizon
# Note: dual_thinking, preserved_thinking, vision_coordination, context_degradation
# are create_provider() kwargs, not TOML driver fields.

# =============================================================================
# Execution Pipeline — Multi-model collaboration
# =============================================================================
[execution.pipeline]
design_driver = "openai_compatible.deepseek_v4_pro"    # Design: V4-Pro
plan_driver = "openai_compatible.glm_52"               # Plan: GLM-5.2 (1M context)
execute_driver = "openai_compatible.deepseek_v4_flash"  # Execute: V4-Flash
review_driver = "openai_compatible.glm_5"              # Review: GLM-5 (thorough)

# =============================================================================
# Smart Routing Configuration
# =============================================================================
[routing]
multimodal_driver = "openai_compatible.minimax_m3"     # Multimodal → M3
desktop_driver = "openai_compatible.minimax_m3"        # Desktop → M3
long_horizon_driver = "openai_compatible.glm_52"      # Long-horizon → GLM-5.2
ultra_context_driver = "openai_compatible.glm_52"     # >200K context → GLM-5.2

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
description = "Quality first: GLM-5.2 for planning, V4-Pro for design/review"
design_driver = "openai_compatible.deepseek_v4_pro"
plan_driver = "openai_compatible.glm_52"
execute_driver = "openai_compatible.glm_52"
review_driver = "openai_compatible.deepseek_v4_pro"

[execution.pipeline.profiles.ultra_context]
description = "Ultra-long context: GLM-5.2 for all stages with 1M context"
design_driver = "openai_compatible.glm_52"
plan_driver = "openai_compatible.glm_52"
execute_driver = "openai_compatible.glm_52"
review_driver = "openai_compatible.glm_52"
```

### Environment Variables

Set these in your `.env` file or environment:

```bash
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxx        # Required for V4-Flash and V4-Pro
MINIMAX_API_KEY=xxxxxxxxxxxx            # Required for M3
GLM_API_KEY=xxxxxxxxxxxx.xxxxxx         # Required for GLM-5 and GLM-5.2
```

> **Note:** GLM-5 and GLM-5.2 share the same API key from Zhipu AI. Both are accessed through the same `base_url`.

---

## Choosing the Right Model for Each Use Case

### By Task Type

| Task Type | Recommended Model | Reason |
|-----------|-------------------|--------|
| Chat / Q&A | V4-Flash | Fast, cheap, good quality |
| Code generation | V4-Flash | Quick response, low cost |
| Debugging | V4-Flash | Fast iteration cycles |
| Design / Architecture | V4-Pro | Deep reasoning, full context |
| Planning (large scope) | GLM-5.2 | 1M context, dual thinking |
| Code review | GLM-5 | Thorough analysis, logic verification |
| Image understanding | M3 | Native multimodal support |
| Video processing | M3 | Native video understanding |
| Desktop automation | M3 | Dedicated desktop API |
| Long-horizon tasks | GLM-5.2 | 1M context + 8h+ autonomous |
| Ultra-large codebase (>200K) | GLM-5.2 or M3 | 1M context window |
| Vision + Code tasks | GLM-5.2 + 5V-Turbo | Vision coordination for coding |
| Preserved thinking plans | GLM-5.2 | PreservedThinking feature |
| Multi-step reasoning | GLM-5.2 (Max mode) | Deepest reasoning capability |

### By Context Size

| Context Size | Recommended Model | Fallback |
|-------------|-------------------|----------|
| < 50K tokens | V4-Flash (cost-effective) | V4-Pro |
| 50K–200K tokens | V4-Pro or GLM-5 | V4-Flash |
| 200K–500K tokens | GLM-5.2 or M3 | V4-Pro |
| 500K–1M tokens | GLM-5.2 | M3 |

### By Cost Sensitivity

| Scenario | Strategy | Pipeline Profile |
|----------|----------|------------------|
| Development / prototyping | Use V4-Flash everywhere | `budget` |
| Production (balanced) | Mix models by stage | `default` |
| Quality-critical | V4-Pro + GLM-5.2 | `quality` |
| Visual tasks | M3 for all stages | `multimodal` |
| Ultra-large codebase | GLM-5.2 for all stages | `ultra_context` |

### GLM-5 vs GLM-5.2 Decision Matrix

| Factor | Choose GLM-5 | Choose GLM-5.2 |
|--------|-------------|----------------|
| Context window needed | ≤ 200K | > 200K (up to 1M) |
| Reasoning depth | Standard deep thinking | High/Max dual thinking |
| Coding with vision | Not needed | Need 5V-Turbo coordination |
| PreservedThinking | Not needed | Need coding plan preservation |
| Cost sensitivity | Higher (cheaper per token) | Lower (premium features) |
| Task duration | ≤ 8 hours | 8h+ with degradation support |
| Codebase size | Small–medium | Large–ultra-large |

---

## Per-Model Feature Guide

### DeepSeek V4 — Thinking Modes & Cache Awareness

DeepSeek V4 supports three thinking modes that control reasoning depth:

| Mode | Compiler Variant | Description |
|------|-----------------|-------------|
| `auto` | Flash | Automatically decides whether to think deeply based on complexity |
| `quick` | Flash | Fast responses with minimal reasoning |
| `deep` | Pro | Extended reasoning with full chain-of-thought |

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

**Cache awareness** is a critical V4 feature. DeepSeek's API has a 12x price difference between cache hits and misses:

- The `DeepSeekV4Compiler` automatically freezes system prompts and tool definitions at the start of the message list
- Enable `cache_aware = true` in your driver configuration
- Keep system prompts and tool definitions stable across requests for maximum cache hit rates

### MiniMax M3 — Multimodal & Desktop Operations

MiniMax M3 provides native multimodal capabilities with Anthropic-compatible interface:

```python
from teragent import TAPRequest, create_provider
from teragent.core.tap import MultimodalContent

m3_provider = create_provider(
    compiler="minimax_m3",
    adapter="minimax_native",
    model="minimax-m3",
    base_url="https://api.minimaxi.com/v1",
    api_key_env="MINIMAX_API_KEY",
)

# Alternatively, use the openai_compatible adapter (text-only, no desktop/video)
m3_provider_text = create_provider(
    compiler="minimax_m3",
    adapter="openai_compatible",
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

**Key M3 features:**
- **7 desktop actions**: screenshot, click, type_text, scroll, hotkey, move_mouse, drag
- **5-layer safety system**: Permission → Safe zones → Rate limiting → Ops cap → Blocked shortcuts
- **MSA efficient**: Full-text injection at 1/20 compute cost via Sparse Attention at 1M context
- **Anthropic-compatible**: Supports `count_tokens` and Anthropic-style message format
- **Token estimation**: Provides accurate token counting for multimodal content

### GLM-5 — Long-Horizon Autonomous Tasks

GLM-5 excels at tasks that require extended autonomous execution:

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

# Alternatively, use the glm_native adapter for Zhipu AI-specific optimizations
glm_provider_native = create_provider(
    compiler="glm_5",
    adapter="glm_native",
    model="glm-5",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
)

# Long-horizon task configuration
request = TAPRequest(
    instruction="Implement a complete user management system with auth, roles, and audit logging",
    long_horizon=LongHorizonConfig(
        max_duration_hours=4,
        checkpoint_interval_minutes=15,
        evaluation_interval_steps=10,
    ),
)
```

**Key GLM-5 features:**
- **8-hour autonomous execution** with goal decomposition (DAG topology)
- **Self-evaluation**: Periodic assessment of goal alignment, output quality, and bottleneck detection
- **Strategy switching**: Auto-detects stagnation and switches approach (decompose/backtrack/skip/replan)
- **Checkpoint recovery**: Automatic state snapshots every N minutes for resumability
- **Context window**: 200K tokens (upgrade to GLM-5.2 for 1M context)

### GLM-5.2 — 1M Context & Dual Thinking

GLM-5.2 is the flagship model for ultra-long context and advanced reasoning:

```python
from teragent import create_provider, TAPRequest
from teragent.core.tap import LongHorizonConfig

glm52_provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
)

# Alternatively, use the glm_native adapter for Zhipu AI-specific optimizations
glm52_provider_native = create_provider(
    compiler="glm_52",
    adapter="glm_native",
    model="glm-5.2",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
)

# 1M context with dual thinking
request = TAPRequest(
    instruction="Analyze this entire codebase and design a migration plan to microservices",
    long_horizon=LongHorizonConfig(
        max_duration_hours=6,
        checkpoint_interval_minutes=20,
    ),
)
```

**Key GLM-5.2 features:**

| Feature | Description |
|---------|-------------|
| **1M context window** | Process up to 1,000,000 tokens in a single request |
| **High thinking** | Standard deep reasoning mode (default) |
| **Max thinking** | Maximum reasoning depth for the hardest problems |
| **PreservedThinking** | Preserves reasoning traces across coding sessions for plan continuity |
| **5V-Turbo coordination** | Coordinates with GLM-5V-Turbo for vision→code→verify cycles |
| **Context degradation** | Auto-downgrades from 1M to 200K under memory pressure |
| **Long-horizon** | Extended autonomous tasks (8h+) with checkpoint recovery |

**Dual thinking mode usage:**

```python
from teragent import create_provider

# High thinking — standard deep reasoning (default)
provider_high = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    thinking_mode="high",    # Balanced speed and depth
)

# Max thinking — maximum reasoning depth
provider_max = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    thinking_mode="max",     # Deepest reasoning, slower but most thorough
)
```

**PreservedThinking for coding plans:**

```python
from teragent import create_provider, TAPRequest

provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    preserved_thinking_enabled=True,  # compiler-level kwarg
)

# First request: create a coding plan
plan_request = TAPRequest(
    instruction="Design the architecture for a real-time chat system with WebSocket support",
    meta={"intent": "plan"},
)

# The PreservedThinking feature retains the reasoning trace
# so subsequent code generation stays aligned with the plan
```

**5V-Turbo vision coordination:**

```python
from teragent import create_provider, TAPRequest
from teragent.core.tap import MultimodalContent

provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    vision_coordination_enabled=True,  # compiler-level kwarg
)

# Vision→Code→Verify cycle
request = TAPRequest(
    instruction="Look at this UI mockup and implement the frontend code",
    multimodal_context=[
        MultimodalContent(
            type="image_url",
            image_url={"url": "https://example.com/mockup.png"},
        ),
    ],
)
# GLM-5.2 coordinates with GLM-5V-Turbo for visual analysis
# and then generates code based on the visual understanding
```

---

## Migration Guide: Single-Model to Multi-Model

### Step 1: Add New Model Drivers

Keep your existing single-model configuration and add the new drivers incrementally:

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

# Add GLM-5.2
[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
max_context_tokens = 1_000_000
thinking_mode = "high"
multimodal_enabled = true
# Note: preserved_thinking_enabled and vision_coordination_enabled
# are create_provider() kwargs, not TOML driver fields.
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
plan_driver = "openai_compatible.glm_52"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.glm_5"
```

### Step 3: Add Routing Configuration

Add the `[routing]` section for automatic model selection:

```toml
[routing]
multimodal_driver = "openai_compatible.minimax_m3"
desktop_driver = "openai_compatible.minimax_m3"
long_horizon_driver = "openai_compatible.glm_52"
ultra_context_driver = "openai_compatible.glm_52"

[routing.monthly_budget]
limit_cny = 500.0
warning_threshold = 0.8
auto_downgrade = true
```

### Step 4: Add Pipeline Profiles

Add named profiles for quick switching:

```toml
[execution.pipeline.profiles.budget]
description = "Maximum cost savings: V4-Flash for everything"
design_driver = "openai_compatible.deepseek_v4_flash"
plan_driver = "openai_compatible.deepseek_v4_flash"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.deepseek_v4_flash"

[execution.pipeline.profiles.ultra_context]
description = "Ultra-long context: GLM-5.2 for everything"
design_driver = "openai_compatible.glm_52"
plan_driver = "openai_compatible.glm_52"
execute_driver = "openai_compatible.glm_52"
review_driver = "openai_compatible.glm_52"
```

### Step 5: Update Degradation Chains

Include GLM-5.2 in your degradation chains:

```toml
[degradation]
heavy = ["deepseek_v4_pro", "glm_52", "glm_5", "deepseek_v4_flash"]
multimodal = ["minimax_m3", "glm_52", "deepseek_v4_pro"]
ultra_context = ["glm_52", "deepseek_v4_pro", "minimax_m3"]
default = ["deepseek_v4_pro", "glm_52", "glm_5", "deepseek_v4_flash"]
```

### Step 6: Test Incrementally

1. Start with the `budget` profile (all V4-Flash) to validate the new models
2. Switch to `default` profile to test per-stage routing
3. Add M3 for multimodal tasks
4. Enable GLM-5 for long-horizon tasks
5. Enable GLM-5.2 for ultra-long context and dual thinking
6. Test degradation chains by temporarily disabling one model
7. Validate the `ultra_context` profile with a large codebase

---

## Common Patterns and Recipes

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
plan_driver = "openai_compatible.glm_52"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.glm_5"
```

### Pattern 3: Ultra-Long Context Codebase Analysis

```toml
[execution.pipeline.profiles.codebase_analysis]
description = "For analyzing codebases >200K tokens"
design_driver = "openai_compatible.glm_52"
plan_driver = "openai_compatible.glm_52"
execute_driver = "openai_compatible.glm_52"
review_driver = "openai_compatible.glm_52"
```

### Pattern 4: Vision-Assisted Development

```python
from teragent import create_provider, TAPRequest
from teragent.core.tap import MultimodalContent

# Use GLM-5.2 with 5V-Turbo for vision→code tasks
provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    vision_coordination_enabled=True,  # compiler-level kwarg
)

request = TAPRequest(
    instruction="Analyze this error screenshot and fix the code",
    multimodal_context=[
        MultimodalContent(
            type="image_url",
            image_url={"url": "https://ci.example.com/error.png"},
        ),
    ],
)
```

### Pattern 5: Runtime Profile Switching

```python
from teragent.router import PipelineManager, PipelineProfile

# Switch to ultra_context profile for large codebase analysis
pipeline_manager.set_active_profile("ultra_context")

# Switch back to default for regular tasks
pipeline_manager.set_active_profile("default")

# Create a custom profile on the fly
pipeline_manager.register_profile(PipelineProfile(
    name="glm52_review",
    description="GLM-5.2 for all stages (1M context review)",
    design_driver="openai_compatible.glm_52",
    plan_driver="openai_compatible.glm_52",
    execute_driver="openai_compatible.glm_52",
    review_driver="openai_compatible.glm_52",
))
```

### Pattern 6: Dual Thinking Mode Selection

```python
from teragent import create_provider

# Use "high" thinking for most tasks (balanced speed and depth)
provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    thinking_mode="high",
)

# Switch to "max" thinking for critical decisions
# (This can be done per-request via meta overrides)
request = TAPRequest(
    instruction="Decide whether to refactor or rewrite this module",
    meta={"thinking_mode": "max"},  # Override per-request
)
```

---

## Best Practices

1. **Start with V4-Flash** for new projects, then upgrade specific stages to Pro, GLM-5, or GLM-5.2
2. **Use the `budget` profile** for development and testing to minimize costs
3. **Enable monthly budget** to prevent unexpected charges from long-running sessions
4. **Let the router handle model selection** — only override when you have specific requirements
5. **Use GLM-5 for long-horizon tasks** under 200K context; use GLM-5.2 for tasks exceeding 200K
6. **Use GLM-5.2 Max thinking** only when High thinking isn't sufficient — Max is slower and more expensive
7. **Route multimodal content to M3** — other models will degrade visual content to text descriptions
8. **Leverage V4 cache awareness** — keep system prompts and tool definitions stable across requests
9. **Use PreservedThinking** when working on multi-step coding tasks to maintain plan continuity
10. **Enable context degradation** on GLM-5.2 to handle memory pressure gracefully (1M → 200K)
11. **Monitor circuit breaker states** — check `ModelCircuitBreakerManager.get_all_states()` regularly
12. **Set up fallback chains** — configure degradation chains so your agent never deadlocks when a model is down
13. **Use pipeline profiles** for different environments (dev/staging/prod)
14. **Test degradation paths** — deliberately disable a model and verify fallback works
15. **Consider cost-performance trade-offs** — GLM-5.2's 1M context is powerful but more expensive per token

---

## Troubleshooting Common Issues

### Model Not Found Error

```
KeyError: "openai_compatible.glm_52"
```

**Solution**: Ensure the driver section exists in `agent.toml` and the API key environment variable is set. Check that the compiler name `glm_52` matches the registered compiler.

### Context Length Exceeded with GLM-5

GLM-5 has a 200K token context window. If your request exceeds this:

**Solution**: The ModelRouter automatically routes requests with >200K context to GLM-5.2 or V4-Pro/M3. Ensure `[routing]` is configured with `ultra_context_driver = "openai_compatible.glm_52"`.

### GLM-5.2 1M Context OOM

If the 1M context consumes too much memory:

**Solution**: Context degradation is handled internally by the AutoCompactor. Ensure `max_context_tokens = 1_000_000` is set in the GLM-5.2 driver configuration. Also ensure your inference endpoint has sufficient resources for 1M context.

### Dual Thinking Mode Not Activating

If the thinking mode doesn't switch between High and Max:

**Solution**: Dual thinking is controlled via `thinking_mode` in the driver config (TOML field) or per-request via `meta={"thinking_mode": "max"}`. The `dual_thinking_enabled` kwarg is for `create_provider()` only (compiler-level), not a TOML field. Check that the `GLM52Compiler` is registered and selected.

### 5V-Turbo Coordination Fails

If vision coordination between GLM-5V-Turbo and GLM-5.2 isn't working:

**Solution**: `vision_coordination_enabled` is a `create_provider()` kwarg, not a TOML field. In TOML, use `multimodal_enabled = true` on the GLM-5.2 driver. Check that the GLM-5V-Turbo service is accessible. Verify that the `GLM52Compiler` supports vision coordination mode. If 5V-Turbo is unavailable, the system will degrade to text-only analysis.

### PreservedThinking Context Lost

If PreservedThinking loses context between sessions:

**Solution**: PreservedThinking is designed for within-session continuity, not cross-session persistence. For cross-session work, include the plan summary in the goal description of the new session. `preserved_thinking_enabled` is a `create_provider()` kwarg (not a TOML field); pass it when creating the provider programmatically.

### Multimodal Content Not Processed

If M3 is not available, multimodal content is degraded to text descriptions by other compilers.

**Solution**: Ensure the M3 driver is configured and the API key is valid. Check routing configuration. For GLM-5.2 with vision, pass `vision_coordination_enabled=True` to `create_provider()` or set `multimodal_enabled = true` in TOML.

### High Costs Despite Budget Configuration

**Possible causes**:
- Budget limit is too high
- `auto_downgrade` is set to `false`
- Many requests bypass the router
- GLM-5.2 Max thinking mode is used excessively

**Solution**: Set a reasonable `limit_cny`, enable `auto_downgrade`, ensure all requests go through the ModelRouter, and use Max thinking mode sparingly.

### Circuit Breaker Opens Too Easily

If a model's circuit breaker opens prematurely:

```python
from teragent.reliability.circuit_breaker import ModelCircuitBreakerManager, ModelBreakerConfig

# Customize breaker thresholds for GLM-5.2
manager = ModelCircuitBreakerManager(configs=[
    ModelBreakerConfig(
        model_name="glm_52",
        max_consecutive_failures=10,     # More tolerant for 1M context
        cooldown_seconds=60.0,           # Longer cooldown for recovery
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

**Solution**: Keep system prompts and tool definitions consistent. Use the `cache_aware = true` driver option. The DeepSeek V4 compiler automatically freezes tool definitions at the beginning of the message list.

### Context Degradation Not Triggering on GLM-5.2

**Solution**: Context degradation is handled internally by the AutoCompactor. Ensure `max_context_tokens = 1_000_000` is set in the driver config. The AutoCompactor triggers degradation when memory utilization exceeds the threshold. Check that your inference endpoint reports memory pressure correctly. The degradation log will show the downgrade event.

---

## Quick Reference Card

### Model Selection Quick Guide

```
Need speed?           → V4-Flash
Need depth?           → V4-Pro (deep) or GLM-5.2 (Max)
Need vision?          → M3
Need desktop?         → M3
Need 8h autonomy?     → GLM-5 or GLM-5.2
Need >200K context?   → GLM-5.2
Need vision + code?   → GLM-5.2 (5V-Turbo)
Need coding plans?    → GLM-5.2 (PreservedThinking)
Budget limited?       → V4-Flash (budget profile)
```

### Key Configuration Paths

```
Model drivers:    [drivers.openai_compatible.<name>]
Pipeline:         [execution.pipeline]
Routing:          [routing]
Budget:           [routing.monthly_budget]
Circuit breakers: [circuit_breaker.models.<name>]
Degradation:      [degradation]
Long-horizon:     [long_horizon]
Profiles:         [execution.pipeline.profiles.<name>]
```

---

*This guide is part of the TerAgent documentation. For model-specific deep dives, see the [GLM-5.2 Guide](glm_52_guide.md), [Long-Horizon Guide](long_horizon_guide.md), and [Multimodal Guide](multimodal_guide.md).*
