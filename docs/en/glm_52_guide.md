# GLM-5.2 Usage Guide

This guide covers TerAgent's GLM-5.2 integration, including 1M context usage, High/Max dual thinking, PreservedThinking for coding plans, 5V-Turbo vision coordination, and production best practices.

---

## Table of Contents

- [Overview](#overview)
- [1M Context Usage and Best Practices](#1m-context-usage-and-best-practices)
- [High/Max Dual Thinking Mode](#highmax-dual-thinking-mode)
- [PreservedThinking for Coding Plans](#preservedthinking-for-coding-plans)
- [5V-Turbo Vision Coordination](#5v-turbo-vision-coordination)
- [Configuration Examples](#configuration-examples)
- [Performance Tips and Stability Considerations](#performance-tips-and-stability-considerations)
- [Context Degradation](#context-degradation)
- [Integration with Pipeline Stages](#integration-with-pipeline-stages)
- [Monitoring and Observability](#monitoring-and-observability)
- [Troubleshooting](#troubleshooting)

---

## Overview

GLM-5.2 is Zhipu AI's flagship model for ultra-long context and advanced reasoning. It extends GLM-5's capabilities with a 1M token context window, dual thinking modes, PreservedThinking, and vision coordination.

| Specification | Value |
|--------------|-------|
| Context window | 1,000,000 tokens |
| Max output | 128,000 tokens |
| Thinking modes | High (default), Max |
| API endpoint | `https://open.bigmodel.cn/api/paas/v4` |
| Compiler | `glm_52` |
| Shared API key | Same as GLM-5 (`GLM_API_KEY`) |

### Key Differentiators from GLM-5

| Feature | GLM-5 | GLM-5.2 |
|---------|-------|---------|
| Context window | 200K | 1M |
| Thinking modes | deep | High / Max |
| PreservedThinking | ❌ | ✅ |
| 5V-Turbo coordination | ❌ | ✅ |
| Context degradation | ❌ | ✅ (1M → 200K) |
| Long-horizon | ✅ (8h) | ✅ (8h+) |
| Cost per token | Lower | Higher |

---

## 1M Context Usage and Best Practices

### When to Use 1M Context

The 1M context window is a powerful feature, but it comes with increased cost and latency. Use it judiciously:

**Good use cases:**
- Analyzing entire large codebases (>200K tokens)
- Processing extensive documentation
- Multi-file refactoring across a large project
- Long conversation history with accumulated context
- Comprehensive audit or review of large systems

**When 200K is sufficient:**
- Single-file or small project tasks
- Tasks where only the most recent context matters
- Short conversations with minimal history
- Budget-constrained scenarios

### Configuring 1M Context

```toml
[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
max_context_tokens = 1_000_000          # Enable 1M context
max_output_tokens = 128_000
# Context degradation is handled internally by the AutoCompactor
```

### Loading Large Context Efficiently

```python
from teragent import create_provider, TAPRequest
from teragent.context import ContextWindow

provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
)

# Load a large codebase into context
context = ContextWindow(
    max_tokens=1_000_000,
    reserved_for_output=128_000,
    reserved_for_system=8_000,
)

# The compiler automatically optimizes context layout for 1M window
request = TAPRequest(
    instruction="Analyze this entire codebase and identify architectural issues",
    context={
        "codebase": large_codebase_text,  # Can be >500K tokens
    },
)
```

### Best Practices for 1M Context

1. **Structure your context hierarchically** — Place the most important information at the beginning and end (recency effect optimization by GLM52Compiler)
2. **Enable context degradation** — Automatically downgrades to 200K under memory pressure
3. **Use prefix caching** — Keep system prompts and tool definitions stable
4. **Monitor memory usage** — Track context utilization via `ContextWindow`
5. **Batch related content** — Group related files or documents together rather than splitting across requests
6. **Prefer High thinking for large context** — Max thinking with 1M context can be very slow; use High by default
7. **Set appropriate timeouts** — 1M context requests take longer; configure `timeout` and `multimodal_timeout`

### 1M Context Considerations

1M context consumes significantly more memory than 200K. Ensure your inference endpoint has sufficient resources, and always enable  as a safety net.

---

## High/Max Dual Thinking Mode

GLM-5.2 introduces dual thinking modes that allow you to balance reasoning depth against speed and cost.

### High Thinking (Default)

High thinking provides standard deep reasoning — similar to GLM-5's `deep` mode but optimized for the 1M context window. It's the right choice for most tasks.

```python
from teragent import create_provider

provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    thinking_mode="high",  # Default; balanced speed and depth
)
```

**Characteristics:**
- Response time: Moderate
- Reasoning depth: Standard deep reasoning
- Token consumption: ~1.2x vs. non-thinking mode
- Best for: Code generation, planning, analysis

### Max Thinking

Max thinking activates the deepest reasoning capability. It's significantly slower but provides the most thorough analysis.

```python
from teragent import create_provider

provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    thinking_mode="max",  # Maximum reasoning depth
)
```

**Characteristics:**
- Response time: Slow (2-5x vs. High thinking)
- Reasoning depth: Maximum chain-of-thought
- Token consumption: ~3-5x vs. High thinking
- Best for: Architecture decisions, complex debugging, critical code review

### Per-Request Thinking Mode Override

You can override the thinking mode on a per-request basis:

```python
from teragent import TAPRequest

# Use High thinking for most requests (default)
request = TAPRequest(
    instruction="Generate a REST API endpoint for user registration",
)

# Override to Max thinking for critical decisions
critical_request = TAPRequest(
    instruction="Decide whether to use microservices or monolith for this project",
    meta={"thinking_mode": "max"},  # Per-request override
)
```

### When to Use Each Mode

| Scenario | Recommended Mode | Reason |
|----------|-----------------|--------|
| Code generation | High | Balanced speed and quality |
| Quick analysis | High | Sufficient reasoning depth |
| Architecture decisions | Max | Maximum reasoning needed |
| Complex debugging | Max | Deep causal analysis required |
| Code review (standard) | High | Good trade-off |
| Code review (critical) | Max | Most thorough analysis |
| Large context analysis | High | Max with 1M can be very slow |
| Budget-constrained | High | Max consumes 3-5x more tokens |

---

## PreservedThinking for Coding Plans

PreservedThinking is a unique GLM-5.2 feature that preserves reasoning traces across coding sessions, ensuring that generated code stays aligned with the original plan.

### How PreservedThinking Works

```
Plan Request → Reasoning Trace Generated → Trace Preserved
                                              ↓
Code Request ← Trace Injected ← PreservedThinking
                                              ↓
Generated code stays aligned with original plan
```

### Enabling PreservedThinking

> **Note:** `preserved_thinking_enabled` is a compiler-level kwarg for `create_provider()`, not a TOML driver field. It cannot be set in `agent.toml`.

Programmatically:

```python
from teragent import create_provider

provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    preserved_thinking_enabled=True,  # compiler-level kwarg
)
```

### Using PreservedThinking for Multi-Step Coding

```python
from teragent import create_provider, TAPRequest

provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    preserved_thinking_enabled=True,  # compiler-level kwarg
)
plan_request = TAPRequest(
    instruction="Design a real-time chat system architecture with: "
                "(1) WebSocket connections, (2) message persistence, "
                "(3) typing indicators, (4) read receipts, "
                "(5) file sharing. Include database schema and API design.",
    meta={"intent": "plan"},
)
plan_response = await provider.execute_tap(plan_request)

# Step 2: Implement based on the plan
# PreservedThinking automatically injects the reasoning trace
impl_request = TAPRequest(
    instruction="Now implement the WebSocket connection handler based on the plan",
    meta={"intent": "create"},
)
impl_response = await provider.execute_tap(impl_request)

# Step 3: Continue implementation
# The reasoning trace is still preserved
db_request = TAPRequest(
    instruction="Implement the message persistence layer with the database schema from the plan",
    meta={"intent": "create"},
)
db_response = await provider.execute_tap(db_request)
```

### PreservedThinking Best Practices

1. **Start with a plan request** — Always begin with a `meta={"intent": "plan"}` request to establish the reasoning trace
2. **Keep the session active** — PreservedThinking works within a single session; don't let sessions expire
3. **Reference the plan explicitly** — In subsequent requests, remind the model about specific plan elements
4. **Don't mix PreservedThinking sessions** — Each provider instance has its own preserved context; don't share between unrelated tasks
5. **Monitor token consumption** — PreservedThinking adds to the context; track total tokens used
6. **Use with High thinking for best results** — Max thinking + PreservedThinking can consume significant tokens

### PreservedThinking Limitations

- **Session-scoped** — PreservedThinking does not persist across sessions or process restarts
- **Token overhead** — Each preserved trace adds ~2-5K tokens to the context
- **Not compatible with all compilers** — Only the `glm_52` compiler supports PreservedThinking
- **Maximum 10 preserved traces** — Older traces are automatically summarized when the limit is reached

---

## 5V-Turbo Vision Coordination

5V-Turbo vision coordination enables GLM-5.2 to work with GLM-5V-Turbo (Zhipu AI's vision model) for vision→code→verify cycles.

### Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  GLM-5V-Turbo│────▶│   GLM-5.2    │────▶│ Verification │
│  (Visual     │     │  (Code       │     │  (5V-Turbo   │
│   Analysis)  │     │   Generation)│     │   Re-check)  │
└──────────────┘     └──────────────┘     └──────────────┘
       ↑                                          │
       └──────────── Feedback Loop ───────────────┘
```

### Enabling 5V-Turbo Coordination

> **Note:** `vision_coordination_enabled` is a compiler-level kwarg for `create_provider()`, not a TOML driver field. In TOML, use `multimodal_enabled = true` on the GLM-5.2 driver and configure the vision model endpoint separately.

```toml
[drivers.openai_compatible.glm_52]
multimodal_enabled = true               # Enable multimodal (vision coordination)

# Optional: Configure the vision model endpoint
[drivers.openai_compatible.glm_5v_turbo]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5v-turbo"
compiler = "glm_5v_turbo"
```

### Using Vision Coordination

```python
from teragent import create_provider, TAPRequest
from teragent.core.tap import MultimodalContent

provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    vision_coordination_enabled=True,  # compiler-level kwarg
)
request = TAPRequest(
    instruction="Look at this UI mockup and implement the React component. "
                "Follow the exact layout, colors, and typography shown.",
    multimodal_context=[
        MultimodalContent(
            type="image_url",
            image_url={"url": "https://example.com/mockup.png"},
        ),
    ],
)

response = await provider.execute_tap(request)
# The system automatically:
# 1. Sends the image to GLM-5V-Turbo for visual analysis
# 2. Provides the visual understanding to GLM-5.2
# 3. GLM-5.2 generates code based on the visual analysis
```

### Vision→Code→Verify Cycle

```python
# Complete vision coordination workflow
async def vision_code_verify(image_url: str, requirement: str):
    # Step 1: Visual analysis (5V-Turbo)
    vision_request = TAPRequest(
        instruction="Describe the UI elements, layout, and styling in this image",
        multimodal_context=[
            MultimodalContent(type="image_url", image_url={"url": image_url}),
        ],
    )

    # Step 2: Code generation (GLM-5.2)
    code_request = TAPRequest(
        instruction=f"Based on the visual analysis, implement: {requirement}",
        meta={"vision_context": True},
    )

    # Step 3: Visual verification (5V-Turbo re-check)
    # After generating code, render and compare with original
    verify_request = TAPRequest(
        instruction="Compare the rendered output with the original mockup. "
                    "Identify any discrepancies in layout, colors, or typography.",
        multimodal_context=[
            MultimodalContent(type="image_url", image_url={"url": image_url}),
            MultimodalContent(type="image_url", image_url={"url": rendered_url}),
        ],
    )
```

### Fallback Behavior

When 5V-Turbo is unavailable:
1. **Circuit breaker detects** 5V-Turbo failure
2. **System degrades** to text-only analysis — GLM-5.2 processes the image description as text
3. **Quality reduction** — Without vision, the model relies on textual descriptions of images
4. **Automatic recovery** — When 5V-Turbo becomes available, the system resumes coordination

---

## Configuration Examples

### Minimal Configuration

```toml
[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
max_context_tokens = 1_000_000
```

### Full Feature Configuration

```toml
[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
max_context_tokens = 1_000_000
max_output_tokens = 128_000
thinking_mode = "high"                    # Default thinking mode
multimodal_enabled = true                 # Enable multimodal (vision coordination)
long_horizon_enabled = true               # Enable long-horizon mode
# Note: dual_thinking_enabled, preserved_thinking_enabled, vision_coordination_enabled,
# and context_degradation_enabled are create_provider() kwargs, not TOML fields.
# Use thinking_mode + per-request overrides for dual thinking control.
# Context degradation is handled internally by the AutoCompactor.
```

### Local Deployment Configuration

```toml
[drivers.openai_compatible.glm_52]
base_url = "http://localhost:8004/v1"     # Local inference endpoint
api_key_env = "GLM_API_KEY"               # Or use "local" for no-auth
model = "glm-5.2"
compiler = "glm_52"
max_context_tokens = 1_000_000
# Context degradation is handled internally by the AutoCompactor

[circuit_breaker.models.glm_52]
max_consecutive_failures = 5               # Standard threshold
cooldown_seconds = 60.0                    # Longer cooldown for 1M context
```

### Budget-Conscious Configuration

```toml
[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
max_context_tokens = 200_000              # Limit to 200K to save cost
thinking_mode = "high"                    # Avoid Max mode to save tokens
# Note: preserved_thinking_enabled and vision_coordination_enabled are
# create_provider() kwargs, not TOML fields. Simply don't pass them to disable.
```

---

## Performance Tips and Stability Considerations

### Performance Tips

1. **Use High thinking by default** — Reserve Max thinking for truly complex decisions
2. **Structure prompts efficiently** — Place key information at the beginning and end
3. **Enable prefix caching** — Keep system prompts consistent across requests
4. **Batch related queries** — Process multiple related items in one request rather than many small ones
5. **Set appropriate timeouts** — 1M context requests may take 30-60 seconds
6. **Monitor context utilization** — High utilization indicates efficient use of the context window
7. **Use context degradation** — Automatically handles memory pressure

### Stability Considerations

1. **1M context stability** — At full 1M context, memory usage is high. Enable degradation as a safety net
2. **Max thinking timeout risk** — Max thinking with 1M context can exceed API timeouts. Configure `timeout=300.0` or higher
3. **5V-Turbo availability** — The vision model may have separate rate limits. Configure circuit breakers appropriately
4. **PreservedThinking memory** — Accumulated preserved traces consume context. Monitor total context usage
5. **Long-horizon checkpoint size** — With 1M context, checkpoints can be large. Ensure sufficient disk space

### Memory Optimization

```python
from teragent import create_provider

# Optimized for memory-constrained environments
provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    max_context_tokens=500_000,          # Reduce from 1M to 500K
    thinking_mode="high",                # Avoid Max mode (less thinking tokens)
    preserved_thinking_enabled=False,    # compiler-level kwarg; disable to save context space
    context_degradation_enabled=True,    # compiler-level kwarg; auto-downgrade when memory is tight
)
```

---

## Context Degradation

GLM-5.2 supports automatic context degradation when the system is under memory pressure.

### How Degradation Works

```
1M Context (full capacity)
    ↓ Memory pressure detected (threshold exceeded)
200K Context (degraded mode)
    ↓ Memory pressure resolved
1M Context (recovered)
```

### Configuring Degradation

> **Note:** Context degradation is handled internally by the AutoCompactor. The fields `context_degradation_enabled`, `context_degradation_threshold`, `context_degradation_target`, and `context_degradation_recovery_threshold` are **not** valid TOML driver config fields — they are compiler-level kwargs for `create_provider()`. In TOML, simply set `max_context_tokens = 1_000_000` and the AutoCompactor will handle degradation automatically.

### What Happens During Degradation

1. **Trigger**: Memory utilization exceeds the configured threshold
2. **Downgrade**: Maximum context is reduced from 1M to 200K
3. **Context compaction**: Existing context is compressed to fit within 200K
4. **Information retention**: The GLM52Compiler preserves the most important context (system prompt, recent messages, tool definitions)
5. **Recovery**: When memory pressure subsides, the system can recover to 1M mode
6. **Logging**: All degradation events are logged with timestamps and memory statistics

### Monitoring Degradation

```python
from teragent.context import ContextWindow

# The ContextWindow tracks utilization
utilization = context_window.usage_ratio()
if utilization > 0.9:
    print("⚠️ Context utilization high, consider enabling auto-compaction")
```

---

## Integration with Pipeline Stages

### GLM-5.2 as Plan Driver

GLM-5.2 excels as the plan driver due to its 1M context and dual thinking:

```toml
[execution.pipeline]
plan_driver = "openai_compatible.glm_52"  # 1M context for comprehensive planning
```

### GLM-5.2 for All Stages (Ultra-Context Profile)

```toml
[execution.pipeline.profiles.ultra_context]
description = "GLM-5.2 for everything — maximum context and reasoning"
design_driver = "openai_compatible.glm_52"
plan_driver = "openai_compatible.glm_52"
execute_driver = "openai_compatible.glm_52"
review_driver = "openai_compatible.glm_52"
```

### Mixed Pipeline with GLM-5.2

```toml
[execution.pipeline]
design_driver = "openai_compatible.deepseek_v4_pro"  # V4-Pro for design
plan_driver = "openai_compatible.glm_52"              # GLM-5.2 for planning (1M context)
execute_driver = "openai_compatible.deepseek_v4_flash" # V4-Flash for execution
review_driver = "openai_compatible.glm_5"             # GLM-5 for review (200K sufficient)
```

---

## Monitoring and Observability

### Key Metrics to Monitor

| Metric | Tool | Threshold |
|--------|------|-----------|
| Request latency | TerAgent logs | > 60s for 1M context |
| Thinking mode token usage | CostTracker | Max mode: 3-5x vs High |
| Context degradation events | ContextWindow / AutoCompactor | Any event warrants investigation |
| 5V-Turbo availability | Circuit breaker state | Open = vision unavailable |
| Checkpoint size | File system | > 100MB per checkpoint |

### Setting Up Alerts

```python
from teragent.context import ContextWindow
from teragent.reliability.circuit_breaker import ModelCircuitBreakerManager

# Monitor context utilization
utilization = context_window.usage_ratio()
if utilization > 0.9:
    print("⚠️ Context utilization high, consider enabling auto-compaction")

# Monitor circuit breaker
breaker_manager = ModelCircuitBreakerManager()

# Custom alert logic
async def check_health():
    utilization = context_window.usage_ratio()
    if utilization > 0.9:
        print("⚠️ Context utilization high, consider enabling auto-compaction")

    breaker_state = breaker_manager.get_state("glm_52")
    if breaker_state == "open":
        print("🔴 GLM-5.2 circuit breaker is open — model unavailable")

    breaker_5v = breaker_manager.get_state("glm_5v_turbo")
    if breaker_5v == "open":
        print("⚠️ 5V-Turbo circuit breaker is open — vision coordination degraded")
```

---

## Troubleshooting

### 1M Context Requests Timing Out

**Cause**: 1M context + Max thinking can exceed API timeouts.

**Solution**: Increase timeout and prefer High thinking for large context:

```python
from teragent.core.adapters.openai_compatible import OpenAICompatibleAdapter

adapter = OpenAICompatibleAdapter(
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key="your-key",
    timeout=300.0,            # 5 minutes for 1M context
)
```

### Context Degradation Triggering Too Often

**Cause**: The inference endpoint has insufficient memory for 1M context.

**Solution**:
1. Reduce `max_context_tokens` to 500K or 200K
2. Ensure your inference endpoint has sufficient resources for 1M context
3. Enable prefix caching to reduce memory overhead

### PreservedThinking Consuming Too Many Tokens

**Cause**: Accumulated preserved traces are using significant context.

**Solution**: 
1. Limit the number of preserved traces (default: 10)
2. Manually clear old traces when starting a new sub-task
3. Disable PreservedThinking for tasks that don't benefit from it

### 5V-Turbo Coordination Returning Poor Results

**Cause**: Vision model may misinterpret complex UI mockups or diagrams.

**Solution**:
1. Provide additional textual context alongside the image
2. Use higher-resolution images
3. Break complex mockups into smaller components
4. Fall back to M3 for complex visual tasks (native multimodal)

### GLM-5.2 Not Using 1M Context

**Cause**: The inference endpoint may not support 1M context.

**Solution**:
1. Ensure `max_context_tokens = 1_000_000` in driver config
2. Verify your inference endpoint supports 1M context and is configured accordingly
3. Check endpoint response for context truncation warnings

---

*This guide is part of the TerAgent documentation. For the complete four-model adaptation guide, see [Adaptation Guide](adaptation_guide.md). For long-horizon task details, see [Long-Horizon Guide](long_horizon_guide.md).*
