# Getting Started with TerAgent

This guide walks you through installing, configuring, and using TerAgent.

## Installation

### Basic Installation

```bash
pip install teragent
```

### Optional Dependencies

TerAgent uses lazy imports — `import teragent` always succeeds. Optional components raise `ImportError` only when actually used.

```bash
pip install teragent[ast]      # CodeIndexer — tree-sitter AST parsing
pip install teragent[graph]    # ReferenceGraph — networkx dependency analysis
pip install teragent[vector]   # VectorIndexer — LanceDB semantic search
pip install teragent[all]      # All optional dependencies
pip install teragent[dev]      # Development tools (pytest, ruff, mypy)
```

**Requirements:** Python 3.10+. On Python 3.10, `tomli` is auto-installed for TOML config support.

## Quick Start

### 1. Create a Provider

A `ModelProvider` composes a Compiler (prompt strategy) and an Adapter (HTTP protocol):

```python
import teragent

# Method 1: Factory function (recommended)
provider = teragent.create_provider(
    compiler="glm",
    adapter="openai_compatible",
    model="glm-5.1",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
)

# Method 2: From config file
full_config = teragent.load_full_config()
drivers = full_config["drivers"]
provider = teragent.create_provider_from_config(drivers["openai_compatible.glm"])

# Method 3: From DriverConfig object
from teragent.config import DriverConfig
driver_cfg = DriverConfig(
    adapter="openai_compatible",
    identity="glm",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
    model="glm-5.1",
    compiler="glm",
)
provider = teragent.create_provider(**driver_cfg.to_create_provider_kwargs())
```

### 2. Execute a TAP Request

```python
response = await provider.execute_tap(teragent.TAPRequest(
    meta={"task_id": "1.1", "intent": "code_generation"},
    instruction="Implement user login module",
    constraints=["Python 3.10+"],
    output_format_hint="<file path='...'>complete code</file>",
))

print(response.raw_text)
print(f"Tokens: {response.total_tokens}")
```

### 3. Extract Files & Run Checks

```python
# Extract files from the model response
files = teragent.extract_files_from_response(response.raw_text, task_id="1.1")

# Run deterministic code quality checks
task_list = [teragent.TaskInfo(id="1.1", title="Login module", status="completed")]
report, data = teragent.run_deterministic_checks("/project", task_list)
```

### 4. Build a Full Agent

```python
from teragent import AgentLoop, ModelProvider, ToolRegistry
from teragent.config import AgentLoopConfig
from teragent.reliability import CircuitBreakerManager, StepBudget
from teragent.security import EnhancedPermissionManager
from teragent.context import ContextWindow, AutoCompactor
from teragent.intent import IntentClassifier
from teragent.streaming import StreamingToolExecutor

# Build the agent loop with all cross-cutting concerns
loop = AgentLoop(
    model=provider,
    tool_registry=my_tool_registry,
    config=AgentLoopConfig(),
    circuit_breaker=CircuitBreakerManager(),
    step_budget=StepBudget(max_steps=50),
    permission_manager=EnhancedPermissionManager(),
    context_window=ContextWindow(model_token_limit=128_000),
    auto_compactor=AutoCompactor(context_window=..., model=provider),
    intent_classifier=IntentClassifier(provider),
    streaming_executor=StreamingToolExecutor(my_tool_registry),
)

# Run the agent
messages = await loop.run("Help me build a Snake game in Python")
```

### 5. Self-RL Data Collection (DPO Pairs)

```python
# Attach a tracer to auto-record all TAP calls
tracer = teragent.TAPTracer(trace_dir="/project/.agent/traces")
provider.set_tracer(tracer)

# ... execute TAP calls ...

# Record checklist results (deterministic PASS/FAIL labels)
await tracer.record_checklist("1.1", checklist_data)

# Export DPO preference pairs for fine-tuning
pairs = tracer.export_dpo_pairs()
tracer.export_dpo_pairs_jsonl()  # Write to JSONL file
```

## Available Compilers & Adapters

### Compilers

| Compiler | Optimization Strategy | Target Models |
|----------|----------------------|---------------|
| `default` | Standard chat messages | Generic OpenAI-protocol models |
| `glm` | Recency effect (key instruction last) | GLM series (Zhipu AI) |
| `anthropic` | XML tag structured + Mode B | Claude series |
| `deepseek` | Minimalist compilation | DeepSeek models |

### Adapters

| Adapter | Protocol | Notes |
|---------|----------|-------|
| `openai_compatible` | OpenAI `/chat/completions` with SSE | Works with GLM, DeepSeek, OpenRouter, etc. |
| `anthropic_native` | Anthropic `/messages` with Anthropic SSE | Direct Anthropic API |
| `mock` | No HTTP calls | For testing |

### Valid Combinations

| Compiler | Adapter | Target | Prompt Strategy |
|----------|---------|--------|-----------------|
| `default` | `openai_compatible` | Generic OpenAI-protocol models | Standard chat messages |
| `glm` | `openai_compatible` | GLM series (Zhipu AI) | Recency effect optimization |
| `anthropic` | `openai_compatible` | Claude via OpenRouter | XML tags + recency |
| `anthropic` | `anthropic_native` | Claude via Anthropic API | XML tags + system/user separation (Mode B) |
| `deepseek` | `openai_compatible` | DeepSeek models | Minimalist compilation |
| `default` | `mock` | Testing | No HTTP calls |

## Next Steps

- [Architecture Guide](architecture.md) — Deep dive into design decisions
- [Security Guide](security.md) — Permission system, sandbox, file writes
- [Configuration Guide](configuration.md) — agent.toml and typed config
- [API Reference](api-reference.md) — Complete module reference
- [Self-RL Guide](self-rl.md) — TAP tracing and DPO pair generation
