
# TerAgent

**Terminal AI Agent Library — TAP IR + Model-Specific Compilation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Version: 0.1.3](https://img.shields.io/badge/version-0.1.3-blue.svg)](https://github.com/teragent/teragent)

**English** | [中文](README_zh.md)

---

TerAgent is a Python library for building production AI agent systems with a **compiler-adapter architecture**. It introduces **TAP IR** (Tool-Augmented Prompt Intermediate Representation) — a model-agnostic in-memory representation that separates *what to ask* from *how to format it*, enabling orthogonal composition of prompt compilers and protocol adapters.

**9 Compilers** × **5 Adapters** = **45 model+protocol combinations** (including test adapter), each optimized for a specific pairing.

Now with **DeepSeek V4**, **MiniMax M3**, **GLM-5**, and **GLM-5.2** deep adaptation — intelligent multi-model routing, long-horizon autonomous tasks, native multimodal understanding, dual thinking modes, and desktop automation.

---

## Table of Contents

- [Documentation](docs/) — Full documentation with guides and API reference
- [Why TerAgent](#why-teragent)
- [Four-Model Deep Adaptation](#four-model-deep-adaptation)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Single-Model](#single-model-quick-start)
  - [Multi-Model](#multi-model-quick-start)
- [Architecture](#architecture)
  - [TAP IR](#tap-ir)
  - [Compiler × Adapter Combinations](#compiler--adapter-combinations)
  - [Data Flow](#data-flow)
- [Cross-Platform Compatibility](#cross-platform-compatibility)
- [Module Reference](#module-reference)
  - [Core (TAP IR + Compiler + Adapter)](#core-tap-ir--compiler--adapter)
  - [Router & Pipeline (Multi-Model)](#router--pipeline-multi-model)
  - [Long-Horizon Tasks](#long-horizon-tasks)
  - [Budget & Cost Tracking](#budget--cost-tracking)
  - [Pipeline Primitives](#pipeline-primitives)
  - [AgentLoop (Central Orchestration)](#agentloop-central-orchestration)
  - [Streaming Execution](#streaming-execution)
  - [Security Architecture](#security-architecture)
  - [Reliability System](#reliability-system)
  - [Context Management](#context-management)
  - [Coordination (Sub-Agents)](#coordination-sub-agents)
  - [Intent Classification](#intent-classification)
  - [Hooks System](#hooks-system)
  - [Session Persistence](#session-persistence)
  - [Self-RL Data Constitution](#self-rl-data-constitution)
  - [Configuration System](#configuration-system)
  - [Event Bus](#event-bus)
- [Configuration](#configuration)
- [How It Was Built](#how-it-was-built)
- [Development](#development)
- [License](#license)

---

## Why TerAgent

| Problem | TerAgent Solution |
|---|---|
| Prompt formats differ across models (GLM, Claude, DeepSeek…) | **Compiler** compiles TAP IR into model-specific prompts |
| API protocols differ (OpenAI, Anthropic native…) | **Adapter** handles protocol-specific HTTP I/O |
| Adding a new model requires changing both prompt format and API calls | **Orthogonal composition**: add a Compiler OR an Adapter, not both |
| No structured way to capture agent interactions for self-improvement | **TAPTracer** records every request→response with DPO pair generation |
| Security is an afterthought in most agent frameworks | **7-layer permission resolution**, **6-layer command defense**, **2PC file writes**, **3-level sandbox** |
| Reliability is missing — agents burn tokens on infinite loops | **4 circuit breakers**, streaming retry with batch fallback, context compaction |

---

## Four-Model Deep Adaptation

TerAgent now supports deep adaptation for four leading Chinese AI models, with intelligent routing that automatically selects the best model for each task:

| Model | Role | Key Capabilities | Context Window |
|-------|------|-----------------|----------------|
| **DeepSeek V4-Flash** | Lightweight tasks | Fast response, low cost, cache-aware | 1M tokens |
| **DeepSeek V4-Pro** | Complex reasoning | Deep thinking mode, cache optimization | 1M tokens |
| **MiniMax M3** | Multimodal & desktop | Image/video understanding, desktop automation, MSA | 1M tokens |
| **GLM-5** | Long-horizon & review | 8-hour autonomous tasks, self-evaluation, strategy switching | 200K tokens |
| **GLM-5.2** | Ultra-long context & dual thinking | 1M context, High/Max dual thinking, PreservedThinking, 5V-Turbo vision coordination | 1M tokens |

### Feature Matrix

| Feature | V4-Flash | V4-Pro | M3 | GLM-5 | GLM-5.2 |
|---------|----------|--------|----|---------|---------|
| Fast code generation | ✓✓✓ | ✓✓ | ✓ | ✓✓ | ✓✓ |
| Deep reasoning | — | ✓✓✓ | ✓ | ✓✓✓ | ✓✓✓ |
| Dual thinking mode | — | — | — | — | ✓✓✓ |
| PreservedThinking | — | — | — | — | ✓✓✓ |
| Multimodal (image) | — | — | ✓✓✓ | — | ✓ (5V-Turbo) |
| Video understanding | — | — | ✓✓✓ | — | — |
| Desktop automation | — | — | ✓✓✓ | — | — |
| Long-horizon tasks | — | — | — | ✓✓✓ | ✓✓✓ |
| Vision→Code workflow | — | — | — | — | ✓✓✓ |
| Cache-aware pricing | ✓✓✓ | ✓✓✓ | — | — | — |
| 1M context window | ✓ | ✓ | ✓ | — | ✓ |
| Cost efficiency | ✓✓✓ | ✓ | ✓ | ✓✓ | ✓✓ |

### Smart Routing (6 Steps)

The `ModelRouter` automatically selects the optimal model through a 6-step decision flow:

1. **Multimodal check** → Route visual/video content to M3 or GLM-5.2 + 5V-Turbo
2. **Context length** → Exclude models with insufficient context (>200K → V4/M3/GLM-5.2)
3. **Long-horizon** → Route extended tasks to GLM-5 or GLM-5.2
4. **Intent matching** → Default routing table (design→V4-Pro, plan→GLM-5.2, execute→GLM-5.2, review→V4-Pro, chat→V4-Flash)
5. **Cost evaluation** → Downgrade if monthly budget is constrained
6. **Degradation** → Fall back if primary model is unavailable

### Pipeline Profiles

Switch between named pipeline configurations at runtime:

| Profile | Design | Plan | Execute | Review | Use Case |
|---------|--------|------|---------|--------|----------|
| `default` | V4-Pro | GLM-5.2 | GLM-5.2 | V4-Pro | Production |
| `budget` | V4-Flash | V4-Flash | V4-Flash | V4-Flash | Development |
| `multimodal` | M3 | M3 | M3 | M3 | Visual tasks |
| `deep_thinking` | GLM-5.2 Max | GLM-5.2 Max | GLM-5.2 Max | GLM-5.2 Max | Complex reasoning |

### Documentation

- 📖 [Four-Model Adaptation Guide](docs/en/adaptation_guide.md) — Configuration, migration, best practices
- 📖 [GLM-5.2 Usage Guide](docs/en/glm_52_guide.md) — 1M context, dual thinking, PreservedThinking, 5V-Turbo
- 📖 [Long-Horizon Task Guide](docs/en/long_horizon_guide.md) — 8-hour autonomous tasks
- 📖 [Multimodal Guide](docs/en/multimodal_guide.md) — Image, video, desktop operations
- 📖 [API Reference](docs/en/api-reference.md) — Full API documentation
- 📖 [Configuration Manual](docs/en/configuration.md) — Complete agent.toml reference
- 📖 [Four-Model Evaluation Report](docs/EVALUATION_FOUR_MODELS.md) — Comprehensive benchmark results
- 📖 [GLM-5.2 Stability Report](docs/glm_52_stability_report.md) — Production stability verification
- 📖 [Ascend Deployment Guide](docs/deployment_guide_ascend.md) — Deploying on Huawei Ascend NPU

---

## Installation

```bash
pip install teragent
```

### Optional Dependencies

```bash
pip install teragent[ast]      # CodeIndexer — tree-sitter AST parsing
pip install teragent[graph]    # ReferenceGraph — networkx dependency analysis
pip install teragent[vector]   # VectorIndexer — LanceDB semantic search
pip install teragent[all]      # All optional dependencies
pip install teragent[dev]      # Development tools (pytest, ruff, mypy)
```

**Requirements:** Python 3.10+. On Python 3.10, `tomli` is auto-installed for TOML config support.

**Type Stubs:** TerAgent includes `.pyi` type stub files and a `py.typed` marker for PEP 561 compliance — IDE autocompletion and mypy type checking work out of the box.

Optional components use lazy imports — `import teragent` always succeeds, and `ImportError` is raised only when an optional component is actually used without its extra installed.

---

## Quick Start

### Single-Model Quick Start

### 1. Create a Provider

```python
import teragent

# Method 1: Factory function (recommended)
provider = teragent.create_provider(
    compiler="glm_5",
    adapter="openai_compatible",
    model="glm-5",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
)

# Method 2: From config file
full_config = teragent.load_full_config()
drivers = full_config["drivers"]
provider = teragent.create_provider_from_config(drivers["openai_compatible.glm_5"])

# Method 3: From DriverConfig object
from teragent.config import DriverConfig
driver_cfg = DriverConfig(
    adapter="openai_compatible",
    identity="glm_5",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
    model="glm-5",
    compiler="glm_5",
)
provider = teragent.create_provider(**driver_cfg.to_create_provider_kwargs())

# Method 4: Async context manager (auto-cleanup)
async with teragent.create_provider(
    compiler="glm_5", adapter="openai_compatible", model="glm-5",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
) as provider:
    response = await provider.execute_tap(teragent.TAPRequest(...))
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
    auto_compactor=AutoCompactor(
        context_window=ContextWindow(model_token_limit=128_000),
        model=provider,
    ),
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

---

### Multi-Model Quick Start

Configure all four models for intelligent routing:

```toml
# agent.toml
[drivers.openai_compatible.deepseek_v4_flash]
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-v4-flash"
compiler = "deepseek_v4"
compiler_variant = "flash"

[drivers.openai_compatible.deepseek_v4_pro]
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-v4-pro"
compiler = "deepseek_v4"
compiler_variant = "pro"

[drivers.openai_compatible.minimax_m3]
base_url = "https://api.minimaxi.com/v1"
api_key_env = "MINIMAX_API_KEY"
model = "minimax-m3"
compiler = "minimax_m3"

[drivers.openai_compatible.glm_5]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5"
compiler = "glm_5"

[execution.pipeline]
design_driver = "openai_compatible.deepseek_v4_pro"
plan_driver = "openai_compatible.deepseek_v4_pro"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.glm_5"

[routing]
multimodal_driver = "openai_compatible.minimax_m3"
desktop_driver = "openai_compatible.minimax_m3"
long_horizon_driver = "openai_compatible.glm_5"

[routing.monthly_budget]
limit_cny = 500.0
warning_threshold = 0.8
auto_downgrade = true
```

```python
import teragent
from teragent.router import ModelRouter, RoutingTable

# Load multi-model configuration
config = teragent.load_full_config()

# The ModelRouter automatically selects the best model
router = ModelRouter(
    available_providers={...},
    routing_table=RoutingTable(),
)

# Route a TAP request — multimodal content goes to M3 automatically
request = teragent.TAPRequest(
    instruction="Analyze this screenshot",
    multimodal_context=[...],
)
decision = router.route(request)
# decision.selected_driver → "openai_compatible.minimax_m3"
```

📖 See the [Four-Model Adaptation Guide](docs/en/adaptation_guide.md) for complete configuration and migration instructions.

---

## Architecture

### TAP IR

TAP (TerAgent Protocol) is an in-memory intermediate representation — like LLVM IR, but for LLM prompts. It is **not** a wire protocol.

```
┌─────────────────────────────────────────────────────────────────┐
│                        TAP IR                                   │
│                                                                 │
│  TAPRequest                          TAPResponse                │
│  ┌──────────────────────┐            ┌───────────────────┐     │
│  │ meta: dict           │            │ raw_text: str     │     │
│  │ context: dict        │            │ usage: dict       │     │
│  │ instruction: str     │            └───────────────────┘     │
│  │ constraints: list    │                                      │
│  │ output_format_hint   │                                      │
│  └──────────────────────┘                                      │
│           │                                                     │
│           ▼                                                     │
│  CompiledPrompt (two mutually exclusive modes)                  │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ Mode A: messages list                                │      │
│  │   [{role, content}, ...]    ← OpenAI / GLM / DeepSeek│      │
│  │                                                      │      │
│  │ Mode B: system_prompt + user_message                 │      │
│  │   system + user separation  ← Anthropic native       │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### Compiler × Adapter Combinations

| Compiler | Adapter | Target | Prompt Strategy |
|----------|---------|--------|-----------------|
| `default` | `openai_compatible` | Generic OpenAI-protocol models | Standard chat messages |
| `glm` | `openai_compatible` | GLM series (Zhipu AI) | Recency effect optimization — key instruction last |
| `glm_5` | `openai_compatible` | GLM-5 (long-horizon) | Deep reasoning + long-horizon task support |
| `glm_5` | `glm_native` | GLM-5 (native features) | Native thinking + cache + async chat |
| `glm_52` | `openai_compatible` | GLM-5.2 (1M context, dual thinking) | 1M context optimization + High/Max dual thinking + PreservedThinking |
| `glm_52` | `glm_native` | GLM-5.2 (native features) | Native thinking + cache + async + reasoning_content |
| `glm_5v_turbo` | `openai_compatible` | GLM-5V-Turbo (vision) | Vision analysis + multimodal prompt |
| `glm_5v_turbo` | `glm_native` | GLM-5V-Turbo (native vision) | Native vision API + image understanding |
| `anthropic` | `openai_compatible` | Claude via OpenRouter | XML tag structured + recency |
| `anthropic` | `anthropic_native` | Claude via Anthropic API | XML tags + system/user separation (Mode B) |
| `deepseek` | `openai_compatible` | DeepSeek V3 models | Minimalist compilation |
| `deepseek_v4` | `openai_compatible` | DeepSeek V4-Flash/Pro | Cache-aware layout + thinking mode + 1M context optimization |
| `minimax_m3` | `openai_compatible` | MiniMax M3 (text) | MSA full-text injection |
| `minimax_m3` | `minimax_native` | MiniMax M3 (multimodal/desktop) | Native multimodal + video + desktop + rate limit tracking |
| `default` | `mock` | Testing | No HTTP calls |

> **Note:** `deepseek_v4_flash` and `deepseek_v4_pro` are not separate compilers — they are variants of `deepseek_v4` set via the `variant` constructor parameter (`DeepSeekV4Compiler(variant="flash")` or `variant="pro"`).

Adding a new model requires only a new Compiler class. Adding a new protocol requires only a new Adapter class. The two are composed orthogonally via `ModelProvider`.

### Data Flow

```
User Input
    │
    ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  TAPRequest  │───▶│    Compiler     │───▶│  CompiledPrompt  │
│  (IR)        │    │  (compile IR)   │    │  (model-specific)│
└──────────────┘    └─────────────────┘    └────────┬─────────┘
                                                     │
                                                     ▼
                                            ┌──────────────────┐
                                            │     Adapter      │
                                            │  (HTTP I/O)      │
                                            └────────┬─────────┘
                                                     │
                                                     ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  TAPResponse │◀───│  ModelProvider  │◀───│  Model API       │
│  (IR)        │    │  (compose C+A)  │    │  (GLM/Claude/…)  │
└──────────────┘    └─────────────────┘    └──────────────────┘
```

---

## Cross-Platform Compatibility

TerAgent supports **Windows**, **macOS**, and **Linux** with platform-specific adaptations:

| Feature | Windows | macOS | Linux |
|---------|---------|-------|-------|
| Sandbox Level 0 | ✅ `CREATE_NEW_PROCESS_GROUP` | ✅ `start_new_session` | ✅ `start_new_session` + `preexec_fn` |
| Sandbox Level 1 (Docker) | ✅ `ContainerUser` | ✅ uid/gid mapping | ✅ uid/gid mapping |
| Sandbox Level 2 (Firecracker) | ❌ KVM required | ❌ KVM required | ✅ Full support |
| Process tree kill | ✅ `taskkill /F /T` | ✅ `os.killpg()` | ✅ `os.killpg()` |
| Windows dangerous commands | ✅ 16 patterns blocked | N/A | N/A |
| Windows system path protection | ✅ `C:\Windows`, `Program Files`, etc. | N/A | N/A |
| Clipboard (X11) | N/A | ✅ `pbcopy`/`pbpaste` | ✅ `xclip` |
| Clipboard (Wayland) | N/A | N/A | ✅ `wl-copy`/`wl-paste` |
| Screenshot | ✅ PIL ImageGrab | ✅ PIL ImageGrab | ✅ `mss` preferred |
| Screen size | ✅ `ctypes` fallback | ✅ `AppKit` fallback | ✅ `mss`/`pyautogui` |
| Config search | `%APPDATA%\teragent\` | `~/Library/Application Support/teragent/` | `~/.config/teragent/` (XDG) |
| `.env` search | CWD → `~/.env` → project | CWD → `~/.env` → project | CWD → `~/.env` → project |
| File atomic write | ✅ 3-step rename + backup | ✅ Atomic `os.replace()` | ✅ Atomic `os.replace()` |
| `shlex` parsing | ✅ `posix=False` | ✅ `posix=True` | ✅ `posix=True` |
| HTTP/2 | ✅ Configurable (default off) | ✅ Configurable | ✅ Configurable |
| SSL verify | ✅ Custom CA support | ✅ Custom CA support | ✅ Custom CA support |

### Key Cross-Platform Features

- **Process Management**: Windows uses `CREATE_NEW_PROCESS_GROUP` + `taskkill /F /T`; Unix uses `start_new_session` + `os.killpg()`
- **Command Safety**: Platform-specific dangerous command blacklists — Unix (`sudo`, `rm -rf /`, `/etc/`) and Windows (`format C:`, `reg delete`, `powershell -enc`, `taskkill`)
- **File Writes**: POSIX uses atomic `os.replace()`; Windows NTFS uses rename-rename-delete 3-step pattern with backup/rollback
- **Desktop Clipboard**: Auto-detects Wayland vs X11 on Linux; uses `pbcopy` on macOS; `clip` on Windows
- **Config Paths**: Searches platform-standard directories (XDG, AppData, Application Support) in addition to CWD and project root

---

## Module Reference

### Core (TAP IR + Compiler + Adapter)

| Module | Key Classes | Description |
|--------|------------|-------------|
| `teragent.core.tap` | `TAPRequest`, `TAPResponse`, `CompiledPrompt`, `TAPCostRecord`, `CostTracker` | TAP IR data structures — the model-agnostic contract between user intent and model API |
| `teragent.core.compiler` | `TAPCompiler` (ABC), `TAPCompilerRegistry` | Compiler abstract base class + name→class registry. Subclasses implement `compile()` to produce model-specific prompts |
| `teragent.core.adapter` | `TAPAdapter` (ABC), `TAPAdapterRegistry` | Adapter abstract base class + name→class registry. Subclasses implement `send()` and `stream()` for protocol-specific HTTP |
| `teragent.core.provider` | `ModelProvider` | Composes Compiler + Adapter. Entry point for `execute_tap()`, `stream_tap()`, `chat()`, `execute_tap_with_retry()`, `chat_with_fallback()` |
| `teragent.core.types` | `Message`, `MessageRole`, `MessageType`, `ToolSafety` | Internal message types and tool safety enumeration (`READ_ONLY`, `SAFE_WRITE`, `DESTRUCTIVE`, `HIGH_RISK`) |
| `teragent.core.compilers.default` | `DefaultCompiler` | Generic OpenAI-compatible prompt compilation |
| `teragent.core.compilers.glm` | `GLMCompiler` | GLM-optimized: recency effect (key instruction last) |
| `teragent.core.compilers.glm_5` | `GLM5Compiler` | GLM-5: recency effect + 200K extreme compression + long-horizon task support |
| `teragent.core.compilers.glm_52` | `GLM52Compiler` | GLM-5.2: 1M context optimization + High/Max dual thinking + PreservedThinking |
| `teragent.core.compilers.glm_5v_turbo` | `GLM5VTurboCompiler` | GLM-5V-Turbo: vision analysis + multimodal prompt optimization |
| `teragent.core.compilers.anthropic` | `AnthropicCompiler` | Claude-optimized: XML tag structure + Mode B (system/user separation) |
| `teragent.core.compilers.deepseek` | `DeepSeekCompiler` | DeepSeek-optimized: minimalist prompt format |
| `teragent.core.compilers.deepseek_v4` | `DeepSeekV4Compiler` | DeepSeek V4: cache-aware layout + thinking mode + Flash/Pro variants + 1M context |
| `teragent.core.compilers.minimax_m3` | `MiniMaxM3Compiler` | MiniMax M3: multimodal + MSA full-text injection + desktop ops |
| `teragent.core.adapters.openai_compatible` | `OpenAICompatibleAdapter` | OpenAI `/chat/completions` protocol with SSE streaming |
| `teragent.core.adapters.anthropic_native` | `AnthropicNativeAdapter` | Anthropic `/messages` protocol with Anthropic-specific SSE |
| `teragent.core.adapters.glm_native` | `GLMNativeAdapter` | GLM native API: thinking + cache + async chat + reasoning_content |
| `teragent.core.adapters.minimax_native` | `MiniMaxNativeAdapter` | MiniMax M3 native: Anthropic-compatible + OpenAI dual interface + video + desktop |
| `teragent.core.adapters.mock` | `MockAdapter` | Testing adapter — no HTTP calls |
| `teragent.core.prompts` | `get_system_prompt_for_intent()`, `list_intents()`, `list_compiler_types()` | Centralized prompt management: 9 intents × 9 compiler variants |

**Prompt intents:** `design`, `plan`, `replan`, `execute`, `review`, `chat`, `chat_friendly`, `sub_agent`, `code_generation` (alias for `execute`)

**Compiler types:** `default`, `glm`, `glm_5`, `glm_52`, `glm_5v_turbo`, `anthropic`, `deepseek`, `deepseek_v4`, `minimax_m3`

### Router & Pipeline (Multi-Model)

| Component | Description |
|-----------|-------------|
| `ModelRouter` | 6-step intelligent routing (multimodal→context→long-horizon→intent→cost→degradation) |
| `RoutingTable` | Configurable routing rules with intent defaults and override maps |
| `RoutingDecision` | Captures routing choice with full trace for debugging |
| `PipelineManager` | Runtime pipeline profile switching (default/budget/multimodal/deep_thinking) |
| `PipelineProfile` | Named stage→driver mapping for quick configuration |

### Long-Horizon Tasks

| Component | Description |
|-----------|-------------|
| `LongHorizonTaskManager` | Orchestrates 8-hour autonomous tasks with GLM-5 |
| `SubGoal` / `PhaseResult` / `LongHorizonResult` | Task decomposition and result tracking |
| `CheckpointStore` | JSON-based checkpoint persistence with auto-cleanup |
| `SelfEvaluator` | Periodic self-assessment (goal alignment, output quality, 1-5 scoring) |
| `StrategySwitcher` | Detects stagnation and triggers strategy changes (6 built-in strategies) |
| `ProgressTracker` / `ProgressReport` | Real-time progress tracking with ETA estimation |
| `LongHorizonRecoveryManager` | Checkpoint-based recovery with exponential backoff |

### Budget & Cost Tracking

| Component | Description |
|-----------|-------------|
| `CrossModelCostTracker` | Multi-model cost tracking with monthly budget control |
| `MonthlyBudgetConfig` | Budget limits with warning thresholds and auto-downgrade |
| `CostRecord` | Per-call cost record with cache savings tracking |
| `ModelCircuitBreakerManager` | Per-model circuit breakers with degradation chains |
| `DegradationChain` | Task-type-aware fallback ordering (heavy/multimodal/default) |
| `RateLimitHandler` | Legacy per-adapter rate limit handling (see `RateLimiter` for centralized abstraction) |

### Rate Limiting

| Component | Description |
|-----------|-------------|
| `TokenBucketRateLimiter` | Token bucket rate limiter — allows bursty traffic, then refills at steady rate |
| `SlidingWindowRateLimiter` | Sliding window rate limiter — tracks request timestamps within a time window |
| `AdaptiveRateLimiter` | Adaptive rate limiter — learns from `X-RateLimit-*` and `Retry-After` response headers, auto-adjusts on 429s |
| `RateLimitConfig` | Configuration for all strategies (token-bucket, sliding-window, adaptive) |
| `RateLimitStrategy` | Strategy enum: `TOKEN_BUCKET`, `SLIDING_WINDOW`, `ADAPTIVE` |
| `RateLimitStatus` | Current rate limit status with `is_limited` and `wait_seconds` properties |
| `create_rate_limiter()` | Factory function — creates the appropriate limiter based on config |

### Pipeline Primitives

| Module | Key Functions | Description |
|--------|--------------|-------------|
| `teragent.pipeline.extractor` | `extract_files_from_response()` | Parse `<file>` tags from model output |
| `teragent.pipeline.prompt_builder` | `build_prompt()`, `build_subagent_prompt()`, `validate_prompt_tokens()` | Template-based prompt construction with token budget validation |
| `teragent.pipeline.checklist` | `run_deterministic_checks()`, `check_code_quality()`, `check_runnable()` | Deterministic code verification (AST, syntax, import, conflict checks) |
| `teragent.pipeline.retry` | `retry_with_backoff()` | Exponential backoff retry with configurable validation |
| `teragent.pipeline.tracing` | `TAPTracer`, `DPOPair`, `DataConstitution`, `TraceStats` | Self-RL trace recording + DPO pair generation (see [Self-RL Data Constitution](#self-rl-data-constitution)) |

### AgentLoop (Central Orchestration)

`AgentLoop` is the main orchestration class that composes all cross-cutting concerns into a cohesive tool-calling loop.

**Lifecycle per user input:**

```
1. IntentClassifier → CHAT / DEBUG / CREATE_PROJECT
2. ConfirmationGate → (if CREATE_PROJECT, ask user approval)
3. Filter tools by intent (from config.intent_tools)
4. SubAgent delegation (if CREATE_PROJECT + SubAgentManager available)
5. Tool loop:
   a. Check step budget
   b. Context compaction (if approaching token limit)
   c. Call model (streaming or batch)
   d. If tool_calls → execute them (StreamingToolExecutor or ToolOrchestrator)
   e. Append tool results, loop to (a)
   f. If text-only → done
6. Emit events, persist session
```

**Cross-cutting concerns integrated into AgentLoop:**

| Concern | Component | Integration Point |
|---------|-----------|-------------------|
| Cost tracking | `CircuitBreakerManager` | Records every model call's token usage |
| Failure protection | `ConsecutiveFailureBreaker` | Opens circuit on N consecutive failures |
| Latency monitoring | `LatencyBreaker` | Warns on consistently slow calls |
| Progress detection | `ProgressDetector` | Detects stuck loops (no meaningful progress) |
| Permission checks | `EnhancedPermissionManager` | Validates tool calls before execution |
| Intent classification | `IntentClassifier` | Routes user input to appropriate behavior |
| Context management | `ContextWindow` + `AutoCompactor` | Compacts context when approaching token limit |
| Streaming mode | `StreamingToolExecutor` | Auto-detects streaming capability, retries on failure, falls back to batch |
| Session persistence | `SessionPersistence` | Saves/restores conversation state |
| Hook system | `HookManager` | Pre/post execution hooks for customization |
| Sub-agent coordination | `SubAgentManager` | Spawns child agents for complex tasks |
| Event bus | `EventBus` | Signal-driven event emission throughout the lifecycle |

### Streaming Execution

`StreamingToolExecutor` processes model stream events and executes tools in real time, significantly reducing latency.

**Dispatch strategy (authoritative):**

| Tool Safety Attributes | Execution Strategy |
|------------------------|--------------------|
| `read_only` + `concurrency_safe` | **Immediate** async execution during stream (no waiting) |
| Non-read-only or non-concurrency-safe | **Queued** for serial execution after stream ends |
| Unknown tool | **Queued** (conservative default) |

**Degradation path:**

```
Streaming + tool_use → Streaming retry → Batch fallback
         ↓ failed         ↓ failed          ↓
     retry N times     fall back to     ToolOrchestrator
                       batch mode       .execute_batch()
```

### Security Architecture

TerAgent provides defense-in-depth security across multiple layers.

#### 7-Layer Permission Resolution

```
Layer 1: user rules     (priority 100) ─┐
Layer 2: config rules   (priority 60)  ─┤ These are PermissionRules
Layer 3: project rules  (priority 50)  ─┤ with glob matching on
Layer 4: system rules   (priority 10)  ─┘ tool_name + path
Layer 5: PermissionLevel check           ← DEFAULT / PLAN / BYPASS / ACCEPT_EDITS / AUTO
Layer 6: AI Classifier (async only)     ← consultative, uses LLM to judge intent
Layer 7: Default DENY                   ← safe default when no rule matches
```

**PermissionRule example:**

```python
from teragent.security import EnhancedPermissionManager, PermissionRule, PermissionEffect

epm = EnhancedPermissionManager()

# User-level DENY: never read /etc
epm.add_rule(PermissionRule(
    effect=PermissionEffect.DENY,
    tool_pattern="read_file",
    path_pattern="/etc/*",
    description="Block reading system directories",
    source="user",  # highest priority
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
# allowed = False, reason = "Denied by rule: Block reading system directories"
```

#### 6-Layer Command Defense

```
Layer 1: Command normalization    ← strip ANSI, null bytes, compress whitespace
Layer 2: Pipeline chain splitting  ← check each sub-command in | && ; chains
Layer 3: 8-category blacklist     ← privilege escalation, reverse shell, inline exec,
                                     system destroy, persistence, encoding bypass,
                                     remote exec, fork bomb / disk write
Layer 4: Dangerous redirect detection ← > /etc/, > /dev/, > /sys/ (fine-grained per sub-command)
Layer 5: Cross-chain detection    ← curl | sh, wget | python (visible only in full command)
Layer 6: Package install warning  ← pip/npm/apt install → log warning, no hard block
```

**Cross-platform extensions:** + platform-specific patterns (Windows 16-pattern blacklist, Windows system path protection)

#### 2-Phase Commit (2PC) File Writes

```
Phase 1: Validate  → check permissions + path traversal + read-before-write contract
Phase 2: Write     → write all files to .tmp suffix
Phase 3: Commit    → os.replace() atomic swap (all succeed or all roll back)
Phase 4: Rollback  → on any commit failure, restore from .bak backups
```

**Key properties:**
- **Atomic**: `os.replace()` is atomic on both POSIX and Windows
- **Crash-safe**: intermediate temp files prevent corruption on crash
- **Consistent**: all files commit or none do (transactional)
- **Concurrent-safe**: readers never see half-written state
- **Path traversal protection**: all paths must be within `workspace_root`
- **NTFS fallback**: with NTFS 3-step fallback (rename-rename-delete with backup/rollback on Windows)

#### 3-Level Sandbox Degradation

| Level | Isolation | Fallback |
|-------|-----------|----------|
| Level 2 | Firecracker microVM | → Docker (Level 1) |
| Level 1 | Docker container (512MB, 1 CPU, 64 PIDs) | → subprocess (Level 0) |
| Level 0 | Subprocess with `rlimit` + `create_subprocess_exec` | — |

**Cross-platform process management:** Windows uses `CREATE_NEW_PROCESS_GROUP` + `taskkill /F /T`; Unix uses `start_new_session` + `os.killpg()`

### Reliability System

Four independent circuit breakers protect against token waste and infinite loops.

| Breaker | What It Detects | Behavior |
|---------|-----------------|----------|
| **CostBudgetTracker** | Token budget approaching limits | Advisory warnings at 70%, critical at 90%, optional hard stop at 100% |
| **ConsecutiveFailureBreaker** | N consecutive API failures | Opens circuit → pauses calls → half-open after cooldown |
| **LatencyBreaker** | Consistently slow model calls | Advisory warning (does not block) |
| **ProgressDetector** | Agent loop making no meaningful progress | Advisory stall warning when ≥80% recent steps had no effect |

**Additional reliability features:**
- **Streaming retry with batch fallback**: auto-retry streaming calls, degrade to batch on persistent failure
- **Context compaction**: automatic `AutoCompactor` when approaching token limit
- **Step budget**: hard limit on total tool-calling steps per conversation
- **Recovery manager**: handles output truncation (`finish_reason="length"`), context overflow errors, and provider fallback

**RecoveryType enum:**

| Type | Trigger |
|------|---------|
| `LENGTH` | Output token truncation → continuation request |
| `CONTEXT_OVERFLOW` | Input context exceeds model token limit → compaction + retry |
| `FALLBACK` | Primary model fails → switch to fallback provider |
| `STREAMING_RETRY` | Streaming call fails → retry or degrade to batch |
| `TOOL_REPAIR` | Tool execution failure → repair retry |

### Context Management

| Component | Description |
|-----------|-------------|
| `ContextWindow` | Token budget estimator with CJK-aware heuristic. Conservative estimation (×1.3 factor) to avoid API overflow. |
| `Microcompactor` | Fine-grained context reduction — removes low-information messages while preserving key context |
| `AutoCompactor` | Automatic compaction trigger based on `ContextWindow.should_compact()` |
| `CodeIndexer` | tree-sitter AST indexing for code structure understanding (`teragent[ast]`) |
| `ReferenceGraph` | networkx-based dependency graph analysis (`teragent[graph]`) |
| `VectorIndexer` | LanceDB semantic code search (`teragent[vector]`) |
| `DependencyReporter` | Generates dependency reports for TAP context (lazy-loaded, requires optional deps) |
| `Memory` | `load_agent_md()` / `save_agent_md()` — persistent project memory via `.agent.md` files |

### Coordination (Sub-Agents)

`SubAgentManager` creates and manages child agent lifecycles with three execution modes:

| Mode | Behavior | Use Case |
|------|----------|----------|
| `SYNC` | Blocks parent until child completes | Simple sub-tasks that must finish before continuing |
| `ASYNC` | Runs in background, notifies parent via `AgentMessageBus` on completion | Long-running background tasks |
| `FORK` | Like SYNC but marks shared system prompt prefix for KV cache optimization | Repeated queries with shared context |

**Safety constraints:**
- Maximum 15 steps per sub-agent (prevents infinite loops)
- Maximum 5 concurrent sub-agents (prevents resource exhaustion)
- Tool whitelist — sub-agents can only use explicitly allowed tools
- Budget tracking — sub-agents respect the global step budget

### Intent Classification

| Component | Description |
|-----------|-------------|
| `IntentClassifier` | Classifies user input into `CHAT`, `DEBUG`, or `CREATE_PROJECT` intents |
| `ConfirmationGate` | Requires explicit user approval before CREATE_PROJECT intent execution |

Intent classification feeds into the tool filtering system — different intents get different tool subsets via `AgentLoopConfig.intent_tools`.

### Hooks System

| Component | Description |
|-----------|-------------|
| `HookManager` | Manages pre/post execution hooks with `HookDecision` (ALLOW / DENY / MODIFY) |
| `Hook` (ABC) | Base class for hooks — `ShellHook` (command hooks) and `PythonHook` (Python callable hooks) |
| `AuditHook` | Built-in hook that logs all tool executions for audit trail |
| `DangerousCommandHook` | Built-in hook that blocks dangerous shell commands using the 6-layer defense |

### Session Persistence

`SessionPersistence` provides full conversation lifecycle management with SQLite-backed storage:

- Create/restore sessions by ID
- Save individual messages per session
- Track step counts
- List session history

### Self-RL Data Constitution

TerAgent includes a complete self-reinforcement learning data pipeline. Every TAP call can be automatically traced and paired with deterministic verification results to produce DPO (Direct Preference Optimization) training pairs.

**Data Constitution Principles:**

1. **TAP traces are a core library output**, independent of specific agent flows
2. **Preference labels come from deterministic checks** (AST, syntax, runnability), not from human annotation
3. **Data belongs to the user** — the library never uploads traces

**DPO Pair Generation:**

```
TAPRequest  →  TAPTracer.record_request()  →  JSONL trace
TAPResponse →  TAPTracer.record_response() →  JSONL trace
Checklist   →  TAPTracer.record_checklist() →  JSONL trace
                                                    ↓
                                     TAPTracer.export_dpo_pairs()
                                                    ↓
                                   (chosen=PASS, rejected=FAIL) pairs
```

**Pairing strategies:**

| Strategy | Description |
|----------|-------------|
| Same-task retry | Same `task_id` has both PASS and FAIL responses from retries → `(chosen=PASS, rejected=FAIL)` |
| Cross-task | Different `task_id`s with the same intent → pair PASS from one with FAIL from another |
| Partial | Only chosen or only rejected available (when `include_partial=True`) |

### Configuration System

TerAgent uses a typed configuration system backed by `agent.toml` files.

**Available config modules:**

| Config Module | Key Class | Controls |
|---------------|-----------|----------|
| `teragent.config.teragent_config` | `TerAgentConfig` | Top-level configuration container |
| `teragent.config.agent_loop_config` | `AgentLoopConfig` | Agent loop behavior (max steps, streaming retries, tool timeouts, intent→tool mapping) |
| `teragent.config.circuit_breaker_config` | `CircuitBreakerConfig` | Budget thresholds, failure limits, latency thresholds, stall detection |
| `teragent.config.streaming_config` | `StreamingConfig` | Streaming mode and retry behavior |
| `teragent.config.permission_config` | `PermissionConfig` | Permission mode and rules |
| `teragent.config.context_management_config` | `ContextManagementConfig` | Context window limits and compaction thresholds |
| `teragent.config.tools_config` | `ToolsConfig` | Tool registry configuration |
| `teragent.config.file_safety_config` | `FileSafetyConfig` | File write safety and 2PC behavior |
| `teragent.config.session_config` | `SessionConfig` | Session persistence settings |
| `teragent.config.hooks_config` | `HooksConfig` | Hook registration |
| `teragent.config.recovery_config` | `RecoveryConfig` | Recovery strategy configuration |
| `teragent.config.coordination_config` | `CoordinationConfig` | Sub-agent coordination settings |
| `teragent.config.execution_pipeline_config` | `ExecutionPipelineConfig` | Pipeline stage driver assignments |
| `teragent.config.model_fallback_config` | `ModelFallbackConfig` | Model fallback chain configuration |
| `teragent.config.driver_config` | `DriverConfig` | Individual model driver (compiler + adapter + model + API key) |
| `teragent.config.api_key_security` | `ApiKeyVault`, `SecurityFinding` | API key resolution, masking, and security auditing |

### Event Bus

`EventBus` is the signal-driven communication backbone of TerAgent.

**Key methods:**

| Method | Description |
|--------|-------------|
| `emit()` | Fire-and-forget event emission (never blocks the main loop) |
| `emit_and_wait()` | Emit event and wait for all handlers to complete |
| `emit_message()` | Emit structured `Message` events with metadata |
| `on()` / `once()` | Subscribe to events (permanent / one-time) |
| `wait_for()` | Wait for a specific event with timeout |

**Design principles:**
- Fire-and-forget: async handlers via `create_task`, sync handlers via `run_in_executor`
- Error isolation: single handler failure does not affect other handlers
- Event history: tracks last 200 events with structured data for debugging

---

## Configuration

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

**API Key Security:** Always use `api_key_env` (environment variable name) rather than `api_key` (direct value). The `ApiKeyVault` resolves keys from environment variables with `.env` file fallback via `python-dotenv`. Use `audit_config_security()` and `audit_env_file()` to scan for leaked keys.

---

## How It Was Built

Every line of code in TerAgent was generated by AI — not a single line was written by hand. The project follows a **Design → Plan → Code → Review** pipeline:

- **Design**: I worked with multiple AI models (including DeepSeek, GLM-5) to define the core architecture — TAP as the IR, compiler/adapter orthogonal decoupling, security layers, and more.
- **Plan**: I directed AI to decompose the system into 95 modules, specifying interfaces and dependency relationships, producing detailed task breakdowns.
- **Code**: I instructed GLM-5 via natural language to generate code module by module, strictly following the plan.
- **Review**: I directed AI to perform syntax checks, dependency validation, and runnability tests. Based on the feedback, I accepted, revised, or rejected the output.

After the above pipeline, AI automatically compiled the project statistics: ~46,900 lines of Python code (17 sub-modules, 99 source files), ~28,300 lines of tests (61 test files), a test-to-source ratio of 60.4%, version 0.1.3 Beta, license Apache-2.0. These figures were also AI-generated.

After publication, GLM-5 conducted an independent third-party evaluation of the entire codebase in a separate session, awarding an overall score of **7.4/10** (Architecture 9.0, Anti-Hallucination Security 7.5, Engineering Standards 6.5). The evaluation identified the core innovation as the TAP IR + Compiler/Adapter orthogonal composition, noted that the security architecture is essentially an "anti-AI-self-destruction" system, and flagged the main gaps: missing intent-action consistency checks, sandbox degradation requiring user confirmation, and no CI/CD. The evaluation reports are available at [`docs/EVALUATION_THREE_MODELS.md`](docs/EVALUATION_THREE_MODELS.md) (three-model) and [`docs/EVALUATION_FOUR_MODELS.md`](docs/EVALUATION_FOUR_MODELS.md) (four-model, also AI-generated).

This development methodology is itself part of TerAgent: the `pipeline` module provides a reusable **Design → Plan → Code → Review** workflow.

---

## Development

```bash
# Install with development dependencies
pip install teragent[dev]

# Run tests
pytest

# Lint
ruff check teragent/

# Type check
mypy teragent/
```

---

## License

Apache License Version 2.0
