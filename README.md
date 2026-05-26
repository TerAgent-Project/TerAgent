
# TerAgent

**Terminal AI Agent Library — TAP IR + Model-Specific Compilation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Version: 0.0.1](https://img.shields.io/badge/version-0.0.1-orange.svg)](https://github.com/teragent/teragent)


---

TerAgent is a Python library for building production AI agent systems with a **compiler-adapter architecture**. It introduces **TAP IR** (Tool-Augmented Prompt Intermediate Representation) — a model-agnostic in-memory representation that separates *what to ask* from *how to format it*, enabling orthogonal composition of prompt compilers and protocol adapters.

**4 Compilers** × **3 Adapters** = **12 model+protocol combinations**, each optimized for a specific pairing.

---

## Table of Contents

- [Documentation](docs/) — Full documentation with guides and API reference
- [Why TerAgent](#why-teragent)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
  - [TAP IR](#tap-ir)
  - [Compiler × Adapter Combinations](#compiler--adapter-combinations)
  - [Data Flow](#data-flow)
- [Module Reference](#module-reference)
  - [Core (TAP IR + Compiler + Adapter)](#core-tap-ir--compiler--adapter)
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

Optional components use lazy imports — `import teragent` always succeeds, and `ImportError` is raised only when an optional component is actually used without its extra installed.

---

## Quick Start

### 1. Create a Provider

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
| `anthropic` | `openai_compatible` | Claude via OpenRouter | XML tag structured + recency |
| `anthropic` | `anthropic_native` | Claude via Anthropic API | XML tags + system/user separation (Mode B) |
| `deepseek` | `openai_compatible` | DeepSeek models | Minimalist compilation |
| `default` | `mock` | Testing | No HTTP calls |

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
| `teragent.core.compilers.anthropic` | `AnthropicCompiler` | Claude-optimized: XML tag structure + Mode B (system/user separation) |
| `teragent.core.compilers.deepseek` | `DeepSeekCompiler` | DeepSeek-optimized: minimalist prompt format |
| `teragent.core.adapters.openai_compatible` | `OpenAICompatibleAdapter` | OpenAI `/chat/completions` protocol with SSE streaming |
| `teragent.core.adapters.anthropic_native` | `AnthropicNativeAdapter` | Anthropic `/messages` protocol with Anthropic-specific SSE |
| `teragent.core.adapters.mock` | `MockAdapter` | Testing adapter — no HTTP calls |
| `teragent.core.prompts` | `get_system_prompt_for_intent()`, `list_intents()`, `list_compiler_types()` | Centralized prompt management: 9 intents × 4 compiler variants |

**Prompt intents:** `design`, `plan`, `replan`, `execute`, `review`, `chat`, `chat_friendly`, `sub_agent`, `code_generation` (alias for `execute`)

**Compiler types:** `default`, `glm`, `anthropic`, `deepseek`

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

#### 3-Level Sandbox Degradation

| Level | Isolation | Fallback |
|-------|-----------|----------|
| Level 2 | Firecracker microVM | → Docker (Level 1) |
| Level 1 | Docker container (512MB, 1 CPU, 64 PIDs) | → subprocess (Level 0) |
| Level 0 | Subprocess with `rlimit` + `create_subprocess_exec` | — |

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

**API Key Security:** Always use `api_key_env` (environment variable name) rather than `api_key` (direct value). The `ApiKeyVault` resolves keys from environment variables with `.env` file fallback via `python-dotenv`. Use `audit_config_security()` and `audit_env_file()` to scan for leaked keys.

---

## How It Was Built

Every line of code in TerAgent was generated by AI — not a single line was written by hand. The project follows a **Design → Plan → Code → Review** pipeline:

- **Design**: I worked with multiple AI models (including DeepSeek, GLM-5.1) to define the core architecture — TAP as the IR, compiler/adapter orthogonal decoupling, security layers, and more.
- **Plan**: I directed AI to decompose the system into 95 modules, specifying interfaces and dependency relationships, producing detailed task breakdowns.
- **Code**: I instructed GLM-5.1 via natural language to generate code module by module, strictly following the plan.
- **Review**: I directed AI to perform syntax checks, dependency validation, and runnability tests. Based on the feedback, I accepted, revised, or rejected the output.

After the above pipeline, AI automatically compiled the project statistics: ~22,207 lines of Python code (14 sub-modules, 83 source files), ~15,071 lines of tests (44 test files), a test-to-source ratio of 67.9%, version 0.0.1 Alpha, license Apache-2.0. These figures were also AI-generated.

After publication, GLM-5.1 conducted an independent third-party evaluation of the entire codebase in a separate session, awarding an overall score of **7.4/10** (Architecture 9.0, Anti-Hallucination Security 7.5, Engineering Standards 6.5). The evaluation identified the core innovation as the TAP IR + Compiler/Adapter orthogonal composition, noted that the security architecture is essentially an "anti-AI-self-destruction" system, and flagged the main gaps: missing intent-action consistency checks, sandbox degradation requiring user confirmation, and no CI/CD. The full evaluation report is available at [`docs/EVALUATION_GLM5.md`](docs/EVALUATION_GLM5.md) (also AI-generated).

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
