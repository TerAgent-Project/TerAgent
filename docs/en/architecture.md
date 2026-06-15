# TerAgent Architecture

This document provides an in-depth look at the TerAgent architecture, design decisions, and module interactions.

## Overview

TerAgent is built around a **compiler-adapter architecture** that introduces **TAP IR** (Tool-Augmented Prompt Intermediate Representation) — a model-agnostic in-memory representation that separates *what to ask* from *how to format it*.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TerAgent Architecture                        │
│                                                                     │
│  ┌──────────┐   ┌───────────┐   ┌────────────────┐   ┌──────────┐ │
│  │ TAPRequest│──▶│ Compiler  │──▶│ CompiledPrompt │──▶│ Adapter  │ │
│  │ (IR)      │   │ (compile) │   │ (model-specific)│   │ (HTTP)  │ │
│  └──────────┘   └───────────┘   └────────────────┘   └────┬─────┘ │
│                                                           │       │
│  ┌───────────┐   ┌───────────────┐   ┌──────────────┐   │       │
│  │ TAPResponse│◀──│ ModelProvider │◀──│  Model API   │◀──┘       │
│  │ (IR)       │   │ (compose C+A)│   │ (GLM/Claude) │           │
│  └───────────┘   └───────────────┘   └──────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Design Principles

### 1. Orthogonal Composition (Compiler × Adapter)

The central insight is that **prompt format** and **API protocol** are two independent dimensions:

| Dimension | Varies By | Examples |
|-----------|-----------|---------|
| Prompt format | Model family | GLM recency effect, Anthropic XML tags, DeepSeek minimalist |
| API protocol | Provider | OpenAI `/chat/completions`, Anthropic `/messages` |

By separating these into **Compiler** and **Adapter**, we get **9 × 5 = 45 combinations** (9 compilers × 5 adapters) instead of 45 separate integrations. In production (excluding the `mock` adapter), that's **9 × 4 = 36 valid combinations**.

**Adding a new model** requires only a new Compiler class. **Adding a new protocol** requires only a new Adapter class. The two compose orthogonally.

### 2. TAP IR — Like LLVM IR, but for Prompts

TAP IR is **not** a wire protocol. It is an in-memory data structure that captures *what* the user wants, not *how* to ask the model. This decoupling enables:

- **Model-agnostic optimization**: The same TAP request can be compiled differently for GLM vs Claude
- **Tool-augmented context**: TAPRequest includes meta, context, instruction, constraints, and output_format_hint
- **Self-RL data collection**: Every TAP request→response pair can be traced for DPO pair generation

### 3. Defense in Depth

Security is not an afterthought — it is woven into every layer:

- **7-layer permission resolution** (user → config → project → system → level → AI → deny)
- **6-layer command defense** (normalization → chain split → blacklist → cross-chain → package → metacharacters) + platform-specific patterns (Windows 16-pattern blacklist, Windows system path protection)
- **2-phase commit file writes** (validate → write temp → atomic swap → rollback) with NTFS 3-step fallback for Windows
- **3-level sandbox degradation** (Firecracker → Docker → subprocess) with cross-platform process management

### 4. Advisory-First Reliability

TerAgent's reliability system defaults to **advisory, not blocking**:

- Budget warnings at 70%, critical at 90%, hard stop only if explicitly enabled
- Circuit breakers open on consecutive failures but auto-recover (half-open)
- Progress detection warns on stall but doesn't kill the loop
- Latency monitoring is purely informational

## Module Dependency Graph

```
                        ┌─────────────────────────┐
                        │      AgentLoop           │
                        │  (Central Orchestration)  │
                        └─────────┬───────────────┘
                                  │
            ┌─────────────────────┼───────────────────────┐
            │                     │                       │
   ┌────────▼────────┐  ┌────────▼────────┐  ┌───────────▼──────────┐
   │   Core (TAP)    │  │   Security      │  │   Reliability         │
   │                 │  │                 │  │                       │
   │ TAPRequest      │  │ PermissionMgr   │  │ CircuitBreakerMgr     │
   │ TAPResponse     │  │ Sandbox         │  │ StepBudget            │
   │ CompiledPrompt  │  │ FileWriter      │  │ RecoveryManager       │
   │ Compiler (ABC)  │  │ AuditLogger     │  │                       │
   │ Adapter (ABC)   │  │ AI Classifier   │  └───────────────────────┘
   │ ModelProvider   │  │                 │
   └────────┬────────┘  └─────────────────┘
            │
   ┌────────▼────────┐  ┌─────────────────┐  ┌──────────────────────┐
   │   Pipeline       │  │   Context       │  │   Coordination       │
   │                 │  │                 │  │                      │
   │ Extractor       │  │ ContextWindow   │  │ SubAgentManager      │
   │ PromptBuilder   │  │ AutoCompactor   │  │ AgentMessageBus      │
   │ Checklist       │  │ Microcompactor  │  │                      │
   │ Retry           │  │ CodeIndexer*    │  └──────────────────────┘
   │ TAPTracer       │  │ ReferenceGraph* │
   └─────────────────┘  │ VectorIndexer*  │
                        └─────────────────┘
                        (* optional dependencies)

   ┌──────────────────────┐  ┌──────────────────────────────────────┐
   │   Router             │  │   Long-Horizon                       │
   │                      │  │                                      │
   │ ModelRouter          │  │ LongHorizonTaskManager               │
   │ RoutingTable         │  │ CheckpointStore                      │
   │ PipelineManager      │  │ SelfEvaluator                        │
   └──────────────────────┘  │ StrategySwitcher                     │
                             │ ProgressTracker                      │
   ┌──────────────────────┐  └──────────────────────────────────────┘
   │   Benchmark          │
   │                      │  ┌──────────────────────────────────────┐
   │ BenchmarkRunner      │  │   Tools                              │
   └──────────────────────┘  │                                      │
                             │ BaseTool, ToolRegistry, Orchestrator │
                             │ DesktopTool (desktop automation)     │
                             └──────────────────────────────────────┘
```

## Data Flow: A Complete TAP Call

### Simple TAP Call (execute_tap)

```python
provider = create_provider(compiler="glm_5", adapter="openai_compatible", ...)
response = await provider.execute_tap(TAPRequest(instruction="..."))
```

1. `TAPRequest` created with meta, context, instruction, constraints, output_format_hint
2. `ModelProvider.execute_tap()` calls `compiler.compile(request)` → `CompiledPrompt`
3. CompiledPrompt is validated against adapter's `required_mode`
4. `adapter.send(compiled, model)` → HTTP request → `TAPResponse`
5. If tracer is attached, request and response are auto-recorded

### Agent Loop TAP Call (with all cross-cutting concerns)

```
User Input
    │
    ▼
┌──────────────────┐
│ IntentClassifier  │  CHAT / DEBUG / CREATE_PROJECT
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│ ConfirmationGate  │  (if CREATE_PROJECT, ask user)
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│ Tool Filtering    │  Filter by intent_tools config
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│ SubAgent Deleg.   │  (if CREATE_PROJECT + SubAgentManager)
└───────┬──────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│           Tool Loop (iterates)            │
│                                          │
│  1. Check StepBudget                     │
│  2. Check ConsecutiveFailureBreaker      │
│  3. Context Compaction (AutoCompactor)   │
│  4. Build API Messages                   │
│  5. Determine streaming vs batch         │
│  6. Call Model                           │
│  7. Handle truncation recovery           │
│  8. Execute tools (Streaming/Orchestrator)│
│  9. Append results, loop to 1            │
│                                          │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────┐
│ Session Persist   │
└──────────────────┘
```

## Registered Compilers & Adapters

### Compilers (9 registered)

| Name | Class | Optimization Strategy |
|------|-------|----------------------|
| `default` | `TAPCompiler` | Standard chat messages |
| `glm` | `TAPCompiler` | Recency effect (key instruction last) |
| `glm_5` | `GLM5Compiler` | Recency effect + long-horizon + self-evaluation |
| `glm_52` | `GLM52Compiler` | 1M context + dual thinking (High/Max) + PreservedThinking + 5V-Turbo coordination |
| `glm_5v_turbo` | `GLM5VTurboCompiler` | Vision analysis for GLM-5V-Turbo model |
| `anthropic` | `TAPCompiler` | XML tag structured + Mode B (system/user separation) |
| `deepseek` | `TAPCompiler` | Minimalist compilation |
| `deepseek_v4` | `DeepSeekV4Compiler` | Cache-aware layout + thinking mode + 1M context optimization (flash/pro via `variant` param) |
| `minimax_m3` | `MiniMaxM3Compiler` | MSA full-text injection + multimodal + desktop context |

> **Note:** `deepseek_v4_flash` and `deepseek_v4_pro` are **NOT** separate compilers — they are variants of `DeepSeekV4Compiler` controlled by the `variant` parameter.

### Adapters (5 registered)

| Name | Class | Protocol |
|------|-------|----------|
| `openai_compatible` | `OpenAICompatibleAdapter` | OpenAI `/chat/completions` with SSE |
| `anthropic_native` | `AnthropicNativeAdapter` | Anthropic `/messages` with Anthropic SSE |
| `glm_native` | `GLMNativeAdapter` | Zhipu AI native API |
| `minimax_native` | `MiniMaxNativeAdapter` | MiniMax native API with rate limit tracking |
| `mock` | `MockAdapter` | No HTTP calls (testing) |

## CompiledPrompt: Two Modes

Compilers produce one of two mutually exclusive prompt formats:

### Mode A: Messages List (OpenAI / GLM / DeepSeek)

```python
CompiledPrompt(
    messages=[
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
    ],
    tools=[...],
    extra={"cache_aware": True},  # Compiler-specific parameters passed to adapter
)
```

### Mode B: System + User Separation (Anthropic Native)

```python
CompiledPrompt(
    system_prompt="...",
    user_message="...",
    tools=[...],
    extra={},  # Extra dict for compiler-specific parameters
)
```

The `CompiledPrompt.mode` property returns `"messages"`, `"system_user"`, or `"empty"`. Adapters check `required_mode` to validate compatibility. The `extra` dict carries compiler-specific parameters (e.g., `cache_aware`, `variant`, `minimax_video_mode`) that adapters use to customize their behavior.

## Compiler Registry Pattern

Compilers and Adapters use a class-level registry pattern:

```python
# Registration (happens at import time)
@classmethod
def register(cls, name: str, compiler_cls: type[TAPCompiler]) -> None:
    cls._get_registry()[name] = compiler_cls

# Creating instances
compiler = TAPCompilerRegistry.create("glm")
adapter = TAPAdapterRegistry.create("openai_compatible", base_url="...", api_key="...")
```

The registry uses `_get_registry()` to ensure subclasses don't share the parent's mutable class variable.

## Event System Design

The `EventBus` follows a **fire-and-forget** design:

- `emit()`: Async handlers scheduled via `create_task`, sync handlers via `run_in_executor`
- `emit_and_wait()`: Waits for all handlers to complete (for critical paths)
- `emit_message()`: Structured Message events with metadata tracking
- `wait_for()`: Await a specific event with timeout
- Error isolation: single handler failure never blocks other handlers
- Event history: tracks last 100 basic events (name + timestamp) and 200 structured events (with full data) for debugging

## Thread Safety

| Component | Thread Safety Mechanism |
|-----------|------------------------|
| `CostTracker` | `threading.Lock` on all operations |
| `TAPTracer` | `threading.Lock` on record writes |
| `EnhancedPermissionManager` | `_rules_dirty` flag + cached sorted rules |
| `CircuitBreakerManager` | Single-threaded async (no lock needed) |
| `EventBus` | Single-threaded async (no lock needed) |
| `AgentLoop` | Single-threaded async (no lock needed) |

## Resource Lifecycle

```
ModelProvider
    ├── Compiler (stateless, reusable)
    ├── Adapter (holds httpx.AsyncClient)
    │       └── close() — must be called to release connections
    ├── CostTracker (thread-safe)
    └── TAPTracer (thread-safe)

AgentLoop
    ├── ToolOrchestrator (per-loop, delegates to ToolRegistry)
    ├── StreamingToolExecutor (per-loop, wraps ToolOrchestrator)
    ├── SessionPersistence (SQLite via aiosqlite)
    └── EventBus (shared across components)

SubAgentManager
    ├── ModelProvider (shared reference)
    ├── ToolRegistry (shared reference)
    └── AgentMessageBus (shared reference)
```

## Cross-Platform Architecture

TerAgent's cross-platform support is built on platform abstraction rather than scattered `if sys.platform` checks:

| Subsystem | Abstraction | Windows | macOS | Linux |
|-----------|-------------|---------|-------|-------|
| Process group | `_kill_process_group()` | `taskkill /F /T` | `os.killpg()` | `os.killpg()` |
| Session creation | Conditional kwargs | `CREATE_NEW_PROCESS_GROUP` | `start_new_session=True` | `start_new_session=True` + `preexec_fn` |
| Command parsing | `shlex.split(cmd, posix=...)` | `posix=False` | `posix=True` | `posix=True` |
| File atomicity | `_sync_atomic_write()` | 3-step rename + backup | `os.replace()` | `os.replace()` |
| Path normalization | `_is_sensitive_path()` | `.lower()` + `\` → `/` | Preserve case | Preserve case |
| Config search | `load_full_config()` | `%APPDATA%\teragent\` | `~/Library/Application Support/teragent/` | `~/.config/teragent/` (XDG) |

**FirecrackerSandbox** raises `RuntimeError` at initialization on non-Linux platforms, providing a clear error message instead of a cryptic runtime failure. The sandbox degradation chain automatically falls back to Level 1 (Docker) or Level 0 (subprocess) on unsupported platforms.
