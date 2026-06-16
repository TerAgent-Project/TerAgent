# TerAgent 架构

本文档深入介绍 TerAgent 的架构、设计决策和模块交互。

## 概述

TerAgent 围绕 **编译器-适配器架构** 构建，引入了 **TAP IR**（Tool-Augmented Prompt Intermediate Representation，工具增强提示中间表示）——一种与模型无关的内存表示，将 *请求什么* 与 *如何格式化* 分离开来。

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

## 核心设计原则

### 1. 正交组合（Compiler × Adapter）

核心洞察在于 **提示格式** 和 **API 协议** 是两个独立的维度：

| 维度 | 随...而变化 | 示例 |
|------|-----------|------|
| 提示格式 | 模型系列 | GLM 近因效应、Anthropic XML 标签、DeepSeek 极简 |
| API 协议 | 服务商 | OpenAI `/chat/completions`、Anthropic `/messages` |

通过将它们分离为 **Compiler** 和 **Adapter**，我们获得 **9 × 5 = 45 种组合**（9 个编译器 × 5 个适配器），而非 45 个独立集成。在生产环境中（排除 `mock` 适配器），则是 **9 × 4 = 36 种有效组合**。

**添加新模型** 只需要一个新的 Compiler 类。**添加新协议** 只需要一个新的 Adapter 类。两者正交组合。

### 2. TAP IR — 类似 LLVM IR，但用于 Prompt

TAP IR **不是** 传输协议。它是一种内存数据结构，捕获用户 *想要什么*，而非 *如何请求模型*。这种解耦实现了：

- **模型无关优化**：同一个 TAP 请求可以为 GLM 和 Claude 进行不同的编译
- **工具增强上下文**：TAPRequest 包含 meta、context、instruction、constraints 和 output_format_hint
- **自强化学习数据采集**：每个 TAP 请求→响应对都可以被追踪，用于 DPO 对生成

### 3. 纵深防御

安全不是事后补充 — 它被编织到每一层中：

- **7 层权限解析**（user → config → project → system → level → AI → deny）
- **6 层命令防御**（normalization → chain split → blacklist → cross-chain → package → metacharacters）
- **2 阶段提交文件写入**（validate → write temp → atomic swap → rollback）
- **3 级沙箱降级**（Firecracker → Docker → subprocess）

### 4. 建议优先的可靠性

TerAgent 的可靠性系统默认为 **建议性而非阻塞性**：

- 预算在 70% 时发出警告，90% 时发出严重警告，仅在显式启用时才硬性停止
- 断路器在连续失败时打开，但会自动恢复（半开状态）
- 进度检测在停滞时发出警告，但不会终止循环
- 延迟监控纯粹是信息性的

## 模块依赖图

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
   │   Pipeline       │  │   Context       │  │   Orchestration      │
   │                 │  │                 │  │                      │
   │ Extractor       │  │ ContextWindow   │  │ Agent                │
   │ PromptBuilder   │  │ AutoCompactor   │  │ Orchestrator         │
   │ Checklist       │  │ Microcompactor  │  │ Handoff              │
   │ Retry           │  │ CodeIndexer*    │  │ SharedState          │
   │ TAPTracer       │  │ ReferenceGraph* │  │ CancellationToken   │
   └─────────────────┘  │ VectorIndexer*  │  │ Guardrail / Approval │
                        └─────────────────┘  │ 5 Patterns           │
                        (* optional deps)    └──────────────────────┘

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

## 已注册的编译器与适配器

### 编译器（9 个已注册）

| 名称 | 类 | 优化策略 |
|------|-----|----------|
| `default` | `TAPCompiler` | 标准聊天消息 |
| `glm` | `TAPCompiler` | 近因效应（关键指令置末） |
| `glm_5` | `GLM5Compiler` | 近因效应 + 长时任务 + 自我评估 |
| `glm_52` | `GLM52Compiler` | 1M 上下文 + 双思考模式（High/Max）+ PreservedThinking + 5V-Turbo 协调 |
| `glm_5v_turbo` | `GLM5VTurboCompiler` | GLM-5V-Turbo 模型的视觉分析 |
| `anthropic` | `TAPCompiler` | XML 标签结构化 + Mode B（system/user 分离） |
| `deepseek` | `TAPCompiler` | 极简编译 |
| `deepseek_v4` | `DeepSeekV4Compiler` | 缓存感知布局 + 思考模式 + 1M 上下文优化（flash/pro 通过 `variant` 参数控制） |
| `minimax_m3` | `MiniMaxM3Compiler` | MSA 全文注入 + 多模态 + 桌面上下文 |

> **注意：** `deepseek_v4_flash` 和 `deepseek_v4_pro` **不是**独立的编译器 —— 它们是 `DeepSeekV4Compiler` 的变体，通过 `variant` 参数控制。

### 适配器（5 个已注册）

| 名称 | 类 | 协议 |
|------|-----|------|
| `openai_compatible` | `OpenAICompatibleAdapter` | OpenAI `/chat/completions` + SSE |
| `anthropic_native` | `AnthropicNativeAdapter` | Anthropic `/messages` + Anthropic SSE |
| `glm_native` | `GLMNativeAdapter` | 智谱 AI 原生 API |
| `minimax_native` | `MiniMaxNativeAdapter` | MiniMax 原生 API + 速率限制追踪 |
| `mock` | `MockAdapter` | 无 HTTP 调用（测试用） |

## 数据流：一次完整的 TAP 调用

### 简单 TAP 调用（execute_tap）

```python
provider = create_provider(compiler="glm_5", adapter="openai_compatible", ...)
response = await provider.execute_tap(TAPRequest(instruction="..."))
```

1. 创建 `TAPRequest`，包含 meta、context、instruction、constraints、output_format_hint
2. `ModelProvider.execute_tap()` 调用 `compiler.compile(request)` → `CompiledPrompt`
3. CompiledPrompt 根据 adapter 的 `required_mode` 进行验证
4. `adapter.send(compiled, model)` → HTTP 请求 → `TAPResponse`
5. 如果附加了 tracer，请求和响应会被自动记录

### Agent Loop TAP 调用（包含所有横切关注点）

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
│ Orchestrator      │  (if multi-agent configured)
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

## CompiledPrompt：两种模式

编译器产生两种互斥的提示格式之一：

### Mode A：消息列表（OpenAI / GLM / DeepSeek）

```python
CompiledPrompt(
    messages=[
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
    ],
    tools=[...],
    extra={"cache_aware": True},  # 编译器特定参数，传递给适配器
)
```

### Mode B：System + User 分离（Anthropic 原生）

```python
CompiledPrompt(
    system_prompt="...",
    user_message="...",
    tools=[...],
    extra={},  # 额外字典，用于编译器特定参数
)
```

`CompiledPrompt.mode` 属性返回 `"messages"`、`"system_user"` 或 `"empty"`。Adapter 通过 `required_mode` 验证兼容性。`extra` 字典携带编译器特定参数（如 `cache_aware`、`variant`、`minimax_video_mode`），供适配器定制行为。

## 编译器注册模式

Compiler 和 Adapter 使用类级别注册模式：

```python
# 注册（在导入时发生）
@classmethod
def register(cls, name: str, compiler_cls: type[TAPCompiler]) -> None:
    cls._get_registry()[name] = compiler_cls

# 创建实例
compiler = TAPCompilerRegistry.create("glm")
adapter = TAPAdapterRegistry.create("openai_compatible", base_url="...", api_key="...")
```

注册表使用 `_get_registry()` 确保子类不会共享父类的可变类变量。

## 事件系统设计

`EventBus` 遵循 **即发即弃** 设计：

- `emit()`：异步处理器通过 `create_task` 调度，同步处理器通过 `run_in_executor` 调度
- `emit_and_wait()`：等待所有处理器完成（用于关键路径）
- `emit_message()`：带元数据追踪的结构化 Message 事件
- `wait_for()`：带超时地等待特定事件
- 错误隔离：单个处理器失败不会阻塞其他处理器
- 事件历史：追踪最近 100 个基本事件（名称 + 时间戳）和 200 个结构化事件（含完整数据），用于调试

## 线程安全

| 组件 | 线程安全机制 |
|------|------------|
| `CostTracker` | 所有操作使用 `threading.Lock` |
| `TAPTracer` | 记录写入使用 `threading.Lock` |
| `EnhancedPermissionManager` | `_rules_dirty` 标志 + 缓存排序规则 |
| `CircuitBreakerManager` | 单线程异步（无需锁） |
| `EventBus` | 单线程异步（无需锁） |
| `AgentLoop` | 单线程异步（无需锁） |

## 资源生命周期

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

Orchestrator
    ├── Agent 实例（独立的 provider、工具集、handoff）
    ├── OrchestrationPattern（Sequential / Swarm / Parallel / Conditional / Loop）
    ├── SharedState（作用域：session / agent / global）
    ├── RunContext + UsageTracker
    └── CancellationToken（线程安全协作式取消）
```
