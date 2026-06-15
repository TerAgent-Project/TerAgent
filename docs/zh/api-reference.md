# API 参考

TerAgent 库的完整模块参考。

## 核心模块（`teragent.core`）

### TAPRequest

```python
from teragent import TAPRequest

request = TAPRequest(
    meta={"task_id": "1.1", "intent": "code_generation"},  # Task metadata
    context={"design": "...", "plan": "...", "memory": "..."},  # Reference material
    instruction="Implement user login module",  # Core instruction
    constraints=["Python 3.10+"],  # Hard constraints
    output_format_hint="<file path='...'>complete code</file>",  # Desired format
    thinking_mode="high",               # 扩展：思考模式（auto/deep/quick/high/max）
    multimodal_context=[...],           # 扩展：MultimodalContent 列表（图像/视频）
    long_horizon=None,                  # 扩展：LongHorizonConfig，用于长时间运行任务
    cache_preference=None,              # 扩展：缓存偏好提示，用于缓存感知编译器
)
```

**方法：**
- `estimate_prompt_tokens() -> int` — 粗略的 Token 数估算

**扩展字段：**
- `thinking_mode: str | None` — 控制推理深度。取值：`"auto"`、`"quick"`、`"deep"`（DeepSeek V4）；`"high"`、`"max"`（GLM-5.2）。请求级别的覆盖，优先于驱动级默认值。
- `multimodal_context: list[MultimodalContent] | None` — 多模态内容项列表（图像、视频），用于支持视觉的模型（M3、GLM-5.2 + 5V-Turbo）。每项包含 `type`（`"image_url"` 或 `"video_url"`）及对应的 URL/数据。
- `long_horizon: LongHorizonConfig | None` — 长时自主任务配置（GLM-5/5.2）。包含 `max_duration_hours`、`checkpoint_interval_minutes`、`evaluation_interval_steps` 等。
- `cache_preference: dict | None` — 缓存感知编译器提示（如 DeepSeek V4）。控制提示布局优化以提高缓存命中率。

### TAPResponse

```python
from teragent import TAPResponse

response = TAPResponse(
    raw_text="...",  # Model's raw text output (None = API error)
    usage={"prompt_tokens": 100, "completion_tokens": 200},  # Token usage
    tool_calls=[...],  # Structured tool calls from API
    finish_reason="stop",  # Why the model stopped
    cache_hit_tokens=3000,        # 扩展：缓存命中的 Token 数（缓存感知模型）
    thinking_content="...",       # 扩展：推理追踪内容（思考模式模型）
    long_horizon_status=None,     # 扩展：长时任务状态信息
)
```

**属性：**
- `prompt_tokens -> int`
- `completion_tokens -> int`
- `total_tokens -> int`

**扩展字段：**
- `cache_hit_tokens: int | None` — 从缓存提供的 Token 数（DeepSeek V4 开启 `cache_aware=true` 时）。用于成本追踪 —— 缓存命中显著更便宜。
- `thinking_content: str | None` — 模型内部推理追踪，在思考模式激活时可用（DeepSeek V4 `deep` 模式、GLM-5.2 `high`/`max` 模式）。
- `long_horizon_status: dict | None` — 长时任务步骤的状态信息，包括检查点信息、子目标进度和策略切换通知。

### CompiledPrompt

```python
from teragent import CompiledPrompt

# Mode A: Messages list (OpenAI / GLM / DeepSeek)
prompt = CompiledPrompt(
    messages=[
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
    ],
    tools=[...],
    extra={"cache_aware": True},  # 编译器特定参数
)

# Mode B: System + User separation (Anthropic native)
prompt = CompiledPrompt(
    system_prompt="...",
    user_message="...",
    tools=[...],
    extra={},  # 额外字典，用于编译器特定参数
)
```

**属性：**
- `mode -> str` — 返回 `"messages"`、`"system_user"` 或 `"empty"`

**`extra` 字典字段：**
`extra` 字典携带编译器特定参数，供适配器定制行为：

| 键 | 编译器 | 适配器 | 描述 |
|-----|----------|---------|-------------|
| `cache_aware` | `deepseek_v4` | `openai_compatible` | 是否冻结工具定义以优化缓存命中率 |
| `variant` | `deepseek_v4` | `openai_compatible` | `"flash"` 或 `"pro"` — 控制提示策略 |
| `minimax_video_mode` | `minimax_m3` | `minimax_native` | `"understand"` 或 `"summarize"` — 视频处理模式 |
| `minimax_frame_sampling` | `minimax_m3` | `minimax_native` | `"auto"`、`"uniform"`、`"keyframe"` 或 `"dense"` |
| `thinking_mode` | `glm_52` | `openai_compatible` | `"high"` 或 `"max"` — 双思考模式 |
| `preserved_thinking` | `glm_52` | `openai_compatible` | 是否注入保留的推理追踪 |
| `vision_coordination` | `glm_52` | `openai_compatible` | 是否激活 5V-Turbo 视觉协调 |

### ModelProvider

```python
from teragent import ModelProvider, create_provider

# Create via factory function
provider = create_provider(
    compiler="glm_5",
    adapter="openai_compatible",
    model="glm-5",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
)
```

**方法：**
- `execute_tap(request) -> TAPResponse` — 执行 TAP 请求（编译 → 发送）
- `stream_tap(request) -> AsyncIterator[str]` — 流式执行 TAP 请求（编译 → 流式）
- `chat(messages, tools=None) -> dict` — 简单聊天（绕过 Compiler）
- `execute_tap_with_retry(request, max_retries=2) -> TAPResponse` — 带重试和熔断器的 TAP
- `chat_with_fallback(messages, tools=None) -> dict` — 带降级提供者的聊天
- `set_tracer(tracer)` — 附加 TAPTracer 用于自动追踪
- `set_fallback(fallback_provider)` — 设置降级提供者
- `get_cost_summary() -> dict` — 按提供者获取聚合成本摘要
- `close()` — 关闭适配器连接

**属性：**
- `tracer -> TAPTracer | None`
- `fallback_provider -> ModelProvider | None`
- `has_fallback -> bool`
- `cost_records -> list[TAPCostRecord]`
- `capabilities -> dict`

### TAPCompiler（ABC）

```python
from teragent import TAPCompiler

class MyCompiler(TAPCompiler):
    def compile(self, request: TAPRequest) -> CompiledPrompt:
        # Transform TAPRequest into model-specific CompiledPrompt
        ...
```

**方法：**
- `compile(request) -> CompiledPrompt` — **抽象方法**。编译 TAP 请求
- `get_system_prompt(intent) -> str` — 获取特定意图的系统提示

### TAPAdapter（ABC）

```python
from teragent import TAPAdapter

class MyAdapter(TAPAdapter):
    async def send(self, compiled: CompiledPrompt, model: str) -> TAPResponse:
        # Send compiled prompt to model API
        ...

    async def stream(self, compiled: CompiledPrompt, model: str) -> AsyncIterator[str]:
        # Stream compiled prompt to model API
        ...
```

**属性：**
- `capabilities -> dict` — 特性检测（streaming、tool_calling 等）
- `required_mode -> str` — 期望的 CompiledPrompt 模式（"any"、"messages"、"system_user"）

## 安全模块（`teragent.security`）

### EnhancedPermissionManager

```python
from teragent.security import EnhancedPermissionManager, PermissionRule, PermissionEffect

epm = EnhancedPermissionManager(
    default_level=PermissionLevel.PLAN,
    default_effect=PermissionEffect.DENY,
    ai_classifier=None,
)
```

**方法：**
- `add_rule(rule)` — 添加权限规则
- `add_rules(rules)` — 批量添加规则
- `remove_rules_by_source(source) -> int` — 按来源移除规则
- `clear_rules()` — 清除所有规则
- `check(tool_name, path="") -> (bool, str)` — 同步权限检查（第 1-5、7 层）
- `acheck(tool_name, path="", context="") -> (bool, str)` — 异步检查（第 1-7 层，包含 AI 分类器）
- `check_tool_params(tool_name, params) -> (bool, str)` — 从工具参数检查（自动提取路径）
- `acheck_tool_params(tool_name, params, context="") -> (bool, str)` — 异步版本
- `elevate(new_level)` — 提升权限级别
- `deactivate()` — 重置为默认级别
- `set_level(level)` — 直接设置级别
- `load_from_config(config)` — 从配置字典加载
- `default_rules() -> list[PermissionRule]` — 获取内置默认规则
- `get_status_report() -> dict` — 用于调试的状态报告
- `get_rules_summary() -> list[dict]` — 按优先级列出所有规则
- `reset()` — 重置所有状态

### Sandbox

```python
from teragent.security import execute_in_sandbox, check_command_safety

# Check command safety (no execution)
is_safe, reason = check_command_safety("rm -rf /")
# → (False, "命令匹配危险模式: ...")

# Execute in sandbox
exit_code, output = await execute_in_sandbox(
    cmd="python script.py",
    workdir="/project",
    level=0,  # 0=subprocess, 1=Docker, 2=Firecracker
    timeout=60,
)
```

### File Writer

```python
from teragent.security import write_files_safely, atomic_write_file

# Write multiple files atomically (2PC)
success, results = write_files_safely(
    files=[
        {"path": "/project/src/main.py", "content": "..."},
        {"path": "/project/src/utils.py", "content": "..."},
    ],
    workspace_root="/project",
)

# Write single file atomically
success = atomic_write_file("/project/src/main.py", "content")
```

## 可靠性模块（`teragent.reliability`）

### CircuitBreakerManager

```python
from teragent.reliability import CircuitBreakerManager

manager = CircuitBreakerManager(bus=event_bus)

# Record a model call
result = manager.record_model_call(
    prompt_tokens=500,
    completion_tokens=200,
    stage="plan",
    latency_ms=3500,
)

# Record success/failure
manager.record_success()
manager.record_failure("API timeout")

# Record agent step progress
manager.record_agent_step("read_file", had_effect=True)

# Check budget before call
result = manager.check_before_call(estimated_prompt_tokens=1000)

# Get status
status = manager.get_status()
```

### StepBudget

```python
from teragent.reliability import StepBudget

budget = StepBudget(max_steps=50)

if budget.consume():  # Returns True if budget remaining
    # Do work
    pass

# Properties
budget.current_steps  # Steps consumed
budget.remaining      # Steps remaining
budget.exhausted      # Whether budget is exhausted
```

### RecoveryManager

```python
from teragent.reliability import RecoveryManager, RecoveryType

manager = RecoveryManager()

# Check if recovery is needed
if manager.should_continue_after_truncation(finish_reason, attempt):
    manager.record_recovery(RecoveryType.LENGTH)

# Check error types
manager.is_context_overflow(error)
manager.is_retryable(error)
manager.should_retry_streaming(attempt)

# Get stats
stats = manager.get_stats()
```

## 上下文模块（`teragent.context`）

### ContextWindow

```python
from teragent.context import ContextWindow

window = ContextWindow(model_token_limit=128_000)

# Estimate tokens
tokens = window.estimate(messages)

# Check if compaction needed
if window.should_compact(messages):
    # Trigger compaction
    pass

# Properties
window.available_budget
window.utilization
window.last_estimated_tokens
```

### AutoCompactor

```python
from teragent.context import AutoCompactor

compactor = AutoCompactor(
    context_window=window,
    model=provider,
    retain_count=8,  # Keep last 8 messages
    max_compacts=5,   # Max 5 compactions per session
)

# Check and compact if needed
compacted = await compactor.maybe_compact(messages, system_prompt)

# Get stats
stats = compactor.get_stats()
```

## 管道模块（`teragent.pipeline`）

### Extractor

```python
from teragent import extract_files_from_response

files = extract_files_from_response(response_text, task_id="1.1")
# → [{"path": "src/main.py", "content": "..."}, ...]
```

### PromptBuilder

```python
from teragent import build_prompt, validate_prompt_tokens

# Build from template
messages = build_prompt(
    system_template="You are {role}. Task: {task}",
    context={"role": "engineer", "task": "implement login"},
)

# Validate token budget
errors = validate_prompt_tokens(messages, max_tokens=4000)
```

### Checklist

```python
from teragent import run_deterministic_checks, TaskInfo

task_list = [TaskInfo(id="1.1", title="Login module", status="completed")]
report, data = run_deterministic_checks("/project", task_list)
```

### Retry

```python
from teragent import retry_with_backoff

async def _call():
    return await provider.chat(messages=[...])

result = await retry_with_backoff(
    fn=_call,
    max_retries=3,
    validate=lambda r: [] if r else ["empty response"],
)
```

### TAPTracer

```python
from teragent import TAPTracer

tracer = TAPTracer(trace_dir="/project/.agent/traces")

# Auto-tracing via ModelProvider
provider.set_tracer(tracer)

# Manual tracing
trace_id = await tracer.record_request(tap_request)
await tracer.record_response(tap_response, task_id="1.1", trace_id=trace_id)
await tracer.record_checklist("1.1", checklist_data)

# Export
pairs = tracer.export_dpo_pairs()
tracer.export_dpo_pairs_jsonl()
traces = tracer.export_traces()
stats = tracer.get_trace_stats()
```

## 流式模块（`teragent.streaming`）

### StreamingToolExecutor

```python
from teragent.streaming import StreamingToolExecutor

executor = StreamingToolExecutor(
    tool_registry=registry,
    permission_level=0,
    max_concurrent=10,
)

# Execute with streaming
results, streaming_result, stats = await executor.execute_streaming(
    stream=model.adapter.stream(compiled, model.model),
    on_text_delta=lambda text: print(text, end=""),
    on_tool_complete=lambda tc, result: print(f"Tool {tc['name']}: {result.success}"),
)

# Batch fallback
results, stats = await executor.execute_batch_fallback(tool_calls)

# Check streaming capability
can_stream = executor.can_stream_with_tools(model)
```

## 协调模块（`teragent.coordination`）

### SubAgentManager

```python
from teragent.coordination import SubAgentManager, AgentMode

manager = SubAgentManager(event_bus, model, tool_registry, message_bus)

# Sync: block until done
result = await manager.spawn("Analyze code quality", mode=AgentMode.SYNC)

# Async: run in background
agent_id = await manager.spawn("Background refactoring", mode=AgentMode.ASYNC)

# FORK: shared prefix (KV cache optimization)
result = await manager.spawn("Quick query", mode=AgentMode.FORK)

# Management
status = manager.get_status(agent_id)
agents = manager.list_active_agents()
await manager.stop(agent_id)
await manager.stop_all()
```

## 工具模块（`teragent.tools`）

### BaseTool

```python
from teragent.tools import BaseTool, ToolResult
from teragent.core.types import ToolSafety

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"
    parameters_schema = {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input text"},
        },
        "required": ["input"],
    }
    _safety = ToolSafety.READ_ONLY
    _concurrency_safe = True

    async def execute(self, params, progress_callback=None):
        return ToolResult(
            success=True,
            data={"result": params["input"].upper()},
            safety=ToolSafety.READ_ONLY,
        )
```

### ToolRegistry

```python
from teragent.tools import ToolRegistry

registry = ToolRegistry()
registry.register(MyTool())

# Query
tool = registry.get("my_tool")
names = registry.list_tool_names()
summary = registry.get_summary()
```

### ToolOrchestrator

```python
from teragent.tools import ToolOrchestrator

orchestrator = ToolOrchestrator(
    tool_registry=registry,
    permission_level=0,
    hook_manager=hook_mgr,
    enhanced_perm_manager=epm,
)

# Execute batch
results = await orchestrator.execute_batch(tool_calls)

# Execute single
result = await orchestrator._execute_single(tool_call_dict)
```

## 意图模块（`teragent.intent`）

### IntentClassifier

```python
from teragent.intent import IntentClassifier, IntentType

classifier = IntentClassifier(provider)

intent = await classifier.classify("Build me a web app")
# → IntentType.CREATE_PROJECT

intent = await classifier.classify("What does this code do?")
# → IntentType.CHAT

intent = await classifier.classify("Fix the bug in main.py")
# → IntentType.DEBUG
```

### ConfirmationGate

```python
from teragent.intent import ConfirmationGate

gate = ConfirmationGate()

confirmed = await gate.confirm_create_project("Build a new web app")
# → True/False (asks user for approval)
```

## Hook 模块（`teragent.hooks`）

### HookManager

```python
from teragent.hooks import HookManager, HookDecision

manager = HookManager()

# Register a hook
manager.register_hook("pre_execute", my_hook)

# Run hooks
decision = await manager.run_hooks("pre_execute", context)
# → HookDecision.ALLOW / DENY / MODIFY
```

### 内置 Hook

- **AuditHook**：记录所有工具执行，用于审计追踪
- **DangerousCommandHook**：使用 6 层防御阻止危险 Shell 命令

## 会话模块（`teragent.session`）

### SessionPersistence

```python
from teragent.session import SessionPersistence

persistence = SessionPersistence(db_path=".agent/sessions.db")

# Create session
session_id = persistence.create(title="My Session", intent="chat")

# Save message
persistence.save_message(session_id, message)

# Restore session
messages = persistence.restore(session_id)

# List sessions
sessions = persistence.list_sessions()
```

## 事件总线（`teragent.event_bus`）

### EventBus

```python
from teragent import EventBus

bus = EventBus()

# Subscribe
bus.on("agent_done", lambda **kw: print("Done!"))

# Subscribe once
bus.once("agent_done", handler)

# Emit (fire-and-forget)
await bus.emit("agent_done", total_steps=10)

# Emit and wait
await bus.emit_and_wait("agent_done", total_steps=10)

# Wait for event
args, kwargs = await bus.wait_for("agent_done", timeout=30)

# Query
names = bus.get_event_names()
history = bus.get_event_history(limit=50)
```

## 路由模块（`teragent.router`）

### RoutingReason

```python
from teragent.router import RoutingReason

# 枚举值，指示模型选择的原因
RoutingReason.INTENT                    # 默认基于意图的路由
RoutingReason.MULTIMODAL_OVERRIDE       # 包含多模态内容 → M3
RoutingReason.DESKTOP_OVERRIDE          # 包含桌面上下文 → M3
RoutingReason.VIDEO_OVERRIDE            # 包含视频内容 → M3
RoutingReason.CONTEXT_LENGTH_OVERRIDE   # 上下文 >200K → V4/M3
RoutingReason.LONG_HORIZON_OVERRIDE     # 长时任务 → GLM-5
RoutingReason.COST_OPTIMIZATION         # 预算约束 → 更便宜的模型
RoutingReason.DEGRADATION               # 主模型不可用 → 降级
RoutingReason.PIPELINE_PROFILE          # 显式管道配置文件分配
RoutingReason.EXPLICIT                  # 用户显式指定模型
```

### RoutingDecision

```python
from teragent.router import RoutingDecision

decision = RoutingDecision(
    selected_driver="openai_compatible.deepseek_v4_pro",
    selected_compiler="deepseek_v4",
    reason=RoutingReason.INTENT,
    intent="design",
)
```

**属性：**
- `selected_driver: str` — 完整驱动名称（如 `"openai_compatible.deepseek_v4_pro"`）
- `selected_compiler: str` — 编译器名称（如 `"deepseek_v4"`）
- `reason: RoutingReason` — 主要路由原因
- `intent: str` — 请求的意图类型
- `trace: list[tuple[str, str, str]]` — 有序的 (原因, 候选, 接受/拒绝) 元组列表
- `timestamp: float` — 决策时间戳（纪元秒）
- `estimated_cost: float` — 此请求的估算成本
- `context_tokens: int` — 估算的上下文 Token 数

**方法：**
- `add_trace(reason, candidate, result)` — 追加调试追踪条目

### RoutingTable

```python
from teragent.router import RoutingTable

table = RoutingTable(
    multimodal_driver="openai_compatible.minimax_m3",
    desktop_driver="openai_compatible.minimax_m3",
    long_horizon_driver="openai_compatible.glm_5",
)
```

**关键属性：**
- `intent_routing: dict[str, str]` — 意图 → 默认驱动名称映射
- `multimodal_driver: str` — 多模态内容驱动（默认：M3）
- `desktop_driver: str` — 桌面上下文驱动（默认：M3）
- `long_horizon_driver: str` — 长时任务驱动（默认：GLM-5）
- `long_context_candidates: list[str]` — 支持 >200K 上下文的模型
- `cost_fallback_order: list[str]` — 从最便宜到最贵的模型顺序
- `degradation_map: dict[str, str]` — 主模型 → 降级映射
- `model_pricing: dict[str, dict[str, float]]` — 每模型每百万 Token CNY 定价
- `max_context_per_model: dict[str, int]` — 每模型最大上下文 Token 数
- `compiler_map: dict[str, str]` — 驱动名称 → 编译器名称映射

**方法：**
- `resolve_compiler(driver_name) -> str` — 从驱动名称解析编译器名称
- `get_intent_driver(intent) -> str` — 获取意图类型的默认驱动
- `get_pricing(driver_name) -> dict[str, float]` — 获取模型的定价字典
- `from_dict(data) -> RoutingTable` — 从配置字典创建 RoutingTable

### ModelRouter

```python
from teragent.router import ModelRouter, RoutingTable, RoutingDecision

router = ModelRouter(
    available_providers={"openai_compatible.glm_5": glm_provider, ...},
    routing_table=RoutingTable(),
)

decision = router.route(tap_request)
provider = router.get_provider(decision.selected_driver)
```

**方法：**
- `route(request) -> RoutingDecision` — 通过 6 步决策流程路由 TAP 请求
- `route_for_stage(stage, request) -> RoutingDecision` — 使用活动管道配置文件按阶段路由
- `get_decision_log() -> list[RoutingDecision]` — 获取所有路由决策日志
- `get_provider(driver_name) -> ModelProvider | None` — 按驱动名称获取提供者
- `set_monthly_budget(limit_cny, warning_threshold, auto_downgrade)` — 配置月度预算

### PipelineProfile

```python
from teragent.router import PipelineProfile

profile = PipelineProfile(
    name="default",
    description="Default pipeline configuration",
    design_driver="openai_compatible.deepseek_v4_pro",
    plan_driver="openai_compatible.glm_5",
    execute_driver="openai_compatible.glm_5",
    review_driver="openai_compatible.deepseek_v4_pro",
)
```

**方法：**
- `get_driver_for_stage(stage) -> str` — 获取管道阶段的驱动名称
- `from_dict(name, data) -> PipelineProfile` — 从配置字典创建

### PipelineManager

```python
from teragent.router import PipelineManager, PipelineProfile

pm = PipelineManager()

# 注册配置文件
pm.register_profile(PipelineProfile(name="budget", ...))

# 运行时切换配置文件
pm.set_active_profile("budget")

# 获取阶段的驱动
driver = pm.get_driver("execute")
```

**方法：**
- `register_profile(profile) -> None` — 注册管道配置文件
- `set_active_profile(name) -> bool` — 切换到指定配置文件（找到则返回 True）
- `get_driver(stage) -> str` — 从活动配置文件获取阶段的驱动名称
- `list_profiles() -> list[str]` — 列出所有已注册的配置文件名称
- `get_profile(name) -> PipelineProfile | None` — 按名称获取配置文件
- `from_config(config, routing_table) -> PipelineManager` — 从 TOML 配置字典创建

**属性：**
- `active_profile_name -> str` — 当前活动配置文件名称
- `active_profile -> PipelineProfile` — 当前活动的 PipelineProfile

---

## 长时任务模块（`teragent.long_horizon`）

### SubGoal

```python
from teragent.long_horizon import SubGoal

goal = SubGoal(
    id="sg_1",
    description="Design database schema",
    completion_criteria="All tables defined with proper constraints",
    estimated_steps=10,
    dependencies=["sg_0"],  # 依赖于 sg_0
    status="pending",        # pending | in_progress | completed | failed
)
```

**属性：**
- `id: str` — 唯一标识符
- `description: str` — 子目标描述
- `completion_criteria: str` — 可衡量的完成标准
- `estimated_steps: int` — 预估步骤数
- `dependencies: list[str]` — 依赖的子目标 ID（DAG 拓扑）
- `status: str` — 当前状态：`pending` | `in_progress` | `completed` | `failed`

### PhaseResult

```python
from teragent.long_horizon import PhaseResult

result = PhaseResult(
    sub_goal_id="sg_1",
    success=True,
    result_text="Database schema designed with 5 tables...",
    steps_taken=8,
    files_created=["src/models/user.py", "src/models/role.py"],
    files_modified=["src/db/init.py"],
    errors=[],
)
```

**属性：**
- `sub_goal_id: str` — 对应的子目标 ID
- `success: bool` — 阶段是否成功
- `result_text: str` — 模型输出文本
- `steps_taken: int` — 此阶段消耗的步骤数
- `files_created: list[str]` — 创建的文件
- `files_modified: list[str]` — 修改的文件
- `errors: list[str]` — 错误消息

### LongHorizonResult

```python
from teragent.long_horizon import LongHorizonResult

result = LongHorizonResult(
    task_id="task_001",
    goal="Implement user management system",
    success=True,
    total_steps=120,
    total_elapsed_minutes=95.5,
    completed_sub_goals=5,
    total_sub_goals=5,
    strategy_switches=1,
    phase_results=[...],
    final_summary="All sub-goals completed successfully",
    checkpoints_saved=6,
)
```

**属性：**
- `task_id: str` — 唯一任务标识符
- `goal: str` — 原始目标描述
- `success: bool` — 整体是否成功
- `total_steps: int` — 消耗的总步骤数
- `total_elapsed_minutes: float` — 总耗时（分钟）
- `completed_sub_goals: int` — 已完成子目标数
- `total_sub_goals: int` — 总子目标数
- `strategy_switches: int` — 策略切换次数
- `phase_results: list[PhaseResult]` — 详细的阶段结果
- `final_summary: str` — 最终摘要文本
- `checkpoints_saved: int` — 保存的检查点数

### LongHorizonTaskManager

```python
from teragent.long_horizon import LongHorizonTaskManager
from teragent.core.tap import LongHorizonConfig

manager = LongHorizonTaskManager(
    model=provider,
    tool_registry=registry,
    event_bus=bus,
    config=LongHorizonConfig(
        max_duration_hours=8.0,
        checkpoint_interval_minutes=15.0,
        evaluation_interval_steps=10,
    ),
)

# 启动长时任务
result = await manager.run("Implement user management system")

# 从检查点恢复
result = await manager.resume_from_checkpoint("task_001")

# 获取状态
status = manager.get_status("task_001")
```

**方法：**
- `run(goal, context=None) -> LongHorizonResult` — 启动长时自主任务
- `resume_from_checkpoint(task_id) -> LongHorizonResult` — 从检查点恢复任务
- `cancel(task_id) -> bool` — 取消运行中的任务
- `get_status(task_id) -> dict` — 获取任务状态
- `list_active_tasks() -> list[str]` — 列出活动任务 ID
- `save_checkpoint(task_id) -> str` — 手动保存检查点

### CheckpointStore

```python
from teragent.long_horizon import CheckpointStore

store = CheckpointStore(base_dir=".teragent/checkpoints")

# 保存检查点
checkpoint_id = store.save(task_id="task_001", state={...})

# 加载检查点
state = store.load(checkpoint_id)

# 列出检查点
checkpoints = store.list_for_task("task_001")

# 清理旧检查点
store.cleanup(task_id="task_001", keep_last=5)
```

### SelfEvaluator

```python
from teragent.long_horizon import SelfEvaluator

evaluator = SelfEvaluator(model=provider)

# 评估当前进度
evaluation = await evaluator.evaluate(
    goal="Implement user management system",
    sub_goals=[...],
    completed_results=[...],
    current_state={...},
)
# → SelfEvaluation(score=0.8, should_continue=True, suggested_strategy="refine", ...)
```

### StrategySwitcher

```python
from teragent.long_horizon import StrategySwitcher

switcher = StrategySwitcher()

# 检查是否需要切换策略
should_switch, new_strategy = switcher.check(
    current_strategy="direct",
    evaluation=evaluation,
    consecutive_similar=3,
)

# 记录策略切换
switcher.record_switch("direct", "decompose", reason="stagnation detected")
```

---

## 基准测试模块（`teragent.benchmark`）

### BenchmarkRunner

```python
from teragent.benchmark import BenchmarkRunner

runner = BenchmarkRunner(
    model=provider,
    tool_registry=registry,
    output_dir=".teragent/benchmarks",
)

# 运行基准测试
results = await runner.run_suite(
    suite_name="code_generation",
    task_configs=[...],
)

# 生成报告
report = runner.generate_report(results)
runner.save_report(report, format="markdown")
```

**方法：**
- `run_suite(suite_name, task_configs) -> list[BenchmarkResult]` — 运行基准测试套件
- `run_single(task_config) -> BenchmarkResult` — 运行单个基准测试
- `generate_report(results) -> BenchmarkReport` — 生成报告
- `save_report(report, format) -> str` — 保存报告到文件
- `compare_reports(report_a, report_b) -> ComparisonReport` — 比较两份报告
