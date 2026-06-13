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
)
```

**方法：**
- `estimate_prompt_tokens() -> int` — 粗略的 Token 数估算

### TAPResponse

```python
from teragent import TAPResponse

response = TAPResponse(
    raw_text="...",  # Model's raw text output (None = API error)
    usage={"prompt_tokens": 100, "completion_tokens": 200},  # Token usage
    tool_calls=[...],  # Structured tool calls from API
    finish_reason="stop",  # Why the model stopped
)
```

**属性：**
- `prompt_tokens -> int`
- `completion_tokens -> int`
- `total_tokens -> int`

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
)

# Mode B: System + User separation (Anthropic native)
prompt = CompiledPrompt(
    system_prompt="...",
    user_message="...",
    tools=[...],
)
```

**属性：**
- `mode -> str` — 返回 `"messages"`、`"system_user"` 或 `"empty"`

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
