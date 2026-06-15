# API Reference

Complete module reference for the TerAgent library.

## Core Module (`teragent.core`)

### TAPRequest

```python
from teragent import TAPRequest

request = TAPRequest(
    meta={"task_id": "1.1", "intent": "code_generation"},  # Task metadata
    context={"design": "...", "plan": "...", "memory": "..."},  # Reference material
    instruction="Implement user login module",  # Core instruction
    constraints=["Python 3.10+"],  # Hard constraints
    output_format_hint="<file path='...'>complete code</file>",  # Desired format
    thinking_mode="high",               # Extended: thinking mode (auto/deep/quick/high/max)
    multimodal_context=[...],           # Extended: list of MultimodalContent (images/video)
    long_horizon=None,                  # Extended: LongHorizonConfig for long-running tasks
    cache_preference=None,              # Extended: cache preference hints for cache-aware compilers
)
```

**Methods:**
- `estimate_prompt_tokens() -> int` — Rough token count estimation

**Extended Fields:**
- `thinking_mode: Optional[Literal["deep", "quick", "auto"]] — Controls reasoning depth. Per-request override of driver-level default. Additional values `"high"`, `"max"` supported for GLM-5.2.
- `multimodal_context: Optional[list[MultimodalContent]]` — List of multimodal content items (images, video) for models that support vision (M3, GLM-5.2 + 5V-Turbo). Each item has `type` (`"image_url"` or `"video_url"`) and corresponding URL/data.
- `long_horizon: Optional[LongHorizonConfig]` — Configuration for long-horizon autonomous tasks (GLM-5/5.2). Includes `max_duration_hours`, `checkpoint_interval_minutes`, `evaluation_interval_steps`, etc.
- `cache_preference: Optional[Literal["auto", "aggressive", "none"]] — Cache preference hint

### TAPResponse

```python
from teragent import TAPResponse

response = TAPResponse(
    raw_text="...",  # Model's raw text output (None = API error)
    usage={"prompt_tokens": 100, "completion_tokens": 200},  # Token usage
    tool_calls=[...],  # Structured tool calls from API
    finish_reason="stop",  # Why the model stopped
    cache_hit_tokens=3000,        # Extended: tokens served from cache (cache-aware models)
    thinking_content="...",       # Extended: reasoning trace content (thinking mode models)
    long_horizon_status=None,     # Extended: status info for long-horizon tasks
)
```

**Properties:**
- `prompt_tokens -> int`
- `completion_tokens -> int`
- `total_tokens -> int`

**Extended Fields:**
- `cache_hit_tokens: int | None` — Number of tokens served from cache (DeepSeek V4 with `cache_aware=true`). Useful for cost tracking — cache hits are significantly cheaper.
- `thinking_content: str | None` — The model's internal reasoning trace, available when thinking mode is active (DeepSeek V4 `deep` mode, GLM-5.2 `high`/`max` mode).
- `long_horizon_status: dict | None` — Status information for long-horizon task steps, including checkpoint info, sub-goal progress, and strategy switch notifications.

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

**Properties:**
- `mode -> str` — Returns `"messages"`, `"system_user"`, or `"empty"`

**`extra` dict field:**
The `extra` dict carries compiler-specific parameters that adapters use to customize their behavior:

| Key | Compiler | Adapter | Description |
|-----|----------|---------|-------------|
| `cache_aware` | `deepseek_v4` | `openai_compatible` | Whether to freeze tool definitions for cache hit optimization |
| `variant` | `deepseek_v4` | `openai_compatible` | `"flash"` or `"pro"` — controls prompt strategy |
| `minimax_video_mode` | `minimax_m3` | `minimax_native` | `"understand"` or `"summarize"` — video processing mode |
| `minimax_frame_sampling` | `minimax_m3` | `minimax_native` | `"auto"`, `"uniform"`, `"keyframe"`, or `"dense"` |
| `thinking_mode` | `glm_52` | `openai_compatible` | `"high"` or `"max"` — dual thinking mode |
| `preserved_thinking` | `glm_52` | `openai_compatible` | Whether to inject preserved reasoning traces |
| `vision_coordination` | `glm_52` | `openai_compatible` | Whether 5V-Turbo vision coordination is active |

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

**Methods:**
- `execute_tap(request) -> TAPResponse` — Execute a TAP request (compile → send)
- `stream_tap(request) -> AsyncIterator[str]` — Stream a TAP request (compile → stream)
- `chat(messages, tools=None) -> dict` — Simple chat (bypasses Compiler)
- `execute_tap_with_retry(request, max_retries=2) -> TAPResponse` — TAP with retry + circuit breaker
- `chat_with_fallback(messages, tools=None) -> dict` — Chat with fallback provider
- `set_tracer(tracer)` — Attach a TAPTracer for auto-tracing
- `set_fallback(fallback_provider)` — Set fallback provider
- `get_cost_summary() -> dict` — Get aggregated cost summary by provider
- `close()` — Close adapter connections

**Properties:**
- `tracer -> TAPTracer | None`
- `fallback_provider -> ModelProvider | None`
- `has_fallback -> bool`
- `cost_records -> list[TAPCostRecord]`
- `capabilities -> dict`

**Async context manager:** `ModelProvider` supports `async with` for automatic resource cleanup:
```python
async with create_provider(...) as provider:
    response = await provider.execute_tap(request)
```

### TAPCompiler (ABC)

```python
from teragent import TAPCompiler

class MyCompiler(TAPCompiler):
    def compile(self, request: TAPRequest) -> CompiledPrompt:
        # Transform TAPRequest into model-specific CompiledPrompt
        ...
```

**Methods:**
- `compile(request) -> CompiledPrompt` — **Abstract**. Compile TAP request
- `get_system_prompt(intent) -> str` — Get intent-specific system prompt

### TAPAdapter (ABC)

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

**Properties:**
- `capabilities -> dict` — Feature detection (streaming, tool_calling, etc.)
- `required_mode -> str` — Expected CompiledPrompt mode ("any", "messages", "system_user")

## Security Module (`teragent.security`)

### EnhancedPermissionManager

```python
from teragent.security import EnhancedPermissionManager, PermissionRule, PermissionEffect

epm = EnhancedPermissionManager(
    default_level=PermissionLevel.PLAN,
    default_effect=PermissionEffect.DENY,
    ai_classifier=None,
)
```

**Methods:**
- `add_rule(rule)` — Add a permission rule
- `add_rules(rules)` — Batch add rules
- `remove_rules_by_source(source) -> int` — Remove rules by source
- `clear_rules()` — Clear all rules
- `check(tool_name, path="") -> (bool, str)` — Sync permission check (Layers 1-5, 7)
- `acheck(tool_name, path="", context="") -> (bool, str)` — Async check (Layers 1-7, includes AI classifier)
- `check_tool_params(tool_name, params) -> (bool, str)` — Check from tool params (auto-extract path)
- `acheck_tool_params(tool_name, params, context="") -> (bool, str)` — Async version
- `elevate(new_level)` — Elevate permission level
- `deactivate()` — Reset to default level
- `set_level(level)` — Directly set level
- `load_from_config(config)` — Load from config dict
- `default_rules() -> list[PermissionRule]` — Get built-in default rules
- `get_status_report() -> dict` — Status report for debugging
- `get_rules_summary() -> list[dict]` — List all rules sorted by priority
- `reset()` — Reset all state

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

## Reliability Module (`teragent.reliability`)

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

## Context Module (`teragent.context`)

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

## Pipeline Module (`teragent.pipeline`)

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

## Streaming Module (`teragent.streaming`)

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

## Coordination Module (`teragent.coordination`)

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

## Tools Module (`teragent.tools`)

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

## Intent Module (`teragent.intent`)

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

## Hooks Module (`teragent.hooks`)

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

### Built-in Hooks

- **AuditHook**: Logs all tool executions for audit trail
- **DangerousCommandHook**: Blocks dangerous shell commands using the 6-layer defense

## Session Module (`teragent.session`)

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

## Router Module (`teragent.router`)

### RoutingReason

```python
from teragent.router import RoutingReason

# Enum values indicating why a model was selected
RoutingReason.INTENT                    # Default intent-based routing
RoutingReason.MULTIMODAL_OVERRIDE       # Has multimodal content → M3
RoutingReason.DESKTOP_OVERRIDE          # Has desktop context → M3
RoutingReason.VIDEO_OVERRIDE            # Has video content → M3
RoutingReason.CONTEXT_LENGTH_OVERRIDE   # Context >200K → V4/M3
RoutingReason.LONG_HORIZON_OVERRIDE     # Long-horizon task → GLM-5
RoutingReason.COST_OPTIMIZATION         # Budget constraint → cheaper model
RoutingReason.DEGRADATION               # Primary unavailable → fallback
RoutingReason.PIPELINE_PROFILE          # Explicit pipeline profile assignment
RoutingReason.EXPLICIT                  # User explicitly specified model
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

**Properties:**
- `selected_driver: str` — Full driver name (e.g., `"openai_compatible.deepseek_v4_pro"`)
- `selected_compiler: str` — Compiler name (e.g., `"deepseek_v4"`)
- `reason: RoutingReason` — Primary routing reason
- `intent: str` — The request's intent type
- `trace: list[tuple[str, str, str]]` — Ordered list of (reason, candidate, accepted/rejected) tuples
- `timestamp: float` — Decision timestamp (epoch seconds)
- `estimated_cost: float` — Estimated cost for this request
- `context_tokens: int` — Estimated context token count

**Methods:**
- `add_trace(reason, candidate, result)` — Append a trace entry for debugging

### RoutingTable

```python
from teragent.router import RoutingTable

table = RoutingTable(
    multimodal_driver="openai_compatible.minimax_m3",
    desktop_driver="openai_compatible.minimax_m3",
    long_horizon_driver="openai_compatible.glm_5",
)
```

**Key attributes:**
- `intent_routing: dict[str, str]` — Maps intent → default driver name
- `multimodal_driver: str` — Driver for multimodal content (default: M3)
- `desktop_driver: str` — Driver for desktop context (default: M3)
- `long_horizon_driver: str` — Driver for long-horizon tasks (default: GLM-5)
- `long_context_candidates: list[str]` — Models supporting >200K context
- `cost_fallback_order: list[str]` — Cheapest-to-most-expensive model order
- `degradation_map: dict[str, str]` — Primary → fallback mapping
- `model_pricing: dict[str, dict[str, float]]` — Per-model CNY pricing per million tokens
- `max_context_per_model: dict[str, int]` — Per-model max context tokens
- `compiler_map: dict[str, str]` — Driver name → compiler name mapping

**Methods:**
- `resolve_compiler(driver_name) -> str` — Resolve compiler name from driver name
- `get_intent_driver(intent) -> str` — Get default driver for an intent type
- `get_pricing(driver_name) -> dict[str, float]` — Get pricing dict for a model
- `from_dict(data) -> RoutingTable` — Create RoutingTable from a config dict

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

**Methods:**
- `route(request) -> RoutingDecision` — Route a TAP request through the 6-step decision flow
- `route_for_stage(stage, request) -> RoutingDecision` — Route using active pipeline profile for a stage
- `get_decision_log() -> list[RoutingDecision]` — Get the log of all routing decisions
- `get_provider(driver_name) -> ModelProvider | None` — Get a provider by driver name
- `set_monthly_budget(limit_cny, warning_threshold, auto_downgrade)` — Configure monthly budget

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

**Methods:**
- `get_driver_for_stage(stage) -> str` — Get driver name for a pipeline stage
- `from_dict(name, data) -> PipelineProfile` — Create from a config dict

### PipelineManager

```python
from teragent.router import PipelineManager, PipelineProfile

pm = PipelineManager()

# Register profiles
pm.register_profile(PipelineProfile(name="budget", ...))

# Switch profiles at runtime
pm.set_active_profile("budget")

# Get driver for a stage
driver = pm.get_driver("execute")
```

**Methods:**
- `register_profile(profile) -> None` — Register a pipeline profile
- `set_active_profile(name) -> bool` — Switch to a named profile (returns True if found)
- `get_driver(stage) -> str` — Get driver name for a stage from active profile
- `list_profiles() -> list[str]` — List all registered profile names
- `get_profile(name) -> PipelineProfile | None` — Get a profile by name
- `from_config(config, routing_table) -> PipelineManager` — Create from TOML config dict

**Properties:**
- `active_profile_name -> str` — Name of the currently active profile
- `active_profile -> PipelineProfile` — The currently active PipelineProfile

---

## Long-Horizon Module (`teragent.long_horizon`)

### SubGoal

```python
from teragent.long_horizon import SubGoal

goal = SubGoal(
    id="sg_1",
    description="Design database schema",
    completion_criteria="All tables defined with proper constraints",
    estimated_steps=10,
    dependencies=["sg_0"],  # Depends on sg_0
    status="pending",        # pending | in_progress | completed | failed
)
```

**Attributes:**
- `id: str` — Unique identifier
- `description: str` — Sub-goal description
- `completion_criteria: str` — Measurable completion criteria
- `estimated_steps: int` — Estimated number of steps
- `dependencies: list[str]` — IDs of sub-goals this depends on (DAG topology)
- `status: str` — Current status: `pending` | `in_progress` | `completed` | `failed`

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

**Attributes:**
- `sub_goal_id: str` — Corresponding sub-goal ID
- `success: bool` — Whether the phase succeeded
- `result_text: str` — Model output text
- `steps_taken: int` — Steps consumed in this phase
- `files_created: list[str]` — Files created
- `files_modified: list[str]` — Files modified
- `errors: list[str]` — Error messages

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

**Attributes:**
- `task_id: str` — Unique task identifier
- `goal: str` — Original goal description
- `success: bool` — Overall success
- `total_steps: int` — Total steps consumed
- `total_elapsed_minutes: float` — Total elapsed time
- `completed_sub_goals: int` — Completed sub-goals count
- `total_sub_goals: int` — Total sub-goals count
- `strategy_switches: int` — Number of strategy switches
- `phase_results: list[PhaseResult]` — Detailed phase results
- `final_summary: str` — Final summary text
- `checkpoints_saved: int` — Number of checkpoints saved

### LongHorizonTaskManager

```python
from teragent.long_horizon import LongHorizonTaskManager
from teragent.core.tap import LongHorizonConfig

manager = LongHorizonTaskManager(
    goal="Implement a complete user management system",
    model_provider=glm_provider,
    config=LongHorizonConfig(max_duration_hours=4),
)

result = await manager.execute_long_task()
```

**Methods:**
- `decompose_goal() -> list[SubGoal]` — Break down the goal into sub-goals
- `execute_phase(sub_goal) -> PhaseResult` — Execute one sub-goal phase
- `execute_long_task() -> LongHorizonResult` — Run the full long-horizon task
- `save_checkpoint() -> str` — Save current state as checkpoint
- `evaluate_progress() -> SelfEvaluationResult` — Run self-evaluation
- `recover_from_checkpoint(checkpoint_id) -> bool` — Resume from checkpoint

### Checkpoint & CheckpointStore

```python
from teragent.long_horizon.checkpoint import Checkpoint, CheckpointStore

store = CheckpointStore(base_dir=".teragent/checkpoints")

# Save checkpoint
checkpoint = Checkpoint(
    checkpoint_id=store.generate_checkpoint_id(),
    task_id="task_001",
    timestamp=store.now_iso(),
    phase="executing",
    completed_sub_goals=["sg_0", "sg_1"],
    current_sub_goal="sg_2",
    steps_completed=50,
    elapsed_minutes=30.0,
    strategy_switches=0,
    state_data={},
)
checkpoint_id = await store.save(checkpoint)

# Load latest
latest = await store.load_latest("task_001")

# List all
checkpoints = await store.list_checkpoints("task_001")

# Cleanup old (keep last 5)
deleted = await store.cleanup("task_001", keep_last=5)
```

### SelfEvaluator

```python
from teragent.long_horizon import SelfEvaluator, SelfEvaluationResult

evaluator = SelfEvaluator(
    model_provider=provider,
    evaluation_interval_steps=10,
    evaluation_interval_minutes=30.0,
)

if evaluator.should_evaluate(steps_since_last=10, minutes_since_last=30.0):
    result = await evaluator.evaluate(goal, progress_report, recent_results)
    if result.should_switch_strategy:
        # Trigger strategy switch
        ...
```

**SelfEvaluationResult attributes:**
- `goal_alignment: int` — Goal alignment score (1-5)
- `output_quality: int` — Output quality score (1-5)
- `bottleneck_identified: str` — Bottleneck description
- `strategy_review: str` — Strategy effectiveness review
- `next_step_plan: str` — Recommended next steps
- `overall_score: float` — Weighted overall score
- `should_switch_strategy: bool` — Whether to switch strategy
- `raw_response: str` — Raw model response text

**SelfEvaluator methods:**
- `evaluate(goal, progress_report, recent_results) -> SelfEvaluationResult` — Run self-evaluation
- `should_evaluate(steps_since_last, minutes_since_last) -> bool` — Check if evaluation is due
- `reset_evaluation_timer(current_steps)` — Reset evaluation timer

### StrategySwitcher

```python
from teragent.long_horizon import StrategySwitcher, StrategySwitchRecord

switcher = StrategySwitcher(
    model_provider=provider,
    stagnation_threshold=3,
    no_progress_threshold=5,
    similarity_threshold=0.8,
)

is_stagnant, reason = switcher.detect_stagnation(recent_results, recent_steps)
if is_stagnant:
    new_strategy, record = await switcher.switch_strategy(
        current_strategy, reason, goal, progress_report
    )
```

**StrategySwitchRecord attributes:**
- `timestamp: str` — ISO format timestamp
- `reason: str` — Switch reason
- `previous_strategy: str` — Previous strategy description
- `new_strategy: str` — New strategy description
- `risk_assessment: str` — Risk assessment
- `effectiveness: str` — Post-hoc effectiveness evaluation

**StrategySwitcher methods:**
- `detect_stagnation(recent_results, recent_steps) -> (bool, str)` — Detect stagnation
- `switch_strategy(current_strategy, reason, goal, progress_report) -> (str, StrategySwitchRecord)` — Execute strategy switch
- `get_switch_history() -> list[StrategySwitchRecord]` — Get switch history
- `assess_switch_effectiveness(record_index, subsequent_results) -> str` — Evaluate switch effectiveness

**Properties:**
- `current_strategy -> str` — Current strategy description

### ProgressTracker & ProgressReport

```python
from teragent.long_horizon.progress import ProgressTracker, ProgressReport

tracker = ProgressTracker(task_id="task_1", goal="Implement user system")
tracker.start_sub_goal("sg_1", "Design database")
tracker.record_step("Create User table")
tracker.complete_sub_goal("sg_1", "User table created")
report = tracker.get_report()
```

**ProgressReport attributes:**
- `task_id: str` — Task identifier
- `goal: str` — Original goal
- `total_sub_goals: int` — Total sub-goals
- `completed_sub_goals: int` — Completed sub-goals
- `current_phase: str` — Current phase (`planning`/`executing`/`evaluating`/`stagnant`)
- `steps_completed: int` — Steps taken
- `elapsed_minutes: float` — Elapsed time
- `strategy_switches: int` — Strategy switch count
- `estimated_remaining_minutes: float` — Estimated time remaining

---

## Budget Module (`teragent.reliability.budget`)

### StepBudget

```python
from teragent.reliability.budget import StepBudget

budget = StepBudget(max_steps=50)

if budget.consume():  # Returns True if budget remaining
    # Do work
    pass

budget.resume(extra_steps=10)  # Add more steps after user confirmation
```

**Properties:**
- `current_steps -> int` — Steps consumed
- `remaining -> int` — Steps remaining
- `exhausted -> bool` — Whether budget is exhausted

### CostRecord

```python
from teragent.reliability.budget import CostRecord

record = CostRecord(
    driver_name="openai_compatible.deepseek_v4_pro",
    compiler="deepseek_v4",
    model="deepseek-v4-pro",
    intent="design",
    prompt_tokens=5000,
    completion_tokens=2000,
    cache_hit_tokens=3000,
    cost_cny=0.052,
    cost_saved_cny=0.012,
    success=True,
    latency_ms=3500.0,
)
```

**Properties:**
- `date_str -> str` — Date string (YYYY-MM-DD) for date-dimension reporting
- `total_tokens -> int` — Total tokens (prompt + completion)

### MonthlyBudgetConfig

```python
from teragent.reliability.budget import MonthlyBudgetConfig

config = MonthlyBudgetConfig(
    limit_cny=500.0,                     # Monthly budget cap (0 = no limit)
    warning_threshold=0.8,                # Warn at 80%
    critical_threshold=0.95,              # Auto-downgrade at 95%
    auto_downgrade_driver="openai_compatible.deepseek_v4_flash",  # Fallback driver
    notify_on_warning=True,               # Emit events on warning
)
```

### CrossModelCostTracker

```python
from teragent.reliability.budget import CrossModelCostTracker, MonthlyBudgetConfig, CostRecord

tracker = CrossModelCostTracker()
tracker.set_monthly_budget(MonthlyBudgetConfig(limit_cny=500.0))

# Record a cost
budget_status = tracker.record(CostRecord(
    driver_name="openai_compatible.deepseek_v4_pro",
    compiler="deepseek_v4",
    model="deepseek-v4-pro",
    intent="design",
    prompt_tokens=5000,
    completion_tokens=2000,
    cost_cny=0.052,
))

# Record from TAP response (convenience method)
status = tracker.record_from_tap_response(
    driver_name="openai_compatible.deepseek_v4_pro",
    compiler="deepseek_v4",
    model="deepseek-v4-pro",
    intent="design",
    prompt_tokens=5000,
    completion_tokens=2000,
    cache_hit_tokens=3000,
    latency_ms=3500.0,
)

# Check budget
status = tracker.check_budget()
# → {"level": "ok"|"warning"|"critical"|"exhausted", "utilization": float, ...}

# Generate report
report = tracker.generate_report(group_by="model")
model_stats = tracker.get_model_stats("openai_compatible.deepseek_v4_pro")
all_stats = tracker.get_all_model_stats()
cache_savings = tracker.get_cache_savings()
```

**Key methods:**
- `record(record) -> dict` — Record a cost entry and check budget
- `record_from_tap_response(...) -> dict` — Record from TAP response data (calculates cost)
- `check_budget() -> dict` — Check current budget status
- `set_monthly_budget(config)` — Configure monthly budget
- `generate_report(group_by, start_date, end_date) -> dict` — Generate cost report
- `get_model_stats(driver_name) -> dict` — Get stats for a specific model
- `get_all_model_stats() -> dict` — Get stats for all models
- `get_cache_savings() -> dict` — Get cache savings statistics

**Properties:**
- `is_budget_warning -> bool` — Whether budget is in warning state
- `is_budget_exhausted -> bool` — Whether budget is exhausted
- `should_auto_downgrade -> bool` — Whether to auto-downgrade
- `total_records -> int` — Number of cost records

---

## Circuit Breaker Module (`teragent.reliability.circuit_breaker`)

### ModelCircuitBreakerManager

```python
from teragent.reliability.circuit_breaker import ModelCircuitBreakerManager, ModelBreakerConfig

manager = ModelCircuitBreakerManager()

# Record a failure (returns fallback model name if breaker just opened)
fallback = manager.record_failure("deepseek_v4_pro", "API timeout")

# Record a success
manager.record_success("deepseek_v4_pro")

# Check if a model can be called
if manager.can_call("deepseek_v4_pro"):
    # Safe to call
    ...

# Get fallback model
fallback = manager.get_fallback("deepseek_v4_pro")

# Get all model states
states = manager.get_all_states()
# → {"deepseek_v4_pro": "closed", "minimax_m3": "half_open", ...}

# Reset a specific model or all
manager.reset("deepseek_v4_pro")  # Reset specific
manager.reset()                    # Reset all
```

### ModelBreakerConfig

```python
from teragent.reliability.circuit_breaker import ModelBreakerConfig

config = ModelBreakerConfig(
    model_name="deepseek_v4_pro",
    max_consecutive_failures=5,       # Open after N consecutive failures
    window_seconds=300.0,            # Sliding window duration
    cooldown_seconds=60.0,           # Time before half-open transition
    failure_threshold_percent=0.5,   # Open if >50% failures in window
    half_open_max_calls=3,           # Test calls allowed in half-open
)
```

---

## Recovery Module (`teragent.reliability.recovery`)

### DegradationChain

```python
from teragent.reliability.recovery import DegradationChain

chain = DegradationChain(breaker_manager=breaker_mgr)

# Get next available fallback
fallback = chain.get_fallback("deepseek_v4_pro", task_type="heavy")
# → "glm_5"

# Get full chain
full_chain = chain.get_full_chain("heavy")
# → ["deepseek_v4_pro", "glm_5", "deepseek_v4_flash"]

# Add custom chain
chain.add_chain("light", ["deepseek_v4_flash", "glm_5"])
```

**Default chains:**
- `"heavy"`: V4-Pro → GLM-5 → V4-Flash
- `"multimodal"`: M3 → V4-Pro (degrades to text-only)
- `"default"`: V4-Pro → GLM-5 → V4-Flash

### LongHorizonRecoveryManager

```python
from teragent.reliability.recovery import LongHorizonRecoveryManager
from teragent.long_horizon.checkpoint import CheckpointStore

store = CheckpointStore()
recovery_mgr = LongHorizonRecoveryManager(checkpoint_store=store)

# Recover from latest checkpoint
success = await recovery_mgr.recover_from_checkpoint(task_manager)

# Check if should downgrade to standard mode
if recovery_mgr.should_downgrade_to_standard(recovery_attempts=3, elapsed_time=1800):
    print("Switching to standard mode")

# Get reconnection delay (exponential backoff with jitter)
delay = recovery_mgr.get_reconnection_delay(attempt=2)
```

### RateLimitHandler & RateLimitInfo

```python
from teragent.reliability.recovery import RateLimitHandler, RateLimitInfo

handler = RateLimitHandler(breaker_manager=breaker_mgr)

# Parse rate limit response (normalizes different provider formats)
info = handler.parse_rate_limit_response(
    model_name="deepseek_v4_pro",
    status_code=429,
    headers={"Retry-After": "30"},
    body=None,
)

# Check if should retry
if handler.should_retry("deepseek_v4_pro", info):
    delay = handler.get_backoff_delay("deepseek_v4_pro", attempt=1, rate_limit_info=info)
```

**RateLimitInfo attributes:**
- `model_name: str` — Model that returned the rate limit response
- `requests_remaining: int | None` — Remaining requests in current window
- `tokens_remaining: int | None` — Remaining tokens in current window
- `reset_time: float | None` — Unix timestamp when window resets
- `retry_after: float | None` — Seconds to wait before retrying

---

## MiniMax Adapter (`teragent.core.adapters.minimax_native`)

### MiniMaxNativeAdapter

```python
from teragent.core.adapters.minimax_native import MiniMaxNativeAdapter

adapter = MiniMaxNativeAdapter(
    base_url="https://api.minimaxi.com/v1",
    api_key="your-api-key",
    group_id="your-group-id",       # Optional: required for some endpoints
    timeout=300.0,
    multimodal_timeout=600.0,       # Longer timeout for video processing
    enable_fake_tools=False,
)
```

**MiniMax-specific methods:**
- `send_desktop_command(command, params, screenshot, interactive_elements, active_window, model) -> dict` — Send desktop command via dedicated endpoint
  - Returns `{"action": ..., "reasoning": ..., "raw_response": ...}`
  - Falls back to chat completions if desktop endpoint unavailable (404)

**MiniMax-specific properties:**
- `billing_summary -> dict` — Cumulative billing tracker (input/output/cache tokens, request count)
- `rate_limit_info -> MiniMaxRateLimitInfo` — Current rate limit information from headers

**Capabilities:**
- `multimodal: True`
- `desktop: True`
- `video: True`
- `msa_efficient: True`
- `max_context_tokens: 1_000_000`

### MiniMaxRateLimitInfo

```python
from teragent.core.adapters.minimax_native import MiniMaxRateLimitInfo

info = MiniMaxRateLimitInfo()
# Updated automatically from response headers
```

**Properties:**
- `limit: int` — Maximum requests in current window
- `remaining: int` — Remaining requests
- `reset: float` — Timestamp when window resets
- `is_exhausted -> bool` — Whether rate limit is exhausted
- `reset_in_seconds -> float` — Seconds until window resets

---

## GLM Adapter (`teragent.core.adapters.glm_native`)

### GLMNativeAdapter

```python
from teragent.core.adapters.glm_native import GLMNativeAdapter

adapter = GLMNativeAdapter(
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key="your-api-key",
    timeout=300.0,
    multimodal_timeout=600.0,       # Longer timeout for 1M context requests
)
```

**GLM-specific capabilities:**
- Supports GLM-5 and GLM-5.2 model endpoints
- Compatible with Zhipu AI's native message format
- Handles High/Max thinking mode response parsing
- Supports PreservedThinking trace injection

**GLM-specific properties:**
- `capabilities -> dict` — Includes `streaming: True`, `tool_calling: True`, `thinking_modes: ["high", "max"]`

**Note:** GLM models can also be used via `openai_compatible` adapter with the appropriate compiler (`glm_5` or `glm_52`). The `glm_native` adapter provides Zhipu AI-specific optimizations.

---

## Compilers (`teragent.core.compilers`)

### DeepSeekV4Compiler

```python
from teragent.core.compilers.deepseek_v4 import DeepSeekV4Compiler

compiler = DeepSeekV4Compiler()
# The variant (flash/pro) is set via the `variant` parameter in compile() or driver config
```

**Features:**
- Cache-aware prompt layout (freezes system prompt and tool definitions at the beginning)
- Thinking mode support (`auto`, `quick`, `deep`)
- 1M context optimization
- Flash/Pro variant switching via `variant` parameter (not separate compiler names)

**Key `extra` dict keys:** `cache_aware`, `variant`

### GLM5Compiler

```python
from teragent.core.compilers.glm_5 import GLM5Compiler

compiler = GLM5Compiler()
```

**Features:**
- Recency effect optimization (key instruction placed last)
- Long-horizon task support with self-evaluation injection
- 200K context optimization

### GLM52Compiler

```python
from teragent.core.compilers.glm_52 import GLM52Compiler

compiler = GLM52Compiler()
```

**Features:**
- 1M context with context degradation support (1M → 200K)
- Dual thinking modes: High (default) and Max
- PreservedThinking: preserves reasoning traces across coding sessions
- 5V-Turbo vision coordination: enables vision→code→verify cycles with GLM-5V-Turbo

**Key `extra` dict keys:** `thinking_mode`, `preserved_thinking`, `vision_coordination`

### GLM5VTurboCompiler

```python
from teragent.core.compilers.glm_5v_turbo import GLM5VTurboCompiler

compiler = GLM5VTurboCompiler()
```

**Features:**
- Vision analysis compilation for GLM-5V-Turbo model
- Converts multimodal content into GLM-5V-Turbo's expected format
- Used in coordination with GLM52Compiler for vision→code→verify cycles

### MiniMaxM3Compiler

```python
from teragent.core.compilers.minimax_m3 import MiniMaxM3Compiler

compiler = MiniMaxM3Compiler()
```

**Features:**
- MSA full-text injection mode (1/20 compute cost at 1M context)
- Multimodal content encoding (image_url, video_url content blocks)
- Desktop context conversion (DesktopContext → M3 desktop operation instructions)
- Video processing hints (automatically added when using MiniMaxNativeAdapter)

**Key `extra` dict keys:** `minimax_video_mode`, `minimax_frame_sampling`

---

## Desktop Tool (`teragent.tools.desktop`)

### DesktopTool

```python
from teragent.tools.desktop import DesktopTool, DesktopSafetyConfig

safety = DesktopSafetyConfig(
    safe_zones=[],               # Forbidden click zones (x1, y1, x2, y2)
    min_interval=0.5,            # Min seconds between operations
    max_consecutive_ops=50,      # Max consecutive operations
    screenshot_quality=75,       # JPEG quality (1-100)
    screenshot_format="jpeg",    # "jpeg" or "png"
)

tool = DesktopTool(safety_config=safety)
```

**Supported actions (7 total):**

| Action | Description | Required Parameters |
|--------|-------------|---------------------|
| `screenshot` | Capture screen screenshot | None |
| `click` | Click at coordinates | `x`, `y`, `button` (optional) |
| `type_text` | Type text string | `text` |
| `scroll` | Scroll in direction | `direction`, `scroll_amount` |
| `hotkey` | Press keyboard shortcut | `keys` (comma-separated, e.g., `"ctrl,c"`) |
| `move_mouse` | Move mouse to coordinates | `x`, `y` |
| `drag` | Drag from start to end | `x`, `y`, `end_x`, `end_y` |

**Safety features (5 layers):**
1. Permission level — All operations require DESTRUCTIVE-level confirmation
2. Safe zones — Configurable forbidden click areas
3. Rate limiting — Minimum interval between operations
4. Consecutive ops cap — Max consecutive operations before forced pause
5. Blocked shortcuts — Dangerous key combinations (Alt+F4, Ctrl+Alt+Del, etc.) are blocked

**Properties:**
- `simulation_mode -> bool` — Whether in simulation mode (pyautogui unavailable)
- `safety_config -> DesktopSafetyConfig` — Current safety configuration

**Example:**

```python
result = await tool.execute({
    "action": "screenshot",
})
# Returns ToolResult with base64-encoded screenshot in data

result = await tool.execute({
    "action": "click",
    "x": 100,
    "y": 200,
    "button": "left",
})
```

---

## Event Bus (`teragent.event_bus`)

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

---

### Rate Limiting — `teragent.core.rate_limiter`

| Class / Function | Description |
|-----------------|-------------|
| `TokenBucketRateLimiter` | Token bucket rate limiter with `acquire()` and `wait_and_acquire()` |
| `SlidingWindowRateLimiter` | Sliding window rate limiter with timestamp tracking |
| `AdaptiveRateLimiter` | Adaptive rate limiter that learns from `X-RateLimit-*` headers; auto-backs off on 429 |
| `RateLimitConfig` | Configuration dataclass (strategy, max_tokens, refill_rate, window, safety_factor) |
| `RateLimitStrategy` | Enum: `TOKEN_BUCKET`, `SLIDING_WINDOW`, `ADAPTIVE` |
| `RateLimitStatus` | Current status with `is_limited` and `wait_seconds` properties |
| `RateLimiter` | Type alias: `TokenBucketRateLimiter \| SlidingWindowRateLimiter \| AdaptiveRateLimiter` |
| `create_rate_limiter(config=None)` | Factory function — returns appropriate limiter based on config |
