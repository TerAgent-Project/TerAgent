# Long-Horizon Task Usage Guide

This guide covers TerAgent's long-horizon task mode, which enables GLM-5 and GLM-5.2 to execute autonomous tasks lasting 8+ hours with checkpoint recovery, self-evaluation, strategy switching, and 1M context support.

---

## Table of Contents

- [What is Long-Horizon Task Mode?](#what-is-long-horizon-task-mode)
- [When to Use Long-Horizon Mode](#when-to-use-long-horizon-mode)
- [Model Selection: GLM-5 vs GLM-5.2](#model-selection-glm-5-vs-glm-52)
- [Configuration](#configuration)
- [Step-by-Step Usage](#step-by-step-usage)
- [Checkpoint Management and Recovery](#checkpoint-management-and-recovery)
- [Self-Evaluation and Strategy Switching](#self-evaluation-and-strategy-switching)
- [1M Context for Long-Horizon Tasks](#1m-context-for-long-horizon-tasks)
- [Best Practices for 8+ Hour Tasks](#best-practices-for-8-hour-tasks)
- [Recovery from Interruptions](#recovery-from-interruptions)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Known Limitations](#known-limitations)

---

## What is Long-Horizon Task Mode?

Long-horizon task mode is a specialized execution mode that allows GLM-5 and GLM-5.2 to autonomously work on complex, multi-step tasks for extended periods (up to 8+ hours). The system provides:

1. **Goal Decomposition** — Breaking large goals into manageable sub-goals with dependency ordering (DAG topology)
2. **Checkpoint Recovery** — Automatic state snapshots that allow resuming from any interruption point
3. **Self-Evaluation** — Periodic self-assessment to detect goal drift and strategy failures
4. **Strategy Switching** — Automatic strategy adjustment when stagnation is detected
5. **Progress Tracking** — Real-time progress reporting with estimated remaining time
6. **1M Context (GLM-5.2)** — Ultra-long context window for analyzing massive codebases during long-running tasks

The core workflow:

```
Goal → Decompose into Sub-Goals → Execute each Sub-Goal → Checkpoint → Evaluate → Continue/Adjust
```

### Key Differences: GLM-5 vs GLM-5.2

| Feature | GLM-5 | GLM-5.2 |
|---------|-------|---------|
| Context window | 200K | 1M |
| Max task duration | 8 hours | 8+ hours (with degradation) |
| Thinking modes | deep | High/Max dual thinking |
| PreservedThinking | ❌ | ✅ |
| Context degradation | ❌ | ✅ (1M → 200K) |
| 5V-Turbo coordination | ❌ | ✅ |
| Cost per token | Lower | Higher |

---

## When to Use Long-Horizon Mode

### Good Use Cases

- **Large codebase refactoring** — Restructuring a project with many interconnected files
- **Full-stack feature development** — Implementing a complete feature from database to UI
- **Comprehensive code review** — Deep analysis of an entire codebase (use GLM-5.2 for >200K)
- **Documentation generation** — Creating extensive documentation for a large project
- **Migration projects** — Migrating from one framework/architecture to another
- **Testing suite creation** — Building comprehensive test suites for an existing codebase
- **Ultra-large codebase analysis** — Analyzing codebases exceeding 200K tokens (GLM-5.2 only)
- **Vision-assisted development** — Long-running tasks with UI mockup implementation (GLM-5.2 + 5V-Turbo)

### When NOT to Use

- Simple Q&A or chat tasks
- Single-file code changes
- Quick debugging sessions
- Tasks that complete in under 10 minutes
- Tasks that don't require autonomous decision-making

---

## Model Selection: GLM-5 vs GLM-5.2

### Choose GLM-5 When:

- Context window needed is ≤ 200K tokens
- Budget is a primary concern (GLM-5 is cheaper per token)
- Task is well-scoped and doesn't need ultra-long context
- Simpler deep reasoning is sufficient

### Choose GLM-5.2 When:

- Context window needed exceeds 200K tokens
- Task requires the deepest reasoning (Max thinking mode)
- You need PreservedThinking for multi-step coding plans
- Vision coordination with 5V-Turbo is needed
- Task may run longer than 8 hours and needs degradation support
- Analyzing ultra-large codebases (500K+ tokens)

### Configuration for Each Model

```toml
# GLM-5 long-horizon configuration
[drivers.openai_compatible.glm_5]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5"
compiler = "glm_5"
max_context_tokens = 200_000
long_horizon_enabled = true

# GLM-5.2 long-horizon configuration (with 1M context)
[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
max_context_tokens = 1_000_000
long_horizon_enabled = true
multimodal_enabled = true
# Note: dual_thinking_enabled, preserved_thinking_enabled are
# create_provider() kwargs, not TOML driver fields.
# Context degradation is handled internally by the AutoCompactor.

# Route long-horizon to the appropriate model
[routing]
long_horizon_driver = "openai_compatible.glm_52"  # Default to GLM-5.2
```

---

## Configuration

### LongHorizonConfig Fields

```python
from teragent.core.tap import LongHorizonConfig

config = LongHorizonConfig(
    max_duration_hours=4,              # Maximum task duration (default: 8)
    checkpoint_interval_minutes=15,    # Auto-checkpoint interval (default: 15)
    evaluation_interval_steps=10,      # Self-evaluate every N steps (default: 10)
    evaluation_interval_minutes=30,    # Self-evaluate every N minutes (default: 30)
)
```

### agent.toml Configuration

```toml
[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
long_horizon_enabled = true             # Required for long-horizon mode

[long_horizon]
max_duration_hours = 8.0
checkpoint_interval_minutes = 15.0
evaluation_interval_steps = 10
evaluation_interval_minutes = 30.0
stagnation_threshold = 3
no_progress_threshold = 5
similarity_threshold = 0.8
max_strategy_switches = 5
checkpoint_base_dir = ".teragent/checkpoints"
checkpoint_keep_last = 5

[routing]
long_horizon_driver = "openai_compatible.glm_52"
```

---

## Step-by-Step Usage

### Basic Usage with GLM-5.2

```python
import asyncio
from teragent import create_provider, TAPRequest
from teragent.core.tap import LongHorizonConfig
from teragent.long_horizon import LongHorizonTaskManager

async def main():
    # 1. Create GLM-5.2 provider with 1M context
    provider = create_provider(
        compiler="glm_52",
        adapter="openai_compatible",
        model="glm-5.2",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="GLM_API_KEY",
    )

    # 2. Configure long-horizon task
    config = LongHorizonConfig(
        max_duration_hours=6,
        checkpoint_interval_minutes=20,
        evaluation_interval_steps=10,
    )

    # 3. Create task manager
    manager = LongHorizonTaskManager(
        goal="Migrate the entire monolithic application to a microservices "
             "architecture. Include: (1) service boundary identification, "
             "(2) API gateway setup, (3) database per service pattern, "
             "(4) inter-service communication, (5) deployment configuration.",
        model_provider=provider,
        config=config,
    )

    # 4. Execute the long-horizon task
    result = await manager.execute_long_task()

    # 5. Review results
    print(f"Success: {result.success}")
    print(f"Total steps: {result.total_steps}")
    print(f"Duration: {result.total_elapsed_minutes:.1f} minutes")
    print(f"Sub-goals completed: {result.completed_sub_goals}/{result.total_sub_goals}")
    print(f"Strategy switches: {result.strategy_switches}")
    print(f"Summary: {result.final_summary}")

asyncio.run(main())
```

### Using TAPRequest with Long-Horizon Config

```python
from teragent import TAPRequest
from teragent.core.tap import LongHorizonConfig

request = TAPRequest(
    instruction="Refactor the entire authentication module to use OAuth2",
    long_horizon=LongHorizonConfig(
        max_duration_hours=2,
        checkpoint_interval_minutes=10,
    ),
)

# The ModelRouter will automatically route this to GLM-5.2
# when [routing].long_horizon_driver is configured
```

### Using PreservedThinking with Long-Horizon

```python
from teragent import create_provider
from teragent.core.tap import LongHorizonConfig
from teragent.long_horizon import LongHorizonTaskManager

provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    preserved_thinking_enabled=True,  # compiler-level kwarg
)

manager = LongHorizonTaskManager(
    goal="Build a complete e-commerce platform with product catalog, "
         "shopping cart, checkout, and order management",
    model_provider=provider,
    config=LongHorizonConfig(
        max_duration_hours=8,
        checkpoint_interval_minutes=15,
    ),
)

# PreservedThinking ensures that the reasoning trace from the
# initial planning phase carries through to all implementation steps
result = await manager.execute_long_task()
```

---

## Checkpoint Management and Recovery

### How Checkpoints Work

Every `checkpoint_interval_minutes`, the system automatically saves a checkpoint containing:
- Completed sub-goal IDs
- Current sub-goal being executed
- Steps completed and elapsed time
- Strategy switch count
- PreservedThinking traces (GLM-5.2 only)
- Context degradation state (GLM-5.2 only)
- Arbitrary state data for resumption

### Checkpoint Storage

Checkpoints are stored as JSON files in the configured directory:

```
.teragent/checkpoints/
├── task_001/
│   ├── a1b2c3d4-...json    # Checkpoint 1
│   ├── e5f6g7h8-...json    # Checkpoint 2
│   └── ...
└── task_002/
    └── ...
```

### Manual Checkpoint Operations

```python
from teragent.long_horizon.checkpoint import CheckpointStore, Checkpoint

store = CheckpointStore(base_dir=".teragent/checkpoints")

# List checkpoints for a task
checkpoints = await store.list_checkpoints("task_001")

# Load the latest checkpoint
latest = await store.load_latest("task_001")
print(f"Phase: {latest.phase}")
print(f"Completed sub-goals: {latest.completed_sub_goals}")
print(f"Steps: {latest.steps_completed}")

# Load a specific checkpoint
checkpoint = await store.load("a1b2c3d4-...")

# Cleanup old checkpoints (keep last 5)
deleted = await store.cleanup("task_001", keep_last=5)
```

### Checkpoint Size Considerations for 1M Context

GLM-5.2 checkpoints with 1M context can be significantly larger:

| Context Size | Approximate Checkpoint Size |
|-------------|---------------------------|
| 200K tokens | 2-5 MB |
| 500K tokens | 5-15 MB |
| 1M tokens | 15-50 MB |

Ensure sufficient disk space for long-running tasks with 1M context.

---

## Self-Evaluation and Strategy Switching

### Self-Evaluation

The `SelfEvaluator` periodically injects an evaluation prompt, asking the model to assess:

| Dimension | Scale | Description |
|-----------|-------|-------------|
| Goal alignment | 1-5 | Is the current direction aligned with the original goal? |
| Output quality | 1-5 | How good is the completed work? |
| Bottleneck identified | Text | What's blocking progress? |
| Strategy review | Text | Is the current strategy effective? |
| Next step plan | Text | What should happen next? |

### Strategy Switching

When stagnation is detected, the model chooses from these strategies:

1. **Decompose** — Break the current sub-goal into smaller steps
2. **Backtrack** — Return to the last successful state and try a different path
3. **Skip** — Skip the stuck sub-goal and complete other parts first
4. **Tool change** — Try a different technical approach or tool
5. **Incremental validation** — Validate each small step to prevent drift
6. **Replan** — Re-evaluate the entire goal decomposition

### Dual Thinking in Long-Horizon Tasks

GLM-5.2's dual thinking mode can be leveraged during long-horizon tasks:

```python
# The task manager can switch thinking modes based on sub-goal complexity
request = TAPRequest(
    instruction="Evaluate the current progress and decide next steps",
    meta={
        "thinking_mode": "max",  # Use Max thinking for evaluation steps
    },
    long_horizon=LongHorizonConfig(
        max_duration_hours=6,
    ),
)
```

---

## 1M Context for Long-Horizon Tasks

### When 1M Context Matters

Long-horizon tasks naturally accumulate context over time. With GLM-5's 200K limit, tasks analyzing large codebases may run out of context before completion. GLM-5.2's 1M window addresses this:

- **Large codebase analysis** — Load the entire codebase once, analyze progressively
- **Cumulative context** — Accumulate findings, plans, and implementation details without truncation
- **PreservedThinking** — Maintain reasoning traces throughout the entire task duration

### Context Degradation During Long-Horizon Tasks

For tasks running 8+ hours, context degradation is critical:

```toml
[drivers.openai_compatible.glm_52]
# Context degradation is handled internally by the AutoCompactor
```

The degradation behavior:
1. Start at 1M context
2. When memory usage exceeds the threshold, degrade to 200K
3. Existing context is compacted (preserve system prompt, recent messages, tool definitions)
4. When memory pressure subsides, recover to 1M

### Best Practices for 1M Context Long-Horizon

1. **Load the full codebase upfront** — Take advantage of 1M window in the initial analysis
2. **Use High thinking for execution** — Save Max thinking for critical decision points
3. **Enable PreservedThinking** — Keep the planning context alive throughout
4. **Set larger checkpoint intervals** — 1M context checkpoints are larger and slower to save
5. **Monitor memory closely** — Set up alerts for context degradation events
6. **Plan for degradation** — Assume the context may degrade during long tasks

---

## Best Practices for 8+ Hour Tasks

### 1. Write Clear, Specific Goals

```python
# Good: Clear, specific, measurable
goal = ("Migrate the monolithic User Service to a standalone microservice. "
        "Include: (1) Extract user domain models, (2) Create REST API with "
        "CRUD endpoints, (3) Implement JWT authentication, (4) Set up "
        "PostgreSQL database, (5) Add Docker configuration, (6) Write "
        "integration tests with >80% coverage.")

# Bad: Vague, open-ended
goal = "Make the user service better"
```

### 2. Set Appropriate Time Limits

| Task Complexity | Recommended Duration | Recommended Model |
|----------------|---------------------|-------------------|
| Small refactoring | 1-2 hours | GLM-5 |
| Feature development | 2-4 hours | GLM-5 or GLM-5.2 |
| Large migration | 4-8 hours | GLM-5.2 |
| Full system rewrite | 8+ hours | GLM-5.2 (with degradation) |

### 3. Configure Checkpoint Frequency

| Task Duration | Context Size | Recommended Checkpoint Interval |
|---------------|-------------|-------------------------------|
| 1-2 hours | 200K | 5 minutes |
| 2-4 hours | 200K | 10 minutes |
| 4-8 hours | 1M | 15 minutes |
| 8+ hours | 1M | 20 minutes |

### 4. Use Budget Controls

```python
from teragent.reliability.budget import CrossModelCostTracker, MonthlyBudgetConfig

tracker = CrossModelCostTracker()
tracker.set_monthly_budget(MonthlyBudgetConfig(
    limit_cny=200.0,  # Cap spending for long-horizon tasks
    auto_downgrade=True,
))
```

### 5. Monitor Progress Regularly

```python
report = tracker.get_report()
print(f"Phase: {report.current_phase}")
print(f"Progress: {report.completed_sub_goals}/{report.total_sub_goals}")
print(f"Elapsed: {report.elapsed_minutes:.1f} min")
print(f"Remaining: {report.estimated_remaining_minutes:.1f} min (est)")
```

### 6. Set Maximum Strategy Switches

Prevent infinite strategy cycling by setting `max_strategy_switches` (default: 5). If this limit is reached, the task will terminate with a summary of what was accomplished.

---

## Recovery from Interruptions

### Automatic Recovery

```python
from teragent.reliability.recovery import LongHorizonRecoveryManager
from teragent.long_horizon.checkpoint import CheckpointStore

store = CheckpointStore(base_dir=".teragent/checkpoints")
recovery_mgr = LongHorizonRecoveryManager(
    checkpoint_store=store,
    max_reconnection_attempts=5,
    reconnection_base_delay=2.0,
)

# Attempt recovery from latest checkpoint
success = await recovery_mgr.recover_from_checkpoint(task_manager)

if not success:
    # Check if we should downgrade to standard (non-long-horizon) mode
    if recovery_mgr.should_downgrade_to_standard(
        recovery_attempts=3,
        elapsed_time=1800,  # 30 minutes elapsed
    ):
        print("Too many failures; switching to standard mode")
```

### Handling Context Degradation Recovery

When GLM-5.2's context degrades during a long-horizon task:

1. **Checkpoint captures the degradation state** — The checkpoint includes the current context mode
2. **On recovery, the degraded mode is preserved** — The task resumes at 200K context
3. **Manual recovery to 1M** — After the task completes or memory pressure subsides, you can restart at 1M

```python
# After recovery, check the context mode
checkpoint = await store.load_latest("task_001")
if hasattr(checkpoint, 'context_mode') and checkpoint.context_mode == "200K":
    print("Task was interrupted during degraded mode")
    print("Consider restarting with fresh 1M context for better results")
```

### Cross-Session Recovery

Long-horizon tasks that span process restarts:

```python
# On process restart:
store = CheckpointStore(base_dir=".teragent/checkpoints")
recovery = LongHorizonRecoveryManager(checkpoint_store=store)

# Resume from the last checkpoint
success = await recovery.recover_from_checkpoint(task_manager)
if success:
    result = await task_manager.execute_long_task()  # Resumes from checkpoint
```

---

## Monitoring and Debugging

### Progress Tracking

```python
from teragent.long_horizon.progress import ProgressTracker

tracker = ProgressTracker(task_id="task_1", goal="My goal")

# During execution:
report = tracker.get_report()
# report.current_phase: "planning" | "executing" | "evaluating" | "stagnant"
# report.completed_sub_goals / report.total_sub_goals
# report.steps_completed
# report.elapsed_minutes
# report.estimated_remaining_minutes
# report.strategy_switches
```

### Context Degradation Monitoring

```python
from teragent.context import ContextWindow

# The ContextWindow tracks utilization
utilization = context_window.usage_ratio()
print(f"Context utilization: {utilization:.1%}")
if utilization > 0.9:
    print("⚠️ Context utilization high, consider enabling auto-compaction")
```

### Circuit Breaker Integration

```python
from teragent.reliability.circuit_breaker import ModelCircuitBreakerManager

breaker = ModelCircuitBreakerManager()
states = breaker.get_all_states()
# → {"glm_52": "closed", "glm_5": "closed", "deepseek_v4_flash": "closed", ...}

if not breaker.can_call("glm_52"):
    fallback = breaker.get_fallback("glm_52")
    print(f"GLM-5.2 unavailable, falling back to {fallback}")
```

---

## Known Limitations

1. **Context window limits** — GLM-5 has 200K context; tasks requiring more will be automatically routed to GLM-5.2 (if configured). If GLM-5.2 is not available, context compaction may affect coherence.

2. **API rate limits** — Long-horizon tasks make many API calls. Ensure your API plan supports the expected request volume. Configure `RateLimitHandler` for automatic backoff.

3. **Checkpoint size** — With GLM-5.2 at 1M context, checkpoints can be 15-50 MB. Ensure sufficient disk space and network bandwidth for checkpoint operations.

4. **Strategy switch limits** — The maximum number of strategy switches (default: 5) prevents infinite cycling but may terminate tasks that genuinely need more adaptation.

5. **No cross-session memory** — Each long-horizon task starts fresh. If you need context from previous tasks, include it in the goal description. PreservedThinking helps within a session but not across sessions.

6. **Self-evaluation cost** — Self-evaluation uses additional API calls. For very frequent evaluation intervals, this can increase costs by 10-20%.

7. **Single-model execution** — Long-horizon tasks currently execute on a single model (GLM-5 or GLM-5.2). Multi-model long-horizon orchestration is not yet supported.

8. **Desktop operations** — Long-horizon tasks cannot use desktop operations (M3 feature). Use M3 separately for visual tasks.

9. **1M context stability** — At full 1M context, GLM-5.2 may experience higher latency and memory pressure. Enable context degradation as a safety net.

10. **PreservedThinking overhead** — PreservedThinking adds ~2-5K tokens per preserved trace. For very long tasks with many traces, this can consume significant context.

---

*This guide is part of the TerAgent documentation. For the complete four-model adaptation guide, see [Adaptation Guide](adaptation_guide.md). For GLM-5.2 specific features, see [GLM-5.2 Guide](glm_52_guide.md).*
