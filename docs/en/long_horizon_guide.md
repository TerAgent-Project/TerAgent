# Long-Horizon Task Usage Guide

This guide covers TerAgent's long-horizon task mode, which enables GLM-5 to execute autonomous tasks lasting up to 8 hours with checkpoint recovery, self-evaluation, and strategy switching.

---

## Table of Contents

- [What is Long-Horizon Task Mode?](#what-is-long-horizon-task-mode)
- [When to Use Long-Horizon Mode](#when-to-use-long-horizon-mode)
- [Configuration](#configuration)
- [Step-by-Step Usage](#step-by-step-usage)
- [Checkpoint Management and Recovery](#checkpoint-management-and-recovery)
- [Self-Evaluation](#self-evaluation)
- [Strategy Switching](#strategy-switching)
- [Best Practices for 8-Hour Autonomous Tasks](#best-practices-for-8-hour-autonomous-tasks)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Known Limitations](#known-limitations)

---

## What is Long-Horizon Task Mode?

Long-horizon task mode is a specialized execution mode that allows GLM-5 to autonomously work on complex, multi-step tasks for extended periods (up to 8 hours). The system provides:

1. **Goal Decomposition** — Breaking large goals into manageable sub-goals with dependency ordering (DAG topology)
2. **Checkpoint Recovery** — Automatic state snapshots that allow resuming from any interruption point
3. **Self-Evaluation** — Periodic self-assessment to detect goal drift and strategy failures
4. **Strategy Switching** — Automatic strategy adjustment when stagnation is detected
5. **Progress Tracking** — Real-time progress reporting with estimated remaining time

The core workflow:

```
Goal → Decompose into Sub-Goals → Execute each Sub-Goal → Checkpoint → Evaluate → Continue/Adjust
```

---

## When to Use Long-Horizon Mode

### Good Use Cases

- **Large codebase refactoring** — Restructuring a project with many interconnected files
- **Full-stack feature development** — Implementing a complete feature from database to UI
- **Comprehensive code review** — Deep analysis of an entire codebase
- **Documentation generation** — Creating extensive documentation for a large project
- **Migration projects** — Migrating from one framework/architecture to another
- **Testing suite creation** — Building comprehensive test suites for an existing codebase

### When NOT to Use

- Simple Q&A or chat tasks
- Single-file code changes
- Quick debugging sessions
- Tasks that complete in under 10 minutes
- Tasks that don't require autonomous decision-making

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
[drivers.openai_compatible.glm_5]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5"
compiler = "glm_5"
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
long_horizon_driver = "openai_compatible.glm_5"  # Route long-horizon to GLM-5
```

---

## Step-by-Step Usage

### Basic Usage

```python
import asyncio
from teragent import create_provider, TAPRequest
from teragent.core.tap import LongHorizonConfig
from teragent.long_horizon import LongHorizonTaskManager

async def main():
    # 1. Create GLM-5 provider
    provider = create_provider(
        compiler="glm_5",
        adapter="openai_compatible",
        model="glm-5",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="GLM_API_KEY",
    )

    # 2. Configure long-horizon task
    config = LongHorizonConfig(
        max_duration_hours=4,
        checkpoint_interval_minutes=15,
        evaluation_interval_steps=10,
    )

    # 3. Create task manager
    manager = LongHorizonTaskManager(
        goal="Implement a complete user authentication system with JWT tokens, "
             "role-based access control, and audit logging. Include registration, "
             "login, password reset, and session management endpoints.",
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
    long_horizon_config=LongHorizonConfig(
        max_duration_hours=2,
        checkpoint_interval_minutes=10,
    ),
)

# The ModelRouter will automatically route this to GLM-5
# when [routing].long_horizon_driver is configured
```

---

## Checkpoint Management and Recovery

### How Checkpoints Work

Every `checkpoint_interval_minutes`, the system automatically saves a checkpoint containing:
- Completed sub-goal IDs
- Current sub-goal being executed
- Steps completed and elapsed time
- Strategy switch count
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

### Recovering from Interruption

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

---

## Self-Evaluation

### How Self-Evaluation Works

The `SelfEvaluator` periodically injects an evaluation prompt into GLM-5, asking it to assess:

| Dimension | Scale | Description |
|-----------|-------|-------------|
| Goal alignment | 1-5 | Is the current direction aligned with the original goal? |
| Output quality | 1-5 | How good is the completed work? |
| Bottleneck identified | Text | What's blocking progress? |
| Strategy review | Text | Is the current strategy effective? |
| Next step plan | Text | What should happen next? |

### Trigger Conditions

Evaluation is triggered when **either** condition is met:
- Steps since last evaluation ≥ `evaluation_interval_steps`
- Minutes since last evaluation ≥ `evaluation_interval_minutes`

### Using Self-Evaluation Results

```python
from teragent.long_horizon import SelfEvaluator

evaluator = SelfEvaluator(
    model_provider=provider,
    evaluation_interval_steps=10,
    evaluation_interval_minutes=30.0,
)

# Check if evaluation is due
if evaluator.should_evaluate(steps_since_last=10, minutes_since_last=30.0):
    result = await evaluator.evaluate(goal, progress_report, recent_results)

    print(f"Goal alignment: {result.goal_alignment}/5")
    print(f"Output quality: {result.output_quality}/5")
    print(f"Overall score: {result.overall_score:.1f}")
    print(f"Bottleneck: {result.bottleneck_identified}")
    print(f"Should switch strategy: {result.should_switch_strategy}")

    # Automatic strategy switch if score is too low
    if result.should_switch_strategy:
        # The task manager will automatically trigger a strategy switch
        pass
```

### Evaluation Score Interpretation

| Score | Meaning | Action |
|-------|---------|--------|
| 4.0-5.0 | On track, high quality | Continue as planned |
| 3.0-3.9 | Acceptable but could improve | Monitor closely |
| 2.0-2.9 | Significant drift or quality issues | Consider strategy switch |
| 1.0-1.9 | Severe problems | Strategy switch required |

---

## Strategy Switching

### When Strategy Switching Occurs

The `StrategySwitcher` detects stagnation through four signals:

1. **Consecutive similar results** — N consecutive PhaseResults with Jaccard similarity > threshold
2. **No file output** — M consecutive steps without new file creation or modification
3. **Consecutive failures** — N consecutive failed sub-goal executions
4. **Self-evaluation recommendation** — The self-evaluator recommends switching

### Available Strategy Directions

When stagnation is detected, the model chooses from these strategies:

1. **Decompose** — Break the current sub-goal into smaller steps
2. **Backtrack** — Return to the last successful state and try a different path
3. **Skip** — Skip the stuck sub-goal and complete other parts first
4. **Tool change** — Try a different technical approach or tool
5. **Incremental validation** — Validate each small step to prevent drift
6. **Replan** — Re-evaluate the entire goal decomposition

### Manual Strategy Switching

```python
from teragent.long_horizon import StrategySwitcher

switcher = StrategySwitcher(
    model_provider=provider,
    stagnation_threshold=3,      # 3 consecutive similar results
    no_progress_threshold=5,     # 5 steps without output
    similarity_threshold=0.8,    # 80% Jaccard similarity
)

# Detect stagnation
is_stagnant, reason = switcher.detect_stagnation(recent_results, recent_steps)

if is_stagnant:
    new_strategy, record = await switcher.switch_strategy(
        current_strategy=switcher.current_strategy,
        reason=reason,
        goal="Original goal description",
        progress_report=tracker.get_report(),
    )
    print(f"New strategy: {new_strategy}")
    print(f"Risk assessment: {record.risk_assessment}")

# Review switch history
for record in switcher.get_switch_history():
    print(f"[{record.timestamp}] {record.previous_strategy[:40]} → {record.new_strategy[:40]}")
```

---

## Best Practices for 8-Hour Autonomous Tasks

### 1. Write Clear, Specific Goals

```python
# Good: Clear, specific, measurable
goal = "Implement a REST API for user management with: (1) POST /register, "
       "(2) POST /login returning JWT, (3) GET /profile with auth, "
       "(4) PUT /profile with auth, (5) POST /logout. Use FastAPI, "
       "PostgreSQL, and include unit tests with >80% coverage."

# Bad: Vague, open-ended
goal = "Make a user system"
```

### 2. Set Appropriate Time Limits

- Start with 2-4 hours for new tasks
- Use 8 hours only for well-understood, large tasks
- Always set a `max_duration_hours` to prevent runaway tasks

### 3. Configure Checkpoint Frequency

| Task Duration | Recommended Checkpoint Interval |
|---------------|-------------------------------|
| 1-2 hours | 5 minutes |
| 2-4 hours | 10 minutes |
| 4-8 hours | 15 minutes |

### 4. Monitor Progress Regularly

```python
# Check progress report periodically
report = tracker.get_report()
print(f"Phase: {report.current_phase}")
print(f"Progress: {report.completed_sub_goals}/{report.total_sub_goals}")
print(f"Elapsed: {report.elapsed_minutes:.1f} min")
print(f"Remaining: {report.estimated_remaining_minutes:.1f} min (est)")
```

### 5. Handle Interruptions Gracefully

```python
# When your process is interrupted:
# 1. The latest checkpoint is already saved
# 2. Use LongHorizonRecoveryManager to resume

store = CheckpointStore()
recovery = LongHorizonRecoveryManager(checkpoint_store=store)

# On restart:
success = await recovery.recover_from_checkpoint(task_manager)
if success:
    result = await task_manager.execute_long_task()  # Resumes from checkpoint
```

### 6. Use Budget Controls

```python
from teragent.reliability.budget import CrossModelCostTracker, MonthlyBudgetConfig

tracker = CrossModelCostTracker()
tracker.set_monthly_budget(MonthlyBudgetConfig(
    limit_cny=100.0,  # Cap spending for long-horizon tasks
    auto_downgrade=True,
))
```

### 7. Set Maximum Strategy Switches

Prevent infinite strategy cycling by setting `max_strategy_switches` (default: 5). If this limit is reached, the task will terminate with a summary of what was accomplished.

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

### Sub-Goal Status

```python
for sg in report.sub_goal_statuses:
    status_icon = {"completed": "✓", "in_progress": "→", "pending": "○", "failed": "✗"}
    print(f"  {status_icon.get(sg['status'], '?')} {sg['id']}: {sg['description']}")
```

### Decision Logging

The `LongHorizonTaskManager` logs all major decisions:
- Goal decomposition
- Checkpoint saves
- Self-evaluation results
- Strategy switches
- Phase completions and failures

### Circuit Breaker Integration

```python
from teragent.reliability.circuit_breaker import ModelCircuitBreakerManager

breaker = ModelCircuitBreakerManager()
states = breaker.get_all_states()
# → {"glm_5": "closed", "deepseek_v4_flash": "closed", ...}

if not breaker.can_call("glm_5"):
    fallback = breaker.get_fallback("glm_5")
    print(f"GLM-5 unavailable, falling back to {fallback}")
```

---

## Known Limitations

1. **Context window** — GLM-5 has a 200K token context window. Tasks requiring more context will be automatically routed to V4-Pro or M3, but this may affect long-horizon coherence.

2. **API rate limits** — Long-horizon tasks make many API calls. Ensure your API plan supports the expected request volume. Configure `RateLimitHandler` for automatic backoff.

3. **Checkpoint size** — Checkpoints are stored as JSON files. Very large state data may slow down checkpoint save/load operations.

4. **Strategy switch limits** — The maximum number of strategy switches (default: 5) prevents infinite cycling but may terminate tasks that genuinely need more adaptation.

5. **No cross-session memory** — Each long-horizon task starts fresh. If you need context from previous tasks, include it in the goal description.

6. **Self-evaluation cost** — Self-evaluation uses additional API calls. For very frequent evaluation intervals, this can increase costs by 10-20%.

7. **Single-model execution** — Long-horizon tasks currently execute on a single model (GLM-5). Multi-model long-horizon orchestration is not yet supported.

8. **Desktop operations** — Long-horizon tasks cannot use desktop operations (M3 feature). Use M3 separately for visual tasks.
