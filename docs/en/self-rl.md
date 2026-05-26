# Self-RL Data Constitution

TerAgent includes a complete self-reinforcement learning data pipeline. Every TAP call can be automatically traced and paired with deterministic verification results to produce DPO (Direct Preference Optimization) training pairs.

## Data Constitution Principles

1. **TAP traces are a core library output**, independent of specific agent flows
2. **Preference labels come from deterministic checks** (AST, syntax, runnability), not from human annotation
3. **Data belongs to the user** — the library never uploads traces

## Architecture

```
TAPRequest  →  TAPTracer.record_request()  →  JSONL trace file
TAPResponse →  TAPTracer.record_response() →  JSONL trace file
Checklist   →  TAPTracer.record_checklist() →  JSONL trace file
                                                    ↓
                                     TAPTracer.export_dpo_pairs()
                                                    ↓
                                   (chosen=PASS, rejected=FAIL) pairs
```

## Quick Start

### 1. Create and Attach a Tracer

```python
from teragent import TAPTracer, create_provider

# Create tracer
tracer = TAPTracer(trace_dir="/project/.agent/traces")

# Attach to provider (auto-traces all TAP calls)
provider = create_provider(
    compiler="glm",
    adapter="openai_compatible",
    model="glm-5.1",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
    tracer=tracer,  # Auto-trace from creation
)

# Or attach later
provider.set_tracer(tracer)
```

### 2. Execute TAP Calls (Auto-Traced)

When a tracer is attached, `execute_tap()` automatically records:

1. **Request** — Before compilation: instruction, constraints, context keys, meta
2. **Response** — After API call: raw text, token usage, emptiness flag

```python
response = await provider.execute_tap(TAPRequest(
    meta={"task_id": "1.1", "intent": "code_generation"},
    instruction="Implement user login module",
    constraints=["Python 3.10+"],
))
# Trace is automatically recorded — no manual steps needed
```

### 3. Record Deterministic Checklist Results

After running code quality checks, record the results to create preference labels:

```python
from teragent import run_deterministic_checks, TaskInfo

task_list = [TaskInfo(id="1.1", title="Login module", status="completed")]
report, data = run_deterministic_checks("/project", task_list)

# Record the checklist result (determines PASS/FAIL label)
await tracer.record_checklist("1.1", {
    "fail_count": 0,        # Number of FAIL issues
    "warn_count": 2,        # Number of WARN issues
    "ok_count": 3,          # Number of OK checks
    "has_critical_warn": False,  # Whether critical warnings exist
    "needs_repair": False,  # Whether code needs repair
    "issues": [...],        # List of issue dicts
})
```

### 4. Export DPO Pairs

```python
# Generate DPO pairs
pairs = tracer.export_dpo_pairs()
# → [{"prompt": ..., "chosen": ..., "rejected": ..., "task_id": ..., ...}]

# Export to JSONL file
tracer.export_dpo_pairs_jsonl()  # → /project/.agent/traces/dpo_pairs_{ts}.jsonl

# Export all traces
tracer.export_traces_jsonl()
```

## DPO Pair Generation Strategies

### Strategy 1: Same-Task Retry Pairs

When a task is retried after failing checklist, both PASS and FAIL responses exist for the same task_id:

```
Task 1.1, Attempt 1 → FAIL (syntax errors)
Task 1.1, Attempt 2 → PASS (all checks pass)
→ DPO Pair: (chosen=Attempt 2, rejected=Attempt 1)
```

**Matching**: Responses and checklists are linked by `trace_id`, not positional index. This prevents incorrect pairings when retry counts differ.

### Strategy 2: Cross-Task Pairs

Different task_ids with the same intent form pairs:

```
Task 1.1 (code_generation) → PASS
Task 2.1 (code_generation) → FAIL
→ DPO Pair: (chosen=Task 1.1 response, rejected=Task 2.1 response)
```

### Strategy 3: Partial Pairs

When `include_partial=True`, tasks with only one outcome (PASS or FAIL) still generate partial pairs:

```
Task 1.1 → PASS (no FAIL attempt)
→ Partial pair: chosen=Task 1.1 response, rejected=""
```

## Data Structures

### TraceRecord

Each JSONL line is a `TraceRecord`:

```json
{
  "trace_id": "1.1_a1b2c3d4",
  "timestamp": 1715000000.0,
  "record_type": "tap_request",  // or "tap_response" or "checklist_result"
  "task_id": "1.1",
  "intent": "code_generation",
  "data": { ... }
}
```

### DPOPair

```json
{
  "prompt": "Instruction: Implement user login module\nConstraints: [...]\nOutput format: ...",
  "chosen": "PASS response text (good answer)",
  "rejected": "FAIL response text (bad answer)",
  "task_id": "1.1",
  "intent": "code_generation",
  "source": "deterministic_check",
  "metadata": {
    "pairing_strategy": "same_task_retry",
    "pass_trace_id": "...",
    "fail_trace_id": "..."
  },
  "constitution": {
    "version": "1.0",
    "principles": [...],
    "preference_source": "deterministic_check",
    "data_ownership": "user",
    "upload_policy": "never"
  }
}
```

### DataConstitution

Every export includes the data constitution metadata:

```python
from teragent.pipeline.tracing import DataConstitution

constitution = DataConstitution()
# → version="1.0"
# → preference_source="deterministic_check"
# → data_ownership="user"
# → upload_policy="never"
```

## Statistics

```python
stats = tracer.get_trace_stats()
# → TraceStats(
#     total_records=15,
#     request_count=5,
#     response_count=5,
#     checklist_count=5,
#     task_ids={"1.1", "1.2", "2.1"},
#     intents={"code_generation", "design"},
#     dpo_pair_count=3,
#     pass_count=4,
#     fail_count=1,
# )
```

## File Management

### Loading Traces

```python
# Load from specific file
count = tracer.load_from_file("/path/to/trace.jsonl")

# Load all trace files from trace_dir
count = tracer.load_all_traces()
```

### Trace File Format

Each trace file is JSONL with a constitution header:

```jsonl
{"type": "constitution", "data": {...}, "session_id": "abc123", "export_time": 1715000000}
{"trace_id": "...", "timestamp": ..., "record_type": "tap_request", ...}
{"trace_id": "...", "timestamp": ..., "record_type": "tap_response", ...}
{"trace_id": "...", "timestamp": ..., "record_type": "checklist_result", ...}
```

## Thread Safety

All write operations are thread-safe via an internal `threading.Lock`. Read operations (export, stats) operate on a snapshot. File writes use `write + flush + fsync` for durability.

## Integration with ModelProvider

When a tracer is attached to a `ModelProvider`, `execute_tap()` automatically:

1. Calls `tracer.record_request(request)` before compilation → returns `trace_id`
2. Calls `tracer.record_response(response, task_id, trace_id, intent)` after API call

**Streaming calls** (`stream_tap()`) are **NOT** traced — partial chunks don't form meaningful DPO pairs. Use `execute_tap()` for traced calls.
