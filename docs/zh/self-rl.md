# 自强化学习数据宪章

TerAgent 包含完整的自强化学习数据管道。每次 TAP 调用都会被自动追踪，并与确定性验证结果配对，生成 DPO（Direct Preference Optimization）训练对。

## 数据宪章原则

1. **TAP 追踪是核心库输出**，独立于特定的 Agent 流程
2. **偏好标签来自确定性检查**（AST、语法、可运行性），而非人工标注
3. **数据属于用户** —— 库绝不会上传追踪数据

## 架构

```
TAPRequest  →  TAPTracer.record_request()  →  JSONL trace file
TAPResponse →  TAPTracer.record_response() →  JSONL trace file
Checklist   →  TAPTracer.record_checklist() →  JSONL trace file
                                                    ↓
                                     TAPTracer.export_dpo_pairs()
                                                    ↓
                                   (chosen=PASS, rejected=FAIL) pairs
```

## 快速上手

### 1. 创建并附加追踪器

```python
from teragent import TAPTracer, create_provider

# Create tracer
tracer = TAPTracer(trace_dir="/project/.agent/traces")

# Attach to provider (auto-traces all TAP calls)
provider = create_provider(
    compiler="glm_5",
    adapter="openai_compatible",
    model="glm-5",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
    tracer=tracer,  # Auto-trace from creation
)

# Or attach later
provider.set_tracer(tracer)
```

### 2. 执行 TAP 调用（自动追踪）

当附加了追踪器时，`execute_tap()` 会自动记录：

1. **请求** — 编译前：指令、约束、上下文键、元数据
2. **响应** — API 调用后：原始文本、Token 用量、空响应标志

```python
response = await provider.execute_tap(TAPRequest(
    meta={"task_id": "1.1", "intent": "code_generation"},
    instruction="Implement user login module",
    constraints=["Python 3.10+"],
))
# Trace is automatically recorded — no manual steps needed
```

### 3. 记录确定性检查清单结果

运行代码质量检查后，记录结果以创建偏好标签：

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

### 4. 导出 DPO 对

```python
# Generate DPO pairs
pairs = tracer.export_dpo_pairs()
# → [{"prompt": ..., "chosen": ..., "rejected": ..., "task_id": ..., ...}]

# Export to JSONL file
tracer.export_dpo_pairs_jsonl()  # → /project/.agent/traces/dpo_pairs_{ts}.jsonl

# Export all traces
tracer.export_traces_jsonl()
```

## DPO 对生成策略

### 策略 1：同任务重试配对

当任务在检查清单失败后重试时，同一 task_id 会同时存在 PASS 和 FAIL 响应：

```
Task 1.1, Attempt 1 → FAIL (syntax errors)
Task 1.1, Attempt 2 → PASS (all checks pass)
→ DPO Pair: (chosen=Attempt 2, rejected=Attempt 1)
```

**匹配方式**：响应和检查清单通过 `trace_id` 关联，而非位置索引。这可以防止重试次数不同时的错误配对。

### 策略 2：跨任务配对

具有相同 intent 的不同 task_id 形成配对：

```
Task 1.1 (code_generation) → PASS
Task 2.1 (code_generation) → FAIL
→ DPO Pair: (chosen=Task 1.1 response, rejected=Task 2.1 response)
```

### 策略 3：部分配对

当 `include_partial=True` 时，只有一种结果（PASS 或 FAIL）的任务仍会生成部分配对：

```
Task 1.1 → PASS (no FAIL attempt)
→ Partial pair: chosen=Task 1.1 response, rejected=""
```

## 数据结构

### TraceRecord

每行 JSONL 是一个 `TraceRecord`：

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

每次导出都包含数据宪章元数据：

```python
from teragent.pipeline.tracing import DataConstitution

constitution = DataConstitution()
# → version="1.0"
# → preference_source="deterministic_check"
# → data_ownership="user"
# → upload_policy="never"
```

## 统计信息

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

## 文件管理

### 加载追踪

```python
# Load from specific file
count = tracer.load_from_file("/path/to/trace.jsonl")

# Load all trace files from trace_dir
count = tracer.load_all_traces()
```

### 追踪文件格式

每个追踪文件为 JSONL 格式，包含宪章头部：

```jsonl
{"type": "constitution", "data": {...}, "session_id": "abc123", "export_time": 1715000000}
{"trace_id": "...", "timestamp": ..., "record_type": "tap_request", ...}
{"trace_id": "...", "timestamp": ..., "record_type": "tap_response", ...}
{"trace_id": "...", "timestamp": ..., "record_type": "checklist_result", ...}
```

## 线程安全

所有写操作通过内部 `threading.Lock` 保证线程安全。读操作（导出、统计）在快照上执行。文件写入使用 `write + flush + fsync` 确保持久性。

## 与 ModelProvider 集成

当追踪器附加到 `ModelProvider` 时，`execute_tap()` 自动执行以下操作：

1. 编译前调用 `tracer.record_request(request)` → 返回 `trace_id`
2. API 调用后调用 `tracer.record_response(response, task_id, trace_id, intent)`

**流式调用**（`stream_tap()`）**不会**被追踪 —— 部分数据块无法形成有意义的 DPO 对。请使用 `execute_tap()` 进行追踪调用。
