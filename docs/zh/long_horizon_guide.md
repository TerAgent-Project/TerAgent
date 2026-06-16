# 长时程任务使用指南

本指南介绍 TerAgent 的长时程任务模式，该模式使 GLM-5 和 GLM-5.2 能够执行持续 8 小时以上的自主任务，支持检查点恢复、自我评估、策略切换和 1M 上下文支持。

---

## 目录

- [什么是长时程任务模式？](#什么是长时程任务模式)
- [何时使用长时程模式](#何时使用长时程模式)
- [模型选择：GLM-5 与 GLM-5.2](#模型选择glm-5-与-glm-52)
- [配置](#配置)
- [分步使用](#分步使用)
- [检查点管理与恢复](#检查点管理与恢复)
- [自我评估与策略切换](#自我评估与策略切换)
- [长时程任务的 1M 上下文](#长时程任务的-1m-上下文)
- [8 小时以上任务的最佳实践](#8-小时以上任务的最佳实践)
- [中断恢复](#中断恢复)
- [监控与调试](#监控与调试)
- [已知限制](#已知限制)

---

## 什么是长时程任务模式？

长时程任务模式是一种专门的执行模式，允许 GLM-5 和 GLM-5.2 在较长时间内（最多 8 小时以上）自主处理复杂的多步骤任务。系统提供以下功能：

1. **目标分解** — 将大型目标拆分为可管理的子目标，并按依赖关系排序（DAG 拓扑）
2. **检查点恢复** — 自动状态快照，允许从任何中断点恢复执行
3. **自我评估** — 定期自我评估，以检测目标偏移和策略失败
4. **策略切换** — 检测到停滞时自动调整策略
5. **进度追踪** — 实时进度报告，附带预估剩余时间
6. **1M 上下文（GLM-5.2）** — 超长上下文窗口，用于在长时间运行的任务中分析大规模代码库

核心工作流程：

```
目标 → 分解为子目标 → 执行每个子目标 → 检查点 → 评估 → 继续/调整
```

### 核心差异：GLM-5 与 GLM-5.2

| 特性 | GLM-5 | GLM-5.2 |
|------|-------|---------|
| 上下文窗口 | 200K | 1M |
| 最大任务时长 | 8 小时 | 8+ 小时（支持降级） |
| 思考模式 | deep | High/Max 双重思考 |
| PreservedThinking | ❌ | ✅ |
| 上下文降级 | ❌ | ✅（1M → 200K） |
| 5V-Turbo 协同 | ❌ | ✅ |
| 每 token 成本 | 较低 | 较高 |

---

## 何时使用长时程模式

### 适用场景

- **大型代码库重构** — 重构包含许多相互关联文件的项目
- **全栈功能开发** — 从数据库到 UI 实现完整功能
- **全面代码审查** — 对整个代码库进行深度分析（超过 200K 时请使用 GLM-5.2）
- **文档生成** — 为大型项目创建详尽的文档
- **迁移项目** — 从一个框架/架构迁移到另一个
- **测试套件创建** — 为现有代码库构建全面的测试套件
- **超大型代码库分析** — 分析超过 200K token 的代码库（仅限 GLM-5.2）
- **视觉辅助开发** — 长时间运行的任务，需要实现 UI 模型（GLM-5.2 + 5V-Turbo）

### 不适用场景

- 简单的问答或聊天任务
- 单文件代码修改
- 快速调试会话
- 10 分钟内可完成的任务
- 不需要自主决策的任务

---

## 模型选择：GLM-5 与 GLM-5.2

### 选择 GLM-5 的场景：

- 所需上下文窗口 ≤ 200K token
- 预算是首要考虑因素（GLM-5 每 token 更便宜）
- 任务范围明确，不需要超长上下文
- 简单的深度推理即可满足需求

### 选择 GLM-5.2 的场景：

- 所需上下文窗口超过 200K token
- 任务需要最深层推理（Max 思考模式）
- 需要使用 PreservedThinking 来管理多步编码计划
- 需要与 5V-Turbo 进行视觉协同
- 任务可能运行超过 8 小时，需要降级支持
- 分析超大型代码库（500K+ token）

### 各模型配置

```toml
# GLM-5 长时程配置
[drivers.openai_compatible.glm_5]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5"
compiler = "glm_5"
max_context_tokens = 200_000
long_horizon_enabled = true

# GLM-5.2 长时程配置（带 1M 上下文）
[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
max_context_tokens = 1_000_000
long_horizon_enabled = true
multimodal_enabled = true
# 注意：dual_thinking_enabled、preserved_thinking_enabled 是
# create_provider() 的 kwargs，不是 TOML 驱动字段。
# 上下文降级由 AutoCompactor 内部处理。

# 将长时程任务路由到相应模型
[routing]
long_horizon_driver = "openai_compatible.glm_52"  # 默认使用 GLM-5.2
```

---

## 配置

### LongHorizonConfig 字段

```python
from teragent.core.tap import LongHorizonConfig

config = LongHorizonConfig(
    max_duration_hours=4,              # 最大任务时长（默认：8）
    checkpoint_interval_minutes=15,    # 自动检查点间隔（默认：15）
    evaluation_interval_steps=10,      # 每 N 步自我评估（默认：10）
    evaluation_interval_minutes=30,    # 每 N 分钟自我评估（默认：30）
)
```

### agent.toml 配置

```toml
[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
long_horizon_enabled = true             # 长时程模式必需

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

## 分步使用

### 使用 GLM-5.2 的基本用法

```python
import asyncio
from teragent import create_provider, TAPRequest
from teragent.core.tap import LongHorizonConfig
from teragent.long_horizon import LongHorizonTaskManager

async def main():
    # 1. 创建带 1M 上下文的 GLM-5.2 提供者
    provider = create_provider(
        compiler="glm_52",
        adapter="openai_compatible",
        model="glm-5.2",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="GLM_API_KEY",
    )

    # 2. 配置长时程任务
    config = LongHorizonConfig(
        max_duration_hours=6,
        checkpoint_interval_minutes=20,
        evaluation_interval_steps=10,
    )

    # 3. 创建任务管理器
    manager = LongHorizonTaskManager(
        goal="Migrate the entire monolithic application to a microservices "
             "architecture. Include: (1) service boundary identification, "
             "(2) API gateway setup, (3) database per service pattern, "
             "(4) inter-service communication, (5) deployment configuration.",
        model_provider=provider,
        config=config,
    )

    # 4. 执行长时程任务
    result = await manager.execute_long_task()

    # 5. 查看结果
    print(f"Success: {result.success}")
    print(f"Total steps: {result.total_steps}")
    print(f"Duration: {result.total_elapsed_minutes:.1f} minutes")
    print(f"Sub-goals completed: {result.completed_sub_goals}/{result.total_sub_goals}")
    print(f"Strategy switches: {result.strategy_switches}")
    print(f"Summary: {result.final_summary}")

asyncio.run(main())
```

### 使用 TAPRequest 搭配长时程配置

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

# 当配置了 [routing].long_horizon_driver 时，
# ModelRouter 会自动将此请求路由到 GLM-5.2
```

### 在长时程任务中使用 PreservedThinking

```python
from teragent import create_provider
from teragent.core.tap import LongHorizonConfig
from teragent.long_horizon import LongHorizonTaskManager

provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    preserved_thinking_enabled=True,  # 编译器级 kwargs
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

# PreservedThinking 确保初始规划阶段的推理链
# 贯穿到所有实现步骤中
result = await manager.execute_long_task()
```

---

## 检查点管理与恢复

### 检查点的工作原理

每隔 `checkpoint_interval_minutes`，系统会自动保存一个检查点，包含：
- 已完成的子目标 ID
- 当前正在执行的子目标
- 已完成的步骤数和已用时间
- 策略切换次数
- PreservedThinking 追踪记录（仅限 GLM-5.2）
- 上下文降级状态（仅限 GLM-5.2）
- 用于恢复的任意状态数据

### 检查点存储

检查点以 JSON 文件形式存储在配置的目录中：

```
.teragent/checkpoints/
├── task_001/
│   ├── a1b2c3d4-...json    # 检查点 1
│   ├── e5f6g7h8-...json    # 检查点 2
│   └── ...
└── task_002/
    └── ...
```

### 手动检查点操作

```python
from teragent.long_horizon.checkpoint import CheckpointStore, Checkpoint

store = CheckpointStore(base_dir=".teragent/checkpoints")

# 列出某个任务的检查点
checkpoints = await store.list_checkpoints("task_001")

# 加载最新检查点
latest = await store.load_latest("task_001")
print(f"Phase: {latest.phase}")
print(f"Completed sub-goals: {latest.completed_sub_goals}")
print(f"Steps: {latest.steps_completed}")

# 加载特定检查点
checkpoint = await store.load("a1b2c3d4-...")

# 清理旧检查点（保留最近 5 个）
deleted = await store.cleanup("task_001", keep_last=5)
```

### 1M 上下文的检查点大小考量

使用 1M 上下文的 GLM-5.2 检查点可能明显更大：

| 上下文大小 | 检查点大致大小 |
|-----------|--------------|
| 200K token | 2-5 MB |
| 500K token | 5-15 MB |
| 1M token | 15-50 MB |

请确保在使用 1M 上下文运行长时间任务时有足够的磁盘空间。

---

## 自我评估与策略切换

### 自我评估

`SelfEvaluator` 定期注入评估提示，要求模型评估以下方面：

| 维度 | 评分范围 | 说明 |
|------|---------|------|
| 目标一致性 | 1-5 | 当前方向是否与原始目标一致？ |
| 输出质量 | 1-5 | 已完成工作的质量如何？ |
| 已识别瓶颈 | 文本 | 阻碍进展的因素是什么？ |
| 策略审查 | 文本 | 当前策略是否有效？ |
| 下一步计划 | 文本 | 接下来应该做什么？ |

### 策略切换

当检测到停滞时，模型可从以下策略中选择：

1. **分解（Decompose）** — 将当前子目标拆分为更小的步骤
2. **回溯（Backtrack）** — 返回到最后一个成功状态，尝试不同的路径
3. **跳过（Skip）** — 跳过卡住的子目标，先完成其他部分
4. **工具更换（Tool change）** — 尝试不同的技术方法或工具
5. **增量验证（Incremental validation）** — 验证每个小步骤以防止偏移
6. **重新规划（Replan）** — 重新评估整个目标分解

### 长时程任务中的双重思考

GLM-5.2 的双重思考模式可在长时程任务中加以利用：

```python
# 任务管理器可以根据子目标复杂度切换思考模式
request = TAPRequest(
    instruction="Evaluate the current progress and decide next steps",
    meta={
        "thinking_mode": "max",  # 评估步骤使用 Max 思考模式
    },
    long_horizon=LongHorizonConfig(
        max_duration_hours=6,
    ),
)
```

---

## 长时程任务的 1M 上下文

### 1M 上下文何时重要

长时程任务会随时间自然累积上下文。在 GLM-5 的 200K 限制下，分析大型代码库的任务可能在完成前耗尽上下文。GLM-5.2 的 1M 窗口解决了这一问题：

- **大型代码库分析** — 一次性加载整个代码库，逐步分析
- **累积上下文** — 累积发现、计划和实现细节，无需截断
- **PreservedThinking** — 在整个任务期间维护推理追踪记录

### 长时程任务中的上下文降级

对于运行 8 小时以上的任务，上下文降级至关重要：

```toml
[drivers.openai_compatible.glm_52]
# 上下文降级由 AutoCompactor 内部处理
```

降级行为：
1. 从 1M 上下文开始
2. 当内存使用率超过阈值时，降级到 200K
3. 现有上下文被压缩（保留系统提示、最近消息、工具定义）
4. 当内存压力缓解后，恢复到 1M

### 1M 上下文长时程任务的最佳实践

1. **预先加载完整代码库** — 在初始分析阶段充分利用 1M 窗口
2. **执行时使用 High 思考** — 将 Max 思考留给关键决策点
3. **启用 PreservedThinking** — 在整个过程中保持规划上下文
4. **设置较大的检查点间隔** — 1M 上下文的检查点更大，保存更慢
5. **密切监控内存** — 为上下文降级事件设置告警
6. **为降级做好计划** — 假设上下文可能在长时间任务中发生降级

---

## 8 小时以上任务的最佳实践

### 1. 编写清晰、具体的目标

```python
# 好的做法：清晰、具体、可衡量
goal = ("Migrate the monolithic User Service to a standalone microservice. "
        "Include: (1) Extract user domain models, (2) Create REST API with "
        "CRUD endpoints, (3) Implement JWT authentication, (4) Set up "
        "PostgreSQL database, (5) Add Docker configuration, (6) Write "
        "integration tests with >80% coverage.")

# 不好的做法：模糊、开放式
goal = "Make the user service better"
```

### 2. 设置适当的时间限制

| 任务复杂度 | 建议时长 | 建议模型 |
|-----------|---------|---------|
| 小型重构 | 1-2 小时 | GLM-5 |
| 功能开发 | 2-4 小时 | GLM-5 或 GLM-5.2 |
| 大型迁移 | 4-8 小时 | GLM-5.2 |
| 完整系统重写 | 8+ 小时 | GLM-5.2（带降级） |

### 3. 配置检查点频率

| 任务时长 | 上下文大小 | 建议检查点间隔 |
|---------|-----------|--------------|
| 1-2 小时 | 200K | 5 分钟 |
| 2-4 小时 | 200K | 10 分钟 |
| 4-8 小时 | 1M | 15 分钟 |
| 8+ 小时 | 1M | 20 分钟 |

### 4. 使用预算控制

```python
from teragent.reliability.budget import CrossModelCostTracker, MonthlyBudgetConfig

tracker = CrossModelCostTracker()
tracker.set_monthly_budget(MonthlyBudgetConfig(
    limit_cny=200.0,  # 为长时程任务设置支出上限
    auto_downgrade=True,
))
```

### 5. 定期监控进度

```python
report = tracker.get_report()
print(f"Phase: {report.current_phase}")
print(f"Progress: {report.completed_sub_goals}/{report.total_sub_goals}")
print(f"Elapsed: {report.elapsed_minutes:.1f} min")
print(f"Remaining: {report.estimated_remaining_minutes:.1f} min (est)")
```

### 6. 设置最大策略切换次数

通过设置 `max_strategy_switches`（默认值：5）来防止无限策略循环。如果达到此限制，任务将终止并附带已完成工作的摘要。

---

## 中断恢复

### 自动恢复

```python
from teragent.reliability.recovery import LongHorizonRecoveryManager
from teragent.long_horizon.checkpoint import CheckpointStore

store = CheckpointStore(base_dir=".teragent/checkpoints")
recovery_mgr = LongHorizonRecoveryManager(
    checkpoint_store=store,
    max_reconnection_attempts=5,
    reconnection_base_delay=2.0,
)

# 尝试从最新检查点恢复
success = await recovery_mgr.recover_from_checkpoint(task_manager)

if not success:
    # 检查是否应降级到标准（非长时程）模式
    if recovery_mgr.should_downgrade_to_standard(
        recovery_attempts=3,
        elapsed_time=1800,  # 已用 30 分钟
    ):
        print("Too many failures; switching to standard mode")
```

### 处理上下文降级恢复

当 GLM-5.2 的上下文在长时程任务中发生降级时：

1. **检查点捕获降级状态** — 检查点包含当前上下文模式
2. **恢复时保留降级模式** — 任务以 200K 上下文恢复
3. **手动恢复到 1M** — 任务完成或内存压力缓解后，可以 1M 重新启动

```python
# 恢复后，检查上下文模式
checkpoint = await store.load_latest("task_001")
if hasattr(checkpoint, 'context_mode') and checkpoint.context_mode == "200K":
    print("Task was interrupted during degraded mode")
    print("Consider restarting with fresh 1M context for better results")
```

### 跨会话恢复

跨越进程重启的长时程任务：

```python
# 进程重启时：
store = CheckpointStore(base_dir=".teragent/checkpoints")
recovery = LongHorizonRecoveryManager(checkpoint_store=store)

# 从上次检查点恢复
success = await recovery.recover_from_checkpoint(task_manager)
if success:
    result = await task_manager.execute_long_task()  # 从检查点恢复执行
```

---

## 监控与调试

### 进度追踪

```python
from teragent.long_horizon.progress import ProgressTracker

tracker = ProgressTracker(task_id="task_1", goal="My goal")

# 执行过程中：
report = tracker.get_report()
# report.current_phase: "planning" | "executing" | "evaluating" | "stagnant"
# report.completed_sub_goals / report.total_sub_goals
# report.steps_completed
# report.elapsed_minutes
# report.estimated_remaining_minutes
# report.strategy_switches
```

### 上下文降级监控

```python
from teragent.context import ContextWindow

# The ContextWindow tracks utilization
utilization = context_window.usage_ratio()
print(f"Context utilization: {utilization:.1%}")
if utilization > 0.9:
    print("⚠️ Context utilization high, consider enabling auto-compaction")
```

### 熔断器集成

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

## 已知限制

1. **上下文窗口限制** — GLM-5 的上下文为 200K；需要更多上下文的任务将自动路由到 GLM-5.2（如果已配置）。如果 GLM-5.2 不可用，上下文压缩可能会影响连贯性。

2. **API 速率限制** — 长时程任务会发出大量 API 调用。请确保您的 API 套餐支持预期的请求量。配置 `RateLimitHandler` 以实现自动退避。

3. **检查点大小** — 使用 1M 上下文的 GLM-5.2 检查点可能达到 15-50 MB。请确保有足够的磁盘空间和网络带宽用于检查点操作。

4. **策略切换限制** — 最大策略切换次数（默认值：5）可防止无限循环，但可能会终止确实需要更多适应的任务。

5. **无跨会话记忆** — 每个长时程任务从头开始。如果需要来自先前任务的上下文，请将其包含在目标描述中。PreservedThinking 在同一会话内有效，但不跨会话。

6. **自我评估成本** — 自我评估使用额外的 API 调用。对于非常频繁的评估间隔，这可能使成本增加 10-20%。

7. **单模型执行** — 长时程任务目前在单个模型上执行（GLM-5 或 GLM-5.2）。尚不支持多模型长时程编排。

8. **桌面操作** — 长时程任务不能使用桌面操作（M3 功能）。请单独使用 M3 处理视觉任务。

9. **1M 上下文稳定性** — 在完整 1M 上下文下，GLM-5.2 可能会经历更高的延迟和内存压力。请启用上下文降级作为安全网。

10. **PreservedThinking 开销** — PreservedThinking 每条保留追踪记录增加约 2-5K token。对于具有许多追踪记录的非常长的任务，这可能会消耗大量上下文。

---

*本指南是 TerAgent 文档的一部分。完整的四模型适配指南请参阅[适配指南](adaptation_guide.md)。有关 GLM-5.2 特定功能，请参阅 [GLM-5.2 指南](glm_52_guide.md)。*
