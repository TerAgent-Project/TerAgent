# TerAgent 快速上手

本指南将带你完成 TerAgent 的安装、配置和使用。

## 安装

### 基础安装

```bash
pip install teragent
```

### 可选依赖

TerAgent 使用延迟导入 — `import teragent` 总是会成功。可选组件仅在实际使用时才会抛出 `ImportError`。

```bash
pip install teragent[ast]      # CodeIndexer — tree-sitter AST 解析
pip install teragent[graph]    # ReferenceGraph — networkx 依赖分析
pip install teragent[vector]   # VectorIndexer — LanceDB 语义搜索
pip install teragent[all]      # 所有可选依赖
pip install teragent[dev]      # 开发工具 (pytest, ruff, mypy)
```

**环境要求：** Python 3.10+。在 Python 3.10 上，`tomli` 会被自动安装以支持 TOML 配置。

## 快速上手

### 1. 创建 Provider

`ModelProvider` 由 Compiler（提示策略）和 Adapter（HTTP 协议）组合而成：

```python
import teragent

# 方式 1：工厂函数（推荐）
provider = teragent.create_provider(
    compiler="glm",
    adapter="openai_compatible",
    model="glm-5.1",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
)

# 方式 2：从配置文件创建
full_config = teragent.load_full_config()
drivers = full_config["drivers"]
provider = teragent.create_provider_from_config(drivers["openai_compatible.glm"])

# 方式 3：从 DriverConfig 对象创建
from teragent.config import DriverConfig
driver_cfg = DriverConfig(
    adapter="openai_compatible",
    identity="glm",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
    model="glm-5.1",
    compiler="glm",
)
provider = teragent.create_provider(**driver_cfg.to_create_provider_kwargs())
```

### 2. 执行 TAP 请求

```python
response = await provider.execute_tap(teragent.TAPRequest(
    meta={"task_id": "1.1", "intent": "code_generation"},
    instruction="实现用户登录模块",
    constraints=["Python 3.10+"],
    output_format_hint="<file path='...'>完整代码</file>",
))

print(response.raw_text)
print(f"Tokens: {response.total_tokens}")
```

### 3. 提取文件 & 运行检查

```python
# 从模型响应中提取文件
files = teragent.extract_files_from_response(response.raw_text, task_id="1.1")

# 运行确定性代码质量检查
task_list = [teragent.TaskInfo(id="1.1", title="登录模块", status="completed")]
report, data = teragent.run_deterministic_checks("/project", task_list)
```

### 4. 构建完整 Agent

```python
from teragent import AgentLoop, ModelProvider, ToolRegistry
from teragent.config import AgentLoopConfig
from teragent.reliability import CircuitBreakerManager, StepBudget
from teragent.security import EnhancedPermissionManager
from teragent.context import ContextWindow, AutoCompactor
from teragent.intent import IntentClassifier
from teragent.streaming import StreamingToolExecutor

# 构建包含所有横切关注点的 Agent 循环
loop = AgentLoop(
    model=provider,
    tool_registry=my_tool_registry,
    config=AgentLoopConfig(),
    circuit_breaker=CircuitBreakerManager(),
    step_budget=StepBudget(max_steps=50),
    permission_manager=EnhancedPermissionManager(),
    context_window=ContextWindow(model_token_limit=128_000),
    auto_compactor=AutoCompactor(context_window=..., model=provider),
    intent_classifier=IntentClassifier(provider),
    streaming_executor=StreamingToolExecutor(my_tool_registry),
)

# 运行 Agent
messages = await loop.run("帮我用 Python 构建一个贪吃蛇游戏")
```

### 5. 自强化学习数据采集（DPO 对）

```python
# 附加追踪器，自动记录所有 TAP 调用
tracer = teragent.TAPTracer(trace_dir="/project/.agent/traces")
provider.set_tracer(tracer)

# ... 执行 TAP 调用 ...

# 记录检查清单结果（确定性 PASS/FAIL 标签）
await tracer.record_checklist("1.1", checklist_data)

# 导出 DPO 偏好对用于微调
pairs = tracer.export_dpo_pairs()
tracer.export_dpo_pairs_jsonl()  # 写入 JSONL 文件
```

## 可用的编译器和适配器

### Compilers

| Compiler | 优化策略 | 目标模型 |
|----------|---------|---------|
| `default` | 标准聊天消息 | 通用 OpenAI 协议模型 |
| `glm` | 近因效应（关键指令置末） | GLM 系列（智谱 AI） |
| `anthropic` | XML 标签结构化 + Mode B | Claude 系列 |
| `deepseek` | 极简编译 | DeepSeek 模型 |

### Adapters

| Adapter | 协议 | 说明 |
|---------|------|------|
| `openai_compatible` | OpenAI `/chat/completions` + SSE | 适用于 GLM、DeepSeek、OpenRouter 等 |
| `anthropic_native` | Anthropic `/messages` + Anthropic SSE | 直连 Anthropic API |
| `mock` | 无 HTTP 调用 | 用于测试 |

### Valid Combinations

| Compiler | Adapter | 目标 | 提示策略 |
|----------|---------|------|---------|
| `default` | `openai_compatible` | 通用 OpenAI 协议模型 | 标准聊天消息 |
| `glm` | `openai_compatible` | GLM 系列（智谱 AI） | 近因效应优化 |
| `anthropic` | `openai_compatible` | 通过 OpenRouter 的 Claude | XML 标签 + 近因效应 |
| `anthropic` | `anthropic_native` | 通过 Anthropic API 的 Claude | XML 标签 + system/user 分离 (Mode B) |
| `deepseek` | `openai_compatible` | DeepSeek 模型 | 极简编译 |
| `default` | `mock` | 测试 | 无 HTTP 调用 |

## 下一步

- [架构指南](architecture.md) — 深入了解设计决策
- [安全指南](security.md) — 权限系统、沙箱、文件写入
- [配置指南](configuration.md) — agent.toml 与类型化配置
- [API 参考](api-reference.md) — 完整模块参考
- [自强化学习指南](self-rl.md) — TAP 追踪与 DPO 对生成
