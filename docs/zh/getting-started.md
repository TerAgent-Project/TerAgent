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
    compiler="glm_5",
    adapter="openai_compatible",
    model="glm-5",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
)

# 方式 2：从配置文件创建
full_config = teragent.load_full_config()
drivers = full_config["drivers"]
provider = teragent.create_provider_from_config(drivers["openai_compatible.glm_5"])

# 方式 3：从 DriverConfig 对象创建
from teragent.config import DriverConfig
driver_cfg = DriverConfig(
    adapter="openai_compatible",
    identity="glm_5",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
    model="glm-5",
    compiler="glm_5",
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

### 编译器

| Compiler | 优化策略 | 目标模型 |
|----------|---------|----------|
| `default` | 标准聊天消息 | 通用 OpenAI 协议模型 |
| `glm` | 近因效应（关键指令置末） | GLM 系列（智谱 AI） |
| `glm_5` | 近因效应 + 长时任务 + 自我评估 | GLM-5 |
| `glm_52` | 1M 上下文 + 双思考模式（High/Max）+ PreservedThinking + 5V-Turbo 协调 | GLM-5.2 |
| `glm_5v_turbo` | GLM-5V-Turbo 视觉分析 | GLM-5V-Turbo（视觉模型） |
| `anthropic` | XML 标签结构化 + Mode B | Claude 系列 |
| `deepseek` | 极简编译 | DeepSeek V3 模型 |
| `deepseek_v4` | 缓存感知布局 + 思考模式 + 1M 上下文优化 | DeepSeek V4-Flash/Pro（通过 `compiler_variant` 控制变体） |
| `minimax_m3` | MSA 全文注入 + 多模态 + 桌面上下文 | MiniMax M3 |

> **注意：** `deepseek_v4_flash` 和 `deepseek_v4_pro` **不是**独立的编译器 —— 它们是 `deepseek_v4` 的变体，通过 `compiler_variant` 参数（`"flash"` 或 `"pro"`）控制。

### Adapters

| Adapter | 协议 | 说明 |
|---------|------|------|
| `openai_compatible` | OpenAI `/chat/completions` + SSE | 适用于 GLM、DeepSeek、OpenRouter 等 |
| `anthropic_native` | Anthropic `/messages` + Anthropic SSE | 直连 Anthropic API |
| `glm_native` | 智谱 AI 原生 API | GLM-5/5.2 智谱 AI 特定优化 |
| `minimax_native` | MiniMax 原生 API + 速率限制追踪 | MiniMax M3 多模态/桌面自动化 |
| `mock` | 无 HTTP 调用 | 用于测试 |

### 有效组合

| Compiler | Adapter | 目标 | 提示策略 |
|----------|---------|------|----------|
| `default` | `openai_compatible` | 通用 OpenAI 协议模型 | 标准聊天消息 |
| `glm` | `openai_compatible` | GLM 系列（智谱 AI） | 近因效应优化 |
| `glm_5` | `openai_compatible` | GLM-5（长时任务） | 深度推理 + 长时任务支持 |
| `glm_5` | `glm_native` | GLM-5 通过智谱 AI 原生 API | 深度推理 + 原生优化 |
| `glm_52` | `openai_compatible` | GLM-5.2（1M + 双思考） | 1M 上下文 + PreservedThinking + 5V-Turbo 协调 |
| `glm_52` | `glm_native` | GLM-5.2 通过智谱 AI 原生 API | 1M 上下文 + 原生优化 |
| `glm_5v_turbo` | `openai_compatible` | GLM-5V-Turbo（视觉） | 视觉分析编译 |
| `anthropic` | `openai_compatible` | 通过 OpenRouter 的 Claude | XML 标签 + 近因效应 |
| `anthropic` | `anthropic_native` | 通过 Anthropic API 的 Claude | XML 标签 + system/user 分离 (Mode B) |
| `deepseek` | `openai_compatible` | DeepSeek V3 模型 | 极简编译 |
| `deepseek_v4` | `openai_compatible` | DeepSeek V4-Flash/Pro | 缓存感知布局 + 思考模式（变体：flash/pro） |
| `minimax_m3` | `openai_compatible` | MiniMax M3（文本） | MSA 全文注入 |
| `minimax_m3` | `minimax_native` | MiniMax M3（多模态/桌面） | 原生多模态 + 速率限制追踪 |
| `default` | `mock` | 测试 | 无 HTTP 调用 |

## 下一步

- [架构指南](architecture.md) — 深入了解设计决策
- [安全指南](security.md) — 权限系统、沙箱、文件写入
- [配置指南](configuration.md) — agent.toml 与类型化配置
- [API 参考](api-reference.md) — 完整模块参考
- [自强化学习指南](self-rl.md) — TAP 追踪与 DPO 对生成
- [四模型适配指南](adaptation_guide.md) — DeepSeek V4、MiniMax M3、GLM-5、GLM-5.2 配置与最佳实践
- [GLM-5.2 使用指南](../en/glm_52_guide.md)（英文） — 1M 上下文、双思考模式、PreservedThinking、5V-Turbo 视觉协调
- [多模态指南](../en/multimodal_guide.md)（英文） — MiniMax M3 图像、视频和桌面操作
- [长时任务指南](../en/long_horizon_guide.md)（英文） — GLM-5/5.2 8小时+自主任务
