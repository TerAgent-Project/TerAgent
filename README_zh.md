
# TerAgent

**终端 AI Agent 库 — TAP IR + 模型专属编译**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Version: 0.1.2](https://img.shields.io/badge/Version-0.1.2-blue.svg)](https://github.com/teragent/teragent)

[English](README.md) | **中文**

---

TerAgent 是一个用于构建生产级 AI Agent 系统的 Python 库，核心采用**编译器-适配器正交架构**。它引入了 **TAP IR**（Tool-Augmented Prompt Intermediate Representation）—— 一种模型无关的内存中间表示，将 *"问什么"* 与 *"怎么格式化"* 解耦，实现 Prompt 编译器与协议适配器的正交组合。

**9 编译器** × **5 适配器** = **45 种模型+协议组合（含测试适配器）**，每种组合都针对特定配对做了优化。

现已支持 **DeepSeek V4**、**MiniMax M3**、**GLM-5**、**GLM-5.2** 深度适配——智能多模型路由、长程自主任务、原生多模态理解和桌面自动化。

---

## 目录

- [文档](docs/) — 完整文档、指南和 API 参考
- [为什么选择 TerAgent](#为什么选择-teragent)
- [四模型深度适配](#四模型深度适配)
- [安装](#安装)
- [快速上手](#快速上手)
  - [单模型](#单模型快速上手)
  - [多模型](#多模型快速上手)
- [架构](#架构)
  - [TAP IR](#tap-ir)
  - [编译器 × 适配器组合](#编译器--适配器组合)
  - [数据流](#数据流)
- [模块参考](#模块参考)
  - [核心层 (TAP IR + 编译器 + 适配器)](#核心层-tap-ir--编译器--适配器)
  - [路由与 Pipeline（多模型）](#路由与-pipeline多模型)
  - [长程任务](#长程任务)
  - [预算与成本追踪](#预算与成本追踪)
  - [基准测试](#基准测试)
  - [Pipeline 原语](#pipeline-原语)
  - [AgentLoop（核心编排）](#agentloop核心编排)
  - [流式执行](#流式执行)
  - [安全体系](#安全体系)
  - [可靠性系统](#可靠性系统)
  - [上下文管理](#上下文管理)
  - [协调层（子 Agent）](#协调层子-agent)
  - [意图分类](#意图分类)
  - [Hook 系统](#hook-系统)
  - [会话持久化](#会话持久化)
  - [自强化学习数据宪章](#自强化学习数据宪章)
  - [配置系统](#配置系统)
  - [事件总线](#事件总线)
- [配置](#配置)
- [构建方式](#构建方式)
- [开发](#开发)
- [许可证](#许可证)

---

📖 **相关文档：**
- [四模型评估报告](docs/EVALUATION_FOUR_MODELS.md) — DeepSeek V4、MiniMax M3、GLM-5、GLM-5.2 评估结果
- [GLM-5.2 稳定性报告](docs/glm_52_stability_report.md) — GLM-5.2 生产稳定性验证
- [昇腾部署指南](docs/deployment_guide_ascend.md) — 在华为昇腾 NPU 上部署 TerAgent

---

## 为什么选择 TerAgent

| 痛点 | TerAgent 的解决方式 |
|------|-------------------|
| 不同模型的 Prompt 格式各异（GLM、Claude、DeepSeek…） | **编译器** 将 TAP IR 编译为模型专属 Prompt |
| API 协议各异（OpenAI、Anthropic native…） | **适配器** 处理协议级别的 HTTP I/O |
| 新增模型需要同时修改 Prompt 格式和 API 调用 | **正交组合**：只需添加编译器或适配器，而非两者 |
| 缺乏结构化的 Agent 交互记录用于自我改进 | **TAPTracer** 记录每次请求→响应对，支持 DPO 对生成 |
| 安全在大多数 Agent 框架中是事后补丁 | **7 层权限解析**、**6 层命令防御**、**2PC 文件写入**、**3 级沙箱降级** |
| 缺少可靠性——Agent 在无限循环中浪费 Token | **4 个熔断器**、流式重试+批量降级、上下文自动压缩 |

---

## 四模型深度适配

TerAgent 现已支持四款国产领先 AI 模型的深度适配，通过智能路由自动为每个任务选择最优模型：

| 模型 | 定位 | 核心能力 | 上下文窗口 |
|------|------|---------|-----------|
| **DeepSeek V4-Flash** | 轻量任务 | 快速响应、低成本、缓存感知 | 1M tokens |
| **DeepSeek V4-Pro** | 复杂推理 | 深度思考模式、缓存优化 | 1M tokens |
| **MiniMax M3** | 多模态 & 桌面 | 图像/视频理解、桌面自动化、MSA | 1M tokens |
| **GLM-5** | 长程 & 审查 | 8 小时自主任务、自评估、策略切换 | 200K tokens |
| **GLM-5.2** | 超长上下文 & 双思考 | 1M 上下文、High/Max 双思考、PreservedThinking、5V-Turbo 视觉协调 | 1M tokens |

### 特性矩阵

| 特性 | V4-Flash | V4-Pro | M3 | GLM-5 | GLM-5.2 |
|------|----------|--------|----|---------|---------|
| 快速代码生成 | ✓✓✓ | ✓✓ | ✓ | ✓✓ | ✓✓ |
| 深度推理 | — | ✓✓✓ | ✓ | ✓✓✓ | ✓✓✓ |
| 双思考模式 | — | — | — | — | ✓✓✓ |
| PreservedThinking | — | — | — | — | ✓✓✓ |
| 多模态（图像） | — | — | ✓✓✓ | — | ✓（5V-Turbo） |
| 视频理解 | — | — | ✓✓✓ | — | — |
| 桌面自动化 | — | — | ✓✓✓ | — | — |
| 长程任务 | — | — | — | ✓✓✓ | ✓✓✓ |
| 视觉→代码工作流 | — | — | — | — | ✓✓✓ |
| 缓存感知定价 | ✓✓✓ | ✓✓✓ | — | — | — |
| 1M 上下文窗口 | ✓ | ✓ | ✓ | — | ✓ |
| 成本效率 | ✓✓✓ | ✓ | ✓ | ✓✓ | ✓✓ |

### 智能路由（6 步）

`ModelRouter` 通过 6 步决策流自动选择最优模型：

1. **多模态检测** → 视觉/视频内容路由到 M3 或 GLM-5.2 + 5V-Turbo
2. **上下文长度** → 排除上下文不足的模型（>200K → V4/M3/GLM-5.2）
3. **长程任务** → 长时间任务路由到 GLM-5 或 GLM-5.2
4. **意图匹配** → 默认路由表（design→V4-Pro, plan→GLM-5.2, execute→GLM-5.2, review→V4-Pro, chat→V4-Flash）
5. **成本评估** → 月度预算受限时降级
6. **降级容灾** → 主模型不可用时自动回退

### Pipeline 配置方案

运行时在命名 Pipeline 配置之间切换：

| 方案 | Design | Plan | Execute | Review | 适用场景 |
|------|--------|------|---------|--------|---------|
| `default` | V4-Pro | GLM-5.2 | GLM-5.2 | V4-Pro | 生产环境 |
| `budget` | V4-Flash | V4-Flash | V4-Flash | V4-Flash | 开发调试 |
| `multimodal` | M3 | M3 | M3 | M3 | 视觉任务 |
| `deep_thinking` | GLM-5.2 Max | GLM-5.2 Max | GLM-5.2 Max | GLM-5.2 Max | 复杂推理 |

### 文档

- 📖 [四模型适配指南](docs/zh/adaptation_guide.md) — 配置、迁移、最佳实践
- 📖 [GLM-5.2 使用指南](docs/en/glm_52_guide.md)（英文）— 1M 上下文、双思考、PreservedThinking、5V-Turbo
- 📖 [长程任务指南](docs/en/long_horizon_guide.md)（英文）— 8 小时自主任务
- 📖 [多模态指南](docs/en/multimodal_guide.md)（英文）— 图像、视频、桌面操作
- 📖 [API 参考](docs/zh/api-reference.md) — 完整 API 文档
- 📖 [配置手册](docs/zh/configuration.md) — 完整 agent.toml 参考
- 📖 [四模型评估报告](docs/EVALUATION_FOUR_MODELS.md) — 综合基准测试结果
- 📖 [GLM-5.2 稳定性报告](docs/glm_52_stability_report.md) — 生产稳定性验证
- 📖 [昇腾部署指南](docs/deployment_guide_ascend.md) — 在华为昇腾 NPU 上部署

---

## 安装

```bash
pip install teragent
```

### 可选依赖

```bash
pip install teragent[ast]      # CodeIndexer — tree-sitter AST 解析
pip install teragent[graph]    # ReferenceGraph — networkx 依赖图分析
pip install teragent[vector]   # VectorIndexer — LanceDB 语义搜索
pip install teragent[all]      # 安装全部可选依赖
pip install teragent[dev]      # 开发工具 (pytest, ruff, mypy)
```

**环境要求：** Python 3.10+。在 Python 3.10 上，`tomli` 会自动安装以支持 TOML 配置。

**类型存根：** TerAgent 包含 `.pyi` 类型存根文件和 `py.typed` 标记，符合 PEP 561 规范 — IDE 自动补全和 mypy 类型检查开箱即用。

可选组件采用懒导入——`import teragent` 始终成功，只有在实际使用未安装的可选组件时才会抛出 `ImportError`。

---

## 快速上手

### 单模型快速上手

### 1. 创建 Provider

```python
import teragent

# 方式一：工厂函数（推荐）
provider = teragent.create_provider(
    compiler="glm_5",
    adapter="openai_compatible",
    model="glm-5",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
)

# 方式二：从配置文件加载
full_config = teragent.load_full_config()
drivers = full_config["drivers"]
provider = teragent.create_provider_from_config(drivers["openai_compatible.glm_5"])

# 方式三：从 DriverConfig 对象创建
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

# 方式四：异步上下文管理器（自动清理资源）
async with teragent.create_provider(
    compiler="glm_5", adapter="openai_compatible", model="glm-5",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
) as provider:
    response = await provider.execute_tap(teragent.TAPRequest(...))
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
print(f"Token 用量: {response.total_tokens}")
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
    auto_compactor=AutoCompactor(
        context_window=ContextWindow(model_token_limit=128_000),
        model=provider,
    ),
    intent_classifier=IntentClassifier(provider),
    streaming_executor=StreamingToolExecutor(my_tool_registry),
)

# 运行 Agent
messages = await loop.run("帮我用 Python 写一个贪吃蛇游戏")
```

### 5. 自强化学习数据采集（DPO 对）

```python
# 附加 Tracer 自动记录所有 TAP 调用
tracer = teragent.TAPTracer(trace_dir="/project/.agent/traces")
provider.set_tracer(tracer)

# ... 执行 TAP 调用 ...

# 记录 Checklist 结果（确定性 PASS/FAIL 标签）
await tracer.record_checklist("1.1", checklist_data)

# 导出 DPO 偏好对用于微调
pairs = tracer.export_dpo_pairs()
tracer.export_dpo_pairs_jsonl()  # 写入 JSONL 文件
```

---

### 多模型快速上手

配置四款模型实现智能路由：

```toml
# agent.toml
[drivers.openai_compatible.deepseek_v4_flash]
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-v4-flash"
compiler = "deepseek_v4"
compiler_variant = "flash"

[drivers.openai_compatible.deepseek_v4_pro]
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-v4-pro"
compiler = "deepseek_v4"
compiler_variant = "pro"

[drivers.openai_compatible.minimax_m3]
base_url = "https://api.minimaxi.com/v1"
api_key_env = "MINIMAX_API_KEY"
model = "minimax-m3"
compiler = "minimax_m3"

[drivers.openai_compatible.glm_5]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5"
compiler = "glm_5"

[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"

[execution.pipeline]
design_driver = "openai_compatible.deepseek_v4_pro"
plan_driver = "openai_compatible.deepseek_v4_pro"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.glm_52"

[routing]
multimodal_driver = "openai_compatible.minimax_m3"
desktop_driver = "openai_compatible.minimax_m3"
long_horizon_driver = "openai_compatible.glm_5"
vision_code_driver = "openai_compatible.glm_52"

[routing.monthly_budget]
limit_cny = 500.0
warning_threshold = 0.8
auto_downgrade = true
```

```python
import teragent
from teragent.router import ModelRouter, RoutingTable

# 加载多模型配置
config = teragent.load_full_config()

# ModelRouter 自动选择最优模型
router = ModelRouter(
    available_providers={...},
    routing_table=RoutingTable(),
)

# 路由 TAP 请求 — 多模态内容自动路由到 M3
request = teragent.TAPRequest(
    instruction="分析这张截图",
    multimodal_context=[...],
)
decision = router.route(request)
# decision.selected_driver → "openai_compatible.minimax_m3"
```

📖 完整配置和迁移说明请参阅[四模型适配指南](docs/zh/adaptation_guide.md)。

---

## 架构

### TAP IR

TAP（TerAgent Protocol）是一种内存中间表示——类似 LLVM IR，但面向 LLM Prompt。它**不是**线缆协议。

```
┌─────────────────────────────────────────────────────────────────┐
│                        TAP IR                                   │
│                                                                 │
│  TAPRequest                          TAPResponse                │
│  ┌──────────────────────┐            ┌───────────────────┐     │
│  │ meta: dict           │            │ raw_text: str     │     │
│  │ context: dict        │            │ usage: dict       │     │
│  │ instruction: str     │            └───────────────────┘     │
│  │ constraints: list    │                                      │
│  │ output_format_hint   │                                      │
│  └──────────────────────┘                                      │
│           │                                                     │
│           ▼                                                     │
│  CompiledPrompt（两种互斥模式）                                  │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ 模式 A: messages 列表                                │      │
│  │   [{role, content}, ...]    ← OpenAI / GLM / DeepSeek│      │
│  │                                                      │      │
│  │ 模式 B: system_prompt + user_message                 │      │
│  │   system + user 分离       ← Anthropic native        │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

**TAPRequest 字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `meta` | `dict` | 任务元数据，如 `{"task_id": "1.1", "intent": "code_generation"}` |
| `context` | `dict` | 参考材料，如 `{"design": "...", "plan": "...", "dependency_report": "..."}` |
| `instruction` | `str` | 核心指令 / 用户请求 |
| `constraints` | `list[str]` | 输出必须满足的硬约束 |
| `output_format_hint` | `str` | 期望的输出格式描述 |

**TAPRequest 深度适配扩展字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `thinking_mode` | `Optional[Literal["deep", "quick", "auto"]]` | 推理深度控制（`"auto"`由编译器决定，`"deep"`完整CoT，`"quick"`跳过CoT） |
| `multimodal_context` | `Optional[list[MultimodalContent]]` | 多模态输入（图像/视频，用于 M3 和 GLM-5V-Turbo） |
| `long_horizon` | `Optional[LongHorizonConfig]` | 长程自主任务配置（GLM-5/5.2） |
| `cache_preference` | `str` | 缓存偏好（DeepSeek V4: `auto` / `aggressive`） |

**TAPResponse 字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `raw_text` | `str \| None` | 模型原始文本输出 |
| `usage` | `dict` | Token 用量，如 `{"prompt_tokens": int, "completion_tokens": int}` |
| `tool_calls` | `list[dict]` | API 响应中的结构化工具调用 |
| `finish_reason` | `str` | 停止原因（`"stop"`、`"length"` 等） |

**TAPResponse 深度适配扩展字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `cache_hit_tokens` | `int` | 缓存命中 Token 数（DeepSeek V4） |
| `thinking_content` | `Optional[str]` | 思考过程内容（GLM-5.2 / DeepSeek V4-Pro） |
| `long_horizon_status` | `Optional[LongHorizonStatus]` | 长程任务状态 |

### 编译器 × 适配器组合

| 编译器 | 适配器 | 目标模型 | Prompt 策略 |
|--------|--------|---------|-------------|
| `default` | `openai_compatible` | 通用 OpenAI 协议模型 | 标准聊天消息 |
| `glm` | `openai_compatible` | GLM 系列（智谱 AI） | 近因效应优化——关键指令置于末尾 |
| `glm_5` | `openai_compatible` | GLM-5（长时任务） | 深度推理 + 长时任务支持 |
| `glm_5` | `glm_native` | GLM-5（原生特性） | 原生 thinking + cache + 异步 chat |
| `glm_52` | `openai_compatible` | GLM-5.2（1M 上下文、双思考模式） | 1M 上下文优化 + High/Max 双思考 + PreservedThinking |
| `glm_52` | `glm_native` | GLM-5.2（原生特性） | 原生 thinking/cache/async + reasoning_content |
| `glm_5v_turbo` | `openai_compatible` | GLM-5V-Turbo（视觉） | 视觉分析 + 多模态 prompt |
| `glm_5v_turbo` | `glm_native` | GLM-5V-Turbo（原生视觉） | 原生视觉 API + 图像理解 |
| `anthropic` | `openai_compatible` | Claude（经 OpenRouter） | XML 标签结构化 + 近因效应 |
| `anthropic` | `anthropic_native` | Claude（Anthropic API） | XML 标签 + system/user 分离（模式 B） |
| `deepseek` | `openai_compatible` | DeepSeek V3 模型 | 极简编译 |
| `deepseek_v4` | `openai_compatible` | DeepSeek V4-Flash/Pro | 缓存感知布局 + 思考模式 + 1M 上下文 |
| `minimax_m3` | `openai_compatible` | MiniMax M3（文本） | MSA 全文注入 |
| `minimax_m3` | `minimax_native` | MiniMax M3（多模态/桌面） | 原生多模态 + 视频输入 + 桌面操作 + 速率限制追踪 |
| `default` | `mock` | 测试 | 无 HTTP 调用 |

新增模型只需编写新的编译器类。新增协议只需编写新的适配器类。二者通过 `ModelProvider` 正交组合。

### 数据流

```
用户输入
    │
    ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  TAPRequest  │───▶│     编译器      │───▶│  CompiledPrompt  │
│  (IR)        │    │  (编译 IR)      │    │  (模型专属)      │
└──────────────┘    └─────────────────┘    └────────┬─────────┘
                                                     │
                                                     ▼
                                            ┌──────────────────┐
                                            │     适配器       │
                                            │  (HTTP I/O)      │
                                            └────────┬─────────┘
                                                     │
                                                     ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  TAPResponse │◀───│  ModelProvider  │◀───│   模型 API       │
│  (IR)        │    │ (组合编译器+    │    │ (GLM/Claude/…)   │
│              │    │  适配器)        │    │                  │
└──────────────┘    └─────────────────┘    └──────────────────┘
```

---

## 跨平台兼容性

TerAgent 支持 **Windows**、**macOS** 和 **Linux**，并针对各平台做了专属适配：

| 特性 | Windows | macOS | Linux |
|------|---------|-------|-------|
| 沙箱 Level 0 | ✅ `CREATE_NEW_PROCESS_GROUP` | ✅ `start_new_session` | ✅ `start_new_session` + `preexec_fn` |
| 沙箱 Level 1 (Docker) | ✅ `ContainerUser` | ✅ uid/gid 映射 | ✅ uid/gid 映射 |
| 沙箱 Level 2 (Firecracker) | ❌ 需要 KVM | ❌ 需要 KVM | ✅ 完整支持 |
| 进程树杀灭 | ✅ `taskkill /F /T` | ✅ `os.killpg()` | ✅ `os.killpg()` |
| Windows 危险命令 | ✅ 16 种模式拦截 | N/A | N/A |
| Windows 系统路径保护 | ✅ `C:\Windows`、`Program Files` 等 | N/A | N/A |
| 剪贴板 (X11) | N/A | ✅ `pbcopy`/`pbpaste` | ✅ `xclip` |
| 剪贴板 (Wayland) | N/A | N/A | ✅ `wl-copy`/`wl-paste` |
| 截图 | ✅ PIL ImageGrab | ✅ PIL ImageGrab | ✅ `mss` 优先 |
| 屏幕尺寸 | ✅ `ctypes` 回退 | ✅ `AppKit` 回退 | ✅ `mss`/`pyautogui` |
| 配置搜索 | `%APPDATA%\teragent\` | `~/Library/Application Support/teragent/` | `~/.config/teragent/` (XDG) |
| `.env` 搜索 | CWD → `~/.env` → 项目 | CWD → `~/.env` → 项目 | CWD → `~/.env` → 项目 |
| 原子文件写入 | ✅ 3 步重命名 + 备份 | ✅ 原子 `os.replace()` | ✅ 原子 `os.replace()` |
| `shlex` 解析 | ✅ `posix=False` | ✅ `posix=True` | ✅ `posix=True` |
| HTTP/2 | ✅ 可配置（默认关闭） | ✅ 可配置 | ✅ 可配置 |
| SSL 证书验证 | ✅ 自定义 CA 支持 | ✅ 自定义 CA 支持 | ✅ 自定义 CA 支持 |

### 关键跨平台特性

- **进程管理**：Windows 使用 `CREATE_NEW_PROCESS_GROUP` + `taskkill /F /T`；Unix 使用 `start_new_session` + `os.killpg()`
- **命令安全**：平台专属危险命令黑名单 — Unix（`sudo`、`rm -rf /`、`/etc/`）和 Windows（`format C:`、`reg delete`、`powershell -enc`、`taskkill`）
- **文件写入**：POSIX 使用原子 `os.replace()`；Windows NTFS 使用重命名-重命名-删除 3 步模式，含备份/回滚
- **桌面剪贴板**：Linux 自动检测 Wayland vs X11；macOS 使用 `pbcopy`；Windows 使用 `clip`
- **配置路径**：搜索平台标准目录（XDG、AppData、Application Support），以及 CWD 和项目根目录

---

## 模块参考

### 核心层 (TAP IR + 编译器 + 适配器)

| 模块 | 关键类 | 说明 |
|------|--------|------|
| `teragent.core.tap` | `TAPRequest`, `TAPResponse`, `CompiledPrompt`, `TAPCostRecord`, `CostTracker` | TAP IR 数据结构——用户意图与模型 API 之间的模型无关契约 |
| `teragent.core.compiler` | `TAPCompiler` (ABC), `TAPCompilerRegistry` | 编译器抽象基类 + 名称→类注册表。子类实现 `compile()` 生成模型专属 Prompt |
| `teragent.core.adapter` | `TAPAdapter` (ABC), `TAPAdapterRegistry` | 适配器抽象基类 + 名称→类注册表。子类实现 `send()` 和 `stream()` 处理协议级 HTTP |
| `teragent.core.provider` | `ModelProvider` | 组合编译器 + 适配器。提供 `execute_tap()`、`stream_tap()`、`chat()`、`execute_tap_with_retry()`、`chat_with_fallback()` |
| `teragent.core.types` | `Message`, `MessageRole`, `MessageType`, `ToolSafety` | 内部消息类型和工具安全枚举（`READ_ONLY`、`SAFE_WRITE`、`DESTRUCTIVE`、`HIGH_RISK`） |
| `teragent.core.compilers.default` | `DefaultCompiler` | 通用 OpenAI 兼容 Prompt 编译 |
| `teragent.core.compilers.glm` | `GLMCompiler` | GLM 优化：近因效应（关键指令置于末尾） |
| `teragent.core.compilers.glm_5` | `GLM5Compiler` | GLM-5：近因效应 + 200K 极限压缩 + 长程任务 |
| `teragent.core.compilers.glm_52` | `GLM52Compiler` | GLM-5.2：1M 上下文 + High/Max 双思考 + PreservedThinking |
| `teragent.core.compilers.glm_5v_turbo` | `GLM5VTurboCompiler` | GLM-5V-Turbo：视觉分析 + 多模态 prompt |
| `teragent.core.compilers.anthropic` | `AnthropicCompiler` | Claude 优化：XML 标签结构 + 模式 B（system/user 分离） |
| `teragent.core.compilers.deepseek` | `DeepSeekCompiler` | DeepSeek 优化：极简 Prompt 格式 |
| `teragent.core.compilers.deepseek_v4` | `DeepSeekV4Compiler` | DeepSeek V4：缓存感知布局 + 思考模式 + Flash/Pro 变体 + 1M 上下文 |
| `teragent.core.compilers.minimax_m3` | `MiniMaxM3Compiler` | MiniMax M3：多模态 + MSA 全文注入 + 桌面操作 |
| `teragent.core.adapters.openai_compatible` | `OpenAICompatibleAdapter` | OpenAI `/chat/completions` 协议，支持 SSE 流式 |
| `teragent.core.adapters.anthropic_native` | `AnthropicNativeAdapter` | Anthropic `/messages` 协议，Anthropic 专属 SSE |
| `teragent.core.adapters.glm_native` | `GLMNativeAdapter` | GLM 原生 API：thinking + cache + 异步 + reasoning_content |
| `teragent.core.adapters.minimax_native` | `MiniMaxNativeAdapter` | MiniMax M3 原生：Anthropic 兼容 + OpenAI 双接口 + 视频 + 桌面 |
| `teragent.core.adapters.mock` | `MockAdapter` | 测试适配器——无 HTTP 调用 |
| `teragent.core.prompts` | `get_system_prompt_for_intent()`, `list_intents()`, `list_compiler_types()` | 集中化 Prompt 管理：9 种意图 × 9 种编译器变体 |

**Prompt 意图：** `design`、`plan`、`replan`、`execute`、`review`、`chat`、`chat_friendly`、`sub_agent`、`code_generation`（`execute` 的别名）

**编译器类型：** `default`、`glm`、`glm_5`、`glm_52`、`glm_5v_turbo`、`anthropic`、`deepseek`、`deepseek_v4`、`minimax_m3`

### 路由与 Pipeline（多模型）

| 组件 | 说明 |
|------|------|
| `ModelRouter` | 6 步智能路由（多模态→上下文→长程→意图→成本→降级容灾） |
| `RoutingTable` | 可配置路由规则，含意图默认值和覆盖映射 |
| `RoutingDecision` | 捕获路由选择，包含完整追踪信息用于调试 |
| `PipelineManager` | 运行时 Pipeline 配置方案切换（default/budget/multimodal/deep_thinking） |
| `PipelineProfile` | 命名阶段→驱动映射，用于快速配置 |

### 长程任务

| 组件 | 说明 |
|------|------|
| `LongHorizonTaskManager` | 编排 8 小时自主任务（基于 GLM-5） |
| `SubGoal` / `PhaseResult` / `LongHorizonResult` | 任务分解和结果追踪 |
| `CheckpointStore` | 基于 JSON 的检查点持久化，含自动清理 |
| `SelfEvaluator` | 周期性自评估（目标对齐、输出质量、1-5 分评分） |
| `StrategySwitcher` | 检测停滞并触发策略切换（6 种内置策略） |
| `ProgressTracker` / `ProgressReport` | 实时进度追踪，含 ETA 估算 |
| `LongHorizonRecoveryManager` | 基于检查点的恢复，含指数退避 |

### 预算与成本追踪

| 组件 | 说明 |
|------|------|
| `CrossModelCostTracker` | 多模型成本追踪，含月度预算控制 |
| `MonthlyBudgetConfig` | 预算限制，含警告阈值和自动降级 |
| `CostRecord` | 单次调用成本记录，含缓存节省追踪 |
| `ModelCircuitBreakerManager` | 按模型的熔断器，含降级链 |
| `DegradationChain` | 任务类型感知的回退排序（heavy/multimodal/default） |
| `RateLimitHandler` | 统一的 V4/M3/GLM-5 速率限制处理 |

### 限流器

| 组件 | 说明 |
|------|------|
| `TokenBucketRateLimiter` | 令牌桶限流器 — 允许突发流量，之后按恒定速率补充 |
| `SlidingWindowRateLimiter` | 滑动窗口限流器 — 在时间窗口内跟踪请求时间戳 |
| `AdaptiveRateLimiter` | 自适应限流器 — 从 `X-RateLimit-*` 和 `Retry-After` 响应头学习，429 时自动降速 |
| `RateLimitConfig` | 所有策略的配置（令牌桶、滑动窗口、自适应） |
| `RateLimitStrategy` | 策略枚举：`TOKEN_BUCKET`、`SLIDING_WINDOW`、`ADAPTIVE` |
| `RateLimitStatus` | 当前限流状态，含 `is_limited` 和 `wait_seconds` 属性 |
| `create_rate_limiter()` | 工厂函数 — 根据配置创建相应的限流器 |

### 基准测试

TerAgent 在四模型配置下进行了全面的基准测试，评估维度包括代码生成质量、推理深度、多模态理解、长程任务完成率和成本效率。

**关键基准结果：**

| 基准 | V4-Flash | V4-Pro | M3 | GLM-5 | GLM-5.2 |
|------|----------|--------|----|---------|---------|
| 代码生成准确率 | 高 | 高 | 中 | 高 | 高 |
| 推理深度 | 低 | 极高 | 中 | 极高 | 极高 |
| 多模态理解 | — | — | 极高 | — | 高（5V-Turbo） |
| 长程任务完成率 | — | — | — | 极高 | 极高 |
| 平均响应延迟 | 低 | 中 | 中 | 中 | 中 |

📖 完整基准测试数据详见 [四模型评估报告](docs/EVALUATION_FOUR_MODELS.md)。

### Pipeline 原语

| 模块 | 关键函数 | 说明 |
|------|---------|------|
| `teragent.pipeline.extractor` | `extract_files_from_response()` | 从模型输出中解析 `<file>` 标签 |
| `teragent.pipeline.prompt_builder` | `build_prompt()`, `build_subagent_prompt()`, `validate_prompt_tokens()` | 基于模板的 Prompt 构建，含 Token 预算验证 |
| `teragent.pipeline.checklist` | `run_deterministic_checks()`, `check_code_quality()`, `check_runnable()` | 确定性代码验证（AST、语法、导入、冲突检查） |
| `teragent.pipeline.retry` | `retry_with_backoff()` | 指数退避重试，支持可配置验证 |
| `teragent.pipeline.tracing` | `TAPTracer`, `DPOPair`, `DataConstitution`, `TraceStats` | 自强化学习追踪记录 + DPO 对生成（详见[自强化学习数据宪章](#自强化学习数据宪章)） |

### AgentLoop（核心编排）

`AgentLoop` 是主编排类，将所有横切关注点组合成一个内聚的工具调用循环。

**每次用户输入的生命周期：**

```
1. IntentClassifier → CHAT / DEBUG / CREATE_PROJECT
2. ConfirmationGate → (若 CREATE_PROJECT，请求用户确认)
3. 按意图过滤工具（来自 config.intent_tools）
4. SubAgent 委派（若 CREATE_PROJECT + SubAgentManager 可用）
5. 工具循环：
   a. 检查步骤预算
   b. 上下文压缩（如接近 Token 限制）
   c. 调用模型（流式或批量）
   d. 若 tool_calls → 执行工具（StreamingToolExecutor 或 ToolOrchestrator）
   e. 追加工具结果，回到 (a)
   f. 若仅文本 → 完成
6. 发射事件，持久化会话
```

**AgentLoop 集成的横切关注点：**

| 关注点 | 组件 | 集成点 |
|--------|------|--------|
| 成本追踪 | `CircuitBreakerManager` | 记录每次模型调用的 Token 用量 |
| 故障保护 | `ConsecutiveFailureBreaker` | 连续 N 次失败后打开熔断器 |
| 延迟监控 | `LatencyBreaker` | 对持续慢速调用发出警告 |
| 进度检测 | `ProgressDetector` | 检测卡死循环（无有意义进展） |
| 权限检查 | `EnhancedPermissionManager` | 执行前验证工具调用 |
| 意图分类 | `IntentClassifier` | 将用户输入路由到适当行为 |
| 上下文管理 | `ContextWindow` + `AutoCompactor` | 接近 Token 限制时压缩上下文 |
| 流式模式 | `StreamingToolExecutor` | 自动检测流式能力，失败重试，降级到批量 |
| 会话持久化 | `SessionPersistence` | 保存/恢复对话状态 |
| Hook 系统 | `HookManager` | 执行前后钩子用于自定义 |
| 子 Agent 协调 | `SubAgentManager` | 为复杂任务生成子 Agent |
| 事件总线 | `EventBus` | 信号驱动的事件发射贯穿整个生命周期 |

### 流式执行

`StreamingToolExecutor` 实时处理模型流事件并执行工具，显著降低延迟。

**调度策略（权威定义）：**

| 工具安全属性 | 执行策略 |
|-------------|---------|
| `read_only` + `concurrency_safe` | **立即执行**——流式期间异步执行（无需等待） |
| 非只读或非并发安全 | **延迟串行**——流结束后串行执行 |
| 未知工具 | **延迟**（保守默认） |

**降级路径：**

```
流式 + tool_use → 流式重试 → 批量降级
       ↓ 失败       ↓ 失败      ↓
    重试 N 次    降级到      ToolOrchestrator
                 批量模式    .execute_batch()
```

### 安全体系

TerAgent 提供多层纵深防御安全体系。

- 6 层命令防御（规范化 → 管道链拆分 → 黑名单 → 跨链检测 → 包警告 → 元字符）+ 平台专属模式（Windows 16 种危险模式黑名单、Windows 系统路径保护）
- 3 级沙箱降级（Firecracker → Docker → 子进程）含跨平台进程管理
- 2 阶段提交文件写入（校验 → 写临时 → 原子替换 → 回滚）含 NTFS 3 步回退

#### 7 层权限解析

```
第 1 层: 用户规则     (优先级 100) ─┐
第 2 层: 配置规则     (优先级 60)  ─┤ 这些是 PermissionRule
第 3 层: 项目规则     (优先级 50)  ─┤ 使用 glob 匹配
第 4 层: 系统规则     (优先级 10)  ─┘ tool_name + path
第 5 层: PermissionLevel 检查       ← DEFAULT / PLAN / BYPASS / ACCEPT_EDITS / AUTO
第 6 层: AI 分类器 (仅异步)        ← 咨询性质，使用 LLM 判断意图
第 7 层: 默认 DENY                  ← 无规则匹配时的安全默认
```

**PermissionRule 示例：**

```python
from teragent.security import EnhancedPermissionManager, PermissionRule, PermissionEffect

epm = EnhancedPermissionManager()

# 用户级 DENY：永远禁止读取 /etc
epm.add_rule(PermissionRule(
    effect=PermissionEffect.DENY,
    tool_pattern="read_file",
    path_pattern="/etc/*",
    description="禁止读取系统目录",
    source="user",  # 最高优先级
))

# 系统级 ALLOW：允许读取项目文件
epm.add_rule(PermissionRule(
    effect=PermissionEffect.ALLOW,
    tool_pattern="read_file",
    description="读取文件始终允许",
    source="system",
))

# 检查权限
allowed, reason = epm.check("read_file", path="/etc/passwd")
# allowed = False, reason = "Denied by rule: 禁止读取系统目录"
```

#### 6 层命令防御

```
第 1 层: 命令归一化         ← 去除 ANSI、null 字节、压缩空白
第 2 层: 管道链拆分         ← 检查 | && ; 链中的每个子命令
第 3 层: 8 类黑名单         ← 提权、反向 Shell、内联执行、
                               系统破坏、持久化、编码绕过、
                               远程执行、Fork 炸弹 / 磁盘写入
第 4 层: 危险重定向检测     ← > /etc/、> /dev/、> /sys/（按子命令细粒度检测）
第 5 层: 跨链检测           ← curl | sh、wget | python（仅在完整命令中可见）
第 6 层: 包安装警告         ← pip/npm/apt install → 记录警告，不硬性阻断
```

#### 2 阶段提交 (2PC) 文件写入

```
阶段 1: 验证    → 检查权限 + 路径穿越 + 先读后写契约
阶段 2: 写入    → 所有文件写入 .tmp 后缀
阶段 3: 提交    → os.replace() 原子交换（全部成功或全部回滚）
阶段 4: 回滚    → 提交失败时从 .bak 备份恢复
```

**关键特性：**
- **原子性**：`os.replace()` 在 POSIX 和 Windows 上均为原子操作
- **崩溃安全**：中间临时文件防止崩溃导致数据损坏
- **一致性**：所有文件要么全部提交，要么全部不提交（事务性）
- **并发安全**：读取者永远不会看到半写状态
- **路径穿越防护**：所有路径必须在 `workspace_root` 内

#### 3 级沙箱降级

| 级别 | 隔离方式 | 降级目标 |
|------|---------|---------|
| Level 2 | Firecracker 微虚拟机 | → Docker（Level 1） |
| Level 1 | Docker 容器（512MB, 1 CPU, 64 PIDs） | → 子进程（Level 0） |
| Level 0 | 子进程 + `rlimit` + `create_subprocess_exec` | — |

### 可靠性系统

四个独立熔断器防止 Token 浪费和无限循环。

| 熔断器 | 检测内容 | 行为 |
|--------|---------|------|
| **CostBudgetTracker** | Token 预算接近限制 | 70% 建议性警告，90% 严重警告，100% 可选硬停止 |
| **ConsecutiveFailureBreaker** | 连续 N 次 API 失败 | 打开熔断器 → 暂停调用 → 冷却后半开 |
| **LatencyBreaker** | 模型调用持续缓慢 | 建议性警告（不阻断） |
| **ProgressDetector** | Agent 循环无有意义进展 | ≥80% 近期步骤无效时发出卡死警告 |

**其他可靠性特性：**
- **流式重试 + 批量降级**：自动重试流式调用，持续失败时降级为批量
- **上下文压缩**：接近 Token 限制时自动触发 `AutoCompactor`
- **步骤预算**：每次对话的工具调用步骤硬上限
- **恢复管理器**：处理输出截断（`finish_reason="length"`）、上下文溢出错误和 Provider 降级

**RecoveryType 枚举：**

| 类型 | 触发条件 |
|------|---------|
| `LENGTH` | 输出 Token 截断 → 续接请求 |
| `CONTEXT_OVERFLOW` | 输入上下文超过模型 Token 限制 → 压缩 + 重试 |
| `FALLBACK` | 主模型失败 → 切换到备用 Provider |
| `STREAMING_RETRY` | 流式调用失败 → 重试或降级为批量 |
| `TOOL_REPAIR` | 工具执行失败 → 修复重试 |

### 上下文管理

| 组件 | 说明 |
|------|------|
| `ContextWindow` | Token 预算估算器，支持 CJK 启发式。保守估算（×1.3 因子）避免 API 溢出 |
| `Microcompactor` | 细粒度上下文裁减——移除低信息量消息，保留关键上下文 |
| `AutoCompactor` | 基于 `ContextWindow.should_compact()` 自动触发压缩 |
| `CodeIndexer` | tree-sitter AST 索引，用于代码结构理解（`teragent[ast]`） |
| `ReferenceGraph` | 基于 networkx 的依赖图分析（`teragent[graph]`） |
| `VectorIndexer` | LanceDB 语义代码搜索（`teragent[vector]`） |
| `DependencyReporter` | 为 TAP 上下文生成依赖报告（懒加载，需要可选依赖） |
| `Memory` | `load_agent_md()` / `save_agent_md()`——通过 `.agent.md` 文件实现持久化项目记忆 |

### 协调层（子 Agent）

`SubAgentManager` 创建和管理子 Agent 生命周期，支持三种执行模式：

| 模式 | 行为 | 使用场景 |
|------|------|---------|
| `SYNC` | 阻塞父 Agent 直到子 Agent 完成 | 必须完成后才能继续的简单子任务 |
| `ASYNC` | 后台运行，完成后通过 `AgentMessageBus` 通知父 Agent | 长时间运行的后台任务 |
| `FORK` | 类似 SYNC，但标记共享系统 Prompt 前缀用于 KV 缓存优化 | 共享上下文的重复查询 |

**安全约束：**
- 每个子 Agent 最多 15 步（防止无限循环）
- 最多 5 个并发子 Agent（防止资源耗尽）
- 工具白名单——子 Agent 只能使用明确允许的工具
- 预算追踪——子 Agent 遵守全局步骤预算

### 意图分类

| 组件 | 说明 |
|------|------|
| `IntentClassifier` | 将用户输入分类为 `CHAT`、`DEBUG` 或 `CREATE_PROJECT` 意图 |
| `ConfirmationGate` | 执行 CREATE_PROJECT 意图前需要用户明确批准 |

意图分类结果输入工具过滤系统——不同意图通过 `AgentLoopConfig.intent_tools` 获得不同的工具子集。

### Hook 系统

| 组件 | 说明 |
|------|------|
| `HookManager` | 管理执行前后钩子，返回 `HookDecision`（ALLOW / DENY / MODIFY） |
| `Hook` (ABC) | 钩子基类——`ShellHook`（命令钩子）和 `PythonHook`（Python 可调用钩子） |
| `AuditHook` | 内置钩子，记录所有工具执行用于审计追踪 |
| `DangerousCommandHook` | 内置钩子，使用 6 层防御阻断危险 Shell 命令 |

### 会话持久化

`SessionPersistence` 提供基于 SQLite 的完整对话生命周期管理：

- 按 ID 创建/恢复会话
- 保存每条消息到会话
- 追踪步骤计数
- 列出会话历史

### 自强化学习数据宪章

TerAgent 包含完整的自强化学习数据管线。每次 TAP 调用可自动追踪，并与确定性验证结果配对，生成 DPO（Direct Preference Optimization）训练对。

**数据宪章原则：**

1. **TAP 追踪是核心库输出**，独立于特定 Agent 流程
2. **偏好标签来自确定性检查**（AST、语法、可运行性），而非人工标注
3. **数据属于用户**——库永远不会上传追踪数据

**DPO 对生成：**

```
TAPRequest  →  TAPTracer.record_request()  →  JSONL 追踪
TAPResponse →  TAPTracer.record_response() →  JSONL 追踪
Checklist   →  TAPTracer.record_checklist() →  JSONL 追踪
                                                    ↓
                                     TAPTracer.export_dpo_pairs()
                                                    ↓
                                   (chosen=PASS, rejected=FAIL) 对
```

**配对策略：**

| 策略 | 说明 |
|------|------|
| 同任务重试 | 同一 `task_id` 有 PASS 和 FAIL 响应（来自重试）→ `(chosen=PASS, rejected=FAIL)` |
| 跨任务 | 相同意图的不同 `task_id` → 将一个的 PASS 与另一个的 FAIL 配对 |
| 部分 | 仅有 chosen 或仅有 rejected（当 `include_partial=True` 时） |

### 配置系统

TerAgent 使用基于 `agent.toml` 文件的类型化配置系统。

**可用配置模块：**

| 配置模块 | 关键类 | 控制范围 |
|---------|--------|---------|
| `teragent.config.teragent_config` | `TerAgentConfig` | 顶层配置容器 |
| `teragent.config.agent_loop_config` | `AgentLoopConfig` | Agent 循环行为（最大步数、流式重试、工具超时、意图→工具映射） |
| `teragent.config.circuit_breaker_config` | `CircuitBreakerConfig` | 预算阈值、失败限制、延迟阈值、卡死检测 |
| `teragent.config.streaming_config` | `StreamingConfig` | 流式模式和重试行为 |
| `teragent.config.permission_config` | `PermissionConfig` | 权限模式和规则 |
| `teragent.config.context_management_config` | `ContextManagementConfig` | 上下文窗口限制和压缩阈值 |
| `teragent.config.tools_config` | `ToolsConfig` | 工具注册配置 |
| `teragent.config.file_safety_config` | `FileSafetyConfig` | 文件写入安全和 2PC 行为 |
| `teragent.config.session_config` | `SessionConfig` | 会话持久化设置 |
| `teragent.config.hooks_config` | `HooksConfig` | Hook 注册 |
| `teragent.config.recovery_config` | `RecoveryConfig` | 恢复策略配置 |
| `teragent.config.coordination_config` | `CoordinationConfig` | 子 Agent 协调设置 |
| `teragent.config.execution_pipeline_config` | `ExecutionPipelineConfig` | Pipeline 阶段驱动分配 |
| `teragent.config.model_fallback_config` | `ModelFallbackConfig` | 模型降级链配置 |
| `teragent.config.driver_config` | `DriverConfig` | 单个模型驱动（编译器 + 适配器 + 模型 + API Key） |
| `teragent.config.api_key_security` | `ApiKeyVault`, `SecurityFinding` | API Key 解析、脱敏和安全审计 |

### 事件总线

`EventBus` 是 TerAgent 的信号驱动通信骨干。

**关键方法：**

| 方法 | 说明 |
|------|------|
| `emit()` | 即发即弃事件发射（永不阻塞主循环） |
| `emit_and_wait()` | 发射事件并等待所有处理函数完成 |
| `emit_message()` | 发射带元数据的结构化 `Message` 事件 |
| `on()` / `once()` | 订阅事件（永久 / 一次性） |
| `wait_for()` | 等待特定事件（带超时） |

**设计原则：**
- 即发即弃：异步处理函数通过 `create_task`，同步处理函数通过 `run_in_executor`
- 错误隔离：单个处理函数失败不影响其他处理函数
- 事件历史：追踪最近 200 个事件及结构化数据，用于调试

---

## 配置

在项目根目录创建 `agent.toml`（完整示例见 `examples/agent.toml`）：

```toml
[drivers.openai_compatible.glm_5]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5"
compiler = "glm_5"

[drivers.anthropic_native.claude]
base_url = "https://api.anthropic.com/v1"
api_key_env = "ANTHROPIC_API_KEY"
model = "claude-sonnet-4-20250514"
compiler = "anthropic"

[drivers.openai_compatible.deepseek]
base_url = "https://api.deepseek.com/v1"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-chat"
compiler = "deepseek"

[execution.pipeline]
design_driver = "openai_compatible.glm_5"
plan_driver = "openai_compatible.glm_5"
execute_driver = "openai_compatible.glm_5"
review_driver = "openai_compatible.glm_5"

[permission]
mode = "plan"
rules = { allow = ["read_file:*", "explore_codebase:*"], deny = ["*:**/.env*", "read_file:/etc/*"] }
```

**API Key 安全：** 始终使用 `api_key_env`（环境变量名）而非 `api_key`（直接值）。`ApiKeyVault` 通过环境变量解析密钥，并通过 `python-dotenv` 支持 `.env` 文件回退。使用 `audit_config_security()` 和 `audit_env_file()` 扫描泄露的密钥。

---

## 构建方式

TerAgent 的所有代码均由 AI 生成，没有一行由人类手写。项目遵循**设计 → 规划 → 编码 → 审查**流水线：

- **设计阶段**：我与多个 AI 模型（包括 DeepSeek、GLM-5）共同讨论，确定 TAP 作为 IR、编译器/适配器正交解耦、安全层等核心架构。
- **规划阶段**：我指挥 AI 将系统拆解为 95 个模块，指定接口与依赖关系，生成详细任务分解。
- **编码阶段**：我通过自然语言指挥 GLM-5 严格按照规划逐模块生成代码。
- **审查阶段**：我指挥 AI 对代码进行语法检查、依赖验证、可运行性检测等审查，并根据反馈决定接受、修改或拒绝。

以上流程执行后，AI 自动统计了项目数据：约 46,900 行 Python 代码（17 个子模块，99 个源文件），测试约 28,300 行（61 个测试文件），测试/源码比 60.4%，版本 0.1.2 Beta，许可证 Apache-2.0。这些数据同样由 AI 生成。

项目发布后，GLM-5 在独立会话中对整个代码库进行了第三方评估，给出综合评分 **7.4/10**（架构设计 9.0，防幻觉安全 7.5，工程规范 6.5）。该评估指出项目的核心创新在于 TAP IR 与 Compiler/Adapter 正交组合，安全体系本质上是一个"防 AI 自毁"系统，主要短板为缺少意图-行动一致性校验、沙箱降级需用户确认、以及无 CI/CD。完整的三模型评估报告存放于 [`docs/EVALUATION_THREE_MODELS.md`](docs/EVALUATION_THREE_MODELS.md)，四模型评估报告存放于 [`docs/EVALUATION_FOUR_MODELS.md`](docs/EVALUATION_FOUR_MODELS.md)（该报告亦由 AI 生成）。

这种构建方式本身也是 TerAgent 的一部分：`pipeline` 模块提供了可复用的**设计 → 规划 → 编码 → 审查**流程。

---

## 开发

```bash
# 安装开发依赖
pip install teragent[dev]

# 运行测试
pytest

# 代码检查
ruff check teragent/

# 类型检查
mypy teragent/
```

---

## 许可证

Apache License Version 2.0
