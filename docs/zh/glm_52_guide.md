# GLM-5.2 使用指南

本指南涵盖 TerAgent 的 GLM-5.2 集成，包括 1M 上下文使用、High/Max 双思维模式、用于编码计划的 PreservedThinking、5V-Turbo 视觉协同，以及生产环境最佳实践。

---

## 目录

- [概述](#概述)
- [1M 上下文使用与最佳实践](#1m-上下文使用与最佳实践)
- [High/Max 双思维模式](#highmax-双思维模式)
- [用于编码计划的 PreservedThinking](#用于编码计划的-preservedthinking)
- [5V-Turbo 视觉协同](#5v-turbo-视觉协同)
- [配置示例](#配置示例)
- [性能提示与稳定性考量](#性能提示与稳定性考量)
- [上下文降级](#上下文降级)
- [与流水线阶段的集成](#与流水线阶段的集成)
- [监控与可观测性](#监控与可观测性)
- [故障排除](#故障排除)

---

## 概述

GLM-5.2 是智谱 AI 旗下面向超长上下文和高级推理的旗舰模型。它在 GLM-5 的基础上扩展了 1M token 上下文窗口、双思维模式、PreservedThinking 和视觉协同等能力。

| 规格参数 | 值 |
|--------------|-------|
| 上下文窗口 | 1,000,000 tokens |
| 最大输出 | 128,000 tokens |
| 思维模式 | High（默认）、Max |
| API 端点 | `https://open.bigmodel.cn/api/paas/v4` |
| Compiler | `glm_52` |
| 共享 API 密钥 | 与 GLM-5 相同（`GLM_API_KEY`） |

### 与 GLM-5 的关键差异

| 特性 | GLM-5 | GLM-5.2 |
|---------|-------|---------|
| 上下文窗口 | 200K | 1M |
| 思维模式 | deep | High / Max |
| PreservedThinking | ❌ | ✅ |
| 5V-Turbo 协同 | ❌ | ✅ |
| 上下文降级 | ❌ | ✅ (1M → 200K) |
| 长时任务 | ✅ (8h) | ✅ (8h+) |
| 每 token 成本 | 较低 | 较高 |

---

## 1M 上下文使用与最佳实践

### 何时使用 1M 上下文

1M 上下文窗口是强大的功能，但会带来更高的成本和延迟。请谨慎使用：

**适用场景：**
- 分析整个大型代码库（>200K tokens）
- 处理大量文档
- 跨大型项目的多文件重构
- 包含累积上下文的长对话历史
- 大型系统的全面审计或审查

**200K 即可满足的场景：**
- 单文件或小型项目任务
- 仅需最新上下文即可完成的任务
- 历史记录较少的短对话
- 预算受限的场景

### 配置 1M 上下文

```toml
[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
max_context_tokens = 1_000_000          # 启用 1M 上下文
max_output_tokens = 128_000
# 上下文降级由 AutoCompactor 内部处理
```

### 高效加载大上下文

```python
from teragent import create_provider, TAPRequest
from teragent.context import ContextWindow

provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
)

# 将大型代码库加载到上下文中
context = ContextWindow(
    max_tokens=1_000_000,
    reserved_for_output=128_000,
    reserved_for_system=8_000,
)

# Compiler 会自动为 1M 窗口优化上下文布局
request = TAPRequest(
    instruction="Analyze this entire codebase and identify architectural issues",
    context={
        "codebase": large_codebase_text,  # 可以超过 500K tokens
    },
)
```

### 1M 上下文最佳实践

1. **按层次组织上下文** — 将最重要的信息放在开头和结尾（GLM52Compiler 的近因效应优化）
2. **启用上下文降级** — 在内存压力下自动降级到 200K
3. **使用前缀缓存** — 保持系统提示和工具定义的稳定性
4. **监控内存使用** — 通过 `ContextWindow` 跟踪上下文利用率
5. **批量处理相关内容** — 将相关文件或文档分组处理，而非拆分到多个请求中
6. **大上下文优先使用 High 思维** — Max 思维配合 1M 上下文可能非常慢；默认使用 High
7. **设置适当的超时** — 1M 上下文请求耗时更长；配置 `timeout` 和 `multimodal_timeout`

### 1M 上下文注意事项

1M 上下文消耗的内存远超 200K，请确保推理端点有足够资源，并启用 AutoCompactor 作为安全网。
---

## High/Max 双思维模式

GLM-5.2 引入了双思维模式，允许你在推理深度与速度和成本之间进行平衡。

### High 思维（默认）

High 思维提供标准的深度推理——类似于 GLM-5 的 `deep` 模式，但针对 1M 上下文窗口进行了优化。它是大多数任务的正确选择。

```python
from teragent import create_provider

provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    thinking_mode="high",  # 默认；速度与深度的平衡
)
```

**特征：**
- 响应时间：中等
- 推理深度：标准深度推理
- Token 消耗：约为非思维模式的 ~1.2 倍
- 最适用于：代码生成、规划、分析

### Max 思维

Max 思维激活最深层的推理能力。它明显更慢，但提供最全面的分析。

```python
from teragent import create_provider

provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    thinking_mode="max",  # 最大推理深度
)
```

**特征：**
- 响应时间：慢（约为 High 思维的 2-5 倍）
- 推理深度：最大链式思维
- Token 消耗：约为 High 思维的 3-5 倍
- 最适用于：架构决策、复杂调试、关键代码审查

### 按请求覆盖思维模式

你可以按请求覆盖思维模式：

```python
from teragent import TAPRequest

# 大多数请求使用 High 思维（默认）
request = TAPRequest(
    instruction="Generate a REST API endpoint for user registration",
)

# 对关键决策覆盖为 Max 思维
critical_request = TAPRequest(
    instruction="Decide whether to use microservices or monolith for this project",
    meta={"thinking_mode": "max"},  # 按请求覆盖
)
```

### 何时使用各模式

| 场景 | 推荐模式 | 原因 |
|----------|-----------------|--------|
| 代码生成 | High | 速度与质量的平衡 |
| 快速分析 | High | 推理深度足够 |
| 架构决策 | Max | 需要最大推理能力 |
| 复杂调试 | Max | 需要深层因果分析 |
| 代码审查（标准） | High | 良好的权衡 |
| 代码审查（关键） | Max | 最全面的分析 |
| 大上下文分析 | High | Max 配合 1M 可能非常慢 |
| 预算受限 | High | Max 消耗 3-5 倍的 token |

---

## 用于编码计划的 PreservedThinking

PreservedThinking 是 GLM-5.2 的独特功能，可在编码会话之间保留推理痕迹，确保生成的代码与原始计划保持一致。

### PreservedThinking 工作原理

```
Plan Request → Reasoning Trace Generated → Trace Preserved
                                              ↓
Code Request ← Trace Injected ← PreservedThinking
                                              ↓
Generated code stays aligned with original plan
```

### 启用 PreservedThinking

> **注意：** `preserved_thinking_enabled` 是 `create_provider()` 的编译器级 kwargs，不是 TOML 驱动字段。无法在 `agent.toml` 中设置。

通过编程方式：

```python
from teragent import create_provider

provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    preserved_thinking_enabled=True,  # 编译器级 kwargs
)
```

### 使用 PreservedThinking 进行多步骤编码

```python
from teragent import create_provider, TAPRequest

provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    preserved_thinking_enabled=True,  # 编译器级 kwargs
)

# 步骤 1：创建架构计划
plan_request = TAPRequest(
    instruction="Design a real-time chat system architecture with: "
                "(1) WebSocket connections, (2) message persistence, "
                "(3) typing indicators, (4) read receipts, "
                "(5) file sharing. Include database schema and API design.",
    meta={"intent": "plan"},
)
plan_response = await provider.execute_tap(plan_request)

# 步骤 2：基于计划进行实现
# PreservedThinking 自动注入推理痕迹
impl_request = TAPRequest(
    instruction="Now implement the WebSocket connection handler based on the plan",
    meta={"intent": "create"},
)
impl_response = await provider.execute_tap(impl_request)

# 步骤 3：继续实现
# 推理痕迹仍然保留
db_request = TAPRequest(
    instruction="Implement the message persistence layer with the database schema from the plan",
    meta={"intent": "create"},
)
db_response = await provider.execute_tap(db_request)
```

### PreservedThinking 最佳实践

1. **以计划请求开始** — 始终以 `meta={"intent": "plan"}` 请求开始，建立推理痕迹
2. **保持会话活跃** — PreservedThinking 在单个会话内工作；不要让会话过期
3. **显式引用计划** — 在后续请求中，提醒模型具体的计划要素
4. **不要混用 PreservedThinking 会话** — 每个 provider 实例有自己的保留上下文；不要在不相关的任务之间共享
5. **监控 token 消耗** — PreservedThinking 会增加上下文；跟踪总 token 使用量
6. **配合 High 思维使用效果最佳** — Max 思维 + PreservedThinking 可能消耗大量 token

### PreservedThinking 限制

- **会话范围** — PreservedThinking 不会跨会话或进程重启持久化
- **Token 开销** — 每条保留痕迹增加约 2-5K tokens 到上下文中
- **不兼容所有 Compiler** — 只有 `glm_52` Compiler 支持 PreservedThinking
- **最多保留 10 条痕迹** — 达到限制时，较早的痕迹会被自动摘要

---

## 5V-Turbo 视觉协同

5V-Turbo 视觉协同使 GLM-5.2 能够与 GLM-5V-Turbo（智谱 AI 的视觉模型）配合工作，实现"视觉→代码→验证"循环。

### 架构

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  GLM-5V-Turbo│────▶│   GLM-5.2    │────▶│ Verification │
│  (Visual     │     │  (Code       │     │  (5V-Turbo   │
│   Analysis)  │     │   Generation)│     │   Re-check)  │
└──────────────┘     └──────────────┘     └──────────────┘
       ↑                                          │
       └──────────── Feedback Loop ───────────────┘
```

### 启用 5V-Turbo 协同

> **注意：** `vision_coordination_enabled` 是 `create_provider()` 的编译器级 kwargs，不是 TOML 驱动字段。在 TOML 中，请在 GLM-5.2 驱动上使用 `multimodal_enabled = true` 并单独配置视觉模型端点。

```toml
[drivers.openai_compatible.glm_52]
multimodal_enabled = true               # 启用多模态（视觉协调）

# 可选：配置视觉模型端点
[drivers.openai_compatible.glm_5v_turbo]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5v-turbo"
compiler = "glm_5v_turbo"
```

### 使用视觉协同

```python
from teragent import create_provider, TAPRequest
from teragent.core.tap import MultimodalContent

provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    vision_coordination_enabled=True,  # 编译器级 kwargs
)

# 视觉辅助编码任务
request = TAPRequest(
    instruction="Look at this UI mockup and implement the React component. "
                "Follow the exact layout, colors, and typography shown.",
    multimodal_context=[
        MultimodalContent(
            type="image_url",
            image_url={"url": "https://example.com/mockup.png"},
        ),
    ],
)

response = await provider.execute_tap(request)
# 系统自动：
# 1. 将图像发送到 GLM-5V-Turbo 进行视觉分析
# 2. 将视觉理解提供给 GLM-5.2
# 3. GLM-5.2 基于视觉分析生成代码
```

### 视觉→代码→验证循环

```python
# 完整的视觉协同工作流
async def vision_code_verify(image_url: str, requirement: str):
    # 步骤 1：视觉分析（5V-Turbo）
    vision_request = TAPRequest(
        instruction="Describe the UI elements, layout, and styling in this image",
        multimodal_context=[
            MultimodalContent(type="image_url", image_url={"url": image_url}),
        ],
    )

    # 步骤 2：代码生成（GLM-5.2）
    code_request = TAPRequest(
        instruction=f"Based on the visual analysis, implement: {requirement}",
        meta={"vision_context": True},
    )

    # 步骤 3：视觉验证（5V-Turbo 复查）
    # 生成代码后，渲染并与原图比较
    verify_request = TAPRequest(
        instruction="Compare the rendered output with the original mockup. "
                    "Identify any discrepancies in layout, colors, or typography.",
        multimodal_context=[
            MultimodalContent(type="image_url", image_url={"url": image_url}),
            MultimodalContent(type="image_url", image_url={"url": rendered_url}),
        ],
    )
```

### 降级行为

当 5V-Turbo 不可用时：
1. **熔断器检测到** 5V-Turbo 故障
2. **系统降级** 为纯文本分析——GLM-5.2 以文本方式处理图像描述
3. **质量降低** — 没有视觉能力时，模型依赖图像的文本描述
4. **自动恢复** — 当 5V-Turbo 恢复可用时，系统自动恢复协同

---

## 配置示例

### 最小配置

```toml
[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
max_context_tokens = 1_000_000
```

### 全功能配置

```toml
[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
max_context_tokens = 1_000_000
max_output_tokens = 128_000
thinking_mode = "high"                    # 默认思维模式
multimodal_enabled = true                 # 启用多模态（视觉协同）
long_horizon_enabled = true               # 启用长时任务模式
# 注意：dual_thinking_enabled、preserved_thinking_enabled、vision_coordination_enabled
# 和 context_degradation_enabled 是 create_provider() 的 kwargs，不是 TOML 字段。
# 使用 thinking_mode + 按请求覆盖来控制双思考模式。
# 上下文降级由 AutoCompactor 内部处理。
```

### 本地部署配置

```toml
[drivers.openai_compatible.glm_52]
base_url = "http://localhost:8004/v1"     # 本地推理端点
api_key_env = "GLM_API_KEY"               # 或使用 "local" 表示无需认证
model = "glm-5.2"
compiler = "glm_52"
max_context_tokens = 1_000_000
# 上下文降级由 AutoCompactor 内部处理

[circuit_breaker.models.glm_52]
max_consecutive_failures = 5               # 标准阈值
cooldown_seconds = 60.0                    # 1M 上下文需更长的冷却时间
```

### 预算优先配置

```toml
[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
max_context_tokens = 200_000              # 限制为 200K 以节省成本
thinking_mode = "high"                    # 避免 Max 模式以节省 token
# 注意：preserved_thinking_enabled 和 vision_coordination_enabled 是
# create_provider() 的 kwargs，不是 TOML 字段。不传即可禁用。
```

---

## 性能提示与稳定性考量

### 性能提示

1. **默认使用 High 思维** — 将 Max 思维留给真正复杂的决策
2. **高效组织提示词** — 将关键信息放在开头和结尾
3. **启用前缀缓存** — 跨请求保持系统提示一致
4. **批量处理相关查询** — 在一个请求中处理多个相关项，而非多个小请求
5. **设置适当的超时** — 1M 上下文请求可能需要 30-60 秒
6. **监控 KV 缓存利用率** — 高利用率表明缓存效率良好
7. **使用上下文降级** — 自动处理内存压力

### 稳定性考量

1. **1M 上下文稳定性** — 在完整 1M 上下文下，内存使用量很高。启用降级作为安全保障
2. **Max 思维超时风险** — Max 思维配合 1M 上下文可能超出 API 超时。配置 `timeout=300.0` 或更高
3. **5V-Turbo 可用性** — 视觉模型可能有独立的速率限制。适当配置熔断器
4. **PreservedThinking 内存** — 累积的保留痕迹消耗上下文。监控总上下文使用量
5. **长时任务检查点大小** — 在 1M 上下文下，检查点可能很大。确保有足够的磁盘空间

### 内存优化

```python
from teragent import create_provider

# 针对内存受限环境优化
provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    max_context_tokens=500_000,          # 从 1M 减少到 500K
    thinking_mode="high",                # 避免 Max 模式（减少思维 token）
    preserved_thinking_enabled=False,    # 编译器级 kwargs；禁用以节省上下文空间
    context_degradation_enabled=True,    # 编译器级 kwargs；内存紧张时自动降级
)
```

---

## 上下文降级

GLM-5.2 支持在系统内存压力下自动进行上下文降级。

### 降级工作原理

```
1M Context (full capacity)
    ↓ Memory pressure detected (阈值超过)
200K Context (degraded mode)
    ↓ Memory pressure resolved
1M Context (recovered)
```

### 配置降级

> **注意：** 上下文降级由 AutoCompactor 内部处理。`context_degradation_enabled`、`context_degradation_threshold`、`context_degradation_target` 和 `context_degradation_recovery_threshold` **不是**有效的 TOML 驱动配置字段——它们是 `create_provider()` 的编译器级 kwargs。在 TOML 中，只需设置 `max_context_tokens = 1_000_000`，AutoCompactor 会自动处理降级。

### 降级期间发生的情况

1. **触发**：内存利用率超过配置的阈值
2. **降级**：最大上下文从 1M 减少到 200K
3. **上下文压缩**：现有上下文被压缩以适应 200K
4. **信息保留**：GLM52Compiler 保留最重要的上下文（系统提示、最近的消息、工具定义）
5. **恢复**：当内存压力缓解时，系统可以恢复到 1M 模式
6. **日志记录**：所有降级事件都会记录时间戳和内存统计信息

### 监控降级

```python
from teragent.context import ContextWindow

# The ContextWindow tracks utilization
utilization = context_window.usage_ratio()
if utilization > 0.9:
    print("⚠️ Context utilization high, consider enabling auto-compaction")
```

---

## 与流水线阶段的集成

### GLM-5.2 作为 Plan Driver

GLM-5.2 凭借 1M 上下文和双思维模式，作为 Plan Driver 表现出色：

```toml
[execution.pipeline]
plan_driver = "openai_compatible.glm_52"  # 1M 上下文用于全面规划
```

### GLM-5.2 用于所有阶段（超上下文配置）

```toml
[execution.pipeline.profiles.ultra_context]
description = "GLM-5.2 for everything — maximum context and reasoning"
design_driver = "openai_compatible.glm_52"
plan_driver = "openai_compatible.glm_52"
execute_driver = "openai_compatible.glm_52"
review_driver = "openai_compatible.glm_52"
```

### 混合流水线中的 GLM-5.2

```toml
[execution.pipeline]
design_driver = "openai_compatible.deepseek_v4_pro"  # V4-Pro 用于设计
plan_driver = "openai_compatible.glm_52"              # GLM-5.2 用于规划（1M 上下文）
execute_driver = "openai_compatible.deepseek_v4_flash" # V4-Flash 用于执行
review_driver = "openai_compatible.glm_5"             # GLM-5 用于审查（200K 足够）
```

---

## 监控与可观测性

### 关键监控指标

| 指标 | 工具 | 阈值 |
|--------|------|-----------|
| 请求延迟 | TerAgent 日志 | 1M 上下文 > 60 秒 |
| 思维模式 token 使用量 | CostTracker | Max 模式：约为 High 的 3-5 倍 |
| 上下文降级事件 | ContextWindow / AutoCompactor | 任何事件都值得关注 |
| 5V-Turbo 可用性 | 熔断器状态 | Open = 视觉不可用 |
| 检查点大小 | 文件系统 | 每个检查点 > 100MB |

### 设置告警

```python
from teragent.context import ContextWindow
from teragent.reliability.circuit_breaker import ModelCircuitBreakerManager

# 监控上下文利用率
utilization = context_window.usage_ratio()
if utilization > 0.9:
    print("⚠️ Context utilization high, consider enabling auto-compaction")

# 监控熔断器
breaker_manager = ModelCircuitBreakerManager()

# 自定义告警逻辑
async def check_health():
    utilization = context_window.usage_ratio()
    if utilization > 0.9:
        print("⚠️ Context utilization high, consider enabling auto-compaction")

    breaker_state = breaker_manager.get_state("glm_52")
    if breaker_state == "open":
        print("🔴 GLM-5.2 circuit breaker is open — model unavailable")

    breaker_5v = breaker_manager.get_state("glm_5v_turbo")
    if breaker_5v == "open":
        print("⚠️ 5V-Turbo circuit breaker is open — vision coordination degraded")
```

---

## 故障排除

### 1M 上下文请求超时

**原因**：1M 上下文 + Max 思维可能超出 API 超时。

**解决方案**：增加超时时间，大上下文优先使用 High 思维：

```python
from teragent.core.adapters.openai_compatible import OpenAICompatibleAdapter

adapter = OpenAICompatibleAdapter(
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key="your-key",
    timeout=300.0,            # 1M 上下文设置 5 分钟
)
```

### 上下文降级触发过于频繁

**原因**：推理端点内存不足以支持 1M 上下文。

**解决方案**：
1. 将 `max_context_tokens` 减少到 500K 或 200K
2. 确保推理端点有足够资源支持 1M 上下文
4. 启用前缀缓存以减少内存开销

### PreservedThinking 消耗过多 Token

**原因**：累积的保留痕迹占用了大量上下文。

**解决方案**：
1. 限制保留痕迹的数量（默认：10）
2. 开始新子任务时手动清除旧痕迹
3. 对不需要 PreservedThinking 的任务禁用该功能

### 5V-Turbo 协同返回效果不佳

**原因**：视觉模型可能误解复杂的 UI 原型图或图表。

**解决方案**：
1. 在图像之外提供额外的文本上下文
2. 使用更高分辨率的图像
3. 将复杂原型图拆分为更小的组件
4. 对复杂视觉任务回退到 M3（原生多模态）

### GLM-5.2 未使用 1M 上下文

**原因**：推理端点可能不支持 1M 上下文。

**解决方案**：
1. 确保配置中 `max_context_tokens = 1_000_000`
2. 确认推理端点支持 1M 上下文并已正确配置
3. 确保 driver 配置中 `max_context_tokens = 1_000_000`
3. 检查端点响应中是否有上下文截断警告

---

*本指南是 TerAgent 文档的一部分。完整的四模型适配指南请参阅[适配指南](adaptation_guide.md)。长时任务详情请参阅[长时任务指南](long_horizon_guide.md)。*
