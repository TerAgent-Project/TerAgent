# 四模型适配指南

> **⚠️ 注意：** 本文件位于 `docs/en/zh/` 目录下，这是一个中文文档放在了英文文档目录中。如果您需要英文版本，请参阅 [Four-Model Adaptation Guide](../adaptation_guide.md)。此文件可能应移动到 `docs/zh/` 目录。

本文档介绍如何配置和使用 TerAgent 的四模型深度适配架构，包括 **DeepSeek V4**、**MiniMax M3**、**GLM-5** 和 **GLM-5.2**。

---

## 目录

- [概述](#概述)
- [模型能力一览](#模型能力一览)
- [四模型配置](#四模型配置)
- [如何为每个场景选择模型](#如何为每个场景选择模型)
- [各模型特性指南](#各模型特性指南)
  - [DeepSeek V4 — 思考模式与缓存感知](#deepseek-v4--思考模式与缓存感知)
  - [MiniMax M3 — 多模态与桌面操作](#minimax-m3--多模态与桌面操作)
  - [GLM-5 — 长程自治任务](#glm-5--长程自治任务)
  - [GLM-5.2 — 1M 上下文与双思考模式](#glm-52--1m-上下文与双思考模式)
- [从单模型迁移到多模型](#从单模型迁移到多模型)
- [常用模式与配置模板](#常用模式与配置模板)
- [最佳实践](#最佳实践)
- [常见问题排查](#常见问题排查)
- [快速参考](#快速参考)

---

## 概述

TerAgent 的四模型架构为每个模型分配最擅长的任务，在最大化质量的同时最小化成本：

| 模型 | 定位 | 核心优势 |
|------|------|----------|
| **DeepSeek V4-Flash** | 轻量级任务 | 快速响应、低成本、1M 上下文 |
| **DeepSeek V4-Pro** | 复杂推理 | 深度思考模式、1M 上下文、缓存感知 |
| **MiniMax M3** | 多模态与桌面 | 图像/视频理解、桌面自动化、1M 上下文 |
| **GLM-5** | 长程与审查 | 8 小时自治任务、自评估、策略切换、200K 上下文 |
| **GLM-5.2** | 超长上下文与双思考 | 1M 上下文、High/Max 双思考、PreservedThinking、5V-Turbo 视觉协调 |

**ModelRouter** 通过 6 步决策流自动选择最优模型：

1. **意图** — 根据任务意图匹配默认模型
2. **多模态** — 将视觉/视频内容路由到 M3
3. **上下文长度** — 排除上下文窗口不足的模型
4. **长程任务** — 将长时间任务路由到 GLM-5 或 GLM-5.2
5. **成本** — 预算紧张时降级到更便宜的模型
6. **降级** — 主模型不可用时回退

### 架构图

```
┌──────────────────────────────────────────────────────────────────────┐
│                      TerAgent 四模型层                               │
│                                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────┐  ┌───────────────┐ │
│  │ DeepSeek V4  │  │  MiniMax M3  │  │  GLM-5  │  │   GLM-5.2     │ │
│  │ Flash + Pro  │  │  多模态      │  │ 8h 自治 │  │ 1M + 双思考   │ │
│  │ 1M 上下文    │  │  桌面操作    │  │ 200K    │  │ 模式          │ │
│  └──────┬───────┘  └──────┬───────┘  └────┬────┘  └───────┬───────┘ │
│         │                  │               │                │         │
│  ┌──────▼──────────────────▼───────────────▼────────────────▼───────┐│
│  │                     ModelRouter（6 维度路由）                      ││
│  │  意图 → 多模态 → 上下文 → 长程 → 成本 → 降级                     ││
│  └──────────────────────────────────────────────────────────────────┘│
│         │                                                            │
│  ┌──────▼───────────────────────────────────────────────────────────┐│
│  │            PipelineManager（设计→规划→执行→审查）                  ││
│  └──────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────┘
```

---

## 模型能力一览

| 特性 | V4-Flash | V4-Pro | M3 | GLM-5 | GLM-5.2 |
|------|:--------:|:------:|:--:|:-----:|:-------:|
| 上下文窗口 | 1M | 1M | 1M | 200K | **1M** |
| 最大输出 | 384K | 384K | 384K | 128K | 128K |
| 思考模式 | auto/quick | deep | — | deep | **High/Max** |
| 多模态 | ❌ | ❌ | ✅ | ❌ | ❌（通过 5V） |
| 桌面操作 | ❌ | ❌ | ✅ | ❌ | ❌ |
| 长程任务 | ❌ | ❌ | ❌ | ✅（8h） | ✅（8h+） |
| 缓存感知 | ✅ | ✅ | MSA | ❌ | ✅ |
| PreservedThinking | ❌ | ❌ | ❌ | ❌ | ✅ |
| 视觉协调 | ❌ | ❌ | ❌ | ❌ | ✅（5V-Turbo） |
| 相对成本 | ★★★★★ | ★★ | ★★★★ | ★★★ | ★★★ |

---

## 四模型配置

### 完整 agent.toml 示例

```toml
# =============================================================================
# DeepSeek V4 — Flash（轻量级任务，默认模型）
# =============================================================================
[drivers.openai_compatible.deepseek_v4_flash]
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-v4-flash"
compiler = "deepseek_v4"
compiler_variant = "flash"              # Flash：极简提示词，快速响应
max_context_tokens = 1_000_000          # V4 支持 1M 上下文
max_output_tokens = 384_000             # V4 最大 384K 输出
thinking_mode = "auto"                  # auto/deep/quick
cache_aware = true                      # 启用缓存感知（12 倍价格差异）

# =============================================================================
# DeepSeek V4 — Pro（复杂任务）
# =============================================================================
[drivers.openai_compatible.deepseek_v4_pro]
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-v4-pro"
compiler = "deepseek_v4"
compiler_variant = "pro"                # Pro：完整提示词 + 推理引导
max_context_tokens = 1_000_000
max_output_tokens = 384_000
thinking_mode = "deep"                  # Pro 默认深度推理
cache_aware = true

# =============================================================================
# MiniMax M3 — 多模态模型
# =============================================================================
[drivers.openai_compatible.minimax_m3]
base_url = "https://api.minimaxi.com/v1"
api_key_env = "MINIMAX_API_KEY"
model = "minimax-m3"
compiler = "minimax_m3"
max_context_tokens = 1_000_000          # M3 支持 1M 上下文
max_output_tokens = 384_000
multimodal_enabled = true               # 启用多模态（图像+视频）
desktop_enabled = true                  # 启用桌面操作
msa_efficient = true                    # MSA 全文注入模式

# =============================================================================
# GLM-5 — 长程任务模型
# =============================================================================
[drivers.openai_compatible.glm_5]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5"
compiler = "glm_5"
max_context_tokens = 200_000            # GLM-5 200K 上下文
max_output_tokens = 128_000
thinking_mode = "deep"                  # GLM-5 默认深度推理
long_horizon_enabled = true             # 启用长程模式（8h 自治）

# =============================================================================
# GLM-5.2 — 1M 上下文 + 双思考模式
# =============================================================================
[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
max_context_tokens = 1_000_000          # GLM-5.2 支持 1M 上下文
max_output_tokens = 128_000
thinking_mode = "high"                  # 默认："high"（标准深度推理）
dual_thinking_enabled = true            # 启用 High/Max 双思考模式
preserved_thinking_enabled = true       # 启用 PreservedThinking（编码计划保持）
vision_coordination_enabled = true      # 启用 5V-Turbo 视觉协调
long_horizon_enabled = true             # GLM-5.2 也支持长程任务
context_degradation_enabled = true      # 压力下自动降级 1M → 200K

# =============================================================================
# 执行流水线 — 多模型协作
# =============================================================================
[execution.pipeline]
design_driver = "openai_compatible.deepseek_v4_pro"    # 设计：V4-Pro
plan_driver = "openai_compatible.glm_52"               # 规划：GLM-5.2（1M 上下文）
execute_driver = "openai_compatible.deepseek_v4_flash"  # 执行：V4-Flash
review_driver = "openai_compatible.glm_5"              # 审查：GLM-5（深度审查）

# =============================================================================
# 智能路由配置
# =============================================================================
[routing]
multimodal_driver = "openai_compatible.minimax_m3"     # 多模态 → M3
desktop_driver = "openai_compatible.minimax_m3"        # 桌面 → M3
long_horizon_driver = "openai_compatible.glm_52"      # 长程 → GLM-5.2
ultra_context_driver = "openai_compatible.glm_52"     # >200K 上下文 → GLM-5.2

[routing.monthly_budget]
limit_cny = 500.0                      # 月度预算上限 ¥500
warning_threshold = 0.8                 # 80% 时警告
auto_downgrade = true                   # 超预算时自动降级

# =============================================================================
# 流水线命名配置
# =============================================================================
[execution.pipeline.profiles.budget]
description = "最大节约：全部使用 V4-Flash"
design_driver = "openai_compatible.deepseek_v4_flash"
plan_driver = "openai_compatible.deepseek_v4_flash"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.deepseek_v4_flash"

[execution.pipeline.profiles.multimodal]
description = "多模态模式：全部使用 M3"
design_driver = "openai_compatible.minimax_m3"
plan_driver = "openai_compatible.minimax_m3"
execute_driver = "openai_compatible.minimax_m3"
review_driver = "openai_compatible.minimax_m3"

[execution.pipeline.profiles.quality]
description = "质量优先：V4-Pro 设计/审查，GLM-5.2 规划"
design_driver = "openai_compatible.deepseek_v4_pro"
plan_driver = "openai_compatible.glm_52"
execute_driver = "openai_compatible.glm_52"
review_driver = "openai_compatible.deepseek_v4_pro"

[execution.pipeline.profiles.ultra_context]
description = "超长上下文：GLM-5.2 全流水线（1M 上下文）"
design_driver = "openai_compatible.glm_52"
plan_driver = "openai_compatible.glm_52"
execute_driver = "openai_compatible.glm_52"
review_driver = "openai_compatible.glm_52"
```

### 环境变量

在 `.env` 文件或环境中设置：

```bash
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxx        # V4-Flash 和 V4-Pro 必需
MINIMAX_API_KEY=xxxxxxxxxxxx            # M3 必需
GLM_API_KEY=xxxxxxxxxxxx.xxxxxx         # GLM-5 和 GLM-5.2 必需
```

> **注意：** GLM-5 和 GLM-5.2 共享同一个智谱 AI 的 API 密钥，通过相同的 `base_url` 访问。

---

## 如何为每个场景选择模型

### 按任务类型

| 任务类型 | 推荐模型 | 原因 |
|---------|---------|------|
| 聊天/问答 | V4-Flash | 快速、低成本、质量好 |
| 代码生成 | V4-Flash | 响应快、成本低 |
| 调试 | V4-Flash | 快速迭代 |
| 设计/架构 | V4-Pro | 深度推理、完整上下文 |
| 规划（大范围） | GLM-5.2 | 1M 上下文、双思考 |
| 代码审查 | GLM-5 | 深度分析、逻辑验证 |
| 图像理解 | M3 | 原生多模态 |
| 视频处理 | M3 | 原生视频理解 |
| 桌面自动化 | M3 | 专用桌面 API |
| 长程任务 | GLM-5.2 | 1M 上下文 + 8h+ 自治 |
| 超大代码库（>200K） | GLM-5.2 或 M3 | 1M 上下文窗口 |
| 视觉+代码任务 | GLM-5.2 + 5V-Turbo | 视觉协调编码 |
| 编码计划保持 | GLM-5.2 | PreservedThinking 特性 |
| 多步推理 | GLM-5.2（Max 模式） | 最深推理能力 |

### 按上下文大小

| 上下文大小 | 推荐模型 | 降级方案 |
|-----------|---------|---------|
| < 50K tokens | V4-Flash（性价比高） | V4-Pro |
| 50K–200K tokens | V4-Pro 或 GLM-5 | V4-Flash |
| 200K–500K tokens | GLM-5.2 或 M3 | V4-Pro |
| 500K–1M tokens | GLM-5.2 | M3 |

### GLM-5 与 GLM-5.2 选择矩阵

| 因素 | 选 GLM-5 | 选 GLM-5.2 |
|------|---------|-----------|
| 需要的上下文窗口 | ≤ 200K | > 200K（最高 1M） |
| 推理深度 | 标准深度思考 | High/Max 双思考 |
| 编码+视觉 | 不需要 | 需要 5V-Turbo 协调 |
| PreservedThinking | 不需要 | 需要编码计划保持 |
| 成本敏感度 | 较高（每 token 更便宜） | 较低（高级特性） |
| 任务时长 | ≤ 8 小时 | 8h+（支持降级） |
| 代码库大小 | 小-中型 | 大型-超大型 |

---

## 各模型特性指南

### DeepSeek V4 — 思考模式与缓存感知

DeepSeek V4 支持三种思考模式，控制推理深度：

| 模式 | 编译器变体 | 描述 |
|------|-----------|------|
| `auto` | Flash | 根据复杂度自动决定是否深度思考 |
| `quick` | Flash | 最少推理的快速响应 |
| `deep` | Pro | 带完整思维链的扩展推理 |

```python
from teragent import create_provider

# 快速思考 — 快速响应
flash_provider = create_provider(
    compiler="deepseek_v4",
    adapter="openai_compatible",
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com",
    api_key_env="DEEPSEEK_API_KEY",
)

# 深度思考 — 扩展推理
pro_provider = create_provider(
    compiler="deepseek_v4",
    adapter="openai_compatible",
    model="deepseek-v4-pro",
    base_url="https://api.deepseek.com",
    api_key_env="DEEPSEEK_API_KEY",
)
```

**缓存感知**是 V4 的关键特性。DeepSeek API 的缓存命中和未命中之间有 12 倍价格差异：

- `DeepSeekV4Compiler` 自动将系统提示和工具定义冻结在消息列表开头
- 在驱动配置中启用 `cache_aware = true`
- 保持系统提示和工具定义跨请求一致，最大化缓存命中率

### MiniMax M3 — 多模态与桌面操作

MiniMax M3 提供原生多模态能力，支持 Anthropic 兼容接口：

```python
from teragent import TAPRequest, create_provider
from teragent.core.tap import MultimodalContent

m3_provider = create_provider(
    compiler="minimax_m3",
    adapter="minimax_native",
    model="minimax-m3",
    base_url="https://api.minimaxi.com/v1",
    api_key_env="MINIMAX_API_KEY",
)

# 图像理解
request = TAPRequest(
    instruction="描述这张图片的内容",
    multimodal_context=[
        MultimodalContent(type="image_url", image_url={"url": "https://example.com/photo.jpg"}),
    ],
)

# 视频处理
request = TAPRequest(
    instruction="总结这个视频的关键内容",
    multimodal_context=[
        MultimodalContent(type="video_url", video_url={"url": "https://example.com/video.mp4"}),
    ],
)
```

**M3 核心特性：**
- **7 种桌面动作**：截图、点击、输入文本、滚动、快捷键、移动鼠标、拖拽
- **5 层安全体系**：权限 → 安全区域 → 频率限制 → 操作上限 → 禁止快捷键
- **MSA 高效模式**：1M 上下文下全文注入，计算成本仅 1/20
- **Anthropic 兼容**：支持 `count_tokens` 和 Anthropic 消息格式
- **Token 估算**：提供多模态内容的精确 token 计数

### GLM-5 — 长程自治任务

GLM-5 擅长需要长时间自治执行的任务：

```python
from teragent import TAPRequest, create_provider
from teragent.core.tap import LongHorizonConfig

glm_provider = create_provider(
    compiler="glm_5",
    adapter="openai_compatible",
    model="glm-5",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
)

# 长程任务配置
request = TAPRequest(
    instruction="实现一个完整的用户管理系统，包含认证、角色和审计日志",
    long_horizon_config=LongHorizonConfig(
        max_duration_hours=4,
        checkpoint_interval_minutes=15,
        evaluation_interval_steps=10,
    ),
)
```

**GLM-5 核心特性：**
- **8 小时自治执行**，带目标分解（DAG 拓扑）
- **自评估**：定期评估目标一致性、输出质量和瓶颈检测
- **策略切换**：自动检测停滞并切换方法（分解/回溯/跳过/重规划）
- **检查点恢复**：每 N 分钟自动保存状态快照
- **上下文窗口**：200K tokens（升级到 GLM-5.2 获得 1M 上下文）

### GLM-5.2 — 1M 上下文与双思考模式

GLM-5.2 是超长上下文和高级推理的旗舰模型：

```python
from teragent import create_provider, TAPRequest
from teragent.core.tap import LongHorizonConfig

glm52_provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="GLM_API_KEY",
)

# 1M 上下文 + 双思考模式
request = TAPRequest(
    instruction="分析整个代码库并设计微服务迁移方案",
    long_horizon_config=LongHorizonConfig(
        max_duration_hours=6,
        checkpoint_interval_minutes=20,
    ),
)
```

**GLM-5.2 核心特性：**

| 特性 | 描述 |
|------|------|
| **1M 上下文窗口** | 单次请求处理最多 1,000,000 tokens |
| **High 思考** | 标准深度推理模式（默认） |
| **Max 思考** | 最大推理深度，解决最难的问题 |
| **PreservedThinking** | 跨编码会话保持推理痕迹，确保计划连续性 |
| **5V-Turbo 协调** | 与 GLM-5V-Turbo 协作进行视觉→代码→验证循环 |
| **上下文降级** | 内存压力下自动从 1M 降级到 200K |
| **长程支持** | 扩展自治任务（8h+）带检查点恢复 |

**双思考模式使用：**

```python
from teragent import create_provider

# High 思考 — 标准深度推理（默认）
provider_high = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    thinking_mode="high",    # 速度与深度平衡
)

# Max 思考 — 最大推理深度
provider_max = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    thinking_mode="max",     # 最深推理，更慢但最彻底
)
```

**PreservedThinking 编码计划保持：**

```python
from teragent import create_provider, TAPRequest

provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    preserved_thinking_enabled=True,
)

# 第一次请求：创建编码计划
plan_request = TAPRequest(
    instruction="设计一个支持 WebSocket 的实时聊天系统架构",
    meta={"intent": "plan"},
)

# PreservedThinking 特性保留推理痕迹
# 后续代码生成将保持与计划一致
```

**5V-Turbo 视觉协调：**

```python
from teragent import create_provider, TAPRequest
from teragent.core.tap import MultimodalContent

provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    vision_coordination_enabled=True,
)

# 视觉→代码→验证循环
request = TAPRequest(
    instruction="查看这个 UI 设计稿并实现前端代码",
    multimodal_context=[
        MultimodalContent(
            type="image_url",
            image_url={"url": "https://example.com/mockup.png"},
        ),
    ],
)
# GLM-5.2 与 GLM-5V-Turbo 协调进行视觉分析
# 然后基于视觉理解生成代码
```

---

## 从单模型迁移到多模型

### 步骤 1：添加新模型驱动

保留现有单模型配置，逐步添加新驱动：

```toml
# 保留现有配置
[drivers.openai_compatible.glm_5]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5"
compiler = "glm_5"

# 添加 V4-Flash
[drivers.openai_compatible.deepseek_v4_flash]
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-v4-flash"
compiler = "deepseek_v4"
compiler_variant = "flash"

# 添加 V4-Pro
[drivers.openai_compatible.deepseek_v4_pro]
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-v4-pro"
compiler = "deepseek_v4"
compiler_variant = "pro"

# 添加 M3
[drivers.openai_compatible.minimax_m3]
base_url = "https://api.minimaxi.com/v1"
api_key_env = "MINIMAX_API_KEY"
model = "minimax-m3"
compiler = "minimax_m3"
multimodal_enabled = true
desktop_enabled = true

# 添加 GLM-5.2
[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
max_context_tokens = 1_000_000
dual_thinking_enabled = true
preserved_thinking_enabled = true
vision_coordination_enabled = true
```

### 步骤 2：更新流水线配置

将单模型流水线替换为多模型分配：

```toml
# 迁移前：全部使用同一模型
[execution.pipeline]
design_driver = "openai_compatible.glm_5"
plan_driver = "openai_compatible.glm_5"
execute_driver = "openai_compatible.glm_5"
review_driver = "openai_compatible.glm_5"

# 迁移后：每个阶段使用专用模型
[execution.pipeline]
design_driver = "openai_compatible.deepseek_v4_pro"
plan_driver = "openai_compatible.glm_52"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.glm_5"
```

### 步骤 3：添加路由配置

添加 `[routing]` 部分实现自动模型选择：

```toml
[routing]
multimodal_driver = "openai_compatible.minimax_m3"
desktop_driver = "openai_compatible.minimax_m3"
long_horizon_driver = "openai_compatible.glm_52"
ultra_context_driver = "openai_compatible.glm_52"

[routing.monthly_budget]
limit_cny = 500.0
warning_threshold = 0.8
auto_downgrade = true
```

### 步骤 4：添加降级链

将 GLM-5.2 加入降级链：

```toml
[degradation]
heavy = ["deepseek_v4_pro", "glm_52", "glm_5", "deepseek_v4_flash"]
multimodal = ["minimax_m3", "glm_52", "deepseek_v4_pro"]
ultra_context = ["glm_52", "deepseek_v4_pro", "minimax_m3"]
default = ["deepseek_v4_pro", "glm_52", "glm_5", "deepseek_v4_flash"]
```

### 步骤 5：增量测试

1. 先用 `budget` 配置（全部 V4-Flash）验证新模型
2. 切换到 `default` 配置测试各阶段路由
3. 添加 M3 测试多模态任务
4. 启用 GLM-5 测试长程任务
5. 启用 GLM-5.2 测试超长上下文和双思考
6. 临时禁用一个模型测试降级链
7. 用大型代码库验证 `ultra_context` 配置

---

## 常用模式与配置模板

### 模式 1：开发环境预算控制

```toml
[execution.pipeline]
design_driver = "openai_compatible.deepseek_v4_flash"
plan_driver = "openai_compatible.deepseek_v4_flash"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.deepseek_v4_flash"

[routing.monthly_budget]
limit_cny = 100.0        # 开发环境紧预算
auto_downgrade = true     # 超预算自动降级
```

### 模式 2：生产环境质量优先

```toml
[execution.pipeline]
design_driver = "openai_compatible.deepseek_v4_pro"
plan_driver = "openai_compatible.glm_52"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.glm_5"
```

### 模式 3：超大代码库分析

```toml
[execution.pipeline.profiles.codebase_analysis]
description = "分析超过 200K tokens 的代码库"
design_driver = "openai_compatible.glm_52"
plan_driver = "openai_compatible.glm_52"
execute_driver = "openai_compatible.glm_52"
review_driver = "openai_compatible.glm_52"
```

### 模式 4：视觉辅助开发

```python
from teragent import create_provider, TAPRequest
from teragent.core.tap import MultimodalContent

# 使用 GLM-5.2 + 5V-Turbo 进行视觉→代码任务
provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    vision_coordination_enabled=True,
)

request = TAPRequest(
    instruction="分析这个错误截图并修复代码",
    multimodal_context=[
        MultimodalContent(
            type="image_url",
            image_url={"url": "https://ci.example.com/error.png"},
        ),
    ],
)
```

### 模式 5：运行时配置切换

```python
from teragent.router import PipelineManager, PipelineProfile

# 切换到超长上下文配置分析大型代码库
pipeline_manager.set_active_profile("ultra_context")

# 切换回默认配置进行常规任务
pipeline_manager.set_active_profile("default")

# 动态创建自定义配置
pipeline_manager.register_profile(PipelineProfile(
    name="glm52_review",
    description="GLM-5.2 全流水线（1M 上下文审查）",
    design_driver="openai_compatible.glm_52",
    plan_driver="openai_compatible.glm_52",
    execute_driver="openai_compatible.glm_52",
    review_driver="openai_compatible.glm_52",
))
```

### 模式 6：双思考模式选择

```python
from teragent import create_provider

# 大多数任务使用 "high" 思考（速度与深度平衡）
provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    thinking_mode="high",
)

# 关键决策时切换到 "max" 思考
request = TAPRequest(
    instruction="决定是重构还是重写这个模块",
    meta={"thinking_mode": "max"},  # 按请求覆盖
)
```

---

## 最佳实践

1. **新项目从 V4-Flash 开始**，然后逐步升级特定阶段到 Pro、GLM-5 或 GLM-5.2
2. **开发测试使用 `budget` 配置**，最小化成本
3. **启用月度预算**，防止长时间会话产生意外费用
4. **让路由器处理模型选择** — 仅在有特定需求时手动覆盖
5. **200K 以下上下文的长程任务使用 GLM-5**；超过 200K 时使用 GLM-5.2
6. **仅在 High 思考不足时使用 GLM-5.2 Max 模式** — Max 更慢、更贵
7. **多模态内容路由到 M3** — 其他模型会将视觉内容降级为文本描述
8. **利用 V4 缓存感知** — 保持系统提示和工具定义跨请求一致
9. **多步编码任务使用 PreservedThinking**，保持计划连续性
10. **启用 GLM-5.2 上下文降级**，优雅处理内存压力（1M → 200K）
11. **定期监控熔断器状态** — 检查 `ModelCircuitBreakerManager.get_all_states()`
12. **配置降级链** — 确保模型不可用时不会死锁
13. **不同环境使用不同流水线配置**（开发/预发布/生产）
14. **测试降级路径** — 故意禁用一个模型验证回退
15. **考虑成本-性能权衡** — GLM-5.2 的 1M 上下文强大但每 token 更贵

---

## 常见问题排查

### 模型未找到错误

```
KeyError: "openai_compatible.glm_52"
```

**解决方案**：确保 `agent.toml` 中存在对应的驱动部分且 API 密钥环境变量已设置。检查编译器名称 `glm_52` 是否与注册的编译器匹配。

### GLM-5 上下文长度超限

GLM-5 的上下文窗口为 200K tokens。如果请求超过此限制：

**解决方案**：ModelRouter 会自动将 >200K 上下文的请求路由到 GLM-5.2 或 V4-Pro/M3。确保 `[routing]` 配置了 `ultra_context_driver = "openai_compatible.glm_52"`。

### GLM-5.2 1M 上下文内存溢出

如果 1M 上下文消耗过多内存：

**解决方案**：在 GLM-5.2 驱动配置中启用 `context_degradation_enabled = true`。这允许在内存压力下自动降级到 200K 上下文。同时确保推理服务器有足够的 NPU 内存（推荐：Ascend 910B ×2 用于 1M 上下文）。

### 双思考模式未激活

如果 High/Max 思考模式未正确切换：

**解决方案**：确认驱动配置中 `dual_thinking_enabled = true`。检查 `GLM52Compiler` 已注册并选中。思考模式可通过驱动配置的 `thinking_mode` 设置，或通过 `meta={"thinking_mode": "max"}` 按请求覆盖。

### 5V-Turbo 视觉协调失败

如果 GLM-5V-Turbo 与 GLM-5.2 的视觉协调不工作：

**解决方案**：确保 `vision_coordination_enabled = true` 已设置。检查 GLM-5V-Turbo 服务可访问。验证 `GLM52Compiler` 支持视觉协调模式。5V-Turbo 不可用时，系统会降级到纯文本分析。

### PreservedThinking 上下文丢失

如果 PreservedThinking 在会话间丢失上下文：

**解决方案**：PreservedThinking 设计用于会话内连续性，不跨会话持久化。跨会话工作时，在新会话的目标描述中包含计划摘要。确认 `preserved_thinking_enabled = true` 已设置。

### 多模态内容未处理

如果 M3 不可用，其他编译器会将多模态内容降级为文本描述。

**解决方案**：确保 M3 驱动已配置且 API 密钥有效。检查路由配置。对于 GLM-5.2 视觉任务，启用 `vision_coordination_enabled = true`。

### 预算控制下成本仍然高

**可能原因**：
- 预算上限过高
- `auto_downgrade` 设为 `false`
- 大量请求绕过路由器
- GLM-5.2 Max 思考模式使用过多

**解决方案**：设置合理的 `limit_cny`，启用 `auto_downgrade`，确保所有请求通过 ModelRouter，谨慎使用 Max 思考模式。

### 熔断器过于频繁触发

```python
from teragent.reliability.circuit_breaker import ModelCircuitBreakerManager, ModelBreakerConfig

# 为 GLM-5.2 自定义熔断器阈值
manager = ModelCircuitBreakerManager(configs=[
    ModelBreakerConfig(
        model_name="glm_52",
        max_consecutive_failures=10,     # 对 1M 上下文更宽容
        cooldown_seconds=60.0,           # 更长的恢复冷却时间
    ),
])
```

### 流水线配置未生效

**解决方案**：确保已调用 `set_active_profile()` 且配置名称完全匹配。使用 `pipeline_manager.list_profiles()` 查看可用配置。

---

## 快速参考

### 模型选择速查

```
需要速度？          → V4-Flash
需要深度？          → V4-Pro（deep）或 GLM-5.2（Max）
需要视觉？          → M3
需要桌面操作？      → M3
需要 8h 自治？     → GLM-5 或 GLM-5.2
需要 >200K 上下文？ → GLM-5.2
需要视觉+代码？    → GLM-5.2（5V-Turbo）
需要编码计划？     → GLM-5.2（PreservedThinking）
预算有限？         → V4-Flash（budget 配置）
```

### 关键配置路径

```
模型驱动：   [drivers.openai_compatible.<name>]
流水线：     [execution.pipeline]
路由：       [routing]
预算：       [routing.monthly_budget]
熔断器：     [circuit_breaker.models.<name>]
降级链：     [degradation]
长程任务：   [long_horizon]
命名配置：   [execution.pipeline.profiles.<name>]
```

### 端口分配（本地部署）

| 端口 | 服务 |
|------|------|
| 8001 | GLM-5 推理 |
| 8002 | DeepSeek V4 Flash 推理 |
| 8003 | MiniMax M3 推理 |
| 8004 | GLM-5.2 推理 |
| 8005 | DeepSeek V4 Pro 推理 |
| 8010 | 桌面操作 API |

---

*本指南是 TerAgent 文档的一部分。如需模型专属深入指南，请参阅 [GLM-5.2 指南](../glm_52_guide.md)、[长程任务指南](../long_horizon_guide.md) 和 [多模态指南](../multimodal_guide.md)。*
