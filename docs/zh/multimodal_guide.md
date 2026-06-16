# 多模态使用指南

本指南涵盖 TerAgent 的多模态能力，包括 MiniMax M3 的原生多模态支持、图像/视频处理、桌面自动化，以及 GLM-5V-Turbo + GLM-5.2 视觉协调。

---

## 目录

- [M3 多模态能力概览](#m3-多模态能力概览)
- [图像输入](#图像输入)
- [视频输入](#视频输入)
- [桌面操作工具](#桌面操作工具)
- [DesktopContext 使用](#desktopcontext-使用)
- [视觉协调：GLM-5V-Turbo + GLM-5.2](#视觉协调glm-5v-turbo--glm-52)
- [配置多模态路由](#配置多模态路由)
- [多模态编译与 Token 估算](#多模态编译与-token-估算)
- [视觉任务最佳实践](#视觉任务最佳实践)
- [降级行为](#降级行为)
- [与审查流水线集成](#与审查流水线集成)

---

## M3 多模态能力概览

MiniMax M3 提供原生多模态支持，具备以下能力：

| 能力 | 描述 | 性能 |
|------|------|------|
| **图像理解** | 分析来自 URL 或 base64 的图像 | 视觉问答高准确率 |
| **视频理解** | 原生处理视频内容 | 支持 MP4、AVI、MOV 等格式 |
| **桌面自动化** | 截图 → 分析 → 点击/输入/滚动 | 7 种操作类型，5 层安全机制 |
| **MSA 高效模式** | 1M 上下文全文注入 | 通过稀疏注意力实现 1/20 计算成本 |
| **Agent 编程** | 代码生成与理解 | SWE-Bench Pro 59.0% |
| **浏览增强** | 网页信息检索 | BrowseComp 83.5 |

**关键规格：**
- 上下文窗口：1,000,000 tokens
- 最大输出：384,000 tokens
- 支持混合内容：同一请求中可包含文本 + 图像 + 视频

### 多模态模型对比

| 特性 | MiniMax M3 | GLM-5.2 + 5V-Turbo |
|------|-----------|---------------------|
| 图像理解 | ✅ 原生支持 | ✅ 通过 5V-Turbo |
| 视频理解 | ✅ 原生支持 | ❌ |
| 桌面操作 | ✅ 原生支持 | ❌ |
| 从设计稿生成代码 | ✅ 良好 | ✅ 优秀（PreservedThinking） |
| 视觉→代码→验证 | ❌ | ✅ 协调循环 |
| 上下文窗口 | 1M | 1M |
| 最佳场景 | 视觉分析、桌面操作 | UI 实现、编码 |

---

## 图像输入

### 使用图像 URL

```python
from teragent import TAPRequest, create_provider
from teragent.core.tap import MultimodalContent

# 创建 M3 提供者
provider = create_provider(
    compiler="minimax_m3",
    adapter="minimax_native",
    model="minimax-m3",
    base_url="https://api.minimaxi.com/v1",
    api_key_env="MINIMAX_API_KEY",
)

# 分析来自 URL 的图像
request = TAPRequest(
    instruction="Describe what you see in this image in detail",
    multimodal_context=[
        MultimodalContent(
            type="image_url",
            image_url={"url": "https://example.com/screenshot.png"},
        ),
    ],
)

response = await provider.execute_tap(request)
print(response.raw_text)
```

### 使用 Base64 编码的图像

```python
import base64

# 读取图像并编码为 base64
with open("screenshot.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

request = TAPRequest(
    instruction="What UI elements are visible in this screenshot?",
    multimodal_context=[
        MultimodalContent(
            type="image_url",
            image_url={"url": f"data:image/png;base64,{image_data}"},
        ),
    ],
)

response = await provider.execute_tap(request)
```

### 在一个请求中使用多张图像

```python
request = TAPRequest(
    instruction="Compare these two screenshots and identify the differences",
    multimodal_context=[
        MultimodalContent(
            type="image_url",
            image_url={"url": "https://example.com/before.png"},
        ),
        MultimodalContent(
            type="image_url",
            image_url={"url": "https://example.com/after.png"},
        ),
    ],
)
```

### 图像与文本上下文结合

```python
request = TAPRequest(
    instruction="Based on this error screenshot and the log output, "
                "identify the root cause of the failure",
    context={
        "logs": "ERROR: Connection refused at 127.0.0.1:5432\n"
                "TRACE: Attempting reconnect in 5s...",
    },
    multimodal_context=[
        MultimodalContent(
            type="image_url",
            image_url={"url": "https://example.com/error_screen.png"},
        ),
    ],
)
```

### 图像尺寸与格式指南

| 格式 | 推荐 | 备注 |
|------|------|------|
| JPEG | ✅ | 最适合截图，token 消耗较低 |
| PNG | ✅ | 更适合包含文字的图表 |
| WebP | ⚠️ | 部分端点可能不支持 |
| GIF | ❌ | 不推荐用于分析 |
| BMP | ❌ | 不推荐，文件体积大 |

**推荐图像尺寸：** 512x512 至 2048x2048 像素。更大的图像会消耗更多 token，但收益递减。

---

## 视频输入

### 视频 URL 处理

```python
from teragent import TAPRequest, create_provider
from teragent.core.tap import MultimodalContent

request = TAPRequest(
    instruction="Summarize the key events in this video",
    multimodal_context=[
        MultimodalContent(
            type="video_url",
            video_url={"url": "https://example.com/demo.mp4"},
        ),
    ],
)

response = await provider.execute_tap(request)
```

### 视频处理提示

使用 `MiniMaxNativeAdapter` 时，MiniMax M3 编译器会自动添加视频处理提示：

```python
# 适配器自动为视频内容添加以下增强信息：
# - minimax_video_mode: "understand"（默认）或 "summarize"
# - minimax_frame_sampling: "auto"（默认）、"uniform"、"keyframe" 或 "dense"

# 这些提示在使用 MiniMaxNativeAdapter 时会自动添加。
# 无需手动设置。
```

### 支持的视频格式

| 格式 | 扩展名 |
|------|--------|
| MPEG-4 | `.mp4` |
| AVI | `.avi` |
| QuickTime | `.mov` |
| Matroska | `.mkv` |
| WebM | `.webm` |
| Flash Video | `.flv` |
| Windows Media | `.wmv` |
| MPEG-4 Part 14 | `.m4v` |

### 视频超时设置

视频处理比文本耗时更长。请设置合适的超时时间：

```python
from teragent.core.adapters.minimax_native import MiniMaxNativeAdapter

adapter = MiniMaxNativeAdapter(
    base_url="https://api.minimaxi.com/v1",
    api_key="your-api-key",
    timeout=300.0,            # 标准超时
    multimodal_timeout=600.0, # 视频扩展超时（10 分钟）
)
```

### 视频最佳实践

1. **视频时长控制在 5 分钟以内** —— 较长的视频会消耗大量 token 和时间
2. **对长视频使用 "summarize" 模式** —— 比逐帧分析更高效
3. **提供上下文** —— 告诉模型在视频中寻找什么
4. **使用直链 URL** —— 由于文件体积大，避免对视频使用 base64 编码

---

## 桌面操作工具

### 概述

`DesktopTool` 通过 M3 的视觉理解提供 7 种桌面自动化操作：

| 操作 | 描述 | 参数 |
|------|------|------|
| `screenshot` | 截取屏幕 | 无 |
| `click` | 点击指定位置 | `x`、`y`、`button`（可选） |
| `type_text` | 输入文本 | `text` |
| `scroll` | 滚动视图 | `direction`、`scroll_amount` |
| `hotkey` | 按下组合键 | `keys`（逗号分隔） |
| `move_mouse` | 移动光标 | `x`、`y` |
| `drag` | 拖拽对象 | `x`、`y`、`end_x`、`end_y` |

### 基本桌面操作

```python
from teragent.tools.desktop import DesktopTool

tool = DesktopTool()

# 1. 先截图
result = await tool.execute({"action": "screenshot"})
screenshot_base64 = result.data.get("screenshot", "")

# 2. 点击指定坐标
result = await tool.execute({
    "action": "click",
    "x": 500,
    "y": 300,
    "button": "left",
})

# 3. 输入文本
result = await tool.execute({
    "action": "type_text",
    "text": "Hello, World!",
})

# 4. 使用键盘快捷键
result = await tool.execute({
    "action": "hotkey",
    "keys": "ctrl,s",  # Ctrl+S（保存）
})

# 5. 向下滚动
result = await tool.execute({
    "action": "scroll",
    "direction": "down",
    "scroll_amount": 3,
})
```

### 桌面安全配置

```python
from teragent.tools.desktop import DesktopTool, DesktopSafetyConfig

safety = DesktopSafetyConfig(
    safe_zones=[
        (0, 0, 100, 50),    # 阻止左上角区域（例如系统菜单）
    ],
    min_interval=0.5,         # 操作间最小间隔 0.5 秒
    max_consecutive_ops=50,   # 最大连续操作数 50
    screenshot_quality=75,    # JPEG 质量（1-100）
    screenshot_format="jpeg", # "jpeg" 或 "png"
)

tool = DesktopTool(safety_config=safety)
```

### 5 层安全机制

1. **权限级别** —— 所有桌面操作需要 DESTRUCTIVE 级别的用户确认
2. **安全区域** —— 可配置的禁止点击区域（不应被点击的坐标）
3. **频率限制** —— 操作间的最小间隔（防止过快的自动化）
4. **连续操作上限** —— 强制暂停前的最大操作数
5. **屏蔽快捷键** —— 危险的组合键会被自动屏蔽：
   - Alt+F4（关闭窗口）
   - Ctrl+Alt+Delete（系统安全）
   - Alt+Tab（切换窗口）
   - Ctrl+Shift+Esc（任务管理器）
   - Win+L / Super+L（锁屏）
   - Cmd+Q（macOS 退出）

---

## DesktopContext 使用

`DesktopContext` 提供当前桌面状态的结构化信息，供 M3 分析：

### 创建 DesktopContext

```python
from teragent import TAPRequest
from teragent.core.tap import DesktopContext

# 手动构建桌面上下文
desktop_ctx = DesktopContext(
    active_window="Chrome - Google Search",
    screen_resolution=(1920, 1080),
    interactive_elements=[
        {"type": "button", "text": "Search", "x": 500, "y": 400},
        {"type": "input", "text": "", "x": 500, "y": 350, "placeholder": "Search..."},
        {"type": "link", "text": "Images", "x": 600, "y": 100},
    ],
    screenshot_base64=screenshot_data,
)

# 在 TAPRequest 中使用
request = TAPRequest(
    instruction="Click on the search input and type 'TerAgent documentation'",
    desktop_context=desktop_ctx,
)
```

### 从截图自动生成 DesktopContext

```python
from teragent.tools.desktop import DesktopTool

tool = DesktopTool()

# 截图 —— 工具可自动检测可交互元素
result = await tool.execute({"action": "screenshot"})

# 结果包含检测到的可交互元素
# 可用于构建 DesktopContext
```

### DesktopContext 与 M3 适配器配合使用

```python
from teragent.core.adapters.minimax_native import MiniMaxNativeAdapter

adapter = MiniMaxNativeAdapter(
    base_url="https://api.minimaxi.com/v1",
    api_key="your-api-key",
)

# 发送带有完整上下文的桌面命令
result = await adapter.send_desktop_command(
    command="click",
    params={"x": 500, "y": 300},
    screenshot=screenshot_base64,
    interactive_elements=[
        {"type": "button", "text": "Submit", "x": 500, "y": 300},
    ],
    active_window="Chrome - Google Search",
    model="minimax-m3",
)

# 结果包含：
# - result["action"]: 推荐的下一步操作
# - result["reasoning"]: 模型对该操作的推理过程
# - result["raw_response"]: 完整 API 响应
```

---

## 视觉协调：GLM-5V-Turbo + GLM-5.2

### 概述

GLM-5.2 可以与 GLM-5V-Turbo 协调进行视觉→代码→验证的循环，将视觉理解与深度推理和代码生成相结合：

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  GLM-5V-Turbo│────▶│   GLM-5.2    │────▶│ Verification │
│  (Visual     │     │  (Code       │     │  (5V-Turbo   │
│   Analysis)  │     │   Generation)│     │   Re-check)  │
└──────────────┘     └──────────────┘     └──────────────┘
       ↑                                          │
       └──────────── Feedback Loop ───────────────┘
```

### 何时使用视觉协调 vs M3

| 场景 | 使用 M3 | 使用 GLM-5.2 + 5V-Turbo |
|------|--------|------------------------|
| 简单图像问答 | ✅ | ❌（杀鸡用牛刀） |
| 视频分析 | ✅ | ❌（5V-Turbo 不支持视频） |
| 桌面自动化 | ✅ | ❌ |
| UI 设计稿→代码 | ⚠️（良好） | ✅（搭配 PreservedThinking 效果优秀） |
| 错误截图→修复 | ⚠️（良好） | ✅（配合代码上下文效果更好） |
| 多步骤视觉编码 | ❌ | ✅（视觉→代码→验证循环） |
| 大型代码库 + 视觉 | ❌ | ✅（1M 上下文 + 视觉） |

### 设置视觉协调

> **注意：** `vision_coordination_enabled` 和 `preserved_thinking_enabled` 是 `create_provider()` 的编译器级 kwargs，不是 TOML 驱动字段。在 TOML 中，使用 `multimodal_enabled = true` 启用多模态支持。

```toml
[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
multimodal_enabled = true               # 启用多模态（视觉协调）
# 注意：preserved_thinking_enabled 是 create_provider() 的 kwargs，不是 TOML 字段
```

### 视觉→代码→验证工作流

```python
from teragent import create_provider, TAPRequest
from teragent.core.tap import MultimodalContent

provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    vision_coordination_enabled=True,  # 编译器级 kwargs
    preserved_thinking_enabled=True,  # 编译器级 kwargs
)

# 步骤 1：分析 UI 设计稿并生成代码
request = TAPRequest(
    instruction="Analyze this UI mockup and implement it as a React component. "
                "Pay attention to layout, colors, typography, and spacing.",
    multimodal_context=[
        MultimodalContent(
            type="image_url",
            image_url={"url": "https://example.com/mockup.png"},
        ),
    ],
)

response = await provider.execute_tap(request)

# 步骤 2：验证实现
# 渲染生成的代码后，与原始设计稿进行对比
verify_request = TAPRequest(
    instruction="Compare this rendered output with the original mockup. "
                "Identify discrepancies and suggest fixes.",
    multimodal_context=[
        MultimodalContent(
            type="image_url",
            image_url={"url": "https://example.com/mockup.png"},
        ),
        MultimodalContent(
            type="image_url",
            image_url={"url": "https://example.com/rendered.png"},
        ),
    ],
)
```

### 结合 M3 和 GLM-5.2

对于复杂的视觉任务，可以将 M3 的原生视觉与 GLM-5.2 的编码能力结合使用：

```python
# 使用 M3 进行初始视觉分析
m3_provider = create_provider(
    compiler="minimax_m3",
    adapter="minimax_native",
    model="minimax-m3",
)

# 使用 GLM-5.2 进行带 PreservedThinking 的代码生成
glm52_provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    preserved_thinking_enabled=True,  # 编译器级 kwargs
)

# 步骤 1：M3 分析截图
visual_analysis = await m3_provider.execute_tap(TAPRequest(
    instruction="Describe every UI element in this screenshot in detail, "
                "including layout, colors, fonts, and spacing",
    multimodal_context=[
        MultimodalContent(type="image_url", image_url={"url": screenshot_url}),
    ],
))

# 步骤 2：GLM-5.2 基于分析结果生成代码
code_request = TAPRequest(
    instruction=f"Based on this visual analysis, implement the React component:\n\n"
                f"{visual_analysis.raw_text}",
)
code_response = await glm52_provider.execute_tap(code_request)
```

---

## 配置多模态路由

### 自动路由

配置后，ModelRouter 会自动将多模态内容路由到 M3：

```toml
[routing]
multimodal_driver = "openai_compatible.minimax_m3"
desktop_driver = "openai_compatible.minimax_m3"
```

### 路由优先级

多模态内容的路由决策流程：

1. **视频内容** → 路由到 M3（multimodal_context 中包含 video_url）
2. **桌面上下文** → 路由到 M3（has_desktop_context 为 True）
3. **图像内容 + 编码意图** → 路由到 GLM-5.2 + 5V-Turbo（如已配置）
4. **图像内容** → 路由到 M3（has_multimodal 为 True）
5. **纯文本** → 遵循正常路由（基于意图）

### 自定义视觉协调路由

```python
from teragent.router import ModelRouter, RoutingTable

# 自定义路由，将视觉编码任务发送到 GLM-5.2
routing_table = RoutingTable(
    multimodal_driver="openai_compatible.minimax_m3",
    desktop_driver="openai_compatible.minimax_m3",
    # 视觉编码任务覆盖
    vision_coding_driver="openai_compatible.glm_52",
)
```

### 使用多模态流水线配置

对于主要涉及视觉的任务，使用 `multimodal` 配置：

```toml
[execution.pipeline.profiles.multimodal]
description = "Multimodal mode: all stages use M3"
design_driver = "openai_compatible.minimax_m3"
plan_driver = "openai_compatible.minimax_m3"
execute_driver = "openai_compatible.minimax_m3"
review_driver = "openai_compatible.minimax_m3"
```

---

## 多模态编译与 Token 估算

### MiniMaxM3Compiler

M3 编译器对多模态内容的处理方式与文本不同：

- **纯文本请求**：使用 MSA 全文注入（无需检索截断）
- **多模态请求**：将内容编码为 OpenAI 格式的 content 数组，包含 `image_url` 和 `video_url` 块
- **桌面上下文**：将 `desktop_context` 转换为 M3 的桌面操作指令格式

### Token 估算

多模态内容比文本消耗更多 token。大致 token 数量：

| 内容类型 | 预估 Token 数 |
|---------|--------------|
| 文本 | 约每 4 个字符 1 个 token（标准） |
| 图像（低细节） | 约 85 tokens |
| 图像（高细节） | 约 170-1105 tokens（取决于分辨率） |
| 视频（1 分钟） | 约 1000-5000 tokens（取决于帧采样） |
| 截图 | 约 170-1105 tokens |

编译器提供 token 估算功能：

```python
from teragent import TAPRequest

request = TAPRequest(
    instruction="Analyze this image",
    multimodal_context=[
        MultimodalContent(type="image_url", image_url={"url": "..."}),
    ],
)

# 估算总 token 数（包括多模态内容）
estimated = request.estimate_prompt_tokens()
```

---

## 视觉任务最佳实践

### 1. 始终先截图

在执行任何桌面操作之前，先截取屏幕截图，让模型能够"看到"当前状态：

```python
# 好的做法：截图 → 分析 → 操作
screenshot = await tool.execute({"action": "screenshot"})
# 让 M3 分析截图并确定下一步操作
```

### 2. 截图使用 JPEG 格式

JPEG 格式截图比 PNG 更节省 token：

```python
safety = DesktopSafetyConfig(
    screenshot_format="jpeg",
    screenshot_quality=75,  # 平衡质量和 token 消耗
)
```

### 3. 为图像提供上下文

为模型提供关于所查看内容的上下文信息：

```python
# 好的做法：清晰的上下文
request = TAPRequest(
    instruction="This is a login form. Find the username and password fields "
                "and identify any validation errors displayed.",
    multimodal_context=[...],
)
```

### 4. 用截图验证操作结果

执行桌面操作后，再次截图以验证结果：

```python
# 1. 操作前截图
before = await tool.execute({"action": "screenshot"})

# 2. 执行操作
await tool.execute({"action": "click", "x": 500, "y": 300})

# 3. 操作后截图验证
after = await tool.execute({"action": "screenshot"})
```

### 5. 为视觉任务选择合适的模型

| 任务 | 模型 | 原因 |
|------|------|------|
| 图像问答 | M3 | 原生视觉，速度快 |
| 视频分析 | M3 | 仅 M3 支持视频 |
| 桌面自动化 | M3 | 仅 M3 支持桌面操作 |
| UI 设计稿→代码 | GLM-5.2 + 5V | PreservedThinking 保留计划 |
| 错误截图→修复 | GLM-5.2 + 5V | 代码上下文 + 视觉 |
| 视觉回归测试 | M3 | 对比前后截图 |

### 6. 使用模拟模式进行测试

当未安装 `pyautogui` 时，DesktopTool 以模拟模式运行：

```python
tool = DesktopTool()
if tool.simulation_mode:
    print("Running in simulation mode — no actual desktop operations")
    # 适用于无副作用的测试
```

---

## 降级行为

### 当 M3 不可用时

如果 MiniMax M3 模型不可用（API 错误、速率限制、熔断器开启），系统会：

1. **熔断器** 检测到 M3 故障并开启熔断
2. **降级链** 路由到下一个可用模型（默认：V4-Pro）
3. **多模态降级** —— 其他编译器（V4、GLM）无法原生处理图像/视频。它们会：
   - 以文本描述图像内容（如果可用）
   - 完全跳过视频内容
   - 记录关于多模态能力降级的警告

### 当 5V-Turbo 不可用时

如果 GLM-5V-Turbo 不可用于视觉协调：

1. **熔断器** 检测到 5V-Turbo 故障
2. **视觉协调降级** —— GLM-5.2 将图像作为文本描述处理
3. **质量降低** —— 没有视觉模型时，从设计稿生成代码的准确度降低
4. **考虑回退到 M3** —— 将视觉任务路由到 M3

### 多模态降级链

```python
from teragent.reliability.recovery import DegradationChain

# 默认多模态降级链
chain = DegradationChain()
# "multimodal" → M3 → GLM-5.2（5V-Turbo）→ V4-Pro（降级为纯文本）
```

---

## 与审查流水线集成

### 审查阶段的视觉验证

在审查阶段使用 M3 对 UI 变更进行视觉验证：

```toml
[execution.pipeline.profiles.quality]
description = "Quality first with visual review"
design_driver = "openai_compatible.deepseek_v4_pro"
plan_driver = "openai_compatible.glm_52"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.minimax_m3"  # M3 用于视觉审查
```

### 自动化视觉回归测试

```python
from teragent import TAPRequest
from teragent.core.tap import MultimodalContent

# 对比前后截图
request = TAPRequest(
    meta={"intent": "review"},
    instruction="Compare the before and after screenshots. "
                "Identify any visual regressions or unexpected changes. "
                "Focus on: layout shifts, color changes, missing elements.",
    multimodal_context=[
        MultimodalContent(
            type="image_url",
            image_url={"url": "https://ci.example.com/before.png"},
        ),
        MultimodalContent(
            type="image_url",
            image_url={"url": "https://ci.example.com/after.png"},
        ),
    ],
)
```

### 基于截图的代码审查

```python
# 代码变更后，截取并审查渲染输出
request = TAPRequest(
    meta={"intent": "review"},
    instruction="Review this screenshot of the rendered page. "
                "Check for: (1) correct layout, (2) proper styling, "
                "(3) responsive design issues, (4) accessibility concerns.",
    context={
        "changes": "Modified header component and navigation bar CSS",
        "expected": "Header should be sticky, navigation links should be centered",
    },
    multimodal_context=[
        MultimodalContent(
            type="image_url",
            image_url={"url": f"data:image/png;base64,{screenshot_base64}"},
        ),
    ],
)
```

---

*本指南是 TerAgent 文档的一部分。如需完整的四模型适配指南，请参阅 [适配指南](adaptation_guide.md)。如需 GLM-5.2 特定功能，请参阅 [GLM-5.2 指南](glm_52_guide.md)。*
