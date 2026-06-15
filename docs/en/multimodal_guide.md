# Multimodal Usage Guide

This guide covers TerAgent's multimodal capabilities, including MiniMax M3's native multimodal support, image/video processing, desktop automation, and GLM-5V-Turbo + GLM-5.2 vision coordination.

---

## Table of Contents

- [M3 Multimodal Capabilities Overview](#m3-multimodal-capabilities-overview)
- [Image Input](#image-input)
- [Video Input](#video-input)
- [Desktop Operation Tools](#desktop-operation-tools)
- [DesktopContext Usage](#desktopcontext-usage)
- [Vision Coordination: GLM-5V-Turbo + GLM-5.2](#vision-coordination-glm-5v-turbo--glm-52)
- [Configuring Multimodal Routing](#configuring-multimodal-routing)
- [Multimodal Compilation and Token Estimation](#multimodal-compilation-and-token-estimation)
- [Best Practices for Visual Tasks](#best-practices-for-visual-tasks)
- [Fallback Behavior](#fallback-behavior)
- [Integration with Review Pipeline](#integration-with-review-pipeline)

---

## M3 Multimodal Capabilities Overview

MiniMax M3 provides native multimodal support with the following capabilities:

| Capability | Description | Performance |
|-----------|-------------|-------------|
| **Image understanding** | Analyze images from URLs or base64 | High accuracy on visual Q&A |
| **Video understanding** | Process video content natively | Supports MP4, AVI, MOV, etc. |
| **Desktop automation** | Screenshot → analyze → click/type/scroll | 7 action types with 5-layer safety |
| **MSA efficient** | Full-text injection at 1M context | 1/20 compute cost via Sparse Attention |
| **Agent programming** | Code generation and understanding | SWE-Bench Pro 59.0% |
| **Browse enhancement** | Web information retrieval | BrowseComp 83.5 |

**Key specifications:**
- Context window: 1,000,000 tokens
- Max output: 384,000 tokens
- Supports mixed content: text + images + video in the same request

### Multimodal Model Comparison

| Feature | MiniMax M3 | GLM-5.2 + 5V-Turbo |
|---------|-----------|---------------------|
| Image understanding | ✅ Native | ✅ Via 5V-Turbo |
| Video understanding | ✅ Native | ❌ |
| Desktop operations | ✅ Native | ❌ |
| Coding from mockups | ✅ Good | ✅ Excellent (PreservedThinking) |
| Vision→Code→Verify | ❌ | ✅ Coordinated cycle |
| Context window | 1M | 1M |
| Best for | Visual analysis, desktop | UI implementation, coding |

---

## Image Input

### Using Image URLs

```python
from teragent import TAPRequest, create_provider
from teragent.core.tap import MultimodalContent

# Create M3 provider
provider = create_provider(
    compiler="minimax_m3",
    adapter="minimax_native",
    model="minimax-m3",
    base_url="https://api.minimaxi.com/v1",
    api_key_env="MINIMAX_API_KEY",
)

# Analyze an image from URL
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

### Using Base64-Encoded Images

```python
import base64

# Read image and encode to base64
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

### Multiple Images in One Request

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

### Image with Text Context

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

### Image Size and Format Guidelines

| Format | Recommended | Notes |
|--------|-------------|-------|
| JPEG | ✅ | Best for screenshots, smaller token cost |
| PNG | ✅ | Better for diagrams with text |
| WebP | ⚠️ | May not be supported by all endpoints |
| GIF | ❌ | Not recommended for analysis |
| BMP | ❌ | Not recommended, large file size |

**Recommended image dimensions:** 512x512 to 2048x2048 pixels. Larger images consume more tokens with diminishing returns.

---

## Video Input

### Video URL Processing

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

### Video with Processing Hints

The MiniMax M3 compiler automatically adds video processing hints when using the `MiniMaxNativeAdapter`:

```python
# The adapter automatically enhances video content with:
# - minimax_video_mode: "understand" (default) or "summarize"
# - minimax_frame_sampling: "auto" (default), "uniform", "keyframe", or "dense"

# These hints are added automatically when using MiniMaxNativeAdapter.
# You don't need to set them manually.
```

### Supported Video Formats

| Format | Extension |
|--------|-----------|
| MPEG-4 | `.mp4` |
| AVI | `.avi` |
| QuickTime | `.mov` |
| Matroska | `.mkv` |
| WebM | `.webm` |
| Flash Video | `.flv` |
| Windows Media | `.wmv` |
| MPEG-4 Part 14 | `.m4v` |

### Video Timeout

Video processing takes longer than text. Set an appropriate timeout:

```python
from teragent.core.adapters.minimax_native import MiniMaxNativeAdapter

adapter = MiniMaxNativeAdapter(
    base_url="https://api.minimaxi.com/v1",
    api_key="your-api-key",
    timeout=300.0,            # Standard timeout
    multimodal_timeout=600.0, # Extended timeout for video (10 minutes)
)
```

### Video Best Practices

1. **Keep videos under 5 minutes** — Longer videos consume significant tokens and time
2. **Use "summarize" mode for long videos** — More efficient than frame-by-frame analysis
3. **Provide context** — Tell the model what to look for in the video
4. **Use direct URLs** — Avoid base64 encoding for videos due to size

---

## Desktop Operation Tools

### Overview

The `DesktopTool` provides 7 desktop automation actions through M3's visual understanding:

| Action | Description | Parameters |
|--------|-------------|------------|
| `screenshot` | Capture screen | None |
| `click` | Click at position | `x`, `y`, `button` (optional) |
| `type_text` | Type text | `text` |
| `scroll` | Scroll view | `direction`, `scroll_amount` |
| `hotkey` | Press key combo | `keys` (comma-separated) |
| `move_mouse` | Move cursor | `x`, `y` |
| `drag` | Drag object | `x`, `y`, `end_x`, `end_y` |

### Basic Desktop Usage

```python
from teragent.tools.desktop import DesktopTool

tool = DesktopTool()

# 1. Take a screenshot first
result = await tool.execute({"action": "screenshot"})
screenshot_base64 = result.data.get("screenshot", "")

# 2. Click at specific coordinates
result = await tool.execute({
    "action": "click",
    "x": 500,
    "y": 300,
    "button": "left",
})

# 3. Type text
result = await tool.execute({
    "action": "type_text",
    "text": "Hello, World!",
})

# 4. Use keyboard shortcut
result = await tool.execute({
    "action": "hotkey",
    "keys": "ctrl,s",  # Ctrl+S (save)
})

# 5. Scroll down
result = await tool.execute({
    "action": "scroll",
    "direction": "down",
    "scroll_amount": 3,
})
```

### Desktop Safety Configuration

```python
from teragent.tools.desktop import DesktopTool, DesktopSafetyConfig

safety = DesktopSafetyConfig(
    safe_zones=[
        (0, 0, 100, 50),    # Block top-left corner (e.g., system menu)
    ],
    min_interval=0.5,         # Min 0.5s between operations
    max_consecutive_ops=50,   # Max 50 consecutive operations
    screenshot_quality=75,    # JPEG quality (1-100)
    screenshot_format="jpeg", # "jpeg" or "png"
)

tool = DesktopTool(safety_config=safety)
```

### 5-Layer Safety System

1. **Permission level** — All desktop operations require DESTRUCTIVE-level user confirmation
2. **Safe zones** — Configurable forbidden click areas (coordinates that should never be clicked)
3. **Rate limiting** — Minimum interval between operations (prevents too-fast automation)
4. **Consecutive ops cap** — Maximum number of operations before forced pause
5. **Blocked shortcuts** — Dangerous key combinations are automatically blocked:
   - Alt+F4 (close window)
   - Ctrl+Alt+Delete (system security)
   - Alt+Tab (window switch)
   - Ctrl+Shift+Esc (task manager)
   - Win+L / Super+L (lock screen)
   - Cmd+Q (macOS quit)

---

## DesktopContext Usage

The `DesktopContext` provides structured information about the current desktop state for M3 to analyze:

### Creating a DesktopContext

```python
from teragent import TAPRequest
from teragent.core.tap import DesktopContext

# Build desktop context manually
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

# Use in a TAPRequest
request = TAPRequest(
    instruction="Click on the search input and type 'TerAgent documentation'",
    desktop_context=desktop_ctx,
)
```

### Automatic DesktopContext from Screenshot

```python
from teragent.tools.desktop import DesktopTool

tool = DesktopTool()

# Take a screenshot — the tool can auto-detect interactive elements
result = await tool.execute({"action": "screenshot"})

# The result includes detected interactive elements
# that can be used to build a DesktopContext
```

### DesktopContext with M3 Adapter

```python
from teragent.core.adapters.minimax_native import MiniMaxNativeAdapter

adapter = MiniMaxNativeAdapter(
    base_url="https://api.minimaxi.com/v1",
    api_key="your-api-key",
)

# Send desktop command with full context
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

# Result contains:
# - result["action"]: Recommended next action
# - result["reasoning"]: Model's reasoning for the action
# - result["raw_response"]: Full API response
```

---

## Vision Coordination: GLM-5V-Turbo + GLM-5.2

### Overview

GLM-5.2 can coordinate with GLM-5V-Turbo for vision→code→verify cycles, combining visual understanding with deep reasoning and code generation:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  GLM-5V-Turbo│────▶│   GLM-5.2    │────▶│ Verification │
│  (Visual     │     │  (Code       │     │  (5V-Turbo   │
│   Analysis)  │     │   Generation)│     │   Re-check)  │
└──────────────┘     └──────────────┘     └──────────────┘
       ↑                                          │
       └──────────── Feedback Loop ───────────────┘
```

### When to Use Vision Coordination vs M3

| Scenario | Use M3 | Use GLM-5.2 + 5V-Turbo |
|----------|--------|------------------------|
| Simple image Q&A | ✅ | ❌ (overkill) |
| Video analysis | ✅ | ❌ (5V-Turbo doesn't support video) |
| Desktop automation | ✅ | ❌ |
| UI mockup → code | ⚠️ (good) | ✅ (excellent with PreservedThinking) |
| Error screenshot → fix | ⚠️ (good) | ✅ (better with code context) |
| Multi-step visual coding | ❌ | ✅ (vision→code→verify cycles) |
| Large codebase + visual | ❌ | ✅ (1M context + vision) |

### Setting Up Vision Coordination

```toml
[drivers.openai_compatible.glm_52]
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "GLM_API_KEY"
model = "glm-5.2"
compiler = "glm_52"
vision_coordination_enabled = true
preserved_thinking_enabled = true    # Recommended for coding from mockups
```

### Vision→Code→Verify Workflow

```python
from teragent import create_provider, TAPRequest
from teragent.core.tap import MultimodalContent

provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    vision_coordination_enabled=True,
    preserved_thinking_enabled=True,
)

# Step 1: Analyze a UI mockup and generate code
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

# Step 2: Verify the implementation
# After rendering the generated code, compare with the original
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

### Combining M3 and GLM-5.2

For complex visual tasks, you can combine M3's native vision with GLM-5.2's coding:

```python
# Use M3 for initial visual analysis
m3_provider = create_provider(
    compiler="minimax_m3",
    adapter="minimax_native",
    model="minimax-m3",
)

# Use GLM-5.2 for code generation with PreservedThinking
glm52_provider = create_provider(
    compiler="glm_52",
    adapter="openai_compatible",
    model="glm-5.2",
    preserved_thinking_enabled=True,
)

# Step 1: M3 analyzes the screenshot
visual_analysis = await m3_provider.execute_tap(TAPRequest(
    instruction="Describe every UI element in this screenshot in detail, "
                "including layout, colors, fonts, and spacing",
    multimodal_context=[
        MultimodalContent(type="image_url", image_url={"url": screenshot_url}),
    ],
))

# Step 2: GLM-5.2 generates code based on the analysis
code_request = TAPRequest(
    instruction=f"Based on this visual analysis, implement the React component:\n\n"
                f"{visual_analysis.raw_text}",
)
code_response = await glm52_provider.execute_tap(code_request)
```

---

## Configuring Multimodal Routing

### Automatic Routing

The ModelRouter automatically routes multimodal content to M3 when configured:

```toml
[routing]
multimodal_driver = "openai_compatible.minimax_m3"
desktop_driver = "openai_compatible.minimax_m3"
```

### Routing Priority

The routing decision flow for multimodal content:

1. **Video content** → Route to M3 (video_url in multimodal_context)
2. **Desktop context** → Route to M3 (has_desktop_context is True)
3. **Image content + coding intent** → Route to GLM-5.2 with 5V-Turbo (if configured)
4. **Image content** → Route to M3 (has_multimodal is True)
5. **Text only** → Follow normal routing (intent-based)

### Custom Routing for Vision Coordination

```python
from teragent.router import ModelRouter, RoutingTable

# Custom routing that sends visual coding tasks to GLM-5.2
routing_table = RoutingTable(
    multimodal_driver="openai_compatible.minimax_m3",
    desktop_driver="openai_compatible.minimax_m3",
    # Override for visual coding tasks
    vision_coding_driver="openai_compatible.glm_52",
)
```

### Using the Multimodal Pipeline Profile

For tasks that are primarily visual, use the `multimodal` profile:

```toml
[execution.pipeline.profiles.multimodal]
description = "Multimodal mode: all stages use M3"
design_driver = "openai_compatible.minimax_m3"
plan_driver = "openai_compatible.minimax_m3"
execute_driver = "openai_compatible.minimax_m3"
review_driver = "openai_compatible.minimax_m3"
```

---

## Multimodal Compilation and Token Estimation

### MiniMaxM3Compiler

The M3 compiler handles multimodal content differently than text:

- **Text-only requests**: Uses MSA full-text injection (no retrieval truncation needed)
- **Multimodal requests**: Encodes content as OpenAI-format content arrays with `image_url` and `video_url` blocks
- **Desktop context**: Converts `desktop_context` into M3's desktop operation instruction format

### Token Estimation

Multimodal content consumes more tokens than text. Approximate token counts:

| Content Type | Estimated Tokens |
|-------------|-----------------|
| Text | ~1 token per 4 characters (standard) |
| Image (low detail) | ~85 tokens |
| Image (high detail) | ~170-1105 tokens (depends on resolution) |
| Video (1 minute) | ~1000-5000 tokens (depends on frame sampling) |
| Screenshot | ~170-1105 tokens |

The compiler provides token estimation:

```python
from teragent import TAPRequest

request = TAPRequest(
    instruction="Analyze this image",
    multimodal_context=[
        MultimodalContent(type="image_url", image_url={"url": "..."}),
    ],
)

# Estimate total tokens (including multimodal content)
estimated = request.estimate_prompt_tokens()
```

---

## Best Practices for Visual Tasks

### 1. Always Start with a Screenshot

Before performing any desktop operation, capture a screenshot so the model can "see" the current state:

```python
# Good: Screenshot → Analyze → Act
screenshot = await tool.execute({"action": "screenshot"})
# Let M3 analyze the screenshot and determine the next action
```

### 2. Use JPEG Format for Screenshots

JPEG is more token-efficient than PNG for screenshots:

```python
safety = DesktopSafetyConfig(
    screenshot_format="jpeg",
    screenshot_quality=75,  # Balance quality and token cost
)
```

### 3. Provide Context with Images

Give the model context about what it's looking at:

```python
# Good: Clear context
request = TAPRequest(
    instruction="This is a login form. Find the username and password fields "
                "and identify any validation errors displayed.",
    multimodal_context=[...],
)
```

### 4. Verify Actions with Screenshots

After performing a desktop action, take another screenshot to verify the result:

```python
# 1. Screenshot before
before = await tool.execute({"action": "screenshot"})

# 2. Perform action
await tool.execute({"action": "click", "x": 500, "y": 300})

# 3. Screenshot after to verify
after = await tool.execute({"action": "screenshot"})
```

### 5. Choose the Right Model for Visual Tasks

| Task | Model | Reason |
|------|-------|--------|
| Image Q&A | M3 | Native vision, fast |
| Video analysis | M3 | Only M3 supports video |
| Desktop automation | M3 | Only M3 supports desktop |
| UI mockup → code | GLM-5.2 + 5V | PreservedThinking keeps plan |
| Error screenshot → fix | GLM-5.2 + 5V | Code context + vision |
| Visual regression test | M3 | Compare before/after screenshots |

### 6. Use Simulation Mode for Testing

When `pyautogui` is not installed, the DesktopTool operates in simulation mode:

```python
tool = DesktopTool()
if tool.simulation_mode:
    print("Running in simulation mode — no actual desktop operations")
    # Useful for testing without side effects
```

---

## Fallback Behavior

### When M3 is Unavailable

If the MiniMax M3 model is unavailable (API error, rate limit, circuit breaker open), the system:

1. **Circuit breaker** detects M3 failure and opens the breaker
2. **Degradation chain** routes to the next available model (default: V4-Pro)
3. **Multimodal degradation** — Other compilers (V4, GLM) cannot process images/video natively. They will:
   - Describe the image content in text (if available)
   - Skip video content entirely
   - Log a warning about degraded multimodal capability

### When 5V-Turbo is Unavailable

If GLM-5V-Turbo is unavailable for vision coordination:

1. **Circuit breaker** detects 5V-Turbo failure
2. **Vision coordination degrades** — GLM-5.2 processes the image as text description
3. **Quality reduction** — Without vision model, coding from mockups is less accurate
4. **Consider M3 fallback** — Route visual tasks to M3 instead

### Degradation Chain for Multimodal

```python
from teragent.reliability.recovery import DegradationChain

# Default multimodal degradation chain
chain = DegradationChain()
# "multimodal" → M3 → GLM-5.2 (5V-Turbo) → V4-Pro (degrades to text-only)
```

---

## Integration with Review Pipeline

### Visual Verification in Review Stage

Use M3 in the review stage for visual verification of UI changes:

```toml
[execution.pipeline.profiles.quality]
description = "Quality first with visual review"
design_driver = "openai_compatible.deepseek_v4_pro"
plan_driver = "openai_compatible.glm_52"
execute_driver = "openai_compatible.deepseek_v4_flash"
review_driver = "openai_compatible.minimax_m3"  # M3 for visual review
```

### Automated Visual Regression Testing

```python
from teragent import TAPRequest
from teragent.core.tap import MultimodalContent

# Compare before and after screenshots
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

### Screenshot-Based Code Review

```python
# After code changes, capture and review the rendered output
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

*This guide is part of the TerAgent documentation. For the complete four-model adaptation guide, see [Adaptation Guide](adaptation_guide.md). For GLM-5.2 specific features, see [GLM-5.2 Guide](glm_52_guide.md).*
