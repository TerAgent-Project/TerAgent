# Multimodal Usage Guide

This guide covers TerAgent's multimodal capabilities powered by MiniMax M3, including image understanding, video processing, and desktop automation.

---

## Table of Contents

- [M3 Multimodal Capabilities Overview](#m3-multimodal-capabilities-overview)
- [Image Understanding](#image-understanding)
- [Video Processing](#video-processing)
- [Desktop Operations](#desktop-operations)
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

---

## Image Understanding

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

---

## Video Processing

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

---

## Desktop Operations

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

### Desktop Command via MiniMaxNativeAdapter

For direct API access to M3's desktop endpoint:

```python
from teragent.core.adapters.minimax_native import MiniMaxNativeAdapter

adapter = MiniMaxNativeAdapter(
    base_url="https://api.minimaxi.com/v1",
    api_key="your-api-key",
    group_id="your-group-id",  # Required for some MiniMax endpoints
)

# Send desktop command with screen context
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

## Configuring Multimodal Routing

### Automatic Routing

The ModelRouter automatically routes multimodal content to M3 when configured:

```toml
[routing]
multimodal_driver = "openai_compatible.minimax_m3"
desktop_driver = "openai_compatible.minimax_m3"
```

When a `TAPRequest` has `multimodal_context` or `desktop_context`, the router overrides the default model and routes to M3.

### Routing Priority

The routing decision flow for multimodal content:

1. **Video content** → Route to M3 (video_url in multimodal_context)
2. **Desktop context** → Route to M3 (has_desktop_context is True)
3. **Image content** → Route to M3 (has_multimodal is True)
4. **Text only** → Follow normal routing (intent-based)

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

```python
from teragent.router import PipelineManager

pm = PipelineManager()
pm.set_active_profile("multimodal")
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

# Bad: No context
request = TAPRequest(
    instruction="What is this?",
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

### 5. Handle Large Videos with Appropriate Frame Sampling

For long videos, use `"summarize"` mode instead of `"understand"` to reduce token consumption:

```python
# The MiniMaxNativeAdapter automatically sets processing hints.
# For longer videos, the adapter uses "auto" frame sampling
# which lets the API decide the optimal sampling strategy.
```

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

### Degradation Chain for Multimodal

```python
from teragent.reliability.recovery import DegradationChain

# Default multimodal degradation chain
chain = DegradationChain()
# "multimodal" → M3 → V4-Pro (degrades to text-only)
```

### Rate Limit Handling

```python
from teragent.reliability.recovery import RateLimitHandler

handler = RateLimitHandler()

# M3 returns rate limits via X-RateLimit-* headers
info = handler.parse_rate_limit_response(
    model_name="minimax_m3",
    status_code=429,
    headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1709000000"},
)

if not handler.should_retry("minimax_m3", info):
    # Fall back to text-only model
    pass
```

---

## Integration with Review Pipeline

### Visual Verification in Review Stage

Use M3 in the review stage for visual verification of UI changes:

```toml
[execution.pipeline.profiles.quality]
description = "Quality first with visual review"
design_driver = "openai_compatible.deepseek_v4_pro"
plan_driver = "openai_compatible.glm_5"
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

response = await provider.execute_tap(request)
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
