# TerAgent Documentation

Welcome to the TerAgent documentation. TerAgent is a Python library for building production AI agent systems with a compiler-adapter architecture.

## Guides

| Guide | Description |
|-------|-------------|
| [Getting Started](getting-started.md) | Installation, quick start, and first steps |
| [Architecture](architecture.md) | Design principles, module dependencies, data flow |
| [Security](security.md) | Permission system, sandbox, 2PC file writes, API key security, cross-platform compatibility |
| [Configuration](configuration.md) | agent.toml, typed config, environment variables, platform-specific paths |
| [Streaming](streaming.md) | Streaming tool execution, dispatch strategy, degradation |
| [Self-RL Data](self-rl.md) | TAP tracing, DPO pair generation, data constitution |
| [Contributing](contributing.md) | Development setup, coding standards, adding modules |
| [Four-Model Adaptation Guide](adaptation_guide.md) | DeepSeek V4, MiniMax M3, GLM-5, GLM-5.2 configuration and best practices |
| [GLM-5.2 Usage Guide](glm_52_guide.md) | 1M context, dual thinking, PreservedThinking, 5V-Turbo coordination |
| [Long-Horizon Task Guide](long_horizon_guide.md) | 8-hour+ autonomous tasks with GLM-5/5.2 |
| [Multimodal Guide](multimodal_guide.md) | Image, video, and desktop operations with MiniMax M3 |

## Reports & Deployment

- [Four-Model Evaluation Report](../EVALUATION_FOUR_MODELS.md) — Comprehensive benchmark results for V4/M3/GLM-5/GLM-5.2
- [GLM-5.2 Stability Report](../glm_52_stability_report.md) — Production stability verification
- [Ascend Deployment Guide](../deployment_guide_ascend.md) — Deploying TerAgent on Huawei Ascend NPU

## API Reference

- [Complete API Reference](api-reference.md) — Module-by-module reference with code examples

## Quick Links

- **Project**: [GitHub](https://github.com/teragent/teragent)
- **License**: Apache License Version 2.0
- **Version**: 0.1.3
