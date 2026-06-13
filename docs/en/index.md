# TerAgent Documentation

Welcome to the TerAgent documentation. TerAgent is a Python library for building production AI agent systems with a compiler-adapter architecture.

## Guides

| Guide | Description |
|-------|-------------|
| [Getting Started](getting-started.md) | Installation, quick start, and first steps |
| [Architecture](architecture.md) | Design principles, module dependencies, data flow |
| [Security](security.md) | Permission system, sandbox, 2PC file writes, API key security |
| [Configuration](configuration.md) | agent.toml, typed config, environment variables |
| [Streaming](streaming.md) | Streaming tool execution, dispatch strategy, degradation |
| [Self-RL Data](self-rl.md) | TAP tracing, DPO pair generation, data constitution |
| [Contributing](contributing.md) | Development setup, coding standards, adding modules |
| [Three-Model Adaptation Guide](three_model_adaptation_guide.md) | DeepSeek V4, MiniMax M3, GLM-5 configuration and best practices |
| [Long-Horizon Task Guide](long_horizon_guide.md) | 8-hour autonomous tasks with GLM-5 |
| [Multimodal Guide](multimodal_guide.md) | Image, video, and desktop operations with MiniMax M3 |

## Reports & Deployment

- [Evaluation Report](../EVALUATION_THREE_MODELS.md) — Three-model evaluation results
- [Ascend Deployment Guide](../deployment_guide_ascend.md) — Deploying TerAgent on Huawei Ascend NPU

## API Reference

- [Complete API Reference](api-reference.md) — Module-by-module reference with code examples

## Quick Links

- **Project**: [GitHub](https://github.com/teragent/teragent)
- **License**: Apache License Version 2.0
- **Version**: 0.1.1
