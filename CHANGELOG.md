# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] — 2025-07-13

### Added

#### Three-Model Deep Adaptation (Phase 3)
- **DeepSeek V4 Compiler** (`deepseek_v4`): Math/code/reasoning enhancement, cache-aware prompt layout, Flash/Pro variants
- **GLM-5 Compiler** (`glm_5`): Recency effect optimization, 200K context compression, thinking mode control, long-horizon task support, self-evaluation injection, strategy switching
- **MiniMax M3 Compiler** (`minimax_m3`): Multimodal content handling, MSA full-text injection, desktop operation degradation
- **MiniMax Native Adapter** (`minimax_native`): Rate limit tracking, billing tracker, video content enhancement, desktop operation API
- **Model Router**: Intent-based routing, cost fallback ordering, degradation chain, context length overrides
- **Pipeline Manager**: Named profiles (budget/multimodal/quality/long_horizon), TOML-driven configuration
- **Cross-Model Cost Tracker**: Unified cost tracking across DeepSeek V4, GLM-5, and MiniMax M3

#### Reliability (Phase 4)
- **ModelCircuitBreakerManager**: Per-model circuit breaker with configurable thresholds
- **DegradationChain**: Automatic model degradation with fallback paths
- **LongHorizonRecoveryManager**: Recovery strategies for long-running tasks (8h autonomous mode)
- **RateLimitHandler**: Automatic rate limit response handling with retry-after support

#### Benchmark Framework (Phase 4)
- 8 benchmark suites: Compilation, Latency, Context Management, Multimodal, Long-Horizon, Cost Efficiency, Router, Fault Recovery
- `BenchmarkRunner` for automated benchmark execution and report generation

#### Documentation
- Three-Model Evaluation Report (`docs/EVALUATION_THREE_MODELS.md`)
- Three-Model Adaptation Guide (`docs/en/three_model_adaptation_guide.md`)
- Long-Horizon Task Guide (`docs/en/long_horizon_guide.md`)
- Multimodal Guide (`docs/en/multimodal_guide.md`)
- Ascend Deployment Guide (`docs/deployment_guide_ascend.md`)
- Local deployment config template (`agent.local.toml`)

#### Infrastructure
- GitHub Actions CI workflow (lint + test on Python 3.10/3.11/3.12 + build check)
- Live API test framework (`pytest.mark.live`)
- MiniMax native adapter test suite

### Changed
- Model identifier defaults from `glm-5.1` to `glm-5` (GLM-5 family compatibility)
- All docstring examples updated from `compiler="glm"` to `compiler="glm_5"` for GLM-5
- All `openai_compatible.glm` references updated to `openai_compatible.glm_5`
- Version bumped from `0.0.1` (Alpha) to `0.1.0` (Beta)
- Version bumped from `0.1.0` to `0.1.1` — ruff lint fixes, PEP 639 compliance, CI green
- Development Status classifier from `3 - Alpha` to `4 - Beta`
- Documentation compiler/adapter tables updated from 4×3 to 9×4

### Fixed
- Broken link to deleted `EVALUATION_GLM5.md` in README (replaced with `EVALUATION_THREE_MODELS.md`)
- Broken anchor in deployment guide (`#3-glm-51-` → `#3-glm-5-`)
- Outdated model name in `examples/simple_chat` (`glm-4-flash` → `glm-5`)
- Missing re-exports for P3/P4 prompt templates and reliability classes in top-level `__init__.py`
- Missing `FirecrackerSandbox` export in `security/__init__.py`
- Missing `DesktopTool`, `SubAgentWorker` exports in their respective `__init__.py`
- `cuda_triton` intent missing from docstrings
- Incomplete compiler/adapter lists in docstrings and comments
