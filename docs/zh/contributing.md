# 为 TerAgent 贡献

感谢你对 TerAgent 贡献的关注！本指南涵盖开发环境搭建、编码规范和贡献流程。

## 开发环境搭建

### 前置条件

- Python 3.10 或更高版本
- [bun](https://bun.sh/) 或 npm（用于运行测试）
- git

### 安装开发依赖

```bash
# Clone the repository
git clone https://github.com/teragent/teragent.git
cd teragent

# Install with development dependencies
pip install -e ".[dev]"

# Install all optional dependencies (for full testing)
pip install -e ".[all]"
```

### 运行测试

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_permission.py

# Run with coverage
pytest --cov=teragent --cov-report=term-missing
```

### 代码检查

```bash
# Run ruff linter
ruff check teragent/

# Auto-fix issues
ruff check --fix teragent/
```

### 类型检查

```bash
mypy teragent/
```

## 代码架构

### 模块组织

```
teragent/
├── core/              # TAP IR + Compiler + Adapter + Provider
│   ├── tap.py         # TAPRequest, TAPResponse, CompiledPrompt
│   ├── compiler.py    # TAPCompiler ABC + Registry
│   ├── adapter.py     # TAPAdapter ABC + Registry
│   ├── provider.py    # ModelProvider (Compiler + Adapter composition)
│   ├── types.py       # Message, MessageRole, MessageType, ToolSafety
│   ├── compilers/     # Concrete compilers: default, glm, glm_5, glm_52, glm_5v_turbo, anthropic, deepseek, deepseek_v4, minimax_m3
│   ├── adapters/      # Concrete adapters: openai_compatible, anthropic_native, glm_native, minimax_native, mock
│   └── prompts/       # Intent-specific system prompts
├── security/          # Security architecture
│   ├── permission.py  # PermissionManager + EnhancedPermissionManager
│   ├── sandbox.py     # Sandbox execution (3 levels)
│   ├── file_writer.py # 2PC file writes
│   ├── file_state.py  # File state tracking
│   ├── audit.py       # Audit logging
│   └── ai_permission_classifier.py  # AI-based permission classification
├── reliability/       # Reliability system
│   ├── circuit_breaker.py  # 4 circuit breakers + manager
│   ├── budget.py      # Step budget + CrossModelCostTracker
│   └── recovery.py    # Recovery manager
├── context/           # Context management
│   ├── context_window.py  # Token budget estimator
│   ├── auto_compact.py    # Automatic context compaction
│   ├── microcompactor.py  # Fine-grained compaction
│   ├── memory.py      # .agent.md persistent memory
│   ├── code_indexer.py    # tree-sitter AST indexing (optional)
│   ├── reference_graph.py # networkx dependency graph (optional)
│   └── vector_indexer.py  # LanceDB semantic search (optional)
├── pipeline/          # Pipeline primitives
│   ├── extractor.py   # File extraction from model output
│   ├── prompt_builder.py  # Template-based prompt construction
│   ├── checklist.py   # Deterministic code verification
│   ├── retry.py       # Exponential backoff retry
│   └── tracing.py     # TAP tracing + DPO pair generation
├── streaming/         # Streaming execution
│   ├── streaming_executor.py  # StreamingToolExecutor
│   └── stream_events.py      # Stream event types + parsers
├── tools/             # Tool system
│   ├── base.py        # BaseTool ABC + ToolResult
│   ├── registry.py    # ToolRegistry
│   ├── orchestrator.py # ToolOrchestrator
│   └── desktop.py     # DesktopTool (desktop automation)
├── router/            # Smart model routing
│   └── model_router.py # ModelRouter, RoutingTable, PipelineManager
├── long_horizon/      # Long-horizon autonomous tasks
│   ├── manager.py     # LongHorizonTaskManager
│   ├── checkpoint.py  # CheckpointStore
│   ├── evaluator.py   # SelfEvaluator
│   ├── strategy.py    # StrategySwitcher
│   └── progress.py    # ProgressTracker
├── benchmark/         # Benchmark framework
│   └── runner.py      # BenchmarkRunner
├── intent/            # Intent classification
│   ├── classifier.py  # IntentClassifier
│   └── confirmation.py # ConfirmationGate
├── hooks/             # Hook system
│   ├── manager.py     # HookManager
│   └── builtin/       # Built-in hooks (audit, dangerous command)
├── coordination/      # Sub-agent coordination
│   ├── sub_agent_manager.py  # SubAgentManager
│   └── message_bus.py        # AgentMessageBus
├── session/           # Session persistence
│   └── persistence.py # SessionPersistence (SQLite)
├── config/            # Configuration system
│   ├── loader.py      # Config loading + provider creation
│   ├── teragent_config.py    # Top-level config
│   ├── driver_config.py      # Driver config
│   └── ...            # 15+ typed config dataclasses
├── event_bus.py       # EventBus (signal-driven communication)
├── utils/             # Utilities
│   ├── exceptions.py  # Custom exceptions
│   ├── token_counter.py # Token estimation
│   ├── text.py        # Text utilities
│   └── tracing.py     # Tracing utilities
└── agent_loop.py      # AgentLoop (central orchestration)
```

### 关键设计原则

1. **正交组合**：Compiler × Adapter —— 添加新模型或新协议，而非同时添加两者
2. **纵深防御**：7 层权限、6 层命令防御、2PC 文件写入、3 级沙箱
3. **建议优先**：熔断器默认发出警告而非阻止
4. **即发即弃事件**：EventBus 永远不会阻塞主循环
5. **延迟导入**：可选依赖仅在使用时抛出 ImportError
6. **线程安全**：CostTracker 和 TAPTracer 使用锁；异步组件为单线程

### 添加新编译器

1. 在 `teragent/core/compilers/` 中创建新文件
2. 继承 `TAPCompiler` 并实现 `compile()`
3. 使用 `TAPCompilerRegistry.register("name", MyCompiler)` 注册
4. 在 `teragent/core/prompts/` 中添加系统提示
5. 在 `tests/test_compilers.py` 中添加测试

### 添加新适配器

1. 在 `teragent/core/adapters/` 中创建新文件
2. 继承 `TAPAdapter` 并实现 `send()` 和 `stream()`
3. 使用 `TAPAdapterRegistry.register("name", MyAdapter)` 注册
4. 在 `teragent/streaming/stream_events.py` 中添加流解析器
5. 在 `tests/test_openai_adapter.py`（或新文件）中添加测试

### 添加新工具

1. 继承 `BaseTool`
2. 定义 `name`、`description`、`parameters_schema`
3. 设置 `_safety` 和 `_concurrency_safe`
4. 实现 `execute(params) -> ToolResult`
5. 使用 `ToolRegistry.register(tool_instance)` 注册

## 编码规范

### Python 风格

- 遵循 PEP 8，行长度最多 100 个字符
- 所有文件使用 `from __future__ import annotations`
- 数据结构使用 dataclasses
- 抽象类使用 `ABC` + `@abstractmethod`
- 优先使用 `async/await` 而非回调
- 全面使用类型提示

### 错误处理

- 工具**绝不**向调用者抛出异常 —— 返回 `ToolResult(success=False, error=...)`
- 可恢复问题使用 `logger.warning()`
- 不可恢复问题使用 `logger.error()`
- 开发/追踪信息使用 `logger.debug()`

### 测试

- 所有新功能必须包含测试
- 使用 `pytest` 和 `pytest-asyncio`（严格模式）
- 同时测试成功和失败路径
- 需要提供者的测试使用 `MockAdapter`
- 保持测试聚焦且独立

## 许可证

通过为 TerAgent 贡献，你同意你的贡献将在 Apache License Version 2.0 下授权。
