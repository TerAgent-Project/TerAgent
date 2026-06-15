# Contributing to TerAgent

Thank you for your interest in contributing to TerAgent! This guide covers the development setup, coding standards, and contribution process.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [bun](https://bun.sh/) or npm (for running tests)
- git

### Install Development Dependencies

```bash
# Clone the repository
git clone https://github.com/teragent/teragent.git
cd teragent

# Install with development dependencies
pip install -e ".[dev]"

# Install all optional dependencies (for full testing)
pip install -e ".[all]"
```

### Running Tests

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

### Linting

```bash
# Run ruff linter
ruff check teragent/

# Auto-fix issues
ruff check --fix teragent/
```

### Type Checking

```bash
mypy teragent/
```

## Code Architecture

### Module Organization

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
│   └── recovery.py    # Recovery manager + DegradationChain + RateLimitHandler
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

### Key Design Principles

1. **Orthogonal Composition**: Compiler × Adapter — add a new model OR protocol, not both
2. **Defense in Depth**: 7-layer permissions, 6-layer command defense, 2PC file writes, 3-level sandbox
3. **Advisory-First**: Circuit breakers warn rather than block by default
4. **Fire-and-Forget Events**: EventBus never blocks the main loop
5. **Lazy Imports**: Optional dependencies raise ImportError only when used
6. **Thread Safety**: CostTracker and TAPTracer use locks; async components are single-threaded

### Adding a New Compiler

1. Create a new file in `teragent/core/compilers/`
2. Subclass `TAPCompiler` and implement `compile()`
3. Register with `TAPCompilerRegistry.register("name", MyCompiler)`
4. Add system prompts in `teragent/core/prompts/`
5. Add tests in `tests/test_compilers.py`

### Adding a New Adapter

1. Create a new file in `teragent/core/adapters/`
2. Subclass `TAPAdapter` and implement `send()` and `stream()`
3. Register with `TAPAdapterRegistry.register("name", MyAdapter)`
4. Add stream parser in `teragent/streaming/stream_events.py`
5. Add tests in `tests/test_openai_adapter.py` (or new file)

### Adding a New Tool

1. Subclass `BaseTool`
2. Define `name`, `description`, `parameters_schema`
3. Set `_safety` and `_concurrency_safe`
4. Implement `execute(params) -> ToolResult`
5. Register with `ToolRegistry.register(tool_instance)`

## Coding Standards

### Python Style

- Follow PEP 8 with line length up to 100 characters
- Use `from __future__ import annotations` in all files
- Use dataclasses for data structures
- Use `ABC` + `@abstractmethod` for abstract classes
- Prefer `async/await` over callbacks
- Use type hints throughout

### Error Handling

- Tools **never** raise exceptions to callers — return `ToolResult(success=False, error=...)`
- Use `logger.warning()` for recoverable issues
- Use `logger.error()` for unrecoverable issues
- Use `logger.debug()` for development/tracing info

### Testing

- All new features must include tests
- Use `pytest` with `pytest-asyncio` (strict mode)
- Test both success and failure paths
- Use `MockAdapter` for tests that need a provider
- Keep tests focused and independent

## License

By contributing to TerAgent, you agree that your contributions will be licensed under the Apache License Version 2.0.
