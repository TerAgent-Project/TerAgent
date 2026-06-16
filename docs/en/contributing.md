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
├── orchestration/     # Multi-Agent orchestration (v0.2.0+)
│   ├── agent.py       # Agent base class — independent provider, tools, handoffs
│   ├── orchestrator.py # Orchestrator — strategy pattern, run() and run_stream()
│   ├── handoff.py     # Handoff + HandoffTool + HandoffInputFilter
│   ├── shared_state.py # SharedState + ScopedState
│   ├── rwlock.py      # AsyncRWLock — writer-priority async read-write lock
│   ├── run_context.py # RunContext + UsageTracker
│   ├── cancellation.py # CancellationToken — thread-safe cooperative cancellation
│   ├── guardrail.py   # Guardrail — input/output, fail-fast parallel check engine
│   ├── checkpoint.py  # OrchestrationCheckpoint — state snapshot save/restore
│   ├── approval.py    # ApprovalGate — human-in-the-loop tool approval
│   ├── agent_hooks.py # AgentHooks — lifecycle hooks (start/end/handoff/tool/model)
│   └── patterns/      # Execution patterns: Sequential, Swarm, Parallel, Conditional, Loop
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
│   ├── vector_indexer.py  # LanceDB semantic search (optional)
│   ├── retention_tracker.py # Context retention tracking
│   ├── dependency_reporter.py # Dependency analysis reporting
│   └── profiles.py    # Context management profiles
├── pipeline/          # Pipeline primitives
│   ├── extractor.py   # File extraction from model output
│   ├── prompt_builder.py  # Template-based prompt construction
│   ├── checklist.py   # Deterministic code verification
│   ├── retry.py       # Exponential backoff retry
│   ├── subagent_worker.py # Sub-agent pipeline worker
│   └── tracing.py     # TAP tracing + DPO pair generation
├── streaming/         # Streaming execution
│   ├── streaming_executor.py  # StreamingToolExecutor
│   └── stream_events.py      # Stream event types + parsers
├── tools/             # Tool system
│   ├── base.py        # BaseTool ABC + ToolResult
│   ├── registry.py    # ToolRegistry (with categories, intent recommendation)
│   ├── orchestrator.py # ToolOrchestrator
│   ├── decorator.py   # @tool decorator — Python function → BaseTool
│   ├── schema_gen.py  # JSON Schema auto-generation from function signatures
│   ├── agent_tool.py  # AgentTool — Agent-as-Tool (delegation, control returns)
│   ├── orchestrator_tool.py # OrchestratorTool — nested multi-Agent orchestration as tool
│   ├── auth.py        # AuthScheme, AuthCredential, AuthManager
│   ├── toolpack.py    # ToolPack — grouped tools with shared lifecycle
│   ├── mcp_toolset.py # MCPToolset — MCP server connection management
│   ├── openapi_toolset.py # OpenAPIToolset — auto-generate tools from OpenAPI spec
│   ├── result_cache.py # ResultCache — TTL + LRU caching for tool results
│   ├── desktop.py     # DesktopTool (desktop automation)
│   ├── hub/           # ToolHubClient — tool marketplace client
│   └── builtin/       # Built-in tools: file, code, web, analysis
├── router/            # Smart model routing
│   └── model_router.py # ModelRouter, RoutingTable, PipelineManager
├── long_horizon/      # Long-horizon autonomous tasks
│   ├── task_manager.py # LongHorizonTaskManager
│   ├── checkpoint.py  # CheckpointStore
│   ├── self_evaluation.py # SelfEvaluator
│   ├── strategy_switch.py # StrategySwitcher
│   ├── progress.py    # ProgressTracker
│   └── types.py       # Long-horizon data types
├── benchmark/         # Benchmark framework
│   └── benchmark.py   # BenchmarkRunner
├── intent/            # Intent classification
│   ├── classifier.py  # IntentClassifier
│   └── confirmation.py # ConfirmationGate
├── hooks/             # Hook system
│   ├── manager.py     # HookManager
│   └── builtin/       # Built-in hooks (audit, dangerous command)
├── coordination/      # DEPRECATED — migrated to orchestration/ in v0.2.0
│   └── __init__.py    # Only deprecation warnings and migration guide
├── session/           # Session persistence
│   └── persistence.py # SessionPersistence (SQLite)
├── config/            # Configuration system
│   ├── loader.py      # Config loading + provider creation
│   ├── teragent_config.py    # Top-level config
│   ├── driver_config.py      # Driver config
│   ├── agent_config.py       # Agent config
│   ├── orchestration_config.py # Orchestration config
│   ├── mcp_config.py        # MCP server config
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
