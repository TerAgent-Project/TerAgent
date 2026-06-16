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
│   ├── compilers/     # 具体编译器：default, glm, glm_5, glm_52, glm_5v_turbo, anthropic, deepseek, deepseek_v4, minimax_m3
│   ├── adapters/      # 具体适配器：openai_compatible, anthropic_native, glm_native, minimax_native, mock
│   └── prompts/       # 意图专属系统提示词
├── orchestration/     # 多 Agent 编排（v0.2.0+）
│   ├── agent.py       # Agent 基类 — 独立 provider、工具集、handoff
│   ├── orchestrator.py # Orchestrator — 策略模式，支持 run() 和 run_stream()
│   ├── handoff.py     # Handoff + HandoffTool + HandoffInputFilter
│   ├── shared_state.py # SharedState + ScopedState
│   ├── rwlock.py      # AsyncRWLock — 写者优先异步读写锁
│   ├── run_context.py # RunContext + UsageTracker
│   ├── cancellation.py # CancellationToken — 线程安全协作式取消
│   ├── guardrail.py   # Guardrail — 输入/输出护栏，fail-fast 并行检查引擎
│   ├── checkpoint.py  # OrchestrationCheckpoint — 编排状态快照保存/恢复
│   ├── approval.py    # ApprovalGate — Human-in-the-loop 工具审批
│   ├── agent_hooks.py # AgentHooks — 生命周期钩子（start/end/handoff/tool/model）
│   └── patterns/      # 执行模式：Sequential, Swarm, Parallel, Conditional, Loop
├── security/          # 安全体系
│   ├── permission.py  # PermissionManager + EnhancedPermissionManager
│   ├── sandbox.py     # 沙箱执行（3 级）
│   ├── file_writer.py # 2PC 文件写入
│   ├── file_state.py  # 文件状态追踪
│   ├── audit.py       # 审计日志
│   └── ai_permission_classifier.py  # AI 权限分类器
├── reliability/       # 可靠性系统
│   ├── circuit_breaker.py  # 4 种熔断器 + 管理器
│   ├── budget.py      # 步数预算 + CrossModelCostTracker
│   └── recovery.py    # 恢复管理器 + DegradationChain + RateLimitHandler
├── context/           # 上下文管理
│   ├── context_window.py  # Token 预算估算器
│   ├── auto_compact.py    # 自动上下文压缩
│   ├── microcompactor.py  # 细粒度压缩
│   ├── memory.py      # .agent.md 持久化记忆
│   ├── code_indexer.py    # tree-sitter AST 索引（可选）
│   ├── reference_graph.py # networkx 依赖图（可选）
│   ├── vector_indexer.py  # LanceDB 语义搜索（可选）
│   ├── retention_tracker.py # 上下文保留追踪
│   ├── dependency_reporter.py # 依赖分析报告
│   └── profiles.py    # 上下文管理配置方案
├── pipeline/          # Pipeline 原语
│   ├── extractor.py   # 从模型输出中提取文件
│   ├── prompt_builder.py  # 基于模板的提示词构建
│   ├── checklist.py   # 确定性代码验证
│   ├── retry.py       # 指数退避重试
│   ├── subagent_worker.py # 子 Agent Pipeline 工作器
│   └── tracing.py     # TAP 追踪 + DPO 对生成
├── streaming/         # 流式执行
│   ├── streaming_executor.py  # StreamingToolExecutor
│   └── stream_events.py      # 流事件类型 + 解析器
├── tools/             # 工具系统
│   ├── base.py        # BaseTool ABC + ToolResult
│   ├── registry.py    # ToolRegistry（分类注册、意图推荐）
│   ├── orchestrator.py # ToolOrchestrator
│   ├── decorator.py   # @tool 装饰器 — Python 函数 → BaseTool
│   ├── schema_gen.py  # 从函数签名自动生成 JSON Schema
│   ├── agent_tool.py  # AgentTool — Agent-as-Tool（委托，控制权返回）
│   ├── orchestrator_tool.py # OrchestratorTool — 嵌套多 Agent 编排作为工具
│   ├── auth.py        # AuthScheme, AuthCredential, AuthManager
│   ├── toolpack.py    # ToolPack — 分组工具，共享生命周期
│   ├── mcp_toolset.py # MCPToolset — MCP 服务器连接管理
│   ├── openapi_toolset.py # OpenAPIToolset — 从 OpenAPI 规范自动生成工具
│   ├── result_cache.py # ResultCache — TTL + LRU 工具结果缓存
│   ├── desktop.py     # DesktopTool（桌面自动化）
│   ├── hub/           # ToolHubClient — 工具市场客户端
│   └── builtin/       # 内置工具：file, code, web, analysis
├── router/            # 智能模型路由
│   └── model_router.py # ModelRouter, RoutingTable, PipelineManager
├── long_horizon/      # 长时自主任务
│   ├── task_manager.py # LongHorizonTaskManager
│   ├── checkpoint.py  # CheckpointStore
│   ├── self_evaluation.py # SelfEvaluator
│   ├── strategy_switch.py # StrategySwitcher
│   ├── progress.py    # ProgressTracker
│   └── types.py       # 长时任务数据类型
├── benchmark/         # 基准测试框架
│   └── benchmark.py   # BenchmarkRunner
├── intent/            # 意图分类
│   ├── classifier.py  # IntentClassifier
│   └── confirmation.py # ConfirmationGate
├── hooks/             # Hook 系统
│   ├── manager.py     # HookManager
│   └── builtin/       # 内置 Hook（审计、危险命令）
├── coordination/      # 已废弃 — v0.2.0 迁移至 orchestration/
│   └── __init__.py    # 仅保留废弃警告及迁移指引
├── session/           # 会话持久化
│   └── persistence.py # SessionPersistence (SQLite)
├── config/            # 配置系统
│   ├── loader.py      # 配置加载 + provider 创建
│   ├── teragent_config.py    # 顶层配置
│   ├── driver_config.py      # Driver 配置
│   ├── agent_config.py       # Agent 配置
│   ├── orchestration_config.py # 编排配置
│   ├── mcp_config.py        # MCP 服务器配置
│   └── ...            # 15+ 类型化配置 dataclass
├── event_bus.py       # EventBus（信号驱动通信）
├── utils/             # 工具函数
│   ├── exceptions.py  # 自定义异常
│   ├── token_counter.py # Token 估算
│   ├── text.py        # 文本工具
│   └── tracing.py     # 追踪工具
└── agent_loop.py      # AgentLoop（中央编排）
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
