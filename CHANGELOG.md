# Changelog

All notable changes to TerAgent are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] — 2025-03-04

> **里程碑版本**：完整实现三阶段（Phase 1/2/3）多 Agent 编排框架，从 `coordination` 模块全面迁移至 `orchestration` 模块。

---

### 🔥 Breaking Changes

#### 1. `teragent.coordination` 模块废弃并移除

- **删除** `teragent/coordination/glm5v_coordinator.py` — GLM-5V + GLM-5.2 跨模型协调工作流（`GLM52VCoordinatedWorkflow`、`CoordinationMode`、`CoordinationConfig` 等 ~800 行）
- **删除** `teragent/coordination/message_bus.py` — 基于 `asyncio.Queue` 邮箱的异步消息传递（`AgentMessageBus`、`AgentMessage`、`BROADCAST`）
- **删除** `teragent/coordination/sub_agent_manager.py` — 子 Agent 生命周期管理（`SubAgentManager`、`AgentMode`、`SubAgentStatus`、`SubAgentInfo`）
- **替换为** `teragent/coordination/__init__.py` — 仅保留废弃警告及迁移指引

**迁移映射：**

| 旧 API | 新 API |
|--------|--------|
| `SubAgentManager` | `Orchestrator(mode=OrchestrationMode.SEQUENTIAL)` |
| `AgentMessageBus` | `EventBus + SharedState` |
| `AgentMode.SYNC/ASYNC/FORK` | `OrchestrationMode.SEQUENTIAL/PARALLEL/SWARM` |
| `SubAgentInfo` | `Agent` |
| `GLM52VCoordinatedWorkflow` | `Orchestrator` + Conditional/Swarm 模式 |

#### 2. 明文 API Key 支持移除

- `ResolvedKey` 不再包含 `is_plaintext` 字段
- `ApiKeyVault.resolve_from_settings()` 不再回退到 `api_key` 明文字段，仅支持 `api_key_env`（环境变量 / .env 文件）
- `audit_config_security()` 不再检测明文密钥和弱密钥
- **影响**：TOML 配置中必须使用 `api_key_env` 替代直接写入 `api_key`

#### 3. 旧配置格式支持移除

- `config/loader.py` 移除 `_OLD_DRIVER_NAME_MAP`、`_parse_old_driver_name()`、`_load_old_format()`、`_is_new_format()` 等旧格式兼容代码
- `config/execution_pipeline_config.py` 不再支持 `design_model`/`plan_model` 等旧键名，必须使用 `design_driver`/`plan_driver`
- `router/model_router.py` 移除旧格式回退逻辑
- **影响**：必须使用 `protocol.identity` 两级嵌套格式

#### 4. 审计日志模块级函数移除

- `security/audit.py` 移除所有模块级向后兼容函数（`set_db_path()`、`log_audit()`、`audit_log()`、`query_by_action()` 等 10 个函数）
- 仅保留 `AuditLogger` 类和 `DEFAULT_DB_PATH` 常量
- **影响**：所有消费者必须使用 `AuditLogger` 实例

#### 5. `EnhancedPermissionManager.ai_classifier` setter 移除

- 运行时重新赋值 AI 分类器不再支持，必须在构造时设置

---

### 🚀 New Features

#### Phase 1（W1–W4）：编排核心

##### 多 Agent 编排框架 — `teragent/orchestration/`（19 个新文件）

| 文件 | 功能 |
|------|------|
| `agent.py` | `Agent` 基类 — 独立 provider、工具集、handoff、guardrail、hooks、MCP 服务器 |
| `orchestrator.py` | `Orchestrator` 主入口 — 策略模式，支持 `run()` 和 `run_stream()` |
| `handoff.py` | `Handoff` + `HandoffTool` + `HandoffInputFilter` — Swarm 模式控制转移 |
| `shared_state.py` | `SharedState` + `ScopedState` — 跨 Agent 有作用域的共享状态 |
| `rwlock.py` | `AsyncRWLock` — 写者优先的异步读写锁 |
| `run_context.py` | `RunContext` + `UsageTracker` — 执行上下文与 Token 用量追踪 |
| `cancellation.py` | `CancellationToken` — 线程安全的协作式取消信号 |
| `guardrail.py` | `Guardrail` — 输入/输出护栏，fail-fast 并行检查引擎 |
| `checkpoint.py` | `OrchestrationCheckpoint` — 编排状态快照保存/恢复（原子写入） |
| `approval.py` | `ApprovalGate` — Human-in-the-loop 工具审批（支持参数修改） |
| `agent_hooks.py` | `AgentHooks` — Agent 生命周期钩子（on_start/end/handoff/tool/model） |
| `patterns/sequential.py` | 顺序执行模式（A→B→C） |
| `patterns/swarm.py` | Swarm 去中心化模式（Agent 驱动 handoff） |
| `patterns/parallel.py` | 并行扇出/扇入模式（fan-out → fan-in） |
| `patterns/conditional.py` | 条件路由模式（受限于 Swarm — 仅 router agent 可 handoff） |
| `patterns/loop.py` | 循环迭代模式（Generator-Critic，支持退出条件） |

**关键架构特性：**
- **策略模式**：`Orchestrator` 通过 `OrchestrationPattern` 接口委托执行
- **Handoff 机制**：LLM 调用 `transfer_to_{agent_name}` → `HandoffTool` 返回 `__handoff__` 标记 → 模式检测并切换 Agent
- **Guardrail fail-fast**：`asyncio.wait(FIRST_COMPLETED)` 并行检查，首个失败立即取消其余
- **SharedState 作用域**：`session`（默认）、`agent`（前缀隔离）、`global`（全局共享）
- **流式编排**：`run_stream()` 返回 `AsyncIterator[OrchestrationEvent]`
- **嵌套编排**：`Orchestrator.as_tool()` → `OrchestratorTool`

##### @tool 装饰器与工具基础设施

| 文件 | 功能 |
|------|------|
| `tools/decorator.py` | `@tool` 装饰器 — Python 函数 → `BaseTool`，自动提取名称/文档/Schema |
| `tools/schema_gen.py` | JSON Schema 自动生成 — 从函数签名推断参数类型（支持 `Optional`、`Union`、`Annotated`） |
| `tools/agent_tool.py` | `AgentTool` — Agent-as-Tool 模式（委托后控制权返回，区别于 Handoff） |

##### 内置工具集 — `tools/builtin/`

| 文件 | 工具 | 安全级别 |
|------|------|----------|
| `builtin/file.py` | `ReadFileTool`、`WriteFileTool`、`ListDirectoryTool`、`SearchFilesTool` | READ_ONLY / SAFE_WRITE |
| `builtin/code.py` | `CodeExecutionTool` — 子进程执行，超时控制 | DESTRUCTIVE |
| `builtin/web.py` | `WebSearchTool`、`WebScrapeTool` — SearXNG/SerpAPI + BeautifulSoup | READ_ONLY |
| `builtin/analysis.py` | `AnalyzeCodeTool`、`SearchCodeSemanticTool` — AST/正则分析 | READ_ONLY |

##### 新配置模块

| 文件 | 功能 |
|------|------|
| `config/agent_config.py` | `AgentConfig` — 映射 `[agents.{name}]` TOML 段 |
| `config/orchestration_config.py` | `OrchestrationConfig` — 映射 `[orchestration]` TOML 段，支持 5 种模式 |

---

#### Phase 2（W5–W8）：协议集成与高级模式

##### MCP 工具集成 — `tools/mcp_toolset.py`

- **`MCPToolset`** — MCP 服务器连接管理，支持三种传输：`stdio`（本地进程）、`sse`（Server-Sent Events）、`streamable_http`
- **`MCPTool`** — 远程 MCP 工具的本地代理
- **`MCPConnectionPool`** — 连接池（LRU，最大 16 连接，空闲超时 300s，健康检查 60s）
- **`MCPServerConfig`** — MCP 服务器连接参数配置

##### OpenAPI 工具自动生成 — `tools/openapi_toolset.py`

- **`OpenAPIToolset`** — 从 OpenAPI 3.0 / Swagger 2.0 规范自动生成工具
- **`OpenAPIOperationTool`** — 单个 HTTP 操作封装为 `BaseTool`
- 自动推断安全级别：GET/HEAD/OPTIONS → `READ_ONLY`，其他 → `SAFE_WRITE`
- 支持 `tool_filter` 过滤指定 operationId

##### ToolPack — `tools/toolpack.py`

- 分组相关工具，共享生命周期和资源
- `on_start()` / `on_stop()` 异步回调，幂等性保证
- 支持 `async with` 上下文管理器

##### Tool Hub 市场客户端 — `tools/hub/`

- **`ToolHubClient`** — 工具市场客户端（搜索、安装、发布、卸载）
- **`HubTool`** — 远程工具代理，通过 Hub API 执行
- **`ToolHubEntry`** — 搜索结果条目（名称/版本/作者/评分/下载量）
- 默认 Hub URL：`https://hub.teragent.dev/api/v1`
- 本地磁盘缓存 `hub_cache.json`

##### 工具注册表增强 — `tools/registry.py`

- **`ToolInfo`** — 扩展元数据（name, category, description, safety, source, tags）
- **分类注册**：`register_category()` / `get_tools_by_category()`
- **意图推荐**：`get_tools_for_intent()` — 按类别/标签/描述模糊匹配
- **便捷注册**：`register_toolpack()` / `register_mcp_toolset()`

---

#### Phase 3（W9–W12）：安全、可靠性与生产就绪

##### 认证基础设施 — `tools/auth.py`

- **`AuthScheme`** — 声明认证类型（bearer / api_key / oauth2 / basic）
- **`AuthCredential`** — 统一凭证存储（api_key, api_key_env, client_id, client_secret, access_token, refresh_token）
- **`AuthManager`** — 集中认证管理器，`apply_auth()` 自动注入 HTTP 头/查询参数
- 安全特性：凭证仅内存存储，`__repr__` 掩码所有敏感字段

##### 嵌套编排 — `tools/orchestrator_tool.py`

- **`OrchestratorTool`** — 将整个 `Orchestrator` 包装为 `BaseTool`
- 支持多级嵌套：Agent → OrchestratorTool → 内部 Orchestrator → ...
- 区别于 `AgentTool`（单 Agent 委托）：执行完整多 Agent 编排流程

##### 工具结果缓存 — `tools/result_cache.py`

- **`ResultCache`** — TTL + LRU 缓存（`OrderedDict` + `asyncio.Lock`）
- 确定性键：`sha256(sorted_json(params))[:16]`
- **`CacheStats`** — 命中率/未命中/淘汰统计
- 默认：`max_size=128`，`default_ttl=60s`
- `AgentTool` 可选集成：执行前缓存查找，成功结果写入，失败跳过

##### 编排检查点 — `orchestration/checkpoint.py`

- `save()` — 原子写入（tempfile + `os.fsync` + `os.replace`）
- `restore()` — 从指定或最新检查点恢复
- `list_checkpoints()` / `delete_checkpoint()` / `cleanup(keep_last=5)`

##### Human-in-the-Loop 审批 — `orchestration/approval.py`

- `request_approval()` — 阻塞等待外部审批
- `approve(id, modified_params?)` — 审批通过，可选修改工具参数
- `reject(id, reason)` — 审批拒绝
- `get_pending_approvals()` — 外部 UI/监控
- 超时机制：默认 300s

---

### 🔄 Changed

#### `teragent/__init__.py`

- **移除**：`AgentMessageBus`、`AgentMessage`、`SubAgentManager`、`AgentMode`、`SubAgentStatus`、`SubAgentInfo`
- **新增**：`Agent`、`Handoff`、`HandoffInputFilter`、`HandoffTool`、`Orchestrator`、`OrchestrationConfig`、`OrchestrationMode`、`OrchestrationResult`、`SharedState`、`RunContext`、`CancellationToken`、`AgentHooks`、`tool`、`DecoratorTool`、`AgentTool`、`all_builtin_tools` 等 20+ 导出

#### `teragent/agent_loop.py`

- **新增** `agent: Agent | None = None` 参数 — 支持 Agent 对象初始化
- **移除** `message_bus`、`sub_agent_manager` 参数
- `model` 和 `tool_registry` 改为 `Optional`
- 无 `model` 也无 `agent` 时抛出 `ValueError`

#### `teragent/core/types.py`

- **新增** `MessageType.HANDOFF = "handoff"` 枚举值
- **新增** `Message.handoff(content, target_agent)` 工厂方法
- **新增** `Message.is_handoff` 便捷属性

#### `teragent/core/compilers/glm_52.py`

- `create_coordinated_workflow()` 标记废弃，返回 `None`，引导使用 `teragent.orchestration.Orchestrator`

#### `teragent/event_bus.py`

- 移除已废弃的 `_shared: dict[str, Any]` 属性

#### `teragent/config/teragent_config.py`

- **新增** `orchestration: OrchestrationConfig` 字段
- **新增** `agents: dict[str, AgentConfig]` 字段
- **新增** `mcp_servers: dict[str, MCPServerConfig]` 字段
- `from_toml()` 修复 frozen dataclass 崩溃（使用 `object.__setattr__`）

#### `teragent/tools/base.py`

- **新增** `needs_approval: bool = False` 类属性
- `check_permissions()` 简化为两层：`READ_ONLY → 始终允许`，其他 → `level ≥ 1`
- `to_registry_metadata()` 包含 `needs_approval` 字段

#### `teragent/security/audit.py`

- 移除所有模块级向后兼容函数，仅保留 `AuditLogger` 类

#### `teragent/security/file_writer.py` / `security/permission.py` / `security/sandbox.py`

- 统一迁移至 `AuditLogger` 实例方式

#### `teragent/hooks/builtin/audit_hook.py`

- 迁移至 `AuditLogger` 实例

#### `teragent/pipeline/subagent_worker.py`

- 移除 `_sync_append_trace()` 旧版 JSONL 追踪
- 移除 `DANGEROUS_PATTERNS` 列表
- 无 tracer 时仅日志跳过，不再回退到旧版追踪

#### `teragent/benchmark/benchmark.py`

- `VisionCoordinationBenchmark` 临时禁用，等待迁移至 orchestration

---

### 🗑️ Removed

| 文件 | 说明 |
|------|------|
| `teragent/coordination/glm5v_coordinator.py` | GLM-5V + GLM-5.2 协调工作流 |
| `teragent/coordination/message_bus.py` | Agent 消息总线 |
| `teragent/coordination/sub_agent_manager.py` | 子 Agent 管理器 |
| `tests/test_glm5v_coordinator.py` | 协调器测试 |
| `tests/test_message_bus.py` | 消息总线测试 |
| `tests/test_sub_agent_manager.py` | 子 Agent 管理器测试 |

---

### 📚 Documentation

#### 移除（引用已废弃 coordination 模块或内容过时）

| 文件 | 说明 |
|------|------|
| `docs/EVALUATION_THREE_MODELS.md` | 三模型评估报告（已被四模型报告替代，两者均引用已废弃 coordination） |
| `docs/EVALUATION_FOUR_MODELS.md` | 四模型评估报告（引用已废弃 5V-Turbo coordination） |
| `docs/EVALUATION_GLM5.md` | 项目评估报告（内容与用户文档定位不符） |
| `docs/glm_52_stability_report.md` | GLM-5.2 稳定性报告（大量引用已废弃 5V-Turbo coordination 稳定性） |

#### 移动

- `docs/deployment_guide_ascend.md` → `docs/zh/deployment_guide_ascend.md` — 中文部署指南移至中文文档目录

#### 新增

- `docs/zh/glm_52_guide.md` — GLM-5.2 使用指南中文版
- `docs/zh/long_horizon_guide.md` — 长时任务指南中文版
- `docs/zh/multimodal_guide.md` — 多模态指南中文版

#### 更新

- `docs/index.md` — 清理已删文件引用，简化为语言选择 + 部署链接，版本号 0.1.3 → 0.2.0
- `docs/en/index.md` — 清理已删文件引用，版本号 0.1.3 → 0.2.0
- `docs/zh/index.md` — 清理已删文件引用，三篇指南从英文链接改为中文链接，版本号 0.1.3 → 0.2.0
- `docs/en/security.md` / `docs/zh/security.md` — 版本号更新
- `README.md` / `README_zh.md` — 版本号徽章 0.1.3 → 0.2.0，How It Was Built / 构建方式 按三阶段重写，移除已删评测报告引用

---

### 🧪 Tests

#### 新增

| 文件 | 测试类数 | 覆盖范围 |
|------|----------|----------|
| `tests/integration/test_orchestration_e2e.py` | 10 | Phase 1 编排 E2E（顺序/Swarm/AgentTool/装饰器/取消/状态/Handoff/Hooks/Usage/Builtin） |
| `tests/test_phase2.py` | 10 | Phase 2 集成（Parallel/Conditional/Loop/Guardrail/ToolPack/OpenAPI/MCP/Registry/Config/模式覆盖） |
| `tests/test_phase3.py` | 12 | Phase 3 集成（Checkpoint/Approval/OrchestratorTool/Hub/Auth/RWLock/Cache/边界/恢复/并发） |

#### 修改

- `tests/test_config_loader.py` — 移除旧格式测试类（`TestParseOldDriverName`、`test_load_old_format` 等）
- `tests/integration/test_e2e.py` — 移除 `TestCoordinationE2E` 类
- `tests/integration/__init__.py` — 添加模块文档字符串

---

### 📊 Statistics

| 指标 | 数值 |
|------|------|
| 新增文件 | 35+ |
| 删除文件 | 9 |
| 修改文件 | 18 |
| 新增代码行数 | ~8,000+ |
| 删除代码行数 | ~2,000+ |
| 新增测试类 | 32 |
| 删除测试类 | 4+ |
| Phase 1 W1–W4 | Agent/Orchestrator/Handoff/工具装饰器/Builtin 工具 |
| Phase 2 W5–W8 | MCP/OpenAPI/ToolPack/Hub/并行/条件/循环模式 |
| Phase 3 W9–W12 | Checkpoint/Approval/Auth/Cache/RWLock/嵌套编排 |

