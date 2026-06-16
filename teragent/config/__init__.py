"""teragent.config — Configuration dataclasses and loaders

This module provides typed configuration structures for the teragent library.

Key classes:
    TerAgentConfig: Top-level configuration (all sections)
    AgentdSectionConfig: Agentd section configuration
    DriverConfig: Per-driver configuration (adapter, compiler, credentials, etc.)
    CircuitBreakerConfig: Circuit breaker configuration (budget, failure, latency, progress)
    BudgetConfig: Budget breaker configuration
    FailureBreakerConfig: Failure breaker configuration
    LatencyBreakerConfig: Latency breaker configuration
    ProgressDetectorConfig: Progress detector configuration
    ContextManagementConfig: Context window and compaction configuration
    ToolsConfig: Tool execution and output persistence configuration
    FileSafetyConfig: File safety and state tracking configuration
    PermissionConfig: Permission system configuration
    HooksConfig: Hook system configuration
    RecoveryConfig: Error recovery configuration
    StreamingConfig: Streaming execution configuration
    CoordinationConfig: Multi-agent coordination configuration
    ExecutionPipelineConfig: Execution pipeline driver assignments
    ModelFallbackConfig: Model fallback/degradation configuration
    SessionConfig: Session persistence configuration
    ApiKeyVault: Centralized API key management (Phase 9)
    ResolvedKey: Resolved API key with metadata (Phase 9)
    SecurityFinding: Security finding from config audit (Phase 9)
    SecuritySeverity: Severity level for security findings (Phase 9)
    SecurityError: Security-related error (Phase 9)
    AgentLoopConfig: AgentLoop tunable parameters (Phase 0.1 migrated)

Key functions:
    load_typed_config(): Load full typed config from agent.toml
    load_driver_configs(): Load driver configs from a TOML dict
    load_pipeline_config(): Load pipeline config from a TOML dict
    load_full_config(): Load full config from a TOML dict
    create_provider_from_config(): Create a ModelProvider from a DriverConfig
    get_driver_config(): Get a single driver config by name
    resolve_api_key(): Resolve an API key by provider (Phase 9)
    resolve_api_key_detailed(): Resolve an API key with detailed metadata (Phase 9)
    infer_compiler(): Infer compiler from model name
    detect_api_key_provider(): Detect API key provider from key string (Phase 9)
    audit_config_security(): Audit config for API key security issues (Phase 9)
    audit_env_file(): Audit .env file for security best practices (Phase 9)
    mask_api_key(): Safely mask an API key for display (Phase 9)
    get_vault(): Get the global ApiKeyVault instance (Phase 9)
"""

# Phase 2: Driver config
# AgentLoop config (Phase 0.1 migrated)
from teragent.config.agent_loop_config import AgentLoopConfig

# Orchestration config (Phase 1 W2)
from teragent.config.agent_config import AgentConfig
from teragent.config.orchestration_config import OrchestrationConfig

# Phase 9: API Key Security
from teragent.config.api_key_security import (
    ApiKeyVault,
    ResolvedKey,
    SecurityError,
    SecurityFinding,
    SecuritySeverity,
    audit_env_file,
    detect_api_key_provider,
    get_vault,
    mask_api_key,
)
from teragent.config.api_key_security import (
    audit_config_security as audit_api_key_security,
)
from teragent.config.circuit_breaker_config import (
    BudgetConfig,
    CircuitBreakerConfig,
    FailureBreakerConfig,
    LatencyBreakerConfig,
    ProgressDetectorConfig,
)
from teragent.config.context_management_config import ContextManagementConfig
from teragent.config.coordination_config import CoordinationConfig
from teragent.config.driver_config import DriverConfig
from teragent.config.execution_pipeline_config import ExecutionPipelineConfig
from teragent.config.file_safety_config import FileSafetyConfig
from teragent.config.hooks_config import HooksConfig

# Phase 2: Loaders (re-exported)
from teragent.config.loader import (
    audit_config_security,
    create_provider_from_config,
    get_driver_config,
    infer_compiler,
    load_driver_configs,
    load_full_config,
    load_pipeline_config,
    load_typed_config,
    resolve_api_key,
    resolve_api_key_detailed,
)
from teragent.config.model_fallback_config import ModelFallbackConfig
from teragent.config.permission_config import PermissionConfig
from teragent.config.recovery_config import RecoveryConfig
from teragent.config.session_config import SessionConfig
from teragent.config.streaming_config import StreamingConfig

# Phase 5: All sub-config dataclasses
from teragent.config.teragent_config import AgentdSectionConfig, TerAgentConfig
from teragent.config.tools_config import ToolsConfig

# Phase 2 W5: MCP Config
from teragent.config.mcp_config import MCPServerConfig

__all__ = [
    # Top-level config
    "TerAgentConfig",
    "AgentdSectionConfig",
    # Driver config (Phase 2)
    "DriverConfig",
    # Sub-config dataclasses (Phase 5)
    "CircuitBreakerConfig",
    "BudgetConfig",
    "FailureBreakerConfig",
    "LatencyBreakerConfig",
    "ProgressDetectorConfig",
    "ContextManagementConfig",
    "ToolsConfig",
    "FileSafetyConfig",
    "SessionConfig",
    "PermissionConfig",
    "HooksConfig",
    "RecoveryConfig",
    "StreamingConfig",
    "CoordinationConfig",
    "ExecutionPipelineConfig",
    "ModelFallbackConfig",
    # AgentLoop config (Phase 0.1 migrated)
    "AgentLoopConfig",
    # Orchestration config (Phase 1 W2)
    "AgentConfig",
    "OrchestrationConfig",
    # Loader functions
    "load_typed_config",
    "load_driver_configs",
    "load_pipeline_config",
    "load_full_config",
    "create_provider_from_config",
    "get_driver_config",
    "resolve_api_key",
    "resolve_api_key_detailed",
    "infer_compiler",
    "audit_config_security",
    # API Key Security (Phase 9)
    "ApiKeyVault",
    "ResolvedKey",
    "SecurityFinding",
    "SecuritySeverity",
    "SecurityError",
    "mask_api_key",
    "detect_api_key_provider",
    "audit_api_key_security",
    "audit_env_file",
    "get_vault",
    # MCP Config (Phase 2 W5)
    "MCPServerConfig",
]
