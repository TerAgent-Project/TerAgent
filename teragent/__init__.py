"""teragent — Terminal AI Agent Library

TAP IR + Model-Specific Compilation — Apache-2.0 Licensed

Quick start:
    import teragent

    # Method 1: Create a provider from DriverConfig (recommended)
    from teragent.config import DriverConfig
    driver_cfg = DriverConfig(
        adapter="openai_compatible",
        identity="glm_5",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="GLM_API_KEY",
        model="glm-5",
        compiler="glm_5",
    )
    provider = teragent.create_provider_from_config(driver_cfg)

    # Method 2: Create a provider manually (Compiler + Adapter auto-combined)
    provider = teragent.create_provider(
        compiler="glm_5",
        adapter="openai_compatible",
        model="glm-5",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="GLM_API_KEY",
    )

    # Method 3: Load from config file
    full_config = teragent.load_full_config()
    drivers = full_config["drivers"]
    provider = teragent.create_provider_from_config(drivers["openai_compatible.glm_5"])

    # Execute a TAP request
    response = await provider.execute_tap(teragent.TAPRequest(
        meta={"task_id": "1.1", "intent": "code_generation"},
        instruction="实现用户登录模块",
        constraints=["Python 3.10+"],
        output_format_hint="<file path='...'>完整代码</file>",
    ))

    # Extract files from response
    files = teragent.extract_files_from_response(response.raw_text, task_id="1.1")

    # Run deterministic code checks
    task_list = [teragent.TaskInfo(id="1.1", title="Login module", status="completed")]
    report, data = teragent.run_deterministic_checks("/project", task_list)

    # Build prompt with custom template
    messages = teragent.build_prompt(
        system_template="You are {role}. Task: {task}",
        context={"role": "engineer", "task": "implement login"},
    )

    # Retry with exponential backoff
    result = await teragent.retry_with_backoff(
        fn=lambda: provider.chat(messages=[...]),
        max_retries=3,
        validate=lambda r: [] if r else ["empty response"],
    )

    # Self-RL data constitution — TAP tracing + DPO pairs
    tracer = teragent.TAPTracer(trace_dir="/project/.agent/traces")
    provider.set_tracer(tracer)  # Auto-trace all TAP calls

    # After running deterministic checks, record the result
    await tracer.record_checklist("1.1", checklist_data)

    # Export DPO preference pairs for fine-tuning
    pairs = tracer.export_dpo_pairs()
    tracer.export_dpo_pairs_jsonl()  # Write to file

Optional dependencies:
    Some components require extra packages that are not installed by default.
    Install them with pip extras:

    - teragent[vector]  — VectorIndexer (LanceDB + numpy)
    - teragent[ast]     — CodeIndexer (tree-sitter)
    - teragent[graph]   — ReferenceGraph (networkx)
    - teragent[all]     — All optional dependencies

    These components are lazily imported and will raise ImportError only
    when actually used without the corresponding extra installed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================================
# Core data types
# ============================================================================

from teragent.core.tap import (
    TAPRequest, TAPResponse, TAPCostRecord, CompiledPrompt,
    MultimodalContent, DesktopContext, LongHorizonConfig, LongHorizonStatus,
    CostTracker,
)

# Compiler ABC + Registry
from teragent.core.compiler import TAPCompiler, TAPCompilerRegistry

# Adapter ABC + Registry
from teragent.core.adapter import TAPAdapter, TAPAdapterRegistry

# Compiler implementations (triggers registration)
from teragent.core.compilers import (
    DefaultCompiler, GLMCompiler, AnthropicCompiler, DeepSeekCompiler,
    DeepSeekV4Compiler, GLM5Compiler, MiniMaxM3Compiler,
)

# ModelProvider (Compiler + Adapter composition)
from teragent.core.provider import ModelProvider

# Core types
from teragent.core.types import ToolSafety
from teragent.core.types import Message, MessageRole, MessageType, messages_to_api_format, messages_from_dicts

# ============================================================================
# Config layer
# ============================================================================

from teragent.config.driver_config import DriverConfig
from teragent.config.loader import (
    load_driver_configs,
    load_pipeline_config,
    load_full_config,
    create_provider_from_config,
    get_driver_config,
    resolve_api_key,
    resolve_api_key_detailed,
    infer_compiler,
    load_typed_config,
    audit_config_security,
)

# API Key security
from teragent.config.api_key_security import (
    ApiKeyVault,
    ResolvedKey,
    SecurityFinding,
    SecuritySeverity,
    SecurityError,
    mask_api_key,
    detect_api_key_provider,
    audit_config_security as audit_api_key_security,
    audit_env_file,
    get_vault,
)

# Typed configuration dataclasses
from teragent.config.teragent_config import TerAgentConfig, AgentdSectionConfig
from teragent.config.agent_loop_config import AgentLoopConfig
from teragent.config.circuit_breaker_config import (
    CircuitBreakerConfig,
    BudgetConfig,
    FailureBreakerConfig,
    LatencyBreakerConfig,
    ProgressDetectorConfig,
)
from teragent.config.context_management_config import ContextManagementConfig
from teragent.config.tools_config import ToolsConfig
from teragent.config.file_safety_config import FileSafetyConfig
from teragent.config.session_config import SessionConfig
from teragent.config.permission_config import PermissionConfig
from teragent.config.hooks_config import HooksConfig
from teragent.config.recovery_config import RecoveryConfig
from teragent.config.streaming_config import StreamingConfig
from teragent.config.coordination_config import CoordinationConfig
from teragent.config.execution_pipeline_config import ExecutionPipelineConfig
from teragent.config.model_fallback_config import ModelFallbackConfig

# ============================================================================
# Pipeline primitives
# ============================================================================

from teragent.pipeline.extractor import extract_files_from_response
from teragent.pipeline.prompt_builder import (
    build_prompt,
    build_subagent_prompt,
    validate_prompt_tokens,
    DEFAULT_SYSTEM_TEMPLATE,
)
from teragent.pipeline.checklist import (
    run_deterministic_checks,
    TaskInfo,
    check_code_quality,
    check_requirements,
    check_runnable,
    check_file_conflicts,
    check_fallback_files,
)
from teragent.pipeline.retry import retry_with_backoff

# TAP Tracing + DPO pair generation
from teragent.pipeline.tracing import (
    TAPTracer,
    TraceRecord,
    DPOPair,
    DataConstitution,
    TraceStats,
)

# ============================================================================
# Prompt selection
# ============================================================================

from teragent.core.prompts import get_system_prompt_for_intent as _get_system_prompt_for_intent
from teragent.core.prompts import list_intents, list_compiler_types

def get_system_prompt_for_intent(
    intent: str,
    compiler_type: str = "default",
) -> str:
    """Get the system prompt for a given intent and compiler type.

    Centralized prompt management.
    All prompts are managed through teragent/core/prompts/ and selected
    based on intent (design, plan, replan, execute, review, chat,
    chat_friendly, sub_agent, cuda_triton) and compiler type (default, glm,
    anthropic, deepseek, deepseek_v4, glm_5, minimax_m3).

    Args:
        intent: One of: design | plan | replan | execute | review |
            chat | chat_friendly | sub_agent | code_generation | cuda_triton
        compiler_type: One of: default | glm | anthropic | deepseek | deepseek_v4 | glm_5 | minimax_m3

    Returns:
        System prompt string for the given intent and compiler type.
        Falls back to "default" compiler if specific one not found.
        Returns empty string if intent is not recognized.

    Examples:
        # Get GLM-optimized design prompt
        prompt = teragent.get_system_prompt_for_intent("design", "glm")

        # Get default chat prompt
        prompt = teragent.get_system_prompt_for_intent("chat")

        # Different compilers produce different prompts for the same intent
        default_prompt = teragent.get_system_prompt_for_intent("review", "default")
        anthropic_prompt = teragent.get_system_prompt_for_intent("review", "anthropic")
        # anthropic_prompt uses XML tags, default_prompt uses plain text
    """
    return _get_system_prompt_for_intent(intent, compiler_type=compiler_type)


# Prompt templates
from teragent.core.prompts import (
    DESIGN_PROMPT_DEFAULT, DESIGN_PROMPT_GLM, DESIGN_PROMPT_ANTHROPIC, DESIGN_PROMPT_DEEPSEEK,
    DESIGN_PROMPT_DEEPSEEK_V4, DESIGN_PROMPT_GLM_5, DESIGN_PROMPT_MINIMAX_M3,
    PLAN_PROMPT_DEFAULT, PLAN_PROMPT_GLM, PLAN_PROMPT_ANTHROPIC, PLAN_PROMPT_DEEPSEEK,
    PLAN_PROMPT_DEEPSEEK_V4, PLAN_PROMPT_GLM_5, PLAN_PROMPT_MINIMAX_M3,
    REPLAN_PROMPT_DEFAULT, REPLAN_PROMPT_GLM, REPLAN_PROMPT_ANTHROPIC, REPLAN_PROMPT_DEEPSEEK,
    REPLAN_PROMPT_DEEPSEEK_V4, REPLAN_PROMPT_GLM_5, REPLAN_PROMPT_MINIMAX_M3,
    REVIEW_PROMPT_DEFAULT, REVIEW_PROMPT_GLM, REVIEW_PROMPT_ANTHROPIC, REVIEW_PROMPT_DEEPSEEK,
    REVIEW_PROMPT_DEEPSEEK_V4, REVIEW_PROMPT_GLM_5, REVIEW_PROMPT_MINIMAX_M3,
    AGENT_PROMPT_DEFAULT, AGENT_PROMPT_GLM, AGENT_PROMPT_ANTHROPIC, AGENT_PROMPT_DEEPSEEK,
    AGENT_PROMPT_DEEPSEEK_V4, AGENT_PROMPT_GLM_5, AGENT_PROMPT_MINIMAX_M3,
    CHAT_PROMPT_DEFAULT, CHAT_PROMPT_GLM, CHAT_PROMPT_ANTHROPIC, CHAT_PROMPT_DEEPSEEK,
    CHAT_PROMPT_DEEPSEEK_V4, CHAT_PROMPT_GLM_5, CHAT_PROMPT_MINIMAX_M3,
    SUB_AGENT_PROMPT_DEFAULT, SUB_AGENT_PROMPT_GLM, SUB_AGENT_PROMPT_ANTHROPIC, SUB_AGENT_PROMPT_DEEPSEEK,
    SUB_AGENT_PROMPT_DEEPSEEK_V4, SUB_AGENT_PROMPT_GLM_5, SUB_AGENT_PROMPT_MINIMAX_M3,
    EXECUTE_PROMPT_DEFAULT, EXECUTE_PROMPT_GLM, EXECUTE_PROMPT_ANTHROPIC, EXECUTE_PROMPT_DEEPSEEK,
    EXECUTE_PROMPT_DEEPSEEK_V4, EXECUTE_PROMPT_GLM_5, EXECUTE_PROMPT_MINIMAX_M3,
    CUDA_TRITON_PROMPT_GLM_5,
)

# ============================================================================
# Security architecture
# ============================================================================

from teragent.security.permission import (
    PermissionManager,
    PermissionLevel,
    EnhancedPermissionManager,
    PermissionRule,
    PermissionEffect,
)
from teragent.security.file_state import FileStateTracker
from teragent.security.file_writer import write_files_safely, atomic_write_file
from teragent.security.ai_permission_classifier import AIPermissionClassifier
from teragent.security.sandbox import execute_in_sandbox, check_command_safety
from teragent.security.audit import AuditLogger
from teragent.security.firecracker_sandbox import FirecrackerSandbox

# ============================================================================
# Context management — core components always available
# ============================================================================

from teragent.context.context_window import ContextWindow
from teragent.context.microcompactor import Microcompactor
from teragent.context.auto_compact import AutoCompactor
from teragent.context.memory import (
    load_agent_md,
    save_agent_md,
    merge_agent_md,
    extract_rules,
)
# DependencyReporter moved to lazy import (requires optional deps)

# ============================================================================
# Reliability
# ============================================================================

from teragent.reliability.circuit_breaker import (
    CircuitBreakerManager,
    CostBudgetTracker,
    CostBudgetConfig,
    BudgetCheckResult,
    BreakerState,
    ConsecutiveFailureBreaker,
    LatencyBreaker,
    ProgressDetector,
    ModelBreakerConfig,
    ModelBreakerState,
    ModelCircuitBreakerManager,
)
from teragent.reliability.budget import StepBudget, DEFAULT_MAX_STEPS
from teragent.reliability.recovery import (
    RecoveryType,
    RecoveryStats,
    RecoveryManagerConfig,
    RecoveryManager,
    is_context_overflow_error,
    is_retryable_error,
    DegradationChain,
    LongHorizonRecoveryManager,
    RateLimitInfo,
    RateLimitHandler,
)

# ============================================================================
# EventBus
# ============================================================================

from teragent.event_bus import EventBus

# ============================================================================
# Coordination
# ============================================================================

from teragent.coordination.message_bus import AgentMessageBus, AgentMessage
from teragent.coordination.sub_agent_manager import (
    SubAgentManager,
    AgentMode,
    SubAgentStatus,
    SubAgentInfo,
)

# ============================================================================
# Intent
# ============================================================================

from teragent.intent.classifier import IntentClassifier, IntentType
from teragent.intent.confirmation import ConfirmationGate

# ============================================================================
# Hooks
# ============================================================================

from teragent.hooks.manager import (
    HookEvent,
    HookDecision,
    HookContext,
    HookResult,
    Hook,
    ShellHook,
    PythonHook,
    HookManager,
)
from teragent.hooks.builtin.audit_hook import AuditHook
from teragent.hooks.builtin.dangerous_command_hook import DangerousCommandHook

# ============================================================================
# Session
# ============================================================================

from teragent.session.persistence import SessionPersistence, SessionData, SessionInfo

# ============================================================================
# Streaming
# ============================================================================

from teragent.streaming.stream_events import (
    StreamEventType,
    StreamEvent,
    ToolCallAccumulator,
    StreamingChatResult,
    OpenAIStreamParser,
    AnthropicStreamParser,
)
from teragent.streaming.streaming_executor import (
    StreamingToolExecutor,
    StreamingExecutionStats,
)

# ============================================================================
# Tools — base abstractions for tool system
# ============================================================================

from teragent.tools.base import BaseTool, ToolResult
from teragent.tools.registry import ToolRegistry
from teragent.tools.orchestrator import ToolOrchestrator, MAX_CONCURRENT_TOOLS

# ============================================================================
# AgentLoop — central orchestration class
# ============================================================================

from teragent.agent_loop import AgentLoop

# ============================================================================
# Trigger compiler/adapter registration
# ============================================================================

import teragent.core.compilers  # noqa: F401 — registers: default, glm, anthropic, deepseek, deepseek_v4, deepseek_v4_flash, deepseek_v4_pro, glm_5, minimax_m3
import teragent.core.adapters   # noqa: F401 — registers: openai_compatible, anthropic_native, minimax_native, mock


# ============================================================================
# Lazy imports for optional-dependency components
# ============================================================================

def __getattr__(name: str):
    """Lazy-load components that require optional dependencies.

    This allows ``import teragent`` to succeed even when optional
    dependencies (lancedb, tree-sitter, networkx) are not installed.
    The components will only raise ImportError when actually accessed.

    Optional extras:
        - teragent[vector] — VectorIndexer
        - teragent[ast]    — CodeIndexer
        - teragent[graph]  — ReferenceGraph
    """
    if name == "DependencyReporter":
        from teragent.context.dependency_reporter import DependencyReporter
        return DependencyReporter
    if name == "TaskProtocol":
        from teragent.context.dependency_reporter import TaskProtocol
        return TaskProtocol
    if name == "CodeIndexer":
        from teragent.context.code_indexer import CodeIndexer
        return CodeIndexer
    if name == "ReferenceGraph":
        from teragent.context.reference_graph import ReferenceGraph
        return ReferenceGraph
    if name == "VectorIndexer":
        from teragent.context.vector_indexer import VectorIndexer
        return VectorIndexer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ============================================================================
# Factory function
# ============================================================================

def create_provider(
    compiler: str | TAPCompiler,
    adapter: str | TAPAdapter,
    model: str,
    base_url: str = "",
    api_key: str = "",
    api_key_env: str = "",
    timeout: float = 300.0,
    extra_headers: dict | None = None,
    enable_fake_tools: bool = False,
    fallback: ModelProvider | None = None,
    circuit_breaker: Any | None = None,
    tracer: Any | None = None,
    compiler_variant: str = "",
    **kwargs,
) -> ModelProvider:
    """Factory function to create a ModelProvider from Registry

    Automatically creates Compiler and Adapter instances from the registry,
    then combines them into a ModelProvider.

    API key resolution:
        - If api_key_env is provided, resolves from environment variable
          (with .env file fallback via python-dotenv)
        - If only api_key is provided, uses it directly
          (logs an info message recommending api_key_env instead)
        - Recommended: use api_key_env for security

    Args:
        compiler: Compiler name (str) or TAPCompiler instance
            Available names: "default", "glm", "anthropic", "deepseek",
                "deepseek_v4", "deepseek_v4_flash", "deepseek_v4_pro",
                "glm_5", "minimax_m3"
        adapter: Adapter name (str) or TAPAdapter instance
            Available names: "openai_compatible", "anthropic_native", "minimax_native", "mock"
        model: Model identifier string (e.g., "glm-5", "claude-sonnet-4-20250514")
        base_url: API base URL (required for non-mock adapters)
        api_key: API key string (direct, not recommended for production)
        api_key_env: Environment variable name for API key (recommended)
        timeout: HTTP request timeout in seconds (default 300.0)
        extra_headers: Additional HTTP headers for the adapter
        enable_fake_tools: Whether to inject fake tools for distillation detection
        fallback: Fallback ModelProvider for automatic degradation
        circuit_breaker: CircuitBreakerManager for cost/failure tracking
        tracer: TAPTracer for auto-tracing TAP calls
        **kwargs: Additional arguments passed to Adapter constructor

    Returns:
        ModelProvider instance with Compiler + Adapter composed

    Raises:
        ValueError: If compiler or adapter name is not registered

    Examples:
        # GLM via OpenAI-compatible protocol (recommended: api_key_env)
        provider = create_provider(
            compiler="glm_5",
            adapter="openai_compatible",
            model="glm-5",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key_env="GLM_API_KEY",
        )

        # Claude via OpenRouter (OpenAI protocol + Anthropic compiler)
        provider = create_provider(
            compiler="anthropic",
            adapter="openai_compatible",
            model="anthropic/claude-sonnet-4-20250514",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
        )

        # Claude via Anthropic native protocol
        provider = create_provider(
            compiler="anthropic",
            adapter="anthropic_native",
            model="claude-sonnet-4-20250514",
            base_url="https://api.anthropic.com/v1",
            api_key_env="ANTHROPIC_API_KEY",
        )

        # Mock for testing
        provider = create_provider(
            compiler="default",
            adapter="mock",
            model="mock-model",
        )

        # From DriverConfig
        from teragent.config import DriverConfig
        cfg = DriverConfig(adapter="openai_compatible", identity="glm_5",
                           model="glm-5", compiler="glm_5",
                           base_url="...", api_key_env="GLM_API_KEY")
        provider = create_provider(**cfg.to_create_provider_kwargs())
    """
    # Resolve Compiler
    if isinstance(compiler, str):
        # Handle compiler_variant for DeepSeek V4 (e.g., compiler="deepseek_v4", variant="flash")
        if compiler_variant and compiler == "deepseek_v4":
            resolved_name = f"deepseek_v4_{compiler_variant}"
            compiler_instance = TAPCompilerRegistry.create(resolved_name, **kwargs)
        else:
            compiler_instance = TAPCompilerRegistry.create(compiler, **kwargs)
    else:
        compiler_instance = compiler

    # Resolve API key (centralized via ApiKeyVault)
    from teragent.config.api_key_security import ApiKeyVault, mask_api_key as _mask
    vault = ApiKeyVault()

    resolved_api_key = ""
    if api_key_env:
        # Preferred path: resolve from environment variable
        resolved = vault.resolve(api_key_env)
        resolved_api_key = resolved.key
        if not resolved.found:
            logger.warning(f"API key not found for env var: {api_key_env}")
    elif api_key:
        # Fallback: direct key (store in vault for tracking, no deprecation
        # warning here since this is a programmatic call, not config)
        resolved = vault.store_direct(api_key, name=f"create_provider:{model}")
        resolved_api_key = resolved.key
        logger.info(
            f"Using directly provided API key for model '{model}' "
            f"(masked: {_mask(resolved_api_key)}). "
            f"Consider using api_key_env for better security."
        )

    # Resolve Adapter
    if isinstance(adapter, str):
        adapter_kwargs: dict[str, Any] = {}
        if adapter == "openai_compatible":
            adapter_kwargs = {
                "base_url": base_url,
                "api_key": resolved_api_key,
                "timeout": timeout,
                "extra_headers": extra_headers,
                "enable_fake_tools": enable_fake_tools,
            }
        elif adapter == "anthropic_native":
            adapter_kwargs = {
                "base_url": base_url,
                "api_key": resolved_api_key,
                "timeout": timeout,
                "enable_fake_tools": enable_fake_tools,
            }
        elif adapter == "mock":
            adapter_kwargs = {}
        else:
            adapter_kwargs = kwargs

        adapter_instance = TAPAdapterRegistry.create(adapter, **adapter_kwargs)
    else:
        adapter_instance = adapter

    return ModelProvider(
        compiler=compiler_instance,
        adapter=adapter_instance,
        model=model,
        fallback=fallback,
        circuit_breaker=circuit_breaker,
        tracer=tracer,
    )


# ============================================================================
# Version
# ============================================================================

__version__ = "0.0.1"


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Version
    "__version__",
    # Core types
    "TAPRequest",
    "TAPResponse",
    "TAPCostRecord",
    "CompiledPrompt",
    # Extended TAP types (V4/M3/GLM-5)
    "MultimodalContent",
    "DesktopContext",
    "LongHorizonConfig",
    "LongHorizonStatus",
    "CostTracker",
    # ABC + Registry
    "TAPCompiler",
    "TAPCompilerRegistry",
    "TAPAdapter",
    "TAPAdapterRegistry",
    # Compiler classes (legacy)
    "DefaultCompiler",
    "GLMCompiler",
    "AnthropicCompiler",
    "DeepSeekCompiler",
    # Compiler classes (new — V4/M3/GLM-5)
    "DeepSeekV4Compiler",
    "GLM5Compiler",
    "MiniMaxM3Compiler",
    # Provider
    "ModelProvider",
    # Factory
    "create_provider",
    # Core types
    "ToolSafety",
    "Message",
    "MessageRole",
    "MessageType",
    "messages_to_api_format",
    "messages_from_dicts",
    # Config layer
    "DriverConfig",
    "load_driver_configs",
    "load_pipeline_config",
    "load_full_config",
    "create_provider_from_config",
    "get_driver_config",
    "resolve_api_key",
    "resolve_api_key_detailed",
    "infer_compiler",
    "audit_config_security",
    # API Key Security
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
    # Typed config
    "TerAgentConfig",
    "AgentdSectionConfig",
    "AgentLoopConfig",
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
    "load_typed_config",
    # Pipeline primitives
    "extract_files_from_response",
    "build_prompt",
    "build_subagent_prompt",
    "validate_prompt_tokens",
    "DEFAULT_SYSTEM_TEMPLATE",
    "run_deterministic_checks",
    "TaskInfo",
    "check_code_quality",
    "check_requirements",
    "check_runnable",
    "check_file_conflicts",
    "check_fallback_files",
    "retry_with_backoff",
    # TAP Tracing + DPO pair generation
    "TAPTracer",
    "TraceRecord",
    "DPOPair",
    "DataConstitution",
    "TraceStats",
    # Prompt selection
    "get_system_prompt_for_intent",
    "list_intents",
    "list_compiler_types",
    # Prompt templates
    "DESIGN_PROMPT_DEFAULT", "DESIGN_PROMPT_GLM", "DESIGN_PROMPT_ANTHROPIC", "DESIGN_PROMPT_DEEPSEEK",
    "DESIGN_PROMPT_DEEPSEEK_V4", "DESIGN_PROMPT_GLM_5", "DESIGN_PROMPT_MINIMAX_M3",
    "PLAN_PROMPT_DEFAULT", "PLAN_PROMPT_GLM", "PLAN_PROMPT_ANTHROPIC", "PLAN_PROMPT_DEEPSEEK",
    "PLAN_PROMPT_DEEPSEEK_V4", "PLAN_PROMPT_GLM_5", "PLAN_PROMPT_MINIMAX_M3",
    "REPLAN_PROMPT_DEFAULT", "REPLAN_PROMPT_GLM", "REPLAN_PROMPT_ANTHROPIC", "REPLAN_PROMPT_DEEPSEEK",
    "REPLAN_PROMPT_DEEPSEEK_V4", "REPLAN_PROMPT_GLM_5", "REPLAN_PROMPT_MINIMAX_M3",
    "REVIEW_PROMPT_DEFAULT", "REVIEW_PROMPT_GLM", "REVIEW_PROMPT_ANTHROPIC", "REVIEW_PROMPT_DEEPSEEK",
    "REVIEW_PROMPT_DEEPSEEK_V4", "REVIEW_PROMPT_GLM_5", "REVIEW_PROMPT_MINIMAX_M3",
    "AGENT_PROMPT_DEFAULT", "AGENT_PROMPT_GLM", "AGENT_PROMPT_ANTHROPIC", "AGENT_PROMPT_DEEPSEEK",
    "AGENT_PROMPT_DEEPSEEK_V4", "AGENT_PROMPT_GLM_5", "AGENT_PROMPT_MINIMAX_M3",
    "CHAT_PROMPT_DEFAULT", "CHAT_PROMPT_GLM", "CHAT_PROMPT_ANTHROPIC", "CHAT_PROMPT_DEEPSEEK",
    "CHAT_PROMPT_DEEPSEEK_V4", "CHAT_PROMPT_GLM_5", "CHAT_PROMPT_MINIMAX_M3",
    "SUB_AGENT_PROMPT_DEFAULT", "SUB_AGENT_PROMPT_GLM", "SUB_AGENT_PROMPT_ANTHROPIC", "SUB_AGENT_PROMPT_DEEPSEEK",
    "SUB_AGENT_PROMPT_DEEPSEEK_V4", "SUB_AGENT_PROMPT_GLM_5", "SUB_AGENT_PROMPT_MINIMAX_M3",
    "EXECUTE_PROMPT_DEFAULT", "EXECUTE_PROMPT_GLM", "EXECUTE_PROMPT_ANTHROPIC", "EXECUTE_PROMPT_DEEPSEEK",
    "EXECUTE_PROMPT_DEEPSEEK_V4", "EXECUTE_PROMPT_GLM_5", "EXECUTE_PROMPT_MINIMAX_M3",
    "CUDA_TRITON_PROMPT_GLM_5",
    # Security
    "PermissionManager",
    "PermissionLevel",
    "EnhancedPermissionManager",
    "PermissionRule",
    "PermissionEffect",
    "FileStateTracker",
    "write_files_safely",
    "atomic_write_file",
    "AIPermissionClassifier",
    "execute_in_sandbox",
    "check_command_safety",
    "AuditLogger",
    "FirecrackerSandbox",
    # Context management — core always available
    "ContextWindow",
    "Microcompactor",
    "AutoCompactor",
    "DependencyReporter",
    "TaskProtocol",
    "load_agent_md",
    "save_agent_md",
    "merge_agent_md",
    "extract_rules",
    # Context management — optional (lazy-loaded)
    "CodeIndexer",
    "ReferenceGraph",
    "VectorIndexer",
    # Reliability
    "CircuitBreakerManager",
    "CostBudgetTracker",
    "CostBudgetConfig",
    "BudgetCheckResult",
    "BreakerState",
    "ConsecutiveFailureBreaker",
    "LatencyBreaker",
    "ProgressDetector",
    "ModelBreakerConfig",
    "ModelBreakerState",
    "ModelCircuitBreakerManager",
    "StepBudget",
    "DEFAULT_MAX_STEPS",
    # Reliability — Recovery
    "RecoveryType",
    "RecoveryStats",
    "RecoveryManagerConfig",
    "RecoveryManager",
    "is_context_overflow_error",
    "is_retryable_error",
    "DegradationChain",
    "LongHorizonRecoveryManager",
    "RateLimitInfo",
    "RateLimitHandler",
    # EventBus
    "EventBus",
    # Coordination
    "AgentMessageBus",
    "AgentMessage",
    "SubAgentManager",
    "AgentMode",
    "SubAgentStatus",
    "SubAgentInfo",
    # Intent
    "IntentClassifier",
    "IntentType",
    "ConfirmationGate",
    # Hooks
    "HookEvent",
    "HookDecision",
    "HookContext",
    "HookResult",
    "Hook",
    "ShellHook",
    "PythonHook",
    "HookManager",
    "AuditHook",
    "DangerousCommandHook",
    # Session
    "SessionPersistence",
    "SessionData",
    "SessionInfo",
    # Streaming
    "StreamEventType",
    "StreamEvent",
    "ToolCallAccumulator",
    "StreamingChatResult",
    "OpenAIStreamParser",
    "AnthropicStreamParser",
    "StreamingToolExecutor",
    "StreamingExecutionStats",
    # Tools — base abstractions
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    "ToolOrchestrator",
    "MAX_CONCURRENT_TOOLS",
    # AgentLoop — central orchestration
    "AgentLoop",
]
