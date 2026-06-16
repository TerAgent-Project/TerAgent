# teragent/agent_loop.py
"""AgentLoop — The central orchestration class that drives the tool-calling loop.

Composes all existing modules and provides the main agent interaction loop.

Architecture (tool loop pattern):
  1. User sends a message
  2. IntentClassifier determines intent (CHAT / DEBUG / CREATE_PROJECT)
  3. If CREATE_PROJECT, ConfirmationGate asks for user approval
  4. Filter tools by intent (from AgentLoopConfig.intent_tools)
  5. Call model (streaming or batch depending on mode)
  6. If model returns tool_calls, execute them
     (via StreamingToolExecutor or ToolOrchestrator)
  7. Add tool results to messages
  8. Loop back to step 5 until: no tool_calls, step budget exhausted,
     or max steps reached
  9. Context compaction at each iteration before model call (proactive, not reactive)
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from teragent.orchestration.agent import Agent

__all__ = [
    "AgentLoop",
]

from teragent.config.agent_loop_config import AgentLoopConfig
from teragent.context.auto_compact import AutoCompactor
from teragent.context.context_window import ContextWindow
# NOTE: teragent.coordination is deprecated; sub-agent coordination
# is now provided by teragent.orchestration.Orchestrator.
from teragent.core.provider import ModelProvider
from teragent.core.tap import LongHorizonConfig, TAPRequest
from teragent.core.types import Message, MessageRole, MessageType
from teragent.event_bus import EventBus
from teragent.hooks.manager import HookManager
from teragent.intent.classifier import IntentClassifier, IntentType
from teragent.intent.confirmation import ConfirmationGate
from teragent.long_horizon.task_manager import LongHorizonTaskManager
from teragent.long_horizon.types import LongHorizonResult
from teragent.reliability.budget import CrossModelCostTracker, StepBudget
from teragent.reliability.circuit_breaker import CircuitBreakerManager
from teragent.reliability.recovery import RecoveryManager, RecoveryType
from teragent.router.model_router import ModelRouter, RoutingDecision, RoutingReason
from teragent.security.permission import EnhancedPermissionManager
from teragent.session.persistence import SessionPersistence
from teragent.streaming.streaming_executor import (
    StreamingToolExecutor,
)
from teragent.tools.base import ToolResult
from teragent.tools.orchestrator import ToolOrchestrator
from teragent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentLoop:
    """The central orchestration class that drives the tool-calling loop.

    Composes ModelProvider, ToolRegistry, and all cross-cutting concerns
    (reliability, context, security, intent, streaming, etc.) into a
    single cohesive agent interaction loop.

    Usage::

        loop = AgentLoop(
            model=provider,
            tool_registry=registry,
            config=AgentLoopConfig(),
        )
        messages = await loop.run("帮我写一个贪吃蛇游戏")
    """

    def __init__(
        self,
        model: ModelProvider | None = None,
        tool_registry: ToolRegistry | None = None,
        config: AgentLoopConfig | None = None,
        event_bus: EventBus | None = None,
        context_window: ContextWindow | None = None,
        auto_compactor: AutoCompactor | None = None,
        step_budget: StepBudget | None = None,
        circuit_breaker: CircuitBreakerManager | None = None,
        recovery_manager: RecoveryManager | None = None,
        permission_manager: EnhancedPermissionManager | None = None,
        intent_classifier: IntentClassifier | None = None,
        confirmation_gate: ConfirmationGate | None = None,
        hook_manager: HookManager | None = None,
        session_persistence: SessionPersistence | None = None,
        streaming_executor: StreamingToolExecutor | None = None,
        long_horizon_manager: LongHorizonTaskManager | None = None,
        model_router: ModelRouter | None = None,
        cost_tracker: CrossModelCostTracker | None = None,
        agent: Agent | None = None,
    ) -> None:
        # --- Agent-based initialization (Phase 1 W3) ---
        if agent is not None:
            # Build provider from agent
            provider = agent.resolve_provider()
            if model is None:
                model = provider
            # Build tool registry from agent tools
            if tool_registry is None:
                tool_registry = ToolRegistry()
                for tool in agent.tools:
                    tool_registry.register(tool)
            # Override config with agent's max_steps if not explicitly configured
            if config is None:
                config = AgentLoopConfig(max_tool_steps=agent.max_steps)

        # --- Core dependencies ---
        if model is None:
            raise ValueError(
                "AgentLoop requires either a 'model' (ModelProvider) or 'agent' parameter. "
                "Pass agent=Agent(...) or model=ModelProvider(...) explicitly."
            )
        if tool_registry is None:
            raise ValueError(
                "AgentLoop requires either a 'tool_registry' or 'agent' parameter. "
                "Pass agent=Agent(...) or tool_registry=ToolRegistry() explicitly."
            )
        self._model = model
        self._tool_registry = tool_registry
        self._config = config or AgentLoopConfig()

        # --- Optional cross-cutting components ---
        self._event_bus = event_bus
        self._context_window = context_window
        self._auto_compactor = auto_compactor
        self._step_budget = step_budget
        self._circuit_breaker = circuit_breaker
        self._recovery_manager = recovery_manager
        self._permission_manager = permission_manager
        self._intent_classifier = intent_classifier
        self._confirmation_gate = confirmation_gate
        self._message_bus = None        # deprecated: use Orchestrator instead
        self._sub_agent_manager = None   # deprecated: use Orchestrator instead
        self._hook_manager = hook_manager
        self._session_persistence = session_persistence
        self._streaming_executor = streaming_executor
        self._long_horizon_manager = long_horizon_manager
        self._model_router = model_router
        self._cost_tracker = cost_tracker

        # --- Validate config ---
        validation_warnings = self._config.validate()
        if validation_warnings:
            for w in validation_warnings:
                logger.warning(f"AgentLoopConfig: {w}")

        # --- Internal tool orchestrator (always created) ---
        self._tool_orchestrator = ToolOrchestrator(
            tool_registry=tool_registry,
            permission_level=0,
            hook_manager=hook_manager,
            enhanced_perm_manager=permission_manager,
        )

        # --- Streaming mode state ---
        self._streaming_mode: str = "auto"  # "auto" | "streaming" | "batch"
        self._max_streaming_retries: int = self._config.max_streaming_retries

        # --- Permission level ---
        self._permission_level: int = 0

        # --- Recovery stats ---
        self._recovery_stats: dict[str, Any] = {
            "streaming_mode": self._streaming_mode,
            "streaming_retries": 0,
            "batch_fallbacks": 0,
            "truncation_recoveries": 0,
            "context_compactions": 0,
        }

        # --- Loop metrics ---
        self._total_steps: int = 0
        self._total_model_calls: int = 0

        # --- Streaming tool results buffer ---
        # Populated by _call_model_streaming(), consumed by _tool_loop()
        # 修复 H9: 添加文档说明 — 此列表非线程安全，不应并发调用 run()
        # 若需并发，需改为 per-invocation 的局部变量（需较大重构）
        self._streaming_tool_results: list[tuple[dict, ToolResult]] = []

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def streaming_mode(self) -> str:
        """Returns current streaming mode: "auto" | "streaming" | "batch"."""
        return self._streaming_mode

    @property
    def model(self) -> ModelProvider:
        """Returns the ModelProvider."""
        return self._model

    @property
    def tool_registry(self) -> ToolRegistry:
        """Returns the ToolRegistry."""
        return self._tool_registry

    @property
    def config(self) -> AgentLoopConfig:
        """Returns the AgentLoopConfig."""
        return self._config

    # ==================================================================
    # Streaming configuration
    # ==================================================================

    def set_streaming_config(
        self,
        mode: str | None = None,
        max_streaming_retries: int | None = None,
    ) -> None:
        """Update streaming configuration at runtime.

        Args:
            mode: "auto" | "streaming" | "batch". None = unchanged.
            max_streaming_retries: Max retry count. None = unchanged.
        """
        if mode is not None:
            if mode in ("auto", "streaming", "batch"):
                self._streaming_mode = mode
            else:
                # Invalid value → fallback to auto
                self._streaming_mode = "auto"
            self._recovery_stats["streaming_mode"] = self._streaming_mode

        if max_streaming_retries is not None:
            self._max_streaming_retries = max_streaming_retries

    # ==================================================================
    # Core entry point
    # ==================================================================

    async def run(
        self,
        user_input: str,
        messages: list[Message] | None = None,
        system_prompt: str = "",
    ) -> list[Message]:
        """Run the agent loop for a user input.

        Full lifecycle:
          1. Classify intent
          2. If CREATE_PROJECT, confirm via gate
          3. Filter tools by intent
          4. Prepend/replace system prompt
          5. If CREATE_PROJECT, delegate to Orchestrator (was SubAgentManager)
          6. Restore session state (if applicable)
          7. Execute tool loop until done
          8. Emit agent_done event
          9. Persist session
          10. Return updated message list

        Args:
            user_input: The user's input text
            messages: Existing conversation messages (for continuation)
            system_prompt: System prompt to prepend

        Returns:
            Updated message list including all new messages from this turn
        """
        start_time = time.time()

        # Initialize message list
        if messages is None:
            messages = []
        else:
            messages = list(messages)

        # 1. Append user message
        user_msg = Message.user_input(user_input)
        messages.append(user_msg)

        # 2. Classify intent
        intent = IntentType.CHAT  # default
        if self._intent_classifier:
            try:
                intent = await self._intent_classifier.classify(user_input)
            except Exception as e:
                logger.warning(f"Intent classification failed: {e}, defaulting to CHAT")

        # Emit intent classified event
        if self._event_bus:
            await self._event_bus.emit(
                "intent_classified",
                intent=intent.value,
                user_input=user_input[:200],
            )

        logger.info(f"Intent classified: {intent.value} for: {user_input[:80]}")

        # 3. If CREATE_PROJECT, ask for confirmation
        if intent == IntentType.CREATE_PROJECT and self._confirmation_gate:
            try:
                confirmed = await self._confirmation_gate.confirm_create_project(
                    user_input
                )
                if not confirmed:
                    messages.append(
                        Message.assistant_text("用户取消了项目创建。")
                    )
                    return messages
            except Exception as e:
                logger.warning(f"Confirmation gate error: {e}, proceeding anyway")

        # 4. Filter tools by intent
        allowed_tools = self._filter_tools_by_intent(intent)

        # 5. Prepend system prompt if provided (replace existing, no accumulation)
        if system_prompt:
            has_system = False
            for i, msg in enumerate(messages):
                if hasattr(msg, 'role') and msg.role == MessageRole.SYSTEM:
                    messages[i] = Message.system_prompt(system_prompt)
                    has_system = True
                    break
                elif isinstance(msg, dict) and msg.get('role') == 'system':
                    messages[i] = Message.system_prompt(system_prompt)
                    has_system = True
                    break
            if not has_system:
                messages.insert(0, Message.system_prompt(system_prompt))

        # 5b. SubAgent delegation for CREATE_PROJECT intent
        # NOTE: SubAgentManager is deprecated; delegation now handled by Orchestrator.
        # When Orchestrator is integrated, insert delegation logic here and
        # return early (similar to the old SubAgentManager pattern).

        # 5c. Session restore (before tool loop)
        session_id = None
        if self._session_persistence:
            try:
                session_id = self._session_persistence.get_current_session_id()
                if not session_id:
                    session_id = self._session_persistence.create(
                        title=user_input[:50],
                        intent=intent.value if intent else "unknown",
                    )
                # Restore from existing session if messages are empty
                # 修复 H7: 放宽条件 — 在 system prompt 添加之前 messages 较短时也应恢复
                if len(messages) <= 2:
                    existing = self._session_persistence.restore(session_id)
                    if existing and len(existing) > 1:
                        # Keep the newly appended user message, prepend restored
                        messages = existing + messages
            except Exception as e:
                logger.debug(f"Session restore error: {e}")

        # 6. Execute tool loop
        try:
            messages = await self._tool_loop(messages, allowed_tools, system_prompt)
        except Exception as e:
            logger.error(f"Agent loop error: {e}", exc_info=True)
            messages.append(Message.system_error(f"Agent loop error: {e}"))

        # 7. Emit agent_done event
        elapsed = time.time() - start_time
        if self._event_bus:
            await self._event_bus.emit(
                "agent_done",
                intent=intent.value,
                total_steps=self._total_steps,
                elapsed_seconds=round(elapsed, 2),
                final_message_count=len(messages),
            )

        logger.info(
            f"Agent loop completed: intent={intent.value}, "
            f"steps={self._total_steps}, "
            f"elapsed={elapsed:.1f}s, "
            f"messages={len(messages)}"
        )

        # 8. Persist session (full lifecycle — all messages + step count)
        if self._session_persistence and session_id:
            try:
                for msg in messages:
                    self._session_persistence.save_message(session_id, msg)
                self._session_persistence.update_step_count(
                    session_id, self._total_steps
                )
            except Exception as e:
                logger.debug(f"Session persistence error: {e}")

        return messages

    # ==================================================================
    # Core tool loop
    # ==================================================================

    async def _tool_loop(
        self,
        messages: list[Message],
        allowed_tools: list[str],
        system_prompt: str,
    ) -> list[Message]:
        """The core tool-calling loop.

        Each iteration:
          1. Check step budget
          2. Check consecutive tool failures
          3. Check max_tool_steps limit
          4. Emit agent_step event
          5. Context compaction if needed
          6. Build API messages from Message list
          7. Determine streaming vs batch mode
          8. Call model
          9. Handle output truncation recovery (if applicable)
          10. Process response:
             - If text only: append and done
             - If tool_calls: execute them, append results, loop
          11. Track circuit breaker progress and notify message bus
        """
        max_steps = self._config.max_tool_steps
        consecutive_failures = 0
        truncation_attempt = 0

        while True:
            # 1. Check step budget
            if self._step_budget and not self._step_budget.consume():
                messages.append(Message.system_warning("Step budget exhausted."))
                logger.warning("AgentLoop: step budget exhausted")
                break

            # Check consecutive tool failures
            if consecutive_failures >= self._config.max_consecutive_tool_failures:
                messages.append(Message.system_warning(
                    f"连续 {consecutive_failures} 次工具执行失败，终止循环"
                ))
                logger.warning(f"AgentLoop: {consecutive_failures} consecutive tool failures")
                break

            # Also check against config max_tool_steps
            self._total_steps += 1
            if self._total_steps >= max_steps:
                messages.append(
                    Message.system_warning(
                        f"Maximum tool steps reached ({max_steps})."
                    )
                )
                logger.warning(
                    f"AgentLoop: max_tool_steps ({max_steps}) reached"
                )
                break

            # 2. Emit agent_step event
            if self._event_bus:
                await self._event_bus.emit(
                    "agent_step",
                    step=self._total_steps,
                    max_steps=max_steps,
                )

            # 3. Context compaction
            if self._context_window and self._auto_compactor:
                try:
                    compacted = await self._auto_compactor.maybe_compact(
                        messages, system_prompt
                    )
                    if compacted is not messages:
                        messages = compacted
                        self._recovery_stats["context_compactions"] += 1
                        if self._event_bus:
                            await self._event_bus.emit(
                                "context_compacted",
                                message_count=len(messages),
                            )
                except Exception as e:
                    logger.warning(f"Context compaction failed: {e}")

            # 4. Build API messages and tool definitions
            api_messages = self._build_api_messages(messages)
            tools = self._build_tools_definition(allowed_tools)

            # 5. Determine streaming vs batch mode
            use_streaming = self._should_use_streaming(allowed_tools)

            # 6. Call model
            if use_streaming:
                content, tool_calls, usage, finish_reason = (
                    await self._call_model_streaming(api_messages, tools)
                )
            else:
                response = await self._call_model_batch(api_messages, tools)
                content = response.get("content", "")
                tool_calls = response.get("tool_calls", [])
                usage = response.get("usage", {})
                finish_reason = response.get("finish_reason", "stop")

            self._total_model_calls += 1

            # Record circuit breaker progress
            # Only record to circuit breaker if we have actual token data
            if self._circuit_breaker and usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                if prompt_tokens > 0 or completion_tokens > 0:  # skip zero-token records
                    self._circuit_breaker.record_model_call(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        stage="agentloop",
                        latency_ms=0,
                    )
                if finish_reason != "error":
                    self._circuit_breaker.record_success()
            elif self._circuit_breaker and finish_reason != "error":
                self._circuit_breaker.record_success()

            # 7. Handle output truncation recovery
            if finish_reason == "length" and self._recovery_manager:
                # Append partial assistant message — 保留 tool_calls 避免上下文丢失
                if tool_calls:
                    messages.append(Message.assistant_tool_call(content, tool_calls))
                else:
                    messages.append(Message.assistant_text(content))

                if self._recovery_manager.should_continue_after_truncation(
                    finish_reason, truncation_attempt
                ):
                    self._recovery_manager.record_recovery(RecoveryType.LENGTH)
                    self._recovery_stats["truncation_recoveries"] += 1
                    truncation_attempt += 1

                    # Ask model to continue
                    messages.append(
                        Message.user_input(
                            "Please continue from where you left off."
                        )
                    )
                    continue
                else:
                    break

            # 8. Process response
            if not tool_calls:
                # Text-only response — append and done
                if content:
                    messages.append(Message.assistant_text(content))
                break

            # 9. Has tool calls — process them
            messages.append(
                Message.assistant_tool_call(content, tool_calls)
            )

            # Execute tool calls
            if use_streaming and self._streaming_executor:
                # Tool execution was already handled by streaming executor
                # in _call_model_streaming. Use those results directly
                # to avoid double execution.
                batch_results = self._streaming_tool_results
                self._streaming_tool_results = []
            else:
                batch_results = await self._execute_tool_calls_batch(tool_calls)

            # Append tool results as messages
            for tool_call, result in batch_results:
                call_id = tool_call.get("id", "")
                # Support both orchestrator format {"name": ...} and OpenAI format {"function": {"name": ...}}
                if "function" in tool_call:
                    tool_name = tool_call["function"].get("name", "unknown")
                else:
                    tool_name = tool_call.get("name", "unknown")

                result_content = self._format_tool_result(result)
                messages.append(
                    Message.tool_result(call_id, tool_name, result_content)
                )

                # Track progress in circuit breaker
                if self._circuit_breaker:
                    had_effect = result.success
                    self._circuit_breaker.record_agent_step(
                        tool_name, had_effect
                    )

                # NOTE: message bus notification removed (coordination deprecated)
                # Use Orchestrator events for inter-agent communication.

            # Per-iteration failure tracking (not per-tool)
            iteration_failed = any(not r.success for _, r in batch_results)
            if iteration_failed:
                consecutive_failures += 1
            else:
                consecutive_failures = 0

            # Reset truncation attempt for next iteration
            truncation_attempt = 0

        return messages

    # ==================================================================
    # Model calling methods
    # ==================================================================

    def _should_use_streaming(self, allowed_tools: list[str]) -> bool:
        """Determine if streaming should be used based on mode and model
        capabilities.

        Returns:
            True if streaming should be used, False for batch mode.

        Logic:
            - mode == "batch" → always False
            - mode == "streaming" → always True
            - mode == "auto" → check model capabilities via
              streaming_executor.can_stream_with_tools()
        """
        if self._streaming_mode == "batch":
            return False
        elif self._streaming_mode == "streaming":
            if self._streaming_executor is None:
                logger.warning(
                    "Streaming mode requested but no StreamingToolExecutor "
                    "provided, falling back to batch"
                )
                return False
            return True
        else:  # "auto"
            if self._streaming_executor:
                return self._streaming_executor.can_stream_with_tools(
                    self._model
                )
            return False

    async def _call_model_batch(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict:
        """Call model in batch (non-streaming) mode.

        Uses ModelProvider.chat() or chat_with_fallback().
        On failure with fallback configured, directly calls
        fallback_provider.chat() as a secondary fallback.
        Returns the model response dict with content, tool_calls,
        usage, finish_reason.
        """
        # Build the messages for the chat API
        # NOTE: chat_messages 在 try 块之前定义，确保 fallback 路径中可访问
        chat_messages = list(messages)

        used_chat_with_fallback = self._model.has_fallback
        try:

            # Use chat_with_fallback when available (handles provider
            # failover internally), otherwise fall back to plain chat().
            if self._model.has_fallback:
                response = await self._model.chat_with_fallback(chat_messages, tools=tools)
            else:
                response = await self._model.chat(chat_messages, tools=tools)

            # Normalize response format
            # chat() returns {"content": str}, we need to add
            # tool_calls, usage, finish_reason
            if not isinstance(response, dict):
                response = {"content": str(response)}

            response.setdefault("tool_calls", [])
            response.setdefault("usage", {})
            response.setdefault("finish_reason", "stop")

            return response

        except Exception as e:
            last_error = str(e)  # Save before Python 3 deletes 'e' (PEP 3110)
            logger.error(f"Batch model call failed: {e}")

            # Record failure in circuit breaker
            if self._circuit_breaker:
                self._circuit_breaker.record_failure(last_error)

            # Record recovery if applicable
            if self._recovery_manager:
                if self._recovery_manager.is_context_overflow(e):
                    self._recovery_manager.record_recovery(
                        RecoveryType.CONTEXT_OVERFLOW
                    )
                elif self._recovery_manager.is_retryable(e):
                    self._recovery_manager.record_recovery(RecoveryType.FALLBACK)

        # Only attempt manual fallback if we didn't already use chat_with_fallback
        # (chat_with_fallback already tries the fallback provider internally).
        # For the plain chat() path, try fallback_provider if available.
        if not used_chat_with_fallback and self._model.fallback_provider is not None:
            try:
                fallback_response = await self._model.fallback_provider.chat(
                    chat_messages, tools=tools
                )
                if self._recovery_manager:
                    self._recovery_manager.record_recovery(
                        RecoveryType.FALLBACK
                    )
                if not isinstance(fallback_response, dict):
                    fallback_response = {"content": str(fallback_response)}
                fallback_response.setdefault("tool_calls", [])
                fallback_response.setdefault("usage", {})
                fallback_response.setdefault("finish_reason", "stop")
                return fallback_response
            except Exception as fallback_err:
                logger.error(f"Fallback model also failed: {fallback_err}")

        # Return an error response — usage is None (not {}) to signal no
        # token data was returned, so the circuit-breaker recording path
        # skips zero-token entries cleanly.
        return {
            "content": f"Model call failed: {last_error}",
            "tool_calls": [],
            "usage": None,
            "finish_reason": "error",
        }

    async def _call_model_streaming(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> tuple[str, list[dict], dict, str]:
        """Call model in streaming mode.

        Uses StreamingToolExecutor to process stream + execute tools.
        Returns (content, tool_calls, usage, finish_reason).
        Handles streaming retry with fallback to batch.
        """
        for attempt in range(self._max_streaming_retries + 1):
            try:
                # Build the stream via model's stream_tap
                from teragent.core.tap import CompiledPrompt

                # Build a compiled prompt from messages
                compiled = CompiledPrompt(
                    messages=messages,
                    tools=tools,
                )

                # Create the stream
                stream = self._model.adapter.stream(
                    compiled, self._model.model
                )

                # Execute streaming with tool executor
                results, streaming_result, stats = (
                    await self._streaming_executor.execute_streaming(
                        stream,
                    )
                )

                content = streaming_result.content
                tool_calls = streaming_result.tool_calls
                usage = streaming_result.usage
                finish_reason = streaming_result.finish_reason or "stop"

                # Store streaming tool results for _tool_loop() to consume
                # This avoids double execution — tools were already executed
                # by the streaming executor during stream processing.
                self._streaming_tool_results = results

                return content, tool_calls, usage, finish_reason

            except Exception as e:
                logger.warning(
                    f"Streaming attempt {attempt + 1} failed: {e}"
                )
                self._recovery_stats["streaming_retries"] += 1

                if self._recovery_manager and self._recovery_manager.should_retry_streaming(
                    attempt
                ):
                    if self._recovery_manager:
                        self._recovery_manager.record_recovery(
                            RecoveryType.STREAMING_RETRY
                        )
                    continue
                break

        # Fallback to batch mode
        logger.info("Streaming failed, falling back to batch mode")
        self._recovery_stats["batch_fallbacks"] += 1
        if self._recovery_manager:
            self._recovery_manager.record_recovery(RecoveryType.FALLBACK)

        response = await self._call_model_batch(messages, tools)
        tool_calls = response.get("tool_calls", [])

        # When falling back to batch, tool execution was NOT performed
        # by the streaming executor. Execute tools via batch path so
        # _streaming_tool_results is populated for _tool_loop().
        if tool_calls:
            self._streaming_tool_results = await self._execute_tool_calls_batch(
                tool_calls
            )
        else:
            self._streaming_tool_results = []

        return (
            response.get("content", ""),
            tool_calls,
            response.get("usage", {}),
            response.get("finish_reason", "stop"),
        )

    # ==================================================================
    # Tool execution
    # ==================================================================

    async def _execute_tool_calls_batch(
        self,
        tool_calls: list[dict],
    ) -> list[tuple[dict, ToolResult]]:
        """Execute tool calls in batch mode via ToolOrchestrator."""
        if not tool_calls:
            return []

        # Convert OpenAI-style tool_calls to orchestrator format
        orchestrator_calls = []
        for tc in tool_calls:
            func_info = tc.get("function", {})
            tool_name = func_info.get("name", "")
            arguments = func_info.get("arguments", {})

            # Parse arguments if it's a string
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            orchestrator_calls.append({
                "name": tool_name,
                "arguments": arguments,
                "id": tc.get("id", ""),
            })

        try:
            results = await self._tool_orchestrator.execute_batch(
                orchestrator_calls
            )
            return results
        except Exception as e:
            logger.error(f"Tool execution batch failed: {e}", exc_info=True)
            # Return error results for all tool calls
            return [
                (
                    tc,
                    ToolResult(
                        success=False,
                        data={},
                        error=f"Tool execution failed: {e}",
                    ),
                )
                for tc in tool_calls
            ]

    # ==================================================================
    # Tool filtering
    # ==================================================================

    def _filter_tools_by_intent(
        self,
        intent: IntentType,
    ) -> list[str]:
        """Return allowed tool names for the given intent from
        config.intent_tools."""
        intent_tools = self._config.intent_tools
        all_tool_names = self._tool_registry.list_tool_names()

        if not intent_tools:
            # No intent tools configured → allow all tools
            return all_tool_names

        # Normalize: try enum key first, then string key
        allowed = None
        if intent in intent_tools:
            allowed = list(intent_tools[intent])
        elif intent.value in intent_tools:
            allowed = list(intent_tools[intent.value])
        else:
            # Try converting all keys to their string values for comparison
            for key, val in intent_tools.items():
                key_str = key.value if isinstance(key, IntentType) else str(key)
                if key_str == intent.value:
                    allowed = list(val)
                    break

        if allowed is None:
            # intent not in config → allow all tools as safe default
            allowed = list(all_tool_names)

        return allowed

    # ==================================================================
    # Permission management
    # ==================================================================

    def set_permission_level(self, level: int) -> None:
        """Update permission level on both orchestrator and
        streaming_executor."""
        self._permission_level = level
        if self._tool_orchestrator:
            self._tool_orchestrator.set_permission_level(level)
        if self._streaming_executor:
            self._streaming_executor.set_permission_level(level)
        if self._permission_manager:
            if level >= 0:
                self._permission_manager.set_level(level)
            # else: keep current level unchanged

    # ==================================================================
    # Status reporting
    # ==================================================================

    def get_status_report(self) -> dict:
        """Return a comprehensive status report."""
        report: dict[str, Any] = {
            "streaming_mode": self._streaming_mode,
            "max_streaming_retries": self._max_streaming_retries,
            "total_steps": self._total_steps,
            "total_model_calls": self._total_model_calls,
            "permission_level": self._permission_level,
            "recovery_stats": dict(self._recovery_stats),
            "config": {
                "max_tool_steps": self._config.max_tool_steps,
                "max_streaming_retries": self._config.max_streaming_retries,
                "tool_execution_timeout": self._config.tool_execution_timeout,
            },
        }

        # Add component status
        if self._step_budget:
            report["step_budget"] = {
                "current": self._step_budget.current_steps,
                "max": self._step_budget.max_steps,
                "remaining": self._step_budget.remaining,
                "exhausted": self._step_budget.exhausted,
            }

        if self._context_window:
            report["context_window"] = {
                "available_budget": self._context_window.available_budget,
                "model_token_limit": self._context_window.model_token_limit,
                "last_estimated_tokens": self._context_window.last_estimated_tokens,
            }

        if self._auto_compactor:
            report["auto_compactor"] = self._auto_compactor.get_stats()

        if self._circuit_breaker:
            report["circuit_breaker"] = self._circuit_breaker.get_status()

        if self._recovery_manager:
            report["recovery_manager"] = self._recovery_manager.get_stats()

        if self._permission_manager:
            report["permission_manager"] = (
                self._permission_manager.get_status_report()
            )

        if self._intent_classifier:
            report["intent_classifier"] = self._intent_classifier.get_stats()

        if self._tool_registry:
            report["tool_registry"] = self._tool_registry.get_summary()

        return report

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _build_api_messages(self, messages: list[Message]) -> list[dict]:
        """Convert Message list to OpenAI API format."""
        api_messages: list[dict] = []
        for msg in messages:
            # Skip tombstone messages
            if msg.message_type == MessageType.TOMBSTONE:
                continue

            api_msg = msg.to_api_format()

            # For tool result messages, ensure tool_call_id is present
            if msg.role == MessageRole.TOOL and msg.tool_call_id:
                api_msg["tool_call_id"] = msg.tool_call_id

            # For assistant messages with tool_calls, ensure format
            if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                api_msg["tool_calls"] = msg.tool_calls

            api_messages.append(api_msg)

        return api_messages

    def _build_tools_definition(self, allowed_tools: list[str]) -> list[dict]:
        """Build OpenAI-format tool definitions for allowed tools."""
        if not allowed_tools:
            return []

        tools: list[dict] = []
        for name in allowed_tools:
            tool = self._tool_registry.get(name)
            if tool:
                tools.append(tool.to_function_definition())

        return tools

    @staticmethod
    def _format_tool_result(result: ToolResult) -> str:
        """Format a ToolResult into a string for the model.

        Includes success status, data, and error information.
        """
        parts: list[str] = []

        if result.success:
            parts.append("Success.")
            if result.data is not None:
                if isinstance(result.data, dict):
                    try:
                        parts.append(json.dumps(result.data, ensure_ascii=False))
                    except (TypeError, ValueError):
                        parts.append(str(result.data))
                else:
                    parts.append(str(result.data))
        else:
            parts.append(f"Error: {result.error}")

        if result.metadata:
            try:
                meta_str = json.dumps(result.metadata, ensure_ascii=False)
            except (TypeError, ValueError):
                meta_str = str(result.metadata)
            parts.append(f"Metadata: {meta_str}")

        return "\n".join(parts) if parts else "Tool completed with no output."

    # ==================================================================
    # Long-horizon task support (GLM-5)
    # ==================================================================

    async def run_long_task(
        self,
        goal: str,
        config: LongHorizonConfig | None = None,
    ) -> LongHorizonResult:
        """执行长程任务（GLM-5 的8小时持续工作能力）

        创建 LongHorizonTaskManager 并委托给它执行。
        如果 AgentLoop 已有 long_horizon_manager，则复用其配置。

        Args:
            goal: 大目标描述
            config: 长程任务配置，默认使用 LongHorizonConfig()

        Returns:
            LongHorizonResult 长程任务最终结果
        """
        # 如果已有 manager，复用其配置
        if self._long_horizon_manager is not None:
            manager = self._long_horizon_manager
            # 更新目标
            manager.goal = goal
            if config is not None:
                manager.config = config
        else:
            manager = LongHorizonTaskManager(
                goal=goal,
                model_provider=self._model,
                config=config,
            )
            self._long_horizon_manager = manager

        return await manager.execute_long_task()

    # ==================================================================
    # Model routing (P3-1) + Cost tracking (P3-3)
    # ==================================================================

    def route_request(self, request: TAPRequest) -> RoutingDecision:
        """Route a TAP request to the optimal model using ModelRouter

        If ModelRouter is configured, uses intelligent routing based on
        intent, multimodal, context length, long-horizon, cost, and
        degradation. Otherwise, returns the default model.

        Args:
            request: The TAP request to route

        Returns:
            RoutingDecision with selected driver and trace
        """
        if self._model_router is None:
            # No router configured → use default model
            return RoutingDecision(
                selected_driver="default",
                selected_compiler=self._model.compiler.__class__.__name__,
                reason=RoutingReason.EXPLICIT,
                intent=request.meta.get("intent", "chat"),
            )

        return self._model_router.route(request)

    def route_request_for_stage(self, stage: str, request: TAPRequest) -> RoutingDecision:
        """Route a TAP request for a specific pipeline stage

        Uses the active PipelineProfile to determine the driver for
        the given stage, then applies override checks.

        Args:
            stage: Pipeline stage ("design", "plan", "execute", "review")
            request: The TAP request to route

        Returns:
            RoutingDecision with pipeline-based routing
        """
        if self._model_router is None:
            return RoutingDecision(
                selected_driver="default",
                selected_compiler=self._model.compiler.__class__.__name__,
                intent=stage,
            )

        return self._model_router.route_for_stage(stage, request)

    async def execute_routed_tap(self, request: TAPRequest) -> Any:
        """Execute a TAP request with intelligent model routing

        Routes the request to the optimal model using ModelRouter,
        then executes it with the selected provider.

        Args:
            request: The TAP request to route and execute

        Returns:
            TAPResponse from the selected provider

        Raises:
            RuntimeError: If no suitable provider is found
        """

        decision = self.route_request(request)

        # Get provider for selected driver
        provider = None
        if self._model_router is not None:
            provider = self._model_router.get_provider(decision.selected_driver)

        if provider is None:
            # Fallback to default model
            provider = self._model

        # Execute with selected provider
        try:
            response = await provider.execute_tap_with_retry(request)

            # Track cost if cost_tracker is configured
            if self._cost_tracker is not None:
                self._cost_tracker.record_from_tap_response(
                    driver_name=decision.selected_driver,
                    compiler=decision.selected_compiler,
                    model=provider.model,
                    intent=decision.intent,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    cache_hit_tokens=response.cache_hit_tokens,
                    latency_ms=0.0,  # Will be filled by retry method
                    success=True,
                    pricing=self._model_router.routing_table.get_pricing(decision.selected_driver) if self._model_router else {},
                )

            return response

        except Exception:
            # Record failed cost
            if self._cost_tracker is not None:
                self._cost_tracker.record_from_tap_response(
                    driver_name=decision.selected_driver,
                    compiler=decision.selected_compiler,
                    model=provider.model,
                    intent=decision.intent,
                    prompt_tokens=0,
                    completion_tokens=0,
                    success=False,
                )
            raise

    def switch_pipeline_profile(self, profile_name: str) -> bool:
        """Switch the active pipeline profile for model routing

        Args:
            profile_name: Profile name ("default", "budget", "multimodal", or custom)

        Returns:
            True if the profile was found and activated, False otherwise
        """
        if self._model_router is None:
            logger.warning("Cannot switch pipeline profile: ModelRouter not configured")
            return False
        return self._model_router.pipeline_manager.set_active_profile(profile_name)

    @property
    def active_pipeline_profile(self) -> str:
        """Name of the currently active pipeline profile"""
        if self._model_router is None:
            return "(no router)"
        return self._model_router.pipeline_manager.active_profile_name

    @property
    def model_router(self) -> ModelRouter | None:
        """Access the ModelRouter (if configured)"""
        return self._model_router

    @property
    def cost_tracker(self) -> CrossModelCostTracker | None:
        """Access the CrossModelCostTracker (if configured)"""
        return self._cost_tracker

    def get_cost_report(self, group_by: str = "model") -> dict:
        """Generate a cost report from the CrossModelCostTracker

        Args:
            group_by: Grouping dimension — "model", "intent", "date", or "driver"

        Returns:
            Cost report dict, or empty dict if cost_tracker not configured
        """
        if self._cost_tracker is None:
            return {}
        return self._cost_tracker.generate_report(group_by=group_by)

    def reset(self) -> None:
        """Reset the agent loop state for a new conversation."""
        self._total_steps = 0
        self._total_model_calls = 0
        self._streaming_tool_results = []
        # 修复 H8: 重置流式相关配置
        self._streaming_mode = "auto"
        self._recovery_stats = {
            "streaming_mode": self._streaming_mode,
            "streaming_retries": 0,
            "batch_fallbacks": 0,
            "truncation_recoveries": 0,
            "context_compactions": 0,
        }
        self.set_permission_level(0)

        if self._step_budget:
            self._step_budget.reset()
        if self._auto_compactor:
            self._auto_compactor.reset()

    def __repr__(self) -> str:
        return (
            f"AgentLoop("
            f"model={self._model.model!r}, "
            f"streaming_mode={self._streaming_mode!r}, "
            f"steps={self._total_steps}, "
            f"tools={len(self._tool_registry)})"
        )
