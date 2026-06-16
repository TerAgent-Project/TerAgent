# teragent/orchestration/patterns/swarm.py
"""去中心化 Swarm 编排模式

Agent 通过 Handoff 工具自主决定转交控制权。
无中心协调器，Agent 自主决定下一步。

参考: OpenAI Agents SDK 的 Swarm 模式, Google ADK 的 Agent.transfer()

执行流程:
  1. 选择初始 Agent
  2. Agent 执行，工具调用中可能包含 HandoffTool
  3. 如果检测到 HandoffTool 结果，切换到目标 Agent
  4. 重复直到无 Handoff 或 max_turns 超限
"""

from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

from teragent.orchestration.patterns.base import OrchestrationPattern, OrchestrationResult
from teragent.core.tap import TAPRequest, TAPResponse
from teragent.tools.base import ToolResult
from teragent.orchestration.guardrail import (
    GuardrailTripwireTriggered,
    run_input_guardrails,
    run_output_guardrails,
)

if TYPE_CHECKING:
    from teragent.orchestration.agent import Agent
    from teragent.orchestration.shared_state import SharedState
    from teragent.orchestration.run_context import RunContext

logger = logging.getLogger(__name__)

__all__ = [
    "SwarmPattern",
]


class SwarmPattern(OrchestrationPattern):
    """去中心化 Swarm 编排模式

    Agent 通过 Handoff 工具自主决定转交控制权。
    无中心协调器，Agent 自主决定下一步。

    参考: OpenAI Agents SDK 的 Swarm 模式, Google ADK 的 Agent.transfer()

    执行流程:
    1. 选择初始 Agent
    2. Agent 执行，工具调用中可能包含 HandoffTool
    3. 如果检测到 HandoffTool 结果，切换到目标 Agent
    4. 重复直到无 Handoff 或 max_turns 超限
    """

    async def run(
        self,
        task: str | TAPRequest,
        agents: list[Agent],
        shared_state: SharedState,
        context: RunContext,
        **kwargs,
    ) -> OrchestrationResult:
        """执行 Swarm 编排

        Agent 通过 Handoff 工具自主决定转交控制权。

        Args:
            task: 任务描述或 TAPRequest
            agents: 参与编排的 Agent 列表
            shared_state: 跨 Agent 共享状态
            context: 运行时上下文

        Returns:
            OrchestrationResult 包含最终输出和各 Agent 结果
        """
        if not agents:
            return OrchestrationResult(final_output="", total_turns=0)

        agent_map = {a.name: a for a in agents}
        # 检测 Agent 名称重复
        if len(agent_map) < len(agents):
            seen = set()
            for a in agents:
                if a.name in seen:
                    logger.warning(f"Duplicate agent name '{a.name}' in Swarm agents list. Later agent will override earlier one.")
                seen.add(a.name)

        current_agent = agents[0]
        messages: list[dict] = []
        turn = 0
        last_response = None
        total_prompt = 0
        total_completion = 0
        handoff_count = 0
        max_handoffs = context.max_turns  # 默认使用 max_turns 作为上限
        if kwargs.get("config") and hasattr(kwargs["config"], "max_handoffs"):
            max_handoffs = kwargs["config"].max_handoffs

        # 保存原始任务指令，确保 handoff 后新 Agent 仍能看到原始任务
        original_instruction = (
            task.instruction if isinstance(task, TAPRequest) else task
        )

        while turn < context.max_turns:
            # 1. 检查取消
            if context.cancellation_token:
                context.cancellation_token.throw_if_cancelled()

            # 2. 更新 RunContext 中的当前 Agent
            context = context.with_agent(current_agent.name, turn=turn)

            # 3. 触发 on_start 钩子
            if current_agent.hooks:
                await current_agent.hooks.on_start(context, current_agent)

            # 3.5 执行输入守卫检查
            if current_agent.input_guardrails:
                try:
                    await run_input_guardrails(
                        current_agent.input_guardrails,
                        current_agent,
                        original_instruction,
                        context,
                    )
                except GuardrailTripwireTriggered as e:
                    logger.warning(
                        f"Swarm input guardrail triggered for agent '{current_agent.name}': "
                        f"{e.output_info}"
                    )
                    return OrchestrationResult(
                        final_output="",
                        last_agent=current_agent.name,
                        agent_outputs={},
                        total_turns=turn + 1,
                        total_prompt_tokens=total_prompt,
                        total_completion_tokens=total_completion,
                        metadata={
                            "guardrail_triggered": True,
                            "guardrail_name": e.guardrail_name,
                            "guardrail_info": e.output_info,
                            "guardrail_mode": "input",
                        },
                    )

            # 4. 获取当前 Agent 的全部工具（包括 handoff 工具）
            all_tools = current_agent.tools + current_agent.get_handoff_tools()

            # 5. 解析 provider
            provider = current_agent.resolve_provider()

            # 6. 构建 tool definitions
            tool_defs = [t.to_function_definition() for t in all_tools]

            # 7. 构建 messages
            chat_messages = self._build_chat_messages(
                task, current_agent, messages, original_instruction, context,
            )

            # 8. 触发 on_model_start 钩子
            system_prompt = current_agent.get_system_prompt(context)
            if current_agent.hooks:
                await current_agent.hooks.on_model_start(context, current_agent, system_prompt)

            # 9. 执行 Agent 循环
            response, tool_results = await self._run_agent_loop(
                current_agent, provider, chat_messages, tool_defs, shared_state, context
            )

            last_response = response

            # 10. 触发 on_model_end 钩子
            if current_agent.hooks and response:
                await current_agent.hooks.on_model_end(context, current_agent, response)

            # 11. 检查是否有 Handoff
            handoff_target = self._detect_handoff(tool_results)
            if handoff_target:
                new_agent = agent_map.get(handoff_target)
                if new_agent:
                    # 追踪 token 使用（Handoff 前，避免 continue 跳过）
                    if response:
                        total_prompt += response.prompt_tokens
                        total_completion += response.completion_tokens
                        context.usage.record(current_agent.name, response.prompt_tokens, response.completion_tokens)

                    # 触发 on_handoff 钩子
                    if new_agent.hooks:
                        await new_agent.hooks.on_handoff(context, new_agent, current_agent)

                    # 应用 HandoffInputFilter
                    handoff = self._find_handoff(current_agent, handoff_target)
                    if handoff and handoff.input_filter:
                        messages = handoff.input_filter.apply(messages)

                    # 触发 on_end 钩子（当前 Agent 结束）
                    if current_agent.hooks:
                        await current_agent.hooks.on_end(
                            context, current_agent, response.raw_text or "" if response else ""
                        )

                    logger.info(f"Swarm handoff: {current_agent.name} → {new_agent.name}")

                    # 发射 handoff 事件
                    if context.event_bus:
                        await context.event_bus.emit(
                            "orchestration_handoff",
                            from_agent=current_agent.name,
                            to_agent=new_agent.name,
                        )

                    current_agent = new_agent
                    turn += 1
                    handoff_count += 1
                    if handoff_count >= max_handoffs:
                        logger.warning(
                            f"Swarm handoff limit reached ({max_handoffs}). "
                            f"Stopping orchestration."
                        )
                        break
                    continue
                else:
                    # Handoff 目标不在 agents 列表中 — 记录警告，不静默失败
                    logger.warning(
                        f"Swarm handoff target '{handoff_target}' not found in agents list. "
                        f"Available agents: {list(agent_map.keys())}. "
                        f"Ignoring handoff and continuing with current agent."
                    )

            # 12. 追踪 token 使用
            if response:
                total_prompt += response.prompt_tokens
                total_completion += response.completion_tokens
                context.usage.record(current_agent.name, response.prompt_tokens, response.completion_tokens)

            # 13. 无 Handoff → Agent 完成
            if current_agent.output_key and response:
                shared_state.set(
                    current_agent.output_key,
                    response.raw_text or "",
                    scope="session",
                    agent_name=current_agent.name,
                )

            # 13.5 执行输出守卫检查
            if current_agent.output_guardrails and response:
                try:
                    await run_output_guardrails(
                        current_agent.output_guardrails,
                        current_agent,
                        response.raw_text or "",
                        context,
                    )
                except GuardrailTripwireTriggered as e:
                    logger.warning(
                        f"Swarm output guardrail triggered for agent '{current_agent.name}': "
                        f"{e.output_info}"
                    )
                    return OrchestrationResult(
                        final_output="",
                        last_agent=current_agent.name,
                        agent_outputs={
                            a.output_key: shared_state.get(a.output_key)
                            for a in agents if a.output_key
                        },
                        total_turns=turn + 1,
                        total_prompt_tokens=total_prompt + (response.prompt_tokens if response else 0),
                        total_completion_tokens=total_completion + (response.completion_tokens if response else 0),
                        metadata={
                            "guardrail_triggered": True,
                            "guardrail_name": e.guardrail_name,
                            "guardrail_info": e.output_info,
                            "guardrail_mode": "output",
                        },
                    )

            # 14. 触发 on_end 钩子
            if current_agent.hooks:
                await current_agent.hooks.on_end(
                    context, current_agent, response.raw_text or "" if response else ""
                )

            # 15. 发射 step 完成事件
            if context.event_bus:
                await context.event_bus.emit(
                    "orchestration_step_completed",
                    agent_name=current_agent.name,
                    step=turn + 1,
                    output_preview=(response.raw_text or "")[:200] if response else "",
                )

            break

        final_output = ""
        if last_response:
            final_output = last_response.raw_text or ""

        return OrchestrationResult(
            final_output=final_output,
            last_agent=current_agent.name,
            agent_outputs={
                a.output_key: shared_state.get(a.output_key)
                for a in agents if a.output_key
            },
            total_turns=turn + 1,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            metadata={"swarm_completed": True, "final_agent": current_agent.name},
        )

    def get_next_agent(self, current: Agent | None, result: Any) -> Agent | None:
        """Swarm 模式通过 Handoff 动态决定下一个 Agent

        此方法始终返回 None，因为 Swarm 模式的 Agent 切换
        是通过 Handoff 工具调用动态决定的。
        """
        return None  # Swarm decides dynamically via Handoff

    def get_execution_plan(self) -> list[dict]:
        """获取执行计划"""
        return [{"type": "swarm", "description": "Agent-driven handoff pattern"}]

    def _detect_handoff(self, tool_results: list[tuple[dict, ToolResult]]) -> str | None:
        """检测工具结果中是否包含 Handoff 标记

        遍历工具调用结果，检查是否有 HandoffTool 返回的
        __handoff__ 标记和 target_agent 字段。

        Args:
            tool_results: 工具调用结果列表 [(tool_call_dict, ToolResult), ...]

        Returns:
            目标 Agent 名称，或 None 表示无 Handoff
        """
        for tool_call, result in tool_results:
            if result.success and result.data.get("__handoff__"):
                return result.data.get("target_agent")
        return None

    def _find_handoff(self, agent: Agent, target_name: str) -> Any | None:
        """查找 Agent 的 Handoff 定义

        在 Agent 的 handoffs 列表中查找目标 Agent 名称匹配的 Handoff 定义。

        Args:
            agent: 当前 Agent
            target_name: 目标 Agent 名称

        Returns:
            匹配的 Handoff 对象，或 None
        """
        for h in agent.handoffs:
            if h.target_agent.name == target_name:
                return h
        return None

    def _build_request(
        self,
        task: str | TAPRequest,
        agent: Agent,
        shared_state: SharedState,
        messages: list[dict],
        context: RunContext | None = None,
    ) -> TAPRequest:
        """构建 TAPRequest（Swarm 模式备用，当前未被调用）

        注意：Swarm 模式使用 chat + 工具调用路径，此方法保留供未来扩展使用。
        如果需要 TAP 编译链路，可通过此方法构建请求。

        Args:
            task: 原始任务描述或 TAPRequest
            agent: 当前要执行的 Agent
            shared_state: 共享状态
            messages: 消息历史
            context: 运行时上下文

        Returns:
            构建好的 TAPRequest（新实例）
        """
        if isinstance(task, TAPRequest):
            request = TAPRequest(
                meta=dict(task.meta),
                context=dict(task.context),
                instruction=task.instruction,
                constraints=list(task.constraints),
                output_format_hint=task.output_format_hint,
            )
        else:
            request = TAPRequest(instruction=task)

        state_dict = shared_state.to_dict()
        if state_dict:
            request.context["shared_state"] = state_dict

        system_prompt = agent.get_system_prompt(context)
        if system_prompt:
            request.context["agent_system_prompt"] = system_prompt

        return request

    def _build_chat_messages(
        self,
        task: str | TAPRequest,
        agent: Agent,
        history: list[dict],
        original_instruction: str = "",
        context: RunContext | None = None,
    ) -> list[dict]:
        """构建 chat 接口的 messages 列表

        按顺序组装：系统提示 → 历史消息 → 当前任务。
        在 handoff 后（有历史消息时），确保原始任务指令仍然可见，
        防止 HandoffInputFilter 过滤掉原始任务后新 Agent 丢失上下文。

        Args:
            task: 原始任务描述或 TAPRequest
            agent: 当前 Agent
            history: 历史消息列表
            original_instruction: 原始任务指令（确保 handoff 后仍可访问）

        Returns:
            OpenAI 格式的消息列表
        """
        messages = []

        # 系统提示
        system_prompt = agent.get_system_prompt(context)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 历史消息
        messages.extend(history)

        # 当前任务
        instruction = task.instruction if isinstance(task, TAPRequest) else task
        if not history:
            # 首次执行，直接添加任务
            if instruction:
                messages.append({"role": "user", "content": instruction})
        else:
            # handoff 后的执行：确保原始任务指令在上下文中可见
            # 检查历史中是否已有包含原始指令的 user 消息
            has_original_task = any(
                m.get("role") == "user" and m.get("content") == original_instruction
                for m in history
            )
            if not has_original_task and original_instruction:
                messages.append({
                    "role": "user",
                    "content": f"[Original task]: {original_instruction}",
                })

        return messages

    async def _run_agent_loop(
        self,
        agent: Agent,
        provider: Any,  # ModelProvider
        chat_messages: list[dict],
        tool_defs: list[dict],
        shared_state: SharedState,
        context: RunContext,
    ) -> tuple[Any, list[tuple[dict, ToolResult]]]:
        """运行 Agent 的工具调用循环

        反复调用模型，处理工具调用，直到模型返回纯文本回复
        或达到 max_steps 限制。

        Args:
            agent: 当前 Agent
            provider: ModelProvider 实例
            chat_messages: 初始消息列表
            tool_defs: 工具定义列表
            shared_state: 共享状态
            context: 运行时上下文

        Returns:
            (最终TAPResponse, 所有工具调用结果列表)
        """
        messages = list(chat_messages)
        all_tool_results: list[tuple[dict, ToolResult]] = []
        assistant_content = ""
        total_prompt = 0
        total_completion = 0

        # 预构建工具注册表（避免在循环中重复创建）
        from teragent.tools.registry import ToolRegistry
        registry = ToolRegistry()
        for tool in agent.tools + agent.get_handoff_tools():
            registry.register(tool)

        for step in range(agent.max_steps):
            # 检查取消
            if context.cancellation_token:
                context.cancellation_token.throw_if_cancelled()

            # 调用模型
            chat_result = await provider.chat(messages, tools=tool_defs if tool_defs else None)

            assistant_content = chat_result.get("content", "")
            tool_calls = chat_result.get("tool_calls", [])

            # 累积 token 使用
            usage = chat_result.get("usage", {})
            total_prompt += usage.get("prompt_tokens", 0)
            total_completion += usage.get("completion_tokens", 0)

            # 追加 assistant 消息
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": assistant_content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if not tool_calls:
                # 纯文本回复 → 结束
                return TAPResponse(
                    raw_text=assistant_content,
                    usage={"prompt_tokens": total_prompt, "completion_tokens": total_completion},
                    finish_reason=chat_result.get("finish_reason", "stop"),
                ), all_tool_results

            # 处理工具调用
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                tool_args_str = func.get("arguments", "{}")
                tool_call_id = tc.get("id", "")

                try:
                    if isinstance(tool_args_str, str):
                        tool_args = json.loads(tool_args_str)
                    else:
                        tool_args = tool_args_str
                except json.JSONDecodeError:
                    tool_args = {}

                # 执行工具
                tool_obj = registry.get(tool_name)
                if tool_obj:
                    # 触发 on_tool_start 钩子
                    if agent.hooks:
                        await agent.hooks.on_tool_start(context, agent, tool_obj)
                    result = await tool_obj.execute(tool_args)
                    # 触发 on_tool_end 钩子
                    if agent.hooks:
                        await agent.hooks.on_tool_end(context, agent, tool_obj, result)
                    all_tool_results.append((tc, result))

                    # 追加工具结果消息
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(result.to_dict(), ensure_ascii=False),
                    })

                    # 注意：Handoff 不在此处立即返回
                    # 需要先处理完本轮所有工具调用，确保消息列表完整
                else:
                    # 工具未找到 — 不调用 on_tool_end（tool_obj 为 None 会导致钩子异常）
                    error_result = ToolResult(success=False, error=f"Tool '{tool_name}' not found")
                    all_tool_results.append((tc, error_result))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(error_result.to_dict(), ensure_ascii=False),
                    })

            # 所有工具调用处理完毕后，检查是否有 Handoff
            # 必须等所有工具结果追加到 messages 后再返回，否则消息列表不完整
            handoff_detected = False
            for tc, result in all_tool_results:
                if result.success and result.data.get("__handoff__"):
                    handoff_detected = True
                    break
            if handoff_detected:
                return TAPResponse(
                    raw_text=assistant_content,
                    usage={"prompt_tokens": total_prompt, "completion_tokens": total_completion},
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                ), all_tool_results

        # 达到 max_steps
        return TAPResponse(
            raw_text=assistant_content,
            usage={"prompt_tokens": total_prompt, "completion_tokens": total_completion},
            finish_reason="max_steps",
        ), all_tool_results
