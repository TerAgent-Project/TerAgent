# teragent/orchestration/patterns/sequential.py
"""顺序编排模式

Agent 按列表顺序执行，前一个的输出通过 SharedState 传递给下一个。

执行流程:
  1. Agent[0] 接收 task + SharedState 注入 → 输出写入 shared_state[agent.output_key]
  2. Agent[1] 接收前驱上下文 + SharedState 注入 → 输出写入 shared_state[agent.output_key]
  3. ... 直到所有 Agent 执行完毕

支持工具调用:
  - 如果 Agent 配置了 tools，使用 provider.chat() + 工具调用循环
  - 如果 Agent 无 tools，使用 provider.execute_tap() (TAP 编译链路)

参考: LangGraph 的 StateGraph(sequential), CrewAI 的 Process.sequential
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
    "SequentialPattern",
]


class SequentialPattern(OrchestrationPattern):
    """顺序编排模式

    Agent 按列表顺序执行，前一个的输出通过 SharedState 传递给下一个。

    执行流程:
    1. Agent[0] 接收 task + SharedState 注入 → 输出写入 shared_state[agent.output_key]
    2. Agent[1] 接收前驱上下文 + SharedState 注入 → 输出写入 shared_state[agent.output_key]
    3. ... 直到所有 Agent 执行完毕

    工具调用支持:
    - 如果 Agent 配置了 tools，使用 chat + 工具调用循环
    - 如果 Agent 无 tools，使用 execute_tap (TAP 编译链路)

    参考: LangGraph 的 StateGraph(sequential), CrewAI 的 Process.sequential
    """

    async def run(
        self,
        task: str | TAPRequest,
        agents: list[Agent],
        shared_state: SharedState,
        context: RunContext,
        **kwargs,
    ) -> OrchestrationResult:
        """执行顺序编排

        按列表顺序依次执行每个 Agent，前驱输出通过 SharedState 传递。
        如果 Agent 配置了 tools，使用 chat + 工具调用循环；
        否则使用 execute_tap (TAP 编译链路)。

        Args:
            task: 任务描述或 TAPRequest
            agents: 按顺序执行的 Agent 列表
            shared_state: 跨 Agent 共享状态
            context: 运行时上下文

        Returns:
            OrchestrationResult 包含最终输出和各 Agent 结果
        """
        if not agents:
            return OrchestrationResult(final_output="", total_turns=0)

        messages: list[str] = []
        total_prompt = 0
        total_completion = 0

        for i, agent in enumerate(agents):
            # 1. 检查取消
            if context.cancellation_token:
                context.cancellation_token.throw_if_cancelled()

            # 2. 触发 on_start 钩子
            if agent.hooks:
                await agent.hooks.on_start(context, agent)

            # 2.5 执行输入守卫检查
            if agent.input_guardrails:
                input_text = task.instruction if isinstance(task, TAPRequest) else task
                try:
                    await run_input_guardrails(
                        agent.input_guardrails,
                        agent,
                        input_text,
                        context,
                    )
                except GuardrailTripwireTriggered as e:
                    logger.warning(
                        f"Sequential input guardrail triggered for agent '{agent.name}': "
                        f"{e.output_info}"
                    )
                    return OrchestrationResult(
                        final_output="",
                        last_agent=agent.name,
                        agent_outputs={
                            a.output_key: shared_state.get(a.output_key)
                            for a in agents if a.output_key
                        },
                        total_turns=i + 1,
                        total_prompt_tokens=total_prompt,
                        total_completion_tokens=total_completion,
                        metadata={
                            "guardrail_triggered": True,
                            "guardrail_name": e.guardrail_name,
                            "guardrail_info": e.output_info,
                            "guardrail_mode": "input",
                        },
                    )

            # 3. 解析 provider
            provider = agent.resolve_provider()

            # 4. 选择执行路径：有工具用 chat + 工具循环，无工具用 execute_tap
            if agent.tools:
                output, prompt_tok, completion_tok = await self._run_agent_with_tools(
                    task, agent, provider, shared_state, context
                )
            else:
                request = self._build_request(task, agent, shared_state, messages, context)
                # 触发 on_model_start 钩子
                system_prompt = agent.get_system_prompt(context)
                if agent.hooks:
                    await agent.hooks.on_model_start(context, agent, system_prompt)
                response = await provider.execute_tap(request)
                # 触发 on_model_end 钩子
                if agent.hooks:
                    await agent.hooks.on_model_end(context, agent, response)
                output = response.raw_text or ""
                prompt_tok = response.prompt_tokens
                completion_tok = response.completion_tokens

            # 5. 写入 SharedState（传入 agent_name）
            if agent.output_key:
                shared_state.set(agent.output_key, output, scope="session", agent_name=agent.name)

            # 5.5 执行输出守卫检查
            if agent.output_guardrails:
                try:
                    await run_output_guardrails(
                        agent.output_guardrails,
                        agent,
                        output,
                        context,
                    )
                except GuardrailTripwireTriggered as e:
                    logger.warning(
                        f"Sequential output guardrail triggered for agent '{agent.name}': "
                        f"{e.output_info}"
                    )
                    return OrchestrationResult(
                        final_output="",
                        last_agent=agent.name,
                        agent_outputs={
                            a.output_key: shared_state.get(a.output_key)
                            for a in agents if a.output_key
                        },
                        total_turns=i + 1,
                        total_prompt_tokens=total_prompt + prompt_tok,
                        total_completion_tokens=total_completion + completion_tok,
                        metadata={
                            "guardrail_triggered": True,
                            "guardrail_name": e.guardrail_name,
                            "guardrail_info": e.output_info,
                            "guardrail_mode": "output",
                        },
                    )

            # 6. 累积对话
            messages.append(output)

            # 7. 追踪 token 使用
            total_prompt += prompt_tok
            total_completion += completion_tok
            context.usage.record(agent.name, prompt_tok, completion_tok)

            # 8. 触发 on_end 钩子
            if agent.hooks:
                await agent.hooks.on_end(context, agent, output)

            # 9. 发射事件
            if context.event_bus:
                await context.event_bus.emit(
                    "orchestration_step_completed",
                    agent_name=agent.name,
                    step=i + 1,
                    total=len(agents),
                    output_preview=output[:200],
                )

            logger.info(
                f"Sequential step {i+1}/{len(agents)}: "
                f"agent={agent.name} tokens={prompt_tok + completion_tok}"
            )

        return OrchestrationResult(
            final_output=messages[-1] if messages else "",
            last_agent=agents[-1].name if agents else "",
            agent_outputs={
                a.output_key: shared_state.get(a.output_key)
                for a in agents if a.output_key
            },
            total_turns=len(agents),
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
        )

    async def _run_agent_with_tools(
        self,
        task: str | TAPRequest,
        agent: Agent,
        provider: Any,
        shared_state: SharedState,
        context: RunContext,
    ) -> tuple[str, int, int]:
        """执行带工具调用的 Agent

        使用 provider.chat() + 工具调用循环，与 SwarmPattern 类似。

        Args:
            task: 任务描述
            agent: 当前 Agent
            provider: ModelProvider 实例
            shared_state: 共享状态
            context: 运行时上下文

        Returns:
            (输出文本, prompt_tokens, completion_tokens) 元组
        """
        from teragent.tools.registry import ToolRegistry

        # 构建工具注册表和定义
        all_tools = agent.tools
        registry = ToolRegistry()
        for t in all_tools:
            registry.register(t)

        tool_defs = [t.to_function_definition() for t in all_tools]

        # 构建 chat messages
        chat_messages = self._build_chat_messages(task, agent, context)

        # 触发 on_model_start 钩子
        system_prompt = agent.get_system_prompt(context)
        if agent.hooks:
            await agent.hooks.on_model_start(context, agent, system_prompt)

        # 执行工具调用循环
        total_prompt = 0
        total_completion = 0

        for step in range(agent.max_steps):
            if context.cancellation_token:
                context.cancellation_token.throw_if_cancelled()

            chat_result = await provider.chat(chat_messages, tools=tool_defs or None)
            assistant_content = chat_result.get("content", "")
            tool_calls = chat_result.get("tool_calls", [])

            # 追踪 token
            usage = chat_result.get("usage", {})
            total_prompt += usage.get("prompt_tokens", 0)
            total_completion += usage.get("completion_tokens", 0)

            # 追加 assistant 消息
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": assistant_content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            chat_messages.append(assistant_msg)

            if not tool_calls:
                # 纯文本回复 → 结束
                # 触发 on_model_end 钩子
                if agent.hooks:
                    tap_response = TAPResponse(
                        raw_text=assistant_content,
                        usage=usage,
                        finish_reason=chat_result.get("finish_reason", "stop"),
                    )
                    await agent.hooks.on_model_end(context, agent, tap_response)
                return assistant_content, total_prompt, total_completion

            # 处理工具调用
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                tool_args_str = func.get("arguments", "{}")
                tool_call_id = tc.get("id", "")

                try:
                    tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                except json.JSONDecodeError:
                    tool_args = {}

                tool_obj = registry.get(tool_name)
                if tool_obj:
                    # 触发 on_tool_start 钩子
                    if agent.hooks:
                        await agent.hooks.on_tool_start(context, agent, tool_obj)
                    result = await tool_obj.execute(tool_args)
                    # 触发 on_tool_end 钩子
                    if agent.hooks:
                        await agent.hooks.on_tool_end(context, agent, tool_obj, result)
                    chat_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(result.to_dict(), ensure_ascii=False),
                    })
                else:
                    error_result = ToolResult(success=False, error=f"Tool '{tool_name}' not found")
                    chat_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(error_result.to_dict(), ensure_ascii=False),
                    })

        # 达到 max_steps — 触发 on_model_end 钩子
        if agent.hooks:
            tap_response = TAPResponse(
                raw_text=assistant_content,
                usage={"prompt_tokens": total_prompt, "completion_tokens": total_completion},
                finish_reason="max_steps",
            )
            await agent.hooks.on_model_end(context, agent, tap_response)

        return assistant_content, total_prompt, total_completion

    def _build_chat_messages(
        self,
        task: str | TAPRequest,
        agent: Agent,
        context: RunContext | None = None,
    ) -> list[dict]:
        """构建 chat 接口的 messages 列表

        Args:
            task: 原始任务描述或 TAPRequest
            agent: 当前 Agent

        Returns:
            OpenAI 格式的消息列表
        """
        messages = []

        # 系统提示
        system_prompt = agent.get_system_prompt(context)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 当前任务
        instruction = task.instruction if isinstance(task, TAPRequest) else task
        if instruction:
            messages.append({"role": "user", "content": instruction})

        return messages

    def get_next_agent(self, current: Agent | None, result: Any) -> Agent | None:
        """顺序获取下一个 Agent

        Sequential pattern 不使用动态路由，此方法始终返回 None。
        Agent 的顺序由 agents 列表决定。
        """
        return None

    def get_execution_plan(self) -> list[dict]:
        """获取执行计划"""
        return [{"type": "sequential", "description": "Execute agents in order"}]

    def _build_request(
        self,
        task: str | TAPRequest,
        agent: Agent,
        shared_state: SharedState,
        previous_outputs: list[str],
        context: RunContext | None = None,
    ) -> TAPRequest:
        """构建 TAPRequest（无工具调用时使用）

        将原始任务、SharedState 上下文、前驱输出和 Agent 系统提示
        注入到 TAPRequest 中。

        注意：始终创建新的 TAPRequest 实例，避免原地修改导致状态泄漏。

        Args:
            task: 原始任务描述或 TAPRequest
            agent: 当前要执行的 Agent
            shared_state: 共享状态
            previous_outputs: 前驱 Agent 的输出列表

        Returns:
            构建好的 TAPRequest（新实例）
        """
        if isinstance(task, TAPRequest):
            # 深拷贝 context，避免原地修改原始 TAPRequest
            request = TAPRequest(
                meta=dict(task.meta),
                context=dict(task.context),
                instruction=task.instruction,
                constraints=list(task.constraints),
                output_format_hint=task.output_format_hint,
            )
        else:
            request = TAPRequest(instruction=task)

        # 注入 SharedState 上下文（注意：只注入 summary 而非全量数据，避免 token 超限）
        state_dict = shared_state.to_dict()
        if state_dict:
            # 限制注入量：只注入前 10 个键值对
            summary = {k: v for i, (k, v) in enumerate(state_dict.items()) if i < 10}
            request.context["shared_state"] = summary
            if len(state_dict) > 10:
                request.context["shared_state_truncated"] = True

        # 注入前驱输出
        if previous_outputs:
            request.context["previous_outputs"] = previous_outputs

        # 注意：不再注入 agent_system_prompt 到 context 中
        # 系统提示已通过 Compiler 在 compile() 时处理，
        # 如果同时注入到 context 会导致重复

        return request
