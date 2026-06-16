# teragent/orchestration/patterns/parallel.py
"""并行扇出/扇入编排模式

多个 Agent 并行执行（fan-out），可选的聚合 Agent 读取所有并行结果后产生最终输出（fan-in）。

执行流程:
  1. 将任务分发给多个 fan-out Agent 并行执行
  2. 每个 Agent 的输出写入 SharedState（通过 output_key）
  3. 如果配置了 fan-in Agent，读取所有并行结果并产生最终输出
  4. 如果没有 fan-in Agent，返回最后一个完成的 Agent 的输出

参考: LangGraph 的 MapReduce, CrewAI 的 Task(async_execution), Airflow 的 FanOut/FanIn

配置参数（通过 kwargs["config"] 传入）:
  - fan_in_agent: 可选的聚合 Agent（名称或 Agent 实例）
  - fan_in_output_key: 聚合 Agent 的 output_key（默认 "fan_in_result"）

注意:
  - 每个 fan-out Agent 必须有唯一的 output_key，用于 SharedState 写入
  - 并行 Agent 之间的 SharedState 写入冲突会被检测并记录
  - CancellationToken 取消时，所有并行任务都会被取消
"""

from __future__ import annotations

import asyncio
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
    "ParallelPattern",
]


class ParallelPattern(OrchestrationPattern):
    """并行扇出/扇入编排模式

    多个 Agent 通过 asyncio.gather 并行执行（fan-out），
    可选的聚合 Agent 读取所有并行结果后产生最终输出（fan-in）。

    参考: LangGraph 的 MapReduce, CrewAI 的 Task(async_execution)

    执行流程:
    1. 将任务分发给多个 fan-out Agent 并行执行
    2. 每个 Agent 的输出写入 SharedState（通过 output_key）
    3. 如果配置了 fan-in Agent，读取所有并行结果并产生最终输出
    4. 如果没有 fan-in Agent，返回最后一个完成的 Agent 的输出
    """

    async def run(
        self,
        task: str | TAPRequest,
        agents: list[Agent],
        shared_state: SharedState,
        context: RunContext,
        **kwargs,
    ) -> OrchestrationResult:
        """执行并行编排

        Args:
            task: 任务描述或 TAPRequest
            agents: 参与并行执行的 Agent 列表
            shared_state: 跨 Agent 共享状态
            context: 运行时上下文
            **kwargs: 额外参数，可包含 config

        Returns:
            OrchestrationResult 包含最终输出和各 Agent 结果
        """
        if not agents:
            return OrchestrationResult(final_output="", total_turns=0)

        config = kwargs.get("config")

        # 解析 fan-out 和 fan-in Agent
        fan_out_agents, fan_in_agent = self._resolve_agents(agents, config)

        if not fan_out_agents:
            return OrchestrationResult(final_output="", total_turns=0)

        # 检查 output_key 唯一性
        self._validate_output_keys(fan_out_agents)

        # 发射并行开始事件
        if context.event_bus:
            await context.event_bus.emit(
                "parallel_fan_out_started",
                agents=[a.name for a in fan_out_agents],
            )

        # 并行执行所有 fan-out Agent
        parallel_tasks = [
            self._run_single(agent, task, shared_state, context)
            for agent in fan_out_agents
        ]

        results = await asyncio.gather(*parallel_tasks, return_exceptions=True)

        # 处理并行结果
        total_prompt = 0
        total_completion = 0
        last_agent_name = ""
        last_output = ""
        agent_outputs: dict[str, Any] = {}
        completed_count = 0
        error_count = 0

        for agent, result in zip(fan_out_agents, results):
            if isinstance(result, Exception):
                error_count += 1
                logger.error(
                    f"Parallel agent '{agent.name}' failed: {result}",
                    exc_info=result if isinstance(result, BaseException) else None,
                )
                # 即使失败也记录错误到 SharedState
                if agent.output_key:
                    shared_state.set(
                        agent.output_key,
                        f"[ERROR] {result}",
                        scope="session",
                        agent_name=agent.name,
                    )
                    agent_outputs[agent.output_key] = f"[ERROR] {result}"
                continue

            output, prompt_tok, completion_tok = result
            total_prompt += prompt_tok
            total_completion += completion_tok
            context.usage.record(agent.name, prompt_tok, completion_tok)

            if agent.output_key:
                shared_state.set(
                    agent.output_key,
                    output,
                    scope="session",
                    agent_name=agent.name,
                )
                agent_outputs[agent.output_key] = output

            last_agent_name = agent.name
            last_output = output
            completed_count += 1

            # 发射 step 完成事件
            if context.event_bus:
                await context.event_bus.emit(
                    "orchestration_step_completed",
                    agent_name=agent.name,
                    output_preview=output[:200] if output else "",
                )

        logger.info(
            f"Parallel fan-out completed: {completed_count}/{len(fan_out_agents)} succeeded, "
            f"{error_count} failed"
        )

        # 发射扇出完成事件
        if context.event_bus:
            await context.event_bus.emit(
                "parallel_fan_out_completed",
                completed=completed_count,
                failed=error_count,
            )

        # Fan-in 阶段
        final_output = last_output
        final_agent_name = last_agent_name
        fan_in_turns = 0

        if fan_in_agent:
            # 检查取消
            if context.cancellation_token:
                context.cancellation_token.throw_if_cancelled()

            fan_in_output, fan_in_prompt, fan_in_completion = await self._run_fan_in(
                fan_in_agent, task, shared_state, context, fan_out_agents,
            )
            total_prompt += fan_in_prompt
            total_completion += fan_in_completion
            context.usage.record(fan_in_agent.name, fan_in_prompt, fan_in_completion)

            final_output = fan_in_output
            final_agent_name = fan_in_agent.name
            fan_in_turns = 1

            # 写入 fan-in 结果到 SharedState
            fan_in_key = getattr(config, "fan_in_output_key", None) or fan_in_agent.output_key or "fan_in_result"
            shared_state.set(fan_in_key, fan_in_output, scope="session", agent_name=fan_in_agent.name)
            agent_outputs[fan_in_key] = fan_in_output

            # 发射 fan-in 完成事件
            if context.event_bus:
                await context.event_bus.emit(
                    "parallel_fan_in_completed",
                    agent_name=fan_in_agent.name,
                    output_preview=fan_in_output[:200] if fan_in_output else "",
                )

        return OrchestrationResult(
            final_output=final_output,
            last_agent=final_agent_name,
            agent_outputs=agent_outputs,
            total_turns=completed_count + fan_in_turns,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            metadata={
                "parallel_completed": True,
                "fan_out_count": len(fan_out_agents),
                "fan_out_succeeded": completed_count,
                "fan_out_failed": error_count,
                "has_fan_in": fan_in_agent is not None,
            },
        )

    def get_next_agent(self, current: Agent | None, result: Any) -> Agent | None:
        """并行模式不使用顺序获取下一个 Agent

        并行模式所有 Agent 同时执行，此方法始终返回 None。
        """
        return None

    def get_execution_plan(self) -> list[dict]:
        """获取执行计划"""
        return [
            {"type": "parallel", "description": "Fan-out/fan-in parallel execution"},
        ]

    # ===== 内部方法 =====

    def _resolve_agents(
        self,
        agents: list[Agent],
        config: Any | None,
    ) -> tuple[list[Agent], Agent | None]:
        """解析 fan-out 和 fan-in Agent

        根据 config 中的 fan_in_agent 配置，将 Agent 列表分为
        fan-out 列表和可选的 fan-in Agent。

        Args:
            agents: 所有参与编排的 Agent 列表
            config: 编排配置，可包含 fan_in_agent 字段

        Returns:
            (fan_out_agents, fan_in_agent) 元组
        """
        fan_in_agent: Agent | None = None
        fan_out_agents = list(agents)

        if config and hasattr(config, "fan_in_agent"):
            fan_in_ref = config.fan_in_agent
            if fan_in_ref is not None:
                # fan_in_agent 可以是 Agent 名称（str）或 Agent 实例
                if isinstance(fan_in_ref, str):
                    # 按名称查找
                    for agent in agents:
                        if agent.name == fan_in_ref:
                            fan_in_agent = agent
                            fan_out_agents = [a for a in agents if a.name != fan_in_ref]
                            break
                    if fan_in_agent is None:
                        logger.warning(
                            f"Parallel fan_in_agent '{fan_in_ref}' not found in agents list. "
                            f"Skipping fan-in phase."
                        )
                else:
                    # 直接传入 Agent 实例
                    fan_in_agent = fan_in_ref
                    fan_out_agents = [a for a in agents if a.name != fan_in_ref.name]

        return fan_out_agents, fan_in_agent

    def _validate_output_keys(self, agents: list[Agent]) -> None:
        """验证 fan-out Agent 的 output_key 唯一性

        并行 Agent 各自写入 SharedState，output_key 冲突会导致
        数据覆盖。此方法检测冲突并发出警告。

        Args:
            agents: fan-out Agent 列表
        """
        seen_keys: dict[str, str] = {}  # key → agent_name
        for agent in agents:
            if not agent.output_key:
                logger.warning(
                    f"Parallel agent '{agent.name}' has no output_key. "
                    f"Its output will not be stored in SharedState."
                )
                continue
            if agent.output_key in seen_keys:
                logger.warning(
                    f"Parallel output_key conflict: agent '{agent.name}' and "
                    f"'{seen_keys[agent.output_key]}' share output_key '{agent.output_key}'. "
                    f"Last writer wins — this may cause data loss."
                )
            seen_keys[agent.output_key] = agent.name

    async def _run_single(
        self,
        agent: Agent,
        task: str | TAPRequest,
        shared_state: SharedState,
        context: RunContext,
    ) -> tuple[str, int, int]:
        """执行单个并行 Agent

        每个 Agent 独立执行，返回 (输出文本, prompt_tokens, completion_tokens)。
        与 SequentialPattern 的 _run_agent_with_tools 逻辑类似，
        但为并行执行场景定制。

        Args:
            agent: 要执行的 Agent
            task: 任务描述
            shared_state: 共享状态
            context: 运行时上下文

        Returns:
            (输出文本, prompt_tokens, completion_tokens) 元组
        """
        # 检查取消
        if context.cancellation_token:
            context.cancellation_token.throw_if_cancelled()

        # 创建该 Agent 的上下文
        agent_context = context.with_agent(agent.name, turn=0)

        # 触发 on_start 钩子
        if agent.hooks:
            await agent.hooks.on_start(agent_context, agent)

        # 执行输入守卫检查
        if agent.input_guardrails:
            input_text = task.instruction if isinstance(task, TAPRequest) else task
            try:
                await run_input_guardrails(
                    agent.input_guardrails,
                    agent,
                    input_text,
                    agent_context,
                )
            except GuardrailTripwireTriggered as e:
                logger.warning(
                    f"Parallel input guardrail triggered for agent '{agent.name}': {e.output_info}"
                )
                # 守卫跳闸：返回空结果
                return "", 0, 0

        # 解析 provider
        provider = agent.resolve_provider()

        # 选择执行路径
        if agent.tools:
            output, prompt_tok, completion_tok = await self._run_agent_with_tools(
                task, agent, provider, shared_state, agent_context
            )
        else:
            # 无工具，使用 TAP 编译链路
            request = self._build_request(task, agent, shared_state, agent_context)
            system_prompt = agent.get_system_prompt(agent_context)

            if agent.hooks:
                await agent.hooks.on_model_start(agent_context, agent, system_prompt)

            response = await provider.execute_tap(request)

            if agent.hooks:
                await agent.hooks.on_model_end(agent_context, agent, response)

            output = response.raw_text or ""
            prompt_tok = response.prompt_tokens
            completion_tok = response.completion_tokens

        # 执行输出守卫检查
        if agent.output_guardrails:
            try:
                await run_output_guardrails(
                    agent.output_guardrails,
                    agent,
                    output,
                    agent_context,
                )
            except GuardrailTripwireTriggered as e:
                logger.warning(
                    f"Parallel output guardrail triggered for agent '{agent.name}': {e.output_info}"
                )
                # 守卫跳闸：返回空结果
                return "", 0, 0

        # 触发 on_end 钩子
        if agent.hooks:
            await agent.hooks.on_end(agent_context, agent, output)

        logger.info(
            f"Parallel agent '{agent.name}' completed: "
            f"tokens={prompt_tok + completion_tok}"
        )

        return output, prompt_tok, completion_tok

    async def _run_fan_in(
        self,
        fan_in_agent: Agent,
        task: str | TAPRequest,
        shared_state: SharedState,
        context: RunContext,
        fan_out_agents: list[Agent],
    ) -> tuple[str, int, int]:
        """执行 fan-in 聚合 Agent

        聚合 Agent 读取所有并行 Agent 的结果，产生最终输出。
        并行结果通过 SharedState 注入到 Agent 的上下文中。

        Args:
            fan_in_agent: 聚合 Agent
            task: 原始任务
            shared_state: 共享状态
            context: 运行时上下文
            fan_out_agents: 已完成的并行 Agent 列表

        Returns:
            (输出文本, prompt_tokens, completion_tokens) 元组
        """
        # 检查取消
        if context.cancellation_token:
            context.cancellation_token.throw_if_cancelled()

        fan_in_context = context.with_agent(fan_in_agent.name, turn=0)

        # 触发 on_start 钩子
        if fan_in_agent.hooks:
            await fan_in_agent.hooks.on_start(fan_in_context, fan_in_agent)

        # 构建聚合提示，将并行结果注入上下文
        instruction = task.instruction if isinstance(task, TAPRequest) else task

        # 收集并行结果摘要
        parallel_results = {}
        for a in fan_out_agents:
            if a.output_key:
                val = shared_state.get(a.output_key)
                if val is not None:
                    # 截断过长的结果避免 token 超限
                    result_str = str(val)
                    if len(result_str) > 2000:
                        result_str = result_str[:2000] + "...[truncated]"
                    parallel_results[a.name] = result_str

        # 将并行结果写入 SharedState 的专用键
        shared_state.set(
            "parallel_results",
            parallel_results,
            scope="session",
            agent_name="parallel_pattern",
        )

        provider = fan_in_agent.resolve_provider()

        if fan_in_agent.tools:
            output, prompt_tok, completion_tok = await self._run_agent_with_tools(
                task, fan_in_agent, provider, shared_state, fan_in_context
            )
        else:
            request = self._build_request(task, fan_in_agent, shared_state, fan_in_context)
            # 注入并行结果到上下文
            request.context["parallel_results"] = parallel_results

            system_prompt = fan_in_agent.get_system_prompt(fan_in_context)
            if fan_in_agent.hooks:
                await fan_in_agent.hooks.on_model_start(fan_in_context, fan_in_agent, system_prompt)

            response = await provider.execute_tap(request)

            if fan_in_agent.hooks:
                await fan_in_agent.hooks.on_model_end(fan_in_context, fan_in_agent, response)

            output = response.raw_text or ""
            prompt_tok = response.prompt_tokens
            completion_tok = response.completion_tokens

        # 触发 on_end 钩子
        if fan_in_agent.hooks:
            await fan_in_agent.hooks.on_end(fan_in_context, fan_in_agent, output)

        logger.info(
            f"Parallel fan-in agent '{fan_in_agent.name}' completed: "
            f"tokens={prompt_tok + completion_tok}"
        )

        return output, prompt_tok, completion_tok

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
        assistant_content = ""

        for step in range(agent.max_steps):
            # 检查取消
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

        # 达到 max_steps
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

    def _build_request(
        self,
        task: str | TAPRequest,
        agent: Agent,
        shared_state: SharedState,
        context: RunContext | None = None,
    ) -> TAPRequest:
        """构建 TAPRequest（无工具调用时使用）

        将原始任务、SharedState 上下文和 Agent 系统提示
        注入到 TAPRequest 中。

        始终创建新的 TAPRequest 实例，避免原地修改导致状态泄漏。

        Args:
            task: 原始任务描述或 TAPRequest
            agent: 当前要执行的 Agent
            shared_state: 共享状态

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

        # 注入 SharedState 上下文
        state_dict = shared_state.to_dict()
        if state_dict:
            summary = {k: v for i, (k, v) in enumerate(state_dict.items()) if i < 10}
            request.context["shared_state"] = summary
            if len(state_dict) > 10:
                request.context["shared_state_truncated"] = True

        return request
