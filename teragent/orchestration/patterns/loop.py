# teragent/orchestration/patterns/loop.py
"""循环迭代编排模式

Generator-Critic 循环模式：Agent 按列表顺序依次执行，每轮迭代后检查退出条件。
退出条件可以是可调用函数、SharedState 中的字符串匹配，或最大迭代次数。

执行流程:
  1. 所有 Agent 按顺序执行（一次迭代）
  2. 检查退出条件
  3. 如果退出条件不满足且未达到最大迭代次数，继续下一轮
  4. 退出条件满足或达到最大迭代次数时，返回最终结果

退出条件类型:
  - callable: 接受 (shared_state, iteration) 参数，返回 True 表示应退出
  - str: 在 SharedState 中查找匹配该字符串的值，找到即退出
  - dict: 包含 key 和 value 字段，在 SharedState 中检查指定键值
  - None: 仅通过 max_iterations 控制退出

参考: LangGraph 的循环图, AutoGen 的 GroupChat (max_rounds), CrewAI 的循环任务

典型场景:
  - Generator-Critic: 生成器 Agent 产出内容，批评者 Agent 评审，
    循环直到评审通过或达到最大迭代次数
  - 迭代优化: Agent 逐步改进输出质量
  - 多轮对话: Agent 轮流发言，直到达成共识
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, TYPE_CHECKING

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
    "LoopPattern",
]


class LoopPattern(OrchestrationPattern):
    """循环迭代编排模式

    Generator-Critic 循环模式：Agent 按列表顺序依次执行，
    每轮迭代后检查退出条件。

    参考: LangGraph 的循环图, AutoGen 的 GroupChat (max_rounds)

    执行流程:
    1. 所有 Agent 按顺序执行（一次迭代）
    2. 检查退出条件
    3. 如果退出条件不满足且未达到最大迭代次数，继续下一轮
    4. 退出条件满足或达到最大迭代次数时，返回最终结果

    退出条件类型:
    - callable: 接受 (shared_state, iteration) 参数，返回 True 表示应退出
    - str: 在 SharedState 中查找匹配该字符串的值，找到即退出
    - dict: 包含 key 和 value 字段，在 SharedState 中检查指定键值
    - None: 仅通过 max_iterations 控制退出
    """

    async def run(
        self,
        task: str | TAPRequest,
        agents: list[Agent],
        shared_state: SharedState,
        context: RunContext,
        **kwargs,
    ) -> OrchestrationResult:
        """执行循环迭代编排

        Agent 按列表顺序依次执行，每轮迭代后检查退出条件。

        Args:
            task: 任务描述或 TAPRequest
            agents: 按顺序执行的 Agent 列表
            shared_state: 跨 Agent 共享状态
            context: 运行时上下文
            **kwargs: 额外参数
                max_iterations: 最大迭代次数（默认 5）
                exit_condition: 退出条件（callable/str/dict/None）

        Returns:
            OrchestrationResult 包含最终输出和各 Agent 结果
        """
        if not agents:
            return OrchestrationResult(final_output="", total_turns=0)

        max_iterations = kwargs.get("max_iterations", 5)
        exit_condition = kwargs.get("exit_condition", None)

        # 发射循环开始事件
        if context.event_bus:
            await context.event_bus.emit(
                "loop_started",
                max_iterations=max_iterations,
                agents=[a.name for a in agents],
                has_exit_condition=exit_condition is not None,
            )

        total_prompt = 0
        total_completion = 0
        last_output = ""
        last_agent_name = agents[-1].name if agents else ""
        iteration = 0
        exit_reason = "max_iterations"

        for iteration in range(max_iterations):
            # 检查取消
            if context.cancellation_token:
                context.cancellation_token.throw_if_cancelled()

            # 发射迭代开始事件
            if context.event_bus:
                await context.event_bus.emit(
                    "loop_iteration_started",
                    iteration=iteration + 1,
                    max_iterations=max_iterations,
                )

            logger.info(
                f"Loop iteration {iteration + 1}/{max_iterations} started"
            )

            # 执行本轮所有 Agent（顺序执行）
            for agent_idx, agent in enumerate(agents):
                # 检查取消
                if context.cancellation_token:
                    context.cancellation_token.throw_if_cancelled()

                # 更新上下文
                agent_context = context.with_agent(
                    agent.name,
                    turn=iteration * len(agents) + agent_idx,
                )

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
                            f"Loop input guardrail triggered for agent '{agent.name}': {e.output_info}"
                        )
                        # 守卫跳闸：跳过该 Agent，继续循环
                        continue

                # 解析 provider
                provider = agent.resolve_provider()

                # 选择执行路径
                if agent.tools:
                    output, prompt_tok, completion_tok = await self._run_agent_with_tools(
                        task, agent, provider, shared_state, agent_context,
                        iteration=iteration,
                    )
                else:
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
                            f"Loop output guardrail triggered for agent '{agent.name}': {e.output_info}"
                        )
                        # 守卫跳闸：跳过该 Agent 的输出，继续循环
                        continue

                # 写入 SharedState
                if agent.output_key:
                    shared_state.set(
                        agent.output_key,
                        output,
                        scope="session",
                        agent_name=agent.name,
                    )

                # 追踪 token 使用
                total_prompt += prompt_tok
                total_completion += completion_tok
                context.usage.record(agent.name, prompt_tok, completion_tok)

                last_output = output
                last_agent_name = agent.name

                # 触发 on_end 钩子
                if agent.hooks:
                    await agent.hooks.on_end(agent_context, agent, output)

                # 发射 step 完成事件
                if context.event_bus:
                    await context.event_bus.emit(
                        "orchestration_step_completed",
                        agent_name=agent.name,
                        iteration=iteration + 1,
                        step=agent_idx + 1,
                        output_preview=output[:200] if output else "",
                    )

                logger.info(
                    f"Loop iteration {iteration + 1}, "
                    f"step {agent_idx + 1}/{len(agents)}: "
                    f"agent={agent.name} tokens={prompt_tok + completion_tok}"
                )

            # 本轮迭代完成，检查退出条件
            if exit_condition and self._check_exit(shared_state, exit_condition, iteration):
                exit_reason = "exit_condition_met"
                logger.info(
                    f"Loop exit condition met at iteration {iteration + 1}"
                )
                break

            # 发射迭代完成事件
            if context.event_bus:
                await context.event_bus.emit(
                    "loop_iteration_completed",
                    iteration=iteration + 1,
                )

        # 发射循环完成事件
        if context.event_bus:
            await context.event_bus.emit(
                "loop_completed",
                iterations=iteration + 1,
                exit_reason=exit_reason,
            )

        return OrchestrationResult(
            final_output=last_output,
            last_agent=last_agent_name,
            agent_outputs={
                a.output_key: shared_state.get(a.output_key)
                for a in agents if a.output_key
            },
            total_turns=(iteration + 1) * len(agents),
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            metadata={
                "loop_completed": True,
                "iterations": iteration + 1,
                "max_iterations": max_iterations,
                "exit_reason": exit_reason,
            },
        )

    def get_next_agent(self, current: Agent | None, result: Any) -> Agent | None:
        """循环模式按固定顺序执行，此方法始终返回 None

        Agent 的执行顺序由 agents 列表和迭代次数决定。
        """
        return None

    def get_execution_plan(self) -> list[dict]:
        """获取执行计划"""
        return [
            {"type": "loop", "description": "Iterative generator-critic loop pattern"},
        ]

    # ===== 内部方法 =====

    def _check_exit(
        self,
        shared_state: SharedState,
        exit_condition: Any,
        iteration: int,
    ) -> bool:
        """检查退出条件

        支持多种退出条件类型:
        - callable: 接受 (shared_state, iteration) 参数，返回 True 表示应退出
        - str: 在 SharedState 的值中查找匹配该字符串的内容
        - dict: 包含 key 和 value 字段，在 SharedState 中检查指定键值

        Args:
            shared_state: 共享状态
            exit_condition: 退出条件
            iteration: 当前迭代次数

        Returns:
            True 表示应退出循环
        """
        # 可调用函数
        if callable(exit_condition):
            try:
                result = exit_condition(shared_state, iteration)
                return bool(result)
            except Exception as e:
                logger.warning(
                    f"Exit condition callable raised exception: {e}. "
                    f"Continuing loop."
                )
                return False

        # 字符串匹配：在 SharedState 的值中查找
        if isinstance(exit_condition, str):
            state_dict = shared_state.to_dict()
            for key, value in state_dict.items():
                if isinstance(value, str) and exit_condition in value:
                    logger.debug(
                        f"Exit condition string '{exit_condition}' found in "
                        f"SharedState key '{key}'"
                    )
                    return True
            return False

        # 字典条件：检查 SharedState 中指定键值
        if isinstance(exit_condition, dict):
            key = exit_condition.get("key", "")
            expected_value = exit_condition.get("value")

            if not key:
                logger.warning(
                    "Exit condition dict missing 'key' field. Ignoring."
                )
                return False

            actual_value = shared_state.get(key)
            if actual_value is None:
                return False

            # 如果指定了期望值，检查是否匹配
            if expected_value is not None:
                try:
                    return actual_value == expected_value
                except Exception:
                    return False

            # 只有 key 没有 value：只要 key 存在就退出
            return True

        # 不支持的退出条件类型
        logger.warning(
            f"Unsupported exit_condition type: {type(exit_condition).__name__}. "
            f"Expected callable, str, or dict."
        )
        return False

    async def _run_agent_with_tools(
        self,
        task: str | TAPRequest,
        agent: Agent,
        provider: Any,
        shared_state: SharedState,
        context: RunContext,
        iteration: int = 0,
    ) -> tuple[str, int, int]:
        """执行带工具调用的 Agent

        使用 provider.chat() + 工具调用循环。

        Args:
            task: 任务描述
            agent: 当前 Agent
            provider: ModelProvider 实例
            shared_state: 共享状态
            context: 运行时上下文
            iteration: 当前迭代次数

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
        chat_messages = self._build_chat_messages(task, agent, context, iteration)

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
        iteration: int = 0,
    ) -> list[dict]:
        """构建 chat 接口的 messages 列表

        在循环模式中，除了系统提示和任务指令外，
        还会注入当前迭代信息，帮助 Agent 理解当前进度。

        Args:
            task: 原始任务描述或 TAPRequest
            agent: 当前 Agent
            context: 运行时上下文
            iteration: 当前迭代次数

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
            # 在后续迭代中，附加迭代信息提示
            if iteration > 0:
                messages.append({
                    "role": "user",
                    "content": f"[Iteration {iteration + 1}] {instruction}",
                })
            else:
                messages.append({"role": "user", "content": instruction})

        return messages

    def _build_request(
        self,
        task: str | TAPRequest,
        agent: Agent,
        shared_state: SharedState,
        context: RunContext | None = None,
        iteration: int = 0,
    ) -> TAPRequest:
        """构建 TAPRequest（无工具调用时使用）

        将原始任务、SharedState 上下文和迭代信息
        注入到 TAPRequest 中。

        始终创建新的 TAPRequest 实例，避免原地修改导致状态泄漏。

        Args:
            task: 原始任务描述或 TAPRequest
            agent: 当前要执行的 Agent
            shared_state: 共享状态
            context: 运行时上下文
            iteration: 当前迭代次数

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

        # 注入迭代信息
        request.context["loop_iteration"] = iteration + 1

        return request
