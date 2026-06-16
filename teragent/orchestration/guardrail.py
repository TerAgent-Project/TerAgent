# teragent/orchestration/guardrail.py
"""Agent 守卫系统 — 输入/输出检查与跳闸机制

参考 OpenAI Agents SDK 的 InputGuardrail / OutputGuardrail，
实现 Agent 输入/输出守卫检查。

核心组件:
  - GuardrailResult: 守卫检查结果
  - GuardrailTripwireTriggered: 跳闸异常（fail-fast）
  - Guardrail: 守卫定义（输入/输出）
  - run_input_guardrails: 执行输入守卫
  - run_output_guardrails: 执行输出守卫

设计原则:
  - 并行执行：多个守卫可并行运行（asyncio.gather）
  - 跳闸模式：任一守卫触发跳闸立即终止（fail-fast）
  - 数据修改：守卫可通过 modified_data 修改输入/输出
  - 非侵入式：守卫检查是可选的，不守卫时不影响原有流程

参考:
  - OpenAI Agents SDK: InputGuardrail, OutputGuardrail, GuardrailFunctionOutput
  - Google ADK: BeforeCallback, AfterCallback
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from teragent.orchestration.agent import Agent
    from teragent.orchestration.run_context import RunContext

logger = logging.getLogger(__name__)

__all__ = [
    "GuardrailResult",
    "GuardrailTripwireTriggered",
    "Guardrail",
    "run_input_guardrails",
    "run_output_guardrails",
]


@dataclass
class GuardrailResult:
    """守卫检查结果

    Attributes:
        passed: 是否通过检查
        output_info: 检查信息（描述检查结果或原因）
        modified_data: 修改后的输入/输出数据（可选，用于数据修改场景）
    """

    passed: bool
    output_info: str = ""
    modified_data: Any | None = None


class GuardrailTripwireTriggered(Exception):
    """守卫触发跳闸 — fail-fast 模式

    参考 OpenAI Agents SDK 的 InputGuardrailTripwireTriggered /
    OutputGuardrailTripwireTriggered。

    当守卫检查未通过时抛出，编排器应捕获此异常并终止当前流程。

    Attributes:
        guardrail_name: 触发跳闸的守卫名称
        output_info: 跳闸原因描述
    """

    def __init__(self, guardrail_name: str, output_info: str):
        self.guardrail_name = guardrail_name
        self.output_info = output_info
        super().__init__(f"Guardrail '{guardrail_name}' triggered: {output_info}")


@dataclass
class Guardrail:
    """Agent 守卫

    参考 OpenAI Agents SDK 的 InputGuardrail / OutputGuardrail。

    支持:
    - 输入守卫：在 Agent 接收输入前检查
    - 输出守卫：在 Agent 产生输出后检查
    - 并行执行：多个守卫可并行运行
    - 跳闸模式：任一守卫失败立即终止（fail-fast）
    - 数据修改：守卫可修改输入/输出数据

    Attributes:
        name: 守卫名称（用于日志和调试）
        check: 守卫检查函数，接收 (agent, data, context) 参数，
               返回 GuardrailResult
        mode: 守卫模式，"input" 或 "output"
        run_in_parallel: 是否可与其他守卫并行执行
    """

    name: str
    check: Callable[..., Awaitable[GuardrailResult]]
    mode: str = "input"  # "input" | "output"
    run_in_parallel: bool = True


async def _run_single_guardrail(
    guardrail: Guardrail,
    agent: Agent,
    data: str,
    context: RunContext,
) -> GuardrailResult:
    """执行单个守卫检查

    Args:
        guardrail: 守卫定义
        agent: 当前 Agent
        data: 输入/输出数据
        context: 运行时上下文

    Returns:
        GuardrailResult 守卫检查结果
    """
    try:
        result = await guardrail.check(agent, data, context)
        logger.debug(
            f"Guardrail '{guardrail.name}' check result: "
            f"passed={result.passed}, info={result.output_info!r}"
        )
        return result
    except Exception as e:
        logger.error(f"Guardrail '{guardrail.name}' check raised exception: {e}")
        # 守卫检查自身异常视为不通过
        return GuardrailResult(
            passed=False,
            output_info=f"Guardrail check exception: {e}",
        )


async def _run_guardrails_with_fail_fast(
    guardrails: list[Guardrail],
    agent: Agent,
    data: str,
    context: RunContext,
) -> list[GuardrailResult]:
    """通用 fail-fast 守卫执行引擎

    并行执行所有守卫，任一守卫触发跳闸则取消其余并抛出异常。
    使用 asyncio.wait(FIRST_COMPLETED) 实现真正的 fail-fast，
    通过 task-to-index 映射避免 as_completed 的索引错位问题。

    Args:
        guardrails: 守卫列表
        agent: 当前 Agent
        data: 输入/输出数据
        context: 运行时上下文

    Returns:
        所有守卫检查结果列表

    Raises:
        GuardrailTripwireTriggered: 任一守卫未通过时抛出
    """
    if not guardrails:
        return []

    logger.info(f"Running {len(guardrails)} guardrails for agent '{agent.name}'")

    # 创建任务，使用 dict 映射 task → (index, guardrail) 避免索引错位
    tasks_to_info: dict[asyncio.Task, tuple[int, Guardrail]] = {}
    for i, g in enumerate(guardrails):
        task = asyncio.create_task(_run_single_guardrail(g, agent, data, context))
        tasks_to_info[task] = (i, g)

    pending = set(tasks_to_info.keys())
    results: list[GuardrailResult | None] = [None] * len(guardrails)
    failed: GuardrailTripwireTriggered | None = None

    try:
        while pending:
            # wait 返回 (done, pending) — 任一完成即返回
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                idx, guardrail = tasks_to_info[task]
                result = task.result()
                results[idx] = result

                if not result.passed:
                    logger.warning(
                        f"Guardrail '{guardrail.name}' triggered for agent '{agent.name}': "
                        f"{result.output_info}"
                    )
                    failed = GuardrailTripwireTriggered(
                        guardrail_name=guardrail.name,
                        output_info=result.output_info,
                    )
                    break

            if failed is not None:
                break
    finally:
        # 取消所有未完成的任务
        for task in pending:
            if not task.done():
                task.cancel()
        # 等待所有任务完成（包括被取消的），避免警告
        all_tasks = set(tasks_to_info.keys())
        await asyncio.gather(*all_tasks, return_exceptions=True)

    if failed is not None:
        raise failed

    return [r for r in results if r is not None]


async def run_input_guardrails(
    guardrails: list[Guardrail],
    agent: Agent,
    input_data: str,
    context: RunContext,
) -> list[GuardrailResult]:
    """执行输入守卫检查

    并行执行所有守卫，任一守卫触发跳闸则取消其余并抛出异常（fail-fast）。

    Args:
        guardrails: 输入守卫列表（只执行 mode="input" 的守卫）
        agent: 当前 Agent
        input_data: 输入数据
        context: 运行时上下文

    Returns:
        所有守卫检查结果列表

    Raises:
        GuardrailTripwireTriggered: 任一守卫未通过时抛出
    """
    input_guardrails = [g for g in guardrails if g.mode == "input"]
    return await _run_guardrails_with_fail_fast(input_guardrails, agent, input_data, context)


async def run_output_guardrails(
    guardrails: list[Guardrail],
    agent: Agent,
    output: str,
    context: RunContext,
) -> list[GuardrailResult]:
    """执行输出守卫检查

    并行执行所有守卫，任一守卫触发跳闸则取消其余并抛出异常（fail-fast）。

    Args:
        guardrails: 输出守卫列表（只执行 mode="output" 的守卫）
        agent: 当前 Agent
        output: Agent 输出数据
        context: 运行时上下文

    Returns:
        所有守卫检查结果列表

    Raises:
        GuardrailTripwireTriggered: 任一守卫未通过时抛出
    """
    output_guardrails = [g for g in guardrails if g.mode == "output"]
    return await _run_guardrails_with_fail_fast(output_guardrails, agent, output, context)
