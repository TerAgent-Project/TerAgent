"""teragent.long_horizon.self_evaluation — GLM-5 自评估模块

在长程任务执行过程中，周期性注入自评估 Prompt，
触发 GLM-5 进行自我评估，检测目标偏移和策略失效。

评估维度：
  1. 目标对齐度 — 当前方向是否正确？
  2. 产出质量 — 已完成的工作质量如何？
  3. 瓶颈识别 — 是否遇到了卡点？
  4. 策略审查 — 当前策略是否有效？
  5. 下一步规划 — 接下来应该做什么？

与 LongHorizonTaskManager 的集成方式::

    evaluator = SelfEvaluator(
        model_provider=provider,
        evaluation_interval_steps=10,
        evaluation_interval_minutes=30.0,
    )

    # 在 execute_long_task() 的阶段循环中
    if evaluator.should_evaluate(steps_since_last, minutes_since_last):
        result = await evaluator.evaluate(goal, progress_report, recent_results)
        if result.should_switch_strategy:
            # 触发策略切换
            ...
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

__all__ = [
    "SelfEvaluationResult",
    "SelfEvaluator",
]

if TYPE_CHECKING:
    from teragent.core.provider import ModelProvider
    from teragent.long_horizon.progress import ProgressReport
    from teragent.long_horizon.types import PhaseResult

logger = logging.getLogger(__name__)


@dataclass
class SelfEvaluationResult:
    """自评估结果

    解析模型输出的结构化自评估结果。
    每个维度使用 1-5 的评分，1 表示最差，5 表示最好。

    Attributes:
        goal_alignment: 目标对齐度 1-5（当前方向与原始目标的对齐程度）
        output_quality: 产出质量 1-5（已完成工作的质量评估）
        bottleneck_identified: 瓶颈识别（文字描述当前卡点）
        strategy_review: 策略审查（文字评估当前策略有效性）
        next_step_plan: 下一步规划（建议的后续行动）
        overall_score: 综合评分（各维度的加权平均）
        should_switch_strategy: 是否应切换策略
        raw_response: 原始响应文本
    """

    goal_alignment: int  # 目标对齐度 1-5
    output_quality: int  # 产出质量 1-5
    bottleneck_identified: str  # 瓶颈识别
    strategy_review: str  # 策略审查
    next_step_plan: str  # 下一步规划
    overall_score: float  # 综合评分
    should_switch_strategy: bool  # 是否应切换策略
    raw_response: str  # 原始响应文本


class SelfEvaluator:
    """自评估执行器

    在长程任务执行过程中，周期性注入自评估 Prompt，
    触发模型进行自我评估。

    评估维度：
    1. 目标对齐度：当前方向是否正确？
    2. 产出质量：已完成的工作质量如何？
    3. 瓶颈识别：是否遇到了卡点？
    4. 策略审查：当前策略是否有效？
    5. 下一步规划：接下来应该做什么？

    使用方式::

        evaluator = SelfEvaluator(model_provider=provider)
        if evaluator.should_evaluate(steps_since_last=10, minutes_since_last=30.0):
            result = await evaluator.evaluate(goal, progress_report, recent_results)

    Attributes:
        model_provider: GLM-5 的 ModelProvider 实例
        evaluation_interval_steps: 每N步评估一次
        evaluation_interval_minutes: 每N分钟评估一次
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        evaluation_interval_steps: int = 10,
        evaluation_interval_minutes: float = 30.0,
    ) -> None:
        """初始化自评估执行器

        Args:
            model_provider: GLM-5 的 ModelProvider 实例
            evaluation_interval_steps: 每N步评估一次，默认10步
            evaluation_interval_minutes: 每N分钟评估一次，默认30分钟
        """
        self.model_provider = model_provider
        self.evaluation_interval_steps = evaluation_interval_steps
        self.evaluation_interval_minutes = evaluation_interval_minutes

        # 追踪上次评估时间
        self._last_evaluation_time: float = time.monotonic()
        self._last_evaluation_steps: int = 0

    async def evaluate(
        self,
        goal: str,
        progress_report: ProgressReport,
        recent_results: list[PhaseResult],
    ) -> SelfEvaluationResult:
        """执行自评估

        流程：
        1. 构建结构化自评估 Prompt
        2. 调用 GLM-5 生成评估
        3. 解析评估结果（1-5 评分 + 文字分析）
        4. 返回 SelfEvaluationResult

        Args:
            goal: 原始大目标描述
            progress_report: 当前进度报告
            recent_results: 最近N个阶段的执行结果

        Returns:
            SelfEvaluationResult 自评估结果
        """
        # 1. 构建自评估 Prompt
        prompt = self._build_evaluation_prompt(goal, progress_report, recent_results)

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个专业的项目自评估专家。请对当前任务执行状态进行客观、"
                    "深入的自我评估。\n"
                    "你必须以 JSON 格式返回评估结果，格式如下：\n"
                    '{\n'
                    '  "goal_alignment": 1-5的整数,\n'
                    '  "output_quality": 1-5的整数,\n'
                    '  "bottleneck_identified": "瓶颈描述",\n'
                    '  "strategy_review": "策略审查描述",\n'
                    '  "next_step_plan": "下一步规划",\n'
                    '  "should_switch_strategy": true或false\n'
                    '}\n'
                    "只返回 JSON 对象，不要包含其他文字。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # 2. 调用模型
        try:
            response = await self.model_provider.chat(messages)
            content = response.get("content", "")
        except Exception as e:
            logger.warning(f"Self-evaluation model call failed: {e}")
            # 模型调用失败时返回默认评估
            content = ""

        # 3. 解析评估结果
        result = self._parse_evaluation_response(content)

        # 4. 更新追踪状态
        self._last_evaluation_time = time.monotonic()

        logger.info(
            f"Self-evaluation completed: overall_score={result.overall_score:.1f} "
            f"goal_alignment={result.goal_alignment} "
            f"should_switch={result.should_switch_strategy}"
        )

        return result

    def should_evaluate(
        self,
        steps_since_last: int,
        minutes_since_last: float,
    ) -> bool:
        """判断是否需要执行自评估

        当满足以下任一条件时触发评估：
        - 距上次评估的步数 >= evaluation_interval_steps
        - 距上次评估的分钟数 >= evaluation_interval_minutes

        Args:
            steps_since_last: 距上次评估的步数
            minutes_since_last: 距上次评估的分钟数

        Returns:
            是否需要执行自评估
        """
        if steps_since_last >= self.evaluation_interval_steps:
            return True
        if minutes_since_last >= self.evaluation_interval_minutes:
            return True
        return False

    def reset_evaluation_timer(self, current_steps: int) -> None:
        """重置评估计时器

        在执行完一次评估后调用，重置步数和时间追踪。

        Args:
            current_steps: 当前的总步骤数
        """
        self._last_evaluation_time = time.monotonic()
        self._last_evaluation_steps = current_steps

    def _build_evaluation_prompt(
        self,
        goal: str,
        progress_report: ProgressReport,
        recent_results: list[PhaseResult],
    ) -> str:
        """构建结构化自评估 Prompt

        包含：
        - 原始目标
        - 当前进度（已完成/进行中/待执行）
        - 最近N个阶段的执行结果
        - 评估维度说明
        - 期望的输出格式

        Args:
            goal: 原始大目标描述
            progress_report: 当前进度报告
            recent_results: 最近N个阶段的执行结果

        Returns:
            结构化的自评估 Prompt 文本
        """
        parts: list[str] = []

        # 原始目标
        parts.append(f"【原始目标】\n{goal}")

        # 当前进度
        parts.append(
            f"\n【当前进度】\n"
            f"- 已完成子目标：{progress_report.completed_sub_goals}/{progress_report.total_sub_goals}\n"
            f"- 已执行步骤：{progress_report.steps_completed}\n"
            f"- 已耗时间：{progress_report.elapsed_minutes:.1f}分钟\n"
            f"- 策略切换次数：{progress_report.strategy_switches}\n"
            f"- 当前阶段：{progress_report.current_phase}"
        )

        # 子目标状态详情
        if progress_report.sub_goal_statuses:
            parts.append("\n【子目标详情】")
            for sg in progress_report.sub_goal_statuses:
                status_icon = {
                    "completed": "✓",
                    "in_progress": "→",
                    "pending": "○",
                    "failed": "✗",
                }.get(sg.get("status", ""), "?")
                parts.append(
                    f"  {status_icon} {sg.get('id', '?')}: "
                    f"{sg.get('description', '')} [{sg.get('status', '?')}]"
                )

        # 最近执行结果
        if recent_results:
            parts.append("\n【最近执行结果】")
            # 只取最近5个结果，避免 Prompt 过长
            for pr in recent_results[-5:]:
                status = "成功" if pr.success else "失败"
                files_info = ""
                if pr.files_created:
                    files_info = f"，创建了: {', '.join(pr.files_created[:3])}"
                result_preview = pr.result_text[:150] if pr.result_text else "(无输出)"
                parts.append(
                    f"  [{status}] {pr.sub_goal_id}: {result_preview}{files_info}"
                )
                if pr.errors:
                    parts.append(f"    错误: {'; '.join(pr.errors[:2])}")

        # 评估维度说明
        parts.append(
            "\n【评估维度】\n"
            "1. goal_alignment (1-5): 当前执行方向是否与原始目标对齐？\n"
            "   1=严重偏移, 2=有偏移, 3=基本对齐, 4=良好对齐, 5=完全对齐\n"
            "2. output_quality (1-5): 已完成工作的质量如何？\n"
            "   1=质量极差, 2=质量较差, 3=质量一般, 4=质量良好, 5=质量优秀\n"
            "3. bottleneck_identified: 当前是否遇到瓶颈？请具体描述。\n"
            "4. strategy_review: 当前策略是否有效？请评估并说明。\n"
            "5. next_step_plan: 建议的下一步行动。\n"
            "6. should_switch_strategy: 是否建议切换策略？(true/false)\n"
            "   当 goal_alignment <= 2 或 output_quality <= 2 时应考虑切换。"
        )

        return "\n".join(parts)

    def _parse_evaluation_response(self, response: str) -> SelfEvaluationResult:
        """解析模型输出的自评估结果

        尝试解析结构化格式（评分 + 分析文本）。
        如果解析失败，使用启发式方法提取关键信息。

        解析策略：
        1. 尝试从 JSON 格式解析
        2. 如果 JSON 解析失败，尝试从文本中提取关键字
        3. 最后使用默认值

        Args:
            response: 模型返回的原始响应文本

        Returns:
            SelfEvaluationResult 自评估结果
        """
        if not response:
            return self._default_evaluation_result(response)

        # 策略1：尝试从 JSON 解析
        json_result = self._try_parse_json(response)
        if json_result is not None:
            return json_result

        # 策略2：启发式提取
        heuristic_result = self._try_heuristic_parse(response)
        if heuristic_result is not None:
            return heuristic_result

        # 策略3：默认值
        return self._default_evaluation_result(response)

    def _try_parse_json(self, response: str) -> SelfEvaluationResult | None:
        """尝试从 JSON 格式解析评估结果

        Args:
            response: 模型返回的原始响应文本

        Returns:
            SelfEvaluationResult 如果解析成功，否则 None
        """
        json_str = response.strip()

        # 提取 markdown 代码块中的内容
        if "```" in json_str:
            match = re.search(r"```(?:json)?\s*\n(.*?)\n```", json_str, re.DOTALL)
            if match:
                json_str = match.group(1).strip()

        # 找到 JSON 对象
        start = json_str.find("{")
        end = json_str.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        json_str = json_str[start:end + 1]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        # 提取各字段，带默认值保护
        goal_alignment = self._clamp_score(data.get("goal_alignment", 3))
        output_quality = self._clamp_score(data.get("output_quality", 3))

        bottleneck = str(data.get("bottleneck_identified", "未识别到明显瓶颈"))
        strategy_review = str(data.get("strategy_review", "当前策略基本有效"))
        next_step_plan = str(data.get("next_step_plan", "继续按当前计划执行"))

        # 解析 should_switch_strategy
        switch_val = data.get("should_switch_strategy", False)
        if isinstance(switch_val, str):
            should_switch = switch_val.lower() in ("true", "yes", "是", "1")
        else:
            should_switch = bool(switch_val)

        # 计算综合评分：目标对齐度权重0.6，产出质量权重0.4
        overall_score = goal_alignment * 0.6 + output_quality * 0.4

        # 如果评分很低但模型没建议切换，自动判断
        if goal_alignment <= 2 or output_quality <= 2:
            should_switch = True

        return SelfEvaluationResult(
            goal_alignment=goal_alignment,
            output_quality=output_quality,
            bottleneck_identified=bottleneck,
            strategy_review=strategy_review,
            next_step_plan=next_step_plan,
            overall_score=overall_score,
            should_switch_strategy=should_switch,
            raw_response=response,
        )

    def _try_heuristic_parse(self, response: str) -> SelfEvaluationResult | None:
        """使用启发式方法从文本中提取评估结果

        当 JSON 解析失败时，尝试从文本中提取关键信息。
        寻找评分关键词和数字模式。

        Args:
            response: 模型返回的原始响应文本

        Returns:
            SelfEvaluationResult 如果提取成功，否则 None
        """
        # 查找评分数字（1-5）
        # 匹配 "目标对齐度：3" 或 "goal_alignment: 4" 等模式
        goal_alignment = 3  # 默认值
        output_quality = 3  # 默认值

        # 目标对齐度
        ga_patterns = [
            r"目标对齐度[：:]\s*(\d)",
            r"goal.alignment[：:]\s*(\d)",
            r"对齐度[：:]\s*(\d)",
            r"方向[：:]\s*(\d)",
        ]
        for pattern in ga_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                goal_alignment = self._clamp_score(int(match.group(1)))
                break

        # 产出质量
        oq_patterns = [
            r"产出质量[：:]\s*(\d)",
            r"output.quality[：:]\s*(\d)",
            r"质量[：:]\s*(\d)",
        ]
        for pattern in oq_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                output_quality = self._clamp_score(int(match.group(1)))
                break

        # 查找"评分"或"分数"关键词
        score_match = re.search(r"(?:综合)?评[分价][：:]\s*(\d+(?:\.\d+)?)", response)
        overall_score = goal_alignment * 0.6 + output_quality * 0.4
        if score_match:
            try:
                overall_score = float(score_match.group(1))
            except ValueError:
                pass

        # 查找策略切换建议
        should_switch = False
        switch_keywords = ["切换策略", "更换策略", "调整策略", "策略失效", "应该切换"]
        for kw in switch_keywords:
            if kw in response:
                should_switch = True
                break

        # 如果评分很低，自动建议切换
        if goal_alignment <= 2 or output_quality <= 2:
            should_switch = True

        # 提取瓶颈描述
        bottleneck = "未识别到明显瓶颈"
        bottleneck_match = re.search(
            r"瓶颈[：:]\s*(.+?)(?:\n|$)", response
        )
        if bottleneck_match:
            bottleneck = bottleneck_match.group(1).strip()

        # 提取策略审查
        strategy_review = "当前策略基本有效"
        strategy_match = re.search(
            r"策略(?:审查|评估|分析)[：:]\s*(.+?)(?:\n|$)", response
        )
        if strategy_match:
            strategy_review = strategy_match.group(1).strip()

        # 提取下一步规划
        next_step_plan = "继续按当前计划执行"
        next_match = re.search(
            r"下一步(?:规划|计划|行动)[：:]\s*(.+?)(?:\n|$)", response
        )
        if next_match:
            next_step_plan = next_match.group(1).strip()

        # 如果连任何评分数字都没找到，返回 None 让默认处理器处理
        if not any(re.search(p, response, re.IGNORECASE) for p in ga_patterns + oq_patterns):
            return None

        return SelfEvaluationResult(
            goal_alignment=goal_alignment,
            output_quality=output_quality,
            bottleneck_identified=bottleneck,
            strategy_review=strategy_review,
            next_step_plan=next_step_plan,
            overall_score=overall_score,
            should_switch_strategy=should_switch,
            raw_response=response,
        )

    def _default_evaluation_result(self, raw_response: str) -> SelfEvaluationResult:
        """生成默认的自评估结果

        当所有解析方法都失败时使用。

        Args:
            raw_response: 原始响应文本

        Returns:
            SelfEvaluationResult 默认评估结果
        """
        return SelfEvaluationResult(
            goal_alignment=3,
            output_quality=3,
            bottleneck_identified="无法解析评估结果",
            strategy_review="评估解析失败，使用默认值",
            next_step_plan="继续执行并观察",
            overall_score=3.0,
            should_switch_strategy=False,
            raw_response=raw_response,
        )

    @staticmethod
    def _clamp_score(value: int) -> int:
        """将评分限制在 1-5 范围内

        Args:
            value: 原始评分值

        Returns:
            限制在 1-5 范围内的评分
        """
        return max(1, min(5, int(value)))
