"""teragent.long_horizon.strategy_switch — GLM-5 策略切换模块

当检测到模型陷入局部最优（停滞）时，通过 Prompt 引导切换策略，
避免长时间在无效方向上消耗算力。

触发条件：
  1. 连续N次工具调用结果相似度 > 阈值 → 停滞
  2. 连续M步无新文件产出 → 停滞
  3. 自评估建议切换策略 → 主动切换
  4. 进度追踪器检测到长时间无进展 → 超时

策略切换流程：
  1. 检测停滞
  2. 注入策略切换 Prompt
  3. 等待模型选择新策略
  4. 记录切换
  5. 继续执行

与 LongHorizonTaskManager 的集成方式::

    switcher = StrategySwitcher(model_provider=provider)

    # 检测停滞
    is_stagnant, reason = switcher.detect_stagnation(recent_results, recent_steps)
    if is_stagnant:
        new_strategy, record = await switcher.switch_strategy(
            current_strategy, reason, goal, progress_report
        )
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from teragent.core.provider import ModelProvider
    from teragent.long_horizon.progress import ProgressReport
    from teragent.long_horizon.types import PhaseResult

logger = logging.getLogger(__name__)


@dataclass
class StrategySwitchRecord:
    """策略切换记录

    记录每次策略切换的详细信息，用于事后分析和效果评估。

    Attributes:
        timestamp: 切换时间（ISO 格式）
        reason: 切换原因
        previous_strategy: 前一个策略描述
        new_strategy: 新策略描述
        risk_assessment: 切换风险评估
        effectiveness: 事后效果评估（后续填充）
    """

    timestamp: str  # ISO format
    reason: str  # 切换原因
    previous_strategy: str  # 前一个策略描述
    new_strategy: str  # 新策略描述
    risk_assessment: str  # 切换风险评估
    effectiveness: str = ""  # 事后效果评估 (filled later)


class StrategySwitcher:
    """策略切换管理器

    当检测到模型陷入局部最优时，通过 prompt 引导切换策略。

    触发条件：
    1. 连续N次工具调用结果相似度 > 阈值 → 停滞
    2. 连续M步无新文件产出 → 停滞
    3. 自评估建议切换策略 → 主动切换
    4. 进度追踪器检测到长时间无进展 → 超时

    策略切换流程：
    1. 检测停滞
    2. 注入策略切换 Prompt
    3. 等待模型选择新策略
    4. 记录切换
    5. 继续执行

    使用方式::

        switcher = StrategySwitcher(model_provider=provider)
        is_stagnant, reason = switcher.detect_stagnation(results, steps)
        if is_stagnant:
            new_strategy, record = await switcher.switch_strategy(
                current_strategy, reason, goal, progress_report
            )

    Attributes:
        model_provider: GLM-5 的 ModelProvider 实例
        stagnation_threshold: 连续N次相同结果判定为停滞
        no_progress_threshold: 连续M步无新产出判定为停滞
        similarity_threshold: 结果相似度阈值
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        stagnation_threshold: int = 3,
        no_progress_threshold: int = 5,
        similarity_threshold: float = 0.8,
    ) -> None:
        """初始化策略切换管理器

        Args:
            model_provider: GLM-5 的 ModelProvider 实例
            stagnation_threshold: 连续N次相同结果判定为停滞，默认3
            no_progress_threshold: 连续M步无新产出判定为停滞，默认5
            similarity_threshold: 结果相似度阈值（Jaccard），默认0.8
        """
        self.model_provider = model_provider
        self.stagnation_threshold = stagnation_threshold
        self.no_progress_threshold = no_progress_threshold
        self.similarity_threshold = similarity_threshold

        # 切换历史
        self._switch_history: list[StrategySwitchRecord] = []

        # 当前策略描述
        self._current_strategy: str = "初始策略"

    @property
    def current_strategy(self) -> str:
        """当前策略描述"""
        return self._current_strategy

    def detect_stagnation(
        self,
        recent_results: list[PhaseResult],
        recent_steps: list[str],
    ) -> tuple[bool, str]:
        """检测是否陷入停滞

        综合多种信号判断是否停滞：
        1. 连续N次结果高度相似（Jaccard 相似度 > 阈值）
        2. 连续M步无新文件产出
        3. 连续失败

        Args:
            recent_results: 最近的阶段执行结果列表
            recent_steps: 最近N步的描述

        Returns:
            (is_stagnant, reason) — 是否停滞及原因
        """
        # 检查1：连续N次结果高度相似
        if len(recent_results) >= self.stagnation_threshold:
            recent_texts = [
                pr.result_text for pr in recent_results[-self.stagnation_threshold:]
                if pr.result_text
            ]
            if len(recent_texts) >= self.stagnation_threshold:
                # 检查所有相邻结果对的相似度
                all_similar = True
                for i in range(len(recent_texts) - 1):
                    sim = self._calculate_similarity(recent_texts[i], recent_texts[i + 1])
                    if sim < self.similarity_threshold:
                        all_similar = False
                        break

                if all_similar:
                    reason = (
                        f"连续{self.stagnation_threshold}次执行结果高度相似"
                        f"（相似度 > {self.similarity_threshold}）"
                    )
                    logger.warning(f"Stagnation detected: {reason}")
                    return True, reason

        # 检查2：连续M步无新文件产出
        no_file_count = 0
        for pr in reversed(recent_results):
            if not pr.files_created and not pr.files_modified:
                no_file_count += 1
            else:
                break

        if no_file_count >= self.no_progress_threshold:
            reason = f"连续{no_file_count}步无新文件产出"
            logger.warning(f"Stagnation detected: {reason}")
            return True, reason

        # 检查3：连续失败
        consecutive_failures = 0
        for pr in reversed(recent_results):
            if not pr.success:
                consecutive_failures += 1
            else:
                break

        if consecutive_failures >= self.stagnation_threshold:
            reason = f"连续{consecutive_failures}次执行失败"
            logger.warning(f"Stagnation detected: {reason}")
            return True, reason

        # 检查4：步骤描述高度重复
        if len(recent_steps) >= self.stagnation_threshold:
            recent_descs = recent_steps[-self.stagnation_threshold:]
            if len(recent_descs) >= self.stagnation_threshold:
                all_similar = True
                for i in range(len(recent_descs) - 1):
                    sim = self._calculate_similarity(recent_descs[i], recent_descs[i + 1])
                    if sim < self.similarity_threshold:
                        all_similar = False
                        break

                if all_similar:
                    reason = f"连续{self.stagnation_threshold}步描述高度相似"
                    logger.warning(f"Stagnation detected: {reason}")
                    return True, reason

        return False, ""

    async def switch_strategy(
        self,
        current_strategy: str,
        reason: str,
        goal: str,
        progress_report: ProgressReport,
    ) -> tuple[str, StrategySwitchRecord]:
        """执行策略切换

        流程：
        1. 构建策略切换 Prompt
        2. 调用 GLM-5 选择新策略
        3. 记录切换
        4. 返回新策略描述和切换记录

        Args:
            current_strategy: 当前策略描述
            reason: 切换原因
            goal: 原始大目标
            progress_report: 当前进度报告

        Returns:
            (new_strategy, switch_record) — 新策略描述和切换记录
        """
        # 构建策略切换 Prompt
        prompt = self._build_strategy_switch_prompt(
            current_strategy, reason, goal, progress_report
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个策略规划专家。当前任务执行陷入停滞，需要切换策略。\n"
                    "请分析停滞原因，提出新的执行策略，并评估切换风险。\n"
                    "你必须以 JSON 格式返回，格式如下：\n"
                    '{\n'
                    '  "new_strategy": "新策略描述",\n'
                    '  "risk_assessment": "切换风险评估",\n'
                    '  "rationale": "选择此策略的理由"\n'
                    '}\n'
                    "只返回 JSON 对象，不要包含其他文字。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # 调用模型
        try:
            response = await self.model_provider.chat(messages)
            content = response.get("content", "")
        except Exception as e:
            logger.warning(f"Strategy switch model call failed: {e}")
            content = ""

        # 解析新策略
        new_strategy, risk_assessment = self._parse_strategy_response(
            content, current_strategy, reason
        )

        # 记录切换
        timestamp = datetime.now(timezone.utc).isoformat()
        record = StrategySwitchRecord(
            timestamp=timestamp,
            reason=reason,
            previous_strategy=current_strategy,
            new_strategy=new_strategy,
            risk_assessment=risk_assessment,
        )
        self._switch_history.append(record)

        # 更新当前策略
        self._current_strategy = new_strategy

        logger.info(
            f"Strategy switched: '{current_strategy[:50]}' -> '{new_strategy[:50]}' "
            f"reason={reason[:80]}"
        )

        return new_strategy, record

    def _build_strategy_switch_prompt(
        self,
        current_strategy: str,
        reason: str,
        goal: str,
        progress_report: ProgressReport,
    ) -> str:
        """构建策略切换引导 Prompt

        包含：
        - 当前策略描述
        - 停滞原因
        - 可选的替代策略列表
        - 切换风险评估提示
        - 期望的输出格式

        Args:
            current_strategy: 当前策略描述
            reason: 停滞原因
            goal: 原始大目标
            progress_report: 当前进度报告

        Returns:
            策略切换引导 Prompt 文本
        """
        parts: list[str] = []

        # 原始目标
        parts.append(f"【任务目标】\n{goal}")

        # 当前策略和停滞原因
        parts.append(
            f"\n【当前策略】\n{current_strategy}\n\n"
            f"【停滞原因】\n{reason}"
        )

        # 当前进度
        parts.append(
            f"\n【当前进度】\n"
            f"- 已完成子目标：{progress_report.completed_sub_goals}/{progress_report.total_sub_goals}\n"
            f"- 已执行步骤：{progress_report.steps_completed}\n"
            f"- 已耗时间：{progress_report.elapsed_minutes:.1f}分钟\n"
            f"- 策略切换次数：{progress_report.strategy_switches}\n"
            f"- 当前阶段：{progress_report.current_phase}"
        )

        # 子目标状态
        if progress_report.sub_goal_statuses:
            parts.append("\n【子目标状态】")
            for sg in progress_report.sub_goal_statuses:
                status = sg.get("status", "?")
                desc = sg.get("description", "")
                parts.append(f"  [{status}] {desc}")

        # 替代策略选项
        parts.append(
            "\n【可选的替代策略方向】\n"
            "1. 分解细化 — 将当前子目标进一步分解为更小的步骤\n"
            "2. 回退重试 — 回到上一个成功状态，尝试不同的路径\n"
            "3. 跳过绕行 — 跳过当前卡住的子目标，先完成其他部分\n"
            "4. 换工具/方法 — 尝试不同的技术方案或工具\n"
            "5. 增量验证 — 每完成一小步就验证，防止方向偏移\n"
            "6. 重新规划 — 重新评估目标分解，调整子目标结构\n"
            "\n请选择一个策略方向（可以是上述之一或自定义），并说明理由。"
        )

        # 风险评估提示
        parts.append(
            "\n【风险评估要点】\n"
            "- 切换策略是否会导致已完成的工作浪费？\n"
            "- 新策略是否可能引入新的风险？\n"
            "- 切换成本（时间、算力）是否可接受？"
        )

        return "\n".join(parts)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度（简单 Jaccard 系数）

        使用词集合的交集/并集计算 Jaccard 相似度，
        用于检测连续结果是否相似（停滞的标志）。

        对于中文文本，按字符拆分；对于英文文本，按空格拆分为词。

        Args:
            text1: 第一个文本
            text2: 第二个文本

        Returns:
            Jaccard 相似度，范围 [0.0, 1.0]
        """
        if not text1 and not text2:
            return 1.0  # 两个空文本视为完全相同
        if not text1 or not text2:
            return 0.0  # 一个空一个非空视为完全不同

        # 将文本转换为词/字符集合
        set1 = self._tokenize(text1)
        set2 = self._tokenize(text2)

        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        # Jaccard 系数 = |交集| / |并集|
        intersection = set1 & set2
        union = set1 | set2

        return len(intersection) / len(union)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """将文本分词为集合

        对中文按字符拆分，对英文按空格拆分。
        统一转为小写，过滤空白和标点。

        Args:
            text: 输入文本

        Returns:
            词/字符集合
        """
        # 简单的混合分词策略：
        # 英文按空格分词，中文按字符分词
        tokens: set[str] = set()

        # 按空格分割（处理英文）
        for word in text.split():
            word = word.strip().lower()
            # 过滤纯标点
            cleaned = re.sub(r'[^\w]', '', word)
            if cleaned:
                tokens.add(cleaned)

        # 对中文字符按单字拆分
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # CJK 统一汉字范围
                tokens.add(char)

        return tokens

    def get_switch_history(self) -> list[StrategySwitchRecord]:
        """获取策略切换历史

        Returns:
            策略切换记录列表，按时间升序排列
        """
        return list(self._switch_history)

    def assess_switch_effectiveness(
        self,
        record_index: int,
        subsequent_results: list[PhaseResult],
    ) -> str:
        """评估某次策略切换的效果

        根据切换后的执行结果，判断策略切换是否有效。
        评估标准：
        - 切换后是否有成功的执行
        - 切换后是否有新的文件产出
        - 切换后是否不再出现连续失败

        Args:
            record_index: 切换记录的索引
            subsequent_results: 切换后的执行结果列表

        Returns:
            效果评估描述
        """
        if record_index < 0 or record_index >= len(self._switch_history):
            return "无效的记录索引"

        if not subsequent_results:
            effectiveness = "无法评估：切换后无执行结果"
            self._switch_history[record_index].effectiveness = effectiveness
            return effectiveness

        # 计算切换后的成功率
        success_count = sum(1 for pr in subsequent_results if pr.success)
        success_rate = success_count / len(subsequent_results)

        # 计算切换后的文件产出
        total_files = sum(
            len(pr.files_created) + len(pr.files_modified)
            for pr in subsequent_results
        )

        # 判断是否仍有连续失败
        consecutive_failures = 0
        max_consecutive = 0
        for pr in subsequent_results:
            if not pr.success:
                consecutive_failures += 1
                max_consecutive = max(max_consecutive, consecutive_failures)
            else:
                consecutive_failures = 0

        # 综合评估
        if success_rate >= 0.8 and total_files > 0:
            effectiveness = f"有效：成功率{success_rate:.0%}，产出{total_files}个文件"
        elif success_rate >= 0.5:
            effectiveness = f"部分有效：成功率{success_rate:.0%}，产出{total_files}个文件"
        elif max_consecutive >= self.stagnation_threshold:
            effectiveness = f"无效：切换后仍连续失败{max_consecutive}次"
        else:
            effectiveness = f"效果有限：成功率{success_rate:.0%}，产出{total_files}个文件"

        # 记录评估结果
        self._switch_history[record_index].effectiveness = effectiveness
        return effectiveness

    def _parse_strategy_response(
        self,
        response: str,
        current_strategy: str,
        reason: str,
    ) -> tuple[str, str]:
        """解析模型返回的策略切换响应

        尝试从 JSON 格式解析，失败时使用启发式方法提取。

        Args:
            response: 模型返回的原始响应文本
            current_strategy: 当前策略（用作回退）
            reason: 切换原因

        Returns:
            (new_strategy, risk_assessment) — 新策略和风险评估
        """
        if not response:
            return (
                f"基于停滞原因({reason})调整执行策略",
                "模型无响应，使用默认策略调整",
            )

        # 尝试 JSON 解析
        json_str = response.strip()

        # 提取 markdown 代码块中的内容
        if "```" in json_str:
            match = re.search(r"```(?:json)?\s*\n(.*?)\n```", json_str, re.DOTALL)
            if match:
                json_str = match.group(1).strip()

        # 找到 JSON 对象
        start = json_str.find("{")
        end = json_str.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = json_str[start:end + 1]
            try:
                data = json.loads(json_str)
                if isinstance(data, dict):
                    new_strategy = str(data.get("new_strategy", "")).strip()
                    risk = str(data.get("risk_assessment", "")).strip()

                    if new_strategy:
                        return new_strategy, risk or "风险评估未提供"
            except json.JSONDecodeError:
                pass

        # 启发式提取：查找"新策略"关键词
        strategy_match = re.search(
            r"新策略[：:]\s*(.+?)(?:\n|$)", response
        )
        if strategy_match:
            new_strategy = strategy_match.group(1).strip()
            risk = "启发式提取，风险未知"
            risk_match = re.search(
                r"风险[：:]\s*(.+?)(?:\n|$)", response
            )
            if risk_match:
                risk = risk_match.group(1).strip()
            return new_strategy, risk

        # 最终回退：使用模型响应的前200字作为新策略
        return (
            response[:200].strip() or f"基于停滞原因({reason})调整执行策略",
            "解析失败，使用模型原始响应",
        )
