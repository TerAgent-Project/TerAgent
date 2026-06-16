"""teragent.core.compilers.glm_52 — GLM52Compiler

GLM-5.2 专属编译器，继承 GLM5Compiler 并扩展：
  1. 1M 上下文智能分区：从"极限压缩"转为"智能分区"，
     不再需要 5:1 的激进压缩比，改为按重要性分层保留
  2. 双思考模式路由：High 模式用于常规任务（成本优化），
     Max 模式用于复杂 Coding（质量优先）
  3. 保留式思考（Preserved Thinking）：在 Coding Plan 场景中
     保留推理内容以提升缓存命中率和推理连续性
  4. 与 GLM-5V-Turbo 协同支持（视觉→编码跨模型工作流框架）

与 GLM5Compiler 的核心区别：
  - max_context_tokens: 200K → 1M
  - 思考模式: 单一 enabled/disabled → High/Max 双模式
  - 压缩策略: 5:1 极限压缩 → 1.2:1 智能分区
  - preserve_thinking: 无 → Coding Plan 场景默认开启
  - 上下文分区: 固定5区 → 动态5区 + 保留率追踪

协同工作流（GLM52VCoordinatedWorkflow）：
  通过 teragent.coordination.glm5v_coordinator 模块提供
  GLM-5V-Turbo + GLM-5.2 的跨模型协同工作流。
  - 视觉分析 → 上下文传递 → 编码执行 → 视觉验证
  - 支持顺序/并行/验证三种模式
  - 支持降级到纯文本模式

设计参考：design.md §6 GLM-5.2 深度适配方案
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from teragent.context.retention_tracker import LongContextRetentionTracker
from teragent.core.compiler import TAPCompiler, TAPCompilerRegistry
from teragent.core.compilers.glm_5 import GLM5Compiler
from teragent.core.tap import CompiledPrompt, TAPRequest

logger = logging.getLogger(__name__)


# ===== 1M 上下文智能分区配置 =====

@dataclass
class GLM52CompactionProfile:
    """GLM-5.2 专属 1M 上下文智能分区配置

    与 GLM5CompactionStrategy（200K 极限压缩）的核心区别：
    - 上下文窗口从 200K 扩展到 1M
    - 不再需要激进的压缩策略
    - 设计文档可完整保留（不再需要压缩为 ADR 摘要）
    - 执行历史可大面积保留

    分区策略（1M tokens）：
    - [0-50K]     系统提示 + 工具定义 + 设计文档(完整)     ← 完整保留
    - [50K-200K]  计划 + 架构决策记录(ADR)                 ← 完整保留
    - [200K-600K] 执行历史（关键步骤+结果）                ← 高度保留
    - [600K-900K] 执行历史（详细记录）                     ← 中度压缩
    - [900K-980K] 最新执行结果 + 错误信息 + review 记录    ← 完整保留
    - [980K-1M]   当前指令 + 约束 + 自评估提示             ← 尾部强化

    压缩比目标：1M+ 信息 → 1M 上下文
    压缩策略：1.2:1（仅需压缩 17%，远优于 GLM-5 的 5:1）
    关键优势：跨文档推理无需丢弃开头信息
    """

    # Token 预算分配
    system_budget: int = 51_200           # [0-50K] ≈ 50K tokens
    plan_budget: int = 153_600           # [50K-200K] ≈ 150K tokens
    execution_high_budget: int = 409_600 # [200K-600K] ≈ 400K tokens
    execution_mid_budget: int = 307_200  # [600K-900K] ≈ 300K tokens
    recent_budget: int = 81_920          # [900K-980K] ≈ 80K tokens
    tail_budget: int = 20_480            # [980K-1M] ≈ 20K tokens

    # 宽松压缩参数（与 GLM-5 形成鲜明对比）
    design_compression_target: float = 1.0      # 设计文档完整保留（1M 空间充裕）
    plan_compression_target: float = 1.0        # 计划完整保留
    execution_high_compression: float = 0.8     # 关键执行步骤保留 80%
    execution_mid_compression: float = 0.4      # 早期执行记录保留 40%
    min_information_retention: float = 0.95     # 压缩后信息保留率最低 95%

    @property
    def total_budget(self) -> int:
        """总 token 预算（应等于 1,024,000）"""
        return (
            self.system_budget
            + self.plan_budget
            + self.execution_high_budget
            + self.execution_mid_budget
            + self.recent_budget
            + self.tail_budget
        )


# ===== 双思考模式路由 =====

ThinkingLevel = Literal["high", "max"]


@dataclass
class ThinkingModeDecision:
    """GLM-5.2 思考模式路由决策结果

    Attributes:
        level: 选择的思考模式 — "high" 或 "max"
        reason: 选择该模式的原因（用于日志和调试）
        preserve_thinking: 是否保留推理内容（Preserved Thinking）
        cost_estimate: 估算的相对成本（1.0 = 基准无思考，4.0 = Max 模式）
    """
    level: ThinkingLevel = "high"
    reason: str = ""
    preserve_thinking: bool = False
    cost_estimate: float = 0.0


# Token 成本乘数（相对于基准无思考模式的成本）
_THINKING_COST_MULTIPLIERS = {
    "high": 1.5,  # High 模式使用约 1.5x tokens
    "max": 4.0,   # Max 模式使用约 4x tokens
}


class ThinkingModeRouter:
    """GLM-5.2 思考模式路由器

    根据任务特征自动选择 High 或 Max 模式：
    - High 模式：常规对话、简单编码、信息查询 → 快速响应 + 低成本
    - Max 模式：复杂 Coding、长程任务、多步推理 → 深度思考 + 高质量

    路由规则（按优先级）：
    1. 预算约束检查（budget_remaining）
    2. 长程任务强制 Max
    3. 多文件编辑（约束中提到多文件）强制 Max
    4. Debug/修复/重构类指令 → Max
    5. 简单查询/格式化 → High
    6. 根据上下文估算 token 比例动态调整
    7. 默认 High（成本优化）
    """

    # 触发 Max 模式的意图关键词
    _MAX_MODE_INTENTS = {"design", "plan", "review"}
    _MAX_MODE_KEYWORDS = {"debug", "fix", "refactor", "优化", "修复", "重构", "复杂"}

    # 触发 High 模式的意图关键词
    _HIGH_MODE_INTENTS = {"chat", "chat_friendly"}
    _HIGH_MODE_KEYWORDS = {"查询", "格式化", "简单", "列出", "query", "format", "simple"}

    def select(
        self,
        request: TAPRequest,
        budget_remaining: float | None = None,
    ) -> ThinkingModeDecision:
        """根据 TAP 请求自动选择思考模式（含成本感知）

        Args:
            request: TAP 请求对象
            budget_remaining: 剩余预算比例（0.0-1.0），None 表示无预算约束
                - < 0.05（5%）：强制 High 模式
                - < 0.20（20%）：优先 High 模式
                - >= 0.20 或 None：正常路由逻辑

        Returns:
            ThinkingModeDecision 包含模式选择、原因、preserve_thinking 和 cost_estimate
        """
        intent = request.meta.get("intent", "execute")
        instruction_lower = request.instruction.lower()

        # --- 成本感知路由 ---

        # 预算极度紧张（<5%）：强制 High 模式，无论任务多复杂
        if budget_remaining is not None and budget_remaining < 0.05:
            return ThinkingModeDecision(
                level="high",
                reason=f"预算极度紧张（剩余 {budget_remaining:.1%}），强制 High 模式",
                preserve_thinking=False,
                cost_estimate=_THINKING_COST_MULTIPLIERS["high"],
            )

        # 预算紧张（<20%）：偏好 High 模式，仅长程任务允许 Max
        if budget_remaining is not None and budget_remaining < 0.20:
            # 长程任务即使预算紧张也用 Max（否则任务可能失败）
            if request.is_long_horizon:
                return ThinkingModeDecision(
                    level="max",
                    reason=f"预算紧张（剩余 {budget_remaining:.1%}）但长程任务必须 Max",
                    preserve_thinking=True,
                    cost_estimate=_THINKING_COST_MULTIPLIERS["max"],
                )
            # 非长程任务在预算紧张时偏好 High
            for keyword in self._MAX_MODE_KEYWORDS:
                if keyword in instruction_lower:
                    return ThinkingModeDecision(
                        level="high",
                        reason=(
                            f"预算紧张（剩余 {budget_remaining:.1%}），"
                            f"指令含 '{keyword}' 但降级为 High 节省成本"
                        ),
                        preserve_thinking=False,
                        cost_estimate=_THINKING_COST_MULTIPLIERS["high"],
                    )
            # 其他情况默认 High
            return ThinkingModeDecision(
                level="high",
                reason=f"预算紧张（剩余 {budget_remaining:.1%}），默认 High 模式",
                preserve_thinking=False,
                cost_estimate=_THINKING_COST_MULTIPLIERS["high"],
            )

        # --- 正常路由逻辑（预算充足或无预算约束） ---

        # 规则 1：长程任务强制 Max + preserve_thinking
        if request.is_long_horizon:
            return ThinkingModeDecision(
                level="max",
                reason="长程任务强制 Max 模式 + 保留推理内容",
                preserve_thinking=True,
                cost_estimate=_THINKING_COST_MULTIPLIERS["max"],
            )

        # 规则 2：Coding Plan 场景（plan 意图 + 大量上下文）→ Max + preserve
        if intent == "plan" and request.context.get("design"):
            return ThinkingModeDecision(
                level="max",
                reason="Coding Plan 场景：Max 模式 + 保留推理内容提升缓存命中",
                preserve_thinking=True,
                cost_estimate=_THINKING_COST_MULTIPLIERS["max"],
            )

        # 规则 3：Debug/修复/重构类指令 → Max
        for keyword in self._MAX_MODE_KEYWORDS:
            if keyword in instruction_lower:
                return ThinkingModeDecision(
                    level="max",
                    reason=f"指令包含复杂任务关键词 '{keyword}'，使用 Max 模式",
                    preserve_thinking=False,
                    cost_estimate=_THINKING_COST_MULTIPLIERS["max"],
                )

        # 规则 4：设计/审查意图 → Max（需要深度推理）
        if intent in self._MAX_MODE_INTENTS:
            return ThinkingModeDecision(
                level="max",
                reason=f"意图 '{intent}' 需要深度推理，使用 Max 模式",
                preserve_thinking=False,
                cost_estimate=_THINKING_COST_MULTIPLIERS["max"],
            )

        # 规则 5：简单查询/格式化 → High
        for keyword in self._HIGH_MODE_KEYWORDS:
            if keyword in instruction_lower:
                return ThinkingModeDecision(
                    level="high",
                    reason=f"指令包含简单任务关键词 '{keyword}'，使用 High 模式",
                    preserve_thinking=False,
                    cost_estimate=_THINKING_COST_MULTIPLIERS["high"],
                )

        # 规则 6：聊天意图 → High
        if intent in self._HIGH_MODE_INTENTS:
            return ThinkingModeDecision(
                level="high",
                reason=f"意图 '{intent}' 适合 High 模式",
                preserve_thinking=False,
                cost_estimate=_THINKING_COST_MULTIPLIERS["high"],
            )

        # 规则 7：执行意图 — 根据上下文大小动态判断
        if intent in ("execute", "code_generation"):
            estimated_tokens = request.estimate_prompt_tokens()
            if estimated_tokens > 100_000:
                return ThinkingModeDecision(
                    level="max",
                    reason=f"上下文估算 {estimated_tokens} tokens > 100K，使用 Max 模式",
                    preserve_thinking=True,
                    cost_estimate=_THINKING_COST_MULTIPLIERS["max"],
                )

        # 默认：High 模式（成本优化）
        return ThinkingModeDecision(
            level="high",
            reason="默认 High 模式（成本优化）",
            preserve_thinking=False,
            cost_estimate=_THINKING_COST_MULTIPLIERS["high"],
        )


# ===== 模式切换记录 =====

@dataclass
class ModeSwitchRecord:
    """模式切换记录

    Attributes:
        from_mode: 切换前的思考模式
        to_mode: 切换后的思考模式
        reason: 切换原因
        step: 切换发生的步骤编号
        timestamp: 切换时间戳
    """
    from_mode: ThinkingLevel
    to_mode: ThinkingLevel
    reason: str
    step: int = 0
    timestamp: float = field(default_factory=time.time)


# ===== Dynamic Thinking Mode Manager =====

class DynamicThinkingModeManager:
    """管理长程任务中的动态思考模式切换

    在长程任务中，子任务的复杂度各不相同：
    - 简单子任务（文件读取、格式化）→ High 模式（节省成本）
    - 复杂子任务（调试、多文件重构）→ Max 模式（保证质量）

    该管理器跟踪当前模式并提供切换逻辑。
    """

    def __init__(self, router: ThinkingModeRouter) -> None:
        self._router = router
        self._current_mode: ThinkingLevel = "high"
        self._mode_history: list[ModeSwitchRecord] = []
        self._max_switches_per_task: int = 10  # 防止模式振荡

    def should_switch(self, request: TAPRequest, current_step: int) -> ThinkingModeDecision:
        """判断是否应切换思考模式

        根据当前子任务的复杂度决定是否切换模式。
        切换决策基于 ThinkingModeRouter 的路由逻辑，
        但受最大切换次数限制以防振荡。

        Args:
            request: 当前 TAP 请求
            current_step: 当前步骤编号

        Returns:
            ThinkingModeDecision，可能包含不同的 level
        """
        # 已达到最大切换次数 → 保持当前模式
        if self.switch_count >= self._max_switches_per_task:
            return ThinkingModeDecision(
                level=self._current_mode,
                reason=f"已达最大切换次数 {self._max_switches_per_task}，保持当前模式",
                preserve_thinking=self._current_mode == "max",
                cost_estimate=_THINKING_COST_MULTIPLIERS.get(self._current_mode, 1.5),
            )

        # 通过路由器获取建议模式
        suggested = self._router.select(request)

        # 如果建议的模式与当前模式相同，无需切换
        if suggested.level == self._current_mode:
            return suggested

        # 检查是否在最近步骤内已切换过（防止快速振荡）
        if self._mode_history:
            last_switch = self._mode_history[-1]
            steps_since_last = current_step - last_switch.step
            if steps_since_last <= 1:
                # 上一步刚切换，这一步不切换
                return ThinkingModeDecision(
                    level=self._current_mode,
                    reason=f"上一步刚切换（步骤 {last_switch.step}），防止振荡保持当前模式",
                    preserve_thinking=self._current_mode == "max",
                    cost_estimate=_THINKING_COST_MULTIPLIERS.get(self._current_mode, 1.5),
                )

        # 允许切换
        return suggested

    def apply_switch(self, decision: ThinkingModeDecision, current_step: int) -> None:
        """应用模式切换

        在实际切换模式后调用此方法记录历史。
        如果 decision.level 与当前模式不同，记录一次切换。

        Args:
            decision: 路由决策结果
            current_step: 当前步骤编号
        """
        if decision.level != self._current_mode:
            self.record_switch(
                from_mode=self._current_mode,
                to_mode=decision.level,
                reason=decision.reason,
                step=current_step,
            )
            self._current_mode = decision.level

    def record_switch(self, from_mode: ThinkingLevel, to_mode: ThinkingLevel, reason: str, step: int = 0) -> None:
        """记录模式切换

        Args:
            from_mode: 切换前的模式
            to_mode: 切换后的模式
            reason: 切换原因
            step: 切换发生的步骤编号
        """
        record = ModeSwitchRecord(
            from_mode=from_mode,
            to_mode=to_mode,
            reason=reason,
            step=step,
        )
        self._mode_history.append(record)
        logger.debug(
            f"DynamicThinkingModeManager: {from_mode} → {to_mode} "
            f"at step {step} ({reason})"
        )

    @property
    def current_mode(self) -> ThinkingLevel:
        """当前思考模式"""
        return self._current_mode

    @property
    def switch_count(self) -> int:
        """当前任务中的模式切换次数"""
        return len(self._mode_history)

    @property
    def mode_history(self) -> list[ModeSwitchRecord]:
        """模式切换历史（只读）"""
        return list(self._mode_history)

    def reset(self) -> None:
        """重置为新任务"""
        self._current_mode = "high"
        self._mode_history.clear()


# ===== Preserved Thinking Manager =====

class PreservedThinkingManager:
    """保留式思考管理器

    GLM-5.2 独有的 Preserved Thinking 功能：
    1. 在多轮对话中保留 reasoning_content，提升推理连续性
    2. 提高缓存命中率，在真实任务中节省 tokens
    3. 在 Coding Plan 端点默认开启，标准 API 端点默认关闭

    注意：
    - reasoning_content 必须完整、未修改地传回 API
    - 连续的 reasoning_content 必须与模型原始生成的序列完全一致
    - 重新排序或修改会降低效果并影响缓存命中

    Phase 2 实现：完整的多轮 reasoning_content 持久化。
    """

    # 多步推理阈值：超过此步骤数时自动开启 preserve_thinking
    MULTI_STEP_THRESHOLD = 3

    def __init__(self) -> None:
        self._reasoning_history: list[str] = []  # 每轮的 reasoning_content
        self._preserve_enabled: bool = False

    def should_preserve(self, request: TAPRequest, thinking_decision: ThinkingModeDecision) -> bool:
        """判断是否应保留推理内容

        判断逻辑：
        1. ThinkingModeRouter 已经决定 preserve → 直接返回
        2. 长程任务 → 开启
        3. 多步执行（meta 中有 step_count） → 开启
        4. Coding Plan 场景 → 开启
        5. 简单对话 → 关闭

        Args:
            request: TAP 请求
            thinking_decision: 思考模式路由决策

        Returns:
            是否应保留推理内容
        """
        # 1. 路由器已决定
        if thinking_decision.preserve_thinking:
            return True

        # 2. 长程任务
        if request.is_long_horizon:
            return True

        # 3. 多步执行
        step_count = request.meta.get("step_count", 0)
        if isinstance(step_count, int) and step_count >= self.MULTI_STEP_THRESHOLD:
            return True

        # 4. Coding Plan 场景（plan 意图 + 有设计文档）
        intent = request.meta.get("intent", "")
        if intent == "plan" and request.context.get("design"):
            return True

        return False

    # --- reasoning_content 多轮持久化 ---

    def record_reasoning(self, reasoning_content: str) -> None:
        """记录模型响应中的推理内容

        内容必须逐字存储，不做任何修改。

        Args:
            reasoning_content: 模型返回的 reasoning_content 原始文本
        """
        if reasoning_content:
            self._reasoning_history.append(reasoning_content)
            logger.debug(
                f"PreservedThinkingManager: 记录推理内容 "
                f"(第 {len(self._reasoning_history)} 轮, "
                f"长度={len(reasoning_content)})"
            )

    def get_preserved_reasoning_messages(self) -> list[dict]:
        """构建包含 reasoning_content 的消息用于下一轮请求

        返回带有 reasoning_content 的 assistant 消息列表，
        使模型可以继续之前的推理链。

        Returns:
            消息字典列表，每条包含 role、reasoning_content、content
        """
        messages = []
        for reasoning in self._reasoning_history:
            # GLM-5.2 格式：assistant 消息带 reasoning_content
            messages.append({
                "role": "assistant",
                "reasoning_content": reasoning,
                "content": "",  # 内容在原始响应中
            })
        return messages

    def inject_preserved_reasoning(self, compiled: CompiledPrompt) -> None:
        """将保留的推理内容注入编译后的 prompt

        在当前用户消息之前插入之前的 reasoning_content 作为 assistant 消息，
        维持推理链的连续性。

        Args:
            compiled: 已编译的 CompiledPrompt 对象
        """
        if not self._reasoning_history:
            return

        preserved_messages = self.get_preserved_reasoning_messages()
        # 在最后一个 user 消息之前插入
        last_user_idx = -1
        for i, msg in enumerate(compiled.messages):
            if msg.get("role") == "user":
                last_user_idx = i

        if last_user_idx >= 0:
            for j, pm in enumerate(preserved_messages):
                compiled.messages.insert(last_user_idx + j, pm)
            logger.debug(
                f"PreservedThinkingManager: 注入 {len(preserved_messages)} 条 "
                f"保留推理消息 (插入位置={last_user_idx})"
            )

    @property
    def reasoning_count(self) -> int:
        """已保留的推理轮数"""
        return len(self._reasoning_history)

    def clear(self) -> None:
        """清除所有保留的推理内容（新对话）"""
        self._reasoning_history.clear()
        self._preserve_enabled = False


# ===== GLM52Compiler 主类 =====

class GLM52Compiler(GLM5Compiler):
    """GLM-5.2 专属 TAP 编译器

    继承 GLM5Compiler 的核心能力：
    - Recency Effect（关键指令放最后）
    - 长程任务引导（实验→分析→优化闭环）
    - 自评估注入
    - 策略切换引导

    新增/覆盖能力：
    1. 1M 上下文智能分区（覆盖 200K 极限压缩）
    2. High/Max 双思考模式（覆盖单一 enabled/disabled）
    3. 保留式思考 Preserved Thinking
    4. 1M 上下文 prompt 模板（告知模型无需压缩）

    Returns CompiledPrompt in Mode A (messages list).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._thinking_mode_router = ThinkingModeRouter()
        self._preserved_thinking_manager = PreservedThinkingManager()
        self._dynamic_mode_manager = DynamicThinkingModeManager(self._thinking_mode_router)
        self._compaction_profile = GLM52CompactionProfile()

        # P2-9: 1M 上下文保留率追踪 + 自动降级
        self._retention_tracker = LongContextRetentionTracker(
            max_context_tokens=self.max_context_tokens
        )
        self._is_downgraded_to_200k = False

    # ----- Capability overrides -----

    @property
    def max_context_tokens(self) -> int:
        """GLM-5.2 支持 1M tokens 上下文"""
        return 1_000_000

    def _get_compiler_type(self) -> str:
        """Compiler type for prompt registry lookup"""
        return "glm_52"

    # ----- Context building (override for 1M smart partitioning) -----

    def _build_glm_context(self, request: TAPRequest) -> str:
        """构建 GLM-5.2 风格的 1M 上下文

        与 GLM5Compiler._build_glm_context 的区别：
        - 设计文档完整保留（不再压缩为 ADR）
        - 计划完整保留
        - 执行历史大面积保留（近期完整，早期适度压缩）
        - 尾部强化区增加 1M 上下文利用提示

        对于 Phase 1（基础版），实现上下文构建逻辑。
        实际的动态分区压缩由 Phase 2 的 AutoCompactor 实现。
        """
        parts: list[str] = []

        # 1. 设计文档 — 完整保留（1M 空间充裕）
        design = request.context.get("design", "N/A")
        if design and design != "N/A":
            # GLM-5.2 的 1M 上下文下，设计文档不再需要压缩
            parts.append(f"<design>\n{design}\n</design>")

        # 2. 执行计划 — 完整保留
        plan = request.context.get("plan", "N/A")
        if plan and plan != "N/A":
            parts.append(f"<plan>\n{plan}\n</plan>")

        # 3. 依赖报告
        dep = request.context.get("dependency_report", "N/A")
        if dep and dep not in ["N/A", ""]:
            parts.append(f"<dependency_report>\n{dep}\n</dependency_report>")

        # 4. 项目记忆
        memory = request.context.get("memory", "N/A")
        if memory and memory != "N/A":
            parts.append(f"<memory>\n{memory}\n</memory>")

        return "\n".join(parts)

    # ----- Thinking mode (override for High/Max dual mode) -----

    def _apply_thinking_mode(self, compiled: CompiledPrompt, request: TAPRequest) -> None:
        """将 thinking_mode 转换为 GLM-5.2 High/Max API 参数

        与 GLM5Compiler 的区别：
        - GLM-5: thinking={"type": "enabled"/"disabled"}
        - GLM-5.2: thinking={"type": "enabled", "level": "high"/"max"}
                   + preserve_thinking 参数

        路由逻辑：
        1. thinking_mode="deep" → Max 模式
        2. thinking_mode="quick" → High 模式
        3. thinking_mode="auto" → ThinkingModeRouter 自动选择
        4. thinking_mode=None → auto 行为
        5. 长程任务 → DynamicThinkingModeManager 动态切换
        """
        explicit_mode = request.effective_thinking_mode

        if explicit_mode == "deep":
            # 用户明确要求深度推理 → Max 模式
            decision = ThinkingModeDecision(
                level="max",
                reason="用户指定 deep 模式 → Max",
                preserve_thinking=True,
                cost_estimate=_THINKING_COST_MULTIPLIERS["max"],
            )
        elif explicit_mode == "quick":
            # 用户明确要求快速响应 → High 模式
            decision = ThinkingModeDecision(
                level="high",
                reason="用户指定 quick 模式 → High",
                preserve_thinking=False,
                cost_estimate=_THINKING_COST_MULTIPLIERS["high"],
            )
        else:
            # auto / None → 动态路由
            if request.is_long_horizon:
                # 长程任务使用 DynamicThinkingModeManager 动态切换
                current_step = request.meta.get("step", 0)
                if isinstance(current_step, str):
                    current_step = int(current_step) if current_step.isdigit() else 0
                decision = self._dynamic_mode_manager.should_switch(request, current_step)
                self._dynamic_mode_manager.apply_switch(decision, current_step)
            else:
                # 非长程任务直接使用 ThinkingModeRouter
                decision = self._thinking_mode_router.select(request)

        # 设置 API 参数
        # GLM-5.2 思考模式 API 参数格式：
        # thinking={"type": "enabled", "level": "high"/"max"}
        compiled.extra["thinking"] = {
            "type": "enabled",
            "level": decision.level,
        }

        # 成本估算
        if decision.cost_estimate > 0:
            compiled.extra["cost_estimate"] = decision.cost_estimate

        # Preserved Thinking 参数
        should_preserve = self._preserved_thinking_manager.should_preserve(
            request, decision
        )
        if should_preserve:
            compiled.extra["preserve_thinking"] = True
            # 注入之前保留的 reasoning_content
            self._preserved_thinking_manager.inject_preserved_reasoning(compiled)

        logger.debug(
            f"GLM52Compiler: thinking_mode={explicit_mode} → "
            f"level={decision.level}, preserve={should_preserve}, "
            f"cost_estimate={decision.cost_estimate}, "
            f"reason={decision.reason}"
        )

    # ----- Tail reinforcement (override for 1M context) -----

    def _build_tail_reinforcement(self, request: TAPRequest) -> str:
        """构建 GLM-5.2 专属尾部强化内容

        与 GLM5Compiler 的区别：
        - 增大尾部预算（20K → 20K，但整体上下文 1M，比例更小）
        - 增加 1M 上下文利用提示
        - 增加 preserve_thinking 状态提示
        - 增加跨文档推理引导

        Args:
            request: TAP 请求对象

        Returns:
            尾部强化文本
        """
        profile = self._compaction_profile
        max_chars = int(profile.tail_budget * 2.5)

        parts: list[str] = []

        # 1. 1M 上下文利用提示（GLM-5.2 独有）
        parts.append("【1M 上下文模式】")
        parts.append("你有 1M tokens 的上下文空间，可以完整保留所有代码和文档，无需压缩。")
        parts.append("注意不同文件/文档之间的交叉引用和一致性。")

        # 2. 当前指令重申
        if request.instruction:
            instruction_preview = request.instruction
            if len(instruction_preview) > 300:
                instruction_preview = instruction_preview[:300] + "..."
            parts.append("")
            parts.append("【当前指令重申】")
            parts.append(instruction_preview)

        # 3. 约束条件重申
        if request.constraints:
            parts.append("")
            parts.append("【关键约束】")
            for i, constraint in enumerate(request.constraints[:5], 1):
                c = constraint if len(constraint) <= 200 else constraint[:200] + "..."
                parts.append(f"  {i}. {c}")

        # 4. 自评估提示
        parts.append("")
        parts.append("【自评估检查】")
        parts.append("完成输出前，请确认：")
        parts.append("  - 是否满足所有约束条件？")
        parts.append("  - 输出格式是否符合要求？")
        parts.append("  - 是否遗漏了指令中的关键要求？")
        parts.append("  - 跨文件引用和一致性是否正确？")

        result = "\n".join(parts)

        # 如果超出预算，截断
        if len(result) > max_chars:
            result = result[:max_chars] + "\n... [尾部强化截断]"

        return result

    # ----- Long-horizon system addition (override for 1M context) -----

    def _build_long_horizon_system_addition(self, request: TAPRequest) -> str:
        """构建 GLM-5.2 长程任务的系统提示追加内容

        与 GLM5Compiler 的区别：
        - 增加 1M 上下文利用提示
        - 增加 preserve_thinking 说明
        - 更强调跨文档推理能力
        """
        base_addition = super()._build_long_horizon_system_addition(request)

        if not base_addition:
            return ""

        # 追加 1M 上下文特有提示
        glm52_additions = [
            "",
            "【1M 上下文优势】",
            "你有 1M tokens 的上下文空间，无需压缩历史信息。",
            "可以跨文件追踪代码修改的影响，完整保留所有设计决策。",
        ]

        return base_addition + "\n".join(glm52_additions)

    # ----- 1M Context Partition Info -----

    def get_context_partition_info(self) -> dict:
        """返回当前上下文分区配置信息

        用于调试和日志，展示 1M 上下文的分区分配。

        Returns:
            分区信息字典
        """
        profile = self._compaction_profile
        return {
            "model": "GLM-5.2",
            "max_context_tokens": self.max_context_tokens,
            "is_downgraded": self._is_downgraded_to_200k,
            "partitions": {
                "system": {"budget": profile.system_budget, "desc": "系统提示+工具定义+设计文档"},
                "plan": {"budget": profile.plan_budget, "desc": "计划+ADR"},
                "execution_high": {"budget": profile.execution_high_budget, "desc": "关键执行步骤"},
                "execution_mid": {"budget": profile.execution_mid_budget, "desc": "详细执行记录"},
                "recent": {"budget": profile.recent_budget, "desc": "最新结果+错误+review"},
                "tail": {"budget": profile.tail_budget, "desc": "当前指令+约束+自评估"},
            },
            "total_budget": profile.total_budget,
            "compression_ratio": "1.2:1",
            "design_retention": "100%",
            "retention_tracker": self._retention_tracker.get_stats(),
        }

    # ----- P2-9: 保留率追踪 + 自动降级 -----

    def check_downgrade_before_build(self) -> bool:
        """编译前检查是否应从 1M 降级到 200K 模式

        当 LongContextRetentionTracker.should_downgrade() 返回 True 时，
        自动切换到 GLM5CompactionStrategy 参数，并记录降级原因。

        Returns:
            是否已降级
        """
        if self._retention_tracker.should_downgrade() and not self._is_downgraded_to_200k:
            reason = self._retention_tracker.get_downgrade_reason()
            self._is_downgraded_to_200k = True
            logger.warning(
                f"GLM52Compiler 自动降级到 200K 模式: {reason}"
            )
        return self._is_downgraded_to_200k

    def record_partition_usage_after_compilation(
        self,
        zone: str,
        tokens_provided: int,
        response_text: str = "",
        zone_content: str = "",
    ) -> None:
        """编译后记录分区使用情况

        在每次编译完成后调用，更新保留率追踪器。
        如果提供了 response_text 和 zone_content，使用基于规则的
        方法估算引用 token 数；否则使用 tokens_provided 作为近似。

        Args:
            zone: 分区名称
            tokens_provided: 提供的 token 数
            response_text: 模型响应文本（可选，用于估算引用率）
            zone_content: 分区内容文本（可选，用于估算引用率）
        """
        if response_text and zone_content:
            tokens_referenced = self._retention_tracker.estimate_referenced_tokens(
                tokens_provided, response_text, zone_content
            )
        else:
            # 无响应文本时，假设 80% 引用率（保守估计）
            tokens_referenced = int(tokens_provided * 0.8)

        self._retention_tracker.record_partition(zone, tokens_provided, tokens_referenced)

        # 每次记录后检查降级
        self.check_downgrade_before_build()

    def get_retention_optimization_suggestions(self) -> list[str]:
        """获取基于保留率追踪的分区优化建议

        Returns:
            优化建议列表
        """
        return self._retention_tracker.get_optimization_suggestions()

    def reset_downgrade(self) -> None:
        """重置降级状态，恢复 1M 模式"""
        self._is_downgraded_to_200k = False
        self._retention_tracker.reset_downgrade()
        logger.info("GLM52Compiler 降级状态已重置，恢复 1M 模式")

    # ----- P2-11: 与 GLM-5V-Turbo 协同集成 (deprecated — migrated to orchestration) -----

    def create_coordinated_workflow(
        self,
        vision_provider: "ModelProvider | None" = None,
        coding_provider: "ModelProvider | None" = None,
        config: "Any | None" = None,
    ) -> "Any":
        """创建 GLM-5V-Turbo + GLM-5.2 协同工作流

        .. deprecated::
            teragent.coordination is removed. Use teragent.orchestration
            for multi-agent coordination instead.

        Raises:
            NotImplementedError: Always — coordination has been migrated to orchestration.
        """
        raise NotImplementedError(
            "GLM52Compiler.create_coordinated_workflow() is deprecated. "
            "Use teragent.orchestration.Orchestrator for multi-agent coordination."
        )

    def compile_with_visual_context(
        self,
        request: TAPRequest,
        visual_analysis: str,
    ) -> CompiledPrompt:
        """编译带视觉分析上下文的请求

        当 GLM-5V-Turbo 完成视觉分析后，调用此方法将分析结果
        注入到 GLM-5.2 的编译上下文中。

        Args:
            request: 原始 TAP 请求
            visual_analysis: GLM-5V-Turbo 的视觉分析文本

        Returns:
            包含视觉上下文的 CompiledPrompt
        """
        # 将视觉分析注入到请求的 context 中
        enhanced_context = dict(request.context)
        enhanced_context["visual_analysis"] = visual_analysis

        # 创建增强请求
        enhanced_request = TAPRequest(
            meta=request.meta,
            context=enhanced_context,
            instruction=request.instruction,
            constraints=request.constraints,
            output_format_hint=request.output_format_hint,
            thinking_mode=request.thinking_mode,
            long_horizon=request.long_horizon,
            cache_preference=request.cache_preference,
        )

        # 使用标准编译流程
        compiled = self.compile(enhanced_request)

        # 在 extra 中标记使用了视觉协同
        compiled.extra["visual_coordination"] = True
        compiled.extra["visual_analysis_length"] = len(visual_analysis)

        return compiled


# Register compiler
TAPCompilerRegistry.register("glm_52", GLM52Compiler)
