"""teragent.coordination.glm5v_coordinator — GLM-5V-Turbo + GLM-5.2 协同工作流

实现 GLM-5V-Turbo 视觉理解与 GLM-5.2 编码执行的跨模型协同工作流。

核心流程：
  1. GLM-5V-Turbo：理解图像/设计稿，提取视觉语义
  2. 语义信息传入 GLM-5.2：基于视觉理解进行编码执行
  3. GLM-5V-Turbo：验证编码结果与设计稿的一致性（可选）

关键挑战：
  - 两模型间的上下文共享机制
  - 视觉语义到代码逻辑的对齐
  - 协同工作流的错误恢复

设计参考：design.md §6.2.5 与 GLM-5V-Turbo 多模态协同
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "CoordinationConfig",
    "CoordinationMode",
    "CoordinationPhase",
    "CoordinationResult",
    "CoordinationStep",
    "GLM52VCoordinatedWorkflow",
]

from teragent.core.compiler import TAPCompiler, TAPCompilerRegistry
from teragent.core.compilers.glm_5v_turbo import (
    GLM5VTurboCompiler,
    VisionAnalysisResult,
    VISION_CODE_GENERATION_HINT,
)
from teragent.core.tap import CompiledPrompt, MultimodalContent, TAPRequest, TAPResponse
from teragent.core.provider import ModelProvider

logger = logging.getLogger(__name__)


# ===== Enums =====


class CoordinationMode(Enum):
    """协同工作流模式"""

    SEQUENTIAL = "sequential"  # 顺序：视觉→编码（默认）
    PARALLEL = "parallel"  # 并行：视觉+编码同时启动（实验性）
    VERIFY = "verify"  # 验证：视觉→编码→视觉验证


class CoordinationPhase(Enum):
    """协同工作流的阶段"""

    PENDING = "pending"
    VISION_ANALYSIS = "vision_analysis"  # 5V-Turbo 分析图像
    CONTEXT_TRANSFER = "context_transfer"  # 上下文传递
    CODE_GENERATION = "code_generation"  # GLM-5.2 生成代码
    VISUAL_VERIFICATION = "visual_verification"  # 5V-Turbo 验证
    COMPLETED = "completed"
    FAILED = "failed"
    DEGRADED = "degraded"  # 降级为纯文本模式


# ===== Coordination Config =====


@dataclass
class CoordinationConfig:
    """协同工作流配置

    Attributes:
        enabled: 是否启用协同工作流
        mode: 协同模式（sequential/parallel/verify）
        vision_model: 视觉模型名称
        coding_model: 编码模型名称
        vision_compiler: 视觉编译器名称
        coding_compiler: 编码编译器名称
        context_sharing: 是否在两模型间共享上下文
        max_verification_rounds: 最大验证轮数（0 = 不验证）
        verification_score_threshold: 验证通过的分数阈值（0-10）
        degrade_on_vision_failure: 视觉模型失败时是否降级为纯文本
        inject_code_generation_hint: 是否在编码时注入视觉→编码提示
    """

    enabled: bool = True
    mode: str = "sequential"
    vision_model: str = "glm-5v-turbo"
    coding_model: str = "glm-5.2"
    vision_compiler: str = "glm_5v_turbo"
    coding_compiler: str = "glm_52"
    context_sharing: bool = True
    max_verification_rounds: int = 1
    verification_score_threshold: float = 7.0
    degrade_on_vision_failure: bool = True
    inject_code_generation_hint: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoordinationConfig:
        """从配置字典创建 CoordinationConfig

        Args:
            data: 配置字典，通常来自 agent.toml

        Returns:
            CoordinationConfig 实例
        """
        return cls(
            enabled=data.get("enabled", True),
            mode=data.get("coordination_mode", "sequential"),
            vision_model=data.get("vision_model", "glm-5v-turbo"),
            coding_model=data.get("coding_model", "glm-5.2"),
            vision_compiler=data.get("vision_compiler", "glm_5v_turbo"),
            coding_compiler=data.get("coding_compiler", "glm_52"),
            context_sharing=data.get("context_sharing", True),
            max_verification_rounds=data.get("max_verification_rounds", 1),
            verification_score_threshold=data.get("verification_score_threshold", 7.0),
            degrade_on_vision_failure=data.get("degrade_on_vision_failure", True),
            inject_code_generation_hint=data.get("inject_code_generation_hint", True),
        )


# ===== Coordination Step Record =====


@dataclass
class CoordinationStep:
    """协同工作流单步记录

    Attributes:
        phase: 当前阶段
        model: 使用的模型
        start_time: 开始时间戳
        end_time: 结束时间戳
        success: 是否成功
        token_usage: token 用量
        error: 错误信息
    """

    phase: str = ""
    model: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    success: bool = True
    token_usage: dict = field(default_factory=dict)
    error: str = ""

    @property
    def duration_ms(self) -> float:
        """耗时（毫秒）"""
        if self.end_time > 0 and self.start_time > 0:
            return (self.end_time - self.start_time) * 1000
        return 0.0


# ===== Coordination Result =====


@dataclass
class CoordinationResult:
    """协同工作流结果

    Attributes:
        final_response: 最终的 TAPResponse（来自 GLM-5.2 编码步骤）
        vision_analysis: 视觉分析结果
        verification_result: 验证结果（如有）
        steps: 所有步骤记录
        phase: 最终阶段
        is_degraded: 是否降级为纯文本模式
        total_tokens: 总 token 用量
        total_duration_ms: 总耗时（毫秒）
    """

    final_response: TAPResponse | None = None
    vision_analysis: VisionAnalysisResult | None = None
    verification_result: TAPResponse | None = None
    steps: list[CoordinationStep] = field(default_factory=list)
    phase: str = CoordinationPhase.PENDING.value
    is_degraded: bool = False
    total_tokens: dict = field(default_factory=dict)
    total_duration_ms: float = 0.0
    error: str | None = None  # 错误信息（如视觉模型不可用等）

    @property
    def success(self) -> bool:
        """工作流是否成功完成"""
        return self.phase in (
            CoordinationPhase.COMPLETED.value,
            CoordinationPhase.DEGRADED.value,
        )

    @property
    def verification_score(self) -> float | None:
        """验证分数（如果进行了验证）"""
        if self.verification_result and self.verification_result.raw_text:
            return _extract_verification_score(self.verification_result.raw_text)
        return None


# ===== Main Workflow Class =====


class GLM52VCoordinatedWorkflow:
    """GLM-5.2 + GLM-5V-Turbo 协同工作流

    场景：设计稿→代码生成、UI 审查、视觉 Debug

    流程：
    1. GLM-5V-Turbo：理解图像/视频/设计稿，提取视觉语义
    2. 语义信息传入 GLM-5.2：基于视觉理解进行编码执行
    3. GLM-5V-Turbo：验证编码结果与设计稿的一致性（可选）

    降级策略：
    - 当 5V-Turbo 不可用时，降级为纯文本模式
    - 纯文本模式：将多模态内容降级为文本描述，仅使用 GLM-5.2

    使用示例::

        workflow = GLM52VCoordinatedWorkflow(
            vision_provider=vision_provider,
            coding_provider=coding_provider,
            config=CoordinationConfig(mode="verify"),
        )
        result = await workflow.execute(tap_request)
        if result.success:
            print(result.final_response.raw_text)
    """

    def __init__(
        self,
        vision_provider: ModelProvider | None = None,
        coding_provider: ModelProvider | None = None,
        config: CoordinationConfig | None = None,
    ) -> None:
        """初始化协同工作流

        Args:
            vision_provider: GLM-5V-Turbo 的 ModelProvider
            coding_provider: GLM-5.2 的 ModelProvider
            config: 协同配置
        """
        self._vision_provider = vision_provider
        self._coding_provider = coding_provider
        self._config = config or CoordinationConfig()
        self._steps: list[CoordinationStep] = []

        # 如果未提供 provider，尝试从编译器创建
        if self._vision_provider is None and self._config.vision_compiler:
            self._vision_provider = self._create_provider_from_compiler(
                self._config.vision_compiler,
                self._config.vision_model,
            )
        if self._coding_provider is None and self._config.coding_compiler:
            self._coding_provider = self._create_provider_from_compiler(
                self._config.coding_compiler,
                self._config.coding_model,
            )

    @property
    def config(self) -> CoordinationConfig:
        """协同配置"""
        return self._config

    @property
    def is_available(self) -> bool:
        """协同工作流是否可用（需要两个 provider 都存在）"""
        return (
            self._vision_provider is not None
            and self._coding_provider is not None
            and self._config.enabled
        )

    # ----- Main execute -----

    async def execute(self, request: TAPRequest) -> CoordinationResult:
        """执行协同工作流

        Args:
            request: 包含多模态内容的 TAP 请求

        Returns:
            CoordinationResult 包含最终结果和过程信息
        """
        result = CoordinationResult()
        start_time = time.time()

        # 检查可用性
        if not self.is_available:
            if self._config.degrade_on_vision_failure:
                logger.warning(
                    "GLM52VCoordinatedWorkflow: 视觉模型不可用，降级为纯文本模式"
                )
                return await self._execute_degraded(request, start_time)
            else:
                result.phase = CoordinationPhase.FAILED.value
                result.error = "视觉模型不可用且未启用降级"
                return result

        # 检查请求是否包含多模态内容
        if not request.has_multimodal:
            # 没有多模态内容，直接使用编码模型
            logger.debug(
                "GLM52VCoordinatedWorkflow: 请求无多模态内容，直接使用编码模型"
            )
            return await self._execute_coding_only(request, start_time)

        # 根据模式执行
        mode = CoordinationMode(self._config.mode)
        if mode == CoordinationMode.SEQUENTIAL:
            return await self._execute_sequential(request, result, start_time)
        elif mode == CoordinationMode.VERIFY:
            return await self._execute_with_verification(request, result, start_time)
        elif mode == CoordinationMode.PARALLEL:
            return await self._execute_parallel(request, result, start_time)
        else:
            return await self._execute_sequential(request, result, start_time)

    # ----- Sequential workflow -----

    async def _execute_sequential(
        self,
        request: TAPRequest,
        result: CoordinationResult,
        start_time: float,
    ) -> CoordinationResult:
        """顺序模式：视觉分析 → 上下文传递 → 编码执行

        Args:
            request: 原始 TAP 请求
            result: 协同结果（持续更新）
            start_time: 工作流开始时间

        Returns:
            CoordinationResult
        """
        # Phase 1: 视觉分析
        step_vision = CoordinationStep(
            phase=CoordinationPhase.VISION_ANALYSIS.value,
            model=self._config.vision_model,
            start_time=time.time(),
        )
        result.phase = CoordinationPhase.VISION_ANALYSIS.value

        try:
            # 创建视觉分析请求
            vision_request = self._build_vision_request(request)
            vision_response = await self._vision_provider.execute_tap(vision_request)

            step_vision.end_time = time.time()
            step_vision.success = True
            step_vision.token_usage = vision_response.usage
            result.steps.append(step_vision)

            # 解析视觉分析结果
            analysis = GLM5VTurboCompiler.parse_analysis_result(
                vision_response.raw_text or ""
            )
            result.vision_analysis = analysis

            logger.info(
                f"GLM52VCoordinatedWorkflow: 视觉分析完成 "
                f"(confidence={analysis.confidence:.2f}, "
                f"elements={analysis.element_count})"
            )

        except Exception as e:
            step_vision.end_time = time.time()
            step_vision.success = False
            step_vision.error = str(e)
            result.steps.append(step_vision)

            logger.error(f"GLM52VCoordinatedWorkflow: 视觉分析失败: {e}")

            if self._config.degrade_on_vision_failure:
                return await self._execute_degraded(request, start_time)
            else:
                result.phase = CoordinationPhase.FAILED.value
                return result

        # Phase 2: 上下文传递
        result.phase = CoordinationPhase.CONTEXT_TRANSFER.value
        step_transfer = CoordinationStep(
            phase=CoordinationPhase.CONTEXT_TRANSFER.value,
            start_time=time.time(),
        )
        step_transfer.end_time = time.time()
        result.steps.append(step_transfer)

        # Phase 3: 编码执行
        return await self._execute_coding_with_vision_context(
            request, analysis, result, start_time
        )

    # ----- With verification workflow -----

    async def _execute_with_verification(
        self,
        request: TAPRequest,
        result: CoordinationResult,
        start_time: float,
    ) -> CoordinationResult:
        """验证模式：视觉分析 → 编码 → 视觉验证

        Args:
            request: 原始 TAP 请求
            result: 协同结果（持续更新）
            start_time: 工作流开始时间

        Returns:
            CoordinationResult
        """
        # 先执行顺序模式
        result = await self._execute_sequential(request, result, start_time)

        if not result.success or result.final_response is None:
            return result

        # 验证循环
        for round_num in range(self._config.max_verification_rounds):
            logger.info(
                f"GLM52VCoordinatedWorkflow: 开始视觉验证 (round {round_num + 1})"
            )

            step_verify = CoordinationStep(
                phase=CoordinationPhase.VISUAL_VERIFICATION.value,
                model=self._config.vision_model,
                start_time=time.time(),
            )

            try:
                # 构建验证请求
                verify_request = self._build_verification_request(
                    request, result.final_response
                )
                verify_response = await self._vision_provider.execute_tap(verify_request)

                step_verify.end_time = time.time()
                step_verify.success = True
                step_verify.token_usage = verify_response.usage
                result.steps.append(step_verify)
                result.verification_result = verify_response

                # 检查验证分数
                score = _extract_verification_score(verify_response.raw_text or "")
                logger.info(
                    f"GLM52VCoordinatedWorkflow: 验证分数 = {score}"
                )

                if score is not None and score >= self._config.verification_score_threshold:
                    logger.info(
                        f"GLM52VCoordinatedWorkflow: 验证通过 (score={score} >= "
                        f"{self._config.verification_score_threshold})"
                    )
                    break

            except Exception as e:
                step_verify.end_time = time.time()
                step_verify.success = False
                step_verify.error = str(e)
                result.steps.append(step_verify)
                logger.warning(f"GLM52VCoordinatedWorkflow: 验证失败: {e}")

        # 更新最终状态
        result.phase = CoordinationPhase.COMPLETED.value
        result.total_duration_ms = (time.time() - start_time) * 1000
        result.total_tokens = _aggregate_token_usage(result.steps)

        return result

    # ----- Parallel workflow (experimental) -----

    async def _execute_parallel(
        self,
        request: TAPRequest,
        result: CoordinationResult,
        start_time: float,
    ) -> CoordinationResult:
        """并行模式：视觉分析和编码同时启动（实验性）

        注意：并行模式下，编码步骤无法获得视觉分析的上下文，
        因此实际效果有限。仅适用于编码和视觉分析独立的场景。

        Args:
            request: 原始 TAP 请求
            result: 协同结果
            start_time: 工作流开始时间

        Returns:
            CoordinationResult
        """
        import asyncio

        # 同时启动视觉分析和编码
        vision_request = self._build_vision_request(request)

        async def _run_vision():
            try:
                return await self._vision_provider.execute_tap(vision_request)
            except Exception as e:
                logger.error(f"并行模式视觉分析失败: {e}")
                return None

        async def _run_coding():
            try:
                return await self._coding_provider.execute_tap(request)
            except Exception as e:
                logger.error(f"并行模式编码执行失败: {e}")
                return None

        vision_resp, coding_resp = await asyncio.gather(
            _run_vision(), _run_coding()
        )

        # 记录视觉分析步骤
        step_vision = CoordinationStep(
            phase=CoordinationPhase.VISION_ANALYSIS.value,
            model=self._config.vision_model,
            start_time=start_time,
            end_time=time.time(),
            success=vision_resp is not None,
            token_usage=vision_resp.usage if vision_resp else {},
            error="" if vision_resp else "视觉分析失败",
        )
        result.steps.append(step_vision)

        if vision_resp:
            result.vision_analysis = GLM5VTurboCompiler.parse_analysis_result(
                vision_resp.raw_text or ""
            )

        # 记录编码步骤
        step_coding = CoordinationStep(
            phase=CoordinationPhase.CODE_GENERATION.value,
            model=self._config.coding_model,
            start_time=start_time,
            end_time=time.time(),
            success=coding_resp is not None,
            token_usage=coding_resp.usage if coding_resp else {},
            error="" if coding_resp else "编码执行失败",
        )
        result.steps.append(step_coding)

        if coding_resp:
            result.final_response = coding_resp
            result.phase = CoordinationPhase.COMPLETED.value
        else:
            result.phase = CoordinationPhase.FAILED.value

        result.total_duration_ms = (time.time() - start_time) * 1000
        result.total_tokens = _aggregate_token_usage(result.steps)

        return result

    # ----- Degraded mode -----

    async def _execute_degraded(
        self,
        request: TAPRequest,
        start_time: float,
    ) -> CoordinationResult:
        """降级模式：仅使用编码模型，多模态内容降级为文本

        Args:
            request: 原始 TAP 请求
            start_time: 工作流开始时间

        Returns:
            CoordinationResult with is_degraded=True
        """
        result = CoordinationResult(is_degraded=True)

        if self._coding_provider is None:
            result.phase = CoordinationPhase.FAILED.value
            return result

        # 将多模态内容降级为文本描述
        degraded_request = self._degrade_multimodal_to_text(request)

        step = CoordinationStep(
            phase=CoordinationPhase.DEGRADED.value,
            model=self._config.coding_model,
            start_time=time.time(),
        )

        try:
            response = await self._coding_provider.execute_tap(degraded_request)
            step.end_time = time.time()
            step.success = True
            step.token_usage = response.usage
            result.steps.append(step)
            result.final_response = response
            result.phase = CoordinationPhase.DEGRADED.value
        except Exception as e:
            step.end_time = time.time()
            step.success = False
            step.error = str(e)
            result.steps.append(step)
            result.phase = CoordinationPhase.FAILED.value

        result.total_duration_ms = (time.time() - start_time) * 1000
        result.total_tokens = _aggregate_token_usage(result.steps)

        return result

    # ----- Coding only (no multimodal) -----

    async def _execute_coding_only(
        self,
        request: TAPRequest,
        start_time: float,
    ) -> CoordinationResult:
        """纯编码模式（请求无多模态内容时使用）

        Args:
            request: 原始 TAP 请求
            start_time: 工作流开始时间

        Returns:
            CoordinationResult
        """
        result = CoordinationResult()

        step = CoordinationStep(
            phase=CoordinationPhase.CODE_GENERATION.value,
            model=self._config.coding_model,
            start_time=time.time(),
        )

        try:
            response = await self._coding_provider.execute_tap(request)
            step.end_time = time.time()
            step.success = True
            step.token_usage = response.usage
            result.steps.append(step)
            result.final_response = response
            result.phase = CoordinationPhase.COMPLETED.value
        except Exception as e:
            step.end_time = time.time()
            step.success = False
            step.error = str(e)
            result.steps.append(step)
            result.phase = CoordinationPhase.FAILED.value

        result.total_duration_ms = (time.time() - start_time) * 1000
        result.total_tokens = _aggregate_token_usage(result.steps)

        return result

    # ----- Coding with vision context -----

    async def _execute_coding_with_vision_context(
        self,
        request: TAPRequest,
        analysis: VisionAnalysisResult,
        result: CoordinationResult,
        start_time: float,
    ) -> CoordinationResult:
        """使用视觉分析结果作为上下文执行编码

        Args:
            request: 原始 TAP 请求
            analysis: 视觉分析结果
            result: 协同结果（持续更新）
            start_time: 工作流开始时间

        Returns:
            CoordinationResult
        """
        step = CoordinationStep(
            phase=CoordinationPhase.CODE_GENERATION.value,
            model=self._config.coding_model,
            start_time=time.time(),
        )
        result.phase = CoordinationPhase.CODE_GENERATION.value

        try:
            # 构建带视觉上下文的编码请求
            coding_request = self._build_coding_request(request, analysis)
            response = await self._coding_provider.execute_tap(coding_request)

            step.end_time = time.time()
            step.success = True
            step.token_usage = response.usage
            result.steps.append(step)
            result.final_response = response
            result.phase = CoordinationPhase.COMPLETED.value

        except Exception as e:
            step.end_time = time.time()
            step.success = False
            step.error = str(e)
            result.steps.append(step)
            result.phase = CoordinationPhase.FAILED.value

        result.total_duration_ms = (time.time() - start_time) * 1000
        result.total_tokens = _aggregate_token_usage(result.steps)

        return result

    # ----- Request builders -----

    def _build_vision_request(self, request: TAPRequest) -> TAPRequest:
        """构建视觉分析请求

        从原始请求中提取多模态内容和指令，构建适合 5V-Turbo 的请求。

        Args:
            request: 原始 TAP 请求

        Returns:
            视觉分析 TAP 请求
        """
        instruction = request.instruction
        if not instruction or ("分析" not in instruction and "analyze" not in instruction.lower()):
            instruction = f"请分析这个设计稿/截图，提取视觉语义信息。\n\n原始指令: {instruction}"

        return TAPRequest(
            meta={**request.meta, "intent": "design", "coordination_phase": "vision_analysis"},
            instruction=instruction,
            constraints=request.constraints,
            multimodal_context=request.multimodal_context,
            desktop_context=request.desktop_context,
        )

    def _build_coding_request(
        self,
        request: TAPRequest,
        analysis: VisionAnalysisResult,
    ) -> TAPRequest:
        """构建带视觉上下文的编码请求

        将视觉分析结果注入到编码请求的 context 中，
        让 GLM-5.2 可以基于视觉理解进行编码。

        Args:
            request: 原始 TAP 请求
            analysis: 视觉分析结果

        Returns:
            带视觉上下文的编码 TAP 请求
        """
        # 构建视觉上下文
        visual_context = analysis.to_context_string()

        # 合并上下文
        enhanced_context = dict(request.context)
        if self._config.context_sharing and visual_context:
            enhanced_context["visual_analysis"] = visual_context

        # 构建增强的指令
        instruction = request.instruction
        if self._config.inject_code_generation_hint and analysis.has_layout_info:
            instruction = f"{instruction}\n\n{VISION_CODE_GENERATION_HINT}"

        return TAPRequest(
            meta={**request.meta, "intent": request.meta.get("intent", "execute"), "coordination_phase": "code_generation"},
            context=enhanced_context,
            instruction=instruction,
            constraints=request.constraints,
            output_format_hint=request.output_format_hint,
            thinking_mode=request.thinking_mode,
            long_horizon=request.long_horizon,
            cache_preference=request.cache_preference,
        )

    def _build_verification_request(
        self,
        original_request: TAPRequest,
        coding_response: TAPResponse,
    ) -> TAPRequest:
        """构建视觉验证请求

        让 5V-Turbo 对比原始设计稿和生成的代码，评估一致性。

        Args:
            original_request: 原始 TAP 请求（包含设计稿图像）
            coding_response: 编码步骤的响应

        Returns:
            视觉验证 TAP 请求
        """
        code_summary = coding_response.raw_text or ""
        if len(code_summary) > 2000:
            code_summary = code_summary[:2000] + "\n... [代码已截断]"

        instruction = (
            "请对比原始设计稿和以下生成的代码实现，评估视觉一致性。\n\n"
            f"生成的代码:\n```\n{code_summary}\n```"
        )

        return TAPRequest(
            meta={
                **original_request.meta,
                "intent": "review",
                "coordination_phase": "visual_verification",
            },
            instruction=instruction,
            multimodal_context=original_request.multimodal_context,
        )

    def _degrade_multimodal_to_text(self, request: TAPRequest) -> TAPRequest:
        """将多模态内容降级为文本描述

        Args:
            request: 原始 TAP 请求

        Returns:
            降级后的 TAP 请求（无多模态内容）
        """
        # 提取文本描述
        descriptions: list[str] = []
        if request.multimodal_context:
            for mc in request.multimodal_context:
                desc = mc.extract_text_description()
                if desc:
                    descriptions.append(desc)

        # 将描述添加到上下文
        enhanced_context = dict(request.context)
        if descriptions:
            enhanced_context["multimodal_description"] = "\n".join(descriptions)

        # 增强指令，提醒模型这是降级模式
        instruction = request.instruction
        if descriptions:
            instruction = (
                f"[注意: 视觉模型不可用，以下是对图像的文字描述]\n"
                f"{'; '.join(descriptions)}\n\n"
                f"{instruction}"
            )

        return TAPRequest(
            meta={**request.meta, "coordination_degraded": True},
            context=enhanced_context,
            instruction=instruction,
            constraints=request.constraints,
            output_format_hint=request.output_format_hint,
            thinking_mode=request.thinking_mode,
            long_horizon=request.long_horizon,
            cache_preference=request.cache_preference,
        )

    # ----- Provider creation -----

    @staticmethod
    def _create_provider_from_compiler(
        compiler_name: str,
        model: str,
    ) -> ModelProvider | None:
        """从编译器名称创建一个简单的 ModelProvider（使用 MockAdapter）

        注意：这主要用于测试。生产环境应通过构造函数传入真实的 provider。

        Args:
            compiler_name: 编译器名称
            model: 模型名称

        Returns:
            ModelProvider 或 None（如果编译器不存在）
        """
        try:
            from teragent.core.adapters.mock import MockAdapter
            compiler_cls = TAPCompilerRegistry.get(compiler_name)
            if compiler_cls is None:
                return None
            compiler = compiler_cls()
            adapter = MockAdapter()
            return ModelProvider(compiler=compiler, adapter=adapter, model=model)
        except Exception as e:
            logger.warning(f"无法创建 provider (compiler={compiler_name}): {e}")
            return None


# ===== Helper functions =====


def _extract_verification_score(text: str) -> float | None:
    """从验证响应中提取分数

    支持的格式：
    - "总体评分: 8/10"
    - "总体评分 (0-10): 8"
    - "总体评分：8.5"
    - "Overall Score: 8/10"

    Args:
        text: 验证响应文本

    Returns:
        分数（0-10），如果未找到则返回 None
    """
    import re

    # 尝试多种模式
    patterns = [
        r"总体评分[：:]\s*(\d+(?:\.\d+)?)\s*/\s*10",
        r"总体评分[：:]\s*(\d+(?:\.\d+)?)",
        r"Overall\s+Score[：:]\s*(\d+(?:\.\d+)?)\s*/\s*10",
        r"Overall\s+Score[：:]\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*/\s*10",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue

    return None


def _aggregate_token_usage(steps: list[CoordinationStep]) -> dict:
    """汇总所有步骤的 token 用量

    Args:
        steps: 步骤列表

    Returns:
        汇总的 token 用量字典
    """
    total_prompt = 0
    total_completion = 0

    for step in steps:
        usage = step.token_usage or {}
        total_prompt += usage.get("prompt_tokens", 0)
        total_completion += usage.get("completion_tokens", 0)

    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
    }
