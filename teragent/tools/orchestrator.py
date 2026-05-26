# teragent/tools/orchestrator.py
"""工具并行编排器

参考 Claude-Code 的 toolOrchestration.ts，实现工具调用的智能编排：

核心策略:
  1. 连续的只读 + concurrency_safe 工具 --> 并行执行
  2. 非只读工具 --> 串行执行
  3. 破坏性工具 --> 串行执行 + 额外日志
  4. 未知工具 --> 串行执行（安全优先）

完整生命周期:
  validate_input --> enhanced_permission_check --> check_permissions --> [PRE_TOOL_USE hooks] --> execute --> [POST_TOOL_USE hooks] --> 收集结果

设计原则:
  - 不信任模型的管理能力 -- 并行/串行由工具安全属性决定，不由模型决定
  - 安全优先 -- 有疑问就串行（宁可慢，不可错）
  - 保持原始顺序 -- 结果按 tool_calls 原始顺序返回
"""
import asyncio
import logging
from typing import Callable, Awaitable, Optional, TYPE_CHECKING

from teragent.tools.base import BaseTool, ToolResult
from teragent.core.types import ToolSafety
from teragent.tools.registry import ToolRegistry

from teragent.hooks.manager import HookManager, HookEvent, HookContext, HookDecision

if TYPE_CHECKING:
    from teragent.security.permission import EnhancedPermissionManager

logger = logging.getLogger(__name__)

# 最大并行工具数
MAX_CONCURRENT_TOOLS = 10


class ToolOrchestrator:
    """工具并行编排器 -- 参考 Claude-Code 的 toolOrchestration.ts

    核心职责:
      - 将 tool_calls 分为并行组（只读）和串行组（写/破坏）
      - 对每个工具调用执行完整生命周期（验证 --> 权限 --> Hook --> 执行 --> Hook --> 收集结果）
      - 收集结果并保持原始顺序

    使用方式::

        orchestrator = ToolOrchestrator(tool_registry, permission_level=1)
        results = await orchestrator.execute_batch(
            tool_calls,
            on_progress=lambda name, progress: print(f"{name}: {progress}")
        )
        for tool_call, result in results:
            if result.success:
                print(f"成功: {result.data}")
            else:
                print(f"失败: {result.error}")
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        permission_level: int = 0,
        max_concurrent: int = MAX_CONCURRENT_TOOLS,
        hook_manager: Optional[HookManager] = None,
        enhanced_perm_manager: Optional["EnhancedPermissionManager"] = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.permission_level = permission_level
        self.max_concurrent = max_concurrent
        # Hook 管理器（None = 不执行 hooks）
        self.hook_manager = hook_manager
        # 增强权限管理器（None = 仅使用 BaseTool.check_permissions）
        self.enhanced_perm_manager = enhanced_perm_manager

    async def execute_batch(
        self,
        tool_calls: list[dict],
        on_progress: Optional[Callable[[str, float], Awaitable[None]]] = None,
    ) -> list[tuple[dict, ToolResult]]:
        """批量执行工具调用

        将工具调用分为并行组（只读）和串行组（写/破坏），
        最大化执行效率，同时保证安全性。

        Args:
            tool_calls: [{"name": ..., "arguments": ..., "id": ...}]
            on_progress: 进度回调 (tool_name, progress_fraction) --> None

        Returns:
            [(tool_call, result), ...] -- 保持原始顺序
        """
        if not tool_calls:
            return []

        # 单个工具调用：直接执行（无需分区）
        if len(tool_calls) == 1:
            result = await self._execute_single(tool_calls[0], on_progress)
            return [(tool_calls[0], result)]

        # 多个工具调用：分区编排
        partitions = self._partition(tool_calls)

        results: list[tuple[dict, ToolResult]] = []
        for partition in partitions:
            if partition["parallel"]:
                # 并行执行
                batch_results = await self._execute_parallel(
                    partition["calls"], on_progress
                )
                results.extend(batch_results)
                logger.info(
                    f"ToolOrchestrator: parallel partition completed, "
                    f"{len(batch_results)} tools executed"
                )
            else:
                # 串行执行
                for call in partition["calls"]:
                    result = await self._execute_single(call, on_progress)
                    results.append((call, result))
                logger.info(
                    f"ToolOrchestrator: serial partition completed, "
                    f"{len(partition['calls'])} tools executed"
                )

        return results

    def _partition(self, tool_calls: list[dict]) -> list[dict]:
        """将工具调用分为并行/串行分区

        策略:
          - 连续的只读 + concurrency_safe 工具 --> 并行
          - 其他 --> 串行
          - 当并行区遇到非并行工具时，当前并行区结束，开始新的分区

        Returns:
            [{"parallel": bool, "calls": [tool_call, ...]}, ...]
        """
        partitions: list[dict] = []
        current_batch: list[dict] = []
        current_parallel = True

        for call in tool_calls:
            tool_name = call.get("name", "")
            tool = self.tool_registry.get(tool_name)

            if not tool:
                # 未知工具：结束当前分区，单独串行执行
                if current_batch:
                    partitions.append({
                        "parallel": current_parallel,
                        "calls": current_batch,
                    })
                    current_batch = []
                partitions.append({"parallel": False, "calls": [call]})
                current_parallel = False
                continue

            can_parallel = tool.is_read_only and tool.is_concurrency_safe

            if can_parallel and current_parallel:
                # 可以加入当前并行区
                current_batch.append(call)
            else:
                # 不能并行或当前区不是并行区 --> 结束当前分区
                if current_batch:
                    partitions.append({
                        "parallel": current_parallel,
                        "calls": current_batch,
                    })
                # 开始新分区
                current_batch = [call]
                current_parallel = can_parallel

        # 最后一个分区
        if current_batch:
            partitions.append({
                "parallel": current_parallel,
                "calls": current_batch,
            })

        # 日志：分区结果
        for i, part in enumerate(partitions):
            mode = "PARALLEL" if part["parallel"] else "SERIAL"
            names = [c.get("name", "?") for c in part["calls"]]
            logger.debug(f"Partition {i} ({mode}): {names}")

        return partitions

    async def _execute_single(
        self,
        tool_call: dict,
        on_progress: Optional[Callable[[str, float], Awaitable[None]]] = None,
    ) -> ToolResult:
        """执行单个工具调用（含完整生命周期 + Hook 集成）

        生命周期:
          1. 查找工具
          2. validate_input -- 输入验证
          3. check_permissions -- 权限检查
          4. PRE_TOOL_USE hooks
          5. 设置进度回调
          6. execute -- 执行工具
          7. POST_TOOL_USE hooks
          8. 返回结果（附带安全级别）

        Args:
            tool_call: {"name": ..., "arguments": ..., "id": ...}
            on_progress: 进度回调

        Returns:
            ToolResult
        """
        tool_name = tool_call.get("name", "")
        params = tool_call.get("arguments", {})
        call_id = tool_call.get("id", "")

        # 1. 查找工具
        tool = self.tool_registry.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                data={},
                error=f"未知工具: {tool_name}",
            )

        # 2. 输入验证（validate_input 包含 validate_params）
        errors = tool.validate_input(params)
        if errors:
            logger.warning(
                f"ToolOrchestrator: input validation failed for {tool_name}: {errors}"
            )
            return ToolResult(
                success=False,
                data={},
                error=f"输入验证失败: {'; '.join(errors)}",
                safety=tool.safety_level,
            )

        # 3a. 增强权限检查（优先于 BaseTool.check_permissions）
        if self.enhanced_perm_manager:
            # 优先使用异步方法（支持 AI 分类器），回退到同步方法
            if hasattr(self.enhanced_perm_manager, 'acheck_tool_params'):
                allowed, reason = await self.enhanced_perm_manager.acheck_tool_params(
                    tool_name, params
                )
            else:
                allowed, reason = self.enhanced_perm_manager.check_tool_params(
                    tool_name, params
                )
            if not allowed:
                logger.warning(
                    f"ToolOrchestrator: enhanced permission denied for "
                    f"{tool_name}: {reason}"
                )
                return ToolResult(
                    success=False,
                    data={},
                    error=f"权限不足（规则拒绝）: {reason}",
                    safety=tool.safety_level,
                )

        # 3b. 基础权限检查（BaseTool 级别检查，作为第二层防护）
        allowed, reason = tool.check_permissions(params, self.permission_level)
        if not allowed:
            logger.warning(
                f"ToolOrchestrator: permission denied for {tool_name}: {reason}"
            )
            return ToolResult(
                success=False,
                data={},
                error=f"权限不足: {reason}",
                safety=tool.safety_level,
            )

        # 4. PRE_TOOL_USE hooks
        if self.hook_manager:
            hook_context = HookContext(
                event=HookEvent.PRE_TOOL_USE,
                tool_name=tool_name,
                params=dict(params) if isinstance(params, dict) else {},
                metadata={"call_id": call_id},
            )
            hook_result = await self.hook_manager.run_hooks(
                HookEvent.PRE_TOOL_USE, hook_context
            )

            if hook_result.decision == HookDecision.DENY:
                logger.warning(
                    f"ToolOrchestrator: PRE_TOOL_USE hook DENIED {tool_name}: "
                    f"{hook_result.reason}"
                )
                return ToolResult(
                    success=False,
                    data={},
                    error=f"Hook denied: {hook_result.reason}",
                    safety=tool.safety_level,
                )

            if hook_result.decision == HookDecision.MODIFY:
                if hook_result.modified_params is not None:
                    params = hook_result.modified_params
                    logger.info(
                        f"ToolOrchestrator: PRE_TOOL_USE hook MODIFIED params for "
                        f"{tool_name}: {hook_result.reason}"
                    )

        # 5. 执行
        logger.info(
            f"ToolOrchestrator: executing {tool_name} "
            f"(safety={tool.safety_level.value}, "
            f"concurrent={tool.is_concurrency_safe})"
        )

        # Pass progress_callback as parameter to avoid race conditions
        # when the same tool instance is used concurrently
        try:
            result = await tool.execute(params, progress_callback=on_progress)
            # 确保结果携带安全级别
            if result.safety == ToolSafety.READ_ONLY and tool.safety_level != ToolSafety.READ_ONLY:
                result.safety = tool.safety_level
        except Exception as e:
            logger.error(
                f"ToolOrchestrator: {tool_name} execution failed: {e}",
                exc_info=True,
            )
            result = ToolResult(
                success=False,
                data={},
                error=str(e),
                safety=tool.safety_level,
            )

        # 6. POST_TOOL_USE hooks
        if self.hook_manager:
            post_context = HookContext(
                event=HookEvent.POST_TOOL_USE,
                tool_name=tool_name,
                params=dict(params) if isinstance(params, dict) else {},
                result={
                    "success": result.success,
                    "data": result.data,
                    "error": result.error,
                } if result else None,
                metadata={"call_id": call_id},
            )
            # POST hooks 不影响执行结果，只做记录/通知
            await self.hook_manager.run_hooks(HookEvent.POST_TOOL_USE, post_context)

        return result

    async def _execute_parallel(
        self,
        tool_calls: list[dict],
        on_progress: Optional[Callable[[str, float], Awaitable[None]]] = None,
    ) -> list[tuple[dict, ToolResult]]:
        """并行执行一组工具调用

        使用信号量限制最大并行数，避免资源过度占用。

        Args:
            tool_calls: 同一分区内的工具调用列表
            on_progress: 进度回调

        Returns:
            [(tool_call, result), ...]
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def limited_execute(call: dict) -> tuple[dict, ToolResult]:
            async with semaphore:
                result = await self._execute_single(call, on_progress)
                return (call, result)

        tasks = [limited_execute(call) for call in tool_calls]
        results = await asyncio.gather(*tasks)
        return list(results)

    def get_execution_plan(self, tool_calls: list[dict]) -> list[dict]:
        """预览编排计划（不执行，用于调试/TUI 展示）

        Returns:
            分区计划 [{"parallel": bool, "calls": [...], "tool_names": [...]}, ...]
        """
        partitions = self._partition(tool_calls)
        plan = []
        for part in partitions:
            plan.append({
                "parallel": part["parallel"],
                "tool_names": [c.get("name", "?") for c in part["calls"]],
                "count": len(part["calls"]),
            })
        return plan

    def set_permission_level(self, level: int) -> None:
        """更新权限级别"""
        self.permission_level = level

    def set_enhanced_perm_manager(
        self, enhanced_perm_manager: "EnhancedPermissionManager"
    ) -> None:
        """设置增强权限管理器

        Args:
            enhanced_perm_manager: EnhancedPermissionManager 实例
        """
        self.enhanced_perm_manager = enhanced_perm_manager

    def set_hook_manager(self, hook_manager: HookManager) -> None:
        """设置 Hook 管理器

        Args:
            hook_manager: HookManager 实例
        """
        self.hook_manager = hook_manager
