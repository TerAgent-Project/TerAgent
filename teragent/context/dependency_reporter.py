# teragent/context/dependency_reporter.py
import asyncio
import logging
import os
from typing import TYPE_CHECKING, Protocol

__all__ = [
    "DependencyReporter",
    "MAX_REPORT_BUDGET",
    "TaskProtocol",
]

if TYPE_CHECKING:
    from teragent.context.code_indexer import CodeIndexer
    from teragent.context.reference_graph import ReferenceGraph

from teragent.utils.token_counter import estimate_tokens


class TaskProtocol(Protocol):
    """Minimal task interface for DependencyReporter"""
    output_files: list[str]


logger = logging.getLogger(__name__)

# 警戒线：单次子任务的依赖报告最多 15000 Tokens
# （25% 百分比限制暂未实现，仅使用绝对上限）
# 超过此线，说明任务拆解粒度过大，必须重新规划
MAX_REPORT_BUDGET = 15000


class DependencyReporter:
    def __init__(self, indexer: "CodeIndexer", graph: "ReferenceGraph", workspace_root: str) -> None:
        self.indexer = indexer
        self.graph = graph
        self.workspace_root = workspace_root

    async def generate_report(self, task: TaskProtocol) -> str:
        """
        根据任务的输出文件，生成依赖分析报告。
        核心原则：只看直接调用者（深度=1），按重要性排序全量生成。
        若直接调用者过多导致超限，则触发主动熔断，拒绝生成残缺报告。
        """
        report_lines: list[str] = []
        current_tokens: int = 0
        explosion_detected: bool = False

        for file_path in task.output_files:
            abs_path = os.path.join(self.workspace_root, file_path)
            if not os.path.exists(abs_path):
                continue

            symbols = await self.indexer.find_symbols_by_file(abs_path)

            for sym in symbols:
                callers = self.graph.get_callers(sym["name"])
                if not callers:
                    continue

                external_callers: list[tuple[str, dict]] = []
                for caller_name in callers:
                    caller_node = self.graph.graph.nodes.get(caller_name)
                    if not caller_node:
                        continue
                    if caller_node.get("file_path") != abs_path:
                        external_callers.append((caller_name, caller_node))

                if not external_callers:
                    continue

                # 按重要性排序（被调用越多的函数越核心）
                external_callers.sort(
                    key=lambda x: len(self.graph.get_callers(x[0])),
                    reverse=True,
                )

                report_lines.append(
                    f"### Impact of modifying `{sym['name']}` in `{file_path}`"
                )
                report_lines.append(
                    f"Found {len(external_callers)} external callers:\n"
                )

                for caller_name, caller_node in external_callers:
                    caller_file = caller_node.get("file_path")
                    if caller_file is None:
                        continue
                    snippet = await self._get_caller_context(caller_file, caller_name)
                    if snippet:
                        line = (
                            f"- **{caller_name}** in `{caller_file}`:\n"
                            f"```python\n{snippet}\n```\n"
                        )
                        line_tokens = estimate_tokens(line)

                        # 核心防护：检查是否即将撑爆预算
                        if current_tokens + line_tokens > MAX_REPORT_BUDGET:
                            explosion_detected = True
                            break

                        report_lines.append(line)
                        current_tokens += line_tokens

                if explosion_detected:
                    break

        if explosion_detected:
            # 主动熔断：绝不返回截断的报告，而是返回明确的错误信号
            error_msg = (
                "DEPENDENCY_EXPLOSION: The impact radius of this task exceeds the context limit. "
                "This task MUST be decomposed into smaller tasks in the PLAN "
                "(e.g., deprecate old interface first, then migrate callers in batches)."
            )
            logger.error(error_msg)
            return error_msg

        return (
            "\n".join(report_lines)
            if report_lines
            else "No external dependencies found. You are safe to modify."
        )

    async def _get_caller_context(self, file_path: str, symbol_name: str) -> str:
        """提取调用者的完整上下文。

        策略：给出函数签名 + 函数内调用的前 15 行代码，让模型看到真实的使用场景。
        严格遵循绝对宪法 #2：async 内部禁止阻塞 I/O，必须使用 run_in_executor。
        """
        if file_path is None:
            return ""
        abs_path = os.path.join(self.workspace_root, file_path)
        symbols = await self.indexer.find_symbols_by_file(abs_path)
        for sym in symbols:
            if sym["name"] == symbol_name:
                signature = sym.get("signature", "")
                line_number = sym.get("line_number")

                if not line_number:
                    return signature

                try:
                    loop = asyncio.get_running_loop()
                    context_code = await loop.run_in_executor(
                        None, self._sync_read_context, abs_path, line_number
                    )
                    return context_code
                except Exception as e:
                    logger.warning(
                        f"Failed to read context for {symbol_name}: {e}"
                    )
                    return signature
        return ""

    @staticmethod
    def _sync_read_context(abs_path: str, line_number: int, context_lines: int = 15) -> str:
        """同步读取文件上下文（供 run_in_executor 调用）"""
        with open(abs_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            start = line_number - 1
            end = min(start + context_lines, len(lines))
            return "".join(lines[start:end])
