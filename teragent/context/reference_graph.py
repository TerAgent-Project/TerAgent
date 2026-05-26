# teragent/context/reference_graph.py
import logging
from collections import deque

import networkx as nx

logger = logging.getLogger(__name__)


class ReferenceGraph:
    """Directed graph tracking symbol definitions and call relationships.

    Nodes represent symbols (functions, classes) with a ``file_path`` attribute.
    Edges represent caller → callee relationships.
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_symbol(self, symbol_name: str, file_path: str) -> None:
        """添加节点，附加文件路径属性"""
        if not self.graph.has_node(symbol_name):
            self.graph.add_node(symbol_name, file_path=file_path)
        else:
            # 更新文件路径（可能被重新定义）
            self.graph.nodes[symbol_name]["file_path"] = file_path

    def add_call(self, caller: str, callee: str) -> None:
        """添加调用边：caller 调用了 callee"""
        if self.graph.has_node(caller) and self.graph.has_node(callee):
            self.graph.add_edge(caller, callee)
        else:
            logger.warning(f"Cannot add edge: node missing ({caller} -> {callee})")

    def remove_symbol(self, symbol_name: str) -> None:
        """Remove a symbol node and all its incident edges.

        Args:
            symbol_name: The symbol to remove.
        """
        if self.graph.has_node(symbol_name):
            self.graph.remove_node(symbol_name)
            logger.debug(f"Removed symbol: {symbol_name}")
        else:
            logger.warning(f"Cannot remove symbol: node not found ({symbol_name})")

    def remove_call(self, caller: str, callee: str) -> None:
        """Remove a specific call edge.

        Args:
            caller: The calling symbol.
            callee: The called symbol.
        """
        if self.graph.has_edge(caller, callee):
            self.graph.remove_edge(caller, callee)
            logger.debug(f"Removed edge: {caller} -> {callee}")
        else:
            logger.warning(f"Cannot remove edge: not found ({caller} -> {callee})")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_callers(self, symbol_name: str) -> list[str]:
        """获取所有直接调用该符号的符号列表（前驱节点）"""
        if not self.graph.has_node(symbol_name):
            return []
        return list(self.graph.predecessors(symbol_name))

    def get_callees(self, symbol_name: str) -> list[str]:
        """获取该符号直接调用的所有符号列表（后继节点）"""
        if not self.graph.has_node(symbol_name):
            return []
        return list(self.graph.successors(symbol_name))

    def get_all_symbols(self) -> list[str]:
        """Return a list of all symbol names in the graph."""
        return list(self.graph.nodes)

    def get_symbols_by_file(self, file_path: str) -> list[str]:
        """Return all symbols defined in the given file.

        Args:
            file_path: The file path to filter by.

        Returns:
            A list of symbol names whose ``file_path`` attribute matches.
        """
        result: list[str] = []
        for node, data in self.graph.nodes(data=True):
            if data.get("file_path") == file_path:
                result.append(node)
        return result

    def get_transitive_callers(self, symbol_name: str, max_depth: int | None = None) -> list[str]:
        """Return all transitive callers (ancestors) of *symbol_name*.

        Performs a BFS upward through caller edges.

        Args:
            symbol_name: The target symbol.
            max_depth: If set, limit the traversal depth.

        Returns:
            A list of caller symbol names (excluding *symbol_name* itself).
        """
        if not self.graph.has_node(symbol_name):
            return []
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque()
        queue.append((symbol_name, 0))
        visited.add(symbol_name)
        result: list[str] = []
        while queue:
            current, depth = queue.popleft()
            for pred in self.graph.predecessors(current):
                if pred not in visited:
                    next_depth = depth + 1
                    if max_depth is not None and next_depth > max_depth:
                        continue
                    visited.add(pred)
                    result.append(pred)
                    queue.append((pred, next_depth))
        return result

    def get_impact_set(self, modified_symbols: set[str]) -> set[str]:
        """Compute the full impact set of modified symbols.

        The impact set includes all symbols that transitively call any
        of the modified symbols (i.e., all transitive callers).

        Args:
            modified_symbols: Set of symbol names that have been modified.

        Returns:
            The set of all symbols potentially affected by the changes.
        """
        impact: set[str] = set()
        for sym in modified_symbols:
            if self.graph.has_node(sym):
                impact.update(self.get_transitive_callers(sym))
        return impact

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the graph to a JSON-friendly dict.

        Returns:
            A dict with keys ``nodes`` and ``edges``.
        """
        nodes = [
            {"name": name, "file_path": data.get("file_path", "")}
            for name, data in self.graph.nodes(data=True)
        ]
        edges = [
            {"caller": u, "callee": v}
            for u, v in self.graph.edges
        ]
        return {"nodes": nodes, "edges": edges}

    @classmethod
    def from_dict(cls, data: dict) -> "ReferenceGraph":
        """Deserialize a graph from a dict produced by :meth:`to_dict`.

        Args:
            data: A dict with ``nodes`` and ``edges`` keys.

        Returns:
            A new ReferenceGraph instance.
        """
        graph = cls()
        for node_data in data.get("nodes", []):
            graph.add_symbol(node_data["name"], node_data.get("file_path", ""))
        for edge_data in data.get("edges", []):
            graph.add_call(edge_data["caller"], edge_data["callee"])
        return graph

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return summary statistics about the graph.

        Returns:
            A dict with keys ``symbol_count``, ``edge_count``,
            ``avg_connectivity``, ``max_callers``, ``max_callees``.
        """
        symbol_count = self.graph.number_of_nodes()
        edge_count = self.graph.number_of_edges()
        return {
            "symbol_count": symbol_count,
            "edge_count": edge_count,
            "avg_connectivity": round(edge_count / symbol_count, 2) if symbol_count > 0 else 0,
            "max_callers": max((d for _, d in self.graph.in_degree()), default=0),
            "max_callees": max((d for _, d in self.graph.out_degree()), default=0),
        }
