"""teragent/context — Context management components

Phase 7: Context compression and management migration
Phase P2-2: Model-specific context profiles for 1M context management

Components:
    - ContextWindow: Token-budget-aware context window manager
    - ContextProfile: Base context profile for context window management
    - DeepSeekV4ContextProfile: DeepSeek V4 1M context profile
    - GLM5ContextProfile: GLM-5 200K extreme compression profile
    - MiniMaxM3ContextProfile: MiniMax M3 1M MSA full-text profile
    - Microcompactor: Tool result micro-compressor
    - AutoCompactor: Automatic context compressor (LLM-based)
    - CodeIndexer: Code symbol indexer (tree-sitter + SQLite) [requires: pip install teragent[ast]]
    - ReferenceGraph: Reference relationship graph (networkx) [requires: pip install teragent[graph]]
    - DependencyReporter: Dependency impact analysis reporter
    - VectorIndexer: Vector semantic search (LanceDB) [requires: pip install teragent[vector]]
    - Memory: AGENT.md project memory management

Optional dependencies are lazily imported — the package itself always installs
without them. Only the specific classes that need optional libraries will raise
ImportError when used without the corresponding extra installed.
"""

from __future__ import annotations

import logging

# Core components — no optional dependencies
from teragent.context.auto_compact import AutoCompactor
from teragent.context.context_window import ContextWindow
from teragent.context.memory import (
    extract_rules,
    load_agent_md,
    merge_agent_md,
    save_agent_md,
)
from teragent.context.microcompactor import Microcompactor
from teragent.context.profiles import (
    ContextProfile,
    DeepSeekV4ContextProfile,
    GLM5CompactionStrategy,
    GLM5ContextProfile,
    MiniMaxM3ContextProfile,
)

# DependencyReporter requires CodeIndexer + ReferenceGraph (optional deps),
# so it must be lazy-loaded too.
logger = logging.getLogger(__name__)


def __getattr__(name: str):
    """Lazy-load optional components that require extra dependencies.

    This allows ``import teragent`` to succeed even when optional
    dependencies (lancedb, tree-sitter, networkx) are not installed.
    """
    if name == "DependencyReporter":
        try:
            from teragent.context.dependency_reporter import DependencyReporter
            return DependencyReporter
        except ImportError:
            raise ImportError(
                f"{name} requires optional dependencies (tree-sitter, networkx). "
                f"Install with: pip install teragent[ast] teragent[graph]"
            )
    if name == "TaskProtocol":
        try:
            from teragent.context.dependency_reporter import TaskProtocol
            return TaskProtocol
        except ImportError:
            raise ImportError(
                f"{name} requires optional dependencies (tree-sitter, networkx). "
                f"Install with: pip install teragent[ast] teragent[graph]"
            )
    if name == "CodeIndexer":
        try:
            from teragent.context.code_indexer import CodeIndexer
            return CodeIndexer
        except ImportError:
            raise ImportError(
                f"{name} requires optional dependency (tree-sitter). "
                f"Install with: pip install teragent[ast]"
            )
    if name == "ReferenceGraph":
        try:
            from teragent.context.reference_graph import ReferenceGraph
            return ReferenceGraph
        except ImportError:
            raise ImportError(
                f"{name} requires optional dependency (networkx). "
                f"Install with: pip install teragent[graph]"
            )
    if name == "VectorIndexer":
        try:
            from teragent.context.vector_indexer import VectorIndexer
            return VectorIndexer
        except ImportError:
            raise ImportError(
                f"{name} requires optional dependency (LanceDB). "
                f"Install with: pip install teragent[vector]"
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core (always available)
    "ContextWindow",
    "ContextProfile",
    "DeepSeekV4ContextProfile",
    "GLM5ContextProfile",
    "GLM5CompactionStrategy",
    "MiniMaxM3ContextProfile",
    "Microcompactor",
    "AutoCompactor",
    "load_agent_md",
    "save_agent_md",
    "merge_agent_md",
    "extract_rules",
    # Lazy-loaded (DependencyReporter/TaskProtocol require CodeIndexer + ReferenceGraph)
    "DependencyReporter",
    "TaskProtocol",
    # Optional (lazy-loaded, requires extra deps)
    "CodeIndexer",
    "ReferenceGraph",
    "VectorIndexer",
]
