# tests/conftest.py
"""Shared pytest fixtures and configuration for TerAgent test suite.

公共 fixture 和配置
"""
import asyncio
import os
import tempfile
import logging
import pytest

# 确保项目根目录在 sys.path 中
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ===== 日志配置 =====

@pytest.fixture(autouse=True)
def _setup_logging():
    """为每个测试配置日志级别，方便调试"""
    logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")


# ===== 事件循环 =====

@pytest.fixture(scope="session")
def event_loop():
    """Session-scoped event loop for async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# ===== 临时工作区 =====

@pytest.fixture
def workspace(tmp_path):
    """创建临时工作区目录，用于文件系统相关测试"""
    return str(tmp_path)


@pytest.fixture
def workspace_with_files(tmp_path):
    """创建包含测试文件的临时工作区"""
    # 创建一些测试文件
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main(): pass\n")
    (tmp_path / "src" / "utils.py").write_text("def helper(): pass\n")
    (tmp_path / "README.md").write_text("# Test Project\n")
    (tmp_path / ".env").write_text("SECRET=abc123\n")
    return str(tmp_path)


# ===== EventBus fixture =====

@pytest.fixture
def event_bus():
    """创建干净的 EventBus 实例"""
    from teragent.event_bus import EventBus
    bus = EventBus()
    yield bus
    bus.clear()


# ===== Permission fixture =====

@pytest.fixture
def enhanced_perm_manager():
    """创建带有默认规则的 EnhancedPermissionManager"""
    from teragent.security.permission import (
        EnhancedPermissionManager, PermissionRule, PermissionEffect,
    )
    mgr = EnhancedPermissionManager()
    for rule in EnhancedPermissionManager.default_rules():
        mgr.add_rule(rule)
    return mgr


# ===== FileStateTracker fixture =====

@pytest.fixture
def file_tracker(workspace):
    """创建 FileStateTracker 实例"""
    from teragent.security.file_state import FileStateTracker
    return FileStateTracker(workspace_root=workspace)


# ===== CircuitBreaker fixtures =====

@pytest.fixture
def budget_tracker():
    """创建 CostBudgetTracker 实例"""
    from teragent.reliability.circuit_breaker import CostBudgetTracker, CostBudgetConfig
    config = CostBudgetConfig(
        max_session_tokens=10_000,
        warning_threshold=0.7,
        critical_threshold=0.9,
    )
    return CostBudgetTracker(config=config)


@pytest.fixture
def failure_breaker():
    """创建 ConsecutiveFailureBreaker 实例"""
    from teragent.reliability.circuit_breaker import ConsecutiveFailureBreaker
    return ConsecutiveFailureBreaker(max_consecutive=3, window_seconds=1.0)


@pytest.fixture
def cb_manager():
    """创建 CircuitBreakerManager 实例"""
    from teragent.reliability.circuit_breaker import CircuitBreakerManager
    config = {
        "budget": {"max_session_tokens": 10_000},
        "failure_breaker": {"max_consecutive": 3, "window_seconds": 1.0},
        "latency_breaker": {"warn_latency_ms": 1000.0, "avg_window": 5},
        "progress_detector": {"stall_threshold": 5},
    }
    return CircuitBreakerManager(config=config)
