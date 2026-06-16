# teragent/tools/result_cache.py
"""工具结果缓存 — AgentTool 结果 TTL 缓存

提供 ResultCache 类，用于缓存工具执行结果，避免重复执行
相同参数的耗时工具调用（特别是 AgentTool 子 Agent 调用）。

功能:
  - TTL (Time-To-Live) 过期机制
  - LRU 淘汰策略（max_size 限制）
  - asyncio.Lock 保证并发安全
  - 缓存键生成（工具名 + 参数哈希）
  - 统计信息（命中/未命中/淘汰计数）

设计参考:
  - Python functools.lru_cache 的 LRU 策略
  - Redis TTL 过期机制
  - asyncio.Lock 并发保护模式

用法::

    from teragent.tools.result_cache import ResultCache

    # 创建缓存
    cache = ResultCache(max_size=128, default_ttl=60.0)

    # 生成缓存键
    key = cache.make_key("use_coder", {"task": "write tests"})

    # 缓存操作
    await cache.set(key, result, ttl=120.0)
    result = await cache.get(key)
    await cache.invalidate(key)
    await cache.clear()

    # 检查缓存
    if await cache.has(key):
        result = await cache.get(key)

    # 统计信息
    stats = cache.stats()
    # {"hits": 10, "misses": 3, "evictions": 1, "size": 5, "max_size": 128}

    # 与 AgentTool 集成
    agent_tool = AgentTool(agent=coder, cache=cache, cache_ttl=60.0)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ResultCache",
    "CacheEntry",
    "CacheStats",
]


@dataclass
class CacheEntry:
    """缓存条目

    存储缓存值及其元数据（创建时间、过期时间）。

    Attributes:
        value: 缓存的值
        created_at: 创建时间戳（time.monotonic）
        expires_at: 过期时间戳（time.monotonic），0 表示永不过期
    """

    value: Any
    created_at: float
    expires_at: float  # 0 = never expires

    @property
    def is_expired(self) -> bool:
        """检查条目是否已过期

        Returns:
            True 如果当前时间超过 expires_at 且 expires_at > 0
        """
        if self.expires_at <= 0:
            return False
        return time.monotonic() > self.expires_at

    @property
    def ttl_remaining(self) -> float:
        """剩余 TTL（秒）

        Returns:
            剩余秒数，0 表示已过期或永不过期
        """
        if self.expires_at <= 0:
            return float("inf")
        remaining = self.expires_at - time.monotonic()
        return max(0.0, remaining)


@dataclass
class CacheStats:
    """缓存统计信息

    Attributes:
        hits: 缓存命中次数
        misses: 缓存未命中次数
        evictions: LRU 淘汰次数
        size: 当前缓存大小
        max_size: 最大缓存大小
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 128


class ResultCache:
    """工具结果缓存

    提供 TTL + LRU 淘汰策略的异步安全缓存，
    适用于缓存 AgentTool 执行结果等耗时操作。

    特性:
    - TTL 过期: 每个条目可设置独立的 TTL，过期自动失效
    - LRU 淘汰: 缓存满时淘汰最近最少使用的条目
    - 异步安全: 使用 asyncio.Lock 保护所有状态变更
    - 统计信息: 跟踪命中/未命中/淘汰次数

    Args:
        max_size: 最大缓存条目数（默认 128），0 表示无限制
        default_ttl: 默认 TTL（秒，默认 60.0），0 表示永不过期
    """

    def __init__(
        self,
        max_size: int = 128,
        default_ttl: float = 60.0,
    ) -> None:
        if max_size < 0:
            raise ValueError(f"max_size must be >= 0, got {max_size}")
        if default_ttl < 0:
            raise ValueError(f"default_ttl must be >= 0, got {default_ttl}")

        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

        # 统计
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    @property
    def max_size(self) -> int:
        """最大缓存条目数"""
        return self._max_size

    @property
    def default_ttl(self) -> float:
        """默认 TTL（秒）"""
        return self._default_ttl

    @staticmethod
    def make_key(tool_name: str, params: dict) -> str:
        """生成缓存键

        将工具名和参数序列化为确定性哈希键。
        参数先排序再序列化，确保相同参数生成相同的键。

        Args:
            tool_name: 工具名称
            params: 工具参数字典

        Returns:
            缓存键字符串，格式为 "{tool_name}:{sha256_hash}"
        """
        # 确定性序列化：排序键 + 紧凑 JSON
        try:
            params_str = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            # 不可序列化的参数使用 repr 作为后备
            params_str = repr(params)

        params_hash = hashlib.sha256(params_str.encode("utf-8")).hexdigest()[:16]
        return f"{tool_name}:{params_hash}"

    async def get(self, key: str) -> Any | None:
        """获取缓存值

        如果缓存命中且未过期，返回缓存值并移动到 LRU 队列尾部
        （标记为最近使用）。如果缓存未命中或已过期，返回 None。

        Args:
            key: 缓存键

        Returns:
            缓存值，或 None（未命中/已过期）
        """
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            # 检查过期
            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                logger.debug("ResultCache: key '%s' expired, evicted", key)
                return None

            # LRU: 移动到队列尾部（最近使用）
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: float | None = None,
    ) -> None:
        """设置缓存值

        如果键已存在，更新值并移动到 LRU 队列尾部。
        如果缓存已满，淘汰最近最少使用的条目。

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间（秒），None 使用 default_ttl，0 表示永不过期
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl
        now = time.monotonic()
        expires_at = (now + effective_ttl) if effective_ttl > 0 else 0

        entry = CacheEntry(
            value=value,
            created_at=now,
            expires_at=expires_at,
        )

        async with self._lock:
            # 如果键已存在，先删除（后续重新插入到尾部）
            if key in self._cache:
                del self._cache[key]

            # LRU 淘汰：如果缓存已满，移除最旧的条目
            if self._max_size > 0 and len(self._cache) >= self._max_size:
                # 淘汰 LRU 条目（OrderedDict 首部）
                evicted_key, _ = self._cache.popitem(last=False)
                self._evictions += 1
                logger.debug("ResultCache: LRU evicted key '%s'", evicted_key)

            self._cache[key] = entry

    async def has(self, key: str) -> bool:
        """检查缓存键是否存在且未过期

        Args:
            key: 缓存键

        Returns:
            True 如果键存在且未过期
        """
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                del self._cache[key]
                return False
            return True

    async def invalidate(self, key: str) -> bool:
        """使指定缓存键失效

        Args:
            key: 缓存键

        Returns:
            True 如果键存在且已被移除，False 如果键不存在
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug("ResultCache: invalidated key '%s'", key)
                return True
            return False

    async def clear(self) -> int:
        """清空所有缓存

        Returns:
            被清除的条目数
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.debug("ResultCache: cleared %d entries", count)
            return count

    async def cleanup_expired(self) -> int:
        """清理所有过期条目

        遍历缓存，移除所有已过期的条目。
        建议在低峰期定期调用此方法，防止过期条目长期占用内存。

        Returns:
            被清理的过期条目数
        """
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                logger.debug(
                    "ResultCache: cleaned up %d expired entries",
                    len(expired_keys),
                )
            return len(expired_keys)

    def stats(self) -> CacheStats:
        """获取缓存统计信息

        Returns:
            CacheStats 实例，包含命中/未命中/淘汰/大小统计
        """
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            size=len(self._cache),
            max_size=self._max_size,
        )

    @property
    def hit_rate(self) -> float:
        """缓存命中率

        Returns:
            命中率（0.0 ~ 1.0），无访问时返回 0.0
        """
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def __len__(self) -> int:
        """当前缓存条目数"""
        return len(self._cache)

    def __repr__(self) -> str:
        return (
            f"ResultCache(size={len(self._cache)}, "
            f"max_size={self._max_size}, "
            f"hit_rate={self.hit_rate:.1%})"
        )
