# teragent/security/ai_permission_classifier.py
"""AI Permission Classifier

Optional component that uses an LLM to classify tool calls into permission
levels when static rules do not provide a clear answer.  This is the final
advisory layer in the permission pipeline, consulted only after:

  Layer 1: User-level rules (source=user)
  Layer 2: Config-level rules (source=config)
  Layer 3: Project-level rules (source=project)
  Layer 4: System-level rules (source=system)
  Layer 5: Permission level check (PermissionLevel)
  Layer 6: AI classifier (this module) -- advisory only
  Layer 7: Default policy (DENY)

Design principles:
  - AI classification is *advisory* -- it never overrides an explicit rule.
  - Conservative default: if the classifier is unsure (confidence below
    threshold), the call is denied.
  - Heuristic fallback is always available when no model is configured or
    when the model times out / errors.
  - Results are cached with a short TTL to avoid redundant LLM calls.

Usage::

    from teragent.security.ai_permission_classifier import AIPermissionClassifier
    from teragent.security.permission import PermissionEffect

    classifier = AIPermissionClassifier(model=my_model_provider)

    effect, confidence, reason = await classifier.classify(
        tool_name="write_file",
        params={"file_path": "/src/main.py", "content": "..."},
        context="User requested code refactoring",
    )
    # effect=PermissionEffect.ALLOW, confidence=0.72, reason="..."

Integration with EnhancedPermissionManager::

    from teragent.security.permission import EnhancedPermissionManager

    epm = EnhancedPermissionManager()
    epm.ai_classifier = AIPermissionClassifier(model=my_model_provider)

    # epm.check() will now consult the AI classifier when no rule matches
    # and the permission level check also doesn't resolve the decision.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

__all__ = [
    "AIPermissionClassifier",
]

from teragent.security.permission import PermissionEffect

if TYPE_CHECKING:
    from teragent.core.provider import ModelProvider

logger = logging.getLogger(__name__)


# ===== Constants =====

_LLM_TIMEOUT_SECONDS: float = 5.0
_CACHE_MAX_SIZE: int = 100
_CACHE_TTL_SECONDS: float = 300.0  # 5 minutes

# Known read-only tool names (supplements ToolSafety lookup)
_READ_ONLY_TOOL_NAMES: frozenset[str] = frozenset({
    "read_file",
    "list_directory",
    "explore_codebase",
    "classify_intent",
    "get_pipeline_status",
    "submit_failure",
})

# Known destructive / high-risk tool names
_DESTRUCTIVE_TOOL_NAMES: frozenset[str] = frozenset({
    "execute_subtask",
})

_HIGH_RISK_TOOL_NAMES: frozenset[str] = frozenset({
    "create_project",
})

# Path patterns that are always denied (high confidence)
_SENSITIVE_PATH_PREFIXES_UNIX: tuple[str, ...] = (
    "/etc/",
    ".env",
    ".ssh/",
    ".git/",
)

def _get_sensitive_path_prefixes() -> tuple[str, ...]:
    """获取平台特定的敏感路径前缀"""
    prefixes = list(_SENSITIVE_PATH_PREFIXES_UNIX)
    if sys.platform == "win32":
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        prefixes.extend([
            system_root.replace("\\", "/") + "/",
            os.path.join(system_root, "System32").replace("\\", "/") + "/",
        ])
    return tuple(prefixes)

_SENSITIVE_PATH_PREFIXES: tuple[str, ...] = _get_sensitive_path_prefixes()


# ===== LRU Cache =====

@dataclass
class _CacheEntry:
    """Single entry in the classification LRU cache."""

    effect: PermissionEffect
    confidence: float
    reason: str
    timestamp: float  # monotonic or wall-clock; we use time.monotonic()

    def is_expired(self, ttl: float = _CACHE_TTL_SECONDS) -> bool:
        """Return True if this entry has exceeded *ttl* seconds."""
        return (time.monotonic() - self.timestamp) > ttl


class _ClassificationCache:
    """Simple LRU cache backed by an OrderedDict.

    - Maximum capacity is bounded by *max_size*.
    - Entries expire after *ttl* seconds.
    - Cache key is ``(tool_name, frozenset_of_sorted_params)``.
    """

    def __init__(
        self,
        max_size: int = _CACHE_MAX_SIZE,
        ttl: float = _CACHE_TTL_SECONDS,
    ) -> None:
        self._max_size = max_size
        self._ttl = ttl
        self._store: OrderedDict[tuple[str, frozenset], _CacheEntry] = OrderedDict()

    @staticmethod
    def make_key(tool_name: str, params: dict) -> tuple[str, frozenset]:
        """Build a hashable cache key from *tool_name* and *params*.

        The key is ``(tool_name, frozenset(sorted(params.items())))`` which
        ensures that two calls with the same tool name and same parameter
        key-value pairs (regardless of insertion order) produce the same key.
        """
        # Only include hashable values; skip non-hashable ones gracefully.
        hashable_items: list[tuple[str, Any]] = []
        for k, v in sorted(params.items()):
            try:
                hash(v)
            except TypeError:
                # Convert unhashable values (lists, dicts) to a type-prefixed
                # JSON string to avoid collision between different types that
                # could produce the same JSON string (e.g. a string "[1,2]"
                # vs a list [1,2]).
                v = ("__json__", json.dumps(v, sort_keys=True, default=str))
            hashable_items.append((k, v))
        return (tool_name, frozenset(hashable_items))

    def get(self, key: tuple[str, frozenset]) -> _CacheEntry | None:
        """Retrieve a cache entry, returning None on miss or expiry."""
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.is_expired(self._ttl):
            # Lazy eviction of stale entry
            del self._store[key]
            return None
        # Move to end (most recently used)
        self._store.move_to_end(key)
        return entry

    def put(self, key: tuple[str, frozenset], entry: _CacheEntry) -> None:
        """Insert or replace a cache entry, evicting the LRU entry if at capacity."""
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = entry
        # Evict oldest entries if over capacity
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    @property
    def size(self) -> int:
        """Current number of entries (including potentially expired ones)."""
        return len(self._store)


# ===== Heuristic Fallback =====

class _HeuristicClassifier:
    """Rule-based heuristic classifier used when no LLM is available.

    Classification rules:
      1. Sensitive paths (``/etc/*``, ``.env*``, ``.ssh/*``, ``.git/*``)
         are always DENIED with high confidence.
      2. READ_ONLY tools are ALLOWED with high confidence.
      3. DESTRUCTIVE / HIGH_RISK tools are DENIED with high confidence.
      4. SAFE_WRITE tools are ALLOWED with medium confidence.
      5. Unknown tools are DENIED with medium confidence (conservative).
    """

    @staticmethod
    def _extract_path(params: dict) -> str:
        """Extract a file-system path from tool parameters.

        Checks common parameter names in order of preference.
        """
        for key in (
            "file_path", "path", "filepath",
            "dir", "directory", "workspace",
            "target", "destination",
        ):
            val = params.get(key, "")
            if val and isinstance(val, str):
                return val
        return ""

    @staticmethod
    def _is_sensitive_path(path: str) -> bool:
        """Return True if *path* matches a sensitive pattern.

        Distinguishes between absolute system paths (like /etc/) and relative
        patterns (like .env, .ssh/):
          - Absolute prefixes (e.g. "/etc/"): only match paths that start
            with the absolute prefix.
          - Relative prefixes (e.g. ".env", ".ssh/"): match any path segment
            that starts with the relative prefix.
        This avoids false positives where a project-internal directory like
        "project/src/etc/" would be incorrectly flagged as "/etc/".
        """
        if not path:
            return False
        if sys.platform == "win32":
            normalized = path.lower().replace("\\", "/")  # Windows: 大小写不敏感，需 .lower()
        else:
            normalized = path.replace("\\", "/")  # Unix: 保留大小写
        parts = normalized.split("/")

        for prefix in _SENSITIVE_PATH_PREFIXES:
            is_absolute = prefix.startswith("/")
            stripped = prefix.rstrip("/")

            if is_absolute:
                # Absolute system path: only match if the path itself starts
                # with this prefix (e.g. /etc/passwd matches /etc/ but
                # project/src/etc/file does NOT match).
                if normalized.startswith(prefix):
                    return True
            else:
                # Relative prefix (.env, .ssh/, .git/): match any path segment
                # that starts with the prefix (e.g. ".env.production",
                # ".ssh/id_rsa", ".git/config").
                for part in parts:
                    if part.startswith(stripped):
                        return True
        return False

    @classmethod
    def classify(
        cls,
        tool_name: str,
        params: dict,
    ) -> tuple[PermissionEffect, float, str]:
        """Classify a tool call using heuristic rules.

        Args:
            tool_name: Name of the tool being called.
            params: Tool call parameters.

        Returns:
            ``(effect, confidence, reason)`` tuple.
        """
        # --- Step 1: Sensitive path check (highest priority) ---
        path = cls._extract_path(params)
        if cls._is_sensitive_path(path):
            return (
                PermissionEffect.DENY,
                0.95,
                f"Heuristic: sensitive path detected ({path})",
            )

        # --- Step 2: Tool safety classification ---
        # Try known tool-name sets first (avoids needing a tool registry)
        if tool_name in _READ_ONLY_TOOL_NAMES:
            return (
                PermissionEffect.ALLOW,
                0.90,
                "Heuristic: read-only tool",
            )

        if tool_name in _HIGH_RISK_TOOL_NAMES:
            return (
                PermissionEffect.DENY,
                0.90,
                "Heuristic: high-risk tool",
            )

        if tool_name in _DESTRUCTIVE_TOOL_NAMES:
            return (
                PermissionEffect.DENY,
                0.85,
                "Heuristic: destructive tool",
            )

        # --- Step 3: Infer from tool name heuristics ---
        name_lower = tool_name.lower()

        # Common read-only patterns
        if any(
            name_lower.startswith(prefix)
            for prefix in ("read_", "list_", "get_", "search_", "find_", "explore_", "show_", "view_")
        ):
            return (
                PermissionEffect.ALLOW,
                0.80,
                "Heuristic: tool name suggests read-only operation",
            )

        # Common destructive / write patterns
        if any(
            name_lower.startswith(prefix)
            for prefix in ("delete_", "remove_", "rm_", "destroy_", "drop_")
        ):
            return (
                PermissionEffect.DENY,
                0.85,
                "Heuristic: tool name suggests destructive operation",
            )

        # Common safe-write patterns
        if any(
            name_lower.startswith(prefix)
            for prefix in ("write_", "create_", "update_", "generate_", "save_", "send_")
        ):
            # Re-check sensitive path for write tools with extra caution
            if path and cls._is_sensitive_path(path):
                return (
                    PermissionEffect.DENY,
                    0.95,
                    f"Heuristic: write to sensitive path ({path})",
                )
            return (
                PermissionEffect.ALLOW,
                0.60,
                "Heuristic: tool name suggests safe-write operation",
            )

        # --- Step 4: Unknown tools -- conservative deny ---
        return (
            PermissionEffect.DENY,
            0.55,
            "Heuristic: unknown tool, conservative deny",
        )


# ===== LLM Prompt Construction =====

_CLASSIFICATION_SYSTEM_PROMPT = """\
You are a security permission classifier for an AI coding agent. Your job is to \
decide whether a tool call should be ALLOWED or DENIED based on the tool name, \
its parameters, and the conversational context.

Classification guidelines:
- Read-only operations (reading files, listing directories, searching code) are \
generally SAFE and should be ALLOWED.
- Writing to user project files is generally SAFE if the path is within the \
project workspace.
- Writing to system directories (/etc/*), environment files (.env*), SSH \
directories (.ssh/*), or Git internals (.git/*) is UNSAFE and must be DENIED.
- Executing arbitrary shell commands, deleting files, or modifying system \
configuration is UNSAFE and must be DENIED unless the context clearly shows \
user intent and the path is safe.
- When in doubt, choose DENY to err on the side of caution.

You MUST respond with a valid JSON object and nothing else. The JSON object \
must have exactly these three fields:
  - "decision": either "allow" or "deny" (string)
  - "confidence": a float between 0.0 and 1.0 indicating your certainty
  - "reason": a brief explanation of your decision (string, max 200 chars)

Example response:
{"decision": "allow", "confidence": 0.92, "reason": "Read-only file access within project workspace"}\
"""


def _build_classification_messages(
    tool_name: str,
    params: dict,
    context: str,
) -> list[dict]:
    """Build the message list for the LLM classification request.

    Args:
        tool_name: Name of the tool being classified.
        params: Tool call parameters.
        context: Optional conversational context.

    Returns:
        List of message dicts suitable for ``ModelProvider.chat()``.
    """
    # Sanitize params for display (truncate large values)
    display_params: dict[str, Any] = {}
    for k, v in params.items():
        s = json.dumps(v, default=str) if not isinstance(v, str) else v
        if len(s) > 500:
            s = s[:500] + "...[truncated]"
        display_params[k] = s

    user_content = (
        f"Tool call to classify:\n"
        f"  Tool name: {tool_name}\n"
        f"  Parameters: {json.dumps(display_params, indent=2)}\n"
    )
    if context:
        user_content += f"  Context: {context}\n"

    user_content += "\nClassify this tool call as allowed or denied."

    return [
        {"role": "system", "content": _CLASSIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_llm_response(raw_text: str) -> tuple[PermissionEffect, float, str] | None:
    """Attempt to parse the structured JSON response from the LLM.

    Args:
        raw_text: Raw text output from the LLM.

    Returns:
        ``(effect, confidence, reason)`` on success, or ``None`` if parsing
        fails or the output is malformed.
    """
    if not raw_text:
        return None

    # Try to extract JSON from the response (the LLM might add extra text)
    text = raw_text.strip()

    # Look for JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        logger.debug("AI classifier: no JSON object found in LLM response")
        return None

    json_str = text[start : end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.debug(f"AI classifier: JSON parse error: {exc}")
        return None

    if not isinstance(data, dict):
        logger.debug("AI classifier: LLM response is not a JSON object")
        return None

    # Extract decision
    decision = data.get("decision", "")
    if not isinstance(decision, str):
        return None
    decision = decision.strip().lower()
    if decision == "allow":
        effect = PermissionEffect.ALLOW
    elif decision == "deny":
        effect = PermissionEffect.DENY
    else:
        logger.debug(f"AI classifier: unknown decision value '{decision}'")
        return None

    # Extract confidence (clamped to [0.0, 1.0])
    confidence = data.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    # Extract reason
    reason = data.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    reason = reason[:200]  # Truncate to match prompt constraint

    return (effect, confidence, reason)


# ===== AIPermissionClassifier =====

class AIPermissionClassifier:
    """LLM-based permission classifier for ambiguous tool calls.

    When the static rule engine in ``EnhancedPermissionManager`` cannot reach
    a decision (no matching rule, permission level check inconclusive), this
    classifier can be consulted as an advisory layer.

    The classifier first checks its local LRU cache.  On a cache miss it
    attempts LLM classification with a 5-second timeout.  If the model is
    unavailable, times out, or returns unparseable output, the heuristic
    fallback is used instead.

    A confidence threshold (default 0.8) enforces a conservative default:
    any classification with confidence below the threshold is converted to
    ``DENY``.

    Args:
        model: Optional ``ModelProvider`` instance for LLM classification.
        confidence_threshold: Minimum confidence required to accept an
            ``ALLOW`` decision.  Classifications below this threshold are
            converted to ``DENY`` (conservative default).

    Attributes:
        model: The ``ModelProvider`` (may be ``None``).
        confidence_threshold: The configured confidence threshold.
    """

    def __init__(
        self,
        model: ModelProvider | None = None,
        confidence_threshold: float = 0.8,
    ) -> None:
        self.model = model
        self.confidence_threshold = confidence_threshold
        self._cache = _ClassificationCache()
        self._heuristic = _HeuristicClassifier()

        # Statistics
        self._total_calls: int = 0
        self._llm_calls: int = 0
        self._llm_timeouts: int = 0
        self._llm_errors: int = 0
        self._heuristic_fallbacks: int = 0
        self._cache_hits: int = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def classify(
        self,
        tool_name: str,
        params: dict,
        context: str = "",
    ) -> tuple[PermissionEffect, float, str]:
        """Classify a tool call as ALLOW or DENY.

        The classification proceeds in the following order:

        1. **Cache lookup** -- if the same ``(tool_name, params)`` combination
           was recently classified and the cache entry has not expired, the
           cached result is returned immediately.
        2. **LLM classification** -- if a ``ModelProvider`` is configured, the
           classifier asks the LLM to decide.  A 5-second timeout is enforced.
        3. **Heuristic fallback** -- if the model is unavailable, times out,
           or returns unparseable output, a rule-based heuristic is used.

        After classification, if the confidence is below
        ``confidence_threshold``, the effect is forced to ``DENY`` (conservative
        default).

        The result is cached for future lookups.

        Args:
            tool_name: Name of the tool being called.
            params: Tool call parameters (may contain ``file_path``, ``path``,
                etc.).
            context: Optional conversational context providing intent
                information (e.g. "User requested code refactoring").

        Returns:
            ``(effect, confidence, reason)`` where:
            - *effect* is ``PermissionEffect.ALLOW`` or ``PermissionEffect.DENY``
            - *confidence* is a float in ``[0.0, 1.0]``
            - *reason* is a human-readable explanation string
        """
        self._total_calls += 1

        # 1. Cache lookup
        cache_key = _ClassificationCache.make_key(tool_name, params)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            logger.debug(
                f"AI classifier cache hit: tool={tool_name} "
                f"effect={cached.effect.value} conf={cached.confidence:.2f}"
            )
            return (cached.effect, cached.confidence, cached.reason)

        # 2. Attempt LLM classification
        effect: PermissionEffect
        confidence: float
        reason: str

        if self.model is not None:
            self._llm_calls += 1
            result = await self._classify_with_llm(tool_name, params, context)
            if result is not None:
                effect, confidence, reason = result
            else:
                # LLM failed -- fall back to heuristic
                self._heuristic_fallbacks += 1
                effect, confidence, reason = self._heuristic.classify(
                    tool_name, params,
                )
        else:
            # No model configured -- use heuristic directly
            self._heuristic_fallbacks += 1
            effect, confidence, reason = self._heuristic.classify(
                tool_name, params,
            )

        # 3. Apply confidence threshold (conservative default)
        if confidence < self.confidence_threshold:
            if effect == PermissionEffect.ALLOW:
                logger.info(
                    f"AI classifier: confidence {confidence:.2f} below threshold "
                    f"{self.confidence_threshold}, overriding ALLOW -> DENY "
                    f"(tool={tool_name})"
                )
                reason = (
                    f"Confidence {confidence:.2f} below threshold "
                    f"{self.confidence_threshold}: {reason}"
                )
                effect = PermissionEffect.DENY

        # 4. Cache the result
        self._cache.put(
            cache_key,
            _CacheEntry(
                effect=effect,
                confidence=confidence,
                reason=reason,
                timestamp=time.monotonic(),
            ),
        )

        logger.info(
            f"AI classifier result: tool={tool_name} "
            f"effect={effect.value} conf={confidence:.2f} "
            f"reason={reason[:80]}"
        )
        return (effect, confidence, reason)

    # ------------------------------------------------------------------ #
    # LLM Classification (private)
    # ------------------------------------------------------------------ #

    async def _classify_with_llm(
        self,
        tool_name: str,
        params: dict,
        context: str,
    ) -> tuple[PermissionEffect, float, str] | None:
        """Attempt to classify using the LLM model.

        Returns:
            ``(effect, confidence, reason)`` on success, or ``None`` on any
            failure (timeout, error, unparseable response).
        """
        messages = _build_classification_messages(tool_name, params, context)

        try:
            # Enforce timeout to avoid blocking the permission pipeline
            response = await asyncio.wait_for(
                self.model.chat(messages=messages),  # type: ignore[union-attr]
                timeout=_LLM_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            self._llm_timeouts += 1
            logger.warning(
                f"AI classifier: LLM call timed out after "
                f"{_LLM_TIMEOUT_SECONDS}s (tool={tool_name})"
            )
            return None
        except Exception as exc:
            self._llm_errors += 1
            logger.warning(
                f"AI classifier: LLM call failed: {exc} (tool={tool_name})"
            )
            return None

        # Extract raw text from the response
        raw_text = ""
        if isinstance(response, dict):
            raw_text = response.get("content", "")
        elif isinstance(response, str):
            raw_text = response
        if not raw_text:
            logger.debug("AI classifier: empty response from LLM")
            return None

        parsed = _parse_llm_response(raw_text)
        if parsed is None:
            logger.debug(
                f"AI classifier: could not parse LLM response "
                f"(tool={tool_name}, response={raw_text[:200]})"
            )
            return None

        return parsed

    # ------------------------------------------------------------------ #
    # Cache Management
    # ------------------------------------------------------------------ #

    def clear_cache(self) -> None:
        """Clear the classification cache."""
        self._cache.clear()
        logger.info("AI classifier cache cleared")

    @property
    def cache_size(self) -> int:
        """Current number of entries in the cache."""
        return self._cache.size

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #

    def get_stats(self) -> dict:
        """Return classification statistics.

        Returns:
            Dictionary with keys: ``total_calls``, ``llm_calls``,
            ``llm_timeouts``, ``llm_errors``, ``heuristic_fallbacks``,
            ``cache_hits``, ``cache_size``, ``confidence_threshold``.
        """
        return {
            "total_calls": self._total_calls,
            "llm_calls": self._llm_calls,
            "llm_timeouts": self._llm_timeouts,
            "llm_errors": self._llm_errors,
            "heuristic_fallbacks": self._heuristic_fallbacks,
            "cache_hits": self._cache_hits,
            "cache_size": self._cache.size,
            "confidence_threshold": self.confidence_threshold,
        }

    def reset_stats(self) -> None:
        """Reset all statistics counters to zero (does not clear cache)."""
        self._total_calls = 0
        self._llm_calls = 0
        self._llm_timeouts = 0
        self._llm_errors = 0
        self._heuristic_fallbacks = 0
        self._cache_hits = 0
