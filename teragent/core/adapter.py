"""teragent.core.adapter — TAPAdapter ABC + TAPAdapterRegistry

An Adapter sends CompiledPrompt to a model API via HTTP.
It only handles network I/O (send/stream), not prompt optimization.

Different adapters target different API protocols:
  - OpenAI-compatible: /chat/completions with SSE streaming
  - Anthropic native: /messages with Anthropic-specific SSE format
  - Mock: Local testing adapter
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator

__all__ = [
    "TAPAdapter",
    "TAPAdapterRegistry",
]

if TYPE_CHECKING:
    from teragent.core.tap import CompiledPrompt, TAPResponse

logger = logging.getLogger(__name__)


class TAPAdapter(ABC):
    """TAP Adapter abstract base class

    Responsibility: Send CompiledPrompt to model API and return TAPResponse.
    Only handles HTTP I/O and response parsing, not prompt optimization.

    Subclasses must implement:
        send(compiled, model) -> TAPResponse
        stream(compiled, model) -> AsyncIterator[str]
    """

    @abstractmethod
    async def send(self, compiled: CompiledPrompt, model: str) -> TAPResponse:
        """Send a compiled prompt to the model API (non-streaming)

        Args:
            compiled: The compiled prompt to send
            model: Model identifier string

        Returns:
            TAPResponse with the model's output
        """
        ...

    @abstractmethod
    async def stream(self, compiled: CompiledPrompt, model: str) -> AsyncIterator[str]:
        """Stream a compiled prompt to the model API

        Args:
            compiled: The compiled prompt to send
            model: Model identifier string

        Yields:
            Text chunks as they arrive from the model
        """
        ...

    async def close(self) -> None:
        """Close any persistent resources (connections, clients, etc.)

        Override if the adapter maintains persistent HTTP connections.
        """
        pass

    @property
    def capabilities(self) -> dict:
        """Return adapter capabilities for feature detection"""
        return {
            "streaming": True,
            "tool_calling": False,
        }

    @property
    def required_mode(self) -> str:
        """The CompiledPrompt mode this adapter expects.

        Returns:
            "any" (accepts both modes), "messages" (Mode A), or "system_user" (Mode B)
        """
        return "any"


class TAPAdapterRegistry:
    """Adapter registry — maps adapter names to Adapter classes

    Note: _adapters is NOT declared as a class-level mutable default to avoid
    shared-state bugs across subclasses. It is lazily created per-class via
    _get_registry(), which checks cls.__dict__ to ensure each subclass gets
    its own independent registry.
    """

    @classmethod
    def _get_registry(cls) -> dict[str, type[TAPAdapter]]:
        """Return the adapter registry for *this* class, creating one if needed.

        Without this, subclasses would share the parent's ``_adapters`` dict
        because mutable class variables are inherited by reference.
        """
        if "_adapters" not in cls.__dict__:
            cls._adapters = {}
        return cls._adapters

    @classmethod
    def register(cls, name: str, adapter_cls: type[TAPAdapter]) -> None:
        """Register an Adapter class under a name"""
        cls._get_registry()[name] = adapter_cls
        logger.debug(f"Registered adapter: {name} -> {adapter_cls.__name__}")

    @classmethod
    def get(cls, name: str) -> type[TAPAdapter] | None:
        """Get a registered Adapter class by name"""
        return cls._get_registry().get(name)

    @classmethod
    def create(cls, name: str, **kwargs) -> TAPAdapter:
        """Create an Adapter instance by name

        Args:
            name: Registered adapter name
            **kwargs: Arguments to pass to the Adapter constructor
                Typically: base_url, api_key, timeout, extra_headers, etc.

        Returns:
            New Adapter instance

        Raises:
            ValueError: If no adapter is registered under the given name
        """
        adapter_cls = cls._get_registry().get(name)
        if adapter_cls is None:
            raise ValueError(
                f"Unknown adapter: {name}. Available: {list(cls._get_registry().keys())}"
            )
        return adapter_cls(**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """List all registered adapter names"""
        return list(cls._get_registry().keys())
