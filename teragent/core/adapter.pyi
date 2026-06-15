"""Type stubs for teragent.core.adapter — TAPAdapter ABC + TAPAdapterRegistry"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncIterator

__all__ = [
    "TAPAdapter",
    "TAPAdapterRegistry",
]

if TYPE_CHECKING:
    from teragent.core.tap import CompiledPrompt, TAPResponse

class TAPAdapter(ABC):
    """TAP Adapter abstract base class."""

    @abstractmethod
    async def send(self, compiled: CompiledPrompt, model: str) -> TAPResponse: ...

    @abstractmethod
    async def stream(
        self, compiled: CompiledPrompt, model: str
    ) -> AsyncIterator[str]: ...

    async def close(self) -> None: ...

    @property
    def capabilities(self) -> dict: ...

    @property
    def required_mode(self) -> str: ...

class TAPAdapterRegistry:
    """Adapter registry — maps adapter names to Adapter classes."""

    _adapters: dict[str, type[TAPAdapter]]

    @classmethod
    def _get_registry(cls) -> dict[str, type[TAPAdapter]]: ...
    @classmethod
    def register(cls, name: str, adapter_cls: type[TAPAdapter]) -> None: ...
    @classmethod
    def get(cls, name: str) -> type[TAPAdapter] | None: ...
    @classmethod
    def create(cls, name: str, **kwargs: Any) -> TAPAdapter: ...
    @classmethod
    def available(cls) -> list[str]: ...
