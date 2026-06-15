"""Type stubs for teragent.core.compiler — TAPCompiler ABC + TAPCompilerRegistry"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

__all__ = [
    "TAPCompiler",
    "TAPCompilerRegistry",
]

if TYPE_CHECKING:
    from teragent.core.tap import CompiledPrompt, TAPRequest

class TAPCompiler(ABC):
    """TAP Compiler abstract base class."""

    @property
    def supports_multimodal(self) -> bool: ...
    @property
    def supports_thinking_mode(self) -> bool: ...
    @property
    def max_context_tokens(self) -> int: ...

    @abstractmethod
    def compile(self, request: TAPRequest) -> CompiledPrompt: ...

    def get_system_prompt(self, intent: str) -> str: ...
    def _get_compiler_type(self) -> str: ...

    # Context injection helpers
    def _inject_context(
        self, messages: list[dict], request: TAPRequest
    ) -> list[dict]: ...
    def _build_context_string(self, request: TAPRequest) -> str: ...

    # Multimodal degradation
    def _handle_multimodal_degradation(self, request: TAPRequest) -> str: ...

class TAPCompilerRegistry:
    """Compiler registry — maps compiler names to Compiler classes."""

    _compilers: dict[str, type[TAPCompiler]]

    @classmethod
    def _get_registry(cls) -> dict[str, type[TAPCompiler]]: ...
    @classmethod
    def register(cls, name: str, compiler_cls: type[TAPCompiler]) -> None: ...
    @classmethod
    def get(cls, name: str) -> type[TAPCompiler] | None: ...
    @classmethod
    def create(cls, name: str, **kwargs: Any) -> TAPCompiler: ...
    @classmethod
    def available(cls) -> list[str]: ...
