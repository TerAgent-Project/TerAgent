"""teragent.core.adapters — TAP Adapter implementations

Importing this package triggers registration of all adapters with
TAPAdapterRegistry. After importing, adapters can be created via:

    from teragent.core.adapter import TAPAdapterRegistry

    adapter = TAPAdapterRegistry.create("openai_compatible", base_url=..., api_key=...)
    adapter = TAPAdapterRegistry.create("anthropic_native", base_url=..., api_key=...)
    adapter = TAPAdapterRegistry.create("mock", delay=0.1)

Available adapters:
  - openai_compatible: OpenAI-compatible /chat/completions API (Mode A)
  - anthropic_native:  Anthropic /messages API (Mode B)
  - mock:              Local testing adapter (no network I/O)
"""

# Import adapter modules to trigger TAPAdapterRegistry.register() calls
from teragent.core.adapters import openai_compatible as _openai  # noqa: F401
from teragent.core.adapters import anthropic_native as _anthropic  # noqa: F401
from teragent.core.adapters import mock as _mock  # noqa: F401

# Re-export adapter classes for convenience
from teragent.core.adapters.openai_compatible import OpenAICompatibleAdapter
from teragent.core.adapters.anthropic_native import AnthropicNativeAdapter
from teragent.core.adapters.mock import MockAdapter

__all__ = [
    "OpenAICompatibleAdapter",
    "AnthropicNativeAdapter",
    "MockAdapter",
]
